from functools import lru_cache
from contextlib import asynccontextmanager
import json

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from .models.asset_classificator_model import AssetClassificatorModel
from .models.suspicion_model import SuspicionScorer
from .models.similarity_module import ListingSimilarityEngine, load_reference_listings
from .config import (
    ASSET_MODEL_PATH,
    SUSPICION_MODEL_PATH,
    SIMILARITY_MODEL_PATH,
    RESULTS_PATH,
    ANALYSES_DB_PATH,
)
from .storage import (
    init_db,
    insert_analysed_listing,
    get_analysed_listings_above_threshold,
    get_last_n_analysed_listings_db
)

MODEL_PATH = ASSET_MODEL_PATH
STAGE2_MODEL_PATH = SUSPICION_MODEL_PATH
REFERENCE_DATA_PATH = RESULTS_PATH
ASSET_METADATA_PATH = ASSET_MODEL_PATH.parent / "assetClassificator_metadata.json"
SUSPICION_METADATA_PATH = SUSPICION_MODEL_PATH.parent / "suspicionScorer_metadata.json"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(ANALYSES_DB_PATH)
    yield #després del yield es posaria codi de shutdown de la api


app = FastAPI(title="Listing Pipeline API", version="0.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Listing title to analyze")
    top_k: int = Field(3, ge=1, le=50, description="Number of similar references to return")


class SimilarityItem(BaseModel):
    reference_id: int
    reference_title: str
    similarity_score: float


class AnalyzeResponse(BaseModel):
    title: str
    stage_1_ran: bool
    similarity_ran: bool
    stage_2_ran: bool
    is_asset: bool
    asset_score:float
    suspicion_flag: bool | None = None
    suspicion_score: float | None = None
    similarity_score: float
    top_k: int
    top_k_most_similar_reference_listings: list[SimilarityItem]


class AnalysedListingRecord(BaseModel):
    id: int
    created_at_utc: str
    title: str
    stage_1_ran: bool
    stage_2_ran: bool
    similarity_ran: bool
    is_asset: bool
    asset_score: float
    suspicion_flag: bool | None = None
    suspicion_score: float | None = None
    similarity_score: float
    top_k: int
    top_k_most_similar_reference_listings: list[SimilarityItem]


@lru_cache(maxsize=1)#Tiene el modelo cargado lazy loading
def _load_stage1_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at '{MODEL_PATH}'. "
            "Run train/asset_classificator.py first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)#Tiene el modelo cargado lazy loading
def _load_similarity_engine() -> ListingSimilarityEngine:
    if SIMILARITY_MODEL_PATH.exists():
        return ListingSimilarityEngine.load(SIMILARITY_MODEL_PATH)

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            "No reference listings data found. "
            f"Expected '{REFERENCE_DATA_PATH}'."
        )

    references = load_reference_listings(REFERENCE_DATA_PATH)
    engine = ListingSimilarityEngine()
    engine.fit(references, title_col="title")
    SIMILARITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine.save(SIMILARITY_MODEL_PATH)
    return engine


@lru_cache(maxsize=1)#Tiene el modelo cargado lazy loading
def _load_stage2_pipeline():
    if not STAGE2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at '{STAGE2_MODEL_PATH}'. "
            "Run train/suspicion_scorer.py first."
        )
    try:
        return joblib.load(STAGE2_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Stage 2 model from '{STAGE2_MODEL_PATH}'. "
            "Re-train with: python train/suspicion_scorer.py"
        ) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage1_model_path": str(MODEL_PATH),
        "stage1_model_exists": MODEL_PATH.exists(),
        "stage2_model_path": str(STAGE2_MODEL_PATH),
        "stage2_model_exists": STAGE2_MODEL_PATH.exists(),
        "similarity_model_path": str(SIMILARITY_MODEL_PATH),
        "similarity_model_exists": SIMILARITY_MODEL_PATH.exists(),
        "reference_data_path": str(REFERENCE_DATA_PATH),
        "reference_data_exists": REFERENCE_DATA_PATH.exists(),
    }


def _positive_class_score(model, title: str) -> float | None:
    proba = model.predict_proba([title])[0]
    idx_pos = list(model.classes_).index(1)
    return float(proba[idx_pos])


@app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
def analyze_listing(payload: AnalyzeRequest):
    try:
        pipeline = _load_stage1_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        similarity_engine = _load_similarity_engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")

    # Similarity always runs, regardless of asset prediction.
    top_k_results = similarity_engine.query(title, top_k=payload.top_k)
    if not top_k_results:
        raise HTTPException(status_code=500, detail="Similarity engine returned no references")
    similarity_score = float(top_k_results[0]["similarity_score"])

    # Stage 1 sempre corre.
    stage1_pred = int(pipeline.predict([title])[0])
    is_asset = stage1_pred == 1
    asset_score = _positive_class_score(pipeline, title)
    if asset_score is None:
        asset_score = float(stage1_pred)

    stage_2_ran = False
    suspicion_flag = None
    suspicion_score = None
    if is_asset:
        try:
            stage2_pipeline = _load_stage2_pipeline()
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        stage2_pred = int(stage2_pipeline.predict([title])[0])
        suspicion_flag = stage2_pred == 1
        suspicion_score = _positive_class_score(stage2_pipeline, title)
        if suspicion_score is None:
            suspicion_score = float(stage2_pred)
        stage_2_ran = True

    response_payload = {
        "title": title,
        "stage_1_ran": True,
        "similarity_ran": True,
        "stage_2_ran": stage_2_ran,
        "is_asset": is_asset,
        "asset_score": asset_score,
        "suspicion_flag": suspicion_flag,
        "suspicion_score": suspicion_score,
        "similarity_score": similarity_score,
        "top_k": payload.top_k,
        "top_k_most_similar_reference_listings": top_k_results,
    }
    insert_analysed_listing(ANALYSES_DB_PATH, response_payload)# a la db
    return AnalyzeResponse(**response_payload)


@app.get("/analyzed-listings/by-threshold", response_model=list[AnalysedListingRecord])
def get_analyzed_listings_by_threshold(
    threshold: float = Query(..., ge=0.0, le=1.0),
    stage: int = Query(..., ge=1, le=2),
):
    """
    Stage 1: returns all previously analyzed listings with asset_score >= threshold.
    Stage 2: returns only asset listings with suspicion_score >= threshold.
    """
    try:
        records = get_analysed_listings_above_threshold(
            ANALYSES_DB_PATH, stage=stage, threshold=threshold
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return records

@app.get("/analyzed-listings/lastN", response_model=list[AnalysedListingRecord])
def get_last_n_analysed_listings(
    n: int = Query(..., ge=1, le=50)
):
    "Returns de last n analysed listings"
    try:
        records =  get_last_n_analysed_listings_db(ANALYSES_DB_PATH, n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return records

@app.get("/model-metadata")
def get_model_metadata(stage: int = Query(..., ge=1, le=2)):
    if stage == 1:
        metadata_path = ASSET_METADATA_PATH
    else:
        metadata_path = SUSPICION_METADATA_PATH

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Metadata file for stage {stage} not found at '{metadata_path}'. "
                "Train the stage model first."
            ),
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid metadata JSON for stage {stage} at '{metadata_path}'.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read metadata for stage {stage} at '{metadata_path}'.",
        ) from exc

    return {
        "stage": stage,
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
