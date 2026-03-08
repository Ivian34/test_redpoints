from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("REDPOINTS_DATA_DIR", BASE_DIR / "data"))
BIN_DIR = Path(os.getenv("REDPOINTS_BIN_DIR", BASE_DIR / "bin"))
STORAGE_DIR = Path(os.getenv("REDPOINTS_STORAGE_DIR", BASE_DIR / "storage"))

RESULTS_PATH = DATA_DIR / "Result_7.tsv"
LABELS_PATH = DATA_DIR / "labels.tsv"

ASSET_MODEL_PATH = BIN_DIR / "asset_classifier_pipeline.joblib"
SUSPICION_MODEL_PATH = BIN_DIR / "suspicion_scorer_pipeline.joblib"
SIMILARITY_MODEL_PATH = BIN_DIR / "similarity_engine.joblib"
ANALYSES_DB_PATH = STORAGE_DIR / "analyses.db"
