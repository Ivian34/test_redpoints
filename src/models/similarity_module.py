from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ListingSimilarityEngine:
    """TF-IDF + cosine nearest-neighbors for listing title similarity."""

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.reference_df: pd.DataFrame | None = None
        self.reference_matrix = None

    def fit(self, reference_df: pd.DataFrame, title_col: str = "title") -> None:
        if title_col not in reference_df.columns:
            raise ValueError(f"Column '{title_col}' not found in reference_df")

        clean_df = reference_df.copy()
        clean_df[title_col] = clean_df[title_col].fillna("").astype(str).str.strip()
        clean_df = clean_df[clean_df[title_col] != ""].reset_index(drop=True)
        clean_df["reference_id"] = clean_df.index

        self.reference_matrix = self.vectorizer.fit_transform(clean_df[title_col])
        self.nn_index.fit(self.reference_matrix)
        self.reference_df = clean_df

    def query(self, listing_title: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.reference_df is None or self.reference_matrix is None:
            raise RuntimeError("Similarity engine is not fitted yet.")

        if top_k <= 0:
            return []

        query_text = str(listing_title or "").strip()
        query_vec = self.vectorizer.transform([query_text])

        n_neighbors = min(top_k, len(self.reference_df))
        distances, indices = self.nn_index.kneighbors(
            query_vec, n_neighbors=n_neighbors, return_distance=True
        )

        results: list[dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            row = self.reference_df.iloc[int(idx)]
            results.append(
                {
                    "reference_id": int(row["reference_id"]),
                    "reference_title": str(row["title"]),
                    "similarity_score": float(1.0 - distance),
                }
            )
        return results

    def save(self, output_path: str | Path) -> None:
        payload = {
            "vectorizer": self.vectorizer,
            "nn_index": self.nn_index,
            "reference_df": self.reference_df,
            "reference_matrix": self.reference_matrix,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "ListingSimilarityEngine":
        payload = joblib.load(model_path)
        engine = cls()
        engine.vectorizer = payload["vectorizer"]
        engine.nn_index = payload["nn_index"]
        engine.reference_df = payload["reference_df"]
        engine.reference_matrix = payload["reference_matrix"]
        return engine


def load_reference_listings(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    if "title" not in df.columns:
        df = pd.read_csv(path, sep=sep, header=None, encoding="utf-8")
        df = df.rename(columns={0: "title"})
    return df


if __name__ == "__main__":
    # Demo source aligned with API configuration.
    source_path = Path("Result_7.tsv")
    if not source_path.exists():
        raise FileNotFoundError(f"Reference data not found: {source_path}")

    references = load_reference_listings(source_path)
    engine = ListingSimilarityEngine()
    engine.fit(references, title_col="title")

    query_title = "Eileen Fisher Tunic Top Women's Size XS Gray Lyocell Tencel"
    top_k_results = engine.query(query_title, top_k=5)

    print(f"Query: {query_title}")
    for rank, item in enumerate(top_k_results, start=1):
        print(
            f"{rank}. score={item['similarity_score']:.4f} "
            f"id={item['reference_id']} title={item['reference_title']}"
        )
