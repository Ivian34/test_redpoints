from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .similarity_module import ListingSimilarityEngine


class SuspicionScorer:
    """
    Stage 2 scorer using:
    - TF-IDF(title)
    - top1_similarity
    - mean_topk_similarity
    """

    def __init__(self, similarity_top_k: int = 3) -> None:
        self.similarity_top_k = similarity_top_k
        self.text_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        self.similarity_engine = ListingSimilarityEngine(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
        )
        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        self.reference_titles_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, titles: Iterable[str], y: Iterable[int]) -> "SuspicionScorer":
        title_list = [str(t) for t in titles]
        y_arr = np.asarray(list(y))

        self.text_vectorizer.fit(title_list)

        self.reference_titles_ = np.asarray(title_list, dtype=object)
        reference_df = pd.DataFrame({"title": title_list})
        self.similarity_engine.fit(reference_df, title_col="title")

        X = self._build_features(title_list, exclude_self=True)
        self.classifier.fit(X, y_arr)
        self.classes_ = self.classifier.classes_
        self.is_fitted_ = True
        return self

    def predict(self, titles: Iterable[str]) -> np.ndarray:
        proba = self.predict_proba(titles)
        positive_idx = list(self.classes_).index(1)
        return (proba[:, positive_idx] >= 0.5).astype(int)

    def predict_proba(self, titles: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        title_list = [str(t) for t in titles]
        X = self._build_features(title_list, exclude_self=False)
        return self.classifier.predict_proba(X)

    def _build_features(self, titles: list[str], exclude_self: bool) -> sparse.csr_matrix:
        text_X = self.text_vectorizer.transform(titles)
        sim_features = self._similarity_features(titles, exclude_self=exclude_self)
        sim_X = sparse.csr_matrix(sim_features)
        return sparse.hstack([text_X, sim_X], format="csr")

    def _similarity_features(self, titles: list[str], exclude_self: bool) -> np.ndarray:
        rows = []
        for query_title in titles:
            requested_k = self.similarity_top_k + (1 if exclude_self else 0)
            top_k_items = self.similarity_engine.query(query_title, top_k=requested_k)
            sims = []
            removed_self = False

            for item in top_k_items:
                sim = float(item["similarity_score"])
                ref_title = str(item["reference_title"])

                if exclude_self and not removed_self and ref_title == query_title:
                    removed_self = True
                    continue

                sims.append(sim)
                if len(sims) == self.similarity_top_k:
                    break

            if not sims:
                sims = [0.0]

            top1_similarity = sims[0]
            mean_topk_similarity = float(np.mean(sims))
            rows.append([top1_similarity, mean_topk_similarity])

        return np.asarray(rows, dtype=np.float64)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("SuspicionScorer is not fitted yet.")
