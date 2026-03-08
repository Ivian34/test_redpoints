from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class AssetClassificatorModel:
    """Stage 1 model: TF-IDF(title) + LogisticRegression."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, titles: Iterable[str], y: Iterable[int]) -> "AssetClassificatorModel":
        title_list = [str(t) for t in titles]
        y_arr = np.asarray(list(y))
        X = self.vectorizer.fit_transform(title_list)
        self.classifier.fit(X, y_arr)
        self.classes_ = self.classifier.classes_
        self.is_fitted_ = True
        return self

    def predict(self, titles: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        title_list = [str(t) for t in titles]
        X = self.vectorizer.transform(title_list)
        return self.classifier.predict(X)

    def predict_proba(self, titles: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        title_list = [str(t) for t in titles]
        X = self.vectorizer.transform(title_list)
        return self.classifier.predict_proba(X)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("AssetClassificatorModel is not fitted yet.")
