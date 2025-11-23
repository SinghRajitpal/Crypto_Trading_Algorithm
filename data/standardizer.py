import numpy as np
from typing import Optional, Dict


class Standardizer:
    """Mean/std scaler for feature matrices (intercept handled by caller)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def stats(self) -> Dict[str, np.ndarray]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer not fitted")
        return {"mean": self.mean_, "std": self.std_}
