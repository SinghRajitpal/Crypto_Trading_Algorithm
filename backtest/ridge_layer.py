from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

import numpy as np

import config
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster
from data.standardizer import Standardizer


@dataclass
class RidgeLayerResult:
    k_per_asset: Dict[str, float]
    msep_per_asset: Dict[str, float]
    rl_vs_ls: Dict[str, Optional[float]]
    t_threshold: float
    samples_per_asset: Dict[str, int]

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "k_per_asset": self.k_per_asset,
                    "msep_per_asset": self.msep_per_asset,
                    "rl_vs_ls": self.rl_vs_ls,
                    "t_threshold": self.t_threshold,
                    "samples_per_asset": self.samples_per_asset,
                },
                f,
                indent=2,
            )


class RidgeLayerSelector:
    """Layer A selector: rolling-origin CV to choose k per asset with min training enforcement."""

    def __init__(
        self,
        k_grid: Optional[List[float]] = None,
        train_min: int = config.REGRESSION_MIN_TRAIN,
        val_len: int = config.REGRESSION_VAL_WINDOW,
        t_threshold: float = config.RIDGE_T_THRESHOLD,
    ) -> None:
        self.k_grid = k_grid or config.RIDGE_K_GRID
        self.train_min = train_min
        self.val_len = val_len
        self.t_threshold = t_threshold

    def select(self, data_engine, universe: Optional[List[str]] = None) -> RidgeLayerResult:
        k_per_asset: Dict[str, float] = {}
        msep_per_asset: Dict[str, float] = {}
        rl_vs_ls: Dict[str, Optional[float]] = {}
        samples_per_asset: Dict[str, int] = {}

        symbols = universe or data_engine.get_active_universe()
        for sym in symbols:
            X, y, ts, cols = data_engine.get_feature_matrix(sym)
            if y.size < self.train_min:
                continue
            X = X[-config.REGRESSION_MAX_BARS :]
            y = y[-config.REGRESSION_MAX_BARS :]
            n = len(y)
            if n < self.train_min:
                continue

            best_k = None
            best_msep = np.inf
            # Rolling-origin splits
            for start in range(0, n - self.train_min, self.val_len):
                train_end = min(start + self.train_min, n - self.val_len)
                if train_end <= start or train_end + self.val_len > n:
                    continue
                X_train = X[start:train_end]
                y_train = y[start:train_end]
                X_val = X[train_end : train_end + self.val_len]
                y_val = y[train_end : train_end + self.val_len]

                scaler = Standardizer().fit(X_train)
                X_train_std = scaler.transform(X_train)
                X_val_std = scaler.transform(X_val)

                X_train_std = self._with_intercept(X_train_std)
                X_val_std = self._with_intercept(X_val_std)

                XtX = X_train_std.T @ X_train_std
                Xty = X_train_std.T @ y_train
                for k in self.k_grid:
                    beta = self._ridge_beta(XtX, Xty, k)
                    preds = X_val_std @ beta
                    msep = float(np.mean((y_val - preds) ** 2))
                    if msep < best_msep:
                        best_msep = msep
                        best_k = k

            if best_k is None:
                continue

            forecaster = RidgeRegressionForecaster(k_grid=[best_k], t_threshold=self.t_threshold)
            ridge_result = forecaster.forecast(sym, X, y)
            if not ridge_result:
                continue

            k_per_asset[sym] = best_k
            msep_per_asset[sym] = best_msep
            rl_vs_ls[sym] = ridge_result.rl_vs_ls
            samples_per_asset[sym] = ridge_result.samples

        return RidgeLayerResult(
            k_per_asset=k_per_asset,
            msep_per_asset=msep_per_asset,
            rl_vs_ls=rl_vs_ls,
            t_threshold=self.t_threshold,
            samples_per_asset=samples_per_asset,
        )

    @staticmethod
    def _with_intercept(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _ridge_beta(XtX: np.ndarray, Xty: np.ndarray, k: float) -> np.ndarray:
        p = XtX.shape[0]
        return np.linalg.pinv(XtX + k * np.eye(p)) @ Xty
