from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from data.standardizer import Standardizer
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RidgeForecast:
    expected_log_return: float
    expected_simple_return: float
    k_best: float
    msep: float
    rl_vs_ls: Optional[float]
    samples: int
    t_threshold: float
    dropped_features: List[str]


class RidgeRegressionForecaster:
    """Per-asset ridge forecaster with rolling-origin CV and optional ridge-selection pruning."""

    def __init__(
        self,
        k_grid: Optional[List[float]] = None,
        t_threshold: float = config.RIDGE_T_THRESHOLD,
    ) -> None:
        self.k_grid = k_grid or config.RIDGE_K_GRID
        self.t_threshold = t_threshold

    def forecast(
        self, symbol: str, X: np.ndarray, y: np.ndarray
    ) -> Optional[RidgeForecast]:
        """Fit ridge on per-asset data and return one-step-ahead expectation."""
        if X.size == 0 or y.size == 0:
            return None

        n_samples = min(len(y), len(X))
        if n_samples < config.REGRESSION_MIN_TRAIN:
            return None

        # Cap history to max bars
        X = X[-config.REGRESSION_MAX_BARS :]
        y = y[-config.REGRESSION_MAX_BARS :]
        n = len(y)
        if n < config.REGRESSION_MIN_TRAIN:
            return None

        # Rolling-origin CV
        train_min = config.REGRESSION_MIN_TRAIN
        val_len = config.REGRESSION_VAL_WINDOW
        best_k = None
        best_msep = np.inf
        gcv_lookup: Dict[float, float] = {}

        # Standardize using train stats each split
        for start in range(0, n - train_min, val_len):
            train_end = min(start + train_min, n - val_len)
            if train_end <= start or train_end + val_len > n:
                continue
            X_train = X[start:train_end]
            y_train = y[start:train_end]
            X_val = X[train_end : train_end + val_len]
            y_val = y[train_end : train_end + val_len]

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
                if best_k is None or k not in gcv_lookup:
                    gcv_lookup[k] = msep  # fallback if no GCV calc

        if best_k is None:
            return None

        # Fit on full standardized data with best_k
        scaler_full = Standardizer().fit(X)
        X_std = self._with_intercept(scaler_full.transform(X))
        beta_ridge = self._ridge_beta(X_std.T @ X_std, X_std.T @ y, best_k)

        # Optional ridge-selection (t-stat pruning)
        dropped: List[str] = []
        beta_final = beta_ridge
        if self.t_threshold and self.t_threshold > 0:
            hat = X_std @ np.linalg.pinv(X_std.T @ X_std + best_k * np.eye(X_std.shape[1])) @ X_std.T
            residuals = y - X_std @ beta_ridge
            sigma2 = float(np.mean(residuals**2))
            XtX = X_std.T @ X_std
            cov_beta = sigma2 * np.linalg.pinv(XtX + best_k * np.eye(XtX.shape[0])) @ XtX @ np.linalg.pinv(
                XtX + best_k * np.eye(XtX.shape[0])
            )
            t_stats = np.abs(beta_ridge) / (np.sqrt(np.diag(cov_beta)) + 1e-12)
            # Skip intercept index 0
            weak_idx = [i for i, t in enumerate(t_stats) if i > 0 and t < self.t_threshold]
            if weak_idx:
                mask = np.ones(X_std.shape[1], dtype=bool)
                mask[weak_idx] = False
                # Intercept preserved
                X_pruned = X_std[:, mask]
                beta_final = self._ridge_beta(X_pruned.T @ X_pruned, X_pruned.T @ y, best_k)
                # Track dropped features (intercept excluded)
                dropped = [f"feature_{i-1}" for i in weak_idx]
                # Reassign for prediction dimension
                beta_ridge = np.zeros_like(beta_ridge)
                beta_ridge[mask] = beta_final

        latest_std = self._with_intercept(scaler_full.transform(X[-1:].reshape(1, -1)))
        y_hat = float(latest_std @ beta_ridge)
        mu_hat = float(np.exp(y_hat) - 1.0)

        # RL vs LS as diagnostic
        rl = None
        try:
            beta_ls, *_ = np.linalg.lstsq(X_std, y, rcond=None)
            y_val = X_std @ beta_ls
            msep_ls = float(np.mean((y - y_val) ** 2))
            rl = best_msep / msep_ls if msep_ls > 0 else None
        except Exception:
            rl = None

        return RidgeForecast(
            expected_log_return=y_hat,
            expected_simple_return=mu_hat,
            k_best=best_k,
            msep=best_msep,
            rl_vs_ls=rl,
            samples=n,
            t_threshold=self.t_threshold,
            dropped_features=dropped,
        )

    @staticmethod
    def _ridge_beta(XtX: np.ndarray, Xty: np.ndarray, k: float) -> np.ndarray:
        p = XtX.shape[0]
        return np.linalg.pinv(XtX + k * np.eye(p)) @ Xty

    @staticmethod
    def _with_intercept(X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.column_stack([intercept, X])
