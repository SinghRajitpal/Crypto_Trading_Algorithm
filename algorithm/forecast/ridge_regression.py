from dataclasses import dataclass
from typing import Dict, List, Optional

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
    gcv: Optional[float]
    rl_vs_ls: Optional[float]
    samples: int
    t_threshold: float
    dropped_features: List[str]
    hat_mean: float
    hat_max: float
    resid_sigma: float
    feature_columns: List[str]
    outliers_dropped: Optional[int] = None


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
        self,
        symbol: str,
        X: np.ndarray,
        y: np.ndarray,
        fixed_k: Optional[float] = None,
        train_window: Optional[int] = None,
        skip_cv: bool = False,
        train_min: Optional[int] = None,
        feature_columns: Optional[List[str]] = None,
        outliers_dropped: Optional[int] = None,
    ) -> Optional[RidgeForecast]:
        """Fit ridge on per-asset data and return one-step-ahead expectation."""
        if X.size == 0 or y.size == 0:
            return None

        n_samples = min(len(y), len(X))
        if n_samples < config.REGRESSION_MIN_TRAIN:
            return None

        # Cap history to max bars
        if config.REGRESSION_MAX_BARS:
            X = X[-config.REGRESSION_MAX_BARS :]
            y = y[-config.REGRESSION_MAX_BARS :]
        if train_window is not None:
            X = X[-train_window:]
            y = y[-train_window:]
        n = len(y)
        eff_train_min = train_min if train_min is not None else config.REGRESSION_MIN_TRAIN
        if train_window is not None:
            eff_train_min = min(eff_train_min, train_window)
        if n < eff_train_min:
            return None

        best_k = fixed_k
        best_msep = np.inf
        best_gcv = np.inf
        gcv_lookup: Dict[float, float] = {}

        run_cv = not skip_cv or best_k is None
        k_grid = [fixed_k] if fixed_k is not None else list(self.k_grid)

        if run_cv:
            train_min = eff_train_min
            val_len = min(config.REGRESSION_VAL_WINDOW, max(1, n - train_min))

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
                for k in k_grid:
                    penalty = self._penalty_matrix(XtX.shape[0], k)
                    beta = self._ridge_beta(XtX, Xty, penalty)
                    preds = X_val_std @ beta
                    msep = float(np.mean((y_val - preds) ** 2))
                    # GCV approximation
                    try:
                        H = X_train_std @ np.linalg.pinv(XtX + penalty) @ X_train_std.T
                        trace_H = float(np.trace(H))
                        gcv = float(np.sum((y_train - X_train_std @ beta) ** 2) / (len(y_train) - trace_H) ** 2)
                    except Exception:
                        gcv = msep
                    if msep < best_msep:
                        best_msep = msep
                        best_k = k
                    if gcv < best_gcv:
                        best_gcv = gcv
                    if best_k is None or k not in gcv_lookup:
                        gcv_lookup[k] = gcv
        else:
            # No CV path: rely on provided k and compute diagnostics after fit.
            best_k = best_k if best_k is not None else (k_grid[0] if k_grid else None)

        if best_k is None:
            return None

        # Fit on full standardized data with best_k
        scaler_full = Standardizer().fit(X)
        X_std = self._with_intercept(scaler_full.transform(X))
        # Downcast to float32 for diagnostics-heavy ops to reduce transient RAM.
        X_std = X_std.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        XtX_full = X_std.T @ X_std
        penalty_full = self._penalty_matrix(X_std.shape[1], best_k)
        beta_ridge = self._ridge_beta(XtX_full, X_std.T @ y, penalty_full)

        # Pruning disabled: keep full ridge fit
        dropped: List[str] = []
        beta_final = beta_ridge

        fit_msep = float(np.mean((y - X_std @ beta_ridge) ** 2))
        if not run_cv:
            best_msep = fit_msep
            best_gcv = None

        latest_std = self._with_intercept(scaler_full.transform(X[-1:].reshape(1, -1))).astype(np.float32)
        y_hat = float((latest_std @ beta_ridge).item())
        mu_hat = float(np.exp(y_hat) - 1.0)

        # Hat/residual diagnostics
        hat_mean = 0.0
        hat_max = 0.0
        resid_sigma = 0.0
        try:
            inv_mat = np.linalg.pinv(XtX_full + best_k * np.eye(XtX_full.shape[0], dtype=XtX_full.dtype))
            hat_diag = self._hat_diag(X_std, inv_mat)
            hat_mean = float(np.mean(hat_diag))
            hat_max = float(np.max(hat_diag))
            resid = y - X_std @ beta_ridge
            resid_sigma = float(np.std(resid, ddof=1))
        except Exception:
            pass

        if best_gcv == np.inf:
            best_gcv = None

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
            gcv=best_gcv,
            rl_vs_ls=rl,
            samples=n,
            t_threshold=self.t_threshold,
            dropped_features=dropped,
            hat_mean=hat_mean,
            hat_max=hat_max,
            resid_sigma=resid_sigma,
            feature_columns=feature_columns or [],
            outliers_dropped=outliers_dropped,
        )

    @staticmethod
    def _penalty_matrix(size: int, k: float) -> np.ndarray:
        """Build ridge penalty with unpenalized intercept (index 0)."""
        pen = np.eye(size) * k
        if size > 0:
            pen[0, 0] = 0.0
        return pen

    @staticmethod
    def _ridge_beta(XtX: np.ndarray, Xty: np.ndarray, penalty: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(XtX + penalty) @ Xty

    @staticmethod
    def _with_intercept(X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.column_stack([intercept, X])

    @staticmethod
    def _hat_diag(X: np.ndarray, inv_xtx: np.ndarray) -> np.ndarray:
        """Return only diagonal of hat matrix without materializing n x n."""
        # (X @ inv_xtx) has shape (n, p); elementwise multiply by X and row-sum -> diag(H)
        return np.sum((X @ inv_xtx) * X, axis=1)
