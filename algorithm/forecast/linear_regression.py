from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionOutput:
    expected_return: float
    beta0: float
    beta1: float
    sample_count: int
    r_squared: float


class LinearRegressionForecaster:
    """Rolling linear regression forecaster for per-asset expected returns."""

    def __init__(self, window: int, min_samples: Optional[int] = None):
        self.window = window
        self.min_samples = min_samples or max(10, window // 2)

    def forecast(
        self, symbol: str, features: List[float], returns: List[float]
    ) -> Optional[RegressionOutput]:
        """Fit rolling OLS and return the one-step-ahead expected return."""
        if len(features) < 2 or len(returns) < 1:
            return None

        paired_samples = min(len(features) - 1, len(returns))
        if paired_samples < self.min_samples:
            return None

        window = min(self.window, paired_samples)
        x_series = np.array(features[-(window + 1) : -1], dtype=float)
        y_series = np.array(returns[-window:], dtype=float)

        if np.isnan(x_series).any() or np.isnan(y_series).any():
            logger.debug(f"[{symbol}] NaN encountered in regression window")
            return None

        X = np.column_stack([np.ones(window), x_series])
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y_series, rcond=None)
        except np.linalg.LinAlgError as exc:
            logger.warning(f"[{symbol}] Regression failed: {exc}")
            return None

        beta0, beta1 = beta
        latest_feature = float(features[-1])
        expected_return = float(beta0 + beta1 * latest_feature)

        if window > 1:
            y_hat = X @ beta
            ss_res = float(np.sum((y_series - y_hat) ** 2))
            ss_tot = float(np.sum((y_series - np.mean(y_series)) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            r_squared = 0.0

        return RegressionOutput(
            expected_return=expected_return,
            beta0=float(beta0),
            beta1=float(beta1),
            sample_count=window,
            r_squared=r_squared,
        )
