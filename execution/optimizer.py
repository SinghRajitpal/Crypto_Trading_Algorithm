from __future__ import annotations

from typing import Dict, List

import numpy as np

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MeanVarianceOptimizer:
    """Single-period mean–variance optimizer with basic constraints."""

    def __init__(
        self,
        risk_aversion: float = config.RISK_AVERSION,
        w_min: float = config.WEIGHT_MIN,
        w_max: float = config.WEIGHT_MAX,
        max_net: float = config.MAX_NET_EXPOSURE,
        max_gross: float = config.MAX_GROSS_EXPOSURE,
    ) -> None:
        self.risk_aversion = max(risk_aversion, 1e-6)
        self.weight_min = w_min
        self.weight_max = w_max
        self.max_net = max_net
        self.max_gross = max_gross

    def optimize(
        self, expected_returns: Dict[str, float], covariance: np.ndarray, symbols: List[str]
    ) -> Dict[str, float]:
        if covariance.shape[0] != len(symbols):
            raise ValueError("Covariance matrix must align with symbol list")

        mu = np.array([expected_returns.get(symbol, 0.0) for symbol in symbols], dtype=float)
        cov = covariance.copy()

        # Regularize covariance for numerical stability
        cov += np.eye(len(symbols)) * 1e-6

        try:
            inv_cov = np.linalg.pinv(cov)
        except np.linalg.LinAlgError as exc:
            logger.error("Optimizer covariance inversion failed: %s", exc)
            return {symbol: 0.0 for symbol in symbols}

        raw_weights = (1.0 / self.risk_aversion) * inv_cov.dot(mu)

        # Clip to bounds
        clipped = np.clip(raw_weights, self.weight_min, self.weight_max)

        # Enforce gross exposure constraint
        gross = np.sum(np.abs(clipped))
        if gross > self.max_gross and gross > 0:
            clipped *= self.max_gross / gross

        # Enforce net exposure constraint
        net = float(np.sum(clipped))
        if abs(net) > self.max_net:
            adjustment = net - np.sign(net) * self.max_net
            clipped -= adjustment / len(clipped)

        weights = {symbol: float(weight) for symbol, weight in zip(symbols, clipped)}
        return weights
