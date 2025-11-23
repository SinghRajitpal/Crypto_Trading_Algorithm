from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MeanVarianceOptimizer:
    """Single-period mean–variance optimizer with basic constraints and optional turnover penalty."""

    def __init__(
        self,
        risk_aversion: float = config.RISK_AVERSION,
        w_min: float = config.WEIGHT_MIN,
        w_max: float = config.WEIGHT_MAX,
        max_net: float = config.MAX_NET_EXPOSURE,
        max_gross: float = config.MAX_GROSS_EXPOSURE,
        turnover_lambda: float = config.TURNOVER_PENALTY_LAMBDA,
        impact_kappa_default: float = config.IMPACT_KAPPA_DEFAULT,
        impact_kappa_overrides: Optional[Dict[str, float]] = None,
        turnover_horizon: int = 1,
    ) -> None:
        self.risk_aversion = max(risk_aversion, 1e-6)
        self.weight_min = w_min
        self.weight_max = w_max
        self.max_net = max_net
        self.max_gross = max_gross
        self.turnover_lambda = max(turnover_lambda, 0.0)
        self.impact_kappa_default = max(impact_kappa_default, 0.0)
        self.impact_kappa_overrides = impact_kappa_overrides or {}
        self.turnover_horizon = max(1, turnover_horizon)

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance: np.ndarray,
        symbols: List[str],
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if covariance.shape[0] != len(symbols):
            raise ValueError("Covariance matrix must align with symbol list")

        mu = np.array([expected_returns.get(symbol, 0.0) for symbol in symbols], dtype=float)
        cov = covariance.copy()

        # Regularize covariance for numerical stability
        cov += np.eye(len(symbols)) * 1e-6

        # Turnover/impact penalty: penalize (w - w_prev)^2 with per-asset impact kappa.
        # Approximate by inflating covariance and nudging mu toward prev weights.
        cov_eff = cov.copy()
        mu_eff = mu.copy()
        if prev_weights and self.turnover_lambda > 0:
            kappas = np.array(
                [
                    self.impact_kappa_overrides.get(symbol, self.impact_kappa_default)
                    for symbol in symbols
                ],
                dtype=float,
            )
            w_prev_vec = np.array([prev_weights.get(symbol, 0.0) for symbol in symbols], dtype=float)
            diag_penalty = (2 * self.turnover_lambda / self.risk_aversion) * np.diag(1.0 + kappas)
            cov_eff += diag_penalty
            mu_eff += (2 * self.turnover_lambda / self.risk_aversion) * w_prev_vec / self.turnover_horizon

        try:
            inv_cov = np.linalg.pinv(cov_eff)
        except np.linalg.LinAlgError as exc:
            logger.error("Optimizer covariance inversion failed: %s", exc)
            return {symbol: 0.0 for symbol in symbols}

        raw_weights = (1.0 / self.risk_aversion) * inv_cov.dot(mu_eff)

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
