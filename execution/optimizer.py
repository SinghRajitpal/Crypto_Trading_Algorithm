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
        slippage_bps_default: float = config.SLIPPAGE_BPS_DEFAULT,
        slippage_bps_overrides: Optional[Dict[str, float]] = None,
        turnover_horizon: int = 1,
    ) -> None:
        self.risk_aversion = max(risk_aversion, 1e-6)
        self.weight_min = w_min
        self.weight_max = w_max
        self.max_net = max_net
        self.max_gross = max_gross
        self.turnover_lambda = max(turnover_lambda, 0.0)
        self.slippage_bps_default = max(slippage_bps_default, 0.0)
        self.slippage_bps_overrides = slippage_bps_overrides or {}
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

        # Turnover penalty: penalize (w - w_prev)^2 with per-asset weights derived from slippage (bps).
        # Map to closed-form: argmax mu'w - (γ/2) w'Σw - λ ||w - w_prev||^2_slip
        # Solution: w = (γΣ + 2λK)^(-1) (mu + 2λK w_prev), K=diag(slippage_scale)
        slippage_scale = np.ones(len(symbols), dtype=float)
        if prev_weights and self.turnover_lambda > 0:
            slippage_bps = np.array(
                [self.slippage_bps_overrides.get(symbol, self.slippage_bps_default) for symbol in symbols],
                dtype=float,
            )
            # Normalize bps to a small, positive scaling factor to weight expensive-to-trade assets higher.
            slippage_scale = 1.0 + np.maximum(slippage_bps, 0.0) / 10000.0
            w_prev_vec = np.array([prev_weights.get(symbol, 0.0) for symbol in symbols], dtype=float)
            mu = mu + (2 * self.turnover_lambda / self.turnover_horizon) * (slippage_scale * w_prev_vec)

        # Regularize covariance for numerical stability
        cov += np.eye(len(symbols)) * 1e-6
        K = np.diag(slippage_scale)
        cov_eff = self.risk_aversion * cov + 2 * self.turnover_lambda * K

        try:
            inv_term = np.linalg.pinv(cov_eff)
        except np.linalg.LinAlgError as exc:
            logger.error("Optimizer covariance inversion failed: %s", exc)
            return {symbol: 0.0 for symbol in symbols}

        raw_weights = inv_term.dot(mu)

        # Clip to per-asset bounds
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

        # Final clamp to avoid numerical bleed
        clipped = np.clip(clipped, self.weight_min, self.weight_max)

        weights = {symbol: float(weight) for symbol, weight in zip(symbols, clipped)}
        return weights
