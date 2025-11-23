from typing import Dict, Tuple, List, Optional

import numpy as np

import config


def compute_kelly_scaler(
    symbols: List[str],
    weights: Dict[str, float],
    mu: Dict[str, float],
    covariance: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Kelly leverage factor for a portfolio and return (f*, mu_p, var_p).

    Symbols define the alignment for weights/mu relative to covariance rows/cols.
    """
    if not weights or covariance.size == 0 or not symbols:
        return 0.0, 0.0, 0.0

    w_vec = np.array([weights.get(s, 0.0) for s in symbols], dtype=float)
    mu_vec = np.array([mu.get(s, 0.0) for s in symbols], dtype=float)

    mu_p = float(mu_vec @ w_vec)
    var_p = float(w_vec @ covariance @ w_vec)
    if var_p <= 0:
        return 0.0, mu_p, var_p

    f_star = mu_p / var_p
    # Cap leverage to avoid over-Kelly
    f_star = max(0.0, min(f_star, config.KELLY_MAX_LEVERAGE))
    return f_star, mu_p, var_p


def apply_fractional_kelly(
    base_weights: Dict[str, float],
    f_star: float,
    drawdown: float,
    lam_base: float = config.KELLY_FRACTION_BASE,
    thresholds=None,
    lambdas=None,
    vol: Optional[float] = None,
) -> Dict[str, float]:
    """Scale weights by fractional Kelly with drawdown-aware lambda."""
    thresholds = thresholds or config.DRAWDOWN_THRESHOLDS
    lambdas = lambdas or config.DRAWDOWN_LAMBDAS

    lam = lam_base
    for th, lval in zip(thresholds, lambdas):
        if drawdown >= th:
            lam = min(lam, lval)
    if vol is not None and vol > 0:
        # Damp scaling when realized vol is elevated
        lam = lam / (1.0 + max(0.0, vol - config.KELLY_VOL_THRESHOLD))
    scaler = lam * max(f_star, 0.0)
    return {s: w * scaler for s, w in base_weights.items()}
