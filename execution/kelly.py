from typing import Dict, Tuple, List, Optional

import numpy as np

import config


def fractional_kelly_scaler(
    f_star: float,
    drawdown: float,
    lam_base: float = config.KELLY_FRACTION_BASE,
    thresholds=None,
    lambdas=None,
    vol: Optional[float] = None,
) -> float:
    """Return the fractional Kelly leverage scaler given drawdown/vol regime."""
    thresholds = thresholds or config.DRAWDOWN_THRESHOLDS
    lambdas = lambdas or config.DRAWDOWN_LAMBDAS

    lam = lam_base
    for th, lval in zip(thresholds, lambdas):
        if drawdown >= th:
            lam = min(lam, lval)
    if vol is not None and vol > 0:
        # Damp scaling when realized vol is elevated
        lam = lam / (1.0 + max(0.0, vol - config.KELLY_VOL_THRESHOLD))
    return lam * max(f_star, 0.0)


def max_constraint_scaler(
    weights: Dict[str, float],
    weight_min: float,
    weight_max: float,
    max_gross: float,
    max_net: Optional[float] = None,
) -> float:
    """Largest scalar that keeps weights within per-asset, gross (and optional net) caps."""
    if not weights:
        return 0.0

    per_name_limits = []
    for w in weights.values():
        if w == 0:
            continue
        if w > 0 and weight_max > 0:
            per_name_limits.append(weight_max / w)
        elif w < 0 and weight_min < 0:
            per_name_limits.append(weight_min / w)
    s_name = min(per_name_limits) if per_name_limits else float("inf")

    gross = sum(abs(w) for w in weights.values())
    s_gross = max_gross / gross if gross > 0 else float("inf")

    s_net = float("inf")
    if max_net is not None and max_net > 0:
        net = sum(weights.values())
        if net != 0:
            s_net = max_net / abs(net)

    candidates = [s for s in (s_name, s_gross, s_net) if s > 0]
    if not candidates:
        return 0.0
    return float(min(candidates))


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
    scaler = fractional_kelly_scaler(
        f_star,
        drawdown,
        lam_base=lam_base,
        thresholds=thresholds,
        lambdas=lambdas,
        vol=vol,
    )
    return {s: w * scaler for s, w in base_weights.items()}
