from dataclasses import dataclass
from typing import Dict
import numpy as np
import math


@dataclass
class ImpactParams:
    kappa: float  # quadratic cost coefficient
    delta: float = 0.5  # concavity for power-law impact (0<delta<1)
    decay: float = 0.5  # propagator decay


def estimate_temp_impact(notional: float, params: ImpactParams) -> float:
    """Temporary impact cost using concave power-law."""
    direction = np.sign(notional)
    size = abs(notional)
    cost = params.kappa * (size ** params.delta)
    return float(direction * cost)


def aggregate_impact(target_weights: Dict[str, float], prev_weights: Dict[str, float], prices: Dict[str, float], nav: float, kappa_overrides: Dict[str, float], default_kappa: float, delta: float = 0.5) -> float:
    total_cost = 0.0
    for sym, tgt in target_weights.items():
        prev = prev_weights.get(sym, 0.0)
        delta_w = tgt - prev
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        notional = delta_w * nav
        kappa = kappa_overrides.get(sym, default_kappa)
        params = ImpactParams(kappa=kappa, delta=delta)
        total_cost += abs(estimate_temp_impact(notional, params))
    return total_cost


def propagator_cost(trade_sizes: Dict[str, float], prices: Dict[str, float], nav: float, kappa_overrides: Dict[str, float], default_kappa: float, delta: float, decay: float) -> float:
    """Estimate impact cost with simple exponential decay propagator across sequential trades."""
    total_cost = 0.0
    for idx, (sym, tgt) in enumerate(trade_sizes.items()):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        kappa = kappa_overrides.get(sym, default_kappa)
        params = ImpactParams(kappa=kappa, delta=delta, decay=decay)
        decay_factor = math.exp(-decay * idx)
        total_cost += abs(estimate_temp_impact(tgt * nav, params)) * decay_factor
    return total_cost
