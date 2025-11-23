from execution.impact_model import aggregate_impact, propagator_cost
from execution.kelly import apply_fractional_kelly
import config


def test_impact_monotonic():
    prices = {"A": 100}
    target = {"A": 0.1}
    prev = {"A": 0.0}
    nav = 1000
    cost_small = aggregate_impact(target, prev, prices, nav, {}, 0.1, delta=0.5)
    target["A"] = 0.2
    cost_large = aggregate_impact(target, prev, prices, nav, {}, 0.1, delta=0.5)
    assert cost_large >= cost_small
    prop_cost = propagator_cost({"A": 0.2}, prices, nav, {}, 0.1, delta=0.5, decay=0.5)
    assert prop_cost >= 0


def test_kelly_dampening():
    base_w = {"A": 0.1}
    # High vol should dampen lambda
    w_high_vol = apply_fractional_kelly(base_w, f_star=1.0, drawdown=0.0, vol=0.2)
    w_low_vol = apply_fractional_kelly(base_w, f_star=1.0, drawdown=0.0, vol=0.01)
    assert abs(w_high_vol["A"]) <= abs(w_low_vol["A"])
