from execution.impact_model import aggregate_impact, propagator_cost
from execution.kelly import apply_fractional_kelly
import config
from execution.execution_engine import ProductionExecutionEngine


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


def test_propagator_decay_reduces_later_trade_costs():
    prices = {"A": 50, "B": 20}
    nav = 10000
    trades = {"A": 0.1, "B": 0.05}  # sequential
    cost_slow_decay = propagator_cost(trades, prices, nav, {}, 0.1, delta=0.5, decay=0.1)
    cost_fast_decay = propagator_cost(trades, prices, nav, {}, 0.1, delta=0.5, decay=2.0)
    assert cost_fast_decay < cost_slow_decay


def test_impact_kappa_overrides_change_costs():
    prices = {"A": 100}
    target = {"A": 0.2}
    prev = {"A": 0.0}
    nav = 1000
    cost_low = aggregate_impact(target, prev, prices, nav, {}, 0.01, delta=0.5)
    cost_high = aggregate_impact(target, prev, prices, nav, {"A": 0.1}, 0.01, delta=0.5)
    assert cost_high > cost_low


def test_realized_slippage_basis_points():
    engine = ProductionExecutionEngine(binance_client=None)
    prices = {"A": 100.0}
    execution = [
        {
            "symbol": "A",
            "side": "buy",
            "quantity": 1.0,
            "response": {"fills": [{"price": 101.0, "qty": 1.0}]},
        }
    ]
    slip_bp = engine._compute_realized_slippage(execution, prices, nav=1000.0)
    assert slip_bp is not None
    assert slip_bp > 0  # buy above mid is positive slippage
