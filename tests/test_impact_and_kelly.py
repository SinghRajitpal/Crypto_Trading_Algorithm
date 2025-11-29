from execution.kelly import (
    apply_fractional_kelly,
    fractional_kelly_scaler,
    max_constraint_scaler,
)
from execution.execution_engine import ProductionExecutionEngine


def test_kelly_dampening():
    base_w = {"A": 0.1}
    # High vol should dampen lambda
    w_high_vol = apply_fractional_kelly(base_w, f_star=1.0, drawdown=0.0, vol=0.2)
    w_low_vol = apply_fractional_kelly(base_w, f_star=1.0, drawdown=0.0, vol=0.01)
    assert abs(w_high_vol["A"]) <= abs(w_low_vol["A"])


def test_constraint_scaler_caps_kelly():
    # Direction already at per-asset bound; constraint scaler should cap leverage
    direction = {"A": 0.3, "B": -0.2}
    s_kelly = fractional_kelly_scaler(
        f_star=1.5, drawdown=0.0, lam_base=1.0, thresholds=[], lambdas=[], vol=0.0
    )
    s_constraints = max_constraint_scaler(
        direction, weight_min=-0.3, weight_max=0.3, max_gross=1.2, max_net=0.25
    )
    s_final = min(s_kelly, s_constraints)
    scaled = {k: v * s_final for k, v in direction.items()}
    assert s_kelly > 1.0
    assert s_constraints == 1.0  # per-name cap binds
    assert all(abs(w) <= 0.3 + 1e-9 for w in scaled.values())
    assert sum(abs(w) for w in scaled.values()) <= 1.2 + 1e-9


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
