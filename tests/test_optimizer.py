import numpy as np
import pytest

from execution.optimizer import MeanVarianceOptimizer


def test_optimizer_bounds_and_turnover():
    symbols = ["A", "B"]
    mu = {"A": 0.01, "B": 0.02}
    cov = np.eye(2) * 0.01
    opt = MeanVarianceOptimizer(w_min=-0.1, w_max=0.1, max_net=0.05, max_gross=0.2, turnover_lambda=0.5)
    prev = {"A": 0.0, "B": 0.0}
    weights = opt.optimize(mu, cov, symbols, prev_weights=prev)
    assert all(-0.1 <= w <= 0.1 for w in weights.values())
    assert abs(sum(weights.values())) <= 0.05 + 1e-6


def test_gross_and_net_exposure_scaling():
    symbols = ["A", "B", "C"]
    mu = {"A": 0.05, "B": 0.04, "C": 0.03}
    cov = np.eye(3) * 0.01
    opt = MeanVarianceOptimizer(w_min=-0.5, w_max=0.5, max_net=0.1, max_gross=0.6)
    weights = opt.optimize(mu, cov, symbols, prev_weights={"A": 0.0, "B": 0.0, "C": 0.0})
    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    assert gross <= 0.6 + 1e-6
    assert abs(net) <= 0.1 + 1e-6


def test_turnover_penalty_keeps_weights_close_to_prev():
    symbols = ["A", "B"]
    mu = {"A": 0.05, "B": 0.05}
    cov = np.eye(2) * 0.01
    prev = {"A": 0.2, "B": -0.2}

    opt_low = MeanVarianceOptimizer(turnover_lambda=0.0)
    opt_high = MeanVarianceOptimizer(turnover_lambda=1.0)
    w_low = opt_low.optimize(mu, cov, symbols, prev_weights=prev)
    w_high = opt_high.optimize(mu, cov, symbols, prev_weights=prev)

    dist_low = sum(abs(w_low[s] - prev[s]) for s in symbols)
    dist_high = sum(abs(w_high[s] - prev[s]) for s in symbols)
    assert dist_high < dist_low  # higher penalty should stay closer to prev weights


def test_slippage_overrides_penalize_specific_assets():
    symbols = ["A", "B"]
    mu = {"A": 0.04, "B": 0.04}
    cov = np.eye(2) * 0.01
    prev = {"A": 0.1, "B": -0.1}
    base = MeanVarianceOptimizer(turnover_lambda=1.0, slippage_bps_default=0.0)
    with_slip = MeanVarianceOptimizer(
        turnover_lambda=1.0,
        slippage_bps_default=0.0,
        slippage_bps_overrides={"A": 50.0, "B": 0.0},
    )
    w_base = base.optimize(mu, cov, symbols, prev_weights=prev)
    w_slip = with_slip.optimize(mu, cov, symbols, prev_weights=prev)
    # High slippage on A should reduce movement vs baseline; B should be similar or unchanged
    delta_a_base = abs(w_base["A"] - prev["A"])
    delta_a_slip = abs(w_slip["A"] - prev["A"])
    delta_b_base = abs(w_base["B"] - prev["B"])
    delta_b_slip = abs(w_slip["B"] - prev["B"])
    assert delta_a_slip <= delta_a_base + 1e-9
    assert delta_b_slip == pytest.approx(delta_b_base)


def test_turnover_horizon_scaling():
    symbols = ["A"]
    mu = {"A": 0.02}
    cov = np.eye(1) * 0.01
    prev = {"A": 0.3}
    opt_fast = MeanVarianceOptimizer(turnover_lambda=0.5, turnover_horizon=1)
    opt_slow = MeanVarianceOptimizer(turnover_lambda=0.5, turnover_horizon=10)
    w_fast = opt_fast.optimize(mu, cov, symbols, prev_weights=prev)["A"]
    w_slow = opt_slow.optimize(mu, cov, symbols, prev_weights=prev)["A"]
    # Longer horizon dampens the pull toward prev weight
    assert abs(w_slow - prev["A"]) >= abs(w_fast - prev["A"])


def test_covariance_alignment_error():
    opt = MeanVarianceOptimizer()
    with pytest.raises(ValueError):
        opt.optimize({"A": 0.01}, np.eye(2), ["A"])


def test_positive_slippage_default_reduces_rebalance_size():
    symbols = ["A", "B"]
    mu = {"A": 0.04, "B": 0.03}
    cov = np.eye(2) * 0.01
    prev = {"A": 0.1, "B": -0.1}
    opt_zero = MeanVarianceOptimizer(turnover_lambda=0.5, slippage_bps_default=0.0)
    opt_high = MeanVarianceOptimizer(turnover_lambda=0.5, slippage_bps_default=50.0)
    w_zero = opt_zero.optimize(mu, cov, symbols, prev_weights=prev)
    w_high = opt_high.optimize(mu, cov, symbols, prev_weights=prev)
    # With high kappa_default, movement from prev weights should be smaller
    move_zero = sum(abs(w_zero[s] - prev[s]) for s in symbols)
    move_high = sum(abs(w_high[s] - prev[s]) for s in symbols)
    assert move_high <= move_zero + 1e-9
