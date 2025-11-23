import numpy as np

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
