import numpy as np

from execution.risk_model import RiskModel


def test_risk_model_covariance_shapes_and_losses():
    symbols = ["A", "B"]
    rng = np.random.default_rng(0)
    returns = rng.normal(scale=0.01, size=(300, 2))
    rm = RiskModel()
    rm.update(symbols, returns)
    cov = rm.get_covariance(symbols)
    assert cov.shape == (2, 2)
    diag = rm.diagnostics()
    assert diag.get("symbols") == 2
    loss = rm.covariance_loss_metrics(symbols, returns)
    assert "cov_port_mse" in loss or "cov_port_qlike" in loss
