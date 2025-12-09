import numpy as np
import pytest

import config
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


def _manual_garch(returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    var = float(np.var(returns, ddof=1) or 1e-6)
    for r in returns:
        var = omega + alpha * (r ** 2) + beta * var
    return max(min(var, 1e2), 1e-10)


def test_risk_model_cov_and_symbol_alignment():
    symbols = ["BTC", "ETH", "SOL"]
    rng = np.random.default_rng(42)
    returns = rng.normal(scale=0.01, size=(300, 3))
    rm = RiskModel()
    rm.update(symbols, returns)

    # Expect pure sample covariance (no shrinkage)
    window = min(config.RISK_COV_WINDOW, returns.shape[0])
    expected = np.cov(returns[-window:].T, ddof=1)

    # get_covariance must reorder rows/cols to requested symbol order
    requested = list(reversed(symbols))
    idx = [symbols.index(sym) for sym in requested]
    expected_reordered = expected[np.ix_(idx, idx)]
    cov = rm.get_covariance(requested)

    assert cov.shape == expected_reordered.shape
    assert np.allclose(cov, expected_reordered)
    assert rm.ready()


def test_risk_model_variance_selection_and_diagnostics_with_garch():
    symbols = ["A"]
    # Heteroskedastic series to exercise GARCH path
    series = np.array([0.001, 0.002, -0.003, 0.01, -0.012, 0.008, -0.006, 0.004, 0.0, -0.002])
    returns = series.reshape(-1, 1)
    rm = RiskModel(use_garch=True)
    rm.update(symbols, returns)

    garch_expected = _manual_garch(
        returns[:, 0],
        config.GARCH_PARAMS["omega"],
        config.GARCH_PARAMS["alpha"],
        config.GARCH_PARAMS["beta"],
    )
    assert rm._select_variance("A") == pytest.approx(garch_expected)
    assert rm.get_volatility("A") > 0

    diag = rm.diagnostics()
    assert diag.get("symbols") == 1
    assert "last_mahalanobis_d2" in diag
    assert "malv_mean" in diag
    # Volatility loss trackers should be populated and finite
    assert diag.get("vol_mse_avg") is not None
    assert diag.get("vol_qlike_avg") is not None
    assert np.isfinite(diag["vol_mse_avg"])
    assert np.isfinite(diag["vol_qlike_avg"])


def test_risk_model_covariance_loss_metrics_positive():
    symbols = ["A", "B", "C"]
    rng = np.random.default_rng(7)
    base = rng.normal(scale=0.02, size=(400, 3))
    # Introduce correlation to stress covariance loss evaluation
    base[:, 1] = base[:, 0] * 0.5 + rng.normal(scale=0.01, size=400)
    base[:, 2] = -base[:, 0] * 0.3 + rng.normal(scale=0.015, size=400)

    rm = RiskModel()
    rm.update(symbols, base)
    metrics = rm.covariance_loss_metrics(symbols, base)

    assert "cov_port_mse" in metrics and metrics["cov_port_mse"] >= 0
    assert "cov_port_qlike" in metrics and np.isfinite(metrics["cov_port_qlike"])


def test_risk_model_handles_near_singular_cov_and_malv_is_finite():
    symbols = ["A", "B"]
    # Perfectly collinear returns → near-singular covariance
    base = np.linspace(-0.01, 0.01, 200)
    returns = np.column_stack([base, base * 2.0])
    rm = RiskModel()
    rm.update(symbols, returns)
    diag = rm.diagnostics()
    # Mahalanobis diagnostics should be present and finite despite pinv fallback
    assert "last_mahalanobis_d2" in diag
    assert np.isfinite(diag["last_mahalanobis_d2"])
    # Covariance still returned (pinv used)
    cov = rm.get_covariance(symbols)
    assert cov.shape == (2, 2)
    assert np.isfinite(cov).all()


def test_risk_model_vol_spike_records_losses():
    symbols = ["A"]
    # Mostly quiet, one spike to test vol loss recording
    quiet = np.zeros(50)
    spike = np.array([0.0, 0.0, 0.1, 0.0, 0.0])
    series = np.concatenate([quiet, spike])
    returns = series.reshape(-1, 1)
    rm = RiskModel()
    rm.update(symbols, returns)
    diag = rm.diagnostics()
    assert "vol_mse_avg" in diag and np.isfinite(diag["vol_mse_avg"])
    assert "vol_qlike_avg" in diag and np.isfinite(diag["vol_qlike_avg"])


def test_risk_model_mismatched_dimensions_raises():
    rm = RiskModel()
    bad_returns = np.random.normal(size=(10, 1))
    with pytest.raises(ValueError):
        rm.update(["A", "B"], bad_returns)


def test_risk_model_empty_input_not_ready():
    rm = RiskModel()
    rm.update([], np.empty((0, 0)))
    assert not rm.ready()
    assert rm.get_covariance([]) is None
