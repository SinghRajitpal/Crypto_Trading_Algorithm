import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, UTC, timedelta

import config
from backtest import ridge_layer
from backtest.ridge_layer import RidgeLayerSelector
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster


class DummyDataEngine:
    def __init__(self, X, y, symbols):
        self._X = X
        self._y = y
        self._symbols = symbols

    def get_active_universe(self):
        return self._symbols

    def get_feature_matrix(self, symbol, window=None, exclude_outliers=True):
        if symbol not in self._symbols:
            return np.empty((0, 0)), np.empty((0,)), [], []
        return self._X, self._y, list(range(len(self._y))), [f"f{i}" for i in range(self._X.shape[1])]


def test_ridge_layer_selects_low_penalty_for_clean_linear():
    n = max(400, config.REGRESSION_MIN_TRAIN + 50)
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = 0.5 * x[:, 0] + 0.1
    de = DummyDataEngine(x, y, symbols=["A"])
    selector = RidgeLayerSelector(k_grid=[0.0, 10.0], train_min=20, val_len=10, t_threshold=0.0)
    res = selector.select(de)
    assert res.k_per_asset.get("A") == 0.0
    assert res.samples_per_asset.get("A") == n


def test_ridge_layer_intercept_not_penalized():
    n = 30
    X = np.empty((n, 0))
    X_int = RidgeLayerSelector._with_intercept(X)
    XtX = X_int.T @ X_int
    y = np.ones(n) * 0.5
    Xty = X_int.T @ y
    penalty = RidgeLayerSelector._penalty_matrix(XtX.shape[0], 1e6)
    beta = RidgeLayerSelector._ridge_beta(XtX, Xty, penalty)
    # Intercept should stay near mean(y) even with huge k
    assert beta.shape[0] == 1
    assert beta[0] == pytest.approx(0.5, rel=1e-3)


def test_ridge_layer_skips_insufficient_history():
    X = np.random.normal(size=(10, 2))
    y = np.random.normal(size=10)
    de = DummyDataEngine(X, y, symbols=["A"])
    selector = RidgeLayerSelector(k_grid=[0.1, 1.0], train_min=50, val_len=10, t_threshold=0.0)
    res = selector.select(de)
    assert "A" not in res.k_per_asset


def test_load_parquet_slice_filters_window(tmp_path):
    pytest.importorskip("pyarrow")
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(10, dtype=float),
            "high": np.arange(10, dtype=float) + 0.5,
            "low": np.arange(10, dtype=float) - 0.5,
            "close": np.arange(10, dtype=float) + 1.0,
            "volume": np.ones(10, dtype=float),
        },
        index=ts,
    )
    path = tmp_path / "TEST-1h.parquet"
    df.to_parquet(path)

    start = ts[3]
    end = ts[6]
    sliced = ridge_layer._load_parquet_slice(path, start, end, ["timestamp", "open", "high", "low", "close", "volume"])
    assert sliced is not None
    assert len(sliced) == 4  # inclusive bounds
    assert sliced.index[0] == start
    assert sliced.index[-1] == end
    assert "open" in sliced.columns and "close" in sliced.columns


def test_load_parquet_window_caps_and_filters(tmp_path):
    pytest.importorskip("pyarrow")
    ts = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(12, dtype=float),
            "high": np.arange(12, dtype=float) + 0.5,
            "low": np.arange(12, dtype=float) - 0.5,
            "close": np.arange(12, dtype=float) + 1.0,
            "volume": np.ones(12, dtype=float),
        },
        index=ts,
    )
    path = tmp_path / "TEST-1h.parquet"
    df.to_parquet(path)

    start = ts[3]
    end = ts[-2]
    bars, stats = ridge_layer._load_parquet_window(
        path, start, end, ["timestamp", "open", "high", "low", "close", "volume"], max_rows=5
    )
    assert stats["rows_read"] >= 0
    assert len(bars) == 5  # capped to last 5 within window
    assert bars[0]["timestamp"] == int(ts[-6].timestamp() * 1000)  # tail of the window


def test_build_engine_trims_to_lookback_and_cap(tmp_path):
    pytest.importorskip("pyarrow")
    data_dir = Path("data/cache")
    data_dir.mkdir(parents=True, exist_ok=True)
    sym = "UNITTESTXX"
    timeframe = "1h"
    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(48, dtype=float),
            "high": np.arange(48, dtype=float) + 0.5,
            "low": np.arange(48, dtype=float) - 0.5,
            "close": np.arange(48, dtype=float) + 1.0,
            "volume": np.ones(48, dtype=float),
        },
        index=ts,
    )
    path = data_dir / f"{sym}-{timeframe}.parquet"
    df.to_parquet(path)

    try:
        selector = RidgeLayerSelector(train_min=5, val_len=2, embargo=0, fast_mode=True)
        lookback_days = 1
        end_ts = ts[-1]
        engine = selector._build_engine_for_tf(
            timeframe,
            [sym],
            base_client=object(),
            w_grid=[4],
            lookback_days=lookback_days,
            end_ts=end_ts,
        )
        assert engine is not None
        price_hist = engine.return_manager.price_history[sym]
        cutoff = end_ts - timedelta(days=lookback_days)
        expected_len = len(df[df.index >= cutoff])
        assert len(price_hist) == expected_len
    finally:
        if path.exists():
            path.unlink()


def test_hat_diag_matches_naive():
    X = np.random.normal(size=(8, 3)).astype(np.float32)
    inv = np.linalg.pinv(X.T @ X + np.eye(X.shape[1], dtype=X.dtype))
    diag_fast = RidgeRegressionForecaster._hat_diag(X, inv)
    H = X @ inv @ X.T
    diag_naive = np.diag(H)
    np.testing.assert_allclose(diag_fast, diag_naive, rtol=1e-5, atol=1e-6)


def test_estimate_buffer_respects_caps(monkeypatch):
    selector = RidgeLayerSelector(train_min=10, val_len=5, embargo=0)
    monkeypatch.setattr(config, "LAYERA_REGRESSION_MAX_BARS", 1_000)
    monkeypatch.setattr(config, "REGRESSION_MAX_BARS", None)
    monkeypatch.setattr(config, "LAYERA_MAX_BARS_BY_TF", {"1m": 800})
    estimate = selector._estimate_buffer("1m", lookback_days=365, w_grid=[5_000])
    min_required = selector.train_min + selector.val_len + selector.embargo + 10
    assert min_required <= estimate <= 800
