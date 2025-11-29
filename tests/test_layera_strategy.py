import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import config
from algorithm.forecast.layera_loader import load_manifest
from algorithm.strategies.layera_forecast import LayerAForecastStrategy


class FakeForecaster:
    def __init__(self, mu=0.1):
        self.mu = mu

    def predict_log_return(self, window: np.ndarray) -> float:
        # Return deterministic log-return; window shape should match lookback x 2
        assert window.ndim == 2
        return float(self.mu)


class StubDataEngine:
    def __init__(self, lookback: int, symbols):
        self.return_manager = SimpleNamespace(
            log_return_history={s: [] for s in symbols},
            volume_history={s: [] for s in symbols},
        )
        self._missing = {s: [] for s in symbols}
        self._candles = {s: None for s in symbols}
        self._universe = symbols

    def process_all_latest_bars(self, timeframe):
        return

    def get_active_universe(self):
        return self._universe

    def get_missing_bars(self, symbol, timeframe):
        return self._missing.get(symbol, [])

    def get_latest_candle(self, symbol, timeframe):
        return self._candles.get(symbol)


@pytest.fixture
def manifest_file(tmp_path):
    # Create dummy artifact dir with a root model file to satisfy resolver
    artefact = tmp_path / "bnbartefact"
    artefact.mkdir(parents=True, exist_ok=True)
    (artefact / "model.pt").write_text("")
    content = [
        {
            "symbol": "BNBUSDT",
            "timeframe": config.GRU_TIMEFRAME,
            "lookback": config.GRU_LOOKBACK,
            "retrain_days": config.GRU_RETRAIN_DAYS,
            "artifact_dir": str(artefact),
        }
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(content))
    return path


def test_load_manifest_uses_loader(monkeypatch, manifest_file):
    calls = {}

    def fake_load(model_dir, device=None):
        calls["dir"] = model_dir
        return FakeForecaster()

    monkeypatch.setattr("algorithm.forecast.layera_loader.GRUForecasterTorch.load", fake_load)
    models = load_manifest(str(manifest_file))
    assert "BNBUSDT" in models
    model = models["BNBUSDT"]
    assert model.lookback == config.GRU_LOOKBACK
    assert calls["dir"] == str(manifest_file.parent / "bnbartefact")


def test_load_manifest_picks_latest_checkpoint(tmp_path, monkeypatch):
    base = tmp_path / "BNBUSDT"
    step0 = base / "checkpoints" / "step_000"
    step2 = base / "checkpoints" / "step_002"
    step0.mkdir(parents=True)
    step2.mkdir(parents=True)
    # Touch model files to satisfy resolver
    (step0 / "model.pt").write_text("")
    (step2 / "model.pt").write_text("")
    manifest = [
        {
            "symbol": "BNBUSDT",
            "timeframe": config.GRU_TIMEFRAME,
            "lookback": config.GRU_LOOKBACK,
            "retrain_days": config.GRU_RETRAIN_DAYS,
            "artifact_dir": str(base),
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    calls = {}

    def fake_load(model_dir, device=None):
        calls["dir"] = model_dir
        return FakeForecaster()

    monkeypatch.setattr("algorithm.forecast.layera_loader.GRUForecasterTorch.load", fake_load)
    models = load_manifest(str(manifest_path))
    assert "BNBUSDT" in models
    # Should choose latest step_002 directory
    assert calls["dir"] == str(step2)
    assert models["BNBUSDT"].artifact_dir == str(step2)


def test_layera_strategy_produces_forecast(monkeypatch, manifest_file):
    mu_log = 0.1

    def fake_load(model_dir, device=None):
        return FakeForecaster(mu=mu_log)

    monkeypatch.setattr("algorithm.forecast.layera_loader.GRUForecasterTorch.load", fake_load)

    lookback = config.GRU_LOOKBACK
    symbols = ["BNBUSDT"]
    engine = StubDataEngine(lookback, symbols)
    ts_base = 1_700_000_000_000
    lr = [(ts_base + i * 1000, 0.001) for i in range(lookback)]
    vol = [(ts_base + i * 1000, 1000.0) for i in range(lookback)]
    engine.return_manager.log_return_history["BNBUSDT"].extend(lr)
    engine.return_manager.volume_history["BNBUSDT"].extend(vol)
    engine._candles["BNBUSDT"] = [ts_base + (lookback - 1) * 1000, 0, 0, 0, 0, 0]

    strategy = LayerAForecastStrategy(
        engine, manifest_path=str(manifest_file), symbols=symbols, timeframe=config.GRU_TIMEFRAME
    )
    forecast = asyncio.run(strategy.calculate_forecast())
    assert forecast is not None
    assert forecast.universe == symbols
    expected_mu = math.exp(mu_log) - 1.0
    assert pytest.approx(forecast.expected_returns["BNBUSDT"], rel=1e-6) == expected_mu
    assert forecast.diagnostics["coverage"] == 1.0
    assert forecast.diagnostics["ready"] == 1
    assert forecast.diagnostics["requested"] == 1
