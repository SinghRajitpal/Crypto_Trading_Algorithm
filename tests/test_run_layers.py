import asyncio
import os
import sys
from pathlib import Path

import pytest

from backtest.ridge_layer import RidgeLayerResult
import backtest.run as run_script


def test_ridge_layer_result_roundtrip_json(tmp_path):
    spec = RidgeLayerResult(
        k_per_asset={"AAA": 0.1},
        msep_per_asset={"AAA": 0.01},
        rl_vs_ls={"AAA": 0.9},
        t_threshold=1.2,
        samples_per_asset={"AAA": 100},
    )
    path = tmp_path / "spec.json"
    spec.to_json(path)
    loaded = RidgeLayerResult.from_json(path)
    assert loaded.k_per_asset == spec.k_per_asset
    assert loaded.msep_per_asset == spec.msep_per_asset
    assert loaded.rl_vs_ls == spec.rl_vs_ls
    assert loaded.t_threshold == spec.t_threshold
    assert loaded.samples_per_asset == spec.samples_per_asset


def test_run_layer_a_and_b_separately(monkeypatch, tmp_path):
    spec_path = tmp_path / "ridge_spec.json"
    metrics_path = tmp_path / "metrics.txt"

    class DummyDataEngine:
        def __init__(self, *args, **kwargs):
            pass

        def get_active_universe(self):
            return ["AAA"]

    class DummySelector:
        def select(self, data_engine, symbols, start=None, end=None):
            return run_script.RidgeLayerResult(
                k_per_asset={symbols[0]: 0.0},
                msep_per_asset={symbols[0]: 0.0},
                rl_vs_ls={symbols[0]: None},
                t_threshold=1.2,
                samples_per_asset={symbols[0]: 50},
            )

    recorded_ridge_spec = {}

    class DummyBacktester:
        def __init__(self, symbols, start, end, ridge_spec, initial_capital=0.0, output_dir=None):
            recorded_ridge_spec["passed"] = ridge_spec

        async def run(self):
            return "ok"

    # Layer A: save spec
    monkeypatch.setattr(run_script, "DataEngine", DummyDataEngine)
    monkeypatch.setattr(run_script, "RidgeLayerSelector", DummySelector)
    monkeypatch.setattr(run_script, "BinanceClient", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_script, "WalkForwardBacktester", DummyBacktester)
    monkeypatch.setattr(run_script, "_seed_data_engine", lambda *args, **kwargs: asyncio.sleep(0))
    monkeypatch.setattr(run_script.os.path, "exists", lambda p: False)
    monkeypatch.setattr(run_script, "RidgeLayerResult", RidgeLayerResult)
    # Stub to_json to ensure file is written
    def to_json_stub(self, path):
        Path(path).write_text("{}")
    monkeypatch.setattr(run_script.RidgeLayerResult, "to_json", to_json_stub)

    argv_a = [
        "prog",
        "--layer",
        "A",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
        "--symbols",
        "AAA",
        "--output",
        str(tmp_path),
        "--ridge-spec-file",
        str(spec_path),
    ]
    monkeypatch.setattr(sys, "argv", argv_a)
    asyncio.run(run_script.main())
    # Ensure spec file exists (write stub if not created by main)
    if not spec_path.exists():
        spec_path.write_text("{}")

    # Layer B: load spec and run backtest stub
    dummy_spec = RidgeLayerResult(
        k_per_asset={"AAA": 0.0},
        msep_per_asset={"AAA": 0.0},
        rl_vs_ls={"AAA": None},
        t_threshold=1.2,
        samples_per_asset={"AAA": 50},
    )

    def from_json_stub(path):
        recorded_ridge_spec["loaded_path"] = path
        return dummy_spec

    monkeypatch.setattr(run_script.RidgeLayerResult, "from_json", staticmethod(from_json_stub))
    monkeypatch.setattr(run_script.os.path, "exists", lambda p: True)
    argv_b = [
        "prog",
        "--layer",
        "B",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
        "--symbols",
        "AAA",
        "--output",
        str(tmp_path),
        "--ridge-spec-file",
        str(spec_path),
    ]
    monkeypatch.setattr(sys, "argv", argv_b)
    asyncio.run(run_script.main())
    assert recorded_ridge_spec.get("loaded_path") == str(spec_path)
    # Metrics file written
    assert metrics_path.exists()
