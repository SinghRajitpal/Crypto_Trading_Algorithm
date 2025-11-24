import json
import os
import tempfile

import pytest

from execution.alerts import check_thresholds
from utils.monitoring_store import MonitoringStore


def test_check_thresholds_warns_on_high_metrics(caplog, monkeypatch):
    caplog.set_level("WARNING")
    result = {
        "risk_cov_loss": {"cov_port_mse": 0.2},
        "risk_diag": {"vol_mse_avg": 0.1},
    }
    # Lower thresholds to trigger warnings
    import config

    monkeypatch.setattr(config, "MONITOR_COV_MSE_WARN", 0.05)
    monkeypatch.setattr(config, "MONITOR_VOL_MSE_WARN", 0.05)
    check_thresholds(result)
    msgs = [rec.message for rec in caplog.records]
    assert any("Covariance MSE exceeded" in m for m in msgs)
    assert any("Vol MSE exceeded" in m for m in msgs)


def test_monitoring_store_appends_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mon.jsonl")
        store = MonitoringStore(path)
        rec1 = {"a": 1}
        rec2 = {"b": 2}
        store.append(rec1)
        store.append(rec2)
        with open(path) as f:
            lines = f.read().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == rec1
        assert json.loads(lines[1]) == rec2
