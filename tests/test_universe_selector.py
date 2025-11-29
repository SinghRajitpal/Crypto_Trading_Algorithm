from datetime import datetime, UTC

import numpy as np

from data.universe_selector import UniverseSelector


def test_universe_selector_defaults_and_refresh():
    default = ["BTCUSDT"]
    selector = UniverseSelector(
        max_rank=2,
        min_dollar_volume=0.0,
        lookback_days=30,
        default_universe=default,
    )
    # Defaults are returned when no market caps/volumes
    assert selector.get_active_universe() == default

    # Update market caps to include a new symbol with higher rank
    selector.update_market_cap_snapshot({"BTCUSDT": 1e12, "ETHUSDT": 5e11, "XRPUSDT": 2e11})
    ts_ms = int(datetime.now(UTC).timestamp() * 1000)
    refreshed = selector.refresh_if_needed(ts_ms)
    # Universe should now be recomputed (max_rank=2) and differ from default
    assert refreshed is True
    assert selector.get_active_universe() == ["BTCUSDT", "ETHUSDT"]


def test_universe_selector_records_volume_and_updates():
    default = ["BTCUSDT"]
    selector = UniverseSelector(
        max_rank=3,
        min_dollar_volume=1.0,
        lookback_days=30,
        default_universe=default,
    )
    ts_ms = int(datetime.now(UTC).timestamp() * 1000)
    # Record sufficient volume for XRP to be eligible
    selector.update_market_cap_snapshot({"BTCUSDT": 1e12, "ETHUSDT": 5e11, "XRPUSDT": 2e11})
    selector.record_bar_metrics("BTCUSDT", ts_ms, close=100.0, volume=1000.0)
    selector.record_bar_metrics("XRPUSDT", ts_ms, close=100.0, volume=1000.0)
    refreshed = selector.refresh_if_needed(ts_ms)
    assert refreshed is True
    # Top 3 by rank with sufficient volume should be BTC and XRP (ETH has no volume, so is skipped)
    assert selector.get_active_universe() == ["BTCUSDT", "XRPUSDT"]
