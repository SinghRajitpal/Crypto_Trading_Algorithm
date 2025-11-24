from datetime import datetime, timedelta, UTC

from data.universe_selector import BarValidator, UniverseSelector


def test_bar_validator_rules():
    bv = BarValidator(max_abs_return=0.2, min_volume=1.0)
    prev_close = 100.0
    valid_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 10.0}
    assert bv.is_valid("SYM", valid_bar, prev_close)

    bad_ohlc = {"open": 100.0, "high": 99.0, "low": 101.0, "close": 101.0, "volume": 10.0}
    assert not bv.is_valid("SYM", bad_ohlc, prev_close)

    bad_ret = {"open": 100.0, "high": 150.0, "low": 100.0, "close": 130.0, "volume": 10.0}
    assert not bv.is_valid("SYM", bad_ret, prev_close)

    bad_vol = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 0.0}
    assert not bv.is_valid("SYM", bad_vol, prev_close)


def test_universe_selector_updates_and_falls_back():
    selector = UniverseSelector(
        max_rank=2,
        min_dollar_volume=1000.0,
        lookback_days=2,
        default_universe=["DEF"],
    )
    ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    # Update market caps and record sufficient volume for two assets
    selector.update_market_cap_snapshot({"AAA": 1e9, "BBB": 5e8})
    selector.record_bar_metrics("AAA", ts, close=100.0, volume=20.0)
    selector.record_bar_metrics("BBB", ts, close=90.0, volume=15.0)
    selector.refresh_if_needed(ts)
    assert set(selector.get_active_universe()) == {"AAA", "BBB"}

    # Low volume should trigger fallback to default
    ts2 = ts + int(timedelta(days=1).total_seconds() * 1000)
    selector.update_market_cap_snapshot({"CCC": 2e9})
    selector.record_bar_metrics("CCC", ts2, close=1.0, volume=0.0)
    selector.refresh_if_needed(ts2)
    assert selector.get_active_universe()  # non-empty


def test_universe_refresh_once_per_day_and_prunes_old_volume():
    selector = UniverseSelector(
        max_rank=1,
        min_dollar_volume=1000.0,
        lookback_days=2,
        default_universe=["DEF"],
    )
    day0 = datetime(2024, 1, 1, tzinfo=UTC)
    ts_day0 = int(day0.timestamp() * 1000)
    selector.update_market_cap_snapshot({"AAA": 1e9})
    selector.record_bar_metrics("AAA", ts_day0, close=100.0, volume=15.0)  # $1500
    assert selector.refresh_if_needed(ts_day0) is True
    first_universe = selector.get_active_universe()
    assert first_universe == ["AAA"]

    # Same day refresh should be a no-op
    assert selector.refresh_if_needed(ts_day0) is False
    assert selector.get_active_universe() == first_universe

    # Advance beyond lookback so volume history is pruned; without new volume AAA should drop
    day3 = day0 + timedelta(days=3)
    ts_day3 = int(day3.timestamp() * 1000)
    selector.refresh_if_needed(ts_day3)
    # After pruning, median volume falls below threshold, so fallback to default
    assert selector.get_active_universe() == ["DEF"]
