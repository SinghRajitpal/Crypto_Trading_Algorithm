import asyncio

from data.processor import DataProcessor


def run(coro):
    return asyncio.run(coro)


def test_aligns_to_grid_and_tracks_missing():
    dp = DataProcessor(max_candles=10)
    symbol = "S"
    tf = "5m"
    # First bar at t=0
    run(dp.update_tracked_candles(symbol, tf, [0, 1, 2, 3, 4, 5]))
    # Skip one grid step, then add next at 10 minutes
    run(dp.update_tracked_candles(symbol, tf, [600000, 1, 2, 3, 4, 5]))
    candles = dp.get_candles(symbol, tf)
    assert len(candles) == 2
    # Missing bar at 5 minutes detected
    missing = dp.get_missing_bars(symbol, tf)
    assert (0 + 300000) in missing


def test_duplicate_and_out_of_order_skipped():
    dp = DataProcessor(max_candles=5)
    symbol = "S"
    tf = "5m"
    ts_aligned = 300000
    run(dp.update_tracked_candles(symbol, tf, [ts_aligned, 1, 2, 3, 4, 5]))
    # Duplicate timestamp should be ignored
    run(dp.update_tracked_candles(symbol, tf, [ts_aligned, 9, 9, 9, 9, 9]))
    # Out-of-order earlier timestamp should be ignored
    run(dp.update_tracked_candles(symbol, tf, [ts_aligned - 300000, 9, 9, 9, 9, 9]))
    candles = dp.get_candles(symbol, tf)
    assert len(candles) == 1
    assert candles[0][0] == ts_aligned


def test_timestamp_alignment_to_grid():
    dp = DataProcessor(max_candles=3)
    symbol = "S"
    tf = "5m"
    # Timestamp not on grid should be floored to grid
    run(dp.update_tracked_candles(symbol, tf, [123456, 1, 2, 3, 4, 5]))
    candles = dp.get_candles(symbol, tf)
    assert len(candles) == 1
    # 5m grid = 300000 ms; floor of 123456 is 0
    assert candles[0][0] == 0
