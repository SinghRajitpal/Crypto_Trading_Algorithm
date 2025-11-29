import asyncio
from types import SimpleNamespace

import numpy as np

from data.data_engine import DataEngine
from data.data_fetcher import DataFetcher


class DummyDataFetcher(DataFetcher):
    """A minimal DataFetcher that returns preset candles."""

    def __init__(self, candles_map):
        # Bypass parent init; we don't need network
        self.data_processor = SimpleNamespace()
        self._candles_map = candles_map
        self.symbol_timeframes = []

    async def _load_history(self, symbol, timeframe):
        # No-op; candles already set
        return

    def get_candles(self, symbol, timeframe):
        return self._candles_map.get((symbol, timeframe), [])

    async def run(self):
        # Not used in tests
        return


def test_backfill_populates_lookback(monkeypatch):
    # Construct synthetic candles with monotonically increasing closes/volumes
    lookback = 5
    candles = []
    base_ts = 1_700_000_000_000
    for i in range(lookback + 2):
        candles.append([base_ts + i * 1_000, 100 + i, 101 + i, 99 + i, 100 + i, 1000 + i])

    engine = DataEngine(binance_client=None, max_candles=100)
    engine.primary_timeframe = "8h"
    # Replace real fetcher with dummy to avoid network/format assumptions
    engine.data_fetcher = DummyDataFetcher({("BTCUSDT", "8h"): candles})
    engine.data_fetcher.symbol_timeframes = [("BTCUSDT", "8h")]
    asyncio.run(engine.backfill_history(timeframe="8h", symbols=["BTCUSDT"]))

    lr_hist = engine.return_manager.log_return_history["BTCUSDT"]
    vol_hist = engine.return_manager.volume_history["BTCUSDT"]
    assert len(lr_hist) >= lookback
    assert len(vol_hist) >= lookback
    # Latest timestamp should match the last candle
    assert lr_hist[-1][0] == candles[-1][0]
