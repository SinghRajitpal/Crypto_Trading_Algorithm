import asyncio
import sys
import types
import time

import numpy as np

# Stub binance modules to avoid dependency issues during import
if "binance" not in sys.modules:
    binance_mod = types.ModuleType("binance")

    class _AsyncClient:
        @classmethod
        async def create(cls, *args, **kwargs):
            return cls()

        async def close_connection(self):
            return None

    class _BinanceSocketManager:
        def __init__(self, *args, **kwargs):
            pass

    enums_mod = types.ModuleType("binance.enums")
    enums_mod.__all__ = []
    binance_mod.AsyncClient = _AsyncClient
    binance_mod.BinanceSocketManager = _BinanceSocketManager
    sys.modules["binance"] = binance_mod
    sys.modules["binance.enums"] = enums_mod

from data.data_fetcher import DataFetcher


class StubBinanceClient:
    def __init__(self, klines):
        class Client:
            def __init__(self, rows):
                self._rows = rows

            async def futures_klines(self, symbol, interval, limit):
                # ignore interval/limit, return provided rows
                return self._rows

        self.client = Client(klines)

    @staticmethod
    def interval_to_milliseconds(interval):
        return 300000 if interval == "5m" else 60000

    @staticmethod
    def _format_symbol(symbol):
        return symbol


def run(coro):
    return asyncio.run(coro)


def test_load_history_uses_only_closed_candles(monkeypatch):
    tf = "5m"
    tf_ms = 300000
    # Three candles, last one should be excluded as "current" bar
    klines = [
        [0, "1", "2", "0.5", "1.5", "10", 0 + tf_ms - 1],
        [tf_ms, "1", "2", "0.5", "1.6", "11", tf_ms + tf_ms - 1],
        [tf_ms * 2, "1", "2", "0.5", "1.7", "12", tf_ms * 3],  # current forming bar
    ]
    stub = StubBinanceClient(klines)
    # Fake current time halfway into third bar
    monkeypatch.setattr(time, "time", lambda: (tf_ms * 2.5) / 1000)

    df = DataFetcher(binance_client=stub, max_candles=5, symbol_timeframes=[("AAA", tf)])
    captured = []

    async def record(symbol, timeframe, candle):
        captured.append((symbol, timeframe, candle))

    df.data_processor.update_tracked_candles = record
    run(df._load_history("AAA", tf))

    # Only first two closed bars ingested
    assert len(captured) == 2
    opens = [c[2][0] for c in captured]
    assert opens == [0, tf_ms]
