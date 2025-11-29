import asyncio
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logging_config import get_logger
from data.processor import DataProcessor
from binance_exchange import BinanceClient

logger = get_logger(__name__)


class DataFetcher:
    """Data Fetcher using python-binance websockets for futures klines."""

    def __init__(self, binance_client=None, max_candles=1000, demo=True, symbol_timeframes=None):
        self.binance = binance_client if binance_client else BinanceClient(demo=demo)
        self.symbol_timeframes = symbol_timeframes or config.symbols
        self.data_processor = DataProcessor(max_candles=max_candles)
        self.should_close_client = binance_client is None

    async def watch_ohlcv(self, symbol, timeframe):
        last_printed = None
        candle_count = 0

        logger.info(f"Starting data collection for {symbol} ({timeframe})")

        try:
            await self._load_history(symbol, timeframe)
        except Exception as e:
            logger.warning(f"[{symbol}] Could not fetch historical data: {e}")
            logger.info(f"[{symbol}] Will start with live data only")

        logger.info(f"[{symbol}] Starting REST polling for live data...")
        timeframe_ms = self.binance.interval_to_milliseconds(timeframe)
        sym_fmt = self.binance._format_symbol(symbol)
        while True:
            try:
                klines = await self.binance.client.futures_klines(symbol=sym_fmt, interval=timeframe, limit=2)
                if not klines:
                    await asyncio.sleep(1)
                    continue
                k = klines[-1]
                # Only accept closed candles: futures kline has close_time at index 6
                close_time = int(k[6])
                now = int(time.time() * 1000)
                # Skip if candle still forming
                if now < close_time:
                    await asyncio.sleep(1)
                    continue
                latest = [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                current_candle_count = len(self.data_processor.get_candles(symbol, timeframe))
                is_new_candle = latest[0] != last_printed
                is_initial_collection = current_candle_count < self.data_processor.max_candles
                if is_new_candle and (is_initial_collection or now - latest[0] > timeframe_ms):
                    candle_count += 1
                    if candle_count <= 10 or candle_count % 10 == 0:
                        total_candles = self.data_processor.max_candles
                        current_candles = len(self.data_processor.get_candles(symbol, timeframe))
                        logger.debug(f"[{symbol}] Collected {current_candles}/{total_candles} candles")
                    await self.data_processor.update_tracked_candles(symbol, timeframe, latest)
                    last_printed = latest[0]
                # Sleep until next poll (conservative)
                await asyncio.sleep(timeframe_ms / 1000 * 0.8)
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}/{timeframe}: {e}")
                await asyncio.sleep(1)

    async def _load_history(self, symbol, timeframe):
        logger.info(f"[{symbol}] Fetching historical data (python-binance)...")
        sym_fmt = self.binance._format_symbol(symbol)
        klines = await self.binance.client.futures_klines(symbol=sym_fmt, interval=timeframe, limit=self.data_processor.max_candles + 1)
        complete_candles = []
        current_time = time.time() * 1000
        timeframe_ms = self.binance.interval_to_milliseconds(timeframe)
        current_candle_start = (current_time // timeframe_ms) * timeframe_ms
        for k in klines:
            open_time = int(k[0])
            if open_time < current_candle_start:
                complete_candles.append([k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
        if len(complete_candles) > self.data_processor.max_candles:
            complete_candles = complete_candles[-self.data_processor.max_candles:]
        for candle in complete_candles:
            await self.data_processor.update_tracked_candles(symbol, timeframe, candle)

    def get_candles(self, symbol, timeframe):
        return self.data_processor.get_candles(symbol, timeframe)

    def get_latest_candle(self, symbol, timeframe):
        return self.data_processor.get_latest_candle(symbol, timeframe)

    async def run(self):
        tasks = []
        for symbol, timeframe in self.symbol_timeframes:
            tasks.append(self.watch_ohlcv(symbol, timeframe))
        try:
            await asyncio.gather(*tasks)
        finally:
            if self.should_close_client:
                await self.binance.close()
