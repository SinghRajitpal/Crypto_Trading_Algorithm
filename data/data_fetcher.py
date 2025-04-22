import asyncio
import time
from datetime import datetime
from collections import deque
import sys
import os

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.processor import DataProcessor
from binance_exchange import BinanceClient


class DataFetcher:
    def __init__(self, binance_client=None, max_candles=1000, testnet=True):
        # Use provided client or create a new one
        self.binance = binance_client if binance_client else BinanceClient(testnet=testnet)
        self.symbol_timeframes = config.symbols
        self.data_processor = DataProcessor(max_candles=max_candles)
        # Track if we need to close the client (only if we created it)
        self.should_close_client = binance_client is None
 
    async def watch_ohlcv(self, symbol, timeframe):
        last_printed = None
        
        while True:
            try:
                # Use BinanceClient for market data
                candles = await self.binance.exchange.watch_ohlcv(symbol, timeframe)
                now = time.time() * 1000
                latest = candles[-1]
                
                # Check if this is a closed candle
                if latest[0] != last_printed and now - latest[0] > self.binance.exchange.parse_timeframe(timeframe) * 1000:
                    # Convert millisecond timestamp to readable date format
                    readable_time = datetime.fromtimestamp(latest[0]/1000).strftime('%H:%M:%S %d/%m/%Y')
                    print(f"{symbol} ({timeframe}) | Time: {readable_time} | Open: {latest[1]} | High: {latest[2]} | Low: {latest[3]} | Close: {latest[4]} | Volume: {latest[5]}")

                    # Update the tracked candles for this specific symbol-timeframe pair
                    await self.data_processor.update_tracked_candles(symbol, timeframe, latest)
                    
                    last_printed = latest[0]
                    
            except Exception as e:
                print(f"Error: {symbol}/{timeframe} - {e}")
                await asyncio.sleep(1)

    def get_candles(self, symbol, timeframe):
        """Helper method to get candles for a specific symbol-timeframe pair"""
        return self.data_processor.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol, timeframe):
        """Helper method to get the latest candle for a specific symbol-timeframe pair"""
        return self.data_processor.get_latest_candle(symbol, timeframe)

    async def run(self):
        tasks = []
    
        # Create a task for each symbol-timeframe pair
        for symbol, timeframe in self.symbol_timeframes:
            tasks.append(self.watch_ohlcv(symbol, timeframe))
        
        try:
            await asyncio.gather(*tasks)
        finally:
            # Only close the client if we created it ourselves
            if self.should_close_client:
                await self.binance.close()




        