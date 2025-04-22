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
    """Data Fetcher for retrieving real-time market data from exchanges.
    
    This class handles the communication with exchanges to fetch candle data
    and provides it to the processor for storage and retrieval.
    
    Attributes:
        binance: Binance client instance for API communication.
        symbol_timeframes: List of symbol-timeframe pairs to monitor.
        data_processor: Processor for storing and managing candle data.
        should_close_client: Whether to close the client when done.
    """
    
    def __init__(self, binance_client=None, max_candles=1000, testnet=True):
        """Initializes the Data Fetcher.
        
        Args:
            binance_client: Optional Binance client instance. If not provided,
                a new one will be created.
            max_candles: Maximum number of candles to store in memory.
            testnet: Whether to use testnet (default: True).
        """
        # Use provided client or create a new one
        self.binance = binance_client if binance_client else BinanceClient(testnet=testnet)
        self.symbol_timeframes = config.symbols
        self.data_processor = DataProcessor(max_candles=max_candles)
        # Track if we need to close the client (only if we created it)
        self.should_close_client = binance_client is None
 
    async def watch_ohlcv(self, symbol, timeframe):
        """Watches for OHLCV (candle) data for a specific symbol-timeframe pair.
        
        This method continuously monitors for new candles from the exchange
        and updates the data processor when a closed candle is received.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Raises:
            Exception: Any exceptions are caught, logged, and the method continues.
        """
        last_printed = None
        candle_count = 0
        
        print(f"\n[{symbol}] Starting data collection for {timeframe} timeframe...")
        
        while True:
            try:
                # Use BinanceClient for market data
                candles = await self.binance.exchange.watch_ohlcv(symbol, timeframe)
                now = time.time() * 1000
                latest = candles[-1]
                
                # Check if this is a closed candle
                if latest[0] != last_printed and now - latest[0] > self.binance.exchange.parse_timeframe(timeframe) * 1000:
                    candle_count += 1
                    # Convert millisecond timestamp to readable date format
                    readable_time = datetime.fromtimestamp(latest[0]/1000).strftime('%H:%M:%S %d/%m/%Y')
                    
                    print(f"\n📊 {symbol} ({timeframe}) Candle #{candle_count}")
                    print(f"Time: {readable_time}")
                    print(f"Open: {latest[1]:.2f} | High: {latest[2]:.2f} | Low: {latest[3]:.2f} | Close: {latest[4]:.2f}")
                    print(f"Volume: {latest[5]:.3f}")

                    # Update the tracked candles for this specific symbol-timeframe pair
                    await self.data_processor.update_tracked_candles(symbol, timeframe, latest)
                    
                    last_printed = latest[0]
                    
            except Exception as e:
                print(f"\n❌ Error collecting data for {symbol}/{timeframe}: {e}")
                await asyncio.sleep(1)

    def get_candles(self, symbol, timeframe):
        """Gets all candles for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            List of candles for the specified symbol and timeframe.
        """
        return self.data_processor.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol, timeframe):
        """Gets the latest candle for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            Latest candle for the specified symbol and timeframe or None if unavailable.
        """
        return self.data_processor.get_latest_candle(symbol, timeframe)

    async def run(self):
        """Runs the data fetcher to collect data for all configured symbol-timeframe pairs.
        
        This method starts separate monitoring tasks for each symbol-timeframe pair
        and manages their execution.
        
        Raises:
            Exception: Exceptions from individual tasks are caught and the client is closed if needed.
        """
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




        