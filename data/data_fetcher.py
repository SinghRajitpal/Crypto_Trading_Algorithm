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
        
        This method first loads historical data, then continuously monitors for new candles
        from the exchange and updates the data processor when new candles are received.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Raises:
            Exception: Any exceptions are caught, logged, and the method continues.
        """
        last_printed = None
        candle_count = 0
        
        print(f"Starting data collection for {symbol} ({timeframe})")
        
        try:
            # First, fetch historical data to populate initial candles
            print(f"[{symbol}] Fetching historical data...")
            historical_candles = await self.binance.exchange.fetch_ohlcv(symbol, timeframe, limit=self.data_processor.max_candles + 1)  # Fetch extra to account for filtering
            
            # Filter out incomplete current candle to prevent look-ahead bias
            import time
            from datetime import datetime
            
            current_time = time.time() * 1000  # Current time in milliseconds
            timeframe_ms = self.binance.exchange.parse_timeframe(timeframe) * 1000  # Timeframe in milliseconds
            
            # Calculate the start of the current candle period
            current_candle_start = (current_time // timeframe_ms) * timeframe_ms
            
            # Filter out candles that are from the current incomplete period
            complete_candles = []
            for candle in historical_candles:
                candle_timestamp = candle[0]
                if candle_timestamp < current_candle_start:  # Only include completed candles
                    complete_candles.append(candle)
            
            # Limit to max_candles after filtering
            if len(complete_candles) > self.data_processor.max_candles:
                complete_candles = complete_candles[-self.data_processor.max_candles:]
            
            print(f"[{symbol}] Filtered {len(historical_candles)} raw candles to {len(complete_candles)} complete candles")
            
            # Process historical candles (only complete ones)
            for candle in complete_candles:
                await self.data_processor.update_tracked_candles(symbol, timeframe, candle)
                candle_count += 1
            
            current_candles = len(self.data_processor.get_candles(symbol, timeframe))
            print(f"[{symbol}] Loaded {current_candles} historical candles")
            
        except Exception as e:
            print(f"[{symbol}] Warning: Could not fetch historical data: {e}")
            print(f"[{symbol}] Will start with live data only")
        
        # Now watch for live data
        print(f"[{symbol}] Starting live data stream...")
        
        while True:
            try:
                # Use BinanceClient for market data
                candles = await self.binance.exchange.watch_ohlcv(symbol, timeframe)
                now = time.time() * 1000
                latest = candles[-1]
                
                # Check if we have new data or if this is initial collection
                current_candle_count = len(self.data_processor.get_candles(symbol, timeframe))
                is_new_candle = latest[0] != last_printed
                is_initial_collection = current_candle_count < self.data_processor.max_candles
                
                # Process candle if it's new OR if we're still doing initial collection
                if is_new_candle and (is_initial_collection or now - latest[0] > self.binance.exchange.parse_timeframe(timeframe) * 1000):
                    candle_count += 1
                    
                    # Only print summary on initial candles or every 10 candles
                    if candle_count <= 10 or candle_count % 10 == 0:
                        total_candles = self.data_processor.max_candles
                        current_candles = len(self.data_processor.get_candles(symbol, timeframe))
                        print(f"[{symbol}] Collected {current_candles}/{total_candles} candles")
                    
                    # Update the tracked candles for this specific symbol-timeframe pair
                    await self.data_processor.update_tracked_candles(symbol, timeframe, latest)
                    
                    last_printed = latest[0]
                    
            except Exception as e:
                print(f"Error collecting data for {symbol}/{timeframe}: {e}")
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




        