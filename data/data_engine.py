import asyncio
from data.data_fetcher import DataFetcher
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import time

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataEngine:
    """Data Engine for collecting and providing market data.
    
    This class is responsible for:
    1. Fetching market data from exchanges
    2. Processing and storing the data in a standardized format
    3. Providing access to the data for analysis
    
    Attributes:
        binance_client: Binance client instance for market data.
        data_fetcher: Market data fetcher instance.
        running: Boolean indicating if the engine is running.
    """
    
    def __init__(self, binance_client, max_candles: int = 100):
        """Initialize the data engine.
        
        Args:
            binance_client: Binance client instance.
            max_candles: Maximum number of candles to store.
        """
        # Import here to avoid circular imports
        from data.data_fetcher import DataFetcher
        
        # Store binance client reference
        self.binance_client = binance_client
        
        # Setup data fetcher with the client
        self.data_fetcher = DataFetcher(binance_client=binance_client, max_candles=max_candles)
        self.running = False
        
        print(f"[DataEngine] Initialized with max_candles={max_candles}")
        
    async def run(self):
        """Run the data engine to continuously collect market data.
        
        This method starts the data fetcher and keeps it running until stopped.
        """
        if self.running:
            print("[DataEngine] Already running")
            return
            
        self.running = True
        print("[DataEngine] Starting data collection")
        
        try:
            await self.data_fetcher.run()
        except asyncio.CancelledError:
            print("[DataEngine] Data collection task cancelled")
            self.running = False
            raise
        except Exception as e:
            print(f"[DataEngine] Error during data collection: {e}")
            self.running = False
            raise
        finally:
            self.running = False
    
    def get_candles(self, symbol: str, timeframe: str) -> List[List[float]]:
        """Get OHLCV candle data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data.
            
        Returns:
            List of OHLCV candles.
        """
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[List[float]]:
        """Get the latest OHLCV candle for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data.
            
        Returns:
            The latest OHLCV candle or None if no data available.
        """
        candles = self.get_candles(symbol, timeframe)
        if candles and len(candles) > 0:
            return candles[-1]
        return None
    
    def get_latest_price(self, symbol: str, timeframe: str = "1m") -> Optional[float]:
        """Get the latest price for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data (default: "1m").
            
        Returns:
            The latest close price or None if no data available.
        """
        latest_candle = self.get_latest_candle(symbol, timeframe)
        if latest_candle and len(latest_candle) >= 5:
            return latest_candle[4]  # Close price
        return None
    
    @staticmethod
    def extract_ohlcv(candle: List[float]) -> Dict[str, float]:
        """Extract OHLCV values from a candle into a dictionary.
        
        Args:
            candle: OHLCV candle data.
            
        Returns:
            Dictionary with timestamp, open, high, low, close, and volume.
        """
        if not candle or len(candle) < 6:
            return {}
            
        return {
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5]
        }
        
    @staticmethod
    def get_candle_change_pct(candle: List[float]) -> float:
        """Calculate the percentage change in a candle.
        
        Args:
            candle: OHLCV candle data.
            
        Returns:
            Percentage change from open to close.
        """
        if not candle or len(candle) < 5 or candle[1] == 0:
            return 0.0
            
        return (candle[4] - candle[1]) / candle[1] * 100  # (close - open) / open * 100


if __name__ == "__main__":
    from binance_exchange import BinanceClient
    
    # Create a standalone instance
    client = BinanceClient(testnet=True)
    data_engine = DataEngine(binance_client=client, max_candles=30)
    
    # Run this to collect some data first
    print("Starting data collection. Press Ctrl+C to stop...")
    try:
        asyncio.run(data_engine.run())
    except KeyboardInterrupt:
        # Stopping the data collection
        print("\nStopped data collection.")
    finally:
        # Make sure we close the connection
        print("Closing connection...")
        asyncio.run(client.close())

