import asyncio
from data.data_fetcher import DataFetcher
import sys
import os

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataEngine:
    """Data Engine for accessing market data.
    
    This class provides a unified interface for accessing candle data
    across multiple symbols and timeframes. It manages the underlying
    data fetcher that communicates with exchanges.
    
    Attributes:
        data_fetcher: DataFetcher instance for retrieving market data.
    """
    
    def __init__(self, binance_client=None, max_candles=30, testnet=True):
        """Initializes the Data Engine.
        
        Args:
            binance_client: Optional Binance client instance. If not provided,
                the DataFetcher will create its own.
            max_candles: Maximum number of candles to store in memory.
            testnet: Whether to use testnet (default: True).
        """
        # Pass the binance client to the data fetcher if provided, otherwise allow the fetcher to create its own
        self.data_fetcher = DataFetcher(binance_client=binance_client, max_candles=max_candles, testnet=testnet)

    def get_candles(self, symbol, timeframe):
        """Gets all candles for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            List of candles for the specified symbol and timeframe.
        """
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol, timeframe):
        """Gets the latest candle for a specific symbol-timeframe pair.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h").
            
        Returns:
            Latest candle for the specified symbol and timeframe or None if unavailable.
        """
        return self.data_fetcher.get_latest_candle(symbol, timeframe)
    
    async def run(self):
        """Runs the data engine to continuously collect market data.
        
        This method starts the data fetcher to continuously collect
        candle data for all configured symbol-timeframe pairs.
        """
        await self.data_fetcher.run()


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

