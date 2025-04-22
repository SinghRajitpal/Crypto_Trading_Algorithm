import asyncio
from data.data_fetcher import DataFetcher
import sys
import os

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataEngine:
    def __init__(self, binance_client=None, max_candles=30, testnet=True):
        # Pass the binance client to the data fetcher if provided, otherwise allow the fetcher to create its own
        self.data_fetcher = DataFetcher(binance_client=binance_client, max_candles=max_candles, testnet=testnet)

    def get_candles(self, symbol, timeframe):
        """Get all candles for a specific symbol-timeframe pair"""
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol, timeframe):
        """Get the latest candle for a specific symbol-timeframe pair"""
        return self.data_fetcher.get_latest_candle(symbol, timeframe)
    
    async def run(self):
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

