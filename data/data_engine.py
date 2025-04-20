import asyncio
from data_fetcher import DataFetcher


class DataEngine:
    def __init__(self, max_candles=3):
        self.data_fetcher = DataFetcher(max_candles=max_candles)

    def get_candles(self, symbol, timeframe):
        """Get all candles for a specific symbol-timeframe pair"""
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol, timeframe):
        """Get the latest candle for a specific symbol-timeframe pair"""
        return self.data_fetcher.get_latest_candle(symbol, timeframe)
    
    """ def print_all_deques(self):

        processor = self.data_fetcher.data_processor
        for key, deque in processor.symbol_candles.items():
            print(f"\nDeque for {key}:")
            for candle in deque:
                print(candle)
            print(f"Total candles: {len(deque)}")
            
    def print_symbols_summary(self):

        processor = self.data_fetcher.data_processor
        print("\nSymbol-Timeframe Summary:")
        for key, deque in processor.symbol_candles.items():
            print(f"{key}: {len(deque)} candles") """

    async def run(self):
        await self.data_fetcher.run()


if __name__ == "__main__":
    data_engine = DataEngine(max_candles=30)  # You can adjust this number based on your needs
    
    # Run this to collect some data first
    print("Starting data collection. Press Ctrl+C to stop and print deques...")
    try:
        asyncio.run(data_engine.run())

    except KeyboardInterrupt:
        # After stopping data collection, print the deques
        print("\nStopped data collection.")
        #data_engine.print_symbols_summary()
        
        # Uncomment the next line to print full deque contents
        #data_engine.print_all_deques()
