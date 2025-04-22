import asyncio
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from binance_exchange import BinanceClient

class TradingAlgorithm:
    def __init__(self, testnet=True):
        # Create a single BinanceClient instance
        self.binance_client = BinanceClient(testnet=testnet)
        
        # Initialize components and pass the client instance
        self.data_engine = DataEngine(binance_client=self.binance_client, max_candles=30)
        self.algo_engine = AlgoEngine()
        self.running = False
    
    async def start(self):
        """Start the trading algorithm"""
        self.running = True
        
        # Start data collection
        data_task = asyncio.create_task(self.data_engine.run())
        
        # Main trading loop
        try:
            while self.running:
                # Get account metrics (for monitoring)
                metrics = await self.binance_client.get_account_metrics()
                print(f"Account metrics: {metrics}")
                
                # Sleep to avoid excessive API calls
                await asyncio.sleep(60)
                
        except Exception as e:
            print(f"Error in trading loop: {e}")
        finally:
            # Clean up
            self.running = False
            data_task.cancel()
            await self.binance_client.close()
    
    async def stop(self):
        """Stop the trading algorithm"""
        self.running = False

if __name__ == "__main__":
    # Create and run the trading algorithm
    algorithm = TradingAlgorithm(testnet=True)
    
    try:
        asyncio.run(algorithm.start())
    except KeyboardInterrupt:
        print("\nTrading algorithm stopped by user.")
