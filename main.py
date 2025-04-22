import asyncio
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.base_strategy import BaseStrategy
from binance_exchange import BinanceClient

class TradingAlgorithm:
    """Main trading algorithm class that orchestrates the entire trading system.
    
    This class brings together all components of the trading system:
    - Data collection and processing
    - Strategy implementation
    - Signal generation
    - Execution handling
    
    It manages the lifecycle of the trading bot and provides a simple interface
    for starting and stopping the trading process.
    
    Attributes:
        binance_client: Binance client for API communication.
        data_engine: Data engine for market data.
        algo_engine: Algorithm engine for signal generation.
        strategy: Trading strategy to use.
        running: Boolean indicating if the algorithm is running.
        data_task: Asyncio task for data collection.
    """
    
    def __init__(self, strategy: BaseStrategy, testnet=True):
        """Initializes the trading algorithm with a specific strategy.
        
        Args:
            strategy: The trading strategy to use.
            testnet: Whether to use testnet or live trading (default: True).
            
        Raises:
            ValueError: If strategy is not an instance of BaseStrategy.
        """
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("strategy must be an instance of BaseStrategy")
            
        # Create a single BinanceClient instance
        self.binance_client = BinanceClient(testnet=testnet)
        
        # Initialize components with enough candles for indicator calculation
        # Use max(100, 5 * slow_ma_period) to ensure enough data for indicators
        max_candles = max(100, 5 * strategy.params.get('slow_ma_period', 20))
        # Make sure we have additional candles for crossover detection
        max_candles += 5
        self.data_engine = DataEngine(binance_client=self.binance_client, max_candles=max_candles)
        self.algo_engine = AlgoEngine(data_engine=self.data_engine, binance_client=self.binance_client)
        
        # Set the algo engine on the strategy
        strategy.set_algo_engine(self.algo_engine)
        self.strategy = strategy
        
        self.running = False
        self.data_task = None
    
    async def start(self):
        """Starts the trading algorithm.
        
        This method initiates the data collection process and begins
        processing trading signals from the strategy.
        
        Raises:
            Exception: Any exceptions are caught, logged, and cleanup is performed.
        """
        if self.running:
            print("Trading algorithm is already running")
            return
            
        self.running = True
        
        try:
            # Start data collection
            self.data_task = asyncio.create_task(self.data_engine.run())
            
            # Run algo engine and process signals
            async for signal in self.algo_engine.run(self.strategy):
                try:
                    print(f"Signal received: {signal.action} {signal.side} {signal.symbol}")
                    print(f"Metadata: {signal.metadata}")
                except Exception as e:
                    print(f"Error processing signal: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\nShutdown requested... Cleaning up...")
            await self.stop()
        except asyncio.CancelledError:
            print("\nTask cancelled... Cleaning up...")
            await self.stop()
        except Exception as e:
            print(f"\nUnexpected error in trading loop: {e}")
            await self.stop()
            raise  # Re-raise the exception after cleanup
    
    async def stop(self):
        """Stops the trading algorithm and cleans up resources.
        
        This method cancels all running tasks, closes positions if needed,
        and ensures proper shutdown of all components.
        
        Raises:
            Exception: Exceptions during cleanup are caught and logged.
        """
        if not self.running:
            return
            
        print("Stopping trading algorithm...")
        self.running = False
        
        try:
            # Cancel data task if it exists
            if self.data_task and not self.data_task.done():
                print("Cancelling data collection task...")
                self.data_task.cancel()
                try:
                    await self.data_task
                except asyncio.CancelledError:
                    pass
                    
            # Close all open positions if needed
            # TODO: Implement position closing logic here
            
            # Close Binance client connection
            print("Closing Binance client connection...")
            await self.binance_client.close()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Cleanup completed. Bot stopped.")

if __name__ == "__main__":
    # Example usage with MA Crossover strategy
    from algorithm.strategies.ma_crossover import MACrossoverStrategy
    
    # Create strategy with custom parameters
    strategy = MACrossoverStrategy(params={
        'fast_ma_period': 2,
        'slow_ma_period': 4
    })
    
    # Create and run the trading algorithm with the strategy
    algorithm = TradingAlgorithm(strategy=strategy, testnet=True)
    
    asyncio.run(algorithm.start())
