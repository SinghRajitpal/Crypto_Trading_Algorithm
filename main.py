import asyncio
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.base_strategy import BaseStrategy
from binance_exchange import BinanceClient
import time
from datetime import datetime

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
            
            # Print header
            print("\n" + "=" * 80)
            print(f"{'CRYPTO TRADING BOT':^80}")
            print(f"{'Strategy: ' + self.strategy.strategy_id:^80}")
            print("=" * 80)
            
            # Track the last processed time to avoid redundant updates
            last_processed_time = {}
            last_signal_info = {}
            positions_info = {}
            
            # Run algo engine and process signals
            async for signal in self.algo_engine.run(self.strategy):
                try:
                    current_time = int(signal.timestamp) if signal.timestamp else int(time.time() * 1000)
                    symbol = signal.symbol
                    
                    # Only process if this is a new signal or if 5 seconds have passed since last update
                    if (symbol not in last_processed_time or 
                        current_time - last_processed_time.get(symbol, 0) > 5000):
                        
                        last_processed_time[symbol] = current_time
                        
                        # Get latest candle data
                        candles = self.data_engine.get_candles(symbol, "1m")
                        if not candles:
                            continue
                            
                        latest_candle = candles[-1]
                        
                        # Get current position info
                        position = await self.binance_client.get_open_positions(symbol)
                        position_size = float(position[0].get('contracts', 0)) if position else 0
                        positions_info[symbol] = {
                            'size': position_size,
                            'entry_price': position[0].get('entryPrice', 'N/A') if position else 'N/A',
                            'leverage': position[0].get('leverage', 'N/A') if position else 'N/A',
                            'unrealized_pnl': position[0].get('unrealizedPnl', 'N/A') if position else 'N/A'
                        }
                        
                        # Output border
                        print("\n" + "-" * 80)
                        
                        # 1. Symbol and timestamp
                        readable_time = datetime.fromtimestamp(latest_candle[0]/1000).strftime('%H:%M:%S %d/%m/%Y')
                        print(f"{symbol:^15} | {readable_time:^25} | Status: {'🟢 ACTIVE':^15}")
                        print("-" * 80)
                        
                        # 2. OHLCV Price Data
                        print("PRICE DATA:")
                        print(f"Open: {latest_candle[1]:.2f} | High: {latest_candle[2]:.2f} | Low: {latest_candle[3]:.2f} | Close: {latest_candle[4]:.2f}")
                        print(f"Volume: {latest_candle[5]:.3f} | Change: {((latest_candle[4]-latest_candle[1])/latest_candle[1]*100):.2f}%")
                        
                        # 3. Data Collection Progress
                        max_candles = self.data_engine.data_fetcher.data_processor.max_candles
                        current_candles = len(candles)
                        required_indicator_candles = max(self.strategy.params.get('slow_ma_period', 0), 
                                                     self.strategy.params.get('fast_ma_period', 0)) + 1
                        
                        print("\nDATA STATUS:")
                        print(f"Candles Collected: {current_candles}/{max_candles} | " +
                              f"Required for Indicators: {required_indicator_candles}")
                        
                        # 4. Signal Information
                        print("\nSIGNAL:")
                        signal_changed = (symbol not in last_signal_info or 
                                         last_signal_info[symbol]['action'] != signal.action or
                                         last_signal_info[symbol]['side'] != signal.side)
                        
                        if signal_changed:
                            last_signal_info[symbol] = {
                                'action': signal.action,
                                'side': signal.side,
                                'reason': signal.metadata.get('reason', 'N/A')
                            }
                            
                        print(f"Action: {signal.action.upper()} | Side: {signal.side.upper() if signal.side != 'none' else 'NONE'}")
                        print(f"Reason: {signal.metadata.get('reason', 'N/A')}")
                        print(f"Confidence: {signal.signal_confidence:.2f}")
                        
                        # 5. Indicator values if available
                        if 'fast_ma' in signal.metadata and 'slow_ma' in signal.metadata:
                            print("\nINDICATORS:")
                            print(f"Fast MA ({signal.metadata.get('fast_ma_period', 'N/A')}): {signal.metadata['fast_ma']:.2f}")
                            print(f"Slow MA ({signal.metadata.get('slow_ma_period', 'N/A')}): {signal.metadata['slow_ma']:.2f}")
                        
                        # 6. Position Information
                        print("\nPOSITION:")
                        if position_size != 0:
                            position_type = "LONG" if position_size > 0 else "SHORT"
                            print(f"Status: OPEN ({position_type}) | Size: {abs(position_size)}")
                            print(f"Entry Price: {positions_info[symbol]['entry_price']}")
                            print(f"Leverage: {positions_info[symbol]['leverage']}x")
                            print(f"Unrealized PnL: {positions_info[symbol]['unrealized_pnl']}")
                        else:
                            print("Status: CLOSED | No open position")
                            
                except Exception as e:
                    print(f"\n❌ Error processing signal: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
            await self.stop()
        except asyncio.CancelledError:
            print("\nTask cancelled")
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
            
        print("\n" + "-" * 80)
        print("SHUTTING DOWN")
        print("-" * 80)
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
            print("Closing exchange connection...")
            await self.binance_client.close()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("-" * 80)
            print("Shutdown complete")
            print("-" * 80)

if __name__ == "__main__":
    # Example usage with MA Crossover strategy
    from algorithm.strategies.ma_crossover import MACrossoverStrategy
    
    # Create strategy with custom parameters
    strategy = MACrossoverStrategy(params={
        'fast_ma_period': 9,
        'slow_ma_period': 21
    })
    
    # Create and run the trading algorithm with the strategy
    algorithm = TradingAlgorithm(strategy=strategy, testnet=True)
    
    asyncio.run(algorithm.start())
