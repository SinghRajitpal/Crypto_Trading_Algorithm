import asyncio
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from execution.execution_engine import ProductionExecutionEngine
from algorithm.strategies.base_strategy import BaseStrategy
from binance_exchange import BinanceClient
import time
from datetime import datetime, timedelta
import traceback
import numpy as np
from utils.logging_config import get_logger, console_log
import config

# Get logger for this module
logger = get_logger(__name__)

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
        execution_engine: Execution engine for order execution and risk management.
        strategy: Trading strategy to use.
        running: Boolean indicating if the algorithm is running.
        data_task: Asyncio task for data collection.
    """
    
    def __init__(self, strategy: BaseStrategy, testnet=True, total_capital=None):
        """Initializes the trading algorithm with a specific strategy.
        
        Args:
            strategy: The trading strategy to use.
            testnet: Whether to use testnet or live trading (default: True).
            total_capital: Total capital in USDT for portfolio management. If None, fetches from exchange.
            
        Raises:
            ValueError: If strategy is not an instance of BaseStrategy.
        """
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("strategy must be an instance of BaseStrategy")
            
        # Create a single BinanceClient instance
        self.binance_client = BinanceClient(testnet=testnet)
        
        # Fetch actual capital from exchange if not provided
        self.total_capital = total_capital  # Will be set properly in start() method
        
        # Initialize components will be done after we fetch actual capital
        self.data_engine = None
        self.algo_engine = None
        self.execution_engine = None
        
        # Set the algo engine on the strategy (will be set after initialization)
        self.strategy = strategy
        
        self.running = False
        self.data_task = None
        
        logger.info(f"Pre-initialized TradingAlgorithm with {strategy.strategy_id} strategy")
    
    async def start(self):
        """Starts the trading algorithm.
        
        This method initiates the data collection process and begins
        processing trading signals from the strategy.
        
        Raises:
            Exception: Any exceptions are caught, logged, and cleanup is performed.
        """
        if self.running:
            logger.warning("TradingAlgorithm is already running")
            return
            
        self.running = True
        
        try:
            # === STEP 1: Fetch actual capital from exchange ===
            if self.total_capital is None:
                logger.info("Fetching actual capital from exchange...")
                account_metrics = await self.binance_client.get_account_metrics()
                self.total_capital = account_metrics['total_wallet_balance']
                logger.success(f"Fetched actual capital: ${self.total_capital:.2f} USDT")
            else:
                logger.info(f"Using provided capital: ${self.total_capital:.2f} USDT")
            
            # === STEP 2: Initialize components with actual capital ===
            logger.info("Initializing components with actual capital...")
            
            # Initialize components with enough candles for indicator calculation
            max_candles = max(100, 5 * self.strategy.params.get('slow_ma_period', 20))
            
            logger.debug(f"Initializing with {max_candles} candles per symbol")
            
            # Initialize data engine for market data collection
            self.data_engine = DataEngine(binance_client=self.binance_client, max_candles=max_candles)
            
            # Initialize algo engine for signal generation only
            self.algo_engine = AlgoEngine(data_engine=self.data_engine)
            
            # Initialize execution engine with actual capital
            self.execution_engine = ProductionExecutionEngine(
                binance_client=self.binance_client,
                total_capital=self.total_capital
            )
            
            # Set the algo engine on the strategy
            self.strategy.set_algo_engine(self.algo_engine)
            
            logger.success(f"Components initialized with {self.strategy.strategy_id} strategy")
            
            # === STEP 3: Setup execution engine and initialize portfolio ===
            await self.execution_engine.setup()
            
            logger.info("Initializing portfolio allocations...")
            
            # Start data collection first to get historical candles
            self.data_task = asyncio.create_task(self.data_engine.run())
            logger.info("Started data collection task")
            
            # Wait for initial data collection to complete
            logger.info("Waiting for historical data collection...")
            await asyncio.sleep(8)  # Give enough time for candles to be fetched
            
            # Initialize portfolio with real market data
            config_symbols = [symbol for symbol, _ in config.symbols]
            
            # Use the integrated portfolio initialization system
            initialization_success = self.execution_engine.portfolio_manager.initialize_market_data(
                self.data_engine, config_symbols
            )
            
            if initialization_success:
                logger.success("Portfolio data initialization successful")
                
                # Force rebalance on startup by setting old timestamp
                self.execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
                
                # Trigger initial rebalance with real data
                logger.info("Executing initial portfolio rebalance...")
                rebalanced = self.execution_engine.process_daily_rebalance()
                
                if rebalanced:
                    logger.success("Initial portfolio allocation completed")
                    # Verify allocation worked
                    summary = self.execution_engine.portfolio_manager.get_portfolio_summary()
                    logger.info(f"Total allocated: ${summary['allocated_capital']:.2f} ({summary['allocation_percentage']:.1%})")
                else:
                    logger.warning("Initial rebalance failed - attempting manual rebalance")
                    # Manual fallback
                    allocations = self.execution_engine.portfolio_manager.rebalance_portfolio(config_symbols)
                    if allocations:
                        total = sum(a.allocated_capital for a in allocations.values())
                        logger.success(f"Manual rebalance successful: ${total:.2f} allocated")
            else:
                logger.error("Portfolio initialization failed - using defaults")
            
            # Wait briefly for any remaining data to stabilize
            await asyncio.sleep(2)
            console_log("\n" + "=" * 80)
            console_log(f"{'CRYPTO TRADING BOT':^80}")
            console_log(f"{'Strategy: ' + self.strategy.strategy_id:^80}")
            console_log("=" * 80)
            
            # Display portfolio and risk information
            portfolio_summary = self.execution_engine.get_portfolio_summary()
            risk_metrics = self.execution_engine.get_risk_metrics()
            
            console_log("\nPORTFOLIO INITIALIZATION:")
            console_log(f"Total Capital: ${portfolio_summary['total_capital']:.2f}")
            console_log(f"Max Allocation: {portfolio_summary['allocation_percentage']*100:.1f}%")
            console_log(f"Risk Status: {risk_metrics['risk_status'].upper()}")
            
            # Display strategy risk parameters
            console_log("\nSTRATEGY RISK PARAMETERS:")
            console_log(f"ATR Stop Loss Multiplier: 1.8x")
            console_log(f"ATR Take Profit Multiplier: 3.6x (1:2 risk-reward)")
            console_log(f"Risk per Trade: 0.8% of allocated capital")
            console_log("-" * 80)
            
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
                        latest_candle = self.data_engine.get_latest_candle(symbol, "1m")
                        if not latest_candle:
                            logger.warning(f"No candle data available for {symbol}, skipping")
                            continue
                            
                        # Extract OHLCV values using the utility method
                        candle_data = self.data_engine.extract_ohlcv(latest_candle)
                        current_price = candle_data["close"]
                        
                        # === CRITICAL FIX: Update market data for allocation system ===
                        # Get ATR value for volatility tracking
                        atr_value = signal.metadata.get('atr_value')
                        if atr_value:
                            # Update volatility data for portfolio allocation
                            self.execution_engine.update_market_data_bar(
                                symbol=symbol,
                                ohlcv_data=candle_data,
                                atr_value=atr_value
                            )
                        
                        # Check and trigger daily rebalancing
                        rebalanced = self.execution_engine.process_daily_rebalance()
                        if rebalanced:
                            logger.info("Portfolio rebalanced")
                        
                        # Validate the signal with risk management
                        if signal.action == "open":
                            risk_result = await self.execution_engine.validate_signal(signal, current_price)
                            signal.metadata['risk_valid'] = risk_result.get('valid', False)
                            signal.metadata['risk_reason'] = risk_result.get('reason', 'Unknown reason')
                        
                        # Process trade execution using execution engine
                        # Update the signal with current_price since process_signal no longer accepts it as a parameter
                        signal.metadata['price'] = current_price
                        execution_result = await self.execution_engine.process_signal(signal)
                        
                        # Get current position info AFTER execution to reflect any new positions
                        position = await self.binance_client.get_open_positions(symbol)
                        position_size = float(position[0].get('contracts', 0)) if position else 0
                        positions_info[symbol] = {
                            'size': position_size,
                            'entry_price': position[0].get('entryPrice', 'N/A') if position else 'N/A',
                            'leverage': position[0].get('leverage', 'N/A') if position else 'N/A',
                            'unrealized_pnl': position[0].get('unrealizedPnl', 'N/A') if position else 'N/A'
                        }
                        
                        # Output border
                        console_log("\n" + "-" * 80)
                        
                        # 1. Symbol and timestamp
                        readable_time = datetime.fromtimestamp(candle_data["timestamp"]/1000).strftime('%H:%M:%S %d/%m/%Y')
                        console_log(f"{symbol:^15} | {readable_time:^25} | Status: {'🟢 ACTIVE':^15}")
                        console_log("-" * 80)
                        
                        # 2. OHLCV Price Data
                        price_change = self.data_engine.get_candle_change_pct(latest_candle)
                        console_log("PRICE DATA:")
                        console_log(f"Open: {candle_data['open']:.2f} | High: {candle_data['high']:.2f} | Low: {candle_data['low']:.2f} | Close: {candle_data['close']:.2f}")
                        console_log(f"Volume: {candle_data['volume']:.3f} | Change: {price_change:.2f}%")
                        
                        # 3. Data Collection Progress
                        candles = self.data_engine.get_candles(symbol, "1m")
                        max_candles = self.data_engine.data_fetcher.data_processor.max_candles
                        current_candles = len(candles)
                        required_indicator_candles = max(self.strategy.params.get('slow_ma_period', 0), 
                                                     self.strategy.params.get('fast_ma_period', 0)) + 1
                        
                        console_log("\nDATA STATUS:")
                        console_log(f"Candles Collected: {current_candles}/{max_candles} | "
                              f"Required for Indicators: {required_indicator_candles}")
                        
                        # 4. Signal Information
                        console_log("\nSIGNAL:")
                        signal_changed = (symbol not in last_signal_info or 
                                         last_signal_info[symbol]['action'] != signal.action or
                                         last_signal_info[symbol]['side'] != signal.side)
                        
                        if signal_changed:
                            last_signal_info[symbol] = {
                                'action': signal.action,
                                'side': signal.side,
                                'reason': signal.metadata.get('reason', 'N/A')
                            }
                            # Log signal changes for detailed tracking
                            logger.debug(f"Signal change for {symbol}: {signal.action.upper()} {signal.side.upper()}, reason: {signal.metadata.get('reason', 'N/A')}")
                            
                        console_log(f"Action: {signal.action.upper()} | Side: {signal.side.upper() if signal.side != 'none' else 'NONE'}")
                        console_log(f"Reason: {signal.metadata.get('reason', 'N/A')}")
                        console_log(f"Confidence: {signal.signal_confidence:.2f}")
                        
                        # 5. Indicator values if available
                        if 'fast_ma' in signal.metadata and 'slow_ma' in signal.metadata:
                            console_log("\nINDICATORS:")
                            console_log(f"Fast MA ({signal.metadata.get('fast_ma_period', 'N/A')}): {signal.metadata['fast_ma']:.2f}")
                            console_log(f"Slow MA ({signal.metadata.get('slow_ma_period', 'N/A')}): {signal.metadata['slow_ma']:.2f}")
                        
                        # 6. Position Information
                        console_log("\nPOSITION:")
                        if position_size != 0:
                            position_type = "LONG" if position_size > 0 else "SHORT"
                            print(f"Status: OPEN ({position_type}) | Size: {abs(position_size)}")
                            print(f"Entry Price: {positions_info[symbol]['entry_price']}")
                            print(f"Leverage: {positions_info[symbol]['leverage']}x")
                            print(f"Unrealized PnL: {positions_info[symbol]['unrealized_pnl']}")
                            
                            # Find stop loss and take profit orders in open orders
                            try:
                                open_orders = await self.binance_client.get_open_orders(symbol)
                                
                                # More comprehensive order type detection
                                sl_order = None
                                tp_order = None
                                
                                for order in open_orders:
                                    order_type = order.get('type', '').lower()
                                    # Stop loss can be 'stop_market', 'stop', or 'stop_loss_limit'
                                    if 'stop' in order_type and 'take' not in order_type:
                                        sl_order = order
                                    # Take profit can be 'take_profit_market', 'limit' or 'take_profit' 
                                    elif ('take' in order_type and 'profit' in order_type) or order_type == 'limit':
                                        tp_order = order
                                
                                # Display Stop Loss and Take Profit with clear formatting
                                print("\nRISK MANAGEMENT ORDERS:")
                                if sl_order:
                                    sl_price = float(sl_order.get('stopPrice') or sl_order.get('price', 0) or 0)
                                    entry_price = float(positions_info[symbol]['entry_price']) if positions_info[symbol]['entry_price'] != 'N/A' else current_price
                                    sl_pct = abs(sl_price - entry_price) / entry_price * 100
                                    print(f"Stop Loss: ${sl_price:.2f} ({sl_pct:.2f}% from entry) [{sl_order.get('type')}]")
                                else:
                                    print("Stop Loss: Not set")
                                    
                                if tp_order:
                                    tp_price = float(tp_order.get('price') or tp_order.get('stopPrice', 0) or 0)
                                    entry_price = float(positions_info[symbol]['entry_price']) if positions_info[symbol]['entry_price'] != 'N/A' else current_price
                                    tp_pct = abs(tp_price - entry_price) / entry_price * 100
                                    print(f"Take Profit: ${tp_price:.2f} ({tp_pct:.2f}% from entry) [{tp_order.get('type')}]")
                                else:
                                    print("Take Profit: Not set")
                                
                                # Calculate and display reward-to-risk ratio if both orders are set
                                if sl_order and tp_order and position_type == "LONG":
                                    sl_price = float(sl_order.get('stopPrice') or sl_order.get('price', 0) or 0)
                                    tp_price = float(tp_order.get('stopPrice') or tp_order.get('price', 0) or 0)
                                    entry_price = float(positions_info[symbol]['entry_price']) if positions_info[symbol]['entry_price'] != 'N/A' else current_price
                                    
                                    risk = entry_price - sl_price
                                    reward = tp_price - entry_price
                                    
                                    if risk > 0:
                                        reward_risk_ratio = reward / risk
                                        print(f"Reward/Risk Ratio: {reward_risk_ratio:.2f}")
                            except Exception as e:
                                print(f"Could not retrieve open orders: {e}")
                                logger.error(f"Failed to retrieve open orders for {symbol}: {e}")
                            
                            # Update daily PnL for risk tracking
                            if positions_info[symbol]['unrealized_pnl'] != 'N/A':
                                self.execution_engine.update_daily_pnl(
                                    symbol, 
                                    float(positions_info[symbol]['unrealized_pnl'])
                                )
                        else:
                            print("Status: CLOSED | No open position")
                            
                        # 7. Risk and Portfolio Information
                        current_portfolio = self.execution_engine.get_portfolio_summary()
                        current_risk = self.execution_engine.get_risk_metrics()
                        
                        # Get symbol-specific allocated capital
                        symbol_allocated_capital = self.execution_engine.portfolio_manager.get_allocated_capital(symbol)
                        
                        print("\nRISK & PORTFOLIO:")
                        print(f"Allocation: {current_portfolio['allocation_percentage']*100:.1f}% | "
                              f"Positions: {current_portfolio['active_positions']}")
                        print(f"Available Capital: ${symbol_allocated_capital:.2f} | "
                              f"Daily PnL: ${current_risk.get('daily_pnl', 0):.2f}")
                        
                        # 8. Risk Assessment for Open Signals
                        if signal.action == "open":
                            print("\nRISK ASSESSMENT:")
                            if signal.metadata.get('risk_valid', False):
                                print(f"✅ Trade meets risk criteria.")
                                if 'position_size' in signal.metadata:
                                    print(f"Position Size: {signal.metadata['position_size']:.6f} | "
                                          f"Leverage: {signal.metadata.get('position_leverage', 'N/A')}x")
                                    
                                    # Display stop loss and take profit with percentages
                                    if 'stop_loss_price' in signal.metadata:
                                        sl_price = signal.metadata['stop_loss_price']
                                        sl_pct = abs(sl_price - current_price) / current_price * 100
                                        print(f"Stop Loss: ${sl_price:.2f} ({sl_pct:.2f}% from entry)")
                                    
                                    if 'take_profit_price' in signal.metadata:
                                        tp_price = signal.metadata['take_profit_price']
                                        tp_pct = abs(tp_price - current_price) / current_price * 100
                                        print(f"Take Profit: ${tp_price:.2f} ({tp_pct:.2f}% from entry)")
                                    
                                    # Show reward-to-risk ratio
                                    if 'reward_risk_ratio' in signal.metadata:
                                        print(f"Reward/Risk Ratio: {signal.metadata['reward_risk_ratio']:.2f}")
                            else:
                                print(f"❌ Trade rejected: {signal.metadata.get('risk_reason', 'Unknown reason')}")
                        
                        # 9. Execution Result
                        if execution_result:
                            print("\nEXECUTION RESULT:")
                            print(f"Status: {execution_result.get('status', 'N/A')}")
                            if 'reason' in execution_result:
                                print(f"Reason: {execution_result['reason']}")
                            
                except Exception as e:
                    print(f"\n❌ [TradingAlgorithm] Error processing signal: {e}")
                    logger.error(f"Error processing signal for {symbol}: {e}", exc_info=True)
                    traceback.print_exc()
                    continue
                    
        except KeyboardInterrupt:
            print("\n[TradingAlgorithm] Shutdown requested by user")
            logger.info("Shutdown requested by user")
            await self.stop()
        except asyncio.CancelledError:
            print("\n[TradingAlgorithm] Task cancelled")
            logger.info("Task cancelled")
            await self.stop()
        except Exception as e:
            print(f"\n❌ [TradingAlgorithm] Unexpected error in trading loop: {e}")
            logger.critical(f"Unexpected error in trading loop: {e}", exc_info=True)
            traceback.print_exc()
            await self.stop()
            raise  # Re-raise the exception after cleanup
    
    async def stop(self):
        """Stops the trading algorithm and cleans up resources.
        
        This method cancels all running tasks and ensures proper shutdown
        of all components while preserving any open positions and their
        associated stop loss and take profit orders.
        
        Raises:
            Exception: Exceptions during cleanup are caught and logged.
        """
        if not self.running:
            return
            
        print("\n" + "-" * 80)
        print("SHUTTING DOWN")
        print("-" * 80)
        self.running = False
        logger.info("Starting shutdown procedure")
        
        try:
            # Cancel data task if it exists
            if self.data_task and not self.data_task.done():
                print("[TradingAlgorithm] Cancelling data collection task...")
                logger.debug("Cancelling data collection task")
                self.data_task.cancel()
                try:
                    await self.data_task
                except asyncio.CancelledError:
                    pass
                    
            # Display current positions that will be preserved
            try:
                positions = await self.binance_client.get_all_positions()
                active_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]
                
                if active_positions:
                    print(f"\n📈 Found {len(active_positions)} active position(s)")
                    print("\n[TradingAlgorithm] Preserving open positions:")
                    for position in active_positions:
                        symbol = position.get('symbol', 'UNKNOWN')
                        # Sanitize symbol format to ensure proper display (handle various formats)
                        display_symbol = symbol.replace(':/USDT', '').replace(':USDT', '').replace(':', '')
                        size = float(position.get('contracts', 0))
                        entry_price = position.get('entryPrice', 'N/A')
                        pnl = position.get('unrealizedPnl', 'N/A')
                        
                        position_type = "LONG" if size > 0 else "SHORT"
                        print(f"- {display_symbol}: {position_type} {abs(size)} contracts @ {entry_price} (PnL: {pnl})")
                        
                        # Display associated SL/TP orders with proper symbol
                        try:
                            orders = await self.binance_client.get_open_orders(display_symbol)
                            sl_order = next((o for o in orders if o.get('type', '').startswith('STOP')), None)
                            tp_order = next((o for o in orders if o.get('type', '').startswith('TAKE_PROFIT')), None)
                            
                            if sl_order:
                                print(f"  Stop Loss: ${float(sl_order.get('stopPrice', 0)):.2f}")
                            if tp_order:
                                print(f"  Take Profit: ${float(tp_order.get('stopPrice', 0)):.2f}")
                        except Exception as order_error:
                            # Silently handle order lookup errors to avoid disrupting shutdown
                            logger.debug(f"Could not retrieve orders for {display_symbol} during shutdown: {order_error}")
                            pass
            except Exception as e:
                print(f"[TradingAlgorithm] Error displaying positions: {e}")
                logger.error(f"Error displaying positions during shutdown: {e}")
            
            # Cleanup execution engine
            if self.execution_engine:
                print("\n[TradingAlgorithm] Cleaning up execution engine...")
                logger.debug("Cleaning up execution engine")
                await self.execution_engine.cleanup()
            
            # Close Binance client connection
            print("\n[TradingAlgorithm] Closing exchange connection...")
            logger.debug("Closing exchange connection")
            await self.binance_client.close()
            
        except Exception as e:
            print(f"❌ [TradingAlgorithm] Error during cleanup: {e}")
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            traceback.print_exc()
        finally:
            print("-" * 80)
            print("SHUTDOWN COMPLETE - Positions and SL/TP orders preserved")
            print("-" * 80)
            logger.info("Shutdown procedure completed")

if __name__ == "__main__":
    logger.info("Starting Crypto Trading Algorithm")
    
    # Example usage with MA Crossover strategy
    from algorithm.strategies.ma_crossover import MACrossoverStrategy
    
    # Create strategy with custom parameters
    strategy = MACrossoverStrategy(params={
        'fast_ma_period': 5,
        'slow_ma_period': 20,
        'stop_loss_pct': 0.001,  # 2% stop loss
        'take_profit_pct': 0.002,  # 4% take profit
        'leverage': 7  # Setting leverage to 20x
    })
    
    logger.info(f"Created {strategy.strategy_id} strategy with parameters: {strategy.params}")
    
    # Create and run the trading algorithm with the strategy
    algorithm = TradingAlgorithm(
        strategy=strategy, 
        testnet=True,
        total_capital=None  # Fetch actual capital from testnet
    )
    
    try:
        asyncio.run(algorithm.start())
    except KeyboardInterrupt:
        logger.info("Algorithm stopped by user")
    except Exception as e:
        logger.critical(f"Algorithm failed with error: {e}", exc_info=True)
        raise
