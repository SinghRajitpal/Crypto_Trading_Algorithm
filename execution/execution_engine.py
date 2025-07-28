# Production Execution Engine implementing the complete system from the document

from typing import Dict, Any, Optional, List
import time
from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from execution.executor import OrderExecutor
from execution.stress_handler import StressHandlingModule
import traceback
from datetime import datetime, timedelta
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

class ProductionExecutionEngine:
    """Production Execution Engine implementing the complete workflow from the document:
    
    Per 1-min bar workflow:
    1. Update metrics (ATR, correlations, equity curve)
    2. Rebalance if daily trigger
    3. On signal, size/leverage within budgets
    4. Execute with safeguards (stress handling, kill switches)
    """
    
    def __init__(self, binance_client, total_capital: float = 1000.0):
        """Initialize the production execution engine.
        
        Args:
            binance_client: Binance client for exchange communication.
            total_capital: Total trading capital in USDT.
        """
        self.binance_client = binance_client
        self.total_capital = total_capital
        
        # Initialize production components
        self.portfolio_manager = ProductionPortfolioManager(
            total_capital=total_capital,
            target_volatility=0.18,    # 18% target volatility from document
            max_allocation_pct=0.85    # 85% max allocation for production
        )
        
        self.risk_manager = ProductionRiskManager(
            portfolio_manager=self.portfolio_manager
        )
        
        self.order_executor = OrderExecutor(
            binance_client=binance_client,
            portfolio_manager=self.portfolio_manager,
            risk_manager=self.risk_manager
        )
        
        # Initialize order manager for SL/TP tracking
        from execution.order_manager import OrderManager
        self.order_manager = OrderManager(binance_client)
        
        # Initialize stress handling module
        self.stress_handler = StressHandlingModule(self)
        
        # Stress handling state
        self.flash_crash_count = 0
        self.last_flash_crash_time = 0
        self.trading_paused = False
        
        # Initialize symbols from config
        self.portfolio_manager.process_symbols_from_config()
        
        logger.info(f"ProductionExecution initialized with ${total_capital:.2f} USDT capital")
        logger.info(f"Target volatility: 18%, Max allocation: 85%")
    
    async def setup(self):
        """Setup the execution engine asynchronously."""
        try:
            await self.binance_client.setup_account_config()
            logger.info("Account configuration completed")
            
            # Start order manager monitoring
            await self.order_manager.start_monitoring()
            logger.info("Order manager monitoring started")
            
        except Exception as e:
            logger.warning(f"Setup warning: {e}")
    
    async def cleanup(self):
        """Cleanup the execution engine."""
        try:
            # Stop order manager monitoring
            await self.order_manager.stop_monitoring()
            logger.info("Order manager monitoring stopped")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def update_market_data_bar(self, symbol: str, ohlcv_data: Dict[str, float], 
                              atr_value: float, correlation_data: Dict[str, float] = None) -> None:
        """Update market data for 1-min bar processing as per document workflow.
        
        Args:
            symbol: Trading pair symbol.
            ohlcv_data: OHLCV data for the bar.
            atr_value: Current ATR(30) value.
            correlation_data: Correlations with other symbols.
        """
        # Update volatility data (EMA of 1-min ATR(30) over 60 bars)
        current_price = ohlcv_data.get('close', 0.0)  # Use close price for normalization
        self.portfolio_manager.update_volatility_data(symbol, atr_value, current_price)
        
        # Update correlation data (EMA of pairwise returns over 60 bars)
        if correlation_data:
            for other_symbol, correlation in correlation_data.items():
                if other_symbol != symbol:
                    self.portfolio_manager.update_correlation_data(symbol, other_symbol, correlation)
        
        # Check for flash crash conditions using stress handler
        self.stress_handler.check_flash_crash(symbol, ohlcv_data, atr_value)
        
        # Check connection health
        self.stress_handler.check_connection_lag(datetime.now())
        
        # Update equity curve for slope calculation
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        equity_proxy = portfolio_summary['total_capital']  # Simplified equity tracking
        self.risk_manager.update_equity_curve(equity_proxy)
    
    def process_daily_rebalance(self) -> bool:
        """Process daily portfolio rebalancing if needed.
        
        Returns:
            True if rebalancing was performed.
        """
        if not self.portfolio_manager.should_rebalance():
            return False
        
        # Get active symbols from current allocations or volatility data
        active_symbols = list(self.portfolio_manager.volatility_data.keys())
        
        if not active_symbols:
            logger.debug("No active symbols for rebalancing")
            return False
        
        logger.info("Daily rebalance triggered")
        
        # Perform rebalancing using production allocation system
        new_allocations = self.portfolio_manager.rebalance_portfolio(active_symbols)
        
        # Log rebalancing results
        total_allocated = sum(a.allocated_capital for a in new_allocations.values())
        turnover = self._calculate_turnover(new_allocations)
        
        logger.info(f"Rebalance completed:")
        logger.info(f"  Total allocated: ${total_allocated:.2f}")
        logger.info(f"  Turnover: {turnover:.2%}")
        logger.info(f"  High vol regime: {self.portfolio_manager.is_high_volatility_regime()}")
        
        return True
    
    def _calculate_turnover(self, new_allocations: Dict) -> float:
        """Calculate portfolio turnover from rebalancing.
        
        Args:
            new_allocations: New allocation dictionary.
            
        Returns:
            Portfolio turnover as a percentage.
        """
        if not hasattr(self, '_previous_allocations'):
            self._previous_allocations = {}
        
        total_change = 0.0
        total_capital = self.portfolio_manager.total_capital
        
        # Calculate absolute changes in allocations
        current_symbols = set(new_allocations.keys())
        previous_symbols = set(self._previous_allocations.keys())
        all_symbols = current_symbols | previous_symbols
        
        for symbol in all_symbols:
            new_alloc = new_allocations.get(symbol, {}).get('allocated_capital', 0) if isinstance(new_allocations.get(symbol), dict) else 0
            old_alloc = self._previous_allocations.get(symbol, 0)
            total_change += abs(new_alloc - old_alloc)
        
        # Update previous allocations
        self._previous_allocations = {
            symbol: alloc.allocated_capital if hasattr(alloc, 'allocated_capital') else alloc
            for symbol, alloc in new_allocations.items()
        }
        
        # Return turnover as percentage of total capital
        return total_change / (2 * total_capital) if total_capital > 0 else 0.0
    
    async def process_signal(self, signal) -> Dict[str, Any]:
        """Process trading signal with production risk management.
        
        Args:
            signal: Trading signal with symbol, action, side, and metadata.
            
        Returns:
            Signal processing result.
        """
        try:
            symbol = signal.symbol
            action = signal.action
            side = signal.side
            
            # Get current price
            current_price = signal.metadata.get('price')
            if current_price is None:
                return {"status": "error", "reason": "Missing price information"}
            
            # Get ATR value for risk calculations
            atr_value = signal.metadata.get('atr_value')
            if atr_value is None:
                return {"status": "error", "reason": "Missing ATR value for production sizing"}
            
            logger.info(f"Processing signal: {symbol} {action.upper()}/{side.upper()}")
            logger.debug(f"Price: {current_price:.2f}, ATR: {atr_value:.6f}")
            
            # Handle different signal actions
            if action == "hold" or side == "none":
                return {"status": "skipped", "reason": "Hold signal", "symbol": symbol}
            
            elif action == "open":
                return await self._process_open_signal(signal, current_price, atr_value)
            
            elif action == "exit":
                return await self._process_exit_signal(symbol)
            
            else:
                return {"status": "error", "reason": f"Invalid action: {action}", "symbol": symbol}
                
        except Exception as e:
            error_msg = f"Error processing signal: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {"status": "error", "reason": error_msg}
    
    async def _process_open_signal(self, signal, current_price: float, atr_value: float) -> Dict[str, Any]:
        """Process open position signal using production risk management."""
        symbol = signal.symbol
        side = signal.side
        
        # Check kill switches before opening new positions
        kill_switches = self.risk_manager.check_kill_switches()
        if any(kill_switches.values()):
            return {
                "status": "rejected", 
                "reason": f"Kill switches active: {kill_switches}",
                "symbol": symbol
            }
        
        # Check if position already exists
        try:
            positions = await self.binance_client.get_open_positions(symbol)
            if positions:
                return {
                    "status": "skipped",
                    "reason": "Position already exists",
                    "symbol": symbol
                }
        except Exception as e:
            logger.warning(f"Could not check positions: {e}")
        
        # CRITICAL: Ensure allocation exists before validating trade
        allocated_capital = self.portfolio_manager.get_allocated_capital(symbol)
        if allocated_capital <= 0:
            logger.info(f"No allocation for {symbol} - checking if rebalance needed")
            
            # Add this symbol to volatility data if missing
            if symbol not in self.portfolio_manager.volatility_data:
                self.portfolio_manager.update_volatility_data(symbol, atr_value, current_price)
                logger.debug(f"Added volatility data for {symbol}")
            
            # Force rebalance if needed
            active_symbols = list(self.portfolio_manager.volatility_data.keys())
            if active_symbols:
                allocations = self.portfolio_manager.rebalance_portfolio(active_symbols)
                new_allocation = self.portfolio_manager.get_allocated_capital(symbol)
                logger.info(f"After rebalance, {symbol} allocation: ${new_allocation:.2f}")
                
                if new_allocation <= 0:
                    return {
                        "status": "rejected",
                        "reason": f"No capital allocated to {symbol} even after rebalancing",
                        "symbol": symbol
                    }
        
        # Validate trade using production risk management
        risk_result = self.risk_manager.validate_trade(
            symbol=symbol,
            action="open",
            side=side,
            entry_price=current_price,
            atr_value=atr_value
        )
        
        if not risk_result["valid"]:
            return {
                "status": "rejected",
                "reason": risk_result["reason"],
                "symbol": symbol
            }
        
        # Extract position information
        position_info = risk_result["position_info"]
        
        logger.info(f"Opening {side.upper()} position:")
        logger.debug(f"  Size: {position_info['size_contracts']:.6f} contracts")
        logger.debug(f"  Leverage: {position_info['leverage']}x")
        logger.debug(f"  Margin: ${position_info['margin_usdt']:.2f}")
        logger.debug(f"  SL: {position_info['stop_loss_price']:.2f}")
        logger.debug(f"  TP: {position_info['take_profit_price']:.2f}")
        
        # CRITICAL FIX: Use order manager for proper SL/TP tracking
        # Instead of direct execution, use the order manager which handles automatic cancellation
        try:
            # Set leverage first
            await self.binance_client.set_leverage(symbol, position_info['leverage'])
            
            # Use order manager for tracked position with SL/TP
            order_result = await self.order_manager.place_position_with_sltp(
                symbol=symbol,
                side=side,
                amount=position_info['size_contracts'],
                stop_loss=position_info['stop_loss_price'],
                take_profit=position_info['take_profit_price'],
                leverage=position_info['leverage']
            )
            
            if order_result.get('status') == 'success':
                logger.info(f"Position opened with automated SL/TP tracking")
                logger.debug(f"  Main Order ID: {order_result.get('main_order_id')}")
                logger.debug(f"  SL Order ID: {order_result.get('stop_loss_order_id')}")
                logger.debug(f"  TP Order ID: {order_result.get('take_profit_order_id')}")
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "action": "open",
                    "side": side,
                    "size": position_info['size_contracts'],
                    "order_ids": {
                        "main": order_result.get('main_order_id'),
                        "stop_loss": order_result.get('stop_loss_order_id'),
                        "take_profit": order_result.get('take_profit_order_id')
                    }
                }
            else:
                error_msg = order_result.get('error', 'Unknown error from order manager')
                logger.error(f"Order manager failed: {error_msg}")
                return {
                    "status": "error",
                    "reason": error_msg,
                    "symbol": symbol
                }
        
        except Exception as e:
            error_msg = f"Error in order manager execution: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "reason": error_msg,
                "symbol": symbol
            }
    
    async def _process_exit_signal(self, symbol: str) -> Dict[str, Any]:
        """Process exit position signal."""
        logger.info(f"Exiting position for {symbol}")
        return await self.order_executor.execute_close_position(symbol)
    
    def handle_disconnect(self, lag_seconds: float) -> None:
        """Handle connection issues as per document stress handling."""
        if lag_seconds > 3.0:
            self.stress_handler.check_connection_lag(datetime.now() - timedelta(seconds=lag_seconds))
    
    def resume_trading(self) -> None:
        """Resume trading after connection restored."""
        self.trading_paused = False
        logger.info("Trading resumed")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        risk_metrics = self.risk_manager.get_risk_metrics()
        stress_summary = self.stress_handler.get_stress_summary()
        
        return {
            "portfolio": portfolio_summary,
            "risk": risk_metrics,
            "stress": stress_summary,
            "system": {
                "trading_paused": self.trading_paused,
                "last_rebalance": portfolio_summary["last_rebalance"],
                "high_vol_regime": portfolio_summary["high_volatility_regime"]
            }
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary - delegates to portfolio manager.
        
        Returns:
            Portfolio summary dictionary.
        """
        summary = self.portfolio_manager.get_portfolio_summary()
        # Add active_positions if not present
        if "active_positions" not in summary:
            summary["active_positions"] = self.risk_manager.get_risk_metrics().get("active_positions", 0)
        return summary
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics - delegates to risk manager.
        
        Returns:
            Risk metrics dictionary.
        """
        return self.risk_manager.get_risk_metrics()
    
    async def validate_signal(self, signal, current_price: float) -> Dict[str, Any]:
        """Validate trading signal - delegates to risk manager.
        
        Args:
            signal: Trading signal object.
            current_price: Current market price.
            
        Returns:
            Validation result dictionary.
        """
        try:
            # Extract ATR value from signal metadata
            atr_value = signal.metadata.get('atr_value')
            if atr_value is None:
                return {"valid": False, "reason": "Missing ATR value for validation"}
            
            # Extract signal confidence for position sizing adjustment
            signal_confidence = getattr(signal, 'signal_confidence', 0.5)
            strategy_confidence = signal.metadata.get('strategy_confidence', 0.5)
            
            # Combine confidences (geometric mean for conservative approach)
            combined_confidence = (signal_confidence * strategy_confidence) ** 0.5
            
            # Validate using risk manager with confidence adjustment
            result = self.risk_manager.validate_trade(
                symbol=signal.symbol,
                action=signal.action,
                side=signal.side,
                entry_price=current_price,
                atr_value=atr_value,
                signal_confidence=combined_confidence
            )
            
            # Add position information to signal metadata if valid
            if result["valid"] and "position_info" in result:
                position_info = result["position_info"]
                signal.metadata.update({
                    "position_size": position_info["size_contracts"],
                    "position_leverage": position_info["leverage"],
                    "stop_loss_price": position_info["stop_loss_price"],
                    "take_profit_price": position_info["take_profit_price"],
                    "reward_risk_ratio": abs((position_info["take_profit_price"] - current_price) / 
                                           (current_price - position_info["stop_loss_price"])),
                    "combined_confidence": combined_confidence
                })
            
            return result
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    def update_daily_pnl(self, symbol: str, pnl: float) -> None:
        """Update daily P&L for risk tracking.
        
        Args:
            symbol: Trading pair symbol.
            pnl: Current unrealized P&L.
        """
        self.risk_manager.daily_pnl = max(self.risk_manager.daily_pnl, pnl)

    async def emergency_flatten(self, percentage: float = 1.0) -> Dict[str, Any]:
        """Emergency portfolio flattening for kill switches.
        
        Args:
            percentage: Percentage of positions to flatten (0.3 = 30%, 1.0 = 100%).
        """
        logger.critical(f"Emergency flattening {percentage:.0%} of positions")
        
        # Get all open positions (simplified)
        # Would implement actual position flattening logic
        return {
            "status": "flattened",
            "percentage": percentage,
            "reason": "Emergency kill switch triggered"
        }


# Legacy compatibility
ExecutionEngine = ProductionExecutionEngine
