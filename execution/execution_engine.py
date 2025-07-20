# This file handles all trading execution, position management, and risk controls

from typing import Dict, Any, Optional
import time
from execution.portfolio import PortfolioManager
from execution.risk_manager import RiskManager
from execution.executor import OrderExecutor
import traceback

class ExecutionEngine:
    """Execution Engine for crypto trading.
    
    This class integrates all execution-related components:
    - Portfolio management (allocation and position tracking)
    - Risk management (position sizing, stop-loss, drawdown protection)
    - Order execution (via Binance client)
    
    The execution engine handles the interface between trading signals
    and actual market orders, applying risk and portfolio constraints.
    
    Attributes:
        binance_client: Binance client for exchange communication.
        portfolio_manager: Manager for capital allocation and tracking.
        risk_manager: Manager for risk parameters and position sizing.
        order_executor: Handler for actual order execution.
        total_capital: Total trading capital in USDT.
        default_leverage: Default leverage for new positions.
        default_margin_type: Default margin type for new positions.
    """
    
    def __init__(self, binance_client, total_capital: float = 1000.0):
        """Initialize the execution engine.
        
        Args:
            binance_client: Binance client for exchange communication.
            total_capital: Total trading capital in USDT.
        """
        self.binance_client = binance_client
        self.total_capital = total_capital
        
        # Default values for leverage and margin type
        self.default_leverage = 10
        self.default_margin_type = "isolated"
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            total_capital=total_capital,
            max_allocation_pct=0.5  # Default to 50% max allocation
        )
        
        # Initialize risk manager with portfolio
        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio_manager
        )
        
        # Initialize order executor with components
        self.order_executor = OrderExecutor(
            binance_client=binance_client,
            portfolio_manager=self.portfolio_manager,
            risk_manager=self.risk_manager
        )
        
        # Initialize signal history tracking
        self.signal_history = []
        
        # Process all symbols from config
        self.portfolio_manager.process_all_symbols()
        
        print(f"[ExecutionEngine] Initialized with ${total_capital:.2f} USDT capital")
    
    async def process_signal(self, signal):
        """Process a trading signal from the algorithm.
        
        Args:
            signal: The trading signal containing action and parameters.
            
        Returns:
            Dict: The result of signal processing.
        """
        try:
            # Extract signal details
            symbol = signal.symbol
            action = signal.action
            side = signal.side
            
            # Get current price from signal metadata or fetch it
            current_price = signal.metadata.get('price')
            if current_price is None:
                print(f"[ExecutionEngine] ⚠️ No price provided in signal metadata for {symbol}")
                return {"status": "error", "reason": "Missing price information"}
            
            print(f"\n[ExecutionEngine] Processing signal for {symbol}: {action.upper()}/{side.upper()}")
            print(f"[ExecutionEngine] Current price: {current_price:.2f} USDT")
            
            # For hold signals, skip execution
            if action == "hold" or side == "none":
                print(f"[ExecutionEngine] HOLD signal - No action required for {symbol}")
                return {
                    "status": "skipped",
                    "reason": "Hold signal, no action required",
                    "symbol": symbol
                }
                
            # For open signals, validate risk before execution
            if action == "open":
                # Check if we already have a position in this symbol
                try:
                    positions = await self.binance_client.get_open_positions(symbol)
                    if positions:
                        print(f"[ExecutionEngine] ⚠️ Already have a position in {symbol}, skipping open signal")
                        return {
                            "status": "skipped",
                            "reason": f"Position already exists for {symbol}",
                            "symbol": symbol
                        }
                except Exception as e:
                    print(f"[ExecutionEngine] ⚠️ Could not check existing positions: {e}")
                
                # Verify if the signal has passed risk validation
                if not signal.metadata.get('risk_valid', False):
                    reason = signal.metadata.get('risk_reason', 'Failed risk validation')
                    print(f"[ExecutionEngine] ❌ Signal rejected: {reason}")
                    return {
                        "status": "rejected",
                        "reason": reason,
                        "symbol": symbol
                    }
                
                # Extract position details from signal metadata
                pos_size = signal.metadata.get('position_size', 0)
                leverage = signal.metadata.get('position_leverage', 10)
                stop_loss = signal.metadata.get('stop_loss_price')
                take_profit = signal.metadata.get('take_profit_price')
                
                print(f"[ExecutionEngine] Opening {side.upper()} position for {symbol}")
                print(f"[ExecutionEngine] Position size: {pos_size:.6f} contracts")
                print(f"[ExecutionEngine] Leverage: {leverage}x")
                if stop_loss is not None:
                    print(f"[ExecutionEngine] Stop loss: {stop_loss:.2f}")
                else:
                    print(f"[ExecutionEngine] Stop loss: None")
                if take_profit is not None:
                    print(f"[ExecutionEngine] Take profit: {take_profit:.2f}")
                else:
                    print(f"[ExecutionEngine] Take profit: None")
                
                # Execute open position with realistic slippage
                # Add 2-5 basis points slippage for liquid crypto pairs (0.02-0.05%)
                slippage_bp = 3.0  # 3 basis points = 0.03% realistic slippage
                return await self.order_executor.execute_open_position(
                    symbol=symbol,
                    side=side,
                    position_size=pos_size,
                    current_price=current_price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    leverage=leverage,
                    slippage_bp=slippage_bp
                )
                
            # For exit signals, close the position
            elif action == "exit":
                print(f"[ExecutionEngine] Exiting position for {symbol}")
                return await self.order_executor.execute_close_position(symbol)
                
            # Invalid action type
            print(f"[ExecutionEngine] ❌ Invalid action type: {action}")
            return {
                "status": "error",
                "reason": f"Invalid action type: {action}",
                "symbol": symbol
            }
            
        except Exception as e:
            error_msg = f"Error processing signal for {signal.symbol if hasattr(signal, 'symbol') else 'unknown'}: {str(e)}"
            print(f"[ExecutionEngine] ❌ {error_msg}")
            print(traceback.format_exc())
            return {
                "status": "error",
                "reason": error_msg,
                "symbol": signal.symbol if hasattr(signal, "symbol") else "unknown"
            }
    
    async def validate_signal(self, signal, current_price: float) -> Dict[str, Any]:
        """Validate a signal against risk parameters.
        
        Args:
            signal: Trading signal to validate.
            current_price: Current market price.
            
        Returns:
            Dictionary with validation result and metadata.
        """
        symbol = signal.symbol
        
        try:
            # For "hold" or "exit" actions, we don't need risk validation
            if signal.action != "open":
                return {
                    "valid": True,
                    "reason": f"No validation needed for {signal.action} action"
                }
                
            # First check if we're already using too much capital across all positions
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            max_allocation = self.portfolio_manager.total_capital * 0.5  # Max 50% of total capital
            
            if portfolio_summary['allocated_capital'] > max_allocation:
                print(f"[ExecutionEngine] ⚠️ Too much capital already allocated: "
                      f"${portfolio_summary['allocated_capital']:.2f} > ${max_allocation:.2f} (50% of capital)")
                return {
                    "valid": False,
                    "reason": f"Capital allocation limit reached: ${portfolio_summary['allocated_capital']:.2f} > ${max_allocation:.2f}"
                }
            
            # Calculate stop loss price if not provided in signal
            stop_loss_price = None
            if 'stop_loss_price' in signal.metadata:
                stop_loss_price = signal.metadata['stop_loss_price']
            elif 'stop_loss_pct' in signal.metadata:
                stop_pct = signal.metadata['stop_loss_pct']
                stop_loss_price = current_price * (1 - stop_pct) if signal.side == "buy" else current_price * (1 + stop_pct)
            else:
                # Use default risk parameters for stop loss
                risk_params = self.risk_manager.get_risk_parameters(symbol)
                stop_pct = risk_params.trailing_stop_pct
                stop_loss_price = current_price * (1 - stop_pct) if signal.side == "buy" else current_price * (1 + stop_pct)
            
            # Validate trade with risk manager
            risk_result = self.risk_manager.validate_trade(
                symbol=symbol,
                action=signal.action,
                side=signal.side,
                price=current_price,
                stop_loss_price=stop_loss_price
            )
            
            # Enhance signal with position information if it's valid
            if risk_result.get('valid', False) and 'position_info' in risk_result:
                position_info = risk_result['position_info']
                signal.metadata['position_size'] = position_info['size_contracts']
                signal.metadata['position_leverage'] = position_info['leverage']
                signal.metadata['stop_loss_price'] = position_info['stop_loss_price']
                signal.metadata['take_profit_price'] = position_info['take_profit_price']
                signal.metadata['position_value'] = position_info['size_usdt']
                
                # Print the position information including take profit
                print(f"[ExecutionEngine] Position details for {symbol}:")
                print(f"  - Size: {position_info['size_contracts']:.6f} contracts (${position_info['size_usdt']:.2f})")
                print(f"  - Leverage: {position_info['leverage']}x")
                print(f"  - Stop loss: {position_info['stop_loss_price']:.2f}")
                print(f"  - Take profit: {position_info['take_profit_price']:.2f}")
            
            return risk_result
            
        except Exception as e:
            print(f"[ExecutionEngine] Error validating signal for {symbol}: {e}")
            return {
                "valid": False,
                "reason": f"Error during validation: {str(e)}"
            }
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close an open position for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Dictionary with execution result.
        """
        try:
            return await self.order_executor.execute_close_position(symbol)
        except Exception as e:
            print(f"[ExecutionEngine] Error closing position for {symbol}: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "symbol": symbol
            }
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions.
        
        This is typically called during shutdown to ensure all positions
        are properly closed before exiting.
        
        Returns:
            Dictionary with execution results by symbol.
        """
        try:
            print("\n[ExecutionEngine] Closing all open positions...")
            
            # Get active symbols from portfolio manager
            active_symbols = self.portfolio_manager.get_active_symbols()
            
            if not active_symbols:
                print("[ExecutionEngine] No active positions to close")
                return {"status": "success", "message": "No active positions to close"}
            
            # Close each position
            results = {}
            for symbol in active_symbols:
                print(f"[ExecutionEngine] Closing position for {symbol}...")
                results[symbol] = await self.close_position(symbol)
            
            # Also check directly with exchange for any remaining positions
            try:
                exchange_positions = await self.binance_client.get_open_positions()
                for position in exchange_positions:
                    symbol = position.get('symbol')
                    if symbol and symbol not in results:
                        print(f"[ExecutionEngine] Found additional position for {symbol}, closing...")
                        results[symbol] = await self.close_position(symbol)
            except Exception as e:
                print(f"[ExecutionEngine] ⚠️ Error checking exchange positions: {e}")
            
            print("[ExecutionEngine] All positions closed")
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Error closing all positions: {str(e)}"
            print(f"[ExecutionEngine] ❌ {error_msg}")
            return {
                "status": "error",
                "reason": error_msg
            }
    
    def update_daily_pnl(self, symbol: str, pnl: float) -> bool:
        """Update daily PnL for risk tracking.
        
        Args:
            symbol: Trading pair symbol.
            pnl: Current unrealized PnL.
            
        Returns:
            True if trading should continue, False if max drawdown hit.
        """
        return self.risk_manager.update_daily_pnl(symbol, pnl)
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics.
        
        Returns:
            Dictionary with risk metrics.
        """
        return self.risk_manager.get_risk_metrics()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.
        
        Returns:
            Dictionary with portfolio summary.
        """
        return self.portfolio_manager.get_portfolio_summary()
    
    def should_close_position(self, symbol: str, entry_price: float, 
                             current_price: float, side: str,
                             trailing_high_low: Optional[float] = None) -> Dict[str, Any]:
        """Check if a position should be closed based on risk parameters.
        
        Args:
            symbol: Trading pair symbol.
            entry_price: Entry price for the position.
            current_price: Current market price.
            side: Position side ("buy" for long, "sell" for short).
            trailing_high_low: Highest/lowest price since entry for trailing stop.
            
        Returns:
            Dictionary with decision and reason.
        """
        return self.risk_manager.should_close_position(
            symbol, entry_price, current_price, side, trailing_high_low
        )
