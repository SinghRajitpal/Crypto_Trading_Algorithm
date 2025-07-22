from typing import Dict, Any, Optional
import time

class OrderExecutor:
    """Handles execution of trading orders.
    
    This class acts as a bridge between trading signals and the exchange client.
    It processes trades while respecting risk and portfolio limitations.
    
    Attributes:
        binance_client: Binance client for executing orders.
        portfolio_manager: Reference to portfolio manager for allocation.
        risk_manager: Reference to risk manager for risk assessment.
        default_leverage: Default leverage to use for positions.
        default_margin_type: Default margin type for positions.
    """
    
    def __init__(self, binance_client, portfolio_manager, risk_manager):
        """Initialize the order executor.
        
        Args:
            binance_client: Binance client for executing orders.
            portfolio_manager: Reference to portfolio manager.
            risk_manager: Reference to risk manager.
        """
        self.binance_client = binance_client
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.execution_history = []
        self.default_leverage = 5  # Default leverage of 5x
        self.default_margin_type = "isolated"  # Default to isolated margin
        
        print("[OrderExecutor] Initialized")
    
    async def execute_open_position(self, symbol, side, position_size, current_price=None, stop_loss_price=None, take_profit_price=None, leverage=None, margin_type=None):
        """Execute opening a position.
        
        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            position_size: Size of the position.
            current_price: Current market price.
            stop_loss_price: Stop loss price level.
            take_profit_price: Take profit price level.
            leverage: Leverage to use (e.g. 5x, 10x)
            margin_type: 'isolated' or 'cross'
            
        Returns:
            Dictionary with execution result.
        """
        try:
            # Default to the preset values if not provided
            if leverage is None:
                leverage = self.default_leverage
                
            if margin_type is None:
                margin_type = self.default_margin_type
            
            # Calculate notional value and **margin** required
            notional_value = position_size * current_price
            margin_required = notional_value / leverage if leverage else notional_value
            
            # --------------------------------------------------------------
            # 1) Reserve portfolio allocation BEFORE sending the order
            # --------------------------------------------------------------
            if not self.portfolio_manager.reserve_allocation(symbol, margin_required):
                print(f"[OrderExecutor] ❌ Allocation reserve failed for {symbol} – exceeds limits")
                return {
                    "status": "rejected",
                    "reason": "Allocation limit reached",
                    "symbol": symbol
                }
            
            # Log the execution
            print(f"[OrderExecutor] Opening {side.upper()} position for {symbol}: {position_size:.6f} contracts")
            print(f"[OrderExecutor] Notional: ${notional_value:.2f} (margin: ${margin_required:.2f}), Leverage: {leverage}x")
            
            # Log Stop Loss and Take Profit details with clear formatting
            print(f"[OrderExecutor] Risk Management:")
            if stop_loss_price:
                # Calculate percentage from entry
                stop_loss_pct = abs(stop_loss_price - current_price) / current_price * 100
                print(f"  - Stop Loss: {stop_loss_price:.2f} ({stop_loss_pct:.2f}% from entry)")
            else:
                print(f"  - Stop Loss: None")
            
            if take_profit_price:
                # Calculate percentage from entry
                take_profit_pct = abs(take_profit_price - current_price) / current_price * 100
                print(f"  - Take Profit: {take_profit_price:.2f} ({take_profit_pct:.2f}% from entry)")
            else:
                print(f"  - Take Profit: None")
            
            # Ensure stop loss and take profit are properly rounded
            if stop_loss_price is not None:
                stop_loss_price = round(stop_loss_price, 2)
            
            if take_profit_price is not None:
                take_profit_price = round(take_profit_price, 2)
            
            # Execute the order through Binance client
            result = await self.binance_client.open_position(
                symbol=symbol,
                side=side,
                amount=position_size,
                price=None,  # Using market order
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                leverage=leverage,
                margin_type=margin_type
            )
            
            # Check if execution was successful
            if result.get("status") == "error":
                error_message = result.get("error", "Unknown error")
                print(f"[OrderExecutor] ❌ Failed to open position for {symbol}: {error_message}")
                
                # Add specific guidance based on error type
                if "notional" in error_message.lower():
                    print(f"[OrderExecutor] 💡 Recommendation: Increase position size or leverage to meet minimum notional value requirements")
                    # Attempt to calculate how much would be needed
                    if current_price and notional_value < 100:
                        min_size = 100 / current_price
                        print(f"[OrderExecutor] 💡 Minimum size needed: {min_size:.6f} at current price")
                elif "lot_size" in error_message.lower():
                    print(f"[OrderExecutor] 💡 Recommendation: Adjust position size to meet lot size requirements")
                elif "hedge" in error_message.lower() or "position side" in error_message.lower():
                    print(f"[OrderExecutor] 💡 Order requires position side parameter. Your account is in hedge mode.")
                
                # Roll-back reserved capital because order failed
                try:
                    self.portfolio_manager.release_allocation(symbol, margin_required)
                except Exception:
                    pass
                
                # Record the failed execution
                execution_record = {
                    "timestamp": int(time.time() * 1000),
                    "symbol": symbol,
                    "side": side,
                    "action": "open",
                    "status": "failed",
                    "size": position_size,
                    "price": current_price,
                    "reason": error_message
                }
                self.execution_history.append(execution_record)
                
                return {
                    "status": "error",
                    "symbol": symbol,
                    "action": "open",
                    "side": side,
                    "reason": error_message,
                    "recommendation": "Adjust position size or leverage to meet exchange requirements"
                }
                
            # Success - position opened
            print(f"[OrderExecutor] ✅ Successfully opened {side.upper()} position for {symbol}")
            
            # Check if stop loss and take profit orders were created
            has_sl = result.get('position', {}).get('stop_loss') is not None
            has_tp = result.get('position', {}).get('take_profit') is not None
            
            if has_sl and has_tp:
                print(f"[OrderExecutor] ✅ Stop loss and take profit orders successfully placed")
            elif has_sl:
                print(f"[OrderExecutor] ✅ Stop loss order placed, but take profit order failed")
            elif has_tp:
                print(f"[OrderExecutor] ✅ Take profit order placed, but stop loss order failed")
            else:
                print(f"[OrderExecutor] ⚠️ Position opened but stop loss and take profit orders failed")
            
            # Record the successful execution
            execution_record = {
                "timestamp": int(time.time() * 1000),
                "symbol": symbol,
                "side": side,
                "action": "open",
                "status": "success",
                "size": position_size,
                "leverage": leverage,
                "price": current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "has_sl_order": has_sl,
                "has_tp_order": has_tp
            }
            self.execution_history.append(execution_record)
            
            return {
                "status": "success",
                "symbol": symbol,
                "action": "open",
                "side": side,
                "size": position_size,
                "order": result.get("order", {}),
                "position": result.get("position", {})
            }
            
        except Exception as e:
            error_message = f"Error executing open position: {str(e)}"
            print(f"[OrderExecutor] ❌ {error_message}")
            
            # Roll-back reserved allocation (best-effort)
            try:
                self.portfolio_manager.release_allocation(symbol, margin_required if 'margin_required' in locals() else None)
            except Exception:
                pass
            
            # Record the failed execution
            execution_record = {
                "timestamp": int(time.time() * 1000),
                "symbol": symbol,
                "side": side,
                "action": "open",
                "status": "failed",
                "size": position_size,
                "price": current_price,
                "reason": str(e)
            }
            self.execution_history.append(execution_record)
            
            return {
                "status": "error",
                "symbol": symbol,
                "action": "open",
                "side": side,
                "reason": error_message
            }
    
    async def execute_close_position(self, symbol: str, side: str = None):
        """Close an existing position.
        
        Args:
            symbol: Trading pair symbol.
            side: Position side to close (default: close all positions).
            
        Returns:
            Dictionary with execution result.
        """
        try:
            print(f"\n[OrderExecutor] 🔒 Closing position for {symbol}")
            if side:
                print(f"[OrderExecutor] Closing {side.upper()} side only")
            
            # Check current allocation before closing
            current_allocation = 0
            try:
                current_allocation = self.portfolio_manager.get_symbol_allocation(symbol)
                if current_allocation > 0:
                    print(f"[OrderExecutor] Current allocation: {current_allocation:.2f} USDT")
                else:
                    print(f"[OrderExecutor] No allocation found for {symbol}")
            except Exception as e:
                print(f"[OrderExecutor] ⚠️ Could not retrieve allocation: {e}")
            
            try:
                positions = await self.binance_client.get_open_positions(symbol)
                if not positions:
                    print(f"[OrderExecutor] No open position found for {symbol}")
                    # Still release allocation if we have it (cleanup orphaned allocations)
                    if current_allocation > 0:
                        released = self.portfolio_manager.release_allocation(symbol)
                        print(f"[OrderExecutor] Released orphaned allocation of {released:.2f} USDT")
                    return {"status": "no_position", "symbol": symbol}
                
                # Log position details before closing
                pos = positions[0]
                position_side = 'long' if float(pos.get('contracts', 0)) > 0 else 'short'
                print(f"[OrderExecutor] Found {position_side.upper()} position:")
                print(f"  - Size: {abs(float(pos.get('contracts', 0))):.6f} contracts")
                print(f"  - Entry Price: {pos.get('entryPrice', 'N/A')}")
                print(f"  - Unrealized PnL: {pos.get('unrealizedPnl', 'N/A')}")
                
            except Exception as e:
                print(f"[OrderExecutor] ⚠️ Error checking positions: {e}")
                # Continue with close attempt even if check fails
            
            # Execute the close with realistic slippage
            print(f"[OrderExecutor] Sending close request to exchange...")
            slippage_bp = 3.0  # 3 basis points = 0.03% realistic slippage
            result = await self.binance_client.close_position(symbol, side, slippage_bp=slippage_bp)
            
            if result.get("status") in ["closed", "no_position"]:
                # Release allocation from portfolio manager
                if current_allocation > 0:
                    released = self.portfolio_manager.release_allocation(symbol)
                    print(f"[OrderExecutor] ✅ Released {released:.2f} USDT allocation for {symbol}")
                
                print(f"[OrderExecutor] ✅ Successfully closed position for {symbol}")
                
                # Add to execution history
                self.execution_history.append({
                    "timestamp": time.time(),
                    "action": "close",
                    "symbol": symbol,
                    "side": side
                })
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "result": result
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"[OrderExecutor] ❌ Failed to close position for {symbol}: {error_msg}")
                return {
                    "status": "error",
                    "reason": error_msg,
                    "symbol": symbol
                }
                
        except Exception as e:
            error_msg = f"Error closing position: {str(e)}"
            print(f"[OrderExecutor] ❌ {error_msg}")
            return {
                "status": "error",
                "reason": error_msg,
                "symbol": symbol
            }
    
    async def process_signal_execution(self, signal, current_price, position_info=None):
        """Process a trading signal for execution.
        
        Args:
            signal: Trading signal object.
            current_price: Current market price.
            position_info: Optional position information from risk validation.
            
        Returns:
            Dictionary with execution result.
        """
        try:
            symbol = signal.symbol
            action = signal.action
            side = signal.side
            
            # Exit if the action is 'hold'
            if action == "hold" or side == "none":
                return {
                    "status": "hold",
                    "reason": "No action to take",
                    "symbol": symbol
                }
                
            # Process open position
            if action == "open":
                # Ensure we have position info for risk management
                if not position_info:
                    print(f"[OrderExecutor] ❌ Cannot open position without risk validation info for {symbol}")
                    return {
                        "status": "error",
                        "reason": "Missing position information",
                        "symbol": symbol
                    }
                
                # Extract position details
                position_size = position_info.get("size_contracts")
                stop_loss_price = position_info.get("stop_loss_price")
                take_profit_price = position_info.get("take_profit_price")
                leverage = position_info.get("leverage", self.default_leverage)
                
                # Print full position details
                print(f"[OrderExecutor] Position details for {symbol}:")
                print(f"  - Size: {position_size:.6f} contracts")
                print(f"  - Leverage: {leverage}x")
                print(f"  - Stop Loss: {stop_loss_price}")
                print(f"  - Take Profit: {take_profit_price}")
                
                # Execute the position
                return await self.execute_open_position(
                    symbol=symbol,
                    side=side,
                    position_size=position_size,
                    current_price=current_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    leverage=leverage
                )
                
            # Process close position
            elif action == "close":
                # For close action, we use the opposite side of the current position
                # which is implicitly handled by the exchange
                return await self.execute_close_position(symbol)
                
            # Unknown action
            else:
                print(f"[OrderExecutor] ❌ Unknown action: {action}")
                return {
                    "status": "error",
                    "reason": f"Unknown action: {action}",
                    "symbol": symbol
                }
                
        except Exception as e:
            print(f"[OrderExecutor] ❌ Error processing signal execution: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "symbol": signal.symbol if hasattr(signal, "symbol") else "unknown"
            }
    
    def get_execution_history(self) -> list:
        """Get execution history.
        
        Returns:
            List of execution records.
        """
        return self.execution_history
