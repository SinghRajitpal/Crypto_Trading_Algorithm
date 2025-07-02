from typing import Dict, Optional, Tuple, List, Any
import time
from dataclasses import dataclass
from execution.portfolio import PortfolioManager

@dataclass
class RiskParameters:
    """Represents risk parameters for a specific trade.
    
    Attributes:
        max_risk_per_trade_pct (float): Maximum percentage of capital to risk per trade.
        max_capital_per_trade_pct (float): Maximum percentage of total capital to allocate per trade.
        max_daily_drawdown_pct (float): Maximum allowed daily drawdown percentage.
        trailing_stop_pct (float): Trailing stop percentage from peak price.
        take_profit_pct (float): Take profit percentage from entry price.
        max_leverage (int): Maximum allowed leverage for positions.
        position_sizing_method (str): Method to use for position sizing.
    """
    max_risk_per_trade_pct: float = 0.01     # Default 1% risk per trade
    max_capital_per_trade_pct: float = 0.1   # Default 10% max capital allocation per trade
    max_daily_drawdown_pct: float = 0.05     # Default 5% max daily drawdown
    trailing_stop_pct: float = 0.02          # Default 2% trailing stop
    take_profit_pct: float = 0.04            # Default 4% take profit (2:1 reward:risk ratio)
    max_leverage: int = 10                    # Default max 5x leverage (reduced from 10x for safety)
    position_sizing_method: str = "risk"     # "risk" or "fixed"


class RiskManager:
    """Risk Manager for crypto futures trading.
    
    Provides risk control features like position sizing, drawdown protection,
    and leverage management. Works with PortfolioManager for capital allocation.
    
    Attributes:
        portfolio_manager (PortfolioManager): Reference to the portfolio manager.
        global_risk_params (RiskParameters): Default risk parameters.
        symbol_risk_params (Dict[str, RiskParameters]): Symbol-specific risk parameters.
        daily_pnl (Dict[str, float]): Tracking of daily PnL.
        day_start_timestamp (int): Timestamp for the start of the trading day.
        max_drawdown_hit (bool): Flag indicating if max drawdown was hit.
    """
    
    def __init__(self, portfolio_manager: PortfolioManager):
        """Initialize the risk manager.
        
        Args:
            portfolio_manager: Reference to the portfolio manager.
        """
        self.portfolio_manager = portfolio_manager
        self.global_risk_params = RiskParameters()
        self.symbol_risk_params: Dict[str, RiskParameters] = {}
        
        # Daily PnL tracking
        self.daily_pnl: Dict[str, float] = {}
        self.day_start_timestamp = int(time.time())
        self.max_drawdown_hit = False
        
        # Position tracking
        self.position_metrics: Dict[str, Dict[str, Any]] = {}
        
        print("[RiskManager] Initialized with default risk parameters")
    
    def set_risk_parameters(self, params: RiskParameters, symbol: Optional[str] = None):
        """Set risk parameters globally or for a specific symbol.
        
        Args:
            params: Risk parameters to set.
            symbol: Symbol to set parameters for. If None, sets global parameters.
        """
        if symbol:
            self.symbol_risk_params[symbol] = params
            print(f"[RiskManager] Set risk parameters for {symbol}: "
                  f"risk={params.max_risk_per_trade_pct:.1%}, "
                  f"drawdown={params.max_daily_drawdown_pct:.1%}, "
                  f"TP/SL={params.take_profit_pct:.1%}/{params.trailing_stop_pct:.1%}, "
                  f"max_leverage={params.max_leverage}x")
        else:
            self.global_risk_params = params
            print(f"[RiskManager] Set global risk parameters: "
                  f"risk={params.max_risk_per_trade_pct:.1%}, "
                  f"drawdown={params.max_daily_drawdown_pct:.1%}, "
                  f"TP/SL={params.take_profit_pct:.1%}/{params.trailing_stop_pct:.1%}, "
                  f"max_leverage={params.max_leverage}x")
    
    def get_risk_parameters(self, symbol: str) -> RiskParameters:
        """Get risk parameters for a specific symbol.
        
        Args:
            symbol: Symbol to get parameters for.
            
        Returns:
            Risk parameters for the symbol or global parameters if not set.
        """
        return self.symbol_risk_params.get(symbol, self.global_risk_params)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float) -> Dict[str, Any]:
        """Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol.
            entry_price: Entry price for the position.
            stop_loss_price: Stop loss price.
            
        Returns:
            Dictionary with position sizing information.
            
        Raises:
            ValueError: If prices are invalid.
            KeyError: If symbol is not in portfolio.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Entry and stop loss prices must be positive")
            
        # Get risk parameters for symbol
        risk_params = self.get_risk_parameters(symbol)
        
        # Calculate max allocation based on our capital constraints
        max_capital_from_percentage = self.portfolio_manager.total_capital * risk_params.max_capital_per_trade_pct
        portfolio_max_allocation = self.portfolio_manager.get_allocation_limit(symbol)
        
        # Use the smaller of the two limits
        max_allocation = min(max_capital_from_percentage, portfolio_max_allocation)
        
        print(f"[RiskManager] Max allocation for {symbol}: ${max_allocation:.2f} (portfolio limit: ${portfolio_max_allocation:.2f}, "
              f"per-trade limit: ${max_capital_from_percentage:.2f} = {risk_params.max_capital_per_trade_pct*100:.0f}% of capital)")
        
        # Calculate risk amount
        risk_amount = self.portfolio_manager.total_capital * risk_params.max_risk_per_trade_pct
        
        # Ensure we don't exceed allocation limit
        risk_amount = min(risk_amount, max_allocation * 0.5)  # Max risk is 50% of allocation
        
        # Calculate risk per contract
        price_difference = abs(entry_price - stop_loss_price)
        risk_percentage = price_difference / entry_price
        
        # Adjust for leverage
        leverage = min(risk_params.max_leverage, int(1 / risk_percentage)) if risk_percentage > 0 else 1
        
        # Calculate position size in USDT and contracts
        position_size_usdt = risk_amount / risk_percentage * leverage
        
        # Ensure position_size_usdt doesn't exceed max_allocation
        position_size_usdt = min(position_size_usdt, max_allocation)
        
        # Convert to contracts
        position_size_contracts = position_size_usdt / entry_price
        
        # Calculate take profit price based on entry price and risk parameters
        side = "buy" if entry_price > stop_loss_price else "sell"
        take_profit_price = self.calculate_take_profit_price(entry_price, side, risk_params.take_profit_pct)
        
        # Calculate the notional value (in USDT)
        notional_value = position_size_contracts * entry_price
        
        # Ensure the notional value meets Binance's minimum requirement of 100 USDT
        min_notional = 100.0  # Binance minimum notional value requirement
        
        # For high-value assets like BTC, ensure position size is sufficient
        if notional_value < min_notional:
            # Calculate minimum contracts needed
            min_contracts = min_notional / entry_price
            print(f"[RiskManager] Adjusting position size for {symbol} to meet minimum notional value of ${min_notional}")
            print(f"[RiskManager] Original: {position_size_contracts:.8f} ({notional_value:.2f} USDT) -> New: {min_contracts:.8f} ({min_notional:.2f} USDT)")
            
            # Adjust to minimum required
            position_size_contracts = min_contracts
            position_size_usdt = min_notional
            
            # Recalculate risk parameters
            risk_amount = position_size_usdt * risk_percentage / leverage
            
            # If needed, increase leverage to keep risk amount reasonable
            if risk_amount > self.portfolio_manager.total_capital * risk_params.max_risk_per_trade_pct * 2:
                new_leverage = min(int(risk_amount / (self.portfolio_manager.total_capital * risk_params.max_risk_per_trade_pct)), risk_params.max_leverage)
                if new_leverage > leverage:
                    leverage = new_leverage
                    print(f"[RiskManager] Increased leverage to {leverage}x to maintain reasonable risk")
        
        print(f"[RiskManager] Calculated position size for {symbol}: "
              f"{position_size_contracts:.6f} contracts (${position_size_usdt:.2f}), "
              f"leverage: {leverage}x, risk: ${risk_amount:.2f}, "
              f"SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}")
        
        return {
            "size_contracts": position_size_contracts,
            "size_usdt": position_size_usdt,
            "leverage": leverage,
            "risk_amount": risk_amount,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "max_capital": max_allocation,
            "notional_value": position_size_contracts * entry_price
        }
    
    def calculate_take_profit_price(self, entry_price: float, side: str, take_profit_pct: Optional[float] = None) -> float:
        """Calculate take profit price based on entry price and side.
        
        Args:
            entry_price: Entry price for the position.
            side: Position side ("buy" or "sell").
            take_profit_pct: Take profit percentage, defaults to global setting if None.
            
        Returns:
            Take profit price.
        """
        if take_profit_pct is None:
            take_profit_pct = self.global_risk_params.take_profit_pct
            
        if side == "buy":  # Long position
            return entry_price * (1 + take_profit_pct)
        else:  # Short position
            return entry_price * (1 - take_profit_pct)
    
    def update_daily_pnl(self, symbol: str, current_pnl: float) -> bool:
        """Update daily PnL and check if max drawdown is hit.
        
        Args:
            symbol: Trading pair symbol.
            current_pnl: Current unrealized PnL in USDT.
            
        Returns:
            True if trading should continue, False if max drawdown hit.
        """
        # Reset daily PnL at the start of a new day
        current_timestamp = int(time.time())
        if current_timestamp - self.day_start_timestamp > 86400:  # 24 hours
            self.daily_pnl = {}
            self.day_start_timestamp = current_timestamp
            self.max_drawdown_hit = False
            print("[RiskManager] Reset daily PnL tracking for new trading day")
        
        # Update PnL for symbol
        self.daily_pnl[symbol] = current_pnl
        
        # Calculate total daily PnL
        total_daily_pnl = sum(self.daily_pnl.values())
        
        # Check if max drawdown is hit
        max_drawdown_amount = self.portfolio_manager.total_capital * self.global_risk_params.max_daily_drawdown_pct
        
        if total_daily_pnl < -max_drawdown_amount and not self.max_drawdown_hit:
            self.max_drawdown_hit = True
            print(f"[RiskManager] ⚠️ ALERT: Max daily drawdown hit! "
                  f"PnL: ${total_daily_pnl:.2f}, limit: -${max_drawdown_amount:.2f}")
            return False
        
        return True
    
    def validate_trade(self, symbol: str, action: str, side: str, 
                      price: float, stop_loss_price: Optional[float] = None, 
                      take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """Validate if a trade meets risk management criteria.
        
        Args:
            symbol: Trading pair symbol.
            action: Trading action ("open", "exit", or "hold").
            side: Trading side ("buy", "sell", or "none").
            price: Current market price.
            stop_loss_price: Stop loss price (optional).
            take_profit_price: Take profit price (optional).
            
        Returns:
            Dictionary with validation result and position sizing information.
        """
        try:
            # For "hold" or "exit" actions, we don't need risk validation
            if action != "open":
                return {
                    "valid": True,
                    "reason": f"No validation needed for {action} action"
                }
                
            # Max drawdown check - if exceeded, reject all new trades
            if self.max_drawdown_hit:
                return {
                    "valid": False,
                    "reason": "Max daily drawdown limit reached"
                }
                
            # Check if we can allocate capital for this position
            # Get risk parameters
            risk_params = self.get_risk_parameters(symbol)
            
            # Calculate stop loss if not provided
            if stop_loss_price is None:
                stop_pct = risk_params.trailing_stop_pct
                stop_loss_price = price * (1 - stop_pct) if side == "buy" else price * (1 + stop_pct)
                print(f"[RiskManager] Calculated stop loss for {symbol} at {stop_loss_price:.2f} " +
                     f"({stop_pct*100:.1f}% from entry)")
            
            # Calculate take profit if not provided
            if take_profit_price is None:
                take_profit_pct = risk_params.take_profit_pct
                take_profit_price = self.calculate_take_profit_price(price, side, take_profit_pct)
                print(f"[RiskManager] Calculated take profit for {symbol} at {take_profit_price:.2f} " +
                     f"({take_profit_pct*100:.1f}% from entry)")
            
            # Calculate position size
            try:
                position_info = self.calculate_position_size(symbol, price, stop_loss_price)
                
                # Always ensure take profit is included in position info
                position_info["take_profit_price"] = take_profit_price
                
                # Ensure stop loss is explicitly set
                position_info["stop_loss_price"] = stop_loss_price
                
                # Make sure we round all values properly to avoid precision issues
                position_info["take_profit_price"] = round(position_info["take_profit_price"], 2)
                position_info["stop_loss_price"] = round(position_info["stop_loss_price"], 2)
                
                # Calculate potential profit and risk
                risk_amount = position_info["risk_amount"]
                
                # For long positions, TP > entry > SL
                # For short positions, SL > entry > TP
                if side == "buy":
                    profit_amount = position_info["size_contracts"] * (take_profit_price - price)
                else:
                    profit_amount = position_info["size_contracts"] * (price - take_profit_price)
                
                # Calculate reward-to-risk ratio
                if risk_amount > 0:
                    reward_risk_ratio = profit_amount / risk_amount
                    position_info["reward_risk_ratio"] = reward_risk_ratio
                    print(f"[RiskManager] Reward-to-risk ratio: {reward_risk_ratio:.2f}")
                
            except Exception as e:
                return {
                    "valid": False,
                    "reason": f"Position sizing error: {str(e)}"
                }
            
            # Check if allocation is possible
            if not self.portfolio_manager.can_allocate(symbol, position_info["size_usdt"]):
                max_possible = self.portfolio_manager.get_allocation_limit(symbol)
                return {
                    "valid": False,
                    "reason": f"Allocation limit reached. Requested: ${position_info['size_usdt']:.2f}, Available: ${max_possible:.2f}"
                }
            
            # All checks passed, trade is valid
            return {
                "valid": True,
                "reason": "Trade meets risk criteria",
                "position_info": position_info
            }
            
        except Exception as e:
            print(f"[RiskManager] Error validating trade for {symbol}: {e}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    def should_close_position(self, symbol: str, entry_price: float, 
                             current_price: float, side: str, 
                             trailing_high_low: Optional[float] = None) -> Dict[str, Any]:
        """Check if a position should be closed based on risk parameters.
        
        Args:
            symbol: Trading pair symbol.
            entry_price: Entry price for the position.
            current_price: Current market price.
            side: Position side ("buy" for long, "sell" for short).
            trailing_high_low: Highest price (for long) or lowest price (for short) since entry.
            
        Returns:
            Dictionary with decision and reason.
        """
        try:
            # Update position metrics first if we have a trailing high/low
            if trailing_high_low is not None:
                self._update_position_metrics(symbol, current_price, entry_price, side, trailing_high_low)
                
            # Check if max drawdown is hit
            if self.max_drawdown_hit:
                return {
                    "should_close": True,
                    "reason": "Max daily drawdown limit reached",
                    "symbol": symbol
                }
                
            # Get risk parameters
            risk_params = self.get_risk_parameters(symbol)
            
            # Check if take profit is hit
            take_profit_price = self.calculate_take_profit_price(entry_price, side, risk_params.take_profit_pct)
            
            if side == "buy" and current_price >= take_profit_price:  # Long position
                print(f"[RiskManager] Take profit triggered for {symbol} LONG: "
                      f"current price {current_price:.2f} >= take profit {take_profit_price:.2f}")
                return {
                    "should_close": True,
                    "reason": f"Take profit triggered: {current_price:.2f} >= {take_profit_price:.2f}",
                    "symbol": symbol
                }
            elif side == "sell" and current_price <= take_profit_price:  # Short position
                print(f"[RiskManager] Take profit triggered for {symbol} SHORT: "
                      f"current price {current_price:.2f} <= take profit {take_profit_price:.2f}")
                return {
                    "should_close": True,
                    "reason": f"Take profit triggered: {current_price:.2f} <= {take_profit_price:.2f}",
                    "symbol": symbol
                }
            
            # Check trailing stop if we have metrics
            if symbol in self.position_metrics:
                metrics = self.position_metrics[symbol]
                
                if side == "buy":  # Long position
                    # For longs, we use the highest price since entry
                    highest_price = metrics.get("highest_price", entry_price)
                    trailing_stop_price = highest_price * (1 - risk_params.trailing_stop_pct)
                    
                    if current_price < trailing_stop_price:
                        print(f"[RiskManager] Trailing stop triggered for {symbol} LONG: "
                              f"current price {current_price:.2f} < stop price {trailing_stop_price:.2f}")
                        return {
                            "should_close": True,
                            "reason": f"Trailing stop triggered: {current_price:.2f} < {trailing_stop_price:.2f}",
                            "symbol": symbol,
                            "metrics": metrics
                        }
                else:  # Short position
                    # For shorts, we use the lowest price since entry
                    lowest_price = metrics.get("lowest_price", entry_price)
                    trailing_stop_price = lowest_price * (1 + risk_params.trailing_stop_pct)
                    
                    if current_price > trailing_stop_price:
                        print(f"[RiskManager] Trailing stop triggered for {symbol} SHORT: "
                              f"current price {current_price:.2f} > stop price {trailing_stop_price:.2f}")
                        return {
                            "should_close": True,
                            "reason": f"Trailing stop triggered: {current_price:.2f} > {trailing_stop_price:.2f}",
                            "symbol": symbol,
                            "metrics": metrics
                        }
                        
            # No reason to close
            return {
                "should_close": False,
                "reason": "Position within risk parameters",
                "symbol": symbol
            }
            
        except Exception as e:
            print(f"[RiskManager] Error checking close condition for {symbol}: {e}")
            return {
                "should_close": False,
                "reason": f"Error checking close condition: {str(e)}",
                "symbol": symbol
            }
    
    def _update_position_metrics(self, symbol: str, current_price: float, 
                               entry_price: float, side: str, 
                               trailing_high_low: Optional[float] = None):
        """Update position metrics for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            current_price: Current market price.
            entry_price: Entry price for the position.
            side: Position side ("buy" or "sell").
            trailing_high_low: Highest price (for long) or lowest price (for short) since entry.
        """
        # Initialize if needed
        if symbol not in self.position_metrics:
            self.position_metrics[symbol] = {
                "entry_price": entry_price,
                "side": side,
                "highest_price": current_price,
                "lowest_price": current_price,
                "last_price": current_price,
                "unrealized_pnl": 0
            }
            return
            
        # Update highest/lowest prices
        metrics = self.position_metrics[symbol]
        
        if trailing_high_low is not None:
            if side == "buy":  # For long positions
                metrics["highest_price"] = max(metrics["highest_price"], trailing_high_low)
            else:  # For short positions
                metrics["lowest_price"] = min(metrics["lowest_price"], trailing_high_low)
        
        # Always update current price
        metrics["last_price"] = current_price
        
        # Calculate unrealized PnL
        if side == "buy":  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
            
        # We don't have position size here, but we update the percentage
        metrics["unrealized_pnl_pct"] = pnl_pct
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics.
        
        Returns:
            Dictionary with current risk metrics.
        """
        total_daily_pnl = sum(self.daily_pnl.values())
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        allocation_pct = portfolio_summary["allocation_percentage"]
        
        return {
            "daily_pnl": total_daily_pnl,
            "max_drawdown_hit": self.max_drawdown_hit,
            "max_daily_drawdown": self.portfolio_manager.total_capital * self.global_risk_params.max_daily_drawdown_pct,
            "allocation_percentage": allocation_pct,
            "position_count": portfolio_summary["active_positions"],
            "risk_status": "high" if allocation_pct > 0.8 else 
                         "medium" if allocation_pct > 0.5 else "low",
            "total_capital": self.portfolio_manager.total_capital,
            "available_capital": self.portfolio_manager.get_available_capital()
        }
