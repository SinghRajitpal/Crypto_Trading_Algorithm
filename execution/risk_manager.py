from typing import Dict, Optional, Tuple, List, Any
import time
import numpy as np
from datetime import datetime, timedelta
import config


class ProductionRiskManager:
    """Production Risk Manager implementing the exact risk engine from the document.
    
    Core Features:
    - ATR-based position sizing with fractional Kelly criterion
    - Dynamic cost adjustments based on volatility
    - Regime-aware leverage scaling with drawdown/Sharpe/slope adjustments
    - Real-time risk monitoring and safeguards
    """
    
    def __init__(self, portfolio_manager):
        """Initialize the production risk manager.
        
        Args:
            portfolio_manager: Reference to the production portfolio manager.
        """
        self.portfolio_manager = portfolio_manager
        
        # Risk tracking data structures
        self.drawdown_history: List[Tuple[datetime, float]] = []  # 3-day drawdown history
        self.sharpe_history: List[Tuple[datetime, float]] = []  # 30-day Sharpe history
        self.equity_curve: List[Tuple[datetime, float]] = []  # 60-bar equity curve for slope
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.max_drawdown_hit = False
        
        print(f"[ProductionRisk] Initialized with {config.RISK_PER_TRADE_PCT*100}% risk, {config.KELLY_FRACTION} Kelly fraction")
    
    @property
    def risk_per_trade(self) -> float:
        """Get risk per trade as a decimal (0.008 for 0.8%)."""
        return config.RISK_PER_TRADE_PCT
    
    def calculate_dynamic_cost_adjustment(self, volatility_norm: float, entry_price: float, 
                                       position_size_contracts: float) -> Dict[str, float]:
        """Calculate comprehensive dynamic cost adjustment based on market conditions.
        
        Includes: trading fees, spread, slippage, commission, and funding costs.
        
        Args:
            volatility_norm: Normalized volatility measure.
            entry_price: Entry price for the position.
            position_size_contracts: Position size in contracts.
            
        Returns:
            Dictionary with detailed cost breakdown and total cost.
        """
        # Base costs (as percentages of notional value)
        trading_fee_pct = config.BASE_TRADING_FEE_PCT
        spread_pct = config.BASE_SPREAD_PCT  
        commission_pct = config.BASE_COMMISSION_PCT
        funding_pct = config.FUNDING_RATE_8H_PCT
        
        # Dynamic slippage increases with volatility and position size
        position_notional = position_size_contracts * entry_price
        
        # Slippage increases with volatility and position size (market impact)
        volatility_slippage = config.BASE_SLIPPAGE_PCT * (1 + volatility_norm * 0.5)
        
        # Market impact: larger positions have higher slippage
        market_impact_factor = min(2.0, max(1.0, position_notional / 10000))  # Scale with position size
        dynamic_slippage_pct = volatility_slippage * market_impact_factor
        
        # Total cost percentage
        total_cost_pct = trading_fee_pct + spread_pct + dynamic_slippage_pct + commission_pct + funding_pct
        
        # Convert to absolute USD costs
        cost_breakdown = {
            "trading_fee_usd": position_notional * trading_fee_pct,
            "spread_cost_usd": position_notional * spread_pct,
            "slippage_cost_usd": position_notional * dynamic_slippage_pct,
            "commission_usd": position_notional * commission_pct,
            "funding_cost_usd": position_notional * funding_pct,
            "total_cost_usd": position_notional * total_cost_pct,
            "total_cost_pct": total_cost_pct,
            "market_impact_factor": market_impact_factor
        }
        
        return cost_breakdown
    
    def calculate_position_size(self, symbol: str, allocated_capital: float, atr_value: float, 
                              entry_price: float, volatility_norm: float = 0.5) -> Dict[str, Any]:
        """Calculate position size using the document's exact formula with comprehensive cost accounting:
        Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001) - total_trading_costs
        
        Args:
            symbol: Trading pair symbol.
            allocated_capital: Capital allocated to this symbol from portfolio manager.
            atr_value: Current ATR value (raw price units).
            entry_price: Entry price for position.
            volatility_norm: Normalized volatility for cost adjustment.
            
        Returns:
            Dictionary with position sizing information and detailed cost breakdown.
        """
        # CRITICAL FIX: Normalize raw ATR to percentage volatility
        # ATR comes from algorithm strategies as raw price units, need to convert to percentage
        normalized_atr = atr_value / entry_price if entry_price > 0 else 0.02
        normalized_atr = max(0.005, min(normalized_atr, 0.15))  # Bound between 0.5% and 15%
        
        # Apply ATR floor from document
        atr_adjusted = max(normalized_atr, config.MIN_ATR_FLOOR)
        
        # STEP 1: Calculate base position size using Kelly formula
        # Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001)
        numerator = config.RISK_PER_TRADE_PCT * allocated_capital * config.KELLY_FRACTION
        base_position_size_usdt = numerator / atr_adjusted
        
        # STEP 2: Initial position size in contracts (for cost calculation)
        initial_position_contracts = base_position_size_usdt / entry_price if entry_price > 0 else 0
        
        # STEP 3: Calculate comprehensive trading costs
        cost_breakdown = self.calculate_dynamic_cost_adjustment(
            volatility_norm, entry_price, initial_position_contracts
        )
        
        # STEP 4: Subtract total costs from position size
        position_size_usdt = base_position_size_usdt - cost_breakdown["total_cost_usd"]
        
        # CRITICAL SAFETY: Ensure position size never exceeds allocated capital
        max_position_size = allocated_capital * 0.95  # Maximum 95% of allocated capital
        if position_size_usdt > max_position_size:
            position_size_usdt = max_position_size
            print(f"[ProductionRisk] Position capped at 95% of allocated capital: ${max_position_size:.2f}")
        
        # Ensure position size is positive
        position_size_usdt = max(position_size_usdt, 0)
        
        # Convert to contracts
        position_size_contracts = position_size_usdt / entry_price if entry_price > 0 else 0
        
        # STEP 5: Recalculate final costs with actual position size
        final_cost_breakdown = self.calculate_dynamic_cost_adjustment(
            volatility_norm, entry_price, position_size_contracts
        )
        
        # Calculate leverage using dynamic leverage engine
        leverage = self.calculate_dynamic_leverage(symbol, atr_adjusted)
        
        # Check minimum notional value requirement (Binance: $100)
        min_notional = 100.0  # Minimum order value in USD
        current_notional = position_size_contracts * entry_price  # Actual notional value
        
        if current_notional < min_notional and position_size_contracts > 0:
            # Scale up position to meet minimum notional requirement, but cap at allocated capital
            scale_factor = min_notional / current_notional
            scaled_position_usdt = position_size_usdt * scale_factor
            
            if scaled_position_usdt <= allocated_capital:
                position_size_usdt = scaled_position_usdt
                position_size_contracts *= scale_factor
                print(f"[ProductionRisk] Position scaled up {scale_factor:.2f}x to meet ${min_notional} minimum")
            else:
                print(f"[ProductionRisk] Cannot scale to minimum notional - would exceed allocated capital")
        
        # Check minimum amount precision (BTCUSDT: 0.001)
        min_amount = 0.001  # Minimum contract size for BTCUSDT
        if position_size_contracts < min_amount and position_size_contracts > 0:
            position_size_contracts = min_amount
            position_size_usdt = position_size_contracts * entry_price
            
            # Ensure this doesn't exceed allocated capital
            if position_size_usdt > allocated_capital:
                position_size_usdt = allocated_capital * 0.95
                position_size_contracts = position_size_usdt / entry_price
            
            print(f"[ProductionRisk] Position size adjusted to minimum precision: {position_size_contracts:.6f} contracts")
        
        # Calculate margin required with safety check
        margin_required = position_size_usdt / leverage if leverage > 0 else position_size_usdt
        
        # Final safety check: margin should not exceed allocated capital
        if margin_required > allocated_capital:
            margin_required = allocated_capital * 0.95
            position_size_usdt = margin_required * leverage
            position_size_contracts = position_size_usdt / entry_price
            print(f"[ProductionRisk] Margin capped at allocated capital")
        
        # Calculate stop loss and take profit distances based on ATR
        stop_loss_distance = config.ATR_STOP_MULTIPLIER * atr_adjusted
        take_profit_distance = config.RISK_REWARD_RATIO * stop_loss_distance
        
        print(f"[ProductionRisk] Position sizing for {symbol}:")
        print(f"[ProductionRisk]   Allocated capital: ${allocated_capital:.2f}")
        print(f"[ProductionRisk]   ATR: {atr_value:.6f} (floor-adjusted: {atr_adjusted:.6f})")
        print(f"[ProductionRisk]   Size: {position_size_contracts:.6f} contracts (${position_size_usdt:.2f})")
        print(f"[ProductionRisk]   Leverage: {leverage}x, Margin: ${margin_required:.2f}")
        print(f"[ProductionRisk]   Total costs: ${final_cost_breakdown['total_cost_usd']:.2f} ({final_cost_breakdown['total_cost_pct']:.2%})")
        print(f"[ProductionRisk]   Cost breakdown:")
        print(f"[ProductionRisk]     - Trading fees: ${final_cost_breakdown['trading_fee_usd']:.2f}")
        print(f"[ProductionRisk]     - Spread cost: ${final_cost_breakdown['spread_cost_usd']:.2f}")
        print(f"[ProductionRisk]     - Slippage: ${final_cost_breakdown['slippage_cost_usd']:.2f}")
        print(f"[ProductionRisk]     - Commission: ${final_cost_breakdown['commission_usd']:.2f}")
        print(f"[ProductionRisk]     - Funding (8h): ${final_cost_breakdown['funding_cost_usd']:.2f}")
        print(f"[ProductionRisk]   Position/Allocated: {(position_size_usdt/allocated_capital)*100:.1f}%")
        
        return {
            "size_contracts": position_size_contracts,
            "size_usdt": position_size_usdt,
            "margin_usdt": margin_required,
            "leverage": leverage,
            "risk_amount": numerator,
            "atr_adjusted": atr_adjusted,
            "stop_loss_distance": stop_loss_distance,
            "take_profit_distance": take_profit_distance,
            "entry_price": entry_price,
            "allocated_capital": allocated_capital,
            "cost_breakdown": final_cost_breakdown,
            "total_costs_usd": final_cost_breakdown["total_cost_usd"],
            "cost_pct": final_cost_breakdown["total_cost_pct"]
        }
    
    def calculate_dynamic_leverage(self, symbol: str, atr_value: float) -> int:
        """Calculate dynamic leverage using the document's formula:
        lev = min(10, 10 × min(1, target_vol/σ) × dd_factor × sharpe_factor × slope_factor)
        
        Args:
            symbol: Trading pair symbol.
            atr_value: Current ATR value (σ).
            
        Returns:
            Dynamic leverage value.
        """
        # FIXED: Base leverage calculation with proper volatility adjustment
        # lev_base = 10 × min(1, target_vol/σ)
        vol_adjustment = min(1.0, config.TARGET_VOLATILITY / max(atr_value, 0.001))
        base_leverage_float = config.MAX_LEVERAGE * vol_adjustment
        base_leverage = max(1, int(base_leverage_float))
        
        # FIXED: Drawdown factor - drawdowns are negative, so check absolute values
        # dd_factor = 0.8 if rolling 3-day DD >10%; 0.5 if >14%
        dd_3d = abs(self.get_rolling_drawdown_3d())  # Get absolute value for comparison
        if dd_3d > 0.14:
            dd_factor = 0.5  # Severely reduce leverage (>14% drawdown)
        elif dd_3d > 0.10:
            dd_factor = 0.8  # Moderately reduce leverage (>10% drawdown)
        else:
            dd_factor = 1.0  # No reduction (<10% drawdown)
        
        # FIXED: Sharpe factor with proper handling of all cases
        # sharpe_factor = max(0.5, min(1, rolling_30d_sharpe / 1.5))
        sharpe_30d = self.get_rolling_sharpe_30d()
        if sharpe_30d > 0:
            sharpe_factor = max(0.5, min(1.0, sharpe_30d / 1.5))
        elif sharpe_30d < 0:
            sharpe_factor = 0.5  # Minimum factor for negative Sharpe
        else:
            sharpe_factor = 0.7  # Conservative default for zero Sharpe
        
        # Equity curve slope factor: scale by 0.7 if slope < -5% over 60 bars
        equity_slope = self.get_equity_curve_slope()
        slope_factor = 0.7 if equity_slope < -0.05 else 1.0
        
        # CRITICAL FIX: Add funding rate adjustment as specified in document
        # "Reduce further by max(0, projected_8h_funding / 5%) for funding drag"
        funding_8h = self.get_projected_funding_8h(symbol)  # Get 8-hour projected funding
        funding_adjustment = max(0, funding_8h / 0.05)  # funding adjustment factor
        
        # FIXED: Apply the exact formula from document
        # lev = min(10, base_leverage × dd_factor × sharpe_factor × slope_factor) - funding_adjustment
        final_leverage_float = base_leverage * dd_factor * sharpe_factor * slope_factor - funding_adjustment
        final_leverage = max(1, min(int(final_leverage_float), config.MAX_LEVERAGE))
        
        print(f"[ProductionRisk] Dynamic leverage for {symbol}: {final_leverage}x")
        print(f"[ProductionRisk]   ATR: {atr_value:.6f}, Vol adjustment: {vol_adjustment:.3f}")
        print(f"[ProductionRisk]   Base: {base_leverage}x, DD factor: {dd_factor:.2f}, "
              f"Sharpe factor: {sharpe_factor:.2f}, Slope factor: {slope_factor:.2f}")
        print(f"[ProductionRisk]   Funding adjustment: {funding_adjustment:.3f}")
        print(f"[ProductionRisk]   DD 3d: {dd_3d:.3f}, Sharpe 30d: {sharpe_30d:.3f}, Equity slope: {equity_slope:.3f}")
        
        return final_leverage
    
    def calculate_stop_loss_take_profit(self, entry_price: float, side: str, atr_adjusted: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices using ATR multipliers.
        
        Document specification:
        - SL = Entry ± 1.8×ATR(30) (trail by 0.8×ATR)
        - TP = Entry ± 2×|Entry-SL| (1:2 risk-reward; partial exit 40% at 1:1)
        
        Args:
            entry_price: Entry price.
            side: Position side ("buy" or "sell").
            atr_adjusted: Floor-adjusted ATR value.
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        # SL = Entry ± 1.8×ATR (exactly as specified in document)
        stop_distance = config.ATR_STOP_MULTIPLIER * atr_adjusted
        
        # TP = Entry ± 2×|Entry-SL| (1:2 risk-reward as specified)
        tp_distance = config.RISK_REWARD_RATIO * stop_distance
        
        if side.lower() in ["buy", "long"]:
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + tp_distance
        else:  # sell/short
            stop_loss_price = entry_price + stop_distance
            take_profit_price = entry_price - tp_distance
        
        # FIXED: Ensure prices are positive and reasonable
        stop_loss_price = max(stop_loss_price, entry_price * 0.01)  # Minimum 1% of entry
        take_profit_price = max(take_profit_price, entry_price * 0.01)  # Minimum 1% of entry
        
        return stop_loss_price, take_profit_price
    
    def validate_trade(self, symbol: str, action: str, side: str, entry_price: float, 
                      atr_value: float) -> Dict[str, Any]:
        """Validate a trade against production risk criteria.
        
        Args:
            symbol: Trading pair symbol.
            action: Trade action ("open" or "close").
            side: Position side ("buy" or "sell").
            entry_price: Entry price for the trade.
            atr_value: Current ATR value (raw ATR in price units).
            
        Returns:
            Validation result dictionary.
        """
        # Get allocated capital from portfolio manager
        allocated_capital = self.portfolio_manager.get_allocated_capital(symbol)
        if allocated_capital <= 0:
            return {
                "valid": False,
                "reason": f"No capital allocated to {symbol}",
                "allocated_capital": 0
            }
        
        print(f"[ProductionRisk] Validating {action} {side} trade for {symbol}")
        print(f"[ProductionRisk]   Entry price: ${entry_price:.2f}")
        print(f"[ProductionRisk]   Allocated capital: ${allocated_capital:.2f}")
        print(f"[ProductionRisk]   Raw ATR: {atr_value:.6f}")
        
        # CRITICAL FIX: Normalize raw ATR to percentage volatility
        # ATR comes from algorithm strategies as raw price units, need to convert to percentage
        normalized_atr = atr_value / entry_price if entry_price > 0 else 0.02
        normalized_atr = max(0.005, min(normalized_atr, 0.15))  # Bound between 0.5% and 15%
        
        print(f"[ProductionRisk]   Normalized ATR: {normalized_atr:.6f} ({normalized_atr*100:.2f}%)")
        
        # Calculate normalized volatility for cost adjustment
        portfolio_avg_vol = self.get_portfolio_average_volatility()
        # Improved volatility normalization with better bounds
        if portfolio_avg_vol > 0:
            volatility_norm = min(max(normalized_atr / portfolio_avg_vol, 0.1), 3.0)  # Bound between 0.1x and 3x
        else:
            volatility_norm = 1.0
        
        # Calculate position size using allocated capital
        position_result = self.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=normalized_atr,  # Use normalized ATR
            entry_price=entry_price
        )
        
        if position_result["size_contracts"] <= 0:
            return {
                "valid": False,
                "reason": f"Invalid position size calculated: {position_result['size_contracts']}",
                "allocated_capital": allocated_capital
            }
        
        # Calculate dynamic leverage
        leverage = self.calculate_dynamic_leverage(symbol, normalized_atr)
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side=side,
            atr_adjusted=normalized_atr * entry_price  # Convert back to price units for SL/TP
        )
        
        # Perform risk checks
        kill_switches = self.check_kill_switches()
        
        if kill_switches["trading_halt"]:
            return {
                "valid": False,
                "reason": "Trading halted due to maximum drawdown",
                "allocated_capital": allocated_capital
            }
        
        if kill_switches["full_flatten"]:
            return {
                "valid": False,
                "reason": "Position flattening due to negative equity slope",
                "allocated_capital": allocated_capital
            }
        
        # All checks passed - return valid trade
        return {
            "valid": True,
            "position_info": {
                "symbol": symbol,
                "side": side,
                "size_contracts": position_result["size_contracts"],  # Fixed key name
                "size_usdt": position_result["size_usdt"],
                "margin_usdt": position_result["margin_usdt"],
                "leverage": position_result["leverage"],
                "entry_price": entry_price,
                "stop_loss_price": stop_loss,  # Fixed key name
                "take_profit_price": take_profit,  # Fixed key name
                "allocated_capital": allocated_capital,
                "atr_value": normalized_atr,
                "volatility_norm": volatility_norm
            },
            "allocated_capital": allocated_capital
        }
    
    def update_drawdown_history(self, drawdown_pct: float) -> None:
        """Update 3-day drawdown history."""
        now = datetime.now()
        self.drawdown_history.append((now, drawdown_pct))
        
        # Keep only last 3 days
        cutoff_time = now - timedelta(days=3)
        self.drawdown_history = [(dt, dd) for dt, dd in self.drawdown_history if dt > cutoff_time]
    
    def update_sharpe_history(self, sharpe_ratio: float) -> None:
        """Update 30-day Sharpe ratio history."""
        now = datetime.now()
        self.sharpe_history.append((now, sharpe_ratio))
        
        # Keep only last 30 days
        cutoff_time = now - timedelta(days=30)
        self.sharpe_history = [(dt, sr) for dt, sr in self.sharpe_history if dt > cutoff_time]
    
    def update_equity_curve(self, equity_value: float) -> None:
        """Update equity curve for slope calculation."""
        now = datetime.now()
        self.equity_curve.append((now, equity_value))
        
        # Keep only last 60 bars
        if len(self.equity_curve) > 60:
            self.equity_curve.pop(0)
    
    def get_rolling_drawdown_3d(self) -> float:
        """Get maximum drawdown over last 3 days.
        
        Returns:
            Maximum drawdown as negative value (e.g., -0.05 for 5% drawdown).
        """
        if not self.drawdown_history:
            return 0.0
        # Return the most negative (worst) drawdown
        return min(dd for _, dd in self.drawdown_history)
    
    def get_rolling_sharpe_30d(self) -> float:
        """Get current 30-day Sharpe ratio."""
        if not self.sharpe_history:
            return 0.0
        return self.sharpe_history[-1][1] if self.sharpe_history else 0.0
    
    def get_equity_curve_slope(self) -> float:
        """Calculate equity curve slope over last 60 bars."""
        if len(self.equity_curve) < 10:
            return 0.0
        
        # Simple linear regression on equity values
        x = np.arange(len(self.equity_curve))
        y = np.array([eq for _, eq in self.equity_curve])
        
        if len(y) < 2:
            return 0.0
            
        slope = np.polyfit(x, y, 1)[0]
        avg_equity = np.mean(y)
        
        # Convert to percentage slope
        slope_pct = (slope * len(x)) / avg_equity if avg_equity > 0 else 0
        return slope_pct
    
    def get_portfolio_average_volatility(self) -> float:
        """Get average volatility across active symbols."""
        active_symbols = list(self.portfolio_manager.volatility_data.keys())
        if not active_symbols:
            return 100.0  # Default volatility in price units (better for crypto)
        
        total_vol = sum(self.portfolio_manager.get_volatility_ema(s) for s in active_symbols)
        return total_vol / len(active_symbols)
    
    def get_projected_funding_8h(self, symbol: str) -> float:
        """Get projected 8-hour funding rate for leverage adjustment.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Projected 8-hour funding rate as decimal (e.g., 0.001 for 0.1%).
        """
        # In a real implementation, this would fetch current funding rates
        # For now, return a conservative estimate
        # Typical crypto funding rates: 0.01% to 0.1% per 8 hours
        default_funding_8h = 0.0001  # 0.01% per 8 hours (conservative)
        
        # Could be enhanced to fetch real funding rates from exchange
        # funding_rate = self.exchange.get_funding_rate(symbol)
        
        return default_funding_8h
    
    def check_kill_switches(self) -> Dict[str, bool]:
        """Check kill switch conditions from document.
        
        Returns:
            Dictionary indicating which kill switches are triggered.
        """
        dd_3d = self.get_rolling_drawdown_3d()
        equity_slope = self.get_equity_curve_slope()
        
        kill_switches = {
            "partial_flatten": dd_3d > 0.14,  # If DD >14%, flatten 30% of positions
            "full_flatten": equity_slope < -0.10,  # If equity slope < -10%, full flatten
            "trading_halt": self.max_drawdown_hit  # If max drawdown hit, halt trading
        }
        
        return kill_switches
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics."""
        kill_switches = self.check_kill_switches()
        
        # Determine risk status based on current conditions
        risk_status = "normal"
        if self.max_drawdown_hit:
            risk_status = "critical"
        elif any(kill_switches.values()):
            risk_status = "warning"
        elif self.daily_pnl < -0.05:  # -5% daily loss
            risk_status = "caution"
        
        return {
            "daily_pnl": self.daily_pnl,
            "max_drawdown_hit": self.max_drawdown_hit,
            "drawdown_3d": self.get_rolling_drawdown_3d(),
            "sharpe_30d": self.get_rolling_sharpe_30d(),
            "equity_slope": self.get_equity_curve_slope(),
            "active_positions": len(self.positions),
            "kill_switches": kill_switches,
            "risk_status": risk_status,
            "risk_parameters": {
                "risk_per_trade": config.RISK_PER_TRADE_PCT,
                "kelly_fraction": config.KELLY_FRACTION,
                "max_leverage": config.MAX_LEVERAGE,
                "base_cost": config.BASE_COST_PCT
            }
        }


# Legacy compatibility
RiskManager = ProductionRiskManager
