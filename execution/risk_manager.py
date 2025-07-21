from typing import Dict, Optional, Tuple, List, Any
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ProductionRiskParameters:
    """Production Risk Parameters implementing the exact specifications from the document."""
    
    # Core position sizing parameters from document
    risk_per_trade_pct: float = 0.008  # 0.8% risk per trade
    kelly_fraction: float = 0.7  # Fractional Kelly criterion
    base_cost_pct: float = 0.0014  # 0.14% base cost (0.04% + 0.1% spread)
    min_atr_floor: float = 0.001  # Minimum ATR floor to prevent excessive sizing
    
    # Stop loss and take profit parameters
    atr_stop_multiplier: float = 1.8  # SL = Entry ± 1.8×ATR
    atr_trail_multiplier: float = 0.8  # Trail by 0.8×ATR
    risk_reward_ratio: float = 2.0  # TP = Entry ± 2×|Entry-SL| (1:2 risk-reward)
    partial_exit_ratio: float = 0.4  # 40% partial exit at 1:1
    
    # Dynamic leverage parameters
    max_leverage: int = 10  # Cap leverage at 10x
    target_volatility: float = 0.18  # Target volatility for scaling


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
        self.risk_params = ProductionRiskParameters()
        
        # Risk tracking data structures
        self.drawdown_history: List[Tuple[datetime, float]] = []  # 3-day drawdown history
        self.sharpe_history: List[Tuple[datetime, float]] = []  # 30-day Sharpe history
        self.equity_curve: List[Tuple[datetime, float]] = []  # 60-bar equity curve for slope
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.max_drawdown_hit = False
        
        print(f"[ProductionRisk] Initialized with 0.8% risk, {self.risk_params.kelly_fraction} Kelly fraction")
    
    def calculate_dynamic_cost_adjustment(self, volatility_norm: float) -> float:
        """Calculate dynamic cost adjustment based on normalized volatility.
        
        Formula from document: dynamic_cost = base_cost × (1 + 0.5 × normalized_volatility)
        
        Args:
            volatility_norm: Normalized volatility measure.
            
        Returns:
            Adjusted cost percentage.
        """
        return self.risk_params.base_cost_pct * (1 + 0.5 * volatility_norm)
    
    def calculate_position_size(self, symbol: str, allocated_capital: float, atr_value: float, 
                              entry_price: float, volatility_norm: float = 0.5) -> Dict[str, Any]:
        """Calculate position size using the document's exact formula:
        Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001) - dynamic_cost
        
        Args:
            symbol: Trading pair symbol.
            allocated_capital: Capital allocated to this symbol from portfolio manager.
            atr_value: Current ATR value.
            entry_price: Entry price for position.
            volatility_norm: Normalized volatility for cost adjustment.
            
        Returns:
            Dictionary with position sizing information.
        """
        # Apply ATR floor from document
        atr_adjusted = max(atr_value, self.risk_params.min_atr_floor)
        
        # Calculate dynamic cost adjustment
        dynamic_cost = self.calculate_dynamic_cost_adjustment(volatility_norm)
        
        # Core formula: Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001) - dynamic_cost
        numerator = self.risk_params.risk_per_trade_pct * allocated_capital * self.risk_params.kelly_fraction
        position_size_usdt = (numerator / atr_adjusted) - (allocated_capital * dynamic_cost)
        
        # Ensure position size is positive
        position_size_usdt = max(position_size_usdt, 0)
        
        # Convert to contracts
        position_size_contracts = position_size_usdt / entry_price if entry_price > 0 else 0
        
        # Calculate leverage using dynamic leverage engine
        leverage = self.calculate_dynamic_leverage(symbol, atr_adjusted)
        
        # Calculate margin required
        margin_required = position_size_usdt / leverage if leverage > 0 else position_size_usdt
        
        # Calculate stop loss and take profit distances based on ATR
        stop_loss_distance = self.risk_params.atr_stop_multiplier * atr_adjusted
        take_profit_distance = self.risk_params.risk_reward_ratio * stop_loss_distance
        
        print(f"[ProductionRisk] Position sizing for {symbol}:")
        print(f"[ProductionRisk]   ATR: {atr_value:.6f} (floor-adjusted: {atr_adjusted:.6f})")
        print(f"[ProductionRisk]   Size: {position_size_contracts:.6f} contracts (${position_size_usdt:.2f})")
        print(f"[ProductionRisk]   Leverage: {leverage}x, Margin: ${margin_required:.2f}")
        print(f"[ProductionRisk]   Dynamic cost: {dynamic_cost:.4%}")
        
        return {
            "size_contracts": position_size_contracts,
            "size_usdt": position_size_usdt,
            "margin_usdt": margin_required,
            "leverage": leverage,
            "risk_amount": numerator,
            "dynamic_cost": dynamic_cost,
            "atr_adjusted": atr_adjusted,
            "stop_loss_distance": stop_loss_distance,
            "take_profit_distance": take_profit_distance,
            "entry_price": entry_price
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
        # Base leverage calculation with volatility adjustment
        vol_adjustment = min(1.0, self.risk_params.target_volatility / max(atr_value, 0.001))
        base_leverage = min(self.risk_params.max_leverage, 
                          int(self.risk_params.max_leverage * vol_adjustment))
        
        # Drawdown factor: 0.8 if rolling 3-day DD >10%; 0.5 if >14%
        dd_3d = self.get_rolling_drawdown_3d()
        if dd_3d > 0.14:
            dd_factor = 0.5
        elif dd_3d > 0.10:
            dd_factor = 0.8
        else:
            dd_factor = 1.0
        
        # Sharpe factor: max(0.5, min(1, rolling_30d_sharpe / 1.5))
        sharpe_30d = self.get_rolling_sharpe_30d()
        sharpe_factor = max(0.5, min(1.0, sharpe_30d / 1.5)) if sharpe_30d > 0 else 0.7
        
        # Equity curve slope factor: scale by 0.7 if slope < -5% over 60 bars
        slope_factor = 0.7 if self.get_equity_curve_slope() < -0.05 else 1.0
        
        # Calculate final leverage
        final_leverage = int(base_leverage * dd_factor * sharpe_factor * slope_factor)
        final_leverage = max(1, min(final_leverage, self.risk_params.max_leverage))
        
        print(f"[ProductionRisk] Dynamic leverage for {symbol}: {final_leverage}x")
        print(f"[ProductionRisk]   Base: {base_leverage}x, DD factor: {dd_factor:.2f}, "
              f"Sharpe factor: {sharpe_factor:.2f}, Slope factor: {slope_factor:.2f}")
        
        return final_leverage
    
    def calculate_stop_loss_take_profit(self, entry_price: float, side: str, atr_adjusted: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices using ATR multipliers.
        
        Args:
            entry_price: Entry price.
            side: Position side ("buy" or "sell").
            atr_adjusted: Floor-adjusted ATR value.
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        # SL = Entry ± 1.8×ATR
        stop_distance = self.risk_params.atr_stop_multiplier * atr_adjusted
        
        # TP = Entry ± 2×|Entry-SL| (1:2 risk-reward)
        tp_distance = self.risk_params.risk_reward_ratio * stop_distance
        
        if side == "buy":
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + tp_distance
        else:  # sell
            stop_loss_price = entry_price + stop_distance
            take_profit_price = entry_price - tp_distance
        
        return stop_loss_price, take_profit_price
    
    def validate_trade(self, symbol: str, action: str, side: str, entry_price: float, 
                      atr_value: float) -> Dict[str, Any]:
        """Validate trade using production risk criteria.
        
        Args:
            symbol: Trading pair symbol.
            action: Trade action ("open", "close", etc.).
            side: Position side ("buy" or "sell").
            entry_price: Entry price.
            atr_value: Current ATR value.
            
        Returns:
            Validation result with position information.
        """
        if action != "open":
            return {"valid": True, "reason": f"No validation needed for {action}"}
        
        # Check if max drawdown hit
        if self.max_drawdown_hit:
            return {"valid": False, "reason": "Max drawdown limit reached"}
        
        # Get allocated capital from portfolio manager
        allocated_capital = self.portfolio_manager.get_allocated_capital(symbol)
        if allocated_capital <= 0:
            return {"valid": False, "reason": "No capital allocated to this symbol"}
        
        # Calculate normalized volatility for cost adjustment
        portfolio_avg_vol = self.get_portfolio_average_volatility()
        volatility_norm = atr_value / max(portfolio_avg_vol, 0.001) if portfolio_avg_vol > 0 else 0.5
        
        # Calculate position size using production formula
        try:
            position_info = self.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocated_capital,
                atr_value=atr_value,
                entry_price=entry_price,
                volatility_norm=volatility_norm
            )
            
            # Calculate stop loss and take profit
            stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                entry_price, side, position_info["atr_adjusted"]
            )
            
            position_info.update({
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "side": side
            })
            
            return {
                "valid": True,
                "reason": "Production risk validation passed",
                "position_info": position_info
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"Position sizing error: {str(e)}"}
    
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
        """Get maximum drawdown over last 3 days."""
        if not self.drawdown_history:
            return 0.0
        return max(dd for _, dd in self.drawdown_history)
    
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
            return 0.02  # Default volatility
        
        total_vol = sum(self.portfolio_manager.get_volatility_ema(s) for s in active_symbols)
        return total_vol / len(active_symbols)
    
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
        
        return {
            "daily_pnl": self.daily_pnl,
            "max_drawdown_hit": self.max_drawdown_hit,
            "drawdown_3d": self.get_rolling_drawdown_3d(),
            "sharpe_30d": self.get_rolling_sharpe_30d(),
            "equity_slope": self.get_equity_curve_slope(),
            "active_positions": len(self.positions),
            "kill_switches": kill_switches,
            "risk_parameters": {
                "risk_per_trade": self.risk_params.risk_per_trade_pct,
                "kelly_fraction": self.risk_params.kelly_fraction,
                "max_leverage": self.risk_params.max_leverage,
                "base_cost": self.risk_params.base_cost_pct
            }
        }


# Legacy compatibility
RiskManager = ProductionRiskManager
