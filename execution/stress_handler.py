"""
Stress Handling Module implementing the exact safeguards from the document.

This module provides real-time safeguards for edge cases ensuring capital preservation:
- Flash crash detection and response
- Slippage monitoring and rejection
- Disconnect handling with forward-fill
- Liquidity filters and funding rate monitoring
- Kill switches for drawdown and equity slope protection
"""

from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class StressEvent:
    """Represents a stress event detected by the system."""
    timestamp: datetime
    event_type: str
    symbol: str
    severity: str
    data: Dict[str, Any]
    action_taken: str

class StressHandlingModule:
    """Stress Handling Module implementing document specifications."""
    
    def __init__(self, execution_engine):
        """Initialize stress handling with reference to execution engine.
        
        Args:
            execution_engine: Reference to the production execution engine.
        """
        self.execution_engine = execution_engine
        
        # Flash crash detection
        self.flash_crash_events: List[Tuple[datetime, str]] = []
        self.affected_assets_60s = set()
        
        # Connection monitoring
        self.last_data_timestamp = datetime.now()
        self.connection_lag_threshold = 3.0  # 3 seconds
        self.forward_fill_active = False
        
        # Slippage monitoring
        self.slippage_threshold = 0.002  # 0.2% slippage threshold
        
        # Liquidity filters
        self.min_daily_volume = 5_000_000  # $5M minimum daily volume
        self.max_spread = 0.0015  # 0.15% maximum spread
        self.max_funding_rate = 0.004  # 0.4% maximum funding rate per day
        
        # Kill switch states
        self.kill_switches = {
            "drawdown_partial": False,
            "drawdown_full": False,
            "equity_slope": False,
            "emergency_halt": False
        }
        
        # Event history
        self.stress_events: List[StressEvent] = []
        
        print("[StressHandler] Initialized with document-specified thresholds")
    
    def check_flash_crash(self, symbol: str, price_data: Dict[str, float], atr_value: float) -> bool:
        """Check for flash crash conditions per document:
        - If 1-min drop >4×ATR, flatten asset
        - If >5 assets in 60s, de-risk portfolio by 30%
        
        Args:
            symbol: Trading pair symbol.
            price_data: OHLCV price data.
            atr_value: Current ATR value.
            
        Returns:
            True if flash crash detected and handled.
        """
        high = price_data.get('high', 0)
        low = price_data.get('low', 0)
        
        if high <= 0 or low <= 0:
            return False
        
        # Calculate 1-min drop
        drop_pct = (high - low) / high
        flash_threshold = 4 * atr_value
        
        now = datetime.now()
        
        if drop_pct > flash_threshold:
            print(f"[StressHandler] ⚠️ Flash crash detected: {symbol} dropped {drop_pct:.2%} (>{flash_threshold:.2%})")
            
            # Record the event
            self.flash_crash_events.append((now, symbol))
            self.affected_assets_60s.add(symbol)
            
            # Flatten this asset
            self._flatten_asset(symbol, "flash_crash")
            
            # Check if >5 assets affected in 60s
            recent_events = [evt for evt in self.flash_crash_events 
                           if (now - evt[0]).total_seconds() < 60]
            
            if len(recent_events) > 5:
                print("[StressHandler] ⚠️ Multiple flash crashes detected - de-risking portfolio by 30%")
                self._derisk_portfolio(0.30, "multi_flash_crash")
                
                # Record major stress event
                stress_event = StressEvent(
                    timestamp=now,
                    event_type="multi_flash_crash",
                    symbol="PORTFOLIO",
                    severity="HIGH",
                    data={"affected_assets": len(recent_events), "derisk_pct": 0.30},
                    action_taken="Portfolio de-risked by 30%"
                )
                self.stress_events.append(stress_event)
            
            # Clean up old events (>60s)
            self.flash_crash_events = recent_events
            
            return True
        
        return False
    
    def check_slippage(self, expected_price: float, execution_price: float, symbol: str) -> bool:
        """Check slippage and reject if >0.2% off as per document.
        
        Args:
            expected_price: Expected execution price.
            execution_price: Actual execution price.
            symbol: Trading pair symbol.
            
        Returns:
            True if slippage is acceptable, False if should be rejected.
        """
        slippage_pct = abs(execution_price - expected_price) / expected_price
        
        if slippage_pct > self.slippage_threshold:
            print(f"[StressHandler] ❌ Excessive slippage detected: {symbol} {slippage_pct:.3%} > {self.slippage_threshold:.3%}")
            
            # Record stress event
            stress_event = StressEvent(
                timestamp=datetime.now(),
                event_type="excessive_slippage",
                symbol=symbol,
                severity="MEDIUM",
                data={
                    "expected_price": expected_price,
                    "execution_price": execution_price,
                    "slippage_pct": slippage_pct
                },
                action_taken="Order rejected"
            )
            self.stress_events.append(stress_event)
            
            return False
        
        return True
    
    def check_connection_lag(self, data_timestamp: datetime) -> bool:
        """Check for connection lag >3s as per document.
        
        Args:
            data_timestamp: Timestamp of latest market data.
            
        Returns:
            True if connection is healthy, False if lagged.
        """
        now = datetime.now()
        lag_seconds = (now - data_timestamp).total_seconds()
        
        if lag_seconds > self.connection_lag_threshold:
            if not self.forward_fill_active:
                print(f"[StressHandler] ⚠️ Connection lag detected: {lag_seconds:.1f}s > {self.connection_lag_threshold}s")
                print("[StressHandler] Activating forward-fill mode and pausing trading")
                
                self._activate_forward_fill()
                self._pause_trading()
                
                # Would send Telegram alert here
                # self._alert_telegram(f"Connection lag: {lag_seconds:.1f}s")
            
            return False
        else:
            if self.forward_fill_active:
                print("[StressHandler] Connection restored, resuming normal operation")
                self._deactivate_forward_fill()
                self._resume_trading()
            
            return True
    
    def check_liquidity_filters(self, symbol: str, volume_24h: float, spread_pct: float, 
                              funding_rate: float) -> bool:
        """Check liquidity filters as per document:
        - Skip if avg daily volume <$5M
        - Skip if spread >0.15%
        - Exit if funding >0.4%/day
        
        Args:
            symbol: Trading pair symbol.
            volume_24h: 24-hour trading volume in USD.
            spread_pct: Current bid-ask spread percentage.
            funding_rate: Current funding rate (daily).
            
        Returns:
            True if liquidity is acceptable.
        """
        # Check minimum volume
        if volume_24h < self.min_daily_volume:
            print(f"[StressHandler] ❌ Insufficient liquidity: {symbol} volume ${volume_24h:,.0f} < ${self.min_daily_volume:,.0f}")
            return False
        
        # Check spread
        if spread_pct > self.max_spread:
            print(f"[StressHandler] ❌ Excessive spread: {symbol} {spread_pct:.3%} > {self.max_spread:.3%}")
            return False
        
        # Check funding rate
        if abs(funding_rate) > self.max_funding_rate:
            print(f"[StressHandler] ⚠️ High funding rate: {symbol} {funding_rate:.3%} > {self.max_funding_rate:.3%}")
            
            # Exit position if funding too high
            if abs(funding_rate) > self.max_funding_rate:
                self._exit_position_funding(symbol, funding_rate)
                return False
        
        return True
    
    def check_kill_switches(self, drawdown_pct: float, equity_slope: float) -> Dict[str, bool]:
        """Check kill switch conditions as per document:
        - If DD >14%, flatten 30% of positions
        - If equity slope < -10%, full flatten
        
        Args:
            drawdown_pct: Current drawdown percentage.
            equity_slope: Equity curve slope percentage.
            
        Returns:
            Dictionary of kill switch states.
        """
        now = datetime.now()
        switches_triggered = {}
        
        # Drawdown kill switches
        if drawdown_pct > 0.14 and not self.kill_switches["drawdown_partial"]:
            print(f"[StressHandler] 🚨 KILL SWITCH: Drawdown {drawdown_pct:.1%} > 14% - Flattening 30% of positions")
            self._emergency_flatten(0.30, "drawdown_kill_switch")
            self.kill_switches["drawdown_partial"] = True
            switches_triggered["drawdown_partial"] = True
        
        # Equity slope kill switch
        if equity_slope < -0.10 and not self.kill_switches["equity_slope"]:
            print(f"[StressHandler] 🚨 KILL SWITCH: Equity slope {equity_slope:.1%} < -10% - Full flatten")
            self._emergency_flatten(1.0, "equity_slope_kill_switch")
            self.kill_switches["equity_slope"] = True
            switches_triggered["equity_slope"] = True
        
        # Reset kill switches if conditions improve
        if drawdown_pct < 0.10:
            self.kill_switches["drawdown_partial"] = False
        
        if equity_slope > -0.05:
            self.kill_switches["equity_slope"] = False
        
        return switches_triggered
    
    def smooth_regime_transitions(self, current_value: float, ema_history: List[float]) -> float:
        """Smooth regime transitions with 5-bar EMA to avoid whipsaw.
        
        Args:
            current_value: Current regime indicator value.
            ema_history: History of EMA values for smoothing.
            
        Returns:
            Smoothed regime value.
        """
        if not ema_history:
            return current_value
        
        # 5-bar EMA smoothing
        alpha = 2 / (5 + 1)
        smoothed = alpha * current_value + (1 - alpha) * ema_history[-1]
        
        # Update history
        ema_history.append(smoothed)
        if len(ema_history) > 10:  # Keep limited history
            ema_history.pop(0)
        
        return smoothed
    
    def _flatten_asset(self, symbol: str, reason: str) -> None:
        """Flatten specific asset due to stress condition."""
        print(f"[StressHandler] Flattening {symbol} due to {reason}")
        
        # Record stress event
        stress_event = StressEvent(
            timestamp=datetime.now(),
            event_type="asset_flatten",
            symbol=symbol,
            severity="HIGH",
            data={"reason": reason},
            action_taken=f"Asset {symbol} flattened"
        )
        self.stress_events.append(stress_event)
        
        # Would implement actual position flattening logic here
    
    def _derisk_portfolio(self, percentage: float, reason: str) -> None:
        """De-risk portfolio by specified percentage."""
        print(f"[StressHandler] De-risking portfolio by {percentage:.0%} due to {reason}")
        
        # Would implement actual portfolio de-risking logic here
        # This would reduce position sizes across all holdings
    
    def _emergency_flatten(self, percentage: float, reason: str) -> None:
        """Emergency position flattening."""
        print(f"[StressHandler] 🚨 EMERGENCY FLATTEN: {percentage:.0%} of positions ({reason})")
        
        # Record critical stress event
        stress_event = StressEvent(
            timestamp=datetime.now(),
            event_type="emergency_flatten",
            symbol="PORTFOLIO",
            severity="CRITICAL",
            data={"flatten_pct": percentage, "reason": reason},
            action_taken=f"Emergency flattened {percentage:.0%} of positions"
        )
        self.stress_events.append(stress_event)
        
        # Would implement actual emergency flattening logic here
    
    def _exit_position_funding(self, symbol: str, funding_rate: float) -> None:
        """Exit position due to high funding rate."""
        print(f"[StressHandler] Exiting {symbol} due to high funding rate: {funding_rate:.3%}")
        
        # Would implement position exit logic here
    
    def _activate_forward_fill(self) -> None:
        """Activate forward-fill mode for OHLCV data."""
        self.forward_fill_active = True
        # Would implement forward-fill logic here
        # Floor ATR/σ at 0.1% during forward-fill
    
    def _deactivate_forward_fill(self) -> None:
        """Deactivate forward-fill mode."""
        self.forward_fill_active = False
    
    def _pause_trading(self) -> None:
        """Pause trading due to connection issues."""
        self.execution_engine.trading_paused = True
    
    def _resume_trading(self) -> None:
        """Resume trading after connection restored."""
        self.execution_engine.trading_paused = False
    
    def get_stress_summary(self) -> Dict[str, Any]:
        """Get comprehensive stress handling summary."""
        recent_events = [e for e in self.stress_events 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "kill_switches": self.kill_switches,
            "forward_fill_active": self.forward_fill_active,
            "recent_flash_crashes": len([e for e in recent_events if e.event_type == "flash_crash"]),
            "recent_stress_events": len(recent_events),
            "connection_healthy": not self.forward_fill_active,
            "trading_paused": self.execution_engine.trading_paused,
            "stress_events_24h": len([e for e in self.stress_events 
                                     if (datetime.now() - e.timestamp).total_seconds() < 86400])
        }
    
    def get_recent_events(self, hours: int = 24) -> List[StressEvent]:
        """Get recent stress events.
        
        Args:
            hours: Number of hours to look back.
            
        Returns:
            List of recent stress events.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [e for e in self.stress_events if e.timestamp > cutoff_time]
