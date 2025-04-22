from typing import Dict, Any, List
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy
from ..trade_signal import TradeSignal

class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy for crypto futures trading.
    
    This strategy generates trading signals based on the crossover of two moving averages:
    - A fast moving average (shorter period)
    - A slow moving average (longer period)
    
    Attributes:
        params: Dictionary containing strategy parameters:
            - fast_ma_period: Period for fast moving average
            - slow_ma_period: Period for slow moving average
            - leverage: Trading leverage
            - position_threshold: Minimum position size to consider as open
    
    Trading Rules:
        1. Buy Signal: Fast MA crosses above Slow MA (bullish crossover)
        2. Sell Signal: Fast MA crosses below Slow MA (bearish crossover)
        3. Exit Signal: When no crossover is detected and position is open
    """
    
    def __init__(self, params: Dict[str, Any] = None, strategy_id: str = "ma_crossover"):
        """Initializes the MA Crossover strategy.
        
        Args:
            params: Strategy parameters:
                - fast_ma_period: Period for fast moving average (default: 2)
                - slow_ma_period: Period for slow moving average (default: 4)
                - leverage: Trading leverage (default: 1.0)
                - position_threshold: Minimum position size to consider as open (default: 0.0001)
            strategy_id: Unique identifier for this strategy instance
            
        Raises:
            ValueError: If parameters are invalid:
                - fast_ma_period must be a positive integer
                - slow_ma_period must be a positive integer
                - fast_ma_period must be less than slow_ma_period
                - leverage must be a positive number
                - position_threshold must be a positive number
        """
        default_params = {
            'fast_ma_period': 2,
            'slow_ma_period': 4,
            'leverage': 1.0,
            'position_threshold': self.DEFAULT_POSITION_THRESHOLD
        }
        
        if params:
            default_params.update(params)
            
        # Validate parameters
        if not isinstance(default_params['fast_ma_period'], int) or default_params['fast_ma_period'] <= 0:
            raise ValueError("fast_ma_period must be a positive integer")
        if not isinstance(default_params['slow_ma_period'], int) or default_params['slow_ma_period'] <= 0:
            raise ValueError("slow_ma_period must be a positive integer")
        if default_params['fast_ma_period'] >= default_params['slow_ma_period']:
            raise ValueError("fast_ma_period must be less than slow_ma_period")
        if not isinstance(default_params['leverage'], (int, float)) or default_params['leverage'] <= 0:
            raise ValueError("leverage must be a positive number")
        if not isinstance(default_params['position_threshold'], (int, float)) or default_params['position_threshold'] <= 0:
            raise ValueError("position_threshold must be a positive number")
            
        super().__init__(default_params, strategy_id)
        
    def get_required_indicators(self) -> List[str]:
        """Gets the list of indicators required by this strategy.
        
        This strategy requires two simple moving averages:
        - A fast MA with period specified in params
        - A slow MA with period specified in params
        
        Returns:
            List of indicator names required for this strategy:
            - sma_{fast_ma_period}
            - sma_{slow_ma_period}
        """
        return [
            f'sma_{self.params["fast_ma_period"]}',
            f'sma_{self.params["slow_ma_period"]}'
        ]
    
    async def _generate_signals(self, data: Dict[str, np.ndarray], indicator_data: Dict[str, np.ndarray], symbol: str) -> TradeSignal:
        """Generates trading signals based on MA crossover.
        
        This method implements the core trading logic:
        1. Checks for bullish crossover (fast MA crosses above slow MA)
        2. Checks for bearish crossover (fast MA crosses below slow MA)
        3. Manages position opening and closing
        
        Args:
            data: Dictionary of numpy arrays for each price component
            indicator_data: Dictionary of calculated indicators
            symbol: Trading pair symbol
            
        Returns:
            A TradeSignal object containing:
                - action: "open", "exit", or "hold"
                - side: "buy", "sell", or "none" (for hold)
                - symbol: Trading pair
                - strategy_id: Strategy identifier
                - metadata: Additional signal information
                - signal_confidence: Confidence level (0.0 to 1.0)
                
        Raises:
            Exception: Any exceptions are caught, logged, and exit signal is returned
        """
        try:
            # Check if we have enough data
            # We need enough candles to calculate indicators plus one previous candle for crossover detection
            min_candles_needed = self.params['slow_ma_period'] + 1
            
            if len(data['close']) < min_candles_needed:
                print(f"\n[{symbol}] 📊 Data Collection Progress: {len(data['close'])}/{min_candles_needed} candles")
                return TradeSignal(
                    action="exit",
                    side="sell" if await self.get_position(symbol) > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Insufficient data"},
                    signal_confidence=0.0
                )
            
            # Get the latest values
            fast_ma_key = f'sma_{self.params["fast_ma_period"]}'
            slow_ma_key = f'sma_{self.params["slow_ma_period"]}'
            
            if fast_ma_key not in indicator_data or slow_ma_key not in indicator_data:
                print(f"\n[{symbol}] ⚠️ Missing indicators: {fast_ma_key} or {slow_ma_key}")
                return TradeSignal(
                    action="exit",
                    side="sell" if await self.get_position(symbol) > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Missing indicator data"},
                    signal_confidence=0.0
                )
            
            fast_ma = indicator_data[fast_ma_key]
            slow_ma = indicator_data[slow_ma_key]
            
            # Check for NaN values
            if np.isnan(fast_ma[-1]) or np.isnan(slow_ma[-1]):
                print(f"\n[{symbol}] ⚠️ Invalid indicator values detected (current)")
                return TradeSignal(
                    action="exit",
                    side="sell" if await self.get_position(symbol) > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Invalid indicator values"},
                    signal_confidence=0.0
                )
                
            # Make sure we have previous values for crossover detection
            if len(fast_ma) < 2 or len(slow_ma) < 2 or np.isnan(fast_ma[-2]) or np.isnan(slow_ma[-2]):
                print(f"\n[{symbol}] ⚠️ Missing previous indicator values for crossover detection")
                return TradeSignal(
                    action="exit",
                    side="sell" if await self.get_position(symbol) > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Insufficient data for crossover detection"},
                    signal_confidence=0.0
                )
            
            current_fast_ma = fast_ma[-1]
            current_slow_ma = slow_ma[-1]
            prev_fast_ma = fast_ma[-2]
            prev_slow_ma = slow_ma[-2]
            current_price = data['close'][-1]
            
            # Get current position from Binance
            current_position = await self.get_position(symbol)
            
            # Print strategy state (only when values change significantly)
            if abs(current_fast_ma - prev_fast_ma) > 0.001 or abs(current_slow_ma - prev_slow_ma) > 0.001:
                print(f"\n=== {symbol} Strategy Update ===")
                print(f"Price: {current_price:.2f} | Position: {current_position}")
                print(f"Fast MA ({self.params['fast_ma_period']}): {current_fast_ma:.2f}")
                print(f"Slow MA ({self.params['slow_ma_period']}): {current_slow_ma:.2f}")
            
            # Create base metadata
            metadata = {
                "reason": "No action needed",
                "fast_ma": current_fast_ma,
                "slow_ma": current_slow_ma,
                "current_price": current_price,
                "current_position": current_position,
                "fast_ma_period": self.params['fast_ma_period'],
                "slow_ma_period": self.params['slow_ma_period'],
                "position_threshold": self.position_threshold
            }
            
            # Bullish crossover
            if prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma:
                if current_position <= 0 and await self.can_open_position(symbol):
                    print(f"\n[{symbol}] 🚀 BULLISH SIGNAL")
                    print(f"Fast MA ({current_fast_ma:.2f}) crossed above Slow MA ({current_slow_ma:.2f})")
                    metadata["reason"] = "Bullish crossover"
                    return TradeSignal(
                        action="open",
                        side="buy",
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        metadata=metadata,
                        signal_confidence=0.8
                    )
            
            # Bearish crossover
            elif prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma:
                if current_position >= 0 and await self.can_open_position(symbol):
                    print(f"\n[{symbol}] 📉 BEARISH SIGNAL")
                    print(f"Fast MA ({current_fast_ma:.2f}) crossed below Slow MA ({current_slow_ma:.2f})")
                    metadata["reason"] = "Bearish crossover"
                    return TradeSignal(
                        action="open",
                        side="sell",
                        symbol=symbol,
                        strategy_id=self.strategy_id,
                        metadata=metadata,
                        signal_confidence=0.8
                    )
            
            # No signal or can't open new position
            if abs(current_position) > self.position_threshold:
                metadata["reason"] = "Managing existing position"
                return TradeSignal(
                    action="exit",
                    side="sell" if current_position > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata=metadata,
                    signal_confidence=0.5
                )
            
            # No position and no new signals - explicit HOLD
            metadata["reason"] = "Market conditions stable - holding"
            return TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata=metadata,
                signal_confidence=0.7
            )
            
        except Exception as e:
            print(f"\n[{symbol}] ❌ Error: {e}")
            return TradeSignal(
                action="exit",
                side="sell" if await self.get_position(symbol) > 0 else "buy",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": f"Error: {str(e)}"},
                signal_confidence=0.0
            ) 