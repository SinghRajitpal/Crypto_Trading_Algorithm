from typing import Dict, Any, List
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy
from ..trade_signal import TradeSignal

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy for crypto futures trading.
    
    This strategy generates trading signals based on the crossover of two moving averages:
    - A fast moving average (shorter period)
    - A slow moving average (longer period)
    
    Trading Rules:
    1. Buy Signal: Fast MA crosses above Slow MA (bullish crossover)
    2. Sell Signal: Fast MA crosses below Slow MA (bearish crossover)
    3. Close Signal: When no crossover is detected and position is open
    
    Key Features:
    - Configurable MA periods
    - Position size management
    - Error handling and validation
    - Detailed signal metadata
    
    Usage:
        strategy = MACrossoverStrategy(
            params={
                'fast_ma_period': 9,
                'slow_ma_period': 21,
                'leverage': 1.0,
                'position_threshold': 0.0001
            },
            strategy_id='ma_crossover_1'
        )
    """
    
    def __init__(self, params: Dict[str, Any] = None, strategy_id: str = "ma_crossover"):
        """
        Initialize the MA Crossover strategy.
        
        Args:
            params (Dict[str, Any]): Strategy parameters:
                - fast_ma_period (int): Period for fast moving average (default: 2)
                - slow_ma_period (int): Period for slow moving average (default: 4)
                - leverage (float): Trading leverage (default: 1.0)
                - position_threshold (float): Minimum position size to consider as open (default: 0.0001)
            strategy_id (str): Unique identifier for this strategy instance
            
        Raises:
            ValueError: If parameters are invalid:
                - fast_ma_period must be a positive integer
                - slow_ma_period must be a positive integer
                - fast_ma_period must be less than slow_ma_period
                - leverage must be a positive number
                - position_threshold must be a positive number
                
        Example:
            strategy = MACrossoverStrategy(
                params={
                    'fast_ma_period': 9,
                    'slow_ma_period': 21,
                    'leverage': 1.0
                }
            )
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
        """
        Get the list of indicators required by this strategy.
        
        This strategy requires two simple moving averages:
        - A fast MA with period specified in params
        - A slow MA with period specified in params
        
        Returns:
            List[str]: List of indicator names:
                - sma_{fast_ma_period}
                - sma_{slow_ma_period}
                
        Example:
            strategy = MACrossoverStrategy(params={'fast_ma_period': 9, 'slow_ma_period': 21})
            indicators = strategy.get_required_indicators()
            # Returns: ['sma_9', 'sma_21']
        """
        return [
            f'sma_{self.params["fast_ma_period"]}',
            f'sma_{self.params["slow_ma_period"]}'
        ]
    
    async def _generate_signals(self, data: Dict[str, np.ndarray], indicator_data: Dict[str, np.ndarray], symbol: str) -> TradeSignal:
        """
        Generate trading signals based on MA crossover.
        
        This method implements the core trading logic:
        1. Checks for bullish crossover (fast MA crosses above slow MA)
        2. Checks for bearish crossover (fast MA crosses below slow MA)
        3. Manages position opening and closing
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary of numpy arrays for each price component
            indicator_data (Dict[str, np.ndarray]): Dictionary of calculated indicators
            symbol (str): Trading pair symbol
            
        Returns:
            TradeSignal: A TradeSignal object containing:
                - action: "open" or "close"
                - side: "buy" or "sell"
                - symbol: Trading pair
                - strategy_id: Strategy identifier
                - metadata: Additional information including:
                    - reason: Signal generation reason
                    - fast_ma: Current fast MA value
                    - slow_ma: Current slow MA value
                    - current_price: Latest closing price
                    - current_position: Current position size
                    - fast_ma_period: Fast MA period
                    - slow_ma_period: Slow MA period
                    - position_threshold: Position threshold
                    
        Example:
            signal = await strategy._generate_signals(data, indicator_data, "BTCUSDT")
            if signal.action == "open":
                print(f"Opening {signal.side} position for {signal.symbol}")
                print(f"Reason: {signal.metadata['reason']}")
        """
        try:
            # Check if we have enough data
            # We need enough candles to calculate indicators plus one previous candle for crossover detection
            min_candles_needed = self.params['slow_ma_period'] + 1
            
            if len(data['close']) < min_candles_needed:
                print(f"\n[{symbol}] 📊 Data Collection Progress: {len(data['close'])}/{min_candles_needed} candles")
                return TradeSignal(
                    action="close",
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
                    action="close",
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
                    action="close",
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
                    action="close",
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
                    action="close",
                    side="sell" if current_position > 0 else "buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata=metadata,
                    signal_confidence=0.5
                )
            
            return TradeSignal(
                action="close",
                side="sell" if current_position > 0 else "buy",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata=metadata,
                signal_confidence=0.0
            )
            
        except Exception as e:
            print(f"\n[{symbol}] ❌ Error: {e}")
            return TradeSignal(
                action="close",
                side="sell" if await self.get_position(symbol) > 0 else "buy",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": f"Error: {str(e)}"},
                signal_confidence=0.0
            ) 