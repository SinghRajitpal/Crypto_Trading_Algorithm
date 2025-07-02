from abc import ABC, abstractmethod
from typing import Dict, Any, List, Deque, Optional, TYPE_CHECKING
import numpy as np
from collections import deque
from ..trade_signal import TradeSignal

if TYPE_CHECKING:
    from ..algo_engine import AlgoEngine

class BaseStrategy(ABC):
    """Base class for all trading strategies.
    
    This class defines the interface that all strategies must implement and provides
    common functionality for data processing, indicator calculation, and position management.
    
    Attributes:
        params (Dict[str, Any]): Strategy-specific parameters.
        strategy_id (str): Unique identifier for this strategy instance.
        algo_engine (Optional[AlgoEngine]): Reference to the algorithm engine.
        position_threshold (float): Minimum position size to consider as open.
        stop_loss_pct (float): Default stop loss percentage.
        take_profit_pct (float): Default take profit percentage.
        max_positions (int): Maximum number of positions allowed.
        
    Signal Types:
        - open/buy: Enter a long position
        - open/sell: Enter a short position
        - exit/buy: Exit a short position
        - exit/sell: Exit a long position
        - hold/none: No action needed, maintain current state
    """
    
    # Default constants
    DEFAULT_POSITION_THRESHOLD = 0.0001  # Consider position as 0 if smaller than this
    
    def __init__(self, params: Dict[str, Any], strategy_id: str):
        """Initializes the strategy with parameters.
        
        Args:
            params: Strategy-specific parameters including:
                position_threshold (float): Minimum position size to consider as open.
                stop_loss_pct (float): Default stop loss percentage.
                take_profit_pct (float): Default take profit percentage.
                max_positions (int): Maximum number of positions allowed.
                Other strategy-specific parameters.
            strategy_id: Unique identifier for this strategy instance.
                
        Raises:
            ValueError: If params is not a dictionary or strategy_id is not a string.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        if not isinstance(strategy_id, str):
            raise ValueError("strategy_id must be a string")
            
        self.params = params
        self.strategy_id = strategy_id
        self.algo_engine: Optional['AlgoEngine'] = None  # Will be set by algo_engine
        self.position_threshold = float(params.get('position_threshold', self.DEFAULT_POSITION_THRESHOLD))
        self.stop_loss_pct = float(params.get('stop_loss_pct', 0.02))
        self.take_profit_pct = float(params.get('take_profit_pct', 0.04))
        self.max_positions = int(params.get('max_positions', 1))
        
    def set_algo_engine(self, algo_engine: 'AlgoEngine') -> None:
        """Sets the algo engine instance for this strategy.
        
        This method is called by the algo engine when the strategy is registered.
        It provides the strategy with access to the Binance client and other
        trading functionality.
        
        Args:
            algo_engine: The algo engine instance.
            
        Raises:
            ValueError: If algo_engine is not an instance of AlgoEngine.
        """
        # Instead of using isinstance, we'll check the class name
        if not hasattr(algo_engine, '__class__') or algo_engine.__class__.__name__ != 'AlgoEngine':
            raise ValueError("algo_engine must be an instance of AlgoEngine")
        self.algo_engine = algo_engine
        
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Gets the list of indicators required by this strategy.
        
        This method must be implemented by all strategies to specify which
        technical indicators they need for signal generation.
        
        Returns:
            List of indicator names required by the strategy.
            These should match the indicator names in data/indicators.py.
        """
        pass
    
    def _convert_deque_to_numpy(self, data: Deque) -> Dict[str, np.ndarray]:
        """Converts deque of candles to numpy arrays for indicator calculation.
        
        Args:
            data: Deque of candle data [timestamp, open, high, low, close, volume].
            
        Returns:
            Dictionary of numpy arrays for each price component.
            
        Raises:
            Exception: If there's an error during conversion, logs the error and returns empty dict.
        """
        if not data:
            return {}
            
        try:
            # Convert deque to numpy array
            candles = np.array(data)
            
            # Extract components
            return {
                'timestamp': candles[:, 0],  # Keep timestamp for reference
                'open': candles[:, 1].astype(float),
                'high': candles[:, 2].astype(float),
                'low': candles[:, 3].astype(float),
                'close': candles[:, 4].astype(float),
                'volume': candles[:, 5].astype(float)
            }
        except Exception as e:
            print(f"[ERROR] Failed to convert candle data: {e}")
            return {}
    
    async def _calculate_indicators(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculates all required indicators for the strategy.
        
        This method orchestrates the calculation of all indicators required by
        the strategy. It ensures all required indicators are available and handles
        any calculation errors.
        
        Args:
            data: Dictionary of numpy arrays for each price component.
            
        Returns:
            Dictionary of calculated indicators.
            
        Raises:
            ValueError: If required indicators are not available.
            Exception: Any other exceptions are caught, logged, and empty dict is returned.
        """
        if not data:
            print(f"[{self.strategy_id}] ⚠️ No data provided for indicator calculation")
            return {}
            
        try:
            # Calculate indicators
            from data.indicators import Indicators
            indicators = Indicators()
            
            # Get the required indicators
            required_indicators = self.get_required_indicators()
            if not required_indicators:
                print(f"[{self.strategy_id}] ❌ No indicators required by strategy")
                raise ValueError("No indicators required by strategy")
            
            print(f"[{self.strategy_id}] Calculating indicators: {', '.join(required_indicators)}")
            
            # Check if we have enough data points for the indicators
            min_data_points = max([int(ind.split('_')[1]) if '_' in ind and ind.split('_')[1].isdigit() else 1 
                                for ind in required_indicators])
                                
            if len(data['close']) < min_data_points:
                print(f"[{self.strategy_id}] ⚠️ Insufficient data for indicator calculation. Need {min_data_points} points, have {len(data['close'])}.")
                return {}
                
            # Create OHLCV array in the correct order for indicator calculation
            ohlcv_data = np.column_stack((
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume']
            ))
            
            print(f"[{self.strategy_id}] Processing {len(ohlcv_data)} candles for indicator calculation")
            
            # Calculate indicators
            indicator_data = await indicators.calculate_indicators(required_indicators, ohlcv_data)
            
            # Validate that all required indicators were calculated
            missing_indicators = [ind for ind in required_indicators if ind not in indicator_data]
            if missing_indicators:
                print(f"[{self.strategy_id}] ⚠️ Missing required indicators: {missing_indicators}")
                return {}
                
            # Check for NaN values in final (most recent) indicator values
            nan_indicators = [ind for ind in required_indicators 
                            if ind in indicator_data and 
                            (isinstance(indicator_data[ind], dict) and any(np.isnan(arr[-1]) for arr in indicator_data[ind].values())
                             or (not isinstance(indicator_data[ind], dict) and np.isnan(indicator_data[ind][-1])))]
                             
            if nan_indicators:
                print(f"[{self.strategy_id}] ⚠️ NaN values detected in indicator(s): {nan_indicators}")
            
            # Print the latest values of each indicator for debugging
            print(f"[{self.strategy_id}] Latest indicator values:")
            for ind in required_indicators:
                if ind in indicator_data:
                    if isinstance(indicator_data[ind], dict):
                        print(f"  - {ind}: " + ", ".join([f"{k}={v[-1]:.4f}" for k, v in indicator_data[ind].items() if not np.isnan(v[-1])]))
                    else:
                        if not np.isnan(indicator_data[ind][-1]):
                            print(f"  - {ind}: {indicator_data[ind][-1]:.4f}")
            
            return indicator_data
        except Exception as e:
            print(f"[{self.strategy_id}] ❌ Failed to calculate indicators: {e}")
            return {}
    
    async def get_position(self, symbol: str) -> float:
        """Gets the current position for a specific symbol from Binance.
        
        This method retrieves the current position size for a trading pair from
        the Binance exchange. It handles errors and returns 0 if the position
        cannot be retrieved.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT").
            
        Returns:
            Current position size:
                - Positive: Long position
                - Negative: Short position
                - Zero: No position
                
        Raises:
            RuntimeError: If algo_engine is not set.
            Exception: Other exceptions are caught, logged, and 0.0 is returned.
        """
        if not self.algo_engine:
            print(f"[ERROR] algo_engine not set for {self.strategy_id}. Call set_algo_engine first.")
            return 0.0
            
        if not hasattr(self.algo_engine, 'binance_client') or self.algo_engine.binance_client is None:
            print(f"[ERROR] binance_client not available in algo_engine for {self.strategy_id}")
            return 0.0
            
        try:
            # Get position from Binance using get_open_positions
            positions = await self.algo_engine.binance_client.get_open_positions(symbol)
            if not positions:
                return 0.0
            return float(positions[0].get('contracts', 0))
        except Exception as e:
            print(f"[ERROR] Failed to get position for {symbol}: {e}")
            return 0.0
    
    async def can_open_position(self, symbol: str) -> bool:
        """Checks if a new position can be opened for a specific symbol.
        
        This method determines if a new position can be opened by checking if
        the current position is below the threshold. It's used to prevent
        opening multiple positions for the same symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            True if a new position can be opened, False otherwise.
            
        Raises:
            Exception: Any exceptions are caught, logged, and False is returned.
        """
        try:
            # Get current position from Binance
            position = await self.get_position(symbol)
            return abs(position) < self.position_threshold
        except Exception as e:
            print(f"[ERROR] Failed to check if position can be opened for {symbol}: {e}")
            return False
    
    async def calculate_signals(self, data: Deque, symbol: str) -> TradeSignal:
        """Calculates trading signals based on candle data.
        
        This is the main entry point for signal generation. It processes
        the raw candle data, calculates indicators, and delegates to the
        strategy-specific signal generation logic.
        
        Args:
            data: Deque of candle data [timestamp, open, high, low, close, volume].
            symbol: Trading pair symbol.
            
        Returns:
            A TradeSignal object with trading instructions.
        """
        try:
            # Skip if we don't have any data
            if not data:
                print(f"[{self.strategy_id}] No candle data available for {symbol}")
                return TradeSignal(
                    action="hold",
                    side="none",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "No candle data available"},
                    signal_confidence=0.0
                )
                
            print(f"[{self.strategy_id}] Calculating signals for {symbol} using {len(data)} candles")
                
            # Convert candle data to numpy arrays
            np_data = self._convert_deque_to_numpy(data)
            if not np_data:
                print(f"[{self.strategy_id}] Failed to convert candle data for {symbol}")
                return TradeSignal(
                    action="hold",
                    side="none",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Failed to convert candle data"},
                    signal_confidence=0.0
                )
                
            # Calculate technical indicators
            indicator_data = await self._calculate_indicators(np_data)
            if not indicator_data:
                print(f"[{self.strategy_id}] No indicator data available for {symbol}")
                return TradeSignal(
                    action="hold",
                    side="none",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "Failed to calculate indicators"},
                    signal_confidence=0.0
                )
                
            # Generate signals based on indicator data
            signal = await self._generate_signals(np_data, indicator_data, symbol)
            
            # Ensure the symbol is set on the signal
            signal.symbol = symbol
            
            # Set the strategy ID
            signal.strategy_id = self.strategy_id
            
            # Set timestamp
            if not signal.timestamp and len(np_data['timestamp']) > 0:
                signal.timestamp = int(np_data['timestamp'][-1])
                
            print(f"[{self.strategy_id}] Generated signal for {symbol}: {signal.action}/{signal.side} (confidence: {signal.signal_confidence:.2f})")
                
            return signal
            
        except Exception as e:
            print(f"[{self.strategy_id}] ❌ Error during signal calculation for {symbol}: {e}")
            # Return hold signal on error
            return TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": f"Error during signal calculation: {str(e)}"},
                signal_confidence=0.0
            )
    
    @abstractmethod
    async def _generate_signals(self, data: Dict[str, np.ndarray], indicator_data: Dict[str, np.ndarray], symbol: str) -> TradeSignal:
        """Generates trading signals based on the calculated indicators.
        
        This method must be implemented by each strategy to define its specific
        trading logic. It should analyze the indicators and current position to
        determine the appropriate trading action.
        
        Args:
            data: Dictionary of numpy arrays for each price component.
            indicator_data: Dictionary of calculated indicators.
            symbol: Trading pair symbol.
            
        Returns:
            A TradeSignal object containing the trading action.
            
        Possible Signal Types:
            1. Enter a position:
               - action="open", side="buy" (go long)
               - action="open", side="sell" (go short)
            2. Exit a position:
               - action="exit", side="sell" (exit long position)
               - action="exit", side="buy" (exit short position)
            3. Hold current state:
               - action="hold", side="none" (no action needed)
        """
        pass