"""
Unit tests for BaseStrategy module.

This test suite covers:
1. Strategy initialization and configuration
2. Data conversion and validation
3. Indicator calculation integration
4. Signal calculation workflow
5. Position management
6. Abstract method implementation
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
import numpy as np
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from algorithm.strategies.base_strategy import BaseStrategy
from algorithm.trade_signal import TradeSignal


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    
    def __init__(self, params=None):
        super().__init__(params or {}, "test_strategy")
        
    def get_required_indicators(self):
        return ["sma_5", "sma_20", "atr_14"]
    
    async def _generate_signals(self, data, indicator_data, symbol):
        """Simple test signal generation."""
        if len(data['close']) < 2:
            return TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": "Insufficient data"},
                signal_confidence=0.0
            )
        
        # Simple logic: if current price > previous price, signal buy
        if data['close'][-1] > data['close'][-2]:
            return TradeSignal(
                action="open",
                side="buy",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": "Price increase detected"},
                signal_confidence=0.7
            )
        else:
            return TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"reason": "No clear signal"},
                signal_confidence=0.3
            )


class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = ConcreteStrategy({"test_param": 100})
        self.mock_algo_engine = Mock()
        self.mock_algo_engine.binance_client = Mock()
        
        # Sample candle data
        self.sample_deque = deque([
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
            [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
            [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
        ])
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        # Test with parameters
        strategy = ConcreteStrategy({"param1": 10, "param2": "value"})
        
        self.assertEqual(strategy.strategy_id, "test_strategy")
        self.assertEqual(strategy.params["param1"], 10)
        self.assertEqual(strategy.params["param2"], "value")
        self.assertIsNone(strategy.algo_engine)
        
        # Test with no parameters
        strategy_no_params = ConcreteStrategy()
        self.assertEqual(strategy_no_params.params, {})
    
    def test_set_algo_engine(self):
        """Test setting the algorithm engine."""
        # Create a mock that mimics AlgoEngine class name
        mock_algo_engine = Mock()
        mock_algo_engine.__class__.__name__ = 'AlgoEngine'
        
        self.strategy.set_algo_engine(mock_algo_engine)
        
        self.assertEqual(self.strategy.algo_engine, mock_algo_engine)
    
    def test_get_required_indicators(self):
        """Test get_required_indicators method."""
        indicators = self.strategy.get_required_indicators()
        
        expected = ["sma_5", "sma_20", "atr_14"]
        self.assertEqual(indicators, expected)
    
    def test_convert_deque_to_numpy(self):
        """Test conversion of deque data to numpy arrays."""
        np_data = self.strategy._convert_deque_to_numpy(self.sample_deque)
        
        self.assertIsInstance(np_data, dict)
        
        # Check all required keys are present
        expected_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for key in expected_keys:
            self.assertIn(key, np_data)
            self.assertIsInstance(np_data[key], np.ndarray)
        
        # Check data integrity
        self.assertEqual(len(np_data['close']), 4)  # 4 candles
        self.assertEqual(np_data['close'][0], 42050.0)  # First close
        self.assertEqual(np_data['close'][-1], 42350.0)  # Last close
    
    def test_convert_deque_to_numpy_edge_cases(self):
        """Test deque conversion with edge cases."""
        # Test empty deque
        empty_data = self.strategy._convert_deque_to_numpy(deque())
        self.assertEqual(empty_data, {})
        
        # Test deque with incomplete candle
        incomplete_deque = deque([[1642680000000, 42000.0, 42100.0]])  # Missing OHLCV
        incomplete_data = self.strategy._convert_deque_to_numpy(incomplete_deque)
        self.assertEqual(incomplete_data, {})
        
        # Test single candle
        single_deque = deque([[1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]])
        single_data = self.strategy._convert_deque_to_numpy(single_deque)
        
        self.assertEqual(len(single_data['close']), 1)
        self.assertEqual(single_data['close'][0], 42050.0)
    
    @patch('data.indicators.Indicators')
    async def test_calculate_indicators(self, mock_indicators_class):
        """Test indicator calculation."""
        # Setup mock
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={
            'sma_5': np.array([42000.0, 42100.0, 42200.0, 42300.0]),
            'sma_20': np.array([41900.0, 42000.0, 42100.0, 42200.0]),
            'atr_14': np.array([50.0, 55.0, 60.0, 65.0])
        })
        
        np_data = self.strategy._convert_deque_to_numpy(self.sample_deque)
        indicator_data = await self.strategy._calculate_indicators(np_data)
        
        self.assertIsInstance(indicator_data, dict)
        self.assertIn('sma_5', indicator_data)
        self.assertIn('sma_20', indicator_data)
        self.assertIn('atr_14', indicator_data)
        
        # Verify indicators were called correctly
        mock_indicators.calculate_indicators.assert_called_once()
    
    @patch('data.indicators.Indicators')
    async def test_calculate_indicators_missing_indicators(self, mock_indicators_class):
        """Test indicator calculation with missing indicators."""
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={
            'sma_5': np.array([42000.0, 42100.0, 42200.0, 42300.0]),
            # Missing sma_20 and atr_14
        })
        
        np_data = self.strategy._convert_deque_to_numpy(self.sample_deque)
        indicator_data = await self.strategy._calculate_indicators(np_data)
        
        # Should return empty dict when required indicators are missing
        self.assertEqual(indicator_data, {})
    
    @patch('data.indicators.Indicators')
    async def test_calculate_indicators_nan_values(self, mock_indicators_class):
        """Test indicator calculation with NaN values."""
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={
            'sma_5': np.array([42000.0, 42100.0, 42200.0, np.nan]),  # NaN in latest value
            'sma_20': np.array([41900.0, 42000.0, 42100.0, 42200.0]),
            'atr_14': np.array([50.0, 55.0, 60.0, 65.0])
        })
        
        np_data = self.strategy._convert_deque_to_numpy(self.sample_deque)
        indicator_data = await self.strategy._calculate_indicators(np_data)
        
        # Should return empty dict when indicators have NaN values
        self.assertEqual(indicator_data, {})
    
    async def test_get_position_with_algo_engine(self):
        """Test get_position method with algorithm engine."""
        # Mock binance client response
        self.mock_algo_engine.binance_client.get_open_positions = AsyncMock(return_value=[
            {'symbol': 'BTCUSDT', 'contracts': '0.5', 'side': 'long'}
        ])
        
        self.strategy.set_algo_engine(self.mock_algo_engine)
        
        position = await self.strategy.get_position('BTCUSDT')
        self.assertEqual(position, 0.5)
    
    async def test_get_position_no_position(self):
        """Test get_position method with no open position."""
        self.mock_algo_engine.binance_client.get_open_positions = AsyncMock(return_value=[])
        self.strategy.set_algo_engine(self.mock_algo_engine)
        
        position = await self.strategy.get_position('BTCUSDT')
        self.assertEqual(position, 0.0)
    
    async def test_get_position_without_algo_engine(self):
        """Test get_position method without algorithm engine."""
        position = await self.strategy.get_position('BTCUSDT')
        self.assertEqual(position, 0.0)
    
    async def test_can_open_position_with_no_existing_position(self):
        """Test can_open_position when no position exists."""
        self.mock_algo_engine.binance_client.get_open_positions = AsyncMock(return_value=[])
        self.strategy.set_algo_engine(self.mock_algo_engine)
        
        can_open = await self.strategy.can_open_position('BTCUSDT')
        self.assertTrue(can_open)
    
    async def test_can_open_position_with_existing_position(self):
        """Test can_open_position when position already exists."""
        self.mock_algo_engine.binance_client.get_open_positions = AsyncMock(return_value=[
            {'symbol': 'BTCUSDT', 'contracts': '0.5', 'side': 'long'}
        ])
        self.strategy.set_algo_engine(self.mock_algo_engine)
        
        can_open = await self.strategy.can_open_position('BTCUSDT')
        self.assertFalse(can_open)
    
    @patch('data.indicators.Indicators')
    async def test_calculate_signals_full_workflow(self, mock_indicators_class):
        """Test the complete calculate_signals workflow."""
        # Setup mocks
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={
            'sma_5': np.array([42000.0, 42100.0, 42200.0, 42300.0]),
            'sma_20': np.array([41900.0, 42000.0, 42100.0, 42200.0]),
            'atr_14': np.array([50.0, 55.0, 60.0, 65.0])
        })
        
        signal = await self.strategy.calculate_signals(self.sample_deque, 'BTCUSDT')
        
        self.assertIsInstance(signal, TradeSignal)
        self.assertEqual(signal.symbol, 'BTCUSDT')
        self.assertEqual(signal.strategy_id, 'test_strategy')
        self.assertIsNotNone(signal.timestamp)
    
    async def test_calculate_signals_no_data(self):
        """Test calculate_signals with no data."""
        empty_deque = deque()
        
        signal = await self.strategy.calculate_signals(empty_deque, 'BTCUSDT')
        
        self.assertEqual(signal.action, 'hold')
        self.assertEqual(signal.side, 'none')
        self.assertEqual(signal.metadata['reason'], 'No candle data available')
        self.assertEqual(signal.signal_confidence, 0.0)
    
    @patch('data.indicators.Indicators')
    async def test_calculate_signals_conversion_failure(self, mock_indicators_class):
        """Test calculate_signals when data conversion fails."""
        # Use malformed data that will fail conversion
        bad_deque = deque([[1642680000000]])  # Incomplete candle
        
        signal = await self.strategy.calculate_signals(bad_deque, 'BTCUSDT')
        
        self.assertEqual(signal.action, 'hold')
        self.assertEqual(signal.side, 'none')
        self.assertEqual(signal.metadata['reason'], 'Failed to convert candle data')
    
    @patch('data.indicators.Indicators')
    async def test_calculate_signals_indicator_failure(self, mock_indicators_class):
        """Test calculate_signals when indicator calculation fails."""
        mock_indicators = Mock()
        mock_indicators_class.return_value = mock_indicators
        mock_indicators.calculate_indicators = AsyncMock(return_value={})  # Empty indicators
        
        signal = await self.strategy.calculate_signals(self.sample_deque, 'BTCUSDT')
        
        self.assertEqual(signal.action, 'hold')
        self.assertEqual(signal.side, 'none')
        self.assertEqual(signal.metadata['reason'], 'Failed to calculate indicators')
    
    async def test_signal_generation_logic(self):
        """Test the specific signal generation logic."""
        # Create data where price increases
        increasing_deque = deque([
            [1642680000000, 42000.0, 42100.0, 41900.0, 42000.0, 100.0],  # Close: 42000
            [1642680060000, 42050.0, 42200.0, 42000.0, 42100.0, 105.0],  # Close: 42100 (increase)
        ])
        
        with patch('data.indicators.Indicators') as mock_indicators_class:
            mock_indicators = Mock()
            mock_indicators_class.return_value = mock_indicators
            mock_indicators.calculate_indicators = AsyncMock(return_value={
                'sma_5': np.array([42000.0, 42050.0]),
                'sma_20': np.array([41950.0, 42000.0]),
                'atr_14': np.array([50.0, 55.0])
            })
            
            signal = await self.strategy.calculate_signals(increasing_deque, 'BTCUSDT')
            
            self.assertEqual(signal.action, 'open')
            self.assertEqual(signal.side, 'buy')
            self.assertEqual(signal.metadata['reason'], 'Price increase detected')
            self.assertEqual(signal.signal_confidence, 0.7)
    
    async def test_error_handling_in_calculate_signals(self):
        """Test error handling in calculate_signals."""
        # Mock an exception in the workflow
        with patch.object(self.strategy, '_convert_deque_to_numpy', side_effect=Exception("Test error")):
            signal = await self.strategy.calculate_signals(self.sample_deque, 'BTCUSDT')
            
            self.assertEqual(signal.action, 'hold')
            self.assertEqual(signal.side, 'none')
            self.assertIn('Error during signal calculation', signal.metadata['reason'])
            self.assertEqual(signal.signal_confidence, 0.0)
    
    def test_abstract_method_enforcement(self):
        """Test that abstract methods must be implemented."""
        # This should not be possible to instantiate directly
        with self.assertRaises(TypeError):
            BaseStrategy({}, "test")
    
    def test_default_position_threshold(self):
        """Test default position threshold constant."""
        self.assertTrue(hasattr(BaseStrategy, 'DEFAULT_POSITION_THRESHOLD'))
        self.assertIsInstance(BaseStrategy.DEFAULT_POSITION_THRESHOLD, (int, float))
        self.assertGreater(BaseStrategy.DEFAULT_POSITION_THRESHOLD, 0)
    
    def test_strategy_parameter_access(self):
        """Test strategy parameter access and modification."""
        strategy = ConcreteStrategy({"initial_param": 100})
        
        # Test parameter access
        self.assertEqual(strategy.params["initial_param"], 100)
        
        # Test parameter modification
        strategy.params["new_param"] = "new_value"
        self.assertEqual(strategy.params["new_param"], "new_value")
        
        # Test parameter update
        strategy.params.update({"batch_param1": 1, "batch_param2": 2})
        self.assertEqual(strategy.params["batch_param1"], 1)
        self.assertEqual(strategy.params["batch_param2"], 2)
    
    def test_strategy_id_immutability(self):
        """Test that strategy_id should remain constant."""
        original_id = self.strategy.strategy_id
        
        # While not technically immutable, it shouldn't be changed
        self.assertEqual(self.strategy.strategy_id, "test_strategy")
        
        # Test that it persists through operations
        self.strategy.params["new_param"] = "value"
        self.assertEqual(self.strategy.strategy_id, original_id)
    
    async def test_position_size_calculation_integration(self):
        """Test integration with position size calculation."""
        self.mock_algo_engine.binance_client.get_open_positions = AsyncMock(return_value=[
            {'symbol': 'BTCUSDT', 'contracts': '0.001', 'side': 'long'}  # Very small position
        ])
        self.strategy.set_algo_engine(self.mock_algo_engine)
        
        position = await self.strategy.get_position('BTCUSDT')
        can_open = await self.strategy.can_open_position('BTCUSDT')
        
        # Should detect small position
        self.assertEqual(position, 0.001)
        
        # Should not allow opening new position due to existing position
        self.assertFalse(can_open)
    
    def test_numpy_data_types(self):
        """Test handling of different numpy data types."""
        # Test with float64 data (default)
        float_deque = deque([
            [1642680000000.0, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
        ])
        
        np_data = self.strategy._convert_deque_to_numpy(float_deque)
        self.assertEqual(np_data['close'][0], 42050.0)
        
        # Test with integer data
        int_deque = deque([
            [1642680000000, 42000, 42100, 41900, 42050, 100]
        ])
        
        np_data_int = self.strategy._convert_deque_to_numpy(int_deque)
        self.assertEqual(np_data_int['close'][0], 42050)
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large deque
        large_deque = deque()
        for i in range(1000):
            candle = [
                1642680000000 + i * 60000,  # Timestamp
                42000.0 + i,                 # Open
                42100.0 + i,                 # High
                41900.0 + i,                 # Low
                42050.0 + i,                 # Close
                100.0 + i                    # Volume
            ]
            large_deque.append(candle)
        
        np_data = self.strategy._convert_deque_to_numpy(large_deque)
        
        self.assertEqual(len(np_data['close']), 1000)
        self.assertEqual(np_data['close'][0], 42050.0)
        self.assertEqual(np_data['close'][-1], 42050.0 + 999)


if __name__ == '__main__':
    unittest.main(verbosity=2)
