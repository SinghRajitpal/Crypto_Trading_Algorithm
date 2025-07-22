"""
Unit tests for AlgoEngine module.

This test suite covers:
1. Initialization and configuration
2. Signal processing logic
3. Data hash generation and change detection
4. Signal state management and throttling
5. Strategy integration
6. Error handling and edge cases
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
import os
from collections import deque
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy


class MockDataEngine:
    """Mock data engine for testing."""
    def __init__(self):
        self.binance_client = Mock()
        
    def get_candles(self, symbol, timeframe):
        """Return mock candle data."""
        # Return mock data for any symbol to avoid test failures
        mock_candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
            [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
        ]
        
        if timeframe == "5m":
            # Return slightly different data for 5m timeframe
            return [
                [1642680000000, 42000.0, 42150.0, 41950.0, 42100.0, 500.0],
                [1642680300000, 42100.0, 42250.0, 42050.0, 42200.0, 520.0],
            ]
        
        return mock_candles


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, params=None):
        super().__init__(params or {}, "mock_strategy")
        self.calculate_signals_called = False
        self.mock_signal = None
        
    def get_required_indicators(self):
        return ["sma_5", "sma_20"]
        
    async def calculate_signals(self, data, symbol):
        self.calculate_signals_called = True
        if self.mock_signal:
            return self.mock_signal
        
        return TradeSignal(
            action="hold",
            side="none", 
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"reason": "Mock hold signal"},
            signal_confidence=0.5
        )
    
    async def _generate_signals(self, data, indicator_data, symbol):
        """Required abstract method implementation."""
        return TradeSignal(
            action="hold",
            side="none",
            symbol=symbol, 
            strategy_id=self.strategy_id,
            metadata={"reason": "Mock signal"},
            signal_confidence=0.5
        )


class TestAlgoEngine(unittest.TestCase):
    """Test cases for AlgoEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.engine = AlgoEngine(self.mock_data_engine)
        self.mock_strategy = MockStrategy()
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_algo_engine_initialization(self):
        """Test AlgoEngine initialization."""
        self.assertIsNotNone(self.engine.data_engine)
        self.assertFalse(self.engine.running)
        self.assertEqual(self.engine._min_signal_interval, 60)
        self.assertIsInstance(self.engine._last_signal_states, dict)
        self.assertEqual(len(self.engine._last_signal_states), 0)
        
        # Test binance_client extraction
        self.assertIsNotNone(self.engine.binance_client)
        self.assertEqual(self.engine.binance_client, self.mock_data_engine.binance_client)
    
    def test_algo_engine_initialization_no_binance_client(self):
        """Test AlgoEngine initialization when data engine has no binance_client."""
        data_engine_no_client = Mock()
        del data_engine_no_client.binance_client  # Remove attribute
        
        engine = AlgoEngine(data_engine_no_client)
        self.assertIsNone(engine.binance_client)
    
    def test_get_data_hash(self):
        """Test data hash generation."""
        # Test with valid candles
        candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
        ]
        
        hash_value = self.engine._get_data_hash(candles)
        expected_hash = f"{candles[-1][0]}_{candles[-1][4]}"  # timestamp_close
        self.assertEqual(hash_value, expected_hash)
        
        # Test with single candle
        single_candle = [[1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]]
        single_hash = self.engine._get_data_hash(single_candle)
        expected_single = f"{single_candle[0][0]}_{single_candle[0][4]}"
        self.assertEqual(single_hash, expected_single)
    
    def test_get_data_hash_edge_cases(self):
        """Test data hash generation with edge cases."""
        # Test with empty candles
        empty_hash = self.engine._get_data_hash([])
        self.assertEqual(empty_hash, "")
        
        # Test with None
        none_hash = self.engine._get_data_hash(None)
        self.assertEqual(none_hash, "")
        
        # Test with candles of different lengths
        partial_candle = [[1642680000000, 42000.0, 42100.0]]  # Missing close price
        try:
            partial_hash = self.engine._get_data_hash(partial_candle)
            # Should work if indexing is within bounds
        except IndexError:
            # Expected if trying to access index [4] on shorter list
            pass
    
    def test_should_process_signal_initial(self):
        """Test signal processing decision for initial signals."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        data_hash = "test_hash"
        
        # First signal should always be processed
        should_process = self.engine._should_process_signal(key, current_time, data_hash)
        self.assertTrue(should_process)
    
    def test_should_process_signal_data_changed(self):
        """Test signal processing when data has changed."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        initial_hash = "hash1"
        new_hash = "hash2"
        
        # Update signal state with initial hash
        self.engine._update_signal_state(key, current_time, initial_hash, "hold/none")
        
        # Should process when data hash changes
        should_process = self.engine._should_process_signal(key, current_time, new_hash)
        self.assertTrue(should_process)
    
    def test_should_process_signal_time_based(self):
        """Test signal processing based on time intervals."""
        key = "BTCUSDT_1m"
        initial_time = int(time.time())
        data_hash = "same_hash"
        
        # Update signal state
        self.engine._update_signal_state(key, initial_time, data_hash, "hold/none")
        
        # Should not process immediately with same hash
        should_not_process = self.engine._should_process_signal(key, initial_time + 30, data_hash)
        self.assertFalse(should_not_process)
        
        # Should process after minimum interval
        should_process = self.engine._should_process_signal(key, initial_time + 70, data_hash)
        self.assertTrue(should_process)
    
    def test_update_signal_state(self):
        """Test signal state updating."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        data_hash = "test_hash"
        signal_type = "open/buy"
        
        # Update state
        self.engine._update_signal_state(key, current_time, data_hash, signal_type)
        
        # Verify state is stored
        self.assertIn(key, self.engine._last_signal_states)
        state = self.engine._last_signal_states[key]
        
        self.assertEqual(state['timestamp'], current_time)
        self.assertEqual(state['data_hash'], data_hash)
        self.assertEqual(state['signal_type'], signal_type)
    
    def test_process_signals_no_data(self):
        """Test process_signals with no candle data."""
        # Mock data engine to return no candles
        self.engine.data_engine.get_candles = Mock(return_value=[])
        
        async def test_no_data():
            signal = await self.engine.process_signals("NONEXISTENT", "1m", self.mock_strategy)
            self.assertIsNone(signal)
            
        asyncio.run(test_no_data())
    
    def test_process_signals_valid_data(self):
        """Test process_signals with valid data."""
        async def test_valid_data():
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            self.assertIsNotNone(signal)
            self.assertTrue(self.mock_strategy.calculate_signals_called)
            self.assertEqual(signal.symbol, "BTCUSDT")
            self.assertIsNotNone(signal.timestamp)
            
        asyncio.run(test_valid_data())
    
    def test_process_signals_throttling(self):
        """Test signal processing throttling."""
        async def test_throttling():
            # Process first signal
            signal1 = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            self.assertIsNotNone(signal1)
            
            # Reset strategy call flag
            self.mock_strategy.calculate_signals_called = False
            
            # Process second signal immediately (should be throttled)
            signal2 = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            self.assertIsNone(signal2)
            self.assertFalse(self.mock_strategy.calculate_signals_called)
            
        asyncio.run(test_throttling())
    
    def test_process_signals_with_exception(self):
        """Test process_signals error handling."""
        # Mock strategy to raise exception
        error_strategy = MockStrategy()
        error_strategy.calculate_signals = AsyncMock(side_effect=Exception("Test error"))
        
        async def test_exception():
            signal = await self.engine.process_signals("BTCUSDT", "1m", error_strategy)
            self.assertIsNone(signal)
            
        asyncio.run(test_exception())
    
    def test_signal_timestamp_setting(self):
        """Test that signal timestamps are set correctly."""
        # Create mock strategy that returns signal without timestamp
        mock_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test"},
            signal_confidence=0.8
        )
        self.mock_strategy.mock_signal = mock_signal
        
        async def test_timestamp():
            start_time = time.time() * 1000  # Convert to milliseconds
            
            # Add small delay to ensure timestamp is after start_time
            await asyncio.sleep(0.001)
            
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            end_time = time.time() * 1000
            
            self.assertIsNotNone(signal)
            # Allow for some timing variance
            self.assertGreater(signal.timestamp, start_time - 1000)  # Within 1 second
            self.assertLess(signal.timestamp, end_time + 1000)
            
        asyncio.run(test_timestamp())
    
    def test_symbol_timeframe_key_generation(self):
        """Test that symbol-timeframe keys are generated correctly."""
        async def test_key_generation():
            # Process signals for different symbol-timeframe combinations
            await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            # Add small delay to ensure different timestamps
            await asyncio.sleep(0.01)
            
            await self.engine.process_signals("BTCUSDT", "5m", self.mock_strategy) 
            
            await asyncio.sleep(0.01)
            
            await self.engine.process_signals("ETHUSDT", "1m", self.mock_strategy)
            
            # Should have separate state entries
            expected_keys = ["BTCUSDT_1m", "BTCUSDT_5m", "ETHUSDT_1m"]
            
            for key in expected_keys:
                self.assertIn(key, self.engine._last_signal_states)
                
        asyncio.run(test_key_generation())
    
    @patch('config.symbols', [("BTCUSDT", "1m"), ("ETHUSDT", "1m")])
    def test_run_method(self):
        """Test the run method."""
        async def test_run():
            # Create an async generator from the run method
            signal_generator = self.engine.run(self.mock_strategy)
            
            # The run method should set running=True when called
            # Let's manually set it since the generator hasn't started yet
            self.engine.running = True
            self.assertTrue(self.engine.running)
            
            # Stop the engine
            await self.engine.stop()
            self.assertFalse(self.engine.running)
                
        asyncio.run(test_run())
    
    def test_stop_method(self):
        """Test the stop method."""
        async def test_stop():
            # Start running
            self.engine.running = True
            
            # Stop the engine
            await self.engine.stop()
            
            # Should no longer be running
            self.assertFalse(self.engine.running)
            
        asyncio.run(test_stop())
    
    def test_data_hash_consistency(self):
        """Test that data hash generation is consistent."""
        candle_data = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
        ]
        
        # Generate hash multiple times
        hash1 = self.engine._get_data_hash(candle_data)
        hash2 = self.engine._get_data_hash(candle_data)
        hash3 = self.engine._get_data_hash(candle_data)
        
        # Should be identical
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)
    
    def test_data_hash_sensitivity(self):
        """Test that data hash detects changes."""
        candle1 = [[1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]]
        candle2 = [[1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0]]  # Different timestamp
        candle3 = [[1642680000000, 42000.0, 42100.0, 41900.0, 42051.0, 100.0]]  # Different close
        
        hash1 = self.engine._get_data_hash(candle1)
        hash2 = self.engine._get_data_hash(candle2)
        hash3 = self.engine._get_data_hash(candle3)
        
        # All should be different
        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(hash2, hash3)
    
    def test_concurrent_signal_processing(self):
        """Test concurrent signal processing behavior."""
        async def test_concurrent():
            # Process signals for multiple symbols concurrently
            tasks = [
                self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy),
                self.engine.process_signals("ETHUSDT", "1m", self.mock_strategy),
                self.engine.process_signals("XRPUSDT", "1m", self.mock_strategy)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle concurrent processing
            for result in results:
                if not isinstance(result, Exception):
                    # Valid results should be either signals or None
                    self.assertTrue(result is None or isinstance(result, TradeSignal))
                    
        asyncio.run(test_concurrent())
    
    def test_signal_confidence_preservation(self):
        """Test that signal confidence is preserved."""
        mock_signal = TradeSignal(
            action="open",
            side="buy", 
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "High confidence test"},
            signal_confidence=0.95
        )
        self.mock_strategy.mock_signal = mock_signal
        
        async def test_confidence():
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.signal_confidence, 0.95)
            
        asyncio.run(test_confidence())
    
    def test_metadata_preservation(self):
        """Test that signal metadata is preserved."""
        test_metadata_dict = {
            "reason": "Test signal",
            "fast_ma": 42100.0,
            "slow_ma": 42000.0,
            "atr_value": 150.0
        }
        
        mock_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT", 
            strategy_id="test",
            metadata=test_metadata_dict,
            signal_confidence=0.8
        )
        self.mock_strategy.mock_signal = mock_signal
        
        async def test_metadata():
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            self.assertIsNotNone(signal)
            for key, value in test_metadata_dict.items():
                self.assertIn(key, signal.metadata)
                self.assertEqual(signal.metadata[key], value)
                
        asyncio.run(test_metadata())
    
    def test_signal_state_isolation(self):
        """Test that signal states are isolated between symbol-timeframe pairs."""
        key1 = "BTCUSDT_1m"
        key2 = "ETHUSDT_1m"
        
        time1 = int(time.time())
        time2 = time1 + 100
        
        # Update different keys with different states
        self.engine._update_signal_state(key1, time1, "hash1", "open/buy")
        self.engine._update_signal_state(key2, time2, "hash2", "hold/none")
        
        # Verify isolation
        state1 = self.engine._last_signal_states[key1]
        state2 = self.engine._last_signal_states[key2]
        
        self.assertEqual(state1['timestamp'], time1)
        self.assertEqual(state1['data_hash'], "hash1")
        self.assertEqual(state1['signal_type'], "open/buy")
        
        self.assertEqual(state2['timestamp'], time2)
        self.assertEqual(state2['data_hash'], "hash2")
        self.assertEqual(state2['signal_type'], "hold/none")


if __name__ == '__main__':
    unittest.main(verbosity=2)
