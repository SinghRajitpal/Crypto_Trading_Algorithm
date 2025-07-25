"""
Comprehensive Algorithm Engine Test Suite  
Senior Quantitative Systems Testing

This test suite provides ultra-detailed coverage of the Algorithm Engine including:
- AlgoEngine signal processing and throttling
- Strategy interface and integration
- TradeSignal validation and metadata handling
- Error handling and resilience
- Performance characteristics and concurrent processing
"""
import unittest
import asyncio
import time
import sys
import os
import gc
import warnings
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from collections import deque
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy
from data.data_engine import DataEngine


class MockDataEngine:
    """Enhanced mock data engine for comprehensive testing."""
    
    def __init__(self, candles_data=None):
        self.binance_client = Mock()
        self.candles_data = candles_data or {}
        self.running = False
        
        # Default sample candles for testing
        self.default_candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
            [1642680120000, 42150.0, 42300.0, 42100.0, 42250.0, 110.0],
            [1642680180000, 42250.0, 42400.0, 42200.0, 42350.0, 115.0],
            [1642680240000, 42350.0, 42500.0, 42300.0, 42450.0, 120.0]
        ]
    
    def get_candles(self, symbol, timeframe):
        """Mock get_candles method."""
        key = (symbol, timeframe)
        if key in self.candles_data:
            return self.candles_data[key].copy()
        # Return default candles for any symbol/timeframe not specifically mocked
        return self.default_candles.copy()
    
    def get_latest_candle(self, symbol, timeframe):
        """Mock get_latest_candle method."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None
    
    def set_candles(self, symbol, timeframe, candles):
        """Helper method to set candles for testing."""
        self.candles_data[(symbol, timeframe)] = candles


class MockStrategy(BaseStrategy):
    """Enhanced mock strategy for comprehensive testing."""
    
    def __init__(self, params=None, strategy_id="mock_strategy"):
        super().__init__(params or {}, strategy_id)
        self.calculate_signals_called = False
        self.call_count = 0
        self.mock_signal = None
        self.exception_to_raise = None
        self.call_history = []
        
    def get_required_indicators(self):
        return ["sma_5", "sma_20", "atr_14"]
    
    async def calculate_signals(self, data, symbol):
        """Mock calculate_signals that tracks calls and can simulate various behaviors."""
        self.calculate_signals_called = True
        self.call_count += 1
        self.call_history.append({
            'symbol': symbol,
            'data_length': len(data) if data else 0,
            'timestamp': time.time()
        })
        
        # Raise exception if configured to do so
        if self.exception_to_raise:
            raise self.exception_to_raise
        
        # Return configured signal if set
        if self.mock_signal:
            return self.mock_signal
        
        # Default behavior: return hold signal
        return TradeSignal(
            action="hold",
            side="none",
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"reason": "Mock hold signal", "call_count": self.call_count},
            signal_confidence=0.5
        )
    
    async def _generate_signals(self, data, indicator_data, symbol):
        """Required abstract method implementation."""
        return await self.calculate_signals(data, symbol)


class MockBuyStrategy(MockStrategy):
    """Mock strategy that always generates buy signals."""
    
    def __init__(self, params=None):
        super().__init__(params, "mock_buy_strategy")
    
    async def calculate_signals(self, data, symbol):
        self.calculate_signals_called = True
        self.call_count += 1
        
        return TradeSignal(
            action="open",
            side="buy",
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"reason": "Mock buy signal", "price": data[-1][4] if data else 0},
            signal_confidence=0.8
        )


class MockSellStrategy(MockStrategy):
    """Mock strategy that always generates sell signals."""
    
    def __init__(self, params=None):
        super().__init__(params, "mock_sell_strategy")
    
    async def calculate_signals(self, data, symbol):
        self.calculate_signals_called = True
        self.call_count += 1
        
        return TradeSignal(
            action="open",
            side="sell",
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"reason": "Mock sell signal", "price": data[-1][4] if data else 0},
            signal_confidence=0.7
        )


class MockErrorStrategy(MockStrategy):
    """Mock strategy that raises exceptions."""
    
    def __init__(self, exception_type=Exception, error_message="Test error"):
        super().__init__({}, "mock_error_strategy")
        self.exception_type = exception_type
        self.error_message = error_message
    
    async def calculate_signals(self, data, symbol):
        self.calculate_signals_called = True
        raise self.exception_type(self.error_message)


class TestAlgoEngine(unittest.TestCase):
    """Comprehensive AlgoEngine testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.engine = AlgoEngine(self.mock_data_engine)
        self.mock_strategy = MockStrategy()
    
    def test_initialization_comprehensive(self):
        """Test AlgoEngine initialization with various scenarios."""
        # Test standard initialization
        engine = AlgoEngine(self.mock_data_engine)
        self.assertEqual(engine.data_engine, self.mock_data_engine)
        self.assertFalse(engine.running)
        self.assertEqual(engine._last_signal_states, {})
        self.assertEqual(engine._min_signal_interval, 60)
        self.assertEqual(engine.binance_client, self.mock_data_engine.binance_client)
        
        # Test with data engine that has no binance_client
        data_engine_no_client = Mock()
        del data_engine_no_client.binance_client  # Remove the attribute
        engine_no_client = AlgoEngine(data_engine_no_client)
        self.assertIsNone(engine_no_client.binance_client)
    
    def test_data_hash_generation_comprehensive(self):
        """Test data hash generation with various inputs."""
        # Test with valid candles
        candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42150.0, 105.0],
        ]
        
        hash_value = self.engine._get_data_hash(candles)
        expected_hash = f"{candles[-1][0]}_{candles[-1][4]}"  # timestamp_close of latest candle
        self.assertEqual(hash_value, expected_hash)
        
        # Test with single candle
        single_candle = [[1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]]
        single_hash = self.engine._get_data_hash(single_candle)
        expected_single = f"{single_candle[0][0]}_{single_candle[0][4]}"
        self.assertEqual(single_hash, expected_single)
        
        # Test with empty candles
        empty_hash = self.engine._get_data_hash([])
        self.assertEqual(empty_hash, "")
        
        # Test with None
        none_hash = self.engine._get_data_hash(None)
        self.assertEqual(none_hash, "")
        
        # Test consistency (same input should produce same hash)
        hash1 = self.engine._get_data_hash(candles)
        hash2 = self.engine._get_data_hash(candles)
        self.assertEqual(hash1, hash2)
        
        # Test different data produces different hash
        different_candles = [
            [1642680000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1642680060000, 42050.0, 42200.0, 42000.0, 42200.0, 105.0],  # Different close
        ]
        different_hash = self.engine._get_data_hash(different_candles)
        self.assertNotEqual(hash_value, different_hash)
    
    def test_should_process_signal_initial_state(self):
        """Test signal processing decision for initial signals."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        data_hash = "test_hash"
        
        # First signal should always be processed
        should_process = self.engine._should_process_signal(key, current_time, data_hash)
        self.assertTrue(should_process)
        
        # Verify no state exists yet
        self.assertNotIn(key, self.engine._last_signal_states)
    
    def test_signal_state_isolation_between_symbols(self):
        """Test signal state isolation between different symbols and timeframes."""
        current_time = int(time.time())
        data_hash = "test_hash"
        signal_type = "open/buy"
        
        # Test different symbol-timeframe combinations
        keys = ["BTCUSDT_1m", "BTCUSDT_5m", "ETHUSDT_1m", "ETHUSDT_5m"]
        
        # Update states for all keys
        for key in keys:
            self.engine._update_signal_state(key, current_time, data_hash, signal_type)
        
        # Verify each key has isolated state
        for key in keys:
            self.assertIn(key, self.engine._last_signal_states)
            state = self.engine._last_signal_states[key]
            self.assertEqual(state['timestamp'], current_time)
            self.assertEqual(state['data_hash'], data_hash)
            self.assertEqual(state['signal_type'], signal_type)
    
    def test_process_signals_no_data(self):
        """Test signal processing with no data available."""
        async def _test_async():
            # Mock empty data
            self.mock_data_engine.set_candles("BTCUSDT", "1m", [])
            
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            # Should return None when no data is available
            self.assertIsNone(signal)
        
        asyncio.run(_test_async())
    
    def test_process_signals_valid_data_flow(self):
        """Test signal processing with valid data flow."""
        async def _test_async():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            # Configure mock strategy to return a specific signal
            expected_signal = TradeSignal(
                action="open",
                side="buy",
                symbol=symbol,
                strategy_id=self.mock_strategy.strategy_id,
                metadata={"test": "valid_flow"},
                signal_confidence=0.8
            )
            self.mock_strategy.mock_signal = expected_signal
            
            # Process signals
            signal = await self.engine.process_signals(symbol, timeframe, self.mock_strategy)
            
            # Verify signal processing
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "open")
            self.assertEqual(signal.side, "buy")
            self.assertEqual(signal.symbol, symbol)
            self.assertTrue(self.mock_strategy.calculate_signals_called)
        
        asyncio.run(_test_async())
    
    def test_process_signals_exception_handling(self):
        """Test signal processing with strategy exceptions."""
        async def _test_async():
            # Configure strategy to raise exception
            error_strategy = MockErrorStrategy(ValueError, "Strategy calculation error")
            
            # Should handle exception gracefully
            signal = await self.engine.process_signals("BTCUSDT", "1m", error_strategy)
            
            # Should return None when strategy raises exception
            self.assertIsNone(signal)
            self.assertTrue(error_strategy.calculate_signals_called)
        
        asyncio.run(_test_async())
    
    def test_process_signals_timestamp_setting(self):
        """Test that signal timestamps are set correctly."""
        async def _test_async():
            # Configure mock strategy to return a signal without timestamp
            # The AlgoEngine should set the timestamp after processing
            self.mock_strategy.mock_signal = TradeSignal(
                action="hold",
                side="none",
                symbol="BTCUSDT",
                strategy_id=self.mock_strategy.strategy_id,
                metadata={"test": "timestamp_check"},
                signal_confidence=0.5
            )
            
            before_time = int(time.time())  # Get seconds
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            after_time = int(time.time())  # Get seconds
            
            # Verify timestamp is set and reasonable
            self.assertIsNotNone(signal)
            self.assertIsNotNone(signal.timestamp)
            
            # Timestamp should be in milliseconds (seconds * 1000)
            signal_seconds = signal.timestamp // 1000
            self.assertGreaterEqual(signal_seconds, before_time)
            self.assertLessEqual(signal_seconds, after_time + 1)  # Allow 1 second tolerance
            
            # Verify it's actually in millisecond format
            self.assertGreater(signal.timestamp, 1000000000000)  # After year 2001 in milliseconds
        
        asyncio.run(_test_async())


class TestTradeSignal(unittest.TestCase):
    """Comprehensive TradeSignal testing."""
    
    def test_valid_signal_creation(self):
        """Test creation of valid TradeSignal instances."""
        # Test basic valid signal
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test_strategy",
            metadata={"reason": "test"},
            signal_confidence=0.8
        )
        
        self.assertEqual(signal.action, "open")
        self.assertEqual(signal.side, "buy")
        self.assertEqual(signal.symbol, "BTCUSDT")
        self.assertEqual(signal.strategy_id, "test_strategy")
        self.assertEqual(signal.signal_confidence, 0.8)
        self.assertIsInstance(signal.timestamp, (int, float))
        # Default timestamp is 0 until set by the engine
        self.assertEqual(signal.timestamp, 0)
    
    def test_signal_validation_invalid_actions(self):
        """Test signal validation with invalid actions."""
        with self.assertRaises(ValueError):
            TradeSignal(
                action="invalid_action",
                side="buy",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata={},
                signal_confidence=0.5
            )
    
    def test_signal_validation_invalid_sides(self):
        """Test signal validation with invalid sides."""
        with self.assertRaises(ValueError):
            TradeSignal(
                action="open",
                side="invalid_side",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata={},
                signal_confidence=0.5
            )
    
    def test_signal_all_valid_combinations(self):
        """Test all valid action-side combinations."""
        valid_combinations = [
            ("open", "buy"),
            ("open", "sell"),
            ("exit", "buy"),
            ("exit", "sell"),
            ("hold", "none")
        ]
        
        for action, side in valid_combinations:
            with self.subTest(action=action, side=side):
                signal = TradeSignal(
                    action=action,
                    side=side,
                    symbol="BTCUSDT",
                    strategy_id="test",
                    metadata={"test": "valid_combination"},
                    signal_confidence=0.6
                )
                self.assertEqual(signal.action, action)
                self.assertEqual(signal.side, side)


class TestAlgorithmEngineHedgeFundStandards(unittest.TestCase):
    """Enhanced Algorithm Engine testing with hedge fund standards."""
    
    def setUp(self):
        """Set up hedge fund standard test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.engine = AlgoEngine(self.mock_data_engine)
        self.mock_strategy = MockStrategy()
    
    def test_market_crash_signal_processing(self):
        """Test signal processing during market crash scenarios."""
        async def _test_async():
            # Create crash scenario candles (20% drop)
            crash_candles = [
                [1642680000000, 50000.0, 50500.0, 49500.0, 50000.0, 1000.0],  # Normal
                [1642680060000, 50000.0, 50100.0, 45000.0, 45000.0, 5000.0],  # Crash
                [1642680120000, 45000.0, 46000.0, 44000.0, 45500.0, 3000.0],  # Recovery
            ]
            
            self.mock_data_engine.set_candles("BTCUSDT", "1m", crash_candles)
            
            # Configure strategy to detect crash
            crash_signal = TradeSignal(
                action="exit",
                side="sell",
                symbol="BTCUSDT",
                strategy_id=self.mock_strategy.strategy_id,
                metadata={
                    "reason": "market_crash_detected",
                    "price_drop": -0.20,
                    "volume_spike": 5.0
                },
                signal_confidence=0.95
            )
            self.mock_strategy.mock_signal = crash_signal
            
            signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "exit")
            self.assertIn("market_crash_detected", signal.metadata.get("reason", ""))
            self.assertGreaterEqual(signal.signal_confidence, 0.9)
        
        asyncio.run(_test_async())
    
    def test_extreme_volatility_signal_throttling(self):
        """Test signal throttling during extreme volatility."""
        async def _test_async():
            # Create high volatility scenario
            volatile_candles = []
            base_price = 42000.0
            for i in range(10):
                # Simulate 5% swings each minute
                swing = 0.05 * (1 if i % 2 == 0 else -1)
                price = base_price * (1 + swing)
                candle = [1642680000000 + i * 60000, base_price, price * 1.02, price * 0.98, price, 1000.0]
                volatile_candles.append(candle)
                base_price = price
            
            self.mock_data_engine.set_candles("BTCUSDT", "1m", volatile_candles)
            
            # Process signals multiple times in short succession
            signals = []
            for _ in range(5):
                signal = await self.engine.process_signals("BTCUSDT", "1m", self.mock_strategy)
                signals.append(signal)
                await asyncio.sleep(0.001)  # Small delay
            
            # Verify throttling is working
            non_none_signals = [s for s in signals if s is not None]
            self.assertLessEqual(len(non_none_signals), 2)  # Should be throttled
        
        asyncio.run(_test_async())


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.main(verbosity=2, buffer=True)
