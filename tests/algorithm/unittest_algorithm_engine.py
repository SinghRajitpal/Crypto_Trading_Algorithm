#!/usr/bin/env python3
"""
Unittest-compatible Test Suite for Algorithm Engine
Senior Quantitative Developer Testing Protocol

Comprehensive unittest-based testing of the algorithm engine functionality.
"""

import unittest
import asyncio
import sys
import os
from typing import Dict, List, Any
from unittest.mock import Mock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy


class MockDataEngineForUnittest:
    """Mock data engine for unittest testing."""
    
    def __init__(self):
        self.call_count = {}
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Generate mock candle data."""
        import time
        current_time = int(time.time() * 1000)
        
        # Track calls for different behavior
        key = f"{symbol}_{timeframe}"
        self.call_count[key] = self.call_count.get(key, 0) + 1
        
        # Generate slightly different data each call to test data hash
        offset = self.call_count[key] * 60000  # 1 minute offset per call
        
        base_price = 50000.0 if 'BTC' in symbol else 3000.0
        
        candles = []
        for i in range(20):
            timestamp = current_time - offset + (i * 60000)
            price = base_price + (i * 10)  # Slight price increase
            
            candles.append([
                timestamp,
                price,         # open
                price + 5,     # high
                price - 5,     # low
                price + 2,     # close
                100.0         # volume
            ])
        
        return candles


class SimpleTestStrategyForUnittest(BaseStrategy):
    """Simple test strategy for unittest."""
    
    def __init__(self, signal_type: str):
        super().__init__(params={}, strategy_id="unittest_strategy")
        self.signal_type = signal_type
        
    def get_required_indicators(self) -> List[str]:
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        if self.signal_type == "buy":
            return {"action": "open", "side": "buy", "confidence": 0.8}
        elif self.signal_type == "sell":
            return {"action": "open", "side": "sell", "confidence": 0.8}
        else:
            return {"action": "hold", "side": "none", "confidence": 0.5}
    
    async def calculate_signals(self, candles, symbol: str):
        """Override to return TradeSignal."""
        result = self._generate_signals(symbol, {})
        
        return TradeSignal(
            action=result["action"],
            side=result["side"],
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"test": True},
            signal_confidence=result["confidence"]
        )


class TestAlgoEngineBasics(unittest.TestCase):
    """Test basic algorithm engine functionality using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngineForUnittest()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_initialization(self):
        """Test algorithm engine initialization."""
        self.assertIsNotNone(self.algo_engine)
        self.assertIsNotNone(self.algo_engine.data_engine)
        self.assertFalse(self.algo_engine.running)
        self.assertEqual(self.algo_engine._last_signal_states, {})
        self.assertGreater(self.algo_engine._min_signal_interval, 0)
    
    def test_data_hash_generation(self):
        """Test data hash generation consistency."""
        candles = self.mock_data_engine.get_candles("BTCUSDT", "1m")
        
        hash1 = self.algo_engine._get_data_hash(candles)
        hash2 = self.algo_engine._get_data_hash(candles)
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertGreater(len(hash1), 0)
    
    def test_signal_state_tracking(self):
        """Test signal state tracking mechanisms."""
        key = "BTCUSDT_1m"
        current_time = 1234567890
        data_hash = "test_hash"
        signal_type = "open/buy"
        
        # Initially should not exist
        self.assertNotIn(key, self.algo_engine._last_signal_states)
        
        # Update state
        self.algo_engine._update_signal_state(key, current_time, data_hash, signal_type)
        
        # Should now exist
        self.assertIn(key, self.algo_engine._last_signal_states)
        state = self.algo_engine._last_signal_states[key]
        
        self.assertEqual(state['timestamp'], current_time)
        self.assertEqual(state['data_hash'], data_hash)
        self.assertEqual(state['signal_type'], signal_type)
    
    def test_should_process_signal_logic(self):
        """Test signal processing decision logic."""
        key = "BTCUSDT_1m"
        base_time = 1234567890
        data_hash = "test_hash"
        
        # Should process when no prior state
        self.assertTrue(self.algo_engine._should_process_signal(key, base_time, data_hash))
        
        # Set initial state
        self.algo_engine._update_signal_state(key, base_time, data_hash, "open/buy")
        
        # Should not process immediately with same data
        self.assertFalse(self.algo_engine._should_process_signal(key, base_time + 30, data_hash))
        
        # Should process with different data
        self.assertTrue(self.algo_engine._should_process_signal(key, base_time + 30, "new_hash"))
        
        # Should process after time interval
        future_time = base_time + self.algo_engine._min_signal_interval + 1
        self.assertTrue(self.algo_engine._should_process_signal(key, future_time, data_hash))


class TestAlgoEngineSignalProcessing(unittest.TestCase):
    """Test signal processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngineForUnittest()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_signal_processing_with_buy_strategy(self):
        """Test signal processing with buy strategy."""
        async def run_test():
            strategy = SimpleTestStrategyForUnittest("buy")
            
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "open")
            self.assertEqual(signal.side, "buy")
            self.assertEqual(signal.symbol, "BTCUSDT")
            self.assertEqual(signal.strategy_id, "unittest_strategy")
            self.assertGreater(signal.signal_confidence, 0)
        
        asyncio.run(run_test())
    
    def test_signal_processing_with_hold_strategy(self):
        """Test signal processing with hold strategy."""
        async def run_test():
            strategy = SimpleTestStrategyForUnittest("hold")
            
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "hold")
            self.assertEqual(signal.side, "none")
        
        asyncio.run(run_test())
    
    def test_multi_symbol_processing(self):
        """Test processing signals for multiple symbols."""
        async def run_test():
            strategy = SimpleTestStrategyForUnittest("buy")
            symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
            
            signals = []
            for symbol in symbols:
                signal = await self.algo_engine.process_signals(symbol, "1m", strategy)
                if signal:
                    signals.append(signal)
            
            self.assertEqual(len(signals), 3)
            
            # Check each signal has correct symbol
            for i, signal in enumerate(signals):
                self.assertEqual(signal.symbol, symbols[i])
                self.assertEqual(signal.action, "open")
                self.assertEqual(signal.side, "buy")
        
        asyncio.run(run_test())


class TestAlgoEngineErrorHandling(unittest.TestCase):
    """Test error handling capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngineForUnittest()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_empty_candle_data_handling(self):
        """Test handling of empty candle data."""
        async def run_test():
            # Mock empty data
            self.mock_data_engine.get_candles = Mock(return_value=[])
            
            strategy = SimpleTestStrategyForUnittest("buy")
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            # Should return None for empty data
            self.assertIsNone(signal)
        
        asyncio.run(run_test())
    
    def test_data_engine_exception_handling(self):
        """Test handling of data engine exceptions."""
        async def run_test():
            # Mock exception
            self.mock_data_engine.get_candles = Mock(side_effect=Exception("Data error"))
            
            strategy = SimpleTestStrategyForUnittest("buy")
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            # Should return None when data engine fails
            self.assertIsNone(signal)
        
        asyncio.run(run_test())


class TestAlgoEnginePerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngineForUnittest()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_rapid_signal_processing(self):
        """Test rapid signal processing performance."""
        async def run_test():
            import time
            
            strategy = SimpleTestStrategyForUnittest("buy")
            start_time = time.time()
            
            # Process multiple signals
            signals_processed = 0
            for i in range(10):
                self.algo_engine._last_signal_states.clear()  # Allow all signals
                signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
                if signal:
                    signals_processed += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process rapidly
            self.assertLess(processing_time, 1.0)  # Under 1 second
            self.assertGreater(signals_processed, 0)
        
        asyncio.run(run_test())


class TestSuiteRunner(unittest.TestCase):
    """Main test suite runner for unittest compatibility."""
    
    def test_run_all_algorithm_tests(self):
        """Run all algorithm engine tests as a single test method."""
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestAlgoEngineBasics,
            TestAlgoEngineSignalProcessing,
            TestAlgoEngineErrorHandling,
            TestAlgoEnginePerformance
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Assert overall success
        self.assertTrue(result.wasSuccessful(), 
                       f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")


if __name__ == '__main__':
    # Set up test discovery and run
    unittest.main(verbosity=2, exit=False)
    
    # Alternative: Run specific test suite
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE ALGORITHM ENGINE UNITTEST SUITE")
    print("="*70)
    
    # Create and run comprehensive test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [TestAlgoEngineBasics, TestAlgoEngineSignalProcessing, 
                       TestAlgoEngineErrorHandling, TestAlgoEnginePerformance]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nFinal Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
