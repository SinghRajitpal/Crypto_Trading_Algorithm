"""
Unittest Compatibility Suite
Senior Quantitative Developer Testing Protocol

Comprehensive unittest-based testing for compatibility with traditional
testing frameworks and enterprise environments.
"""

import unittest
import asyncio
import sys
import os
import time
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from execution.execution_engine import ProductionExecutionEngine
from tests.utils.mock_objects import MockDataEngine, MockBinanceClient, MockStrategy, MockErrorStrategy
from tests.utils.test_data import generate_market_data_bar, create_test_signal_metadata


class TestAlgorithmEngineUnittest(unittest.TestCase):
    """Unittest-based tests for algorithm engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def tearDown(self):
        """Clean up after tests."""
        self.algo_engine.running = False
    
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
        
        # Should now exist with correct values
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


class TestSignalProcessingUnittest(unittest.TestCase):
    """Unittest-based tests for signal processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_signal_processing_with_buy_strategy(self):
        """Test signal processing with buy strategy."""
        async def run_test():
            strategy = MockStrategy(["buy"], "buy_test")
            
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "open")
            self.assertEqual(signal.side, "buy")
            self.assertEqual(signal.symbol, "BTCUSDT")
            self.assertEqual(signal.strategy_id, "buy_test")
            self.assertGreater(signal.signal_confidence, 0)
        
        asyncio.run(run_test())
    
    def test_signal_processing_with_hold_strategy(self):
        """Test signal processing with hold strategy."""
        async def run_test():
            strategy = MockStrategy(["hold"], "hold_test")
            
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "hold")
            self.assertEqual(signal.side, "none")
        
        asyncio.run(run_test())
    
    def test_multi_symbol_processing(self):
        """Test processing signals for multiple symbols."""
        async def run_test():
            strategy = MockStrategy(["buy"], "multi_symbol_test")
            symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
            
            signals = []
            for symbol in symbols:
                self.algo_engine._last_signal_states.clear()
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


class TestErrorHandlingUnittest(unittest.TestCase):
    """Unittest-based tests for error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_strategy_error_handling(self):
        """Test handling of strategy errors."""
        async def run_test():
            error_strategy = MockErrorStrategy()
            
            # Should handle error gracefully and return None
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", error_strategy)
            self.assertIsNone(signal)
        
        asyncio.run(run_test())
    
    def test_empty_candle_data_handling(self):
        """Test handling of empty candle data."""
        async def run_test():
            # Mock empty data
            self.mock_data_engine.get_candles = lambda symbol, timeframe: []
            
            strategy = MockStrategy(["buy"], "empty_data_test")
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            # Should return None for empty data
            self.assertIsNone(signal)
        
        asyncio.run(run_test())
    
    def test_data_engine_exception_handling(self):
        """Test handling of data engine exceptions."""
        async def run_test():
            # Mock exception
            def raise_exception(*args, **kwargs):
                raise Exception("Data engine error")
            
            self.mock_data_engine.get_candles = raise_exception
            
            strategy = MockStrategy(["buy"], "exception_test")
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            # Should return None when data engine fails
            self.assertIsNone(signal)
        
        asyncio.run(run_test())


class TestExecutionEngineUnittest(unittest.TestCase):
    """Unittest-based tests for execution engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_binance_client = MockBinanceClient()
    
    def test_execution_engine_initialization(self):
        """Test execution engine initialization."""
        async def run_test():
            engine = ProductionExecutionEngine(self.mock_binance_client, total_capital=10000.0)
            await engine.setup()
            
            self.assertTrue(self.mock_binance_client.account_config_called)
            self.assertIsNotNone(engine.portfolio_manager)
            self.assertIsNotNone(engine.risk_manager)
        
        asyncio.run(run_test())
    
    def test_portfolio_manager_initialization(self):
        """Test portfolio manager initialization."""
        async def run_test():
            engine = ProductionExecutionEngine(self.mock_binance_client, total_capital=10000.0)
            await engine.setup()
            
            portfolio_manager = engine.portfolio_manager
            
            self.assertIsNotNone(portfolio_manager)
            self.assertEqual(portfolio_manager.total_capital, 10000.0)
            self.assertEqual(portfolio_manager.target_volatility, 0.18)
            self.assertEqual(portfolio_manager.max_allocation_pct, 0.85)
        
        asyncio.run(run_test())
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        async def run_test():
            engine = ProductionExecutionEngine(self.mock_binance_client, total_capital=10000.0)
            await engine.setup()
            
            risk_manager = engine.risk_manager
            
            self.assertIsNotNone(risk_manager)
            self.assertEqual(risk_manager.risk_per_trade, 0.008)  # 0.8%
            self.assertEqual(risk_manager.kelly_fraction, 0.7)
            self.assertEqual(risk_manager.base_leverage, 3.0)
        
        asyncio.run(run_test())


class TestIntegrationUnittest(unittest.TestCase):
    """Unittest-based integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.mock_binance_client = MockBinanceClient()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_signal_validation_integration(self):
        """Test signal validation integration."""
        async def run_test():
            engine = ProductionExecutionEngine(self.mock_binance_client, total_capital=10000.0)
            await engine.setup()
            
            # Set up market data
            engine.update_market_data_bar("BTCUSDT", generate_market_data_bar("BTCUSDT"), 0.02)
            
            # Create test signal
            signal = TradeSignal(
                action="open",
                side="buy",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata=create_test_signal_metadata(),
                signal_confidence=0.8
            )
            
            # Test validation
            validation_result = await engine.validate_signal(signal, 50000.0)
            
            self.assertIsInstance(validation_result, dict)
            self.assertIn('valid', validation_result)
        
        asyncio.run(run_test())


class TestPerformanceUnittest(unittest.TestCase):
    """Unittest-based performance tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
    
    def test_rapid_signal_processing(self):
        """Test rapid signal processing performance."""
        async def run_test():
            strategy = MockStrategy(["buy"], "performance_test")
            start_time = time.time()
            
            # Process multiple signals
            signals_processed = 0
            for i in range(10):
                self.algo_engine._last_signal_states.clear()
                signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
                if signal:
                    signals_processed += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process rapidly
            self.assertLess(processing_time, 1.0)  # Under 1 second
            self.assertGreater(signals_processed, 0)
        
        asyncio.run(run_test())


class UnittestSuiteRunner(unittest.TestCase):
    """Main test suite runner for unittest compatibility."""
    
    def test_run_all_unittest_suite(self):
        """Run all unittest-based tests as a comprehensive suite."""
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestAlgorithmEngineUnittest,
            TestSignalProcessingUnittest,
            TestErrorHandlingUnittest,
            TestExecutionEngineUnittest,
            TestIntegrationUnittest,
            TestPerformanceUnittest
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        # Assert overall success
        self.assertTrue(result.wasSuccessful(), 
                       f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        
        # Print summary
        print(f"\nUnittest Suite Results:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")


def run_unittest_suite():
    """Run the complete unittest suite with detailed reporting."""
    print("=" * 80)
    print("COMPREHENSIVE UNITTEST SUITE FOR CRYPTO TRADING ALGORITHM")
    print("=" * 80)
    
    # Create and run comprehensive test suite
    suite = unittest.TestSuite()
    
    # Add all test classes except the runner
    test_classes = [
        TestAlgorithmEngineUnittest,
        TestSignalProcessingUnittest,
        TestErrorHandlingUnittest,
        TestExecutionEngineUnittest,
        TestIntegrationUnittest,
        TestPerformanceUnittest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("UNITTEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFailure Details:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nError Details:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run the comprehensive unittest suite
    success = run_unittest_suite()
    exit(0 if success else 1)
