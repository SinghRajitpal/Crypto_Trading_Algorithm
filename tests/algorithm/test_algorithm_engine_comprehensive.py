#!/usr/bin/env python3
"""
Algorithm Engine Comprehensive Test Suite
Senior Quantitative Developer Testing Protocol

Tests for signal generation, processing logic, state management, and edge cases.
Focuses on behavior-first debugging and interface validation.
"""

import os
import sys
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_test_framework import ComprehensiveTestFramework
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.base_strategy import BaseStrategy
from algorithm.trade_signal import TradeSignal
from data.data_engine import DataEngine

class MockDataEngine:
    """Mock data engine for isolated algorithm testing."""
    
    def __init__(self):
        self.candle_data = {}
        self.binance_client = None
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Get mock candle data."""
        key = f"{symbol}_{timeframe}"
        if key not in self.candle_data:
            # Generate default mock data
            framework = ComprehensiveTestFramework()
            self.candle_data[key] = framework.generate_mock_ohlcv_data(symbol, 100)
        return self.candle_data[key]
    
    def set_candles(self, symbol: str, timeframe: str, candles: List[List]):
        """Set specific candle data for testing."""
        key = f"{symbol}_{timeframe}"
        self.candle_data[key] = candles

class MockStrategy(BaseStrategy):
    """Mock strategy for testing algorithm engine behavior."""
    
    def __init__(self, signal_type: str = "hold", confidence: float = 0.8, **kwargs):
        # Provide required parameters for BaseStrategy
        params = kwargs.get('params', {})
        strategy_id = kwargs.get('strategy_id', 'mock_strategy')
        super().__init__(params, strategy_id)
        
        self.signal_type = signal_type  # "buy", "sell", "hold"
        self.confidence = confidence
        self.call_count = 0
        self.last_candles = None
    
    def get_required_indicators(self) -> List[str]:
        """Return empty list for mock strategy."""
        return []
    
    async def _generate_signals(self, data: Dict[str, Any], indicator_data: Dict[str, Any], symbol: str) -> Optional[TradeSignal]:
        """Generate mock signals based on configured type."""
        self.call_count += 1
        
        if self.signal_type == "hold":
            return TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"mock": True},
                signal_confidence=self.confidence,
                timestamp=int(time.time() * 1000)
            )
        elif self.signal_type == "buy":
            return TradeSignal(
                action="open",
                side="buy",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"mock": True},
                signal_confidence=self.confidence,
                timestamp=int(time.time() * 1000)
            )
        elif self.signal_type == "sell":
            return TradeSignal(
                action="open",
                side="sell",
                symbol=symbol,
                strategy_id=self.strategy_id,
                metadata={"mock": True},
                signal_confidence=self.confidence,
                timestamp=int(time.time() * 1000)
            )
        
        return None
        
    async def calculate_signals(self, candles: List[List], symbol: str) -> Optional[TradeSignal]:
        """Generate mock signals for testing."""
        self.last_candles = candles
        
        # Mock the indicator calculation and signal generation
        data = {}
        indicator_data = {}
        
        return await self._generate_signals(data, indicator_data, symbol)

class AlgorithmEngineTestSuite:
    """Comprehensive test suite for Algorithm Engine."""
    
    def __init__(self, framework: ComprehensiveTestFramework):
        self.framework = framework
        self.mock_data_engine = MockDataEngine()
        
    def test_algo_engine_initialization(self) -> Dict[str, Any]:
        """Test algorithm engine initialization and basic properties."""
        errors = []
        details = {}
        
        try:
            # Test basic initialization
            algo_engine = AlgoEngine(self.mock_data_engine)
            
            # Verify initial state
            if not hasattr(algo_engine, 'data_engine'):
                errors.append("AlgoEngine missing data_engine attribute")
            
            if not hasattr(algo_engine, 'running'):
                errors.append("AlgoEngine missing running attribute")
            
            if algo_engine.running != False:
                errors.append("AlgoEngine should initialize with running=False")
            
            if not hasattr(algo_engine, '_last_signal_states'):
                errors.append("AlgoEngine missing _last_signal_states attribute")
            
            if not isinstance(algo_engine._last_signal_states, dict):
                errors.append("_last_signal_states should be a dictionary")
            
            # Test min signal interval
            if not hasattr(algo_engine, '_min_signal_interval'):
                errors.append("AlgoEngine missing _min_signal_interval attribute")
            
            if algo_engine._min_signal_interval != 60:
                errors.append(f"Expected _min_signal_interval=60, got {algo_engine._min_signal_interval}")
            
            details['initial_state'] = {
                'running': algo_engine.running,
                'signal_states': len(algo_engine._last_signal_states),
                'min_interval': algo_engine._min_signal_interval
            }
            
        except Exception as e:
            errors.append(f"Initialization failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_data_hash_generation(self) -> Dict[str, Any]:
        """Test data hash generation for change detection."""
        errors = []
        details = {}
        
        try:
            algo_engine = AlgoEngine(self.mock_data_engine)
            
            # Test with empty candles
            hash_empty = algo_engine._get_data_hash([])
            if hash_empty != "":
                errors.append(f"Empty candles should return empty hash, got '{hash_empty}'")
            
            # Test with single candle
            candle1 = [1640995200000, 50000, 50100, 49900, 50050, 100]  # [timestamp, o, h, l, c, v]
            hash1 = algo_engine._get_data_hash([candle1])
            expected_hash1 = f"{candle1[0]}_{candle1[4]}"  # timestamp_close
            
            if hash1 != expected_hash1:
                errors.append(f"Expected hash '{expected_hash1}', got '{hash1}'")
            
            # Test with multiple candles (should use last candle)
            candle2 = [1640995260000, 50050, 50150, 49950, 50100, 120]
            hash2 = algo_engine._get_data_hash([candle1, candle2])
            expected_hash2 = f"{candle2[0]}_{candle2[4]}"
            
            if hash2 != expected_hash2:
                errors.append(f"Expected hash '{expected_hash2}' for multiple candles, got '{hash2}'")
            
            # Test hash changes with data changes
            candle3 = [1640995320000, 50100, 50200, 50000, 50150, 150]
            hash3 = algo_engine._get_data_hash([candle1, candle2, candle3])
            
            if hash3 == hash2:
                errors.append("Hash should change when new candle is added")
            
            details['hash_tests'] = {
                'empty_hash': hash_empty,
                'single_candle_hash': hash1,
                'multi_candle_hash': hash2,
                'changed_hash': hash3
            }
            
        except Exception as e:
            errors.append(f"Data hash test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    def test_signal_throttling_logic(self) -> Dict[str, Any]:
        """Test signal throttling and deduplication logic."""
        errors = []
        details = {}
        
        try:
            algo_engine = AlgoEngine(self.mock_data_engine)
            
            current_time = int(time.time())
            test_key = "BTCUSDT_1m"
            test_hash = "test_hash_123"
            
            # Test first signal (should process)
            should_process_1 = algo_engine._should_process_signal(test_key, current_time, test_hash)
            if not should_process_1:
                errors.append("First signal should always be processed")
            
            # Update signal state
            algo_engine._update_signal_state(test_key, current_time, test_hash, "open/buy")
            
            # Test immediate duplicate (should not process)
            should_process_2 = algo_engine._should_process_signal(test_key, current_time, test_hash)
            if should_process_2:
                errors.append("Immediate duplicate signal should not be processed")
            
            # Test with changed data (should process)
            new_hash = "test_hash_456"
            should_process_3 = algo_engine._should_process_signal(test_key, current_time, new_hash)
            if not should_process_3:
                errors.append("Signal with changed data should be processed")
            
            # Test with time elapsed but same data (should process after interval)
            future_time = current_time + 61  # Beyond min_signal_interval
            should_process_4 = algo_engine._should_process_signal(test_key, future_time, test_hash)
            if not should_process_4:
                errors.append("Signal should be processed after time interval")
            
            # Test within interval with same data (should not process)
            near_future = current_time + 30  # Within min_signal_interval
            should_process_5 = algo_engine._should_process_signal(test_key, near_future, test_hash)
            if should_process_5:
                errors.append("Signal within time interval with same data should not be processed")
            
            details['throttling_tests'] = {
                'first_signal': should_process_1,
                'immediate_duplicate': should_process_2,
                'changed_data': should_process_3,
                'time_elapsed': should_process_4,
                'within_interval': should_process_5
            }
            
        except Exception as e:
            errors.append(f"Signal throttling test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_signal_processing_pipeline(self) -> Dict[str, Any]:
        """Test the complete signal processing pipeline."""
        errors = []
        details = {}
        warnings = []
        
        try:
            algo_engine = AlgoEngine(self.mock_data_engine)
            
            # Test with buy signal strategy
            buy_strategy = MockStrategy(signal_type="buy", confidence=0.8)
            
            # Set up mock data
            test_candles = self.framework.generate_mock_ohlcv_data("BTCUSDT", 50)
            self.mock_data_engine.set_candles("BTCUSDT", "1m", test_candles)
            
            # Process signal
            signal = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
            
            # Verify signal generation
            if signal is None:
                errors.append("Expected buy signal but got None")
            else:
                if signal.symbol != "BTCUSDT":
                    errors.append(f"Expected symbol 'BTCUSDT', got '{signal.symbol}'")
                
                if signal.action != "open":
                    errors.append(f"Expected action 'open', got '{signal.action}'")
                
                if signal.side != "buy":
                    errors.append(f"Expected side 'buy', got '{signal.side}'")
                
                if signal.signal_confidence != 0.8:
                    errors.append(f"Expected confidence 0.8, got {signal.signal_confidence}")
                
                if not signal.timestamp:
                    errors.append("Signal should have timestamp set")
            
            # Test signal state tracking
            key = "BTCUSDT_1m"
            if key not in algo_engine._last_signal_states:
                errors.append("Signal state should be tracked after processing")
            else:
                state = algo_engine._last_signal_states[key]
                if state['signal_type'] != "open/buy":
                    errors.append(f"Expected signal_type 'open/buy', got '{state['signal_type']}'")
            
            # Test duplicate signal filtering
            signal_2 = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
            if signal_2 is not None:
                warnings.append("Duplicate signal was not filtered out")
            
            # Test with sell signal strategy
            sell_strategy = MockStrategy(signal_type="sell", confidence=0.9)
            
            # Use different data to trigger new signal
            new_candles = self.framework.generate_mock_ohlcv_data("BTCUSDT", 51)
            self.mock_data_engine.set_candles("BTCUSDT", "1m", new_candles)
            
            sell_signal = await algo_engine.process_signals("BTCUSDT", "1m", sell_strategy)
            
            if sell_signal is None:
                errors.append("Expected sell signal but got None")
            elif sell_signal.side != "sell":
                errors.append(f"Expected sell side, got '{sell_signal.side}'")
            
            # Test with hold strategy (hold signal)
            hold_strategy = MockStrategy(signal_type="hold")
            
            # Use different data to ensure signal is processed
            hold_candles = self.framework.generate_mock_ohlcv_data("BTCUSDT", 52)  
            self.mock_data_engine.set_candles("BTCUSDT", "1m", hold_candles)
            
            hold_signal = await algo_engine.process_signals("BTCUSDT", "1m", hold_strategy)
            
            # Hold strategy returns a signal with action="hold", not None
            if hold_signal is None:
                errors.append("Hold strategy should return a hold signal")
            elif hold_signal.action != "hold":
                errors.append(f"Hold strategy should return hold action, got '{hold_signal.action}'")
            
            details['pipeline_tests'] = {
                'buy_signal_generated': signal is not None,
                'sell_signal_generated': sell_signal is not None,
                'hold_signal_generated': hold_signal is not None and hold_signal.action == "hold",
                'strategy_call_count': buy_strategy.call_count,
                'signal_states_tracked': len(algo_engine._last_signal_states)
            }
            
        except Exception as e:
            errors.append(f"Signal processing pipeline test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': details
        }
    
    async def test_multiple_symbol_timeframe_handling(self) -> Dict[str, Any]:
        """Test handling of multiple symbols and timeframes."""
        errors = []
        details = {}
        
        try:
            algo_engine = AlgoEngine(self.mock_data_engine)
            strategy = MockStrategy(signal_type="buy", confidence=0.7)
            
            # Test different symbols
            symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
            timeframes = ["1m", "5m"]
            
            signals_generated = {}
            
            for symbol in symbols:
                for timeframe in timeframes:
                    # Set unique mock data for each symbol-timeframe
                    candles = self.framework.generate_mock_ohlcv_data(symbol, 30)
                    self.mock_data_engine.set_candles(symbol, timeframe, candles)
                    
                    # Process signal
                    signal = await algo_engine.process_signals(symbol, timeframe, strategy)
                    key = f"{symbol}_{timeframe}"
                    signals_generated[key] = signal is not None
            
            # Verify each symbol-timeframe combination is tracked separately
            expected_keys = [f"{s}_{t}" for s in symbols for t in timeframes]
            
            for key in expected_keys:
                if key not in algo_engine._last_signal_states:
                    errors.append(f"Signal state not tracked for {key}")
            
            # Test signal independence (changing one shouldn't affect others)
            # Change data for BTCUSDT_1m only
            new_candles = self.framework.generate_mock_ohlcv_data("BTCUSDT", 31)
            self.mock_data_engine.set_candles("BTCUSDT", "1m", new_candles)
            
            # Process signal for BTCUSDT_1m (should generate new signal)
            new_signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
            if new_signal is None:
                errors.append("New signal should be generated for changed data")
            
            # Process signal for ETHUSDT_1m (should not generate due to no data change)
            no_signal = await algo_engine.process_signals("ETHUSDT", "1m", strategy)
            if no_signal is not None:
                errors.append("No signal should be generated for unchanged data")
            
            details['multi_symbol_tests'] = {
                'symbols_tested': len(symbols),
                'timeframes_tested': len(timeframes),
                'total_combinations': len(expected_keys),
                'signals_generated': sum(signals_generated.values()),
                'state_keys_tracked': len(algo_engine._last_signal_states)
            }
            
        except Exception as e:
            errors.append(f"Multiple symbol/timeframe test failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'details': details
        }
    
    async def test_edge_cases_and_error_handling(self) -> Dict[str, Any]:
        """Test edge cases and error handling robustness."""
        errors = []
        details = {}
        warnings = []
        
        try:
            algo_engine = AlgoEngine(self.mock_data_engine)
            
            # Test with empty candles
            self.mock_data_engine.set_candles("EMPTY", "1m", [])
            empty_signal = await algo_engine.process_signals("EMPTY", "1m", MockStrategy())
            
            if empty_signal is not None:
                warnings.append("Empty candles should return None signal")
            
            # Test with malformed strategy
            class BrokenStrategy(BaseStrategy):
                def __init__(self):
                    super().__init__({}, "broken")
                
                def get_required_indicators(self):
                    return []
                
                async def _generate_signals(self, data, indicator_data, symbol):
                    raise ValueError("Intentional strategy error")
                
                async def calculate_signals(self, candles, symbol):
                    raise ValueError("Intentional strategy error")
            
            broken_strategy = BrokenStrategy()
            candles = self.framework.generate_mock_ohlcv_data("BTCUSDT", 20)
            self.mock_data_engine.set_candles("BTCUSDT", "1m", candles)
            
            error_signal = await algo_engine.process_signals("BTCUSDT", "1m", broken_strategy)
            
            if error_signal is not None:
                warnings.append("Broken strategy should return None signal")
            
            # Test with None candles return from data engine
            class NoneDataEngine:
                def get_candles(self, symbol, timeframe):
                    return None
            
            none_engine = AlgoEngine(NoneDataEngine())
            none_signal = await none_engine.process_signals("BTCUSDT", "1m", MockStrategy())
            
            if none_signal is not None:
                warnings.append("None candles should return None signal")
            
            # Test data engine error handling
            class ErrorDataEngine:
                def get_candles(self, symbol, timeframe):
                    raise ConnectionError("Mock connection error")
            
            error_engine = AlgoEngine(ErrorDataEngine())
            conn_error_signal = await error_engine.process_signals("BTCUSDT", "1m", MockStrategy())
            
            if conn_error_signal is not None:
                warnings.append("Data engine error should return None signal")
            
            details['edge_case_tests'] = {
                'empty_candles_handled': empty_signal is None,
                'strategy_error_handled': error_signal is None,
                'none_candles_handled': none_signal is None,
                'data_error_handled': conn_error_signal is None
            }
            
        except Exception as e:
            errors.append(f"Edge case testing failed: {str(e)}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': details
        }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all algorithm engine tests."""
        test_results = []
        
        # Basic functionality tests
        test_results.append(await self.framework.run_test(
            self.test_algo_engine_initialization, 
            "algorithm_engine_initialization"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_data_hash_generation, 
            "algorithm_data_hash_generation"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_signal_throttling_logic, 
            "algorithm_signal_throttling"
        ))
        
        # Signal processing tests
        test_results.append(await self.framework.run_test(
            self.test_signal_processing_pipeline, 
            "algorithm_signal_processing_pipeline"
        ))
        
        test_results.append(await self.framework.run_test(
            self.test_multiple_symbol_timeframe_handling, 
            "algorithm_multi_symbol_timeframe"
        ))
        
        # Edge cases and robustness
        test_results.append(await self.framework.run_test(
            self.test_edge_cases_and_error_handling, 
            "algorithm_edge_cases_error_handling"
        ))
        
        return test_results

async def main():
    """Run algorithm engine test suite."""
    print("🔬 ALGORITHM ENGINE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    framework = ComprehensiveTestFramework(verbose=True)
    test_suite = AlgorithmEngineTestSuite(framework)
    
    # Run all tests
    await test_suite.run_all_tests()
    
    # Print detailed report
    framework.print_test_report()

if __name__ == "__main__":
    asyncio.run(main())
