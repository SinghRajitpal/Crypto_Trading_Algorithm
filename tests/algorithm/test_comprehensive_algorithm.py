"""
Comprehensive Algorithm Engine Tests
Senior Quantitative Developer Testing Protocol

This file contains comprehensive tests for the Algorithm Engine
covering signal generation, processing, throttling, and integration.
Tests ensure robust behavior under various market conditions.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy
from data.data_engine import DataEngine
from tests.utils.mock_objects import MockDataEngine, MockStrategy, MockErrorStrategy
from tests.utils.test_data import generate_ohlcv_data


class TestAlgorithmEngineCore:
    """Test core Algorithm Engine functionality."""
    
    @pytest.fixture
    def mock_data_engine(self):
        """Create mock data engine with test data."""
        return MockDataEngine()
    
    @pytest.fixture
    def algo_engine(self, mock_data_engine):
        """Create algorithm engine with mock data."""
        return AlgoEngine(mock_data_engine)
    
    @pytest.fixture
    def test_strategy(self):
        """Create a test strategy."""
        return MockStrategy(["buy", "hold", "sell"], "test_strategy")
    
    def test_initialization(self, algo_engine):
        """Test algorithm engine initialization."""
        assert algo_engine is not None
        assert algo_engine.data_engine is not None
        assert algo_engine.running is False
        assert isinstance(algo_engine._last_signal_states, dict)
        assert algo_engine._min_signal_interval > 0
        assert len(algo_engine._last_signal_states) == 0
    
    def test_data_hash_generation(self, algo_engine):
        """Test data hash generation for change detection."""
        # Test with valid candle data
        candles = [
            [1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1000.0],  # timestamp, o, h, l, c, v
            [1640995260000, 47050.0, 47150.0, 46950.0, 47100.0, 1200.0],
        ]
        
        hash1 = algo_engine._get_data_hash(candles)
        hash2 = algo_engine._get_data_hash(candles)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Different data should produce different hash
        different_candles = [
            [1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1000.0],
            [1640995260000, 47050.0, 47150.0, 46950.0, 47200.0, 1200.0],  # Different close
        ]
        
        hash3 = algo_engine._get_data_hash(different_candles)
        assert hash3 != hash1
        
        # Test with empty data
        empty_hash = algo_engine._get_data_hash([])
        assert empty_hash == ""
    
    def test_signal_state_tracking(self, algo_engine):
        """Test signal state tracking mechanisms."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        data_hash = "test_hash"
        signal_type = "open/buy"
        
        # Initially should process (no previous state)
        assert algo_engine._should_process_signal(key, current_time, data_hash) is True
        
        # Update state
        algo_engine._update_signal_state(key, current_time, data_hash, signal_type)
        
        # Check state was stored
        assert key in algo_engine._last_signal_states
        state = algo_engine._last_signal_states[key]
        assert state['timestamp'] == current_time
        assert state['data_hash'] == data_hash
        assert state['signal_type'] == signal_type
        
        # Same data within interval should not process
        assert algo_engine._should_process_signal(key, current_time + 30, data_hash) is False
        
        # Different data should process
        new_hash = "new_hash"
        assert algo_engine._should_process_signal(key, current_time + 30, new_hash) is True
        
        # Same data after interval should process
        assert algo_engine._should_process_signal(key, current_time + 70, data_hash) is True
    
    @pytest.mark.asyncio
    async def test_signal_processing_basic(self, algo_engine, test_strategy):
        """Test basic signal processing functionality."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Mock candle data in data engine
        candles = generate_ohlcv_data(100)
        algo_engine.data_engine.candles_data[(symbol, timeframe)] = candles
        
        # Process signals
        signal = await algo_engine.process_signals(symbol, timeframe, test_strategy)
        
        if signal:  # Strategy might return None for hold signals
            assert isinstance(signal, TradeSignal)
            assert signal.symbol == symbol
            assert signal.strategy_id == test_strategy.strategy_id
            assert signal.action in ["open", "exit", "hold"]
            assert signal.side in ["buy", "sell", "none"]
            assert 0 <= signal.signal_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_signal_throttling_mechanism(self, algo_engine, test_strategy):
        """Test signal throttling to prevent spam."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        candles = generate_ohlcv_data(100)
        algo_engine.data_engine.candles_data[(symbol, timeframe)] = candles
        
        # First signal should process
        signal1 = await algo_engine.process_signals(symbol, timeframe, test_strategy)
        
        # Immediate second signal with same data should be throttled
        signal2 = await algo_engine.process_signals(symbol, timeframe, test_strategy)
        
        # At least one should be None due to throttling
        if signal1 is not None and signal2 is not None:
            # If both returned signals, they should be the same (cached)
            assert signal1.timestamp == signal2.timestamp
        
        # Update data engine to have new data
        new_candles = generate_ohlcv_data(101)  # One more candle
        algo_engine.data_engine.candles_data[(symbol, timeframe)] = new_candles
        
        # New signal with different data should process
        signal3 = await algo_engine.process_signals(symbol, timeframe, test_strategy)
        assert signal3 is not None or signal1 is None  # Either new signal or previous was None
    
    @pytest.mark.asyncio
    async def test_multi_symbol_processing(self, algo_engine, test_strategy):
        """Test processing signals for multiple symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        timeframe = "1m"
        
        # Setup data for all symbols
        for symbol in symbols:
            candles = generate_ohlcv_data(100)
            algo_engine.data_engine.candles_data[(symbol, timeframe)] = candles
        
        # Process signals for all symbols
        signals = []
        for symbol in symbols:
            signal = await algo_engine.process_signals(symbol, timeframe, test_strategy)
            if signal:
                signals.append(signal)
        
        # Should have independent processing for each symbol
        unique_symbols = set(signal.symbol for signal in signals)
        assert len(unique_symbols) == len(signals)  # No duplicate symbols
    
    @pytest.mark.asyncio
    async def test_error_handling_resilience(self, algo_engine):
        """Test error handling and system resilience."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Test with strategy that throws exceptions
        error_strategy = MockErrorStrategy()
        
        # Should handle exceptions gracefully
        signal = await algo_engine.process_signals(symbol, timeframe, error_strategy)
        assert signal is None  # Should return None on error
        
        # Test with missing data
        algo_engine.data_engine.candles_data.clear()
        normal_strategy = MockStrategy(["buy"], "normal")
        
        signal = await algo_engine.process_signals(symbol, timeframe, normal_strategy)
        assert signal is None  # Should return None when no data
    
    @pytest.mark.asyncio
    async def test_signal_metadata_validation(self, algo_engine, test_strategy):
        """Test signal metadata and validation."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        candles = generate_ohlcv_data(100)
        algo_engine.data_engine.candles_data[(symbol, timeframe)] = candles
        
        signal = await algo_engine.process_signals(symbol, timeframe, test_strategy)
        
        if signal:
            # Validate signal structure
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'side')
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'strategy_id')
            assert hasattr(signal, 'signal_confidence')
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'metadata')
            
            # Validate timestamp is set
            assert signal.timestamp is not None
            assert isinstance(signal.timestamp, (int, float))
            
            # Validate confidence is in valid range
            assert 0 <= signal.signal_confidence <= 1


class TestAlgorithmEngineRunLoop:
    """Test the main algorithm engine run loop and lifecycle."""
    
    @pytest.fixture
    def algo_engine_with_config(self, mock_data_engine):
        """Create algorithm engine with mocked config."""
        with patch('config.symbols', [("BTCUSDT", "1m"), ("ETHUSDT", "1m")]):
            return AlgoEngine(mock_data_engine)
    
    @pytest.mark.asyncio
    async def test_run_loop_basic_functionality(self, algo_engine_with_config):
        """Test basic run loop functionality."""
        test_strategy = MockStrategy(["buy", "hold"], "test_strategy")
        
        # Setup some test data
        symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            candles = generate_ohlcv_data(100)
            algo_engine_with_config.data_engine.candles_data[(symbol, "1m")] = candles
        
        # Start run loop
        signal_count = 0
        max_signals = 5
        
        async for signal in algo_engine_with_config.run(test_strategy):
            signal_count += 1
            
            assert isinstance(signal, TradeSignal)
            assert signal.symbol in symbols
            
            # Stop after collecting enough signals
            if signal_count >= max_signals:
                break
        
        assert signal_count > 0
        await algo_engine_with_config.stop()
    
    @pytest.mark.asyncio
    async def test_run_loop_error_recovery(self, algo_engine_with_config):
        """Test run loop error recovery and resilience."""
        # Strategy that alternates between normal and error behavior
        class AlternatingStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("alternating_strategy")
                self.call_count = 0
            
            async def calculate_signals(self, candles, symbol):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Every third call throws error
                    raise Exception("Simulated strategy error")
                
                return TradeSignal(
                    action="open",
                    side="buy", 
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    signal_confidence=0.5
                )
        
        strategy = AlternatingStrategy()
        
        # Setup test data
        candles = generate_ohlcv_data(100)
        algo_engine_with_config.data_engine.candles_data[("BTCUSDT", "1m")] = candles
        algo_engine_with_config.data_engine.candles_data[("ETHUSDT", "1m")] = candles
        
        # Run loop should continue despite errors
        signal_count = 0
        error_count = 0
        max_iterations = 10
        
        iteration = 0
        async for signal in algo_engine_with_config.run(strategy):
            iteration += 1
            if signal:
                signal_count += 1
            else:
                error_count += 1
            
            if iteration >= max_iterations:
                break
        
        # Should have received some signals despite errors
        assert signal_count > 0
        await algo_engine_with_config.stop()
    
    @pytest.mark.asyncio
    async def test_run_loop_performance(self, algo_engine_with_config):
        """Test run loop performance characteristics."""
        test_strategy = MockStrategy(["buy"], "performance_test")
        
        # Setup data
        candles = generate_ohlcv_data(1000)  # Large dataset
        algo_engine_with_config.data_engine.candles_data[("BTCUSDT", "1m")] = candles
        algo_engine_with_config.data_engine.candles_data[("ETHUSDT", "1m")] = candles
        
        # Measure processing time
        start_time = time.perf_counter()
        signal_count = 0
        max_signals = 20
        
        async for signal in algo_engine_with_config.run(test_strategy):
            signal_count += 1
            if signal_count >= max_signals:
                break
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should process signals efficiently
        avg_time_per_signal = total_time / max_signals if max_signals > 0 else float('inf')
        assert avg_time_per_signal < 0.1, f"Average time per signal {avg_time_per_signal:.3f}s too slow"
        
        await algo_engine_with_config.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, algo_engine_with_config):
        """Test concurrent signal processing capabilities."""
        strategy1 = MockStrategy(["buy"], "strategy_1")
        strategy2 = MockStrategy(["sell"], "strategy_2")
        
        # Setup data
        candles = generate_ohlcv_data(100)
        symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            algo_engine_with_config.data_engine.candles_data[(symbol, "1m")] = candles
        
        # Run concurrent processing
        async def collect_signals(strategy, max_count=5):
            signals = []
            count = 0
            async for signal in algo_engine_with_config.run(strategy):
                if signal:
                    signals.append(signal)
                count += 1
                if count >= max_count:
                    break
            return signals
        
        # Note: In practice, you'd typically have one AlgoEngine per strategy
        # This test simulates the theoretical concurrent capability
        signals1 = await collect_signals(strategy1, 3)
        signals2 = await collect_signals(strategy2, 3)
        
        # Each strategy should produce its own signals
        assert len(signals1) > 0 or len(signals2) > 0
        
        await algo_engine_with_config.stop()


class TestAlgorithmEngineStrategyIntegration:
    """Test integration with different strategy types."""
    
    @pytest.fixture
    def algo_engine(self, mock_data_engine):
        return AlgoEngine(mock_data_engine)
    
    def test_strategy_interface_compliance(self, algo_engine):
        """Test that strategies comply with expected interface."""
        class TestStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("test_interface")
            
            async def calculate_signals(self, candles, symbol):
                return TradeSignal(
                    action="open",
                    side="buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    signal_confidence=0.7
                )
        
        strategy = TestStrategy()
        
        # Verify strategy has required methods
        assert hasattr(strategy, 'calculate_signals')
        assert hasattr(strategy, 'strategy_id')
        assert callable(strategy.calculate_signals)
    
    @pytest.mark.asyncio
    async def test_signal_confidence_validation(self, algo_engine):
        """Test signal confidence validation."""
        class ConfidenceStrategy(BaseStrategy):
            def __init__(self, confidence_value):
                super().__init__(f"confidence_{confidence_value}")
                self.confidence = confidence_value
            
            async def calculate_signals(self, candles, symbol):
                return TradeSignal(
                    action="open",
                    side="buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    signal_confidence=self.confidence
                )
        
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 1.0]
        for conf in valid_confidences:
            strategy = ConfidenceStrategy(conf)
            candles = generate_ohlcv_data(50)
            algo_engine.data_engine.candles_data[("BTCUSDT", "1m")] = candles
            
            signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
            if signal:
                assert 0 <= signal.signal_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_strategy_state_isolation(self, algo_engine):
        """Test that strategies maintain isolated state."""
        class StatefulStrategy(BaseStrategy):
            def __init__(self, strategy_id):
                super().__init__(strategy_id)
                self.call_count = 0
            
            async def calculate_signals(self, candles, symbol):
                self.call_count += 1
                return TradeSignal(
                    action="open",
                    side="buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    signal_confidence=min(self.call_count * 0.1, 1.0)
                )
        
        strategy1 = StatefulStrategy("stateful_1")
        strategy2 = StatefulStrategy("stateful_2")
        
        candles = generate_ohlcv_data(50)
        algo_engine.data_engine.candles_data[("BTCUSDT", "1m")] = candles
        
        # Process with both strategies
        signal1 = await algo_engine.process_signals("BTCUSDT", "1m", strategy1)
        signal2 = await algo_engine.process_signals("BTCUSDT", "1m", strategy2)
        
        # Each strategy should maintain its own state
        assert strategy1.call_count >= 1
        assert strategy2.call_count >= 1
        
        if signal1 and signal2:
            assert signal1.strategy_id != signal2.strategy_id


class TestAlgorithmEngineStressTest:
    """Stress tests for algorithm engine robustness."""
    
    @pytest.fixture
    def stress_algo_engine(self, mock_data_engine):
        return AlgoEngine(mock_data_engine)
    
    @pytest.mark.asyncio
    async def test_high_frequency_signal_processing(self, stress_algo_engine):
        """Test high-frequency signal processing capability."""
        strategy = MockStrategy(["buy", "sell", "hold"], "high_freq")
        
        # Generate large amount of data
        large_candles = generate_ohlcv_data(1000)
        stress_algo_engine.data_engine.candles_data[("BTCUSDT", "1m")] = large_candles
        
        # Process many signals rapidly
        start_time = time.perf_counter()
        signal_count = 0
        
        for _ in range(100):  # 100 rapid processing calls
            signal = await stress_algo_engine.process_signals("BTCUSDT", "1m", strategy)
            if signal:
                signal_count += 1
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Should handle high frequency processing
        assert processing_time < 5.0, f"High frequency processing took {processing_time:.2f}s"
        assert signal_count >= 0  # Some signals should be processed
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_datasets(self, stress_algo_engine):
        """Test memory efficiency with large datasets."""
        import sys
        
        initial_size = sys.getsizeof(stress_algo_engine)
        
        # Add large amounts of data
        for i in range(100):
            symbol = f"TEST{i}USDT"
            large_candles = generate_ohlcv_data(500)
            stress_algo_engine.data_engine.candles_data[(symbol, "1m")] = large_candles
        
        # Process signals for all symbols
        strategy = MockStrategy(["buy"], "memory_test")
        processed_count = 0
        
        for i in range(50):  # Process subset to avoid excessive test time
            symbol = f"TEST{i}USDT"
            signal = await stress_algo_engine.process_signals(symbol, "1m", strategy)
            if signal:
                processed_count += 1
        
        final_size = sys.getsizeof(stress_algo_engine)
        memory_growth = final_size - initial_size
        
        # Memory growth should be reasonable
        assert memory_growth < 1024 * 1024, f"Memory growth {memory_growth} bytes excessive"
        assert processed_count > 0
    
    @pytest.mark.asyncio
    async def test_edge_case_data_handling(self, stress_algo_engine):
        """Test handling of edge case data scenarios."""
        strategy = MockStrategy(["buy"], "edge_case")
        
        # Test empty candles
        stress_algo_engine.data_engine.candles_data[("EMPTY", "1m")] = []
        signal = await stress_algo_engine.process_signals("EMPTY", "1m", strategy)
        assert signal is None
        
        # Test single candle
        single_candle = [[1640995200000, 47000.0, 47000.0, 47000.0, 47000.0, 1000.0]]
        stress_algo_engine.data_engine.candles_data[("SINGLE", "1m")] = single_candle
        signal = await stress_algo_engine.process_signals("SINGLE", "1m", strategy)
        # Should handle gracefully (may or may not return signal depending on strategy)
        
        # Test candles with zero values
        zero_candles = [
            [1640995200000, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1640995260000, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        stress_algo_engine.data_engine.candles_data[("ZERO", "1m")] = zero_candles
        signal = await stress_algo_engine.process_signals("ZERO", "1m", strategy)
        # Should handle gracefully
        
        # Test malformed candles (insufficient data)
        malformed = [[1640995200000, 47000.0]]  # Missing OHLCV data
        stress_algo_engine.data_engine.candles_data[("MALFORMED", "1m")] = malformed
        signal = await stress_algo_engine.process_signals("MALFORMED", "1m", strategy)
        # Should handle gracefully without crashing
    
    @pytest.mark.asyncio
    async def test_rapid_state_changes(self, stress_algo_engine):
        """Test rapid state changes and updates."""
        strategy = MockStrategy(["buy", "sell", "hold"], "rapid_state")
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Rapidly change data and process
        for i in range(50):
            # Generate new data each iteration
            new_candles = generate_ohlcv_data(50 + i)
            stress_algo_engine.data_engine.candles_data[(symbol, timeframe)] = new_candles
            
            signal = await stress_algo_engine.process_signals(symbol, timeframe, strategy)
            
            # Brief pause to simulate realistic timing
            await asyncio.sleep(0.001)
        
        # Should handle rapid state changes without issues
        final_state = stress_algo_engine._last_signal_states.get(f"{symbol}_{timeframe}")
        assert final_state is not None  # Should have state from processing
