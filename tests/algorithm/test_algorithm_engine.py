"""
Algorithm Engine Tests
Senior Quantitative Developer Testing Protocol

Comprehensive pytest-based testing for the algorithm engine including:
- Initialization and configuration
- Signal generation and processing
- Throttling and deduplication
- Multi-symbol and multi-timeframe handling
- Error handling and resilience
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from tests.utils.mock_objects import MockStrategy, MockErrorStrategy
from tests.utils.test_data import generate_ohlcv_data


class TestAlgorithmEngineInitialization:
    """Test algorithm engine initialization and basic functionality."""
    
    def test_algo_engine_initialization(self, algo_engine):
        """Test that AlgoEngine initializes correctly."""
        assert algo_engine is not None
        assert algo_engine.data_engine is not None
        assert algo_engine.running is False
        assert algo_engine._last_signal_states == {}
        assert algo_engine._min_signal_interval > 0
    
    def test_data_hash_generation(self, algo_engine):
        """Test data hash generation consistency."""
        candles = algo_engine.data_engine.get_candles("BTCUSDT", "1m")
        
        hash1 = algo_engine._get_data_hash(candles)
        hash2 = algo_engine._get_data_hash(candles)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    def test_signal_state_tracking(self, algo_engine):
        """Test signal state tracking mechanisms."""
        key = "BTCUSDT_1m"
        current_time = 1234567890
        data_hash = "test_hash"
        signal_type = "open/buy"
        
        # Initially should not exist
        assert key not in algo_engine._last_signal_states
        
        # Update state
        algo_engine._update_signal_state(key, current_time, data_hash, signal_type)
        
        # Should now exist with correct values
        assert key in algo_engine._last_signal_states
        state = algo_engine._last_signal_states[key]
        
        assert state['timestamp'] == current_time
        assert state['data_hash'] == data_hash
        assert state['signal_type'] == signal_type


class TestAlgorithmEngineSignalProcessing:
    """Test signal processing functionality."""
    
    @pytest.mark.asyncio
    async def test_signal_processing_with_buy_strategy(self, algo_engine, buy_strategy):
        """Test signal processing with buy strategy."""
        signal = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
        
        assert signal is not None
        assert isinstance(signal, TradeSignal)
        assert signal.action == "open"
        assert signal.side == "buy"
        assert signal.symbol == "BTCUSDT"
        assert signal.strategy_id == "buy_strategy"
        assert signal.signal_confidence > 0
    
    @pytest.mark.asyncio
    async def test_signal_processing_with_hold_strategy(self, algo_engine, hold_strategy):
        """Test signal processing with hold strategy."""
        signal = await algo_engine.process_signals("BTCUSDT", "1m", hold_strategy)
        
        assert signal is not None
        assert signal.action == "hold"
        assert signal.side == "none"
        assert signal.symbol == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_multi_symbol_processing(self, algo_engine, buy_strategy, test_symbols):
        """Test processing signals for multiple symbols."""
        signals = []
        
        for symbol in test_symbols:
            # Clear throttling for each symbol
            algo_engine._last_signal_states.clear()
            signal = await algo_engine.process_signals(symbol, "1m", buy_strategy)
            if signal:
                signals.append(signal)
        
        assert len(signals) == len(test_symbols)
        
        # Check each signal has correct symbol
        for i, signal in enumerate(signals):
            assert signal.symbol == test_symbols[i]
            assert signal.action == "open"
            assert signal.side == "buy"
    
    @pytest.mark.asyncio
    async def test_signal_throttling_mechanism(self, algo_engine, buy_strategy):
        """Test signal throttling mechanism."""
        # Test that throttling infrastructure exists
        assert hasattr(algo_engine, '_last_signal_states')
        assert hasattr(algo_engine, '_min_signal_interval')
        assert hasattr(algo_engine, '_should_process_signal')
        
        # Generate first signal
        signal1 = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
        assert signal1 is not None
        
        # Verify throttling mechanism exists
        assert algo_engine._min_signal_interval > 0


class TestAlgorithmEngineErrorHandling:
    """Test error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, algo_engine):
        """Test handling of strategy errors."""
        error_strategy = MockErrorStrategy()
        
        # Should handle error gracefully and return None
        signal = await algo_engine.process_signals("BTCUSDT", "1m", error_strategy)
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_empty_candle_data_handling(self, algo_engine, buy_strategy):
        """Test handling of empty candle data."""
        # Mock empty data
        algo_engine.data_engine.get_candles = lambda symbol, timeframe: []
        
        signal = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_data_engine_exception_handling(self, algo_engine, buy_strategy):
        """Test handling of data engine exceptions."""
        # Mock exception
        def raise_exception(*args, **kwargs):
            raise Exception("Data engine error")
        
        algo_engine.data_engine.get_candles = raise_exception
        
        signal = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
        assert signal is None


class TestAlgorithmEnginePerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_rapid_signal_processing(self, algo_engine, buy_strategy):
        """Test rapid signal processing performance."""
        start_time = time.time()
        
        # Process multiple signals rapidly
        signals_processed = 0
        for i in range(10):
            algo_engine._last_signal_states.clear()  # Allow all signals for testing
            signal = await algo_engine.process_signals("BTCUSDT", "1m", buy_strategy)
            if signal:
                signals_processed += 1
        
        execution_time = time.time() - start_time
        
        # Should process rapidly
        assert execution_time < 1.0  # Under 1 second
        assert signals_processed > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, algo_engine, buy_strategy, test_symbols):
        """Test concurrent processing of multiple symbols."""
        # Process signals for all symbols concurrently
        tasks = []
        for symbol in test_symbols:
            algo_engine._last_signal_states.clear()
            task = algo_engine.process_signals(symbol, "1m", buy_strategy)
            tasks.append(task)
        
        signals = await asyncio.gather(*tasks)
        
        # Should get signals for all symbols
        valid_signals = [s for s in signals if s is not None]
        assert len(valid_signals) > 0
        
        # Check signal properties
        for signal in valid_signals:
            assert isinstance(signal, TradeSignal)
            assert signal.action == "open"
            assert signal.side == "buy"


class TestSignalDecisionLogic:
    """Test signal processing decision logic."""
    
    def test_should_process_signal_logic(self, algo_engine):
        """Test signal processing decision logic."""
        key = "BTCUSDT_1m"
        base_time = 1234567890
        data_hash = "test_hash"
        
        # Should process when no prior state
        assert algo_engine._should_process_signal(key, base_time, data_hash) is True
        
        # Set initial state
        algo_engine._update_signal_state(key, base_time, data_hash, "open/buy")
        
        # Should not process immediately with same data
        assert algo_engine._should_process_signal(key, base_time + 30, data_hash) is False
        
        # Should process with different data
        assert algo_engine._should_process_signal(key, base_time + 30, "new_hash") is True
        
        # Should process after time interval
        future_time = base_time + algo_engine._min_signal_interval + 1
        assert algo_engine._should_process_signal(key, future_time, data_hash) is True
    
    @pytest.mark.asyncio
    async def test_mixed_signal_sequence(self, algo_engine, mixed_strategy):
        """Test processing of mixed signal sequences."""
        signals = []
        
        # Process multiple signals with mixed strategy
        for i in range(4):
            algo_engine._last_signal_states.clear()
            signal = await algo_engine.process_signals("BTCUSDT", "1m", mixed_strategy)
            if signal:
                signals.append(signal)
        
        assert len(signals) == 4
        
        # Should follow the sequence: buy, hold, sell, hold
        expected_actions = ["open", "hold", "open", "hold"]
        expected_sides = ["buy", "none", "sell", "none"]
        
        for i, signal in enumerate(signals):
            assert signal.action == expected_actions[i]
            assert signal.side == expected_sides[i]
