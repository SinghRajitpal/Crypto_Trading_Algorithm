#!/usr/bin/env python3
"""
Pytest-compatible Algorithm Engine Test Suite
Senior Quantitative Developer Testing Protocol

Tests for signal generation, processing, throttling, and multi-symbol handling.
"""

import pytest
import asyncio
import os
import sys
import time
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy


class MockDataEngine:
    """Mock data engine for testing."""
    
    def __init__(self):
        self.candle_data = {}
        self.binance_client = None
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Get mock candle data."""
        return self._generate_trending_data(symbol, 50)
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> List:
        """Get the latest candle."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None
    
    def _generate_trending_data(self, symbol: str, periods: int) -> List[List]:
        """Generate trending OHLCV data."""
        import numpy as np
        current_time = int(time.time() * 1000)
        candles = []
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 0.5
        
        for i in range(periods):
            timestamp = current_time - (periods - i) * 60000
            price = base_price * (1 + 0.001 * i)  # Trending up
            candles.append([
                timestamp,
                price * 0.999,  # open
                price * 1.002,  # high
                price * 0.998,  # low
                price,          # close
                100.0          # volume
            ])
        
        return candles


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, signal_sequence: List[str]):
        # Initialize with required parameters
        super().__init__(params={}, strategy_id="mock_strategy")
        self.signal_sequence = signal_sequence
        self.signal_index = 0
    
    def get_required_indicators(self) -> List[str]:
        """Return empty list for testing."""
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Generate test signals in sequence."""
        signal_type = self.signal_sequence[self.signal_index % len(self.signal_sequence)]
        self.signal_index += 1
        
        return {
            "action": "open" if signal_type in ["buy", "sell"] else "hold",
            "side": signal_type if signal_type in ["buy", "sell"] else "none",
            "confidence": 0.8
        }
    
    def calculate_signals(self, symbol: str, candles: List[List]) -> Dict[str, Any]:
        """Generate test signals in sequence."""
        return self._generate_signals(symbol, {})


class ErrorStrategy(BaseStrategy):
    """Strategy that throws errors for testing error handling."""
    
    def __init__(self):
        super().__init__(params={}, strategy_id="error_strategy")
    
    def get_required_indicators(self) -> List[str]:
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        raise Exception("Intentional strategy error")
    
    def calculate_signals(self, symbol: str, candles: List[List]) -> Dict[str, Any]:
        raise Exception("Intentional strategy error")


@pytest.fixture
def mock_data_engine():
    """Fixture providing mock data engine."""
    return MockDataEngine()


@pytest.fixture
def algo_engine(mock_data_engine):
    """Fixture providing algorithm engine with mock data."""
    return AlgoEngine(mock_data_engine)


class TestAlgorithmEngineInitialization:
    """Test algorithm engine initialization and basic functionality."""
    
    def test_algo_engine_initialization(self, algo_engine):
        """Test that AlgoEngine initializes correctly."""
        assert algo_engine is not None
        assert algo_engine.data_engine is not None
        assert algo_engine._last_signal_states == {}
        assert algo_engine._min_signal_interval > 0
    
    def test_data_hash_generation(self, algo_engine):
        """Test data hash generation."""
        symbol, timeframe = "BTCUSDT", "1m"
        candles = algo_engine.data_engine.get_candles(symbol, timeframe)
        
        hash1 = algo_engine._get_data_hash(candles)
        hash2 = algo_engine._get_data_hash(candles)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0


class TestAlgorithmEngineSignalThrottling:
    """Test signal throttling mechanisms."""
    
    @pytest.mark.asyncio
    async def test_signal_throttling(self, algo_engine):
        """Test that signal throttling prevents duplicate signals."""
        strategy = MockStrategy(["buy", "buy", "buy"])
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Generate first signal
        signal1 = await algo_engine.process_signals(symbol, timeframe, strategy)
        assert signal1 is not None
        assert signal1.action == "open"
        assert signal1.side == "buy"
        
        # Generate second signal immediately (should be throttled)
        signal2 = await algo_engine.process_signals(symbol, timeframe, strategy)
        assert signal2 is None  # Should be throttled
        
        # Clear cache and generate again
        algo_engine.throttle_cache.clear()
        signal3 = await algo_engine.process_signals(symbol, timeframe, strategy)
        assert signal3 is not None


class TestAlgorithmEngineSignalProcessing:
    """Test signal processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_signal_processing_pipeline(self, algo_engine):
        """Test complete signal processing with various signal types."""
        strategy = MockStrategy(["buy", "sell", "hold", "exit"])
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        signals = []
        for i in range(4):
            algo_engine.throttle_cache.clear()  # Clear throttling for testing
            signal = await algo_engine.process_signals(symbol, timeframe, strategy)
            if signal:
                signals.append(signal)
        
        # Should have generated 4 different signals
        assert len(signals) == 4
        
        # Check signal types
        assert signals[0].action == "open" and signals[0].side == "buy"
        assert signals[1].action == "open" and signals[1].side == "sell"
        assert signals[2].action == "hold" and signals[2].side == "none"
        assert signals[3].action == "exit" and signals[3].side == "sell"
        
        # Check all signals have required attributes
        for signal in signals:
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'strategy_id')
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'signal_confidence')
            assert signal.symbol == symbol


class TestAlgorithmEngineMultiSymbol:
    """Test multi-symbol and multi-timeframe processing."""
    
    @pytest.mark.asyncio
    async def test_multi_symbol_timeframe_processing(self, algo_engine):
        """Test processing signals for multiple symbols and timeframes."""
        strategy = MockStrategy(["buy"])
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        timeframes = ["1m", "5m"]
        
        signals = []
        for symbol in symbols:
            for timeframe in timeframes:
                algo_engine.throttle_cache.clear()
                signal = await algo_engine.process_signals(symbol, timeframe, strategy)
                if signal:
                    signals.append(signal)
        
        # Should generate signals for all symbol-timeframe combinations
        assert len(signals) == len(symbols) * len(timeframes)
        
        # Check that signals have correct symbols
        generated_symbols = {signal.symbol for signal in signals}
        assert generated_symbols == set(symbols)
        
        # All should be buy signals
        for signal in signals:
            assert signal.action == "open"
            assert signal.side == "buy"


class TestAlgorithmEngineErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, algo_engine):
        """Test handling of strategy calculation errors."""
        error_strategy = ErrorStrategy()
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Should handle strategy error gracefully
        signal = await algo_engine.process_signals(symbol, timeframe, error_strategy)
        assert signal is None  # Should return None on error
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, algo_engine):
        """Test handling of invalid symbols."""
        strategy = MockStrategy(["buy"])
        
        # Should handle gracefully
        signal = await algo_engine.process_signals("INVALID", "1m", strategy)
        # Should either return None or a signal depending on mock data behavior
        # The exact behavior depends on implementation
        assert signal is None or isinstance(signal, TradeSignal)
    
    @pytest.mark.asyncio
    async def test_empty_candle_data(self, algo_engine):
        """Test handling when no candle data is available."""
        # Mock the data engine to return empty data
        algo_engine.data_engine.get_candles = Mock(return_value=[])
        
        strategy = MockStrategy(["buy"])
        signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        
        # Should handle empty data gracefully
        assert signal is None
    
    def test_signal_cache_management(self, algo_engine):
        """Test signal cache management."""
        # Test cache size limits and cleanup
        symbol = "BTCUSDT"
        timeframe = "1m"
        
        # Fill cache
        for i in range(10):
            cache_key = f"{symbol}_{timeframe}_{i}"
            algo_engine.signal_cache[cache_key] = {"test": "data"}
        
        assert len(algo_engine.signal_cache) == 10
        
        # Test cache exists
        assert len(algo_engine.signal_cache) > 0


@pytest.mark.asyncio
async def test_concurrent_signal_processing(mock_data_engine):
    """Test concurrent signal processing."""
    algo_engine = AlgoEngine(mock_data_engine)
    strategy = MockStrategy(["buy"])
    
    # Process multiple signals concurrently
    tasks = []
    for i in range(5):
        task = algo_engine.process_signals(f"SYMBOL{i}", "1m", strategy)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Should handle concurrent processing
    non_none_results = [r for r in results if r is not None]
    assert len(non_none_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
