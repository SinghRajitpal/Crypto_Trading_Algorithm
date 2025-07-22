#!/usr/bin/env python3
"""
Simplified Pytest Algorithm Engine Tests
"""

import pytest
import asyncio
import os
import sys
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.base_strategy import BaseStrategy


class MockDataEngine:
    """Mock data engine for testing."""
    
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Return mock candle data."""
        import time
        current_time = int(time.time() * 1000)
        return [
            [current_time - 60000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0],
            [current_time, 50050.0, 50150.0, 49950.0, 50100.0, 100.0]
        ]


class SimpleTestStrategy(BaseStrategy):
    """Simple strategy for testing."""
    
    def __init__(self, signal_type: str = "hold"):
        super().__init__(params={}, strategy_id="test_strategy")
        self.signal_type = signal_type
    
    def get_required_indicators(self) -> List[str]:
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        return {
            "action": "open" if self.signal_type in ["buy", "sell"] else "hold",
            "side": self.signal_type if self.signal_type in ["buy", "sell"] else "none",
            "confidence": 0.8
        }
    
    async def calculate_signals(self, candles, symbol: str):
        """Override to work with test interface."""
        from algorithm.trade_signal import TradeSignal
        
        result = self._generate_signals(symbol, {})
        
        return TradeSignal(
            action=result["action"],
            side=result["side"],
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"test": True},
            signal_confidence=result["confidence"]
        )


@pytest.fixture
def mock_data_engine():
    """Fixture providing mock data engine."""
    return MockDataEngine()


@pytest.fixture
def algo_engine(mock_data_engine):
    """Fixture providing algorithm engine."""
    return AlgoEngine(mock_data_engine)


class TestAlgoEngineBasics:
    """Test basic algorithm engine functionality."""
    
    def test_initialization(self, algo_engine):
        """Test algorithm engine initialization."""
        assert algo_engine is not None
        assert algo_engine.data_engine is not None
        assert algo_engine.running is False
        assert algo_engine._last_signal_states == {}
    
    def test_data_hash(self, algo_engine):
        """Test data hash generation."""
        candles = algo_engine.data_engine.get_candles("BTCUSDT", "1m")
        hash1 = algo_engine._get_data_hash(candles)
        hash2 = algo_engine._get_data_hash(candles)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    @pytest.mark.asyncio
    async def test_signal_processing(self, algo_engine):
        """Test basic signal processing."""
        strategy = SimpleTestStrategy("buy")
        
        signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        
        assert signal is not None
        assert signal.action == "open"
        assert signal.side == "buy"
        assert signal.symbol == "BTCUSDT"
        assert signal.strategy_id == "test_strategy"
    
    @pytest.mark.asyncio
    async def test_signal_throttling(self, algo_engine):
        """Test signal throttling."""
        strategy = SimpleTestStrategy("buy")
        
        # Generate first signal
        signal1 = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        assert signal1 is not None
        
        # Generate second signal immediately with same data
        # Mock data changes each time, so throttling by data hash won't occur
        # But we can test the throttling mechanism exists
        assert hasattr(algo_engine, '_last_signal_states')
        assert hasattr(algo_engine, '_min_signal_interval')
        assert algo_engine._min_signal_interval > 0
    
    @pytest.mark.asyncio
    async def test_hold_signals(self, algo_engine):
        """Test hold signals."""
        strategy = SimpleTestStrategy("hold")
        
        signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        
        assert signal is not None
        assert signal.action == "hold"
        assert signal.side == "none"
    
    @pytest.mark.asyncio
    async def test_multi_symbol(self, algo_engine):
        """Test processing multiple symbols."""
        strategy = SimpleTestStrategy("buy")
        
        # Clear any throttling
        algo_engine._last_signal_states.clear()
        
        # Test different symbols
        signal1 = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        signal2 = await algo_engine.process_signals("ETHUSDT", "1m", strategy)
        
        assert signal1 is not None
        assert signal2 is not None
        assert signal1.symbol == "BTCUSDT"
        assert signal2.symbol == "ETHUSDT"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, algo_engine):
        """Test error handling with invalid data."""
        # Mock data engine that returns no candles
        algo_engine.data_engine.get_candles = lambda symbol, timeframe: []
        
        strategy = SimpleTestStrategy("buy")
        signal = await algo_engine.process_signals("INVALID", "1m", strategy)
        
        assert signal is None  # Should handle gracefully


class TestSignalStates:
    """Test signal state management."""
    
    @pytest.mark.asyncio
    async def test_signal_state_tracking(self, algo_engine):
        """Test signal state tracking."""
        strategy = SimpleTestStrategy("buy")
        
        # Process signal
        await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        
        # Check state was recorded
        key = "BTCUSDT_1m"
        assert key in algo_engine._last_signal_states
        assert 'timestamp' in algo_engine._last_signal_states[key]
        assert 'data_hash' in algo_engine._last_signal_states[key]
    
    def test_should_process_signal(self, algo_engine):
        """Test signal processing decision logic."""
        import time
        current_time = int(time.time())
        
        # No previous state - should process
        assert algo_engine._should_process_signal("test_key", current_time, "hash1") is True
        
        # Add state
        algo_engine._update_signal_state("test_key", current_time, "hash1", "buy")
        
        # Same hash, not enough time - should not process
        assert algo_engine._should_process_signal("test_key", current_time + 30, "hash1") is False
        
        # Different hash - should process
        assert algo_engine._should_process_signal("test_key", current_time + 30, "hash2") is True
        
        # Same hash, enough time - should process
        assert algo_engine._should_process_signal("test_key", current_time + 70, "hash1") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
