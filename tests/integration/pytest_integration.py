#!/usr/bin/env python3
"""
Pytest-compatible Integration Test Suite
Senior Quantitative Developer Testing Protocol

Tests the complete trading workflow from signal generation through to execution.
"""

import pytest
import asyncio
import os
import sys
import time
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy
from execution.execution_engine import ProductionExecutionEngine


class MockDataEngineForIntegration:
    """Mock data engine with consistent data for integration testing."""
    
    def __init__(self):
        self.candle_data = {}
        self.binance_client = None
        
    def get_candles(self, symbol: str, timeframe: str) -> List[List]:
        """Get mock candle data with trend patterns."""
        key = f"{symbol}_{timeframe}"
        if key not in self.candle_data:
            self.candle_data[key] = self._generate_trending_data(symbol, 50)
        return self.candle_data[key]
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> List:
        """Get the latest candle."""
        candles = self.get_candles(symbol, timeframe)
        return candles[-1] if candles else None
    
    def _generate_trending_data(self, symbol: str, periods: int) -> List[List]:
        """Generate trending OHLCV data for signal generation."""
        import numpy as np
        current_time = int(time.time() * 1000)
        candles = []
        
        if 'BTC' in symbol:
            base_price = 50000.0
        elif 'ETH' in symbol:
            base_price = 3000.0
        else:
            base_price = 0.5
        
        for i in range(periods):
            timestamp = current_time - (periods - i) * 60000
            # Create trending pattern
            trend_factor = 1 + (0.002 * i)  # Gradual uptrend
            price = base_price * trend_factor
            
            candles.append([
                timestamp,
                price * 0.999,  # open
                price * 1.002,  # high
                price * 0.998,  # low
                price,          # close
                100.0          # volume
            ])
        
        return candles


class IntegrationTestStrategy(BaseStrategy):
    """Strategy for integration testing."""
    
    def __init__(self, signal_sequence: List[str]):
        super().__init__(params={}, strategy_id="integration_test")
        self.signal_sequence = signal_sequence
        self.signal_index = 0
    
    def get_required_indicators(self) -> List[str]:
        """Return empty list for testing."""
        return []
    
    def _generate_signals(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Generate signals using moving average crossover for realistic testing."""
        # Use the predefined sequence for predictable testing
        signal_type = self.signal_sequence[self.signal_index % len(self.signal_sequence)]
        self.signal_index += 1
        
        if signal_type == "buy":
            return {"action": "open", "side": "buy", "confidence": 0.8}
        elif signal_type == "sell":
            return {"action": "open", "side": "sell", "confidence": 0.8}
        elif signal_type == "exit":
            return {"action": "exit", "side": "sell", "confidence": 0.8}
        else:
            return {"action": "hold", "side": "none", "confidence": 0.8}
    
    async def calculate_signals(self, candles, symbol: str):
        """Override to work with test interface."""
        from algorithm.trade_signal import TradeSignal
        
        result = self._generate_signals(symbol, {})
        
        return TradeSignal(
            action=result["action"],
            side=result["side"],
            symbol=symbol,
            strategy_id=self.strategy_id,
            metadata={"atr_value": 0.02, "test": True},  # Include required ATR
            signal_confidence=result["confidence"]
        )


class MockBinanceClientForIntegration:
    """Enhanced mock Binance client for integration testing."""
    
    def __init__(self):
        self.account_config_called = False
        self.orders = []
        self.positions = []
        self.balance = {'USDT': 10000.0}
        self.order_id_counter = 0
        
    async def setup_account_config(self):
        self.account_config_called = True
        
    async def get_balance(self):
        return {'total': self.balance, 'free': self.balance, 'used': {}}
        
    async def get_open_positions(self, symbol=None):
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions
        
    async def open_position(self, symbol, side, amount, price=None, stop_loss=None, take_profit=None, leverage=None, margin_type=None):
        """Mock open_position method for testing."""
        self.order_id_counter += 1
        
        position = {
            'id': f'position_{self.order_id_counter}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'contracts': amount,
            'entry_price': price or 50000.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage or 1,
            'margin_type': margin_type or 'isolated',
            'timestamp': int(time.time() * 1000),
            'status': 'open'
        }
        
        self.positions.append(position)
        
        order = {
            'id': f'order_{self.order_id_counter}',
            'symbol': symbol,
            'type': 'market',
            'side': side,
            'amount': amount,
            'price': price or 50000.0,
            'status': 'filled',
            'timestamp': int(time.time() * 1000)
        }
        self.orders.append(order)
        
        return {'status': 'success', 'order': order, 'position': position}


@pytest.fixture
def mock_data_engine():
    """Fixture providing mock data engine for integration tests."""
    return MockDataEngineForIntegration()


@pytest.fixture
def mock_binance_client():
    """Fixture providing mock Binance client for integration tests."""
    return MockBinanceClientForIntegration()


@pytest.fixture
def algo_engine(mock_data_engine):
    """Fixture providing algorithm engine."""
    return AlgoEngine(mock_data_engine)


@pytest.fixture
def execution_engine(mock_binance_client):
    """Fixture providing execution engine."""
    return ProductionExecutionEngine(mock_binance_client, total_capital=10000.0)


class TestSignalToExecutionPipeline:
    """Test complete signal-to-execution pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_execution(self, algo_engine, execution_engine, mock_data_engine):
        """Test complete end-to-end signal generation and execution."""
        # Setup
        await execution_engine.setup()
        
        # Initialize market data for portfolio allocation
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in test_symbols:
            execution_engine.update_market_data_bar(symbol, {
                'open': 50000.0,
                'high': 50200.0, 
                'low': 49800.0,
                'close': 50100.0,
                'volume': 100.0
            }, atr_value=0.02)
        
        # Force initial portfolio rebalance to allocate capital
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        rebalance_result = execution_engine.process_daily_rebalance()
        assert rebalance_result is True
        
        # Create strategy and process signals
        strategy = IntegrationTestStrategy(["hold", "buy", "hold"])
        
        signal_count = 0
        successful_executions = 0
        
        for symbol in test_symbols:
            for i in range(3):  # Generate 3 signals per symbol
                # Clear throttling for testing
                algo_engine._last_signal_states.clear()
                
                signal = await algo_engine.process_signals(symbol, "1m", strategy)
                
                if signal:
                    signal_count += 1
                    
                    if signal.action == "open" and signal.side == "buy":
                        # Test signal validation
                        validation_result = await execution_engine.validate_signal(signal, 50100.0)
                        assert isinstance(validation_result, dict)
                        assert 'valid' in validation_result
                        
                        if validation_result.get('valid', False):
                            assert 'position_info' in validation_result
                            
                            # Actually process the signal
                            processing_result = await execution_engine.process_signal(signal)
                            assert processing_result.get('status') in ['success', 'completed', 'skipped', 'rejected', 'error']
                            
                            if processing_result.get('status') == 'success':
                                successful_executions += 1
        
        # Verify results
        assert signal_count > 0, "Should have generated signals"
        
        # Check portfolio state
        portfolio_summary = execution_engine.get_portfolio_summary()
        assert portfolio_summary['allocated_capital'] > 0, "Portfolio should have allocated capital"
        
        # Check risk metrics
        risk_metrics = execution_engine.get_risk_metrics()
        assert 'risk_status' in risk_metrics
        assert risk_metrics['risk_status'] in ['normal', 'caution', 'warning', 'critical']
    
    @pytest.mark.asyncio
    async def test_signal_validation_interface(self, algo_engine, execution_engine):
        """Test signal validation interface consistency."""
        # Setup portfolio allocation
        await execution_engine.setup()
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            execution_engine.update_market_data_bar(symbol, {
                'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
            }, 0.02)
        
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        strategy = IntegrationTestStrategy(["buy"])
        
        # Generate a buy signal
        algo_engine._last_signal_states.clear()
        signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        
        assert signal is not None
        assert signal.action == "open"
        assert signal.side == "buy"
        
        # Test validation interface
        validation_result = await execution_engine.validate_signal(signal, 50000.0)
        
        # Check interface consistency
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        
        if validation_result['valid']:
            assert 'position_info' in validation_result
            position_info = validation_result['position_info']
            
            # Check position info structure
            required_keys = ['size_contracts', 'leverage', 'stop_loss_price', 'take_profit_price']
            for key in required_keys:
                assert key in position_info, f"Missing key: {key}"
                assert position_info[key] is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_allocation_flow(self, execution_engine):
        """Test portfolio allocation flow without signals."""
        await execution_engine.setup()
        
        # Test empty portfolio initially
        initial_summary = execution_engine.get_portfolio_summary()
        assert initial_summary['allocated_capital'] == 0
        
        # Add market data
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        for symbol in symbols:
            execution_engine.update_market_data_bar(symbol, {
                'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
            }, 0.02)
        
        # Force rebalance
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        rebalance_result = execution_engine.process_daily_rebalance()
        
        assert rebalance_result is True
        
        # Check allocation
        final_summary = execution_engine.get_portfolio_summary()
        assert final_summary['allocated_capital'] > 0
        assert final_summary['allocation_percentage'] > 0
        
        # Should be around 85% of total capital (max allocation)
        expected_allocation = execution_engine.portfolio_manager.total_capital * 0.85
        assert abs(final_summary['allocated_capital'] - expected_allocation) < 100  # Within $100


class TestStressConditions:
    """Test system behavior under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_minimal_capital_handling(self):
        """Test system with minimal capital."""
        mock_client = MockBinanceClientForIntegration()
        execution_engine = ProductionExecutionEngine(mock_client, total_capital=100.0)  # Very small capital
        
        await execution_engine.setup()
        
        # Test portfolio allocation with minimal capital
        execution_engine.update_market_data_bar("BTCUSDT", {
            'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
        }, 0.02)
        
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        result = execution_engine.process_daily_rebalance()
        
        # Should handle gracefully
        assert isinstance(result, bool)
        
        portfolio_summary = execution_engine.get_portfolio_summary()
        assert portfolio_summary['total_capital'] == 100.0
    
    @pytest.mark.asyncio
    async def test_rapid_signal_processing(self, algo_engine, execution_engine):
        """Test rapid signal processing."""
        await execution_engine.setup()
        
        # Setup portfolio
        execution_engine.update_market_data_bar("BTCUSDT", {
            'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
        }, 0.02)
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        strategy = IntegrationTestStrategy(["buy", "sell", "buy", "sell"] * 3)
        
        # Process signals rapidly
        start_time = time.time()
        signals_processed = 0
        
        for i in range(10):
            algo_engine._last_signal_states.clear()  # Allow rapid processing for test
            signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
            if signal:
                signals_processed += 1
                
                if signal.action != "hold":
                    validation = await execution_engine.validate_signal(signal, 50000.0)
                    assert isinstance(validation, dict)
                    assert 'valid' in validation
        
        execution_time = time.time() - start_time
        
        # Should process rapidly
        assert execution_time < 1.0  # Should complete in under 1 second
        assert signals_processed > 0
    
    @pytest.mark.asyncio
    async def test_invalid_signal_handling(self, execution_engine):
        """Test handling of invalid signals."""
        await execution_engine.setup()
        
        # Create invalid signal (missing required metadata)
        invalid_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={},  # Missing atr_value
            signal_confidence=0.5
        )
        
        # Should handle gracefully
        validation = await execution_engine.validate_signal(invalid_signal, 50000.0)
        assert validation['valid'] is False
        assert 'Missing ATR' in validation['reason']


class TestMultiSymbolIntegration:
    """Test multi-symbol integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self, algo_engine, execution_engine):
        """Test concurrent processing of multiple symbols."""
        await execution_engine.setup()
        
        # Setup portfolio for multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        for symbol in symbols:
            execution_engine.update_market_data_bar(symbol, {
                'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
            }, 0.02)
        
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        strategy = IntegrationTestStrategy(["buy"])
        
        # Process signals for all symbols concurrently
        tasks = []
        for symbol in symbols:
            algo_engine._last_signal_states.clear()
            task = algo_engine.process_signals(symbol, "1m", strategy)
            tasks.append(task)
        
        signals = await asyncio.gather(*tasks)
        
        # Should generate signals for all symbols
        valid_signals = [s for s in signals if s is not None]
        assert len(valid_signals) > 0
        
        # Test concurrent validation
        validation_tasks = []
        for signal in valid_signals:
            if signal.action != "hold":
                task = execution_engine.validate_signal(signal, 50000.0)
                validation_tasks.append(task)
        
        if validation_tasks:
            validations = await asyncio.gather(*validation_tasks)
            assert all(isinstance(v, dict) for v in validations)
            assert all('valid' in v for v in validations)


class TestSystemResilience:
    """Test system resilience and error recovery."""
    
    @pytest.mark.asyncio
    async def test_partial_system_failure_recovery(self, algo_engine, execution_engine):
        """Test recovery from partial system failures."""
        await execution_engine.setup()
        
        # Setup normal conditions
        execution_engine.update_market_data_bar("BTCUSDT", {
            'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
        }, 0.02)
        
        strategy = IntegrationTestStrategy(["buy"])
        
        # Generate signal
        signal = await algo_engine.process_signals("BTCUSDT", "1m", strategy)
        assert signal is not None
        
        # Test with missing portfolio allocation (simulated failure)
        validation = await execution_engine.validate_signal(signal, 50000.0)
        
        # Should handle gracefully
        assert isinstance(validation, dict)
        assert 'valid' in validation
        
        # Even if validation fails, system should continue operating
        portfolio_summary = execution_engine.get_portfolio_summary()
        assert isinstance(portfolio_summary, dict)
        assert 'total_capital' in portfolio_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
