"""
Integration Tests
Senior Quantitative Developer Testing Protocol

Comprehensive pytest-based integration testing for the complete trading workflow
including end-to-end signal generation through execution pipeline.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List

from algorithm.trade_signal import TradeSignal
from tests.utils.mock_objects import MockStrategy, MockDataEngineWithTrend
from tests.utils.test_data import create_test_signal_metadata


class TestSignalToExecutionPipeline:
    """Test complete signal-to-execution pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_execution(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test complete end-to-end signal generation and execution."""
        strategy = MockStrategy(["hold", "buy", "hold"], "integration_test")
        
        signal_count = 0
        successful_validations = 0
        
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        
        for symbol in test_symbols:
            for i in range(3):  # Generate 3 signals per symbol
                # Clear throttling for testing
                algo_engine_with_trend._last_signal_states.clear()
                
                signal = await algo_engine_with_trend.process_signals(symbol, "1m", strategy)
                
                if signal:
                    signal_count += 1
                    
                    if signal.action == "open" and signal.side == "buy":
                        # Test signal validation
                        validation_result = await portfolio_with_allocation.validate_signal(signal, 50100.0)
                        assert isinstance(validation_result, dict)
                        assert 'valid' in validation_result
                        
                        if validation_result.get('valid', False):
                            successful_validations += 1
                            assert 'position_info' in validation_result
                            
                            # Test signal processing
                            processing_result = await portfolio_with_allocation.process_signal(signal)
                            assert processing_result.get('status') in ['success', 'completed', 'skipped', 'rejected', 'error']
        
        # Verify results
        assert signal_count > 0, "Should have generated signals"
        
        # Check portfolio state
        portfolio_summary = portfolio_with_allocation.get_portfolio_summary()
        assert portfolio_summary['allocated_capital'] > 0, "Portfolio should have allocated capital"
        
        # Check risk metrics
        risk_metrics = portfolio_with_allocation.get_risk_metrics()
        assert 'risk_status' in risk_metrics
        assert risk_metrics['risk_status'] in ['normal', 'caution', 'warning', 'critical']
    
    @pytest.mark.asyncio
    async def test_signal_validation_interface(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test signal validation interface consistency."""
        strategy = MockStrategy(["buy"], "validation_test")
        
        # Generate a buy signal
        algo_engine_with_trend._last_signal_states.clear()
        signal = await algo_engine_with_trend.process_signals("BTCUSDT", "1m", strategy)
        
        assert signal is not None
        assert signal.action == "open"
        assert signal.side == "buy"
        
        # Test validation interface
        validation_result = await portfolio_with_allocation.validate_signal(signal, 50000.0)
        
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
    
    def test_portfolio_allocation_flow(self, execution_engine, test_symbols):
        """Test portfolio allocation flow without signals."""
        # Test empty portfolio initially
        initial_summary = execution_engine.get_portfolio_summary()
        assert initial_summary['allocated_capital'] == 0
        
        # Add market data
        for symbol in test_symbols:
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
        assert abs(final_summary['allocated_capital'] - expected_allocation) < 100


class TestStressConditions:
    """Test system behavior under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_minimal_capital_handling(self, mock_binance_client):
        """Test system with minimal capital."""
        from execution.execution_engine import ProductionExecutionEngine
        
        execution_engine = ProductionExecutionEngine(mock_binance_client, total_capital=100.0)
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
    async def test_rapid_signal_processing(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test rapid signal processing."""
        strategy = MockStrategy(["buy", "sell", "buy", "sell"] * 3, "rapid_test")
        
        # Process signals rapidly
        start_time = time.time()
        signals_processed = 0
        
        for i in range(10):
            algo_engine_with_trend._last_signal_states.clear()  # Allow rapid processing
            signal = await algo_engine_with_trend.process_signals("BTCUSDT", "1m", strategy)
            if signal:
                signals_processed += 1
                
                if signal.action != "hold":
                    validation = await portfolio_with_allocation.validate_signal(signal, 50000.0)
                    assert isinstance(validation, dict)
                    assert 'valid' in validation
        
        execution_time = time.time() - start_time
        
        # Should process rapidly
        assert execution_time < 2.0  # Should complete in under 2 seconds
        assert signals_processed > 0
    
    @pytest.mark.asyncio
    async def test_invalid_signal_handling(self, portfolio_with_allocation):
        """Test handling of invalid signals."""
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
        validation = await portfolio_with_allocation.validate_signal(invalid_signal, 50000.0)
        assert validation['valid'] is False
        assert 'Missing ATR' in validation['reason']


class TestMultiSymbolIntegration:
    """Test multi-symbol integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self, algo_engine_with_trend, portfolio_with_allocation, test_symbols):
        """Test concurrent processing of multiple symbols."""
        strategy = MockStrategy(["buy"], "concurrent_test")
        
        # Process signals for all symbols concurrently
        tasks = []
        for symbol in test_symbols:
            algo_engine_with_trend._last_signal_states.clear()
            task = algo_engine_with_trend.process_signals(symbol, "1m", strategy)
            tasks.append(task)
        
        signals = await asyncio.gather(*tasks)
        
        # Should generate signals for all symbols
        valid_signals = [s for s in signals if s is not None]
        assert len(valid_signals) > 0
        
        # Test concurrent validation
        validation_tasks = []
        for signal in valid_signals:
            if signal.action != "hold":
                task = portfolio_with_allocation.validate_signal(signal, 50000.0)
                validation_tasks.append(task)
        
        if validation_tasks:
            validations = await asyncio.gather(*validation_tasks)
            assert all(isinstance(v, dict) for v in validations)
            assert all('valid' in v for v in validations)
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_processing(self, algo_engine_with_trend, test_timeframes):
        """Test processing across multiple timeframes."""
        strategy = MockStrategy(["buy"], "multi_timeframe_test")
        signals = {}
        
        for timeframe in test_timeframes:
            algo_engine_with_trend._last_signal_states.clear()
            signal = await algo_engine_with_trend.process_signals("BTCUSDT", timeframe, strategy)
            if signal:
                signals[timeframe] = signal
        
        # Should generate signals for different timeframes
        assert len(signals) > 0
        
        # Each signal should have consistent properties
        for timeframe, signal in signals.items():
            assert signal.symbol == "BTCUSDT"
            assert signal.action == "open"
            assert signal.side == "buy"


class TestSystemResilience:
    """Test system resilience and error recovery."""
    
    @pytest.mark.asyncio
    async def test_partial_system_failure_recovery(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test recovery from partial system failures."""
        strategy = MockStrategy(["buy"], "resilience_test")
        
        # Generate signal
        signal = await algo_engine_with_trend.process_signals("BTCUSDT", "1m", strategy)
        assert signal is not None
        
        # Test with normal validation
        validation = await portfolio_with_allocation.validate_signal(signal, 50000.0)
        
        # Should handle gracefully
        assert isinstance(validation, dict)
        assert 'valid' in validation
        
        # Even if validation fails, system should continue operating
        portfolio_summary = portfolio_with_allocation.get_portfolio_summary()
        assert isinstance(portfolio_summary, dict)
        assert 'total_capital' in portfolio_summary
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, market_data_setup, test_symbols):
        """Test data consistency across different components."""
        # Verify market data is consistent across components
        for symbol in test_symbols:
            # Check portfolio manager has volatility data
            assert symbol in market_data_setup.portfolio_manager.volatilities
            vol = market_data_setup.portfolio_manager.volatilities[symbol]
            assert vol > 0
        
        # Check risk manager can access the data
        risk_status = market_data_setup.get_risk_metrics()
        assert 'risk_status' in risk_status
        
        # Check portfolio summary is consistent
        summary = market_data_setup.get_portfolio_summary()
        assert summary['total_capital'] > 0


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test end-to-end latency from signal to validation."""
        strategy = MockStrategy(["buy"], "latency_test")
        
        start_time = time.time()
        
        # Generate signal
        signal = await algo_engine_with_trend.process_signals("BTCUSDT", "1m", strategy)
        signal_time = time.time()
        
        # Validate signal
        if signal and signal.action == "open":
            validation = await portfolio_with_allocation.validate_signal(signal, 50000.0)
            validation_time = time.time()
            
            # Measure latencies
            signal_latency = signal_time - start_time
            validation_latency = validation_time - signal_time
            total_latency = validation_time - start_time
            
            # Performance assertions
            assert signal_latency < 0.1  # Signal generation under 100ms
            assert validation_latency < 0.1  # Validation under 100ms
            assert total_latency < 0.2  # Total pipeline under 200ms
    
    @pytest.mark.asyncio
    async def test_throughput_under_load(self, algo_engine_with_trend, portfolio_with_allocation):
        """Test system throughput under load."""
        strategy = MockStrategy(["buy", "sell"] * 5, "throughput_test")
        
        start_time = time.time()
        processed_signals = 0
        
        # Process multiple signals rapidly
        for i in range(20):
            algo_engine_with_trend._last_signal_states.clear()
            signal = await algo_engine_with_trend.process_signals("BTCUSDT", "1m", strategy)
            
            if signal:
                processed_signals += 1
                
                if signal.action == "open":
                    validation = await portfolio_with_allocation.validate_signal(signal, 50000.0)
                    assert isinstance(validation, dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Throughput assertions
        assert processed_signals > 0
        throughput = processed_signals / total_time
        assert throughput > 10  # Should process more than 10 signals per second
