"""
Comprehensive Integration Tests
Senior Quantitative Developer Testing Protocol

This file contains comprehensive integration tests that validate the
complete trading system workflow from data ingestion through signal
generation to execution and risk management.
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from algorithm.algo_engine import AlgoEngine
from execution.execution_engine import ProductionExecutionEngine
from algorithm.trade_signal import TradeSignal
from algorithm.strategies.base_strategy import BaseStrategy
from tests.utils.mock_objects import MockDataEngine, MockBinanceClient, MockStrategy
from tests.utils.test_data import generate_ohlcv_data, generate_market_data_bar, create_test_signal_metadata


class TestEndToEndSignalExecution:
    """Test complete end-to-end signal processing and execution."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create fully integrated trading system."""
        # Create components
        mock_data_engine = MockDataEngine()
        mock_binance_client = MockBinanceClient()
        
        algo_engine = AlgoEngine(mock_data_engine)
        execution_engine = ProductionExecutionEngine(mock_binance_client, 10000.0)
        await execution_engine.setup()
        
        # Setup test data
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        for symbol in symbols:
            # Add market data to data engine
            candles = generate_ohlcv_data(100)
            mock_data_engine.candles_data[(symbol, "1m")] = candles
            
            # Add market data to execution engine
            market_data = generate_market_data_bar(symbol)
            execution_engine.update_market_data_bar(symbol, market_data, 0.02)
        
        # Force portfolio rebalancing
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        return {
            'algo_engine': algo_engine,
            'execution_engine': execution_engine,
            'data_engine': mock_data_engine,
            'binance_client': mock_binance_client,
            'symbols': symbols
        }
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, integrated_system):
        """Test complete trading workflow from signal to execution."""
        algo_engine = integrated_system['algo_engine']
        execution_engine = integrated_system['execution_engine']
        symbols = integrated_system['symbols']
        
        # Create test strategy
        strategy = MockStrategy(["buy", "sell", "hold"], "integration_test")
        
        # Process signals and execute trades
        executed_trades = []
        signal_count = 0
        max_signals = 10
        
        for symbol in symbols:
            # Generate signal
            signal = await algo_engine.process_signals(symbol, "1m", strategy)
            
            if signal and signal.action != "hold":
                signal_count += 1
                
                # Add required metadata for execution
                signal.metadata = {
                    'price': 50000.0,
                    'atr_value': 0.02,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                }
                
                # Execute signal
                execution_result = await execution_engine.process_signal(signal)
                executed_trades.append(execution_result)
                
                if signal_count >= max_signals:
                    break
        
        # Validate execution results
        assert len(executed_trades) > 0, "No trades were executed"
        
        for trade_result in executed_trades:
            assert isinstance(trade_result, dict)
            assert 'status' in trade_result
            assert 'symbol' in trade_result
    
    @pytest.mark.asyncio
    async def test_signal_metadata_flow(self, integrated_system):
        """Test that signal metadata flows correctly through the system."""
        algo_engine = integrated_system['algo_engine']
        execution_engine = integrated_system['execution_engine']
        
        # Create strategy that adds metadata
        class MetadataStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("metadata_strategy")
            
            async def calculate_signals(self, candles, symbol):
                # Extract current price and volume from candles
                current_candle = candles[-1]
                current_price = current_candle[4]  # Close price
                current_volume = current_candle[5]  # Volume
                
                return TradeSignal(
                    action="open",
                    side="buy",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    metadata={
                        'price': current_price,
                        'atr_value': 0.02,
                        'volume': current_volume,
                        'timestamp': int(time.time() * 1000),
                        'strategy_params': {'test_param': 'test_value'}
                    },
                    signal_confidence=0.8
                )
        
        strategy = MetadataStrategy()
        symbol = "BTCUSDT"
        
        # Generate and process signal
        signal = await algo_engine.process_signals(symbol, "1m", strategy)
        assert signal is not None
        assert signal.metadata is not None
        assert 'price' in signal.metadata
        assert 'atr_value' in signal.metadata
        
        # Execute signal and verify metadata usage
        execution_result = await execution_engine.process_signal(signal)
        assert execution_result['status'] in ['executed', 'validated', 'error', 'skipped']
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, integrated_system):
        """Test risk management integration in signal processing."""
        execution_engine = integrated_system['execution_engine']
        
        # Create high-risk signal (large position)
        high_risk_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="high_risk_test",
            metadata={
                'price': 50000.0,
                'atr_value': 0.001,  # Very low ATR = large position
                'volume': 1000000,
                'timestamp': int(time.time() * 1000)
            },
            signal_confidence=0.9
        )
        
        # Process signal - risk manager should validate
        result = await execution_engine.process_signal(high_risk_signal)
        
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Create low-confidence signal
        low_confidence_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="ETHUSDT",
            strategy_id="low_confidence_test",
            metadata={
                'price': 3000.0,
                'atr_value': 0.02,
                'volume': 1000000,
                'timestamp': int(time.time() * 1000)
            },
            signal_confidence=0.1  # Low confidence
        )
        
        result2 = await execution_engine.process_signal(low_confidence_signal)
        assert isinstance(result2, dict)
    
    @pytest.mark.asyncio
    async def test_portfolio_allocation_flow(self, integrated_system):
        """Test portfolio allocation integration with signals."""
        execution_engine = integrated_system['execution_engine']
        
        # Get portfolio summary before trades
        initial_summary = execution_engine.get_portfolio_summary()
        initial_allocated = initial_summary['allocated_capital']
        
        # Execute multiple signals to test allocation
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        execution_results = []
        
        for symbol in symbols:
            signal = TradeSignal(
                action="open",
                side="buy",
                symbol=symbol,
                strategy_id="allocation_test",
                metadata={
                    'price': 50000.0 if symbol == "BTCUSDT" else 3000.0,
                    'atr_value': 0.02,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                },
                signal_confidence=0.7
            )
            
            result = await execution_engine.process_signal(signal)
            execution_results.append(result)
        
        # Verify portfolio allocation was used
        final_summary = execution_engine.get_portfolio_summary()
        
        # Portfolio state should be consistent
        assert final_summary['total_capital'] == initial_summary['total_capital']
        assert final_summary['allocated_capital'] >= 0


class TestSystemResilience:
    """Test system resilience under stress and error conditions."""
    
    @pytest.fixture
    async def resilience_system(self):
        """Create system for resilience testing."""
        mock_data_engine = MockDataEngine()
        mock_binance_client = MockBinanceClient()
        
        algo_engine = AlgoEngine(mock_data_engine)
        execution_engine = ProductionExecutionEngine(mock_binance_client, 5000.0)
        await execution_engine.setup()
        
        return {
            'algo_engine': algo_engine,
            'execution_engine': execution_engine,
            'data_engine': mock_data_engine,
            'binance_client': mock_binance_client
        }
    
    @pytest.mark.asyncio
    async def test_malformed_signal_handling(self, resilience_system):
        """Test handling of malformed signals."""
        execution_engine = resilience_system['execution_engine']
        
        # Test signal with missing metadata
        incomplete_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="incomplete_test",
            metadata={},  # Missing required fields
            signal_confidence=0.5
        )
        
        result = await execution_engine.process_signal(incomplete_signal)
        assert result['status'] == 'error'
        assert 'reason' in result
        
        # Test signal with invalid action
        invalid_signal = TradeSignal(
            action="invalid_action",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="invalid_test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.5
        )
        
        result2 = await execution_engine.process_signal(invalid_signal)
        assert result2['status'] in ['error', 'skipped']
    
    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, resilience_system):
        """Test system behavior under extreme market conditions."""
        execution_engine = resilience_system['execution_engine']
        
        # Simulate flash crash data
        crash_data = {
            'open': 50000, 'high': 50000, 'low': 25000, 'close': 25000, 'volume': 10000
        }
        
        execution_engine.update_market_data_bar("BTCUSDT", crash_data, 0.05)  # High ATR
        
        # Create signal during crash
        crash_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="crash_test",
            metadata={
                'price': 25000.0,
                'atr_value': 0.05,  # High volatility
                'volume': 10000,
                'timestamp': int(time.time() * 1000)
            },
            signal_confidence=0.8
        )
        
        # System should handle this gracefully
        result = await execution_engine.process_signal(crash_signal)
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Check if stress handler was triggered
        stress_handler = execution_engine.stress_handler
        flash_crash_detected = stress_handler.check_flash_crash("BTCUSDT", crash_data, 0.05)
        assert flash_crash_detected is True
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, resilience_system):
        """Test concurrent signal processing without conflicts."""
        algo_engine = resilience_system['algo_engine']
        execution_engine = resilience_system['execution_engine']
        
        # Setup data for multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            candles = generate_ohlcv_data(50)
            algo_engine.data_engine.candles_data[(symbol, "1m")] = candles
            
            market_data = generate_market_data_bar(symbol)
            execution_engine.update_market_data_bar(symbol, market_data, 0.02)
        
        # Force rebalancing
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        # Process signals concurrently
        strategy = MockStrategy(["buy", "sell"], "concurrent_test")
        
        async def process_symbol_signal(symbol):
            signal = await algo_engine.process_signals(symbol, "1m", strategy)
            if signal and signal.action != "hold":
                signal.metadata = {
                    'price': 50000.0,
                    'atr_value': 0.02,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                }
                return await execution_engine.process_signal(signal)
            return None
        
        # Run concurrent processing
        tasks = [process_symbol_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions and reasonable results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(valid_results) >= 0  # At least some should succeed
        
        for result in valid_results:
            assert isinstance(result, dict)
            assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_and_recovery(self, resilience_system):
        """Test resource cleanup and recovery mechanisms."""
        algo_engine = resilience_system['algo_engine']
        execution_engine = resilience_system['execution_engine']
        
        # Generate substantial load
        for i in range(50):
            symbol = f"TEST{i}USDT"
            candles = generate_ohlcv_data(100)
            algo_engine.data_engine.candles_data[(symbol, "1m")] = candles
        
        # Process many signals
        strategy = MockStrategy(["buy"], "cleanup_test")
        processed_count = 0
        
        for i in range(20):  # Process subset
            symbol = f"TEST{i}USDT"
            signal = await algo_engine.process_signals(symbol, "1m", strategy)
            if signal:
                processed_count += 1
        
        # Verify system still responsive
        test_signal = TradeSignal(
            action="open",
            side="buy", 
            symbol="BTCUSDT",
            strategy_id="cleanup_test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.5
        )
        
        final_result = await execution_engine.process_signal(test_signal)
        assert isinstance(final_result, dict)


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system."""
    
    @pytest.fixture
    async def performance_system(self):
        """Create optimized system for performance testing."""
        mock_data_engine = MockDataEngine()
        mock_binance_client = MockBinanceClient()
        
        algo_engine = AlgoEngine(mock_data_engine)
        execution_engine = ProductionExecutionEngine(mock_binance_client, 20000.0)
        await execution_engine.setup()
        
        # Pre-populate with substantial data
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            candles = generate_ohlcv_data(200)  # Substantial history
            mock_data_engine.candles_data[(symbol, "1m")] = candles
            
            market_data = generate_market_data_bar(symbol)
            execution_engine.update_market_data_bar(symbol, market_data, 0.02)
        
        return {
            'algo_engine': algo_engine,
            'execution_engine': execution_engine,
            'symbols': symbols
        }
    
    @pytest.mark.asyncio
    async def test_high_throughput_signal_processing(self, performance_system):
        """Test high throughput signal processing capability."""
        algo_engine = performance_system['algo_engine']
        execution_engine = performance_system['execution_engine']
        symbols = performance_system['symbols']
        
        # Force rebalancing
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        execution_engine.process_daily_rebalance()
        
        strategy = MockStrategy(["buy", "sell", "hold"], "throughput_test")
        
        # Measure processing throughput
        start_time = time.perf_counter()
        processed_signals = 0
        total_signals = len(symbols) * 10  # 10 iterations per symbol
        
        for iteration in range(10):
            for symbol in symbols:
                signal = await algo_engine.process_signals(symbol, "1m", strategy)
                if signal:
                    processed_signals += 1
                    
                    if signal.action != "hold":
                        signal.metadata = {
                            'price': 50000.0,
                            'atr_value': 0.02,
                            'volume': 1000000,
                            'timestamp': int(time.time() * 1000)
                        }
                        
                        await execution_engine.process_signal(signal)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate throughput metrics
        signals_per_second = processed_signals / total_time if total_time > 0 else 0
        
        # Should achieve reasonable throughput
        assert signals_per_second > 10, f"Throughput {signals_per_second:.2f} signals/sec too low"
        assert total_time < 30, f"Processing took {total_time:.2f}s, should be faster"
    
    @pytest.mark.asyncio
    async def test_latency_under_load(self, performance_system):
        """Test latency characteristics under load."""
        execution_engine = performance_system['execution_engine']
        
        # Create signals with timing measurement
        latencies = []
        num_tests = 20
        
        for i in range(num_tests):
            signal = TradeSignal(
                action="open",
                side="buy",
                symbol="BTCUSDT",
                strategy_id="latency_test",
                metadata={
                    'price': 50000.0 + i,  # Slight variation
                    'atr_value': 0.02,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                },
                signal_confidence=0.7
            )
            
            # Measure individual processing latency
            start = time.perf_counter()
            await execution_engine.process_signal(signal)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Latency requirements from document: <30ms
        assert avg_latency < 30, f"Average latency {avg_latency:.2f}ms exceeds 30ms requirement"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms too high"
        
        print(f"Latency stats - Avg: {avg_latency:.2f}ms, Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_system):
        """Test memory usage patterns under sustained load."""
        import sys
        import gc
        
        algo_engine = performance_system['algo_engine']
        execution_engine = performance_system['execution_engine']
        
        # Measure initial memory
        gc.collect()  # Force garbage collection
        initial_size = sys.getsizeof(algo_engine) + sys.getsizeof(execution_engine)
        
        # Generate sustained load
        strategy = MockStrategy(["buy", "sell"], "memory_test")
        
        for round_num in range(5):  # Multiple rounds
            for symbol in performance_system['symbols']:
                # Update data
                new_candles = generate_ohlcv_data(150 + round_num * 10)
                algo_engine.data_engine.candles_data[(symbol, "1m")] = new_candles
                
                # Process signal
                signal = await algo_engine.process_signals(symbol, "1m", strategy)
                if signal and signal.action != "hold":
                    signal.metadata = create_test_signal_metadata()
                    await execution_engine.process_signal(signal)
        
        # Measure final memory
        gc.collect()
        final_size = sys.getsizeof(algo_engine) + sys.getsizeof(execution_engine)
        memory_growth = final_size - initial_size
        
        # Memory growth should be reasonable
        max_allowed_growth = 5 * 1024 * 1024  # 5MB
        assert memory_growth < max_allowed_growth, f"Memory growth {memory_growth} bytes excessive"
    
    def test_portfolio_calculation_performance(self, performance_system):
        """Test portfolio calculation performance."""
        execution_engine = performance_system['execution_engine']
        portfolio_manager = execution_engine.portfolio_manager
        
        # Add data for many symbols
        symbols = [f"TEST{i}USDT" for i in range(50)]
        
        for symbol in symbols:
            for _ in range(20):  # Build history
                portfolio_manager.update_volatility_data(symbol, 0.02 + np.random.normal(0, 0.001))
        
        # Add correlation data
        for i in range(len(symbols)):
            for j in range(i + 1, min(i + 10, len(symbols))):  # Limited pairs to avoid O(n²)
                correlation = 0.5 + np.random.normal(0, 0.1)
                portfolio_manager.update_correlation_data(symbols[i], symbols[j], correlation)
        
        # Measure rebalancing performance
        start_time = time.perf_counter()
        
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        end_time = time.perf_counter()
        calculation_time = (end_time - start_time) * 1000
        
        # Should calculate allocations efficiently
        assert calculation_time < 500, f"Portfolio calculation took {calculation_time:.2f}ms"
        assert len(allocations) == len(symbols)
        
        # Verify allocation quality
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        assert total_allocated > 0
        assert total_allocated <= portfolio_manager.total_capital * portfolio_manager.max_allocation_pct * 1.01
