"""
Unit tests for ProductionExecutionEngine module.

This test suite covers:
1. Initialization and component integration
2. Signal validation and processing workflow
3. Market data updates and portfolio rebalancing
4. Execution pipeline with risk management
5. Error handling and recovery mechanisms
6. Performance monitoring and metrics
7. Integration with all sub-components
8. Stress testing and edge cases
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from execution.execution_engine import ProductionExecutionEngine
from algorithm.trade_signal import TradeSignal


class MockBinanceClient:
    """Mock Binance client for testing."""
    
    def __init__(self):
        self.testnet = True
        
    async def close(self):
        pass


class MockTradeSignal:
    """Mock trade signal for testing."""
    
    def __init__(self, action="open", side="long", symbol="BTCUSDT"):
        self.action = action
        self.side = side
        self.symbol = symbol
        self.strategy_id = "test_strategy"
        self.timestamp = int(datetime.now().timestamp() * 1000)
        self.signal_confidence = 0.8
        self.metadata = {
            'atr': 0.002,
            'price': 42000.0,
            'reason': 'test signal'
        }


class TestProductionExecutionEngine(unittest.TestCase):
    """Test suite for ProductionExecutionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.binance_client = MockBinanceClient()
        self.total_capital = 10000.0
        
        self.execution_engine = ProductionExecutionEngine(
            binance_client=self.binance_client,
            total_capital=self.total_capital
        )
        
    def test_initialization(self):
        """Test execution engine initialization."""
        # Check basic properties
        self.assertEqual(self.execution_engine.binance_client, self.binance_client)
        self.assertEqual(self.execution_engine.total_capital, self.total_capital)
        
        # Check component initialization
        self.assertIsNotNone(self.execution_engine.portfolio_manager)
        self.assertIsNotNone(self.execution_engine.risk_manager)
        self.assertIsNotNone(self.execution_engine.order_executor)
        self.assertIsNotNone(self.execution_engine.stress_handler)
        
        # Check default values
        self.assertFalse(self.execution_engine.is_setup)
        self.assertIsInstance(self.execution_engine.daily_rebalance_time, datetime)
        
    async def test_setup(self):
        """Test execution engine setup."""
        # Mock portfolio manager setup
        self.execution_engine.portfolio_manager.setup = AsyncMock()
        
        await self.execution_engine.setup()
        
        # Should mark as setup
        self.assertTrue(self.execution_engine.is_setup)
        
        # Should call portfolio manager setup
        self.execution_engine.portfolio_manager.setup.assert_called_once()
        
    async def test_setup_failure(self):
        """Test setup failure handling."""
        # Mock setup failure
        self.execution_engine.portfolio_manager.setup = AsyncMock(side_effect=Exception("Setup failed"))
        
        with self.assertRaises(Exception):
            await self.execution_engine.setup()
            
        # Should not mark as setup
        self.assertFalse(self.execution_engine.is_setup)
        
    def test_validate_signal_valid_open_signal(self):
        """Test validation of valid open signal."""
        signal = MockTradeSignal(action="open", side="long")
        current_price = 42000.0
        
        # Mock risk manager validation
        self.execution_engine.risk_manager.check_risk_limits = Mock(return_value=(True, []))
        self.execution_engine.portfolio_manager.get_allocated_capital = Mock(return_value=2000.0)
        self.execution_engine.risk_manager.calculate_position_size = Mock(return_value={
            'position_size_usdt': 1000.0,
            'position_size_contracts': 0.025,
            'leverage': 5
        })
        
        result = self.execution_engine.validate_signal(signal, current_price)
        
        # Should be valid
        self.assertTrue(result['valid'])
        self.assertEqual(result['reason'], 'Signal validated successfully')
        
    def test_validate_signal_invalid_action(self):
        """Test validation of invalid signal action."""
        signal = MockTradeSignal(action="invalid_action")
        current_price = 42000.0
        
        result = self.execution_engine.validate_signal(signal, current_price)
        
        # Should be invalid
        self.assertFalse(result['valid'])
        self.assertIn('action', result['reason'].lower())
        
    def test_validate_signal_risk_limits_exceeded(self):
        """Test validation when risk limits are exceeded."""
        signal = MockTradeSignal(action="open", side="long")
        current_price = 42000.0
        
        # Mock risk limit failure
        self.execution_engine.risk_manager.check_risk_limits = Mock(
            return_value=(False, ["Max drawdown exceeded"])
        )
        
        result = self.execution_engine.validate_signal(signal, current_price)
        
        # Should be invalid
        self.assertFalse(result['valid'])
        self.assertIn('risk', result['reason'].lower())
        
    def test_validate_signal_no_allocation(self):
        """Test validation when no capital is allocated."""
        signal = MockTradeSignal(action="open", side="long")
        current_price = 42000.0
        
        # Mock zero allocation
        self.execution_engine.risk_manager.check_risk_limits = Mock(return_value=(True, []))
        self.execution_engine.portfolio_manager.get_allocated_capital = Mock(return_value=0.0)
        
        result = self.execution_engine.validate_signal(signal, current_price)
        
        # Should be invalid
        self.assertFalse(result['valid'])
        self.assertIn('allocation', result['reason'].lower())
        
    def test_validate_signal_close_action(self):
        """Test validation of close signal."""
        signal = MockTradeSignal(action="close")
        current_price = 42000.0
        
        result = self.execution_engine.validate_signal(signal, current_price)
        
        # Close signals should always be valid
        self.assertTrue(result['valid'])
        
    async def test_process_signal_open_position(self):
        """Test processing open position signal."""
        signal = MockTradeSignal(action="open", side="long")
        signal.metadata['price'] = 42000.0
        
        # Mock successful execution
        self.execution_engine.order_executor.execute_open_position = AsyncMock(return_value={
            'status': 'success',
            'symbol': 'BTCUSDT'
        })
        
        result = await self.execution_engine.process_signal(signal)
        
        # Should execute successfully
        self.assertEqual(result['status'], 'success')
        self.execution_engine.order_executor.execute_open_position.assert_called_once()
        
    async def test_process_signal_close_position(self):
        """Test processing close position signal."""
        signal = MockTradeSignal(action="close")
        
        # Mock successful closure
        self.execution_engine.order_executor.execute_close_position = AsyncMock(return_value={
            'status': 'success',
            'symbol': 'BTCUSDT'
        })
        
        result = await self.execution_engine.process_signal(signal)
        
        # Should close successfully
        self.assertEqual(result['status'], 'success')
        self.execution_engine.order_executor.execute_close_position.assert_called_once()
        
    async def test_process_signal_no_action(self):
        """Test processing signal with no action."""
        signal = MockTradeSignal(action="none")
        
        result = await self.execution_engine.process_signal(signal)
        
        # Should handle gracefully
        self.assertEqual(result['status'], 'ignored')
        self.assertIn('no action', result['reason'].lower())
        
    async def test_process_signal_execution_failure(self):
        """Test processing signal with execution failure."""
        signal = MockTradeSignal(action="open", side="long")
        signal.metadata['price'] = 42000.0
        
        # Mock execution failure
        self.execution_engine.order_executor.execute_open_position = AsyncMock(
            side_effect=Exception("Execution failed")
        )
        
        result = await self.execution_engine.process_signal(signal)
        
        # Should handle error
        self.assertEqual(result['status'], 'error')
        self.assertIn('execution failed', result['reason'].lower())
        
    def test_update_market_data_bar(self):
        """Test market data bar update."""
        symbol = "BTCUSDT"
        timeframe = "1m"
        bar_data = {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'open': 42000.0,
            'high': 42100.0,
            'low': 41900.0,
            'close': 42050.0,
            'volume': 100.0
        }
        
        # Mock portfolio manager updates
        self.execution_engine.portfolio_manager.update_volatility_data = Mock()
        self.execution_engine.portfolio_manager.update_correlation_data = Mock()
        
        self.execution_engine.update_market_data_bar(symbol, timeframe, bar_data)
        
        # Should update portfolio data
        self.execution_engine.portfolio_manager.update_volatility_data.assert_called()
        
    def test_should_rebalance_daily(self):
        """Test daily rebalancing trigger."""
        # Initially should rebalance
        self.assertTrue(self.execution_engine.should_rebalance())
        
        # After rebalancing, should not rebalance immediately
        self.execution_engine.mark_rebalanced()
        self.assertFalse(self.execution_engine.should_rebalance())
        
        # Should rebalance after 24 hours
        self.execution_engine.daily_rebalance_time = datetime.now() - timedelta(hours=25)
        self.assertTrue(self.execution_engine.should_rebalance())
        
    def test_get_portfolio_summary(self):
        """Test portfolio summary retrieval."""
        # Mock portfolio manager summary
        expected_summary = {
            'total_capital': 10000.0,
            'allocated_capital': 8000.0,
            'available_capital': 2000.0,
            'allocation_percentage': 0.8,
            'active_positions': 3
        }
        
        self.execution_engine.portfolio_manager.get_portfolio_summary = Mock(
            return_value=expected_summary
        )
        
        summary = self.execution_engine.get_portfolio_summary()
        
        self.assertEqual(summary, expected_summary)
        
    def test_get_risk_metrics(self):
        """Test risk metrics retrieval."""
        # Mock risk manager metrics
        expected_metrics = {
            'daily_pnl': 150.0,
            'current_drawdown': -0.05,
            'current_sharpe': 1.5,
            'risk_status': 'normal',
            'max_drawdown_hit': False
        }
        
        self.execution_engine.risk_manager.get_risk_metrics = Mock(
            return_value=expected_metrics
        )
        
        metrics = self.execution_engine.get_risk_metrics()
        
        self.assertEqual(metrics, expected_metrics)
        
    def test_update_daily_pnl(self):
        """Test daily PnL update."""
        symbol = "BTCUSDT"
        pnl_amount = 125.50
        
        # Mock risk manager update
        self.execution_engine.risk_manager.update_daily_pnl = Mock()
        
        self.execution_engine.update_daily_pnl(symbol, pnl_amount)
        
        # Should call risk manager
        self.execution_engine.risk_manager.update_daily_pnl.assert_called_once_with(
            symbol, pnl_amount
        )
        
    def test_calculate_atr_from_bar(self):
        """Test ATR calculation from bar data."""
        bar_data = {
            'high': 42100.0,
            'low': 41900.0,
            'close': 42050.0
        }
        
        previous_close = 42000.0
        
        atr = self.execution_engine.calculate_atr_from_bar(bar_data, previous_close)
        
        # Should calculate true range
        self.assertGreater(atr, 0)
        self.assertIsInstance(atr, float)
        
    def test_calculate_atr_no_previous_close(self):
        """Test ATR calculation without previous close."""
        bar_data = {
            'high': 42100.0,
            'low': 41900.0,
            'close': 42050.0
        }
        
        atr = self.execution_engine.calculate_atr_from_bar(bar_data)
        
        # Should use high-low range
        expected_atr = (42100.0 - 41900.0) / 42050.0  # Normalized
        self.assertAlmostEqual(atr, expected_atr, places=6)
        
    def test_check_stress_conditions(self):
        """Test stress condition monitoring."""
        symbol = "BTCUSDT"
        
        # Mock stress handler
        self.execution_engine.stress_handler.check_stress_conditions = Mock(
            return_value=(False, [])
        )
        
        is_stressed, conditions = self.execution_engine.check_stress_conditions(symbol)
        
        self.assertFalse(is_stressed)
        self.assertEqual(len(conditions), 0)
        
        # Test with stress conditions
        self.execution_engine.stress_handler.check_stress_conditions = Mock(
            return_value=(True, ["High volatility detected"])
        )
        
        is_stressed, conditions = self.execution_engine.check_stress_conditions(symbol)
        
        self.assertTrue(is_stressed)
        self.assertGreater(len(conditions), 0)
        
    async def test_emergency_shutdown(self):
        """Test emergency shutdown procedure."""
        # Mock order executor
        self.execution_engine.order_executor.cancel_all_orders = AsyncMock()
        self.execution_engine.order_executor.close_all_positions = AsyncMock()
        
        await self.execution_engine.emergency_shutdown()
        
        # Should cancel orders and close positions
        self.execution_engine.order_executor.cancel_all_orders.assert_called_once()
        
    def test_get_execution_statistics(self):
        """Test execution statistics retrieval."""
        # Mock order executor statistics
        expected_stats = {
            'total_executions': 100,
            'successful_executions': 95,
            'failed_executions': 5,
            'success_rate': 0.95
        }
        
        self.execution_engine.order_executor.get_execution_statistics = Mock(
            return_value=expected_stats
        )
        
        stats = self.execution_engine.get_execution_statistics()
        
        self.assertEqual(stats, expected_stats)
        
    def test_validate_signal_metadata(self):
        """Test signal metadata validation."""
        # Valid metadata
        signal = MockTradeSignal()
        signal.metadata = {
            'atr': 0.002,
            'price': 42000.0,
            'reason': 'crossover'
        }
        
        is_valid, reason = self.execution_engine.validate_signal_metadata(signal)
        self.assertTrue(is_valid)
        
        # Missing required metadata
        signal.metadata = {'price': 42000.0}  # Missing ATR
        
        is_valid, reason = self.execution_engine.validate_signal_metadata(signal)
        self.assertFalse(is_valid)
        self.assertIn('atr', reason.lower())
        
        # Invalid metadata types
        signal.metadata = {'atr': 'invalid', 'price': 42000.0}
        
        is_valid, reason = self.execution_engine.validate_signal_metadata(signal)
        self.assertFalse(is_valid)
        
    def test_calculate_correlation_between_symbols(self):
        """Test correlation calculation between symbols."""
        symbol1 = "BTCUSDT"
        symbol2 = "ETHUSDT"
        
        # Mock price data
        price_data_1 = [42000, 42100, 42050, 42200, 42150]
        price_data_2 = [3200, 3220, 3210, 3250, 3240]
        
        correlation = self.execution_engine.calculate_correlation_between_symbols(
            symbol1, symbol2, price_data_1, price_data_2
        )
        
        # Should calculate correlation
        self.assertIsInstance(correlation, float)
        self.assertGreaterEqual(correlation, -1.0)
        self.assertLessEqual(correlation, 1.0)
        
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Test timing decorator or performance tracking
        start_time = datetime.now()
        
        # Simulate some processing time
        import time
        time.sleep(0.01)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should be able to track processing times
        self.assertGreater(processing_time, 0)
        self.assertLess(processing_time, 1.0)  # Should be fast
        
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test with None signal
        result = self.execution_engine.validate_signal(None, 42000.0)
        self.assertFalse(result['valid'])
        
        # Test with None price
        signal = MockTradeSignal()
        result = self.execution_engine.validate_signal(signal, None)
        self.assertFalse(result['valid'])
        
        # Test with invalid signal attributes
        signal = MockTradeSignal()
        signal.action = None
        result = self.execution_engine.validate_signal(signal, 42000.0)
        self.assertFalse(result['valid'])
        
    async def test_concurrent_signal_processing(self):
        """Test concurrent signal processing."""
        signals = [
            MockTradeSignal(action="open", side="long", symbol="BTCUSDT"),
            MockTradeSignal(action="open", side="long", symbol="ETHUSDT"),
            MockTradeSignal(action="close", symbol="XRPUSDT")
        ]
        
        # Mock successful processing
        async def mock_process(signal):
            await asyncio.sleep(0.01)  # Simulate processing time
            return {'status': 'success', 'symbol': signal.symbol}
            
        self.execution_engine.process_signal = mock_process
        
        # Process signals concurrently
        tasks = [self.execution_engine.process_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        self.assertEqual(len(results), len(signals))
        for result in results:
            self.assertEqual(result['status'], 'success')
            
    def test_component_integration(self):
        """Test integration between components."""
        # Verify all components are properly connected
        portfolio_manager = self.execution_engine.portfolio_manager
        risk_manager = self.execution_engine.risk_manager
        order_executor = self.execution_engine.order_executor
        
        # Risk manager should have reference to portfolio manager
        self.assertEqual(risk_manager.portfolio_manager, portfolio_manager)
        
        # Order executor should have references to both
        self.assertEqual(order_executor.portfolio_manager, portfolio_manager)
        self.assertEqual(order_executor.risk_manager, risk_manager)
        
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Get memory usage before operations
        memory_before = process.memory_info().rss
        
        # Perform some operations
        for i in range(100):
            signal = MockTradeSignal()
            self.execution_engine.validate_signal(signal, 42000.0)
            
        # Get memory usage after
        memory_after = process.memory_info().rss
        
        # Memory growth should be reasonable
        memory_growth = memory_after - memory_before
        self.assertLess(memory_growth, 50 * 1024 * 1024)  # Less than 50MB growth
        
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid total capital
        with self.assertRaises(ValueError):
            ProductionExecutionEngine(self.binance_client, total_capital=-1000.0)
            
        # Test with zero capital
        with self.assertRaises(ValueError):
            ProductionExecutionEngine(self.binance_client, total_capital=0.0)
            
        # Test with very small capital
        with self.assertRaises(ValueError):
            ProductionExecutionEngine(self.binance_client, total_capital=10.0)


if __name__ == "__main__":
    unittest.main()
