"""
Unit tests for OrderExecutor module.

This test suite covers:
1. Initialization and configuration
2. Order execution workflow and validation
3. Position opening with risk parameters
4. Position closing and management
5. Stop loss and take profit order handling
6. Leverage and margin management
7. Order status tracking and monitoring
8. Error handling and recovery
9. Integration with portfolio and risk managers
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
import asyncio
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from execution.executor import OrderExecutor


class MockBinanceClient:
    """Mock Binance client for testing."""
    
    def __init__(self):
        self.positions = {}
        self.orders = {}
        self.order_id_counter = 1000
        
    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        """Mock order creation."""
        order_id = str(self.order_id_counter)
        self.order_id_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'params': params
        }
        
        self.orders[order_id] = order
        return order
        
    async def get_open_positions(self, symbol=None):
        """Mock position retrieval."""
        if symbol:
            return [self.positions.get(symbol, {})] if symbol in self.positions else []
        return list(self.positions.values())
        
    async def close_position(self, symbol, side=None):
        """Mock position closure."""
        if symbol in self.positions:
            del self.positions[symbol]
            return {'status': 'closed', 'symbol': symbol}
        return {'status': 'not_found', 'symbol': symbol}
        
    async def cancel_order(self, order_id, symbol):
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
            return {'status': 'canceled', 'id': order_id}
        return {'status': 'not_found', 'id': order_id}
        
    async def set_leverage(self, symbol, leverage):
        """Mock leverage setting."""
        return {'symbol': symbol, 'leverage': leverage}
        
    async def set_margin_type(self, symbol, margin_type):
        """Mock margin type setting."""
        return {'symbol': symbol, 'margin_type': margin_type}


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self):
        self.allocations = {}
        self.reservations = {}
        
    def get_allocated_capital(self, symbol):
        return self.allocations.get(symbol, 2000.0)
        
    def get_allocation_percentage(self, symbol):
        return 0.2  # 20%
        
    def reserve_allocation(self, symbol, amount):
        self.reservations[symbol] = self.reservations.get(symbol, 0) + amount
        return True
        
    def release_allocation(self, symbol, amount):
        if symbol in self.reservations:
            self.reservations[symbol] = max(0, self.reservations[symbol] - amount)
        return True


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def __init__(self):
        self.risk_limits_passed = True
        
    def validate_position_size(self, symbol, position_data):
        return self.risk_limits_passed, "Position validated" if self.risk_limits_passed else "Risk limits exceeded"
        
    def calculate_position_size(self, symbol, allocated_capital, atr_value, entry_price, volatility_norm=0.5):
        return {
            'position_size_usdt': 1000.0,
            'position_size_contracts': 0.025,
            'margin_required': 200.0,
            'leverage': 5,
            'atr_adjusted': atr_value,
            'dynamic_cost': 0.0014,
            'risk_amount': 16.0
        }
        
    def calculate_stop_loss_take_profit(self, entry_price, atr_value, side):
        if side == "long":
            return {
                'stop_loss': entry_price * 0.98,
                'take_profit': entry_price * 1.04,
                'risk_amount': entry_price * 0.02,
                'reward_amount': entry_price * 0.04,
                'reward_risk_ratio': 2.0
            }
        else:
            return {
                'stop_loss': entry_price * 1.02,
                'take_profit': entry_price * 0.96,
                'risk_amount': entry_price * 0.02,
                'reward_amount': entry_price * 0.04,
                'reward_risk_ratio': 2.0
            }
            
    def check_risk_limits(self, symbol):
        return self.risk_limits_passed, [] if self.risk_limits_passed else ["Risk limit exceeded"]


class TestOrderExecutor(unittest.TestCase):
    """Test suite for OrderExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.binance_client = MockBinanceClient()
        self.portfolio_manager = MockPortfolioManager()
        self.risk_manager = MockRiskManager()
        
        self.executor = OrderExecutor(
            self.binance_client,
            self.portfolio_manager,
            self.risk_manager
        )
        
    def test_initialization(self):
        """Test order executor initialization."""
        self.assertEqual(self.executor.binance_client, self.binance_client)
        self.assertEqual(self.executor.portfolio_manager, self.portfolio_manager)
        self.assertEqual(self.executor.risk_manager, self.risk_manager)
        
        # Check default values
        self.assertEqual(self.executor.default_leverage, 5)
        self.assertEqual(self.executor.default_margin_type, "isolated")
        self.assertIsInstance(self.executor.execution_history, list)
        
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_execute_open_position_success(self, mock_sleep):
        """Test successful position opening."""
        symbol = "BTCUSDT"
        side = "buy"
        position_size = 0.025
        current_price = 42000.0
        stop_loss_price = 41000.0
        take_profit_price = 44000.0
        leverage = 5
        
        result = await self.executor.execute_open_position(
            symbol=symbol,
            side=side,
            position_size=position_size,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            leverage=leverage
        )
        
        # Check successful execution
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['symbol'], symbol)
        self.assertIn('main_order', result)
        self.assertIn('stop_loss_order', result)
        self.assertIn('take_profit_order', result)
        
        # Verify allocation was reserved
        self.assertGreater(self.portfolio_manager.reservations.get(symbol, 0), 0)
        
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_execute_open_position_allocation_failed(self, mock_sleep):
        """Test position opening with allocation failure."""
        symbol = "BTCUSDT"
        
        # Mock allocation failure
        self.portfolio_manager.reserve_allocation = Mock(return_value=False)
        
        result = await self.executor.execute_open_position(
            symbol=symbol,
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        # Should reject due to allocation failure
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('allocation', result['reason'].lower())
        
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_execute_open_position_with_defaults(self, mock_sleep):
        """Test position opening with default parameters."""
        symbol = "BTCUSDT"
        
        result = await self.executor.execute_open_position(
            symbol=symbol,
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        # Should use default leverage and margin type
        self.assertEqual(result['status'], 'success')
        # Check that leverage was set (would be in binance client calls)
        
    async def test_execute_open_position_order_failure(self):
        """Test position opening with order creation failure."""
        symbol = "BTCUSDT"
        
        # Mock order creation failure
        self.binance_client.create_order = AsyncMock(side_effect=Exception("Order failed"))
        
        result = await self.executor.execute_open_position(
            symbol=symbol,
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        # Should handle error gracefully
        self.assertEqual(result['status'], 'error')
        self.assertIn('order failed', result['reason'].lower())
        
    async def test_execute_close_position_success(self):
        """Test successful position closing."""
        symbol = "BTCUSDT"
        
        # Add mock position
        self.binance_client.positions[symbol] = {
            'symbol': symbol,
            'contracts': 0.025,
            'side': 'long'
        }
        
        result = await self.executor.execute_close_position(symbol)
        
        # Check successful closure
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['symbol'], symbol)
        
        # Position should be removed
        self.assertNotIn(symbol, self.binance_client.positions)
        
    async def test_execute_close_position_not_found(self):
        """Test closing non-existent position."""
        symbol = "BTCUSDT"
        
        result = await self.executor.execute_close_position(symbol)
        
        # Should handle gracefully
        self.assertEqual(result['status'], 'not_found')
        
    async def test_execute_close_position_with_order_cancellation(self):
        """Test position closing with order cancellation."""
        symbol = "BTCUSDT"
        
        # Add mock position and orders
        self.binance_client.positions[symbol] = {'symbol': symbol, 'contracts': 0.025}
        self.binance_client.orders['1001'] = {
            'id': '1001',
            'symbol': symbol,
            'type': 'STOP',
            'status': 'open'
        }
        self.binance_client.orders['1002'] = {
            'id': '1002',
            'symbol': symbol,
            'type': 'TAKE_PROFIT',
            'status': 'open'
        }
        
        # Mock get_open_orders
        self.binance_client.get_open_orders = AsyncMock(return_value=[
            self.binance_client.orders['1001'],
            self.binance_client.orders['1002']
        ])
        
        result = await self.executor.execute_close_position(symbol)
        
        # Should cancel orders and close position
        self.assertEqual(result['status'], 'success')
        self.assertEqual(self.binance_client.orders['1001']['status'], 'canceled')
        self.assertEqual(self.binance_client.orders['1002']['status'], 'canceled')
        
    def test_calculate_notional_and_margin(self):
        """Test notional value and margin calculations."""
        position_size = 0.025
        current_price = 42000.0
        leverage = 5
        
        notional, margin = self.executor.calculate_notional_and_margin(
            position_size, current_price, leverage
        )
        
        expected_notional = position_size * current_price
        expected_margin = expected_notional / leverage
        
        self.assertAlmostEqual(notional, expected_notional, places=2)
        self.assertAlmostEqual(margin, expected_margin, places=2)
        
    def test_calculate_notional_and_margin_no_leverage(self):
        """Test calculations without leverage."""
        position_size = 0.025
        current_price = 42000.0
        
        notional, margin = self.executor.calculate_notional_and_margin(
            position_size, current_price, None
        )
        
        expected_notional = position_size * current_price
        
        self.assertAlmostEqual(notional, expected_notional, places=2)
        self.assertAlmostEqual(margin, expected_notional, places=2)  # No leverage = full margin
        
    def test_validate_order_parameters(self):
        """Test order parameter validation."""
        # Valid parameters
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="BTCUSDT",
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Parameters valid")
        
        # Invalid symbol
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="",
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("symbol", reason.lower())
        
        # Invalid side
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="BTCUSDT",
            side="invalid",
            position_size=0.025,
            current_price=42000.0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("side", reason.lower())
        
        # Invalid position size
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="BTCUSDT",
            side="buy",
            position_size=0,
            current_price=42000.0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("position size", reason.lower())
        
        # Invalid price
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="BTCUSDT",
            side="buy",
            position_size=0.025,
            current_price=0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("price", reason.lower())
        
    async def test_setup_leverage_and_margin(self):
        """Test leverage and margin setup."""
        symbol = "BTCUSDT"
        leverage = 10
        margin_type = "cross"
        
        result = await self.executor.setup_leverage_and_margin(symbol, leverage, margin_type)
        
        # Should complete successfully
        self.assertTrue(result)
        
    async def test_setup_leverage_and_margin_failure(self):
        """Test leverage setup failure handling."""
        symbol = "BTCUSDT"
        
        # Mock leverage setting failure
        self.binance_client.set_leverage = AsyncMock(side_effect=Exception("Leverage failed"))
        
        result = await self.executor.setup_leverage_and_margin(symbol, 10, "isolated")
        
        # Should handle error gracefully
        self.assertFalse(result)
        
    def test_create_stop_loss_order_params(self):
        """Test stop loss order parameter creation."""
        side = "buy"
        stop_price = 41000.0
        position_size = 0.025
        
        params = self.executor.create_stop_loss_order_params(side, stop_price, position_size)
        
        # Check required parameters
        self.assertIn('stopPrice', params)
        self.assertIn('closePosition', params)
        self.assertEqual(params['stopPrice'], stop_price)
        
        # For buy position, stop loss should be sell
        self.assertEqual(params.get('side'), 'SELL')
        
    def test_create_take_profit_order_params(self):
        """Test take profit order parameter creation."""
        side = "buy"
        tp_price = 44000.0
        position_size = 0.025
        
        params = self.executor.create_take_profit_order_params(side, tp_price, position_size)
        
        # Check required parameters
        self.assertIn('stopPrice', params)
        self.assertIn('closePosition', params)
        self.assertEqual(params['stopPrice'], tp_price)
        
        # For buy position, take profit should be sell
        self.assertEqual(params.get('side'), 'SELL')
        
    def test_log_execution_details(self):
        """Test execution detail logging."""
        execution_data = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'position_size': 0.025,
            'current_price': 42000.0,
            'leverage': 5
        }
        
        # Should not raise exception
        try:
            self.executor.log_execution_details(execution_data)
        except Exception as e:
            self.fail(f"log_execution_details raised {e}")
            
    def test_add_to_execution_history(self):
        """Test execution history tracking."""
        initial_length = len(self.executor.execution_history)
        
        execution_record = {
            'timestamp': '2024-01-01T00:00:00Z',
            'symbol': 'BTCUSDT',
            'action': 'open_position',
            'status': 'success'
        }
        
        self.executor.add_to_execution_history(execution_record)
        
        # Should add to history
        self.assertEqual(len(self.executor.execution_history), initial_length + 1)
        self.assertEqual(self.executor.execution_history[-1], execution_record)
        
    def test_get_execution_history(self):
        """Test execution history retrieval."""
        # Add some history
        records = [
            {'symbol': 'BTCUSDT', 'action': 'open', 'status': 'success'},
            {'symbol': 'ETHUSDT', 'action': 'close', 'status': 'success'}
        ]
        
        for record in records:
            self.executor.add_to_execution_history(record)
            
        # Get all history
        history = self.executor.get_execution_history()
        self.assertGreaterEqual(len(history), len(records))
        
        # Get filtered history
        btc_history = self.executor.get_execution_history(symbol='BTCUSDT')
        btc_records = [r for r in btc_history if r['symbol'] == 'BTCUSDT']
        self.assertEqual(len(btc_records), len(btc_history))
        
        # Get limited history
        limited_history = self.executor.get_execution_history(limit=1)
        self.assertEqual(len(limited_history), 1)
        
    async def test_monitor_order_execution(self):
        """Test order execution monitoring."""
        symbol = "BTCUSDT"
        order_id = "1001"
        
        # Mock order that becomes filled
        self.binance_client.orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'status': 'filled'
        }
        
        # Mock get_order method
        async def mock_get_order(order_id, symbol):
            return self.binance_client.orders.get(order_id)
            
        self.binance_client.get_order = mock_get_order
        
        status = await self.executor.monitor_order_execution(order_id, symbol, timeout=1)
        
        # Should detect filled status
        self.assertEqual(status['status'], 'filled')
        
    async def test_monitor_order_execution_timeout(self):
        """Test order monitoring timeout."""
        symbol = "BTCUSDT"
        order_id = "1001"
        
        # Mock order that stays open
        self.binance_client.orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'status': 'open'
        }
        
        async def mock_get_order(order_id, symbol):
            return self.binance_client.orders.get(order_id)
            
        self.binance_client.get_order = mock_get_order
        
        status = await self.executor.monitor_order_execution(order_id, symbol, timeout=0.1)
        
        # Should timeout
        self.assertEqual(status['status'], 'timeout')
        
    def test_calculate_order_size_precision(self):
        """Test order size precision calculation."""
        # Test with standard precision
        size = 0.12345678
        precision = 6
        
        adjusted_size = self.executor.calculate_order_size_precision(size, precision)
        self.assertEqual(adjusted_size, 0.123456)
        
        # Test with zero precision (integer)
        size = 123.456
        precision = 0
        
        adjusted_size = self.executor.calculate_order_size_precision(size, precision)
        self.assertEqual(adjusted_size, 123.0)
        
    def test_validate_stop_loss_take_profit_prices(self):
        """Test stop loss and take profit price validation."""
        current_price = 42000.0
        
        # Valid long position prices
        is_valid, reason = self.executor.validate_stop_loss_take_profit_prices(
            current_price=current_price,
            stop_loss_price=41000.0,
            take_profit_price=44000.0,
            side="buy"
        )
        
        self.assertTrue(is_valid)
        
        # Invalid long position - SL above current price
        is_valid, reason = self.executor.validate_stop_loss_take_profit_prices(
            current_price=current_price,
            stop_loss_price=43000.0,
            take_profit_price=44000.0,
            side="buy"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("stop loss", reason.lower())
        
        # Valid short position prices
        is_valid, reason = self.executor.validate_stop_loss_take_profit_prices(
            current_price=current_price,
            stop_loss_price=43000.0,
            take_profit_price=40000.0,
            side="sell"
        )
        
        self.assertTrue(is_valid)
        
        # Invalid short position - TP above current price
        is_valid, reason = self.executor.validate_stop_loss_take_profit_prices(
            current_price=current_price,
            stop_loss_price=43000.0,
            take_profit_price=44000.0,
            side="sell"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("take profit", reason.lower())
        
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test with None values
        is_valid, reason = self.executor.validate_order_parameters(
            symbol=None,
            side="buy",
            position_size=0.025,
            current_price=42000.0
        )
        
        self.assertFalse(is_valid)
        
        # Test with negative values
        is_valid, reason = self.executor.validate_order_parameters(
            symbol="BTCUSDT",
            side="buy",
            position_size=-0.025,
            current_price=42000.0
        )
        
        self.assertFalse(is_valid)
        
    async def test_cleanup_failed_orders(self):
        """Test cleanup of failed orders."""
        symbol = "BTCUSDT"
        
        # Add some mock orders
        order_ids = ["1001", "1002", "1003"]
        for order_id in order_ids:
            self.binance_client.orders[order_id] = {
                'id': order_id,
                'symbol': symbol,
                'status': 'open'
            }
            
        # Mock get_open_orders
        self.binance_client.get_open_orders = AsyncMock(return_value=[
            self.binance_client.orders[oid] for oid in order_ids
        ])
        
        await self.executor.cleanup_failed_orders(symbol, order_ids)
        
        # All orders should be cancelled
        for order_id in order_ids:
            self.assertEqual(self.binance_client.orders[order_id]['status'], 'canceled')
            
    def test_get_execution_statistics(self):
        """Test execution statistics generation."""
        # Add some execution history
        executions = [
            {'status': 'success', 'action': 'open'},
            {'status': 'success', 'action': 'close'},
            {'status': 'error', 'action': 'open'},
            {'status': 'success', 'action': 'open'}
        ]
        
        for execution in executions:
            self.executor.add_to_execution_history(execution)
            
        stats = self.executor.get_execution_statistics()
        
        # Check statistics
        self.assertIn('total_executions', stats)
        self.assertIn('successful_executions', stats)
        self.assertIn('failed_executions', stats)
        self.assertIn('success_rate', stats)
        
        self.assertEqual(stats['total_executions'], len(executions))
        self.assertEqual(stats['successful_executions'], 3)
        self.assertEqual(stats['failed_executions'], 1)
        self.assertEqual(stats['success_rate'], 0.75)


if __name__ == "__main__":
    unittest.main()
