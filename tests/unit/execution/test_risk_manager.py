"""
Unit tests for ProductionRiskManager module.

This test suite covers:
1. Initialization and risk parameter validation
2. Position sizing calculations with ATR and Kelly criterion
3. Dynamic cost adjustments based on volatility
4. Stop loss and take profit calculations
5. Leverage calculations and constraints
6. Risk monitoring and drawdown tracking
7. Sharpe ratio and equity curve management
8. Emergency risk controls and circuit breakers
9. Edge cases and error handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from execution.risk_manager import ProductionRiskManager, ProductionRiskParameters
from execution.portfolio import ProductionPortfolioManager


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self, total_capital=10000.0):
        self.total_capital = total_capital
        self.target_volatility = 0.18
        
    def get_allocated_capital(self, symbol):
        return 2000.0  # Mock allocated capital
        
    def get_allocation_percentage(self, symbol):
        return 0.2  # 20% allocation


class TestProductionRiskParameters(unittest.TestCase):
    """Test suite for ProductionRiskParameters."""
    
    def test_default_parameters(self):
        """Test default risk parameters match document specifications."""
        params = ProductionRiskParameters()
        
        # Core position sizing parameters
        self.assertEqual(params.risk_per_trade_pct, 0.008)  # 0.8%
        self.assertEqual(params.kelly_fraction, 0.7)  # 70% Kelly
        self.assertEqual(params.base_cost_pct, 0.0014)  # 0.14%
        self.assertEqual(params.min_atr_floor, 0.001)  # 0.1%
        
        # Stop loss and take profit parameters
        self.assertEqual(params.atr_stop_multiplier, 1.8)  # 1.8x ATR
        self.assertEqual(params.atr_trail_multiplier, 0.8)  # 0.8x ATR
        self.assertEqual(params.risk_reward_ratio, 2.0)  # 1:2 ratio
        self.assertEqual(params.partial_exit_ratio, 0.4)  # 40%
        
        # Dynamic leverage parameters
        self.assertEqual(params.max_leverage, 10)  # 10x max
        self.assertEqual(params.target_volatility, 0.18)  # 18%


class TestProductionRiskManager(unittest.TestCase):
    """Test suite for ProductionRiskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio_manager = MockPortfolioManager()
        self.risk_manager = ProductionRiskManager(self.portfolio_manager)
        
    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertIsInstance(self.risk_manager.risk_params, ProductionRiskParameters)
        self.assertEqual(self.risk_manager.portfolio_manager, self.portfolio_manager)
        
        # Check initialized data structures
        self.assertIsInstance(self.risk_manager.drawdown_history, list)
        self.assertIsInstance(self.risk_manager.sharpe_history, list)
        self.assertIsInstance(self.risk_manager.equity_curve, list)
        self.assertIsInstance(self.risk_manager.positions, dict)
        
        # Check initial values
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertFalse(self.risk_manager.max_drawdown_hit)
        
    def test_calculate_dynamic_cost_adjustment(self):
        """Test dynamic cost adjustment calculation."""
        base_cost = self.risk_manager.risk_params.base_cost_pct
        
        # Test with normal volatility
        volatility_norm = 0.5
        adjusted_cost = self.risk_manager.calculate_dynamic_cost_adjustment(volatility_norm)
        expected_cost = base_cost * (1 + 0.5 * volatility_norm)
        self.assertAlmostEqual(adjusted_cost, expected_cost, places=6)
        
        # Test with zero volatility
        adjusted_cost = self.risk_manager.calculate_dynamic_cost_adjustment(0.0)
        self.assertEqual(adjusted_cost, base_cost)
        
        # Test with high volatility
        volatility_norm = 2.0
        adjusted_cost = self.risk_manager.calculate_dynamic_cost_adjustment(volatility_norm)
        expected_cost = base_cost * (1 + 0.5 * 2.0)
        self.assertAlmostEqual(adjusted_cost, expected_cost, places=6)
        
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        symbol = "BTCUSDT"
        allocated_capital = 2000.0
        atr_value = 0.002
        entry_price = 42000.0
        
        result = self.risk_manager.calculate_position_size(
            symbol, allocated_capital, atr_value, entry_price
        )
        
        # Check result structure
        required_keys = [
            'position_size_usdt', 'position_size_contracts', 'margin_required',
            'atr_adjusted', 'dynamic_cost', 'risk_amount', 'leverage'
        ]
        for key in required_keys:
            self.assertIn(key, result)
            
        # Verify calculations
        expected_risk_amount = 0.008 * allocated_capital * 0.7  # 0.8% * allocated * 0.7 Kelly
        self.assertAlmostEqual(result['risk_amount'], expected_risk_amount, places=2)
        
        # Position size should be positive
        self.assertGreater(result['position_size_usdt'], 0)
        self.assertGreater(result['position_size_contracts'], 0)
        
    def test_calculate_position_size_with_atr_floor(self):
        """Test position sizing with ATR floor applied."""
        symbol = "BTCUSDT"
        allocated_capital = 2000.0
        atr_value = 0.0005  # Below ATR floor
        entry_price = 42000.0
        
        result = self.risk_manager.calculate_position_size(
            symbol, allocated_capital, atr_value, entry_price
        )
        
        # ATR should be adjusted to floor
        self.assertEqual(result['atr_adjusted'], self.risk_manager.risk_params.min_atr_floor)
        
        # Position size calculation should use adjusted ATR
        expected_risk = 0.008 * allocated_capital * 0.7
        expected_dynamic_cost = self.risk_manager.calculate_dynamic_cost_adjustment(0.5)
        expected_base_size = expected_risk / self.risk_manager.risk_params.min_atr_floor
        expected_position_size = expected_base_size * (1 - expected_dynamic_cost)
        
        self.assertAlmostEqual(result['position_size_usdt'], expected_position_size, places=2)
        
    def test_calculate_position_size_zero_allocated_capital(self):
        """Test position sizing with zero allocated capital."""
        symbol = "BTCUSDT"
        allocated_capital = 0.0
        atr_value = 0.002
        entry_price = 42000.0
        
        result = self.risk_manager.calculate_position_size(
            symbol, allocated_capital, atr_value, entry_price
        )
        
        # Should return zero position size
        self.assertEqual(result['position_size_usdt'], 0.0)
        self.assertEqual(result['position_size_contracts'], 0.0)
        self.assertEqual(result['risk_amount'], 0.0)
        
    def test_calculate_stop_loss_take_profit_long(self):
        """Test stop loss and take profit calculation for long positions."""
        entry_price = 42000.0
        atr_value = 0.002
        side = "long"
        
        sl_tp = self.risk_manager.calculate_stop_loss_take_profit(entry_price, atr_value, side)
        
        # Check structure
        self.assertIn('stop_loss', sl_tp)
        self.assertIn('take_profit', sl_tp)
        self.assertIn('risk_amount', sl_tp)
        self.assertIn('reward_amount', sl_tp)
        self.assertIn('reward_risk_ratio', sl_tp)
        
        # For long: SL below entry, TP above entry
        self.assertLess(sl_tp['stop_loss'], entry_price)
        self.assertGreater(sl_tp['take_profit'], entry_price)
        
        # Verify ATR multipliers
        expected_sl = entry_price - (atr_value * self.risk_manager.risk_params.atr_stop_multiplier)
        expected_tp = entry_price + (abs(entry_price - expected_sl) * self.risk_manager.risk_params.risk_reward_ratio)
        
        self.assertAlmostEqual(sl_tp['stop_loss'], expected_sl, places=2)
        self.assertAlmostEqual(sl_tp['take_profit'], expected_tp, places=2)
        
        # Check risk-reward ratio
        self.assertAlmostEqual(sl_tp['reward_risk_ratio'], 2.0, places=1)
        
    def test_calculate_stop_loss_take_profit_short(self):
        """Test stop loss and take profit calculation for short positions."""
        entry_price = 42000.0
        atr_value = 0.002
        side = "short"
        
        sl_tp = self.risk_manager.calculate_stop_loss_take_profit(entry_price, atr_value, side)
        
        # For short: SL above entry, TP below entry
        self.assertGreater(sl_tp['stop_loss'], entry_price)
        self.assertLess(sl_tp['take_profit'], entry_price)
        
        # Verify ATR multipliers
        expected_sl = entry_price + (atr_value * self.risk_manager.risk_params.atr_stop_multiplier)
        expected_tp = entry_price - (abs(expected_sl - entry_price) * self.risk_manager.risk_params.risk_reward_ratio)
        
        self.assertAlmostEqual(sl_tp['stop_loss'], expected_sl, places=2)
        self.assertAlmostEqual(sl_tp['take_profit'], expected_tp, places=2)
        
    def test_calculate_dynamic_leverage(self):
        """Test dynamic leverage calculation."""
        volatility_norm = 0.5
        
        leverage = self.risk_manager.calculate_dynamic_leverage(volatility_norm)
        
        # Should be positive and within max leverage
        self.assertGreater(leverage, 0)
        self.assertLessEqual(leverage, self.risk_manager.risk_params.max_leverage)
        
        # Test with high volatility (should reduce leverage)
        high_vol_leverage = self.risk_manager.calculate_dynamic_leverage(2.0)
        self.assertLessEqual(high_vol_leverage, leverage)
        
        # Test with low volatility (should allow higher leverage)
        low_vol_leverage = self.risk_manager.calculate_dynamic_leverage(0.1)
        self.assertGreaterEqual(low_vol_leverage, leverage)
        
    def test_calculate_dynamic_leverage_with_regime_adjustments(self):
        """Test dynamic leverage with regime adjustments."""
        # Add some drawdown history
        self.risk_manager.drawdown_history = [
            (datetime.now() - timedelta(days=1), -0.05),
            (datetime.now() - timedelta(days=2), -0.03),
            (datetime.now(), -0.08)  # 8% drawdown
        ]
        
        # Add Sharpe history
        self.risk_manager.sharpe_history = [
            (datetime.now() - timedelta(days=10), 0.5),
            (datetime.now() - timedelta(days=5), 0.8),
            (datetime.now(), 1.2)
        ]
        
        # Add equity curve for slope calculation
        base_equity = 10000.0
        for i in range(60):
            equity = base_equity + i * 50  # Positive slope
            timestamp = datetime.now() - timedelta(minutes=60-i)
            self.risk_manager.equity_curve.append((timestamp, equity))
            
        volatility_norm = 0.5
        leverage = self.risk_manager.calculate_dynamic_leverage(volatility_norm)
        
        # Should be adjusted based on regime factors
        self.assertGreater(leverage, 0)
        self.assertLessEqual(leverage, self.risk_manager.risk_params.max_leverage)
        
    def test_validate_position_size(self):
        """Test position size validation."""
        symbol = "BTCUSDT"
        position_data = {
            'position_size_usdt': 1000.0,
            'position_size_contracts': 0.025,
            'leverage': 5,
            'margin_required': 200.0
        }
        
        # Valid position should pass
        is_valid, reason = self.risk_manager.validate_position_size(symbol, position_data)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Position size validated")
        
        # Test with zero position size
        position_data['position_size_usdt'] = 0.0
        is_valid, reason = self.risk_manager.validate_position_size(symbol, position_data)
        self.assertFalse(is_valid)
        self.assertIn("zero", reason.lower())
        
        # Test with excessive leverage
        position_data['position_size_usdt'] = 1000.0
        position_data['leverage'] = 25  # Above max
        is_valid, reason = self.risk_manager.validate_position_size(symbol, position_data)
        self.assertFalse(is_valid)
        self.assertIn("leverage", reason.lower())
        
    def test_update_drawdown_history(self):
        """Test drawdown history management."""
        initial_length = len(self.risk_manager.drawdown_history)
        
        # Add drawdown
        self.risk_manager.update_drawdown_history(-0.05)
        
        self.assertEqual(len(self.risk_manager.drawdown_history), initial_length + 1)
        
        # Add multiple drawdowns beyond 3-day window
        for i in range(5):
            self.risk_manager.update_drawdown_history(-0.02 * (i + 1))
            
        # Should maintain 3-day window
        self.assertLessEqual(len(self.risk_manager.drawdown_history), 3 * 24 * 60)  # 3 days in minutes
        
    def test_update_sharpe_history(self):
        """Test Sharpe ratio history management."""
        initial_length = len(self.risk_manager.sharpe_history)
        
        # Add Sharpe ratio
        self.risk_manager.update_sharpe_history(1.5)
        
        self.assertEqual(len(self.risk_manager.sharpe_history), initial_length + 1)
        
        # Test rolling window
        for i in range(35):
            self.risk_manager.update_sharpe_history(1.0 + i * 0.1)
            
        # Should maintain 30-day window
        self.assertLessEqual(len(self.risk_manager.sharpe_history), 30)
        
    def test_update_equity_curve(self):
        """Test equity curve management."""
        initial_length = len(self.risk_manager.equity_curve)
        
        # Add equity point
        self.risk_manager.update_equity_curve(10500.0)
        
        self.assertEqual(len(self.risk_manager.equity_curve), initial_length + 1)
        
        # Test rolling window
        for i in range(65):
            self.risk_manager.update_equity_curve(10000.0 + i * 100)
            
        # Should maintain 60-bar window
        self.assertEqual(len(self.risk_manager.equity_curve), 60)
        
    def test_get_current_drawdown(self):
        """Test current drawdown calculation."""
        # Add equity curve with drawdown
        peak_equity = 12000.0
        current_equity = 10000.0
        
        self.risk_manager.equity_curve = [
            (datetime.now() - timedelta(minutes=10), peak_equity),
            (datetime.now() - timedelta(minutes=5), 11500.0),
            (datetime.now(), current_equity)
        ]
        
        drawdown = self.risk_manager.get_current_drawdown()
        expected_drawdown = (current_equity - peak_equity) / peak_equity
        
        self.assertAlmostEqual(drawdown, expected_drawdown, places=3)
        self.assertLess(drawdown, 0)  # Should be negative
        
    def test_get_current_sharpe(self):
        """Test current Sharpe ratio calculation."""
        # Add Sharpe history
        sharpe_values = [1.2, 1.5, 1.8, 1.3, 1.6]
        for i, sharpe in enumerate(sharpe_values):
            timestamp = datetime.now() - timedelta(days=len(sharpe_values)-i)
            self.risk_manager.sharpe_history.append((timestamp, sharpe))
            
        current_sharpe = self.risk_manager.get_current_sharpe()
        
        # Should return most recent Sharpe ratio
        self.assertEqual(current_sharpe, sharpe_values[-1])
        
    def test_get_equity_slope(self):
        """Test equity curve slope calculation."""
        # Add equity curve with positive slope
        base_equity = 10000.0
        for i in range(10):
            equity = base_equity + i * 100  # +100 per period
            timestamp = datetime.now() - timedelta(minutes=10-i)
            self.risk_manager.equity_curve.append((timestamp, equity))
            
        slope = self.risk_manager.get_equity_slope()
        
        # Should detect positive slope
        self.assertGreater(slope, 0)
        
    def test_check_risk_limits(self):
        """Test risk limit monitoring."""
        symbol = "BTCUSDT"
        
        # Test with normal conditions
        is_safe, violations = self.risk_manager.check_risk_limits(symbol)
        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
        
        # Add excessive drawdown
        self.risk_manager.drawdown_history = [
            (datetime.now(), -0.25)  # 25% drawdown
        ]
        
        is_safe, violations = self.risk_manager.check_risk_limits(symbol)
        self.assertFalse(is_safe)
        self.assertGreater(len(violations), 0)
        
    def test_emergency_flatten_signal(self):
        """Test emergency flatten signal generation."""
        symbol = "BTCUSDT"
        
        # Test with extreme drawdown
        self.risk_manager.drawdown_history = [
            (datetime.now(), -0.30)  # 30% drawdown
        ]
        
        should_flatten = self.risk_manager.should_emergency_flatten(symbol)
        self.assertTrue(should_flatten)
        
        # Test with normal conditions
        self.risk_manager.drawdown_history = [
            (datetime.now(), -0.05)  # 5% drawdown
        ]
        
        should_flatten = self.risk_manager.should_emergency_flatten(symbol)
        self.assertFalse(should_flatten)
        
    def test_update_daily_pnl(self):
        """Test daily PnL tracking."""
        symbol = "BTCUSDT"
        pnl_amount = 150.0
        
        initial_pnl = self.risk_manager.daily_pnl
        self.risk_manager.update_daily_pnl(symbol, pnl_amount)
        
        # Should update daily PnL
        self.assertEqual(self.risk_manager.daily_pnl, initial_pnl + pnl_amount)
        
    def test_get_risk_metrics(self):
        """Test risk metrics summary."""
        # Setup some data
        self.risk_manager.daily_pnl = 200.0
        self.risk_manager.drawdown_history = [(datetime.now(), -0.08)]
        self.risk_manager.sharpe_history = [(datetime.now(), 1.5)]
        
        metrics = self.risk_manager.get_risk_metrics()
        
        # Check required fields
        required_fields = [
            'daily_pnl', 'current_drawdown', 'current_sharpe',
            'risk_status', 'max_drawdown_hit', 'positions_count'
        ]
        
        for field in required_fields:
            self.assertIn(field, metrics)
            
        # Check values
        self.assertEqual(metrics['daily_pnl'], 200.0)
        self.assertEqual(metrics['current_drawdown'], -0.08)
        self.assertEqual(metrics['current_sharpe'], 1.5)
        
    def test_position_tracking(self):
        """Test position tracking functionality."""
        symbol = "BTCUSDT"
        position_data = {
            'side': 'long',
            'size': 0.025,
            'entry_price': 42000.0,
            'stop_loss': 41000.0,
            'take_profit': 44000.0,
            'leverage': 5
        }
        
        # Add position
        self.risk_manager.add_position(symbol, position_data)
        
        # Check position is tracked
        self.assertIn(symbol, self.risk_manager.positions)
        self.assertEqual(self.risk_manager.positions[symbol], position_data)
        
        # Update position
        position_data['size'] = 0.03
        self.risk_manager.update_position(symbol, position_data)
        
        self.assertEqual(self.risk_manager.positions[symbol]['size'], 0.03)
        
        # Remove position
        self.risk_manager.remove_position(symbol)
        
        self.assertNotIn(symbol, self.risk_manager.positions)
        
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        symbol = "BTCUSDT"
        
        # Test with None values
        result = self.risk_manager.calculate_position_size(
            symbol, None, 0.002, 42000.0
        )
        self.assertEqual(result['position_size_usdt'], 0.0)
        
        # Test with negative values
        result = self.risk_manager.calculate_position_size(
            symbol, -1000.0, 0.002, 42000.0
        )
        self.assertEqual(result['position_size_usdt'], 0.0)
        
        # Test invalid side for stop loss calculation
        sl_tp = self.risk_manager.calculate_stop_loss_take_profit(
            42000.0, 0.002, "invalid_side"
        )
        self.assertIn('error', sl_tp)
        
    def test_risk_parameter_updates(self):
        """Test dynamic risk parameter updates."""
        # Update risk per trade
        new_risk_pct = 0.01  # 1%
        self.risk_manager.update_risk_parameters(risk_per_trade_pct=new_risk_pct)
        
        self.assertEqual(self.risk_manager.risk_params.risk_per_trade_pct, new_risk_pct)
        
        # Update multiple parameters
        self.risk_manager.update_risk_parameters(
            kelly_fraction=0.8,
            max_leverage=8,
            atr_stop_multiplier=2.0
        )
        
        self.assertEqual(self.risk_manager.risk_params.kelly_fraction, 0.8)
        self.assertEqual(self.risk_manager.risk_params.max_leverage, 8)
        self.assertEqual(self.risk_manager.risk_params.atr_stop_multiplier, 2.0)


if __name__ == "__main__":
    unittest.main()
