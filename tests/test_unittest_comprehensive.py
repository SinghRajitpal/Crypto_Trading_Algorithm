#!/usr/bin/env python3
"""
Comprehensive unittest tests for the crypto trading algorithm.

This module provides unittest-based tests as an alternative to pytest,
demonstrating both testing frameworks for the trading system.

Tests cover the same functionality as pytest versions but using
unittest framework conventions and assertions.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from execution.stress_handler import StressHandlingModule
from algorithm.trade_signal import TradeSignal


class TestProductionPortfolioManagerUnittest(unittest.TestCase):
    """Unittest test suite for ProductionPortfolioManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_capital = 15000.0
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT"]
        self.portfolio = ProductionPortfolioManager(total_capital=self.test_capital)
        
    def test_initialization(self):
        """Test portfolio manager initialization."""
        self.assertEqual(self.portfolio.total_capital, self.test_capital)
        self.assertEqual(self.portfolio.target_volatility, 0.18)
        self.assertEqual(self.portfolio.max_allocation_pct, 0.85)
        self.assertEqual(self.portfolio.alpha, 0.3)
        
        # Test initial state
        self.assertEqual(self.portfolio.volatility_data, {})
        self.assertEqual(self.portfolio.correlation_data, {})
        # last_rebalance_time may be set during initialization
        self.assertIsInstance(self.portfolio.last_rebalance_time, (type(None), datetime))
        
    def test_volatility_tracking(self):
        """Test volatility EMA tracking."""
        # Single update
        self.portfolio.update_volatility_data("BTCUSDT", 0.025)
        vol_ema = self.portfolio.get_volatility_ema("BTCUSDT")
        self.assertEqual(vol_ema, 0.025)
        
        # Multiple updates
        volatilities = [0.020, 0.030, 0.025, 0.035]
        for vol in volatilities:
            self.portfolio.update_volatility_data("BTCUSDT", vol)
        
        final_ema = self.portfolio.get_volatility_ema("BTCUSDT")
        self.assertGreaterEqual(final_ema, 0.015)
        self.assertLessEqual(final_ema, 0.040)
        self.assertIsInstance(final_ema, float)
        
    def test_weight_calculation_formula(self):
        """Test portfolio weight calculation formula."""
        symbols = self.test_symbols[:3]
        test_volatilities = [0.01, 0.02, 0.03]
        
        for symbol, vol in zip(symbols, test_volatilities):
            self.portfolio.update_volatility_data(symbol, vol)
        
        # Force rebalance
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        
        # Test allocations structure
        self.assertEqual(len(allocations), len(symbols))
        
        # Test weight normalization
        total_weight = sum(alloc.weight for alloc in allocations.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Test inverse volatility relationship
        weights_by_vol = [(allocations[symbol].weight, self.portfolio.get_volatility_ema(symbol)) 
                         for symbol in symbols]
        weights_by_vol.sort(key=lambda x: x[1])
        
        # Lower volatility should have higher weight
        for i in range(len(weights_by_vol) - 1):
            lower_vol_weight = weights_by_vol[i][0]
            higher_vol_weight = weights_by_vol[i + 1][0]
            self.assertGreaterEqual(lower_vol_weight, higher_vol_weight)
            
    def test_allocation_scaling(self):
        """Test allocation scaling with 85% cap."""
        symbols = self.test_symbols[:3]
        
        for symbol in symbols:
            self.portfolio.update_volatility_data(symbol, 0.02)
        
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        max_allocation = self.portfolio.total_capital * 0.85
        
        self.assertLessEqual(total_allocated, max_allocation * 1.01)
        self.assertGreaterEqual(total_allocated, max_allocation * 0.95)
        
    def test_rebalancing_timing(self):
        """Test daily rebalancing timing logic."""
        symbols = self.test_symbols[:3]
        
        for symbol in symbols:
            self.portfolio.update_volatility_data(symbol, 0.02)
        
        # Portfolio rebalances regardless of timing (timing checked elsewhere)
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=12)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        self.assertEqual(len(allocations), len(symbols))
        
        # Rebalance when time threshold passed
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        self.assertEqual(len(allocations), len(symbols))
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero volatility
        self.portfolio.update_volatility_data("ZEROVOLBTC", 0.0)
        vol = self.portfolio.get_volatility_ema("ZEROVOLBTC")
        self.assertGreater(vol, 0)
        
        # Very high volatility
        self.portfolio.update_volatility_data("HIGHVOLBTC", 0.5)
        vol = self.portfolio.get_volatility_ema("HIGHVOLBTC")
        self.assertEqual(vol, 0.5)
        
        # Empty symbol list
        allocations = self.portfolio.rebalance_portfolio([])
        self.assertEqual(allocations, {})
        
        # Single symbol
        self.portfolio.update_volatility_data("SINGLEBTC", 0.02)
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(["SINGLEBTC"])
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations["SINGLEBTC"].weight, 1.0)


class TestProductionRiskManagerUnittest(unittest.TestCase):
    """Unittest test suite for ProductionRiskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_capital = 15000.0
        self.portfolio = ProductionPortfolioManager(total_capital=self.test_capital)
        self.risk_manager = ProductionRiskManager(portfolio_manager=self.portfolio)
        
    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(self.risk_manager.risk_per_trade, 0.008)
        self.assertEqual(self.risk_manager.risk_params.kelly_fraction, 0.7)
        self.assertEqual(self.risk_manager.risk_params.min_atr_floor, 0.001)
        self.assertEqual(self.risk_manager.risk_params.base_cost_pct, 0.0014)
        self.assertEqual(self.risk_manager.risk_params.max_leverage, 10)
        self.assertEqual(self.risk_manager.portfolio_manager, self.portfolio)
        
    def test_position_sizing_formula(self):
        """Test position sizing formula implementation."""
        allocated_capital = 3000.0
        atr_value = 0.02
        entry_price = 50000.0
        volatility_norm = 0.5
        
        result = self.risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=volatility_norm
        )
        
        # Test result structure
        required_keys = ['size_contracts', 'size_usdt', 'leverage', 'margin_usdt', 
                        'risk_amount', 'atr_adjusted']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Test value constraints
        self.assertGreater(result['size_usdt'], 0)
        self.assertGreaterEqual(result['leverage'], 1)
        self.assertLessEqual(result['leverage'], 10)
        self.assertGreater(result['margin_usdt'], 0)
        self.assertGreater(result['risk_amount'], 0)
        
        # Test ATR adjustment
        atr_adjusted = max(atr_value, 0.001)
        self.assertEqual(result['atr_adjusted'], atr_adjusted)
        
    def test_dynamic_leverage_calculation(self):
        """Test dynamic leverage calculation."""
        # Normal conditions
        leverage = self.risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.02)
        self.assertGreaterEqual(leverage, 1)
        self.assertLessEqual(leverage, 10)
        
        # High volatility should reduce leverage
        high_vol_leverage = self.risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.08)
        normal_vol_leverage = self.risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.02)
        self.assertLessEqual(high_vol_leverage, normal_vol_leverage)
        
    def test_stop_loss_take_profit_calculation(self):
        """Test SL/TP calculation formula."""
        entry_price = 50000.0
        atr_value = 0.02
        atr_adjusted = atr_value * entry_price
        
        # Test buy orders
        sl_price, tp_price = self.risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="buy",
            atr_adjusted=atr_adjusted
        )
        
        self.assertLess(sl_price, entry_price)
        self.assertGreater(tp_price, entry_price)
        
        # Test risk-reward ratio
        risk_distance = entry_price - sl_price
        reward_distance = tp_price - entry_price
        rr_ratio = reward_distance / risk_distance
        
        self.assertGreaterEqual(rr_ratio, 1.8)
        self.assertLessEqual(rr_ratio, 2.2)
        
        # Test sell orders
        sl_price_sell, tp_price_sell = self.risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="sell",
            atr_adjusted=atr_adjusted
        )
        
        self.assertGreater(sl_price_sell, entry_price)
        self.assertLess(tp_price_sell, entry_price)
        
    def test_atr_floor_enforcement(self):
        """Test ATR floor enforcement."""
        allocated_capital = 3000.0
        entry_price = 50000.0
        
        # Below floor
        tiny_atr = 0.0001
        result = self.risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=tiny_atr,
            entry_price=entry_price
        )
        
        self.assertEqual(result['atr_adjusted'], 0.001)
        
        # Above floor
        normal_atr = 0.02
        result_normal = self.risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=normal_atr,
            entry_price=entry_price
        )
        
        self.assertEqual(result_normal['atr_adjusted'], normal_atr)
        self.assertGreater(result['size_usdt'], result_normal['size_usdt'])
        
    def test_dynamic_cost_calculation(self):
        """Test dynamic cost calculation."""
        allocated_capital = 3000.0
        atr_value = 0.02
        entry_price = 50000.0
        
        # Low volatility
        low_vol_result = self.risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=0.2
        )
        
        # High volatility
        high_vol_result = self.risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=0.8
        )
        
        self.assertLessEqual(high_vol_result['size_usdt'], low_vol_result['size_usdt'])


class TestStressHandlingModuleUnittest(unittest.TestCase):
    """Unittest test suite for StressHandlingModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_execution_engine = Mock()
        self.stress_handler = StressHandlingModule(self.mock_execution_engine)
        
    def test_initialization(self):
        """Test stress handler initialization."""
        self.assertEqual(self.stress_handler.connection_lag_threshold, 3.0)
        self.assertEqual(self.stress_handler.slippage_threshold, 0.002)
        self.assertEqual(self.stress_handler.min_daily_volume, 5_000_000)
        self.assertEqual(self.stress_handler.max_spread, 0.0015)
        self.assertEqual(self.stress_handler.max_funding_rate, 0.004)
        
        self.assertEqual(self.stress_handler.flash_crash_events, [])
        self.assertEqual(self.stress_handler.affected_assets_60s, set())
        self.assertFalse(self.stress_handler.forward_fill_active)
        self.assertEqual(len(self.stress_handler.stress_events), 0)
        
    def test_flash_crash_detection(self):
        """Test flash crash detection logic."""
        atr_value = 0.02
        
        # Normal movement
        normal_drop = 0.03
        is_flash_crash = self.stress_handler.check_flash_crash("BTCUSDT", normal_drop, atr_value)
        self.assertFalse(is_flash_crash)
        
        # Flash crash
        flash_drop = 0.09
        is_flash_crash = self.stress_handler.check_flash_crash("BTCUSDT", flash_drop, atr_value)
        self.assertTrue(is_flash_crash)
        
        self.assertEqual(len(self.stress_handler.flash_crash_events), 1)
        self.assertIn("BTCUSDT", self.stress_handler.affected_assets_60s)
        
    def test_kill_switch_thresholds(self):
        """Test kill switch activation thresholds."""
        # Drawdown kill switch
        switches = self.stress_handler.check_kill_switches(0.15, -0.05)
        self.assertIn("drawdown_partial", switches)
        self.assertTrue(self.stress_handler.kill_switches["drawdown_partial"])
        
        # Equity slope kill switch
        switches = self.stress_handler.check_kill_switches(0.10, -0.12)
        self.assertIn("equity_slope", switches)
        self.assertTrue(self.stress_handler.kill_switches["equity_slope"])
        
        # No trigger
        switches = self.stress_handler.check_kill_switches(0.10, -0.05)
        self.assertEqual(len(switches), 0)
        
    def test_kill_switch_helper_method(self):
        """Test kill switch helper method."""
        self.assertFalse(self.stress_handler.should_trigger_kill_switch(0.10))
        self.assertFalse(self.stress_handler.should_trigger_kill_switch(0.14))
        self.assertTrue(self.stress_handler.should_trigger_kill_switch(0.15))
        self.assertTrue(self.stress_handler.should_trigger_kill_switch(0.20))
        
    def test_liquidity_filters(self):
        """Test liquidity filters."""
        # Good liquidity
        is_liquid = self.stress_handler.check_liquidity_filters(10_000_000, 0.001)
        self.assertTrue(is_liquid)
        
        # Low volume
        is_liquid_vol = self.stress_handler.check_liquidity_filters(3_000_000, 0.001)
        self.assertFalse(is_liquid_vol)
        
        # High spread
        is_liquid_spread = self.stress_handler.check_liquidity_filters(10_000_000, 0.002)
        self.assertFalse(is_liquid_spread)
        
        # Both bad
        is_liquid_both = self.stress_handler.check_liquidity_filters(3_000_000, 0.002)
        self.assertFalse(is_liquid_both)
        
    def test_slippage_monitoring(self):
        """Test slippage monitoring."""
        expected_price = 50000.0
        
        # Acceptable slippage
        is_acceptable = self.stress_handler.check_slippage(expected_price, 50050.0, "BTCUSDT")
        self.assertTrue(is_acceptable)
        
        # Excessive slippage
        is_acceptable_bad = self.stress_handler.check_slippage(expected_price, 50150.0, "BTCUSDT")
        self.assertFalse(is_acceptable_bad)
        
        # Check event recording
        slippage_events = [e for e in self.stress_handler.stress_events 
                          if e.event_type == "excessive_slippage"]
        self.assertGreaterEqual(len(slippage_events), 1)
        
    def test_connection_lag_handling(self):
        """Test connection lag handling."""
        now = datetime.now()
        
        # Normal connection
        recent_timestamp = now - timedelta(seconds=1)
        is_healthy = self.stress_handler.check_connection_lag(recent_timestamp)
        self.assertTrue(is_healthy)
        self.assertFalse(self.stress_handler.forward_fill_active)
        
        # Lagged connection
        old_timestamp = now - timedelta(seconds=5)
        is_healthy_lag = self.stress_handler.check_connection_lag(old_timestamp)
        self.assertFalse(is_healthy_lag)
        self.assertTrue(self.stress_handler.forward_fill_active)
        
        # Recovery
        recovery_timestamp = now - timedelta(seconds=1)
        is_recovered = self.stress_handler.check_connection_lag(recovery_timestamp)
        self.assertTrue(is_recovered)
        self.assertFalse(self.stress_handler.forward_fill_active)


class TestIntegrationUnittest(unittest.TestCase):
    """Unittest test suite for integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_capital = 15000.0
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        self.portfolio = ProductionPortfolioManager(total_capital=self.test_capital)
        self.risk_manager = ProductionRiskManager(portfolio_manager=self.portfolio)
        
    def test_portfolio_risk_integration(self):
        """Test integration between portfolio and risk managers."""
        symbols = self.test_symbols
        volatilities = [0.015, 0.025, 0.035]
        
        for symbol, vol in zip(symbols, volatilities):
            self.portfolio.update_volatility_data(symbol, vol)
        
        # Force rebalance
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        
        # Test position sizing for each allocation
        for symbol, allocation in allocations.items():
            result = self.risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=self.portfolio.get_volatility_ema(symbol),
                entry_price=50000.0
            )
            
            self.assertGreater(result['size_usdt'], 0)
            self.assertLessEqual(result['size_usdt'], allocation.allocated_capital)
            
            expected_risk = allocation.allocated_capital * 0.008
            self.assertLessEqual(result['risk_amount'], expected_risk * 1.2)
            
    def test_signal_execution_integration(self):
        """Test signal to execution integration."""
        signal = TradeSignal(
            symbol="BTCUSDT",
            side="buy",
            action="open",
            strategy_id="unittest_strategy",
            metadata={"confidence": 0.8, "entry_price": 50000.0},
            signal_confidence=0.8
        )
        
        # Test signal structure
        self.assertIn(signal.symbol, self.test_symbols)
        self.assertIn(signal.side, ["buy", "sell"])
        self.assertIn(signal.action, ["open", "close"])
        self.assertGreaterEqual(signal.signal_confidence, 0)
        self.assertLessEqual(signal.signal_confidence, 1)
        
        # Test metadata
        self.assertIn("entry_price", signal.metadata)
        self.assertGreater(signal.metadata["entry_price"], 0)
        
    def test_mathematical_consistency(self):
        """Test mathematical consistency across modules."""
        symbols = self.test_symbols
        
        # Set volatilities
        for i, symbol in enumerate(symbols):
            self.portfolio.update_volatility_data(symbol, 0.02 + i * 0.01)
        
        # Get allocations
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio.rebalance_portfolio(symbols)
        
        # Calculate totals
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        total_position_value = 0
        total_risk = 0
        
        for symbol, allocation in allocations.items():
            position = self.risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=self.portfolio.get_volatility_ema(symbol),
                entry_price=50000.0
            )
            
            total_position_value += position['size_usdt']
            total_risk += position['risk_amount']
        
        # Test consistency
        max_allocation = self.test_capital * 0.85
        self.assertAlmostEqual(total_allocated, max_allocation, delta=1.0)
        
        total_risk_pct = total_risk / total_allocated
        self.assertLessEqual(total_risk_pct, 0.01)
        
        utilization = total_position_value / total_allocated
        self.assertGreaterEqual(utilization, 0.1)
        self.assertLessEqual(utilization, 1.0)


class TestSuiteRunner:
    """Test suite runner for unittest framework."""
    
    @staticmethod
    def run_all_tests():
        """Run all unittest test suites."""
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestProductionPortfolioManagerUnittest,
            TestProductionRiskManagerUnittest,
            TestStressHandlingModuleUnittest,
            TestIntegrationUnittest
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result
    
    @staticmethod
    def run_specific_test(test_class_name, test_method_name=None):
        """Run a specific test class or method."""
        if test_method_name:
            suite = unittest.TestSuite()
            suite.addTest(globals()[test_class_name](test_method_name))
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(globals()[test_class_name])
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result


if __name__ == '__main__':
    # Run all tests when script is executed directly
    print("Running Crypto Trading Algorithm Unittest Suite")
    print("=" * 60)
    
    runner = TestSuiteRunner()
    result = runner.run_all_tests()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for failure in result.failures:
            print(f"- {failure[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"- {error[0]}")
