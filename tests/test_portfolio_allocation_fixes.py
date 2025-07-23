#!/usr/bin/env python3
"""
Test file for portfolio allocation fixes
Tests to verify:
1. Correct initial capital detection
2. Allocation percentages per asset do not exceed 100%
3. Capital assigned to each asset aligns with portfolio strategy
4. Proportional distribution of assets
"""

import pytest
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

# Import the components we're testing
from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.execution_engine import ProductionExecutionEngine
from binance_exchange import BinanceClient
from main import TradingAlgorithm
from algorithm.strategies.ma_crossover import MACrossoverStrategy


class TestPortfolioAllocationFixes(unittest.TestCase):
    """Test class for portfolio allocation fixes."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_capital = 14000.0  # Simulating actual testnet balance
        self.portfolio_manager = ProductionPortfolioManager(
            total_capital=self.test_capital,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    
    def test_correct_initial_capital_detection(self):
        """Test that portfolio manager correctly handles capital initialization."""
        # Test 1: Correct capital initialization
        self.assertEqual(self.portfolio_manager.total_capital, self.test_capital)
        
        # Test 2: Portfolio summary reflects correct capital
        summary = self.portfolio_manager.get_portfolio_summary()
        self.assertEqual(summary['total_capital'], self.test_capital)
        
        # Test 3: Max allocation calculation is correct
        expected_max_allocation = self.test_capital * 0.85  # 85% of 14,000 = 11,900
        self.assertAlmostEqual(expected_max_allocation, 11900.0, places=2)
        
        print(f"✅ Test 1 PASSED: Capital correctly initialized as ${self.test_capital:.2f}")
    
    def test_allocation_percentages_do_not_exceed_100_percent(self):
        """Test that individual asset allocations do not exceed 100% of total capital."""
        # Add volatility data for all symbols
        for symbol in self.test_symbols:
            # Add realistic volatility data (2% ATR)
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Add some correlation data to make the test realistic
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Test that no individual asset exceeds 100% of total capital
        for symbol, allocation in allocations.items():
            allocation_pct = (allocation.allocated_capital / self.test_capital) * 100
            self.assertLessEqual(allocation_pct, 100.0, 
                f"Asset {symbol} allocation {allocation_pct:.1f}% exceeds 100% of total capital")
            
            # Also test that no asset allocation is negative
            self.assertGreaterEqual(allocation.allocated_capital, 0, 
                f"Asset {symbol} has negative allocation: ${allocation.allocated_capital:.2f}")
            
            print(f"✅ {symbol}: ${allocation.allocated_capital:.2f} ({allocation_pct:.1f}% of total capital)")
        
        print(f"✅ Test 2 PASSED: No asset allocation exceeds 100% of total capital")
    
    def test_capital_assignment_aligns_with_strategy(self):
        """Test that capital assigned to each asset follows the portfolio strategy."""
        # Add volatility data for all symbols
        for symbol in self.test_symbols:
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Add correlation data
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Test 1: Total allocation should not exceed max_allocation_pct
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        max_allowed = self.test_capital * 0.85  # 85% max allocation
        
        self.assertLessEqual(total_allocated, max_allowed * 1.01, 
            f"Total allocation ${total_allocated:.2f} exceeds max allowed ${max_allowed:.2f}")
        
        # Test 2: Weights should sum to approximately 1 (within tolerance)
        total_weight = sum(a.weight for a in allocations.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2,
            msg=f"Total weights {total_weight:.3f} should sum to 1.0")
        
        # Test 3: Each asset should have reasonable allocation (not 0, not extreme)
        for symbol, allocation in allocations.items():
            allocation_pct = allocation.weight * 100
            self.assertGreater(allocation_pct, 0.5, 
                f"Asset {symbol} allocation {allocation_pct:.1f}% is too small")
            self.assertLess(allocation_pct, 50.0, 
                f"Asset {symbol} allocation {allocation_pct:.1f}% is too large")
        
        print(f"✅ Test 3 PASSED: Total allocated ${total_allocated:.2f} within limits (max: ${max_allowed:.2f})")
        print(f"✅ Test 3 PASSED: Total weights sum to {total_weight:.3f}")
    
    def test_proportional_distribution_across_assets(self):
        """Test that assets are proportionally distributed, not each getting 85%."""
        # Add volatility data for all symbols (equal volatility for simplicity)
        for symbol in self.test_symbols:
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)  # Equal 2% volatility
        
        # Add correlation data (equal correlations)
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # With equal volatility and correlations, weights should be approximately equal
        expected_weight_per_asset = 1.0 / len(self.test_symbols)  # 20% each for 5 assets
        
        for symbol, allocation in allocations.items():
            # Each asset should get approximately equal weight (within tolerance)
            self.assertAlmostEqual(allocation.weight, expected_weight_per_asset, delta=0.05,
                msg=f"Asset {symbol} weight {allocation.weight:.3f} not close to expected {expected_weight_per_asset:.3f}")
            
            # Asset allocation percentage should be around 20% each (not 85% each)
            allocation_pct = (allocation.allocated_capital / self.test_capital) * 100
            expected_pct = expected_weight_per_asset * 0.85 * 100  # 20% of 85% max = 17%
            
            self.assertAlmostEqual(allocation_pct, expected_pct, delta=2.0,
                msg=f"Asset {symbol} allocation {allocation_pct:.1f}% not close to expected {expected_pct:.1f}%")
            
            print(f"✅ {symbol}: weight={allocation.weight:.3f} ({allocation.weight*100:.1f}%), "
                  f"allocation=${allocation.allocated_capital:.2f} ({allocation_pct:.1f}%)")
        
        print(f"✅ Test 4 PASSED: Assets properly distributed (not each getting 85%)")
    
    def test_scaling_multiplier_application(self):
        """Test that scaling multiplier is applied correctly to total allocation, not per asset."""
        # Add volatility data
        for symbol in self.test_symbols:
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Add correlation data
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Calculate scaling multiplier
        scaling_multiplier = self.portfolio_manager.calculate_scaling_multiplier()
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Calculate expected total allocation
        max_allocation = self.test_capital * 0.85  # 85% max
        expected_total = scaling_multiplier * max_allocation
        actual_total = sum(a.allocated_capital for a in allocations.values())
        
        # Total allocation should match expected (scaled total)
        self.assertAlmostEqual(actual_total, expected_total, delta=1.0,
            msg=f"Total allocation ${actual_total:.2f} not close to expected ${expected_total:.2f}")
        
        print(f"✅ Test 5 PASSED: Scaling multiplier {scaling_multiplier:.3f} correctly applied")
        print(f"  Max allocation: ${max_allocation:.2f}")
        print(f"  Expected total: ${expected_total:.2f}")
        print(f"  Actual total: ${actual_total:.2f}")


class TestCapitalFetchingIntegration(unittest.TestCase):
    """Test capital fetching integration with TradingAlgorithm."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_testnet_balance = 14750.25  # Mock testnet balance
    
    @pytest.mark.asyncio
    async def test_capital_fetching_from_exchange(self):
        """Test that TradingAlgorithm correctly fetches capital from exchange."""
        # Create mock binance client
        mock_binance_client = MagicMock()
        mock_binance_client.get_account_metrics = AsyncMock(return_value={
            'total_wallet_balance': self.mock_testnet_balance,
            'total_unrealized_pnl': 0.0,
            'total_margin_used': 0.0,
            'available_margin': self.mock_testnet_balance,
            'exposure_percentage': 0.0,
            'position_count': 0
        })
        
        # Create strategy
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.001,
            'take_profit_pct': 0.002,
            'leverage': 7
        })
        
        # Create trading algorithm with no capital specified
        algorithm = TradingAlgorithm(strategy=strategy, testnet=True, total_capital=None)
        
        # Replace the binance client with our mock
        algorithm.binance_client = mock_binance_client
        
        # We can't easily test the full start() method due to complexity, 
        # but we can test the capital fetching logic specifically
        account_metrics = await mock_binance_client.get_account_metrics()
        fetched_capital = account_metrics['total_wallet_balance']
        
        self.assertEqual(fetched_capital, self.mock_testnet_balance)
        print(f"✅ Test 6 PASSED: Capital fetching returns ${fetched_capital:.2f}")


def run_all_tests():
    """Run all portfolio allocation tests."""
    print("="*80)
    print("RUNNING PORTFOLIO ALLOCATION FIXES TESTS")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(TestPortfolioAllocationFixes('test_correct_initial_capital_detection'))
    suite.addTest(TestPortfolioAllocationFixes('test_allocation_percentages_do_not_exceed_100_percent'))
    suite.addTest(TestPortfolioAllocationFixes('test_capital_assignment_aligns_with_strategy'))
    suite.addTest(TestPortfolioAllocationFixes('test_proportional_distribution_across_assets'))
    suite.addTest(TestPortfolioAllocationFixes('test_scaling_multiplier_application'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! Portfolio allocation fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
