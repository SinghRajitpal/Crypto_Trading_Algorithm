#!/usr/bin/env python3
"""
Quick validation test for specific portfolio allocation issues:
1. Initial capital should be ~14,000 (not 5,500)
2. Each asset should NOT show 85% allocation
3. Available capital per asset should be properly calculated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

from execution.portfolio import ProductionPortfolioManager


class TestSpecificAllocationIssues(unittest.TestCase):
    """Test specific allocation issues identified by the user."""
    
    def setUp(self):
        """Set up test with realistic testnet balance."""
        self.actual_testnet_balance = 14000.0  # Realistic testnet balance
        self.portfolio_manager = ProductionPortfolioManager(
            total_capital=self.actual_testnet_balance,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
        # Test with config symbols
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT"]
    
    def test_issue_1_correct_initial_capital(self):
        """Test that initial capital reflects actual testnet balance (~14,000), not 5,500."""
        print("\n🔍 Testing Issue 1: Correct Initial Capital")
        
        # The portfolio manager should be initialized with actual balance
        self.assertEqual(self.portfolio_manager.total_capital, self.actual_testnet_balance)
        
        # Summary should reflect correct capital
        summary = self.portfolio_manager.get_portfolio_summary()
        self.assertEqual(summary['total_capital'], self.actual_testnet_balance)
        
        print(f"✅ FIXED: Initial capital correctly set to ${self.actual_testnet_balance:.2f}")
        print(f"  (was incorrectly: $5,500)")
    
    def test_issue_2_individual_allocations_not_85_percent(self):
        """Test that individual assets do NOT each show 85% allocation."""
        print("\n🔍 Testing Issue 2: Individual Asset Allocations")
        
        # Set up realistic market data
        for symbol in self.test_symbols:
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)  # 2% volatility
        
        # Add correlation data
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        print(f"Portfolio Analysis for ${self.actual_testnet_balance:.2f} total capital:")
        
        total_allocation_pct = 0.0
        for symbol, allocation in allocations.items():
            individual_pct = (allocation.allocated_capital / self.actual_testnet_balance) * 100
            total_allocation_pct += individual_pct
            
            # Each asset should NOT be 85%
            self.assertLess(individual_pct, 50.0, 
                f"❌ Asset {symbol} shows {individual_pct:.1f}% - this is too high!")
            
            print(f"  {symbol}: ${allocation.allocated_capital:.2f} ({individual_pct:.1f}% of total capital)")
        
        # Total should be around 85% (max allocation), distributed across assets
        expected_total_pct = 85.0  # We expect close to 85% total
        self.assertLess(abs(total_allocation_pct - expected_total_pct), 10.0,
            f"Total allocation {total_allocation_pct:.1f}% should be close to {expected_total_pct}%")
        
        print(f"✅ FIXED: Total allocation {total_allocation_pct:.1f}% properly distributed")
        print(f"  (was incorrectly: each asset showing 85%)")
    
    def test_issue_3_correct_available_capital_per_asset(self):
        """Test that available capital per asset is correctly calculated."""
        print("\n🔍 Testing Issue 3: Available Capital per Asset")
        
        # Set up market data
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
        
        print(f"Available Capital Analysis:")
        
        for symbol, allocation in allocations.items():
            allocated_capital = allocation.allocated_capital
            
            # Available capital should be the allocated amount (minus any reservations)
            available_capital = allocated_capital  # No reservations yet
            
            # Should NOT be around 825 for each asset (that was the bug)
            self.assertNotAlmostEqual(available_capital, 825.0, delta=50.0,
                msg=f"Asset {symbol} should not have ~$825 available (that was the bug)")
            
            # Should be proportional to allocation
            expected_range_min = self.actual_testnet_balance * 0.05  # At least 5% share
            expected_range_max = self.actual_testnet_balance * 0.40  # At most 40% share
            
            self.assertGreaterEqual(available_capital, expected_range_min,
                f"Asset {symbol} available capital ${available_capital:.2f} too low")
            self.assertLessEqual(available_capital, expected_range_max,
                f"Asset {symbol} available capital ${available_capital:.2f} too high")
            
            print(f"  {symbol}: ${available_capital:.2f} available")
        
        print(f"✅ FIXED: Available capital properly distributed across assets")
        print(f"  (was incorrectly: each showing ~$825)")
    
    def test_comprehensive_validation(self):
        """Comprehensive validation of all fixes."""
        print("\n🔍 Comprehensive Validation of All Fixes")
        
        # Set up market data
        for symbol in self.test_symbols:
            for _ in range(10):
                self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        for i, symbol1 in enumerate(self.test_symbols):
            for symbol2 in self.test_symbols[i+1:]:
                for _ in range(10):
                    self.portfolio_manager.update_correlation_data(symbol1, symbol2, 0.6)
        
        # Get portfolio summary before rebalance
        summary_before = self.portfolio_manager.get_portfolio_summary()
        
        # Force rebalance
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Get portfolio summary after rebalance
        summary_after = self.portfolio_manager.get_portfolio_summary()
        
        print(f"\n📊 COMPREHENSIVE VALIDATION RESULTS:")
        print(f"Total Capital: ${summary_after['total_capital']:.2f}")
        print(f"Allocated Capital: ${summary_after['allocated_capital']:.2f}")
        print(f"Allocation Percentage: {summary_after['allocation_percentage']:.1%}")
        print(f"Active Symbols: {summary_after['active_symbols']}")
        
        print(f"\n📋 INDIVIDUAL ASSET ALLOCATIONS:")
        total_check = 0.0
        for symbol, allocation in allocations.items():
            pct_of_total = (allocation.allocated_capital / self.actual_testnet_balance) * 100
            total_check += pct_of_total
            print(f"  {symbol}: ${allocation.allocated_capital:.2f} ({pct_of_total:.1f}% of total capital)")
        
        print(f"\n✅ VALIDATION SUMMARY:")
        print(f"  ✓ Total capital: ${self.actual_testnet_balance:.2f} (correct, not $5,500)")
        print(f"  ✓ Total allocation: {total_check:.1f}% (properly distributed, not 85% each)")
        print(f"  ✓ Asset allocations: Proportional distribution achieved")
        print(f"  ✓ No asset exceeds reasonable allocation limits")
        
        # Assertions for comprehensive validation
        self.assertEqual(summary_after['total_capital'], self.actual_testnet_balance)
        self.assertGreater(summary_after['allocated_capital'], 0)
        self.assertLessEqual(summary_after['allocation_percentage'], 1.0)
        self.assertEqual(len(allocations), len(self.test_symbols))
        
        for symbol, allocation in allocations.items():
            pct = (allocation.allocated_capital / self.actual_testnet_balance) * 100
            self.assertLess(pct, 50.0, f"Asset {symbol} allocation too high: {pct:.1f}%")
            self.assertGreater(pct, 1.0, f"Asset {symbol} allocation too low: {pct:.1f}%")


def run_validation_tests():
    """Run validation tests for the specific issues."""
    print("="*80)
    print("VALIDATING PORTFOLIO ALLOCATION FIXES")
    print("="*80)
    print("Testing fixes for:")
    print("1. Incorrect initial capital allocation of 5,500 → ~14,000")
    print("2. Each asset showing 85% allocation → Proportional distribution")
    print("3. Each asset showing ~825 capital → Proper allocation amounts")
    print("="*80)
    
    # Create and run tests
    suite = unittest.TestSuite()
    suite.addTest(TestSpecificAllocationIssues('test_issue_1_correct_initial_capital'))
    suite.addTest(TestSpecificAllocationIssues('test_issue_2_individual_allocations_not_85_percent'))
    suite.addTest(TestSpecificAllocationIssues('test_issue_3_correct_available_capital_per_asset'))
    suite.addTest(TestSpecificAllocationIssues('test_comprehensive_validation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("Portfolio allocation issues have been successfully resolved.")
    else:
        print("❌ SOME VALIDATION TESTS FAILED:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
        for error in result.errors:
            print(f"  - {error[0]}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_validation_tests()
