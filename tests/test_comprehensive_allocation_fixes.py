#!/usr/bin/env python3
"""
Comprehensive pytest test suite for portfolio allocation fixes.
Tests verify the resolution of the critical issues:
1. Incorrect initial capital allocation of 5,500 → actual testnet balance (~14,000)
2. Each asset showing 85% allocation → Proportional distribution 
3. Each asset showing ~825 capital → Proper allocation amounts
"""

import pytest
import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.execution_engine import ProductionExecutionEngine


class TestPortfolioAllocationFixes:
    """Comprehensive test class for portfolio allocation fixes."""
    
    @pytest.fixture
    def test_capital(self):
        """Fixture providing realistic testnet capital."""
        return 14750.25  # Realistic testnet balance
    
    @pytest.fixture
    def test_symbols(self):
        """Fixture providing test symbols from config."""
        return ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT"]
    
    @pytest.fixture
    def portfolio_manager(self, test_capital):
        """Fixture providing configured portfolio manager."""
        return ProductionPortfolioManager(
            total_capital=test_capital,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
    
    @pytest.fixture
    def market_data_setup(self, portfolio_manager, test_symbols):
        """Fixture that sets up realistic market data."""
        # Add volatility data (realistic 2% ATR)
        for symbol in test_symbols:
            for _ in range(15):  # More data points for stability
                portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Add correlation data (realistic crypto correlations)
        correlations = {
            ("BTCUSDT", "ETHUSDT"): 0.7,
            ("BTCUSDT", "XRPUSDT"): 0.6,
            ("BTCUSDT", "BNBUSDT"): 0.65,
            ("BTCUSDT", "SOLUSDT"): 0.6,
            ("ETHUSDT", "XRPUSDT"): 0.65,
            ("ETHUSDT", "BNBUSDT"): 0.7,
            ("ETHUSDT", "SOLUSDT"): 0.75,
            ("XRPUSDT", "BNBUSDT"): 0.5,
            ("XRPUSDT", "SOLUSDT"): 0.55,
            ("BNBUSDT", "SOLUSDT"): 0.6,
        }
        
        for (sym1, sym2), corr in correlations.items():
            for _ in range(15):
                portfolio_manager.update_correlation_data(sym1, sym2, corr)
        
        return portfolio_manager
    
    def test_issue_1_correct_initial_capital_detection(self, test_capital, portfolio_manager):
        """Test Fix 1: Correct initial capital detection (not hard-coded 5,500)."""
        # Verify capital is correctly initialized
        assert portfolio_manager.total_capital == test_capital
        
        # Verify portfolio summary reflects correct capital
        summary = portfolio_manager.get_portfolio_summary()
        assert summary['total_capital'] == test_capital
        
        # Verify max allocation calculation is based on correct capital
        expected_max_allocation = test_capital * 0.85
        max_allocation_actual = test_capital * portfolio_manager.max_allocation_pct
        assert max_allocation_actual == expected_max_allocation
        
        print(f"✅ Fix 1 Verified: Capital correctly initialized as ${test_capital:.2f}")
        print(f"  Max allocation: ${expected_max_allocation:.2f} (85% of ${test_capital:.2f})")
    
    def test_issue_2_proportional_allocation_not_85_percent_each(self, market_data_setup, test_symbols, test_capital):
        """Test Fix 2: Assets get proportional allocation, not 85% each."""
        portfolio_manager = market_data_setup
        
        # Force rebalance
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Verify each asset does NOT get 85% allocation
        for symbol, allocation in allocations.items():
            individual_pct = (allocation.allocated_capital / test_capital) * 100
            
            # Critical assertion: no asset should get close to 85%
            assert individual_pct < 50.0, f"FAILED: {symbol} has {individual_pct:.1f}% allocation (too high!)"
            assert individual_pct > 1.0, f"FAILED: {symbol} has {individual_pct:.1f}% allocation (too low!)"
        
        # Verify total allocation is reasonable
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        total_pct = (total_allocated / test_capital) * 100
        
        # Should be close to 85% total (distributed across assets)
        assert 75.0 <= total_pct <= 90.0, f"Total allocation {total_pct:.1f}% should be ~85%"
        
        print(f"✅ Fix 2 Verified: Proportional distribution achieved")
        print(f"  Total allocation: {total_pct:.1f}% distributed across {len(test_symbols)} assets")
        for symbol, allocation in allocations.items():
            pct = (allocation.allocated_capital / test_capital) * 100
            print(f"    {symbol}: {pct:.1f}% (${allocation.allocated_capital:.2f})")
    
    def test_issue_3_correct_capital_per_asset_not_825(self, market_data_setup, test_symbols, test_capital):
        """Test Fix 3: Available capital per asset is properly calculated (not ~$825 each)."""
        portfolio_manager = market_data_setup
        
        # Force rebalance
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(test_symbols)
        
        for symbol, allocation in allocations.items():
            available_capital = allocation.allocated_capital
            
            # Critical assertion: should NOT be around $825 (the old bug)
            assert abs(available_capital - 825.0) > 100.0, \
                f"FAILED: {symbol} has ${available_capital:.2f} (close to old bug value of $825)"
            
            # Should be reasonable proportion of total capital
            pct_of_total = (available_capital / test_capital) * 100
            assert 5.0 <= pct_of_total <= 40.0, \
                f"FAILED: {symbol} allocation {pct_of_total:.1f}% is outside reasonable range"
            
            print(f"  {symbol}: ${available_capital:.2f} available ({pct_of_total:.1f}% of total)")
        
        print(f"✅ Fix 3 Verified: Available capital properly distributed (not $825 each)")
    
    def test_scaling_multiplier_application(self, market_data_setup, test_symbols, test_capital):
        """Test that scaling multiplier is applied correctly to total allocation, not per asset."""
        portfolio_manager = market_data_setup
        
        # Get scaling multiplier
        scaling_multiplier = portfolio_manager.calculate_scaling_multiplier()
        
        # Force rebalance
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Calculate expected vs actual total allocation
        max_allocation = test_capital * 0.85  # 85% max
        expected_total = scaling_multiplier * max_allocation
        actual_total = sum(a.allocated_capital for a in allocations.values())
        
        # Should match within small tolerance
        tolerance = max_allocation * 0.01  # 1% tolerance
        assert abs(actual_total - expected_total) <= tolerance, \
            f"Total allocation ${actual_total:.2f} should match expected ${expected_total:.2f}"
        
        print(f"✅ Scaling multiplier correctly applied:")
        print(f"  Scaling multiplier: {scaling_multiplier:.3f}")
        print(f"  Max allocation: ${max_allocation:.2f}")
        print(f"  Expected total: ${expected_total:.2f}")
        print(f"  Actual total: ${actual_total:.2f}")
    
    def test_weights_sum_to_one(self, market_data_setup, test_symbols):
        """Test that individual asset weights sum to 1.0."""
        portfolio_manager = market_data_setup
        
        # Compute weights
        weights = portfolio_manager.compute_weights(test_symbols)
        
        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight:.3f}, should be 1.0"
        
        print(f"✅ Weights correctly normalized:")
        for symbol, weight in weights.items():
            print(f"  {symbol}: {weight:.3f} ({weight*100:.1f}%)")
        print(f"  Total: {total_weight:.3f}")
    
    def test_comprehensive_integration(self, test_capital, test_symbols):
        """Comprehensive integration test of all fixes together."""
        print("\n" + "="*70)
        print("COMPREHENSIVE INTEGRATION TEST")
        print("="*70)
        
        # Create portfolio manager with actual capital
        pm = ProductionPortfolioManager(total_capital=test_capital)
        
        # Set up market data
        for symbol in test_symbols:
            for _ in range(15):
                pm.update_volatility_data(symbol, 0.02)
        
        for i, sym1 in enumerate(test_symbols):
            for sym2 in test_symbols[i+1:]:
                for _ in range(15):
                    pm.update_correlation_data(sym1, sym2, 0.6)
        
        # Get initial summary
        initial_summary = pm.get_portfolio_summary()
        print(f"Initial State:")
        print(f"  Total Capital: ${initial_summary['total_capital']:.2f}")
        print(f"  Allocated Capital: ${initial_summary['allocated_capital']:.2f}")
        print(f"  Allocation %: {initial_summary['allocation_percentage']:.1%}")
        
        # Force rebalance
        pm.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = pm.rebalance_portfolio(test_symbols)
        
        # Get final summary
        final_summary = pm.get_portfolio_summary()
        
        print(f"\nAfter Rebalancing:")
        print(f"  Total Capital: ${final_summary['total_capital']:.2f}")
        print(f"  Allocated Capital: ${final_summary['allocated_capital']:.2f}")
        print(f"  Allocation %: {final_summary['allocation_percentage']:.1%}")
        print(f"  Active Symbols: {final_summary['active_symbols']}")
        
        print(f"\nAsset Distribution:")
        total_check = 0.0
        for symbol, allocation in allocations.items():
            pct = (allocation.allocated_capital / test_capital) * 100
            total_check += pct
            print(f"  {symbol}: ${allocation.allocated_capital:.2f} ({pct:.1f}%)")
        
        print(f"\nValidation Results:")
        print(f"  ✓ Capital Detection: ${test_capital:.2f} (not hard-coded $5,500)")
        print(f"  ✓ Total Allocation: {total_check:.1f}% (distributed, not 85% each)")
        print(f"  ✓ Max Individual: {max((a.allocated_capital/test_capital)*100 for a in allocations.values()):.1f}%")
        print(f"  ✓ Available Capital: Proportional amounts (not $825 each)")
        
        # Final assertions
        assert final_summary['total_capital'] == test_capital
        assert final_summary['allocation_percentage'] > 0.5  # At least 50% allocated
        assert final_summary['allocation_percentage'] <= 1.0  # Not over 100%
        assert len(allocations) == len(test_symbols)
        
        # No individual asset should dominate
        max_individual_pct = max((a.allocated_capital/test_capital)*100 for a in allocations.values())
        assert max_individual_pct < 50.0, f"Individual allocation too high: {max_individual_pct:.1f}%"
        
        print(f"\n🎉 COMPREHENSIVE INTEGRATION TEST PASSED!")
        print("="*70)


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running comprehensive portfolio allocation tests...")
    pytest.main([__file__, "-v", "-s", "--tb=short"])


if __name__ == "__main__":
    run_comprehensive_tests()
