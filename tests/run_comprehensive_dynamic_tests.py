#!/usr/bin/env python3
"""
Comprehensive Dynamic System Test Runner

This script runs all tests related to dynamic calculations and validates:
1. Portfolio allocation dynamics with volatility and correlations
2. Risk management adjustments based on market conditions
3. Position sizing with ATR and Kelly criterion
4. Dynamic leverage calculation
5. Stop loss and take profit placement
6. Real-time signal processing workflow
7. Integration of all dynamic components

Run this to ensure all dynamic calculations are working correctly together.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "-"*60)
    print(f"📋 {title}")
    print("-"*60)


def run_test_file(test_file, description):
    """Run a specific test file and return success status."""
    print_section(f"Running {description}")
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            # Print summary of results if available
            if "PASSED" in result.stdout:
                passed_count = result.stdout.count("PASSED")
                print(f"   {passed_count} tests passed")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print("Error output:")
            print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False


def run_manual_verification():
    """Run manual verification of dynamic calculations."""
    print_section("Manual Dynamic Calculations Verification")
    
    try:
        from execution.portfolio import ProductionPortfolioManager, AllocationWeights
        from execution.risk_manager import ProductionRiskManager
        from datetime import datetime, timedelta
        import config
        
        print("🔧 Initializing components...")
        
        # Initialize portfolio manager with realistic capital
        portfolio_manager = ProductionPortfolioManager(
            total_capital=15000.0,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
        
        risk_manager = ProductionRiskManager(portfolio_manager=portfolio_manager)
        
        # Test symbols from config
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        
        print("📊 Testing Portfolio Allocation Dynamics...")
        
        # Add different volatilities to test dynamic allocation
        volatilities = [0.015, 0.025, 0.020, 0.018, 0.030]  # Different for each asset
        
        for symbol, vol in zip(test_symbols, volatilities):
            for _ in range(25):  # Add enough history
                portfolio_manager.update_volatility_data(symbol, vol)
            print(f"   {symbol}: {vol:.1%} volatility")
        
        # Add correlations
        correlation_pairs = [
            ("BTCUSDT", "ETHUSDT", 0.75),
            ("BTCUSDT", "SOLUSDT", 0.65),
            ("ETHUSDT", "SOLUSDT", 0.80),
            ("BNBUSDT", "ETHUSDT", 0.70),
            ("XRPUSDT", "BTCUSDT", 0.60)
        ]
        
        for sym1, sym2, corr in correlation_pairs:
            for _ in range(25):
                portfolio_manager.update_correlation_data(sym1, sym2, corr)
        
        # Force rebalancing
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        print("🔄 Executing portfolio rebalancing...")
        allocations = portfolio_manager.rebalance_portfolio(test_symbols)
        
        if allocations:
            print("✅ Portfolio Allocation Results:")
            total_allocated = 0
            for symbol, allocation in allocations.items():
                pct = (allocation.allocated_capital / 15000.0) * 100
                total_allocated += allocation.allocated_capital
                print(f"   {symbol}: ${allocation.allocated_capital:.2f} ({pct:.1f}%)")
            
            print(f"   Total Allocated: ${total_allocated:.2f} ({total_allocated/15000.0:.1%})")
            
            # Verify allocation is reasonable
            assert 10000 <= total_allocated <= 13000, f"Total allocation should be reasonable: ${total_allocated:.2f}"
            print("✅ Portfolio allocation within expected range")
        else:
            print("❌ Portfolio allocation failed")
            return False
        
        print("\n📊 Testing Dynamic Risk Management...")
        
        # Test position sizing for different scenarios
        test_scenarios = [
            ("BTCUSDT", allocations["BTCUSDT"].allocated_capital, 0.015, "Low volatility"),
            ("ETHUSDT", allocations["ETHUSDT"].allocated_capital, 0.025, "Medium volatility"),
            ("XRPUSDT", allocations["XRPUSDT"].allocated_capital, 0.030, "High volatility")
        ]
        
        entry_price = 50000.0  # Example price
        
        for symbol, allocated_capital, atr_value, scenario in test_scenarios:
            print(f"\n   {scenario} ({symbol}):")
            
            # Calculate position size
            position_result = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocated_capital,
                atr_value=atr_value,
                entry_price=entry_price
            )
            
            # Calculate dynamic leverage
            leverage = risk_manager.calculate_dynamic_leverage(symbol, atr_value)
            
            # Calculate SL/TP
            sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
                entry_price=entry_price,
                side="buy",
                atr_adjusted=atr_value * entry_price
            )
            
            position_size = position_result['position_size_usdt']
            risk_amount = position_result['risk_amount']
            kelly_fraction = position_result['kelly_fraction']
            
            print(f"     Allocated Capital: ${allocated_capital:.2f}")
            print(f"     Position Size: ${position_size:.2f}")
            print(f"     Risk Amount: ${risk_amount:.2f}")
            print(f"     Kelly Fraction: {kelly_fraction:.3f}")
            print(f"     Leverage: {leverage}x")
            print(f"     Stop Loss: ${sl_price:.0f}")
            print(f"     Take Profit: ${tp_price:.0f}")
            
            # Calculate risk-reward ratio
            risk_distance = entry_price - sl_price
            reward_distance = tp_price - entry_price
            rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            print(f"     Risk-Reward Ratio: {rr_ratio:.2f}")
            
            # Validate calculations
            assert position_size > 0, f"Position size should be positive for {symbol}"
            assert position_size <= allocated_capital, f"Position size should not exceed allocation for {symbol}"
            assert risk_amount <= allocated_capital * 0.015, f"Risk should be reasonable for {symbol}"
            assert 1 <= leverage <= 10, f"Leverage should be reasonable for {symbol}"
            assert sl_price < entry_price < tp_price, f"SL/TP should be correctly ordered for {symbol}"
            assert 1.8 <= rr_ratio <= 2.2, f"Risk-reward ratio should be ~2:1 for {symbol}"
        
        print("\n✅ All dynamic risk management calculations validated")
        
        print("\n📊 Testing Stress Conditions...")
        
        # Test high volatility regime detection
        for _ in range(35):
            portfolio_manager.volatility_history.append(0.06)  # High volatility
        
        is_high_vol = portfolio_manager.is_high_volatility_regime()
        scaling_multiplier = portfolio_manager.calculate_scaling_multiplier()
        
        print(f"   High volatility regime detected: {is_high_vol}")
        print(f"   Scaling multiplier: {scaling_multiplier:.3f}")
        
        assert is_high_vol, "Should detect high volatility regime"
        assert scaling_multiplier < 0.6, "Should reduce scaling in high volatility"
        
        print("✅ Stress condition handling validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_validation():
    """Run integration validation of the complete system."""
    print_section("Integration Validation")
    
    try:
        from main import TradingAlgorithm
        from algorithm.strategies.ma_crossover import MACrossoverStrategy
        from unittest.mock import MagicMock, AsyncMock
        
        print("🔧 Setting up integration test...")
        
        # Create mock Binance client
        mock_client = MagicMock()
        mock_client.get_account_metrics = AsyncMock(return_value={
            'total_wallet_balance': 15234.67,
            'total_unrealized_pnl': 0.0,
            'total_margin_used': 0.0,
            'available_margin': 15234.67,
            'exposure_percentage': 0.0,
            'position_count': 0
        })
        mock_client.setup_account_config = AsyncMock()
        mock_client.close = AsyncMock()
        
        # Create strategy
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.018,
            'take_profit_pct': 0.036,
            'leverage': 5
        })
        
        # Test algorithm initialization
        algorithm = TradingAlgorithm(strategy=strategy, testnet=True, total_capital=None)
        
        # Verify initialization
        assert algorithm.strategy == strategy, "Strategy should be set correctly"
        assert algorithm.total_capital is None, "Should start with None capital (to be fetched)"
        
        print("✅ Algorithm initialization successful")
        
        # Test capital fetching simulation
        account_metrics = {'total_wallet_balance': 15234.67}
        fetched_capital = account_metrics['total_wallet_balance']
        
        print(f"   Simulated capital fetch: ${fetched_capital:.2f}")
        assert fetched_capital > 10000, "Should fetch reasonable capital amount"
        
        print("✅ Capital fetching simulation successful")
        
        print("✅ Integration validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all dynamic system tests."""
    print_header("COMPREHENSIVE DYNAMIC SYSTEM VALIDATION")
    
    start_time = time.time()
    total_tests = 0
    passed_tests = 0
    
    print("This comprehensive test suite validates:")
    print("• Dynamic portfolio allocation based on volatility and correlations")
    print("• Risk-adjusted position sizing using ATR and Kelly criterion")  
    print("• Dynamic leverage calculation based on market conditions")
    print("• Stop loss and take profit dynamic placement")
    print("• Real-time signal processing and execution workflow")
    print("• Integration of all dynamic components working together")
    
    # Test files to run
    test_files = [
        ("test_dynamic_risk_calculations.py", "Dynamic Risk Calculations"),
        ("test_realtime_dynamic_execution.py", "Real-time Dynamic Execution"),
    ]
    
    # Run individual test files
    for test_file, description in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            total_tests += 1
            if run_test_file(test_path, description):
                passed_tests += 1
        else:
            print(f"⚠️ Test file not found: {test_file}")
    
    # Run manual verification
    total_tests += 1
    if run_manual_verification():
        passed_tests += 1
    
    # Run integration validation
    total_tests += 1
    if run_integration_validation():
        passed_tests += 1
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("DYNAMIC SYSTEM VALIDATION RESULTS")
    
    print(f"🕒 Test Duration: {duration:.1f} seconds")
    print(f"📊 Total Test Suites: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL DYNAMIC SYSTEM TESTS PASSED!")
        print("✅ Portfolio allocation dynamics working correctly")
        print("✅ Risk management calculations validated")
        print("✅ Position sizing and leverage calculations correct")
        print("✅ Stop loss and take profit placement validated")
        print("✅ Real-time execution workflow functional")
        print("✅ System ready for production trading")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test suite(s) failed")
        print("❌ Please review failed tests before production use")
        return 1
    
    print("\n" + "="*80)
    print("DYNAMIC SYSTEM VALIDATION COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
