#!/usr/bin/env python3
"""
Comprehensive Backtesting Accuracy Test Suite

This script tests the fundamental accuracy of our backtesting system by:
1. Creating synthetic trades with known outcomes
2. Verifying equity curve calculations are correct
3. Testing metric calculations for consistency
4. Identifying discrepancies between actual vs calculated performance

The goal is to ensure our backtesting system produces mathematically correct results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

from backtest.visualizer import QuantStatsVisualizer
from backtest.broker import SimBroker

def create_synthetic_price_data(start_date, end_date, initial_price=100.0, volatility=0.02):
    """Create synthetic price data for testing."""
    dates = pd.date_range(start_date, end_date, freq='5min')
    
    # Generate random returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility, len(dates))
    
    # Create price series
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    }, index=dates)
    
    return df

def create_test_trades_simple_long():
    """Create a simple long trade with known PnL."""
    trades = []
    
    # Open long position: Buy 1 BTC at $100
    trades.append({
        'timestamp': '2024-01-01 10:00:00',
        'symbol': 'BTCUSDT',
        'type': 'open',
        'side': 'buy',
        'contracts': 1.0,
        'price': 100.0,
        'leverage': 1,
        'margin': 100.0,  # 1x leverage, so margin = notional
        'fee': 0.1  # 0.1% fee
    })
    
    # Close long position: Sell 1 BTC at $110
    trades.append({
        'timestamp': '2024-01-01 11:00:00',
        'symbol': 'BTCUSDT',
        'type': 'close',
        'side': 'sell',
        'contracts': 1.0,
        'price': 110.0,
        'leverage': 1,
        'margin': 100.0,
        'fee': 0.11,  # 0.1% of 110
        'pnl': 10.0  # (110 - 100) * 1
    })
    
    return pd.DataFrame(trades)

def create_test_trades_simple_short():
    """Create a simple short trade with known PnL."""
    trades = []
    
    # Open short position: Sell 1 BTC at $100
    trades.append({
        'timestamp': '2024-01-01 10:00:00',
        'symbol': 'BTCUSDT',
        'type': 'open',
        'side': 'sell',
        'contracts': 1.0,
        'price': 100.0,
        'leverage': 1,
        'margin': 100.0,
        'fee': 0.1
    })
    
    # Close short position: Buy 1 BTC at $90
    trades.append({
        'timestamp': '2024-01-01 11:00:00',
        'symbol': 'BTCUSDT',
        'type': 'close',
        'side': 'buy',
        'contracts': 1.0,
        'price': 90.0,
        'leverage': 1,
        'margin': 100.0,
        'fee': 0.09,
        'pnl': 10.0  # (100 - 90) * 1
    })
    
    return pd.DataFrame(trades)

def create_test_trades_losing():
    """Create losing trades to test negative returns."""
    trades = []
    
    # Losing long trade
    trades.append({
        'timestamp': '2024-01-01 10:00:00',
        'symbol': 'BTCUSDT',
        'type': 'open',
        'side': 'buy',
        'contracts': 1.0,
        'price': 100.0,
        'leverage': 1,
        'margin': 100.0,
        'fee': 0.1
    })
    
    trades.append({
        'timestamp': '2024-01-01 11:00:00',
        'symbol': 'BTCUSDT',
        'type': 'close',
        'side': 'sell',
        'contracts': 1.0,
        'price': 80.0,
        'leverage': 1,
        'margin': 100.0,
        'fee': 0.08,
        'pnl': -20.0  # (80 - 100) * 1
    })
    
    return pd.DataFrame(trades)

def create_test_trades_multiple():
    """Create multiple trades with mixed outcomes."""
    trades = []
    
    # Trade 1: Winning long
    trades.extend([
        {
            'timestamp': '2024-01-01 10:00:00',
            'symbol': 'BTCUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 1.0,
            'price': 100.0,
            'leverage': 1,
            'margin': 100.0,
            'fee': 0.1
        },
        {
            'timestamp': '2024-01-01 11:00:00',
            'symbol': 'BTCUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 1.0,
            'price': 105.0,
            'leverage': 1,
            'margin': 100.0,
            'fee': 0.105,
            'pnl': 5.0
        }
    ])
    
    # Trade 2: Losing short
    trades.extend([
        {
            'timestamp': '2024-01-02 10:00:00',
            'symbol': 'BTCUSDT',
            'type': 'open',
            'side': 'sell',
            'contracts': 1.0,
            'price': 105.0,
            'leverage': 1,
            'margin': 105.0,
            'fee': 0.105
        },
        {
            'timestamp': '2024-01-02 11:00:00',
            'symbol': 'BTCUSDT',
            'type': 'close',
            'side': 'buy',
            'contracts': 1.0,
            'price': 108.0,
            'leverage': 1,
            'margin': 105.0,
            'fee': 0.108,
            'pnl': -3.0  # (105 - 108) * 1
        }
    ])
    
    return pd.DataFrame(trades)

def test_equity_curve_calculation():
    """Test that equity curve calculation is mathematically correct."""
    print("\n" + "="*60)
    print("TESTING EQUITY CURVE CALCULATION")
    print("="*60)
    
    # Create synthetic price data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    price_data = create_synthetic_price_data(start_date, end_date, initial_price=100.0)
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    
    # Test 1: Simple winning long trade
    print("\nTest 1: Simple Winning Long Trade")
    print("-" * 40)
    trades = create_test_trades_simple_long()
    print("Trade details:")
    print(trades.to_string())
    
    expected_pnl = 10.0
    expected_fees = 0.1 + 0.11
    expected_final_equity = 1000.0 + expected_pnl - expected_fees
    print(f"Expected final equity: ${expected_final_equity:.2f}")
    
    # Calculate equity curve
    equity_curve = visualizer._trades_to_equity_curve(trades, price_data)
    actual_final_equity = equity_curve.iloc[-1]
    print(f"Calculated final equity: ${actual_final_equity:.2f}")
    
    discrepancy = abs(actual_final_equity - expected_final_equity)
    print(f"Discrepancy: ${discrepancy:.2f}")
    
    if discrepancy < 0.01:
        print("✅ PASSED: Equity curve calculation is correct")
    else:
        print("❌ FAILED: Equity curve calculation is incorrect")
    
    # Test 2: Simple losing trade
    print("\nTest 2: Simple Losing Trade")
    print("-" * 40)
    trades = create_test_trades_losing()
    print("Trade details:")
    print(trades.to_string())
    
    expected_pnl = -20.0
    expected_fees = 0.1 + 0.08
    expected_final_equity = 1000.0 + expected_pnl - expected_fees
    print(f"Expected final equity: ${expected_final_equity:.2f}")
    
    equity_curve = visualizer._trades_to_equity_curve(trades, price_data)
    actual_final_equity = equity_curve.iloc[-1]
    print(f"Calculated final equity: ${actual_final_equity:.2f}")
    
    discrepancy = abs(actual_final_equity - expected_final_equity)
    print(f"Discrepancy: ${discrepancy:.2f}")
    
    if discrepancy < 0.01:
        print("✅ PASSED: Losing trade calculation is correct")
    else:
        print("❌ FAILED: Losing trade calculation is incorrect")

def test_returns_calculation():
    """Test that returns calculation matches expected values."""
    print("\n" + "="*60)
    print("TESTING RETURNS CALCULATION")
    print("="*60)
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    
    # Test with multiple trades
    trades = create_test_trades_multiple()
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    price_data = create_synthetic_price_data(start_date, end_date)
    
    # Calculate expected total return manually
    total_pnl = trades[trades['type'] == 'close']['pnl'].sum()
    total_fees = trades['fee'].sum()
    expected_total_return = (total_pnl - total_fees) / 1000.0
    
    print(f"Manual calculation:")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Total Fees: ${total_fees:.2f}")
    print(f"Expected total return: {expected_total_return:.4f} ({expected_total_return*100:.2f}%)")
    
    # Calculate using visualizer
    returns, _ = visualizer._trades_to_returns(trades, price_data)
    
    if not returns.empty:
        calculated_total_return = (1 + returns).prod() - 1
        print(f"Calculated total return: {calculated_total_return:.4f} ({calculated_total_return*100:.2f}%)")
        
        discrepancy = abs(calculated_total_return - expected_total_return)
        print(f"Discrepancy: {discrepancy:.4f}")
        
        if discrepancy < 0.01:
            print("✅ PASSED: Returns calculation is correct")
        else:
            print("❌ FAILED: Returns calculation is incorrect")
    else:
        print("❌ FAILED: No returns calculated")

def test_metrics_consistency():
    """Test that metrics are mathematically consistent."""
    print("\n" + "="*60)
    print("TESTING METRICS CONSISTENCY")
    print("="*60)
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    
    # Test case: Losing trades should have negative Sharpe and Sortino
    print("\nTest: Losing Trades - Metrics Should Be Negative")
    print("-" * 50)
    
    trades = create_test_trades_losing()
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    price_data = create_synthetic_price_data(start_date, end_date)
    
    returns, _ = visualizer._trades_to_returns(trades, price_data)
    metrics = visualizer._extract_quantstats_metrics(returns)
    
    print("Calculated metrics:")
    for key, value in metrics.items():
        if key in ['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio']:
            print(f"{key}: {value}")
    
    # Validation checks
    total_return = metrics.get('Total Return (%)', 0)
    sharpe = metrics.get('Sharpe Ratio', 0)
    sortino = metrics.get('Sortino Ratio', 0)
    
    print("\nValidation:")
    
    # Check 1: Negative returns should result in negative total return
    if total_return < 0:
        print("✅ PASSED: Total return is negative for losing trade")
    else:
        print("❌ FAILED: Total return should be negative for losing trade")
    
    # Check 2: For consistently losing strategies, Sharpe should be negative
    if returns.mean() < 0 and sharpe <= 0:
        print("✅ PASSED: Sharpe ratio is negative/zero for losing strategy")
    elif returns.mean() < 0 and sharpe > 0:
        print("❌ FAILED: Sharpe ratio should be negative for consistently losing strategy")
    else:
        print("⚠️  WARNING: Edge case - need to investigate Sharpe calculation")
    
    # Check 3: Sortino should follow similar logic
    if returns.mean() < 0 and sortino <= 0:
        print("✅ PASSED: Sortino ratio is negative/zero for losing strategy")
    elif returns.mean() < 0 and sortino > 0:
        print("❌ FAILED: Sortino ratio should be negative for consistently losing strategy")
    else:
        print("⚠️  WARNING: Edge case - need to investigate Sortino calculation")

def test_broker_accuracy():
    """Test the broker's trade execution and PnL calculation."""
    print("\n" + "="*60)
    print("TESTING BROKER ACCURACY")
    print("="*60)
    
    broker = SimBroker(initial_capital=10000.0)
    
    # Test 1: Simple long position
    print("\nTest 1: Long Position PnL Calculation")
    print("-" * 40)
    
    # Open long position
    result = broker.open_position(
        symbol="BTCUSDT",
        side="buy",
        contracts=1.0,
        price=100.0,
        leverage=1,
        stop_loss=95.0,
        take_profit=110.0
    )
    
    print(f"Position opened: {result}")
    print(f"Available balance: ${broker.available_balance:.2f}")
    print(f"Reserved margin: ${broker.reserved_margin:.2f}")
    
    # Close position at profit
    close_result = broker.close_position("BTCUSDT", 110.0, slippage_bp=0)
    print(f"Position closed: {close_result}")
    print(f"Final balance: ${broker.available_balance:.2f}")
    
    # Expected: Started with $10,000, should have ~$10,009.80 after fees
    expected_pnl = 10.0  # (110 - 100) * 1
    expected_fees = 0.1 + 0.11  # 0.1% each way
    expected_final = 10000 + expected_pnl - expected_fees
    
    print(f"Expected final balance: ${expected_final:.2f}")
    discrepancy = abs(broker.available_balance - expected_final)
    
    if discrepancy < 0.01:
        print("✅ PASSED: Broker PnL calculation is correct")
    else:
        print("❌ FAILED: Broker PnL calculation is incorrect")

def test_edge_cases():
    """Test edge cases that might cause issues."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    
    # Test 1: Empty trades
    print("\nTest 1: Empty Trades DataFrame")
    print("-" * 35)
    
    empty_trades = pd.DataFrame()
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    price_data = create_synthetic_price_data(start_date, end_date)
    
    try:
        returns, _ = visualizer._trades_to_returns(empty_trades, price_data)
        metrics = visualizer._extract_quantstats_metrics(returns)
        print("✅ PASSED: Empty trades handled gracefully")
    except Exception as e:
        print(f"❌ FAILED: Empty trades caused error: {e}")
    
    # Test 2: Single trade
    print("\nTest 2: Single Trade")
    print("-" * 20)
    
    single_trade = create_test_trades_simple_long().iloc[:1]  # Just the open
    
    try:
        equity_curve = visualizer._trades_to_equity_curve(single_trade, price_data)
        print("✅ PASSED: Single trade handled gracefully")
    except Exception as e:
        print(f"❌ FAILED: Single trade caused error: {e}")

def run_full_test_suite():
    """Run the complete test suite."""
    print("="*80)
    print("BACKTESTING ACCURACY TEST SUITE")
    print("="*80)
    print("Testing fundamental accuracy of backtesting calculations...")
    
    # Run all tests
    test_equity_curve_calculation()
    test_returns_calculation()
    test_metrics_consistency()
    test_broker_accuracy()
    test_edge_cases()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\nReview the results above to identify any calculation errors.")
    print("All discrepancies should be investigated and fixed before production use.")

if __name__ == "__main__":
    run_full_test_suite()
