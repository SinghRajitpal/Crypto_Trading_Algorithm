#!/usr/bin/env python3
"""
Focused Backtesting Test - Identifying Core Issues

This script focuses on the specific issues found:
1. Daily resampling destroying returns calculation
2. Returns calculation discrepancies
3. Metrics showing impossible values (positive Sharpe with negative returns)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

from backtest.visualizer import QuantStatsVisualizer

def create_simple_test_data():
    """Create minimal test data to isolate the problem."""
    # Create 2 days of 5-minute data
    start = datetime(2024, 1, 1, 9, 0)
    dates = pd.date_range(start, periods=10, freq='5min')
    
    # Simple price data: starts at 100, ends at 110
    prices = np.linspace(100, 110, 10)
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    return df

def create_simple_trades():
    """Create one simple winning trade."""
    trades = pd.DataFrame([
        {
            'timestamp': '2024-01-01 09:00:00',
            'symbol': 'BTCUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 1.0,
            'price': 100.0,
            'leverage': 1,
            'margin': 100.0,
            'fee': 0.0  # Remove fees to simplify
        },
        {
            'timestamp': '2024-01-01 09:30:00',
            'symbol': 'BTCUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 1.0,
            'price': 110.0,
            'leverage': 1,
            'margin': 100.0,
            'fee': 0.0,  # Remove fees to simplify
            'pnl': 10.0  # Simple $10 profit
        }
    ])
    
    return trades

def debug_returns_calculation():
    """Debug the returns calculation step by step."""
    print("="*60)
    print("DEBUGGING RETURNS CALCULATION")
    print("="*60)
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    
    # Create simple test data
    price_data = create_simple_test_data()
    trades = create_simple_trades()
    
    print("Price Data:")
    print(price_data.head())
    print(f"Price data shape: {price_data.shape}")
    
    print("\nTrades:")
    print(trades)
    
    # Step 1: Calculate equity curve
    print("\n" + "-"*40)
    print("STEP 1: EQUITY CURVE CALCULATION")
    print("-"*40)
    
    equity_curve = visualizer._trades_to_equity_curve(trades, price_data)
    print(f"Equity curve shape: {equity_curve.shape}")
    print(f"Initial equity: ${equity_curve.iloc[0]:.2f}")
    print(f"Final equity: ${equity_curve.iloc[-1]:.2f}")
    print(f"Expected final equity: $1010.00 (1000 + 10 PnL)")
    
    # Step 2: Manual returns calculation
    print("\n" + "-"*40)
    print("STEP 2: MANUAL RETURNS CALCULATION")
    print("-"*40)
    
    manual_total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    print(f"Manual total return: {manual_total_return:.4f} ({manual_total_return*100:.2f}%)")
    
    # Step 3: Visualizer returns calculation
    print("\n" + "-"*40)
    print("STEP 3: VISUALIZER RETURNS CALCULATION")
    print("-"*40)
    
    returns, _ = visualizer._trades_to_returns(trades, price_data)
    
    if not returns.empty:
        print(f"Returns shape: {returns.shape}")
        print("Returns values:")
        print(returns)
        
        calculated_total_return = (1 + returns).prod() - 1
        print(f"Calculated total return: {calculated_total_return:.4f} ({calculated_total_return*100:.2f}%)")
        
        # Check if resampling is the issue
        print("\n" + "-"*20)
        print("INVESTIGATING DAILY RESAMPLING")
        print("-"*20)
        
        # Check the time span
        time_span = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 3600  # hours
        obs_per_day = len(equity_curve) / (time_span / 24)
        print(f"Time span: {time_span:.1f} hours")
        print(f"Observations per day: {obs_per_day:.1f}")
        
        if obs_per_day > 2:  # This triggers daily resampling
            print("⚠️  Daily resampling will be triggered!")
            
            # Manual daily resampling
            daily_equity = equity_curve.resample('D').last().dropna()
            print(f"Daily equity after resampling:")
            print(daily_equity)
            
            if len(daily_equity) > 1:
                daily_returns = daily_equity.pct_change().dropna()
                print(f"Daily returns:")
                print(daily_returns)
                
                daily_total_return = (1 + daily_returns).prod() - 1
                print(f"Daily resampled total return: {daily_total_return:.4f} ({daily_total_return*100:.2f}%)")
            else:
                print("Only one daily observation - no returns possible!")
    else:
        print("❌ No returns calculated!")

def test_metrics_with_known_data():
    """Test metrics calculation with known expected results."""
    print("\n" + "="*60)
    print("TESTING METRICS WITH KNOWN DATA")
    print("="*60)
    
    # Create manual returns that we know should be negative
    negative_returns = pd.Series([-0.01, -0.02, -0.015], name='returns')  # -1%, -2%, -1.5%
    
    visualizer = QuantStatsVisualizer(initial_capital=1000.0)
    metrics = visualizer._extract_quantstats_metrics(negative_returns)
    
    print("Test returns (should all be negative):")
    print(negative_returns)
    
    cumulative_return = (1 + negative_returns).prod() - 1
    print(f"Manual cumulative return: {cumulative_return:.4f} ({cumulative_return*100:.2f}%)")
    
    print("\nCalculated metrics:")
    for key, value in metrics.items():
        if key in ['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio']:
            print(f"{key}: {value}")
    
    # Validation
    total_return = metrics.get('Total Return (%)', 0)
    if total_return < 0:
        print("✅ PASSED: Total return is correctly negative")
    else:
        print("❌ FAILED: Total return should be negative")

def run_focused_test():
    """Run the focused test suite."""
    print("="*60)
    print("FOCUSED BACKTESTING ACCURACY TEST")
    print("="*60)
    
    debug_returns_calculation()
    test_metrics_with_known_data()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_focused_test()
