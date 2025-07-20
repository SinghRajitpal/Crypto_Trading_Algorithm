#!/usr/bin/env python3
"""
Test NaN Handling Fix

Test that the visualizer correctly handles NaN values in PnL calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

from backtest.visualizer import QuantStatsVisualizer

def test_nan_pnl_handling():
    """Test that NaN PnL values are handled correctly."""
    print("="*60)
    print("TESTING NaN PnL HANDLING")
    print("="*60)
    
    # Create test trades with NaN PnL values (like real data)
    trades_data = [
        # Open trades (should have NaN PnL)
        {
            'timestamp': '2024-01-01 10:00:00',
            'symbol': 'BTCUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 0.1,
            'price': 100000.0,
            'leverage': 10,
            'margin': 1000.0,
            'fee': 4.0,
            'pnl': np.nan  # This should be NaN for open trades
        },
        # Close trades (should have real PnL)
        {
            'timestamp': '2024-01-02 10:00:00',
            'symbol': 'BTCUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 0.1,
            'price': 102000.0,
            'leverage': 10,
            'margin': 1000.0,
            'fee': 4.08,
            'pnl': 200.0  # (102000 - 100000) * 0.1
        },
        # Another open trade with NaN
        {
            'timestamp': '2024-01-03 10:00:00',
            'symbol': 'ETHUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 3.0,
            'price': 3000.0,
            'leverage': 10,
            'margin': 900.0,
            'fee': 3.6,
            'pnl': np.nan  # This should be NaN for open trades
        },
        # Close with loss
        {
            'timestamp': '2024-01-04 10:00:00',
            'symbol': 'ETHUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 3.0,
            'price': 2900.0,
            'leverage': 10,
            'margin': 900.0,
            'fee': 3.48,
            'pnl': -300.0  # (2900 - 3000) * 3
        }
    ]
    
    trades_df = pd.DataFrame(trades_data)
    
    # Create simple price data
    dates = pd.date_range('2024-01-01', '2024-01-05', freq='H')
    price_data = pd.DataFrame({
        'BTCUSDT': [100000 + i * 10 for i in range(len(dates))],
        'ETHUSDT': [3000 - i * 1 for i in range(len(dates))]
    }, index=dates)
    
    print("Test setup:")
    print(f"Trades: {len(trades_df)} total")
    print(f"Open trades: {len(trades_df[trades_df['type'] == 'open'])}")
    print(f"Close trades: {len(trades_df[trades_df['type'] == 'close'])}")
    print(f"NaN PnL values: {trades_df['pnl'].isna().sum()}")
    print()
    
    # Expected calculation
    close_trades = trades_df[trades_df['type'] == 'close']
    expected_pnl = close_trades['pnl'].sum()  # Should be 200 - 300 = -100
    expected_fees = trades_df['fee'].sum()  # Should be 4 + 4.08 + 3.6 + 3.48 = 15.16
    expected_final = 10000 + expected_pnl - expected_fees  # 10000 - 100 - 15.16 = 9884.84
    
    print("Expected results:")
    print(f"Total PnL: ${expected_pnl:.2f}")
    print(f"Total fees: ${expected_fees:.2f}")
    print(f"Expected final equity: ${expected_final:.2f}")
    print()
    
    # Test the visualizer
    visualizer = QuantStatsVisualizer(initial_capital=10000.0)
    
    print("Testing visualizer calculation:")
    try:
        returns, benchmark_returns = visualizer._trades_to_returns(trades_df, price_data)
        print("✅ _trades_to_returns completed without error")
        
        equity_curve = visualizer._trades_to_equity_curve(trades_df, price_data)
        calculated_final = equity_curve.iloc[-1]
        
        print(f"Calculated final equity: ${calculated_final:.2f}")
        
        discrepancy = abs(calculated_final - expected_final)
        print(f"Discrepancy: ${discrepancy:.2f}")
        
        if discrepancy < 0.01:
            print("✅ PASSED: NaN handling is working correctly")
        else:
            print("❌ FAILED: Still have discrepancy with NaN handling")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_real_trades_data():
    """Test with the actual trades data from the last backtest."""
    print("\n" + "="*60)
    print("TESTING WITH REAL TRADES DATA")
    print("="*60)
    
    # Load the actual trades
    trades_file = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/backtest/results/ma_crossover/20250720_180436/trade_log.csv"
    
    if not os.path.exists(trades_file):
        print("❌ Cannot find real trades file")
        return
    
    trades_df = pd.read_csv(trades_file)
    print(f"Loaded {len(trades_df)} real trades")
    
    # Load some price data for testing
    cache_dir = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/data/cache"
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    
    price_data = {}
    for symbol in symbols:
        cache_file = os.path.join(cache_dir, f"{symbol}-5m.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            price_data[symbol] = df['close']
    
    if not price_data:
        print("❌ Could not load price data")
        return
    
    price_df = pd.DataFrame(price_data).ffill()
    
    # Expected from the terminal output
    expected_pnl = -1644.89
    expected_fees = 283.49
    expected_final = 8071.63
    
    print(f"Expected from terminal: PnL=${expected_pnl:.2f}, Fees=${expected_fees:.2f}, Final=${expected_final:.2f}")
    
    # Test the visualizer
    visualizer = QuantStatsVisualizer(initial_capital=10000.0)
    
    try:
        returns, benchmark_returns = visualizer._trades_to_returns(trades_df, price_df)
        print("✅ Real data processing completed")
        
        # The terminal output should now match more closely
        
    except Exception as e:
        print(f"❌ ERROR processing real data: {e}")
        import traceback
        traceback.print_exc()

def run_nan_handling_tests():
    """Run all NaN handling tests."""
    print("🧪 TESTING NaN PnL HANDLING FIXES")
    
    test_nan_pnl_handling()
    test_real_trades_data()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_nan_handling_tests()
