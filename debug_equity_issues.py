#!/usr/bin/env python3
"""
Advanced Equity Curve Debugging

This script will identify and fix the $274.02 discrepancy issue
by thoroughly testing the equity curve calculation logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

from backtest.visualizer import QuantStatsVisualizer

def create_realistic_test_data():
    """Create realistic test data that mimics the actual backtest scenario."""
    # Create 7 days of 5-minute data (like the actual backtest)
    start = datetime(2024, 1, 1, 0, 0)
    dates = pd.date_range(start, periods=2016, freq='5min')  # 7 days * 288 intervals
    
    # Create realistic price movements for multiple symbols
    np.random.seed(42)
    
    symbols_data = {}
    
    # BTCUSDT: slight uptrend
    btc_prices = [100000]
    for i in range(2015):
        change = np.random.normal(0.0001, 0.005)  # Small drift, realistic volatility
        btc_prices.append(btc_prices[-1] * (1 + change))
    
    symbols_data['BTCUSDT'] = pd.Series(btc_prices, index=dates)
    
    # ETHUSDT: slight downtrend
    eth_prices = [3000]
    for i in range(2015):
        change = np.random.normal(-0.0001, 0.008)
        eth_prices.append(eth_prices[-1] * (1 + change))
    
    symbols_data['ETHUSDT'] = pd.Series(eth_prices, index=dates)
    
    # XRPUSDT: volatile
    xrp_prices = [3.5]
    for i in range(2015):
        change = np.random.normal(-0.0002, 0.01)
        xrp_prices.append(xrp_prices[-1] * (1 + change))
    
    symbols_data['XRPUSDT'] = pd.Series(xrp_prices, index=dates)
    
    return pd.DataFrame(symbols_data)

def create_realistic_trade_sequence():
    """Create a realistic trade sequence that could cause the $274 discrepancy."""
    trades = []
    
    # Scenario: Multiple overlapping positions with fees
    # This mimics the actual backtest pattern
    
    # Trade 1: ETHUSDT long position (profitable)
    trades.extend([
        {
            'timestamp': '2024-01-01 10:00:00',
            'symbol': 'ETHUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 3.0,
            'price': 3000.0,
            'leverage': 10,
            'margin': 900.0,
            'fee': 3.6  # 0.04% of notional
        },
        {
            'timestamp': '2024-01-02 10:00:00',
            'symbol': 'ETHUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 3.0,
            'price': 3100.0,
            'leverage': 10,
            'margin': 900.0,
            'fee': 3.72,  # 0.04% of notional
            'pnl': 300.0  # (3100 - 3000) * 3
        }
    ])
    
    # Trade 2: BTCUSDT long position (losing)
    trades.extend([
        {
            'timestamp': '2024-01-02 14:00:00',
            'symbol': 'BTCUSDT',
            'type': 'open',
            'side': 'buy',
            'contracts': 0.1,
            'price': 100000.0,
            'leverage': 10,
            'margin': 1000.0,
            'fee': 4.0
        },
        {
            'timestamp': '2024-01-03 14:00:00',
            'symbol': 'BTCUSDT',
            'type': 'close',
            'side': 'sell',
            'contracts': 0.1,
            'price': 98000.0,
            'leverage': 10,
            'margin': 1000.0,
            'fee': 3.92,
            'pnl': -200.0  # (98000 - 100000) * 0.1
        }
    ])
    
    # Trade 3: XRPUSDT short position (overlapping with others)
    trades.extend([
        {
            'timestamp': '2024-01-01 16:00:00',
            'symbol': 'XRPUSDT',
            'type': 'open',
            'side': 'sell',
            'contracts': 1000.0,
            'price': 3.5,
            'leverage': 10,
            'margin': 350.0,
            'fee': 1.4
        },
        {
            'timestamp': '2024-01-04 16:00:00',
            'symbol': 'XRPUSDT',
            'type': 'close',
            'side': 'buy',
            'contracts': 1000.0,
            'price': 3.4,
            'leverage': 10,
            'margin': 350.0,
            'fee': 1.36,
            'pnl': 100.0  # (3.5 - 3.4) * 1000
        }
    ])
    
    return pd.DataFrame(trades)

def debug_equity_curve_discrepancy():
    """Debug the specific $274 discrepancy issue."""
    print("="*60)
    print("DEBUGGING EQUITY CURVE DISCREPANCY")
    print("="*60)
    
    visualizer = QuantStatsVisualizer(initial_capital=10000.0)
    
    # Use realistic test data
    price_data = create_realistic_test_data()
    trades = create_realistic_trade_sequence()
    
    print("Test Data Overview:")
    print(f"Price data: {price_data.shape[0]} rows, {price_data.shape[1]} symbols")
    print(f"Trades: {len(trades)} total trades")
    print()
    
    # Calculate ground truth manually
    total_pnl = trades[trades['type'] == 'close']['pnl'].sum()
    total_fees = trades['fee'].sum()
    expected_final_equity = 10000.0 + total_pnl - total_fees
    
    print("Manual Ground Truth Calculation:")
    print(f"Initial capital: $10,000.00")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Total fees: ${total_fees:.2f}")
    print(f"Expected final equity: ${expected_final_equity:.2f}")
    print()
    
    # Test the visualizer's calculation
    print("Visualizer Calculation:")
    equity_curve = visualizer._trades_to_equity_curve(trades, price_data)
    calculated_final_equity = equity_curve.iloc[-1]
    
    discrepancy = abs(calculated_final_equity - expected_final_equity)
    print(f"Calculated final equity: ${calculated_final_equity:.2f}")
    print(f"Discrepancy: ${discrepancy:.2f}")
    
    if discrepancy > 0.01:
        print("❌ ISSUE FOUND: Significant discrepancy detected")
        
        # Detailed debugging
        print("\nDetailed Trade Analysis:")
        for idx, trade in trades.iterrows():
            print(f"Trade {idx}: {trade['timestamp']} - {trade['symbol']} {trade['type']} {trade['side']}")
            if trade['type'] == 'close':
                print(f"  PnL: ${trade['pnl']:.2f}, Fee: ${trade['fee']:.2f}")
            else:
                print(f"  Fee: ${trade['fee']:.2f}")
        
        # Check for specific issues
        analyze_equity_curve_issues(visualizer, trades, price_data, expected_final_equity)
    else:
        print("✅ PASSED: Equity curve calculation is accurate")

def analyze_equity_curve_issues(visualizer, trades, price_data, expected_final_equity):
    """Analyze specific issues in the equity curve calculation."""
    print("\n" + "-"*40)
    print("DETAILED EQUITY CURVE ANALYSIS")
    print("-"*40)
    
    # Recreate the equity curve step by step for debugging
    trades_df = trades.copy()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp')
    
    equity_curve = pd.Series(index=price_data.index, dtype=float, name='equity')
    equity_curve.iloc[:] = 10000.0
    
    cumulative_realized_pnl = 0.0
    cumulative_fees = 0.0
    open_positions = {}
    
    print("Step-by-step processing:")
    
    for trade_num, (_, trade) in enumerate(trades_df.iterrows()):
        print(f"\nProcessing Trade {trade_num + 1}: {trade['timestamp']}")
        print(f"  Symbol: {trade['symbol']}, Type: {trade['type']}, Side: {trade['side']}")
        
        timestamp = trade['timestamp']
        symbol = trade['symbol']
        trade_type = trade['type']
        
        # Find the index closest to this trade timestamp
        try:
            trade_idx = equity_curve.index.get_indexer([timestamp], method='nearest')[0]
            print(f"  Trade index: {trade_idx}")
        except (IndexError, KeyError):
            print(f"  ❌ Could not find index for timestamp {timestamp}")
            continue
            
        if trade_type == 'open':
            fee = trade.get('fee', 0)
            cumulative_fees += fee
            print(f"  Opening position, fee: ${fee:.2f}, cumulative fees: ${cumulative_fees:.2f}")
            
            open_positions[symbol] = {
                'contracts': trade['contracts'] if trade['side'] == 'buy' else -trade['contracts'],
                'entry_price': trade['price'],
            }
            print(f"  Open position: {open_positions[symbol]}")
            
        elif trade_type == 'close':
            if symbol in open_positions:
                realized_pnl = trade.get('pnl', 0)
                fee = trade.get('fee', 0)
                
                cumulative_realized_pnl += realized_pnl
                cumulative_fees += fee
                
                print(f"  Closing position, PnL: ${realized_pnl:.2f}, fee: ${fee:.2f}")
                print(f"  Cumulative realized PnL: ${cumulative_realized_pnl:.2f}")
                print(f"  Cumulative fees: ${cumulative_fees:.2f}")
                
                del open_positions[symbol]
            else:
                print(f"  ❌ Warning: Trying to close non-existent position for {symbol}")
        
        # Calculate equity at this point
        current_equity = 10000.0 + cumulative_realized_pnl - cumulative_fees
        print(f"  Current equity (realized): ${current_equity:.2f}")
        
        # Calculate unrealized P&L for open positions
        unrealized_pnl = 0.0
        if open_positions:
            current_prices = price_data.loc[equity_curve.index[trade_idx]]
            for pos_symbol, pos in open_positions.items():
                if pos_symbol in current_prices:
                    current_price = current_prices[pos_symbol]
                    if not pd.isna(current_price):
                        unrealized = (current_price - pos['entry_price']) * pos['contracts']
                        unrealized_pnl += unrealized
                        print(f"    {pos_symbol} unrealized P&L: ${unrealized:.2f}")
        
        total_equity = current_equity + unrealized_pnl
        print(f"  Total equity (realized + unrealized): ${total_equity:.2f}")
    
    # Final comparison
    final_equity_manual = 10000.0 + cumulative_realized_pnl - cumulative_fees
    print(f"\nFinal Manual Calculation:")
    print(f"  Initial: $10,000.00")
    print(f"  Realized P&L: ${cumulative_realized_pnl:.2f}")
    print(f"  Fees: ${cumulative_fees:.2f}")
    print(f"  Final: ${final_equity_manual:.2f}")
    print(f"  Expected: ${expected_final_equity:.2f}")
    print(f"  Discrepancy: ${abs(final_equity_manual - expected_final_equity):.2f}")

def run_comprehensive_equity_debugging():
    """Run comprehensive debugging of the equity curve issues."""
    print("="*60)
    print("COMPREHENSIVE EQUITY CURVE DEBUG")
    print("="*60)
    
    debug_equity_curve_discrepancy()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_comprehensive_equity_debugging()
