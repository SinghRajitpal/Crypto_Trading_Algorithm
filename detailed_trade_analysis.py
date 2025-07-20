#!/usr/bin/env python3
"""
Trade-by-Trade Equity Analysis

Examine each trade in the equity curve calculation to find
where the $274 discrepancy comes from.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

def load_real_trades_and_prices():
    """Load the real trades and price data."""
    trades_file = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/backtest/results/ma_crossover/20250720_180436/trade_log.csv"
    
    if not os.path.exists(trades_file):
        print("❌ Cannot find trades file")
        return None, None
    
    trades_df = pd.read_csv(trades_file)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Load price data
    cache_dir = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/data/cache"
    symbols = trades_df['symbol'].unique()
    
    price_data = {}
    for symbol in symbols:
        cache_file = os.path.join(cache_dir, f"{symbol}-5m.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            price_data[symbol] = df['close']
    
    if not price_data:
        print("❌ Could not load price data")
        return trades_df, None
    
    price_df = pd.DataFrame(price_data).ffill()
    return trades_df, price_df

def manual_equity_curve_calculation(trades_df, price_data):
    """Manually calculate equity curve step by step to debug."""
    print("="*60)
    print("MANUAL EQUITY CURVE CALCULATION")
    print("="*60)
    
    # Sort trades by timestamp
    trades_df = trades_df.copy()
    trades_df = trades_df.sort_values('timestamp')
    
    # Start with initial capital
    equity = 10000.0
    cumulative_realized_pnl = 0.0
    cumulative_fees = 0.0
    open_positions = {}
    
    print("Initial capital: $10,000.00")
    print()
    
    print("Processing trades chronologically:")
    
    for idx, trade in trades_df.iterrows():
        timestamp = trade['timestamp']
        symbol = trade['symbol']
        trade_type = trade['type']
        
        print(f"\n--- Trade {idx}: {timestamp} ---")
        print(f"Symbol: {symbol}, Type: {trade_type}, Side: {trade['side']}")
        
        if trade_type == 'open':
            fee = trade.get('fee', 0)
            if pd.isna(fee):
                fee = 0
                
            cumulative_fees += fee
            
            # Track open position
            contracts = trade['contracts'] if trade['side'] == 'buy' else -trade['contracts']
            open_positions[symbol] = {
                'contracts': contracts,
                'entry_price': trade['price'],
                'timestamp': timestamp
            }
            
            print(f"Opening position: {contracts:.6f} contracts at ${trade['price']:.2f}")
            print(f"Fee: ${fee:.2f}, Cumulative fees: ${cumulative_fees:.2f}")
            
        elif trade_type == 'close':
            fee = trade.get('fee', 0)
            pnl = trade.get('pnl', 0)
            
            if pd.isna(fee):
                fee = 0
            if pd.isna(pnl):
                pnl = 0
                
            cumulative_fees += fee
            cumulative_realized_pnl += pnl
            
            print(f"Closing position: PnL=${pnl:.2f}, Fee=${fee:.2f}")
            print(f"Cumulative PnL: ${cumulative_realized_pnl:.2f}, Cumulative fees: ${cumulative_fees:.2f}")
            
            if symbol in open_positions:
                del open_positions[symbol]
                print(f"Removed {symbol} from open positions")
            else:
                print(f"⚠️ WARNING: {symbol} not in open positions!")
        
        # Calculate current equity
        current_equity = 10000.0 + cumulative_realized_pnl - cumulative_fees
        print(f"Current equity (realized): ${current_equity:.2f}")
        
        # Calculate unrealized P&L
        total_unrealized = 0.0
        if open_positions and price_data is not None:
            try:
                current_prices = price_data.loc[timestamp]
                print(f"Open positions: {list(open_positions.keys())}")
                
                for pos_symbol, pos in open_positions.items():
                    if pos_symbol in current_prices:
                        current_price = current_prices[pos_symbol]
                        if not pd.isna(current_price):
                            unrealized = (current_price - pos['entry_price']) * pos['contracts']
                            total_unrealized += unrealized
                            print(f"  {pos_symbol}: {pos['contracts']:.6f} @ ${pos['entry_price']:.2f}, current: ${current_price:.2f}, unrealized: ${unrealized:.2f}")
                        else:
                            print(f"  {pos_symbol}: No price data available")
                    else:
                        print(f"  {pos_symbol}: Not in price data columns")
            except (KeyError, IndexError):
                print(f"  Cannot get prices for timestamp {timestamp}")
        
        total_equity = current_equity + total_unrealized
        print(f"Total unrealized P&L: ${total_unrealized:.2f}")
        print(f"Total equity: ${total_equity:.2f}")
    
    # Final summary
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    print(f"Initial capital: $10,000.00")
    print(f"Cumulative realized P&L: ${cumulative_realized_pnl:.2f}")
    print(f"Cumulative fees: ${cumulative_fees:.2f}")
    print(f"Final equity (without unrealized): ${10000.0 + cumulative_realized_pnl - cumulative_fees:.2f}")
    
    # Expected values from terminal
    expected_pnl = -1644.89
    expected_fees = 283.49
    expected_final = 8071.63
    
    print(f"\nExpected from terminal:")
    print(f"PnL: ${expected_pnl:.2f}, Fees: ${expected_fees:.2f}, Final: ${expected_final:.2f}")
    
    print(f"\nDiscrepancies:")
    print(f"PnL difference: ${abs(cumulative_realized_pnl - expected_pnl):.2f}")
    print(f"Fees difference: ${abs(cumulative_fees - expected_fees):.2f}")
    
    # Identify the issue
    if abs(cumulative_realized_pnl - expected_pnl) > 10:
        print("🔍 ISSUE: Large PnL discrepancy - some trades may be missing or calculated incorrectly")
    
    if abs(cumulative_fees - expected_fees) > 5:
        print("🔍 ISSUE: Fees discrepancy - fee calculation may be incorrect")

def analyze_pnl_by_symbol():
    """Analyze PnL by symbol to identify discrepancies."""
    print("\n" + "="*60)
    print("PnL ANALYSIS BY SYMBOL")
    print("="*60)
    
    trades_df, _ = load_real_trades_and_prices()
    if trades_df is None:
        return
    
    close_trades = trades_df[trades_df['type'] == 'close'].copy()
    
    print("Expected PnL by symbol (from terminal):")
    expected_pnl = {
        'SOLUSDT': -313.34,
        'XRPUSDT': -844.38,
        'BNBUSDT': -585.87,
        'BTCUSDT': 349.76,
        'ETHUSDT': -251.07
    }
    
    print("Calculated PnL by symbol (from trades):")
    total_calculated = 0
    for symbol in expected_pnl.keys():
        symbol_trades = close_trades[close_trades['symbol'] == symbol]
        symbol_pnl = symbol_trades['pnl'].fillna(0).sum()
        expected = expected_pnl[symbol]
        difference = abs(symbol_pnl - expected)
        
        print(f"{symbol}: Expected=${expected:.2f}, Calculated=${symbol_pnl:.2f}, Diff=${difference:.2f}")
        if difference > 1.0:
            print(f"  ⚠️ Significant difference for {symbol}")
        
        total_calculated += symbol_pnl
    
    total_expected = sum(expected_pnl.values())
    print(f"\nTotals:")
    print(f"Expected: ${total_expected:.2f}")
    print(f"Calculated: ${total_calculated:.2f}")
    print(f"Overall difference: ${abs(total_calculated - total_expected):.2f}")

def run_detailed_trade_analysis():
    """Run detailed trade-by-trade analysis."""
    print("🔍 DETAILED TRADE-BY-TRADE ANALYSIS")
    
    trades_df, price_data = load_real_trades_and_prices()
    if trades_df is None:
        return
    
    print(f"Loaded {len(trades_df)} trades")
    
    manual_equity_curve_calculation(trades_df, price_data)
    analyze_pnl_by_symbol()
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_detailed_trade_analysis()
