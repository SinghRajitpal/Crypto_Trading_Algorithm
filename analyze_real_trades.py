#!/usr/bin/env python3
"""
Real Trades Equity Analysis

Analyze the actual trades from the backtest to identify
the source of the $274.02 discrepancy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append('/Users/singhs/Documents/Coding/Crypto Trading Algorithm')

def load_latest_trades():
    """Load the most recent trades from the backtest results."""
    # Find the most recent results directory
    results_dir = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/backtest/results/ma_crossover"
    
    if os.path.exists(results_dir):
        # Get the most recent subdirectory
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs)
            trades_file = os.path.join(results_dir, latest_subdir, "trade_log.csv")
            
            if os.path.exists(trades_file):
                print(f"Loading trades from: {trades_file}")
                return pd.read_csv(trades_file)
    
    print("❌ Could not find trades file")
    return None

def analyze_actual_trades_discrepancy():
    """Analyze the actual trades to understand the discrepancy."""
    print("="*60)
    print("ACTUAL TRADES DISCREPANCY ANALYSIS")
    print("="*60)
    
    trades = load_latest_trades()
    if trades is None:
        return
    
    print(f"Loaded {len(trades)} trade records")
    print(f"Trade types: {trades['type'].value_counts().to_dict()}")
    print()
    
    # Calculate the ground truth from terminal output
    print("GROUND TRUTH FROM TERMINAL OUTPUT:")
    print("Portfolio: PnL: $-1,644.89, Fees: $283.49")
    print("Expected final equity: $8,071.63")
    print()
    
    # Calculate what we should get from the trades DataFrame
    print("TRADES DATAFRAME CALCULATION:")
    
    # Filter close trades for PnL
    close_trades = trades[trades['type'] == 'close']
    total_pnl_from_df = close_trades['pnl'].sum() if 'pnl' in close_trades.columns else 0
    
    # All trades for fees
    total_fees_from_df = trades['fee'].sum() if 'fee' in trades.columns else 0
    
    expected_from_df = 10000.0 + total_pnl_from_df - total_fees_from_df
    
    print(f"PnL from trades DF: ${total_pnl_from_df:.2f}")
    print(f"Fees from trades DF: ${total_fees_from_df:.2f}")
    print(f"Expected final from DF: ${expected_from_df:.2f}")
    print()
    
    # Identify the discrepancy sources
    print("DISCREPANCY ANALYSIS:")
    terminal_pnl = -1644.89
    terminal_fees = 283.49
    terminal_final = 8071.63
    
    pnl_diff = abs(total_pnl_from_df - terminal_pnl)
    fees_diff = abs(total_fees_from_df - terminal_fees)
    final_diff = abs(expected_from_df - terminal_final)
    
    print(f"PnL difference: ${pnl_diff:.2f}")
    print(f"Fees difference: ${fees_diff:.2f}")
    print(f"Final equity difference: ${final_diff:.2f}")
    
    if pnl_diff > 1.0 or fees_diff > 1.0:
        print("\n❌ ISSUE FOUND: Significant difference in PnL or fees calculation")
        
        # Detailed breakdown by symbol
        print("\nPER-SYMBOL BREAKDOWN:")
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            symbol_closes = symbol_trades[symbol_trades['type'] == 'close']
            
            symbol_pnl = symbol_closes['pnl'].sum() if len(symbol_closes) > 0 and 'pnl' in symbol_closes.columns else 0
            symbol_fees = symbol_trades['fee'].sum() if 'fee' in symbol_trades.columns else 0
            
            print(f"{symbol}:")
            print(f"  Trades: {len(symbol_trades)} total, {len(symbol_closes)} closes")
            print(f"  PnL: ${symbol_pnl:.2f}")
            print(f"  Fees: ${symbol_fees:.2f}")
    
    # Check for missing or problematic data
    print("\nDATA QUALITY CHECKS:")
    
    # Check for NaN values
    if 'pnl' in trades.columns:
        nan_pnl = trades['pnl'].isna().sum()
        print(f"NaN PnL values: {nan_pnl}")
        if nan_pnl > 0:
            print("❌ Warning: Found NaN PnL values")
    
    if 'fee' in trades.columns:
        nan_fees = trades['fee'].isna().sum()
        print(f"NaN fee values: {nan_fees}")
        if nan_fees > 0:
            print("❌ Warning: Found NaN fee values")
    
    # Check for zero fees (which might indicate missing fee calculation)
    if 'fee' in trades.columns:
        zero_fees = (trades['fee'] == 0).sum()
        print(f"Zero fee trades: {zero_fees}")
        if zero_fees > len(trades) * 0.1:  # More than 10% have zero fees
            print("⚠️ Warning: Many trades have zero fees")
    
    # Check column data types
    print(f"\nColumn data types:")
    if 'pnl' in trades.columns:
        print(f"PnL column type: {trades['pnl'].dtype}")
    if 'fee' in trades.columns:
        print(f"Fee column type: {trades['fee'].dtype}")
    
    return trades

def examine_fee_calculation_logic():
    """Examine the fee calculation logic in detail."""
    print("\n" + "="*60)
    print("FEE CALCULATION EXAMINATION")
    print("="*60)
    
    trades = load_latest_trades()
    if trades is None:
        return
    
    # Group by type to analyze fee patterns
    print("FEES BY TRADE TYPE:")
    for trade_type in trades['type'].unique():
        type_trades = trades[trades['type'] == trade_type]
        if 'fee' in trades.columns:
            avg_fee = type_trades['fee'].mean()
            total_fee = type_trades['fee'].sum()
            print(f"{trade_type}: {len(type_trades)} trades, avg fee: ${avg_fee:.2f}, total: ${total_fee:.2f}")
    
    # Check fee calculation consistency
    if 'fee' in trades.columns and 'contracts' in trades.columns and 'price' in trades.columns:
        print("\nFEE CALCULATION CONSISTENCY:")
        
        # Calculate expected fee rates
        trades['notional'] = trades['contracts'] * trades['price']
        trades['fee_rate'] = trades['fee'] / trades['notional'] * 100
        
        print(f"Fee rate statistics (%):")
        print(f"  Mean: {trades['fee_rate'].mean():.4f}%")
        print(f"  Std: {trades['fee_rate'].std():.4f}%")
        print(f"  Min: {trades['fee_rate'].min():.4f}%")
        print(f"  Max: {trades['fee_rate'].max():.4f}%")
        
        # Check for outliers
        q1 = trades['fee_rate'].quantile(0.25)
        q3 = trades['fee_rate'].quantile(0.75)
        iqr = q3 - q1
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        
        outliers = trades[(trades['fee_rate'] < outlier_threshold_low) | 
                         (trades['fee_rate'] > outlier_threshold_high)]
        
        if len(outliers) > 0:
            print(f"\n⚠️ Found {len(outliers)} fee rate outliers:")
            for idx, outlier in outliers.iterrows():
                print(f"  {outlier['symbol']} {outlier['type']}: {outlier['fee_rate']:.4f}% fee rate")

def examine_pnl_calculation():
    """Examine PnL calculation in detail."""
    print("\n" + "="*60)
    print("PNL CALCULATION EXAMINATION")
    print("="*60)
    
    trades = load_latest_trades()
    if trades is None:
        return
    
    # Focus on close trades since they have PnL
    close_trades = trades[trades['type'] == 'close'].copy()
    
    if len(close_trades) == 0:
        print("❌ No close trades found")
        return
    
    print(f"Analyzing {len(close_trades)} close trades")
    
    # PnL statistics
    if 'pnl' in close_trades.columns:
        total_pnl = close_trades['pnl'].sum()
        positive_trades = close_trades[close_trades['pnl'] > 0]
        negative_trades = close_trades[close_trades['pnl'] <= 0]
        
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Profitable trades: {len(positive_trades)} (${positive_trades['pnl'].sum():.2f})")
        print(f"Losing trades: {len(negative_trades)} (${negative_trades['pnl'].sum():.2f})")
        
        # Per-symbol PnL breakdown
        print(f"\nPER-SYMBOL PNL:")
        for symbol in close_trades['symbol'].unique():
            symbol_closes = close_trades[close_trades['symbol'] == symbol]
            symbol_pnl = symbol_closes['pnl'].sum()
            print(f"{symbol}: ${symbol_pnl:.2f} ({len(symbol_closes)} trades)")
    
    else:
        print("❌ No PnL column found in close trades")

def run_real_trades_analysis():
    """Run comprehensive analysis of actual trades data."""
    print("🔍 ANALYZING REAL TRADES FOR DISCREPANCY")
    
    analyze_actual_trades_discrepancy()
    examine_fee_calculation_logic()
    examine_pnl_calculation()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_real_trades_analysis()
