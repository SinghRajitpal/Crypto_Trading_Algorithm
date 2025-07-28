#!/usr/bin/env python3
"""
BACKTEST AUDIT DIAGNOSTICS SUITE
Critical validation checks for crypto trading backtest accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt


class BacktestAuditor:
    """Comprehensive backtest validation and diagnostic tool."""
    
    def __init__(self, trade_log_path: str):
        """Initialize with trade log path."""
        self.trade_log_path = trade_log_path
        self.trade_log = None
        self.load_trade_log()
        
    def load_trade_log(self):
        """Load and validate trade log."""
        try:
            self.trade_log = pd.read_csv(self.trade_log_path, parse_dates=['timestamp'])
            print(f"✅ Loaded {len(self.trade_log)} trade entries")
        except Exception as e:
            print(f"❌ Failed to load trade log: {e}")
            
    def audit_funding_application(self) -> Dict[str, Any]:
        """CRITICAL: Audit funding rate application logic."""
        print("\n" + "="*60)
        print("🔍 FUNDING RATE APPLICATION AUDIT")
        print("="*60)
        
        funding_entries = self.trade_log[self.trade_log['type'] == 'funding'].copy()
        position_entries = self.trade_log[self.trade_log['type'].isin(['open', 'close'])].copy()
        
        print(f"📊 Trade Type Distribution:")
        type_counts = self.trade_log['type'].value_counts()
        for trade_type, count in type_counts.items():
            print(f"   {trade_type}: {count:,}")
            
        # Check funding frequency
        funding_by_symbol = funding_entries.groupby('symbol').size()
        print(f"\n📈 Funding Entries by Symbol:")
        for symbol, count in funding_by_symbol.items():
            print(f"   {symbol}: {count:,}")
            
        # Analyze funding timing
        funding_hours = funding_entries['timestamp'].dt.hour.value_counts().sort_index()
        print(f"\n⏰ Funding Application Hours:")
        for hour, count in funding_hours.items():
            print(f"   {hour:02d}:00 UTC: {count:,} applications")
            
        # CRITICAL: Check for funding without positions
        issues = []
        
        for _, funding in funding_entries.iterrows():
            symbol = funding['symbol']
            timestamp = funding['timestamp']
            
            # Find if position existed at funding time
            symbol_positions = position_entries[position_entries['symbol'] == symbol].copy()
            
            if symbol_positions.empty:
                issues.append(f"❌ Funding applied to {symbol} with NO POSITIONS EVER")
                continue
                
            # Check if position was open at funding time
            opens = symbol_positions[symbol_positions['type'] == 'open']
            closes = symbol_positions[symbol_positions['type'] == 'close']
            
            position_open = False
            for _, open_trade in opens.iterrows():
                # Find corresponding close (if any)
                later_closes = closes[closes['timestamp'] > open_trade['timestamp']]
                
                if later_closes.empty:
                    # Position still open
                    if timestamp >= open_trade['timestamp']:
                        position_open = True
                        break
                else:
                    # Position was closed
                    first_close = later_closes.iloc[0]
                    if open_trade['timestamp'] <= timestamp < first_close['timestamp']:
                        position_open = True
                        break
                        
            if not position_open:
                issues.append(f"❌ Funding applied to {symbol} at {timestamp} WITHOUT OPEN POSITION")
                
        return {
            'total_funding_entries': len(funding_entries),
            'total_position_entries': len(position_entries),
            'funding_without_position_issues': len(issues),
            'issues': issues[:10],  # First 10 issues
            'funding_hours': funding_hours.to_dict(),
            'funding_by_symbol': funding_by_symbol.to_dict()
        }
    
    def audit_position_sizing(self) -> Dict[str, Any]:
        """Audit position sizing logic and reasonableness."""
        print("\n" + "="*60)
        print("💰 POSITION SIZING AUDIT")
        print("="*60)
        
        open_trades = self.trade_log[self.trade_log['type'] == 'open'].copy()
        
        if open_trades.empty:
            return {'error': 'No open trades found'}
            
        # Analyze position sizes
        position_stats = {
            'count': len(open_trades),
            'total_notional': open_trades['notional'].sum(),
            'avg_notional': open_trades['notional'].mean(),
            'min_notional': open_trades['notional'].min(),
            'max_notional': open_trades['notional'].max(),
            'avg_contracts': open_trades['contracts'].mean(),
            'avg_leverage': open_trades['leverage'].mean() if 'leverage' in open_trades.columns else 1.0
        }
        
        print(f"📊 Position Sizing Statistics:")
        print(f"   Total Trades: {position_stats['count']}")
        print(f"   Total Notional: ${position_stats['total_notional']:,.2f}")
        print(f"   Average Notional: ${position_stats['avg_notional']:,.2f}")
        print(f"   Min/Max Notional: ${position_stats['min_notional']:,.2f} / ${position_stats['max_notional']:,.2f}")
        print(f"   Average Leverage: {position_stats['avg_leverage']:.2f}x")
        
        # Check for unreasonably small positions
        small_positions = open_trades[open_trades['notional'] < 100]  # Less than $100
        if not small_positions.empty:
            print(f"⚠️  {len(small_positions)} positions < $100 (may indicate sizing issues)")
            
        return position_stats
    
    def audit_cost_application(self) -> Dict[str, Any]:
        """Audit trading cost application and realism."""
        print("\n" + "="*60)
        print("💸 TRADING COST AUDIT")
        print("="*60)
        
        open_trades = self.trade_log[self.trade_log['type'] == 'open'].copy()
        close_trades = self.trade_log[self.trade_log['type'] == 'close'].copy()
        
        if open_trades.empty:
            return {'error': 'No trades to analyze costs'}
            
        # Analyze fees
        open_trades['fee_pct'] = (open_trades['fee'] / open_trades['notional']) * 100
        
        cost_stats = {
            'total_open_fees': open_trades['fee'].sum(),
            'avg_fee_pct': open_trades['fee_pct'].mean(),
            'min_fee_pct': open_trades['fee_pct'].min(),
            'max_fee_pct': open_trades['fee_pct'].max(),
        }
        
        if not close_trades.empty and 'fee' in close_trades.columns:
            cost_stats['total_close_fees'] = close_trades['fee'].sum()
            cost_stats['total_fees'] = cost_stats['total_open_fees'] + cost_stats['total_close_fees']
        else:
            cost_stats['total_close_fees'] = 0
            cost_stats['total_fees'] = cost_stats['total_open_fees']
            
        print(f"💰 Cost Analysis:")
        print(f"   Total Opening Fees: ${cost_stats['total_open_fees']:.4f}")
        print(f"   Average Fee %: {cost_stats['avg_fee_pct']:.4f}%")
        print(f"   Fee Range: {cost_stats['min_fee_pct']:.4f}% - {cost_stats['max_fee_pct']:.4f}%")
        
        # Expected fee range for crypto futures: 0.02% - 0.06%
        if cost_stats['avg_fee_pct'] < 0.01:
            print("⚠️  Fees appear too low (expect 0.02-0.06% for crypto futures)")
        elif cost_stats['avg_fee_pct'] > 0.1:
            print("⚠️  Fees appear too high (expect 0.02-0.06% for crypto futures)")
            
        return cost_stats
    
    def audit_pnl_calculation(self) -> Dict[str, Any]:
        """Audit P&L calculations for accuracy."""
        print("\n" + "="*60)
        print("📈 P&L CALCULATION AUDIT")
        print("="*60)
        
        open_trades = self.trade_log[self.trade_log['type'] == 'open'].copy()
        close_trades = self.trade_log[self.trade_log['type'] == 'close'].copy()
        funding_trades = self.trade_log[self.trade_log['type'] == 'funding'].copy()
        
        # Calculate realized P&L
        realized_pnl = close_trades['pnl'].sum() if 'pnl' in close_trades.columns else 0
        
        # Calculate funding costs
        funding_costs = funding_trades['payment'].sum() if 'payment' in funding_trades.columns else 0
        
        # Calculate total fees
        total_fees = 0
        if 'fee' in open_trades.columns:
            total_fees += open_trades['fee'].sum()
        if 'fee' in close_trades.columns:
            total_fees += close_trades['fee'].sum()
            
        # Net P&L
        net_pnl = realized_pnl - funding_costs - total_fees
        
        pnl_stats = {
            'realized_pnl': realized_pnl,
            'funding_costs': funding_costs,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'total_trades': len(close_trades),
            'avg_pnl_per_trade': realized_pnl / len(close_trades) if len(close_trades) > 0 else 0
        }
        
        print(f"💰 P&L Breakdown:")
        print(f"   Realized P&L: ${pnl_stats['realized_pnl']:,.2f}")
        print(f"   Funding Costs: ${pnl_stats['funding_costs']:,.2f}")
        print(f"   Total Fees: ${pnl_stats['total_fees']:,.2f}")
        print(f"   Net P&L: ${pnl_stats['net_pnl']:,.2f}")
        print(f"   Avg P&L per Trade: ${pnl_stats['avg_pnl_per_trade']:,.2f}")
        
        return pnl_stats
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        print("🔍 COMPREHENSIVE BACKTEST AUDIT REPORT")
        print("="*80)
        
        funding_audit = self.audit_funding_application()
        sizing_audit = self.audit_position_sizing()
        cost_audit = self.audit_cost_application()
        pnl_audit = self.audit_pnl_calculation()
        
        # Calculate severity score
        severity_score = 0
        issues = []
        
        # Funding issues (most critical)
        if funding_audit['funding_without_position_issues'] > 0:
            severity_score += 50
            issues.append("CRITICAL: Funding applied without positions")
            
        # Low trade count issues
        trade_count = sizing_audit.get('count', 0) if isinstance(sizing_audit, dict) else 0
        if trade_count < 20:
            severity_score += 30
            issues.append("HIGH: Very few trades generated")
            
        # Cost issues
        if isinstance(cost_audit, dict):
            avg_fee = cost_audit.get('avg_fee_pct', 0)
            if avg_fee < 0.01 or avg_fee > 0.1:
                severity_score += 20
                issues.append("MEDIUM: Fee percentages outside expected range")
        
        print(f"\n🚨 AUDIT SUMMARY:")
        print(f"   Severity Score: {severity_score}/100")
        print(f"   Issues Found: {len(issues)}")
        
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
            
        recommendation = "REJECT" if severity_score > 70 else "REVIEW" if severity_score > 30 else "ACCEPT"
        print(f"\n📋 RECOMMENDATION: {recommendation} backtest results")
        
        return {
            'funding_audit': funding_audit,
            'sizing_audit': sizing_audit,
            'cost_audit': cost_audit,
            'pnl_audit': pnl_audit,
            'severity_score': severity_score,
            'issues': issues,
            'recommendation': recommendation
        }


def main():
    """Run backtest audit."""
    # Update path to your latest backtest results
    trade_log_path = "/Users/singhs/Documents/Coding/Crypto Trading Algorithm/backtest/results/ma_crossover/20250724_132222/trade_log.csv"
    
    auditor = BacktestAuditor(trade_log_path)
    report = auditor.generate_audit_report()
    
    return report


if __name__ == "__main__":
    main()
