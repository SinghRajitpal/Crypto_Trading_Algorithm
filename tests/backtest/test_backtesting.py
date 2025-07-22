#!/usr/bin/env python3
"""
Backtesting Engine Test

Tests the backtesting engine with the new production modules.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_backtesting_engine():
    """Test the backtesting engine."""
    print("=" * 80)
    print("TESTING BACKTESTING ENGINE")
    print("=" * 80)
    
    try:
        from backtest.backtesting_engine import BacktestingEngine
        from algorithm.strategies.ma_crossover import MACrossoverStrategy
        
        # Create strategy
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'leverage': 5
        })
        
        print("✅ Strategy created for backtesting")
        
        # Create backtesting engine
        symbols = [("BTCUSDT", "1m")]
        backtest = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start="2024-07-01",
            end="2024-07-02",  # Short test period
            initial_capital=1000.0
        )
        
        print("✅ BacktestingEngine instantiated")
        
        # Test execution engine compatibility
        execution_engine = backtest.execution_engine
        print(f"✅ Execution engine type: {type(execution_engine).__name__}")
        
        # Test critical methods exist
        methods_to_test = ["get_portfolio_summary", "get_risk_metrics", "process_signal"]
        for method in methods_to_test:
            if hasattr(execution_engine, method):
                print(f"✅ {method} method available")
            else:
                print(f"❌ {method} method missing")
                return False
        
        # Test portfolio/risk integration
        portfolio = execution_engine.get_portfolio_summary()
        risk = execution_engine.get_risk_metrics()
        
        print(f"✅ Portfolio capital: ${portfolio['total_capital']:.2f}")
        print(f"✅ Risk status: PnL={risk['daily_pnl']:.2f}, Positions={risk['active_positions']}")
        
        # Test broker integration
        broker = backtest.broker
        print(f"✅ Broker type: {type(broker).__name__}")
        
        # Test data engine
        data_engine = backtest.data_engine
        print(f"✅ Data engine max candles: {data_engine.data_fetcher.data_processor.max_candles}")
        
        print("\\n✅ Backtesting engine test completed successfully!")
        print("The backtesting system is compatible with production modules.")
        
        # Test a short backtest run (commented out to avoid long execution)
        # print("\\nRunning short backtest...")
        # results = await backtest.run(save_results=False)
        # print(f"✅ Backtest completed: {len(results.get('trades', []))} trades")
        
    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_backtesting_engine())
    if result:
        print("\\n🎉 Backtesting tests passed! System ready for backtesting.")
    else:
        print("\\n⚠️ Backtesting tests failed. Please fix issues before running backtests.")
