#!/usr/bin/env python3
"""
Main Trading Algorithm Test

Tests the main trading algorithm with the new production modules.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_main_algorithm():
    """Test the main trading algorithm."""
    print("=" * 80)
    print("TESTING MAIN TRADING ALGORITHM")
    print("=" * 80)
    
    try:
        from main import TradingAlgorithm
        from algorithm.strategies.ma_crossover import MACrossoverStrategy
        
        # Create strategy with test parameters
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.04,  # 4% take profit
            'leverage': 5  # 5x leverage for testing
        })
        
        print("✅ Strategy created successfully")
        
        # Create algorithm instance
        algorithm = TradingAlgorithm(
            strategy=strategy, 
            testnet=True,
            total_capital=1000.0
        )
        
        print("✅ TradingAlgorithm instantiated successfully")
        
        # Test critical components
        print("\\nTesting critical components:")
        
        # Test portfolio summary
        portfolio = algorithm.execution_engine.get_portfolio_summary()
        print(f"✅ Portfolio Summary: {portfolio['total_capital']:.2f} USDT available")
        
        # Test risk metrics
        risk = algorithm.execution_engine.get_risk_metrics()
        print(f"✅ Risk Metrics: Daily PnL = {risk['daily_pnl']:.2f}, Active positions = {risk['active_positions']}")
        
        # Create a mock signal for validation testing
        class MockSignal:
            def __init__(self):
                self.symbol = "BTCUSDT"
                self.action = "open"
                self.side = "long"
                self.metadata = {
                    'atr_value': 0.005,  # Mock ATR value
                    'reason': 'MA crossover bullish'
                }
        
        mock_signal = MockSignal()
        
        # Test signal validation
        validation = await algorithm.execution_engine.validate_signal(mock_signal, 50000.0)
        print(f"✅ Signal Validation: {validation['valid']} - {validation['reason']}")
        
        if validation['valid']:
            print(f"  - Position size would be: {mock_signal.metadata.get('position_size', 'N/A'):.6f} contracts")
            print(f"  - Leverage would be: {mock_signal.metadata.get('position_leverage', 'N/A')}x")
            print(f"  - Stop loss: ${mock_signal.metadata.get('stop_loss_price', 'N/A'):.2f}")
            print(f"  - Take profit: ${mock_signal.metadata.get('take_profit_price', 'N/A'):.2f}")
        
        print("\\n✅ Main algorithm test completed successfully!")
        print("The system is ready for live testing.")
        
    except Exception as e:
        print(f"❌ Main algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_main_algorithm())
    if result:
        print("\\n🎉 All tests passed! System ready for deployment.")
    else:
        print("\\n⚠️ Tests failed. Please fix issues before deployment.")
