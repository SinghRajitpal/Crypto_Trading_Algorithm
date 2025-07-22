#!/usr/bin/env python3

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from execution.risk_manager import ProductionRiskManager
from execution.portfolio import ProductionPortfolioManager

def test_risk_manager_comprehensive():
    """Run comprehensive risk manager tests."""
    
    print("🧪 Starting comprehensive risk manager tests...")
    
    # Test 1: Initialization
    portfolio_manager = ProductionPortfolioManager(total_capital=10000.0)
    risk_manager = ProductionRiskManager(portfolio_manager)
    assert risk_manager.portfolio_manager == portfolio_manager
    assert risk_manager.risk_params.risk_per_trade_pct == 0.008  # 0.8%
    assert risk_manager.risk_params.kelly_fraction == 0.7
    print("✅ Test 1: Initialization passed")
    
    # Test 2: ATR data management
    symbol = "BTCUSDT"
    atr_values = [100, 110, 95, 120, 105]
    
    for atr in atr_values:
        risk_manager.update_atr_data(symbol, atr)
    
    assert symbol in risk_manager.atr_data
    assert len(risk_manager.atr_data[symbol]) == len(atr_values)
    assert all(val in risk_manager.atr_data[symbol] for val in atr_values)
    print("✅ Test 2: ATR data management passed")
    
    # Test 3: Position sizing with volatility
    symbol_data = {
        "BTCUSDT": {"volatility": 0.002, "price": 50000.0},
        "ETHUSDT": {"volatility": 0.003, "price": 3000.0},
        "XRPUSDT": {"volatility": 0.0025, "price": 0.5}
    }
    
    for sym, data in symbol_data.items():
        # Add ATR data
        for _ in range(10):
            risk_manager.update_atr_data(sym, data["price"] * 0.02)  # 2% ATR
    
    # Test position sizing
    for sym, data in symbol_data.items():
        size = risk_manager.calculate_position_size(
            symbol=sym,
            volatility=data["volatility"],
            allocation_weight=0.3
        )
        assert size > 0
        assert size <= risk_manager.total_capital * risk_manager.max_allocation_per_position
        print(f"   {sym}: Position size ${size:,.2f}")
    
    print("✅ Test 3: Position sizing with volatility passed")
    
    # Test 4: Kelly criterion position sizing
    kelly_size = risk_manager.calculate_kelly_position_size(
        win_rate=0.65,
        avg_win=1.5,
        avg_loss=1.0,
        current_allocation=0.3
    )
    assert kelly_size > 0
    assert kelly_size <= 1.0  # Should not exceed 100%
    print(f"   Kelly fraction: {kelly_size:.3f}")
    print("✅ Test 4: Kelly criterion position sizing passed")
    
    # Test 5: Stop loss calculation
    symbol = "BTCUSDT"
    price = 50000.0
    volatility = 0.002
    
    stop_loss = risk_manager.calculate_stop_loss(
        symbol=symbol,
        entry_price=price,
        volatility=volatility,
        position_side="long"
    )
    
    assert stop_loss < price  # Stop loss should be below entry for long
    assert (price - stop_loss) / price <= 0.05  # Should not exceed max stop loss
    print(f"   Long stop loss: ${stop_loss:,.2f} (risk: {(price - stop_loss)/price:.2%})")
    
    # Test short position
    stop_loss_short = risk_manager.calculate_stop_loss(
        symbol=symbol,
        entry_price=price,
        volatility=volatility,
        position_side="short"
    )
    
    assert stop_loss_short > price  # Stop loss should be above entry for short
    print(f"   Short stop loss: ${stop_loss_short:,.2f} (risk: {(stop_loss_short - price)/price:.2%})")
    print("✅ Test 5: Stop loss calculation passed")
    
    # Test 6: Take profit calculation
    take_profit = risk_manager.calculate_take_profit(
        symbol=symbol,
        entry_price=price,
        volatility=volatility,
        position_side="long"
    )
    
    assert take_profit > price  # Take profit should be above entry for long
    profit_ratio = (take_profit - price) / price
    assert profit_ratio >= 0.02  # Should have reasonable profit target
    print(f"   Long take profit: ${take_profit:,.2f} (target: {profit_ratio:.2%})")
    print("✅ Test 6: Take profit calculation passed")
    
    # Test 7: Dynamic leverage calculation
    market_conditions = [
        {"volatility": 0.001, "expected_leverage": "higher"},  # Low vol = higher leverage
        {"volatility": 0.005, "expected_leverage": "lower"}    # High vol = lower leverage
    ]
    
    leverages = []
    for condition in market_conditions:
        leverage = risk_manager.calculate_dynamic_leverage(
            base_leverage=5.0,
            volatility=condition["volatility"]
        )
        leverages.append(leverage)
        assert 1.0 <= leverage <= risk_manager.max_leverage
        print(f"   Vol {condition['volatility']:.3f}: Leverage {leverage:.2f}x")
    
    # Lower volatility should allow higher leverage
    assert leverages[0] >= leverages[1]
    print("✅ Test 7: Dynamic leverage calculation passed")
    
    # Test 8: Daily P&L tracking
    trades = [
        {"pnl": 150.0, "timestamp": datetime.now() - timedelta(hours=2)},
        {"pnl": -80.0, "timestamp": datetime.now() - timedelta(hours=1)},
        {"pnl": 200.0, "timestamp": datetime.now() - timedelta(minutes=30)},
    ]
    
    for trade in trades:
        risk_manager.update_daily_pnl(trade["pnl"], trade["timestamp"])
    
    daily_pnl = risk_manager.get_daily_pnl()
    expected_pnl = sum(trade["pnl"] for trade in trades)
    assert abs(daily_pnl - expected_pnl) < 0.01
    print(f"   Daily P&L: ${daily_pnl:,.2f}")
    print("✅ Test 8: Daily P&L tracking passed")
    
    # Test 9: Risk limit checks
    # Test daily loss limit
    within_limits = risk_manager.check_risk_limits()
    assert within_limits == True  # Should be within limits
    
    # Simulate large loss to trigger limit
    large_loss = -risk_manager.total_capital * risk_manager.max_daily_loss * 1.5
    risk_manager.update_daily_pnl(large_loss)
    
    within_limits_after_loss = risk_manager.check_risk_limits()
    assert within_limits_after_loss == False  # Should exceed limits
    print("✅ Test 9: Risk limit checks passed")
    
    # Test 10: Drawdown monitoring
    # Reset for clean test
    risk_manager = ProductionRiskManager(total_capital=10000.0)
    
    # Simulate equity curve
    equity_values = [10000, 9800, 9500, 9200, 9400, 9600, 9300, 8800]
    for equity in equity_values:
        risk_manager.update_drawdown_monitoring(equity)
    
    current_dd = risk_manager.get_current_drawdown()
    max_dd = risk_manager.get_max_drawdown()
    
    assert current_dd >= 0  # Drawdown should be positive
    assert max_dd >= current_dd  # Max DD should be >= current
    print(f"   Current drawdown: {current_dd:.2%}")
    print(f"   Maximum drawdown: {max_dd:.2%}")
    print("✅ Test 10: Drawdown monitoring passed")
    
    # Test 11: Sharpe ratio tracking
    returns = [0.01, -0.005, 0.015, -0.002, 0.008, 0.003, -0.001, 0.012]
    for ret in returns:
        risk_manager.update_performance_metrics(ret)
    
    sharpe = risk_manager.get_sharpe_ratio()
    assert sharpe is not None
    print(f"   Sharpe ratio: {sharpe:.3f}")
    print("✅ Test 11: Sharpe ratio tracking passed")
    
    # Test 12: Emergency position sizing
    emergency_mode = True
    normal_size = 1000.0
    emergency_size = risk_manager.apply_emergency_sizing(normal_size, emergency_mode)
    
    assert emergency_size < normal_size
    assert emergency_size == normal_size * risk_manager.emergency_scale_factor
    print(f"   Normal size: ${normal_size:,.2f} -> Emergency size: ${emergency_size:,.2f}")
    print("✅ Test 12: Emergency position sizing passed")
    
    # Test 13: Risk summary
    summary = risk_manager.get_risk_summary()
    
    required_fields = [
        'total_capital', 'daily_pnl', 'daily_pnl_pct', 'current_drawdown',
        'max_drawdown', 'sharpe_ratio', 'within_risk_limits'
    ]
    
    for field in required_fields:
        assert field in summary
        
    print("✅ Test 13: Risk summary passed")
    
    print("🎉 All comprehensive risk manager tests passed!")
    print(f"📊 Final risk state:")
    for key, value in summary.items():
        if isinstance(value, float) and abs(value) < 1:
            print(f"   {key}: {value:.3f}")
        elif isinstance(value, float):
            print(f"   {key}: ${value:,.2f}" if 'capital' in key or 'pnl' in key and 'pct' not in key else f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    test_risk_manager_comprehensive()
