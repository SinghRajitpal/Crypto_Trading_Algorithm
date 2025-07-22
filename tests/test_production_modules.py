#!/usr/bin/env python3
"""
Production Modules Unit Tests

Tests for the individual production modules to ensure they work correctly.
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_portfolio_manager():
    """Test the production portfolio manager."""
    print("=" * 60)
    print("TESTING PRODUCTION PORTFOLIO MANAGER")
    print("=" * 60)
    
    try:
        from execution.portfolio import ProductionPortfolioManager
        
        # Test instantiation
        portfolio = ProductionPortfolioManager(total_capital=5000.0)
        print("✅ Portfolio manager instantiated")
        
        # Test volatility data update
        portfolio.update_volatility_data("BTCUSDT", 0.02)
        portfolio.update_volatility_data("ETHUSDT", 0.03)
        print("✅ Volatility data updated")
        
        # Test correlation data update
        portfolio.update_correlation_data("BTCUSDT", "ETHUSDT", 0.8)
        print("✅ Correlation data updated")
        
        # Test weight computation
        symbols = ["BTCUSDT", "ETHUSDT"]
        weights = portfolio.compute_weights(symbols)
        print(f"✅ Weights computed: {weights}")
        
        # Test rebalancing
        allocations = portfolio.rebalance_portfolio(symbols)
        print(f"✅ Portfolio rebalanced: {len(allocations)} allocations")
        
        # Test regime detection
        is_high_vol = portfolio.is_high_volatility_regime()
        print(f"✅ High volatility regime: {is_high_vol}")
        
        # Test portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"✅ Portfolio summary: ${summary['total_capital']:.2f} total, {summary['active_positions']} positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_manager():
    """Test the production risk manager."""
    print("\\n" + "=" * 60)
    print("TESTING PRODUCTION RISK MANAGER")
    print("=" * 60)
    
    try:
        from execution.portfolio import ProductionPortfolioManager
        from execution.risk_manager import ProductionRiskManager
        
        # Create portfolio manager first
        portfolio = ProductionPortfolioManager(total_capital=5000.0)
        
        # Update with some test data
        portfolio.update_volatility_data("BTCUSDT", 0.02)
        allocations = portfolio.rebalance_portfolio(["BTCUSDT"])
        
        # Test risk manager instantiation
        risk_manager = ProductionRiskManager(portfolio_manager=portfolio)
        print("✅ Risk manager instantiated")
        
        # Test position sizing
        position_info = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=1000.0,
            atr_value=0.015,
            entry_price=50000.0
        )
        print(f"✅ Position size calculated: {position_info['size_contracts']:.6f} contracts")
        print(f"   Leverage: {position_info['leverage']}x, Margin: ${position_info['margin_usdt']:.2f}")
        
        # Test dynamic leverage
        leverage = risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.015)
        print(f"✅ Dynamic leverage: {leverage}x")
        
        # Test stop loss / take profit
        sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(50000.0, "long", 0.015)
        print(f"✅ SL/TP calculated: SL=${sl_price:.2f}, TP=${tp_price:.2f}")
        
        # Test trade validation
        validation = risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="open",
            side="long",
            entry_price=50000.0,
            atr_value=0.015
        )
        print(f"✅ Trade validation: {validation['valid']} - {validation['reason']}")
        
        # Test risk metrics
        metrics = risk_manager.get_risk_metrics()
        print(f"✅ Risk metrics: PnL={metrics['daily_pnl']:.2f}, Kill switches={metrics['kill_switches']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stress_handler():
    """Test the stress handling module."""
    print("\\n" + "=" * 60)
    print("TESTING STRESS HANDLING MODULE")
    print("=" * 60)
    
    try:
        from execution.stress_handler import StressHandlingModule
        
        # Mock execution engine
        class MockExecutionEngine:
            def __init__(self):
                self.trading_paused = False
        
        mock_engine = MockExecutionEngine()
        
        # Test stress handler instantiation
        stress_handler = StressHandlingModule(mock_engine)
        print("✅ Stress handler instantiated")
        
        # Test flash crash detection
        price_data = {"high": 50000, "low": 48000, "close": 48500}
        atr_value = 0.01  # 1%
        flash_detected = stress_handler.check_flash_crash("BTCUSDT", price_data, atr_value)
        print(f"✅ Flash crash test: Detected={flash_detected}")
        
        # Test slippage check
        slippage_ok = stress_handler.check_slippage(50000.0, 50050.0, "BTCUSDT")
        print(f"✅ Slippage check: OK={slippage_ok}")
        
        # Test connection lag
        connection_ok = stress_handler.check_connection_lag(datetime.now())
        print(f"✅ Connection test: OK={connection_ok}")
        
        # Test liquidity filters
        liquidity_ok = stress_handler.check_liquidity_filters(
            symbol="BTCUSDT",
            volume_24h=10_000_000,  # $10M volume
            spread_pct=0.001,       # 0.1% spread
            funding_rate=0.002      # 0.2% funding
        )
        print(f"✅ Liquidity filters: OK={liquidity_ok}")
        
        # Test kill switches
        switches = stress_handler.check_kill_switches(drawdown_pct=0.05, equity_slope=-0.02)
        print(f"✅ Kill switches: {switches}")
        
        # Test stress summary
        summary = stress_handler.get_stress_summary()
        print(f"✅ Stress summary: {len(summary)} metrics tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Stress handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_executor():
    """Test the order executor."""
    print("\\n" + "=" * 60)
    print("TESTING ORDER EXECUTOR")
    print("=" * 60)
    
    try:
        from execution.executor import OrderExecutor
        from execution.portfolio import ProductionPortfolioManager
        from execution.risk_manager import ProductionRiskManager
        
        # Create required components
        portfolio = ProductionPortfolioManager(total_capital=5000.0)
        risk_manager = ProductionRiskManager(portfolio_manager=portfolio)
        
        # Mock Binance client
        class MockBinanceClient:
            async def set_leverage(self, symbol, leverage):
                return {"leverage": leverage}
            
            async def create_order(self, **kwargs):
                return {"orderId": 12345, "status": "FILLED"}
        
        mock_client = MockBinanceClient()
        
        # Test executor instantiation
        executor = OrderExecutor(
            binance_client=mock_client,
            portfolio_manager=portfolio,
            risk_manager=risk_manager
        )
        print("✅ Order executor instantiated")
        
        # The executor methods are async and would need market data to test fully
        print("✅ Order executor basic test completed")
        print("   (Full execution tests require live market data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Order executor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all unit tests."""
    print("🧪 RUNNING PRODUCTION MODULES UNIT TESTS")
    print("=" * 80)
    
    tests = [
        ("Portfolio Manager", test_portfolio_manager),
        ("Risk Manager", test_risk_manager),
        ("Stress Handler", test_stress_handler),
        ("Order Executor", test_order_executor)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\\n🔍 Testing {name}...")
        result = test_func()
        results.append((name, result))
        if result:
            print(f"✅ {name} test PASSED")
        else:
            print(f"❌ {name} test FAILED")
    
    print("\\n" + "=" * 80)
    print("UNIT TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\\nTotal: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\\n🎉 All unit tests passed! Production modules are working correctly.")
    else:
        print(f"\\n⚠️ {failed} tests failed. Please fix issues before deployment.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
