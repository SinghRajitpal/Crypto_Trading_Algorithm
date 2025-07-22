#!/usr/bin/env python3
"""
System Diagnostics for Crypto Trading Algorithm

This script performs comprehensive diagnostics to identify integration issues
caused by the recent risk management module updates.
"""

import os
import sys
import inspect
import traceback
from typing import Dict, List, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-" * 60)
    print(f"{title}")
    print("-" * 60)

def test_imports():
    """Test all critical imports."""
    print_section("IMPORT DIAGNOSTICS")
    
    imports_to_test = [
        ("execution.execution_engine", "ExecutionEngine"),
        ("execution.portfolio", "ProductionPortfolioManager"),
        ("execution.risk_manager", "ProductionRiskManager"),
        ("execution.stress_handler", "StressHandlingModule"),
        ("execution.executor", "OrderExecutor"),
        ("backtest.backtesting_engine", "BacktestingEngine"),
        ("algorithm.strategies.ma_crossover", "MACrossoverStrategy"),
        ("data.data_engine", "DataEngine"),
        ("binance_exchange", "BinanceClient"),
    ]
    
    results = {}
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results[f"{module_name}.{class_name}"] = {"status": "SUCCESS", "class": cls}
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            results[f"{module_name}.{class_name}"] = {"status": "FAILED", "error": str(e)}
            print(f"❌ {module_name}.{class_name}: {e}")
    
    return results

def test_execution_engine_api():
    """Test ExecutionEngine API compatibility with main.py and backtesting."""
    print_section("EXECUTION ENGINE API COMPATIBILITY")
    
    try:
        from execution.execution_engine import ExecutionEngine
        
        # Get all methods from ExecutionEngine
        methods = [method for method in dir(ExecutionEngine) 
                  if not method.startswith('_') and callable(getattr(ExecutionEngine, method))]
        
        print("Available ExecutionEngine methods:")
        for method in sorted(methods):
            print(f"  - {method}")
        
        # Methods expected by main.py
        expected_by_main = [
            "get_portfolio_summary",
            "get_risk_metrics", 
            "validate_signal",
            "process_signal",
            "update_daily_pnl"
        ]
        
        print("\nMethods expected by main.py:")
        missing_methods = []
        for method in expected_by_main:
            if hasattr(ExecutionEngine, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING!")
                missing_methods.append(method)
        
        # Test instantiation
        print("\nTesting ExecutionEngine instantiation:")
        try:
            # Mock binance client
            class MockBinanceClient:
                pass
            
            engine = ExecutionEngine(MockBinanceClient(), total_capital=1000.0)
            print("  ✅ ExecutionEngine instantiated successfully")
            
            # Test if new production methods exist
            production_methods = [
                "update_market_data_bar",
                "process_daily_rebalance", 
                "get_comprehensive_metrics",
                "emergency_flatten"
            ]
            
            print("\nProduction methods:")
            for method in production_methods:
                if hasattr(engine, method):
                    print(f"  ✅ {method}")
                else:
                    print(f"  ❌ {method}")
            
        except Exception as e:
            print(f"  ❌ Failed to instantiate ExecutionEngine: {e}")
            traceback.print_exc()
        
        return {"missing_methods": missing_methods, "methods": methods}
        
    except Exception as e:
        print(f"❌ Failed to import ExecutionEngine: {e}")
        return {"error": str(e)}

def test_portfolio_manager_integration():
    """Test portfolio manager integration."""
    print_section("PORTFOLIO MANAGER INTEGRATION")
    
    try:
        from execution.portfolio import ProductionPortfolioManager
        
        print("Testing ProductionPortfolioManager instantiation:")
        portfolio = ProductionPortfolioManager(total_capital=1000.0)
        print("  ✅ ProductionPortfolioManager instantiated")
        
        # Test expected methods
        expected_methods = [
            "get_portfolio_summary",
            "update_volatility_data",
            "update_correlation_data", 
            "rebalance_portfolio",
            "should_rebalance",
            "is_high_volatility_regime"
        ]
        
        print("\\nChecking expected methods:")
        for method in expected_methods:
            if hasattr(portfolio, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING!")
        
        # Test portfolio summary structure
        print("\\nTesting portfolio summary:")
        try:
            summary = portfolio.get_portfolio_summary()
            expected_keys = ["total_capital", "allocated_capital", "allocation_percentage", "active_positions"]
            print("  Portfolio summary keys:")
            for key in expected_keys:
                if key in summary:
                    print(f"    ✅ {key}: {summary[key]}")
                else:
                    print(f"    ❌ {key} - MISSING!")
        except Exception as e:
            print(f"  ❌ Failed to get portfolio summary: {e}")
        
    except Exception as e:
        print(f"❌ Failed to test ProductionPortfolioManager: {e}")
        traceback.print_exc()

def test_risk_manager_integration():
    """Test risk manager integration."""
    print_section("RISK MANAGER INTEGRATION")
    
    try:
        from execution.portfolio import ProductionPortfolioManager
        from execution.risk_manager import ProductionRiskManager
        
        print("Testing ProductionRiskManager instantiation:")
        portfolio = ProductionPortfolioManager(total_capital=1000.0)
        risk_manager = ProductionRiskManager(portfolio_manager=portfolio)
        print("  ✅ ProductionRiskManager instantiated")
        
        # Test expected methods
        expected_methods = [
            "get_risk_metrics",
            "validate_trade",
            "calculate_position_size",
            "calculate_dynamic_leverage",
            "check_kill_switches"
        ]
        
        print("\\nChecking expected methods:")
        for method in expected_methods:
            if hasattr(risk_manager, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING!")
        
        # Test risk metrics structure
        print("\\nTesting risk metrics:")
        try:
            metrics = risk_manager.get_risk_metrics()
            print("  Risk metrics keys:")
            for key in metrics:
                print(f"    ✅ {key}: {metrics[key]}")
        except Exception as e:
            print(f"  ❌ Failed to get risk metrics: {e}")
        
    except Exception as e:
        print(f"❌ Failed to test ProductionRiskManager: {e}")
        traceback.print_exc()

def test_backtesting_compatibility():
    """Test backtesting engine compatibility."""
    print_section("BACKTESTING COMPATIBILITY")
    
    try:
        from backtest.backtesting_engine import BacktestingEngine
        from algorithm.strategies.ma_crossover import MACrossoverStrategy
        
        print("Testing BacktestingEngine instantiation:")
        
        # Create a simple strategy
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'leverage': 5
        })
        
        symbols = [("BTCUSDT", "1m")]
        backtest = BacktestingEngine(
            symbols=symbols,
            strategy=strategy,
            start="2024-01-01",
            end="2024-01-02",
            initial_capital=1000.0
        )
        print("  ✅ BacktestingEngine instantiated successfully")
        
        # Check if it has access to execution engine
        if hasattr(backtest, 'execution_engine'):
            print("  ✅ BacktestingEngine has execution_engine attribute")
            
            # Check execution engine type
            exec_engine = backtest.execution_engine
            print(f"  ✅ Execution engine type: {type(exec_engine).__name__}")
        else:
            print("  ❌ BacktestingEngine missing execution_engine attribute")
        
    except Exception as e:
        print(f"❌ Failed to test BacktestingEngine: {e}")
        traceback.print_exc()

def test_main_algorithm_compatibility():
    """Test main algorithm compatibility."""
    print_section("MAIN ALGORITHM COMPATIBILITY")
    
    try:
        # Import the main TradingAlgorithm class
        from main import TradingAlgorithm
        from algorithm.strategies.ma_crossover import MACrossoverStrategy
        
        print("Testing TradingAlgorithm instantiation:")
        
        strategy = MACrossoverStrategy(params={
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'leverage': 5
        })
        
        algorithm = TradingAlgorithm(
            strategy=strategy,
            testnet=True,
            total_capital=1000.0
        )
        print("  ✅ TradingAlgorithm instantiated successfully")
        
        # Test execution engine access
        if hasattr(algorithm, 'execution_engine'):
            engine = algorithm.execution_engine
            print(f"  ✅ Has execution_engine: {type(engine).__name__}")
            
            # Test expected methods
            expected_methods = ["get_portfolio_summary", "get_risk_metrics", "validate_signal", "process_signal"]
            for method in expected_methods:
                if hasattr(engine, method):
                    print(f"    ✅ {method}")
                else:
                    print(f"    ❌ {method} - MISSING!")
        else:
            print("  ❌ TradingAlgorithm missing execution_engine")
        
    except Exception as e:
        print(f"❌ Failed to test TradingAlgorithm: {e}")
        traceback.print_exc()

def test_method_signatures():
    """Test method signatures for compatibility."""
    print_section("METHOD SIGNATURE COMPATIBILITY")
    
    try:
        from execution.execution_engine import ExecutionEngine
        
        # Mock binance client
        class MockBinanceClient:
            pass
        
        engine = ExecutionEngine(MockBinanceClient(), total_capital=1000.0)
        
        # Test process_signal signature
        if hasattr(engine, 'process_signal'):
            sig = inspect.signature(engine.process_signal)
            print(f"process_signal signature: {sig}")
            
            # Check if it's async
            if inspect.iscoroutinefunction(engine.process_signal):
                print("  ✅ process_signal is async")
            else:
                print("  ❌ process_signal is not async")
        else:
            print("  ❌ process_signal method missing")
        
        # Test other critical methods
        methods_to_check = ["validate_signal", "update_daily_pnl"]
        for method_name in methods_to_check:
            if hasattr(engine, method_name):
                method = getattr(engine, method_name)
                sig = inspect.signature(method)
                print(f"{method_name} signature: {sig}")
            else:
                print(f"  ❌ {method_name} method missing")
        
    except Exception as e:
        print(f"❌ Failed to test method signatures: {e}")
        traceback.print_exc()

def generate_fix_recommendations():
    """Generate recommendations for fixing the issues."""
    print_section("FIX RECOMMENDATIONS")
    
    print("Based on the diagnostic results, here are the recommended fixes:")
    print()
    print("1. MISSING EXECUTION ENGINE METHODS:")
    print("   - Add get_portfolio_summary() method to delegate to portfolio_manager")
    print("   - Add get_risk_metrics() method to delegate to risk_manager")  
    print("   - Add validate_signal() method to delegate to risk_manager")
    print("   - Add update_daily_pnl() method to delegate to risk_manager")
    print()
    print("2. BACKTESTING COMPATIBILITY:")
    print("   - Ensure ExecutionEngine works with SimBroker")
    print("   - Test backtesting with production modules")
    print()
    print("3. INTEGRATION TESTING:")
    print("   - Create unit tests for each module")
    print("   - Create integration tests for module interactions")
    print("   - Test full system end-to-end")
    print()
    print("4. ERROR HANDLING:")
    print("   - Add proper error handling for missing dependencies")
    print("   - Add validation for input parameters")
    print("   - Add logging for debugging")

def main():
    """Run all diagnostics."""
    print_section("CRYPTO TRADING ALGORITHM SYSTEM DIAGNOSTICS")
    print("Analyzing system for integration issues...")
    
    # Run all diagnostic tests
    test_imports()
    test_execution_engine_api()
    test_portfolio_manager_integration()
    test_risk_manager_integration()
    test_backtesting_compatibility()
    test_main_algorithm_compatibility()
    test_method_signatures()
    generate_fix_recommendations()
    
    print_section("DIAGNOSTICS COMPLETE")
    print("Review the results above to identify and fix integration issues.")

if __name__ == "__main__":
    main()
