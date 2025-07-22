#!/usr/bin/env python3
"""
Integration tests for the new risk management system.

This test suite identifies and isolates issues caused by the recent integration
of production risk management, portfolio allocation, leverage management, 
and stress handling components.
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def print_test_header(test_name: str):
    """Print formatted test header."""
    print("\n" + "="*80)
    print(f" {test_name}")
    print("="*80)

def print_test_result(test_name: str, success: bool, error: str = None):
    """Print formatted test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status}: {test_name}")
    if error:
        print(f"   Error: {error}")

class ExecutionEngineIntegrationTests:
    """Test suite for execution engine integration issues."""
    
    def __init__(self):
        self.results = {}
        
    async def run_all_tests(self):
        """Run all integration tests."""
        print_test_header("EXECUTION ENGINE INTEGRATION TESTS")
        
        # Test individual components first
        await self.test_component_imports()
        await self.test_component_instantiation()
        await self.test_component_interfaces()
        
        # Test integration points
        await self.test_execution_engine_integration()
        await self.test_signal_processing()
        await self.test_market_data_flow()
        
        # Test live trading compatibility
        await self.test_live_trading_compatibility()
        
        # Test backtesting compatibility
        await self.test_backtesting_compatibility()
        
        # Print summary
        self.print_test_summary()
    
    async def test_component_imports(self):
        """Test that all new components can be imported."""
        print_test_header("Component Import Tests")
        
        components = [
            ("ProductionPortfolioManager", "execution.portfolio"),
            ("ProductionRiskManager", "execution.risk_manager"),
            ("StressHandlingModule", "execution.stress_handler"),
            ("ProductionExecutionEngine", "execution.execution_engine"),
            ("OrderExecutor", "execution.executor")
        ]
        
        for component_name, module_path in components:
            try:
                module = __import__(module_path, fromlist=[component_name])
                component_class = getattr(module, component_name)
                print_test_result(f"Import {component_name}", True)
                self.results[f"import_{component_name}"] = True
            except Exception as e:
                print_test_result(f"Import {component_name}", False, str(e))
                self.results[f"import_{component_name}"] = False
    
    async def test_component_instantiation(self):
        """Test that all components can be instantiated."""
        print_test_header("Component Instantiation Tests")
        
        try:
            # Test portfolio manager
            from execution.portfolio import ProductionPortfolioManager
            portfolio = ProductionPortfolioManager(total_capital=1000.0)
            print_test_result("ProductionPortfolioManager instantiation", True)
            self.results["portfolio_instantiation"] = True
            
            # Test risk manager
            from execution.risk_manager import ProductionRiskManager
            risk_manager = ProductionRiskManager(portfolio_manager=portfolio)
            print_test_result("ProductionRiskManager instantiation", True)
            self.results["risk_manager_instantiation"] = True
            
            # Test execution engine (mock binance client)
            from execution.execution_engine import ExecutionEngine
            
            class MockBinanceClient:
                async def close(self):
                    pass
                    
                async def get_open_positions(self, symbol=None):
                    return []
                    
                async def get_all_positions(self):
                    return []
            
            mock_client = MockBinanceClient()
            execution_engine = ExecutionEngine(
                binance_client=mock_client,
                total_capital=1000.0
            )
            print_test_result("ExecutionEngine instantiation", True)
            self.results["execution_engine_instantiation"] = True
            
        except Exception as e:
            print_test_result("Component instantiation", False, str(e))
            traceback.print_exc()
            self.results["component_instantiation"] = False
    
    async def test_component_interfaces(self):
        """Test that components have expected interfaces."""
        print_test_header("Component Interface Tests")
        
        try:
            from execution.portfolio import ProductionPortfolioManager
            from execution.risk_manager import ProductionRiskManager
            
            portfolio = ProductionPortfolioManager(total_capital=1000.0)
            risk_manager = ProductionRiskManager(portfolio_manager=portfolio)
            
            # Test portfolio manager interface
            required_portfolio_methods = [
                "get_portfolio_summary",
                "update_volatility_data", 
                "update_correlation_data",
                "rebalance_portfolio",
                "should_rebalance"
            ]
            
            for method in required_portfolio_methods:
                if hasattr(portfolio, method):
                    print_test_result(f"Portfolio.{method}", True)
                    self.results[f"portfolio_{method}"] = True
                else:
                    print_test_result(f"Portfolio.{method}", False, "Method missing")
                    self.results[f"portfolio_{method}"] = False
            
            # Test risk manager interface
            required_risk_methods = [
                "get_risk_metrics",
                "validate_trade",
                "calculate_position_size",
                "calculate_dynamic_leverage",
                "check_kill_switches"
            ]
            
            for method in required_risk_methods:
                if hasattr(risk_manager, method):
                    print_test_result(f"RiskManager.{method}", True)
                    self.results[f"risk_manager_{method}"] = True
                else:
                    print_test_result(f"RiskManager.{method}", False, "Method missing")
                    self.results[f"risk_manager_{method}"] = False
                    
        except Exception as e:
            print_test_result("Component interfaces", False, str(e))
            traceback.print_exc()
            self.results["component_interfaces"] = False
    
    async def test_execution_engine_integration(self):
        """Test execution engine integration with new components."""
        print_test_header("Execution Engine Integration Tests")
        
        try:
            from execution.execution_engine import ExecutionEngine
            
            class MockBinanceClient:
                async def close(self):
                    pass
                async def get_open_positions(self, symbol=None):
                    return []
                async def get_all_positions(self):
                    return []
            
            mock_client = MockBinanceClient()
            execution_engine = ExecutionEngine(
                binance_client=mock_client,
                total_capital=1000.0
            )
            
            # Test market data update
            try:
                execution_engine.update_market_data_bar(
                    symbol="BTCUSDT",
                    ohlcv_data={"open": 50000, "high": 51000, "low": 49000, "close": 50500, "volume": 1000},
                    atr_value=0.02,
                    correlation_data={"ETHUSDT": 0.8}
                )
                print_test_result("Market data update", True)
                self.results["market_data_update"] = True
            except Exception as e:
                print_test_result("Market data update", False, str(e))
                self.results["market_data_update"] = False
            
            # Test portfolio summary
            try:
                summary = execution_engine.get_portfolio_summary()
                expected_keys = ["total_capital", "allocated_capital", "allocation_percentage"]
                all_keys_present = all(key in summary for key in expected_keys)
                print_test_result("Portfolio summary", all_keys_present)
                self.results["portfolio_summary"] = all_keys_present
            except Exception as e:
                print_test_result("Portfolio summary", False, str(e))
                self.results["portfolio_summary"] = False
            
            # Test risk metrics
            try:
                metrics = execution_engine.get_risk_metrics()
                print_test_result("Risk metrics", True)
                self.results["risk_metrics"] = True
            except Exception as e:
                print_test_result("Risk metrics", False, str(e))
                self.results["risk_metrics"] = False
                
        except Exception as e:
            print_test_result("Execution engine integration", False, str(e))
            traceback.print_exc()
            self.results["execution_engine_integration"] = False
    
    async def test_signal_processing(self):
        """Test signal processing with new risk management."""
        print_test_header("Signal Processing Tests")
        
        try:
            from execution.execution_engine import ExecutionEngine
            from algorithm.trade_signal import TradeSignal
            
            class MockBinanceClient:
                async def close(self):
                    pass
                async def get_open_positions(self, symbol=None):
                    return []
                async def get_all_positions(self):
                    return []
            
            mock_client = MockBinanceClient()
            execution_engine = ExecutionEngine(
                binance_client=mock_client,
                total_capital=1000.0
            )
            
            # Create a test signal
            signal = TradeSignal(
                action="open",
                side="buy",  # should be "buy" not "long" for futures
                symbol="BTCUSDT",
                strategy_id="test_strategy",
                metadata={
                    "price": 50000.0,
                    "atr_value": 0.02,
                    "reason": "MA crossover"
                },
                signal_confidence=0.8
            )
            
            # Test signal processing
            try:
                result = await execution_engine.process_signal(signal)
                print_test_result("Signal processing", True)
                self.results["signal_processing"] = True
                print(f"   Result: {result}")
            except Exception as e:
                print_test_result("Signal processing", False, str(e))
                self.results["signal_processing"] = False
                
        except Exception as e:
            print_test_result("Signal processing setup", False, str(e))
            traceback.print_exc()
            self.results["signal_processing_setup"] = False
    
    async def test_market_data_flow(self):
        """Test market data flow through the system."""
        print_test_header("Market Data Flow Tests")
        
        try:
            from execution.execution_engine import ExecutionEngine
            
            class MockBinanceClient:
                async def close(self):
                    pass
                async def get_open_positions(self, symbol=None):
                    return []
                async def get_all_positions(self):
                    return []
            
            mock_client = MockBinanceClient()
            execution_engine = ExecutionEngine(
                binance_client=mock_client,
                total_capital=1000.0
            )
            
            # Test multiple market data updates
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            
            for i, symbol in enumerate(symbols):
                try:
                    execution_engine.update_market_data_bar(
                        symbol=symbol,
                        ohlcv_data={
                            "open": 50000 + i*1000, 
                            "high": 51000 + i*1000, 
                            "low": 49000 + i*1000, 
                            "close": 50500 + i*1000, 
                            "volume": 1000
                        },
                        atr_value=0.02 + i*0.005,
                        correlation_data={other: 0.7 for other in symbols if other != symbol}
                    )
                    print_test_result(f"Market data flow - {symbol}", True)
                except Exception as e:
                    print_test_result(f"Market data flow - {symbol}", False, str(e))
                    self.results[f"market_data_{symbol}"] = False
                    continue
            
            # Test rebalancing trigger
            try:
                rebalance_result = execution_engine.process_daily_rebalance()
                print_test_result("Daily rebalance", True)
                self.results["daily_rebalance"] = True
                print(f"   Rebalance triggered: {rebalance_result}")
            except Exception as e:
                print_test_result("Daily rebalance", False, str(e))
                self.results["daily_rebalance"] = False
                
        except Exception as e:
            print_test_result("Market data flow setup", False, str(e))
            traceback.print_exc()
            self.results["market_data_flow"] = False
    
    async def test_live_trading_compatibility(self):
        """Test compatibility with live trading system."""
        print_test_header("Live Trading Compatibility Tests")
        
        try:
            from main import TradingAlgorithm
            from algorithm.strategies.ma_crossover import MACrossoverStrategy
            
            strategy = MACrossoverStrategy(params={
                'fast_ma_period': 5,
                'slow_ma_period': 20,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02,
                'leverage': 7
            })
            
            algorithm = TradingAlgorithm(
                strategy=strategy, 
                testnet=True,
                total_capital=1000.0
            )
            
            print_test_result("Live trading algorithm instantiation", True)
            self.results["live_trading_instantiation"] = True
            
            # Test that required methods exist
            required_methods = ["start", "stop"]
            for method in required_methods:
                if hasattr(algorithm, method):
                    print_test_result(f"Live trading - {method} method", True)
                    self.results[f"live_trading_{method}"] = True
                else:
                    print_test_result(f"Live trading - {method} method", False, "Method missing")
                    self.results[f"live_trading_{method}"] = False
            
        except Exception as e:
            print_test_result("Live trading compatibility", False, str(e))
            traceback.print_exc()
            self.results["live_trading_compatibility"] = False
    
    async def test_backtesting_compatibility(self):
        """Test compatibility with backtesting system."""
        print_test_header("Backtesting Compatibility Tests")
        
        try:
            from backtest.backtesting_engine import BacktestingEngine
            from algorithm.strategies.ma_crossover import MACrossoverStrategy
            
            strategy = MACrossoverStrategy(params={
                'fast_ma_period': 5,
                'slow_ma_period': 20,
            })
            
            symbols = [('BTCUSDT', '1m')]
            engine = BacktestingEngine(
                symbols=symbols,
                strategy=strategy,
                start='2024-01-01',
                end='2024-01-02',
                initial_capital=1000.0
            )
            
            print_test_result("Backtesting engine instantiation", True)
            self.results["backtesting_instantiation"] = True
            
            # Test that required methods exist
            required_methods = ["run"]
            for method in required_methods:
                if hasattr(engine, method):
                    print_test_result(f"Backtesting - {method} method", True)
                    self.results[f"backtesting_{method}"] = True
                else:
                    print_test_result(f"Backtesting - {method} method", False, "Method missing")
                    self.results[f"backtesting_{method}"] = False
                    
        except Exception as e:
            print_test_result("Backtesting compatibility", False, str(e))
            traceback.print_exc()
            self.results["backtesting_compatibility"] = False
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print_test_header("TEST SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_name, result in self.results.items():
                if not result:
                    print(f"  ❌ {test_name}")
        
        print("\nTEST ANALYSIS:")
        self.analyze_failures()
    
    def analyze_failures(self):
        """Analyze failure patterns to identify root causes."""
        failed_tests = [name for name, result in self.results.items() if not result]
        
        # Categorize failures
        import_failures = [name for name in failed_tests if name.startswith("import_")]
        instantiation_failures = [name for name in failed_tests if "instantiation" in name]
        interface_failures = [name for name in failed_tests if "portfolio_" in name or "risk_manager_" in name]
        integration_failures = [name for name in failed_tests if "integration" in name]
        
        if import_failures:
            print("📁 IMPORT ISSUES:")
            print("   - Some components cannot be imported")
            print("   - Check for syntax errors or missing dependencies")
        
        if instantiation_failures:
            print("⚙️ INSTANTIATION ISSUES:")
            print("   - Components have initialization problems")
            print("   - Check __init__ methods and required parameters")
        
        if interface_failures:
            print("🔌 INTERFACE ISSUES:")
            print("   - Components missing expected methods")
            print("   - Check method signatures and implementations")
        
        if integration_failures:
            print("🔄 INTEGRATION ISSUES:")
            print("   - Components don't work together properly")
            print("   - Check data flow and component communication")


async def main():
    """Run the integration test suite."""
    test_suite = ExecutionEngineIntegrationTests()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
