#!/usr/bin/env python3
"""
Comprehensive Test Framework for Trading Algorithm
Senior Quantitative Developer Testing Protocol

This framework implements a systematic testing approach for:
1. Algorithm Engine (Signal Generation & Processing)
2. Execution Engine (Portfolio Allocation & Risk Management)
3. Integration Testing (End-to-End Signal → Execution Pipeline)

Testing Philosophy:
- Behavior-first debugging with interface validation
- Stress simulations for edge cases
- Statistical validation of calculations
- State consistency verification
"""

import os
import sys
import asyncio
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TestResult:
    """Structured test result with detailed diagnostics."""
    test_name: str
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

class ComprehensiveTestFramework:
    """Comprehensive testing framework for algorithm and execution engines."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the test framework.
        
        Args:
            verbose: Enable detailed logging during tests.
        """
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.mock_data_cache: Dict[str, Any] = {}
        
        # Test configuration
        self.test_capital = 10000.0
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        self.test_timeframes = ["1m", "5m"]
        
        print("🔬 [TestFramework] Initialized Comprehensive Testing Framework")
        print(f"🔬 [TestFramework] Test Capital: ${self.test_capital}")
        print(f"🔬 [TestFramework] Test Symbols: {self.test_symbols}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and level."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"🔬 [{timestamp}] [{level}] {message}")
    
    def generate_mock_ohlcv_data(self, symbol: str, periods: int = 100, 
                                base_price: float = 50000.0, volatility: float = 0.02) -> List[List]:
        """Generate realistic OHLCV data for testing.
        
        Args:
            symbol: Trading pair symbol.
            periods: Number of candles to generate.
            base_price: Starting price.
            volatility: Price volatility (standard deviation).
            
        Returns:
            List of OHLCV candles [timestamp, open, high, low, close, volume].
        """
        cache_key = f"{symbol}_{periods}_{base_price}_{volatility}"
        if cache_key in self.mock_data_cache:
            return self.mock_data_cache[cache_key]
        
        current_time = int(time.time() * 1000)
        candles = []
        current_price = base_price
        
        for i in range(periods):
            timestamp = current_time - (periods - i) * 60000  # 1-minute candles
            
            # Generate realistic price movement
            price_change = np.random.normal(0, volatility * current_price)
            new_price = max(current_price + price_change, current_price * 0.95)  # Prevent negative prices
            
            # Generate OHLC
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
            volume = np.random.uniform(100, 1000)
            
            candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
            current_price = new_price
        
        self.mock_data_cache[cache_key] = candles
        return candles
    
    def calculate_atr(self, candles: List[List], period: int = 14) -> float:
        """Calculate Average True Range for testing."""
        if len(candles) < period + 1:
            return 0.001  # Return minimum ATR
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i][2]
            low = candles[i][3]
            prev_close = candles[i-1][4]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Calculate ATR as simple moving average of true ranges
        if len(true_ranges) >= period:
            return np.mean(true_ranges[-period:])
        else:
            return np.mean(true_ranges) if true_ranges else 0.001

    async def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """Run a single test with error handling and timing.
        
        Args:
            test_func: Test function to execute.
            test_name: Name of the test.
            *args, **kwargs: Arguments to pass to test function.
            
        Returns:
            TestResult with detailed diagnostics.
        """
        start_time = time.time()
        errors = []
        warnings = []
        details = {}
        metrics = {}
        passed = False
        
        try:
            self.log(f"Running test: {test_name}")
            
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(*args, **kwargs)
            else:
                result = test_func(*args, **kwargs)
            
            if isinstance(result, dict):
                passed = result.get('passed', False)
                details = result.get('details', {})
                errors = result.get('errors', [])
                warnings = result.get('warnings', [])
                metrics = result.get('metrics', {})
            else:
                passed = bool(result)
                
        except Exception as e:
            errors.append(f"Test execution failed: {str(e)}")
            details['exception'] = traceback.format_exc()
            self.log(f"Test {test_name} failed with exception: {e}", "ERROR")
        
        execution_time = time.time() - start_time
        
        test_result = TestResult(
            test_name=test_name,
            passed=passed,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
        
        self.results.append(test_result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.log(f"Test {test_name}: {status} ({execution_time:.3f}s)")
        
        if errors:
            for error in errors:
                self.log(f"  Error: {error}", "ERROR")
        
        if warnings:
            for warning in warnings:
                self.log(f"  Warning: {warning}", "WARNING")
        
        return test_result
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report with statistics."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        # Calculate test coverage metrics
        algorithm_tests = [r for r in self.results if 'algorithm' in r.test_name.lower()]
        execution_tests = [r for r in self.results if 'execution' in r.test_name.lower()]
        integration_tests = [r for r in self.results if 'integration' in r.test_name.lower()]
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time,
                'average_execution_time': avg_time
            },
            'coverage': {
                'algorithm_engine_tests': len(algorithm_tests),
                'execution_engine_tests': len(execution_tests),
                'integration_tests': len(integration_tests)
            },
            'failed_tests': [
                {
                    'name': r.test_name,
                    'errors': r.errors,
                    'details': r.details
                }
                for r in self.results if not r.passed
            ],
            'performance_metrics': {
                'fastest_test': min(self.results, key=lambda x: x.execution_time).test_name if self.results else None,
                'slowest_test': max(self.results, key=lambda x: x.execution_time).test_name if self.results else None,
                'tests_over_1s': len([r for r in self.results if r.execution_time > 1.0])
            }
        }
        
        return report
    
    def print_test_report(self):
        """Print formatted test report."""
        report = self.generate_test_report()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Summary
        summary = report['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Execution Time: {summary['total_execution_time']:.3f}s")
        print(f"Average Test Time: {summary['average_execution_time']:.3f}s")
        
        # Coverage
        coverage = report['coverage']
        print(f"\nTest Coverage:")
        print(f"  Algorithm Engine Tests: {coverage['algorithm_engine_tests']}")
        print(f"  Execution Engine Tests: {coverage['execution_engine_tests']}")
        print(f"  Integration Tests: {coverage['integration_tests']}")
        
        # Failed tests detail
        if report['failed_tests']:
            print(f"\nFailed Tests Detail:")
            for failed in report['failed_tests']:
                print(f"  ❌ {failed['name']}")
                for error in failed['errors']:
                    print(f"    Error: {error}")
        
        # Performance metrics
        perf = report['performance_metrics']
        print(f"\nPerformance Metrics:")
        if perf['fastest_test']:
            print(f"  Fastest Test: {perf['fastest_test']}")
        if perf['slowest_test']:
            print(f"  Slowest Test: {perf['slowest_test']}")
        print(f"  Tests >1s: {perf['tests_over_1s']}")
        
        print("=" * 80)

if __name__ == "__main__":
    # Basic framework test
    framework = ComprehensiveTestFramework()
    
    def sample_test():
        return {'passed': True, 'details': {'message': 'Sample test passed'}}
    
    async def main():
        await framework.run_test(sample_test, "sample_test")
        framework.print_test_report()
    
    asyncio.run(main())
