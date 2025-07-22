"""
Comprehensive Test Suite Runner for Data and Algorithm Engines.

This script runs all unit and integration tests for the data and algorithm
components of the trading system, providing detailed diagnostics and
performance metrics.

Usage:
    python tests/test_runner_data_algorithm.py [options]

Options:
    --verbose, -v: Verbose output
    --unit-only: Run only unit tests
    --integration-only: Run only integration tests
    --performance: Include performance tests
    --coverage: Show coverage report (if coverage.py is installed)
"""

import sys
import os
import unittest
import time
import traceback
import asyncio
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test discovery patterns
UNIT_TEST_PATTERNS = [
    'tests.unit.data.test_data_engine',
    'tests.unit.data.test_data_processor',
    'tests.unit.algorithm.test_algo_engine',
    'tests.unit.algorithm.test_trade_signal',
    'tests.unit.algorithm.test_base_strategy',
]

INTEGRATION_TEST_PATTERNS = [
    'tests.integration.data_algorithm.test_data_algorithm_integration',
]

class TestResult:
    """Container for test results and metrics."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.start_time = None
        self.end_time = None
        self.failures = []
        self.errors = []
        
    @property
    def success_rate(self):
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    @property
    def execution_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class ComprehensiveTestRunner:
    """Comprehensive test runner with diagnostics."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = TestResult()
        
    def print_header(self, title):
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)
        
    def print_section(self, title):
        """Print formatted section header."""
        print(f"\n{'-' * 60}")
        print(f"{title}")
        print(f"{'-' * 60}")
        
    def run_unit_tests(self):
        """Run all unit tests."""
        self.print_section("UNIT TESTS")
        
        for test_module in UNIT_TEST_PATTERNS:
            try:
                print(f"\n🧪 Running {test_module}...")
                
                # Load and run test module
                suite = unittest.TestLoader().loadTestsFromName(test_module)
                stream = StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream,
                    verbosity=2 if self.verbose else 1
                )
                
                start_time = time.time()
                result = runner.run(suite)
                end_time = time.time()
                
                # Collect results
                self.results.total_tests += result.testsRun
                self.results.passed_tests += result.testsRun - len(result.failures) - len(result.errors)
                self.results.failed_tests += len(result.failures)
                self.results.error_tests += len(result.errors)
                self.results.failures.extend(result.failures)
                self.results.errors.extend(result.errors)
                
                # Print summary for this module
                execution_time = end_time - start_time
                print(f"   ✅ Tests run: {result.testsRun}")
                print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                
                if result.failures:
                    print(f"   ❌ Failures: {len(result.failures)}")
                    if self.verbose:
                        for test, failure in result.failures:
                            print(f"      FAIL: {test}")
                            print(f"      {failure}")
                
                if result.errors:
                    print(f"   💥 Errors: {len(result.errors)}")
                    if self.verbose:
                        for test, error in result.errors:
                            print(f"      ERROR: {test}")
                            print(f"      {error}")
                            
                if not result.failures and not result.errors:
                    print(f"   🎉 All tests passed!")
                    
            except Exception as e:
                print(f"   💥 Failed to run {test_module}: {e}")
                if self.verbose:
                    traceback.print_exc()
    
    def run_integration_tests(self):
        """Run all integration tests."""
        self.print_section("INTEGRATION TESTS")
        
        for test_module in INTEGRATION_TEST_PATTERNS:
            try:
                print(f"\n🔗 Running {test_module}...")
                
                # Load and run test module
                suite = unittest.TestLoader().loadTestsFromName(test_module)
                stream = StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream,
                    verbosity=2 if self.verbose else 1
                )
                
                start_time = time.time()
                result = runner.run(suite)
                end_time = time.time()
                
                # Collect results
                self.results.total_tests += result.testsRun
                self.results.passed_tests += result.testsRun - len(result.failures) - len(result.errors)
                self.results.failed_tests += len(result.failures)
                self.results.error_tests += len(result.errors)
                self.results.failures.extend(result.failures)
                self.results.errors.extend(result.errors)
                
                # Print summary for this module
                execution_time = end_time - start_time
                print(f"   ✅ Tests run: {result.testsRun}")
                print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                
                if result.failures:
                    print(f"   ❌ Failures: {len(result.failures)}")
                
                if result.errors:
                    print(f"   💥 Errors: {len(result.errors)}")
                    
                if not result.failures and not result.errors:
                    print(f"   🎉 All tests passed!")
                    
            except Exception as e:
                print(f"   💥 Failed to run {test_module}: {e}")
                if self.verbose:
                    traceback.print_exc()
    
    def run_performance_tests(self):
        """Run performance and stress tests."""
        self.print_section("PERFORMANCE TESTS")
        
        try:
            # Import modules for performance testing
            from data.data_engine import DataEngine
            from algorithm.algo_engine import AlgoEngine
            
            print("\n⚡ Running performance tests...")
            
            # Test 1: Large dataset processing
            print("   📊 Testing large dataset processing...")
            start_time = time.time()
            
            # Create large mock dataset
            large_dataset = []
            for i in range(1000):
                candle = [
                    1642680000000 + i * 60000,  # Timestamp
                    42000.0 + i * 0.1,           # Open
                    42100.0 + i * 0.1,           # High
                    41900.0 + i * 0.1,           # Low
                    42050.0 + i * 0.1,           # Close
                    100.0 + i                    # Volume
                ]
                large_dataset.append(candle)
            
            processing_time = time.time() - start_time
            print(f"      ✅ Processed 1000 candles in {processing_time:.3f}s")
            
            # Test 2: Memory usage test
            print("   💾 Testing memory usage patterns...")
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple engines
            engines = []
            for i in range(10):
                mock_client = type('MockClient', (), {'testnet': True})()
                with unittest.mock.patch('data.data_engine.DataFetcher'):
                    engine = DataEngine(mock_client, max_candles=100)
                    engines.append(engine)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"      ✅ Memory increase for 10 engines: {memory_increase:.1f}MB")
            
            # Test 3: Concurrent processing
            print("   🔄 Testing concurrent processing...")
            
            async def concurrent_test():
                tasks = []
                for i in range(10):
                    task = asyncio.create_task(asyncio.sleep(0.01))  # Simulate work
                    tasks.append(task)
                
                start = time.time()
                await asyncio.gather(*tasks)
                return time.time() - start
            
            concurrent_time = asyncio.run(concurrent_test())
            print(f"      ✅ Concurrent processing of 10 tasks: {concurrent_time:.3f}s")
            
        except Exception as e:
            print(f"   💥 Performance test error: {e}")
            if self.verbose:
                traceback.print_exc()
    
    def run_diagnostic_tests(self):
        """Run diagnostic tests for system health."""
        self.print_section("DIAGNOSTIC TESTS")
        
        try:
            print("\n🔍 Running diagnostic tests...")
            
            # Test 1: Module import health
            print("   📦 Testing module imports...")
            modules_to_test = [
                'data.data_engine',
                'data.processor', 
                'data.data_fetcher',
                'algorithm.algo_engine',
                'algorithm.trade_signal',
                'algorithm.strategies.base_strategy',
                'binance_exchange'
            ]
            
            import_failures = []
            for module in modules_to_test:
                try:
                    __import__(module)
                    print(f"      ✅ {module}")
                except ImportError as e:
                    print(f"      ❌ {module}: {e}")
                    import_failures.append((module, e))
            
            if import_failures:
                print(f"   💥 {len(import_failures)} import failures detected")
            else:
                print(f"   🎉 All modules imported successfully")
            
            # Test 2: Configuration validation
            print("   ⚙️  Testing configuration...")
            try:
                import config
                if hasattr(config, 'symbols') and config.symbols:
                    print(f"      ✅ Configuration loaded, {len(config.symbols)} symbols configured")
                else:
                    print(f"      ⚠️  Configuration loaded but no symbols found")
            except ImportError as e:
                print(f"      ❌ Configuration import failed: {e}")
            
            # Test 3: External dependencies
            print("   🔗 Testing external dependencies...")
            dependencies = ['ccxt', 'numpy', 'pandas', 'talib']
            
            for dep in dependencies:
                try:
                    __import__(dep)
                    print(f"      ✅ {dep}")
                except ImportError:
                    print(f"      ❌ {dep} - Not available")
            
        except Exception as e:
            print(f"   💥 Diagnostic test error: {e}")
            if self.verbose:
                traceback.print_exc()
    
    def print_final_summary(self):
        """Print final test summary."""
        self.print_header("TEST EXECUTION SUMMARY")
        
        print(f"📊 Total Tests:     {self.results.total_tests}")
        print(f"✅ Passed:          {self.results.passed_tests}")
        print(f"❌ Failed:          {self.results.failed_tests}")
        print(f"💥 Errors:          {self.results.error_tests}")
        print(f"⏭️  Skipped:         {self.results.skipped_tests}")
        print(f"📈 Success Rate:    {self.results.success_rate:.1f}%")
        print(f"⏱️  Execution Time: {self.results.execution_time:.2f}s")
        
        if self.results.success_rate == 100.0 and self.results.total_tests > 0:
            print(f"\n🎉 ALL TESTS PASSED! Data and Algorithm engines are healthy.")
        elif self.results.success_rate >= 90.0:
            print(f"\n✅ Tests mostly passed with {self.results.success_rate:.1f}% success rate.")
        elif self.results.success_rate >= 70.0:
            print(f"\n⚠️  Tests passed with warnings. Success rate: {self.results.success_rate:.1f}%")
        else:
            print(f"\n❌ CRITICAL: Low success rate of {self.results.success_rate:.1f}%. System needs attention.")
        
        # Print detailed failures if any
        if self.results.failures or self.results.errors:
            print(f"\n📋 DETAILED FAILURE REPORT:")
            
            for test, failure in self.results.failures:
                print(f"\n❌ FAILURE: {test}")
                print(f"   {failure[:200]}...")  # Truncate long failures
                
            for test, error in self.results.errors:
                print(f"\n💥 ERROR: {test}")
                print(f"   {error[:200]}...")  # Truncate long errors
    
    def run_all_tests(self, unit_only=False, integration_only=False, include_performance=False):
        """Run all tests with options."""
        self.results.start_time = time.time()
        
        self.print_header("DATA & ALGORITHM ENGINE TEST SUITE")
        
        if not integration_only:
            self.run_unit_tests()
            
        if not unit_only:
            self.run_integration_tests()
            
        if include_performance:
            self.run_performance_tests()
            
        self.run_diagnostic_tests()
        
        self.results.end_time = time.time()
        self.print_final_summary()
        
        return self.results.success_rate >= 90.0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for Data and Algorithm engines"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--performance", action="store_true",
                       help="Include performance tests")
    parser.add_argument("--coverage", action="store_true",
                       help="Show coverage report")
    
    args = parser.parse_args()
    
    # Run with coverage if requested
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            runner = ComprehensiveTestRunner(verbose=args.verbose)
            success = runner.run_all_tests(
                unit_only=args.unit_only,
                integration_only=args.integration_only,
                include_performance=args.performance
            )
            
            cov.stop()
            cov.save()
            
            print("\n📊 COVERAGE REPORT:")
            cov.report()
            
        except ImportError:
            print("Coverage.py not installed. Running without coverage.")
            runner = ComprehensiveTestRunner(verbose=args.verbose)
            success = runner.run_all_tests(
                unit_only=args.unit_only,
                integration_only=args.integration_only,
                include_performance=args.performance
            )
    else:
        # Run without coverage
        runner = ComprehensiveTestRunner(verbose=args.verbose)
        success = runner.run_all_tests(
            unit_only=args.unit_only,
            integration_only=args.integration_only,
            include_performance=args.performance
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
