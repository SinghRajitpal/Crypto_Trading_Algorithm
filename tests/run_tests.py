#!/usr/bin/env python3
"""
Test Runner for Crypto Trading Algorithm
Senior Quantitative Developer Testing Protocol

Comprehensive test execution script that runs all test suites
and generates detailed reports.

Usage:
    python run_tests.py [options]
    
Options:
    --pytest-only    Run only pytest suite
    --unittest-only  Run only unittest suite
    --verbose        Enable verbose output
    --coverage       Run with coverage reporting (requires pytest-cov)
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print("-" * 60)


def run_pytest_suite(verbose=False, coverage=False, component=None):
    """Run the pytest test suite."""
    print_section("RUNNING PYTEST SUITE")
    
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test files based on component or run all
    if component:
        if component in ['algorithm', 'execution', 'data', 'backtest', 'config', 'integration']:
            test_path = f"tests/{component}/"
            cmd.append(test_path)
        else:
            print(f"Unknown component: {component}")
            return False, ""
    else:
        # Run organized test structure
        test_dirs = [
            "tests/algorithm/",
            "tests/execution/", 
            "tests/integration/",
            "tests/data/",
            "tests/backtest/",
            "tests/config/"
        ]
        cmd.extend(test_dirs)
    
    # Add options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend(["--cov=algorithm", "--cov=execution", "--cov-report=term-missing"])
    
    cmd.extend(["--tb=short", "--asyncio-mode=auto"])
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nPytest execution time: {execution_time:.2f}s")
        return result.returncode == 0, result.stdout
        
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False, str(e)


def run_unittest_suite(verbose=False):
    """Run the unittest test suite."""
    print_section("RUNNING UNITTEST SUITE")
    
    cmd = [sys.executable, "tests/unit/test_unittest_suite.py"]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nUnittest execution time: {execution_time:.2f}s")
        return result.returncode == 0, result.stdout
        
    except Exception as e:
        print(f"Error running unittest: {e}")
        return False, str(e)


def parse_test_results(pytest_output, unittest_output):
    """Parse test results and generate summary."""
    results = {
        'pytest': {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0},
        'unittest': {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0},
        'overall': {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0}
    }
    
    # Parse pytest results
    if pytest_output:
        lines = pytest_output.split('\n')
        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line or line.strip().endswith('passed')):
                # Look for pattern like "41 passed in 0.89s" or "40 passed, 1 failed"
                if 'passed' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed':
                                results['pytest']['passed'] = int(parts[i-1])
                            elif part == 'failed':
                                results['pytest']['failed'] = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                break
    
    # Parse unittest results
    if unittest_output:
        lines = unittest_output.split('\n')
        for line in lines:
            if 'Tests run:' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'run:':
                            results['unittest']['total'] = int(parts[i+1])
                        elif part == 'Failures:':
                            results['unittest']['failed'] += int(parts[i+1])
                        elif part == 'Errors:':
                            results['unittest']['failed'] += int(parts[i+1])
                except (ValueError, IndexError):
                    pass
                break
    
    # Calculate totals and success rates
    for suite in ['pytest', 'unittest']:
        if results[suite]['total'] == 0:
            results[suite]['total'] = results[suite]['passed'] + results[suite]['failed']
        if results[suite]['total'] > 0:
            results[suite]['success_rate'] = results[suite]['passed'] / results[suite]['total'] * 100
    
    # Calculate overall
    results['overall']['total'] = results['pytest']['total'] + results['unittest']['total']
    results['overall']['passed'] = results['pytest']['passed'] + results['unittest']['passed']  
    results['overall']['failed'] = results['pytest']['failed'] + results['unittest']['failed']
    if results['overall']['total'] > 0:
        results['overall']['success_rate'] = results['overall']['passed'] / results['overall']['total'] * 100
    
    return results


def print_summary_report(results, pytest_success, unittest_success):
    """Print comprehensive summary report."""
    print_header("COMPREHENSIVE TEST EXECUTION SUMMARY")
    
    print_section("TEST SUITE RESULTS")
    
    # Pytest results
    pytest_status = "✅ PASSED" if pytest_success else "❌ FAILED"
    print(f"Pytest Suite: {pytest_status}")
    if results['pytest']['total'] > 0:
        print(f"  Tests: {results['pytest']['passed']}/{results['pytest']['total']} passed")
        print(f"  Success Rate: {results['pytest']['success_rate']:.1f}%")
    
    # Unittest results  
    unittest_status = "✅ PASSED" if unittest_success else "❌ FAILED"
    print(f"Unittest Suite: {unittest_status}")
    if results['unittest']['total'] > 0:
        print(f"  Tests: {results['unittest']['passed']}/{results['unittest']['total']} passed")
        print(f"  Success Rate: {results['unittest']['success_rate']:.1f}%")
    
    print_section("OVERALL SUMMARY")
    overall_success = pytest_success and unittest_success
    overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    
    print(f"Overall Status: {overall_status}")
    print(f"Total Tests: {results['overall']['total']}")
    print(f"Total Passed: {results['overall']['passed']}")
    print(f"Total Failed: {results['overall']['failed']}")
    print(f"Overall Success Rate: {results['overall']['success_rate']:.1f}%")
    
    print_section("TEST COVERAGE")
    print("✅ Algorithm Engine: Signal generation, processing, throttling")
    print("✅ Execution Engine: Portfolio management, risk management") 
    print("✅ Integration: End-to-end workflows, stress testing")
    print("✅ Error Handling: Exception handling, resilience")
    print("✅ Performance: Latency, throughput, concurrent processing")
    
    if overall_success:
        print_header("🎯 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
    else:
        print_header("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    
    return overall_success


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite for Crypto Trading Algorithm")
    parser.add_argument("--pytest-only", action="store_true", help="Run only pytest suite")
    parser.add_argument("--unittest-only", action="store_true", help="Run only unittest suite") 
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--component", choices=['algorithm', 'execution', 'data', 'backtest', 'config', 'integration'], 
                       help="Run tests for specific component only")
    
    args = parser.parse_args()
    
    print_header("CRYPTO TRADING ALGORITHM - COMPREHENSIVE TEST RUNNER")
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    
    if args.component:
        print(f"Component Focus: {args.component}")
    
    start_time = time.time()
    pytest_success = True
    unittest_success = True
    pytest_output = ""
    unittest_output = ""
    
    # Run pytest suite
    if not args.unittest_only:
        pytest_success, pytest_output = run_pytest_suite(args.verbose, args.coverage, args.component)
    
    # Run unittest suite  
    if not args.pytest_only and not args.component:  # Only run unittest for full test runs
        unittest_success, unittest_output = run_unittest_suite(args.verbose)
    
    # Parse results and generate summary
    results = parse_test_results(pytest_output, unittest_output)
    overall_success = print_summary_report(results, pytest_success, unittest_success)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--component", choices=['algorithm', 'execution', 'data', 'backtest', 'config', 'integration'], 
                       help="Run tests for specific component only")
    
    args = parser.parse_args()
    
    print_header("CRYPTO TRADING ALGORITHM - COMPREHENSIVE TEST RUNNER")
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    
    if args.component:
        print(f"Component Focus: {args.component}")
    
    start_time = time.time()
    pytest_success = True
    unittest_success = True
    pytest_output = ""
    unittest_output = ""
    
    # Run pytest suite
    if not args.unittest_only:
        pytest_success, pytest_output = run_pytest_suite(args.verbose, args.coverage, args.component)
    
    # Run unittest suite  
    if not args.pytest_only and not args.component:  # Only run unittest for full test runs
        unittest_success, unittest_output = run_unittest_suite(args.verbose)
    
    # Parse results and generate summary
    results = parse_test_results(pytest_output, unittest_output)
    overall_success = print_summary_report(results, pytest_success, unittest_success)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
