#!/usr/bin/env python3
"""
Comprehensive Test Runner for Crypto Trading Algorithm.

This script provides a unified interface to run all types of tests:
- pytest unit tests
- pytest integration tests  
- unittest comprehensive tests
- live testnet validation
- stress tests
- mathematical validation

Usage:
    python run_comprehensive_tests.py [options]
    
Options:
    --unit          Run unit tests only
    --integration   Run integration tests only
    --stress        Run stress tests only
    --live          Run live testnet tests only
    --unittest      Run unittest suite only
    --mathematical  Run mathematical validation only
    --all           Run all tests (default)
    --fast          Skip slow tests
    --verbose       Increase verbosity
    --no-capture    Don't capture output
"""

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self, verbose: bool = False, no_capture: bool = False):
        self.verbose = verbose
        self.no_capture = no_capture
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"
        
        if level == "ERROR":
            print(f"\033[91m{prefix} {message}\033[0m")
        elif level == "SUCCESS":
            print(f"\033[92m{prefix} {message}\033[0m")
        elif level == "WARNING":
            print(f"\033[93m{prefix} {message}\033[0m")
        else:
            print(f"{prefix} {message}")
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(command)}")
        
        start_time = time.time()
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Run command
            result = subprocess.run(
                command,
                capture_output=not self.no_capture,
                text=True,
                env=env,
                timeout=300,  # 5 minutes timeout
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            if success:
                self.log(f"✅ {description} completed in {duration:.2f}s", "SUCCESS")
                self.passed_tests += 1
            else:
                self.log(f"❌ {description} failed in {duration:.2f}s", "ERROR")
                if result.stdout and self.verbose:
                    self.log(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    self.log(f"STDERR:\n{result.stderr}")
                self.failed_tests += 1
            
            return {
                'success': success,
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'description': description
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.log(f"⏰ {description} timed out after {duration:.2f}s", "ERROR")
            self.failed_tests += 1
            return {
                'success': False,
                'returncode': -1,
                'duration': duration,
                'stdout': '',
                'stderr': 'Command timed out',
                'description': description
            }
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"💥 {description} crashed: {str(e)}", "ERROR")
            self.failed_tests += 1
            return {
                'success': False,
                'returncode': -2,
                'duration': duration,
                'stdout': '',
                'stderr': str(e),
                'description': description
            }
    
    def run_pytest_tests(self, markers: List[str] = None, extra_args: List[str] = None) -> Dict[str, Any]:
        """Run pytest with specified markers."""
        command = ['python3', '-m', 'pytest']
        
        if markers:
            for marker in markers:
                command.extend(['-m', marker])
        
        # Add standard pytest args
        command.extend([
            '--tb=short',
            '--strict-markers',
            '-v' if self.verbose else '-q'
        ])
        
        if self.no_capture:
            command.append('-s')
        
        if extra_args:
            command.extend(extra_args)
        
        # Add test directory
        command.append('tests/')
        
        description = f"pytest tests"
        if markers:
            description += f" (markers: {', '.join(markers)})"
        
        return self.run_command(command, description)
    
    def run_unittest_tests(self) -> Dict[str, Any]:
        """Run unittest suite."""
        command = [
            'python3', '-m', 'unittest',
            'tests.test_unittest_comprehensive',
            '-v' if self.verbose else ''
        ]
        # Remove empty string if not verbose
        command = [arg for arg in command if arg]
        
        return self.run_command(command, "unittest comprehensive tests")
    
    def run_specific_test_file(self, file_path: str, description: str = None) -> Dict[str, Any]:
        """Run a specific test file."""
        if not description:
            description = f"Test file: {os.path.basename(file_path)}"
        
        command = ['python3', '-m', 'pytest', file_path, '-v' if self.verbose else '-q']
        if self.no_capture:
            command.append('-s')
        
        return self.run_command(command, description)
    
    def print_summary(self):
        """Print comprehensive test summary."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE TEST EXECUTION SUMMARY")
        print("="*80)
        
        # Overall stats
        total_executed = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_executed * 100) if total_executed > 0 else 0
        
        print(f"📊 Total Test Suites: {total_executed}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Duration: {duration:.2f} seconds")
        
        # Result details
        if self.results:
            print("\n📋 Detailed Results:")
            print("-" * 80)
            
            for test_name, result in self.results.items():
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                duration_str = f"{result['duration']:.2f}s"
                print(f"{status:<8} {test_name:<50} {duration_str:>8}")
                
                if not result['success'] and result['stderr']:
                    print(f"         Error: {result['stderr'][:100]}...")
        
        # Production readiness assessment
        print("\n🏭 PRODUCTION READINESS ASSESSMENT:")
        print("-" * 50)
        
        if success_rate >= 95:
            print("🟢 PRODUCTION READY - All critical systems validated")
        elif success_rate >= 85:
            print("🟡 CONDITIONALLY READY - Minor issues detected")
        else:
            print("🔴 NOT PRODUCTION READY - Critical failures detected")
        
        # Recommendations
        print("\n📝 RECOMMENDATIONS:")
        if self.failed_tests > 0:
            print("- Review failed test outputs above")
            print("- Fix failing components before deployment")
            print("- Re-run tests after fixes")
        else:
            print("- All tests passing - system ready for deployment")
            print("- Consider running stress tests under load")
            print("- Monitor system performance in production")
        
        print("="*80)
        
        return success_rate >= 95


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Crypto Trading Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_comprehensive_tests.py --all          # Run all tests
    python run_comprehensive_tests.py --unit         # Run unit tests only
    python run_comprehensive_tests.py --live         # Run live testnet tests
    python run_comprehensive_tests.py --fast -v      # Fast tests with verbose output
        """
    )
    
    # Test selection arguments
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--stress', action='store_true', help='Run stress tests only')
    parser.add_argument('--live', action='store_true', help='Run live testnet tests only')
    parser.add_argument('--unittest', action='store_true', help='Run unittest suite only')
    parser.add_argument('--mathematical', action='store_true', help='Run mathematical validation only')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    
    # Execution options
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')
    parser.add_argument('--no-capture', action='store_true', help="Don't capture output")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test type selected
    if not any([args.unit, args.integration, args.stress, args.live, args.unittest, args.mathematical]):
        args.all = True
    
    # Initialize test runner
    runner = TestRunner(verbose=args.verbose, no_capture=args.no_capture)
    runner.start_time = time.time()
    
    runner.log("🚀 Starting Comprehensive Test Execution")
    runner.log(f"Arguments: {vars(args)}")
    
    # Run selected tests
    try:
        if args.unit or args.all:
            runner.results['Unit Tests'] = runner.run_pytest_tests(['unit'])
        
        if args.integration or args.all:
            runner.results['Integration Tests'] = runner.run_pytest_tests(['integration'])
        
        if args.stress or args.all:
            runner.results['Stress Tests'] = runner.run_pytest_tests(['stress'])
        
        if args.mathematical or args.all:
            runner.results['Mathematical Tests'] = runner.run_pytest_tests(['mathematical'])
        
        if args.unittest or args.all:
            runner.results['Unittest Suite'] = runner.run_unittest_tests()
        
        if args.live or args.all:
            if not args.fast:
                runner.results['Live Testnet'] = runner.run_specific_test_file(
                    'tests/test_live_testnet_validation.py',
                    'Live Testnet Validation'
                )
            else:
                runner.log("⏩ Skipping live testnet tests (fast mode)", "WARNING")
        
        # Additional comprehensive tests
        if args.all:
            runner.results['Portfolio Manager'] = runner.run_specific_test_file(
                'tests/test_portfolio_manager_pytest.py',
                'Portfolio Manager Tests'
            )
            
            runner.results['Risk Manager'] = runner.run_specific_test_file(
                'tests/test_risk_manager_pytest.py', 
                'Risk Manager Tests'
            )
            
            runner.results['Stress Handler'] = runner.run_specific_test_file(
                'tests/test_stress_handler_pytest.py',
                'Stress Handler Tests'
            )
    
    except KeyboardInterrupt:
        runner.log("⚠️ Test execution interrupted by user", "WARNING")
        return 1
    except Exception as e:
        runner.log(f"💥 Unexpected error during test execution: {str(e)}", "ERROR")
        return 1
    
    # Print summary and determine exit code
    production_ready = runner.print_summary()
    
    if production_ready:
        runner.log("🎉 All tests completed successfully - System is production ready!", "SUCCESS")
        return 0
    else:
        runner.log("⚠️ Some tests failed - System needs attention before production", "WARNING")
        return 1


if __name__ == "__main__":
    sys.exit(main())
