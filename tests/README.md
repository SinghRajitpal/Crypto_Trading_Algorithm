# Crypto Trading Algorithm - Test Suite

This test package provides comprehensive testing for the crypto trading algorithm using both pytest and unittest frameworks.

## Test Structure

```
tests/
├── __init__.py                 # Package initialization and configuration
├── conftest.py                 # Pytest configuration and shared fixtures
├── run_tests.py               # Main test runner script
├── utils/
│   ├── __init__.py            # Test utilities package
│   ├── mock_objects.py        # Mock implementations for testing
│   └── test_data.py           # Test data generators
├── test_algorithm_engine.py   # Algorithm engine tests (pytest)
├── test_execution_engine.py   # Execution engine tests (pytest)
├── test_integration.py        # Integration tests (pytest)
└── test_unittest_suite.py     # Unittest compatibility suite
```

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py --verbose

# Run only pytest suite
python tests/run_tests.py --pytest-only

# Run only unittest suite
python tests/run_tests.py --unittest-only

# Run with coverage reporting
python tests/run_tests.py --coverage
```

### Individual Test Suites

#### Pytest Suite
```bash
# Run pytest tests directly
pytest tests/test_algorithm_engine.py tests/test_execution_engine.py tests/test_integration.py -v

# Run with coverage
pytest tests/test_algorithm_engine.py tests/test_execution_engine.py tests/test_integration.py --cov=algorithm --cov=execution --cov-report=term-missing
```

#### Unittest Suite
```bash
# Run unittest suite directly
python tests/test_unittest_suite.py
```

## Test Categories

### Algorithm Engine Tests (`test_algorithm_engine.py`)
- **Initialization**: Engine setup, configuration validation
- **Signal Processing**: Signal generation, validation, throttling
- **Strategy Management**: Strategy loading, execution, error handling
- **Performance**: Latency measurement, throughput testing
- **Error Handling**: Exception management, resilience testing

### Execution Engine Tests (`test_execution_engine.py`)
- **Portfolio Management**: Position tracking, balance calculations
- **Risk Management**: Risk validation, position sizing, limits
- **Order Management**: Order creation, validation, execution
- **Market Data**: Real-time data processing, feed management
- **Stress Testing**: High-frequency scenarios, concurrent processing
- **Performance**: Execution latency, throughput benchmarks
- **Error Handling**: Network failures, API errors, recovery

### Integration Tests (`test_integration.py`)
- **Signal-to-Execution**: End-to-end signal processing
- **Portfolio Integration**: Complete workflow testing
- **Error Recovery**: System resilience under failure conditions
- **Performance Integration**: Full system performance testing
- **Stress Testing**: System behavior under extreme conditions

### Unittest Compatibility (`test_unittest_suite.py`)
- **Traditional Structure**: xUnit-style test organization
- **Enterprise Compatibility**: Standard unittest framework usage
- **Comprehensive Coverage**: All major components tested
- **Standalone Execution**: Independent test runner

## Test Utilities

### Mock Objects (`utils/mock_objects.py`)
- **MockDataEngine**: Simulates market data feeds
- **MockBinanceClient**: Mocks exchange API interactions
- **MockStrategy**: Provides test trading strategies
- **MockRiskManager**: Simulates risk management decisions

### Test Data (`utils/test_data.py`)
- **Market Data Generators**: Creates realistic test data
- **Signal Generators**: Generates test trading signals
- **Portfolio Scenarios**: Creates test portfolio states

## Configuration

### Pytest Configuration (`conftest.py`)
- **Shared Fixtures**: Common test setup and teardown
- **Event Loop**: Async testing support
- **Mock Configurations**: Standardized mock objects
- **Test Environment**: Isolated testing environment

### Test Configuration (`__init__.py`)
- **Test Settings**: Centralized test configuration
- **Environment Variables**: Test-specific settings
- **Import Management**: Graceful import handling

## Test Coverage

The test suite covers:

### Functional Testing
- ✅ Algorithm Engine: Signal generation and processing
- ✅ Execution Engine: Trade execution and portfolio management
- ✅ Risk Management: Position sizing and risk controls
- ✅ Market Data: Real-time data processing
- ✅ Error Handling: Exception management and recovery

### Performance Testing
- ✅ Latency: Signal processing and execution timing
- ✅ Throughput: High-frequency processing capabilities
- ✅ Concurrent Processing: Multi-threading performance
- ✅ Memory Usage: Resource utilization monitoring

### Integration Testing
- ✅ End-to-End Workflows: Complete trading pipelines
- ✅ Component Integration: Inter-module communication
- ✅ External Dependencies: Exchange API interactions
- ✅ Error Propagation: System-wide error handling

### Stress Testing
- ✅ High-Frequency Scenarios: Rapid signal processing
- ✅ Resource Limits: Memory and CPU constraints
- ✅ Network Failures: Connectivity issues
- ✅ API Rate Limits: Exchange API limitations

## Test Results

Expected test coverage:
- **Total Tests**: ~51 test cases
- **Success Rate**: 100% (all tests passing)
- **Coverage**: Core algorithm and execution components
- **Performance**: Sub-millisecond signal processing
- **Reliability**: Comprehensive error handling

## Development Guidelines

### Adding New Tests
1. Follow the existing test structure
2. Use appropriate fixtures from `conftest.py`
3. Mock external dependencies using `utils/mock_objects.py`
4. Add both pytest and unittest versions if needed
5. Update this README with new test descriptions

### Test Best Practices
- **Isolation**: Each test should be independent
- **Mocking**: Mock all external dependencies
- **Assertions**: Use specific, meaningful assertions
- **Performance**: Include timing assertions for critical paths
- **Documentation**: Document test purpose and expected outcomes

### CI/CD Integration
The test runner (`run_tests.py`) is designed for CI/CD integration:
- Returns appropriate exit codes (0 for success, 1 for failure)
- Provides structured output for parsing
- Supports coverage reporting
- Handles timeouts and resource constraints

## Dependencies

### Required Packages
```python
# Core testing
pytest>=7.0.0
pytest-asyncio>=0.20.0

# Coverage (optional)
pytest-cov>=4.0.0

# Standard library
unittest  # Built-in
asyncio   # Built-in
```

### Development Dependencies
```python
# For enhanced testing
pytest-xdist      # Parallel test execution
pytest-benchmark  # Performance benchmarking
pytest-mock      # Enhanced mocking capabilities
```

## Troubleshooting

### Common Issues

#### Import Errors
- Ensure project root is in Python path
- Check that all `__init__.py` files are present
- Verify package structure matches imports

#### Async Test Issues
- Use `pytest-asyncio` plugin
- Mark async tests with `@pytest.mark.asyncio`
- Use proper event loop fixtures

#### Mock Configuration
- Check mock object setup in `conftest.py`
- Verify mock methods return expected values
- Ensure mocks are properly reset between tests

#### Performance Test Failures
- Adjust timing thresholds for slower systems
- Consider system load when running performance tests
- Use relative performance comparisons

### Getting Help
1. Check test output for specific error messages
2. Run individual test files to isolate issues
3. Use verbose mode (`--verbose`) for detailed output
4. Review mock configurations and test data

## Contributing

When contributing to the test suite:
1. Maintain existing test structure and patterns
2. Add comprehensive test coverage for new features
3. Update documentation for new test categories
4. Ensure all tests pass before submitting changes
5. Follow the coding standards used in existing tests
