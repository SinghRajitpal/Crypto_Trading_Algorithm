"""
Tests package for Crypto Trading Algorithm
Senior Quantitative Developer Testing Protocol

This package contains comprehensive tests for:
- Algorithm Engine (Signal Generation & Processing)
- Execution Engine (Portfolio & Risk Management)
- Integration Testing (End-to-End Workflows)
- Performance & Stress Testing

Test Structure:
├── conftest.py              # Shared pytest fixtures and configuration
├── test_algorithm_engine.py # Algorithm engine tests (pytest)
├── test_execution_engine.py # Execution engine tests (pytest)
├── test_integration.py      # Integration tests (pytest)
├── test_unittest_suite.py   # Unittest compatibility suite
└── utils/                   # Test utilities and helpers
    ├── __init__.py
    ├── mock_objects.py      # Mock implementations
    └── test_data.py         # Test data generators

Usage:
    # Run all tests
    pytest tests/
    
    # Run specific test module
    pytest tests/test_algorithm_engine.py
    
    # Run unittest suite
    python tests/test_unittest_suite.py
"""

__version__ = "1.0.0"
__author__ = "Senior Quantitative Developer"

# Test configuration
TEST_CONFIG = {
    "test_capital": 10000.0,
    "test_symbols": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    "test_timeframes": ["1m", "5m"],
    "mock_data_periods": 100,
    "test_timeout": 30.0
}

__all__ = ["TEST_CONFIG"]

# Import commonly used test utilities
try:
    from .utils.mock_objects import (
        MockDataEngine,
        MockBinanceClient,
        MockStrategy
    )
    __all__.extend(["MockDataEngine", "MockBinanceClient", "MockStrategy"])
except ImportError:
    # Handle import errors gracefully
    pass
