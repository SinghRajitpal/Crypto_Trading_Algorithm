"""
Pytest configuration and shared fixtures for Crypto Trading Algorithm tests.

This file contains all the shared pytest fixtures and configuration
used across the test suite.
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.algo_engine import AlgoEngine
from execution.execution_engine import ProductionExecutionEngine
from tests.utils.mock_objects import (
    MockDataEngine,
    MockBinanceClient,
    MockStrategy,
    MockDataEngineWithTrend
)
from tests.utils.test_data import generate_market_data_bar


# Pytest configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_data_engine():
    """Fixture providing mock data engine."""
    return MockDataEngine()


@pytest.fixture
def mock_data_engine_with_trend():
    """Fixture providing mock data engine with trending data."""
    return MockDataEngineWithTrend()


@pytest.fixture
def mock_binance_client():
    """Fixture providing mock Binance client."""
    return MockBinanceClient()


@pytest.fixture
def algo_engine(mock_data_engine):
    """Fixture providing algorithm engine with mock data."""
    return AlgoEngine(mock_data_engine)


@pytest.fixture
def algo_engine_with_trend(mock_data_engine_with_trend):
    """Fixture providing algorithm engine with trending mock data."""
    return AlgoEngine(mock_data_engine_with_trend)


@pytest.fixture
async def execution_engine(mock_binance_client):
    """Fixture providing execution engine with mock client."""
    engine = ProductionExecutionEngine(mock_binance_client, total_capital=10000.0)
    await engine.setup()
    return engine


@pytest.fixture
def test_symbols():
    """Fixture providing standard test symbols."""
    return ["BTCUSDT", "ETHUSDT", "XRPUSDT"]


@pytest.fixture
def test_timeframes():
    """Fixture providing standard test timeframes."""
    return ["1m", "5m"]


@pytest.fixture
def test_capital():
    """Fixture providing standard test capital amount."""
    return 10000.0


@pytest.fixture
def buy_strategy():
    """Fixture providing a buy signal strategy."""
    return MockStrategy(["buy"], "buy_strategy")


@pytest.fixture
def sell_strategy():
    """Fixture providing a sell signal strategy."""
    return MockStrategy(["sell"], "sell_strategy")


@pytest.fixture
def hold_strategy():
    """Fixture providing a hold signal strategy."""
    return MockStrategy(["hold"], "hold_strategy")


@pytest.fixture
def mixed_strategy():
    """Fixture providing a mixed signal strategy."""
    return MockStrategy(["buy", "hold", "sell", "hold"], "mixed_strategy")


@pytest.fixture
def portfolio_manager(execution_engine):
    """Fixture providing portfolio manager from execution engine."""
    return execution_engine.portfolio_manager


@pytest.fixture
def risk_manager(execution_engine):
    """Fixture providing risk manager from execution engine."""
    return execution_engine.risk_manager


@pytest.fixture
def market_data_setup(execution_engine, test_symbols):
    """Fixture that sets up market data for testing."""
    for symbol in test_symbols:
        execution_engine.update_market_data_bar(
            symbol, 
            generate_market_data_bar(symbol),
            atr_value=0.02
        )
    return execution_engine


@pytest.fixture
def portfolio_with_allocation(market_data_setup):
    """Fixture providing execution engine with portfolio allocation."""
    from datetime import datetime, timedelta
    
    # Force portfolio rebalance
    market_data_setup.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
    result = market_data_setup.process_daily_rebalance()
    assert result is True
    
    return market_data_setup


# Test configuration constants
TEST_CONFIG = {
    "test_capital": 10000.0,
    "test_symbols": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    "test_timeframes": ["1m", "5m"],
    "mock_data_periods": 100,
    "test_timeout": 30.0,
    "atr_test_value": 0.02,
    "min_position_size": 0.001,
    "max_leverage": 3.0
}


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TEST_CONFIG
