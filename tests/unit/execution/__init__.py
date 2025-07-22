"""
Execution Component Unit Tests

Tests for execution-related modules:
- ExecutionEngine: Main execution coordination
- Portfolio: Portfolio management and allocation
- RiskManager: Risk management and validation
- OrderManager: Order creation and management
- Executor: Trade execution logic
"""

import sys
import os
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class MockBinanceClient:
    """Shared mock Binance client for execution tests."""
    
    def __init__(self, testnet=True):
        self.testnet = testnet
        self.positions = {}
        self.orders = {}
        self.order_id_counter = 1000
        
    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        """Mock order creation."""
        order_id = str(self.order_id_counter)
        self.order_id_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'params': params
        }
        
        self.orders[order_id] = order
        return order
        
    async def get_open_positions(self, symbol=None):
        """Mock position retrieval."""
        if symbol:
            return [self.positions.get(symbol, {})] if symbol in self.positions else []
        return list(self.positions.values())
        
    async def close_position(self, symbol, side=None):
        """Mock position closure."""
        if symbol in self.positions:
            del self.positions[symbol]
            return {'status': 'closed', 'symbol': symbol}
        return {'status': 'not_found', 'symbol': symbol}
        
    async def cancel_order(self, order_id, symbol):
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
            return {'status': 'canceled', 'id': order_id}
        return {'status': 'not_found', 'id': order_id}
        
    async def set_leverage(self, symbol, leverage):
        """Mock leverage setting."""
        return {'symbol': symbol, 'leverage': leverage}
        
    async def set_margin_type(self, symbol, margin_type):
        """Mock margin type setting."""
        return {'symbol': symbol, 'margin_type': margin_type}
        
    async def close(self):
        """Mock client closure."""
        pass


class MockTradeSignal:
    """Shared mock trade signal for execution tests."""
    
    def __init__(self, action="open", side="long", symbol="BTCUSDT"):
        self.action = action
        self.side = side
        self.symbol = symbol
        self.strategy_id = "test_strategy"
        self.timestamp = 1640995200000  # Fixed timestamp for testing
        self.signal_confidence = 0.8
        self.metadata = {
            'atr': 0.002,
            'price': 42000.0,
            'reason': 'test signal',
            'fast_ma': 42000.0,
            'slow_ma': 41950.0
        }


def create_test_bar_data(symbol="BTCUSDT", base_price=42000.0, volatility=0.02):
    """Create test bar data for market data tests."""
    return {
        'symbol': symbol,
        'timestamp': 1640995200000,
        'open': base_price,
        'high': base_price * (1 + volatility),
        'low': base_price * (1 - volatility),
        'close': base_price * (1 + volatility * 0.5),
        'volume': 100.0
    }


def create_test_position_data(symbol="BTCUSDT", side="long", size=0.025):
    """Create test position data."""
    return {
        'symbol': symbol,
        'side': side,
        'size': size,
        'entry_price': 42000.0,
        'unrealized_pnl': 150.0,
        'leverage': 5,
        'margin': 200.0
    }


def create_test_allocation_weights(symbol="BTCUSDT", weight=0.4, capital=4000.0):
    """Create test allocation weights."""
    from execution.portfolio import AllocationWeights
    
    return AllocationWeights(
        symbol=symbol,
        weight=weight,
        allocated_capital=capital,
        volatility=0.002,
        avg_correlation=0.6,
        raw_weight=0.35
    )


class ExecutionTestConstants:
    """Constants for execution tests."""
    
    # Test symbols
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT"]
    
    # Test prices
    BTC_PRICE = 42000.0
    ETH_PRICE = 3200.0
    XRP_PRICE = 0.85
    
    # Test parameters
    DEFAULT_CAPITAL = 10000.0
    DEFAULT_LEVERAGE = 5
    DEFAULT_ATR = 0.002
    DEFAULT_ALLOCATION = 0.2
    
    # Risk parameters
    RISK_PER_TRADE = 0.008  # 0.8%
    KELLY_FRACTION = 0.7
    MAX_LEVERAGE = 10
    STOP_LOSS_MULTIPLIER = 1.8
    TAKE_PROFIT_RATIO = 2.0


# Test data generators
def generate_volatility_data(periods=60, base_vol=0.02, spike_factor=1.0):
    """Generate test volatility data."""
    import random
    
    data = []
    for i in range(periods):
        vol = base_vol * (1 + random.uniform(-0.2, 0.2)) * spike_factor
        data.append(max(vol, 0.001))  # Minimum volatility floor
        
    return data


def generate_correlation_data(periods=60, base_corr=0.6, volatility=0.1):
    """Generate test correlation data."""
    import random
    
    data = []
    for i in range(periods):
        corr = base_corr + random.uniform(-volatility, volatility)
        corr = max(-1.0, min(1.0, corr))  # Clamp to valid range
        data.append(corr)
        
    return data


def generate_price_series(periods=100, start_price=42000.0, volatility=0.02):
    """Generate test price series with realistic movements."""
    import random
    import math
    
    prices = [start_price]
    
    for i in range(1, periods):
        # Random walk with mean reversion
        change = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, start_price * 0.5))  # Prevent extreme drops
        
    return prices


# Test utilities
def assert_valid_execution_result(test_case, result):
    """Assert that an execution result has valid structure."""
    test_case.assertIn('status', result)
    test_case.assertIn('symbol', result)
    
    if result['status'] == 'success':
        test_case.assertIn('timestamp', result)
    elif result['status'] == 'error':
        test_case.assertIn('reason', result)


def assert_valid_portfolio_summary(test_case, summary):
    """Assert that a portfolio summary has valid structure."""
    required_fields = [
        'total_capital', 'allocated_capital', 'available_capital',
        'allocation_percentage', 'active_positions'
    ]
    
    for field in required_fields:
        test_case.assertIn(field, summary)
        
    # Validate numeric constraints
    test_case.assertGreaterEqual(summary['total_capital'], 0)
    test_case.assertGreaterEqual(summary['allocated_capital'], 0)
    test_case.assertLessEqual(summary['allocated_capital'], summary['total_capital'])
    test_case.assertGreaterEqual(summary['active_positions'], 0)


def assert_valid_risk_metrics(test_case, metrics):
    """Assert that risk metrics have valid structure."""
    required_fields = [
        'daily_pnl', 'current_drawdown', 'current_sharpe',
        'risk_status', 'max_drawdown_hit'
    ]
    
    for field in required_fields:
        test_case.assertIn(field, metrics)
        
    # Validate constraints
    test_case.assertLessEqual(metrics['current_drawdown'], 0)  # Drawdown should be negative
    test_case.assertIn(metrics['risk_status'], ['normal', 'warning', 'critical'])


def create_stress_test_scenario(scenario_type="high_volatility"):
    """Create test scenarios for stress testing."""
    scenarios = {
        "high_volatility": {
            "volatility_multiplier": 5.0,
            "correlation_spike": 0.95,
            "drawdown": -0.20
        },
        "market_crash": {
            "volatility_multiplier": 10.0,
            "correlation_spike": 0.98,
            "drawdown": -0.30
        },
        "low_liquidity": {
            "bid_ask_spread": 50.0,
            "order_book_depth": 10.0,
            "trade_frequency": 1.0
        }
    }
    
    return scenarios.get(scenario_type, scenarios["high_volatility"])
