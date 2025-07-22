#!/usr/bin/env python3
"""
Pytest-compatible Execution Engine Test Suite
Senior Quantitative Developer Testing Protocol

Tests for portfolio management, risk calculations, and execution engine integration.
"""

import pytest
import asyncio
import os
import sys
import time
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from execution.execution_engine import ProductionExecutionEngine
from execution.executor import OrderExecutor


class MockBinanceClient:
    """Mock Binance client for testing."""
    
    def __init__(self):
        self.account_config_called = False
        self.orders = []
        self.positions = []
        self.balance = {'USDT': 10000.0}
        
    async def setup_account_config(self):
        self.account_config_called = True
        
    async def get_balance(self):
        return {'total': self.balance, 'free': self.balance, 'used': {}}
        
    async def get_open_positions(self, symbol=None):
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions
        
    async def open_position(self, symbol, side, amount, **kwargs):
        position = {
            'id': f'pos_{len(self.positions)}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'status': 'open'
        }
        self.positions.append(position)
        return {'status': 'success', 'position': position}


@pytest.fixture
def mock_binance_client():
    """Fixture providing mock Binance client."""
    return MockBinanceClient()


@pytest.fixture
def portfolio_manager():
    """Fixture providing portfolio manager."""
    return ProductionPortfolioManager(total_capital=10000.0)


@pytest.fixture
def risk_manager(portfolio_manager):
    """Fixture providing risk manager."""
    return ProductionRiskManager(portfolio_manager)


@pytest.fixture
def execution_engine(mock_binance_client):
    """Fixture providing execution engine."""
    return ProductionExecutionEngine(mock_binance_client, total_capital=10000.0)


class TestPortfolioManagerInitialization:
    """Test portfolio manager initialization and basic functionality."""
    
    def test_portfolio_manager_initialization(self, portfolio_manager):
        """Test portfolio manager initializes correctly."""
        assert portfolio_manager.total_capital == 10000.0
        assert portfolio_manager.target_volatility == 0.18
        assert portfolio_manager.max_allocation_pct == 0.85
        assert portfolio_manager.volatility_data == {}
        assert portfolio_manager.correlation_data == {}
        assert portfolio_manager.allocation_weights == {}
    
    def test_reserved_allocations_initialization(self, portfolio_manager):
        """Test reserved allocations tracking."""
        assert hasattr(portfolio_manager, 'reserved_allocations')
        assert portfolio_manager.reserved_allocations == {}


class TestPortfolioManagerVolatilityCorrelation:
    """Test volatility and correlation data management."""
    
    def test_volatility_data_update(self, portfolio_manager):
        """Test updating volatility data."""
        symbol = "BTCUSDT"
        atr_value = 0.02
        
        # Update volatility data
        portfolio_manager.update_volatility_data(symbol, atr_value)
        
        # Check data was stored
        assert symbol in portfolio_manager.volatility_data
        assert len(portfolio_manager.volatility_data[symbol]) == 1
        assert portfolio_manager.volatility_data[symbol][0] == atr_value
    
    def test_correlation_data_update(self, portfolio_manager):
        """Test updating correlation data."""
        symbol1 = "BTCUSDT"
        symbol2 = "ETHUSDT"
        correlation = 0.75
        
        # Update correlation data
        portfolio_manager.update_correlation_data(symbol1, symbol2, correlation)
        
        # Check data was stored
        key = (symbol1, symbol2)
        assert key in portfolio_manager.correlation_data
        assert len(portfolio_manager.correlation_data[key]) == 1
        assert portfolio_manager.correlation_data[key][0] == correlation
    
    def test_volatility_ema_calculation(self, portfolio_manager):
        """Test volatility EMA calculation."""
        symbol = "BTCUSDT"
        
        # Add multiple data points
        for atr in [0.02, 0.021, 0.019, 0.022]:
            portfolio_manager.update_volatility_data(symbol, atr)
        
        # Get EMA
        ema = portfolio_manager.get_volatility_ema(symbol)
        assert isinstance(ema, float)
        assert ema > 0
    
    def test_average_correlation_calculation(self, portfolio_manager):
        """Test average correlation calculation."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Add correlation data
        portfolio_manager.update_correlation_data("BTCUSDT", "ETHUSDT", 0.75)
        portfolio_manager.update_correlation_data("BTCUSDT", "XRPUSDT", 0.65)
        
        # Calculate average correlation
        avg_corr = portfolio_manager.get_average_correlation("BTCUSDT", symbols)
        assert isinstance(avg_corr, float)
        assert 0 <= avg_corr <= 1


class TestPortfolioManagerAllocationWeights:
    """Test allocation weight calculation."""
    
    def test_weight_computation(self, portfolio_manager):
        """Test weight computation with inverse volatility."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Add volatility data
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Compute weights
        weights = portfolio_manager.compute_weights(symbols)
        
        assert len(weights) == len(symbols)
        assert all(symbol in weights for symbol in symbols)
        assert all(weight > 0 for weight in weights.values())
        
        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_scaling_multiplier_calculation(self, portfolio_manager):
        """Test scaling multiplier calculation."""
        # Test normal regime
        multiplier = portfolio_manager.calculate_scaling_multiplier()
        assert isinstance(multiplier, float)
        assert multiplier > 0
        
        # Test with high volatility regime
        # Add historical volatility data to trigger high vol regime
        for _ in range(50):
            portfolio_manager.volatility_history.append(0.05)  # High volatility
        
        multiplier_high_vol = portfolio_manager.calculate_scaling_multiplier()
        assert multiplier_high_vol <= multiplier  # Should be reduced in high vol
    
    def test_portfolio_rebalancing(self, portfolio_manager):
        """Test complete portfolio rebalancing."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Add required data
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Force rebalance by setting old timestamp
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Perform rebalancing
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        assert len(allocations) == len(symbols)
        assert all(symbol in allocations for symbol in symbols)
        
        # Check total allocated capital
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        expected_max = portfolio_manager.total_capital * portfolio_manager.max_allocation_pct
        assert abs(total_allocated - expected_max) < 1.0  # Within $1


class TestPortfolioManagerReservationSystem:
    """Test allocation reservation system."""
    
    def test_reserve_allocation_success(self, portfolio_manager):
        """Test successful allocation reservation."""
        symbol = "BTCUSDT"
        
        # Set up allocation
        portfolio_manager.allocation_weights[symbol] = type('obj', (object,), {
            'allocated_capital': 1000.0
        })()
        
        # Reserve allocation
        result = portfolio_manager.reserve_allocation(symbol, 500.0)
        
        assert result is True
        assert portfolio_manager.reserved_allocations[symbol] == 500.0
    
    def test_reserve_allocation_exceed_limit(self, portfolio_manager):
        """Test allocation reservation exceeding limits."""
        symbol = "BTCUSDT"
        
        # Set up allocation
        portfolio_manager.allocation_weights[symbol] = type('obj', (object,), {
            'allocated_capital': 1000.0
        })()
        
        # Try to reserve more than allocated
        result = portfolio_manager.reserve_allocation(symbol, 1500.0)
        
        assert result is False
        assert symbol not in portfolio_manager.reserved_allocations
    
    def test_release_allocation(self, portfolio_manager):
        """Test allocation release."""
        symbol = "BTCUSDT"
        
        # Set up reservation
        portfolio_manager.reserved_allocations[symbol] = 500.0
        
        # Release part of allocation
        result = portfolio_manager.release_allocation(symbol, 200.0)
        
        assert result is True
        assert portfolio_manager.reserved_allocations[symbol] == 300.0
        
        # Release remaining
        portfolio_manager.release_allocation(symbol, 300.0)
        assert portfolio_manager.reserved_allocations[symbol] == 0.0


class TestRiskManagerInitialization:
    """Test risk manager initialization and basic functionality."""
    
    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initializes correctly."""
        assert risk_manager.portfolio_manager is not None
        assert risk_manager.risk_params is not None
        assert risk_manager.risk_params.risk_per_trade_pct == 0.008
        assert risk_manager.risk_params.kelly_fraction == 0.7
        assert risk_manager.drawdown_history == []
        assert risk_manager.positions == {}


class TestRiskManagerPositionSizing:
    """Test position sizing calculations."""
    
    def test_position_sizing_calculation(self, risk_manager):
        """Test position sizing with production formula."""
        symbol = "BTCUSDT"
        allocated_capital = 5000.0
        atr_value = 0.02
        entry_price = 50000.0
        
        # Calculate position size
        position_info = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price
        )
        
        assert 'size_contracts' in position_info
        assert 'size_usdt' in position_info
        assert 'leverage' in position_info
        assert 'margin_usdt' in position_info
        
        # Verify calculations are reasonable
        assert position_info['size_contracts'] > 0
        assert position_info['size_usdt'] > 0
        assert position_info['leverage'] >= 1
        assert position_info['margin_usdt'] > 0
    
    def test_dynamic_leverage_calculation(self, risk_manager):
        """Test dynamic leverage calculation."""
        symbol = "BTCUSDT"
        base_leverage = 3
        
        # Test normal conditions
        leverage = risk_manager.calculate_dynamic_leverage(symbol, base_leverage)
        assert isinstance(leverage, int)
        assert 1 <= leverage <= risk_manager.risk_params.max_leverage
        
        # Test with different drawdown conditions
        risk_manager.drawdown_history = [(datetime.now(), -0.05)]  # 5% drawdown
        leverage_dd = risk_manager.calculate_dynamic_leverage(symbol, base_leverage)
        assert leverage_dd <= leverage  # Should be reduced with drawdown


class TestRiskManagerValidation:
    """Test trade validation functionality."""
    
    def test_trade_validation_no_capital(self, risk_manager):
        """Test trade validation when no capital allocated."""
        result = risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="open",
            side="buy",
            entry_price=50000.0,
            atr_value=0.02
        )
        
        assert result['valid'] is False
        assert 'no capital allocated' in result['reason'].lower()
    
    def test_trade_validation_with_capital(self, risk_manager):
        """Test trade validation with allocated capital."""
        # Set up allocated capital
        symbol = "BTCUSDT"
        risk_manager.portfolio_manager.allocation_weights[symbol] = type('obj', (object,), {
            'allocated_capital': 5000.0
        })()
        
        result = risk_manager.validate_trade(
            symbol=symbol,
            action="open",
            side="buy",
            entry_price=50000.0,
            atr_value=0.02
        )
        
        assert result['valid'] is True
        assert 'position_info' in result
        assert result['position_info']['size_contracts'] > 0
    
    def test_trade_validation_non_open_action(self, risk_manager):
        """Test validation for non-open actions."""
        result = risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="close",
            side="sell",
            entry_price=50000.0,
            atr_value=0.02
        )
        
        assert result['valid'] is True
        assert result['reason'] == "No validation needed for close"


class TestExecutionEngineIntegration:
    """Test execution engine integration."""
    
    @pytest.mark.asyncio
    async def test_execution_engine_setup(self, execution_engine):
        """Test execution engine setup."""
        await execution_engine.setup()
        assert execution_engine.binance_client.account_config_called is True
    
    def test_market_data_update(self, execution_engine):
        """Test market data bar update."""
        symbol = "BTCUSDT"
        ohlcv_data = {
            'open': 50000.0,
            'high': 50200.0,
            'low': 49800.0,
            'close': 50100.0,
            'volume': 100.0
        }
        atr_value = 0.02
        
        # Update market data
        execution_engine.update_market_data_bar(symbol, ohlcv_data, atr_value)
        
        # Check volatility data was updated
        assert symbol in execution_engine.portfolio_manager.volatility_data
        assert len(execution_engine.portfolio_manager.volatility_data[symbol]) > 0
    
    def test_portfolio_rebalancing_trigger(self, execution_engine):
        """Test portfolio rebalancing trigger."""
        # Add market data first
        symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            execution_engine.update_market_data_bar(symbol, {
                'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0
            }, 0.02)
        
        # Force rebalance by setting old timestamp
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Trigger rebalance
        result = execution_engine.process_daily_rebalance()
        
        assert result is True
        
        # Check portfolio allocation
        portfolio_summary = execution_engine.get_portfolio_summary()
        assert portfolio_summary['allocated_capital'] > 0
    
    def test_portfolio_summary(self, execution_engine):
        """Test portfolio summary generation."""
        summary = execution_engine.get_portfolio_summary()
        
        assert 'total_capital' in summary
        assert 'allocated_capital' in summary
        assert 'allocation_percentage' in summary
        assert 'target_volatility' in summary
        assert 'active_symbols' in summary
        
        assert summary['total_capital'] == 10000.0
    
    def test_risk_metrics(self, execution_engine):
        """Test risk metrics generation."""
        metrics = execution_engine.get_risk_metrics()
        
        assert 'risk_status' in metrics
        assert 'daily_pnl' in metrics
        assert 'active_positions' in metrics  # Changed from total_positions
        assert 'max_drawdown_hit' in metrics
        
        assert metrics['risk_status'] in ['normal', 'caution', 'warning', 'critical']


@pytest.mark.asyncio
async def test_concurrent_execution_operations(mock_binance_client):
    """Test concurrent execution operations."""
    execution_engine = ProductionExecutionEngine(mock_binance_client, total_capital=10000.0)
    
    # Test concurrent market data updates
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    tasks = []
    
    for symbol in symbols:
        task = asyncio.create_task(asyncio.to_thread(
            execution_engine.update_market_data_bar,
            symbol,
            {'open': 50000.0, 'high': 50200.0, 'low': 49800.0, 'close': 50100.0, 'volume': 100.0},
            0.02
        ))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    # All symbols should have volatility data
    for symbol in symbols:
        assert symbol in execution_engine.portfolio_manager.volatility_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
