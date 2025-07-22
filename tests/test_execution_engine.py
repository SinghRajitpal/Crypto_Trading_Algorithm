"""
Execution Engine Tests
Senior Quantitative Developer Testing Protocol

Comprehensive pytest-based testing for the execution engine including:
- Portfolio management and allocation
- Risk management and position sizing
- Dynamic leverage calculations
- Market data processing
- Integration with algorithm signals
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from execution.execution_engine import ProductionExecutionEngine
from algorithm.trade_signal import TradeSignal
from tests.utils.test_data import generate_market_data_bar, create_test_signal_metadata


class TestPortfolioManagerInitialization:
    """Test portfolio manager initialization and basic functionality."""
    
    def test_portfolio_manager_initialization(self, portfolio_manager, test_capital):
        """Test portfolio manager initialization."""
        assert portfolio_manager is not None
        assert portfolio_manager.total_capital == test_capital
        assert portfolio_manager.target_volatility == 0.18  # 18%
        assert portfolio_manager.max_allocation_pct == 0.85  # 85%
        assert hasattr(portfolio_manager, 'symbol_allocations')
        assert hasattr(portfolio_manager, 'reserved_allocations')
    
    def test_reserved_allocations_initialization(self, portfolio_manager):
        """Test that reserved allocations are properly initialized."""
        assert portfolio_manager.reserved_allocations == {}
        assert hasattr(portfolio_manager, 'reserve_allocation')
        assert hasattr(portfolio_manager, 'release_allocation')


class TestPortfolioManagerVolatilityCorrelation:
    """Test volatility and correlation tracking."""
    
    def test_volatility_data_update(self, portfolio_manager, test_symbols):
        """Test volatility data updates."""
        for symbol in test_symbols:
            volatility = 0.02  # 2% volatility
            portfolio_manager.update_volatility(symbol, volatility)
            
            assert symbol in portfolio_manager.volatilities
            assert portfolio_manager.volatilities[symbol] == volatility
    
    def test_correlation_data_update(self, portfolio_manager, test_symbols):
        """Test correlation data updates."""
        # Update correlation between first two symbols
        symbol1, symbol2 = test_symbols[0], test_symbols[1]
        correlation = 0.5
        
        portfolio_manager.update_correlation(symbol1, symbol2, correlation)
        
        assert symbol1 in portfolio_manager.correlations
        assert symbol2 in portfolio_manager.correlations[symbol1]
        assert portfolio_manager.correlations[symbol1][symbol2] == correlation
    
    def test_volatility_ema_calculation(self, portfolio_manager):
        """Test volatility EMA calculation."""
        symbol = "BTCUSDT"
        initial_vol = 0.02
        new_vol = 0.03
        
        # Set initial volatility
        portfolio_manager.update_volatility(symbol, initial_vol)
        
        # Update with new volatility (should use EMA)
        portfolio_manager.update_volatility(symbol, new_vol)
        
        # Result should be between initial and new volatility (EMA effect)
        result_vol = portfolio_manager.volatilities[symbol]
        assert initial_vol < result_vol < new_vol
    
    def test_average_correlation_calculation(self, portfolio_manager, test_symbols):
        """Test average correlation calculation."""
        # Set up correlation matrix
        portfolio_manager.update_correlation(test_symbols[0], test_symbols[1], 0.6)
        portfolio_manager.update_correlation(test_symbols[0], test_symbols[2], 0.4)
        portfolio_manager.update_correlation(test_symbols[1], test_symbols[2], 0.5)
        
        avg_corr = portfolio_manager.get_average_correlation()
        
        # Should be average of correlations
        expected_avg = (0.6 + 0.4 + 0.5) / 3
        assert abs(avg_corr - expected_avg) < 0.01


class TestPortfolioManagerAllocationWeights:
    """Test allocation weight computation."""
    
    def test_weight_computation(self, market_data_setup, test_symbols):
        """Test portfolio weight computation."""
        portfolio_manager = market_data_setup.portfolio_manager
        
        weights = portfolio_manager.compute_weights(test_symbols)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(test_symbols)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be positive
        for weight in weights.values():
            assert weight > 0
    
    def test_scaling_multiplier_calculation(self, market_data_setup):
        """Test scaling multiplier calculation."""
        portfolio_manager = market_data_setup.portfolio_manager
        
        multiplier = portfolio_manager.get_scaling_multiplier()
        
        assert isinstance(multiplier, float)
        assert multiplier > 0
        assert multiplier <= 2.0  # Should not exceed reasonable bounds
    
    def test_portfolio_rebalancing(self, market_data_setup, test_symbols):
        """Test portfolio rebalancing functionality."""
        portfolio_manager = market_data_setup.portfolio_manager
        
        # Force rebalance by setting old timestamp
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        result = portfolio_manager.rebalance_portfolio()
        
        assert result is True
        assert portfolio_manager.total_allocated > 0
        assert len(portfolio_manager.symbol_allocations) > 0
        
        # Check that allocations are reasonable
        for symbol in test_symbols:
            if symbol in portfolio_manager.symbol_allocations:
                allocation = portfolio_manager.symbol_allocations[symbol]
                assert allocation > 0
                assert allocation <= portfolio_manager.total_capital


class TestPortfolioManagerReservationSystem:
    """Test allocation reservation system."""
    
    def test_reserve_allocation_success(self, portfolio_with_allocation):
        """Test successful allocation reservation."""
        portfolio_manager = portfolio_with_allocation.portfolio_manager
        symbol = "BTCUSDT"
        amount = 1000.0
        
        result = portfolio_manager.reserve_allocation(symbol, amount)
        
        assert result is True
        assert symbol in portfolio_manager.reserved_allocations
        assert portfolio_manager.reserved_allocations[symbol] == amount
    
    def test_reserve_allocation_exceed_limit(self, portfolio_with_allocation):
        """Test allocation reservation exceeding limits."""
        portfolio_manager = portfolio_with_allocation.portfolio_manager
        symbol = "BTCUSDT"
        
        # Try to reserve more than allocated
        if symbol in portfolio_manager.symbol_allocations:
            allocated_amount = portfolio_manager.symbol_allocations[symbol]
            excessive_amount = allocated_amount * 2
            
            result = portfolio_manager.reserve_allocation(symbol, excessive_amount)
            assert result is False
    
    def test_release_allocation(self, portfolio_with_allocation):
        """Test allocation release."""
        portfolio_manager = portfolio_with_allocation.portfolio_manager
        symbol = "BTCUSDT"
        amount = 1000.0
        
        # Reserve first
        portfolio_manager.reserve_allocation(symbol, amount)
        assert symbol in portfolio_manager.reserved_allocations
        
        # Release
        portfolio_manager.release_allocation(symbol, amount)
        
        # Should be removed or reduced
        if symbol in portfolio_manager.reserved_allocations:
            assert portfolio_manager.reserved_allocations[symbol] < amount
        else:
            assert symbol not in portfolio_manager.reserved_allocations


class TestRiskManagerInitialization:
    """Test risk manager initialization and configuration."""
    
    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager is not None
        assert risk_manager.risk_per_trade == 0.008  # 0.8%
        assert risk_manager.kelly_fraction == 0.7
        assert risk_manager.base_leverage == 3.0
        assert hasattr(risk_manager, 'calculate_position_size')
        assert hasattr(risk_manager, 'validate_trade')


class TestRiskManagerPositionSizing:
    """Test position sizing calculations."""
    
    def test_position_sizing_calculation(self, risk_manager):
        """Test position sizing calculation."""
        symbol = "BTCUSDT"
        price = 50000.0
        atr_value = 0.02
        allocated_capital = 5000.0
        
        result = risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            atr=atr_value,
            allocated_capital=allocated_capital
        )
        
        assert isinstance(result, dict)
        assert 'size_contracts' in result
        assert 'leverage' in result
        assert 'stop_loss_price' in result
        assert 'take_profit_price' in result
        
        # Verify reasonable values
        assert result['size_contracts'] > 0
        assert 1 <= result['leverage'] <= 3
        assert result['stop_loss_price'] < price
        assert result['take_profit_price'] > price
    
    def test_dynamic_leverage_calculation(self, risk_manager):
        """Test dynamic leverage calculation."""
        symbol = "BTCUSDT"
        
        leverage = risk_manager.calculate_dynamic_leverage(symbol)
        
        assert isinstance(leverage, (int, float))
        assert 1 <= leverage <= 3  # Within reasonable bounds


class TestRiskManagerValidation:
    """Test trade validation functionality."""
    
    def test_trade_validation_no_capital(self, risk_manager):
        """Test trade validation with no allocated capital."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.8
        )
        
        result = risk_manager.validate_trade(signal, allocated_capital=0)
        
        assert isinstance(result, dict)
        assert result.get('valid') is False
        assert 'reason' in result
    
    def test_trade_validation_with_capital(self, risk_manager):
        """Test trade validation with allocated capital."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.8
        )
        
        result = risk_manager.validate_trade(signal, allocated_capital=5000.0)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        
        if result['valid']:
            assert 'position_info' in result
    
    def test_trade_validation_non_open_action(self, risk_manager):
        """Test trade validation for non-open actions."""
        signal = TradeSignal(
            action="hold",
            side="none",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.5
        )
        
        result = risk_manager.validate_trade(signal, allocated_capital=5000.0)
        
        assert isinstance(result, dict)
        assert result.get('valid') is False


class TestExecutionEngineIntegration:
    """Test execution engine integration functionality."""
    
    @pytest.mark.asyncio
    async def test_execution_engine_setup(self, mock_binance_client, test_capital):
        """Test execution engine setup."""
        engine = ProductionExecutionEngine(mock_binance_client, total_capital=test_capital)
        
        await engine.setup()
        
        assert mock_binance_client.account_config_called is True
        assert engine.portfolio_manager is not None
        assert engine.risk_manager is not None
    
    def test_market_data_update(self, execution_engine):
        """Test market data updates."""
        symbol = "BTCUSDT"
        market_bar = generate_market_data_bar(symbol)
        atr_value = 0.02
        
        execution_engine.update_market_data_bar(symbol, market_bar, atr_value)
        
        # Verify data was updated
        assert symbol in execution_engine.portfolio_manager.volatilities
        assert execution_engine.portfolio_manager.volatilities[symbol] > 0
    
    def test_portfolio_rebalancing_trigger(self, market_data_setup):
        """Test portfolio rebalancing trigger."""
        # Force rebalance by setting old timestamp
        market_data_setup.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        result = market_data_setup.process_daily_rebalance()
        
        assert result is True
        assert market_data_setup.portfolio_manager.total_allocated > 0
    
    def test_portfolio_summary(self, portfolio_with_allocation):
        """Test portfolio summary generation."""
        summary = portfolio_with_allocation.get_portfolio_summary()
        
        assert isinstance(summary, dict)
        assert 'total_capital' in summary
        assert 'allocated_capital' in summary
        assert 'allocation_percentage' in summary
        assert 'symbol_allocations' in summary
        
        assert summary['total_capital'] > 0
        assert summary['allocated_capital'] > 0
        assert 0 <= summary['allocation_percentage'] <= 100
    
    def test_risk_metrics(self, execution_engine):
        """Test risk metrics generation."""
        metrics = execution_engine.get_risk_metrics()
        
        assert isinstance(metrics, dict)
        assert 'risk_status' in metrics
        assert 'daily_pnl' in metrics
        assert 'active_positions' in metrics
        assert 'max_drawdown_hit' in metrics
        
        assert metrics['risk_status'] in ['normal', 'caution', 'warning', 'critical']


class TestExecutionEngineSignalProcessing:
    """Test signal processing in execution engine."""
    
    @pytest.mark.asyncio
    async def test_signal_validation_interface(self, portfolio_with_allocation):
        """Test signal validation interface."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.8
        )
        
        validation_result = await portfolio_with_allocation.validate_signal(signal, 50000.0)
        
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        
        if validation_result['valid']:
            assert 'position_info' in validation_result
            position_info = validation_result['position_info']
            
            required_keys = ['size_contracts', 'leverage', 'stop_loss_price', 'take_profit_price']
            for key in required_keys:
                assert key in position_info
                assert position_info[key] is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_operations(self, portfolio_with_allocation):
        """Test concurrent execution operations."""
        # Test that multiple operations can run concurrently
        tasks = []
        
        # Create multiple signals
        for i in range(3):
            signal = TradeSignal(
                action="open",
                side="buy",
                symbol=f"TEST{i}USDT",
                strategy_id="test",
                metadata=create_test_signal_metadata(),
                signal_confidence=0.8
            )
            task = portfolio_with_allocation.validate_signal(signal, 50000.0)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent operations without errors
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
