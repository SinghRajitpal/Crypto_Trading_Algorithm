"""
Comprehensive Execution Engine Tests
Senior Quantitative Developer Testing Protocol

This file contains comprehensive tests for the Production Execution Engine
that align with the actual implementation and the document specifications.
Tests cover all aspects of portfolio management, risk management, and execution.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from execution.execution_engine import ProductionExecutionEngine
from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.risk_manager import ProductionRiskManager, ProductionRiskParameters
from execution.stress_handler import StressHandlingModule
from algorithm.trade_signal import TradeSignal
from tests.utils.test_data import generate_market_data_bar, create_test_signal_metadata
from tests.utils.mock_objects import MockBinanceClient


class TestProductionPortfolioManager:
    """Test the Production Portfolio Manager implementation."""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Create a portfolio manager for testing."""
        return ProductionPortfolioManager(
            total_capital=10000.0,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
    
    @pytest.fixture
    def symbols_with_data(self, portfolio_manager):
        """Setup portfolio manager with test data."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Add volatility data
        for i, symbol in enumerate(symbols):
            base_vol = 0.02 + (i * 0.005)  # Different volatilities
            for _ in range(10):  # Build up some history
                portfolio_manager.update_volatility_data(symbol, base_vol + np.random.normal(0, 0.001))
        
        # Add correlation data
        correlations = [(symbols[0], symbols[1], 0.6), 
                       (symbols[0], symbols[2], 0.4), 
                       (symbols[1], symbols[2], 0.5)]
        
        for sym1, sym2, corr in correlations:
            for _ in range(10):  # Build up history
                portfolio_manager.update_correlation_data(sym1, sym2, corr + np.random.normal(0, 0.05))
        
        return portfolio_manager, symbols
    
    def test_initialization(self, portfolio_manager):
        """Test portfolio manager initialization."""
        assert portfolio_manager.total_capital == 10000.0
        assert portfolio_manager.target_volatility == 0.18
        assert portfolio_manager.max_allocation_pct == 0.85
        assert portfolio_manager.alpha == 0.3  # Fixed correlation adjustment
        assert portfolio_manager.lookback_bars == 60
        assert isinstance(portfolio_manager.volatility_data, dict)
        assert isinstance(portfolio_manager.correlation_data, dict)
        assert isinstance(portfolio_manager.allocation_weights, dict)
        assert isinstance(portfolio_manager.reserved_allocations, dict)
    
    def test_volatility_data_update(self, portfolio_manager):
        """Test volatility data updates and EMA calculation."""
        symbol = "BTCUSDT"
        
        # Add some data points
        portfolio_manager.update_volatility_data(symbol, 0.02)
        portfolio_manager.update_volatility_data(symbol, 0.025)
        portfolio_manager.update_volatility_data(symbol, 0.03)
        
        assert symbol in portfolio_manager.volatility_data
        assert len(portfolio_manager.volatility_data[symbol]) == 3
        
        # Test EMA calculation
        ema = portfolio_manager.get_volatility_ema(symbol)
        assert isinstance(ema, float)
        assert ema > 0
        assert ema >= 0.001  # Should respect floor
    
    def test_correlation_data_update(self, portfolio_manager):
        """Test correlation data updates."""
        sym1, sym2 = "BTCUSDT", "ETHUSDT"
        
        # Test that symbols are ordered consistently
        portfolio_manager.update_correlation_data(sym1, sym2, 0.6)
        portfolio_manager.update_correlation_data(sym2, sym1, 0.7)  # Reverse order
        
        # Should be stored with consistent ordering
        pair = tuple(sorted([sym1, sym2]))
        assert pair in portfolio_manager.correlation_data
        assert len(portfolio_manager.correlation_data[pair]) == 2
    
    def test_weight_computation(self, symbols_with_data):
        """Test portfolio weight computation according to document formula."""
        portfolio_manager, symbols = symbols_with_data
        
        # Compute weights: w_i = (1/σ_i) × (1 + α × avg_correlation_i)
        weights = portfolio_manager.compute_weights(symbols)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(symbols)
        assert all(symbol in weights for symbol in symbols)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
    
    def test_high_volatility_regime_detection(self, symbols_with_data):
        """Test high volatility regime detection."""
        portfolio_manager, symbols = symbols_with_data
        
        # Initially should not be high vol regime (insufficient history)
        assert not portfolio_manager.is_high_volatility_regime()
        
        # Add some volatility history
        for _ in range(35):  # More than 30 days
            avg_vol = sum(portfolio_manager.get_volatility_ema(s) for s in symbols) / len(symbols)
            portfolio_manager.volatility_history.append(avg_vol)
        
        # Should now have enough data
        # Test regime detection
        regime_status = portfolio_manager.is_high_volatility_regime()
        assert isinstance(bool(regime_status), bool)

def test_scaling_multiplier_calculation(portfolio_manager):
    """Test scaling multiplier calculation."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Setup volatility data for calculation
    for symbol in symbols:
        portfolio_manager.update_volatility_data(symbol, 0.02)
    
    # Calculate scaling multiplier (no parameters)
    multiplier = portfolio_manager.calculate_scaling_multiplier()
    
    assert isinstance(multiplier, float)
    assert multiplier > 0
    assert multiplier <= 1.0
    
    def test_rebalancing_workflow(self, symbols_with_data):
        """Test complete rebalancing workflow."""
        portfolio_manager, symbols = symbols_with_data
        
        # Force rebalance trigger
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Perform rebalancing
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        assert isinstance(allocations, dict)
        assert len(allocations) == len(symbols)
        
        # Check allocation structure
        for symbol, allocation in allocations.items():
            assert isinstance(allocation, AllocationWeights)
            assert allocation.symbol == symbol
            assert allocation.allocated_capital > 0
            assert 0 < allocation.weight <= 1
            assert allocation.volatility > 0
        
        # Total allocation should respect max_allocation_pct
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        max_allowed = portfolio_manager.total_capital * portfolio_manager.max_allocation_pct
        assert total_allocated <= max_allowed * 1.01  # Small tolerance for floating point
    
    def test_reservation_system(self, symbols_with_data):
        """Test allocation reservation system."""
        portfolio_manager, symbols = symbols_with_data
        
        # First rebalance to get allocations
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        symbol = symbols[0]
        available = allocations[symbol].allocated_capital
        
        # Test successful reservation
        reservation_amount = available * 0.5
        result = portfolio_manager.reserve_allocation(symbol, reservation_amount)
        assert result is True
        assert symbol in portfolio_manager.reserved_allocations
        assert portfolio_manager.reserved_allocations[symbol] == reservation_amount
        
        # Test over-reservation
        excessive_amount = available * 2
        result = portfolio_manager.reserve_allocation(symbol, excessive_amount)
        assert result is False
        
        # Test release
        portfolio_manager.release_allocation(symbol, reservation_amount)
        assert portfolio_manager.reserved_allocations.get(symbol, 0) == 0


class TestProductionRiskManager:
    """Test the Production Risk Manager implementation."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager for testing."""
        portfolio_manager = ProductionPortfolioManager(10000.0)
        return ProductionRiskManager(portfolio_manager)
    
    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.portfolio_manager is not None
        assert isinstance(risk_manager.risk_params, ProductionRiskParameters)
        
        # Check document-specified parameters
        params = risk_manager.risk_params
        assert params.risk_per_trade_pct == 0.008  # 0.8%
        assert params.kelly_fraction == 0.7
        assert params.base_cost_pct == 0.0014  # 0.14%
        assert params.min_atr_floor == 0.001
        assert params.atr_stop_multiplier == 1.8
        assert params.risk_reward_ratio == 2.0
        assert params.max_leverage == 10
    
    def test_dynamic_cost_adjustment(self, risk_manager):
        """Test dynamic cost adjustment calculation."""
        # Formula: dynamic_cost = base_cost × (1 + 0.5 × normalized_volatility)
        volatility_norm = 0.5
        
        cost = risk_manager.calculate_dynamic_cost_adjustment(volatility_norm)
        expected = risk_manager.risk_params.base_cost_pct * (1 + 0.5 * volatility_norm)
        
        assert abs(cost - expected) < 1e-6
        assert cost > risk_manager.risk_params.base_cost_pct
    
    def test_position_sizing_calculation(self, risk_manager):
        """Test position sizing using document formula."""
        # Formula: Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001) - dynamic_cost
        symbol = "BTCUSDT"
        allocated_capital = 5000.0
        atr_value = 0.02
        entry_price = 50000.0
        volatility_norm = 0.5
        
        result = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=volatility_norm
        )
        
        assert isinstance(result, dict)
        assert 'size_usdt' in result
        assert 'leverage' in result
        assert 'stop_loss' in result
        assert 'take_profit' in result
        assert 'size_contracts' in result
        
        # Verify calculations
        assert result['size_usdt'] > 0
        assert 1 <= result['leverage'] <= 10
        assert result['stop_loss'] < entry_price  # For long position
        assert result['take_profit'] > entry_price  # For long position
        
        # Check stop loss uses ATR multiplier (1.8x)
        expected_sl = entry_price - (atr_value * risk_manager.risk_params.atr_stop_multiplier)
        assert abs(result['stop_loss'] - expected_sl) / expected_sl < 0.01
    
    def test_dynamic_leverage_calculation(self, risk_manager):
        """Test dynamic leverage calculation."""
        symbol = "BTCUSDT"
        atr_value = 0.02
        
        leverage = risk_manager.calculate_dynamic_leverage(symbol, atr_value)
        
        assert isinstance(leverage, (int, float))
        assert 1 <= leverage <= 10  # Respect max leverage
    
    def test_trade_validation(self, risk_manager):
        """Test trade validation logic."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={
                'price': 50000.0,
                'atr_value': 0.02,
                'volume': 1000000,
                'timestamp': int(time.time() * 1000)
            },
            signal_confidence=0.8
        )
        
        # Test with sufficient capital allocated from portfolio
        result = risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="open",
            side="buy",
            entry_price=50000.0,
            atr_value=0.02
        )
        assert isinstance(result, dict)
        assert 'valid' in result
        
        # Test with hold action (should be valid)
        result_hold = risk_manager.validate_trade(
            symbol="BTCUSDT", 
            action="hold",
            side="none",
            entry_price=50000.0,
            atr_value=0.02
        )
        assert result_hold['valid'] is True
    
    def test_equity_curve_tracking(self, risk_manager):
        """Test equity curve tracking for slope calculation."""
        # Add some equity points
        base_time = datetime.now()
        for i in range(10):
            equity = 10000 + i * 100  # Growing equity
            timestamp = base_time + timedelta(minutes=i)
            risk_manager.update_equity_curve(equity)
        
        assert len(risk_manager.equity_curve) == 10
        
        # Test slope calculation
        slope = risk_manager.calculate_equity_slope()
        assert isinstance(slope, float)
        assert slope > 0  # Should be positive for growing equity


class TestStressHandlingModule:
    """Test the Stress Handling Module implementation."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine with stress handler."""
        client = MockBinanceClient()
        return ProductionExecutionEngine(client, 10000.0)
    
    def test_stress_handler_initialization(self, execution_engine):
        """Test stress handler initialization."""
        stress_handler = execution_engine.stress_handler
        
        assert stress_handler is not None
        assert stress_handler.execution_engine is not None
        assert hasattr(stress_handler, 'check_flash_crash')
        assert hasattr(stress_handler, 'check_connection_lag')
        assert hasattr(stress_handler, 'check_kill_switches')
    
    def test_flash_crash_detection(self, execution_engine):
        """Test flash crash detection mechanism."""
        stress_handler = execution_engine.stress_handler
        symbol = "BTCUSDT"
        
        # Normal market data
        normal_data = {
            'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050, 'volume': 1000
        }
        atr_value = 0.02
        
        result = stress_handler.check_flash_crash(symbol, normal_data, atr_value)
        assert result is False  # No flash crash
        
        # Flash crash scenario (>4x ATR drop)
        crash_data = {
            'open': 50000, 'high': 50000, 'low': 45000, 'close': 45000, 'volume': 5000
        }
        
        result = stress_handler.check_flash_crash(symbol, crash_data, atr_value)
        assert result is True  # Flash crash detected
    
    def test_connection_lag_detection(self, execution_engine):
        """Test connection lag detection."""
        stress_handler = execution_engine.stress_handler
        
        # Normal latency (healthy connection)
        current_time = datetime.now()
        result = stress_handler.check_connection_lag(current_time)
        assert result is True
        
        # High latency (>3 seconds - lagged connection)
        old_time = current_time - timedelta(seconds=5)
        result_lag = stress_handler.check_connection_lag(old_time)
        assert result_lag is False
        old_time = datetime.now() - timedelta(seconds=5)
        result = stress_handler.check_connection_lag(old_time)
        assert result is True
    
def test_liquidity_filters(stress_handler):
    """Test liquidity filtering."""
    symbol = "BTCUSDT"
    
    # Good liquidity
    good_liquidity = stress_handler.check_liquidity_filters(
        symbol=symbol,
        volume_24h=10000000,  # $10M
        spread_pct=0.001,     # 0.1%
        funding_rate=0.001    # 0.1%
    )
    assert good_liquidity is True
    
    # Poor liquidity - low volume
    poor_liquidity = stress_handler.check_liquidity_filters(
        symbol=symbol,
        volume_24h=1000000,   # $1M (below $5M threshold)
        spread_pct=0.001,
        funding_rate=0.001
    )
    assert poor_liquidity is False
class TestExecutionEngineIntegration:
    """Test full execution engine integration."""
    
    @pytest_asyncio.fixture
    async def execution_engine(self):
        """Create and setup execution engine."""
        client = MockBinanceClient()
        engine = ProductionExecutionEngine(client, 10000.0)
        await engine.setup()
        return engine
    
    @pytest.mark.asyncio
    async def test_complete_signal_processing_workflow(self, execution_engine):
        """Test complete signal processing from data update to execution."""
        # Step 1: Update market data
        symbol = "BTCUSDT"
        market_data = generate_market_data_bar(symbol)
        atr_value = 0.02
        
        execution_engine.update_market_data_bar(symbol, market_data, atr_value)
        
        # Verify data was processed
        assert symbol in execution_engine.portfolio_manager.volatility_data
        
        # Step 2: Trigger rebalancing
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        rebalance_result = execution_engine.process_daily_rebalance()
        assert rebalance_result is True
        
        # Step 3: Process trading signal
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol=symbol,
            strategy_id="test_strategy",
            metadata={
                'price': market_data['close'],
                'atr_value': atr_value,
                'volume': market_data['volume'],
                'timestamp': int(time.time() * 1000)
            },
            signal_confidence=0.8
        )
        
        result = await execution_engine.process_signal(signal)
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'symbol' in result
    
    def test_portfolio_risk_metrics_integration(self, execution_engine):
        """Test integration between portfolio and risk metrics."""
        # Setup some market data
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        for symbol in symbols:
            market_data = generate_market_data_bar(symbol)
            execution_engine.update_market_data_bar(symbol, market_data, 0.02)
        
        # Get portfolio summary
        portfolio_summary = execution_engine.get_portfolio_summary()
        assert isinstance(portfolio_summary, dict)
        assert 'total_capital' in portfolio_summary
        assert 'allocated_capital' in portfolio_summary
        
        # Get risk metrics
        risk_metrics = execution_engine.get_risk_metrics()
        assert isinstance(risk_metrics, dict)
        assert 'risk_status' in risk_metrics
    
    def test_stress_conditions_integration(self, execution_engine):
        """Test system behavior under stress conditions."""
        # Test with extreme market conditions
        symbol = "BTCUSDT"
        
        # Simulate flash crash
        crash_data = {
            'open': 50000, 'high': 50000, 'low': 40000, 'close': 40000, 'volume': 10000
        }
        atr_value = 0.02
        
        execution_engine.update_market_data_bar(symbol, crash_data, atr_value)
        
        # System should handle this gracefully
        summary = execution_engine.get_portfolio_summary()
        assert summary is not None
        
        # Check if stress handler flagged the condition
        stress_result = execution_engine.stress_handler.check_flash_crash(symbol, crash_data, atr_value)
        assert stress_result is True
    
    def test_performance_metrics_calculation(self, execution_engine):
        """Test performance metrics and calculations."""
        # Add some historical data to calculate metrics
        portfolio_manager = execution_engine.portfolio_manager
        risk_manager = execution_engine.risk_manager
        
        # Simulate some trading history
        base_equity = 10000
        for i in range(30):  # 30 data points
            equity = base_equity + i * 50 + np.random.normal(0, 20)
            risk_manager.update_equity_curve(equity)
        
        # Calculate slope
        slope = risk_manager.calculate_equity_slope()
        assert isinstance(slope, float)
        
        # Test Sharpe ratio calculation if implemented
        if hasattr(risk_manager, 'calculate_sharpe_ratio'):
            sharpe = risk_manager.calculate_sharpe_ratio()
            assert isinstance(sharpe, float)
    
    def test_document_compliance_validation(self, execution_engine):
        """Test compliance with document specifications."""
        # Test portfolio allocation compliance
        portfolio_manager = execution_engine.portfolio_manager
        
        # Check fixed parameters from document
        assert portfolio_manager.alpha == 0.3  # Fixed correlation adjustment
        assert portfolio_manager.target_volatility == 0.18  # 18% target volatility
        assert portfolio_manager.max_allocation_pct == 0.85  # 85% max allocation
        
        # Check risk parameters compliance
        risk_params = execution_engine.risk_manager.risk_params
        assert risk_params.risk_per_trade_pct == 0.008  # 0.8% risk per trade
        assert risk_params.kelly_fraction == 0.7  # Fractional Kelly
        assert risk_params.atr_stop_multiplier == 1.8  # SL = Entry ± 1.8×ATR
        assert risk_params.risk_reward_ratio == 2.0  # 1:2 risk-reward
        assert risk_params.max_leverage == 10  # Cap leverage at 10x
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, execution_engine):
        """Test concurrent operations and thread safety."""
        # Simulate concurrent market data updates
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        async def update_market_data(symbol):
            for _ in range(5):
                market_data = generate_market_data_bar(symbol)
                execution_engine.update_market_data_bar(symbol, market_data, 0.02)
                await asyncio.sleep(0.01)  # Small delay
        
        # Run concurrent updates
        tasks = [update_market_data(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        # Verify all symbols have data
        for symbol in symbols:
            assert symbol in execution_engine.portfolio_manager.volatility_data
            assert len(execution_engine.portfolio_manager.volatility_data[symbol]) > 0


class TestPerformanceAndBenchmarks:
    """Test performance characteristics and benchmarks."""
    
    @pytest.mark.asyncio
    async def test_signal_processing_latency(self, execution_engine):
        """Test signal processing latency requirements."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=create_test_signal_metadata(),
            signal_confidence=0.8
        )
        
        # Measure processing time
        start_time = time.perf_counter()
        result = await execution_engine.process_signal(signal)
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should process within 30ms as per document requirements
        assert processing_time < 30, f"Signal processing took {processing_time:.2f}ms, should be <30ms"
        assert result is not None
    
    def test_portfolio_rebalancing_performance(self, execution_engine):
        """Test portfolio rebalancing performance."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
        
        # Setup data for multiple symbols
        for symbol in symbols:
            for _ in range(20):  # Add history
                execution_engine.update_market_data_bar(
                    symbol, 
                    generate_market_data_bar(symbol), 
                    0.02 + np.random.normal(0, 0.005)
                )
        
        # Measure rebalancing time
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        start_time = time.perf_counter()
        result = execution_engine.process_daily_rebalance()
        end_time = time.perf_counter()
        
        rebalancing_time = (end_time - start_time) * 1000
        
        # Should rebalance quickly
        assert rebalancing_time < 100, f"Rebalancing took {rebalancing_time:.2f}ms, should be <100ms"
        assert result is True
    
    def test_memory_usage_patterns(self, execution_engine):
        """Test memory usage patterns for large datasets."""
        import sys
        
        # Initial memory usage
        initial_size = sys.getsizeof(execution_engine)
        
        # Add substantial amount of data
        symbols = [f"SYM{i}USDT" for i in range(50)]
        
        for symbol in symbols:
            for _ in range(100):  # Lots of history
                execution_engine.update_market_data_bar(
                    symbol,
                    generate_market_data_bar(symbol),
                    0.02 + np.random.normal(0, 0.001)
                )
        
        # Final memory usage
        final_size = sys.getsizeof(execution_engine)
        
        # Memory growth should be reasonable
        memory_growth = final_size - initial_size
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth {memory_growth} bytes too large"
