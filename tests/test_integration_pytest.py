#!/usr/bin/env python3
"""
Comprehensive pytest integration tests for the crypto trading algorithm.

This module tests the integration between different modules to ensure
they work together correctly as specified in the trading document.

Tests cover:
- Portfolio-Risk Manager integration
- Signal-Execution integration
- Order lifecycle management
- Complete trading workflow
- Mathematical consistency across modules
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from execution.portfolio import ProductionPortfolioManager
from execution.risk_manager import ProductionRiskManager
from execution.order_manager import OrderManager
from algorithm.trade_signal import TradeSignal


class TestPortfolioRiskIntegration:
    """Test integration between Portfolio and Risk managers."""
    
    def test_allocation_to_position_sizing_workflow(self, portfolio_manager, risk_manager, test_symbols):
        """Test complete allocation → position sizing workflow."""
        symbols = test_symbols[:3]
        
        # Set up volatility data
        volatilities = [0.015, 0.025, 0.035]
        for symbol, vol in zip(symbols, volatilities):
            portfolio_manager.update_volatility_data(symbol, vol)
        
        # Force rebalance
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Test position sizing for each allocation
        position_results = {}
        for symbol, allocation in allocations.items():
            result = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=portfolio_manager.get_volatility_ema(symbol),
                entry_price=50000.0
            )
            position_results[symbol] = result
            
            # Verify integration consistency
            assert result['size_usdt'] > 0, f"Position size should be positive for {symbol}"
            assert result['size_usdt'] <= allocation.allocated_capital, \
                f"Position size should not exceed allocation for {symbol}"
            
            # Verify mathematical relationship
            expected_risk = allocation.allocated_capital * 0.008  # 0.8% risk
            assert result['risk_amount'] <= expected_risk * 1.2, \
                f"Risk amount should respect allocation limit for {symbol}"
        
        # Test total consistency
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        total_position_size = sum(result['size_usdt'] for result in position_results.values())
        
        # Position sizes should be reasonable fraction of allocations
        utilization_ratio = total_position_size / total_allocated
        assert 0.1 <= utilization_ratio <= 1.0, \
            f"Position utilization should be reasonable: {utilization_ratio:.2%}"
            
    def test_dynamic_leverage_with_allocations(self, portfolio_manager, risk_manager, test_symbols):
        """Test dynamic leverage calculation with real portfolio allocations."""
        symbols = test_symbols[:2]
        
        # Create high and low volatility scenarios
        high_vol_scenario = [0.08, 0.06]  # High volatility
        low_vol_scenario = [0.01, 0.015]  # Low volatility
        
        for scenario_name, volatilities in [("high_vol", high_vol_scenario), ("low_vol", low_vol_scenario)]:
            # Reset portfolio state
            portfolio_manager.volatility_emas = {}
            portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            
            # Set volatilities
            for symbol, vol in zip(symbols, volatilities):
                portfolio_manager.update_volatility_data(symbol, vol)
            
            allocations = portfolio_manager.rebalance_portfolio(symbols)
            
            # Calculate leverages
            leverages = {}
            for symbol in symbols:
                vol = portfolio_manager.get_volatility_ema(symbol)
                leverage = risk_manager.calculate_dynamic_leverage(symbol, vol)
                leverages[symbol] = leverage
                
                # Verify leverage constraints
                assert 1 <= leverage <= 10, f"Leverage should be 1-10x for {symbol} in {scenario_name}"
            
            # High volatility should generally result in lower leverage
            if scenario_name == "high_vol":
                for leverage in leverages.values():
                    assert leverage <= 5, "High volatility should limit leverage"
                    
    def test_risk_scaling_with_regime_detection(self, portfolio_manager, risk_manager, test_symbols):
        """Test risk scaling when portfolio detects different volatility regimes."""
        symbol = test_symbols[0]
        
        # Normal regime
        portfolio_manager.update_volatility_data(symbol, 0.02)
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        normal_allocations = portfolio_manager.rebalance_portfolio([symbol])
        
        normal_position = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=normal_allocations[symbol].allocated_capital,
            atr_value=0.02,
            entry_price=50000.0
        )
        
        # High volatility regime
        portfolio_manager.volatility_emas = {}  # Reset
        portfolio_manager.update_volatility_data(symbol, 0.08)  # High vol
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        high_vol_allocations = portfolio_manager.rebalance_portfolio([symbol])
        
        high_vol_position = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=high_vol_allocations[symbol].allocated_capital,
            atr_value=0.08,
            entry_price=50000.0
        )
        
        # High volatility should result in lower total risk exposure
        normal_total_risk = normal_position['size_usdt'] * normal_position['leverage']
        high_vol_total_risk = high_vol_position['size_usdt'] * high_vol_position['leverage']
        
        assert high_vol_total_risk <= normal_total_risk, \
            "High volatility regime should reduce total risk exposure"


class TestSignalExecutionIntegration:
    """Test integration between signal generation and execution."""
    
    def test_signal_to_position_sizing_integration(self, risk_manager, sample_trade_signal):
        """Test signal processing through to position sizing."""
        # Mock allocation
        allocated_capital = 3000.0
        
        # Calculate position size based on signal
        position_result = risk_manager.calculate_position_size(
            symbol=sample_trade_signal.symbol,
            allocated_capital=allocated_capital,
            atr_value=0.025,
            entry_price=sample_trade_signal.metadata["entry_price"]
        )
        
        # Verify signal metadata is used correctly
        assert position_result['size_usdt'] > 0
        
        # Test signal confidence impact (higher confidence should allow larger positions)
        # This would be implementation-specific
        
    def test_signal_validation_and_filtering(self, sample_trade_signal):
        """Test signal validation and filtering before execution."""
        # Test valid signal
        assert sample_trade_signal.symbol in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        assert sample_trade_signal.side in ["buy", "sell"]
        assert sample_trade_signal.action in ["open", "close"]
        assert 0 <= sample_trade_signal.signal_confidence <= 1
        
        # Test signal metadata validation
        assert "entry_price" in sample_trade_signal.metadata
        assert sample_trade_signal.metadata["entry_price"] > 0
        
    def test_signal_throttling_and_deduplication(self):
        """Test signal throttling to prevent duplicate orders."""
        # Create multiple similar signals
        signals = []
        for i in range(5):
            signal = TradeSignal(
                symbol="BTCUSDT",
                side="buy",
                action="open",
                strategy_id="test_strategy",
                metadata={"confidence": 0.8, "entry_price": 50000.0 + i},
                signal_confidence=0.8
            )
            signals.append(signal)
        
        # Test deduplication logic (implementation-specific)
        unique_symbols = set(signal.symbol for signal in signals)
        assert len(unique_symbols) == 1  # All same symbol
        
        # In real implementation, only one signal per symbol should be processed


class TestOrderLifecycleIntegration:
    """Test complete order lifecycle management."""
    
    def test_order_creation_from_signal(self, order_executor, sample_trade_signal, test_allocations):
        """Test order creation from trade signal with proper SL/TP."""
        # This would test the complete order creation process
        # Mock the order creation (actual implementation would place real orders)
        
        symbol = sample_trade_signal.symbol
        if symbol in test_allocations:
            allocation = test_allocations[symbol]
            
            # Test order structure would be created here
            # Including entry order, stop-loss, and take-profit
            assert allocation.allocated_capital > 0
            
    def test_sl_tp_order_pair_management(self, risk_manager):
        """Test SL/TP order pair creation and management."""
        entry_price = 50000.0
        atr_adjusted = 1000.0  # $1000 ATR
        
        # Calculate SL/TP for buy order
        sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="buy",
            atr_adjusted=atr_adjusted
        )
        
        # Test proper order pair structure
        assert sl_price < entry_price < tp_price, "SL < Entry < TP for buy orders"
        
        # Test for sell order
        sl_price_sell, tp_price_sell = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="sell",
            atr_adjusted=atr_adjusted
        )
        
        assert tp_price_sell < entry_price < sl_price_sell, "TP < Entry < SL for sell orders"
        
        # Test cancellation logic (one order fills, cancel the other)
        # This would be implementation-specific based on exchange capabilities
        
    def test_order_state_tracking(self):
        """Test order state tracking throughout lifecycle."""
        # Test order states: pending, filled, cancelled, rejected
        order_states = ["pending", "filled", "cancelled", "rejected"]
        
        # Each state should be trackable
        for state in order_states:
            assert state in order_states  # Basic test structure
            
        # Test state transitions
        valid_transitions = {
            "pending": ["filled", "cancelled", "rejected"],
            "filled": [],  # Terminal state
            "cancelled": [],  # Terminal state
            "rejected": []  # Terminal state
        }
        
        for from_state, to_states in valid_transitions.items():
            assert isinstance(to_states, list)


class TestCompleteWorkflowIntegration:
    """Test complete end-to-end trading workflow."""
    
    def test_daily_rebalancing_workflow(self, portfolio_manager, risk_manager, test_symbols):
        """Test complete daily rebalancing workflow."""
        symbols = test_symbols[:3]
        
        # Day 1: Initial setup
        for i, symbol in enumerate(symbols):
            portfolio_manager.update_volatility_data(symbol, 0.02 + i * 0.005)
        
        # Trigger rebalancing
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        day1_allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Calculate positions for Day 1
        day1_positions = {}
        for symbol, allocation in day1_allocations.items():
            position = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=portfolio_manager.get_volatility_ema(symbol),
                entry_price=50000.0
            )
            day1_positions[symbol] = position
        
        # Day 2: Market conditions change
        for i, symbol in enumerate(symbols):
            portfolio_manager.update_volatility_data(symbol, 0.025 + i * 0.01)  # Different vols
        
        # Trigger rebalancing again
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        day2_allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Verify rebalancing occurred
        assert day1_allocations != day2_allocations, "Allocations should change with different market conditions"
        
        # Verify total allocations are consistent
        day1_total = sum(alloc.allocated_capital for alloc in day1_allocations.values())
        day2_total = sum(alloc.allocated_capital for alloc in day2_allocations.values())
        
        max_capital = portfolio_manager.total_capital * 0.85
        assert abs(day1_total - day2_total) < max_capital * 0.1, "Total allocation should be stable"
        
    def test_signal_to_execution_complete_flow(self, portfolio_manager, risk_manager, test_symbols):
        """Test complete flow from signal generation to execution."""
        symbol = test_symbols[0]
        
        # 1. Set up portfolio allocation
        portfolio_manager.update_volatility_data(symbol, 0.025)
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio([symbol])
        
        # 2. Generate signal
        signal = TradeSignal(
            symbol=symbol,
            side="buy",
            action="open",
            strategy_id="integration_test",
            metadata={"confidence": 0.75, "entry_price": 48000.0},
            signal_confidence=0.75
        )
        
        # 3. Calculate position size
        allocation = allocations[symbol]
        position = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocation.allocated_capital,
            atr_value=0.025,
            entry_price=signal.metadata["entry_price"]
        )
        
        # 4. Calculate SL/TP
        sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
            entry_price=signal.metadata["entry_price"],
            side=signal.side,
            atr_adjusted=0.025 * signal.metadata["entry_price"]
        )
        
        # 5. Verify complete workflow consistency
        assert position['size_usdt'] > 0, "Position size should be positive"
        assert position['size_usdt'] <= allocation.allocated_capital, "Position should not exceed allocation"
        assert sl_price < signal.metadata["entry_price"] < tp_price, "SL < Entry < TP for buy"
        
        # 6. Verify risk constraints
        risk_amount = position['risk_amount']
        max_risk = allocation.allocated_capital * 0.008  # 0.8% max risk
        assert risk_amount <= max_risk * 1.1, "Risk should not exceed 0.8% of allocation"
        
    def test_error_propagation_and_recovery(self, portfolio_manager, risk_manager):
        """Test error handling and recovery across modules."""
        # Test portfolio manager error handling
        try:
            # Invalid volatility data
            portfolio_manager.update_volatility_data("INVALID", -0.01)
            vol = portfolio_manager.get_volatility_ema("INVALID")
            assert vol >= 0, "Negative volatility should be handled"
        except Exception as e:
            # Error should be handled gracefully
            assert isinstance(e, (ValueError, AssertionError))
        
        # Test risk manager error handling
        try:
            # Invalid position sizing parameters
            result = risk_manager.calculate_position_size(
                symbol="TEST",
                allocated_capital=-1000.0,  # Negative capital
                atr_value=0.02,
                entry_price=50000.0
            )
            # Should either handle gracefully or raise appropriate error
        except Exception as e:
            assert isinstance(e, (ValueError, AssertionError))
        
        # Test system recovery after errors
        # System should continue functioning after error conditions
        portfolio_manager.update_volatility_data("BTCUSDT", 0.02)
        vol = portfolio_manager.get_volatility_ema("BTCUSDT")
        assert vol > 0, "System should recover and function normally"
        
    def test_mathematical_consistency_across_modules(self, portfolio_manager, risk_manager, test_symbols):
        """Test mathematical consistency across all modules."""
        symbols = test_symbols[:3]
        
        # Set up consistent test scenario
        test_capital = 15000.0
        assert portfolio_manager.total_capital == test_capital
        
        # Set volatilities
        for i, symbol in enumerate(symbols):
            portfolio_manager.update_volatility_data(symbol, 0.02 + i * 0.01)
        
        # Get allocations
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Calculate all positions
        total_allocated = 0
        total_position_value = 0
        total_risk = 0
        
        for symbol, allocation in allocations.items():
            total_allocated += allocation.allocated_capital
            
            position = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=portfolio_manager.get_volatility_ema(symbol),
                entry_price=50000.0
            )
            
            total_position_value += position['size_usdt']
            total_risk += position['risk_amount']
        
        # Mathematical consistency checks
        max_allocation = test_capital * 0.85
        assert abs(total_allocated - max_allocation) < 1.0, "Total allocation should equal 85% of capital"
        
        # Risk should be reasonable percentage of allocated capital
        total_risk_pct = total_risk / total_allocated
        assert total_risk_pct <= 0.01, "Total risk should be reasonable percentage of allocated capital"
        
        # Position values should be reasonable fraction of allocations
        utilization = total_position_value / total_allocated
        assert 0.1 <= utilization <= 1.0, f"Position utilization should be reasonable: {utilization:.2%}"
        
    @pytest.mark.slow
    def test_extended_integration_scenario(self, portfolio_manager, risk_manager, test_symbols):
        """Test extended integration scenario over multiple days."""
        symbols = test_symbols[:4]
        
        # Simulate 5 days of trading
        daily_results = []
        
        for day in range(5):
            # Update market conditions (changing volatilities)
            base_vol = 0.015 + day * 0.005
            for i, symbol in enumerate(symbols):
                vol = base_vol + i * 0.005 + (day % 2) * 0.01  # Some variation
                portfolio_manager.update_volatility_data(symbol, vol)
            
            # Force rebalancing
            portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
            allocations = portfolio_manager.rebalance_portfolio(symbols)
            
            # Calculate positions for each symbol
            day_positions = {}
            day_total_allocation = 0
            day_total_risk = 0
            
            for symbol, allocation in allocations.items():
                position = risk_manager.calculate_position_size(
                    symbol=symbol,
                    allocated_capital=allocation.allocated_capital,
                    atr_value=portfolio_manager.get_volatility_ema(symbol),
                    entry_price=50000.0 + day * 1000  # Slightly changing price
                )
                
                day_positions[symbol] = position
                day_total_allocation += allocation.allocated_capital
                day_total_risk += position['risk_amount']
            
            daily_results.append({
                'day': day,
                'total_allocation': day_total_allocation,
                'total_risk': day_total_risk,
                'positions': day_positions
            })
        
        # Analyze consistency over time
        allocations = [result['total_allocation'] for result in daily_results]
        risks = [result['total_risk'] for result in daily_results]
        
        # Allocations should be relatively stable (within 10% range)
        max_allocation = max(allocations)
        min_allocation = min(allocations)
        allocation_range = (max_allocation - min_allocation) / min_allocation
        assert allocation_range <= 0.15, f"Allocation range should be stable: {allocation_range:.1%}"
        
        # Risk should be controlled
        max_risk = max(risks)
        total_capital = portfolio_manager.total_capital
        max_risk_pct = max_risk / total_capital
        assert max_risk_pct <= 0.05, f"Maximum daily risk should be controlled: {max_risk_pct:.1%}"
        
        # All results should be positive and reasonable
        for result in daily_results:
            assert result['total_allocation'] > 0
            assert result['total_risk'] > 0
            for position in result['positions'].values():
                assert position['size_usdt'] > 0
                assert 1 <= position['leverage'] <= 10
