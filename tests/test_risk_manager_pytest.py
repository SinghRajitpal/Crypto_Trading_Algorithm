#!/usr/bin/env python3
"""
Comprehensive pytest unit tests for Risk Manager.

This module tests the ProductionRiskManager implementation against
the mathematical formulas specified in the trading document.

Tests cover:
- Position sizing formula validation
- Dynamic leverage calculation
- Stop-loss and take-profit calculations
- Kelly criterion implementation
- ATR floor enforcement
- Dynamic cost adjustments
"""

import pytest
import math
from execution.risk_manager import ProductionRiskManager, ProductionRiskParameters
from execution.portfolio import ProductionPortfolioManager


class TestProductionRiskManager:
    """Test suite for ProductionRiskManager unit tests."""
    
    def test_initialization(self):
        """Test risk manager initialization with correct parameters."""
        portfolio_manager = ProductionPortfolioManager(total_capital=10000.0)
        risk_manager = ProductionRiskManager(portfolio_manager)
        
        # Test basic initialization
        assert risk_manager.portfolio_manager == portfolio_manager
        assert risk_manager.risk_params.risk_per_trade_pct == 0.008  # 0.8% risk
        assert risk_manager.risk_params.kelly_fraction == 0.7    # 70% Kelly fraction
        assert risk_manager.risk_params.base_cost_pct == 0.0014  # 0.14% base cost
        assert risk_manager.risk_params.min_atr_floor == 0.001   # 0.1% ATR floor
        
        # Test data structure initialization
        assert risk_manager.drawdown_history == []
        assert risk_manager.sharpe_history == []
        assert risk_manager.equity_curve == []
        assert risk_manager.positions == {}
        assert risk_manager.daily_pnl == 0.0
        
    def test_position_sizing_formula(self, risk_manager):
        """Test position sizing formula: Size = (0.8% × Allocated × 0.7) / max(ATR, 0.001) - dynamic_cost"""
        # Test parameters
        allocated_capital = 3000.0
        atr_value = 0.02
        entry_price = 50000.0
        volatility_norm = 0.5
        
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=volatility_norm
        )
        
        # Verify result structure
        required_keys = ['size_contracts', 'size_usdt', 'leverage', 'margin_usdt', 
                        'risk_amount', 'atr_adjusted']
        for key in required_keys:
            assert key in result, f"Missing key in position size result: {key}"
        
        # Test formula components
        risk_per_trade = 0.008  # 0.8%
        kelly_fraction = 0.7
        atr_floor = 0.001
        
        atr_adjusted = max(atr_value, atr_floor)
        numerator = risk_per_trade * allocated_capital * kelly_fraction
        
        # Dynamic cost calculation
        base_cost = 0.0014  # 0.04% + 0.1% spread
        dynamic_cost = base_cost * (1 + 0.5 * volatility_norm)
        
        # Verify calculations
        assert result['atr_adjusted'] == atr_adjusted
        assert result['size_usdt'] > 0, "Position size should be positive"
        assert result['leverage'] >= 1, "Leverage should be at least 1"
        assert result['leverage'] <= 10, "Leverage should not exceed 10x"
        assert result['margin_usdt'] > 0, "Margin should be positive"
        assert result['risk_amount'] > 0, "Risk amount should be positive"
        
        # Test risk per trade constraint
        max_risk = allocated_capital * risk_per_trade
        assert result['risk_amount'] <= max_risk * 1.1, "Risk should not exceed target significantly"
        
    def test_dynamic_leverage_calculation(self, risk_manager):
        """Test dynamic leverage calculation with various market conditions."""
        # Test normal conditions
        leverage = risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.02)
        assert 1 <= leverage <= 10, f"Leverage should be 1-10x, got {leverage}"
        
        # Test high volatility (should reduce leverage)
        high_vol_leverage = risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.08)
        normal_vol_leverage = risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.02)
        assert high_vol_leverage <= normal_vol_leverage, "High volatility should reduce leverage"
        
        # Test very low volatility
        low_vol_leverage = risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.005)
        assert low_vol_leverage <= 10, "Leverage should be capped at 10x"
        
    def test_stop_loss_take_profit_calculation(self, risk_manager):
        """Test SL/TP formula: SL = Entry ± 1.8×ATR, TP = Entry ± 2×|Entry-SL|"""
        entry_price = 50000.0
        atr_value = 0.02
        atr_adjusted = atr_value * entry_price  # Convert to price terms
        
        # Test buy orders
        sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="buy",
            atr_adjusted=atr_adjusted
        )
        
        # For buy orders: SL below entry, TP above entry
        assert sl_price < entry_price, "Stop loss should be below entry for buy"
        assert tp_price > entry_price, "Take profit should be above entry for buy"
        
        # Test ATR multiplier (1.8x from document)
        atr_multiplier = 1.8
        expected_sl_distance = atr_multiplier * atr_adjusted
        actual_sl_distance = entry_price - sl_price
        
        tolerance = atr_adjusted * 0.05  # 5% tolerance
        assert abs(actual_sl_distance - expected_sl_distance) <= tolerance, \
            f"SL distance should be ~{expected_sl_distance}, got {actual_sl_distance}"
        
        # Test risk-reward ratio (~2:1 from document)
        risk_distance = entry_price - sl_price
        reward_distance = tp_price - entry_price
        rr_ratio = reward_distance / risk_distance
        
        assert 1.8 <= rr_ratio <= 2.2, f"Risk-reward ratio should be ~2:1, got {rr_ratio:.2f}"
        
        # Test sell orders
        sl_price_sell, tp_price_sell = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side="sell",
            atr_adjusted=atr_adjusted
        )
        
        # For sell orders: SL above entry, TP below entry
        assert sl_price_sell > entry_price, "Stop loss should be above entry for sell"
        assert tp_price_sell < entry_price, "Take profit should be below entry for sell"
        
    def test_atr_floor_enforcement(self, risk_manager):
        """Test ATR floor enforcement at 0.001 (0.1%)."""
        allocated_capital = 3000.0
        entry_price = 50000.0
        
        # Test with ATR below floor
        tiny_atr = 0.0001  # 0.01% - below 0.1% floor
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=tiny_atr,
            entry_price=entry_price
        )
        
        assert result['atr_adjusted'] == 0.001, "ATR floor should be enforced at 0.001"
        
        # Test with ATR above floor
        normal_atr = 0.02  # 2% - above floor
        result_normal = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=normal_atr,
            entry_price=entry_price
        )
        
        assert result_normal['atr_adjusted'] == normal_atr, "ATR above floor should be unchanged"
        
        # Verify different position sizes
        assert result['size_usdt'] > result_normal['size_usdt'], \
            "Lower ATR (floored) should result in larger position size"
            
    def test_dynamic_cost_calculation(self, risk_manager):
        """Test dynamic cost calculation based on volatility."""
        allocated_capital = 3000.0
        atr_value = 0.02
        entry_price = 50000.0
        
        # Test low volatility scenario
        low_vol_result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=0.2  # Low volatility
        )
        
        # Test high volatility scenario
        high_vol_result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=0.8  # High volatility
        )
        
        # High volatility should result in smaller position due to higher costs
        assert high_vol_result['size_usdt'] <= low_vol_result['size_usdt'], \
            "High volatility should reduce position size due to higher dynamic costs"
            
    def test_kelly_fraction_implementation(self, risk_manager):
        """Test Kelly fraction (0.7) implementation in position sizing."""
        allocated_capital = 3000.0
        atr_value = 0.02
        entry_price = 50000.0
        
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price
        )
        
        # Calculate expected position without Kelly fraction
        risk_per_trade = 0.008
        base_position = (risk_per_trade * allocated_capital) / atr_value
        
        # With Kelly fraction, position should be smaller
        kelly_adjusted = base_position * 0.7
        
        # Allow some tolerance for dynamic costs
        tolerance = kelly_adjusted * 0.2  # 20% tolerance
        assert abs(result['size_usdt'] - kelly_adjusted) <= tolerance + 100, \
            "Position size should reflect Kelly fraction adjustment"
            
    def test_leverage_constraints(self, risk_manager):
        """Test leverage constraints and boundary conditions."""
        # Test leverage never exceeds maximum
        test_cases = [
            (0.005, 1000.0),   # Very low ATR
            (0.001, 5000.0),   # ATR floor
            (0.1, 500.0),      # Very high ATR
        ]
        
        for atr, capital in test_cases:
            result = risk_manager.calculate_position_size(
                symbol="TESTBTC",
                allocated_capital=capital,
                atr_value=atr,
                entry_price=50000.0
            )
            
            assert result['leverage'] <= 10, f"Leverage should not exceed 10x, got {result['leverage']}"
            assert result['leverage'] >= 1, f"Leverage should be at least 1x, got {result['leverage']}"
            
    def test_edge_cases_and_error_handling(self, risk_manager):
        """Test edge cases and error handling."""
        # Test zero allocated capital - should return zero position
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=0.0,
            atr_value=0.02,
            entry_price=50000.0
        )
        assert result['size_usdt'] == 0.0, "Zero capital should result in zero position"
        assert result['size_contracts'] == 0.0, "Zero capital should result in zero contracts"
        
        # Test negative ATR (should be handled gracefully)
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=1000.0,
            atr_value=-0.01,  # Negative ATR
            entry_price=50000.0
        )
        # Should use ATR floor
        assert result['atr_adjusted'] >= 0.001
        
        # Test very high entry price
        result_high_price = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=1000.0,
            atr_value=0.02,
            entry_price=1000000.0  # Very high price
        )
        assert result_high_price['size_usdt'] > 0
        assert result_high_price['size_contracts'] > 0
        
    @pytest.mark.parametrize("atr_value,expected_atr", [
        (0.0001, 0.001),  # Below floor
        (0.001, 0.001),   # At floor
        (0.002, 0.002),   # Above floor
        (0.05, 0.05),     # High ATR
        (0.1, 0.1),       # Very high ATR
    ])
    def test_atr_floor_parametrized(self, risk_manager, atr_value, expected_atr):
        """Parametrized test for ATR floor enforcement."""
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=1000.0,
            atr_value=atr_value,
            entry_price=50000.0
        )
        
        assert result['atr_adjusted'] == expected_atr
        
    @pytest.mark.parametrize("side,entry_price", [
        ("buy", 50000.0),
        ("sell", 50000.0),
        ("buy", 1000.0),    # Low price
        ("sell", 100000.0), # High price
    ])
    def test_sl_tp_parametrized(self, risk_manager, side, entry_price):
        """Parametrized test for SL/TP calculations."""
        atr_adjusted = 0.02 * entry_price
        
        sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            side=side,
            atr_adjusted=atr_adjusted
        )
        
        if side == "buy":
            assert sl_price < entry_price, "SL should be below entry for buy"
            assert tp_price > entry_price, "TP should be above entry for buy"
        else:  # sell
            assert sl_price > entry_price, "SL should be above entry for sell"
            assert tp_price < entry_price, "TP should be below entry for sell"
            
        # Test risk-reward ratio
        if side == "buy":
            risk = entry_price - sl_price
            reward = tp_price - entry_price
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price
            
        rr_ratio = reward / risk
        assert 1.8 <= rr_ratio <= 2.2, f"Risk-reward should be ~2:1, got {rr_ratio:.2f}"
        
    def test_mathematical_consistency(self, risk_manager):
        """Test mathematical consistency across different scenarios."""
        base_params = {
            "symbol": "BTCUSDT",
            "allocated_capital": 5000.0,
            "atr_value": 0.025,
            "entry_price": 45000.0
        }
        
        # Test multiple calculations with same parameters
        results = []
        for _ in range(5):
            result = risk_manager.calculate_position_size(**base_params)
            results.append(result)
        
        # All results should be identical (deterministic)
        for key in results[0].keys():
            values = [r[key] for r in results]
            assert all(v == values[0] for v in values), f"Inconsistent results for {key}"
        
        # Test scaling properties
        double_capital = base_params.copy()
        double_capital["allocated_capital"] *= 2
        
        normal_result = risk_manager.calculate_position_size(**base_params)
        double_result = risk_manager.calculate_position_size(**double_capital)
        
        # Position size should roughly double with double capital
        size_ratio = double_result['size_usdt'] / normal_result['size_usdt']
        assert 1.8 <= size_ratio <= 2.2, f"Position size should scale with capital, ratio: {size_ratio}"
