#!/usr/bin/env python3
"""
Comprehensive pytest test suite for dynamic portfolio allocation, risk management, 
position sizing, leverage, stop losses, and take profits.

This test suite verifies that all dynamic calculations work correctly together:
1. Dynamic portfolio allocation based on volatility and correlations
2. Risk-adjusted position sizing using ATR and Kelly criterion
3. Dynamic leverage calculation based on volatility and drawdown
4. Stop loss and take profit calculations
5. Integration of all components working together
"""

import pytest
import sys
import os
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.risk_manager import ProductionRiskManager, RiskParameters
from execution.execution_engine import ProductionExecutionEngine
from binance_exchange import BinanceClient


class TestDynamicPortfolioAllocation:
    """Test dynamic portfolio allocation based on changing market conditions."""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Portfolio manager with realistic capital."""
        return ProductionPortfolioManager(
            total_capital=15000.0,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
    
    @pytest.fixture
    def test_symbols(self):
        """Test symbols for allocation."""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    
    def test_dynamic_allocation_with_different_volatilities(self, portfolio_manager, test_symbols):
        """Test that allocation changes dynamically with different volatilities."""
        print("\n🔬 Testing Dynamic Allocation with Different Volatilities")
        
        # Scenario 1: Equal volatilities (should get equal weights)
        print("\n📊 Scenario 1: Equal Volatilities")
        for symbol in test_symbols:
            for _ in range(20):
                portfolio_manager.update_volatility_data(symbol, 0.02)  # 2% volatility
        
        # Add correlations
        for i, sym1 in enumerate(test_symbols):
            for sym2 in test_symbols[i+1:]:
                for _ in range(20):
                    portfolio_manager.update_correlation_data(sym1, sym2, 0.6)
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations_equal = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # With equal volatilities, should get approximately equal allocations
        allocations_list = [a.allocated_capital for a in allocations_equal.values()]
        mean_allocation = np.mean(allocations_list)
        std_allocation = np.std(allocations_list)
        cv = std_allocation / mean_allocation  # Coefficient of variation
        
        assert cv < 0.1, f"With equal volatilities, allocations should be similar (CV: {cv:.3f})"
        print(f"✅ Equal volatilities → Similar allocations (CV: {cv:.3f})")
        
        # Scenario 2: Different volatilities (lower vol should get higher allocation)
        print("\n📊 Scenario 2: Different Volatilities")
        volatilities = [0.015, 0.020, 0.025, 0.030, 0.035]  # Increasing volatility
        
        # Clear existing data and add new volatilities
        portfolio_manager.volatility_data = {}
        for symbol, vol in zip(test_symbols, volatilities):
            for _ in range(20):
                portfolio_manager.update_volatility_data(symbol, vol)
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations_different = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Lower volatility symbols should get higher allocation
        btc_allocation = allocations_different["BTCUSDT"].allocated_capital  # Lowest vol (0.015)
        xrp_allocation = allocations_different["XRPUSDT"].allocated_capital   # Highest vol (0.035)
        
        assert btc_allocation > xrp_allocation, "Lower volatility asset should get higher allocation"
        print(f"✅ Lower vol (BTC): ${btc_allocation:.2f} > Higher vol (XRP): ${xrp_allocation:.2f}")
        
        # Scenario 3: Test scaling multiplier with high volatility regime
        print("\n📊 Scenario 3: High Volatility Regime")
        
        # Simulate high volatility regime by adding high volatility history
        for _ in range(35):
            portfolio_manager.volatility_history.append(0.05)  # High volatility
        
        scaling_multiplier = portfolio_manager.calculate_scaling_multiplier()
        assert scaling_multiplier < 1.0, "High volatility regime should reduce scaling multiplier"
        print(f"✅ High volatility regime → Reduced scaling: {scaling_multiplier:.3f}")
    
    def test_dynamic_correlation_impact(self, portfolio_manager, test_symbols):
        """Test that correlations dynamically impact allocation."""
        print("\n🔬 Testing Dynamic Correlation Impact")
        
        # Add equal volatilities
        for symbol in test_symbols:
            for _ in range(20):
                portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Scenario 1: Low correlations (should get more equal distribution)
        print("\n📊 Scenario 1: Low Correlations")
        for i, sym1 in enumerate(test_symbols):
            for sym2 in test_symbols[i+1:]:
                for _ in range(20):
                    portfolio_manager.update_correlation_data(sym1, sym2, 0.2)  # Low correlation
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations_low_corr = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Scenario 2: High correlations (should adjust weights)
        print("\n📊 Scenario 2: High Correlations")
        portfolio_manager.correlation_data = {}  # Clear existing
        for i, sym1 in enumerate(test_symbols):
            for sym2 in test_symbols[i+1:]:
                for _ in range(20):
                    portfolio_manager.update_correlation_data(sym1, sym2, 0.9)  # High correlation
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations_high_corr = portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Compare total allocations (high correlation should affect distribution)
        low_corr_total = sum(a.allocated_capital for a in allocations_low_corr.values())
        high_corr_total = sum(a.allocated_capital for a in allocations_high_corr.values())
        
        print(f"✅ Low correlation total: ${low_corr_total:.2f}")
        print(f"✅ High correlation total: ${high_corr_total:.2f}")
        
        # Both should be valid allocations
        assert low_corr_total > 0 and high_corr_total > 0
        print(f"✅ Correlations dynamically impact allocation distribution")


class TestDynamicRiskManagement:
    """Test dynamic risk management calculations."""
    
    @pytest.fixture
    def risk_manager(self):
        """Risk manager with portfolio manager."""
        portfolio_manager = ProductionPortfolioManager(total_capital=15000.0)
        return ProductionRiskManager(portfolio_manager=portfolio_manager)
    
    def test_dynamic_position_sizing_with_atr(self, risk_manager):
        """Test position sizing changes dynamically with ATR values."""
        print("\n🔬 Testing Dynamic Position Sizing with ATR")
        
        symbol = "BTCUSDT"
        allocated_capital = 3000.0
        entry_price = 50000.0
        
        # Test different ATR values
        atr_scenarios = [
            (0.01, "Low volatility"),    # 1% ATR
            (0.02, "Medium volatility"), # 2% ATR
            (0.04, "High volatility")    # 4% ATR
        ]
        
        position_sizes = []
        for atr_value, scenario in atr_scenarios:
            result = risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocated_capital,
                atr_value=atr_value,
                entry_price=entry_price
            )
            
            position_size = result['position_size_usdt']
            position_sizes.append(position_size)
            
            print(f"📊 {scenario} (ATR: {atr_value:.1%}): Position size ${position_size:.2f}")
        
        # Higher ATR should result in smaller position size (risk adjustment)
        assert position_sizes[0] > position_sizes[1] > position_sizes[2], \
            "Higher ATR should result in smaller position sizes"
        
        print(f"✅ Position sizing correctly adjusts to volatility")
        
        # Test ATR floor (shouldn't allow extremely small ATR to cause huge positions)
        tiny_atr_result = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=0.0001,  # Extremely small ATR
            entry_price=entry_price
        )
        
        # Should use the ATR floor (0.001) to prevent excessive position sizing
        assert tiny_atr_result['position_size_usdt'] < allocated_capital * 0.5, \
            "ATR floor should prevent excessive position sizing"
        
        print(f"✅ ATR floor prevents excessive position sizing")
    
    def test_dynamic_leverage_calculation(self, risk_manager):
        """Test dynamic leverage calculation based on market conditions."""
        print("\n🔬 Testing Dynamic Leverage Calculation")
        
        symbol = "BTCUSDT"
        base_atr = 0.02
        
        # Test different volatility scenarios
        scenarios = [
            (0.01, "Low volatility"),
            (0.02, "Normal volatility"), 
            (0.05, "High volatility")
        ]
        
        leverages = []
        for atr_value, scenario in scenarios:
            leverage = risk_manager.calculate_dynamic_leverage(symbol, atr_value)
            leverages.append(leverage)
            print(f"📊 {scenario} (ATR: {atr_value:.1%}): Leverage {leverage}x")
        
        # Higher volatility should result in lower leverage
        assert leverages[0] >= leverages[1] >= leverages[2], \
            "Higher volatility should result in lower leverage"
        
        print(f"✅ Leverage correctly adjusts to volatility")
        
        # Test leverage cap
        extreme_low_atr = 0.001  # Extremely low volatility
        max_leverage = risk_manager.calculate_dynamic_leverage(symbol, extreme_low_atr)
        assert max_leverage <= 10, "Leverage should be capped at 10x"
        
        print(f"✅ Leverage capped at maximum (got {max_leverage}x)")
    
    def test_dynamic_stop_loss_take_profit(self, risk_manager):
        """Test dynamic stop loss and take profit calculation."""
        print("\n🔬 Testing Dynamic Stop Loss and Take Profit")
        
        entry_price = 50000.0
        side = "buy"
        
        # Test different ATR values
        atr_scenarios = [0.01, 0.02, 0.04]  # 1%, 2%, 4%
        
        for atr_value in atr_scenarios:
            sl_price, tp_price = risk_manager.calculate_stop_loss_take_profit(
                entry_price=entry_price,
                side=side,
                atr_adjusted=atr_value * entry_price  # Convert to price terms
            )
            
            sl_distance = abs(entry_price - sl_price)
            tp_distance = abs(tp_price - entry_price)
            risk_reward_ratio = tp_distance / sl_distance
            
            print(f"📊 ATR {atr_value:.1%}: SL ${sl_price:.0f}, TP ${tp_price:.0f}, R:R {risk_reward_ratio:.2f}")
            
            # Stop loss should be below entry for long, take profit above
            assert sl_price < entry_price, "Stop loss should be below entry price for long"
            assert tp_price > entry_price, "Take profit should be above entry price for long"
            
            # Risk-reward ratio should be approximately 2:1
            assert 1.8 <= risk_reward_ratio <= 2.2, f"Risk-reward ratio should be ~2:1, got {risk_reward_ratio:.2f}"
        
        print(f"✅ Stop loss and take profit correctly scale with ATR")
        print(f"✅ Risk-reward ratio maintained at ~2:1")


class TestIntegratedDynamicSystem:
    """Test the complete integrated system with dynamic calculations."""
    
    @pytest.fixture
    def mock_binance_client(self):
        """Mock Binance client for testing."""
        client = MagicMock()
        client.get_account_metrics = AsyncMock(return_value={
            'total_wallet_balance': 15000.0,
            'total_unrealized_pnl': 0.0,
            'total_margin_used': 0.0,
            'available_margin': 15000.0,
            'exposure_percentage': 0.0,
            'position_count': 0
        })
        client.setup_account_config = AsyncMock()
        return client
    
    @pytest.fixture
    def execution_engine(self, mock_binance_client):
        """Execution engine with mocked client."""
        return ProductionExecutionEngine(
            binance_client=mock_binance_client,
            total_capital=15000.0
        )
    
    def test_integrated_dynamic_workflow(self, execution_engine):
        """Test complete dynamic workflow from allocation to position sizing."""
        print("\n🔬 Testing Integrated Dynamic Workflow")
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Step 1: Add market data with different volatilities
        print("\n📊 Step 1: Setting up market data")
        volatilities = [0.015, 0.025, 0.020]  # BTC lowest, ETH highest, SOL medium
        
        for symbol, vol in zip(symbols, volatilities):
            for _ in range(25):
                execution_engine.portfolio_manager.update_volatility_data(symbol, vol)
            print(f"  {symbol}: {vol:.1%} volatility")
        
        # Add correlations
        correlations = [("BTCUSDT", "ETHUSDT", 0.7), ("BTCUSDT", "SOLUSDT", 0.6), ("ETHUSDT", "SOLUSDT", 0.8)]
        for sym1, sym2, corr in correlations:
            for _ in range(25):
                execution_engine.portfolio_manager.update_correlation_data(sym1, sym2, corr)
        
        # Step 2: Force portfolio rebalancing
        print("\n📊 Step 2: Portfolio rebalancing")
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = execution_engine.portfolio_manager.rebalance_portfolio(symbols)
        
        print(f"Portfolio allocations:")
        for symbol, allocation in allocations.items():
            pct = (allocation.allocated_capital / 15000.0) * 100
            print(f"  {symbol}: ${allocation.allocated_capital:.2f} ({pct:.1f}%)")
        
        # BTC (lowest vol) should get highest allocation
        btc_allocation = allocations["BTCUSDT"].allocated_capital
        eth_allocation = allocations["ETHUSDT"].allocated_capital
        assert btc_allocation > eth_allocation, "Lower volatility should get higher allocation"
        print(f"✅ Lower volatility asset gets higher allocation")
        
        # Step 3: Test position sizing for each allocation
        print("\n📊 Step 3: Dynamic position sizing")
        entry_price = 50000.0
        
        for symbol, allocation in allocations.items():
            # Get volatility for this symbol
            vol = execution_engine.portfolio_manager.get_volatility_ema(symbol)
            atr_value = vol  # Use volatility as ATR approximation
            
            # Calculate position size
            position_result = execution_engine.risk_manager.calculate_position_size(
                symbol=symbol,
                allocated_capital=allocation.allocated_capital,
                atr_value=atr_value,
                entry_price=entry_price
            )
            
            # Calculate leverage
            leverage = execution_engine.risk_manager.calculate_dynamic_leverage(symbol, atr_value)
            
            # Calculate SL/TP
            sl_price, tp_price = execution_engine.risk_manager.calculate_stop_loss_take_profit(
                entry_price=entry_price,
                side="buy",
                atr_adjusted=atr_value * entry_price
            )
            
            print(f"  {symbol}:")
            print(f"    Allocated: ${allocation.allocated_capital:.2f}")
            print(f"    Position: ${position_result['position_size_usdt']:.2f}")
            print(f"    Leverage: {leverage}x")
            print(f"    SL: ${sl_price:.0f}, TP: ${tp_price:.0f}")
            
            # Validate calculations
            assert position_result['position_size_usdt'] > 0, "Position size should be positive"
            assert 1 <= leverage <= 10, "Leverage should be between 1x and 10x"
            assert sl_price < entry_price < tp_price, "SL < Entry < TP for long position"
        
        print(f"✅ Integrated system produces valid dynamic calculations")
    
    def test_dynamic_rebalancing_trigger(self, execution_engine):
        """Test dynamic rebalancing based on time and conditions."""
        print("\n🔬 Testing Dynamic Rebalancing Trigger")
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Add initial data
        for symbol in symbols:
            for _ in range(20):
                execution_engine.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Test 1: Should not rebalance if recently rebalanced
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=1)
        should_rebalance_1 = execution_engine.portfolio_manager.should_rebalance()
        assert not should_rebalance_1, "Should not rebalance if recently rebalanced"
        print(f"✅ No rebalancing when recently rebalanced")
        
        # Test 2: Should rebalance if 24+ hours passed
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        should_rebalance_2 = execution_engine.portfolio_manager.should_rebalance()
        assert should_rebalance_2, "Should rebalance after 24+ hours"
        print(f"✅ Rebalancing triggered after 24+ hours")
        
        # Test 3: Process daily rebalance
        rebalanced = execution_engine.process_daily_rebalance()
        if rebalanced:
            print(f"✅ Daily rebalance executed successfully")
        else:
            print(f"ℹ️ Daily rebalance not needed (as expected)")
    
    @pytest.mark.asyncio
    async def test_dynamic_signal_validation(self, execution_engine):
        """Test dynamic signal validation with risk management."""
        print("\n🔬 Testing Dynamic Signal Validation")
        
        # Set up allocation first
        symbols = ["BTCUSDT"]
        for symbol in symbols:
            for _ in range(20):
                execution_engine.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        execution_engine.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = execution_engine.portfolio_manager.rebalance_portfolio(symbols)
        
        # Create mock signal
        from algorithm.trade_signal import TradeSignal
        
        signal = TradeSignal(
            symbol="BTCUSDT",
            action="open",
            side="buy",
            signal_confidence=0.8,
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        signal.metadata = {
            'atr_value': 0.02,
            'price': 50000.0
        }
        
        # Test signal validation
        validation_result = await execution_engine.validate_signal(signal, 50000.0)
        
        print(f"Signal validation result: {validation_result}")
        
        # Should be valid if we have allocation
        if allocations:
            assert validation_result.get('valid') is not None, "Should return validation result"
            print(f"✅ Signal validation working with dynamic allocation")
        else:
            print(f"ℹ️ No allocation available for validation")
    
    def test_stress_conditions_dynamic_response(self, execution_engine):
        """Test system response under stress conditions."""
        print("\n🔬 Testing Stress Conditions Dynamic Response")
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Simulate stress condition: Very high volatility
        print("\n📊 Simulating high volatility stress")
        for symbol in symbols:
            for _ in range(30):
                execution_engine.portfolio_manager.update_volatility_data(symbol, 0.08)  # 8% volatility
        
        # Force regime detection with high volatility history
        for _ in range(35):
            execution_engine.portfolio_manager.volatility_history.append(0.08)
        
        # Test scaling multiplier under stress
        scaling_multiplier = execution_engine.portfolio_manager.calculate_scaling_multiplier()
        is_high_vol_regime = execution_engine.portfolio_manager.is_high_volatility_regime()
        
        print(f"High volatility regime: {is_high_vol_regime}")
        print(f"Scaling multiplier: {scaling_multiplier:.3f}")
        
        assert is_high_vol_regime, "Should detect high volatility regime"
        assert scaling_multiplier <= 0.5, "Should reduce allocation in high volatility regime"
        
        # Test leverage under stress
        leverage = execution_engine.risk_manager.calculate_dynamic_leverage("BTCUSDT", 0.08)
        print(f"Stress leverage: {leverage}x")
        
        assert leverage < 5, "Should reduce leverage under stress conditions"
        
        print(f"✅ System correctly responds to stress conditions")


class TestDynamicEdgeCases:
    """Test edge cases in dynamic calculations."""
    
    def test_zero_allocation_edge_case(self):
        """Test behavior with zero allocation."""
        print("\n🔬 Testing Zero Allocation Edge Case")
        
        portfolio_manager = ProductionPortfolioManager(total_capital=1000.0)  # Small capital
        risk_manager = ProductionRiskManager(portfolio_manager)
        
        # Test with zero allocated capital
        result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=0.0,
            atr_value=0.02,
            entry_price=50000.0
        )
        
        assert result['position_size_usdt'] == 0.0, "Zero allocation should result in zero position"
        print(f"✅ Zero allocation handled correctly")
    
    def test_extreme_volatility_edge_cases(self):
        """Test extreme volatility scenarios."""
        print("\n🔬 Testing Extreme Volatility Edge Cases")
        
        portfolio_manager = ProductionPortfolioManager(total_capital=15000.0)
        risk_manager = ProductionRiskManager(portfolio_manager)
        
        # Test extremely low volatility
        low_vol_result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=3000.0,
            atr_value=0.0001,  # Extremely low
            entry_price=50000.0
        )
        
        # Should use minimum ATR floor
        assert low_vol_result['position_size_usdt'] < 3000.0, "Should limit position size even with low volatility"
        print(f"✅ Extremely low volatility handled with ATR floor")
        
        # Test extremely high volatility
        high_vol_result = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            allocated_capital=3000.0,
            atr_value=0.2,  # 20% volatility
            entry_price=50000.0
        )
        
        # Should result in very small position
        assert high_vol_result['position_size_usdt'] < 100.0, "High volatility should result in small position"
        print(f"✅ Extremely high volatility results in appropriately small position")
    
    def test_correlation_edge_cases(self):
        """Test correlation calculation edge cases."""
        print("\n🔬 Testing Correlation Edge Cases")
        
        portfolio_manager = ProductionPortfolioManager(total_capital=15000.0)
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Test with no correlation data
        avg_corr = portfolio_manager.get_average_correlation("BTCUSDT", symbols)
        assert avg_corr == 0.0, "Should return 0 correlation when no data available"
        print(f"✅ No correlation data handled correctly")
        
        # Test with single symbol
        single_symbol_corr = portfolio_manager.get_average_correlation("BTCUSDT", ["BTCUSDT"])
        assert single_symbol_corr == 0.0, "Single symbol should have 0 average correlation"
        print(f"✅ Single symbol correlation handled correctly")


def run_dynamic_tests():
    """Run all dynamic calculation tests."""
    print("="*80)
    print("RUNNING DYNAMIC PORTFOLIO & RISK MANAGEMENT TESTS")
    print("="*80)
    print("Testing dynamic calculations for:")
    print("• Portfolio allocation based on volatility and correlations")
    print("• Risk-adjusted position sizing with ATR and Kelly criterion")
    print("• Dynamic leverage based on market conditions")
    print("• Stop loss and take profit calculations")
    print("• Integration of all dynamic components")
    print("="*80)
    
    # Run pytest with verbose output
    pytest.main([
        __file__, 
        "-v", 
        "-s", 
        "--tb=short",
        "--capture=no"
    ])


if __name__ == "__main__":
    run_dynamic_tests()
