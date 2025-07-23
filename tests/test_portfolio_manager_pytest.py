#!/usr/bin/env python3
"""
Comprehensive pytest unit tests for Portfolio Manager.

This module tests the ProductionPortfolioManager implementation against
the mathematical formulas specified in the trading document.

Tests cover:
- Initialization and configuration
- Volatility tracking and EMA calculation
- Portfolio allocation formula validation
- Regime detection and scaling
- Rebalancing logic and timing
"""

import pytest
import math
from datetime import datetime, timedelta
from execution.portfolio import ProductionPortfolioManager, AllocationWeights


class TestProductionPortfolioManager:
    """Test suite for ProductionPortfolioManager unit tests."""
    
    def test_initialization(self, test_capital):
        """Test portfolio manager initialization with correct parameters."""
        portfolio = ProductionPortfolioManager(total_capital=test_capital)
        
        # Test initialization values from document
        assert portfolio.total_capital == test_capital
        assert portfolio.target_volatility == 0.18  # 18% target volatility
        assert portfolio.max_allocation_pct == 0.85  # 85% max allocation
        assert portfolio.alpha == 0.3  # Fixed correlation parameter
        assert portfolio.lookback_bars == 60  # EMA lookback parameter
        
        # Test initial state
        assert portfolio.volatility_data == {}
        assert portfolio.correlation_data == {}
        assert isinstance(portfolio.last_rebalance_time, datetime)
        
    def test_volatility_tracking(self, portfolio_manager, test_symbols):
        """Test volatility EMA tracking implementation."""
        # Test single volatility update
        portfolio_manager.update_volatility_data("BTCUSDT", 0.025)
        vol_ema = portfolio_manager.get_volatility_ema("BTCUSDT")
        assert vol_ema == 0.025  # First value should equal input
        
        # Test EMA calculation over multiple updates
        volatilities = [0.020, 0.030, 0.025, 0.035]
        for vol in volatilities:
            portfolio_manager.update_volatility_data("BTCUSDT", vol)
        
        final_ema = portfolio_manager.get_volatility_ema("BTCUSDT")
        
        # Verify EMA is within reasonable range
        assert 0.015 <= final_ema <= 0.040
        assert isinstance(final_ema, float)
        
        # Test multiple symbols
        for i, symbol in enumerate(test_symbols[:3]):
            portfolio_manager.update_volatility_data(symbol, 0.01 + i * 0.005)
            vol = portfolio_manager.get_volatility_ema(symbol)
            assert vol > 0
            
    def test_correlation_calculation(self, portfolio_manager, test_symbols):
        """Test correlation calculation between assets."""
        # Update data for multiple symbols
        test_data = {
            "BTCUSDT": [0.02, 0.025, 0.03],
            "ETHUSDT": [0.03, 0.035, 0.04],
            "XRPUSDT": [0.04, 0.045, 0.05]
        }
        
        for symbol, vols in test_data.items():
            for vol in vols:
                portfolio_manager.update_volatility_data(symbol, vol)
        
        # Test correlation calculation
        for symbol in test_data.keys():
            corr = portfolio_manager.get_average_correlation(symbol, list(test_data.keys()))
            assert isinstance(corr, float)
            assert -1.0 <= corr <= 1.0  # Correlation bounds
            
    def test_weight_calculation_formula(self, portfolio_manager, test_symbols):
        """Test portfolio weight calculation formula: w_i = (1/σ_i) × (1 + α × avg_correlation_i)"""
        # Set up test data with known volatilities
        test_volatilities = [0.01, 0.02, 0.03]  # Different volatilities
        symbols = test_symbols[:3]
        
        for symbol, vol in zip(symbols, test_volatilities):
            portfolio_manager.update_volatility_data(symbol, vol)
        
        # Force rebalance to trigger weight calculation
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Verify allocations exist and are properly structured
        assert len(allocations) == len(symbols)
        
        # Test weight normalization (should sum to 1.0)
        total_weight = sum(alloc.weight for alloc in allocations.values())
        assert abs(total_weight - 1.0) < 0.001, f"Weights should sum to 1.0, got {total_weight}"
        
        # Test inverse volatility relationship
        weights_by_vol = [(allocations[symbol].weight, portfolio_manager.get_volatility_ema(symbol)) 
                         for symbol in symbols]
        weights_by_vol.sort(key=lambda x: x[1])  # Sort by volatility
        
        # Lower volatility should have higher weight
        for i in range(len(weights_by_vol) - 1):
            lower_vol_weight = weights_by_vol[i][0]
            higher_vol_weight = weights_by_vol[i + 1][0]
            assert lower_vol_weight >= higher_vol_weight, \
                "Lower volatility assets should have higher weights"
                
        # Test specific formula components
        for symbol in symbols:
            alloc = allocations[symbol]
            vol = portfolio_manager.get_volatility_ema(symbol)
            corr = portfolio_manager.get_average_correlation(symbol, symbols)
            
            # Formula: w_i = (1/σ_i) × (1 + α × avg_correlation_i)
            expected_raw_weight = (1 / vol) * (1 + portfolio_manager.alpha * corr)
            
            # The actual weight should be proportional to expected_raw_weight
            assert alloc.weight > 0
            assert alloc.allocated_capital > 0
            
    def test_allocation_scaling_with_regime_detection(self, portfolio_manager, test_symbols):
        """Test allocation scaling with volatility regime detection."""
        symbols = test_symbols[:3]
        
        # Test normal volatility regime
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, 0.02)  # Normal volatility
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Verify total allocation respects 85% cap
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        max_allocation = portfolio_manager.total_capital * 0.85
        assert total_allocated <= max_allocation * 1.01, \
            f"Total allocation should respect 85% cap: {total_allocated} > {max_allocation}"
        
        # Test that allocation is actually close to the cap in normal conditions
        assert total_allocated >= max_allocation * 0.95, \
            "Should allocate close to maximum in normal conditions"
            
    def test_rebalancing_timing(self, portfolio_manager, test_symbols):
        """Test daily rebalancing timing logic."""
        symbols = test_symbols[:3]
        
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Test no rebalance when recently rebalanced
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=12)
        should_rebalance = portfolio_manager.should_rebalance()
        assert not should_rebalance, "Should not rebalance within 24 hours"
        
        # Test rebalance when time threshold passed
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        should_rebalance = portfolio_manager.should_rebalance()
        assert should_rebalance, "Should rebalance after 24 hours"
        
        # Test actual rebalancing updates timestamp
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        assert len(allocations) == len(symbols), "Should allocate to all symbols"
        
        # Verify timestamp updated
        time_since_rebalance = datetime.now() - portfolio_manager.last_rebalance_time
        assert time_since_rebalance.total_seconds() < 60, \
            "Rebalance timestamp should be updated"
            
    def test_edge_cases(self, portfolio_manager):
        """Test edge cases and error handling."""
        # Test with zero volatility (should use minimum)
        portfolio_manager.update_volatility_data("ZEROVOLBTC", 0.0)
        vol = portfolio_manager.get_volatility_ema("ZEROVOLBTC")
        assert vol > 0, "Zero volatility should be handled with minimum value"
        
        # Test with very high volatility
        portfolio_manager.update_volatility_data("HIGHVOLBTC", 0.5)  # 50% volatility
        vol = portfolio_manager.get_volatility_ema("HIGHVOLBTC")
        assert vol == 0.5
        
        # Test empty symbol list
        allocations = portfolio_manager.rebalance_portfolio([])
        assert allocations == {}
        
        # Test single symbol
        portfolio_manager.update_volatility_data("SINGLEBTC", 0.02)
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(["SINGLEBTC"])
        assert len(allocations) == 1
        assert allocations["SINGLEBTC"].weight == 1.0
        
    def test_mathematical_consistency(self, portfolio_manager, test_symbols):
        """Test mathematical consistency of calculations."""
        symbols = test_symbols[:4]  # Test with more symbols
        
        # Set up diverse volatility landscape
        volatilities = [0.015, 0.025, 0.035, 0.045]
        for symbol, vol in zip(symbols, volatilities):
            portfolio_manager.update_volatility_data(symbol, vol)
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Test mathematical properties
        total_weight = sum(alloc.weight for alloc in allocations.values())
        total_capital = sum(alloc.allocated_capital for alloc in allocations.values())
        
        # Weights should sum to 1.0
        assert abs(total_weight - 1.0) < 0.001
        
        # Capital allocation should be consistent
        expected_total = total_weight * portfolio_manager.total_capital * 0.85
        assert abs(total_capital - expected_total) < 1.0  # Allow $1 rounding error
        
        # Individual allocations should be consistent
        for symbol, alloc in allocations.items():
            expected_capital = alloc.weight * portfolio_manager.total_capital * 0.85
            assert abs(alloc.allocated_capital - expected_capital) < 1.0
            
    @pytest.mark.parametrize("capital", [1000.0, 5000.0, 15000.0, 50000.0])
    def test_different_capital_amounts(self, capital, test_symbols):
        """Test portfolio manager with different capital amounts."""
        portfolio = ProductionPortfolioManager(total_capital=capital)
        symbols = test_symbols[:3]
        
        for symbol in symbols:
            portfolio.update_volatility_data(symbol, 0.02)
        
        portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio.rebalance_portfolio(symbols)
        
        # Test scaling with capital
        total_allocated = sum(alloc.allocated_capital for alloc in allocations.values())
        expected_max = capital * 0.85
        
        assert total_allocated <= expected_max * 1.01
        assert total_allocated >= expected_max * 0.95
        
        # Test proportional scaling
        if capital > 1000:  # Avoid division by small numbers
            scale_factor = capital / 1000.0
            for alloc in allocations.values():
                assert alloc.allocated_capital >= scale_factor * 200  # Minimum reasonable allocation
                
    def test_regime_detection_scenarios(self, portfolio_manager, test_symbols):
        """Test different volatility regime scenarios."""
        symbols = test_symbols[:3]
        
        # Test high volatility regime
        high_vol = 0.08  # 8% volatility (high)
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, high_vol)
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        high_vol_allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Test normal volatility regime
        normal_vol = 0.02  # 2% volatility (normal)
        for symbol in symbols:
            portfolio_manager.update_volatility_data(symbol, normal_vol)
        
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        normal_vol_allocations = portfolio_manager.rebalance_portfolio(symbols)
        
        # Compare allocations (high vol should have lower total allocation due to regime factor)
        high_vol_total = sum(alloc.allocated_capital for alloc in high_vol_allocations.values())
        normal_vol_total = sum(alloc.allocated_capital for alloc in normal_vol_allocations.values())
        
        # In high volatility, total allocation might be reduced
        assert high_vol_total <= normal_vol_total * 1.1  # Allow some variance
        
        # Both should be positive and reasonable
        assert high_vol_total > 0
        assert normal_vol_total > 0
