#!/usr/bin/env python3

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from execution.portfolio import ProductionPortfolioManager, AllocationWeights

def test_portfolio_comprehensive():
    """Run comprehensive portfolio tests."""
    
    print("🧪 Starting comprehensive portfolio tests...")
    
    # Test 1: Initialization
    portfolio = ProductionPortfolioManager(total_capital=10000.0)
    assert portfolio.total_capital == 10000.0
    assert portfolio.target_volatility == 0.18
    assert portfolio.max_allocation_pct == 0.85
    print("✅ Test 1: Initialization passed")
    
    # Test 2: Volatility data management
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    volatilities = [0.002, 0.003, 0.0025]
    
    for symbol, vol in zip(symbols, volatilities):
        for _ in range(10):  # Add multiple values
            portfolio.update_volatility_data(symbol, vol)
    
    # Verify data is stored
    for symbol in symbols:
        assert symbol in portfolio.volatility_data
        assert len(portfolio.volatility_data[symbol]) == 10
    print("✅ Test 2: Volatility data management passed")
    
    # Test 3: Correlation data management
    correlations = [
        (("BTCUSDT", "ETHUSDT"), 0.8),
        (("BTCUSDT", "XRPUSDT"), 0.6),
        (("ETHUSDT", "XRPUSDT"), 0.7)
    ]
    
    for (s1, s2), corr in correlations:
        for _ in range(10):
            portfolio.update_correlation_data(s1, s2, corr)
    
    # Check pair ordering and storage
    for (s1, s2), corr in correlations:
        expected_pair = (s1, s2) if s1 < s2 else (s2, s1)
        assert expected_pair in portfolio.correlation_data
        assert len(portfolio.correlation_data[expected_pair]) == 10
    print("✅ Test 3: Correlation data management passed")
    
    # Test 4: Volatility EMA calculation
    btc_ema = portfolio.get_volatility_ema("BTCUSDT")
    eth_ema = portfolio.get_volatility_ema("ETHUSDT")
    xrp_ema = portfolio.get_volatility_ema("XRPUSDT")
    
    assert btc_ema > 0
    assert eth_ema > 0
    assert xrp_ema > 0
    
    # BTC has lowest volatility, should have lowest EMA
    assert btc_ema < eth_ema  # 0.002 < 0.003
    assert btc_ema < xrp_ema  # 0.002 < 0.0025
    print("✅ Test 4: Volatility EMA calculation passed")
    
    # Test 5: Average correlation calculation
    btc_avg_corr = portfolio.get_average_correlation("BTCUSDT", symbols)
    eth_avg_corr = portfolio.get_average_correlation("ETHUSDT", symbols)
    
    assert btc_avg_corr > 0
    assert eth_avg_corr > 0
    print("✅ Test 5: Average correlation calculation passed")
    
    # Test 6: Weight computation
    weights = portfolio.compute_weights(symbols)
    
    assert len(weights) == len(symbols)
    for symbol in symbols:
        assert symbol in weights
        assert weights[symbol] > 0
    
    # Weights should sum to 1
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01  # Allow small floating point errors
    
    # BTC should have highest weight (lowest volatility)
    assert weights["BTCUSDT"] > weights["ETHUSDT"]
    assert weights["BTCUSDT"] > weights["XRPUSDT"]
    print("✅ Test 6: Weight computation passed")
    
    # Test 7: Portfolio rebalancing
    allocations = portfolio.rebalance_portfolio(symbols)
    
    assert len(allocations) == len(symbols)
    for symbol in symbols:
        assert symbol in allocations
        allocation = allocations[symbol]
        assert isinstance(allocation, AllocationWeights)
        assert allocation.allocated_capital > 0
        assert allocation.weight > 0
    
    # Total allocation should not exceed max allocation percentage
    total_allocated = sum(a.allocated_capital for a in allocations.values())
    max_allowed = portfolio.total_capital * portfolio.max_allocation_pct
    assert total_allocated <= max_allowed * 1.01  # Small tolerance
    print("✅ Test 7: Portfolio rebalancing passed")
    
    # Test 8: Allocation retrieval
    for symbol in symbols:
        allocated = portfolio.get_allocated_capital(symbol)
        assert allocated > 0
        assert allocated == allocations[symbol].allocated_capital
    print("✅ Test 8: Allocation retrieval passed")
    
    # Test 9: Portfolio summary
    summary = portfolio.get_portfolio_summary()
    
    required_fields = [
        'total_capital', 'allocated_capital', 'allocation_percentage', 
        'target_volatility', 'active_symbols'
    ]
    
    for field in required_fields:
        assert field in summary
    
    assert summary["total_capital"] == portfolio.total_capital
    assert summary["active_symbols"] == len(symbols)
    assert summary["allocated_capital"] > 0
    print("✅ Test 9: Portfolio summary passed")
    
    # Test 10: Rebalancing triggers
    # Initially should rebalance
    assert portfolio.should_rebalance() == False  # Just rebalanced
    
    # After 25 hours, should rebalance
    portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
    assert portfolio.should_rebalance() == True
    print("✅ Test 10: Rebalancing triggers passed")
    
    # Test 11: Regime detection and scaling
    multiplier = portfolio.calculate_scaling_multiplier()
    assert multiplier > 0
    assert multiplier <= 1.0  # Typically should not exceed 1
    
    regime = portfolio.is_high_volatility_regime()
    assert isinstance(regime, bool)
    print("✅ Test 11: Regime detection and scaling passed")
    
    print("🎉 All comprehensive portfolio tests passed!")
    print(f"📊 Final portfolio state:")
    print(f"   Total capital: ${portfolio.total_capital:,.2f}")
    print(f"   Total allocated: ${sum(a.allocated_capital for a in portfolio.allocation_weights.values()):,.2f}")
    print(f"   Active symbols: {len(portfolio.allocation_weights)}")
    print(f"   High vol regime: {portfolio.is_high_volatility_regime()}")

if __name__ == "__main__":
    test_portfolio_comprehensive()
