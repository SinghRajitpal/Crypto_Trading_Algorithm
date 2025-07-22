#!/usr/bin/env python3

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from execution.portfolio import ProductionPortfolioManager, AllocationWeights

def test_basic():
    # Test basic initialization
    portfolio = ProductionPortfolioManager(total_capital=10000.0)
    
    assert portfolio.total_capital == 10000.0
    assert portfolio.target_volatility == 0.18
    assert portfolio.max_allocation_pct == 0.85
    
    print("✅ Basic initialization test passed")
    
    # Test volatility data update
    portfolio.update_volatility_data("BTCUSDT", 0.002)
    assert "BTCUSDT" in portfolio.volatility_data
    assert len(portfolio.volatility_data["BTCUSDT"]) == 1
    
    print("✅ Volatility data update test passed")
    
    # Test correlation data update
    portfolio.update_correlation_data("BTCUSDT", "ETHUSDT", 0.7)
    pair = ("BTCUSDT", "ETHUSDT")
    assert pair in portfolio.correlation_data
    
    print("✅ Correlation data update test passed")
    
    # Test weight computation
    symbols = ["BTCUSDT"]
    weights = portfolio.compute_weights(symbols)
    assert len(weights) == 1
    assert "BTCUSDT" in weights
    assert weights["BTCUSDT"] == 1.0  # Single symbol should get 100%
    
    print("✅ Weight computation test passed")
    
    print("🎉 All basic tests passed!")

if __name__ == "__main__":
    test_basic()
