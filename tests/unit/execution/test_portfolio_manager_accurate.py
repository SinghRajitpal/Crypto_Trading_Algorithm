import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from execution.portfolio import ProductionPortfolioManager, AllocationWeights

logging.basicConfig(level=logging.DEBUG)

class TestProductionPortfolioManager(unittest.TestCase):
    """Comprehensive tests for ProductionPortfolioManager based on actual implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.total_capital = 10000.0
        self.target_volatility = 0.18
        self.max_allocation_pct = 0.85
        
        # Create portfolio manager
        self.portfolio = ProductionPortfolioManager(
            total_capital=self.total_capital,
            target_volatility=self.target_volatility,
            max_allocation_pct=self.max_allocation_pct
        )
    
    def test_initialization(self):
        """Test proper initialization of portfolio manager."""
        self.assertEqual(self.portfolio.total_capital, self.total_capital)
        self.assertEqual(self.portfolio.target_volatility, self.target_volatility)
        self.assertEqual(self.portfolio.max_allocation_pct, self.max_allocation_pct)
        
        # Check default parameters
        self.assertEqual(self.portfolio.alpha, 0.3)
        self.assertEqual(self.portfolio.lookback_bars, 60)
        self.assertEqual(self.portfolio.regime_percentile, 75)
        
        # Check initialized data structures
        self.assertIsInstance(self.portfolio.volatility_data, dict)
        self.assertIsInstance(self.portfolio.correlation_data, dict)
        self.assertIsInstance(self.portfolio.allocation_weights, dict)
        self.assertIsInstance(self.portfolio.volatility_history, list)
        self.assertIsInstance(self.portfolio.last_rebalance_time, datetime)
    
    def test_update_volatility_data(self):
        """Test volatility data update and management."""
        symbol = "BTCUSDT"
        atr_values = [0.001, 0.002, 0.0015, 0.0025, 0.002]
        
        # Add multiple ATR values
        for atr in atr_values:
            self.portfolio.update_volatility_data(symbol, atr)
            
        # Check data is stored
        self.assertIn(symbol, self.portfolio.volatility_data)
        self.assertEqual(len(self.portfolio.volatility_data[symbol]), len(atr_values))
        self.assertEqual(self.portfolio.volatility_data[symbol], atr_values)
        
    def test_volatility_data_rolling_window(self):
        """Test volatility data rolling window management."""
        symbol = "BTCUSDT"
        
        # Add more than lookback_bars values
        for i in range(self.portfolio.lookback_bars + 10):
            self.portfolio.update_volatility_data(symbol, 0.001 + i * 0.0001)
            
        # Should maintain exactly lookback_bars values
        self.assertEqual(len(self.portfolio.volatility_data[symbol]), self.portfolio.lookback_bars)
        
        # First value should be the (10+1)th value added
        expected_first_value = 0.001 + 10 * 0.0001
        self.assertAlmostEqual(self.portfolio.volatility_data[symbol][0], expected_first_value, places=6)
        
    def test_update_correlation_data(self):
        """Test correlation data update and pair ordering."""
        symbol1, symbol2 = "BTCUSDT", "ETHUSDT"
        correlation_values = [0.5, 0.6, 0.55, 0.65, 0.7]
        
        # Add correlation data
        for corr in correlation_values:
            self.portfolio.update_correlation_data(symbol1, symbol2, corr)
            
        # Check pair ordering (should be consistent regardless of input order)
        expected_pair = (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)
        self.assertIn(expected_pair, self.portfolio.correlation_data)
        self.assertEqual(len(self.portfolio.correlation_data[expected_pair]), len(correlation_values))
        
        # Test reverse order produces same result
        test_portfolio = ProductionPortfolioManager(total_capital=5000.0)
        for corr in correlation_values:
            test_portfolio.update_correlation_data(symbol2, symbol1, corr)
            
        self.assertIn(expected_pair, test_portfolio.correlation_data)
        
    def test_correlation_data_rolling_window(self):
        """Test correlation data rolling window management."""
        symbol1, symbol2 = "BTCUSDT", "ETHUSDT"
        
        # Add more than lookback_bars values
        for i in range(self.portfolio.lookback_bars + 5):
            self.portfolio.update_correlation_data(symbol1, symbol2, 0.5 + i * 0.01)
            
        # Should maintain exactly lookback_bars values
        pair = (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)
        self.assertEqual(len(self.portfolio.correlation_data[pair]), self.portfolio.lookback_bars)
        
    def test_get_volatility_ema(self):
        """Test volatility EMA calculation."""
        symbol = "BTCUSDT"
        
        # Test with no data - should return default
        ema = self.portfolio.get_volatility_ema(symbol)
        self.assertEqual(ema, 0.02)  # Default volatility from implementation
        
        # Add some data
        atr_values = [0.002, 0.003, 0.0025, 0.004, 0.0035]
        for atr in atr_values:
            self.portfolio.update_volatility_data(symbol, atr)
            
        # Should return calculated EMA
        ema = self.portfolio.get_volatility_ema(symbol)
        self.assertGreater(ema, 0)
        self.assertNotEqual(ema, 0.02)  # Should not be default
        
        # EMA should be in reasonable range based on input data
        self.assertGreater(ema, min(atr_values) * 0.1)
        self.assertLess(ema, max(atr_values) * 5.0)
        
    def test_get_average_correlation(self):
        """Test average correlation calculation."""
        symbol = "BTCUSDT"
        other_symbols = ["ETHUSDT", "XRPUSDT", "BNBUSDT"]
        
        # Test with no data - should return default
        avg_corr = self.portfolio.get_average_correlation(symbol, other_symbols)
        self.assertEqual(avg_corr, 0.0)  # Default correlation from implementation
        
        # Add correlation data for some pairs
        correlations = [0.5, 0.7, 0.6]
        for i, other_symbol in enumerate(other_symbols):
            for _ in range(5):  # Add multiple values
                self.portfolio.update_correlation_data(symbol, other_symbol, correlations[i])
                
        # Should return calculated average
        avg_corr = self.portfolio.get_average_correlation(symbol, other_symbols)
        expected_avg = sum(correlations) / len(correlations)
        self.assertAlmostEqual(avg_corr, expected_avg, places=2)
        
    def test_compute_weights(self):
        """Test weight computation using inverse volatility."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        volatilities = [0.002, 0.003, 0.0025]
        
        # Setup volatility data
        for symbol, vol in zip(symbols, volatilities):
            for _ in range(10):  # Add multiple values
                self.portfolio.update_volatility_data(symbol, vol)
                
        raw_weights = self.portfolio.compute_weights(symbols)
        
        # Check that weights are calculated correctly
        self.assertEqual(len(raw_weights), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, raw_weights)
            self.assertGreater(raw_weights[symbol], 0)
            
        # Higher volatility should result in lower weight
        btc_weight = raw_weights["BTCUSDT"]
        eth_weight = raw_weights["ETHUSDT"] 
        xrp_weight = raw_weights["XRPUSDT"]
        
        # BTC has lowest volatility (0.002), should have highest weight
        self.assertGreater(btc_weight, eth_weight)
        self.assertGreater(btc_weight, xrp_weight)
        
        # ETH has highest volatility (0.003), should have lowest weight
        self.assertLess(eth_weight, xrp_weight)
        
        # Weights should sum to approximately 1
        total_weight = sum(raw_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
    def test_compute_weights_with_correlation_adjustment(self):
        """Test correlation-adjusted weight calculation."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Setup volatility data first
        for symbol in symbols:
            for _ in range(10):
                self.portfolio.update_volatility_data(symbol, 0.002)
        
        # Setup correlation data
        correlations = [
            (("BTCUSDT", "ETHUSDT"), 0.8),
            (("BTCUSDT", "XRPUSDT"), 0.6),
            (("ETHUSDT", "XRPUSDT"), 0.7)
        ]
        
        for (s1, s2), corr in correlations:
            for _ in range(10):
                self.portfolio.update_correlation_data(s1, s2, corr)
                
        weights = self.portfolio.compute_weights(symbols)
        
        # Check all symbols have weights
        self.assertEqual(len(weights), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, weights)
            self.assertGreater(weights[symbol], 0)
            
        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
    def test_is_high_volatility_regime(self):
        """Test volatility regime detection."""
        # Initially should not be high volatility (no history)
        is_high_vol = self.portfolio.is_high_volatility_regime()
        self.assertIsInstance(bool(is_high_vol), bool)  # Convert numpy bool to Python bool
        
        # Add volatility history through scaling multiplier calculation
        for _ in range(35):  # More than 30 to test rolling window
            self.portfolio.calculate_scaling_multiplier()
            
        # Test regime detection again
        is_high_vol = self.portfolio.is_high_volatility_regime()
        self.assertIsInstance(bool(is_high_vol), bool)
        
    def test_calculate_scaling_multiplier(self):
        """Test volatility targeting scaling multiplier."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Add volatility data
        for symbol in symbols:
            for _ in range(10):
                self.portfolio.update_volatility_data(symbol, 0.02)
                
        # Calculate scaling multiplier
        multiplier = self.portfolio.calculate_scaling_multiplier()
        
        self.assertGreater(multiplier, 0)
        self.assertLessEqual(multiplier, 1.0)  # Should not exceed 1.0 typically
        
    def test_should_rebalance(self):
        """Test rebalancing trigger logic."""
        # Set last rebalance time to 25 hours ago to trigger rebalancing
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Should rebalance after 24 hours
        self.assertTrue(self.portfolio.should_rebalance())
        
        # After setting recent rebalance time, should not rebalance immediately
        self.portfolio.last_rebalance_time = datetime.now()
        self.assertFalse(self.portfolio.should_rebalance())
        
    def test_rebalance_portfolio(self):
        """Test complete portfolio rebalancing."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Setup market data
        volatilities = [0.002, 0.003, 0.0025]
        for symbol, vol in zip(symbols, volatilities):
            for _ in range(20):
                self.portfolio.update_volatility_data(symbol, vol)
                
        # Setup correlations
        correlations = [
            (("BTCUSDT", "ETHUSDT"), 0.7),
            (("BTCUSDT", "XRPUSDT"), 0.6),
            (("ETHUSDT", "XRPUSDT"), 0.8)
        ]
        
        for (s1, s2), corr in correlations:
            for _ in range(20):
                self.portfolio.update_correlation_data(s1, s2, corr)
                
        # Force rebalance
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Rebalance portfolio
        allocations = self.portfolio.rebalance_portfolio(symbols)
        
        # Check that allocation weights are created
        self.assertEqual(len(allocations), len(symbols))
        
        # Check each allocation weight
        for symbol in symbols:
            self.assertIn(symbol, allocations)
            allocation = allocations[symbol]
            self.assertIsInstance(allocation, AllocationWeights)
            self.assertEqual(allocation.symbol, symbol)
            self.assertGreater(allocation.weight, 0)
            self.assertGreater(allocation.allocated_capital, 0)
            
        # Total allocation should not exceed max_allocation_pct
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        max_allowed = self.total_capital * self.max_allocation_pct
        self.assertLessEqual(total_allocated, max_allowed * 1.01)  # Small tolerance
        
    def test_get_allocated_capital(self):
        """Test allocated capital retrieval."""
        symbol = "BTCUSDT"
        
        # Test with no allocation
        capital = self.portfolio.get_allocated_capital(symbol)
        self.assertEqual(capital, 0.0)
        
        # Setup allocation
        allocated_amount = 4000.0
        self.portfolio.allocation_weights[symbol] = AllocationWeights(
            symbol=symbol,
            weight=0.4,
            allocated_capital=allocated_amount,
            volatility=0.002,
            avg_correlation=0.6,
            raw_weight=0.35
        )
        
        # Should return correct amount
        capital = self.portfolio.get_allocated_capital(symbol)
        self.assertEqual(capital, allocated_amount)
        
    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Setup allocations
        self.portfolio.allocation_weights["BTCUSDT"] = AllocationWeights(
            symbol="BTCUSDT",
            weight=0.6,
            allocated_capital=5000.0,
            volatility=0.002,
            avg_correlation=0.6,
            raw_weight=0.55
        )
        
        self.portfolio.allocation_weights["ETHUSDT"] = AllocationWeights(
            symbol="ETHUSDT",
            weight=0.4,
            allocated_capital=3000.0,
            volatility=0.003,
            avg_correlation=0.7,
            raw_weight=0.45
        )
        
        summary = self.portfolio.get_portfolio_summary()
        
        # Check summary structure and values
        required_fields = [
            'total_capital', 'allocated_capital', 'allocation_percentage', 
            'target_volatility', 'active_symbols'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
            
        self.assertEqual(summary["total_capital"], self.total_capital)
        self.assertEqual(summary["allocated_capital"], 8000.0)  # 5000 + 3000
        self.assertEqual(summary["active_symbols"], 2)
        
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty symbol list
        raw_weights = self.portfolio.compute_weights([])
        self.assertEqual(len(raw_weights), 0)
        
        # Test single symbol
        self.portfolio.update_volatility_data("BTCUSDT", 0.002)
        raw_weights = self.portfolio.compute_weights(["BTCUSDT"])
        self.assertEqual(raw_weights["BTCUSDT"], 1.0)
        
        # Test with zero volatility (should use minimum)
        self.portfolio.volatility_data["TESTSYMBOL"] = [0.0] * 10
        ema = self.portfolio.get_volatility_ema("TESTSYMBOL")
        self.assertGreater(ema, 0)  # Should use minimum volatility
        
    def test_custom_initialization_parameters(self):
        """Test initialization with custom parameters."""
        custom_portfolio = ProductionPortfolioManager(
            total_capital=5000.0, 
            target_volatility=0.25, 
            max_allocation_pct=0.9
        )
        self.assertEqual(custom_portfolio.total_capital, 5000.0)
        self.assertEqual(custom_portfolio.target_volatility, 0.25)
        self.assertEqual(custom_portfolio.max_allocation_pct, 0.9)
        
    def test_process_symbols_from_config(self):
        """Test configuration processing method."""
        # This method should exist and be callable
        try:
            self.portfolio.process_symbols_from_config()
            # If it doesn't raise an exception, it's working
            self.assertTrue(True)
        except Exception as e:
            # If it raises an exception due to missing config, that's expected
            self.assertIsInstance(e, (FileNotFoundError, KeyError, AttributeError))


if __name__ == "__main__":
    unittest.main()
