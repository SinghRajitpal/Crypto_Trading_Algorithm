import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from execution.portfolio import ProductionPortfolioManager, AllocationWeights

logging.basicConfig(level=logging.DEBUG)

class TestProductionPortfolioManager(unittest.TestCase):
    """Comprehensive tests for ProductionPortfolioManager."""
    
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
        
        # Check default values
        self.assertEqual(self.portfolio.alpha, 0.3)
        self.assertEqual(self.portfolio.lookback_bars, 60)
        
        # Check data structures are initialized
        self.assertIsInstance(self.portfolio.volatility_data, dict)
        self.assertIsInstance(self.portfolio.correlation_data, dict)
        self.assertIsInstance(self.portfolio.allocation_weights, dict)
        self.assertIsInstance(self.portfolio.volatility_history, list)
        self.assertIsNotNone(self.portfolio.last_rebalance_time)
    
    def test_update_volatility_data(self):
        """Test volatility data updates."""
        symbol = 'BTCUSDT'
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        # Initial update
        self.portfolio.update_volatility_data(symbol, returns)
        
        self.assertIn(symbol, self.portfolio.volatility_data)
        self.assertIsInstance(self.portfolio.volatility_data[symbol], pd.Series)
        
        # Check EMA calculation exists
        volatility_ema = self.portfolio.volatility_data[symbol]
        self.assertEqual(len(volatility_ema), len(returns))
        
        # Second update should append
        new_returns = pd.Series([0.008, -0.012])
        self.portfolio.update_volatility_data(symbol, new_returns)
        
        updated_volatility = self.portfolio.volatility_data[symbol]
        self.assertEqual(len(updated_volatility), len(returns) + len(new_returns))
    
    def test_update_correlation_data(self):
        """Test correlation data updates."""
        returns_data = {
            'BTCUSDT': pd.Series([0.01, -0.005, 0.02, -0.01, 0.015]),
            'ETHUSDT': pd.Series([0.008, -0.003, 0.018, -0.008, 0.012]),
            'SOLUSDT': pd.Series([0.012, -0.007, 0.025, -0.015, 0.020])
        }
        
        self.portfolio.update_correlation_data(returns_data)
        
        # Check correlation matrix exists
        self.assertFalse(self.portfolio.correlation_data.empty)
        
        # Check symmetric correlation matrix
        corr_matrix = self.portfolio.correlation_data
        symbols = list(returns_data.keys())
        
        for symbol in symbols:
            self.assertIn(symbol, corr_matrix.index)
            self.assertIn(symbol, corr_matrix.columns)
        
        # Diagonal should be close to 1
        for symbol in symbols:
            self.assertAlmostEqual(corr_matrix.loc[symbol, symbol], 1.0, places=2)
    
    def test_calculate_volatility_ema(self):
        """Test EMA volatility calculation."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012])
        alpha = 0.1
        
        volatility_ema = self.portfolio._calculate_volatility_ema(returns, alpha)
        
        self.assertEqual(len(volatility_ema), len(returns))
        self.assertTrue(all(vol >= 0 for vol in volatility_ema))
        
        # EMA should react to recent high volatility
        high_vol_returns = pd.Series([0.05, -0.04, 0.03, -0.06])
        high_vol_ema = self.portfolio._calculate_volatility_ema(high_vol_returns, alpha)
        
        # Last value should be relatively high
        self.assertGreater(high_vol_ema.iloc[-1], volatility_ema.iloc[-1])
    
    def test_calculate_target_weights_equal_weight(self):
        """Test equal weight calculation."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Setup volatility data
        for symbol in symbols:
            self.portfolio.volatility_data[symbol] = pd.Series([0.02] * 20)
        
        # Setup correlation data (identity matrix for simplicity)
        corr_data = pd.DataFrame(
            np.eye(len(symbols)),
            index=symbols,
            columns=symbols
        )
        self.portfolio.correlation_data = corr_data
        
        weights = self.portfolio.calculate_target_weights(symbols, method='equal_weight')
        
        expected_weight = 1.0 / len(symbols)
        for symbol in symbols:
            self.assertAlmostEqual(weights[symbol], expected_weight, places=3)
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
    
    def test_calculate_target_weights_risk_parity(self):
        """Test risk parity weight calculation."""
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Setup different volatilities
        self.portfolio.volatility_data['BTCUSDT'] = pd.Series([0.03] * 20)
        self.portfolio.volatility_data['ETHUSDT'] = pd.Series([0.02] * 20)
        
        # Setup correlation data
        corr_data = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=symbols,
            columns=symbols
        )
        self.portfolio.correlation_data = corr_data
        
        weights = self.portfolio.calculate_target_weights(symbols, method='risk_parity')
        
        # Lower volatility asset should have higher weight
        self.assertGreater(weights['ETHUSDT'], weights['BTCUSDT'])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
    
    def test_calculate_target_weights_volatility_targeting(self):
        """Test volatility targeting weight calculation."""
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Setup volatility data
        self.portfolio.volatility_data['BTCUSDT'] = pd.Series([0.04] * 20)
        self.portfolio.volatility_data['ETHUSDT'] = pd.Series([0.02] * 20)
        
        # Setup correlation data
        corr_data = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]],
            index=symbols,
            columns=symbols
        )
        self.portfolio.correlation_data = corr_data
        
        weights = self.portfolio.calculate_target_weights(symbols, method='volatility_targeting')
        
        # Check weights are normalized
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        
        # Check individual weight constraints
        for symbol in symbols:
            self.assertGreaterEqual(weights[symbol], self.portfolio.min_weight)
            self.assertLessEqual(weights[symbol], self.portfolio.max_weight)
    
    def test_calculate_allocations(self):
        """Test allocation calculation from weights."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        weights = {
            'BTCUSDT': 0.4,
            'ETHUSDT': 0.35,
            'SOLUSDT': 0.25
        }
        
        allocations = self.portfolio.calculate_allocations(weights)
        
        # Check types and structure
        self.assertIsInstance(allocations, dict)
        for symbol in symbols:
            self.assertIn(symbol, allocations)
            self.assertIsInstance(allocations[symbol], AllocationWeights)
        
        # Check allocation calculations
        expected_total = self.total_capital * self.max_allocation_pct
        
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        self.assertLessEqual(total_allocated, expected_total * 1.01)  # Small tolerance
        
        # Check individual allocations
        for symbol in symbols:
            allocation = allocations[symbol]
            expected_capital = expected_total * weights[symbol]
            self.assertAlmostEqual(allocation.allocated_capital, expected_capital, places=2)
            
            # Weight should match input
            self.assertAlmostEqual(allocation.weight, weights[symbol], places=3)
    
    def test_should_rebalance(self):
        """Test rebalancing trigger logic."""
        # Set last rebalance time to 25 hours ago to trigger rebalancing
        self.portfolio.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Should rebalance after 24 hours
        self.assertTrue(self.portfolio.should_rebalance())
        
        # After setting recent rebalance time, should not rebalance immediately
        self.portfolio.last_rebalance_time = datetime.now()
        self.assertFalse(self.portfolio.should_rebalance())
    
    def test_needs_rebalancing_threshold(self):
        """Test weight drift detection for rebalancing."""
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Set current weights
        self.portfolio.current_weights = {
            'BTCUSDT': 0.6,
            'ETHUSDT': 0.4
        }
        
        # Target weights with small drift (should not rebalance)
        target_weights = {
            'BTCUSDT': 0.62,
            'ETHUSDT': 0.38
        }
        
        self.assertFalse(self.portfolio.needs_rebalancing(target_weights))
        
        # Target weights with large drift (should rebalance)
        target_weights_large_drift = {
            'BTCUSDT': 0.7,
            'ETHUSDT': 0.3
        }
        
        self.assertTrue(self.portfolio.needs_rebalancing(target_weights_large_drift))
    
    def test_update_current_state(self):
        """Test updating current portfolio state."""
        weights = {
            'BTCUSDT': 0.5,
            'ETHUSDT': 0.3,
            'SOLUSDT': 0.2
        }
        
        allocations = self.portfolio.calculate_allocations(weights)
        
        # Update state
        self.portfolio.update_current_state(weights, allocations)
        
        # Check state is updated
        self.assertEqual(self.portfolio.current_weights, weights)
        self.assertEqual(self.portfolio.current_allocations, allocations)
        self.assertIsNotNone(self.portfolio.last_rebalance_time)
        
        # Check timestamp is recent
        time_diff = datetime.now() - self.portfolio.last_rebalance_time
        self.assertLess(time_diff.total_seconds(), 5)  # Within 5 seconds
    
    def test_get_current_portfolio_value(self):
        """Test portfolio value calculation."""
        # Setup mock current allocations
        allocations = {
            'BTCUSDT': AllocationWeights(weight=0.5, allocated_capital=4000.0),
            'ETHUSDT': AllocationWeights(weight=0.3, allocated_capital=2400.0),
            'SOLUSDT': AllocationWeights(weight=0.2, allocated_capital=1600.0)
        }
        self.portfolio.current_allocations = allocations
        
        # Mock market values (simulate price changes)
        market_values = {
            'BTCUSDT': 4200.0,  # +5%
            'ETHUSDT': 2280.0,  # -5%
            'SOLUSDT': 1600.0   # No change
        }
        
        total_value = self.portfolio.get_current_portfolio_value(market_values)
        expected_value = sum(market_values.values())
        
        self.assertAlmostEqual(total_value, expected_value, places=2)
    
    def test_get_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        # Setup portfolio state
        weights = {'BTCUSDT': 0.6, 'ETHUSDT': 0.4}
        allocations = self.portfolio.calculate_allocations(weights)
        self.portfolio.update_current_state(weights, allocations)
        
        # Setup volatility data
        for symbol in weights.keys():
            self.portfolio.volatility_data[symbol] = pd.Series([0.02, 0.025, 0.018])
        
        stats = self.portfolio.get_portfolio_statistics()
        
        # Check required statistics
        required_fields = [
            'total_capital', 'allocated_capital', 'available_capital',
            'num_positions', 'weights', 'average_volatility'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Check calculations
        total_allocated = sum(a.allocated_capital for a in allocations.values())
        self.assertAlmostEqual(stats['allocated_capital'], total_allocated, places=2)
        self.assertAlmostEqual(
            stats['available_capital'], 
            self.total_capital - total_allocated, 
            places=2
        )
        self.assertEqual(stats['num_positions'], len(weights))
    
    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        symbols = ['BTCUSDT', 'ETHUSDT']
        weights = {'BTCUSDT': 0.6, 'ETHUSDT': 0.4}
        
        # Setup volatility data
        self.portfolio.volatility_data['BTCUSDT'] = pd.Series([0.03] * 10)
        self.portfolio.volatility_data['ETHUSDT'] = pd.Series([0.02] * 10)
        
        # Setup correlation data
        corr_data = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=symbols,
            columns=symbols
        )
        self.portfolio.correlation_data = corr_data
        
        portfolio_vol = self.portfolio.calculate_portfolio_volatility(weights)
        
        # Portfolio volatility should be positive and reasonable
        self.assertGreater(portfolio_vol, 0)
        self.assertLess(portfolio_vol, max(0.03, 0.02))  # Should be less than max individual vol
    
    def test_detect_regime_change(self):
        """Test regime change detection."""
        # Test with stable returns (no regime change)
        stable_returns = pd.Series([0.001] * 30)
        is_change, current_vol, baseline_vol = self.portfolio.detect_regime_change(stable_returns)
        
        self.assertFalse(is_change)
        self.assertGreater(current_vol, 0)
        self.assertGreater(baseline_vol, 0)
        
        # Test with volatile returns (regime change)
        volatile_returns = pd.Series([0.001] * 20 + [0.05, -0.04, 0.06, -0.05] * 3)
        is_change, current_vol, baseline_vol = self.portfolio.detect_regime_change(volatile_returns)
        
        self.assertTrue(is_change)
        self.assertGreater(current_vol, baseline_vol * 1.5)  # Should detect significant increase


if __name__ == '__main__':
    unittest.main()
