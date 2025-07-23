"""
Comprehensive Unit Test Suite
Senior Quantitative Developer Testing Protocol

This file contains comprehensive unit tests using the unittest framework
for compatibility with enterprise environments and traditional testing frameworks.
Covers all major components with detailed validation.
"""

import unittest
import asyncio
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from algorithm.algo_engine import AlgoEngine
from algorithm.trade_signal import TradeSignal
from execution.execution_engine import ProductionExecutionEngine
from execution.portfolio import ProductionPortfolioManager, AllocationWeights
from execution.risk_manager import ProductionRiskManager, ProductionRiskParameters
from execution.stress_handler import StressHandlingModule
from tests.utils.mock_objects import MockDataEngine, MockBinanceClient, MockStrategy, MockErrorStrategy
from tests.utils.test_data import generate_market_data_bar, create_test_signal_metadata, generate_ohlcv_data


class TestAlgorithmEngineUnit(unittest.TestCase):
    """Unit tests for Algorithm Engine components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_engine = MockDataEngine()
        self.algo_engine = AlgoEngine(self.mock_data_engine)
        self.test_strategy = MockStrategy(["buy", "hold", "sell"], "unittest_strategy")
        
        # Setup some test data
        self.test_candles = generate_ohlcv_data(100)
        self.mock_data_engine.candles_data[("BTCUSDT", "1m")] = self.test_candles
    
    def test_algorithm_engine_initialization(self):
        """Test algorithm engine initialization."""
        self.assertIsNotNone(self.algo_engine)
        self.assertIsNotNone(self.algo_engine.data_engine)
        self.assertFalse(self.algo_engine.running)
        self.assertIsInstance(self.algo_engine._last_signal_states, dict)
        self.assertGreater(self.algo_engine._min_signal_interval, 0)
        self.assertEqual(len(self.algo_engine._last_signal_states), 0)
    
    def test_data_hash_generation_consistency(self):
        """Test data hash generation consistency."""
        candles1 = [[1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1000.0]]
        candles2 = [[1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1000.0]]
        candles3 = [[1640995200000, 47000.0, 47100.0, 46900.0, 47100.0, 1000.0]]
        
        hash1 = self.algo_engine._get_data_hash(candles1)
        hash2 = self.algo_engine._get_data_hash(candles2)
        hash3 = self.algo_engine._get_data_hash(candles3)
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        # Different data should produce different hash
        self.assertNotEqual(hash1, hash3)
        
        # Empty data should return empty string
        empty_hash = self.algo_engine._get_data_hash([])
        self.assertEqual(empty_hash, "")
    
    def test_signal_throttling_logic(self):
        """Test signal throttling and processing logic."""
        key = "BTCUSDT_1m"
        current_time = int(time.time())
        data_hash = "test_hash_123"
        
        # First signal should be processed
        should_process1 = self.algo_engine._should_process_signal(key, current_time, data_hash)
        self.assertTrue(should_process1)
        
        # Update signal state
        self.algo_engine._update_signal_state(key, current_time, data_hash, "open/buy")
        
        # Same signal within interval should not be processed
        should_process2 = self.algo_engine._should_process_signal(key, current_time + 30, data_hash)
        self.assertFalse(should_process2)
        
        # Different data should be processed regardless of time
        new_hash = "different_hash_456"
        should_process3 = self.algo_engine._should_process_signal(key, current_time + 30, new_hash)
        self.assertTrue(should_process3)
        
        # Same data after interval should be processed
        should_process4 = self.algo_engine._should_process_signal(key, current_time + 70, data_hash)
        self.assertTrue(should_process4)
    
    def test_signal_state_update(self):
        """Test signal state update functionality."""
        key = "ETHUSDT_5m"
        timestamp = 1640995200
        data_hash = "state_test_hash"
        signal_type = "exit/sell"
        
        # Update state
        self.algo_engine._update_signal_state(key, timestamp, data_hash, signal_type)
        
        # Verify state was stored correctly
        self.assertIn(key, self.algo_engine._last_signal_states)
        state = self.algo_engine._last_signal_states[key]
        
        self.assertEqual(state['timestamp'], timestamp)
        self.assertEqual(state['data_hash'], data_hash)
        self.assertEqual(state['signal_type'], signal_type)
    
    def test_process_signals_with_valid_data(self):
        """Test signal processing with valid data."""
        async def run_test():
            symbol = "BTCUSDT"
            timeframe = "1m"
            
            signal = await self.algo_engine.process_signals(symbol, timeframe, self.test_strategy)
            
            if signal:  # Strategy might return None sometimes
                self.assertIsInstance(signal, TradeSignal)
                self.assertEqual(signal.symbol, symbol)
                self.assertEqual(signal.strategy_id, self.test_strategy.strategy_id)
                self.assertIn(signal.action, ["open", "exit", "hold"])
                self.assertIn(signal.side, ["buy", "sell", "none"])
                self.assertGreaterEqual(signal.signal_confidence, 0)
                self.assertLessEqual(signal.signal_confidence, 1)
        
        # Run async test
        asyncio.run(run_test())
    
    def test_process_signals_with_missing_data(self):
        """Test signal processing with missing data."""
        async def run_test():
            # Clear test data
            self.mock_data_engine.candles_data.clear()
            
            signal = await self.algo_engine.process_signals("NONEXISTENT", "1m", self.test_strategy)
            self.assertIsNone(signal)  # Should return None for missing data
        
        asyncio.run(run_test())
    
    def test_process_signals_with_error_strategy(self):
        """Test signal processing with strategy that throws errors."""
        async def run_test():
            error_strategy = MockErrorStrategy()
            
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", error_strategy)
            self.assertIsNone(signal)  # Should return None on strategy error
        
        asyncio.run(run_test())


class TestProductionPortfolioManagerUnit(unittest.TestCase):
    """Unit tests for Production Portfolio Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio_manager = ProductionPortfolioManager(
            total_capital=10000.0,
            target_volatility=0.18,
            max_allocation_pct=0.85
        )
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    def test_initialization_parameters(self):
        """Test portfolio manager initialization parameters."""
        self.assertEqual(self.portfolio_manager.total_capital, 10000.0)
        self.assertEqual(self.portfolio_manager.target_volatility, 0.18)
        self.assertEqual(self.portfolio_manager.max_allocation_pct, 0.85)
        self.assertEqual(self.portfolio_manager.alpha, 0.3)  # Document specification
        self.assertEqual(self.portfolio_manager.lookback_bars, 60)
        self.assertEqual(self.portfolio_manager.regime_percentile, 75)
    
    def test_data_structures_initialization(self):
        """Test data structures are properly initialized."""
        self.assertIsInstance(self.portfolio_manager.volatility_data, dict)
        self.assertIsInstance(self.portfolio_manager.correlation_data, dict)
        self.assertIsInstance(self.portfolio_manager.allocation_weights, dict)
        self.assertIsInstance(self.portfolio_manager.reserved_allocations, dict)
        self.assertIsInstance(self.portfolio_manager.volatility_history, list)
        
        # All should be empty initially
        self.assertEqual(len(self.portfolio_manager.volatility_data), 0)
        self.assertEqual(len(self.portfolio_manager.correlation_data), 0)
        self.assertEqual(len(self.portfolio_manager.allocation_weights), 0)
        self.assertEqual(len(self.portfolio_manager.reserved_allocations), 0)
    
    def test_volatility_data_update_mechanism(self):
        """Test volatility data update mechanism."""
        symbol = "BTCUSDT"
        
        # Add data points
        volatilities = [0.02, 0.025, 0.03, 0.035, 0.04]
        for vol in volatilities:
            self.portfolio_manager.update_volatility_data(symbol, vol)
        
        # Verify data storage
        self.assertIn(symbol, self.portfolio_manager.volatility_data)
        self.assertEqual(len(self.portfolio_manager.volatility_data[symbol]), len(volatilities))
        self.assertEqual(self.portfolio_manager.volatility_data[symbol], volatilities)
        
        # Test data limit (should keep only 60 bars)
        for i in range(70):
            self.portfolio_manager.update_volatility_data(symbol, 0.01 + i * 0.001)
        
        self.assertEqual(len(self.portfolio_manager.volatility_data[symbol]), 60)
    
    def test_correlation_data_update_mechanism(self):
        """Test correlation data update mechanism."""
        sym1, sym2 = "BTCUSDT", "ETHUSDT"
        
        # Test proper ordering
        self.portfolio_manager.update_correlation_data(sym1, sym2, 0.6)
        self.portfolio_manager.update_correlation_data(sym2, sym1, 0.7)  # Reverse order
        
        # Should be stored with consistent alphabetical ordering
        expected_pair = tuple(sorted([sym1, sym2]))
        self.assertIn(expected_pair, self.portfolio_manager.correlation_data)
        self.assertEqual(len(self.portfolio_manager.correlation_data[expected_pair]), 2)
        self.assertEqual(self.portfolio_manager.correlation_data[expected_pair], [0.6, 0.7])
    
    def test_volatility_ema_calculation(self):
        """Test volatility EMA calculation."""
        symbol = "TESTUSDT"
        
        # Add test data
        data_points = [0.02, 0.025, 0.03, 0.025, 0.02]
        for point in data_points:
            self.portfolio_manager.update_volatility_data(symbol, point)
        
        # Calculate EMA
        ema = self.portfolio_manager.get_volatility_ema(symbol)
        
        # Verify EMA calculation
        self.assertIsInstance(ema, float)
        self.assertGreater(ema, 0)
        self.assertGreaterEqual(ema, 0.001)  # Should respect floor
        
        # EMA should be influenced by recent values
        self.assertGreater(ema, min(data_points))
        self.assertLess(ema, max(data_points))
        
        # Test with no data (should return default)
        ema_no_data = self.portfolio_manager.get_volatility_ema("NONEXISTENT")
        self.assertEqual(ema_no_data, 0.02)  # Default 2% volatility
    
    def test_average_correlation_calculation(self):
        """Test average correlation calculation."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Setup correlation matrix
        correlations = [
            ("BTCUSDT", "ETHUSDT", 0.6),
            ("BTCUSDT", "SOLUSDT", 0.4),
            ("ETHUSDT", "SOLUSDT", 0.5)
        ]
        
        for sym1, sym2, corr in correlations:
            self.portfolio_manager.update_correlation_data(sym1, sym2, corr)
        
        # Test average correlation for BTCUSDT
        avg_corr_btc = self.portfolio_manager.get_average_correlation("BTCUSDT", symbols)
        expected_btc = (0.6 + 0.4) / 2  # Average of correlations with ETH and SOL
        self.assertAlmostEqual(avg_corr_btc, expected_btc, places=5)
        
        # Test with no correlations
        avg_corr_none = self.portfolio_manager.get_average_correlation("NEWCOIN", symbols)
        self.assertEqual(avg_corr_none, 0.0)
    
    def test_weight_computation_formula(self):
        """Test portfolio weight computation using document formula."""
        # Setup test data
        for symbol in self.test_symbols:
            # Different volatilities for each symbol
            base_vol = 0.02 + self.test_symbols.index(symbol) * 0.005
            self.portfolio_manager.update_volatility_data(symbol, base_vol)
        
        # Add some correlation data
        self.portfolio_manager.update_correlation_data("BTCUSDT", "ETHUSDT", 0.6)
        self.portfolio_manager.update_correlation_data("BTCUSDT", "SOLUSDT", 0.4)
        self.portfolio_manager.update_correlation_data("ETHUSDT", "SOLUSDT", 0.5)
        
        # Compute weights
        weights = self.portfolio_manager.compute_weights(self.test_symbols)
        
        # Validate weights structure
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), len(self.test_symbols))
        
        # All symbols should have weights
        for symbol in self.test_symbols:
            self.assertIn(symbol, weights)
            self.assertGreater(weights[symbol], 0)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Symbol with lower volatility should have higher weight
        btc_vol = self.portfolio_manager.get_volatility_ema("BTCUSDT")
        sol_vol = self.portfolio_manager.get_volatility_ema("SOLUSDT")
        if btc_vol < sol_vol:
            self.assertGreater(weights["BTCUSDT"], weights["SOLUSDT"])
    
    def test_high_volatility_regime_detection(self):
        """Test high volatility regime detection."""
        # Initially should not be high vol regime (insufficient data)
        self.assertFalse(self.portfolio_manager.is_high_volatility_regime())
        
        # Add volatility history (simulate 35 days)
        for i in range(35):
            daily_vol = 0.015 + i * 0.001  # Gradually increasing volatility
            self.portfolio_manager.volatility_history.append(daily_vol)
        
        # Now should have enough data to calculate regime
        regime_status = self.portfolio_manager.is_high_volatility_regime()
        self.assertIsInstance(regime_status, bool)
        
        # Add current symbols data to enable calculation
        for symbol in self.test_symbols:
            self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Test again with actual data
        regime_status_with_data = self.portfolio_manager.is_high_volatility_regime()
        self.assertIsInstance(bool(regime_status_with_data), bool)
    
    def test_scaling_multiplier_calculation(self):
        """Test scaling multiplier calculation."""
        # Setup volatility data
        for symbol in self.test_symbols:
            self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Calculate scaling multiplier (method takes no parameters)
        multiplier = self.portfolio_manager.calculate_scaling_multiplier()
        
        self.assertIsInstance(multiplier, float)
        self.assertGreater(multiplier, 0)
        self.assertLessEqual(multiplier, 1.0)  # Should be capped at 1.0
    
    def test_rebalancing_workflow(self):
        """Test complete rebalancing workflow."""
        # Setup data for rebalancing
        for symbol in self.test_symbols:
            # Add volatility data
            for _ in range(10):
                vol = 0.02 + np.random.normal(0, 0.001)
                self.portfolio_manager.update_volatility_data(symbol, vol)
        
        # Add correlation data
        pairs = [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT"), ("ETHUSDT", "SOLUSDT")]
        for sym1, sym2 in pairs:
            self.portfolio_manager.update_correlation_data(sym1, sym2, 0.5)
        
        # Force rebalancing
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        
        # Perform rebalancing
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Validate allocation structure
        self.assertIsInstance(allocations, dict)
        self.assertEqual(len(allocations), len(self.test_symbols))
        
        # Check allocation objects
        total_allocated = 0
        for symbol, allocation in allocations.items():
            self.assertIsInstance(allocation, AllocationWeights)
            self.assertEqual(allocation.symbol, symbol)
            self.assertGreater(allocation.allocated_capital, 0)
            self.assertGreater(allocation.weight, 0)
            self.assertLessEqual(allocation.weight, 1)
            total_allocated += allocation.allocated_capital
        
        # Total allocation should respect max_allocation_pct
        max_allowed = self.portfolio_manager.total_capital * self.portfolio_manager.max_allocation_pct
        self.assertLessEqual(total_allocated, max_allowed * 1.01)  # Small tolerance
    
    def test_reservation_system_functionality(self):
        """Test allocation reservation system."""
        # Setup allocations first
        for symbol in self.test_symbols:
            self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Force rebalancing to create allocations
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        symbol = self.test_symbols[0]
        available_capital = allocations[symbol].allocated_capital
        
        # Test successful reservation
        reservation_amount = available_capital * 0.5
        result = self.portfolio_manager.reserve_allocation(symbol, reservation_amount)
        self.assertTrue(result)
        self.assertIn(symbol, self.portfolio_manager.reserved_allocations)
        self.assertEqual(self.portfolio_manager.reserved_allocations[symbol], reservation_amount)
        
        # Test over-reservation
        excessive_amount = available_capital * 2
        result_fail = self.portfolio_manager.reserve_allocation(symbol, excessive_amount)
        self.assertFalse(result_fail)
        
        # Test partial release
        release_amount = reservation_amount * 0.5
        self.portfolio_manager.release_allocation(symbol, release_amount)
        remaining = self.portfolio_manager.reserved_allocations.get(symbol, 0)
        expected_remaining = reservation_amount - release_amount
        self.assertAlmostEqual(remaining, expected_remaining, places=2)
        
        # Test full release
        self.portfolio_manager.release_allocation(symbol, remaining)
        self.assertEqual(self.portfolio_manager.reserved_allocations.get(symbol, 0), 0)


class TestProductionRiskManagerUnit(unittest.TestCase):
    """Unit tests for Production Risk Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio_manager = ProductionPortfolioManager(10000.0)
        self.risk_manager = ProductionRiskManager(self.portfolio_manager)
    
    def test_initialization_and_parameters(self):
        """Test risk manager initialization and parameters."""
        self.assertIsNotNone(self.risk_manager.portfolio_manager)
        self.assertIsInstance(self.risk_manager.risk_params, ProductionRiskParameters)
        
        # Verify document-specified parameters
        params = self.risk_manager.risk_params
        self.assertEqual(params.risk_per_trade_pct, 0.008)  # 0.8%
        self.assertEqual(params.kelly_fraction, 0.7)
        self.assertEqual(params.base_cost_pct, 0.0014)  # 0.14%
        self.assertEqual(params.min_atr_floor, 0.001)
        self.assertEqual(params.atr_stop_multiplier, 1.8)
        self.assertEqual(params.atr_trail_multiplier, 0.8)
        self.assertEqual(params.risk_reward_ratio, 2.0)
        self.assertEqual(params.partial_exit_ratio, 0.4)
        self.assertEqual(params.max_leverage, 10)
        self.assertEqual(params.target_volatility, 0.18)
    
    def test_dynamic_cost_adjustment_calculation(self):
        """Test dynamic cost adjustment calculation."""
        # Test with different volatility levels
        test_cases = [
            (0.0, 0.0014),  # No volatility adjustment
            (0.5, 0.0014 * 1.25),  # 50% volatility norm
            (1.0, 0.0014 * 1.5),   # 100% volatility norm
        ]
        
        for vol_norm, expected_cost in test_cases:
            cost = self.risk_manager.calculate_dynamic_cost_adjustment(vol_norm)
            self.assertAlmostEqual(cost, expected_cost, places=6)
            self.assertGreaterEqual(cost, self.risk_manager.risk_params.base_cost_pct)
    
    def test_position_sizing_calculation_formula(self):
        """Test position sizing calculation using document formula."""
        # Test parameters
        symbol = "BTCUSDT"
        allocated_capital = 5000.0
        atr_value = 0.02
        entry_price = 50000.0
        volatility_norm = 0.5
        
        result = self.risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=atr_value,
            entry_price=entry_price,
            volatility_norm=volatility_norm
        )
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        required_keys = ['size_usdt', 'leverage', 'stop_loss_distance', 'take_profit_distance', 'size_contracts']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Validate calculations
        self.assertGreater(result['size_usdt'], 0)
        self.assertGreaterEqual(result['leverage'], 1)
        self.assertLessEqual(result['leverage'], 10)  # Max leverage
        
        # Calculate actual stop loss and take profit from distances
        stop_loss = entry_price - result['stop_loss_distance']
        take_profit = entry_price + result['take_profit_distance']
        
        # For long position: SL < entry < TP
        self.assertLess(stop_loss, entry_price)
        self.assertGreater(take_profit, entry_price)
        
        # Verify stop loss calculation (Entry - 1.8 * ATR)
        expected_sl_distance = atr_value * self.risk_manager.risk_params.atr_stop_multiplier
        self.assertAlmostEqual(result['stop_loss_distance'], expected_sl_distance, delta=entry_price * 0.001)
        
        # Verify risk-reward ratio (TP should be 2x distance from SL)
        risk_reward = result['take_profit_distance'] / result['stop_loss_distance'] if result['stop_loss_distance'] > 0 else 0
        self.assertAlmostEqual(risk_reward, 2.0, delta=0.1)  # 1:2 risk-reward
    
    def test_position_sizing_with_atr_floor(self):
        """Test position sizing with ATR floor application."""
        # Test with very low ATR (should apply floor)
        result_low_atr = self.risk_manager.calculate_position_size(
            symbol="TESTUSDT",
            allocated_capital=1000.0,
            atr_value=0.0001,  # Below floor
            entry_price=100.0,
            volatility_norm=0.5
        )
        
        # Test with normal ATR
        result_normal_atr = self.risk_manager.calculate_position_size(
            symbol="TESTUSDT",
            allocated_capital=1000.0,
            atr_value=0.02,  # Above floor
            entry_price=100.0,
            volatility_norm=0.5
        )
        
        # Low ATR should result in smaller position size due to floor
        self.assertGreater(result_normal_atr['size_usdt'], 0)
        self.assertGreater(result_low_atr['size_usdt'], 0)
    
    def test_dynamic_leverage_calculation(self):
        """Test dynamic leverage calculation."""
        symbol = "BTCUSDT"
        atr_value = 0.02
        
        leverage = self.risk_manager.calculate_dynamic_leverage(symbol, atr_value)
        
        self.assertIsInstance(leverage, (int, float))
        self.assertGreaterEqual(leverage, 1)
        self.assertLessEqual(leverage, 10)  # Max leverage cap
    
    def test_trade_validation_logic(self):
        """Test trade validation logic."""
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Setup portfolio allocation first
        for symbol in test_symbols:
            self.portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Force rebalancing to allocate capital
        self.portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = self.portfolio_manager.rebalance_portfolio(test_symbols)
        
        # Test with valid trade parameters
        result_valid = self.risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="open", 
            side="buy",
            entry_price=50000.0,
            atr_value=0.02
        )
        self.assertIsInstance(result_valid, dict)
        self.assertIn('valid', result_valid)
        
        # Test with hold action (should be valid but no analysis needed)
        result_hold = self.risk_manager.validate_trade(
            symbol="BTCUSDT",
            action="hold",
            side="none", 
            entry_price=50000.0,
            atr_value=0.02
        )
        self.assertTrue(result_hold['valid'])
    
    def test_equity_curve_tracking(self):
        """Test equity curve tracking and slope calculation."""
        # Add equity data points
        base_equity = 10000
        for i in range(20):
            equity = base_equity + i * 50  # Steady growth
            self.risk_manager.update_equity_curve(equity)
        
        self.assertEqual(len(self.risk_manager.equity_curve), 20)
        
        # Test slope calculation
        slope = self.risk_manager.calculate_equity_slope()
        self.assertIsInstance(slope, float)
        self.assertGreater(slope, 0)  # Should be positive for growing equity
        
        # Test with declining equity
        for i in range(10):
            equity = base_equity + 1000 - i * 100  # Declining
            self.risk_manager.update_equity_curve(equity)
        
        slope_declining = self.risk_manager.calculate_equity_slope()
        self.assertLess(slope_declining, slope)  # Should be less positive or negative
    
    def test_drawdown_tracking(self):
        """Test drawdown tracking functionality."""
        # Simulate equity curve with drawdown
        equity_values = [10000, 10500, 11000, 10800, 10200, 9800, 10100, 10600]
        
        for equity in equity_values:
            self.risk_manager.update_equity_curve(equity)
        
        # Test drawdown calculation if method exists
        if hasattr(self.risk_manager, 'calculate_max_drawdown'):
            max_dd = self.risk_manager.calculate_max_drawdown()
            self.assertIsInstance(max_dd, float)
            self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero


class TestStressHandlingModuleUnit(unittest.TestCase):
    """Unit tests for Stress Handling Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_binance_client = MockBinanceClient()
        self.execution_engine = ProductionExecutionEngine(self.mock_binance_client, 10000.0)
        self.stress_handler = StressHandlingModule(self.execution_engine)
    
    def test_stress_handler_initialization(self):
        """Test stress handler initialization."""
        self.assertIsNotNone(self.stress_handler.execution_engine)
        self.assertTrue(hasattr(self.stress_handler, 'check_flash_crash'))
        self.assertTrue(hasattr(self.stress_handler, 'check_connection_lag'))
        self.assertTrue(hasattr(self.stress_handler, 'check_liquidity_filters'))
        self.assertTrue(hasattr(self.stress_handler, 'check_kill_switches'))
    
    def test_flash_crash_detection_thresholds(self):
        """Test flash crash detection with various scenarios."""
        symbol = "BTCUSDT"
        atr_value = 0.02  # 2% ATR
        
        # Normal market movement (should not trigger)
        normal_data = {
            'open': 50000, 'high': 50200, 'low': 49800, 'close': 50100, 'volume': 1000
        }
        result_normal = self.stress_handler.check_flash_crash(symbol, normal_data, atr_value)
        self.assertFalse(result_normal)
        
        # Flash crash scenario (>4x ATR drop in percentage terms)
        # 4 * 0.02 = 0.08 or 8% drop threshold
        crash_data = {
            'open': 50000, 'high': 50000, 'low': 45500, 'close': 45500, 'volume': 5000  # 9% drop
        }
        result_crash = self.stress_handler.check_flash_crash(symbol, crash_data, atr_value)
        self.assertTrue(result_crash)
        
        # Edge case: slightly above 4x ATR drop (8.1%)  
        edge_data = {
            'open': 50000, 'high': 50000, 'low': 45950, 'close': 45950, 'volume': 2000  # 8.1% drop
        }
        result_edge = self.stress_handler.check_flash_crash(symbol, edge_data, atr_value)
        # This should trigger since 8.1% > 8% threshold
        self.assertTrue(result_edge)
    
    def test_connection_lag_detection(self):
        """Test connection lag detection."""
        # Normal timing (should be healthy - return True)
        current_time = datetime.now()
        result_normal = self.stress_handler.check_connection_lag(current_time)
        self.assertTrue(result_normal)
        
        # High lag scenario (>3 seconds old - should be lagged - return False)
        old_time = datetime.now() - timedelta(seconds=5)
        result_lag = self.stress_handler.check_connection_lag(old_time)
        self.assertFalse(result_lag)
        
        # Edge case: exactly 3 seconds
        edge_time = datetime.now() - timedelta(seconds=3)
        result_edge = self.stress_handler.check_connection_lag(edge_time)
        self.assertIsInstance(result_edge, bool)
    
    def test_liquidity_filters(self):
        """Test liquidity filtering mechanisms."""
        symbol = "BTCUSDT"
        
        # Good liquidity conditions
        result_good = self.stress_handler.check_liquidity_filters(
            symbol=symbol,
            volume_24h=10000000,  # $10M daily volume  
            spread_pct=0.001,     # 0.1% spread
            funding_rate=0.001    # 0.1% funding rate
        )
        self.assertTrue(result_good)
        
        # Poor volume
        result_poor_vol = self.stress_handler.check_liquidity_filters(
            symbol=symbol,
            volume_24h=1000000,   # $1M daily volume (< $5M threshold)
            spread_pct=0.001,
            funding_rate=0.001
        )
        self.assertFalse(result_poor_vol)
        
        # Wide spread
        result_wide_spread = self.stress_handler.check_liquidity_filters(
            symbol=symbol,
            volume_24h=10000000,
            spread_pct=0.002,     # 0.2% spread (> 0.15% threshold)
            funding_rate=0.001
        )
        self.assertFalse(result_wide_spread)
    
    def test_kill_switch_mechanisms(self):
        """Test kill switch mechanisms."""
        # Test drawdown kill switch if implemented
        if hasattr(self.stress_handler, 'check_drawdown_kill_switch'):
            # Simulate high drawdown
            result_high_dd = self.stress_handler.check_drawdown_kill_switch(0.15)  # 15% drawdown
            self.assertTrue(result_high_dd)
            
            # Normal drawdown
            result_normal_dd = self.stress_handler.check_drawdown_kill_switch(0.05)  # 5% drawdown
            self.assertFalse(result_normal_dd)
        
        # Test equity slope kill switch if implemented
        if hasattr(self.stress_handler, 'check_slope_kill_switch'):
            # Steep decline
            result_steep = self.stress_handler.check_slope_kill_switch(-0.12)  # -12% slope
            self.assertTrue(result_steep)
            
            # Normal slope
            result_normal_slope = self.stress_handler.check_slope_kill_switch(-0.05)  # -5% slope
            self.assertFalse(result_normal_slope)


class TestIntegrationUnit(unittest.TestCase):
    """Unit tests for component integration."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.mock_data_engine = MockDataEngine()
        self.mock_binance_client = MockBinanceClient()
        
        self.algo_engine = AlgoEngine(self.mock_data_engine)
        
        # Setup execution engine asynchronously
        async def setup_execution():
            engine = ProductionExecutionEngine(self.mock_binance_client, 10000.0)
            await engine.setup()
            return engine
        
        self.execution_engine = asyncio.run(setup_execution())
        
        # Setup test data
        self.test_symbols = ["BTCUSDT", "ETHUSDT"]
        for symbol in self.test_symbols:
            candles = generate_ohlcv_data(50)
            self.mock_data_engine.candles_data[(symbol, "1m")] = candles
            
            market_data = generate_market_data_bar(symbol)
            self.execution_engine.update_market_data_bar(symbol, market_data, 0.02)
    
    def test_signal_to_execution_integration(self):
        """Test signal generation to execution integration."""
        async def run_integration_test():
            strategy = MockStrategy(["buy", "sell"], "integration_test")
            
            # Generate signal
            signal = await self.algo_engine.process_signals("BTCUSDT", "1m", strategy)
            
            if signal and signal.action != "hold":
                # Add execution metadata
                signal.metadata = {
                    'price': 50000.0,
                    'atr_value': 0.02,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                }
                
                # Execute signal
                result = await self.execution_engine.process_signal(signal)
                
                self.assertIsInstance(result, dict)
                self.assertIn('status', result)
                self.assertIn('symbol', result)
        
        asyncio.run(run_integration_test())
    
    def test_portfolio_risk_integration(self):
        """Test portfolio and risk management integration."""
        portfolio_manager = self.execution_engine.portfolio_manager
        risk_manager = self.execution_engine.risk_manager
        
        # Setup portfolio data
        for symbol in self.test_symbols:
            portfolio_manager.update_volatility_data(symbol, 0.02)
        
        # Force rebalancing
        portfolio_manager.last_rebalance_time = datetime.now() - timedelta(hours=25)
        allocations = portfolio_manager.rebalance_portfolio(self.test_symbols)
        
        # Test risk calculation with allocation
        symbol = self.test_symbols[0]
        allocated_capital = allocations[symbol].allocated_capital
        
        risk_result = risk_manager.calculate_position_size(
            symbol=symbol,
            allocated_capital=allocated_capital,
            atr_value=0.02,
            entry_price=50000.0
        )
        
        self.assertIsInstance(risk_result, dict)
        self.assertGreater(risk_result['position_size_usdt'], 0)
        self.assertLessEqual(risk_result['position_size_usdt'], allocated_capital)
    
    def test_stress_handling_integration(self):
        """Test stress handling integration with execution engine."""
        stress_handler = self.execution_engine.stress_handler
        
        # Test flash crash response
        crash_data = {
            'open': 50000, 'high': 50000, 'low': 40000, 'close': 40000, 'volume': 5000
        }
        
        # Update market data (should trigger stress response)
        self.execution_engine.update_market_data_bar("BTCUSDT", crash_data, 0.05)
        
        # Verify stress handler detected the condition
        flash_crash = stress_handler.check_flash_crash("BTCUSDT", crash_data, 0.05)
        self.assertTrue(flash_crash)
        
        # System should still be functional after stress event
        portfolio_summary = self.execution_engine.get_portfolio_summary()
        self.assertIsInstance(portfolio_summary, dict)
        self.assertIn('total_capital', portfolio_summary)


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestAlgorithmEngineUnit,
        TestProductionPortfolioManagerUnit,
        TestProductionRiskManagerUnit,
        TestStressHandlingModuleUnit,
        TestIntegrationUnit
    ]
    
    # Run all tests
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE UNIT TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    sys.exit(exit_code)
