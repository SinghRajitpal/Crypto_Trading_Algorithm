"""
Unit tests for StressHandlingModule.

This test suite covers:
1. Stress condition detection and monitoring
2. Circuit breaker mechanisms
3. Emergency protocols and responses
4. Kill switch functionality
5. Market stress indicators
6. System health monitoring
7. Recovery procedures
8. Performance under stress conditions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from execution.stress_handler import StressHandlingModule


class MockExecutionEngine:
    """Mock execution engine for testing."""
    
    def __init__(self):
        self.portfolio_manager = Mock()
        self.risk_manager = Mock()
        self.order_executor = Mock()
        
        # Setup default mock returns
        self.portfolio_manager.get_portfolio_summary.return_value = {
            'total_capital': 10000.0,
            'allocated_capital': 8000.0,
            'active_positions': 3
        }
        
        self.risk_manager.get_risk_metrics.return_value = {
            'daily_pnl': 100.0,
            'current_drawdown': -0.05,
            'current_sharpe': 1.5,
            'max_drawdown_hit': False
        }


class TestStressHandlingModule(unittest.TestCase):
    """Test suite for StressHandlingModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.execution_engine = MockExecutionEngine()
        self.stress_handler = StressHandlingModule(self.execution_engine)
        
    def test_initialization(self):
        """Test stress handler initialization."""
        self.assertEqual(self.stress_handler.execution_engine, self.execution_engine)
        
        # Check default thresholds
        self.assertEqual(self.stress_handler.max_drawdown_threshold, 0.15)  # 15%
        self.assertEqual(self.stress_handler.volatility_spike_threshold, 3.0)  # 3x normal
        self.assertEqual(self.stress_handler.min_sharpe_threshold, 0.5)
        self.assertEqual(self.stress_handler.max_correlation_threshold, 0.9)
        
        # Check initial state
        self.assertFalse(self.stress_handler.circuit_breaker_active)
        self.assertFalse(self.stress_handler.kill_switch_active)
        self.assertIsInstance(self.stress_handler.stress_events, list)
        self.assertIsInstance(self.stress_handler.volatility_history, list)
        
    def test_check_stress_conditions_normal(self):
        """Test stress condition checking under normal conditions."""
        symbol = "BTCUSDT"
        
        is_stressed, conditions = self.stress_handler.check_stress_conditions(symbol)
        
        # Should not be stressed under normal conditions
        self.assertFalse(is_stressed)
        self.assertEqual(len(conditions), 0)
        
    def test_check_stress_conditions_high_drawdown(self):
        """Test stress detection with high drawdown."""
        symbol = "BTCUSDT"
        
        # Mock high drawdown
        self.execution_engine.risk_manager.get_risk_metrics.return_value = {
            'daily_pnl': -1500.0,
            'current_drawdown': -0.20,  # 20% drawdown
            'current_sharpe': 0.3,
            'max_drawdown_hit': True
        }
        
        is_stressed, conditions = self.stress_handler.check_stress_conditions(symbol)
        
        # Should detect stress
        self.assertTrue(is_stressed)
        self.assertGreater(len(conditions), 0)
        self.assertTrue(any('drawdown' in condition.lower() for condition in conditions))
        
    def test_check_stress_conditions_low_sharpe(self):
        """Test stress detection with low Sharpe ratio."""
        symbol = "BTCUSDT"
        
        # Mock low Sharpe ratio
        self.execution_engine.risk_manager.get_risk_metrics.return_value = {
            'daily_pnl': 50.0,
            'current_drawdown': -0.03,
            'current_sharpe': 0.2,  # Below threshold
            'max_drawdown_hit': False
        }
        
        is_stressed, conditions = self.stress_handler.check_stress_conditions(symbol)
        
        # Should detect stress
        self.assertTrue(is_stressed)
        self.assertTrue(any('sharpe' in condition.lower() for condition in conditions))
        
    def test_check_volatility_spike(self):
        """Test volatility spike detection."""
        symbol = "BTCUSDT"
        
        # Add normal volatility history
        normal_volatility = 0.02
        for _ in range(30):
            self.stress_handler.volatility_history.append(normal_volatility)
            
        # Test normal volatility
        current_volatility = 0.025
        is_spike = self.stress_handler.check_volatility_spike(symbol, current_volatility)
        self.assertFalse(is_spike)
        
        # Test volatility spike
        spike_volatility = 0.08  # 4x normal
        is_spike = self.stress_handler.check_volatility_spike(symbol, spike_volatility)
        self.assertTrue(is_spike)
        
    def test_check_correlation_breakdown(self):
        """Test correlation breakdown detection."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        
        # Normal correlations
        normal_correlations = {
            ("BTCUSDT", "ETHUSDT"): 0.7,
            ("BTCUSDT", "XRPUSDT"): 0.6,
            ("ETHUSDT", "XRPUSDT"): 0.8
        }
        
        is_breakdown = self.stress_handler.check_correlation_breakdown(symbols, normal_correlations)
        self.assertFalse(is_breakdown)
        
        # High correlations (breakdown)
        high_correlations = {
            ("BTCUSDT", "ETHUSDT"): 0.95,
            ("BTCUSDT", "XRPUSDT"): 0.92,
            ("ETHUSDT", "XRPUSDT"): 0.96
        }
        
        is_breakdown = self.stress_handler.check_correlation_breakdown(symbols, high_correlations)
        self.assertTrue(is_breakdown)
        
    def test_activate_circuit_breaker(self):
        """Test circuit breaker activation."""
        reason = "High drawdown detected"
        
        self.stress_handler.activate_circuit_breaker(reason)
        
        # Should activate circuit breaker
        self.assertTrue(self.stress_handler.circuit_breaker_active)
        self.assertIsInstance(self.stress_handler.circuit_breaker_time, datetime)
        
        # Should log stress event
        self.assertGreater(len(self.stress_handler.stress_events), 0)
        self.assertEqual(self.stress_handler.stress_events[-1]['type'], 'circuit_breaker')
        
    def test_deactivate_circuit_breaker(self):
        """Test circuit breaker deactivation."""
        # First activate
        self.stress_handler.activate_circuit_breaker("Test reason")
        self.assertTrue(self.stress_handler.circuit_breaker_active)
        
        # Then deactivate
        self.stress_handler.deactivate_circuit_breaker()
        self.assertFalse(self.stress_handler.circuit_breaker_active)
        self.assertIsNone(self.stress_handler.circuit_breaker_time)
        
    def test_circuit_breaker_timeout(self):
        """Test circuit breaker automatic timeout."""
        # Activate circuit breaker
        self.stress_handler.activate_circuit_breaker("Test reason")
        
        # Set time to past timeout
        self.stress_handler.circuit_breaker_time = datetime.now() - timedelta(minutes=35)
        
        # Check if should deactivate
        should_deactivate = self.stress_handler.should_deactivate_circuit_breaker()
        self.assertTrue(should_deactivate)
        
        # Test within timeout
        self.stress_handler.circuit_breaker_time = datetime.now() - timedelta(minutes=15)
        should_deactivate = self.stress_handler.should_deactivate_circuit_breaker()
        self.assertFalse(should_deactivate)
        
    def test_activate_kill_switch(self):
        """Test kill switch activation."""
        reason = "Critical system failure"
        
        # Mock emergency shutdown
        self.execution_engine.emergency_shutdown = Mock()
        
        self.stress_handler.activate_kill_switch(reason)
        
        # Should activate kill switch
        self.assertTrue(self.stress_handler.kill_switch_active)
        self.assertIsInstance(self.stress_handler.kill_switch_time, datetime)
        
        # Should trigger emergency shutdown
        self.execution_engine.emergency_shutdown.assert_called_once()
        
        # Should log critical event
        self.assertGreater(len(self.stress_handler.stress_events), 0)
        self.assertEqual(self.stress_handler.stress_events[-1]['type'], 'kill_switch')
        
    def test_check_system_health(self):
        """Test system health monitoring."""
        # Normal health
        health = self.stress_handler.check_system_health()
        
        expected_fields = ['overall_status', 'drawdown_status', 'sharpe_status', 
                          'volatility_status', 'circuit_breaker_active', 'kill_switch_active']
        
        for field in expected_fields:
            self.assertIn(field, health)
            
        # Should be healthy initially
        self.assertEqual(health['overall_status'], 'healthy')
        
    def test_check_system_health_stressed(self):
        """Test system health under stress."""
        # Activate circuit breaker
        self.stress_handler.activate_circuit_breaker("Test stress")
        
        health = self.stress_handler.check_system_health()
        
        # Should show stress
        self.assertEqual(health['overall_status'], 'stressed')
        self.assertTrue(health['circuit_breaker_active'])
        
    def test_add_stress_event(self):
        """Test stress event logging."""
        event_type = "volatility_spike"
        description = "Volatility exceeded 3x normal levels"
        symbol = "BTCUSDT"
        
        initial_count = len(self.stress_handler.stress_events)
        
        self.stress_handler.add_stress_event(event_type, description, symbol)
        
        # Should add event
        self.assertEqual(len(self.stress_handler.stress_events), initial_count + 1)
        
        # Check event details
        event = self.stress_handler.stress_events[-1]
        self.assertEqual(event['type'], event_type)
        self.assertEqual(event['description'], description)
        self.assertEqual(event['symbol'], symbol)
        self.assertIn('timestamp', event)
        
    def test_get_stress_events(self):
        """Test stress event retrieval."""
        # Add some events
        events = [
            ("volatility_spike", "High volatility", "BTCUSDT"),
            ("drawdown", "High drawdown", "ETHUSDT"),
            ("correlation", "Correlation breakdown", None)
        ]
        
        for event_type, description, symbol in events:
            self.stress_handler.add_stress_event(event_type, description, symbol)
            
        # Get all events
        all_events = self.stress_handler.get_stress_events()
        self.assertGreaterEqual(len(all_events), len(events))
        
        # Get filtered events
        btc_events = self.stress_handler.get_stress_events(symbol="BTCUSDT")
        btc_count = len([e for e in all_events if e.get('symbol') == "BTCUSDT"])
        self.assertEqual(len(btc_events), btc_count)
        
        # Get limited events
        limited_events = self.stress_handler.get_stress_events(limit=2)
        self.assertEqual(len(limited_events), 2)
        
    def test_calculate_stress_score(self):
        """Test stress score calculation."""
        symbol = "BTCUSDT"
        
        # Normal conditions
        score = self.stress_handler.calculate_stress_score(symbol)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Should be low under normal conditions
        self.assertLess(score, 0.5)
        
    def test_calculate_stress_score_high_stress(self):
        """Test stress score under high stress."""
        symbol = "BTCUSDT"
        
        # Mock high stress conditions
        self.execution_engine.risk_manager.get_risk_metrics.return_value = {
            'daily_pnl': -2000.0,
            'current_drawdown': -0.25,  # 25% drawdown
            'current_sharpe': 0.1,      # Very low Sharpe
            'max_drawdown_hit': True
        }
        
        # Add volatility spike
        for _ in range(20):
            self.stress_handler.volatility_history.append(0.02)
        self.stress_handler.volatility_history.append(0.08)  # Spike
        
        score = self.stress_handler.calculate_stress_score(symbol)
        
        # Should be high under stress
        self.assertGreater(score, 0.7)
        
    def test_should_reduce_exposure(self):
        """Test exposure reduction decision."""
        symbol = "BTCUSDT"
        
        # Normal conditions - should not reduce
        should_reduce = self.stress_handler.should_reduce_exposure(symbol)
        self.assertFalse(should_reduce)
        
        # High stress - should reduce
        self.stress_handler.activate_circuit_breaker("High stress")
        should_reduce = self.stress_handler.should_reduce_exposure(symbol)
        self.assertTrue(should_reduce)
        
    def test_get_recommended_position_size_adjustment(self):
        """Test position size adjustment recommendations."""
        symbol = "BTCUSDT"
        base_size = 1000.0
        
        # Normal conditions
        adjusted_size = self.stress_handler.get_recommended_position_size_adjustment(
            symbol, base_size
        )
        self.assertEqual(adjusted_size, base_size)  # No adjustment
        
        # Stressed conditions
        self.stress_handler.activate_circuit_breaker("Stress test")
        adjusted_size = self.stress_handler.get_recommended_position_size_adjustment(
            symbol, base_size
        )
        self.assertLess(adjusted_size, base_size)  # Should reduce
        
    def test_monitor_market_microstructure(self):
        """Test market microstructure monitoring."""
        symbol = "BTCUSDT"
        
        # Mock market data
        bid_ask_spread = 0.5
        order_book_depth = 1000.0
        trade_frequency = 50.0
        
        alerts = self.stress_handler.monitor_market_microstructure(
            symbol, bid_ask_spread, order_book_depth, trade_frequency
        )
        
        # Should return list of alerts
        self.assertIsInstance(alerts, list)
        
    def test_monitor_market_microstructure_alerts(self):
        """Test market microstructure alerts."""
        symbol = "BTCUSDT"
        
        # Abnormal market conditions
        bid_ask_spread = 50.0    # Very wide spread
        order_book_depth = 10.0  # Thin book
        trade_frequency = 1.0    # Low frequency
        
        alerts = self.stress_handler.monitor_market_microstructure(
            symbol, bid_ask_spread, order_book_depth, trade_frequency
        )
        
        # Should generate alerts
        self.assertGreater(len(alerts), 0)
        
    def test_update_volatility_history(self):
        """Test volatility history management."""
        symbol = "BTCUSDT"
        
        initial_length = len(self.stress_handler.volatility_history)
        
        # Add volatility data
        self.stress_handler.update_volatility_history(symbol, 0.025)
        
        self.assertEqual(len(self.stress_handler.volatility_history), initial_length + 1)
        
        # Test rolling window (add more than max history)
        for i in range(200):
            self.stress_handler.update_volatility_history(symbol, 0.02 + i * 0.001)
            
        # Should maintain maximum history length
        self.assertLessEqual(len(self.stress_handler.volatility_history), 100)
        
    def test_emergency_protocols(self):
        """Test emergency protocol execution."""
        symbol = "BTCUSDT"
        
        # Mock order executor methods
        self.execution_engine.order_executor.cancel_all_orders = Mock()
        self.execution_engine.order_executor.reduce_all_positions = Mock()
        
        # Execute emergency protocols
        self.stress_handler.execute_emergency_protocols(symbol, "test_emergency")
        
        # Should execute emergency actions
        self.execution_engine.order_executor.cancel_all_orders.assert_called()
        
    def test_recovery_procedures(self):
        """Test recovery procedures."""
        # Activate stress conditions
        self.stress_handler.activate_circuit_breaker("Test stress")
        
        # Mock recovery conditions
        self.execution_engine.risk_manager.get_risk_metrics.return_value = {
            'daily_pnl': 200.0,
            'current_drawdown': -0.02,  # Improved
            'current_sharpe': 1.8,      # Good Sharpe
            'max_drawdown_hit': False
        }
        
        # Check recovery
        can_recover = self.stress_handler.check_recovery_conditions()
        self.assertTrue(can_recover)
        
        # Execute recovery
        self.stress_handler.execute_recovery_procedures()
        
        # Should deactivate circuit breaker
        self.assertFalse(self.stress_handler.circuit_breaker_active)
        
    def test_performance_under_stress(self):
        """Test performance monitoring under stress."""
        symbol = "BTCUSDT"
        
        # Simulate multiple stress checks
        start_time = datetime.now()
        
        for _ in range(1000):
            self.stress_handler.check_stress_conditions(symbol)
            
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete quickly even under load
        self.assertLess(processing_time, 1.0)  # Less than 1 second for 1000 checks
        
    def test_stress_event_history_management(self):
        """Test stress event history management."""
        # Add many events
        for i in range(150):
            self.stress_handler.add_stress_event(
                "test_event", 
                f"Event {i}", 
                "BTCUSDT"
            )
            
        # Should maintain reasonable history size
        self.assertLessEqual(len(self.stress_handler.stress_events), 100)
        
    def test_configuration_updates(self):
        """Test dynamic configuration updates."""
        # Update thresholds
        new_drawdown_threshold = 0.20
        new_volatility_threshold = 2.5
        
        self.stress_handler.update_thresholds(
            max_drawdown_threshold=new_drawdown_threshold,
            volatility_spike_threshold=new_volatility_threshold
        )
        
        # Should update thresholds
        self.assertEqual(self.stress_handler.max_drawdown_threshold, new_drawdown_threshold)
        self.assertEqual(self.stress_handler.volatility_spike_threshold, new_volatility_threshold)
        
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test with None symbol
        is_stressed, conditions = self.stress_handler.check_stress_conditions(None)
        self.assertFalse(is_stressed)
        
        # Test with invalid volatility data
        is_spike = self.stress_handler.check_volatility_spike("BTCUSDT", None)
        self.assertFalse(is_spike)
        
        # Test with empty volatility history
        self.stress_handler.volatility_history = []
        is_spike = self.stress_handler.check_volatility_spike("BTCUSDT", 0.05)
        self.assertFalse(is_spike)  # Should handle gracefully
        
    def test_multi_symbol_stress_monitoring(self):
        """Test stress monitoring across multiple symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT"]
        
        # Check stress across all symbols
        overall_stress = {}
        for symbol in symbols:
            is_stressed, conditions = self.stress_handler.check_stress_conditions(symbol)
            overall_stress[symbol] = is_stressed
            
        # Should handle multiple symbols
        self.assertEqual(len(overall_stress), len(symbols))
        
        # Calculate overall system stress
        system_stress_level = self.stress_handler.calculate_system_stress_level(symbols)
        self.assertGreaterEqual(system_stress_level, 0.0)
        self.assertLessEqual(system_stress_level, 1.0)


if __name__ == "__main__":
    unittest.main()
