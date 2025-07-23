#!/usr/bin/env python3
"""
Comprehensive pytest unit tests for Stress Handler.

This module tests the StressHandlingModule implementation against
the safeguards specified in the trading document.

Tests cover:
- Flash crash detection and response
- Kill switch activation thresholds
- Liquidity filters
- Connection lag handling
- Slippage monitoring
- Emergency shutdown procedures
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from execution.stress_handler import StressHandlingModule, StressEvent


@pytest.fixture
def stress_handler():
    """Fixture providing a StressHandlingModule instance for testing."""
    mock_execution_engine = Mock()
    return StressHandlingModule(mock_execution_engine)


class TestStressHandlingModule:
    """Test suite for StressHandlingModule unit tests."""
    
    def test_initialization(self):
        """Test stress handler initialization with correct thresholds."""
        mock_execution_engine = Mock()
        stress_handler = StressHandlingModule(mock_execution_engine)
        
        # Test initialization values from document
        assert stress_handler.connection_lag_threshold == 3.0  # 3 seconds
        assert stress_handler.slippage_threshold == 0.002     # 0.2%
        assert stress_handler.min_daily_volume == 5_000_000   # $5M
        assert stress_handler.max_spread == 0.0015           # 0.15%
        assert stress_handler.max_funding_rate == 0.004      # 0.4%
        
        # Test initial state
        assert stress_handler.flash_crash_events == []
        assert stress_handler.affected_assets_60s == set()
        assert not stress_handler.forward_fill_active
        assert len(stress_handler.stress_events) == 0
        
        # Test kill switch initial states
        expected_switches = ["drawdown_partial", "drawdown_full", "equity_slope", "emergency_halt"]
        for switch in expected_switches:
            assert switch in stress_handler.kill_switches
            assert not stress_handler.kill_switches[switch]
            
    def test_flash_crash_detection(self, stress_handler):
        """Test flash crash detection: If 1-min drop >4×ATR, flatten asset."""
        atr_value = 0.02  # 2% ATR
        
        # Test normal price movement (should not trigger)
        normal_drop = 0.03  # 3% drop < 4×ATR (8%)
        is_flash_crash = stress_handler.check_flash_crash("BTCUSDT", normal_drop, atr_value)
        assert not is_flash_crash, "Normal price movement should not trigger flash crash"
        
        # Test flash crash (should trigger)
        flash_drop = 0.09  # 9% drop > 4×ATR (8%)
        is_flash_crash = stress_handler.check_flash_crash("BTCUSDT", flash_drop, atr_value)
        assert is_flash_crash, "Large drop should trigger flash crash detection"
        
        # Verify event was recorded
        assert len(stress_handler.flash_crash_events) == 1
        assert "BTCUSDT" in stress_handler.affected_assets_60s
        
        # Test with price data dictionary format
        price_data = {"high": 50000.0, "low": 45000.0}  # 10% drop
        is_flash_crash_dict = stress_handler.check_flash_crash("ETHUSDT", price_data, atr_value)
        assert is_flash_crash_dict, "Dictionary format should also work"
        
    def test_multiple_flash_crashes(self, stress_handler):
        """Test portfolio de-risking when >5 assets flash crash in 60s."""
        atr_value = 0.02
        flash_drop = 0.09  # 9% drop > 4×ATR
        
        # Trigger flash crashes on multiple assets
        symbols = ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "DOT"]  # 7 symbols
        
        for symbol in symbols:
            stress_handler.check_flash_crash(symbol, flash_drop, atr_value)
        
        # Should trigger portfolio de-risking after 6th asset (>5)
        assert len(stress_handler.flash_crash_events) == 7
        
        # Check for major stress event recording
        major_events = [e for e in stress_handler.stress_events 
                       if e.event_type == "multi_flash_crash"]
        assert len(major_events) >= 1, "Should record major stress event for multiple crashes"
        
    def test_kill_switch_thresholds(self, stress_handler):
        """Test kill switch activation: DD >14% flatten 30%, slope <-10% full flatten."""
        # Test drawdown kill switch
        drawdown_15_percent = 0.15  # 15% > 14% threshold
        switches = stress_handler.check_kill_switches(drawdown_15_percent, -0.05)
        
        assert "drawdown_partial" in switches, "Should trigger drawdown kill switch at 15%"
        assert stress_handler.kill_switches["drawdown_partial"], "Drawdown kill switch should be active"
        
        # Test equity slope kill switch
        bad_slope = -0.12  # -12% < -10% threshold
        switches = stress_handler.check_kill_switches(0.10, bad_slope)
        
        assert "equity_slope" in switches, "Should trigger equity slope kill switch at -12%"
        assert stress_handler.kill_switches["equity_slope"], "Equity slope kill switch should be active"
        
        # Test no trigger below thresholds
        switches = stress_handler.check_kill_switches(0.10, -0.05)  # Below thresholds
        assert len(switches) == 0, "Should not trigger kill switches below thresholds"
        
    def test_kill_switch_helper_method(self, stress_handler):
        """Test helper method for kill switch detection."""
        # Test below threshold
        assert not stress_handler.should_trigger_kill_switch(0.10), "Should not trigger at 10%"
        assert not stress_handler.should_trigger_kill_switch(0.14), "Should not trigger at exactly 14%"
        
        # Test above threshold
        assert stress_handler.should_trigger_kill_switch(0.15), "Should trigger at 15%"
        assert stress_handler.should_trigger_kill_switch(0.20), "Should trigger at 20%"
        
    def test_liquidity_filters(self, stress_handler):
        """Test liquidity filters: volume <$5M or spread >0.15%."""
        # Test sufficient liquidity
        good_volume = 10_000_000  # $10M > $5M threshold
        good_spread = 0.001       # 0.1% < 0.15% threshold
        
        is_liquid = stress_handler.check_liquidity_filters(good_volume, good_spread)
        assert is_liquid, "Should pass liquidity filters with good volume and spread"
        
        # Test insufficient volume
        low_volume = 3_000_000   # $3M < $5M threshold
        is_liquid_vol = stress_handler.check_liquidity_filters(low_volume, good_spread)
        assert not is_liquid_vol, "Should fail liquidity filter with low volume"
        
        # Test excessive spread
        high_spread = 0.002      # 0.2% > 0.15% threshold
        is_liquid_spread = stress_handler.check_liquidity_filters(good_volume, high_spread)
        assert not is_liquid_spread, "Should fail liquidity filter with high spread"
        
        # Test both bad
        is_liquid_both = stress_handler.check_liquidity_filters(low_volume, high_spread)
        assert not is_liquid_both, "Should fail with both low volume and high spread"
        
    def test_slippage_monitoring(self, stress_handler):
        """Test slippage monitoring: reject if >0.2% off expected price."""
        expected_price = 50000.0
        
        # Test acceptable slippage
        good_execution = 50050.0  # 0.1% slippage
        is_acceptable = stress_handler.check_slippage(expected_price, good_execution, "BTCUSDT")
        assert is_acceptable, "Should accept slippage within 0.2% threshold"
        
        # Test excessive slippage
        bad_execution = 50150.0  # 0.3% slippage > 0.2%
        is_acceptable_bad = stress_handler.check_slippage(expected_price, bad_execution, "BTCUSDT")
        assert not is_acceptable_bad, "Should reject slippage above 0.2% threshold"
        
        # Verify stress event recorded
        slippage_events = [e for e in stress_handler.stress_events 
                          if e.event_type == "excessive_slippage"]
        assert len(slippage_events) >= 1, "Should record slippage stress event"
        
    def test_connection_lag_handling(self, stress_handler):
        """Test connection lag handling: pause if lag >3s."""
        now = datetime.now()
        
        # Test normal connection
        recent_timestamp = now - timedelta(seconds=1)
        is_healthy = stress_handler.check_connection_lag(recent_timestamp)
        assert is_healthy, "Should report healthy connection with 1s lag"
        assert not stress_handler.forward_fill_active, "Forward fill should not be active"
        
        # Test lagged connection
        old_timestamp = now - timedelta(seconds=5)  # 5s > 3s threshold
        is_healthy_lag = stress_handler.check_connection_lag(old_timestamp)
        assert not is_healthy_lag, "Should report unhealthy connection with 5s lag"
        assert stress_handler.forward_fill_active, "Forward fill should be activated"
        
        # Test connection recovery
        recovery_timestamp = now - timedelta(seconds=1)
        is_recovered = stress_handler.check_connection_lag(recovery_timestamp)
        assert is_recovered, "Should report recovered connection"
        assert not stress_handler.forward_fill_active, "Forward fill should be deactivated"
        
    def test_stress_event_recording(self, stress_handler):
        """Test stress event recording and management."""
        # Initial state
        assert len(stress_handler.stress_events) == 0
        
        # Trigger various stress events
        stress_handler.check_slippage(50000.0, 50200.0, "BTCUSDT")  # Excessive slippage
        stress_handler.check_flash_crash("ETHUSDT", 0.09, 0.02)      # Flash crash
        
        # Verify events recorded
        assert len(stress_handler.stress_events) >= 2, "Should record multiple stress events"
        
        # Check event structure
        for event in stress_handler.stress_events:
            assert isinstance(event, StressEvent)
            assert hasattr(event, 'timestamp')
            assert hasattr(event, 'event_type')
            assert hasattr(event, 'symbol')
            assert hasattr(event, 'severity')
            assert hasattr(event, 'data')
            assert hasattr(event, 'action_taken')
            
    def test_regime_transition_smoothing(self, stress_handler):
        """Test regime transition smoothing with 5-bar EMA."""
        # Test with empty history
        smoothed = stress_handler.smooth_regime_transitions(0.5, [])
        assert smoothed == 0.5, "Should return current value with empty history"
        
        # Test with history
        history = [0.3, 0.4, 0.5, 0.6]
        current = 0.8
        smoothed = stress_handler.smooth_regime_transitions(current, history)
        
        # Should be smoothed (between current and historical average)
        assert isinstance(smoothed, float)
        assert 0.3 <= smoothed <= 0.8, "Smoothed value should be within reasonable range"
        
    def test_edge_cases_and_error_handling(self, stress_handler):
        """Test edge cases and error handling."""
        # Test with None values
        is_flash_crash = stress_handler.check_flash_crash("TESTSYM", 0.0, 0.02)
        assert not is_flash_crash, "Should handle zero drop gracefully"
        
        # Test with negative values
        negative_result = stress_handler.check_slippage(50000.0, -1000.0, "TESTSYM")
        assert not negative_result, "Should reject negative execution price"
        
        # Test with extreme values
        extreme_vol = stress_handler.check_liquidity_filters(1e12, 0.5)  # Extreme values
        assert not extreme_vol, "Should handle extreme values appropriately"
        
        # Test kill switch reset conditions
        stress_handler.kill_switches["drawdown_partial"] = True
        stress_handler.check_kill_switches(0.05, 0.02)  # Good conditions
        # Kill switch should eventually reset (implementation dependent)
        
    @pytest.mark.parametrize("drop_pct,atr,should_trigger", [
        (0.02, 0.01, False),   # 2% drop, 1% ATR -> 2x < 4x
        (0.04, 0.01, True),    # 4% drop, 1% ATR -> 4x = 4x (trigger)
        (0.05, 0.01, True),    # 5% drop, 1% ATR -> 5x > 4x
        (0.08, 0.02, True),    # 8% drop, 2% ATR -> 4x = 4x
        (0.06, 0.02, False),   # 6% drop, 2% ATR -> 3x < 4x
    ])
    def test_flash_crash_parametrized(self, stress_handler, drop_pct, atr, should_trigger):
        """Parametrized test for flash crash detection thresholds."""
        result = stress_handler.check_flash_crash("TESTBTC", drop_pct, atr)
        assert result == should_trigger, \
            f"Drop {drop_pct:.1%} with ATR {atr:.1%} should {'trigger' if should_trigger else 'not trigger'}"
            
    @pytest.mark.parametrize("drawdown,should_trigger", [
        (0.10, False),  # 10% < 14%
        (0.14, False),  # 14% = 14% (not greater)
        (0.15, True),   # 15% > 14%
        (0.20, True),   # 20% > 14%
    ])
    def test_kill_switch_parametrized(self, stress_handler, drawdown, should_trigger):
        """Parametrized test for kill switch thresholds."""
        result = stress_handler.should_trigger_kill_switch(drawdown)
        assert result == should_trigger, \
            f"Drawdown {drawdown:.1%} should {'trigger' if should_trigger else 'not trigger'} kill switch"
            
    @pytest.mark.parametrize("volume,spread,expected", [
        (10_000_000, 0.001, True),   # Good volume, good spread
        (3_000_000, 0.001, False),   # Bad volume, good spread
        (10_000_000, 0.002, False),  # Good volume, bad spread
        (3_000_000, 0.002, False),   # Bad volume, bad spread
        (5_000_000, 0.0015, True),   # Exactly at thresholds (should pass)
    ])
    def test_liquidity_filters_parametrized(self, stress_handler, volume, spread, expected):
        """Parametrized test for liquidity filters."""
        result = stress_handler.check_liquidity_filters(volume, spread)
        assert result == expected, \
            f"Volume ${volume:,.0f} and spread {spread:.3%} should {'pass' if expected else 'fail'}"
            
    def test_stress_handler_state_management(self, stress_handler):
        """Test stress handler state management and consistency."""
        # Test initial state
        initial_state = stress_handler.get_system_status()
        assert "total_stress_events" in initial_state
        assert "recent_events_count" in initial_state
        assert "active_connections" in initial_state
        
        # Modify state through various stress events
        stress_handler.check_kill_switches(0.16, -0.05)  # Trigger drawdown
        stress_handler.check_connection_lag(datetime.now() - timedelta(seconds=5))  # Trigger lag
        
        # Check updated state
        updated_state = stress_handler.get_system_status()
        # Verify actual kill switch is activated in the internal state
        assert stress_handler.kill_switches["drawdown_partial"], "Drawdown kill switch should be active"
        assert updated_state["total_stress_events"] > 0
        
        # Test state consistency
        assert isinstance(updated_state, dict)
        assert all(isinstance(v, (bool, int, dict, datetime, str)) for v in updated_state.values())
        
    def test_concurrent_stress_conditions(self, stress_handler):
        """Test handling of multiple concurrent stress conditions."""
        # Trigger multiple stress conditions simultaneously
        current_time = datetime.now()
        
        # Flash crash
        stress_handler.check_flash_crash("BTCUSDT", 0.10, 0.02)
        
        # High slippage
        stress_handler.check_slippage(50000.0, 50200.0, "ETHUSDT")
        
        # Connection lag
        stress_handler.check_connection_lag(current_time - timedelta(seconds=5))
        
        # Kill switch
        stress_handler.check_kill_switches(0.16, -0.05)
        
        # Verify system handles multiple stress conditions
        assert len(stress_handler.stress_events) >= 2, "Should record multiple stress events"
        assert stress_handler.forward_fill_active, "Forward fill should be active"
        assert stress_handler.kill_switches["drawdown_partial"], "Kill switch should be active"
        
        # System should still be functional
        system_status = stress_handler.get_system_status()
        assert isinstance(system_status, dict)
        assert "total_stress_events" in system_status
