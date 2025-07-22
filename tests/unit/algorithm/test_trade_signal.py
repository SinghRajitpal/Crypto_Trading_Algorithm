"""
Unit tests for TradeSignal module.

This test suite covers:
1. Signal initialization and validation
2. Post-initialization validation rules
3. Signal data integrity
4. Edge cases and error handling
5. Signal combinations for futures trading
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from algorithm.trade_signal import TradeSignal


class TestTradeSignal(unittest.TestCase):
    """Test cases for TradeSignal class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_metadata = {
            "reason": "MA crossover",
            "fast_ma": 42100.0,
            "slow_ma": 42000.0,
            "atr_value": 150.0
        }
    
    def test_valid_signal_creation(self):
        """Test creation of valid trade signals."""
        # Test open/buy signal
        signal_open_buy = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="ma_crossover",
            metadata=self.valid_metadata,
            signal_confidence=0.8
        )
        
        self.assertEqual(signal_open_buy.action, "open")
        self.assertEqual(signal_open_buy.side, "buy")
        self.assertEqual(signal_open_buy.symbol, "BTCUSDT")
        self.assertEqual(signal_open_buy.strategy_id, "ma_crossover")
        self.assertEqual(signal_open_buy.signal_confidence, 0.8)
        self.assertEqual(signal_open_buy.metadata, self.valid_metadata)
        self.assertEqual(signal_open_buy.timestamp, 0)  # Default value
    
    def test_all_valid_signal_combinations(self):
        """Test all valid signal combinations for futures trading."""
        valid_combinations = [
            ("open", "buy"),    # Enter long position
            ("open", "sell"),   # Enter short position
            ("exit", "sell"),   # Exit long position
            ("exit", "buy"),    # Exit short position
            ("hold", "none"),   # No action needed
        ]
        
        for action, side in valid_combinations:
            signal = TradeSignal(
                action=action,
                side=side,
                symbol="BTCUSDT",
                strategy_id="test",
                metadata={"reason": f"Test {action}/{side}"},
                signal_confidence=0.5
            )
            
            self.assertEqual(signal.action, action)
            self.assertEqual(signal.side, side)
    
    def test_signal_with_timestamp(self):
        """Test signal creation with custom timestamp."""
        timestamp = 1642680000000  # Millisecond timestamp
        
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test with timestamp"},
            signal_confidence=0.7,
            timestamp=timestamp
        )
        
        self.assertEqual(signal.timestamp, timestamp)
    
    def test_signal_confidence_values(self):
        """Test various signal confidence values."""
        confidence_values = [0.0, 0.1, 0.5, 0.99, 1.0]
        
        for confidence in confidence_values:
            signal = TradeSignal(
                action="hold",
                side="none",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata={"reason": f"Confidence {confidence}"},
                signal_confidence=confidence
            )
            
            self.assertEqual(signal.signal_confidence, confidence)
    
    def test_invalid_action_validation(self):
        """Test validation of invalid action values."""
        invalid_actions = ["invalid", "buy", "sell", "", "OPEN", "EXIT"]
        
        for invalid_action in invalid_actions:
            with self.assertRaises(ValueError) as context:
                TradeSignal(
                    action=invalid_action,
                    side="buy",
                    symbol="BTCUSDT",
                    strategy_id="test",
                    metadata={"reason": "Test"},
                    signal_confidence=0.5
                )
            
            self.assertIn("Action must be either 'open', 'exit', or 'hold'", str(context.exception))
    
    def test_invalid_side_for_hold_validation(self):
        """Test validation of side when action is hold."""
        invalid_sides_for_hold = ["buy", "sell", "long", "short", ""]
        
        for invalid_side in invalid_sides_for_hold:
            with self.assertRaises(ValueError) as context:
                TradeSignal(
                    action="hold",
                    side=invalid_side,
                    symbol="BTCUSDT",
                    strategy_id="test",
                    metadata={"reason": "Test"},
                    signal_confidence=0.5
                )
            
            self.assertIn("Side must be 'none' when action is 'hold'", str(context.exception))
    
    def test_invalid_side_for_open_exit_validation(self):
        """Test validation of side for open/exit actions."""
        invalid_sides = ["none", "long", "short", "invalid", ""]
        actions = ["open", "exit"]
        
        for action in actions:
            for invalid_side in invalid_sides:
                with self.assertRaises(ValueError) as context:
                    TradeSignal(
                        action=action,
                        side=invalid_side,
                        symbol="BTCUSDT",
                        strategy_id="test",
                        metadata={"reason": "Test"},
                        signal_confidence=0.5
                    )
                
                self.assertIn("Side must be either 'buy' or 'sell' for open/exit actions", str(context.exception))
    
    def test_empty_symbol_validation(self):
        """Test validation of empty symbol."""
        empty_symbols = ["", None]
        
        for empty_symbol in empty_symbols:
            with self.assertRaises(ValueError) as context:
                TradeSignal(
                    action="open",
                    side="buy",
                    symbol=empty_symbol,
                    strategy_id="test",
                    metadata={"reason": "Test"},
                    signal_confidence=0.5
                )
            
            self.assertIn("Symbol must be specified", str(context.exception))
    
    def test_empty_strategy_id_validation(self):
        """Test validation of empty strategy_id."""
        empty_strategy_ids = ["", None]
        
        for empty_id in empty_strategy_ids:
            with self.assertRaises(ValueError) as context:
                TradeSignal(
                    action="open",
                    side="buy",
                    symbol="BTCUSDT",
                    strategy_id=empty_id,
                    metadata={"reason": "Test"},
                    signal_confidence=0.5
                )
            
            self.assertIn("Strategy ID must be specified", str(context.exception))
    
    def test_metadata_types(self):
        """Test various metadata types."""
        test_metadata_cases = [
            {},  # Empty dict
            {"reason": "Simple string"},
            {"price": 42000.0, "volume": 100, "active": True},
            {"nested": {"key": "value"}, "list": [1, 2, 3]},
            {"complex": {"prices": [42000.0, 42100.0], "indicators": {"ma": 42050.0}}}
        ]
        
        for metadata in test_metadata_cases:
            signal = TradeSignal(
                action="hold",
                side="none",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata=metadata,
                signal_confidence=0.5
            )
            
            self.assertEqual(signal.metadata, metadata)
    
    def test_signal_immutability_intention(self):
        """Test that signal fields can be modified (dataclass is mutable by default)."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Original"},
            signal_confidence=0.5
        )
        
        # Dataclass fields are mutable by default
        signal.timestamp = 1642680000000
        signal.metadata["reason"] = "Modified"
        
        self.assertEqual(signal.timestamp, 1642680000000)
        self.assertEqual(signal.metadata["reason"], "Modified")
    
    def test_signal_string_representation(self):
        """Test string representation of signals."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="ma_crossover",
            metadata={"reason": "MA crossover bullish"},
            signal_confidence=0.85,
            timestamp=1642680000000
        )
        
        signal_str = str(signal)
        
        # Should contain key information
        self.assertIn("open", signal_str)
        self.assertIn("buy", signal_str)
        self.assertIn("BTCUSDT", signal_str)
        self.assertIn("ma_crossover", signal_str)
    
    def test_signal_equality(self):
        """Test signal equality comparison."""
        signal1 = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test"},
            signal_confidence=0.8,
            timestamp=1642680000000
        )
        
        signal2 = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test"},
            signal_confidence=0.8,
            timestamp=1642680000000
        )
        
        signal3 = TradeSignal(
            action="exit",
            side="sell",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Different"},
            signal_confidence=0.6,
            timestamp=1642680060000
        )
        
        # Same signals should be equal
        self.assertEqual(signal1, signal2)
        
        # Different signals should not be equal
        self.assertNotEqual(signal1, signal3)
    
    def test_signal_hashing(self):
        """Test signal hashing capability."""
        signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test"},
            signal_confidence=0.8
        )
        
        # Should be able to hash (needed for sets, dict keys)
        try:
            hash(signal)
        except TypeError:
            # Expected - dataclass with mutable metadata can't be hashed by default
            pass
    
    def test_extreme_confidence_values(self):
        """Test extreme confidence values."""
        # Test values outside typical range (though not explicitly validated)
        extreme_values = [-1.0, 2.0, 100.0, -0.5]
        
        for confidence in extreme_values:
            signal = TradeSignal(
                action="hold",
                side="none",
                symbol="BTCUSDT",
                strategy_id="test",
                metadata={"reason": f"Extreme confidence {confidence}"},
                signal_confidence=confidence
            )
            
            self.assertEqual(signal.signal_confidence, confidence)
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        special_cases = [
            ("BTC/USDT", "slash_symbol"),
            ("BTC-USDT", "dash_symbol"),
            ("strategy_with_underscore", "underscore_strategy"),
            ("策略", "unicode_strategy"),  # Chinese characters
            ("🚀📈", "emoji_strategy"),    # Emojis
        ]
        
        for symbol, strategy_id in special_cases:
            signal = TradeSignal(
                action="hold",
                side="none",
                symbol=symbol,
                strategy_id=strategy_id,
                metadata={"reason": f"Test with {symbol}"},
                signal_confidence=0.5
            )
            
            self.assertEqual(signal.symbol, symbol)
            self.assertEqual(signal.strategy_id, strategy_id)
    
    def test_large_metadata(self):
        """Test handling of large metadata objects."""
        large_metadata = {}
        
        # Create large metadata
        for i in range(1000):
            large_metadata[f"key_{i}"] = f"value_{i}"
            large_metadata[f"price_{i}"] = 42000.0 + i
        
        signal = TradeSignal(
            action="hold",
            side="none",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata=large_metadata,
            signal_confidence=0.5
        )
        
        self.assertEqual(len(signal.metadata), 2000)  # 1000 string keys + 1000 price keys
    
    def test_signal_copying(self):
        """Test signal copying behavior."""
        import copy
        
        original = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"nested": {"key": "value"}},
            signal_confidence=0.8
        )
        
        # Shallow copy
        shallow_copy = copy.copy(original)
        self.assertEqual(original, shallow_copy)
        self.assertIs(original.metadata, shallow_copy.metadata)  # Same reference
        
        # Deep copy
        deep_copy = copy.deepcopy(original)
        self.assertEqual(original, deep_copy)
        self.assertIsNot(original.metadata, deep_copy.metadata)  # Different reference
    
    def test_post_init_validation_timing(self):
        """Test that post-init validation happens after field assignment."""
        # This should work - all fields set before validation
        valid_signal = TradeSignal(
            action="open",
            side="buy",
            symbol="BTCUSDT",
            strategy_id="test",
            metadata={"reason": "Test"},
            signal_confidence=0.8
        )
        
        self.assertIsNotNone(valid_signal)
        
        # Direct instantiation bypassing __init__ should still trigger __post_init__
        signal = TradeSignal.__new__(TradeSignal)
        signal.action = "invalid"
        signal.side = "buy"
        signal.symbol = "BTCUSDT"
        signal.strategy_id = "test"
        signal.metadata = {"reason": "Test"}
        signal.signal_confidence = 0.8
        signal.timestamp = 0
        
        with self.assertRaises(ValueError):
            signal.__post_init__()


if __name__ == '__main__':
    unittest.main(verbosity=2)
