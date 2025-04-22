"""Algorithm module for trading strategies and signal generation.

This module provides components for implementing trading strategies,
generating trading signals, and processing market data for decision making.
"""

# Algorithm package initializer
from .algo_engine import AlgoEngine
from .trade_signal import TradeSignal

__all__ = ['AlgoEngine', 'TradeSignal'] 