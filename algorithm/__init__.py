"""Algorithm module for trading strategies and signal generation.

This module provides components for implementing trading strategies,
generating trading signals, and processing market data for decision making.
"""

# Algorithm package initializer
from .algo_engine import AlgoEngine
from .strategies.mean_variance import MeanVarianceForecastStrategy
from .forecast.linear_regression import LinearRegressionForecaster
from .forecast.forecast_result import ForecastResult

__all__ = [
    'AlgoEngine',
    'MeanVarianceForecastStrategy',
    'LinearRegressionForecaster',
    'ForecastResult',
] 
