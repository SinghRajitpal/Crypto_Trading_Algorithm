"""Data module for market data processing."""

# Lazily import heavy dependencies to allow partial usage in test environments.
try:
    from .data_engine import DataEngine
    from .data_fetcher import DataFetcher
    from .return_manager import ReturnManager
    from .processor import DataProcessor
    from .historical_data import HistoricalDataFetcher
except Exception:
    DataEngine = None  # type: ignore
    DataFetcher = None  # type: ignore
    ReturnManager = None  # type: ignore
    DataProcessor = None  # type: ignore
    HistoricalDataFetcher = None  # type: ignore

__all__ = [
    "DataEngine",
    "DataFetcher",
    "ReturnManager",
    "DataProcessor",
    "HistoricalDataFetcher",
]
