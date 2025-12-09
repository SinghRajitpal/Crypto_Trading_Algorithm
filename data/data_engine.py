import asyncio
from data.data_fetcher import DataFetcher
from data.return_manager import ReturnManager
from data.universe_selector import UniverseSelector
from data.universe_data import UniverseDataFetcher
from data.feature_builder import FeatureEngineer, FeatureWindowStore
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import config
import time

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

class DataEngine:
    """Collects market data, builds features, manages universe selection, and serves rolling windows."""
    
    def __init__(self, binance_client, max_candles: int = 100, track_timestamps: bool = True):
        """Initialize the data engine.
        
        Args:
            binance_client: Binance client instance.
            max_candles: Maximum number of candles to store.
        """
        
        # Store binance client reference
        self.binance_client = binance_client
        self.primary_timeframe = config.PRIMARY_TIMEFRAME
        self.default_symbols = [symbol for symbol, _ in config.symbols]

        # Ensure we always keep enough candles for risk and feature lookbacks
        required_candles = max(max_candles, config.RISK_WINDOW + 50, config.GRU_LOOKBACK + 50)
        
        # Setup data fetcher with the client
        self.data_fetcher = DataFetcher(
            binance_client=binance_client,
            max_candles=required_candles,
            symbol_timeframes=config.symbols,
        )
        self.running = False

        # Rolling data helpers
        self.return_manager = ReturnManager(
            risk_window=config.RISK_WINDOW,
            track_timestamps=track_timestamps,
        )
        self.feature_engine = FeatureEngineer()
        self.feature_store = FeatureWindowStore(maxlen=config.GRU_LOOKBACK + 10)
        self._live_funding: Dict[str, float] = {}
        self.universe_selector = UniverseSelector(
            max_rank=config.UNIVERSE_MAX_RANK,
            min_dollar_volume=config.UNIVERSE_MIN_DOLLAR_VOLUME,
            lookback_days=config.UNIVERSE_LOOKBACK_DAYS,
            default_universe=self.default_symbols,
        )
        self.universe_data_fetcher = UniverseDataFetcher(config.UNIVERSE_MAX_RANK)
        
        logger.info(f"DataEngine initialized with max_candles={required_candles}")
        
    async def run(self):
        """Run the data fetcher loop until cancelled."""
        if self.running:
            logger.warning("DataEngine already running")
            return
            
        self.running = True
        logger.info("Starting data collection")
        
        try:
            await self.data_fetcher.run()
        except asyncio.CancelledError:
            logger.info("Data collection task cancelled")
            self.running = False
            raise
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            self.running = False
            raise
        finally:
            self.running = False
    
    def get_candles(self, symbol: str, timeframe: str) -> List[List[float]]:
        """Return OHLCV candles for a symbol/timeframe pair."""
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[List[float]]:
        """Return the most recent candle or None."""
        candles = self.get_candles(symbol, timeframe)
        if candles and len(candles) > 0:
            return candles[-1]
        return None
    
    def get_latest_price(
        self, symbol: str, timeframe: Optional[str] = None
    ) -> Optional[float]:
        """Return the latest close price for the symbol/timeframe, if present."""
        timeframe = timeframe or self.primary_timeframe
        latest_candle = self.get_latest_candle(symbol, timeframe)
        if latest_candle and len(latest_candle) >= 5:
            return latest_candle[4]  # Close price
        return None

    def process_latest_bar(
        self, symbol: str, timeframe: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """Validate and ingest the latest bar for the requested symbol."""
        timeframe = timeframe or self.primary_timeframe
        candle = self.get_latest_candle(symbol, timeframe)
        if not candle:
            return None

        bar = self.extract_ohlcv(candle)
        prev_close = self.return_manager.get_last_close(symbol)

        self.return_manager.update(symbol, bar)
        aux = {
            "funding": self._live_funding.pop(symbol, 0.0),
        }
        feats = self.feature_engine.update(symbol, bar, aux=aux)
        self.feature_store.append(symbol, feats, self.feature_engine.schema)
        self.universe_selector.record_bar_metrics(
            symbol, bar["timestamp"], bar["close"], bar["volume"]
        )
        self.universe_selector.refresh_if_needed(bar["timestamp"])
        return bar

    def process_all_latest_bars(self, timeframe: Optional[str] = None) -> None:
        """Ingest the most recent bar for every configured symbol at the timeframe."""
        timeframe = timeframe or self.primary_timeframe
        for symbol, tf in self.data_fetcher.symbol_timeframes:
            if tf != timeframe:
                continue
            self.process_latest_bar(symbol, timeframe)

    def update_market_cap_snapshot(self, market_caps: Dict[str, float]) -> None:
        """Forward market-cap snapshots to the universe selector."""
        self.universe_selector.update_market_cap_snapshot(market_caps)
        self.universe_selector.refresh_if_needed(int(time.time() * 1000))

    async def refresh_universe_snapshot(self) -> bool:
        """Fetch and apply a new market-cap snapshot."""
        try:
            snapshot = await self.universe_data_fetcher.fetch_market_caps()
            if not snapshot:
                logger.warning("Universe snapshot fetch returned empty")
                return False
            self.universe_selector.update_market_cap_snapshot(snapshot)
            self.universe_selector.refresh_if_needed(int(time.time() * 1000))
            logger.info("Universe refreshed with %d assets", len(snapshot))
            return True
        except Exception as exc:
            logger.warning(f"Universe snapshot refresh failed: {exc}")
            return False

    def get_active_universe(self) -> List[str]:
        """Return the currently tradable universe."""
        return self.universe_selector.get_active_universe()

    def get_return_matrix(
        self, symbols: Optional[List[str]] = None, window: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Return an aligned return matrix and the symbols that met the data requirement."""
        target_symbols = symbols or self.get_active_universe()
        return self.return_manager.get_return_matrix(
            target_symbols, window or config.RISK_WINDOW
        )

    async def backfill_history(self, timeframe: Optional[str] = None, symbols: Optional[List[str]] = None) -> None:
        """Load historical candles into ReturnManager to satisfy lookback before live loop starts."""
        timeframe = timeframe or self.primary_timeframe
        target_symbols = symbols or [sym for sym, tf in self.data_fetcher.symbol_timeframes if tf == timeframe]
        for sym in target_symbols:
            try:
                # Populate data_processor with historical candles
                await self.data_fetcher._load_history(sym, timeframe)  # pylint: disable=protected-access
                candles = self.data_fetcher.get_candles(sym, timeframe)
                if not candles:
                    continue
                # Ingest into ReturnManager
                self.return_manager.load_from_candles(sym, candles)
                for candle in candles:
                    if len(candle) < 6:
                        continue
                    bar = self.extract_ohlcv(candle)
                    feats = self.feature_engine.update(sym, bar)
                    self.feature_store.append(sym, feats, self.feature_engine.schema)
                # Record volume for universe selector
                for candle in candles:
                    if len(candle) < 6:
                        continue
                    ts_ms, _, _, _, close, volume = candle[:6]
                    self.universe_selector.record_bar_metrics(sym, int(ts_ms), float(close), float(volume))
                # Refresh universe snapshot after backfill
                latest_ts = int(candles[-1][0]) if candles else int(time.time() * 1000)
                self.universe_selector.refresh_if_needed(latest_ts)
            except Exception as exc:
                logger.warning("Backfill failed for %s: %s", sym, exc)

    def get_missing_bars(self, symbol: str, timeframe: Optional[str] = None):
        """Expose missing bar timestamps detected by DataProcessor."""
        timeframe = timeframe or self.primary_timeframe
        return self.data_fetcher.data_processor.get_missing_bars(symbol, timeframe)

    def data_quality_report(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        """Return simple data quality metrics per symbol."""
        report: Dict[str, Dict[str, int]] = {}
        target_symbols = symbols or self.get_active_universe()
        for sym in target_symbols:
            missing = len(self.get_missing_bars(sym, self.primary_timeframe))
            report[sym] = {"missing_bars": missing, "outliers": 0}
        return report
    
    def get_return_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the rolling simple returns for the symbol."""
        return self.return_manager.get_return_series(
            symbol, length or config.RISK_WINDOW
        )

    def get_price_series(self, symbol: str, length: Optional[int] = None) -> List[float]:
        """Return the rolling close prices for diagnostics."""
        return self.return_manager.get_price_series(symbol, length)

    def get_latest_return(self, symbol: str) -> Optional[float]:
        """Expose the latest validated simple return for the symbol."""
        return self.return_manager.get_latest_return(symbol)

    def get_feature_window(self, symbol: str, lookback: Optional[int] = None) -> Optional[np.ndarray]:
        """Return a ready-to-infer feature window for GRU."""
        lookback = lookback or config.GRU_LOOKBACK
        return self.feature_store.get_window(symbol, lookback)

    def feature_ready(self, symbol: str, lookback: Optional[int] = None) -> bool:
        """Return True if a finite feature window of the requested length exists."""
        return self.get_feature_window(symbol, lookback) is not None

    def set_latest_funding(self, symbol: str, value: float) -> None:
        """Inject latest funding rate for next feature update (per symbol)."""
        self._live_funding[symbol.upper()] = float(value)


    def dispose(self) -> None:
        """Release stored state to help long-running batch jobs."""
        try:
            self.return_manager.clear()
        except Exception:
            pass
        try:
            self.data_fetcher.data_processor.clear()
        except Exception:
            pass
    
    @staticmethod
    def extract_ohlcv(candle: List[float]) -> Dict[str, float]:
        """Extract OHLCV values from a candle into a dictionary.
        
        Args:
            candle: OHLCV candle data.
            
        Returns:
            Dictionary with timestamp, open, high, low, close, and volume.
        """
        if not candle or len(candle) < 6:
            return {}
            
        return {
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5]
        }
        
    @staticmethod
    def get_candle_change_pct(candle: List[float]) -> float:
        """Calculate the percentage change in a candle.
        
        Args:
            candle: OHLCV candle data.
            
        Returns:
            Percentage change from open to close.
        """
        if not candle or len(candle) < 5 or candle[1] == 0:
            return 0.0
            
        return (candle[4] - candle[1]) / candle[1] * 100  # (close - open) / open * 100


if __name__ == "__main__":
    from binance_exchange import BinanceClient
    
    # Create a standalone instance
    client = BinanceClient(demo=True)
    data_engine = DataEngine(binance_client=client, max_candles=100)
    
    # Run this to collect some data first
    logger.info("Starting data collection. Press Ctrl+C to stop...")
    try:
        asyncio.run(data_engine.run())
    except KeyboardInterrupt:
        # Stopping the data collection
        logger.info("Stopped data collection.")
    finally:
        # Make sure we close the connection
        logger.info("Closing connection...")
        asyncio.run(client.close())
