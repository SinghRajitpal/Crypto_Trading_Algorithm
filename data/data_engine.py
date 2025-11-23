import asyncio
from data.data_fetcher import DataFetcher
from data.return_manager import ReturnManager
from data.universe_selector import UniverseSelector, BarValidator
from data.universe_data import UniverseDataFetcher
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
    """Data Engine for collecting and providing market data.
    
    This class is responsible for:
    1. Fetching market data from exchanges
    2. Processing and storing the data in a standardized format
    3. Providing access to the data for analysis
    
    Attributes:
        binance_client: Binance client instance for market data.
        data_fetcher: Market data fetcher instance.
        running: Boolean indicating if the engine is running.
    """
    
    def __init__(self, binance_client, max_candles: int = 100):
        """Initialize the data engine.
        
        Args:
            binance_client: Binance client instance.
            max_candles: Maximum number of candles to store.
        """
        
        # Store binance client reference
        self.binance_client = binance_client
        self.primary_timeframe = config.PRIMARY_TIMEFRAME
        self.default_symbols = [symbol for symbol, _ in config.symbols]

        # Ensure we always keep enough candles for regression/risk windows
        required_candles = max(max_candles, config.RISK_WINDOW + 50)
        
        # Setup data fetcher with the client
        self.data_fetcher = DataFetcher(
            binance_client=binance_client,
            max_candles=required_candles,
            symbol_timeframes=config.symbols,
        )
        self.running = False

        # Rolling data helpers
        self.bar_validator = BarValidator(
            max_abs_return=config.BAR_RETURN_ABS_THRESHOLD,
            min_volume=config.MIN_BAR_VOLUME,
        )
        self.return_manager = ReturnManager(
            regression_window=config.REGRESSION_WINDOW,
            risk_window=config.RISK_WINDOW,
            feature_mode=config.REGRESSION_FEATURE_MODE,
        )
        self.universe_selector = UniverseSelector(
            max_rank=config.UNIVERSE_MAX_RANK,
            min_dollar_volume=config.UNIVERSE_MIN_DOLLAR_VOLUME,
            lookback_days=config.UNIVERSE_LOOKBACK_DAYS,
            default_universe=self.default_symbols,
        )
        self.universe_data_fetcher = UniverseDataFetcher(config.UNIVERSE_MAX_RANK)
        
        logger.info(f"DataEngine initialized with max_candles={required_candles}")
        
    async def run(self):
        """Run the data engine to continuously collect market data.
        
        This method starts the data fetcher and keeps it running until stopped.
        """
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
        """Get OHLCV candle data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data.
            
        Returns:
            List of OHLCV candles.
        """
        return self.data_fetcher.get_candles(symbol, timeframe)
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[List[float]]:
        """Get the latest OHLCV candle for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data.
            
        Returns:
            The latest OHLCV candle or None if no data available.
        """
        candles = self.get_candles(symbol, timeframe)
        if candles and len(candles) > 0:
            return candles[-1]
        return None
    
    def get_latest_price(
        self, symbol: str, timeframe: Optional[str] = None
    ) -> Optional[float]:
        """Get the latest price for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data (default: "1m").
            
        Returns:
            The latest close price or None if no data available.
        """
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

        if not self.bar_validator.is_valid(symbol, bar, prev_close):
            return None

        self.return_manager.update(symbol, bar)
        self.universe_selector.record_bar_metrics(
            symbol, bar["timestamp"], bar["close"], bar["volume"]
        )
        self.universe_selector.refresh_if_needed(bar["timestamp"])
        return bar

    def process_bar_on_grid(
        self, symbol: str, candle: List[float], timeframe: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """Process an arbitrary candle ensuring alignment to global bar grid."""
        timeframe = timeframe or self.primary_timeframe
        if timeframe != config.BAR_GRID_TIMEFRAME:
            return None
        bar = self.extract_ohlcv(candle)
        prev_close = self.return_manager.get_last_close(symbol)

        if not self.bar_validator.is_valid(symbol, bar, prev_close):
            return None

        self.return_manager.update(symbol, bar)
        self.universe_selector.record_bar_metrics(
            symbol, bar["timestamp"], bar["close"], bar["volume"]
        )
        self.universe_selector.refresh_if_needed(bar["timestamp"])
        return bar

    def process_all_latest_bars(self, timeframe: Optional[str] = None) -> None:
        """Convenience helper to ingest the most recent bar for every configured symbol."""
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

    def get_feature_series(
        self, symbol: str, length: Optional[int] = None
    ) -> List[float]:
        """Return the feature vector used by the regression forecaster."""
        return self.return_manager.get_feature_series(
            symbol, length or config.REGRESSION_WINDOW
        )

    def get_feature_matrix(
        self, symbol: str, length: Optional[int] = None
    ):
        """Return standardized-ready feature matrix (X, y, ts, columns) for ridge/OLS."""
        return self.return_manager.get_feature_matrix(
            symbol, length or config.REGRESSION_MAX_BARS
        )

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
            outliers = len(self.return_manager.flag_outliers(sym))
            report[sym] = {"missing_bars": missing, "outliers": outliers}
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
    
    def _bootstrap_return_history(self, symbol: str, timeframe: Optional[str] = None) -> None:
        """Populate return history for a symbol using stored candles."""
        timeframe = timeframe or self.primary_timeframe
        candles = self.get_candles(symbol, timeframe)
        if not candles:
            return

        history_window = max(config.RISK_WINDOW, config.REGRESSION_WINDOW) + 10
        subset = candles[-(history_window + 1):]
        self.return_manager.load_from_candles(symbol, subset)
    
    def initialize_portfolio_volatilities(self, symbols: List[str]) -> Dict[str, float]:
        """Initialize volatility data for portfolio allocation using return history.
        
        This method converts existing return history (or bootstrapped history) 
        into simple standard deviations for each asset.
        
        Args:
            symbols: List of symbols to calculate volatilities for.
            
        Returns:
            Dictionary mapping symbols to their volatility values.
        """
        volatilities = {}
        
        # Default volatilities as fallback (based on crypto market characteristics)
        default_volatilities = {
            'BTCUSDT': 0.015,   # 1.5% - typically lower volatility
            'ETHUSDT': 0.025,   # 2.5% - medium volatility  
            'XRPUSDT': 0.035,   # 3.5% - higher volatility
            'BNBUSDT': 0.018,   # 1.8% - lower volatility
            'SOLUSDT': 0.030    # 3.0% - higher volatility
        }
        logger.info("Calculating return-based volatilities for portfolio initialization...")
        
        for symbol in symbols:
            try:
                returns = self.return_manager.get_return_series(symbol, config.RISK_WINDOW)
                if len(returns) < 2:
                    self._bootstrap_return_history(symbol)
                    returns = self.return_manager.get_return_series(symbol, config.RISK_WINDOW)
                
                if returns and len(returns) >= 2:
                    vol = float(np.std(np.array(returns), ddof=1))
                    vol = max(0.001, min(vol, 0.25))  # Bound to reasonable limits
                    volatilities[symbol] = vol
                    logger.info(f"{symbol}: Return volatility = {vol:.4f} ({vol*100:.2f}%)")
                    continue
                
                default_vol = default_volatilities.get(symbol, 0.02)
                volatilities[symbol] = default_vol
                logger.warning(f"{symbol}: Using default volatility = {default_vol:.4f} ({default_vol*100:.2f}%) [insufficient data]")
                
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
                default_vol = default_volatilities.get(symbol, 0.02)
                volatilities[symbol] = default_vol
                logger.warning(f"{symbol}: Using default volatility = {default_vol:.4f} ({default_vol*100:.2f}%) [error fallback]")
        
        return volatilities


if __name__ == "__main__":
    from binance_exchange import BinanceClient
    
    # Create a standalone instance
    client = BinanceClient(testnet=True)
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
