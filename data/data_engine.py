import asyncio
from data.data_fetcher import DataFetcher
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
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
        # Import here to avoid circular imports
        from data.data_fetcher import DataFetcher
        
        # Store binance client reference
        self.binance_client = binance_client
        
        # Setup data fetcher with the client
        self.data_fetcher = DataFetcher(binance_client=binance_client, max_candles=max_candles)
        self.running = False
        
        logger.info(f"DataEngine initialized with max_candles={max_candles}")
        
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
    
    def get_latest_price(self, symbol: str, timeframe: str = "1m") -> Optional[float]:
        """Get the latest price for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe for the data (default: "1m").
            
        Returns:
            The latest close price or None if no data available.
        """
        latest_candle = self.get_latest_candle(symbol, timeframe)
        if latest_candle and len(latest_candle) >= 5:
            return latest_candle[4]  # Close price
        return None
    
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
    
    def calculate_atr_volatility(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate ATR-based volatility for a symbol using real market data.
        
        This method calculates the Average True Range (ATR) and converts it to 
        percentage volatility for portfolio allocation purposes.
        
        Args:
            symbol: Trading pair symbol.
            period: ATR calculation period (default: 14).
            
        Returns:
            ATR as percentage of current price, or None if insufficient data.
        """
        try:
            candles = self.get_candles(symbol, "1m")
            if not candles or len(candles) < period + 5:
                logger.warning(f"Insufficient candles for {symbol} ATR calculation: {len(candles) if candles else 0} available, need {period + 5}")
                return None
            
            # Import indicators for ATR calculation
            from data.indicators import Indicators
            import numpy as np
            
            indicators = Indicators()
            
            # Convert candles to OHLCV format for ATR calculation
            ohlcv_data = []
            for candle in candles[-(period + 10):]:  # Use extra candles for stable calculation
                if len(candle) >= 6:
                    try:
                        ohlcv_data.append([
                            candle[0],          # timestamp
                            float(candle[1]),   # open
                            float(candle[2]),   # high  
                            float(candle[3]),   # low
                            float(candle[4]),   # close
                            float(candle[5])    # volume
                        ])
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error processing candle data for {symbol}: {e}")
                        continue
            
            if len(ohlcv_data) < period:
                logger.warning(f"Insufficient valid OHLCV data for {symbol}: {len(ohlcv_data)}")
                return None
            
            # Extract price arrays for ATR calculation
            high_prices = np.array([row[2] for row in ohlcv_data])
            low_prices = np.array([row[3] for row in ohlcv_data])
            close_prices = np.array([row[4] for row in ohlcv_data])
            
            # Calculate ATR using the indicators module
            atr_values = indicators.atr(high_prices, low_prices, close_prices, period)
            if atr_values is None or len(atr_values) == 0:
                logger.warning(f"ATR calculation returned no values for {symbol}")
                return None
            
            # Get latest ATR value and current price
            latest_atr = atr_values[-1]
            current_price = float(candles[-1][4])  # Latest close price
            
            # Convert ATR to percentage volatility
            if current_price > 0 and latest_atr > 0:
                atr_percentage = latest_atr / current_price
                
                # Apply realistic bounds for crypto markets
                atr_percentage = max(0.005, min(atr_percentage, 0.15))  # 0.5% to 15%
                
                logger.debug(f"{symbol} ATR calculation: ATR={latest_atr:.2f}, Price={current_price:.2f}, Vol={atr_percentage:.4f}")
                return atr_percentage
            else:
                logger.warning(f"Invalid price data for {symbol}: ATR={latest_atr}, Price={current_price}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating ATR volatility for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def initialize_portfolio_volatilities(self, symbols: List[str]) -> Dict[str, float]:
        """Initialize volatility data for portfolio allocation using real market data.
        
        This method fetches historical candles and calculates ATR-based volatility
        for accurate portfolio allocation weights.
        
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
        
        logger.info("Calculating real-time volatilities for portfolio initialization...")
        
        for symbol in symbols:
            try:
                # Get sufficient historical candles for ATR calculation
                candles = self.get_candles(symbol, "1m")
                
                if candles and len(candles) >= 20:  # Need minimum candles for ATR(14)
                    # Calculate ATR-based volatility using real market data
                    atr_volatility = self.calculate_atr_volatility(symbol, period=14)
                    
                    if atr_volatility is not None and atr_volatility > 0:
                        # Apply realistic bounds for crypto volatility
                        atr_volatility = max(0.005, min(atr_volatility, 0.10))  # 0.5% to 10%
                        volatilities[symbol] = atr_volatility
                        logger.info(f"{symbol}: Real ATR volatility = {atr_volatility:.4f} ({atr_volatility*100:.2f}%)")
                        continue
                
                # Fallback to default if real calculation fails
                default_vol = default_volatilities.get(symbol, 0.02)
                volatilities[symbol] = default_vol
                logger.warning(f"{symbol}: Using default volatility = {default_vol:.4f} ({default_vol*100:.2f}%) [insufficient data]")
                
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
                # Use default volatility as fallback
                default_vol = default_volatilities.get(symbol, 0.02)
                volatilities[symbol] = default_vol
                logger.warning(f"{symbol}: Using default volatility = {default_vol:.4f} ({default_vol*100:.2f}%) [error fallback]")
        
        return volatilities


if __name__ == "__main__":
    from binance_exchange import BinanceClient
    
    # Create a standalone instance
    client = BinanceClient(testnet=True)
    data_engine = DataEngine(binance_client=client, max_candles=30)
    
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

