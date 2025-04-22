import talib
import numpy as np
from typing import Dict, List, Any, Union

class Indicators:
    def __init__(self):
        """Initialize the Indicators class."""
        pass
        
    async def calculate_indicators(self, required_indicators: List[str], numpy_array: np.ndarray) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Calculate all required indicators based on OHLCV data.
        
        Args:
            required_indicators (List[str]): List of indicator names to calculate
            numpy_array (np.ndarray): Array of OHLCV data in order: [open, high, low, close, volume]
            
        Returns:
            Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]: Dictionary of calculated indicators
            
        Raises:
            ValueError: If data is invalid or indicator calculation fails
        """
        results = {}
        
        try:
            # Extract OHLCV components
            if numpy_array.shape[1] < 5:  # Ensure we have enough columns
                raise ValueError(f"Invalid data shape: {numpy_array.shape}. Expected at least 5 columns for OHLCV")
                
            # Data is already in OHLCV order from BaseStrategy._calculate_indicators
            open_prices = numpy_array[:, 0]  # Open
            high_prices = numpy_array[:, 1]  # High
            low_prices = numpy_array[:, 2]   # Low
            close_prices = numpy_array[:, 3]  # Close
            volume = numpy_array[:, 4]        # Volume
            
            # Calculate each required indicator
            for indicator in required_indicators:
                try:
                    if indicator.startswith('sma_'):
                        period = int(indicator.split('_')[1])
                        if period <= 0:
                            raise ValueError(f"Invalid period for SMA: {period}")
                        
                        # Check if we have enough data for SMA calculation
                        if len(close_prices) < period:
                            print(f"Warning: Not enough data for {indicator}. Need {period} points, have {len(close_prices)}")
                            results[indicator] = np.full_like(close_prices, np.nan)
                        else:
                            # Calculate SMA
                            results[indicator] = talib.SMA(close_prices, timeperiod=period)
                            
                            # Validate the output - sometimes talib returns NaN for the first 'period' elements
                            if np.isnan(results[indicator][-1]):
                                print(f"Warning: SMA calculation returned NaN for most recent value of {indicator}")
                    elif indicator.startswith('ema_'):
                        period = int(indicator.split('_')[1])
                        if period <= 0:
                            raise ValueError(f"Invalid period for EMA: {period}")
                        results[indicator] = talib.EMA(close_prices, timeperiod=period)
                    elif indicator.startswith('rsi_'):
                        period = int(indicator.split('_')[1])
                        if period <= 0:
                            raise ValueError(f"Invalid period for RSI: {period}")
                        results[indicator] = self.rsi(close_prices, period)
                    elif indicator == 'macd':
                        results[indicator] = self.macd(close_prices)
                    elif indicator == 'bollinger_bands':
                        results[indicator] = self.bollinger_bands(close_prices)
                    elif indicator == 'atr':
                        results[indicator] = self.atr(high_prices, low_prices, close_prices)
                    elif indicator == 'obv':
                        if volume is not None:
                            results[indicator] = self.obv(close_prices, volume)
                        else:
                            raise ValueError("Volume data required for OBV calculation")
                    elif indicator == 'stoch':
                        results[indicator] = self.stochastic(high_prices, low_prices, close_prices)
                    elif indicator == 'adx':
                        results[indicator] = self.adx(high_prices, low_prices, close_prices)
                    elif indicator == 'ichimoku':
                        results[indicator] = self.ichimoku(high_prices, low_prices, close_prices)
                    elif indicator == 'mfi':
                        if volume is not None:
                            results[indicator] = self.mfi(high_prices, low_prices, close_prices, volume)
                        else:
                            raise ValueError("Volume data required for MFI calculation")
                    elif indicator == 'vwap':
                        if volume is not None:
                            results[indicator] = self.vwap(high_prices, low_prices, close_prices, volume)
                        else:
                            raise ValueError("Volume data required for VWAP calculation")
                    elif indicator == 'heikin_ashi':
                        results[indicator] = self.heikin_ashi(open_prices, high_prices, low_prices, close_prices)
                    else:
                        raise ValueError(f"Unknown indicator: {indicator}")
                except Exception as e:
                    print(f"Error calculating {indicator}: {e}")
                    results[indicator] = np.full_like(close_prices, np.nan)
            
            return results
        except Exception as e:
            print(f"Error in calculate_indicators: {e}")
            return {}
            
    # Moving Averages
    def sma(self, prices, period=14):
        """Simple Moving Average"""
        return talib.SMA(prices, timeperiod=period)
    
    def ema(self, prices, period=14):
        """Exponential Moving Average"""
        return talib.EMA(prices, timeperiod=period)
    
    # Oscillators
    def rsi(self, prices, period=14):
        """Relative Strength Index"""
        return talib.RSI(prices, timeperiod=period)
    
    def macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Moving Average Convergence Divergence"""
        macd, signal, hist = talib.MACD(prices, fastperiod=fast_period, 
                                      slowperiod=slow_period, signalperiod=signal_period)
        return {'macd': macd, 'signal': signal, 'histogram': hist}
    
    def stochastic(self, high, low, close, k_period=14, d_period=3, slowing=3):
        """Stochastic Oscillator"""
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, 
                          slowk_period=slowing, slowk_matype=0, 
                          slowd_period=d_period, slowd_matype=0)
        return {'k': k, 'd': d}
    
    # Volatility Indicators
    def bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(prices, timeperiod=period, 
                                          nbdevup=std_dev, nbdevdn=std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def atr(self, high, low, close, period=14):
        """Average True Range"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    # Volume Indicators
    def obv(self, close, volume):
        """On Balance Volume"""
        return talib.OBV(close, volume)
    
    def mfi(self, high, low, close, volume, period=14):
        """Money Flow Index"""
        return talib.MFI(high, low, close, volume, timeperiod=period)
    
    def vwap(self, high, low, close, volume):
        """Volume Weighted Average Price (intraday)"""
        typical_price = (high + low + close) / 3
        return np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    # Trend Indicators
    def adx(self, high, low, close, period=14):
        """Average Directional Index"""
        return talib.ADX(high, low, close, timeperiod=period)
    
    def ichimoku(self, high, low, close, 
                conversion_period=9, base_period=26, 
                span_b_period=52, displacement=26):
        """Ichimoku Cloud"""
        conversion = (talib.MAX(high, conversion_period) + 
                     talib.MIN(low, conversion_period)) / 2
        
        base = (talib.MAX(high, base_period) + 
               talib.MIN(low, base_period)) / 2
        
        span_a = (conversion + base) / 2
        
        span_b = (talib.MAX(high, span_b_period) + 
                 talib.MIN(low, span_b_period)) / 2
        
        # Shift for cloud
        span_a_shifted = np.zeros_like(span_a)
        span_b_shifted = np.zeros_like(span_b)
        
        if len(span_a) > displacement:
            span_a_shifted[displacement:] = span_a[:-displacement]
            span_b_shifted[displacement:] = span_b[:-displacement]
        
        return {
            'conversion_line': conversion,
            'base_line': base,
            'span_a': span_a_shifted,
            'span_b': span_b_shifted,
            'lagging_span': np.concatenate((np.full(displacement, np.nan), close[:-displacement]))
        }
    
    # Candlestick Patterns
    def heikin_ashi(self, open_prices, high, low, close):
        """Heikin-Ashi Candles"""
        ha_close = (open_prices + high + low + close) / 4
        
        # First HA open equals first open
        ha_open = np.zeros_like(open_prices)
        ha_open[0] = open_prices[0]
        
        # Calculate HA open as average of previous HA open and HA close
        for i in range(1, len(open_prices)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        
        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
        
        return {
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close
        }
        