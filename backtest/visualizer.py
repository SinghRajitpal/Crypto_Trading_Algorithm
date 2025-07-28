"""backtest/visualizer.py

QuantStats-Lumi based visualization system for backtesting analysis.
Uses quantstats-lumi for reliable pandas 2.0+ compatibility and enhanced features.
Provides production-grade performance analytics and comprehensive tear sheet reports.
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import quantstats_lumi as qs
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

# Suppress only FutureWarning for cleaner output (keep RuntimeWarnings visible to fix issues)
warnings.filterwarnings("ignore", category=FutureWarning)

# Do NOT suppress RuntimeWarnings - we need to see and fix them!

# Constants
_CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "cache")

# Configure quantstats-lumi
qs.extend_pandas()


class QuantStatsVisualizer:
    """
    Production-grade backtesting visualization system with QuantStats-Lumi metrics.
    
    Features:
    - Comprehensive QuantStats-Lumi metrics (25+ professional indicators)
    - Native quantstats-lumi HTML report generation (full pandas 2.0+ compatibility)
    - Professional HTML reports with embedded charts and analytics
    - Enhanced metrics including Ulcer Index, Gain to Pain Ratio, Information Ratio
    - Benchmark comparison with Alpha, Beta, R-Squared analysis
    - Clean CSV/JSON exports with all performance data
    - Production-ready visualization with improved performance and reliability
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        
    def _load_price_data(self, symbol: str, timeframe: str = "5m", 
                        start: Optional[datetime] = None, 
                        end: Optional[datetime] = None) -> pd.Series:
        """Load price data from cache."""
        file_safe = symbol.replace("/", "")
        cache_path = os.path.join(_CACHE_DIR, f"{file_safe}-{timeframe}.csv")
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Price data not found: {cache_path}")
            
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        close_series = df['close'].copy()
        
        # Ensure proper timezone handling
        if close_series.index.tz is None:
            close_series.index = close_series.index.tz_localize('UTC')
        elif close_series.index.tz != pd.Timestamp.now().tz:
            close_series.index = close_series.index.tz_convert('UTC')
        
        # Filter by date range with proper timezone handling
        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=pd.Timestamp.now().tz)
            close_series = close_series[close_series.index >= start]
        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=pd.Timestamp.now().tz)
            close_series = close_series[close_series.index <= end]
            
        return close_series

    def _load_multi_asset_prices(self, symbols: List[str], timeframe: str = "5m",
                                start: Optional[datetime] = None,
                                end: Optional[datetime] = None) -> pd.DataFrame:
        """Load price data for multiple assets and align timestamps."""
        price_data = {}
        
        for symbol in symbols:
            try:
                price_data[symbol] = self._load_price_data(symbol, timeframe, start, end)
            except FileNotFoundError:
                logger.warning(f"Price data missing for {symbol} - skipping")
                continue
                
        if not price_data:
            raise ValueError("No price data found for any symbols")
            
        df = pd.DataFrame(price_data)
        return df.ffill()

    def _trades_to_returns(self, trades_df: pd.DataFrame, 
                          price_data: pd.DataFrame, 
                          benchmark_symbol: Optional[str] = None) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Convert trade log to returns time series for quantstats analysis."""
        if trades_df.empty:
            zero_returns = pd.Series(0, index=price_data.index, name='returns')
            benchmark_returns = self._get_benchmark_returns(price_data, benchmark_symbol) if benchmark_symbol else None
            return zero_returns, benchmark_returns
            
        # Calculate actual final portfolio value from trades - this is the ground truth
        close_trades = trades_df[trades_df['type'] == 'close']
        total_pnl = close_trades['pnl'].fillna(0).sum() if 'pnl' in trades_df.columns and len(close_trades) > 0 else 0
        total_fees = trades_df['fee'].fillna(0).sum() if 'fee' in trades_df.columns else 0
        actual_final_equity = self.initial_capital + total_pnl - total_fees
        
        logger.debug(f"Ground truth from trades: Initial: ${self.initial_capital}, PnL: ${total_pnl:.2f}, Fees: ${total_fees:.2f}")
        logger.debug(f"Actual final equity from trades: ${actual_final_equity:.2f}")
        
        equity_curve = self._trades_to_equity_curve(trades_df, price_data)
        
        # Validate equity curve matches actual trades
        if not equity_curve.empty:
            calculated_final_equity = equity_curve.iloc[-1]
            discrepancy = abs(calculated_final_equity - actual_final_equity)
            
            if discrepancy > 1.0:  # More than $1 difference
                logger.warning(f"Equity curve discrepancy: calculated ${calculated_final_equity:.2f} vs actual ${actual_final_equity:.2f}")
                logger.warning(f"Discrepancy: ${discrepancy:.2f} - this indicates an issue with equity curve calculation")
            else:
                logger.debug(f"Equity curve validation passed: ${calculated_final_equity:.2f} matches actual trades")
        
        # Convert to returns - fix the calculation to be mathematically correct
        if not equity_curve.empty and len(equity_curve) > 1:
            # Calculate the correct expected total return from actual trades
            expected_total_return = (actual_final_equity / self.initial_capital) - 1
            
            # Check if we have intraday data that spans less than a day
            time_span_hours = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 3600
            time_span_days = max(1, time_span_hours / 24)
            
            logger.debug(f"Time span: {time_span_hours:.2f} hours ({time_span_days:.2f} days)")
            
            # For very short backtests (less than 1 day), create a single return
            if time_span_days < 1.0:
                logger.info(f"Short backtest detected ({time_span_hours:.2f} hours)")
                logger.debug(f"Creating single return from initial to final equity")
                
                # Create a single return that captures the total performance
                single_return = expected_total_return
                returns = pd.Series([single_return], 
                                  index=[equity_curve.index[-1]], 
                                  name='Strategy')
                
                logger.debug(f"Created single return: {single_return:.4f} ({single_return*100:.2f}%)")
                
            else:
                # For longer backtests, check if we need daily resampling
                obs_per_day = len(equity_curve) / time_span_days
                
                if obs_per_day > 10 and time_span_days > 2:  # Only resample if we have many observations over multiple days
                    logger.debug(f"High-frequency data detected ({obs_per_day:.1f} obs/day), resampling to daily")
                    daily_equity = equity_curve.resample('D').last().dropna()
                    
                    if len(daily_equity) > 1:
                        returns = daily_equity.pct_change().dropna()
                        calculated_total_return = (1 + returns).prod() - 1
                        
                        # Validate the resampled returns match expected
                        discrepancy = abs(calculated_total_return - expected_total_return)
                        if discrepancy > 0.05:  # More than 5% difference
                            logger.warning(f"Daily resampling caused {discrepancy*100:.2f}% discrepancy")
                            logger.debug(f"Falling back to period-based returns")
                            # Fall back to a simple period return
                            returns = pd.Series([expected_total_return], 
                                              index=[equity_curve.index[-1]], 
                                              name='Strategy')
                        else:
                            logger.debug(f"Daily resampling successful: {len(returns)} daily returns")
                    else:
                        logger.debug(f"Daily resampling yielded only one point, using period return")
                        returns = pd.Series([expected_total_return], 
                                          index=[equity_curve.index[-1]], 
                                          name='Strategy')
                else:
                    # Use the equity curve directly but validate it
                    returns = equity_curve.pct_change().dropna()
                    calculated_total_return = (1 + returns).prod() - 1
                    
                    discrepancy = abs(calculated_total_return - expected_total_return)
                    if discrepancy > 0.05:  # More than 5% difference indicates equity curve issues
                        logger.warning(f"Equity curve returns don't match expected")
                        logger.warning(f"Calculated: {calculated_total_return:.4f}, Expected: {expected_total_return:.4f}")
                        logger.debug(f"Using expected return to ensure accuracy")
                        # Use the known correct return instead of the faulty equity curve returns
                        returns = pd.Series([expected_total_return], 
                                          index=[equity_curve.index[-1]], 
                                          name='Strategy')
                    else:
                        logger.debug(f"Using {len(returns)} direct returns from equity curve")
            
            # Clean and validate returns
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = returns.clip(lower=-0.99, upper=10.0)  # Reasonable bounds
            returns.name = 'Strategy'
            
            # Remove timezone for quantstats compatibility
            if returns.index.tz is not None:
                returns.index = returns.index.tz_localize(None)
                
            # Final validation
            final_cumulative_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
            logger.debug(f"Final returns: {len(returns)} observations")
            if len(returns) > 0:
                logger.debug(f"Returns range: {returns.min():.4f} to {returns.max():.4f}")
            logger.debug(f"Final cumulative return: {final_cumulative_return:.4f} ({final_cumulative_return*100:.2f}%)")
            logger.debug(f"Expected return: {expected_total_return:.4f} ({expected_total_return*100:.2f}%)")
            
        else:
            # Handle empty or single-value equity curve case
            logger.warning("Empty or insufficient equity curve data")
            returns = pd.Series(dtype=float, name='Strategy')
        
        benchmark_returns = None
        if benchmark_symbol and benchmark_symbol in price_data.columns:
            benchmark_returns = self._get_benchmark_returns(price_data, benchmark_symbol)
        
        return returns, benchmark_returns

    def _get_benchmark_returns(self, price_data: pd.DataFrame, benchmark_symbol: str) -> pd.Series:
        """Calculate benchmark returns from price data."""
        if benchmark_symbol not in price_data.columns:
            return None
            
        benchmark_prices = price_data[benchmark_symbol].dropna()
        
        if len(benchmark_prices) < 2:
            logger.warning(f"Insufficient benchmark data for {benchmark_symbol}")
            return None
        
        # Check if we need to resample to daily frequency
        time_span_days = max(1, (benchmark_prices.index[-1] - benchmark_prices.index[0]).days + 1)
        obs_per_day = len(benchmark_prices) / time_span_days
        
        if obs_per_day > 2:  # High-frequency data
            logger.debug(f"Resampling benchmark {benchmark_symbol} to daily frequency")
            daily_benchmark_prices = benchmark_prices.resample('D').last().dropna()
            if len(daily_benchmark_prices) > 1:
                benchmark_returns = daily_benchmark_prices.pct_change().dropna()
            else:
                benchmark_returns = benchmark_prices.pct_change().dropna()
        else:
            benchmark_returns = benchmark_prices.pct_change().dropna()
        
        benchmark_returns.name = f'{benchmark_symbol}_benchmark'
        
        # Remove timezone info for quantstats compatibility
        if benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)
        
        logger.debug(f"Generated {len(benchmark_returns)} benchmark returns for {benchmark_symbol}")
        return benchmark_returns

    def _trades_to_equity_curve(self, trades_df: pd.DataFrame, 
                               price_data: pd.DataFrame) -> pd.Series:
        """Convert trade log to equity curve time series with proper accounting."""
        if trades_df.empty:
            return pd.Series(self.initial_capital, index=price_data.index, name='equity')
            
        trades_df = trades_df.copy()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
        
        logger.debug(f"Processing trades: {len(trades_df[trades_df['type'] == 'open'])} opens, {len(trades_df[trades_df['type'] == 'close'])} closes, {len(trades_df[trades_df['type'] == 'funding'])} funding")
        
        # Start with initial capital - this should remain constant until there's actual P&L
        equity_curve = pd.Series(index=price_data.index, dtype=float, name='equity')
        equity_curve.iloc[:] = self.initial_capital
        
        # Track realized P&L and fees separately
        cumulative_realized_pnl = 0.0
        cumulative_fees = 0.0
        
        # Track open positions for unrealized P&L calculation
        open_positions = {}
        
        # Process each trade chronologically
        for _, trade in trades_df.iterrows():
            timestamp = trade['timestamp']
            symbol = trade['symbol']
            trade_type = trade['type']
            
            # Find the index closest to this trade timestamp
            try:
                trade_idx = equity_curve.index.get_indexer([timestamp], method='nearest')[0]
            except (IndexError, KeyError):
                continue
                
            if trade_type == 'open':
                # Opening position: only pay fees (reduce equity), margin doesn't affect equity
                fee = trade.get('fee', 0)
                if pd.isna(fee):
                    fee = 0
                cumulative_fees += fee
                
                # Track open position for unrealized PnL calculation
                open_positions[symbol] = {
                    'contracts': trade['contracts'] if trade['side'] == 'buy' else -trade['contracts'],
                    'entry_price': trade['price'],
                }
                
            elif trade_type == 'close':
                # Closing position: realize P&L and pay closing fees
                # Handle NaN PnL values - replace with 0 if NaN
                realized_pnl = trade.get('pnl', 0)
                if pd.isna(realized_pnl):
                    realized_pnl = 0
                
                fee = trade.get('fee', 0)
                if pd.isna(fee):
                    fee = 0
                
                # Always account for P&L and fees, even if position tracking is out of sync
                cumulative_realized_pnl += realized_pnl
                cumulative_fees += fee
                
                # Remove from open positions if it exists (for unrealized P&L calculation)
                if symbol in open_positions:
                    del open_positions[symbol]
                else:
                    # Position not tracked in open_positions - this can happen with overlapping trades
                    # but we still need to account for the P&L and fees
                    pass
                    
            elif trade_type == 'funding':
                # Funding payment: treat as a cost (negative) or income (positive)
                payment = trade.get('payment', 0)
                if pd.isna(payment):
                    payment = 0
                cumulative_fees += payment  # funding payments are costs
            
            # Calculate current equity = initial capital + realized P&L - fees + unrealized P&L
            current_unrealized_pnl = 0.0
            
            # Calculate unrealized P&L for open positions at each timestamp after the trade
            for i in range(trade_idx, len(equity_curve)):
                timestamp_i = equity_curve.index[i]
                unrealized_pnl_at_i = 0.0
                
                for pos_symbol, pos in open_positions.items():
                    if pos_symbol in price_data.columns:
                        try:
                            current_price = price_data.loc[timestamp_i, pos_symbol]
                            if not pd.isna(current_price):
                                unrealized_pnl = (current_price - pos['entry_price']) * pos['contracts']
                                unrealized_pnl_at_i += unrealized_pnl
                        except (KeyError, IndexError):
                            continue
                
                # Update equity: initial capital + realized P&L - fees + unrealized P&L
                equity_curve.iloc[i] = self.initial_capital + cumulative_realized_pnl - cumulative_fees + unrealized_pnl_at_i
        
        logger.debug(f"Equity curve built: ${equity_curve.iloc[0]:.2f} → ${equity_curve.iloc[-1]:.2f}")
        logger.debug(f"Cumulative realized P&L: ${cumulative_realized_pnl:.2f}")
        logger.debug(f"Cumulative fees: ${cumulative_fees:.2f}")
        
        # Calculate final unrealized P&L
        final_unrealized_pnl = equity_curve.iloc[-1] - self.initial_capital - cumulative_realized_pnl + cumulative_fees
        logger.debug(f"Final unrealized P&L: ${final_unrealized_pnl:.2f}")
        
        return equity_curve
    def generate_portfolio_report(self, trades_df: pd.DataFrame, symbols: List[str],
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 timeframe: str = "5m",
                                 benchmark_symbol: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate comprehensive portfolio performance report using quantstats.
        
        Args:
            trades_df: Trade log DataFrame
            symbols: List of symbols traded
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            benchmark_symbol: Optional benchmark symbol (e.g., 'BTCUSDT' for crypto strategies)
            
        Returns:
            Tuple of (HTML tear sheet path, metrics dictionary)
        """
        logger.info(f"Generating portfolio report for {len(symbols)} symbols")
        
        # Load price data
        try:
            price_data = self._load_multi_asset_prices(symbols, timeframe, start_date, end_date)
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            return None, {}
        
        # Convert trades to returns
        try:
            returns, benchmark_returns = self._trades_to_returns(trades_df, price_data, benchmark_symbol)
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return None, {}
        
        # Generate quantstats metrics
        metrics = self._extract_quantstats_metrics(returns, benchmark_returns)
        
        return returns, metrics

    def generate_asset_report(self, trades_df: pd.DataFrame, symbol: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             timeframe: str = "5m",
                             benchmark_symbol: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate individual asset performance report using quantstats."""
        
        # Filter trades for this asset
        asset_trades = trades_df[trades_df['symbol'] == symbol].copy()
        
        if asset_trades.empty:
            logger.warning(f"No trades found for {symbol}")
            return None, {}
        
        logger.info(f"Generating asset report for {symbol}")
        
        # Load price data for this asset
        try:
            price_series = self._load_price_data(symbol, timeframe, start_date, end_date)
            price_df = pd.DataFrame({'close': price_series})
        except Exception as e:
            logger.error(f"Error loading price data for {symbol}: {e}")
            return None, {}
        
        # Convert trades to returns
        try:
            returns, benchmark_returns = self._trades_to_returns(asset_trades, price_df, benchmark_symbol)
        except Exception as e:
            logger.error(f"Error calculating returns for {symbol}: {e}")
            return None, {}
        
        # Generate quantstats metrics
        metrics = self._extract_quantstats_metrics(returns, benchmark_returns, symbol)
        
        return returns, metrics

    def _extract_quantstats_metrics(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series] = None,
                                   symbol: str = "Portfolio") -> Dict[str, Any]:
        """Extract comprehensive metrics using quantstats-lumi built-in metrics reports."""
        
        logger.debug(f"Extracting QuantStats-Lumi metrics for {symbol} with {len(returns)} returns")
        
        # Validate returns data first
        if returns.empty or returns.isna().all() or len(returns) == 0:
            logger.warning(f"No valid returns data for {symbol}")
            return {
                'Symbol': symbol,
                'Total Return (%)': 0.0,
                'CAGR (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Final Equity': self.initial_capital,
            }
        
        # Clean the returns data to prevent numerical issues
        returns_cleaned = returns.copy()
        
        # Remove infinite values
        inf_mask = np.isinf(returns_cleaned)
        if inf_mask.any():
            logger.warning(f"Removing {inf_mask.sum()} infinite values from returns")
            returns_cleaned = returns_cleaned[~inf_mask]
        
        # Remove NaN values
        nan_mask = returns_cleaned.isna()
        if nan_mask.any():
            logger.warning(f"Removing {nan_mask.sum()} NaN values from returns")
            returns_cleaned = returns_cleaned.dropna()
        
        # Check if we still have valid data after cleaning
        if returns_cleaned.empty or len(returns_cleaned) == 0:
            logger.warning(f"No valid returns left after cleaning for {symbol}")
            return {
                'Symbol': symbol,
                'Total Return (%)': 0.0,
                'CAGR (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Final Equity': self.initial_capital,
            }
        
        # Check for all-zero returns (no variation)
        if (returns_cleaned == 0).all():
            logger.warning(f"All returns are zero for {symbol}")
            return {
                'Symbol': symbol,
                'Total Return (%)': 0.0,
                'CAGR (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Final Equity': self.initial_capital,
            }
        
        # Check for edge cases that cause QuantStats issues
        returns_std = returns_cleaned.std()
        negative_returns = returns_cleaned[returns_cleaned < 0]
        positive_returns = returns_cleaned[returns_cleaned > 0]
        
        # If there's no volatility (all returns are the same), many metrics will be undefined
        if returns_std == 0 or pd.isna(returns_std):
            logger.warning(f"Zero volatility detected for {symbol} - using simplified metrics")
            total_return = returns_cleaned.iloc[0] if len(returns_cleaned) > 0 else 0
            return {
                'Symbol': symbol,
                'Total Return (%)': round(total_return * 100, 2),
                'CAGR (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Calmar Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Final Equity': round(self.initial_capital * (1 + total_return), 2),
            }
        
        # If there are no negative returns, some risk metrics can't be calculated
        has_negative_returns = len(negative_returns) > 0
        has_positive_returns = len(positive_returns) > 0
        
        logger.debug(f"Returns analysis for {symbol}: std={returns_std:.6f}, neg_count={len(negative_returns)}, pos_count={len(positive_returns)}")
        
        # Use quantstats-lumi's comprehensive metrics report
        try:
            # Safe calculation wrapper
            def safe_calc(func, default_value=0.0):
                try:
                    result = func()
                    if pd.isna(result) or np.isinf(result):
                        return default_value
                    return result
                except (ZeroDivisionError, ValueError, RuntimeError) as e:
                    logger.debug(f"Metric calculation failed: {e}")
                    return default_value
            
            # Calculate final equity from cumulative returns
            total_return = safe_calc(lambda: qs.stats.comp(returns_cleaned))
            final_equity = self.initial_capital * (1 + total_return)
            
            # Calculate metrics safely, with special handling for edge cases
            metrics = {
                'Symbol': symbol,
                'Total Return (%)': round(total_return * 100, 2),
                'CAGR (%)': round(safe_calc(lambda: qs.stats.cagr(returns_cleaned)) * 100, 2),
                'Expected Return (%)': round(safe_calc(lambda: qs.stats.expected_return(returns_cleaned)) * 100, 3),
                'Volatility (%)': round(returns_std * 100, 2),  # Calculate directly to avoid QuantStats issues
                'Final Equity': round(final_equity, 2),
            }
            
            # Only calculate risk-adjusted metrics if we have sufficient variation
            if returns_std > 1e-8:  # Minimum threshold for meaningful calculations
                metrics.update({
                    'Sharpe Ratio': round(safe_calc(lambda: qs.stats.sharpe(returns_cleaned)), 3),
                    'Max Drawdown (%)': round(safe_calc(lambda: abs(qs.stats.max_drawdown(returns_cleaned))) * 100, 2),
                    'Ulcer Index': round(safe_calc(lambda: qs.stats.ulcer_index(returns_cleaned)), 4),
                })
                
                # Only calculate Sortino if we have negative returns
                if has_negative_returns:
                    metrics['Sortino Ratio'] = round(safe_calc(lambda: qs.stats.sortino(returns_cleaned)), 3)
                else:
                    logger.debug(f"No negative returns for {symbol}, skipping Sortino calculation")
                    metrics['Sortino Ratio'] = 0.0
                    
                # Only calculate Calmar if we have drawdown
                max_dd = safe_calc(lambda: abs(qs.stats.max_drawdown(returns_cleaned)))
                if max_dd > 0:
                    metrics['Calmar Ratio'] = round(safe_calc(lambda: qs.stats.calmar(returns_cleaned)), 3)
                else:
                    metrics['Calmar Ratio'] = 0.0
                    
                # Recovery factor - only meaningful if we have drawdown
                max_dd = safe_calc(lambda: abs(qs.stats.max_drawdown(returns_cleaned)))
                if max_dd > 1e-6:  # Only calculate if we have meaningful drawdown
                    metrics['Recovery Factor'] = round(safe_calc(lambda: qs.stats.recovery_factor(returns_cleaned)), 2)
                else:
                    metrics['Recovery Factor'] = 0.0  # No drawdown = no recovery needed
                
            else:
                # Zero volatility case
                metrics.update({
                    'Sharpe Ratio': 0.0,
                    'Sortino Ratio': 0.0,
                    'Calmar Ratio': 0.0,
                    'Max Drawdown (%)': 0.0,
                    'Ulcer Index': 0.0,
                    'Recovery Factor': 0.0,
                })
            
            # VaR and CVaR calculations - only if we have enough data and variation
            if len(returns_cleaned) >= 20 and has_negative_returns:  # Need sufficient data AND negative returns for meaningful VaR/CVaR
                # For CVaR, we need to ensure there are returns below the VaR threshold
                var_95 = safe_calc(lambda: qs.stats.var(returns_cleaned))
                returns_below_var = returns_cleaned[returns_cleaned < var_95]
                
                if len(returns_below_var) > 0:  # Only calculate CVaR if we have returns below VaR
                    metrics.update({
                        'VaR (95%) (%)': round(var_95 * 100, 2),
                        'CVaR (95%) (%)': round(safe_calc(lambda: qs.stats.cvar(returns_cleaned)) * 100, 2),
                    })
                else:
                    # Not enough extreme negative returns for CVaR
                    metrics.update({
                        'VaR (95%) (%)': round(var_95 * 100, 2),
                        'CVaR (95%) (%)': round(var_95 * 100, 2),  # Use VaR as approximation
                    })
            else:
                logger.debug(f"Insufficient data for VaR/CVaR calculation for {symbol}: len={len(returns_cleaned)}, has_neg={has_negative_returns}")
                metrics.update({
                    'VaR (95%) (%)': 0.0,
                    'CVaR (95%) (%)': 0.0,
                })
            
            # Win rate and trading metrics
            win_rate = (returns_cleaned > 0).mean() * 100 if len(returns_cleaned) > 0 else 0
            metrics['Win Rate (%)'] = round(win_rate, 2)
            
            # Profit factor - only if we have both positive and negative returns
            if has_positive_returns and has_negative_returns:
                metrics['Profit Factor'] = round(safe_calc(lambda: qs.stats.profit_factor(returns_cleaned), 1.0), 3)
                metrics['Gain to Pain Ratio'] = round(safe_calc(lambda: qs.stats.gain_to_pain_ratio(returns_cleaned)), 3)
                metrics['Tail Ratio'] = round(safe_calc(lambda: qs.stats.tail_ratio(returns_cleaned)), 3)
            else:
                logger.debug(f"Insufficient return variation for {symbol}, skipping profit factor calculations")
                metrics.update({
                    'Profit Factor': 1.0 if has_positive_returns else 0.0,
                    'Gain to Pain Ratio': 0.0,
                    'Tail Ratio': 0.0,
                })
            
            # Kelly criterion
            metrics['Kelly Criterion'] = round(safe_calc(lambda: qs.stats.kelly_criterion(returns_cleaned)), 3)
            
            # Basic statistics
            metrics.update({
                'Best Day (%)': round(returns_cleaned.max() * 100, 2) if len(returns_cleaned) > 0 else 0.0,
                'Worst Day (%)': round(returns_cleaned.min() * 100, 2) if len(returns_cleaned) > 0 else 0.0,
                'Skewness': round(safe_calc(lambda: returns_cleaned.skew()), 3),
                'Kurtosis': round(safe_calc(lambda: returns_cleaned.kurtosis()), 3),
            })
            
            standardized_metrics = metrics
            
            # Add benchmark comparison metrics if benchmark is provided
            if benchmark_returns is not None and not benchmark_returns.empty and len(benchmark_returns) > 1:
                logger.debug(f"Adding benchmark comparison metrics")
                try:
                    # Clean and align benchmark data
                    benchmark_cleaned = benchmark_returns.copy()
                    benchmark_cleaned = benchmark_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not benchmark_cleaned.empty and len(benchmark_cleaned) > 1:
                        aligned_benchmark = benchmark_cleaned.reindex(returns_cleaned.index).ffill().dropna()
                        
                        if not aligned_benchmark.empty and len(aligned_benchmark) > 1 and aligned_benchmark.std() > 0:
                            # Use quantstats-lumi functions safely
                            information_ratio = safe_calc(lambda: qs.stats.information_ratio(returns_cleaned, aligned_benchmark))
                            r_squared = safe_calc(lambda: qs.stats.r_squared(returns_cleaned, aligned_benchmark))
                            
                            # Calculate beta and alpha manually (with error handling)
                            try:
                                covariance = returns_cleaned.cov(aligned_benchmark)
                                benchmark_var = aligned_benchmark.var()
                                beta = covariance / benchmark_var if benchmark_var != 0 and not pd.isna(benchmark_var) else 0
                                
                                portfolio_return = returns_cleaned.mean() * len(returns_cleaned)
                                benchmark_return = aligned_benchmark.mean() * len(aligned_benchmark)
                                alpha = (portfolio_return - beta * benchmark_return) * 100
                                
                                correlation = returns_cleaned.corr(aligned_benchmark)
                                
                                # Only add if values are valid
                                if not pd.isna(alpha) and not np.isinf(alpha):
                                    standardized_metrics['Alpha (%)'] = round(alpha, 2)
                                if not pd.isna(beta) and not np.isinf(beta):
                                    standardized_metrics['Beta'] = round(beta, 3)
                                if not pd.isna(information_ratio) and not np.isinf(information_ratio):
                                    standardized_metrics['Information Ratio'] = round(information_ratio, 3)
                                if not pd.isna(r_squared) and not np.isinf(r_squared):
                                    standardized_metrics['R-Squared'] = round(r_squared, 3)
                                if not pd.isna(correlation) and not np.isinf(correlation):
                                    standardized_metrics['Correlation'] = round(correlation, 3)
                                
                                logger.debug(f"Added benchmark metrics successfully")
                            except Exception as e:
                                logger.debug(f"Benchmark calculation error: {e}")
                        else:
                            logger.debug("Insufficient benchmark data for comparison")
                except Exception as e:
                    logger.warning(f"Benchmark comparison failed: {e}")
            
            logger.debug(f"QuantStats-Lumi metrics extracted for {symbol}: Return: {standardized_metrics['Total Return (%)']}%, Sharpe: {standardized_metrics['Sharpe Ratio']}")
            return standardized_metrics
            
        except Exception as e:
            logger.error(f"Error using QuantStats-Lumi metrics report: {e}")
            # Fallback to individual metric calculations
            return self._fallback_metrics_calculation(returns_cleaned, benchmark_returns, symbol)
    
    def _fallback_metrics_calculation(self, returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None,
                                    symbol: str = "Portfolio") -> Dict[str, Any]:
        """Fallback individual metrics calculation if full report fails."""
        
        def safe_calc(func, default=0.0):
            try:
                result = func()
                if pd.isna(result) or np.isinf(result):
                    return default
                return result
            except Exception:
                return default
        
        final_equity = self.initial_capital * (1 + safe_calc(lambda: qs.stats.comp(returns)))
        
        return {
            'Symbol': symbol,
            'Total Return (%)': round(safe_calc(lambda: qs.stats.comp(returns) * 100), 2),
            'CAGR (%)': round(safe_calc(lambda: qs.stats.cagr(returns) * 100), 2),
            'Volatility (%)': round(safe_calc(lambda: qs.stats.volatility(returns) * 100), 2),
            'Sharpe Ratio': round(safe_calc(lambda: qs.stats.sharpe(returns)), 3),
            'Sortino Ratio': round(safe_calc(lambda: qs.stats.sortino(returns)), 3),
            'Max Drawdown (%)': round(safe_calc(lambda: abs(qs.stats.max_drawdown(returns)) * 100), 2),
            'Final Equity': round(final_equity, 2),
        }

    def generate_comprehensive_plots(self, returns: pd.Series, 
                                   equity_curve: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   output_dir: str = "plots") -> Dict[str, str]:
        """
        Generate comprehensive performance plots.
        
        This method creates the directory structure and prepares plot file paths
        that quantstats-lumi will use for enhanced visualization generation.
        """
        
        logger.debug(f"Setting up plot generation in: {output_dir}")
        
        # Create output directory structure
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different plot types
        performance_dir = os.path.join(output_dir, "performance")
        risk_dir = os.path.join(output_dir, "risk")
        distributions_dir = os.path.join(output_dir, "distributions")
        
        for dir_path in [performance_dir, risk_dir, distributions_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Define plot file paths that quantstats will use
        plot_files = {
            "equity_curve": os.path.join(performance_dir, "equity_curve.png"),
            "rolling_sharpe": os.path.join(performance_dir, "rolling_sharpe.png"),
            "rolling_volatility": os.path.join(performance_dir, "rolling_volatility.png"),
            "drawdown": os.path.join(risk_dir, "drawdown.png"),
            "rolling_returns": os.path.join(performance_dir, "rolling_returns.png"),
            "monthly_returns": os.path.join(performance_dir, "monthly_returns.png"),
            "distribution": os.path.join(distributions_dir, "returns_distribution.png")
        }
        
        logger.debug(f"Prepared {len(plot_files)} plot paths for quantstats generation")
        return plot_files

    def generate_html_report(self, returns: pd.Series, equity_curve: pd.Series,
                           metrics: Dict[str, Any], plot_files: Optional[Dict[str, str]],
                           benchmark_returns: Optional[pd.Series] = None,
                           output_file: str = "performance_report.html") -> str:
        """Generate comprehensive HTML performance report using quantstats-lumi."""
        
        if "portfolio" in output_file.lower():
            title = "Portfolio Backtest Performance Report"
        elif any(symbol in output_file.lower() for symbol in ["btc", "eth", "ada", "sol", "xrp"]):
            filename = os.path.basename(output_file).lower()
            for symbol in ["btc", "eth", "ada", "sol", "xrp", "bnb", "doge", "trx", "usdc", "hype"]:
                if symbol in filename:
                    title = f"{symbol.upper()} Individual Asset Backtest Report"
                    break
            else:
                title = "Individual Asset Backtest Performance Report"
        else:
            title = "Backtest Performance Report"
        
        try:
            # Use quantstats-lumi's native HTML report generation
            # Production-ready with full pandas 2.0+ compatibility and enhanced features
            logger.info(f"Generating {title}")
            logger.debug(f"Output: {output_file}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Clean and prepare returns data
            if returns.empty:
                logger.warning("Empty returns series provided")
                returns = pd.Series([0.0], index=[pd.Timestamp.now()], name="Strategy Returns")
            
            # Clean returns data - remove infinite and extreme values
            orig_len = len(returns)
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) != orig_len:
                logger.warning(f"Cleaned {orig_len - len(returns)} invalid returns")
            
            # Ensure proper datetime index
            if not isinstance(returns.index, pd.DatetimeIndex):
                if len(returns) > 0:
                    returns.index = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
            
            # Prepare benchmark if provided
            if benchmark_returns is not None and not benchmark_returns.empty:
                if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
                    if len(benchmark_returns) > 0:
                        benchmark_returns.index = pd.date_range(start='2023-01-01', periods=len(benchmark_returns), freq='D')
                # Align benchmark with returns index
                benchmark_returns = benchmark_returns.reindex(returns.index, method='ffill').dropna()
            
            # Generate comprehensive HTML report using quantstats-lumi
            # Enhanced with professional-grade analytics and visualizations
            # NOTE: QuantStats may create its own plots directory - this is internal to QuantStats
            qs.reports.html(
                returns,
                output=output_file,
                title=title,
                benchmark=benchmark_returns,
                rf=0.0,  # Risk-free rate
                download_filename=os.path.basename(output_file),
                figfmt='png',
                template_path=None,
                compounding=True,
                periods_per_year=252  # Trading days per year
            )
            
            logger.info("HTML report generated successfully!")
            logger.debug(f"Report saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return None
    def save_results(self, portfolio_returns: pd.Series, portfolio_metrics: Dict[str, Any],
                    asset_results: Dict[str, Tuple[pd.Series, Dict[str, Any]]],
                    trades_df: pd.DataFrame, save_dir: str,
                    start_date: datetime, end_date: datetime,
                    benchmark_symbol: Optional[str] = None,
                    generate_individual_plots: bool = False):
        """Save all results with comprehensive HTML reports.
        
        Args:
            generate_individual_plots: If True, generates individual asset HTML reports.
                                     If False, only generates portfolio reports and saves metrics.
        """
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        port_dir = os.path.join(save_dir, "portfolio_performance")
        indiv_dir = os.path.join(save_dir, "individual_asset_performance")
        # Only create plots directory if individual plots are requested
        plots_dir = os.path.join(save_dir, "plots") if generate_individual_plots else None
        Path(port_dir).mkdir(parents=True, exist_ok=True)
        Path(indiv_dir).mkdir(parents=True, exist_ok=True)
        if plots_dir:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        benchmark_returns = None
        if benchmark_symbol:
            try:
                if not trades_df.empty:
                    timeframe = "5m"
                    benchmark_prices = self._load_price_data(benchmark_symbol, timeframe, start_date, end_date)
                    # Convert benchmark to daily returns to match our strategy returns
                    daily_benchmark_prices = benchmark_prices.resample('D').last().dropna()
                    benchmark_returns = daily_benchmark_prices.pct_change().dropna()
                    benchmark_returns.name = f'{benchmark_symbol}_benchmark'
                    # Remove timezone for consistency
                    if benchmark_returns.index.tz is not None:
                        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
            except Exception:
                pass
        
        # Always generate portfolio reports
        if portfolio_returns is not None and not portfolio_returns.empty:
            equity_curve = (1 + portfolio_returns).cumprod() * self.initial_capital
            
            # Only create plots if individual plots are requested
            plot_files = None
            if generate_individual_plots and plots_dir:
                portfolio_plots_dir = os.path.join(plots_dir, "portfolio")
                Path(portfolio_plots_dir).mkdir(exist_ok=True)
                
                plot_files = self.generate_comprehensive_plots(
                    portfolio_returns, equity_curve, benchmark_returns, portfolio_plots_dir
                )
            
            # Always generate portfolio HTML report (but without plots if not requested)
            portfolio_html = os.path.join(port_dir, "portfolio_performance_report.html")
            self.generate_html_report(
                portfolio_returns, equity_curve, portfolio_metrics, 
                plot_files, benchmark_returns, portfolio_html
            )
            
            # Always save portfolio stats CSV
            portfolio_df = pd.Series(portfolio_metrics, name='Portfolio')
            portfolio_df.to_csv(os.path.join(port_dir, "portfolio_stats.csv"))
        
        # Process individual assets - always calculate metrics, only generate plots if requested
        asset_metrics = {}
        for symbol, (returns, metrics) in asset_results.items():
            if returns is not None and not returns.empty:
                # Always store the metrics
                asset_metrics[symbol] = metrics
                
                # Only generate HTML reports and plots if individual plots are requested
                if generate_individual_plots and plots_dir:
                    asset_plots_dir = os.path.join(plots_dir, symbol)
                    Path(asset_plots_dir).mkdir(exist_ok=True)
                    
                    asset_equity = (1 + returns).cumprod() * self.initial_capital
                    
                    asset_plot_files = self.generate_comprehensive_plots(
                        returns, asset_equity, benchmark_returns, asset_plots_dir
                    )
                    
                    asset_html = os.path.join(indiv_dir, f"{symbol}_performance_report.html")
                    self.generate_html_report(
                        returns, asset_equity, metrics, 
                        asset_plot_files, benchmark_returns, asset_html
                    )
        
        # Always save individual asset metrics CSV
        if asset_metrics:
            asset_df = pd.DataFrame(asset_metrics).T
            asset_df.to_csv(os.path.join(indiv_dir, "per_asset_stats.csv"))
        
        # Prepare summary with conditional HTML reports list
        asset_html_reports = []
        if generate_individual_plots:
            asset_html_reports = [f"individual_asset_performance/{symbol}_performance_report.html" 
                                for symbol in asset_results.keys()]
        
        summary = {
            "metrics": portfolio_metrics,
            "per_asset_stats": asset_metrics,
            "timespan": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "benchmark": benchmark_symbol,
            "reports_generated": {
                "portfolio_html_report": "portfolio_performance/portfolio_performance_report.html",
                "asset_html_reports": asset_html_reports,
                "portfolio_stats": "portfolio_performance/portfolio_stats.csv",
                "asset_stats": "individual_asset_performance/per_asset_stats.csv",
                "plots_directory": "plots/" if generate_individual_plots else None
            }
        }
        
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        trades_df.to_csv(os.path.join(save_dir, "trade_log.csv"), index=False)

    def print_summary(self, portfolio_metrics: Dict[str, Any], 
                     asset_metrics: Dict[str, Dict[str, Any]]):
        """Print a formatted summary of key metrics."""
        
        logger.info("\n" + "="*80)
        logger.info("PORTFOLIO PERFORMANCE SUMMARY (QuantStats-Lumi Enhanced)")
        logger.info("="*80)
        
        # Console output for portfolio summary
        console_log("\n" + "="*80, "INFO")
        console_log("PORTFOLIO PERFORMANCE SUMMARY", "INFO")
        console_log("="*80, "INFO")
        
        # Core performance metrics
        core_metrics = ['Total Return (%)', 'CAGR (%)', 'Expected Return (%)', 'Volatility (%)']
        for metric in core_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                logger.info(f"{metric:<30} {value:>12.2f}")
                console_log(f"{metric:<30} {value:>12.2f}", "INFO")
        
        logger.info("\n" + "-"*80)
        logger.info("RISK-ADJUSTED METRICS")
        logger.info("-"*80)
        
        console_log("\n" + "-"*80, "INFO")
        console_log("RISK-ADJUSTED METRICS", "INFO")
        console_log("-"*80, "INFO")
        
        risk_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
        for metric in risk_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                logger.info(f"{metric:<30} {value:>12.3f}")
                console_log(f"{metric:<30} {value:>12.3f}", "INFO")
        
        logger.info("\n" + "-"*80)
        logger.info("RISK METRICS")
        logger.info("-"*80)
        
        console_log("\n" + "-"*80, "INFO")
        console_log("RISK METRICS", "INFO")
        console_log("-"*80, "INFO")
        
        risk_detail_metrics = ['Max Drawdown (%)', 'Ulcer Index', 'VaR (95%) (%)', 'CVaR (95%) (%)']
        for metric in risk_detail_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                logger.info(f"{metric:<30} {value:>12.2f}")
                console_log(f"{metric:<30} {value:>12.2f}", "INFO")
        
        logger.info("\n" + "-"*80)
        logger.info("TRADING METRICS")
        logger.info("-"*80)
        
        console_log("\n" + "-"*80, "INFO")
        console_log("TRADING METRICS", "INFO")
        console_log("-"*80, "INFO")
        
        trading_metrics = ['Win Rate (%)', 'Profit Factor', 'Gain to Pain Ratio', 'Payoff Ratio', 'Kelly Criterion']
        for metric in trading_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                logger.info(f"{metric:<30} {value:>12.3f}")
                console_log(f"{metric:<30} {value:>12.3f}", "INFO")

        if asset_metrics:
            logger.info("\n" + "="*80)
            logger.info("INDIVIDUAL ASSET PERFORMANCE")
            logger.info("="*80)
            
            console_log("\n" + "="*80, "INFO")
            console_log("INDIVIDUAL ASSET PERFORMANCE", "INFO")
            console_log("="*80, "INFO")
            
            for symbol, metrics in asset_metrics.items():
                logger.info(f"\n📊 {symbol}:")
                console_log(f"\n📊 {symbol}:", "INFO")
                key_asset_metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                                   'Win Rate (%)', 'Profit Factor']
                for metric in key_asset_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        logger.info(f"  {metric:<28} {value:>10.2f}")
                        console_log(f"  {metric:<28} {value:>10.2f}", "INFO")
        
        logger.info("="*80)
        console_log("="*80, "INFO")