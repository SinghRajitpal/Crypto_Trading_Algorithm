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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
                print(f"[QuantStats] ⚠️ Price data missing for {symbol} - skipping")
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
        
        print(f"[Visualizer] Ground truth from trades: Initial: ${self.initial_capital}, PnL: ${total_pnl:.2f}, Fees: ${total_fees:.2f}")
        print(f"[Visualizer] Actual final equity from trades: ${actual_final_equity:.2f}")
        
        equity_curve = self._trades_to_equity_curve(trades_df, price_data)
        
        # Validate equity curve matches actual trades
        if not equity_curve.empty:
            calculated_final_equity = equity_curve.iloc[-1]
            discrepancy = abs(calculated_final_equity - actual_final_equity)
            
            if discrepancy > 1.0:  # More than $1 difference
                print(f"[Visualizer] Warning: Equity curve discrepancy: calculated ${calculated_final_equity:.2f} vs actual ${actual_final_equity:.2f}")
                print(f"[Visualizer] Discrepancy: ${discrepancy:.2f} - this indicates an issue with equity curve calculation")
            else:
                print(f"[Visualizer] Equity curve validation passed: ${calculated_final_equity:.2f} matches actual trades")
        
        # Convert to returns - fix the calculation to be mathematically correct
        if not equity_curve.empty and len(equity_curve) > 1:
            # Calculate the correct expected total return from actual trades
            expected_total_return = (actual_final_equity / self.initial_capital) - 1
            
            # Check if we have intraday data that spans less than a day
            time_span_hours = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 3600
            time_span_days = max(1, time_span_hours / 24)
            
            print(f"[Visualizer] Time span: {time_span_hours:.2f} hours ({time_span_days:.2f} days)")
            
            # For very short backtests (less than 1 day), create a single return
            if time_span_days < 1.0:
                print(f"[Visualizer] Short backtest detected ({time_span_hours:.2f} hours)")
                print(f"[Visualizer] Creating single return from initial to final equity")
                
                # Create a single return that captures the total performance
                single_return = expected_total_return
                returns = pd.Series([single_return], 
                                  index=[equity_curve.index[-1]], 
                                  name='Strategy')
                
                print(f"[Visualizer] Created single return: {single_return:.4f} ({single_return*100:.2f}%)")
                
            else:
                # For longer backtests, check if we need daily resampling
                obs_per_day = len(equity_curve) / time_span_days
                
                if obs_per_day > 10 and time_span_days > 2:  # Only resample if we have many observations over multiple days
                    print(f"[Visualizer] High-frequency data detected ({obs_per_day:.1f} obs/day), resampling to daily")
                    daily_equity = equity_curve.resample('D').last().dropna()
                    
                    if len(daily_equity) > 1:
                        returns = daily_equity.pct_change().dropna()
                        calculated_total_return = (1 + returns).prod() - 1
                        
                        # Validate the resampled returns match expected
                        discrepancy = abs(calculated_total_return - expected_total_return)
                        if discrepancy > 0.05:  # More than 5% difference
                            print(f"[Visualizer] Warning: Daily resampling caused {discrepancy*100:.2f}% discrepancy")
                            print(f"[Visualizer] Falling back to period-based returns")
                            # Fall back to a simple period return
                            returns = pd.Series([expected_total_return], 
                                              index=[equity_curve.index[-1]], 
                                              name='Strategy')
                        else:
                            print(f"[Visualizer] Daily resampling successful: {len(returns)} daily returns")
                    else:
                        print(f"[Visualizer] Daily resampling yielded only one point, using period return")
                        returns = pd.Series([expected_total_return], 
                                          index=[equity_curve.index[-1]], 
                                          name='Strategy')
                else:
                    # Use the equity curve directly but validate it
                    returns = equity_curve.pct_change().dropna()
                    calculated_total_return = (1 + returns).prod() - 1
                    
                    discrepancy = abs(calculated_total_return - expected_total_return)
                    if discrepancy > 0.05:  # More than 5% difference indicates equity curve issues
                        print(f"[Visualizer] Warning: Equity curve returns don't match expected")
                        print(f"[Visualizer] Calculated: {calculated_total_return:.4f}, Expected: {expected_total_return:.4f}")
                        print(f"[Visualizer] Using expected return to ensure accuracy")
                        # Use the known correct return instead of the faulty equity curve returns
                        returns = pd.Series([expected_total_return], 
                                          index=[equity_curve.index[-1]], 
                                          name='Strategy')
                    else:
                        print(f"[Visualizer] Using {len(returns)} direct returns from equity curve")
            
            # Clean and validate returns
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = returns.clip(lower=-0.99, upper=10.0)  # Reasonable bounds
            returns.name = 'Strategy'
            
            # Remove timezone for quantstats compatibility
            if returns.index.tz is not None:
                returns.index = returns.index.tz_localize(None)
                
            # Final validation
            final_cumulative_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
            print(f"[Visualizer] Final returns: {len(returns)} observations")
            if len(returns) > 0:
                print(f"[Visualizer] Returns range: {returns.min():.4f} to {returns.max():.4f}")
            print(f"[Visualizer] Final cumulative return: {final_cumulative_return:.4f} ({final_cumulative_return*100:.2f}%)")
            print(f"[Visualizer] Expected return: {expected_total_return:.4f} ({expected_total_return*100:.2f}%)")
            
        else:
            # Handle empty or single-value equity curve case
            print("[Visualizer] Warning: Empty or insufficient equity curve data")
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
            print(f"[Visualizer] Insufficient benchmark data for {benchmark_symbol}")
            return None
        
        # Check if we need to resample to daily frequency
        time_span_days = max(1, (benchmark_prices.index[-1] - benchmark_prices.index[0]).days + 1)
        obs_per_day = len(benchmark_prices) / time_span_days
        
        if obs_per_day > 2:  # High-frequency data
            print(f"[Visualizer] Resampling benchmark {benchmark_symbol} to daily frequency")
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
        
        print(f"[Visualizer] Generated {len(benchmark_returns)} benchmark returns for {benchmark_symbol}")
        return benchmark_returns

    def _trades_to_equity_curve(self, trades_df: pd.DataFrame, 
                               price_data: pd.DataFrame) -> pd.Series:
        """Convert trade log to equity curve time series with proper accounting."""
        if trades_df.empty:
            return pd.Series(self.initial_capital, index=price_data.index, name='equity')
            
        trades_df = trades_df.copy()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
        
        print(f"[Visualizer] Processing trades: {len(trades_df[trades_df['type'] == 'open'])} opens, {len(trades_df[trades_df['type'] == 'close'])} closes, {len(trades_df[trades_df['type'] == 'funding'])} funding")
        
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
        
        print(f"[Visualizer] Equity curve built: ${equity_curve.iloc[0]:.2f} → ${equity_curve.iloc[-1]:.2f}")
        print(f"[Visualizer] Cumulative realized P&L: ${cumulative_realized_pnl:.2f}")
        print(f"[Visualizer] Cumulative fees: ${cumulative_fees:.2f}")
        
        # Calculate final unrealized P&L
        final_unrealized_pnl = equity_curve.iloc[-1] - self.initial_capital - cumulative_realized_pnl + cumulative_fees
        print(f"[Visualizer] Final unrealized P&L: ${final_unrealized_pnl:.2f}")
        
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
        print(f"[QuantStats] Generating portfolio report for {len(symbols)} symbols")
        
        # Load price data
        try:
            price_data = self._load_multi_asset_prices(symbols, timeframe, start_date, end_date)
        except Exception as e:
            print(f"[QuantStats] Error loading price data: {e}")
            return None, {}
        
        # Convert trades to returns
        try:
            returns, benchmark_returns = self._trades_to_returns(trades_df, price_data, benchmark_symbol)
        except Exception as e:
            print(f"[QuantStats] Error calculating returns: {e}")
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
            print(f"[QuantStats] No trades found for {symbol}")
            return None, {}
        
        print(f"[QuantStats] Generating asset report for {symbol}")
        
        # Load price data for this asset
        try:
            price_series = self._load_price_data(symbol, timeframe, start_date, end_date)
            price_df = pd.DataFrame({'close': price_series})
        except Exception as e:
            print(f"[QuantStats] Error loading price data for {symbol}: {e}")
            return None, {}
        
        # Convert trades to returns
        try:
            returns, benchmark_returns = self._trades_to_returns(asset_trades, price_df, benchmark_symbol)
        except Exception as e:
            print(f"[QuantStats] Error calculating returns for {symbol}: {e}")
            return None, {}
        
        # Generate quantstats metrics
        metrics = self._extract_quantstats_metrics(returns, benchmark_returns, symbol)
        
        return returns, metrics

    def _extract_quantstats_metrics(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series] = None,
                                   symbol: str = "Portfolio") -> Dict[str, Any]:
        """Extract comprehensive metrics using quantstats-lumi built-in metrics reports."""
        
        print(f"[Visualizer] Extracting QuantStats-Lumi metrics for {symbol} with {len(returns)} returns")
        
        if returns.empty or returns.isna().all() or len(returns) == 0:
            print(f"[Visualizer] No valid returns data for {symbol}")
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
        
        # Use quantstats-lumi's comprehensive metrics report
        try:
            # Get full metrics report from quantstats-lumi
            full_metrics_report = qs.reports.metrics(returns, mode='full', display=False, benchmark=benchmark_returns)
            
            # Convert the series to a more usable dictionary format
            metrics_dict = {}
            if isinstance(full_metrics_report, pd.Series):
                for key, value in full_metrics_report.items():
                    if pd.notna(value) and value != '':
                        try:
                            # Try to convert to float if it's a numeric string
                            if isinstance(value, str):
                                # Handle percentage strings
                                if value.endswith('%'):
                                    metrics_dict[key] = float(value.rstrip('%'))
                                # Handle numeric strings
                                elif value.replace('.', '').replace('-', '').isdigit():
                                    metrics_dict[key] = float(value)
                                else:
                                    metrics_dict[key] = value
                            else:
                                metrics_dict[key] = float(value) if isinstance(value, (int, float)) else value
                        except (ValueError, TypeError):
                            metrics_dict[key] = value
            
            # Calculate final equity from cumulative returns
            final_equity = self.initial_capital * (1 + qs.stats.comp(returns))
            
            # Map quantstats-lumi metrics to our standardized format
            standardized_metrics = {
                'Symbol': symbol,
                'Total Return (%)': round(qs.stats.comp(returns) * 100, 2),
                'CAGR (%)': round(qs.stats.cagr(returns) * 100, 2),
                'Expected Return (%)': round(qs.stats.expected_return(returns) * 100, 3),
                'Volatility (%)': round(qs.stats.volatility(returns) * 100, 2),
                'Sharpe Ratio': round(qs.stats.sharpe(returns), 3),
                'Sortino Ratio': round(qs.stats.sortino(returns), 3),
                'Calmar Ratio': round(qs.stats.calmar(returns), 3),
                'Max Drawdown (%)': round(abs(qs.stats.max_drawdown(returns)) * 100, 2),
                'Ulcer Index': round(qs.stats.ulcer_index(returns), 4),
                'Recovery Factor': round(qs.stats.recovery_factor(returns), 2),
                'VaR (95%) (%)': round(qs.stats.var(returns) * 100, 2),
                'CVaR (95%) (%)': round(qs.stats.cvar(returns) * 100, 2),
                'Win Rate (%)': round(qs.stats.win_rate(returns) * 100, 2),
                'Profit Factor': round(qs.stats.profit_factor(returns), 3),
                'Gain to Pain Ratio': round(qs.stats.gain_to_pain_ratio(returns), 3),
                'Tail Ratio': round(qs.stats.tail_ratio(returns), 3),
                'Kelly Criterion': round(qs.stats.kelly_criterion(returns), 3),
                'Best Day (%)': round(returns.max() * 100, 2),
                'Worst Day (%)': round(returns.min() * 100, 2),
                'Skewness': round(returns.skew(), 3),
                'Kurtosis': round(returns.kurtosis(), 3),
                'Final Equity': round(final_equity, 2),
            }
            
            # Add benchmark comparison metrics if benchmark is provided
            if benchmark_returns is not None and not benchmark_returns.empty and len(benchmark_returns) > 1:
                print(f"[Visualizer] Adding benchmark comparison metrics")
                try:
                    aligned_benchmark = benchmark_returns.reindex(returns.index).ffill().dropna()
                    if not aligned_benchmark.empty and len(aligned_benchmark) > 1:
                        # Use quantstats-lumi functions where available
                        information_ratio = qs.stats.information_ratio(returns, aligned_benchmark)
                        r_squared = qs.stats.r_squared(returns, aligned_benchmark)
                        
                        # Calculate beta and alpha manually (not directly available)
                        covariance = returns.cov(aligned_benchmark)
                        benchmark_var = aligned_benchmark.var()
                        beta = covariance / benchmark_var if benchmark_var != 0 else 0
                        
                        portfolio_return = returns.mean() * len(returns)
                        benchmark_return = aligned_benchmark.mean() * len(aligned_benchmark)
                        alpha = (portfolio_return - beta * benchmark_return) * 100
                        
                        correlation = returns.corr(aligned_benchmark)
                        
                        standardized_metrics.update({
                            'Alpha (%)': round(alpha, 2),
                            'Beta': round(beta, 3),
                            'Information Ratio': round(information_ratio, 3),
                            'R-Squared': round(r_squared, 3),
                            'Correlation': round(correlation, 3),
                        })
                        
                        print(f"[Visualizer] Added benchmark metrics: Alpha: {alpha:.2f}%, Beta: {beta:.3f}, Info Ratio: {information_ratio:.3f}")
                except Exception as e:
                    print(f"[Visualizer] Benchmark comparison failed: {e}")
            
            print(f"[Visualizer] QuantStats-Lumi metrics extracted for {symbol}: Return: {standardized_metrics['Total Return (%)']}%, Sharpe: {standardized_metrics['Sharpe Ratio']}")
            return standardized_metrics
            
        except Exception as e:
            print(f"[Visualizer] Error using QuantStats-Lumi metrics report: {e}")
            # Fallback to individual metric calculations
            return self._fallback_metrics_calculation(returns, benchmark_returns, symbol)
    
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
        
        print(f"[Visualizer] Setting up plot generation in: {output_dir}")
        
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
        
        print(f"[Visualizer] Prepared {len(plot_files)} plot paths for quantstats generation")
        return plot_files

    def generate_html_report(self, returns: pd.Series, equity_curve: pd.Series,
                           metrics: Dict[str, Any], plot_files: Dict[str, str],
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
            print(f"🔧 Generating {title}")
            print(f"📁 Output: {output_file}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Clean and prepare returns data
            if returns.empty:
                print("⚠️  Warning: Empty returns series provided")
                returns = pd.Series([0.0], index=[pd.Timestamp.now()], name="Strategy Returns")
            
            # Clean returns data - remove infinite and extreme values
            orig_len = len(returns)
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) != orig_len:
                print(f"⚠️  Cleaned {orig_len - len(returns)} invalid returns")
            
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
            
            print(f"✅ HTML report generated successfully!")
            print(f"📄 Report saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return None
    def save_results(self, portfolio_returns: pd.Series, portfolio_metrics: Dict[str, Any],
                    asset_results: Dict[str, Tuple[pd.Series, Dict[str, Any]]],
                    trades_df: pd.DataFrame, save_dir: str,
                    start_date: datetime, end_date: datetime,
                    benchmark_symbol: Optional[str] = None):
        """Save all results with comprehensive HTML reports."""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        port_dir = os.path.join(save_dir, "portfolio_performance")
        indiv_dir = os.path.join(save_dir, "individual_asset_performance")
        plots_dir = os.path.join(save_dir, "plots")
        Path(port_dir).mkdir(parents=True, exist_ok=True)
        Path(indiv_dir).mkdir(parents=True, exist_ok=True)
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
        
        if portfolio_returns is not None and not portfolio_returns.empty:
            equity_curve = (1 + portfolio_returns).cumprod() * self.initial_capital
            
            portfolio_plots_dir = os.path.join(plots_dir, "portfolio")
            Path(portfolio_plots_dir).mkdir(exist_ok=True)
            
            plot_files = self.generate_comprehensive_plots(
                portfolio_returns, equity_curve, benchmark_returns, portfolio_plots_dir
            )
            
            portfolio_html = os.path.join(port_dir, "portfolio_performance_report.html")
            self.generate_html_report(
                portfolio_returns, equity_curve, portfolio_metrics, 
                plot_files, benchmark_returns, portfolio_html
            )
            
            portfolio_df = pd.Series(portfolio_metrics, name='Portfolio')
            portfolio_df.to_csv(os.path.join(port_dir, "portfolio_stats.csv"))
        
        asset_metrics = {}
        for symbol, (returns, metrics) in asset_results.items():
            if returns is not None and not returns.empty:
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
                
                asset_metrics[symbol] = metrics
        
        if asset_metrics:
            asset_df = pd.DataFrame(asset_metrics).T
            asset_df.to_csv(os.path.join(indiv_dir, "per_asset_stats.csv"))
        
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
                "asset_html_reports": [f"individual_asset_performance/{symbol}_performance_report.html" 
                                     for symbol in asset_results.keys()],
                "portfolio_stats": "portfolio_performance/portfolio_stats.csv",
                "asset_stats": "individual_asset_performance/per_asset_stats.csv",
                "plots_directory": "plots/"
            }
        }
        
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        trades_df.to_csv(os.path.join(save_dir, "trade_log.csv"), index=False)

    def print_summary(self, portfolio_metrics: Dict[str, Any], 
                     asset_metrics: Dict[str, Dict[str, Any]]):
        """Print a formatted summary of key metrics."""
        
        print("\n" + "="*80)
        print("PORTFOLIO PERFORMANCE SUMMARY (QuantStats-Lumi Enhanced)")
        print("="*80)
        
        # Core performance metrics
        core_metrics = ['Total Return (%)', 'CAGR (%)', 'Expected Return (%)', 'Volatility (%)']
        for metric in core_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                print(f"{metric:<30} {value:>12.2f}")
        
        print("\n" + "-"*80)
        print("RISK-ADJUSTED METRICS")
        print("-"*80)
        
        risk_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
        for metric in risk_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                print(f"{metric:<30} {value:>12.3f}")
        
        print("\n" + "-"*80)
        print("RISK METRICS")
        print("-"*80)
        
        risk_detail_metrics = ['Max Drawdown (%)', 'Ulcer Index', 'VaR (95%) (%)', 'CVaR (95%) (%)']
        for metric in risk_detail_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                print(f"{metric:<30} {value:>12.2f}")
        
        print("\n" + "-"*80)
        print("TRADING METRICS")
        print("-"*80)
        
        trading_metrics = ['Win Rate (%)', 'Profit Factor', 'Gain to Pain Ratio', 'Payoff Ratio', 'Kelly Criterion']
        for metric in trading_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                print(f"{metric:<30} {value:>12.3f}")

        if asset_metrics:
            print("\n" + "="*80)
            print("INDIVIDUAL ASSET PERFORMANCE")
            print("="*80)
            
            for symbol, metrics in asset_metrics.items():
                print(f"\n📊 {symbol}:")
                key_asset_metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                                   'Win Rate (%)', 'Profit Factor']
                for metric in key_asset_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        print(f"  {metric:<28} {value:>10.2f}")
        
        print("="*80)