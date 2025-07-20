"""backtest/visualizer.py

QuantStats-based visualization system for backtesting analysis.
Uses a custom quantstats workaround to bypass pandas 2.0+ compatibility issues.
Provides professional-grade performance analytics and tear sheet reports.
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import quantstats as qs

# Import the quantstats workaround for HTML generation
try:
    from .quantstats_workaround import create_working_quantstats_html_report
except ImportError:
    from quantstats_workaround import create_working_quantstats_html_report

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
_CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "cache")

# Configure quantstats
qs.extend_pandas()


class QuantStatsVisualizer:
    """
    Professional backtesting visualization system with QuantStats metrics.
    
    Features:
    - Comprehensive QuantStats metrics (18+ professional indicators)
    - Custom quantstats HTML report generation (bypasses pandas 2.0+ issues)
    - Professional HTML reports with embedded charts
    - Clean CSV/JSON exports with all data
    - Reliable visualization without matplotlib dependencies
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
        
        if start is not None:
            close_series = close_series[close_series.index >= start]
        if end is not None:
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
            
        equity_curve = self._trades_to_equity_curve(trades_df, price_data)
        
        # Ensure final equity matches actual portfolio value from trades
        # Calculate final portfolio value from trades
        total_pnl = trades_df[trades_df['type'] == 'close']['pnl'].sum() if 'pnl' in trades_df.columns else 0
        total_fees = trades_df['fee'].sum() if 'fee' in trades_df.columns else 0
        final_equity_from_trades = self.initial_capital + total_pnl - total_fees
        
        # Adjust the equity curve to match the final trade value
        if not equity_curve.empty and final_equity_from_trades != equity_curve.iloc[-1]:
            adjustment_factor = final_equity_from_trades / equity_curve.iloc[-1]
            equity_curve = equity_curve * adjustment_factor
        
        returns = equity_curve.pct_change().dropna()
        returns.name = 'Strategy'
        
        # Remove timezone info for quantstats compatibility
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        benchmark_returns = None
        if benchmark_symbol and benchmark_symbol in price_data.columns:
            benchmark_returns = self._get_benchmark_returns(price_data, benchmark_symbol)
        
        return returns, benchmark_returns

    def _get_benchmark_returns(self, price_data: pd.DataFrame, benchmark_symbol: str) -> pd.Series:
        """Calculate benchmark returns from price data."""
        if benchmark_symbol not in price_data.columns:
            return None
            
        benchmark_prices = price_data[benchmark_symbol].dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_returns.name = f'{benchmark_symbol}_benchmark'
        
        # Remove timezone info for quantstats compatibility
        if benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)
        
        return benchmark_returns

    def _trades_to_equity_curve(self, trades_df: pd.DataFrame, 
                               price_data: pd.DataFrame) -> pd.Series:
        """Convert trade log to equity curve time series."""
        if trades_df.empty:
            return pd.Series(self.initial_capital, index=price_data.index, name='equity')
            
        trades_df = trades_df.copy()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
        
        cash_flows = []
        current_positions = {}
        
        for _, trade in trades_df.iterrows():
            timestamp = trade['timestamp']
            symbol = trade['symbol']
            trade_type = trade['type']
            
            if trade_type == 'open':
                margin = trade.get('margin', 0)
                fee = trade.get('fee', 0)
                cash_flows.append({
                    'timestamp': timestamp,
                    'cash_flow': -(margin + fee),
                    'symbol': symbol,
                    'type': 'open'
                })
                
                current_positions[symbol] = {
                    'contracts': trade['contracts'] if trade['side'] == 'buy' else -trade['contracts'],
                    'entry_price': trade['price'],
                    'margin': margin
                }
                
            elif trade_type == 'close':
                if symbol in current_positions:
                    pos = current_positions[symbol]
                    margin_released = pos['margin']
                    pnl = trade.get('pnl', 0)
                    fee = trade.get('fee', 0)
                    
                    cash_flows.append({
                        'timestamp': timestamp,
                        'cash_flow': margin_released + pnl - fee,
                        'symbol': symbol,
                        'type': 'close'
                    })
                    
                    del current_positions[symbol]
                    
            elif trade_type == 'funding':
                payment = trade.get('payment', 0)
                cash_flows.append({
                    'timestamp': timestamp,
                    'cash_flow': -payment,
                    'symbol': symbol,
                    'type': 'funding'
                })
        
        if not cash_flows:
            return pd.Series(self.initial_capital, index=price_data.index, name='equity')
            
        cf_df = pd.DataFrame(cash_flows)
        cf_df = cf_df.set_index('timestamp')
        cf_series = cf_df.groupby('timestamp')['cash_flow'].sum()
        
        equity_curve = pd.Series(index=price_data.index, dtype=float, name='equity')
        current_equity = self.initial_capital
        equity_curve = equity_curve.fillna(current_equity)
        
        for timestamp, cash_flow in cf_series.items():
            try:
                closest_idx = equity_curve.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx < len(equity_curve):
                    current_equity += cash_flow
                    equity_curve.iloc[closest_idx:] = current_equity
            except (IndexError, KeyError):
                continue
        
        equity_curve = equity_curve.ffill()
        
        # Add unrealized PnL for open positions
        for i, timestamp in enumerate(equity_curve.index):
            unrealized_pnl = 0
            for symbol, pos in current_positions.items():
                if symbol in price_data.columns:
                    try:
                        current_price = price_data.loc[timestamp, symbol]
                        if not pd.isna(current_price):
                            unrealized_pnl += (current_price - pos['entry_price']) * pos['contracts']
                    except (KeyError, IndexError):
                        continue
            
            equity_curve.iloc[i] += unrealized_pnl
        
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
        """Extract comprehensive metrics using quantstats."""
        
        if returns.empty or returns.isna().all():
            return {
                'Symbol': symbol,
                'Total Return (%)': 0.0,
                'CAGR (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Calmar Ratio': 0.0,
                'Total Trades': 0,
                'Win Rate (%)': 0.0,
            }
        
        def safe_calc(func, default=0.0):
            try:
                result = func()
                if pd.isna(result) or np.isinf(result):
                    return default
                return result
            except (ZeroDivisionError, ValueError, RuntimeWarning):
                return default
        
        total_return = safe_calc(lambda: qs.stats.comp(returns) * 100)
        cagr = safe_calc(lambda: qs.stats.cagr(returns) * 100)
        volatility = safe_calc(lambda: qs.stats.volatility(returns) * 100)
        
        sharpe = safe_calc(lambda: qs.stats.sharpe(returns))
        sortino = safe_calc(lambda: qs.stats.sortino(returns))
        calmar = safe_calc(lambda: qs.stats.calmar(returns))
        
        max_drawdown = safe_calc(lambda: abs(qs.stats.max_drawdown(returns)) * 100)
        var = safe_calc(lambda: qs.stats.var(returns) * 100, -1.0)
        cvar = safe_calc(lambda: qs.stats.cvar(returns) * 100, -1.0)
        
        win_rate = safe_calc(lambda: qs.stats.win_rate(returns) * 100)
        win_loss_ratio = safe_calc(lambda: qs.stats.win_loss_ratio(returns), 1.0)
        kelly = safe_calc(lambda: qs.stats.kelly_criterion(returns))
        payoff_ratio = safe_calc(lambda: qs.stats.payoff_ratio(returns), 1.0)
        
        best_month = safe_calc(lambda: qs.stats.best(returns) * 100)
        worst_month = safe_calc(lambda: qs.stats.worst(returns) * 100)
        
        skew = safe_calc(lambda: qs.stats.skew(returns))
        kurtosis = safe_calc(lambda: qs.stats.kurtosis(returns))
        
        metrics = {
            'Symbol': symbol,
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Calmar Ratio': round(calmar, 3),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'VaR (95%) (%)': round(var, 2),
            'CVaR (95%) (%)': round(cvar, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Win/Loss Ratio': round(win_loss_ratio, 3),
            'Payoff Ratio': round(payoff_ratio, 3),
            'Kelly Criterion': round(kelly, 3),
            'Best Month (%)': round(best_month, 2),
            'Worst Month (%)': round(worst_month, 2),
            'Skewness': round(skew, 3),
            'Kurtosis': round(kurtosis, 3),
        }
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                aligned_benchmark = benchmark_returns.reindex(returns.index).ffill().dropna()
                if not aligned_benchmark.empty:
                    try:
                        if hasattr(qs.stats, 'beta'):
                            beta = qs.stats.beta(returns, aligned_benchmark)
                        else:
                            covariance = returns.cov(aligned_benchmark)
                            benchmark_var = aligned_benchmark.var()
                            beta = covariance / benchmark_var if benchmark_var != 0 else 0
                            
                        alpha = qs.stats.alpha(returns, aligned_benchmark) * 100
                        r_squared = qs.stats.r_squared(returns, aligned_benchmark)
                        
                        metrics.update({
                            'Alpha (%)': round(alpha, 2),
                            'Beta': round(beta, 3),
                            'R-Squared': round(r_squared, 3),
                        })
                    except Exception:
                        try:
                            alpha = qs.stats.alpha(returns, aligned_benchmark) * 100
                            r_squared = qs.stats.r_squared(returns, aligned_benchmark)
                            metrics.update({
                                'Alpha (%)': round(alpha, 2),
                                'R-Squared': round(r_squared, 3),
                            })
                        except:
                            pass
            except Exception:
                pass
        
        return metrics

    def generate_comprehensive_plots(self, returns: pd.Series, 
                                   equity_curve: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   output_dir: str = "plots") -> Dict[str, str]:
        """
        Simplified plot generation that relies on quantstats workaround.
        
        The quantstats workaround handles all plotting internally, so we just
        create the output directory and return an empty plot files dict.
        All visualization is handled by the HTML report generation.
        """
        
        print(f"[Visualizer] Using quantstats workaround for plotting - no matplotlib required")
        
        # Create output directory (may be used by quantstats internally)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Return empty dict since quantstats workaround handles all plotting
        print(f"[Visualizer] All plots will be generated by quantstats workaround in HTML report")
        return {}

    def generate_html_report(self, returns: pd.Series, equity_curve: pd.Series,
                           metrics: Dict[str, Any], plot_files: Dict[str, str],
                           benchmark_returns: Optional[pd.Series] = None,
                           output_file: str = "performance_report.html") -> str:
        """Generate comprehensive HTML performance report using the quantstats workaround."""
        
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
        
        success = create_working_quantstats_html_report(
            returns=returns,
            output_file=output_file,
            title=title,
            benchmark=benchmark_returns,
            initial_capital=self.initial_capital
        )
        
        return output_file if success else None
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
                    benchmark_returns = benchmark_prices.pct_change().dropna()
                    benchmark_returns.name = f'{benchmark_symbol}_benchmark'
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
        
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*60)
        
        key_metrics = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 
                      'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 
                      'Win Rate (%)', 'Calmar Ratio']
        
        for metric in key_metrics:
            if metric in portfolio_metrics:
                value = portfolio_metrics[metric]
                print(f"{metric:<25} {value:>10.2f}")
        
        if asset_metrics:
            print("\n" + "-"*60)
            print("INDIVIDUAL ASSET PERFORMANCE")
            print("-"*60)
            
            for symbol, metrics in asset_metrics.items():
                print(f"\n{symbol}:")
                for metric in ['Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)']:
                    if metric in metrics:
                        value = metrics[metric]
                        print(f"  {metric:<23} {value:>10.2f}")
        
        print("="*60)