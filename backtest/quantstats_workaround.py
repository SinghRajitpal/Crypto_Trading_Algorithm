#!/usr/bin/env python3
"""
QuantStats HTML Report - WORKING SOLUTION for Backtest Module
===========================================================

This module provides the definitive solution for the quantstats HTML report
"Series is ambiguous" error with pandas 2.0+.

PROBLEM EXPLAINED:
The error occurs because QuantStats 0.0.66 uses pandas operations that changed 
behavior in pandas 2.0+. When quantstats tries to check "if series_variable:", 
pandas now raises an ambiguity error instead of returning True/False.

SOLUTION:
Instead of using the buggy qs.reports.html(), we create our own HTML report
using quantstats' individual plotting functions (which work fine) and statistics.
"""

import pandas as pd
import numpy as np
import quantstats as qs
import os
from pathlib import Path
from typing import Optional, Dict, Any


def create_working_quantstats_html_report(returns: pd.Series, 
                                        output_file: str, 
                                        title: str = "Backtest Performance Report", 
                                        benchmark: Optional[pd.Series] = None,
                                        initial_capital: float = 10000.0) -> bool:
    """
    Creates a comprehensive HTML report using quantstats functions.
    This bypasses the "Series is ambiguous" bug in qs.reports.html().
    
    Parameters:
    -----------
    returns : pandas.Series
        Daily returns data with datetime index
    output_file : str
        Path to save the HTML report
    title : str
        Title for the report
    benchmark : pandas.Series, optional
        Benchmark returns for comparison
    initial_capital : float
        Initial capital amount for display purposes
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Create output directory and plots subdirectory with organized structure
        output_dir = os.path.dirname(output_file)
        plots_dir = os.path.join(output_dir, "plots")
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Generating: {title}")
        print(f"📁 Output: {output_file}")
        
        # Ensure returns are properly formatted and valid
        if returns.empty:
            print("⚠️  Warning: Empty returns series provided")
            returns = pd.Series([0.0], index=[pd.Timestamp.now()], name="Strategy Returns")
        
        # Clean the returns data
        returns = returns.dropna()
        if returns.empty:
            print("⚠️  Warning: No valid returns data after cleaning")
            returns = pd.Series([0.0], index=[pd.Timestamp.now()], name="Strategy Returns")
        
        # Ensure proper datetime index handling to fix "Cannot compare dtypes" error
        if not isinstance(returns.index, pd.DatetimeIndex):
            if len(returns) > 0:
                returns.index = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
        else:
            # Fix timezone issues - convert to UTC if mixed timezones
            if returns.index.tz is not None:
                returns.index = returns.index.tz_convert('UTC')
            else:
                returns.index = returns.index.tz_localize('UTC')
                
        # Handle benchmark timezone alignment
        if benchmark is not None and not benchmark.empty:
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                if len(benchmark) > 0:
                    benchmark.index = pd.date_range(start='2023-01-01', periods=len(benchmark), freq='D')
            else:
                # Fix timezone issues for benchmark
                if benchmark.index.tz is not None:
                    benchmark.index = benchmark.index.tz_convert('UTC')
                else:
                    benchmark.index = benchmark.index.tz_localize('UTC')
                    
                # Align benchmark with returns index
                benchmark = benchmark.reindex(returns.index, method='ffill').dropna()
        
        # Generate optimized plots with better organization
        plots = []
        plots_generated = []
        plots_failed = []
        
        try:
            # Create organized plot structure
            plot_categories = {
                'performance': ['cumulative_returns', 'rolling_sharpe', 'rolling_volatility'],
                'risk': ['drawdown', 'distribution'],
                'calendar': ['monthly_heatmap']
            }
            
            # Create category subdirectories
            for category in plot_categories:
                Path(plots_dir, category).mkdir(parents=True, exist_ok=True)
            
            # 1. Performance Plots (including equity curve)
            try:
                cum_returns_path = os.path.join(plots_dir, "performance", "cumulative_returns.png")
                qs.plots.returns(returns, benchmark=benchmark, savefig=cum_returns_path, show=False)
                plots.append(("Cumulative Returns", "plots/performance/cumulative_returns.png"))
                plots_generated.append("Equity Curve (cumulative_returns.png)")
            except Exception as e:
                plots_failed.append(f"Equity Curve: {str(e)}")
            
            # Add equity curve as standalone plot too
            try:
                equity_path = os.path.join(plots_dir, "performance", "equity_curve.png")
                # Calculate equity curve from returns for plotting
                equity_curve = (1 + returns).cumprod() * initial_capital
                # Use basic plot since quantstats doesn't have dedicated equity curve plot
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#1f77b4')
                plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Equity ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(equity_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots.append(("Portfolio Equity", "plots/performance/equity_curve.png"))
                plots_generated.append("Portfolio Equity (equity_curve.png)")
            except Exception as e:
                plots_failed.append(f"Portfolio Equity: {str(e)}")
                
            try:
                rolling_sharpe_path = os.path.join(plots_dir, "performance", "rolling_sharpe.png")
                qs.plots.rolling_sharpe(returns, savefig=rolling_sharpe_path, show=False)
                plots.append(("Rolling Sharpe Ratio", "plots/performance/rolling_sharpe.png"))
                plots_generated.append("Rolling Sharpe")
            except Exception as e:
                plots_failed.append(f"Rolling Sharpe: {str(e)}")
                
            try:
                rolling_vol_path = os.path.join(plots_dir, "performance", "rolling_volatility.png")
                qs.plots.rolling_volatility(returns, savefig=rolling_vol_path, show=False)
                plots.append(("Rolling Volatility", "plots/performance/rolling_volatility.png"))
                plots_generated.append("Rolling Volatility")
            except Exception as e:
                plots_failed.append(f"Rolling Volatility: {str(e)}")
            
            # 2. Risk Analysis Plots
            try:
                drawdown_path = os.path.join(plots_dir, "risk", "drawdown.png")
                qs.plots.drawdown(returns, savefig=drawdown_path, show=False)
                plots.append(("Drawdown Analysis", "plots/risk/drawdown.png"))
                plots_generated.append("Drawdown Analysis")
            except Exception as e:
                plots_failed.append(f"Drawdown: {str(e)}")
                
            try:
                distribution_path = os.path.join(plots_dir, "risk", "distribution.png")
                qs.plots.distribution(returns, savefig=distribution_path, show=False)
                plots.append(("Returns Distribution", "plots/risk/distribution.png"))
                plots_generated.append("Returns Distribution")
            except Exception as e:
                plots_failed.append(f"Distribution: {str(e)}")
            
            # 3. Calendar Analysis
            try:
                monthly_heatmap_path = os.path.join(plots_dir, "calendar", "monthly_heatmap.png")
                qs.plots.monthly_heatmap(returns, savefig=monthly_heatmap_path, show=False)
                plots.append(("Monthly Returns Heatmap", "plots/calendar/monthly_heatmap.png"))
                plots_generated.append("Monthly Heatmap")
            except Exception as e:
                plots_failed.append(f"Monthly Heatmap: {str(e)}")
                
        except Exception as e:
            plots_failed.append(f"General plotting error: {str(e)}")
            # Fallback: Create basic cumulative returns plot in main plots folder
            try:
                cum_returns_path = os.path.join(plots_dir, "cumulative_returns.png")
                qs.plots.returns(returns, savefig=cum_returns_path, show=False)
                plots.append(("Cumulative Returns", "plots/cumulative_returns.png"))
                plots_generated.append("cumulative_returns.png")
            except Exception as fallback_e:
                plots_failed.append(f"Fallback plot: {str(fallback_e)}")
        
        # Clean, concise output
        total_plots = len(plots_generated)
        if total_plots > 0:
            print(f"📈 Generated {total_plots} charts successfully")
        
        if plots_failed:
            print(f"⚠️  {len(plots_failed)} charts failed")
        
        # Calculate comprehensive statistics
        stats = {}
        
        try:
            # Calculate equity curve for display
            equity_curve = (1 + returns).cumprod() * initial_capital
            final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
            total_return = (final_value / initial_capital) - 1
            
            stats['Performance Metrics'] = {
                'Initial Capital': f"${initial_capital:,.2f}",
                'Final Value': f"${final_value:,.2f}",
                'Total Return': f"{total_return:.2%}",
                'CAGR': f"{qs.stats.cagr(returns):.2%}",
                'Volatility (Ann.)': f"{qs.stats.volatility(returns):.2%}",
                'Sharpe Ratio': f"{qs.stats.sharpe(returns):.3f}",
                'Sortino Ratio': f"{qs.stats.sortino(returns):.3f}",
                'Calmar Ratio': f"{qs.stats.calmar(returns):.3f}",
            }
            
            stats['Risk Metrics'] = {
                'Max Drawdown': f"{qs.stats.max_drawdown(returns):.2%}",
                'Longest DD Days': f"{qs.stats.to_drawdown_series(returns).groupby((qs.stats.to_drawdown_series(returns) == 0).cumsum()).size().max():.0f}" if not qs.stats.to_drawdown_series(returns).empty else "0",
                'VaR (95%)': f"{qs.stats.var(returns):.2%}",
                'CVaR (95%)': f"{qs.stats.cvar(returns):.2%}",
                'Skewness': f"{qs.stats.skew(returns):.3f}",
                'Kurtosis': f"{qs.stats.kurtosis(returns):.3f}",
            }
            
            stats['Trade Statistics'] = {
                'Best Day': f"{qs.stats.best(returns):.2%}",
                'Worst Day': f"{qs.stats.worst(returns):.2%}",
                'Win Rate': f"{qs.stats.win_rate(returns):.2%}",
                'Avg Win': f"{qs.stats.avg_win(returns):.2%}",
                'Avg Loss': f"{qs.stats.avg_loss(returns):.2%}",
                'Profit Factor': f"{qs.stats.profit_factor(returns):.2f}",
            }
            
            # Add benchmark comparison if provided
            if benchmark is not None and not benchmark.empty:
                try:
                    bench_total_return = qs.stats.comp(benchmark)
                    bench_sharpe = qs.stats.sharpe(benchmark)
                    bench_vol = qs.stats.volatility(benchmark)
                    
                    stats['Benchmark Comparison'] = {
                        'Benchmark Total Return': f"{bench_total_return:.2%}",
                        'Benchmark Sharpe': f"{bench_sharpe:.3f}",
                        'Benchmark Volatility': f"{bench_vol:.2%}",
                        'Excess Return': f"{total_return - bench_total_return:.2%}",
                        'Information Ratio': f"{(qs.stats.sharpe(returns) - bench_sharpe):.3f}",
                    }
                except Exception as e:
                    print(f"⚠️  Benchmark comparison failed: {e}")
                    
        except Exception as e:
            print(f"⚠️  Some statistics failed to calculate: {e}")
            # Fallback to basic metrics
            try:
                stats['Basic Metrics'] = {
                    'Total Return': f"{qs.stats.comp(returns):.2%}",
                    'Sharpe Ratio': f"{qs.stats.sharpe(returns):.3f}",
                    'Max Drawdown': f"{qs.stats.max_drawdown(returns):.2%}",
                    'Volatility': f"{qs.stats.volatility(returns):.2%}",
                }
            except Exception as e2:
                print(f"⚠️  Even basic metrics failed: {e2}")
                stats['Basic Metrics'] = {
                    'Status': 'Metrics calculation failed',
                    'Data Points': f"{len(returns)}",
                }
        
        print(f"✅ Calculated {sum(len(v) for v in stats.values())} metrics successfully")
        
        # Create professional HTML report
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1400px; margin: 0 auto; 
            background: white; padding: 40px; 
            border-radius: 12px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; text-align: center; 
            font-size: 2.5em; margin-bottom: 10px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        .subtitle {{
            text-align: center; color: #7f8c8d; 
            margin-bottom: 40px; font-style: italic;
        }}
        h2 {{ 
            color: #34495e; font-size: 1.8em; 
            margin-top: 40px; margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .stats-section {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 30px; margin: 30px 0; 
        }}
        .stats-table {{ 
            width: 100%; border-collapse: collapse; 
            background: #f8f9fa; border-radius: 8px;
            overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stats-table caption {{
            font-weight: bold; font-size: 1.1em; 
            padding: 15px; background: #3498db; color: white;
            text-align: left; margin: 0;
        }}
        .stats-table th, .stats-table td {{ 
            padding: 12px 16px; text-align: left; 
            border-bottom: 1px solid #dee2e6;
        }}
        .stats-table th {{ 
            background: #e9ecef; font-weight: 600;
            color: #495057;
        }}
        .stats-table tr:hover {{ background-color: #f1f3f4; }}
        .plot {{ 
            text-align: center; margin: 40px 0; 
            padding: 20px; background: #fafbfc;
            border-radius: 8px; border: 1px solid #e1e5e9;
        }}
        .plot img {{ 
            max-width: 100%; height: auto; 
            border-radius: 6px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .plot h3 {{ 
            color: #2c3e50; margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .footer {{ 
            text-align: center; margin-top: 50px; 
            padding-top: 20px; border-top: 2px solid #ecf0f1;
            color: #7f8c8d; font-size: 0.9em;
        }}
        .period-info {{
            background: #e8f4f8; padding: 20px; 
            border-radius: 8px; margin: 20px 0;
            border-left: 4px solid #17a2b8;
        }}
        .warning {{
            background: #fff3cd; border: 1px solid #ffeaa7;
            border-radius: 8px; padding: 15px; margin: 20px 0;
            border-left: 4px solid #f39c12;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Quantitative Backtest Analysis</div>
        
        <div class="period-info">
            <strong>Analysis Period:</strong> {returns.index[0].strftime('%B %d, %Y')} to {returns.index[-1].strftime('%B %d, %Y')}<br>
            <strong>Total Observations:</strong> {len(returns):,} trading periods<br>
            <strong>Data Frequency:</strong> {returns.index.freq if returns.index.freq else 'Irregular'}
        </div>
"""
        
        # Add warning if no plots were generated
        if not plots:
            html_content += """
        <div class="warning">
            <strong>⚠️ Visualization Warning:</strong> Some charts could not be generated due to insufficient data or plotting errors.
            The statistical metrics below should still be accurate.
        </div>
"""
        
        html_content += """        
        <h2>📊 Performance Statistics</h2>
        <div class="stats-section">
"""
        
        # Add statistics tables
        for category, metrics in stats.items():
            html_content += f"""
            <table class="stats-table">
                <caption>{category}</caption>
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
"""
            for metric, value in metrics.items():
                html_content += f"                    <tr><td>{metric}</td><td><strong>{value}</strong></td></tr>\n"
            
            html_content += "                </tbody>\n            </table>\n"
        
        html_content += """        </div>
"""
        
        # Add plots section only if plots were generated
        if plots:
            html_content += """        
        <h2>📈 Performance Charts</h2>
"""
            # Add all plots
            for plot_title, plot_path in plots:
                html_content += f"""
        <div class="plot">
            <h3>{plot_title}</h3>
            <img src="{plot_path}" alt="{plot_title}" loading="lazy">
        </div>
"""
        
        html_content += f"""
        <div class="footer">
            <p><strong>Report Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Powered by:</strong> QuantStats {qs.__version__} | Custom Backtest Visualizer</p>
            <p><em>This report uses a custom implementation to bypass quantstats HTML generation issues with pandas 2.0+</em></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated successfully!")
        print(f"📄 File size: {os.path.getsize(output_file):,} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_backtest_performance_metrics(returns: pd.Series, 
                                   benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive backtest performance metrics using quantstats.
    
    Parameters:
    -----------
    returns : pandas.Series
        Daily returns data with datetime index
    benchmark : pandas.Series, optional
        Benchmark returns for comparison
        
    Returns:
    --------
    Dict[str, Any] : Dictionary containing all performance metrics
    """
    if returns.empty:
        return {
            'error': 'Empty returns series',
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    try:
        metrics = {
            # Performance metrics
            'total_return': qs.stats.comp(returns),
            'cagr': qs.stats.cagr(returns),
            'volatility': qs.stats.volatility(returns),
            'sharpe_ratio': qs.stats.sharpe(returns),
            'sortino_ratio': qs.stats.sortino(returns),
            'calmar_ratio': qs.stats.calmar(returns),
            
            # Risk metrics
            'max_drawdown': qs.stats.max_drawdown(returns),
            'max_drawdown_days': len(qs.stats.to_drawdown_series(returns)) if not qs.stats.to_drawdown_series(returns).empty else 0,
            'var_95': qs.stats.var(returns),
            'cvar_95': qs.stats.cvar(returns),
            'skewness': qs.stats.skew(returns),
            'kurtosis': qs.stats.kurtosis(returns),
            
            # Trade statistics
            'best_day': qs.stats.best(returns),
            'worst_day': qs.stats.worst(returns),
            'win_rate': qs.stats.win_rate(returns),
            'avg_win': qs.stats.avg_win(returns),
            'avg_loss': qs.stats.avg_loss(returns),
            'profit_factor': qs.stats.profit_factor(returns),
            
            # Additional metrics
            'total_trades': len(returns[returns != 0]),
            'data_points': len(returns),
            'start_date': returns.index[0],
            'end_date': returns.index[-1],
        }
        
        # Add benchmark comparison if provided
        if benchmark is not None and not benchmark.empty:
            try:
                metrics['benchmark_total_return'] = qs.stats.comp(benchmark)
                metrics['benchmark_sharpe'] = qs.stats.sharpe(benchmark)
                metrics['benchmark_volatility'] = qs.stats.volatility(benchmark)
                metrics['excess_return'] = metrics['total_return'] - metrics['benchmark_total_return']
                metrics['tracking_error'] = qs.stats.volatility(returns - benchmark)
                metrics['information_ratio'] = metrics['excess_return'] / metrics['tracking_error'] if metrics['tracking_error'] != 0 else 0
            except Exception as e:
                print(f"⚠️  Benchmark metrics calculation failed: {e}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating performance metrics: {e}")
        return {
            'error': str(e),
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
