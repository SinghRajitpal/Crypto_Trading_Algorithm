"""backtest/metrics.py

Enhanced metrics extraction for backtesting using QuantStats-Lumi.
Provides comprehensive performance analysis with QuantStats integration.
"""

import pandas as pd
import numpy as np
from datetime import UTC
from typing import Dict, Any, Optional
import quantstats_lumi as qs


class Metrics:
    """
    Enhanced metrics class leveraging QuantStats-Lumi for comprehensive analysis.
    
    This class processes trade data and provides both basic summary metrics
    and full QuantStats-powered analysis including returns, risk metrics,
    and performance indicators.
    """

    def __init__(self, trade_log: pd.DataFrame, initial_capital: float):
        self.trade_log = trade_log.copy()
        self.initial_capital = initial_capital
        self._prepare()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        """Return the equity curve for QuantStats-Lumi analysis."""
        return self._equity

    def returns_series(self) -> pd.Series:
        """Return the returns series for QuantStats-Lumi analysis."""
        return self._equity.pct_change().dropna()

    def quantstats_metrics(self, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics using QuantStats-Lumi functions.
        
        Args:
            benchmark: Optional benchmark returns series for comparison.
            
        Returns:
            Dictionary of QuantStats metrics.
        """
        returns = self.returns_series()
        
        if len(returns) == 0:
            return {"error": "No returns data available"}
        
        try:
            # Core performance metrics
            metrics = {
                # Return metrics
                "total_return": qs.stats.comp(returns),
                "cagr": qs.stats.cagr(returns),
                "annualized_return": qs.stats.cagr(returns),
                
                # Risk metrics
                "volatility": qs.stats.volatility(returns),
                "max_drawdown": qs.stats.max_drawdown(returns),
                "calmar_ratio": qs.stats.calmar(returns),
                "ulcer_index": qs.stats.ulcer_index(returns),
                
                # Risk-adjusted returns
                "sharpe_ratio": qs.stats.sharpe(returns),
                "sortino_ratio": qs.stats.sortino(returns),
                
                # Trading metrics
                "win_rate": qs.stats.win_rate(returns),
                "avg_win": qs.stats.avg_win(returns),
                "avg_loss": qs.stats.avg_loss(returns),
                "profit_factor": qs.stats.profit_factor(returns),
                "payoff_ratio": qs.stats.payoff_ratio(returns),
                
                # Risk metrics
                "var_95": qs.stats.var(returns, confidence=0.95),
                "cvar_95": qs.stats.cvar(returns, confidence=0.95),
                "kelly_criterion": qs.stats.kelly_criterion(returns),
                
                # Advanced metrics
                "gain_to_pain_ratio": qs.stats.gain_to_pain_ratio(returns),
                "recovery_factor": qs.stats.recovery_factor(returns),
                "risk_return_ratio": qs.stats.risk_return_ratio(returns),
                
                # Additional trading stats
                "total_trades": int((self.trade_log["type"] == "close").sum()),
                "winning_trades": int((self.trade_log.loc[self.trade_log["type"] == "close", "pnl"] > 0).sum()),
                "final_equity": float(self._equity.iloc[-1]),
                "initial_capital": float(self.initial_capital),
            }
            
            # Add benchmark comparison if provided
            if benchmark is not None and len(benchmark) > 0:
                try:
                    # Align returns and benchmark
                    aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')
                    
                    if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
                        metrics.update({
                            "alpha": qs.stats.alpha(aligned_returns, aligned_benchmark),
                            "beta": qs.stats.beta(aligned_returns, aligned_benchmark),
                            "information_ratio": qs.stats.information_ratio(aligned_returns, aligned_benchmark),
                            "r_squared": qs.stats.r_squared(aligned_returns, aligned_benchmark),
                        })
                except Exception as e:
                    metrics["benchmark_error"] = f"Benchmark comparison failed: {e}"
            
            # Calculate drawdown periods and other time-based metrics
            try:
                drawdown = qs.stats.to_drawdown_series(returns)
                metrics.update({
                    "avg_drawdown": drawdown.mean(),
                    "drawdown_details": qs.stats.drawdown_details(drawdown).to_dict() if len(drawdown) > 0 else {}
                })
                
                # Note: max_drawdown_duration might not be available in this version
                # Using alternative calculation if needed
                try:
                    metrics["max_drawdown_duration"] = qs.stats.max_drawdown_duration(returns)
                except AttributeError:
                    # Calculate manually if function not available
                    dd_series = qs.stats.to_drawdown_series(returns)
                    if len(dd_series) > 0:
                        # Find longest consecutive period below peak
                        underwater = dd_series < 0
                        if underwater.any():
                            underwater_periods = underwater.astype(int).groupby((~underwater).cumsum()).sum()
                            metrics["max_drawdown_duration"] = underwater_periods.max()
                        else:
                            metrics["max_drawdown_duration"] = 0
                    else:
                        metrics["max_drawdown_duration"] = 0
                        
            except Exception as e:
                metrics["drawdown_analysis_error"] = f"Drawdown analysis failed: {e}"
            
            return metrics
            
        except Exception as e:
            return {"error": f"QuantStats calculation failed: {e}"}

    def summary(self) -> Dict[str, Any]:
        """
        Return enhanced summary combining basic metrics with key QuantStats indicators.
        
        This provides immediate feedback while full QuantStats analysis is generated.
        """
        # Basic trade statistics
        total_trades = int((self.trade_log["type"] == "close").sum())
        winning_trades = int((self.trade_log.loc[self.trade_log["type"] == "close", "pnl"] > 0).sum())
        
        basic_stats = {
            "final_equity": float(self._equity.iloc[-1]),
            "total_return_pct": float((self._equity.iloc[-1] / self.initial_capital - 1) * 100),
            "trade_count": total_trades,
            "winning_trades": winning_trades,
            "win_rate_pct": float((winning_trades / total_trades * 100) if total_trades > 0 else 0),
            "initial_capital": float(self.initial_capital),
        }
        
        # Add key QuantStats metrics for immediate insight
        try:
            returns = self.returns_series()
            if len(returns) > 0:
                basic_stats.update({
                    "sharpe_ratio": float(qs.stats.sharpe(returns)),
                    "max_drawdown_pct": float(qs.stats.max_drawdown(returns) * 100),
                    "volatility_pct": float(qs.stats.volatility(returns) * 100),
                    "sortino_ratio": float(qs.stats.sortino(returns)),
                })
        except Exception as e:
            basic_stats["quantstats_error"] = f"QuantStats summary failed: {e}"
        
        return basic_stats

    def comprehensive_report(self, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report combining all metrics.
        
        Args:
            benchmark: Optional benchmark for comparison.
            
        Returns:
            Complete performance analysis report.
        """
        return {
            "basic_summary": self.summary(),
            "quantstats_metrics": self.quantstats_metrics(benchmark),
            "equity_curve_stats": self._analyze_equity_curve(),
            "trade_analysis": self._analyze_trades(),
            "period_analysis": self._analyze_periods(),
        }

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _prepare(self):
        """Prepare equity curve from trade log with comprehensive cost accounting."""
        tl = self.trade_log
        tl["timestamp"] = pd.to_datetime(tl["timestamp"], utc=True)
        tl.set_index("timestamp", inplace=True)

        # Cash flow series: comprehensive cost accounting
        cash_flow = pd.Series(0.0, index=tl.index.unique()).sort_index()

        # Trading fees (always negative)
        if "fee" in tl.columns:
            cash_flow = cash_flow.add(tl["fee"].fillna(0).mul(-1), fill_value=0)

        # Realised PnL from close rows
        if "pnl" in tl.columns:
            cash_flow = cash_flow.add(tl.loc[tl["type"] == "close", "pnl"], fill_value=0)

        # Funding payments (can be positive or negative)
        if "payment" in tl.columns:
            cash_flow = cash_flow.add(tl.loc[tl["type"] == "funding", "payment"], fill_value=0)

        # Additional cost accounting if available
        if "slippage_cost" in tl.columns:
            cash_flow = cash_flow.add(tl["slippage_cost"].fillna(0).mul(-1), fill_value=0)
            
        if "commission" in tl.columns:
            cash_flow = cash_flow.add(tl["commission"].fillna(0).mul(-1), fill_value=0)

        cash_flow = cash_flow.sort_index()
        equity = cash_flow.cumsum().add(self.initial_capital)
        self._equity = equity

    def _analyze_equity_curve(self) -> Dict[str, Any]:
        """Analyze equity curve characteristics."""
        equity = self._equity
        
        return {
            "min_equity": float(equity.min()),
            "max_equity": float(equity.max()),
            "equity_volatility": float(equity.pct_change().std() * np.sqrt(252)),
            "underwater_periods": len(equity[equity < equity.cummax()]),
            "total_periods": len(equity),
            "time_underwater_pct": float(len(equity[equity < equity.cummax()]) / len(equity) * 100),
        }

    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze individual trade characteristics."""
        close_trades = self.trade_log[self.trade_log["type"] == "close"]
        
        if len(close_trades) == 0:
            return {"error": "No closed trades found"}
        
        pnl_trades = close_trades["pnl"].dropna()
        
        return {
            "avg_trade_pnl": float(pnl_trades.mean()),
            "median_trade_pnl": float(pnl_trades.median()),
            "best_trade": float(pnl_trades.max()),
            "worst_trade": float(pnl_trades.min()),
            "trade_pnl_std": float(pnl_trades.std()),
            "profitable_trades": int((pnl_trades > 0).sum()),
            "losing_trades": int((pnl_trades < 0).sum()),
            "breakeven_trades": int((pnl_trades == 0).sum()),
        }

    def _analyze_periods(self) -> Dict[str, Any]:
        """Analyze performance over different time periods."""
        returns = self.returns_series()
        
        if len(returns) == 0:
            return {"error": "No returns data"}
        
        try:
            # Monthly analysis if enough data
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            analysis = {
                "total_periods": len(returns),
                "positive_periods": int((returns > 0).sum()),
                "negative_periods": int((returns < 0).sum()),
                "avg_daily_return": float(returns.mean()),
                "daily_return_std": float(returns.std()),
            }
            
            if len(monthly_returns) > 0:
                analysis.update({
                    "monthly_periods": len(monthly_returns),
                    "positive_months": int((monthly_returns > 0).sum()),
                    "negative_months": int((monthly_returns < 0).sum()),
                    "avg_monthly_return": float(monthly_returns.mean()),
                    "best_month": float(monthly_returns.max()),
                    "worst_month": float(monthly_returns.min()),
                })
            
            return analysis
            
        except Exception as e:
            return {"error": f"Period analysis failed: {e}"}
