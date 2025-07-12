import pandas as pd
import numpy as np
from datetime import UTC


class Metrics:
    """Compute basic back-test performance statistics."""

    def __init__(self, trade_log: pd.DataFrame, initial_capital: float):
        self.trade_log = trade_log.copy()
        self.initial_capital = initial_capital
        self._prepare()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        return self._equity

    def summary(self) -> dict:
        return {
            "final_equity": self._equity.iloc[-1],
            "total_return_pct": (self._equity.iloc[-1] / self.initial_capital - 1) * 100,
            "max_drawdown_pct": self._max_dd * 100,
            "sharpe": self._sharpe,
            "trade_count": int((self.trade_log["type"] == "close").sum()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare(self):
        tl = self.trade_log
        tl["timestamp"] = pd.to_datetime(tl["timestamp"], utc=True)
        tl.set_index("timestamp", inplace=True)

        # Cash flow series: fees and realised PnL (from close rows) + funding payments
        cash_flow = pd.Series(0.0, index=tl.index.unique()).sort_index()

        # Fees (always negative)
        if "fee" in tl.columns:
            cash_flow = cash_flow.add(tl["fee"].fillna(0).mul(-1), fill_value=0)

        # Realised PnL from close rows
        if "pnl" in tl.columns:
            cash_flow = cash_flow.add(tl.loc[tl["type"] == "close", "pnl"], fill_value=0)

        # Funding payments (can be positive or negative)
        if "payment" in tl.columns:
            cash_flow = cash_flow.add(tl.loc[tl["type"] == "funding", "payment"], fill_value=0)

        cash_flow = cash_flow.sort_index()
        equity = cash_flow.cumsum().add(self.initial_capital)
        self._equity = equity

        # Drawdown
        roll_max = equity.cummax()
        dd = 1 - equity / roll_max
        self._max_dd = dd.max()

        # --------------------------------------------------------------
        # Sharpe ratio (annualised – match vectorbt)
        # --------------------------------------------------------------
        # VectorBT annualises returns by multiplying by ``sqrt(freq)`` where
        # ``freq`` is the number of observations per *year* inferred from the
        # index (e.g. 365-days × 24h × 12 × 5-minute bars = 105 120).
        # We replicate that formula here so that the Sharpe values in our
        # custom Metrics summary match those shown in the VectorBT stats
        # table and in most portfolio-analysis literature.

        returns = equity.pct_change().dropna()

        if not returns.empty and returns.std() != 0:
            periods_per_year = 365 * 1440  # fallback assumption: 1-minute bars

            try:
                freq_str = pd.infer_freq(equity.index)
                if freq_str is not None:
                    offset = pd.tseries.frequencies.to_offset(freq_str)
                    seconds = offset.delta.total_seconds()
                    if seconds > 0:
                        periods_per_year = int(31557600 // seconds)  # 365.25 days
            except Exception:
                pass  # keep default

            self._sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            self._sharpe = np.nan
