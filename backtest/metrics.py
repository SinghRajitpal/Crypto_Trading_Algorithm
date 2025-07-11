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

        # Sharpe (daily)
        returns = equity.pct_change().dropna()
        if not returns.empty:
            self._sharpe = (returns.mean() / returns.std()) * np.sqrt(1440)  # assuming minute data ~1440 per day
        else:
            self._sharpe = np.nan
