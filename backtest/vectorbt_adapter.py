"""backtest/vectorbt_adapter.py
Helper utilities to visualise a SimBroker trade-log with *vectorbt*.

Two public helpers are exposed:

    load_close_series(symbol, timeframe)
        → pd.Series of close prices indexed by UTC timestamp.

    portfolio_from_trades(trades, close_ser, init_cash=10_000)
        → vbt.Portfolio object that can be inspected or plotted.

The implementation purposely keeps assumptions minimal:
• Only the *close* column from cached OHLCV CSV is required.
• The trade-log rows must include: timestamp, side (buy/sell), contracts, fee.
• We aggregate multiple orders that fall on the same bar.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# Third-party – fail early with helpful message
try:
    import vectorbt as vbt  # noqa: F401
except ImportError as e:  # pragma: no cover – handled by caller
    raise


# ---------------------------------------------------------------------------
# Paths & CSV loading helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "cache")


def _ohlcv_cache_path(symbol: str, timeframe: str) -> str:
    """Return absolute path to cached OHLCV CSV for *symbol*-*timeframe*."""

    file_safe = symbol.replace("/", "")
    return os.path.join(_CACHE_DIR, f"{file_safe}-{timeframe}.csv")


def load_close_series(symbol: str, timeframe: str, start=None, end=None) -> pd.Series:
    """Load close-price series from cached CSV and return as pd.Series (float).

    Optional *start* / *end* parameters (datetime or str) limit the returned
    series to the desired date range so that vectorbt stats reflect exactly
    the simulated back-test period.
    """

    csv_path = _ohlcv_cache_path(symbol, timeframe)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cached OHLCV file not found: {csv_path}")

    # Accept both the new header format and legacy files without it
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    except ValueError:
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        df.index.name = "timestamp"

    # Ensure UTC timezone awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    ser = df["close"].astype(float)

    if start is not None:
        ser = ser.loc[pd.to_datetime(start, utc=True):]
    if end is not None:
        ser = ser.loc[:pd.to_datetime(end, utc=True)]

    return ser


# ---------------------------------------------------------------------------
# Internal helpers – trade-log aggregation
# ---------------------------------------------------------------------------


def _clip_trades_to_index(trades: pd.DataFrame, price_index: pd.Index) -> pd.DataFrame:
    """Return *trades* limited to the min/max timestamp of *price_index*.

    SimBroker keeps recording rows even before/after the requested period
    (e.g. when the CSV price cache is trimmed by ``--days``).  Any order that
    falls outside the close-price range would be skipped later in
    ``_build_order_size_series`` and thus silently disappear – leading to an
    inconsistent portfolio (few orders, wrong equity).

    By clipping explicitly we make the behaviour obvious *and* allow us to
    report how many rows were discarded for transparency.
    """

    if trades.empty:
        return trades

    ts_start = price_index[0]
    ts_end = price_index[-1]

    # Ensure datetime & UTC tz for comparison
    ts_col = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    mask = (ts_col >= ts_start) & (ts_col <= ts_end)

    dropped = (~mask).sum()
    if dropped:
        print(f"[VectorbtAdapter] ⚠️ Dropping {dropped} trade-log rows outside price range ({ts_start} → {ts_end}).")

    return trades.loc[mask].copy()


def _build_order_size_series(trades: pd.DataFrame, price_index: pd.Index) -> pd.Series:
    """Aggregate trade-log rows into signed size orders aligned to *price_index*."""

    # Using 0.0 avoids NaN propagation during in-place additions which can
    # otherwise lead to all‐zero order arrays and hence NaN stats in
    # vectorbt outputs.  We still align to the same *price_index*.
    size_series = pd.Series(0.0, index=price_index, dtype=float)

    if trades.empty:
        return size_series

    for _, row in trades.iterrows():
        # Skip funding rows or any without 'side'
        side_val = row.get("side")
        if not isinstance(side_val, str):
            continue

        ts_raw = pd.to_datetime(row["timestamp"], utc=True)

        # Map trade timestamp to the nearest *existing* bar index so that the
        # order is captured by vectorbt.  Use "nearest" method with \n
        if ts_raw in size_series.index:
            ts_bar = ts_raw
        else:
            # Find insertion point; pick the previous bar (most conservative)
            pos = size_series.index.searchsorted(ts_raw, side="right")
            if pos == 0:
                # Trade occurred before first bar – skip
                continue
            ts_bar = size_series.index[pos - 1]

        size = float(row["contracts"])
        if side_val.lower() == "sell":
            size *= -1  # sell → negative size

        # Multiple trades in same bar simply add up
        size_series.loc[ts_bar] += size

    return size_series


def _build_fee_series(trades: pd.DataFrame, price_index: pd.Index) -> pd.Series:
    fee_series = pd.Series(0.0, index=price_index, dtype=float)
    if "fee" not in trades.columns or trades["fee"].isna().all():
        return fee_series

    for _, row in trades.iterrows():
        ts_raw = pd.to_datetime(row["timestamp"], utc=True)
        if ts_raw in fee_series.index:
            ts_bar = ts_raw
        else:
            pos = fee_series.index.searchsorted(ts_raw, side="right")
            if pos == 0:
                continue
            ts_bar = fee_series.index[pos - 1]

        fee_series.loc[ts_bar] += float(row.get("fee", 0.0))

    return fee_series


def _build_funding_series(trades: pd.DataFrame, price_index: pd.Index) -> pd.Series:
    """Aggregate *funding* payments into cash-flow per bar (positive → cash IN).

    Vectorbt represents cash movement via the *cash_flow* array where
    positive numbers increase equity and negative decrease.  SimBroker stores
    a *payment* column (positive when we **pay** funding).  Therefore the sign
    is reversed here so that a payment (cash outflow) becomes a **negative**
    cash_flow value.
    """

    cash_ser = pd.Series(0.0, index=price_index, dtype=float)

    if "type" not in trades.columns or "payment" not in trades.columns:
        return cash_ser

    funding_rows = trades[trades["type"] == "funding"]
    if funding_rows.empty:
        return cash_ser

    for _, row in funding_rows.iterrows():
        ts_raw = pd.to_datetime(row["timestamp"], utc=True)
        if ts_raw in cash_ser.index:
            ts_bar = ts_raw
        else:
            pos = cash_ser.index.searchsorted(ts_raw, side="right")
            if pos == 0:
                continue
            ts_bar = cash_ser.index[pos - 1]

        cash_ser.loc[ts_bar] += -float(row.get("payment", 0.0))  # negative for payment

    return cash_ser


def portfolio_from_trades(
    trades: pd.DataFrame,
    close_ser: pd.Series,
    init_cash: float = 10_000.0,
    fees: Optional[pd.Series] = None,
):
    """Convert SimBroker *trades* DataFrame into a *vectorbt* Portfolio.

    Parameters
    ----------
    trades : pd.DataFrame
        Broker trade-log with at least columns: timestamp, side, type, contracts, fee.
    close_ser : pd.Series
        Close price series aligned to desired bar frequency.
    init_cash : float, default 10_000.0
        Starting equity for the portfolio.
    fees : pd.Series, optional
        Pre-computed fee series aligned to *close_ser* index; if omitted it will
        be derived from the trade-log.
    """

    # Ensure the price series is chronological & unique
    close_ser = close_ser.sort_index()

    # ------------------------------------------------------------------
    # 1) Restrict trade-log to the same period as *close_ser*
    # ------------------------------------------------------------------
    trades = _clip_trades_to_index(trades, close_ser.index)

    # ------------------------------------------------------------------
    # 2) Build order size / fee / funding series
    # ------------------------------------------------------------------

    size_ser = _build_order_size_series(trades, close_ser.index)
    if fees is None:
        fees = _build_fee_series(trades, close_ser.index)

    # ------------------------------------------------------------------
    # Fees and funding – both treated as absolute *fixed_fees*
    # ------------------------------------------------------------------

    funding_ser = _build_funding_series(trades, close_ser.index)

    # fixed_fee = commissions + funding (funding already signed: payment -> +)
    fixed_fee_ser = fees.add(funding_ser, fill_value=0.0)

    size_arr = size_ser.fillna(0.0).values
    # No percentage-based fees
    fee_arr = pd.Series(0.0, index=close_ser.index).values
    fixed_fee_arr = fixed_fee_ser.values

    # Infer frequency string for vectorbt (e.g. '5T') – fall back to None
    try:
        freq_inferred = pd.infer_freq(close_ser.index)
    except ValueError:
        freq_inferred = None

    # Debugging statistics ---------------------------------------------------
    nonzero_orders = (size_arr != 0).sum()
    total_fees = fixed_fee_arr.sum()
    print("[VectorbtAdapter] Debug:")
    print(f"  Bars: {len(close_ser)}, Non-zero orders: {nonzero_orders}")
    print(f"  Total contracts traded: {abs(size_arr).sum():.4f}")
    print(f"  Total fees (USDT): {total_fees:.4f}")

    # Build portfolio – add funding via fixed_fees array
    pf = vbt.Portfolio.from_orders(
        close=close_ser,
        size=size_arr,
        price=close_ser,
        size_type="amount",
        fees=fee_arr,          # zero – we use fixed_fees only
        fixed_fees=fixed_fee_arr,
        freq=freq_inferred,
        cash_sharing=True,
        init_cash=init_cash,
    )

    return pf

# ===========================================================================
# Multi-asset helpers (experimental) ========================================
# ===========================================================================


def load_close_dataframe(symbol_pairs, start=None, end=None):
    """Load close-price history for *multiple* symbol / timeframe pairs.

    Parameters
    ----------
    symbol_pairs : list[tuple[str, str]]
        Sequence of ``(symbol, timeframe)`` pairs, e.g. ``[("BTCUSDT", "5m"), ("ETHUSDT", "5m")]``.
        All series will be aligned to a **union** index and forward-filled so that vectorbt
        receives a dense price matrix.
    start, end : datetime | str | None, optional
        Optional date bounds identical to :pyfunc:`load_close_series`.

    Returns
    -------
    pd.DataFrame
        Columns named by *symbol* (string) and a UTC-indexed DateTimeIndex.
    """

    close_sers = {}
    for sym, tf in symbol_pairs:
        try:
            close_sers[sym] = load_close_series(sym, tf, start=start, end=end)
        except FileNotFoundError as exc:
            print(f"[VectorbtAdapter] ⚠️  Price cache missing for {sym} {tf}: {exc}")
            continue

    if not close_sers:
        raise ValueError("No close-price series available – check cache downloads.")

    # Build DataFrame on the *union* of all indices – then fill tiny gaps.
    df = pd.concat(close_sers, axis=1).sort_index()

    # Forward-fill, then back-fill to remove leading NaNs that would otherwise
    # propagate into vectorbt and show up as NaNs in statistics.
    df = df.ffill().bfill().dropna(how="all")

    # Flatten the column MultiIndex (if any) to simple symbol names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def portfolio_from_trades_multi(trades, close_df, init_cash=10_000.0):
    """Build a *vectorbt* Portfolio for **multiple** assets simultaneously.

    This mirrors :pyfunc:`portfolio_from_trades` but accepts:

    * ``close_df`` – *DataFrame* with one column per symbol.
    * Trades for *all* symbols.

    Each symbol column is treated as a separate asset; cash is shared across all.
    """

    import numpy as np

    # Ensure chronological order & unique index
    close_df = close_df.sort_index()

    # ------------------------------------------------------------------
    # Build MultiIndex columns (symbol, field)
    # ------------------------------------------------------------------

    symbols = list(close_df.columns)
    close_df.columns = pd.MultiIndex.from_product([symbols, ["close"]])
    close_df.columns.set_names(["asset", "field"], inplace=True)

    price_index = close_df.index

    # Empty matrices
    size_df = pd.DataFrame(0.0, index=price_index, columns=close_df.columns, dtype=float)
    fixed_fee_df = pd.DataFrame(0.0, index=price_index, columns=close_df.columns, dtype=float)

    if not trades.empty:
        for _, row in trades.iterrows():
            sym = row.get("symbol")
            if sym not in symbols:
                continue

            ts_raw = pd.to_datetime(row["timestamp"], utc=True)
            # Map to nearest previous bar
            if ts_raw in price_index:
                ts_bar = ts_raw
            else:
                pos = price_index.searchsorted(ts_raw, side="right")
                if pos == 0:
                    continue
                ts_bar = price_index[pos - 1]

            col_key = (sym, "close")

            # -------- sizes --------
            size = float(row.get("contracts", 0.0))
            side_val = row.get("side")
            if isinstance(side_val, str) and side_val.lower() == "sell":
                size *= -1
            size_df.at[ts_bar, col_key] += size

            # -------- fees --------
            fee_val = row.get("fee")
            if fee_val and not np.isnan(fee_val):
                fixed_fee_df.at[ts_bar, col_key] += float(fee_val)

            if row.get("type") == "funding":
                payment = float(row.get("payment", 0.0))
                # payment positive means we paid – negative cash flow
                fixed_fee_df.at[ts_bar, col_key] += -payment

    # ------------------------------------------------------------------
    # Create vectorbt Portfolio
    # ------------------------------------------------------------------

    try:
        freq_inferred = pd.infer_freq(price_index)
    except ValueError:
        freq_inferred = None

    pf = vbt.Portfolio.from_orders(
        close=close_df,
        size=size_df.values,
        price=close_df.values,
        size_type="amount",
        fees=np.zeros_like(size_df.values),
        fixed_fees=fixed_fee_df.values,
        init_cash=init_cash,
        cash_sharing=True,
        freq=freq_inferred,
        group_by=True,  # group by first level 'asset'
    )

    return pf