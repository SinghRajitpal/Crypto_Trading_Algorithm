"""data/historical_data.py

Utilities for downloading and caching historical market data that will be
consumed by the back-testing engine.  The goals are:

1. Provide an asynchronous API for fetching large OHLCV datasets from Binance
   USD-M futures (or any ccxt supported exchange).
2. Automatically split the download into batches respecting the exchange
   limit (1500 rows / call for Binance) and rate-limit.
3. Persist the result to `csv` so subsequent back-tests start instantly.
4. Offer helper functions to load the cached data directly into pandas
   DataFrames or NumPy arrays.
5. Include a placeholder for funding-rate download so we can integrate it with
   the simulator later.

This module does **not** try to be a full data-engineering pipeline – it is a
thin convenience layer that keeps the rest of our codebase exchange-agnostic.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, UTC
from typing import List, Optional, Union

import pandas as pd

# ccxt has both sync and async versions; we stick to async for consistency
import ccxt.async_support as ccxt

# Helper functions

def _to_ms(dt: Union[str, datetime, int, None]) -> Optional[int]:
    """Convert various datetime representations to milliseconds since epoch."""
    if dt is None:
        return None

    if isinstance(dt, int):  # already ms
        return dt

    if isinstance(dt, str):
        # Accept common formats, delegate to ccxt
        return ccxt.Exchange.parse8601(dt)

    if isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)

    raise ValueError(f"Unsupported date format: {type(dt)} → {dt}")


def _normalize_symbol(symbol: str) -> str:
    """Ensure symbol is in CCXT slash format, e.g. BTC/USDT."""
    if "/" in symbol:
        return symbol

    # naive handling – if it ends with USDT, BTC, etc.
    quote_currencies = ["USDT", "BUSD", "USDC", "BTC", "ETH"]
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}"

    # default
    return f"{symbol}/USDT"

# Main class


class HistoricalDataFetcher:
    """Download & cache OHLCV / funding-rate data for back-testing.

    Parameters
    ----------
    data_dir : str
        Folder where csv files will be cached.  Will be created if missing.
    exchange_id : str
        ccxt exchange identifier (defaults to 'binanceusdm').  Must support
        `fetch_ohlcv`.
    enable_rate_limit : bool
        Pass-through for ccxt; keep True unless you manage your own pacing.
    """

    def __init__(
        self,
        data_dir: str = os.path.join(os.path.dirname(__file__), "cache"),
        exchange_id: str = "binanceusdm",
        enable_rate_limit: bool = True,
    ) -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls({"enableRateLimit": enable_rate_limit})

        # Binance USD-M supports 1500 rows max per call for 1m data
        self.max_rows_per_call = self.exchange.rateLimit // 1000 or 1500  # fallback

    # Public API

    async def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: Union[str, datetime, int, None] = None,
        end: Union[str, datetime, int, None] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Download OHLCV and cache to CSV; return as DataFrame.

        If the file already exists it will be **appended** (or skipped) unless
        `force=True` which triggers a full re-download.
        """

        symbol_norm = _normalize_symbol(symbol)
        file_path = self._cache_path(symbol, timeframe)

        # Determine existing cached range (if any)
        existing_df: Optional[pd.DataFrame] = None
        if os.path.exists(file_path) and not force:
            existing_df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
            existing_df.index = pd.to_datetime(existing_df.index, utc=True)

        since_ms = _to_ms(start)
        end_ms = _to_ms(end)

        # If we have cached data, adjust the missing range accordingly
        if existing_df is not None and not existing_df.empty:
            earliest_cached = int(existing_df.index[0].timestamp() * 1000)
            latest_cached = int(existing_df.index[-1].timestamp() * 1000)

            if since_ms is None or since_ms < earliest_cached:
                since_ms_needed = since_ms
            else:
                since_ms_needed = latest_cached + 1  # start just after cache

            # If end is before latest cached we already have full data
            if end_ms is not None and end_ms <= latest_cached:
                print(
                    f"[HistoricalData] Cache hit – returning {file_path} without download."
                )
                return existing_df
        else:
            since_ms_needed = since_ms

        # Download loop
        all_rows: List[List] = []
        fetch_since = since_ms_needed

        print(
            f"[HistoricalData] Downloading {symbol_norm} {timeframe} starting "
            f"{datetime.fromtimestamp(fetch_since/1000, UTC) if fetch_since else 'from earliest'} "
            f"→ {'up to ' + str(datetime.fromtimestamp(end_ms/1000, UTC)) if end_ms else 'latest'}"
        )

        while True:
            try:
                batch = await self.exchange.fetch_ohlcv(
                    symbol_norm,
                    timeframe=timeframe,
                    since=fetch_since,
                    limit=1500,  # hard-code for Binance; ccxt will clamp otherwise
                )
            except Exception as e:
                # Close connection to avoid socket leaks then re-raise
                await self.exchange.close()
                raise e

            if not batch:
                break  # no more data

            all_rows.extend(batch)

            # ccxt returns [timestamp, open, high, low, close, volume]
            last_ts = batch[-1][0]

            # stop if we've reached the desired end date
            if end_ms is not None and last_ts >= end_ms:
                break

            # increment since for next loop – add 1 ms to avoid overlap
            fetch_since = last_ts + 1

            # be nice to Binance
            await asyncio.sleep(self.exchange.rateLimit / 1000)

        # Combine with cache and deduplicate
        new_df = self._rows_to_df(all_rows)

        if existing_df is not None:
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        # Persist
        combined.to_csv(file_path, index=True, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
        print(f"[HistoricalData] Saved {len(new_df)} rows → {file_path}")

        return combined

    async def fetch_funding_rate(
        self,
        symbol: str,
        start: Union[str, datetime, int, None] = None,
        end: Union[str, datetime, int, None] = None,
        force: bool = False,
    ) -> pd.Series:
        """Placeholder for funding-rate download (Binance endpoint)."""
        # TODO: implement using self.exchange.fapiPublic_get_fundingrate if needed
        raise NotImplementedError("Funding-rate fetch not implemented yet.")

    async def close(self):
        await self.exchange.close()

    # Internal helpers

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        file_safe = symbol.replace("/", "")
        file_name = f"{file_safe}-{timeframe}.csv"
        return os.path.join(self.data_dir, file_name)

    @staticmethod
    def _rows_to_df(rows: List[List]) -> pd.DataFrame:
        """Convert raw OHLCV rows to pandas DataFrame indexed by timestamp."""
        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], dtype=float
            )

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

# CLI usage example (run once to populate the cache)

if __name__ == "__main__":

    async def _main():
        fetcher = HistoricalDataFetcher()
        try:
            # Example: download last 7 days of BTCUSDT 1-minute candles
            end_dt = datetime.now(UTC)
            start_dt = end_dt - timedelta(weeks=4)
            await fetcher.download_ohlcv("BTCUSDT", "5m", start=start_dt, end=end_dt)
        finally:
            await fetcher.close()

    asyncio.run(_main())
