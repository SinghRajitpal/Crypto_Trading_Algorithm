"""data/historical_data.py

Async utilities for downloading and caching historical futures data using python-binance."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, UTC
from typing import List, Optional, Union, Any, Dict

import pandas as pd
from binance import AsyncClient
from utils.logging_config import get_logger, console_log

logger = get_logger(__name__)


def _to_ms(dt: Union[str, datetime, int, None]) -> Optional[int]:
    if dt is None:
        return None
    if isinstance(dt, int):
        return dt
    if isinstance(dt, str):
        return int(pd.to_datetime(dt, utc=True).timestamp() * 1000)
    if isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)
    raise ValueError(f"Unsupported date format: {type(dt)} → {dt}")


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


class HistoricalDataFetcher:
    """Download & cache OHLCV / funding-rate data for back-testing."""

    def __init__(
        self,
        data_dir: str = os.path.join(os.path.dirname(__file__), "cache"),
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ) -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client: Optional[AsyncClient] = None

    async def _ensure_client(self) -> AsyncClient:
        if self.client:
            return self.client
        self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
        return self.client

    async def close(self) -> None:
        if self.client:
            await self.client.close_connection()
            self.client = None

    async def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: Union[str, datetime, int, None] = None,
        end: Union[str, datetime, int, None] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        symbol_norm = _normalize_symbol(symbol)
        file_path = self._cache_path(symbol, timeframe)

        existing_df: Optional[pd.DataFrame] = None
        if os.path.exists(file_path) and not force:
            try:
                existing_df = pd.read_parquet(file_path)
                if "timestamp" in existing_df.columns:
                    existing_df.set_index("timestamp", inplace=True)
            except Exception:
                # Backward compatibility: try legacy CSV if parquet read fails
                csv_path = file_path.replace(".parquet", ".csv")
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            if existing_df is not None:
                existing_df.index = pd.to_datetime(existing_df.index, utc=True)

        since_ms = _to_ms(start)
        end_ms = _to_ms(end)

        if existing_df is not None and not existing_df.empty:
            earliest_cached = int(existing_df.index[0].timestamp() * 1000)
            latest_cached = int(existing_df.index[-1].timestamp() * 1000)
            if since_ms is None or since_ms < earliest_cached:
                since_ms_needed = since_ms
            else:
                since_ms_needed = latest_cached + 1
            if end_ms is not None and end_ms <= latest_cached:
                logger.info(f"Cache hit – returning {file_path} without download.")
                return existing_df
        else:
            since_ms_needed = since_ms

        all_rows: List[List] = []
        fetch_since = since_ms_needed

        logger.info(
            f"Downloading {symbol_norm} {timeframe} starting "
            f"{datetime.fromtimestamp(fetch_since/1000, UTC) if fetch_since else 'from earliest'} "
            f"→ {'up to ' + str(datetime.fromtimestamp(end_ms/1000, UTC)) if end_ms else 'latest'}"
        )

        try:
            while True:
                klines = await self._futures_klines(symbol_norm, timeframe, fetch_since, end_ms)
                if not klines:
                    break
                batch = []
                for k in klines:
                    batch.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
                all_rows.extend(batch)
                last_ts = batch[-1][0]
                if end_ms is not None and last_ts >= end_ms:
                    break
                fetch_since = last_ts + 1
                await asyncio.sleep(0.2)
        except Exception as exc:
            if existing_df is not None:
                logger.warning(f"Download failed ({exc}); returning cached data at {file_path}")
                return existing_df
            raise

        new_df = self._rows_to_df(all_rows)
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined.index = pd.to_datetime(combined.index, utc=True)
        combined.to_parquet(file_path, index=True)
        logger.info(f"Saved {len(new_df)} rows → {file_path}")
        return combined

    async def fetch_funding_rate(
        self,
        symbol: str,
        start: Union[str, datetime, int, None] = None,
        end: Union[str, datetime, int, None] = None,
        force: bool = False,
    ) -> pd.Series:
        symbol_uc = _normalize_symbol(symbol)
        file_path = os.path.join(self.data_dir, f"{symbol_uc}-funding.csv")

        existing: Optional[pd.Series] = None
        if os.path.exists(file_path) and not force:
            try:
                existing = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")["rate"]
            except ValueError:
                existing = pd.read_csv(file_path, parse_dates=[0], index_col=0)["rate"]
                existing.index.name = "timestamp"
            if existing.index.inferred_type != "datetime64" and existing.index.dtype.kind != "M":
                existing.index = pd.to_datetime(existing.index, utc=True, errors="coerce", format="ISO8601")
            existing = existing[existing.index.notnull()].dropna()

        since_ms = _to_ms(start)
        end_ms = _to_ms(end)
        if existing is not None and not existing.empty:
            earliest_cached = int(existing.index[0].timestamp() * 1000)
            latest_cached = int(existing.index[-1].timestamp() * 1000)
            if since_ms is None or since_ms < earliest_cached:
                since_needed = since_ms
            else:
                since_needed = latest_cached + 1
            if end_ms is not None and end_ms <= latest_cached:
                logger.info(f"Funding cache hit – returning {file_path}")
                return existing
        else:
            since_needed = since_ms

        logger.info(
            f"Downloading funding rates for {symbol_uc} starting "
            f"{datetime.fromtimestamp(since_needed/1000, UTC) if since_needed else 'from earliest'}"
            f" → {'up to ' + str(datetime.fromtimestamp(end_ms/1000, UTC)) if end_ms else 'latest'}"
        )

        all_rows: List[List] = []
        fetch_since = since_needed
        while True:
            batch = await self._funding_rates(symbol_uc, fetch_since, end_ms)
            if not batch:
                break
            for item in batch:
                ts = int(item["fundingTime"])
                rate = float(item["fundingRate"])
                all_rows.append([ts, rate])
            last_ts = all_rows[-1][0]
            if end_ms is not None and last_ts >= end_ms:
                break
            fetch_since = last_ts + 1
            await asyncio.sleep(0.2)

        funding_series = pd.Series(
            {pd.to_datetime(ts, unit="ms", utc=True): rate for ts, rate in all_rows},
            name="rate",
        )
        funding_series.sort_index(inplace=True)
        funding_series.to_csv(file_path, index=True, index_label="timestamp")
        return funding_series

    # Helpers
    def _cache_path(self, symbol: str, timeframe: str) -> str:
        symbol_uc = _normalize_symbol(symbol)
        fname = f"{symbol_uc}-{timeframe}.parquet"
        return os.path.join(self.data_dir, fname)

    def _rows_to_df(self, rows: List[List]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def _futures_klines(self, symbol: str, timeframe: str, start_ms: Optional[int], end_ms: Optional[int]):
        client = await self._ensure_client()
        params: Dict[str, Any] = {"symbol": symbol, "interval": timeframe, "limit": 1500}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        return await client.futures_klines(**params)

    async def _funding_rates(self, symbol: str, start_ms: Optional[int], end_ms: Optional[int]):
        client = await self._ensure_client()
        params: Dict[str, Any] = {"symbol": symbol, "limit": 1000}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        return await client.futures_funding_rate(**params)


# CLI driver remains similar, omitted for brevity
