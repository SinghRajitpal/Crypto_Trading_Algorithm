from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import gc
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

import numpy as np
import pandas as pd

import config
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster
from data.standardizer import Standardizer
from data.data_engine import DataEngine
from data.return_manager import ReturnManager
from data.universe_selector import UniverseSelector
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _mem_usage_mb() -> Optional[float]:
    """Return current RSS in MB if available."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource

            # ru_maxrss is KB on Linux, bytes on macOS; normalize best-effort.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > 10_000_000:  # likely bytes
                return rss / (1024 * 1024)
            return rss / 1024
        except Exception:
            return None


def _log_mem(stage: str, **extra: Any) -> None:
    """Lightweight memory logger gated by config flag."""
    if not getattr(config, "LAYERA_DEBUG_MEM", False):
        return
    rss = _mem_usage_mb()
    if rss is None:
        return
    payload = {"rss_mb": round(rss, 1)}
    if extra:
        payload.update(extra)
    logger.info("[LayerA][mem] %s | %s", stage, payload)


def _bars_per_day(timeframe: str) -> float:
    """Approximate bars per day for a timeframe string like '1m', '1h', '1d', '1w'."""
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith("h"):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith("d"):
        minutes = int(timeframe[:-1]) * 24 * 60
    elif timeframe.endswith("w"):
        minutes = int(timeframe[:-1]) * 7 * 24 * 60
    else:
        minutes = 60
    return max(1.0, (24 * 60) / minutes)


def _infer_last_ts_from_parquet(path: Path) -> Optional[pd.Timestamp]:
    """Use parquet metadata to get the max timestamp without loading the file."""
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        max_ts = None
        for rg_idx in range(pf.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                if col.path_in_schema != "timestamp":
                    continue
                stats = col.statistics
                if stats is None or stats.max is None:
                    continue
                ts_val = stats.max
                ts_val = pd.to_datetime(ts_val, utc=True, errors="coerce")
                if max_ts is None or ts_val > max_ts:
                    max_ts = ts_val
        return max_ts
    except Exception:
        return None


def _infer_min_max_ts_from_parquet(path: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Use parquet metadata to get min/max timestamp without full load."""
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        min_ts = None
        max_ts = None
        for rg_idx in range(pf.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                if col.path_in_schema not in ("timestamp", "__index_level_0__"):
                    continue
                stats = col.statistics
                if stats is None:
                    continue
                if stats.min is not None:
                    ts_min = pd.to_datetime(stats.min, utc=True, errors="coerce")
                    if min_ts is None or ts_min < min_ts:
                        min_ts = ts_min
                if stats.max is not None:
                    ts_max = pd.to_datetime(stats.max, utc=True, errors="coerce")
                    if max_ts is None or ts_max > max_ts:
                        max_ts = ts_max
        return min_ts, max_ts
    except Exception:
        return None, None


def _load_parquet_slice(
    path: Path,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    columns: List[str],
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Load a filtered slice from parquet with pushdown; fallback to pandas read.

    Note: returns a materialized DataFrame (used by tests); for streaming ingestion,
    prefer `_load_parquet_window` which enforces caps without full materialization.
    """
    if not path.exists():
        return None

    start_ts = pd.to_datetime(start_ts, utc=True) if start_ts is not None else None
    end_ts = pd.to_datetime(end_ts, utc=True) if end_ts is not None else None

    df: Optional[pd.DataFrame] = None
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path))
        available_cols = set(dataset.schema.names)
        ts_col = "timestamp" if "timestamp" in available_cols else "__index_level_0__" if "__index_level_0__" in available_cols else None
        if ts_col is None:
            raise ValueError("timestamp column not present in parquet")

        ts_field = ds.field(ts_col)
        filters = None
        if start_ts is not None and end_ts is not None:
            filters = (ts_field >= pa.scalar(start_ts)) & (ts_field <= pa.scalar(end_ts))
        elif start_ts is not None:
            filters = ts_field >= pa.scalar(start_ts)
        elif end_ts is not None:
            filters = ts_field <= pa.scalar(end_ts)

        read_cols = list(columns)
        if ts_col != "timestamp":
            read_cols = [c if c != "timestamp" else ts_col for c in read_cols]
        # Ensure we include ts_col even if not requested explicitly
        if ts_col not in read_cols:
            read_cols.append(ts_col)

        table = dataset.to_table(columns=read_cols, filter=filters)
        df = table.to_pandas()
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = None

    if df is None or df.empty:
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        return None

    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    if max_rows:
        df = df.tail(max_rows)

    return df.sort_index()


def _load_csv_slice(
    path: Path,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    columns: List[str],
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Stream a CSV file in chunks and keep only the requested window."""
    if not path.exists():
        return None

    start_ts = pd.to_datetime(start_ts, utc=True) if start_ts is not None else None
    end_ts = pd.to_datetime(end_ts, utc=True) if end_ts is not None else None

    frames: List[pd.DataFrame] = []
    usecols = columns if "timestamp" in columns else columns + ["timestamp"]
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            if "timestamp" not in chunk.columns:
                continue
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk = chunk.set_index("timestamp")
            if start_ts is not None:
                chunk = chunk[chunk.index >= start_ts]
            if end_ts is not None:
                chunk = chunk[chunk.index <= end_ts]
            if not chunk.empty:
                frames.append(chunk)
    except Exception:
        return None

    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    if max_rows:
        df = df.tail(max_rows)
    keep_cols = [c for c in columns if c in df.columns]
    if keep_cols:
        df = df[keep_cols]
    return df


def _extend_rows_from_df(
    df: pd.DataFrame,
    rows: "deque[dict]",
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> int:
    """Append bars from a filtered/sorted DataFrame into a bounded deque."""
    df = df.copy()
    if "timestamp" not in df.columns:
        if "__index_level_0__" in df.columns:
            df["timestamp"] = df["__index_level_0__"]
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})
        else:
            return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    if df.empty:
        return 0

    ts_ms = (df["timestamp"].astype("int64") // 1_000_000).astype(np.int64)
    vals = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    for ts_val, row in zip(ts_ms, vals):
        rows.append(
            {
                "timestamp": int(ts_val),
                "open": float(row[0]),
                "high": float(row[1]),
                "low": float(row[2]),
                "close": float(row[3]),
                "volume": float(row[4]),
            }
        )
    return len(df)


def _load_parquet_window(
    path: Path,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    columns: List[str],
    max_rows: Optional[int],
    batch_size: int = 25_000,
) -> Tuple[List[dict], Dict[str, int]]:
    """Stream parquet with pushdown filters and return at most max_rows bars (tail)."""
    if not path.exists():
        return [], {"rows_read": 0, "rows_kept": 0}

    start_ts = pd.to_datetime(start_ts, utc=True) if start_ts is not None else None
    end_ts = pd.to_datetime(end_ts, utc=True) if end_ts is not None else None

    rows: "deque[dict]" = deque(maxlen=max_rows if max_rows else None)
    rows_read = 0
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path))
        available_cols = set(dataset.schema.names)
        ts_col = "timestamp" if "timestamp" in available_cols else "__index_level_0__" if "__index_level_0__" in available_cols else None
        if ts_col is None:
            raise ValueError("timestamp column not present in parquet")

        ts_field = ds.field(ts_col)
        filters = None
        if start_ts is not None and end_ts is not None:
            filters = (ts_field >= pa.scalar(start_ts)) & (ts_field <= pa.scalar(end_ts))
        elif start_ts is not None:
            filters = ts_field >= pa.scalar(start_ts)
        elif end_ts is not None:
            filters = ts_field <= pa.scalar(end_ts)

        read_cols = list(columns)
        if ts_col != "timestamp":
            read_cols = [c if c != "timestamp" else ts_col for c in read_cols]
        if ts_col not in read_cols:
            read_cols.append(ts_col)

        scanner = dataset.scanner(columns=read_cols, filter=filters, batch_size=batch_size, use_threads=True)
        for batch in scanner.to_batches():
            try:
                table = pa.Table.from_batches([batch])
                pdf = table.to_pandas()
            except Exception:
                continue
            rows_read += len(pdf)
            _extend_rows_from_df(pdf, rows, start_ts, end_ts)
    except Exception:
        try:
            df = pd.read_parquet(path)
            rows_read = len(df)
            _extend_rows_from_df(df, rows, start_ts, end_ts)
        except Exception:
            return [], {"rows_read": 0, "rows_kept": 0}

    # If Arrow path returned zero rows (unexpected), fall back to pandas filtering
    if rows_read == 0 or len(rows) == 0:
        try:
            df = pd.read_parquet(path, columns=columns or None)
            rows_read = len(df)
            _extend_rows_from_df(df, rows, start_ts, end_ts)
        except Exception:
            pass

    return list(rows), {"rows_read": rows_read, "rows_kept": len(rows)}


def _load_csv_window(
    path: Path,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    columns: List[str],
    max_rows: Optional[int],
    chunksize: int = 200_000,
) -> Tuple[List[dict], Dict[str, int]]:
    """Stream CSV in chunks and return at most max_rows bars (tail)."""
    if not path.exists():
        return [], {"rows_read": 0, "rows_kept": 0}

    start_ts = pd.to_datetime(start_ts, utc=True) if start_ts is not None else None
    end_ts = pd.to_datetime(end_ts, utc=True) if end_ts is not None else None
    rows: "deque[dict]" = deque(maxlen=max_rows if max_rows else None)
    rows_read = 0

    usecols = columns if "timestamp" in columns else columns + ["timestamp"]
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            rows_read += len(chunk)
            if "timestamp" not in chunk.columns:
                continue
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk = chunk.sort_values("timestamp")
            if start_ts is not None:
                chunk = chunk[chunk["timestamp"] >= start_ts]
            if end_ts is not None:
                chunk = chunk[chunk["timestamp"] <= end_ts]
            if chunk.empty:
                continue
            _extend_rows_from_df(chunk, rows, start_ts, end_ts)
            if end_ts is not None and chunk["timestamp"].iloc[-1] > end_ts:
                break
    except Exception:
        return [], {"rows_read": rows_read, "rows_kept": len(rows)}

    return list(rows), {"rows_read": rows_read, "rows_kept": len(rows)}


def _load_cached_bars_window(
    parquet_path: Path,
    csv_path: Path,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    columns: List[str],
    max_rows: Optional[int],
) -> Tuple[List[dict], Dict[str, Any]]:
    """Load cached OHLCV in a bounded tail window; prefers parquet, falls back to CSV."""
    if parquet_path.exists():
        bars, stats = _load_parquet_window(parquet_path, start_ts, end_ts, columns, max_rows=max_rows)
        if bars:
            stats["path"] = str(parquet_path)
            stats["source"] = "parquet"
            return bars, stats
    if csv_path.exists():
        bars, stats = _load_csv_window(csv_path, start_ts, end_ts, columns, max_rows=max_rows)
        if bars:
            stats["path"] = str(csv_path)
            stats["source"] = "csv"
            return bars, stats
    return [], {"rows_read": 0, "rows_kept": 0, "path": None, "source": None}


@dataclass
class RidgeLayerResult:
    k_per_asset: Dict[str, float] = field(default_factory=dict)
    msep_per_asset: Dict[str, float] = field(default_factory=dict)
    rl_vs_ls: Dict[str, Optional[float]] = field(default_factory=dict)
    t_threshold: float = config.RIDGE_T_THRESHOLD
    samples_per_asset: Dict[str, int] = field(default_factory=dict)
    timeframe: str = config.PRIMARY_TIMEFRAME
    w_per_asset: Dict[str, int] = field(default_factory=dict)
    lookback_days: Optional[int] = None

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "timeframe": self.timeframe,
                    "k_per_asset": self.k_per_asset,
                    "msep_per_asset": self.msep_per_asset,
                    "rl_vs_ls": self.rl_vs_ls,
                    "t_threshold": self.t_threshold,
                    "samples_per_asset": self.samples_per_asset,
                    "w_per_asset": self.w_per_asset,
                    "lookback_days": self.lookback_days,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: str) -> "RidgeLayerResult":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            timeframe=data.get("timeframe", config.PRIMARY_TIMEFRAME),
            k_per_asset={k: float(v) for k, v in data.get("k_per_asset", {}).items()},
            msep_per_asset={k: float(v) for k, v in data.get("msep_per_asset", {}).items()},
            rl_vs_ls=data.get("rl_vs_ls", {}),
            t_threshold=float(data.get("t_threshold", config.RIDGE_T_THRESHOLD)),
            samples_per_asset={k: int(v) for k, v in data.get("samples_per_asset", {}).items()},
            w_per_asset={k: int(v) for k, v in data.get("w_per_asset", {}).items()},
            lookback_days=(int(data["lookback_days"]) if data.get("lookback_days") is not None else None),
        )


class RidgeLayerSelector:
    """Layer A selector: rolling-origin CV to choose k per asset with min training enforcement."""

    def __init__(
        self,
        k_grid: Optional[List[float]] = None,
        train_min: int = config.REGRESSION_MIN_TRAIN,
        val_len: int = config.REGRESSION_VAL_WINDOW,
        t_threshold: float = config.RIDGE_T_THRESHOLD,
        embargo: int = config.REGRESSION_EMBARGO_BARS,
        lookback_days_grid: Optional[Dict[str, List[Optional[int]]]] = None,
        tf_candidates: Optional[List[str]] = None,
        max_splits: int = 50,
        fast_mode: bool = False,
    ) -> None:
        self.fast_mode = fast_mode
        self.k_grid = k_grid or ([1e-4, 1e-3, 1e-2, 1e-1] if fast_mode else config.RIDGE_K_GRID)
        self.train_min = train_min
        self.val_len = val_len
        self.t_threshold = t_threshold
        self.embargo = embargo
        self.lookback_days_grid = lookback_days_grid or getattr(config, "LAYERA_LOOKBACK_DAYS_GRID", {}) or {}
        self.tf_candidates = tf_candidates or config.TF_CANDIDATES
        self.max_splits = min(max_splits, 10) if fast_mode else max_splits

    def select(
        self,
        data_engine,
        universe: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> RidgeLayerResult:
        symbols = universe or data_engine.get_active_universe()
        default_ws = [self.train_min, max(self.train_min * 2, self.val_len)]
        result: Optional[RidgeLayerResult] = None

        # Prefer cached history selection when a full DataEngine is provided
        if isinstance(data_engine, DataEngine):
            result = self._select_with_cached_history(data_engine, symbols, default_ws, start)

        # Fallback to direct feature-provider selection (used in tests or pre-seeded engines)
        if result is None:
            result = self._select_from_feature_provider(data_engine, symbols, default_ws)

        if result is None:
            raise ValueError("Layer A selection produced no assets across all timeframes.")
        return result

    def _select_with_cached_history(
        self, data_engine: DataEngine, symbols: List[str], default_ws: List[int], start: Optional[datetime]
    ) -> Optional[RidgeLayerResult]:
        base_client = getattr(data_engine, "binance_client", None)
        if base_client is None:
            return None

        total_tf = len(self.tf_candidates)
        total_sym = len(symbols)
        logger.info("[LayerA] Starting selection over %d timeframes x %d assets", total_tf, total_sym)
        _log_mem("selection-start", tf_count=total_tf, assets=total_sym)
        tf_scores: Dict[Tuple[str, Optional[int]], float] = {}
        tf_best: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}

        # Dynamically adjust lookbacks based on cached coverage relative to start
        coverage = self._scan_cache_coverage(symbols)
        adjusted_lb, tf_earliest = self._adjust_lookbacks_by_coverage(coverage, start)

        # Report earliest feasible start across TFs for transparency
        feasible_starts = []
        for tf, ts in tf_earliest.items():
            lbs = adjusted_lb.get(tf) or []
            finite_lbs = [lb for lb in lbs if lb is not None]
            if ts is not None and finite_lbs:
                min_lb_days = min(finite_lbs)
                feasible_starts.append(ts + pd.Timedelta(days=min_lb_days))
        if feasible_starts:
            overall_earliest = max(feasible_starts)
            logger.info("[LayerA] Earliest feasible backtest start across TFs given cache: %s", overall_earliest)
        else:
            logger.warning("[LayerA] No feasible lookbacks found in cache coverage; Layer A may produce empty spec.")

        for tf_idx, tf in enumerate(self.tf_candidates, 1):
            w_grid = config.W_CANDIDATES_BY_TF.get(tf, default_ws)
            if self.fast_mode and len(w_grid) > 1:
                w_grid = w_grid[:1]
            lookback_grid = adjusted_lb.get(tf) or [None]
            if not lookback_grid:
                logger.warning("[LayerA] Skipping TF=%s because no usable lookbacks after coverage adjustment", tf)
                continue
            if self.fast_mode and len(lookback_grid) > 1:
                lookback_grid = lookback_grid[:1]
            for lb_idx, lb_days in enumerate(lookback_grid, 1):
                logger.info(
                    "[LayerA] Timeframe %s (%d/%d) lookback=%s (%d/%d) starting",
                    tf,
                    tf_idx,
                    total_tf,
                    lb_days,
                    lb_idx,
                    len(lookback_grid),
                )
                _log_mem("tf-start", tf=tf, lookback=lb_days)
                tf_engine = self._build_engine_for_tf(
                    tf, symbols, base_client, w_grid, lookback_days=lb_days, end_ts=start
                )
                if tf_engine is None:
                    logger.warning("[LayerA] Skipping TF=%s lookback=%s due to missing cached data", tf, lb_days)
                    continue

                k_per_asset: Dict[str, float] = {}
                msep_per_asset: Dict[str, float] = {}
                rl_vs_ls: Dict[str, Optional[float]] = {}
                samples_per_asset: Dict[str, int] = {}
                w_per_asset: Dict[str, int] = {}

                # Parallelize per asset using threads (no pickle overhead) but keep worker count low to cap memory.
                max_workers = min(2, len(symbols)) or 1
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(self._select_for_asset, tf, sym, tf_engine, w_grid): sym
                        for sym in symbols
                    }
                    done_count = 0
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res is None:
                            continue
                        sym_out, best_k, best_w, best_msep, rl, samples = res
                        k_per_asset[sym_out] = best_k
                        msep_per_asset[sym_out] = best_msep
                        rl_vs_ls[sym_out] = rl
                        samples_per_asset[sym_out] = samples
                        w_per_asset[sym_out] = best_w
                        done_count += 1
                        logger.info(
                            "[LayerA] TF=%s lookback=%s progress %d/%d assets complete",
                            tf,
                            lb_days,
                            done_count,
                            len(symbols),
                        )

                if k_per_asset:
                    tf_key = (tf, lb_days)
                    tf_scores[tf_key] = float(np.mean(list(msep_per_asset.values())))
                    tf_best[tf_key] = {
                        "k_per_asset": k_per_asset,
                        "msep_per_asset": msep_per_asset,
                        "rl_vs_ls": rl_vs_ls,
                        "samples_per_asset": samples_per_asset,
                        "w_per_asset": w_per_asset,
                        "lookback_days": lb_days,
                    }
                    logger.info(
                        "[LayerA] Timeframe %s (%d/%d) lookback=%s complete | assets=%d | avg_msep=%.4e",
                        tf,
                        tf_idx,
                        total_tf,
                        lb_days,
                        len(k_per_asset),
                        tf_scores[tf_key],
                    )

                # Release per-timeframe engines promptly to avoid RSS accumulation across TFs.
                if tf_engine is not None:
                    tf_engine.dispose()
                    _log_mem("tf-dispose", tf=tf, lookback=lb_days)
                    del tf_engine
                gc.collect()

        if not tf_scores:
            return None

        tf_star, lb_star = min(tf_scores, key=tf_scores.get)
        best = tf_best[(tf_star, lb_star)]
        return RidgeLayerResult(
            timeframe=tf_star,
            k_per_asset=best["k_per_asset"],
            msep_per_asset=best["msep_per_asset"],
            rl_vs_ls=best["rl_vs_ls"],
            t_threshold=self.t_threshold,
            samples_per_asset=best["samples_per_asset"],
            w_per_asset=best["w_per_asset"],
            lookback_days=best.get("lookback_days"),
        )

    def _scan_cache_coverage(self, symbols: List[str]) -> Dict[str, Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]]:
        """Inspect parquet/CSV cache to find min/max ts per symbol per timeframe."""
        data_dir = Path("data/cache")
        coverage: Dict[str, Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]] = {}
        for sym in symbols:
            coverage[sym] = {}
            for tf in self.tf_candidates:
                parquet_path = data_dir / f"{sym}-{tf}.parquet"
                csv_path = data_dir / f"{sym}-{tf}.csv"
                mn = mx = None
                if parquet_path.exists():
                    mn, mx = _infer_min_max_ts_from_parquet(parquet_path)
                if (mn is None or mx is None) and csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
                        if not df.empty:
                            mn = df["timestamp"].min()
                            mx = df["timestamp"].max()
                    except Exception:
                        pass
                coverage[sym][tf] = (mn, mx)
        return coverage

    def _adjust_lookbacks_by_coverage(
        self,
        coverage: Dict[str, Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]],
        start: Optional[datetime],
    ) -> Tuple[Dict[str, List[Optional[int]]], Dict[str, Optional[pd.Timestamp]]]:
        """Clip lookback grids per TF based on earliest available cached ts for requested start."""
        adjusted: Dict[str, List[Optional[int]]] = {}
        tf_earliest: Dict[str, Optional[pd.Timestamp]] = {}
        if start is None:
            for tf in self.tf_candidates:
                adjusted[tf] = self.lookback_days_grid.get(tf) or [None]
                tf_earliest[tf] = None
            return adjusted, tf_earliest

        start_ts = pd.to_datetime(start, utc=True)
        for tf in self.tf_candidates:
            lb_grid = self.lookback_days_grid.get(tf) or [None]
            earliest_ts = None
            for sym_cov in coverage.values():
                mn, _ = sym_cov.get(tf, (None, None))
                if mn is not None:
                    if earliest_ts is None or mn < earliest_ts:
                        earliest_ts = mn
            tf_earliest[tf] = earliest_ts
            if earliest_ts is None:
                logger.warning("[LayerA] No cached coverage for TF=%s across symbols; using configured lookbacks", tf)
                adjusted[tf] = lb_grid
                continue

            max_usable_days = int((start_ts - earliest_ts).days)
            usable: List[Optional[int]] = []
            for lb in lb_grid:
                if lb is None:
                    usable.append(None)
                elif lb <= max_usable_days:
                    usable.append(lb)
            if not usable:
                if max_usable_days > 0:
                    clipped = min([lb for lb in lb_grid if lb is not None], default=max_usable_days)
                    usable = [min(clipped, max_usable_days)]
                    logger.warning(
                        "[LayerA] TF=%s lookbacks clipped to %d days due to coverage (%s → start %s)",
                        tf,
                        usable[0],
                        earliest_ts,
                        start_ts,
                    )
                else:
                    logger.warning(
                        "[LayerA] TF=%s has no usable lookback; earliest=%s start=%s", tf, earliest_ts, start_ts
                    )
            adjusted[tf] = usable

            logger.info(
                "[LayerA] TF=%s coverage earliest=%s start=%s max_usable_days=%d effective_lookbacks=%s",
                tf,
                earliest_ts,
                start_ts,
                max_usable_days,
                usable,
            )

        return adjusted, tf_earliest

    def _select_from_feature_provider(
        self, feature_provider, symbols: List[str], default_ws: List[int]
    ) -> Optional[RidgeLayerResult]:
        tf_label = getattr(feature_provider, "primary_timeframe", config.PRIMARY_TIMEFRAME)
        logger.info("[LayerA] Using preloaded features for timeframe %s", tf_label)

        k_per_asset: Dict[str, float] = {}
        msep_per_asset: Dict[str, float] = {}
        rl_vs_ls: Dict[str, Optional[float]] = {}
        samples_per_asset: Dict[str, int] = {}
        w_per_asset: Dict[str, int] = {}

        for sym in symbols:
            res = self._select_for_asset(tf_label, sym, feature_provider, default_ws)
            if res is None:
                continue
            sym_out, best_k, best_w, best_msep, rl, samples = res
            k_per_asset[sym_out] = best_k
            msep_per_asset[sym_out] = best_msep
            rl_vs_ls[sym_out] = rl
            samples_per_asset[sym_out] = samples
            w_per_asset[sym_out] = best_w

        if not k_per_asset:
            return RidgeLayerResult(timeframe=tf_label, lookback_days=None)

        return RidgeLayerResult(
            timeframe=tf_label,
            k_per_asset=k_per_asset,
            msep_per_asset=msep_per_asset,
            rl_vs_ls=rl_vs_ls,
            t_threshold=self.t_threshold,
            samples_per_asset=samples_per_asset,
            w_per_asset=w_per_asset,
            lookback_days=None,
        )

    def _build_engine_for_tf(
        self,
        timeframe: str,
        symbols: List[str],
        base_client,
        w_grid: List[int],
        lookback_days: Optional[int] = None,
        end_ts: Optional[datetime] = None,
    ) -> Optional[DataEngine]:
        """Build a fresh DataEngine loaded with cached history for the given timeframe.

        Uses cached parquet/CSV history and optionally trims by lookback_days to search over horizons.
        """
        data_dir = Path("data/cache")
        window_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        buffer_estimate = self._estimate_buffer(timeframe, lookback_days, w_grid)
        engine = DataEngine(binance_client=base_client, max_candles=buffer_estimate)
        engine.primary_timeframe = timeframe
        engine.data_fetcher.symbol_timeframes = [(s, timeframe) for s in symbols]
        _log_mem("pre-load", tf=timeframe, lookback=lookback_days, buffer=buffer_estimate)

        ingested = 0
        for sym in symbols:
            parquet_path = data_dir / f"{sym}-{timeframe}.parquet"
            csv_path = data_dir / f"{sym}-{timeframe}.csv"

            effective_end = end_ts
            if effective_end is None and lookback_days is not None and parquet_path.exists():
                effective_end = _infer_last_ts_from_parquet(parquet_path)

            start_cutoff = None
            if lookback_days is not None and effective_end is not None:
                start_cutoff = effective_end - pd.Timedelta(days=lookback_days)

            tf_cap = getattr(config, "LAYERA_MAX_BARS_BY_TF", {}).get(timeframe)
            max_rows = tf_cap if tf_cap is not None else buffer_estimate

            bars, stats = _load_cached_bars_window(
                parquet_path,
                csv_path,
                start_cutoff,
                effective_end,
                window_columns,
                max_rows=max_rows,
            )
            if not bars:
                missing_reason = "no file" if not parquet_path.exists() and not csv_path.exists() else "no rows in window"
                logger.warning(
                    "[LayerA] Missing or empty cached data for %s-%s (%s) window=[%s→%s]",
                    sym,
                    timeframe,
                    missing_reason,
                    start_cutoff,
                    effective_end,
                )
                continue

            for bar in bars:
                engine.return_manager.update(sym, bar)

            ingested += 1
            logger.info(
                "[LayerA] TF=%s lookback=%s asset=%s window=[%s→%s] rows_read=%d rows_kept=%d from=%s",
                timeframe,
                lookback_days,
                sym,
                start_cutoff,
                effective_end,
                stats.get("rows_read", 0),
                stats.get("rows_kept", len(bars)),
                stats.get("path"),
            )
            _log_mem(
                "asset-load",
                tf=timeframe,
                lookback=lookback_days,
                asset=sym,
                rows_read=stats.get("rows_read", 0),
                rows_kept=stats.get("rows_kept", len(bars)),
                path=stats.get("path"),
            )
            del bars

        _log_mem("post-load", tf=timeframe, lookback=lookback_days, assets_ingested=ingested)

        if ingested == 0:
            return None

        return engine

    def _estimate_buffer(self, timeframe: str, lookback_days: Optional[int], w_grid: List[int]) -> int:
        """Estimate max_candles needed for ReturnManager without over-allocating."""
        bars_day = _bars_per_day(timeframe)
        default_lb_days = None
        tf_lb_map = getattr(config, "TIMEFRAME_LOOKBACK_DAYS", {})
        if timeframe in tf_lb_map:
            default_lb_days = tf_lb_map.get(timeframe)
        if default_lb_days is None:
            default_lb_days = getattr(config, "RIDGE_TRAIN_LOOKBACK_DAYS", None)

        effective_lb_days = lookback_days if lookback_days is not None else default_lb_days
        lookback_bars = int(ceil((effective_lb_days or 0) * bars_day)) if effective_lb_days else None

        cv_need = max(max(w_grid or [self.train_min]), self.train_min + self.val_len + self.embargo)
        estimate = max(cv_need, lookback_bars or cv_need) + 10

        min_required = self.train_min + self.val_len + self.embargo + 10
        estimate = max(int(estimate), min_required)

        tf_cap = getattr(config, "LAYERA_MAX_BARS_BY_TF", {}).get(timeframe)
        reg_caps = [getattr(config, "LAYERA_REGRESSION_MAX_BARS", None), getattr(config, "REGRESSION_MAX_BARS", None)]
        hard_caps = [c for c in reg_caps + [tf_cap] if c is not None]
        if hard_caps:
            estimate = min(estimate, min(hard_caps))
            estimate = max(estimate, min_required)
        return int(estimate)

    @staticmethod
    def _with_intercept(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _penalty_matrix(size: int, k: float) -> np.ndarray:
        pen = np.eye(size) * k
        if size > 0:
            pen[0, 0] = 0.0  # Do not penalize intercept
        return pen

    @staticmethod
    def _ridge_beta(XtX: np.ndarray, Xty: np.ndarray, penalty: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(XtX + penalty) @ Xty

    def _select_for_asset(self, tf: str, sym: str, tf_engine: DataEngine, w_grid: List[int]):
        """Run CV for a single asset (used in multiprocessing pool)."""
        X_raw, y_raw, ts, cols = tf_engine.get_feature_matrix(sym)
        _log_mem("asset-cv-start", tf=tf, asset=sym, bars=len(y_raw), features=X_raw.shape[1] if X_raw.size else 0)
        if y_raw.size < self.train_min:
            return None
        # Apply global cap and optional Layer-A per-timeframe cap
        caps = [config.REGRESSION_MAX_BARS, getattr(config, "LAYERA_REGRESSION_MAX_BARS", None)]
        tf_cap = getattr(config, "LAYERA_MAX_BARS_BY_TF", {}).get(tf)
        if tf_cap:
            caps.append(tf_cap)
        cap = min([c for c in caps if c is not None], default=None)
        if cap:
            X_raw = X_raw[-cap:]
            y_raw = y_raw[-cap:]
        n = len(y_raw)
        if n < self.train_min:
            return None
        scaler_full = Standardizer().fit(X_raw)
        X_std_full = scaler_full.transform(X_raw)
        X_std_full = self._with_intercept(X_std_full)
        y_full = y_raw

        best_k = None
        best_msep = np.inf
        best_w = None
        val_len = min(self.val_len, max(1, n - self.train_min))
        k_vec = np.array(self.k_grid, dtype=float)
        eye_penalty = np.eye(X_std_full.shape[1])
        eye_penalty[0, 0] = 0.0

        for W in w_grid:
            if n < W + val_len + self.embargo:
                continue
            splits_done = 0
            for start in range(0, n - W - val_len, val_len):
                train_end = start + W
                val_start = train_end + self.embargo
                val_end = val_start + val_len
                if val_end > n:
                    break
                X_train_std = X_std_full[start:train_end]
                y_train = y_full[start:train_end]
                X_val_std = X_std_full[val_start:val_end]
                y_val = y_full[val_start:val_end]

                XtX = X_train_std.T @ X_train_std
                Xty = X_train_std.T @ y_train
                mats = XtX[None, :, :] + k_vec[:, None, None] * eye_penalty
                rhs = np.broadcast_to(Xty, (k_vec.size, Xty.shape[0]))[..., None]
                try:
                    betas = np.linalg.solve(mats, rhs).squeeze(-1)
                except np.linalg.LinAlgError:
                    betas = np.array([np.linalg.pinv(mats[i]) @ Xty for i in range(k_vec.size)])

                preds = betas @ X_val_std.T  # shape (k, val_len)
                errors = y_val[None, :] - preds
                mseps = np.mean(errors * errors, axis=1)
                idx = int(np.argmin(mseps))
                msep = float(mseps[idx])
                if msep < best_msep:
                    best_msep = msep
                    best_k = float(k_vec[idx])
                    best_w = W
                splits_done += 1
                if self.max_splits and splits_done >= self.max_splits:
                    break

        if best_k is None or best_w is None:
            return None

        forecaster = RidgeRegressionForecaster(k_grid=[best_k], t_threshold=self.t_threshold)
        ridge_result = forecaster.forecast(sym, X_raw, y_raw)
        if not ridge_result:
            return None

        logger.info(
            "[LayerA] TF=%s asset=%s k=%.4g W=%d msep=%.4e samples=%d",
            tf,
            sym,
            best_k,
            best_w,
            best_msep,
            ridge_result.samples,
        )
        _log_mem("asset-cv-end", tf=tf, asset=sym, bars=n, best_k=best_k, best_w=best_w)
        return sym, best_k, best_w, best_msep, ridge_result.rl_vs_ls, ridge_result.samples
