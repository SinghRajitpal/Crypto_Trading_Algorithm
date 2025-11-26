from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import gc
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
import csv

import numpy as np
import pandas as pd

import config
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster
from data.standardizer import Standardizer
from data.data_engine import DataEngine
from utils.logging_config import get_logger

logger = get_logger(__name__)
_GLOBAL_ARTIFACTS: Optional["LayerAArtifacts"] = None


class LayerAArtifacts:
    """Helper for persisting Layer A diagnostics and audit artifacts."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_path = self.base_dir / "coverage_report.json"
        self.timeframe_candidates_path = self.base_dir / "timeframe_candidates.jsonl"
        self.warnings_path = self.base_dir / "warnings_and_skips.jsonl"
        self.mem_trace_path = self.base_dir / "memory_trace.jsonl"
        self.cv_grid_dir = self.base_dir / "cv_grid"
        self.model_diag_dir = self.base_dir / "model_diagnostics"
        self.summary_path = self.base_dir / "layerA_summary.json"
        self.cv_grid_dir.mkdir(parents=True, exist_ok=True)
        self.model_diag_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def log_mem(self, stage: str, payload: Dict[str, Any]) -> None:
        record = {"stage": stage, "timestamp": datetime.utcnow().isoformat(), **payload}
        self._append_jsonl(self.mem_trace_path, record)

    def warn(self, payload: Dict[str, Any]) -> None:
        payload = {"timestamp": datetime.utcnow().isoformat(), **payload}
        self._append_jsonl(self.warnings_path, payload)

    def record_timeframe_candidate(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.timeframe_candidates_path, payload)

    def write_coverage_report(self, payload: Dict[str, Any]) -> None:
        self._write_json(self.coverage_path, payload)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        self._write_json(self.summary_path, payload)


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
    if _GLOBAL_ARTIFACTS:
        _GLOBAL_ARTIFACTS.log_mem(stage, payload)
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
    train_samples_used: Dict[str, int] = field(default_factory=dict)
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
                    "train_samples_used": self.train_samples_used,
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
            train_samples_used={k: int(v) for k, v in data.get("train_samples_used", {}).items()},
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
        min_splits: int = 2,
        min_effective_lookback_days: int = 30,
    ) -> None:
        self.fast_mode = fast_mode
        if k_grid is None:
            # Bias toward smaller k; cap maximum to 1e2 for Layer A
            base_grid = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            self.k_grid = [float(v) for v in base_grid]
        else:
            # Preserve caller-provided ordering but ensure floats
            self.k_grid = [float(v) for v in k_grid]
        self.train_min = train_min
        self.val_len = val_len
        self.t_threshold = t_threshold
        self.embargo = embargo
        self.lookback_days_grid = lookback_days_grid or getattr(config, "LAYERA_LOOKBACK_DAYS_GRID", {}) or {}
        self.tf_candidates = tf_candidates or config.TF_CANDIDATES
        self.max_splits = min(max_splits, 10) if fast_mode else max_splits
        self.min_splits = max(1, min_splits)
        self.min_effective_lookback_days = max(1, min_effective_lookback_days)

    def select(
        self,
        data_engine,
        universe: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        artifacts_dir: Optional[str] = None,
    ) -> RidgeLayerResult:
        global _GLOBAL_ARTIFACTS
        artifacts: Optional[LayerAArtifacts] = None
        if artifacts_dir:
            artifacts = LayerAArtifacts(Path(artifacts_dir))
            _GLOBAL_ARTIFACTS = artifacts

        symbols = universe or data_engine.get_active_universe()
        default_ws = [self.train_min, max(self.train_min * 2, self.val_len)]
        result: Optional[RidgeLayerResult] = None
        try:
            # Prefer cached history selection when a full DataEngine is provided
            if isinstance(data_engine, DataEngine):
                result = self._select_with_cached_history(data_engine, symbols, default_ws, start, artifacts, end)

            # Fallback to direct feature-provider selection (used in tests or pre-seeded engines)
            if result is None:
                result = self._select_from_feature_provider(data_engine, symbols, default_ws, artifacts)

            if result is None:
                raise ValueError("Layer A selection produced no assets across all timeframes.")
            return result
        finally:
            _GLOBAL_ARTIFACTS = None

    def _resolve_bar_cap(self, timeframe: str) -> Optional[int]:
        """Determine the effective bar cap for a timeframe using Layer A + global caps."""
        caps = [
            getattr(config, "LAYERA_MAX_BARS_BY_TF", {}).get(timeframe),
            getattr(config, "LAYERA_REGRESSION_MAX_BARS", None),
            getattr(config, "REGRESSION_MAX_BARS", None),
        ]
        caps = [int(c) for c in caps if c is not None]
        if not caps:
            return None
        return min(caps)

    def _select_with_cached_history(
        self,
        data_engine: DataEngine,
        symbols: List[str],
        default_ws: List[int],
        start: Optional[datetime],
        artifacts: Optional[LayerAArtifacts],
        end: Optional[datetime],
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
        tf_asset_windows: Dict[Tuple[str, Optional[int]], Dict[str, Dict[str, Any]]] = {}

        # Dynamically adjust lookbacks based on cached coverage relative to start
        coverage = self._scan_cache_coverage(symbols)
        adjusted_lb, tf_earliest = self._adjust_lookbacks_by_coverage(coverage, start, artifacts)

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
                logger.warning("[LayerA] Skipping TF=%s because effective lookback grid is empty", tf)
                continue
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
                tf_engine, asset_windows = self._build_engine_for_tf(
                    tf, symbols, base_client, w_grid, lookback_days=lb_days, end_ts=start, artifacts=artifacts
                )
                if tf_engine is None:
                    logger.warning("[LayerA] Skipping TF=%s lookback=%s due to missing cached data", tf, lb_days)
                    continue
                tf_asset_windows[(tf, lb_days)] = asset_windows

                k_per_asset: Dict[str, float] = {}
                msep_per_asset: Dict[str, float] = {}
                rl_vs_ls: Dict[str, Optional[float]] = {}
                samples_per_asset: Dict[str, int] = {}
                w_per_asset: Dict[str, int] = {}
                train_samples_used: Dict[str, int] = {}
                ic_per_asset: Dict[str, Optional[float]] = {}

                # Parallelize per asset using threads (no pickle overhead) but keep worker count low to cap memory.
                max_workers = min(2, len(symbols)) or 1
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(self._select_for_asset, tf, sym, tf_engine, w_grid, artifacts, lb_days): sym
                        for sym in symbols
                    }
                    done_count = 0
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res is None:
                            continue
                        sym_out, best_k, best_w, best_msep, rl, samples, best_ic = res
                        k_per_asset[sym_out] = best_k
                        msep_per_asset[sym_out] = best_msep
                        rl_vs_ls[sym_out] = rl
                        samples_per_asset[sym_out] = samples
                        w_per_asset[sym_out] = best_w
                        train_samples_used[sym_out] = min(best_w, samples)
                        ic_per_asset[sym_out] = best_ic
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
                    coverage_ratio = len(k_per_asset) / max(1, len(symbols))
                    avg_msep = float(np.mean(list(msep_per_asset.values())))
                    ic_vals = [v for v in ic_per_asset.values() if v is not None]
                    mean_ic = float(np.mean(ic_vals)) if ic_vals else None
                    # Penalize missing assets so the chosen TF represents the full universe.
                    coverage_penalty = max(coverage_ratio, 0.05)
                    asset_win = tf_asset_windows.get((tf, lb_days), {})
                    starts = [v.get("start_ts") for v in asset_win.values() if v.get("start_ts") is not None]
                    ends = [v.get("end_ts") for v in asset_win.values() if v.get("end_ts") is not None]
                    any_cap = any(v.get("cap_hit") for v in asset_win.values())
                    realized_days = None
                    if starts and ends:
                        realized_days = (max(ends) - min(starts)).days
                    if any_cap:
                        logger.warning("[LayerA] Skipping TF=%s lookback=%s because cap hit on at least one asset", tf, lb_days)
                        if artifacts:
                            artifacts.warn(
                                {
                                    "type": "candidate_skipped_cap",
                                    "timeframe": tf,
                                    "lookback_days": lb_days,
                                }
                            )
                        continue
                    if lb_days is not None and realized_days is not None and realized_days < 0.8 * lb_days:
                        logger.warning(
                            "[LayerA] Skipping TF=%s lookback=%s because realized days %.1f < 80%% of requested",
                            tf,
                            lb_days,
                            realized_days,
                        )
                        if artifacts:
                            artifacts.warn(
                                {
                                    "type": "candidate_skipped_short_window",
                                    "timeframe": tf,
                                    "lookback_days": lb_days,
                                    "realized_days": realized_days,
                                }
                            )
                        continue
                    if mean_ic is not None and np.isfinite(mean_ic):
                        raw_score = -mean_ic / coverage_penalty
                        score = raw_score  # IC is scale-free
                    else:
                        raw_score = avg_msep / coverage_penalty
                        score = raw_score * _bars_per_day(tf)
                    tf_scores[tf_key] = score
                    tf_best[tf_key] = {
                        "k_per_asset": k_per_asset,
                        "msep_per_asset": msep_per_asset,
                        "rl_vs_ls": rl_vs_ls,
                        "samples_per_asset": samples_per_asset,
                        "w_per_asset": w_per_asset,
                        "train_samples_used": train_samples_used,
                        "lookback_days": lb_days,
                        "coverage_ratio": coverage_ratio,
                        "tf_score_raw": raw_score,
                        "ic_per_asset": ic_per_asset,
                        "mean_ic": mean_ic,
                    }
                    logger.info(
                        "[LayerA] Timeframe %s (%d/%d) lookback=%s complete | assets=%d/%d | avg_msep=%.4e | score=%.4e",
                        tf,
                        tf_idx,
                        total_tf,
                        lb_days,
                        len(k_per_asset),
                        len(symbols),
                        avg_msep,
                        score,
                    )
                    if artifacts:
                        asset_win = tf_asset_windows.get((tf, lb_days), {})
                        starts = [v.get("start_ts") for v in asset_win.values() if v.get("start_ts") is not None]
                        ends = [v.get("end_ts") for v in asset_win.values() if v.get("end_ts") is not None]
                        any_cap = any(v.get("cap_hit") for v in asset_win.values())
                        artifacts.record_timeframe_candidate(
                            {
                                "timeframe": tf,
                                "lookback_days": lb_days,
                                "w_grid": w_grid,
                                "k_grid": list(self.k_grid),
                                "avg_msep": avg_msep,
                                "tf_score_normalized": score,
                                "tf_score_raw": raw_score,
                                "mean_ic": mean_ic,
                                "coverage_ratio": coverage_ratio,
                                "asset_count": len(k_per_asset),
                                "universe_size": len(symbols),
                                "bar_cap": self._resolve_bar_cap(tf),
                                "effective_lookback_grid": adjusted_lb.get(tf),
                                "data_window_estimate": {
                                    "start": (start - pd.Timedelta(days=lb_days)).isoformat()
                                    if start is not None and lb_days is not None
                                    else None,
                                    "end": start.isoformat() if start is not None else None,
                                },
                                "data_window_actual": {
                                    "start": min(starts).isoformat() if starts else None,
                                    "end": max(ends).isoformat() if ends else None,
                                    "cap_hit_any": any_cap,
                                    "realized_days": (max(ends) - min(starts)).days if starts and ends else None,
                                },
                            }
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
        if artifacts:
            per_asset = {}
            for sym, k_val in best["k_per_asset"].items():
                per_asset[sym] = {
                    "k": k_val,
                    "w": best["w_per_asset"].get(sym),
                    "msep": best["msep_per_asset"].get(sym),
                    "rl_vs_ls": best["rl_vs_ls"].get(sym),
                    "samples_available": best["samples_per_asset"].get(sym),
                    "train_samples_used": best.get("train_samples_used", {}).get(sym),
                    "ic": best.get("ic_per_asset", {}).get(sym),
                }
            summary = {
                "requested_start": start,
                "requested_end": end,
                "fast_mode": self.fast_mode,
                "train_min": self.train_min,
                "val_len": self.val_len,
                "embargo": self.embargo,
                "k_grid": list(self.k_grid),
                "tf_candidates": list(self.tf_candidates),
                "lookback_days_grid": self.lookback_days_grid,
                "effective_lookback_grid": adjusted_lb,
                "max_splits": self.max_splits,
                "min_splits": self.min_splits,
                "chosen_timeframe": tf_star,
                "chosen_lookback_days": lb_star,
                "tf_score_normalized": tf_scores[(tf_star, lb_star)],
                "tf_score_raw": tf_best[(tf_star, lb_star)].get("tf_score_raw"),
                "mean_ic": tf_best[(tf_star, lb_star)].get("mean_ic"),
                "coverage_ratio": best.get("coverage_ratio"),
                "chosen_data_window": tf_asset_windows.get((tf_star, lb_star), {}),
                "bar_caps": {
                    "layerA_max": getattr(config, "LAYERA_MAX_BARS_BY_TF", {}),
                    "layerA_regression_max_bars": getattr(config, "LAYERA_REGRESSION_MAX_BARS", None),
                    "regression_max_bars": getattr(config, "REGRESSION_MAX_BARS", None),
                },
                "per_asset": per_asset,
            }
            artifacts.write_summary(summary)
        return RidgeLayerResult(
            timeframe=tf_star,
            k_per_asset=best["k_per_asset"],
            msep_per_asset=best["msep_per_asset"],
            rl_vs_ls=best["rl_vs_ls"],
            t_threshold=self.t_threshold,
            samples_per_asset=best["samples_per_asset"],
            w_per_asset=best["w_per_asset"],
            train_samples_used=best.get("train_samples_used", {}),
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
        artifacts: Optional[LayerAArtifacts],
    ) -> Tuple[Dict[str, List[Optional[int]]], Dict[str, Optional[pd.Timestamp]]]:
        """Clip lookback grids per TF based on earliest available cached ts for requested start."""
        adjusted: Dict[str, List[Optional[int]]] = {}
        tf_earliest: Dict[str, Optional[pd.Timestamp]] = {}
        report: Dict[str, Any] = {"requested_start": start, "coverage": coverage, "tf_summary": {}}
        if start is None:
            for tf in self.tf_candidates:
                adjusted[tf] = self.lookback_days_grid.get(tf) or [None]
                tf_earliest[tf] = None
                report["tf_summary"][tf] = {
                    "earliest_ts": None,
                    "max_usable_days": None,
                    "effective_lookbacks": adjusted[tf],
                }
            if artifacts:
                artifacts.write_coverage_report(report)
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
                report["tf_summary"][tf] = {
                    "earliest_ts": None,
                    "max_usable_days": None,
                    "effective_lookbacks": adjusted[tf],
                    "clipped": False,
                }
                continue

            max_usable_days = int((start_ts - earliest_ts).days)
            if max_usable_days < self.min_effective_lookback_days:
                logger.warning(
                    "[LayerA] Skipping TF=%s because max usable lookback %d < minimum %d",
                    tf,
                    max_usable_days,
                    self.min_effective_lookback_days,
                )
                if artifacts:
                    artifacts.warn(
                        {
                            "type": "lookback_too_short",
                            "timeframe": tf,
                            "max_usable_days": max_usable_days,
                            "min_required_days": self.min_effective_lookback_days,
                            "earliest_ts": earliest_ts,
                            "requested_start": start_ts,
                        }
                    )
                adjusted[tf] = []
                report["tf_summary"][tf] = {
                    "earliest_ts": earliest_ts,
                    "max_usable_days": max_usable_days,
                    "effective_lookbacks": [],
                    "clipped": True,
                }
                continue
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
                    if artifacts:
                        artifacts.warn(
                            {
                                "type": "lookback_clipped",
                                "timeframe": tf,
                                "earliest_ts": earliest_ts,
                                "requested_start": start_ts,
                                "max_usable_days": max_usable_days,
                                "original_grid": lb_grid,
                                "effective_grid": usable,
                            }
                        )
                else:
                    logger.warning(
                        "[LayerA] TF=%s has no usable lookback; earliest=%s start=%s", tf, earliest_ts, start_ts
                    )
                    if artifacts:
                        artifacts.warn(
                            {
                                "type": "no_usable_lookback",
                                "timeframe": tf,
                                "earliest_ts": earliest_ts,
                                "requested_start": start_ts,
                                "max_usable_days": max_usable_days,
                                "original_grid": lb_grid,
                            }
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
            report["tf_summary"][tf] = {
                "earliest_ts": earliest_ts,
                "max_usable_days": max_usable_days,
                "effective_lookbacks": usable,
                "clipped": usable != lb_grid,
            }

        if artifacts:
            artifacts.write_coverage_report(report)
        return adjusted, tf_earliest

    def _select_from_feature_provider(
        self, feature_provider, symbols: List[str], default_ws: List[int], artifacts: Optional[LayerAArtifacts]
    ) -> Optional[RidgeLayerResult]:
        tf_label = getattr(feature_provider, "primary_timeframe", config.PRIMARY_TIMEFRAME)
        logger.info("[LayerA] Using preloaded features for timeframe %s", tf_label)

        k_per_asset: Dict[str, float] = {}
        msep_per_asset: Dict[str, float] = {}
        rl_vs_ls: Dict[str, Optional[float]] = {}
        samples_per_asset: Dict[str, int] = {}
        w_per_asset: Dict[str, int] = {}
        train_samples_used: Dict[str, int] = {}

        for sym in symbols:
            res = self._select_for_asset(tf_label, sym, feature_provider, default_ws, artifacts, None)
            if res is None:
                continue
            sym_out, best_k, best_w, best_msep, rl, samples, best_ic = res
            k_per_asset[sym_out] = best_k
            msep_per_asset[sym_out] = best_msep
            rl_vs_ls[sym_out] = rl
            samples_per_asset[sym_out] = samples
            w_per_asset[sym_out] = best_w
            train_samples_used[sym_out] = min(best_w, samples)

        if not k_per_asset:
            return RidgeLayerResult(timeframe=tf_label, lookback_days=None)

        res = RidgeLayerResult(
            timeframe=tf_label,
            k_per_asset=k_per_asset,
            msep_per_asset=msep_per_asset,
            rl_vs_ls=rl_vs_ls,
            t_threshold=self.t_threshold,
            samples_per_asset=samples_per_asset,
            w_per_asset=w_per_asset,
            train_samples_used=train_samples_used,
            lookback_days=None,
        )
        if artifacts:
            artifacts.write_summary(
                {
                    "requested_start": None,
                    "requested_end": None,
                    "fast_mode": self.fast_mode,
                    "train_min": self.train_min,
                    "val_len": self.val_len,
                    "embargo": self.embargo,
                    "k_grid": list(self.k_grid),
                    "tf_candidates": [tf_label],
                    "lookback_days_grid": {tf_label: [None]},
                    "max_splits": self.max_splits,
                    "min_splits": self.min_splits,
                    "chosen_timeframe": tf_label,
                    "chosen_lookback_days": None,
                    "tf_score": None,
                    "coverage_ratio": len(k_per_asset) / max(1, len(symbols)),
                    "bar_caps": {
                        "layerA_max": getattr(config, "LAYERA_MAX_BARS_BY_TF", {}),
                        "layerA_regression_max_bars": getattr(config, "LAYERA_REGRESSION_MAX_BARS", None),
                        "regression_max_bars": getattr(config, "REGRESSION_MAX_BARS", None),
                    },
                    "per_asset": {
                        sym: {
                            "k": k_per_asset.get(sym),
                            "w": w_per_asset.get(sym),
                            "msep": msep_per_asset.get(sym),
                            "rl_vs_ls": rl_vs_ls.get(sym),
                            "samples": samples_per_asset.get(sym),
                        }
                        for sym in k_per_asset
                    },
                }
            )
        return res

    def _build_engine_for_tf(
        self,
        timeframe: str,
        symbols: List[str],
        base_client,
        w_grid: List[int],
        lookback_days: Optional[int] = None,
        end_ts: Optional[datetime] = None,
        artifacts: Optional[LayerAArtifacts] = None,
    ) -> Tuple[Optional[DataEngine], Dict[str, Dict[str, Any]]]:
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
        asset_windows: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            parquet_path = data_dir / f"{sym}-{timeframe}.parquet"
            csv_path = data_dir / f"{sym}-{timeframe}.csv"

            effective_end = end_ts
            if effective_end is None and lookback_days is not None and parquet_path.exists():
                effective_end = _infer_last_ts_from_parquet(parquet_path)

            start_cutoff = None
            if lookback_days is not None and effective_end is not None:
                start_cutoff = effective_end - pd.Timedelta(days=lookback_days)

            tf_cap = self._resolve_bar_cap(timeframe)
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
                if artifacts:
                    artifacts.warn(
                        {
                            "type": "missing_data",
                            "symbol": sym,
                            "timeframe": timeframe,
                            "reason": missing_reason,
                            "start_cutoff": start_cutoff,
                            "end_ts": effective_end,
                        }
                    )
                continue

            ts_vals = [bar["timestamp"] for bar in bars if "timestamp" in bar]
            start_ts_val = min(ts_vals) if ts_vals else None
            end_ts_val = max(ts_vals) if ts_vals else None
            cap_hit = max_rows is not None and stats.get("rows_read", 0) > stats.get("rows_kept", 0) >= max_rows
            asset_windows[sym] = {
                "start_ts": pd.to_datetime(start_ts_val, unit="ms", utc=True) if start_ts_val else None,
                "end_ts": pd.to_datetime(end_ts_val, unit="ms", utc=True) if end_ts_val else None,
                "rows_read": stats.get("rows_read", 0),
                "rows_kept": stats.get("rows_kept", len(bars)),
                "cap_hit": cap_hit,
                "cap": max_rows,
            }
            if cap_hit and artifacts:
                artifacts.warn(
                    {
                        "type": "cap_hit",
                        "symbol": sym,
                        "timeframe": timeframe,
                        "lookback_days": lookback_days,
                        "rows_read": stats.get("rows_read", 0),
                        "rows_kept": stats.get("rows_kept", len(bars)),
                        "cap": max_rows,
                    }
                )
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
            return None, {}

        return engine, asset_windows

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

        cap = self._resolve_bar_cap(timeframe)
        if cap is not None:
            estimate = min(estimate, cap)
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

    def _select_for_asset(
        self,
        tf: str,
        sym: str,
        tf_engine: DataEngine,
        w_grid: List[int],
        artifacts: Optional[LayerAArtifacts] = None,
        lookback_days: Optional[int] = None,
    ):
        """Run CV for a single asset (used in multiprocessing pool)."""
        X_raw, y_raw, ts, cols = tf_engine.get_feature_matrix(sym)
        _log_mem("asset-cv-start", tf=tf, asset=sym, bars=len(y_raw), features=X_raw.shape[1] if X_raw.size else 0)
        if y_raw.size < self.train_min:
            if artifacts:
                artifacts.warn(
                    {
                        "type": "insufficient_history",
                        "symbol": sym,
                        "timeframe": tf,
                        "lookback_days": lookback_days,
                        "available_samples": int(y_raw.size),
                        "train_min": self.train_min,
                    }
                )
            return None
        # Apply unified cap once to keep sample count consistent across CV and final fit.
        cap = self._resolve_bar_cap(tf)
        if cap:
            X_raw = X_raw[-cap:]
            y_raw = y_raw[-cap:]
            ts = ts[-cap:]
        n = len(y_raw)
        if n < self.train_min:
            if artifacts:
                artifacts.warn(
                    {
                        "type": "insufficient_history_after_cap",
                        "symbol": sym,
                        "timeframe": tf,
                        "lookback_days": lookback_days,
                        "available_samples": int(n),
                        "train_min": self.train_min,
                        "cap": cap,
                    }
                )
            return None
        available_samples = n

        k_vec = np.array(self.k_grid, dtype=float)
        best_k = None
        best_msep = np.inf
        best_w = None
        best_ic = None
        best_rl = None
        split_rows: List[Dict[str, Any]] = []

        def _cv_blocks(n_obs: int, window: int, v_len: int) -> List[Tuple[int, int, int, int]]:
            splits: List[Tuple[int, int, int, int]] = []
            start_idx = 0
            while True:
                train_end = start_idx + window
                val_start = train_end + self.embargo
                val_end = val_start + v_len
                if val_end > n_obs:
                    break
                splits.append((start_idx, train_end, val_start, val_end))
                start_idx += v_len
            return splits

        for W in w_grid:
            max_val = n - W - self.embargo
            if max_val <= 0 or n < max(W, self.train_min):
                continue
            v_len = min(self.val_len, max(max_val, 1), max(3, W // 2))
            if W < 2 * v_len:
                continue
            splits = _cv_blocks(n, W, v_len)
            if len(splits) < self.min_splits and v_len > 1:
                v_len = max(1, v_len // 2)
                splits = _cv_blocks(n, W, v_len)
            if self.max_splits and len(splits) > self.max_splits:
                splits = splits[: self.max_splits]
            if not splits:
                continue

            eye_penalty = None
            mseps_accum = np.zeros_like(k_vec, dtype=float)
            counts = np.zeros_like(k_vec, dtype=float)
            ic_accum = np.zeros_like(k_vec, dtype=float)
            ic_counts = np.zeros_like(k_vec, dtype=float)
            for split_idx, (start_idx, train_end, val_start, val_end) in enumerate(splits):
                X_train = X_raw[start_idx:train_end]
                y_train = y_raw[start_idx:train_end]
                X_val = X_raw[val_start:val_end]
                y_val = y_raw[val_start:val_end]

                scaler = Standardizer().fit(X_train)
                X_train_std = self._with_intercept(scaler.transform(X_train))
                X_val_std = self._with_intercept(scaler.transform(X_val))

                XtX = X_train_std.T @ X_train_std
                Xty = X_train_std.T @ y_train
                if eye_penalty is None or eye_penalty.shape[0] != XtX.shape[0]:
                    eye_penalty = np.eye(XtX.shape[0])
                    eye_penalty[0, 0] = 0.0
                mats = XtX[None, :, :] + k_vec[:, None, None] * eye_penalty
                rhs = np.broadcast_to(Xty, (k_vec.size, Xty.shape[0]))[..., None]
                try:
                    betas = np.linalg.solve(mats, rhs).squeeze(-1)
                except np.linalg.LinAlgError:
                    betas = np.array([np.linalg.pinv(mats[i]) @ Xty for i in range(k_vec.size)])

                preds = betas @ X_val_std.T  # shape (k, val_len)
                errors = y_val[None, :] - preds
                mseps = np.mean(errors * errors, axis=1)
                mseps_accum += mseps
                counts += 1.0
                # IC per k
                for ki, pred_row in enumerate(preds):
                    if pred_row.size < 3:
                        continue
                    try:
                        if np.std(pred_row) < 1e-12 or np.std(y_val) < 1e-12:
                            continue
                        ic_val = np.corrcoef(pred_row, y_val)[0, 1]
                        if not np.isfinite(ic_val):
                            # Spearman fallback
                            ic_val = pd.Series(pred_row).corr(pd.Series(y_val), method="spearman")
                    except Exception:
                        ic_val = np.nan
                    if np.isfinite(ic_val):
                        ic_accum[ki] += ic_val
                        ic_counts[ki] += 1.0
                if artifacts:
                    train_start_ts = ts[start_idx] if start_idx < len(ts) else None
                    train_end_ts = ts[train_end - 1] if train_end - 1 < len(ts) else None
                    val_start_ts = ts[val_start] if val_start < len(ts) else None
                    val_end_ts = ts[val_end - 1] if val_end - 1 < len(ts) else None
                    for k_val, msep_val in zip(k_vec, mseps):
                        split_rows.append(
                            {
                                "window": W,
                                "k": float(k_val),
                                "split_idx": split_idx,
                                "train_start_ts": train_start_ts,
                                "train_end_ts": train_end_ts,
                                "val_start_ts": val_start_ts,
                                "val_end_ts": val_end_ts,
                                "val_len": val_end - val_start,
                                "msep": float(msep_val),
                            }
                        )

            if np.any(counts == 0):
                continue
            mean_msep = mseps_accum / counts
            mean_ic = np.full_like(mean_msep, -np.inf)
            valid_ic = ic_counts > 0
            mean_ic[valid_ic] = ic_accum[valid_ic] / ic_counts[valid_ic]

            # Objective: maximize IC, tie-break by msep then smaller k
            positive_ic_mask = mean_ic > 0
            candidates_mask = positive_ic_mask
            if np.any(candidates_mask):
                idx = int(np.nanargmax(mean_ic))
                # tie-break on msep within IC tolerance
                ic_best = mean_ic[idx]
                close_ic = np.isclose(mean_ic, ic_best, rtol=1e-6, atol=1e-9) & candidates_mask
                if np.any(close_ic):
                    msep_candidates = mean_msep.copy()
                    msep_candidates[~close_ic] = np.inf
                    idx = int(np.nanargmin(msep_candidates))
                    # tie-break by smallest k
                    best_msep_tie = msep_candidates[idx]
                    close_msep = np.isclose(mean_msep, best_msep_tie, rtol=1e-6, atol=1e-9) & close_ic
                    if np.any(close_msep):
                        k_candidates = k_vec.copy()
                        k_candidates[~close_msep] = np.inf
                        idx = int(np.nanargmin(k_candidates))
                msep = float(mean_msep[idx])
                ic_val = float(mean_ic[idx]) if np.isfinite(mean_ic[idx]) else None
            else:
                # No positive IC: fall back to OLS if available
                zero_k_idx = np.where(k_vec == 0.0)[0]
                if zero_k_idx.size:
                    idx = int(zero_k_idx[0])
                else:
                    idx = int(np.nanargmin(mean_msep))
                msep = float(mean_msep[idx])
                ic_val = None

            if msep < best_msep or best_k is None:
                best_msep = msep
                best_k = float(k_vec[idx])
                best_w = W
                best_ic = ic_val

        if best_k is None or best_w is None:
            if artifacts:
                artifacts.warn(
                    {
                        "type": "no_valid_cv_splits",
                        "symbol": sym,
                        "timeframe": tf,
                        "lookback_days": lookback_days,
                        "candidate_windows": w_grid,
                    }
                )
            return None

        forecaster = RidgeRegressionForecaster(k_grid=[best_k], t_threshold=self.t_threshold)
        # Fit with the chosen k; enforce RL guard post-fit
        def _fit_with_k(k_val: float) -> Optional[RidgeForecast]:
            return forecaster.forecast(
                sym,
                X_raw,
                y_raw,
                fixed_k=k_val,
                train_window=best_w,
                skip_cv=True,
                train_min=self.train_min,
                feature_columns=cols,
            )

        ridge_result = _fit_with_k(best_k)
        if not ridge_result:
            return None

        if ridge_result.rl_vs_ls is None or ridge_result.rl_vs_ls > 1.001:
            # Try k=0 (OLS) as fallback
            if best_k != 0.0:
                fallback = _fit_with_k(0.0)
                if fallback and fallback.rl_vs_ls is not None and fallback.rl_vs_ls <= 1.001:
                    ridge_result = fallback
                    best_k = 0.0
            # If still bad, warn and skip this asset
            if ridge_result.rl_vs_ls is not None and ridge_result.rl_vs_ls > 1.001:
                if artifacts:
                    artifacts.warn(
                        {
                            "type": "ridge_underperforms_ls",
                            "symbol": sym,
                            "timeframe": tf,
                            "lookback_days": lookback_days,
                            "best_k": best_k,
                            "best_w": best_w,
                            "rl_vs_ls": ridge_result.rl_vs_ls,
                        }
                    )
                return None

        if artifacts and split_rows:
            def _ts_iso(ts_val: Optional[int]) -> Optional[str]:
                if ts_val is None:
                    return None
                try:
                    return pd.to_datetime(ts_val, unit="ms", utc=True).isoformat()
                except Exception:
                    return str(ts_val)

            path = artifacts.cv_grid_dir / f"{sym}-{tf}-lb{lookback_days if lookback_days is not None else 'all'}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "window",
                "k",
                "split_idx",
                "train_start_ts",
                "train_end_ts",
                "val_start_ts",
                "val_end_ts",
                "val_len",
                "msep",
            ]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in split_rows:
                    row = dict(row)
                    row["train_start_ts"] = _ts_iso(row["train_start_ts"])
                    row["train_end_ts"] = _ts_iso(row["train_end_ts"])
                    row["val_start_ts"] = _ts_iso(row["val_start_ts"])
                    row["val_end_ts"] = _ts_iso(row["val_end_ts"])
                    writer.writerow(row)
        if artifacts and ridge_result:
            diag_path = artifacts.model_diag_dir / f"{sym}-{tf}-lb{lookback_days if lookback_days is not None else 'all'}.json"
            diag_payload = {
                "symbol": sym,
                "timeframe": tf,
                "lookback_days": lookback_days,
                "best_k": best_k,
                "best_window": best_w,
                "best_msep": best_msep,
                "gcv": ridge_result.gcv,
                "rl_vs_ls": ridge_result.rl_vs_ls,
                "samples": ridge_result.samples,
                "t_threshold": ridge_result.t_threshold,
                "dropped_features": ridge_result.dropped_features,
                "hat_mean": ridge_result.hat_mean,
                "hat_max": ridge_result.hat_max,
                "resid_sigma": ridge_result.resid_sigma,
                "feature_columns": ridge_result.feature_columns,
                "outliers_dropped": ridge_result.outliers_dropped,
                "ic": best_ic,
            }
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w") as f:
                json.dump(diag_payload, f, indent=2, default=str)

        logger.info(
            "[LayerA] TF=%s asset=%s k=%.4g W=%d msep=%.4e samples=%d",
            tf,
            sym,
            best_k,
            best_w,
            best_msep,
            available_samples,
        )
        _log_mem("asset-cv-end", tf=tf, asset=sym, bars=n, best_k=best_k, best_w=best_w)
        return sym, best_k, best_w, best_msep, ridge_result.rl_vs_ls, available_samples, best_ic
