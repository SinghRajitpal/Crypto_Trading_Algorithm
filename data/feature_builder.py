"""Feature engineering utilities for GRU inputs.

Computes log_return/log_volume plus momentum/volatility/funding/flow z-features
with winsorization and expanding z-scores. Supports batch (offline) and online
updates while enforcing lag discipline on 8h bars.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config


def _clip(value: float) -> float:
    if value is None or not math.isfinite(value):
        return float("nan")
    return float(value)


class FeatureEngineer:
    """Compute and store GRU feature vectors per symbol.

    Features (ordered by config.GRU_FEATURE_SCHEMA):
        - log_return (raw)
        - log_volume (raw, log1p)
        - tsmom_fast_vol_z (k=3)
        - tsmom_med_vol_z (k=15)
        - rsi12_innov_z (RSI12 - EMA20(RSI12), z-scored)
        - atrpct_innov_z (ATR14% - EMA20(ATR%), z-scored)
        - funding_z (z-scored)
        - ofi_z (computed via OBV tilt here)
    """

    def __init__(
        self,
        warmup: Optional[int] = None,
        winsor_pct: Optional[float] = None,
        adv_window: Optional[int] = None,
        schema: Optional[List[str]] = None,
    ) -> None:
        self.warmup = warmup if warmup is not None else config.FEATURE_Z_WARMUP
        self.winsor_pct = winsor_pct if winsor_pct is not None else config.FEATURE_WINSOR_PCT
        self.adv_window = adv_window if adv_window is not None else config.FEATURE_ADV_WINDOW
        self.schema = schema or list(config.GRU_FEATURE_SCHEMA)

        self._close: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._high: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._low: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._volume: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._log_returns: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._dollar_vol: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.adv_window))
        self._funding: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._ofi: Dict[str, Deque[float]] = defaultdict(lambda: deque())

        # Raw history per feature for winsor/z-score
        self._raw_hist: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # ------------------------------ Public APIs ------------------------------
    def update(self, symbol: str, bar: Dict[str, float], aux: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Ingest one completed bar and return feature vector for that bar.

        Args:
            symbol: asset symbol
            bar: dict with timestamp, open, high, low, close, volume (floats)
            aux: optional dict with keys funding (float)
        """
        sym = symbol.upper()
        aux = aux or {}
        close = _clip(bar.get("close"))
        open_p = _clip(bar.get("open", close))
        high = _clip(bar.get("high", close))
        low = _clip(bar.get("low", close))
        volume = _clip(bar.get("volume", 0.0))

        # Append price/volume history
        self._close[sym].append(close)
        self._high[sym].append(high)
        self._low[sym].append(low)
        self._volume[sym].append(volume)
        self._dollar_vol[sym].append(close * volume if math.isfinite(close) else 0.0)

        # Log return (uses previous close)
        lr_val = self._compute_log_return(sym)
        if lr_val is not None and math.isfinite(lr_val):
            self._log_returns[sym].append(lr_val)

        # Funding / OBV tilt storage
        fund_val = _clip(aux.get("funding", 0.0))
        self._funding[sym].append(fund_val)
        obv = self._compute_obv_tilt(sym)
        self._ofi[sym].append(obv)

        feats = self._compute_features(sym)
        return feats

    def compute_batch(
        self,
        df: pd.DataFrame,
        funding: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute features for a full history DataFrame (indexed by timestamp)."""
        records: List[Dict[str, float]] = []
        funding = funding.reindex(df.index).fillna(0.0) if funding is not None else None

        for ts, row in df.iterrows():
            bar = {
                "timestamp": int(pd.Timestamp(ts).timestamp() * 1000),
                "open": row.get("open", np.nan),
                "high": row.get("high", np.nan),
                "low": row.get("low", np.nan),
                "close": row.get("close", np.nan),
                "volume": row.get("volume", 0.0),
            }
            aux = {
                "funding": funding.loc[ts] if funding is not None else 0.0,
            }
            feats = self.update("_BATCH", bar, aux=aux)
            feats["timestamp"] = ts
            records.append(feats)

        out = pd.DataFrame(records).set_index("timestamp")
        return out

    # ---------------------------- Feature helpers ---------------------------
    def _compute_features(self, sym: str) -> Dict[str, float]:
        lr = self._log_returns[sym]
        vol = self._volume[sym]
        close = self._close[sym]
        high = self._high[sym]
        low = self._low[sym]

        log_return = lr[-1] if lr else float("nan")
        log_volume = math.log1p(vol[-1]) if vol else float("nan")

        fast = self._tsmom(sym, config.FEATURE_TSMOM_FAST_K)
        med = self._tsmom(sym, config.FEATURE_TSMOM_MED_K)
        rsi_innov = self._rsi_innovation(sym)
        atr_innov = self._atrpct_innovation(sym)
        funding = self._funding[sym][-1] if self._funding[sym] else float("nan")
        ofi = self._ofi[sym][-1] if self._ofi[sym] else float("nan")

        raw_map = {
            "log_return": log_return,
            "log_volume": log_volume,
            "tsmom_fast_vol_z": fast,
            "tsmom_med_vol_z": med,
            "rsi12_innov_z": rsi_innov,
            "atrpct_innov_z": atr_innov,
            "funding_z": funding,
            "ofi_z": ofi,
        }

        # Apply winsor+z for z-features (skip raw log_return/log_volume)
        feats: Dict[str, float] = {}
        for name in self.schema:
            val = raw_map.get(name, float("nan"))
            if name in ("log_return", "log_volume"):
                feats[name] = val
                continue
            feats[name] = self._winsor_z(sym, name, val)
        return feats

    def _winsor_z(self, sym: str, feature: str, value: float) -> float:
        hist = self._raw_hist[sym][feature]
        hist.append(value)
        arr_all = np.array(hist, dtype=float)
        finite = arr_all[np.isfinite(arr_all)]
        if finite.size == 0:
            return float("nan")
        if finite.size < self.warmup:
            return float("nan")
        p_low = np.nanpercentile(finite, self.winsor_pct * 100)
        p_high = np.nanpercentile(finite, 100 - self.winsor_pct * 100)
        latest = arr_all[-1]
        latest = float(np.clip(latest, p_low, p_high)) if math.isfinite(latest) else float("nan")
        clipped = np.clip(finite, p_low, p_high)
        mu = np.nanmean(clipped)
        sigma = np.nanstd(clipped, ddof=0)
        if sigma == 0 or math.isnan(sigma):
            sigma = 1.0
        return float((latest - mu) / sigma)

    # ------------------------------ Indicators ------------------------------
    def _compute_log_return(self, sym: str) -> Optional[float]:
        closes = self._close[sym]
        if len(closes) < 2:
            return None
        prev = closes[-2]
        curr = closes[-1]
        if prev is None or prev <= 0 or not math.isfinite(prev) or not math.isfinite(curr):
            return None
        return math.log(curr) - math.log(prev)

    def _tsmom(self, sym: str, k: int) -> float:
        lr = self._log_returns[sym]
        if len(lr) == 0:
            return float("nan")
        k_eff = min(k, len(lr))
        window = list(lr)[-k_eff:]
        s_val = float(np.nansum(window))
        v_val = float(math.sqrt(np.nansum(np.square(window)))) or 1e-9
        return s_val / v_val

    def _rsi_innovation(self, sym: str) -> float:
        closes = self._close[sym]
        if len(closes) < 2:
            return float("nan")
        arr = np.array(closes, dtype=float)
        win = max(1, min(config.FEATURE_RSI_WINDOW, len(closes) - 1))
        rsi_series = self._wilder_rsi(arr, win)
        if rsi_series.size == 0:
            return float("nan")
        rsi_val = rsi_series[-1]
        ema_span = max(1, min(config.FEATURE_RSI_EMA, len(rsi_series)))
        ema = self._ema(pd.Series(rsi_series), ema_span)
        ema_val = ema.iloc[-1]
        return float(rsi_val - ema_val)

    def _atrpct_innovation(self, sym: str) -> float:
        high = np.array(self._high[sym], dtype=float)
        low = np.array(self._low[sym], dtype=float)
        close = np.array(self._close[sym], dtype=float)
        if close.size < 2:
            return float("nan")
        tr = self._true_range(high, low, close)
        win = max(1, min(config.FEATURE_ATR_WINDOW, tr.size))
        atr = self._wilder(tr, win)
        if atr.size == 0 or close[-1] == 0 or math.isnan(close[-1]):
            return float("nan")
        atr_pct = atr[-1] / close[-1]
        ema_span = max(1, min(config.FEATURE_ATR_EMA, len(atr)))
        ema = self._ema(pd.Series(atr / np.where(close == 0, np.nan, close)), ema_span)
        ema_val = ema.iloc[-1]
        return float(atr_pct - ema_val)

    def _compute_obv_tilt(self, sym: str) -> float:
        close = self._close[sym]
        vol = self._volume[sym]
        dollar = self._dollar_vol[sym]
        if len(close) < 2 or not vol:
            return float("nan")
        adv = float(np.nanmean(dollar)) if dollar else 0.0
        if adv == 0:
            return 0.0
        sign = 0.0
        if math.isfinite(close[-1]) and math.isfinite(close[-2]):
            if close[-1] > close[-2]:
                sign = 1.0
            elif close[-1] < close[-2]:
                sign = -1.0
        return sign * vol[-1] / adv

    # ------------------------------ Math utils ------------------------------
    @staticmethod
    def _wilder(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0:
            return np.array([])
        out = np.empty_like(values)
        out[:] = np.nan
        if values.size < window:
            return out
        out[window - 1] = np.nanmean(values[:window])
        alpha = 1.0 / window
        for i in range(window, values.size):
            prev = out[i - 1]
            out[i] = prev + alpha * (values[i] - prev)
        return out

    def _wilder_rsi(self, close: np.ndarray, window: int) -> np.ndarray:
        diff = np.diff(close)
        if diff.size == 0:
            return np.array([])
        up = np.where(diff > 0, diff, 0.0)
        down = np.where(diff < 0, -diff, 0.0)
        avg_up = self._wilder(up, window)
        avg_down = self._wilder(down, window)
        rsi = np.empty_like(close)
        rsi[:] = np.nan
        start = max(window, 1)
        for i in range(start, close.size):
            up_val = avg_up[i - 1] if i - 1 < avg_up.size else np.nan
            down_val = avg_down[i - 1] if i - 1 < avg_down.size else np.nan
            if down_val == 0 or math.isnan(down_val):
                # If no down moves, treat RSI as overbought (100) when up moves exist, else 50 neutral
                if not math.isnan(up_val) and up_val > 0:
                    rsi[i] = 100.0
                elif not math.isnan(up_val) and up_val == 0:
                    rsi[i] = 50.0
                else:
                    rsi[i] = np.nan
                continue
            rs = up_val / down_val
            rsi[i] = 100 - (100 / (1 + rs)) if rs is not None else np.nan
        return rsi

    @staticmethod
    def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        prev_close = np.concatenate(([np.nan], close[:-1]))
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
        return tr

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=1).mean()


class FeatureWindowStore:
    """Keep rolling feature windows per symbol for GRU inference."""

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._store: Dict[str, Deque[List[float]]] = defaultdict(lambda: deque(maxlen=maxlen))

    def append(self, symbol: str, feature_vec: Dict[str, float], schema: Iterable[str]) -> None:
        ordered = [feature_vec.get(name, float("nan")) for name in schema]
        self._store[symbol.upper()].append(ordered)

    def get_window(self, symbol: str, lookback: int) -> Optional[np.ndarray]:
        sym = symbol.upper()
        window = self._store.get(sym)
        if not window or len(window) < lookback:
            return None
        arr = np.array(list(window)[-lookback:], dtype=float)
        if arr.shape != (lookback, len(window[0])):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr
