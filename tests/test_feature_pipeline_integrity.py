import os
import numpy as np
import pandas as pd

import config
from data.feature_builder import FeatureEngineer
from data.gru_sequence_builder import build_sequences_with_index, _compute_features


def _load_cached_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{symbol}-{timeframe}.parquet")
    df = pd.read_parquet(cache_path)
    df = df.head(500)
    if "timestamp" in df.columns:
        df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _load_cached_funding(symbol: str) -> pd.Series:
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{symbol}-funding.csv")
    if not os.path.exists(cache_path):
        return None
    s = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")["rate"]
    s = s.head(500)
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce", format="mixed")
    s = s[s.index.notnull()]
    return s


def test_feature_engineer_on_cached_data_no_nans_after_warmup():
    symbol = config.DEFAULT_UNIVERSE[0]
    timeframe = config.GRU_TIMEFRAME
    df = _load_cached_ohlcv(symbol, timeframe)
    funding = _load_cached_funding(symbol)
    fe = FeatureEngineer(warmup=3, schema=list(config.GRU_FEATURE_SCHEMA))
    feats = fe.compute_batch(df, funding=funding)
    feats = feats[config.GRU_FEATURE_SCHEMA]
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Drop initial warmup rows
    feats = feats.iloc[3:]
    assert feats.notna().all().all()
    # Basic sanity: non-zero variance for z-features
    z_cols = [c for c in config.GRU_FEATURE_SCHEMA if c not in ("log_return", "log_volume")]
    for col in z_cols:
        assert np.nanstd(feats[col]) > 0


def test_sequence_builder_matches_feature_engineer(monkeypatch):
    symbol = config.DEFAULT_UNIVERSE[0]
    timeframe = config.GRU_TIMEFRAME
    df = _load_cached_ohlcv(symbol, timeframe)
    funding = _load_cached_funding(symbol)
    lookback = min(8, config.GRU_LOOKBACK)  # keep small for test runtime

    # Ensure warmup matches between direct FeatureEngineer use and sequence builder
    monkeypatch.setattr(config, "FEATURE_Z_WARMUP", 3)

    # Use the same computation path as build_sequences (via _compute_features)
    feats_df = _compute_features(df, funding=funding)
    feats_df = feats_df[config.GRU_FEATURE_SCHEMA]

    X, y, ts = build_sequences_with_index(df, lookback=lookback, funding=funding)
    if X.size == 0:
        raise AssertionError("No sequences built; check cached data length")

    # Reconstruct expected window directly from the same feature computation
    # used by build_sequences_with_index
    # build_sequences_with_index uses window = values[idx-lookback:idx] with target at idx
    # so the last window excludes the last row (target). Reconstruct accordingly.
    vals = feats_df.to_numpy()
    window_from_df = vals[-(lookback + 1) : -1]
    window_from_X = X[-1]
    assert window_from_X.shape == window_from_df.shape
    assert np.allclose(window_from_X, window_from_df, atol=1e-6, rtol=1e-6)
