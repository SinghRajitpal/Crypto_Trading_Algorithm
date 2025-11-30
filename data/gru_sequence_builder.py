"""Utilities for building GRU training sequences from OHLCV data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

import config
from data.feature_builder import FeatureEngineer


def _compute_features(df: pd.DataFrame, funding: Optional[pd.Series] = None) -> pd.DataFrame:
    """Compute full GRU feature set using FeatureEngineer."""
    if not {"close", "volume"}.issubset(set(df.columns)):
        raise ValueError("DataFrame must contain 'close' and 'volume' columns")
    fe = FeatureEngineer()
    feats = fe.compute_batch(df, funding=funding)
    feats = feats[config.GRU_FEATURE_SCHEMA]
    # Keep rows where all features are finite
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def build_sequences(
    df: pd.DataFrame,
    lookback: int = 64,
    funding: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate supervised sequences for GRU training.

    Args:
        df: OHLCV dataframe indexed by timestamp with at least close and volume.
        lookback: number of timesteps in each input sequence.

    Returns:
        X: ndarray (n_samples, lookback, n_features) where each window ends at t-1
        y: ndarray (n_samples,) where each target is log_return at t
    """
    feats = _compute_features(df, funding=funding)
    values = feats[config.GRU_FEATURE_SCHEMA].to_numpy()
    X_list = []
    y_list = []
    for idx in range(lookback, len(values)):
        X_list.append(values[idx - lookback : idx])
        y_list.append(values[idx, 0])  # Next log return
    if not X_list:
        return np.empty((0, lookback, values.shape[1] if values.size else 0)), np.empty((0,))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=float)


def build_sequences_with_index(
    df: pd.DataFrame,
    lookback: int = 64,
    funding: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate supervised sequences plus target timestamps.

    Returns:
        X: ndarray (n_samples, lookback, n_features) where each window ends at t-1
        y: ndarray (n_samples,)
        ts: ndarray of pandas.Timestamp for each target (aligned with y)
    """
    feats = _compute_features(df, funding=funding)
    values = feats[config.GRU_FEATURE_SCHEMA].to_numpy()
    idx = feats.index.to_numpy()
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    ts_list: list[np.datetime64] = []
    for i in range(lookback, len(values)):
        X_list.append(values[i - lookback : i])
        y_list.append(values[i, 0])
        ts_list.append(idx[i])
    if not X_list:
        return np.empty((0, lookback, values.shape[1] if values.size else 0)), np.empty((0,)), np.empty((0,))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=float), np.array(ts_list)
