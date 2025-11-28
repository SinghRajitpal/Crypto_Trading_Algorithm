"""Utilities for building GRU training sequences from OHLCV data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log return and log-volume features."""
    if not {"close", "volume"}.issubset(set(df.columns)):
        raise ValueError("DataFrame must contain 'close' and 'volume' columns")
    close = df["close"].astype(float)
    volume = df["volume"].fillna(0.0).astype(float)
    log_return = np.log(close).diff()
    log_volume = np.log1p(volume)
    features = pd.DataFrame(
        {
            "log_return": log_return,
            "log_volume": log_volume,
        },
        index=df.index,
    )
    features = features.dropna()
    return features


def build_sequences(df: pd.DataFrame, lookback: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Generate supervised sequences for GRU training.

    Args:
        df: OHLCV dataframe indexed by timestamp with at least close and volume.
        lookback: number of timesteps in each input sequence.

    Returns:
        X: ndarray (n_samples, lookback, 2)
        y: ndarray (n_samples,)
    """
    feats = _compute_features(df)
    values = feats[["log_return", "log_volume"]].to_numpy()
    X_list = []
    y_list = []
    for idx in range(lookback, len(values)):
        X_list.append(values[idx - lookback : idx])
        y_list.append(values[idx, 0])  # Next log return
    if not X_list:
        return np.empty((0, lookback, 2)), np.empty((0,))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=float)


def build_sequences_with_index(df: pd.DataFrame, lookback: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate supervised sequences plus target timestamps.

    Returns:
        X: ndarray (n_samples, lookback, 2)
        y: ndarray (n_samples,)
        ts: ndarray of pandas.Timestamp for each target
    """
    feats = _compute_features(df)
    values = feats[["log_return", "log_volume"]].to_numpy()
    idx = feats.index.to_numpy()
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    ts_list: list[np.datetime64] = []
    for i in range(lookback, len(values)):
        X_list.append(values[i - lookback : i])
        y_list.append(values[i, 0])
        ts_list.append(idx[i])
    if not X_list:
        return np.empty((0, lookback, 2)), np.empty((0,)), np.empty((0,))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=float), np.array(ts_list)
