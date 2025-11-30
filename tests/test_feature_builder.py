import numpy as np
import pandas as pd

import config
from data.feature_builder import FeatureEngineer, FeatureWindowStore
from data import gru_sequence_builder


def _sample_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="8H")
    close = np.linspace(100.0, 110.0, n)
    volume = np.linspace(1_000, 2_000, n)
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": volume,
    }, index=idx)
    return df


def test_feature_engineer_batch_outputs_schema_and_no_nans_after_warmup():
    schema = list(config.GRU_FEATURE_SCHEMA)
    fe = FeatureEngineer(warmup=3, winsor_pct=0.0, adv_window=3, schema=schema)
    df = _sample_df(10)
    feats = fe.compute_batch(df)
    assert list(feats.columns) == schema
    last_row = feats.iloc[-1]
    assert last_row.notna().all()
    # log_return should be finite and match diff of logs
    expected_lr = np.log(df.iloc[-1]["close"]) - np.log(df.iloc[-2]["close"])
    assert np.isclose(last_row["log_return"], expected_lr)


def test_obv_tilt_feature_is_finite_and_nonzero():
    schema = list(config.GRU_FEATURE_SCHEMA)
    fe = FeatureEngineer(warmup=3, winsor_pct=0.0, adv_window=3, schema=schema)
    df = _sample_df(10)
    feats = fe.compute_batch(df)
    ofi_last = feats.iloc[-1]["ofi_z"]
    assert np.isfinite(ofi_last)
    assert ofi_last != 0.0


def test_funding_feature_is_ingested_and_finite():
    schema = list(config.GRU_FEATURE_SCHEMA)
    fe = FeatureEngineer(warmup=3, winsor_pct=0.0, adv_window=3, schema=schema)
    df = _sample_df(10)
    # Create a funding series with a ramp to avoid zero variance
    funding = pd.Series(np.linspace(0.0, 0.01, len(df)), index=df.index)
    feats = fe.compute_batch(df, funding=funding)
    fund_last = feats.iloc[-1]["funding_z"]
    assert np.isfinite(fund_last)
    assert fund_last != 0.0


def test_feature_window_store_returns_valid_window():
    schema = list(config.GRU_FEATURE_SCHEMA)
    store = FeatureWindowStore(maxlen=5)
    feature_vec = {name: float(i) for i, name in enumerate(schema)}
    for _ in range(5):
        store.append("BTCUSDT", feature_vec, schema)
    window = store.get_window("BTCUSDT", 3)
    assert window is not None
    assert window.shape == (3, len(schema))
    assert np.isfinite(window).all()


def test_gru_sequence_builder_uses_full_schema(monkeypatch):
    # Reduce warmup so that features exist in the small sample
    monkeypatch.setattr(config, "FEATURE_Z_WARMUP", 3)
    monkeypatch.setattr(config, "FEATURE_WINSOR_PCT", 0.0)
    df = _sample_df(12)
    lookback = 4
    X, y = gru_sequence_builder.build_sequences(df, lookback=lookback)
    assert X.shape[2] == len(config.GRU_FEATURE_SCHEMA)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
