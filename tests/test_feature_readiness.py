import numpy as np

from data.feature_builder import FeatureWindowStore


def test_feature_window_store_readiness_by_length():
    store = FeatureWindowStore(maxlen=5)
    schema_len = 3
    for _ in range(4):
        store.append("BTCUSDT", {"a": 1, "b": 2, "c": 3}, ["a", "b", "c"])
    assert store.get_window("BTCUSDT", 5) is None
    store.append("BTCUSDT", {"a": 1, "b": 2, "c": 3}, ["a", "b", "c"])
    window = store.get_window("BTCUSDT", 5)
    assert window is not None
    assert window.shape == (5, schema_len)
    assert np.isfinite(window).all()
