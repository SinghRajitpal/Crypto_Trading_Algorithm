import numpy as np

from data.standardizer import Standardizer


def test_standardizer_fit_transform():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = Standardizer().fit(X)
    Xz = scaler.transform(X)
    assert np.allclose(Xz.mean(axis=0), [0.0, 0.0])
    assert np.allclose(Xz.std(axis=0, ddof=0), [1.0, 1.0])


def test_standardizer_zero_std_handling():
    X = np.array([[1.0, 1.0], [1.0, 1.0]])
    scaler = Standardizer().fit(X)
    Xz = scaler.transform(X)
    assert np.allclose(Xz, np.zeros_like(X))
