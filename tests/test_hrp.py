import pytest
import numpy as np
import pandas as pd
from src.hrp import compute_distance_matrix, recursive_bisection, HRP


@pytest.fixture
def rng_returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    data = rng.standard_normal((300, 5)) * 0.01
    return pd.DataFrame(data, index=dates,
                        columns=['SPY', 'TLT', 'GLD', 'VNQ', 'DBC'])


def test_distance_perfect_corr():
    corr = np.array([[1., 1.], [1., 1.]])
    assert np.allclose(compute_distance_matrix(corr)[0, 1], 0.0)


def test_distance_zero_corr():
    corr = np.array([[1., 0.], [0., 1.]])
    assert np.allclose(compute_distance_matrix(corr)[0, 1], np.sqrt(0.5))


def test_distance_negative_corr():
    corr = np.array([[1., -1.], [-1., 1.]])
    assert np.allclose(compute_distance_matrix(corr)[0, 1], 1.0)


def test_bisection_sums_to_one():
    w = recursive_bisection(np.eye(4) * 0.01, list(range(4)))
    assert np.isclose(w.sum(), 1.0)


def test_bisection_non_negative():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    cov = A @ A.T / 100
    assert np.all(recursive_bisection(cov, list(range(6))) >= 0)


def test_hrp_weights_sum_one(rng_returns):
    assert np.isclose(HRP().fit(rng_returns).sum(), 1.0, atol=1e-6)


def test_hrp_non_negative(rng_returns):
    assert np.all(HRP().fit(rng_returns) >= 0)


def test_hrp_correct_index(rng_returns):
    w = HRP().fit(rng_returns)
    assert list(w.index) == list(rng_returns.columns)


@pytest.mark.parametrize('estimator', ['empirical', 'ledoit_wolf', 'oas'])
def test_hrp_all_estimators(rng_returns, estimator):
    w = HRP().fit(rng_returns, cov_estimator=estimator)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_get_weights_before_fit():
    with pytest.raises(ValueError):
        HRP().get_weights()
