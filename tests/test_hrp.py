"""
Unit tests for HRP module.
"""

import pytest
import numpy as np
import pandas as pd
from src.hrp import (
    compute_distance_matrix,
    hierarchical_clustering,
    quasi_diagonalization,
    recursive_bisection,
    HRP,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    data = np.random.randn(300, 5) * 0.01
    return pd.DataFrame(data, index=dates,
                        columns=['SPY', 'TLT', 'GLD', 'VNQ', 'DBC'])


class TestDistanceMatrix:
    def test_perfect_correlation(self):
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        dist = compute_distance_matrix(corr)
        assert np.allclose(dist[0, 1], 0.0)

    def test_zero_correlation(self):
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        dist = compute_distance_matrix(corr)
        assert np.allclose(dist[0, 1], np.sqrt(0.5))

    def test_negative_correlation(self):
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        dist = compute_distance_matrix(corr)
        assert np.allclose(dist[0, 1], 1.0)

    def test_values_in_zero_one(self):
        np.random.seed(0)
        corr = np.clip(np.random.randn(5, 5) * 0.5, -1, 1)
        np.fill_diagonal(corr, 1.0)
        dist = compute_distance_matrix(corr)
        assert np.all(dist >= 0)
        assert np.all(dist <= 1.0 + 1e-10)


class TestClustering:
    def test_linkage_shape(self):
        n = 6
        dist = np.random.rand(n, n)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        linkage = hierarchical_clustering(dist)
        assert linkage.shape == (n - 1, 4)

    def test_sorted_indices_length(self):
        n = 7
        dist = np.random.rand(n, n)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        linkage = hierarchical_clustering(dist)
        idx = quasi_diagonalization(linkage)
        assert len(idx) == n
        assert set(idx) == set(range(n))


class TestRecursiveBisection:
    def test_weights_sum_to_one(self):
        n = 4
        cov = np.eye(n) * 0.01
        weights = recursive_bisection(cov, list(range(n)))
        assert np.isclose(weights.sum(), 1.0)

    def test_all_non_negative(self):
        n = 5
        cov = np.eye(n) * 0.01 + np.ones((n, n)) * 0.001
        weights = recursive_bisection(cov, list(range(n)))
        assert np.all(weights >= 0)

    def test_single_asset(self):
        cov = np.array([[0.01]])
        weights = recursive_bisection(cov, [0])
        assert np.isclose(weights[0], 1.0)


class TestHRPClass:
    def test_weights_sum_to_one(self, sample_returns):
        hrp = HRP()
        weights = hrp.fit(sample_returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_all_weights_non_negative(self, sample_returns):
        hrp = HRP()
        weights = hrp.fit(sample_returns)
        assert np.all(weights >= 0)

    def test_returns_series_with_correct_index(self, sample_returns):
        hrp = HRP()
        weights = hrp.fit(sample_returns)
        assert isinstance(weights, pd.Series)
        assert list(weights.index) == list(sample_returns.columns)

    def test_all_cov_estimators(self, sample_returns):
        for estimator in ['empirical', 'ledoit_wolf', 'oas']:
            hrp = HRP()
            w = hrp.fit(sample_returns, cov_estimator=estimator)
            assert np.isclose(w.sum(), 1.0, atol=1e-6)

    def test_get_weights_before_fit_raises(self):
        with pytest.raises(ValueError, match='fit'):
            HRP().get_weights()

    def test_get_weights_dict(self, sample_returns):
        hrp = HRP()
        hrp.fit(sample_returns)
        d = hrp.get_weights()
        assert isinstance(d, dict)
        assert len(d) == len(sample_returns.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
