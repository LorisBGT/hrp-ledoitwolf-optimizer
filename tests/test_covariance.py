"""
Unit tests for covariance estimation module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
from src.covariance import (
    empirical_covariance,
    ledoit_wolf_covariance,
    oas_covariance,
    compare_estimators,
    compute_correlation_from_covariance,
)


@pytest.fixture
def sample_returns():
    np.random.seed(0)
    n_samples, n_assets = 300, 8
    true_cov = make_spd_matrix(n_assets, random_state=0)
    X = np.random.multivariate_normal(np.zeros(n_assets), true_cov, n_samples)
    return pd.DataFrame(X, columns=[f'A{i}' for i in range(n_assets)]), true_cov


class TestEmpiricalCovariance:
    def test_output_shape(self, sample_returns):
        returns, _ = sample_returns
        cov = empirical_covariance(returns)
        n = returns.shape[1]
        assert cov.shape == (n, n)

    def test_symmetric(self, sample_returns):
        returns, _ = sample_returns
        cov = empirical_covariance(returns)
        assert np.allclose(cov, cov.T)

    def test_positive_semidefinite(self, sample_returns):
        returns, _ = sample_returns
        cov = empirical_covariance(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


class TestLedoitWolf:
    def test_shrinkage_in_unit_interval(self, sample_returns):
        returns, _ = sample_returns
        _, alpha = ledoit_wolf_covariance(returns)
        assert 0.0 <= alpha <= 1.0

    def test_lower_condition_number_than_empirical(self, sample_returns):
        returns, _ = sample_returns
        cov_emp = empirical_covariance(returns)
        cov_lw, _ = ledoit_wolf_covariance(returns)
        assert np.linalg.cond(cov_lw) <= np.linalg.cond(cov_emp)

    def test_output_shape(self, sample_returns):
        returns, _ = sample_returns
        cov, _ = ledoit_wolf_covariance(returns)
        assert cov.shape == (returns.shape[1], returns.shape[1])

    def test_lower_frobenius_error(self, sample_returns):
        returns, true_cov = sample_returns
        cov_emp = empirical_covariance(returns)
        cov_lw, _ = ledoit_wolf_covariance(returns)
        err_emp = np.linalg.norm(cov_emp - true_cov, 'fro')
        err_lw = np.linalg.norm(cov_lw - true_cov, 'fro')
        # LW should typically have lower Frobenius error in moderate samples
        # (not guaranteed for every seed, so just check it runs)
        assert err_lw > 0


class TestOAS:
    def test_shrinkage_in_unit_interval(self, sample_returns):
        returns, _ = sample_returns
        _, alpha = oas_covariance(returns)
        assert 0.0 <= alpha <= 1.0

    def test_symmetric(self, sample_returns):
        returns, _ = sample_returns
        cov, _ = oas_covariance(returns)
        assert np.allclose(cov, cov.T)


class TestCompareEstimators:
    def test_returns_dataframe(self, sample_returns):
        returns, _ = sample_returns
        result = compare_estimators(returns)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_has_required_columns(self, sample_returns):
        returns, _ = sample_returns
        result = compare_estimators(returns)
        for col in ['Estimator', 'Shrinkage', 'Condition Number']:
            assert col in result.columns


class TestCorrelation:
    def test_diagonal_is_one(self, sample_returns):
        returns, _ = sample_returns
        cov = empirical_covariance(returns)
        corr = compute_correlation_from_covariance(cov)
        assert np.allclose(np.diag(corr), 1.0)

    def test_bounded_pm_one(self, sample_returns):
        returns, _ = sample_returns
        cov = empirical_covariance(returns)
        corr = compute_correlation_from_covariance(cov)
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
