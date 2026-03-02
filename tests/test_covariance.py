import pytest
import numpy as np
import pandas as pd
from src.covariance import (
    empirical_covariance, ledoit_wolf_covariance,
    oas_covariance, compare_estimators,
    compute_correlation_from_covariance,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    n, T = 8, 400
    A = rng.standard_normal((n, n))
    true_cov = A @ A.T / n + np.eye(n) * 0.1
    X = rng.multivariate_normal(np.zeros(n), true_cov, T)
    returns = pd.DataFrame(X, columns=[f'A{i}' for i in range(n)])
    return returns, true_cov


def test_empirical_shape(sample_data):
    r, _ = sample_data
    assert empirical_covariance(r).shape == (r.shape[1], r.shape[1])


def test_empirical_symmetric(sample_data):
    r, _ = sample_data
    cov = empirical_covariance(r)
    assert np.allclose(cov, cov.T)


def test_lw_shrinkage_range(sample_data):
    r, _ = sample_data
    _, alpha = ledoit_wolf_covariance(r)
    assert 0.0 <= alpha <= 1.0


def test_lw_lower_condition_number(sample_data):
    r, _ = sample_data
    cov_emp = empirical_covariance(r)
    cov_lw, _ = ledoit_wolf_covariance(r)
    assert np.linalg.cond(cov_lw) <= np.linalg.cond(cov_emp)


def test_oas_shrinkage_range(sample_data):
    r, _ = sample_data
    _, alpha = oas_covariance(r)
    assert 0.0 <= alpha <= 1.0


def test_compare_returns_df(sample_data):
    r, true_cov = sample_data
    df = compare_estimators(r, true_cov=true_cov)
    assert isinstance(df, pd.DataFrame) and len(df) == 3
    assert 'frob_error' in df.columns


def test_correlation_diagonal(sample_data):
    r, _ = sample_data
    corr = compute_correlation_from_covariance(empirical_covariance(r))
    assert np.allclose(np.diag(corr), 1.0)
