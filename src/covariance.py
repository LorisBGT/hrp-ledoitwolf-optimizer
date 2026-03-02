"""
Covariance estimation module.

Implements empirical covariance, Ledoit-Wolf shrinkage, and OAS
for robust portfolio covariance estimation.

References
----------
Ledoit & Wolf (2004). "Honey, I Shrunk the Sample Covariance Matrix".
Journal of Portfolio Management, 30(4), 110-119.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from sklearn.covariance import empirical_covariance as _sklearn_emp


def empirical_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute standard sample covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns, shape (n_samples, n_assets).

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (n_assets, n_assets).

    Notes
    -----
    Σ = (1/T) X'X   (centered)
    """
    return _sklearn_emp(returns.values)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    assume_centered: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Compute Ledoit-Wolf shrunk covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    assume_centered : bool
        If True, skip centering.

    Returns
    -------
    cov : np.ndarray
        Shrunk covariance matrix.
    shrinkage : float
        Optimal shrinkage coefficient α ∈ [0, 1].

    Notes
    -----
    Σ_LW = (1 - α)·Σ_emp + α·(trace(Σ)/n)·I
    The coefficient α minimises the Frobenius-norm MSE.

    References
    ----------
    Ledoit & Wolf (2004). SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=433840
    """
    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(returns.values)
    return lw.covariance_, lw.shrinkage_


def oas_covariance(
    returns: pd.DataFrame,
    assume_centered: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Compute Oracle Approximating Shrinkage (OAS) covariance.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    assume_centered : bool
        If True, skip centering.

    Returns
    -------
    cov : np.ndarray
        OAS shrunk covariance matrix.
    shrinkage : float
        OAS shrinkage coefficient.

    References
    ----------
    Chen et al. (2010). "Shrinkage Algorithms for MMSE Covariance Estimation".
    """
    oas = OAS(assume_centered=assume_centered)
    oas.fit(returns.values)
    return oas.covariance_, oas.shrinkage_


def compare_estimators(
    returns: pd.DataFrame,
    true_cov: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare covariance estimators on key numerical metrics.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    true_cov : np.ndarray, optional
        True covariance (for simulation studies — computes Frobenius error).

    Returns
    -------
    pd.DataFrame
        Metrics table: shrinkage, condition number, min/max eigenvalue,
        Frobenius error (if true_cov provided).
    """
    cov_emp = empirical_covariance(returns)
    cov_lw, alpha_lw = ledoit_wolf_covariance(returns)
    cov_oas, alpha_oas = oas_covariance(returns)

    rows = []
    for name, cov, alpha in [
        ('Empirical', cov_emp, 0.0),
        ('Ledoit-Wolf', cov_lw, alpha_lw),
        ('OAS', cov_oas, alpha_oas),
    ]:
        eig = np.linalg.eigvalsh(cov)
        row = {
            'Estimator': name,
            'Shrinkage': alpha,
            'Condition Number': np.linalg.cond(cov),
            'Min Eigenvalue': eig.min(),
            'Max Eigenvalue': eig.max(),
        }
        if true_cov is not None:
            row['Frobenius Error'] = np.linalg.norm(cov - true_cov, 'fro')
        rows.append(row)

    return pd.DataFrame(rows)


def compute_correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


if __name__ == "__main__":
    np.random.seed(42)
    n, T = 10, 500
    true_cov = np.eye(n) + 0.5 * np.ones((n, n))
    X = np.random.multivariate_normal(np.zeros(n), true_cov, T)
    returns = pd.DataFrame(X, columns=[f'A{i}' for i in range(n)])

    print(compare_estimators(returns, true_cov=true_cov).to_string(index=False))
