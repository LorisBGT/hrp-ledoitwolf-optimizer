"""
Covariance estimation: sample, Ledoit-Wolf shrinkage, OAS.

Ledoit-Wolf shrinks toward a scaled identity matrix with analytically
optimal intensity — reduces condition number, stabilises downstream allocation.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from sklearn.covariance import empirical_covariance as _emp_cov


def empirical_covariance(returns: pd.DataFrame) -> np.ndarray:
    return _emp_cov(returns.values)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    assume_centered: bool = False
) -> Tuple[np.ndarray, float]:
    """Returns (shrunk covariance, shrinkage intensity alpha in [0,1])."""
    lw = LedoitWolf(assume_centered=assume_centered).fit(returns.values)
    return lw.covariance_, lw.shrinkage_


def oas_covariance(
    returns: pd.DataFrame,
    assume_centered: bool = False
) -> Tuple[np.ndarray, float]:
    """OAS — better convergence than LW under Gaussian assumption."""
    oas = OAS(assume_centered=assume_centered).fit(returns.values)
    return oas.covariance_, oas.shrinkage_


def compare_estimators(
    returns: pd.DataFrame,
    true_cov: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Condition number, eigenvalue range, and Frobenius error across estimators."""
    cov_emp = empirical_covariance(returns)
    cov_lw, a_lw = ledoit_wolf_covariance(returns)
    cov_oas, a_oas = oas_covariance(returns)

    rows = []
    for name, cov, alpha in [
        ('empirical', cov_emp, 0.0),
        ('ledoit_wolf', cov_lw, a_lw),
        ('oas', cov_oas, a_oas),
    ]:
        eig = np.linalg.eigvalsh(cov)
        row = {
            'estimator': name,
            'shrinkage': round(alpha, 4),
            'cond_number': round(np.linalg.cond(cov), 1),
            'min_eig': round(eig.min(), 6),
            'max_eig': round(eig.max(), 4),
        }
        if true_cov is not None:
            row['frob_error'] = round(np.linalg.norm(cov - true_cov, 'fro'), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)
