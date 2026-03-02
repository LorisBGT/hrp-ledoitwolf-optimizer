"""
Baseline allocators for comparison with HRP.

Equal weight, inverse variance, Markowitz min-variance.
Markowitz is included to show what happens with and without Ledoit-Wolf
shrinkage — better covariance, still inverts it.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _CVXPY = True
except ImportError:
    from scipy.optimize import minimize as _sp_minimize
    _CVXPY = False


def equal_weight_portfolio(n: int) -> np.ndarray:
    return np.ones(n) / n


def inverse_variance_portfolio(cov: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.diag(cov)
    return inv / inv.sum()


def markowitz_min_variance(
    cov: np.ndarray,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> np.ndarray:
    """Global minimum variance. Uses CVXPY if available, scipy SLSQP otherwise."""
    n = cov.shape[0]

    if _CVXPY:
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cov)),
            [cp.sum(w) == 1, w >= min_w, w <= max_w]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f'CVXPY failed: {prob.status}')
        return np.array(w.value)
    else:
        res = _sp_minimize(
            fun=lambda w: w @ cov @ w,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=[(min_w, max_w)] * n,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )
        if not res.success:
            raise RuntimeError(f'scipy failed: {res.message}')
        return res.x


class BaseAllocator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.weights_: Optional[pd.Series] = None

    def fit(self, returns, cov=None, mean_returns=None) -> pd.Series:
        raise NotImplementedError

    def get_weights(self) -> Dict[str, float]:
        if self.weights_ is None:
            raise ValueError('call fit() first')
        return self.weights_.to_dict()

    def get_name(self) -> str:
        return self.name


class EqualWeightAllocator(BaseAllocator):
    def __init__(self):
        super().__init__('equal_weight')

    def fit(self, returns, cov=None, mean_returns=None):
        names = returns.columns.tolist()
        self.weights_ = pd.Series(equal_weight_portfolio(len(names)), index=names)
        return self.weights_


class InverseVarianceAllocator(BaseAllocator):
    def __init__(self, cov_estimator: str = 'empirical'):
        super().__init__(f'inv_var_{cov_estimator}')
        self.cov_estimator = cov_estimator

    def fit(self, returns, cov=None, mean_returns=None):
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )
        if cov is None:
            cov = {
                'empirical': empirical_covariance,
                'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                'oas': lambda r: oas_covariance(r)[0],
            }[self.cov_estimator](returns)
        names = returns.columns.tolist()
        self.weights_ = pd.Series(inverse_variance_portfolio(cov), index=names)
        return self.weights_


class MarkowitzMinVarianceAllocator(BaseAllocator):
    def __init__(self, cov_estimator: str = 'empirical',
                 min_w: float = 0.0, max_w: float = 1.0):
        super().__init__(f'markowitz_{cov_estimator}')
        self.cov_estimator = cov_estimator
        self.min_w = min_w
        self.max_w = max_w

    def fit(self, returns, cov=None, mean_returns=None):
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )
        if cov is None:
            cov = {
                'empirical': empirical_covariance,
                'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                'oas': lambda r: oas_covariance(r)[0],
            }[self.cov_estimator](returns)
        names = returns.columns.tolist()
        self.weights_ = pd.Series(
            markowitz_min_variance(cov, self.min_w, self.max_w), index=names
        )
        return self.weights_
