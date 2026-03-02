"""
Baseline allocators for comparison with HRP.

Equal weight, inverse variance, Markowitz min-variance, Risk Parity.
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
    allow_short: bool = False,
    max_leverage: float = 1.0,
) -> np.ndarray:
    """
    Global minimum variance.

    allow_short : if False, enforces w >= 0 (long-only)
    max_leverage : sum of absolute weights upper bound (1.0 = fully invested, no leverage)
    Uses CVXPY if available, scipy SLSQP otherwise.
    """
    n = cov.shape[0]
    effective_min = min_w if allow_short else max(0.0, min_w)

    if _CVXPY:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w <= max_w]
        if not allow_short:
            constraints.append(w >= 0)
        else:
            constraints.append(w >= min_w)
        if max_leverage < 10:
            constraints.append(cp.norm1(w) <= max_leverage)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f'CVXPY failed: {prob.status}')
        return np.array(w.value)
    else:
        from scipy.optimize import minimize as _sp_minimize
        bounds = [(effective_min, max_w)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        if max_leverage < 10:
            constraints.append({'type': 'ineq', 'fun': lambda w: max_leverage - np.abs(w).sum()})
        res = _sp_minimize(
            fun=lambda w: w @ cov @ w,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if not res.success:
            raise RuntimeError(f'scipy failed: {res.message}')
        return res.x


def risk_parity_portfolio(
    cov: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 500
) -> np.ndarray:
    """
    Equal Risk Contribution (ERC) portfolio.

    Each asset contributes equally to total portfolio variance.
    Solved iteratively via Spinu (2013) / Maillard et al. (2010).
    """
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ cov @ w)
        marginal = cov @ w / sigma_p
        rc = w * marginal
        target = sigma_p / n

        w_new = w * (target / rc)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w_new


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
    def __init__(
        self,
        cov_estimator: str = 'empirical',
        min_w: float = 0.0,
        max_w: float = 1.0,
        allow_short: bool = False,
        max_leverage: float = 1.0,
    ):
        name = f'markowitz_{cov_estimator}'
        if allow_short:
            name += '_short'
        if max_leverage != 1.0:
            name += f'_lev{max_leverage}'
        super().__init__(name)
        self.cov_estimator = cov_estimator
        self.min_w = min_w
        self.max_w = max_w
        self.allow_short = allow_short
        self.max_leverage = max_leverage

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
            markowitz_min_variance(
                cov, self.min_w, self.max_w,
                allow_short=self.allow_short,
                max_leverage=self.max_leverage
            ),
            index=names
        )
        return self.weights_


class RiskParityAllocator(BaseAllocator):
    def __init__(self, cov_estimator: str = 'empirical'):
        super().__init__(f'risk_parity_{cov_estimator}')
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
        self.weights_ = pd.Series(risk_parity_portfolio(cov), index=names)
        return self.weights_
