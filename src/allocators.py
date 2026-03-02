"""
Alternative portfolio allocation methods for comparison with HRP.

Implements Markowitz mean-variance, inverse variance, and equal-weight
portfolios using a common BaseAllocator interface.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    from scipy.optimize import minimize


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """Equal-weight (1/N) allocation."""
    return np.ones(n_assets) / n_assets


def inverse_variance_portfolio(cov: np.ndarray) -> np.ndarray:
    """
    Inverse-variance portfolio: w_i ∝ 1/σ_i².

    Parameters
    ----------
    cov : np.ndarray  shape (n, n)

    Returns
    -------
    np.ndarray  shape (n,)  — weights summing to 1.
    """
    inv_var = 1.0 / np.diag(cov)
    return inv_var / inv_var.sum()


def markowitz_min_variance(
    cov: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Global minimum-variance portfolio via quadratic programming.

    Solves:
        min   w' Σ w
        s.t.  Σ wᵢ = 1,   min_weight ≤ wᵢ ≤ max_weight

    Parameters
    ----------
    cov : np.ndarray
    min_weight : float
    max_weight : float

    Returns
    -------
    np.ndarray  shape (n,)

    Notes
    -----
    Uses CVXPY if available, otherwise scipy.optimize fallback.

    References
    ----------
    Markowitz, H. (1952). Journal of Finance, 7(1), 77-91.
    """
    n = cov.shape[0]

    if CVXPY_AVAILABLE:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise ValueError(f"Optimization failed: {prob.status}")
        return np.array(w.value)

    else:
        # Scipy fallback
        def port_var(w):
            return w @ cov @ w

        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(min_weight, max_weight)] * n
        w0 = np.ones(n) / n
        result = minimize(port_var, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError(f"Scipy optimization failed: {result.message}")
        return result.x


# ─── Allocator classes (unified interface) ──────────────────────────────────

class BaseAllocator:
    """Abstract base class for portfolio allocators."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.weights_: Optional[pd.Series] = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[np.ndarray] = None,
        mean_returns: Optional[np.ndarray] = None
    ) -> pd.Series:
        raise NotImplementedError

    def get_weights(self) -> Dict[str, float]:
        if self.weights_ is None:
            raise ValueError("Call fit() first")
        return self.weights_.to_dict()

    def get_name(self) -> str:
        return self.name


class EqualWeightAllocator(BaseAllocator):
    """Equal-weight (1/N) allocator."""

    def __init__(self) -> None:
        super().__init__("Equal Weight")

    def fit(self, returns, cov=None, mean_returns=None):
        names = returns.columns.tolist()
        self.weights_ = pd.Series(equal_weight_portfolio(len(names)), index=names)
        return self.weights_


class InverseVarianceAllocator(BaseAllocator):
    """Inverse-variance (simplified risk parity) allocator."""

    def __init__(self, cov_estimator: str = 'empirical') -> None:
        super().__init__(f"Inverse Variance ({cov_estimator})")
        self.cov_estimator = cov_estimator

    def fit(self, returns, cov=None, mean_returns=None):
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )
        if cov is None:
            cov = {'empirical': empirical_covariance,
                   'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                   'oas': lambda r: oas_covariance(r)[0]
                   }[self.cov_estimator](returns)

        names = returns.columns.tolist()
        self.weights_ = pd.Series(inverse_variance_portfolio(cov), index=names)
        return self.weights_


class MarkowitzMinVarianceAllocator(BaseAllocator):
    """Markowitz global minimum-variance allocator."""

    def __init__(
        self,
        cov_estimator: str = 'empirical',
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> None:
        super().__init__(f"Markowitz ({cov_estimator})")
        self.cov_estimator = cov_estimator
        self.min_weight = min_weight
        self.max_weight = max_weight

    def fit(self, returns, cov=None, mean_returns=None):
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )
        if cov is None:
            cov = {'empirical': empirical_covariance,
                   'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                   'oas': lambda r: oas_covariance(r)[0]
                   }[self.cov_estimator](returns)

        names = returns.columns.tolist()
        self.weights_ = pd.Series(
            markowitz_min_variance(cov, self.min_weight, self.max_weight),
            index=names
        )
        return self.weights_
