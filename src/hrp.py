"""
Hierarchical Risk Parity (HRP) portfolio optimizer.

Implements the full HRP algorithm from López de Prado (2016):
  1. Compute correlation-based distance matrix
  2. Hierarchical clustering (Ward linkage)
  3. Quasi-diagonalization (seriation)
  4. Recursive bisection weight allocation

References
----------
López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out of Sample". Journal of Portfolio Management, 42(4), 59-69.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def compute_distance_matrix(correlation: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    d_ij = sqrt(0.5 · (1 - ρ_ij))

    Parameters
    ----------
    correlation : np.ndarray  shape (n, n)

    Returns
    -------
    np.ndarray  shape (n, n)
        Distance matrix with values in [0, 1].

    Notes
    -----
    - ρ = +1  →  d = 0   (identical assets)
    - ρ =  0  →  d = √½  (uncorrelated)
    - ρ = -1  →  d = 1   (perfectly anti-correlated)
    """
    return np.sqrt(0.5 * (1.0 - correlation))


def hierarchical_clustering(
    distance_matrix: np.ndarray,
    method: str = 'ward'
) -> np.ndarray:
    """
    Perform hierarchical clustering on a distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray  shape (n, n)
    method : str
        Linkage algorithm: 'ward', 'single', 'complete', 'average'.

    Returns
    -------
    np.ndarray
        Scipy linkage matrix, shape (n-1, 4).
    """
    condensed = squareform(distance_matrix, checks=False)
    return sch.linkage(condensed, method=method)


def quasi_diagonalization(linkage: np.ndarray) -> List[int]:
    """
    Reorder assets by dendrogram leaves (seriation).

    Parameters
    ----------
    linkage : np.ndarray
        Linkage matrix from hierarchical_clustering.

    Returns
    -------
    list of int
        Sorted asset indices grouping similar assets together.
    """
    return sch.leaves_list(linkage).tolist()


def _cluster_variance(cov: np.ndarray, indices: List[int]) -> float:
    """
    Compute cluster variance using inverse-variance weights.

    Parameters
    ----------
    cov : np.ndarray
        Full covariance matrix.
    indices : list of int
        Asset indices in this cluster.

    Returns
    -------
    float
        Portfolio variance of cluster.
    """
    sub_cov = cov[np.ix_(indices, indices)]
    inv_var = 1.0 / np.diag(sub_cov)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def recursive_bisection(
    cov: np.ndarray,
    sorted_indices: List[int]
) -> np.ndarray:
    """
    Allocate weights via recursive bisection (core of HRP).

    At each level, the cluster is split in two. Capital is allocated
    inversely proportional to each sub-cluster's variance.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    sorted_indices : list of int
        Asset indices in dendrogram order.

    Returns
    -------
    np.ndarray
        Portfolio weights, shape (n_assets,), summing to 1.

    References
    ----------
    López de Prado (2016), Algorithm 2.
    """
    n = len(sorted_indices)
    weights = np.ones(n)
    clusters = [list(range(n))]  # work with positional indices

    while clusters:
        # Process only clusters with more than 1 element
        new_clusters = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            # Translate positional → original covariance indices
            left_orig = [sorted_indices[i] for i in left]
            right_orig = [sorted_indices[i] for i in right]

            var_left = _cluster_variance(cov, left_orig)
            var_right = _cluster_variance(cov, right_orig)

            # Fraction allocated to left sub-cluster
            alpha = var_right / (var_left + var_right)

            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    return weights


class HRP:
    """
    Hierarchical Risk Parity portfolio optimizer.

    Encapsulates the full HRP pipeline: covariance estimation →
    distance matrix → hierarchical clustering → quasi-diagonalization →
    recursive bisection allocation.

    Attributes
    ----------
    weights_ : pd.Series
        Optimized weights (available after fit).
    linkage_ : np.ndarray
        Scipy linkage matrix.
    sorted_indices_ : list of int
        Asset order from dendrogram leaves.
    asset_names_ : list of str
        Asset names.

    Examples
    --------
    >>> hrp = HRP()
    >>> weights = hrp.fit(returns, cov_estimator='ledoit_wolf')
    >>> hrp.plot_dendrogram()
    """

    def __init__(self) -> None:
        self.weights_: Optional[pd.Series] = None
        self.linkage_: Optional[np.ndarray] = None
        self.sorted_indices_: Optional[List[int]] = None
        self.asset_names_: Optional[List[str]] = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[np.ndarray] = None,
        cov_estimator: str = 'ledoit_wolf'
    ) -> pd.Series:
        """
        Compute HRP portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns, shape (n_samples, n_assets).
        cov : np.ndarray, optional
            Pre-computed covariance matrix. Estimated if None.
        cov_estimator : str
            'empirical', 'ledoit_wolf', or 'oas'.

        Returns
        -------
        pd.Series
            Weights indexed by asset name.
        """
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )

        self.asset_names_ = returns.columns.tolist()

        if cov is None:
            estimators = {
                'empirical': lambda r: empirical_covariance(r),
                'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                'oas': lambda r: oas_covariance(r)[0],
            }
            if cov_estimator not in estimators:
                raise ValueError(
                    f"cov_estimator must be one of {list(estimators.keys())}"
                )
            cov = estimators[cov_estimator](returns)

        # Correlation-based distance
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        distance = compute_distance_matrix(corr)

        # Clustering & seriation
        self.linkage_ = hierarchical_clustering(distance, method='ward')
        self.sorted_indices_ = quasi_diagonalization(self.linkage_)

        # Recursive bisection
        raw_weights = recursive_bisection(cov, self.sorted_indices_)
        self.weights_ = pd.Series(raw_weights, index=self.asset_names_)

        return self.weights_

    def get_weights(self) -> Dict[str, float]:
        """Return weights as {ticker: weight} dictionary."""
        if self.weights_ is None:
            raise ValueError("Call fit() before get_weights()")
        return self.weights_.to_dict()

    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (12, 5),
        title: str = 'HRP Hierarchical Clustering Dendrogram',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the hierarchical clustering dendrogram.

        Parameters
        ----------
        figsize : tuple
        title : str
        save_path : str, optional
            If provided, saves figure to this path.
        """
        if self.linkage_ is None:
            raise ValueError("Call fit() before plot_dendrogram()")

        fig, ax = plt.subplots(figsize=figsize)
        sch.dendrogram(
            self.linkage_,
            labels=self.asset_names_,
            ax=ax,
            leaf_font_size=10,
            leaf_rotation=45
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=11)
        ax.set_ylabel('Distance', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    from src.data_loader import download_price_data, compute_returns

    tickers = ['SPY', 'TLT', 'GLD', 'VNQ', 'DBC', 'EFA', 'EEM']
    prices = download_price_data(tickers, '2020-01-01', '2024-01-01')
    returns = compute_returns(prices)

    hrp = HRP()
    weights = hrp.fit(returns, cov_estimator='ledoit_wolf')

    print("\nHRP Weights (Ledoit-Wolf):")
    print(weights.map('{:.2%}'.format).to_string())
    print(f"\nSum of weights: {weights.sum():.6f}")
    hrp.plot_dendrogram()
