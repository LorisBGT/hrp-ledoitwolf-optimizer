"""
Hierarchical Risk Parity (López de Prado, 2016).

  1. Distance matrix from correlation: d_ij = sqrt(0.5 * (1 - rho_ij))
  2. Ward hierarchical clustering
  3. Quasi-diagonalization: reorder assets by dendrogram leaf order
  4. Recursive bisection: allocate capital inversely to cluster variance

No matrix inversion involved — that's the point.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def compute_distance_matrix(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(0.5 * (1 - rho_ij)), values in [0, 1]."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))


def hierarchical_clustering(dist: np.ndarray, method: str = 'ward') -> np.ndarray:
    return sch.linkage(squareform(dist, checks=False), method=method)


def quasi_diagonalization(linkage: np.ndarray) -> List[int]:
    return sch.leaves_list(linkage).tolist()


def _cluster_var(cov: np.ndarray, idx: List[int]) -> float:
    """Inverse-variance weighted variance of a cluster."""
    sub = cov[np.ix_(idx, idx)]
    inv = 1.0 / np.diag(sub)
    w = inv / inv.sum()
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, sorted_idx: List[int]) -> np.ndarray:
    """
    Recursively split the dendrogram, allocating capital inversely
    to cluster variance. Returns weights summing to 1.
    """
    n = len(sorted_idx)
    weights = np.ones(n)
    stack = [list(range(n))]

    while stack:
        cluster = stack.pop()
        if len(cluster) < 2:
            continue
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]

        v_left = _cluster_var(cov, [sorted_idx[i] for i in left])
        v_right = _cluster_var(cov, [sorted_idx[i] for i in right])

        alpha = v_right / (v_left + v_right)
        weights[left] *= alpha
        weights[right] *= (1.0 - alpha)

        if len(left) > 1:
            stack.append(left)
        if len(right) > 1:
            stack.append(right)

    return weights


class HRP:
    """
    HRP portfolio optimizer.

    hrp = HRP()
    weights = hrp.fit(returns, cov_estimator='ledoit_wolf')
    hrp.plot_dendrogram()
    """

    def __init__(self) -> None:
        self.weights_: Optional[pd.Series] = None
        self.linkage_: Optional[np.ndarray] = None
        self.sorted_idx_: Optional[List[int]] = None
        self.asset_names_: Optional[List[str]] = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[np.ndarray] = None,
        cov_estimator: str = 'ledoit_wolf'
    ) -> pd.Series:
        """
        Fit HRP and return portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
        cov : np.ndarray, optional  pre-computed covariance matrix
        cov_estimator : str         'empirical', 'ledoit_wolf', or 'oas'
        """
        from src.covariance import (
            empirical_covariance, ledoit_wolf_covariance, oas_covariance
        )

        self.asset_names_ = returns.columns.tolist()

        if cov is None:
            cov = {
                'empirical': lambda r: empirical_covariance(r),
                'ledoit_wolf': lambda r: ledoit_wolf_covariance(r)[0],
                'oas': lambda r: oas_covariance(r)[0],
            }[cov_estimator](returns)

        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        dist = compute_distance_matrix(corr)

        self.linkage_ = hierarchical_clustering(dist)
        self.sorted_idx_ = quasi_diagonalization(self.linkage_)

        raw = recursive_bisection(cov, self.sorted_idx_)
        self.weights_ = pd.Series(raw, index=self.asset_names_)
        return self.weights_

    def get_weights(self) -> Dict[str, float]:
        if self.weights_ is None:
            raise ValueError('call fit() first')
        return self.weights_.to_dict()

    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> None:
        if self.linkage_ is None:
            raise ValueError('call fit() first')
        fig, ax = plt.subplots(figsize=figsize)
        sch.dendrogram(self.linkage_, labels=self.asset_names_,
                       ax=ax, leaf_font_size=10, leaf_rotation=45)
        ax.set_title('HRP — hierarchical clustering (Ward)')
        ax.set_ylabel('distance')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
