"""
Walk-forward backtest engine.

For each window: fit on training data, apply weights to test period.
Runs every (allocator x covariance estimator) combination.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.data_loader import split_data_rolling
from src.hrp import HRP
from src.allocators import (
    EqualWeightAllocator,
    InverseVarianceAllocator,
    MarkowitzMinVarianceAllocator,
)


class Backtester:
    """
    Parameters
    ----------
    returns : pd.DataFrame
    allocators : list of str   'hrp', 'markowitz', 'inverse_variance', 'equal_weight'
    cov_estimators : list of str   'empirical', 'ledoit_wolf', 'oas'
    train_window : int   default 3*252
    test_window : int    default 252
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        allocators: Optional[List[str]] = None,
        cov_estimators: Optional[List[str]] = None,
        train_window: int = 3 * 252,
        test_window: int = 252,
        rebalance_freq: str = 'monthly'
    ) -> None:
        self.returns = returns
        self.allocators = allocators or ['hrp', 'markowitz', 'inverse_variance', 'equal_weight']
        self.cov_estimators = cov_estimators or ['empirical', 'ledoit_wolf']
        self.train_window = train_window
        self.test_window = test_window
        self.rebalance_freq = rebalance_freq

    def _make_allocator(self, name: str, cov_est: str):
        if name == 'equal_weight':
            return EqualWeightAllocator()
        if name == 'inverse_variance':
            return InverseVarianceAllocator(cov_estimator=cov_est)
        if name == 'markowitz':
            return MarkowitzMinVarianceAllocator(cov_estimator=cov_est)
        if name == 'hrp':
            return HRP()
        raise ValueError(f'unknown allocator: {name}')

    def _get_weights(self, train: pd.DataFrame, name: str, cov_est: str) -> pd.Series:
        alloc = self._make_allocator(name, cov_est)
        if name == 'hrp':
            return alloc.fit(train, cov_estimator=cov_est)
        return alloc.fit(train)

    def run_backtest(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the full walk-forward backtest.
        Returns a DataFrame of daily returns, one column per strategy.
        """
        splits = split_data_rolling(self.returns, self.train_window, self.test_window)
        out: Dict[str, List[pd.Series]] = {}

        for alloc_name in self.allocators:
            for cov_est in self.cov_estimators:
                if alloc_name == 'equal_weight' and cov_est != self.cov_estimators[0]:
                    continue

                key = alloc_name if alloc_name == 'equal_weight' else f'{alloc_name}_{cov_est}'
                if verbose:
                    print(f'  {key}')

                chunks = []
                for train, test in splits:
                    try:
                        w = self._get_weights(train, alloc_name, cov_est)
                        chunks.append((test * w).sum(axis=1))
                    except Exception as e:
                        if verbose:
                            print(f'    skipped: {e}')

                if chunks:
                    out[key] = pd.concat(chunks).sort_index()

        return pd.DataFrame(out)

    @staticmethod
    def compute_metrics(results: pd.DataFrame, ann: int = 252) -> pd.DataFrame:
        """Annualised performance metrics: return, vol, Sharpe, max drawdown, Calmar, Sortino."""
        rows = {}
        for col in results.columns:
            r = results[col].dropna()
            mu = r.mean() * ann
            sigma = r.std() * np.sqrt(ann)
            sharpe = mu / sigma if sigma > 0 else np.nan

            cum = (1 + r).cumprod()
            max_dd = ((cum / cum.cummax()) - 1).min()
            calmar = mu / abs(max_dd) if max_dd < 0 else np.nan
            down = r[r < 0].std() * np.sqrt(ann)
            sortino = mu / down if down > 0 else np.nan

            rows[col] = {
                'ann_return': f'{mu:.2%}',
                'ann_vol': f'{sigma:.2%}',
                'sharpe': f'{sharpe:.3f}',
                'max_dd': f'{max_dd:.2%}',
                'calmar': f'{calmar:.3f}',
                'sortino': f'{sortino:.3f}',
            }
        return pd.DataFrame(rows).T

    def plot_cumulative_returns(
        self,
        results: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        import matplotlib.pyplot as plt
        ax = (1 + results).cumprod().plot(figsize=figsize, linewidth=1.5)
        ax.set_title('cumulative returns (out-of-sample)')
        ax.set_ylabel('growth of $1')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
