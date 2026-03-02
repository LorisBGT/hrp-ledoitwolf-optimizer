"""
Walk-forward backtest engine with comprehensive performance analysis.
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
    transaction_cost : float   percentage cost per rebalance (default 0.1%)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        allocators: Optional[List[str]] = None,
        cov_estimators: Optional[List[str]] = None,
        train_window: int = 3 * 252,
        test_window: int = 252,
        rebalance_freq: str = 'monthly',
        transaction_cost: float = 0.001
    ) -> None:
        self.returns = returns
        self.allocators = allocators or ['hrp', 'markowitz', 'inverse_variance', 'equal_weight']
        self.cov_estimators = cov_estimators or ['empirical', 'ledoit_wolf']
        self.train_window = train_window
        self.test_window = test_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.weights_history_: Dict[str, pd.DataFrame] = {}

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
                weights_list = []
                for train, test in splits:
                    try:
                        w = self._get_weights(train, alloc_name, cov_est)
                        chunks.append((test * w).sum(axis=1))
                        weights_list.append(pd.DataFrame([w.values], columns=w.index, index=[test.index[0]]))
                    except Exception as e:
                        if verbose:
                            print(f'    skipped: {e}')

                if chunks:
                    out[key] = pd.concat(chunks).sort_index()
                    self.weights_history_[key] = pd.concat(weights_list).sort_index()

        return pd.DataFrame(out)

    @staticmethod
    def compute_metrics(results: pd.DataFrame, ann: int = 252, rf_rate: float = 0.0) -> pd.DataFrame:
        """Comprehensive performance metrics."""
        rows = {}
        for col in results.columns:
            r = results[col].dropna()
            mu = r.mean() * ann
            sigma = r.std() * np.sqrt(ann)
            sharpe = (mu - rf_rate) / sigma if sigma > 0 else np.nan

            cum = (1 + r).cumprod()
            max_dd = ((cum / cum.cummax()) - 1).min()
            calmar = mu / abs(max_dd) if max_dd < 0 else np.nan
            
            down = r[r < 0].std() * np.sqrt(ann)
            sortino = (mu - rf_rate) / down if down > 0 else np.nan

            skew = r.skew()
            kurt = r.kurtosis()
            
            positive_months = (r > 0).sum() / len(r) if len(r) > 0 else np.nan

            rows[col] = {
                'ann_return': f'{mu:.2%}',
                'ann_vol': f'{sigma:.2%}',
                'sharpe': f'{sharpe:.3f}',
                'sortino': f'{sortino:.3f}',
                'max_dd': f'{max_dd:.2%}',
                'calmar': f'{calmar:.3f}',
                'skewness': f'{skew:.3f}',
                'kurtosis': f'{kurt:.3f}',
                'hit_rate': f'{positive_months:.2%}',
            }
        return pd.DataFrame(rows).T

    def compute_weight_stability(self) -> pd.DataFrame:
        """Weight stability metrics: variance, turnover, concentration."""
        if not self.weights_history_:
            raise ValueError('run run_backtest() first')
        
        rows = {}
        for strat, weights_df in self.weights_history_.items():
            weight_var = weights_df.var(axis=0).mean()
            
            turnover_list = []
            for i in range(1, len(weights_df)):
                turnover_list.append(np.abs(weights_df.iloc[i].values - weights_df.iloc[i-1].values).sum())
            avg_turnover = np.mean(turnover_list) if turnover_list else 0
            
            herfindahl = (weights_df ** 2).sum(axis=1).mean()
            
            rows[strat] = {
                'avg_weight_variance': f'{weight_var:.4f}',
                'avg_turnover': f'{avg_turnover:.2%}',
                'avg_herfindahl': f'{herfindahl:.3f}',
                'avg_concentration': f'{1/herfindahl:.1f}',
            }
        return pd.DataFrame(rows).T

    def compute_rolling_sharpe(self, results: pd.DataFrame, window: int = 252, ann: int = 252) -> pd.DataFrame:
        """Rolling Sharpe ratio."""
        rolling = pd.DataFrame()
        for col in results.columns:
            r = results[col].dropna()
            roll_mean = r.rolling(window).mean() * ann
            roll_std = r.rolling(window).std() * np.sqrt(ann)
            rolling[col] = roll_mean / roll_std
        return rolling.dropna()

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

    def plot_rolling_sharpe(
        self,
        results: pd.DataFrame,
        window: int = 252,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        import matplotlib.pyplot as plt
        rolling_sharpe = self.compute_rolling_sharpe(results, window=window)
        ax = rolling_sharpe.plot(figsize=figsize, linewidth=1.5, alpha=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'rolling Sharpe ratio ({window}d window)')
        ax.set_ylabel('Sharpe')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_drawdown(
        self,
        results: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        for col in results.columns:
            cum = (1 + results[col]).cumprod()
            dd = (cum / cum.cummax() - 1) * 100
            ax.plot(dd.index, dd.values, label=col, linewidth=1.5, alpha=0.8)
        ax.set_title('drawdown (%)')
        ax.set_ylabel('drawdown')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
