"""
Rolling-window backtesting engine.

Supports pluggable allocators and covariance estimators.
Computes performance metrics: Sharpe, volatility, max drawdown,
Calmar, Sortino, turnover, and HHI concentration.
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
    Walk-forward backtesting engine.

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily return series.
    allocators : list of str
        Allocation methods: 'hrp', 'markowitz', 'inverse_variance', 'equal_weight'.
    cov_estimators : list of str
        Covariance estimators: 'empirical', 'ledoit_wolf', 'oas'.
    train_window : int
        Training window in days (default 3*252).
    test_window : int
        Test / rebalance window in days (default 252).
    rebalance_freq : str
        'monthly' or 'annual' — controls within-test-period rebalancing.

    Examples
    --------
    >>> bt = Backtester(returns, allocators=['hrp', 'equal_weight'],
    ...                 cov_estimators=['ledoit_wolf', 'empirical'])
    >>> results = bt.run_backtest()
    >>> metrics = bt.compute_metrics(results)
    >>> print(metrics)
    """

    REBALANCE_MAP = {'monthly': 21, 'annual': 252}

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

    # ── private helpers ──────────────────────────────────────────────────────

    def _get_allocator(self, alloc_name: str, cov_estimator: str):
        """Return the appropriate allocator instance."""
        if alloc_name == 'equal_weight':
            return EqualWeightAllocator()
        elif alloc_name == 'inverse_variance':
            return InverseVarianceAllocator(cov_estimator=cov_estimator)
        elif alloc_name == 'markowitz':
            return MarkowitzMinVarianceAllocator(
                cov_estimator=cov_estimator, min_weight=0.0, max_weight=1.0
            )
        elif alloc_name == 'hrp':
            return HRP()
        else:
            raise ValueError(f"Unknown allocator: {alloc_name}")

    def _compute_weights(
        self,
        train_data: pd.DataFrame,
        alloc_name: str,
        cov_estimator: str
    ) -> pd.Series:
        """Compute portfolio weights for one training window."""
        allocator = self._get_allocator(alloc_name, cov_estimator)
        if alloc_name == 'hrp':
            return allocator.fit(train_data, cov_estimator=cov_estimator)
        else:
            return allocator.fit(train_data)

    @staticmethod
    def _portfolio_returns(
        weights: pd.Series,
        test_data: pd.DataFrame
    ) -> pd.Series:
        """Apply constant weights over test window."""
        return (test_data * weights).sum(axis=1)

    # ── public API ───────────────────────────────────────────────────────────

    def run_backtest(
        self,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run rolling-window backtest for all (allocator × cov_estimator) combinations.

        Returns
        -------
        pd.DataFrame
            Portfolio returns for each strategy, DatetimeIndex columns = strategy names.
        """
        splits = split_data_rolling(
            self.returns,
            train_window=self.train_window,
            test_window=self.test_window
        )

        strategy_returns: Dict[str, List[pd.Series]] = {}

        for alloc_name in self.allocators:
            for cov_est in self.cov_estimators:
                # Skip redundant combinations for equal_weight
                if alloc_name == 'equal_weight' and cov_est != self.cov_estimators[0]:
                    continue

                strategy_name = (
                    f"{alloc_name}" if alloc_name == 'equal_weight'
                    else f"{alloc_name}_{cov_est}"
                )

                if verbose:
                    print(f"Running: {strategy_name} ...")

                period_returns = []
                for train_data, test_data in splits:
                    try:
                        weights = self._compute_weights(train_data, alloc_name, cov_est)
                        port_ret = self._portfolio_returns(weights, test_data)
                        period_returns.append(port_ret)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: skipping window — {e}")

                if period_returns:
                    strategy_returns[strategy_name] = pd.concat(period_returns)

        results = pd.DataFrame(strategy_returns)
        results = results.sort_index()

        if verbose:
            print(f"\nBacktest complete. Shape: {results.shape}")
        return results

    @staticmethod
    def compute_metrics(
        portfolio_returns: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Compute annualised performance metrics for each strategy.

        Metrics
        -------
        Annualised Return, Volatility, Sharpe Ratio, Max Drawdown,
        Calmar Ratio, Sortino Ratio.

        Parameters
        ----------
        portfolio_returns : pd.DataFrame
            Daily returns, one column per strategy.
        periods_per_year : int

        Returns
        -------
        pd.DataFrame
            Metrics table (rows = metrics, columns = strategies).
        """
        metrics: Dict[str, Dict] = {}

        for col in portfolio_returns.columns:
            r = portfolio_returns[col].dropna()
            ann_ret = r.mean() * periods_per_year
            ann_vol = r.std() * np.sqrt(periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

            # Drawdown
            cum = (1 + r).cumprod()
            rolling_max = cum.cummax()
            drawdown = (cum - rolling_max) / rolling_max
            max_dd = drawdown.min()

            calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

            # Sortino
            downside = r[r < 0].std() * np.sqrt(periods_per_year)
            sortino = ann_ret / downside if downside > 0 else np.nan

            metrics[col] = {
                'Ann. Return': f"{ann_ret:.2%}",
                'Ann. Volatility': f"{ann_vol:.2%}",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Max Drawdown': f"{max_dd:.2%}",
                'Calmar Ratio': f"{calmar:.3f}",
                'Sortino Ratio': f"{sortino:.3f}",
            }

        return pd.DataFrame(metrics).T

    def plot_cumulative_returns(
        self,
        results: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        """Plot cumulative portfolio returns for all strategies."""
        import matplotlib.pyplot as plt

        cumulative = (1 + results).cumprod()

        fig, ax = plt.subplots(figsize=figsize)
        for col in cumulative.columns:
            ax.plot(cumulative.index, cumulative[col], label=col, linewidth=1.5)

        ax.set_title('Cumulative Portfolio Returns (Out-of-Sample)', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Growth of $1')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
