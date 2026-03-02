"""
Bootstrap and Monte Carlo simulation for robustness testing.

The idea: rather than trusting a single backtest path, resample returns
to see how much the results vary under different market realisations.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def bootstrap_returns(
    returns: pd.DataFrame,
    n_samples: int,
    block_size: int = 20,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Stationary block bootstrap: resample blocks of consecutive returns
    to preserve short-term autocorrelation structure.

    block_size : length of each block in days (default 20 = ~1 month)
    """
    rng = np.random.default_rng(seed)
    n, p = returns.shape
    sampled = []

    while len(sampled) < n_samples:
        start = rng.integers(0, n - block_size)
        block = returns.iloc[start: start + block_size].values
        sampled.append(block)

    arr = np.vstack(sampled)[:n_samples]
    return pd.DataFrame(arr, columns=returns.columns)


def bootstrap_sharpe(
    returns: pd.DataFrame,
    allocator_fn: Callable[[pd.DataFrame], pd.Series],
    n_bootstrap: int = 500,
    block_size: int = 20,
    ann: int = 252,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Bootstrap distribution of the Sharpe ratio for a given allocator.

    allocator_fn : function that takes a returns DataFrame and returns portfolio weights

    Returns a Series of n_bootstrap Sharpe estimates.
    """
    sharpes = []
    for i in range(n_bootstrap):
        resampled = bootstrap_returns(returns, n_samples=len(returns),
                                      block_size=block_size, seed=seed + i if seed else None)
        try:
            w = allocator_fn(resampled)
            port_ret = (resampled * w).sum(axis=1)
            mu = port_ret.mean() * ann
            sigma = port_ret.std() * np.sqrt(ann)
            sharpes.append(mu / sigma if sigma > 0 else np.nan)
        except Exception:
            sharpes.append(np.nan)
    return pd.Series(sharpes, name='sharpe_bootstrap')


def compare_allocators_bootstrap(
    returns: pd.DataFrame,
    allocator_fns: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    n_bootstrap: int = 500,
    block_size: int = 20,
    ann: int = 252,
    seed: int = 42
) -> pd.DataFrame:
    """
    Bootstrap Sharpe distribution for multiple allocators.

    Returns a DataFrame with columns = allocator names, rows = bootstrap iterations.
    """
    results = {}
    for name, fn in allocator_fns.items():
        print(f'  bootstrapping {name}...')
        results[name] = bootstrap_sharpe(
            returns, fn, n_bootstrap=n_bootstrap,
            block_size=block_size, ann=ann, seed=seed
        ).values
    return pd.DataFrame(results)


def bootstrap_summary(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for bootstrap Sharpe distributions:
    mean, std, 5th/95th percentiles, probability of Sharpe > 0.
    """
    rows = {}
    for col in bootstrap_df.columns:
        s = bootstrap_df[col].dropna()
        rows[col] = {
            'mean_sharpe': f'{s.mean():.3f}',
            'std_sharpe': f'{s.std():.3f}',
            'p5': f'{s.quantile(0.05):.3f}',
            'p95': f'{s.quantile(0.95):.3f}',
            'prob_positive': f'{(s > 0).mean():.2%}',
        }
    return pd.DataFrame(rows).T


def plot_bootstrap_distributions(
    bootstrap_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """KDE + histogram of bootstrap Sharpe distributions per strategy."""
    import matplotlib.pyplot as plt

    n = len(bootstrap_df.columns)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, bootstrap_df.columns):
        data = bootstrap_df[col].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='none')
        ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1, label=f'mean={data.mean():.2f}')
        ax.axvline(0, color='red', linestyle=':', linewidth=1)
        ax.set_title(col)
        ax.set_xlabel('Sharpe')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('density')
    plt.suptitle('bootstrap Sharpe distributions', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
