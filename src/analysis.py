"""
Advanced robustness and stability analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_weight_heatmap(
    weights_history: pd.DataFrame,
    title: str = 'portfolio weights over time',
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Heatmap showing how portfolio weights evolve across rebalances.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(weights_history.T, cmap='RdYlGn', center=0, cbar_kws={'label': 'weight'},
                linewidths=0, ax=ax, vmin=0, vmax=weights_history.max().max())
    ax.set_title(title)
    ax.set_xlabel('rebalance date')
    ax.set_ylabel('asset')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_weight_evolution(
    weights_history: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
) -> None:
    """
    Line plot of weight evolution for each asset.
    """
    fig, ax = plt.subplots(figsize=figsize)
    weights_history.plot(ax=ax, linewidth=1.5, alpha=0.7)
    ax.set_title('weight evolution over time')
    ax.set_ylabel('weight')
    ax.set_xlabel('date')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def stress_test_subperiods(
    results: pd.DataFrame,
    periods: Dict[str, Tuple[str, str]],
    ann: int = 252
) -> pd.DataFrame:
    """
    Compute metrics for specific stress periods.
    
    Example periods:
    {
        'GFC': ('2007-10-01', '2009-03-31'),
        'COVID': ('2020-02-01', '2020-04-30'),
        'Rate shock': ('2022-01-01', '2022-12-31'),
    }
    """
    rows = []
    for period_name, (start, end) in periods.items():
        subset = results.loc[start:end]
        for col in subset.columns:
            r = subset[col].dropna()
            if len(r) == 0:
                continue
            mu = r.mean() * ann
            sigma = r.std() * np.sqrt(ann)
            sharpe = mu / sigma if sigma > 0 else np.nan
            cum = (1 + r).cumprod()
            max_dd = ((cum / cum.cummax()) - 1).min()
            total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else np.nan
            
            rows.append({
                'period': period_name,
                'strategy': col,
                'total_return': f'{total_ret:.2%}',
                'ann_return': f'{mu:.2%}',
                'ann_vol': f'{sigma:.2%}',
                'sharpe': f'{sharpe:.3f}',
                'max_dd': f'{max_dd:.2%}',
            })
    return pd.DataFrame(rows)


def rolling_window_sensitivity(
    returns: pd.DataFrame,
    allocator_name: str,
    cov_estimator: str,
    train_windows: List[int],
    test_window: int = 252,
    ann: int = 252
) -> pd.DataFrame:
    """
    Test sensitivity to training window length.
    """
    from src.backtester import Backtester
    
    rows = []
    for tw in train_windows:
        bt = Backtester(
            returns=returns,
            allocators=[allocator_name],
            cov_estimators=[cov_estimator],
            train_window=tw,
            test_window=test_window
        )
        results = bt.run_backtest(verbose=False)
        metrics = bt.compute_metrics(results, ann=ann)
        
        key = allocator_name if allocator_name == 'equal_weight' else f'{allocator_name}_{cov_estimator}'
        if key in metrics.index:
            rows.append({
                'train_window': tw,
                'sharpe': float(metrics.loc[key, 'sharpe']),
                'calmar': float(metrics.loc[key, 'calmar']),
                'max_dd': metrics.loc[key, 'max_dd'],
            })
    return pd.DataFrame(rows)


def compare_weight_stability(
    weights_histories: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Compare weight variance across strategies.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    variances = {}
    turnovers = {}
    
    for strat, wh in weights_histories.items():
        variances[strat] = wh.var(axis=0).mean()
        turnover_list = []
        for i in range(1, len(wh)):
            turnover_list.append(np.abs(wh.iloc[i].values - wh.iloc[i-1].values).sum())
        turnovers[strat] = np.mean(turnover_list) if turnover_list else 0
    
    pd.Series(variances).plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('avg weight variance')
    axes[0].set_ylabel('variance')
    axes[0].grid(alpha=0.3, axis='y')
    
    pd.Series(turnovers).plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('avg turnover per rebalance')
    axes[1].set_ylabel('turnover')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
