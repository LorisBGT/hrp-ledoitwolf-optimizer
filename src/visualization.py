"""
Visualization helpers.

Dendrograms, correlation heatmaps, cumulative returns, drawdowns, rolling Sharpe.
Mostly called from notebooks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch


def plot_dendrogram(
    linkage: np.ndarray,
    labels: List[str],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sch.dendrogram(linkage, labels=labels, ax=ax,
                   leaf_font_size=10, leaf_rotation=45)
    ax.set_title('hierarchical clustering — Ward linkage')
    ax.set_ylabel('distance')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(
    corr: np.ndarray,
    labels: List[str],
    ordered_idx: Optional[List[int]] = None,
    title: str = 'correlation matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    if ordered_idx is not None:
        corr = corr[np.ix_(ordered_idx, ordered_idx)]
        labels = [labels[i] for i in ordered_idx]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=-1, vmax=1, center=0,
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.4, annot_kws={'size': 8})
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_weights_comparison(
    weights_dict: Dict[str, pd.Series],
    figsize: Tuple[int, int] = (13, 5),
    save_path: Optional[str] = None
) -> None:
    pd.DataFrame(weights_dict).plot(kind='bar', figsize=figsize, width=0.75)
    plt.title('weight allocation by strategy')
    plt.ylabel('weight')
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> None:
    ax = (1 + returns_df).cumprod().plot(figsize=figsize, linewidth=1.6)
    ax.set_ylabel('growth of $1')
    ax.set_title('cumulative returns')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_drawdown(
    returns_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for col in returns_df.columns:
        r = returns_df[col].dropna()
        dd = (1 + r).cumprod()
        dd = dd / dd.cummax() - 1
        ax.fill_between(dd.index, dd, 0, alpha=0.25, label=col)
        ax.plot(dd.index, dd, linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_title('drawdowns')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_sharpe(
    returns_df: pd.DataFrame,
    window: int = 252,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    rs = (returns_df.rolling(window).mean()
          / returns_df.rolling(window).std()) * np.sqrt(252)
    ax = rs.plot(figsize=figsize, linewidth=1.5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axhline(1, color='gray', lw=0.7, ls=':')
    ax.set_title(f'rolling Sharpe ({window}d)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
