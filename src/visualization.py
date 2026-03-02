"""
Visualization utilities for portfolio analysis.

Provides dendrograms, correlation heatmaps, weight comparisons,
cumulative returns, drawdowns, and rolling Sharpe charts.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy.cluster.hierarchy as sch


def plot_dendrogram(
    linkage: np.ndarray,
    labels: List[str],
    figsize: Tuple[int, int] = (12, 5),
    title: str = 'HRP Hierarchical Clustering',
    save_path: Optional[str] = None
) -> None:
    """
    Plot hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage : np.ndarray
        Scipy linkage matrix.
    labels : list of str
        Asset names for leaf labels.
    figsize : tuple
    title : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=figsize)
    sch.dendrogram(linkage, labels=labels, ax=ax,
                   leaf_font_size=10, leaf_rotation=45)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Assets')
    ax.set_ylabel('Ward Distance')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(
    corr: np.ndarray,
    labels: List[str],
    ordered_indices: Optional[List[int]] = None,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix heatmap (optionally reordered by dendrogram).

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix.
    labels : list of str
    ordered_indices : list of int, optional
        Dendrogram ordering — enables quasi-diagonal visualisation.
    title : str
    figsize : tuple
    save_path : str, optional
    """
    if ordered_indices is not None:
        corr = corr[np.ix_(ordered_indices, ordered_indices)]
        labels = [labels[i] for i in ordered_indices]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=-1, vmax=1, center=0,
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, annot_kws={'size': 8}
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_weights_comparison(
    weights_dict: Dict[str, pd.Series],
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Portfolio Weights Comparison',
    save_path: Optional[str] = None
) -> None:
    """
    Bar chart comparing weights across multiple strategies.

    Parameters
    ----------
    weights_dict : dict
        Mapping strategy name → pd.Series of weights.
    figsize : tuple
    title : str
    save_path : str, optional
    """
    weights_df = pd.DataFrame(weights_dict)
    ax = weights_df.plot(kind='bar', figsize=figsize, width=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Asset')
    ax.set_ylabel('Weight')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 7),
    title: str = 'Cumulative Portfolio Returns',
    save_path: Optional[str] = None
) -> None:
    """
    Plot cumulative (compound) returns for multiple strategies.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, one column per strategy.
    figsize : tuple
    title : str
    save_path : str, optional
    """
    cumulative = (1 + returns_df).cumprod()

    fig, ax = plt.subplots(figsize=figsize)
    colors = cm.tab10(np.linspace(0, 1, len(returns_df.columns)))

    for col, color in zip(cumulative.columns, colors):
        ax.plot(cumulative.index, cumulative[col], label=col,
                linewidth=1.8, color=color)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_drawdown(
    returns_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Portfolio Drawdowns',
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling drawdown for each strategy.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns.
    figsize : tuple
    title : str
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = cm.tab10(np.linspace(0, 1, len(returns_df.columns)))

    for col, color in zip(returns_df.columns, colors):
        r = returns_df[col].dropna()
        cum = (1 + r).cumprod()
        drawdown = (cum / cum.cummax()) - 1
        ax.fill_between(drawdown.index, drawdown, 0,
                        alpha=0.3, color=color, label=col)
        ax.plot(drawdown.index, drawdown, color=color, linewidth=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_sharpe(
    returns_df: pd.DataFrame,
    window: int = 252,
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Rolling Sharpe Ratio (1-Year Window)',
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns.
    window : int
        Rolling window in days.
    figsize : tuple
    title : str
    save_path : str, optional
    """
    rolling_sharpe = (
        returns_df.rolling(window).mean()
        / returns_df.rolling(window).std()
    ) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=figsize)
    for col in rolling_sharpe.columns:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col],
                label=col, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.8, linestyle=':')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
