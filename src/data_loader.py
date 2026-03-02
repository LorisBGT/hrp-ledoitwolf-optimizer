"""
Data loading and preprocessing for walk-forward backtesting.
"""

from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import yfinance as yf


def download_price_data(
    tickers: Union[List[str], str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    print(f'downloading {len(tickers)} tickers ({start_date} to {end_date})')
    raw = yf.download(tickers=tickers, start=start_date, end=end_date,
                      progress=False, auto_adjust=True)

    prices = raw['Close'] if len(tickers) > 1 else raw[['Close']].rename(
        columns={'Close': tickers[0]})

    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f'  warning: missing tickers {missing}')

    print(f'  {len(prices)} rows x {len(prices.columns)} assets')
    return prices


def clean_price_data(
    prices: pd.DataFrame,
    fill_method: str = 'ffill',
    max_missing: float = 0.05
) -> pd.DataFrame:
    """
    Drop assets exceeding the missing data threshold, then forward-fill.

    fill_method: 'ffill', 'bfill', or 'interpolate'
    max_missing: fraction threshold (default 5%)
    """
    missing_frac = prices.isna().sum() / len(prices)
    to_drop = missing_frac[missing_frac > max_missing].index.tolist()
    if to_drop:
        print(f'  dropping {to_drop} (>{max_missing:.0%} missing)')

    clean = prices.drop(columns=to_drop).copy()

    if fill_method == 'ffill':
        clean = clean.ffill().bfill()
    elif fill_method == 'bfill':
        clean = clean.bfill().ffill()
    elif fill_method == 'interpolate':
        clean = clean.interpolate('linear').bfill()
    else:
        raise ValueError(f'unknown fill_method: {fill_method}')

    return clean.dropna()


def compute_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """Daily returns. method: 'log' or 'simple'."""
    if method == 'log':
        return np.log(prices / prices.shift(1)).iloc[1:]
    elif method == 'simple':
        return prices.pct_change().iloc[1:]
    raise ValueError("method must be 'log' or 'simple'")


def split_data_rolling(
    returns: pd.DataFrame,
    train_window: int = 3 * 252,
    test_window: int = 252,
    step_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling train/test splits for walk-forward backtesting."""
    step = step_size or test_window
    splits = []
    for i in range(0, len(returns) - train_window - test_window + 1, step):
        train = returns.iloc[i: i + train_window]
        test = returns.iloc[i + train_window: i + train_window + test_window]
        splits.append((train, test))
    print(f'{len(splits)} splits (train={train_window}d test={test_window}d step={step}d)')
    return splits


def get_default_etf_universe() -> List[str]:
    """14 liquid ETFs: equities, bonds, commodities, real estate."""
    return [
        'SPY', 'QQQ', 'IWM',
        'EFA', 'EEM',
        'TLT', 'AGG', 'LQD', 'HYG',
        'GLD', 'SLV', 'DBC',
        'VNQ',
        'TIP',
    ]
