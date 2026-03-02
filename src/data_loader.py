"""
Data loading and preprocessing module for portfolio optimization.

Downloads price data from Yahoo Finance, handles missing values,
computes returns, and prepares rolling windows for backtesting.
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

    Parameters
    ----------
    tickers : list of str or str
        Ticker symbols to download.
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Prices with DatetimeIndex, columns = tickers.

    Examples
    --------
    >>> prices = download_price_data(['SPY', 'TLT', 'GLD'], '2008-01-01', '2024-01-01')
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers=tickers, start=start_date, end=end_date,
                       progress=False, auto_adjust=True)

    if len(tickers) == 1:
        prices = data[['Close']].rename(columns={'Close': tickers[0]})
    else:
        prices = data['Close']

    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"Warning: could not download {missing}")

    print(f"Downloaded {len(prices)} rows for {len(prices.columns)} tickers")
    return prices


def clean_price_data(
    prices: pd.DataFrame,
    fill_method: str = 'ffill',
    max_missing: float = 0.05
) -> pd.DataFrame:
    """
    Clean price data: remove assets with too many NaNs, forward-fill the rest.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price data.
    fill_method : str
        'ffill', 'bfill', or 'interpolate'.
    max_missing : float
        Maximum fraction of missing values allowed per asset.

    Returns
    -------
    pd.DataFrame
        Cleaned prices.
    """
    missing_pct = prices.isna().sum() / len(prices)
    cols_to_keep = missing_pct[missing_pct <= max_missing].index
    cols_to_drop = missing_pct[missing_pct > max_missing].index

    if len(cols_to_drop):
        print(f"Dropping {list(cols_to_drop)} (>{max_missing*100:.0f}% missing)")

    clean = prices[cols_to_keep].copy()

    if fill_method == 'ffill':
        clean = clean.ffill().bfill()
    elif fill_method == 'bfill':
        clean = clean.bfill().ffill()
    elif fill_method == 'interpolate':
        clean = clean.interpolate('linear').bfill()
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    clean = clean.dropna()
    print(f"Cleaned data shape: {clean.shape}")
    return clean


def compute_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """
    Compute returns from price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    method : str
        'log' for log-returns or 'simple' for arithmetic returns.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame (first row dropped).

    Examples
    --------
    >>> returns = compute_returns(prices, method='log')
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    return returns.iloc[1:]


def split_data_rolling(
    returns: pd.DataFrame,
    train_window: int = 3 * 252,
    test_window: int = 252,
    step_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate rolling train/test splits for walk-forward backtesting.

    Parameters
    ----------
    returns : pd.DataFrame
        Return time series.
    train_window : int
        Training window in days (default 3 years = 756).
    test_window : int
        Test window in days (default 1 year = 252).
    step_size : int, optional
        Rolling step in days. Defaults to test_window.

    Returns
    -------
    list of (train, test) DataFrame tuples.
    """
    if step_size is None:
        step_size = test_window

    splits = []
    for start in range(0, len(returns) - train_window - test_window + 1, step_size):
        train = returns.iloc[start: start + train_window]
        test = returns.iloc[start + train_window: start + train_window + test_window]
        splits.append((train, test))

    print(f"Generated {len(splits)} rolling splits "
          f"(train={train_window}d, test={test_window}d, step={step_size}d)")
    return splits


def get_default_etf_universe() -> List[str]:
    """
    Return a diversified multi-asset ETF universe.

    Returns
    -------
    list of str
        Ticker symbols covering equities, bonds, commodities, real estate.
    """
    return [
        'SPY', 'QQQ', 'IWM',   # US equities
        'EFA', 'EEM',           # International equities
        'TLT', 'AGG', 'LQD', 'HYG',  # Fixed income
        'GLD', 'SLV', 'DBC',   # Commodities
        'VNQ',                  # Real estate
        'TIP',                  # Inflation-protected
    ]


if __name__ == "__main__":
    tickers = get_default_etf_universe()
    prices = download_price_data(tickers, '2008-01-01', '2024-01-01')
    prices_clean = clean_price_data(prices)
    returns = compute_returns(prices_clean)
    splits = split_data_rolling(returns)
    print(f"Ready: {returns.shape}, {len(splits)} backtest splits")
