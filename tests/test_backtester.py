import pytest
import numpy as np
import pandas as pd
from src.backtester import Backtester


@pytest.fixture
def small_returns():
    rng = np.random.default_rng(99)
    dates = pd.date_range('2015-01-01', periods=1200, freq='B')
    data = rng.standard_normal((1200, 5)) * 0.01
    return pd.DataFrame(data, index=dates,
                        columns=['SPY', 'TLT', 'GLD', 'VNQ', 'EFA'])


def test_backtest_returns_df(small_returns):
    bt = Backtester(small_returns, ['equal_weight'], ['empirical'],
                   train_window=252, test_window=63)
    res = bt.run_backtest(verbose=False)
    assert isinstance(res, pd.DataFrame) and len(res) > 0


def test_equal_weight_column(small_returns):
    bt = Backtester(small_returns, ['equal_weight'], ['empirical'],
                   train_window=252, test_window=63)
    res = bt.run_backtest(verbose=False)
    assert 'equal_weight' in res.columns


def test_metrics_has_sharpe(small_returns):
    bt = Backtester(small_returns, ['equal_weight', 'inverse_variance'],
                   ['empirical'], train_window=252, test_window=63)
    res = bt.run_backtest(verbose=False)
    metrics = bt.compute_metrics(res)
    assert 'sharpe' in metrics.columns and 'max_dd' in metrics.columns
