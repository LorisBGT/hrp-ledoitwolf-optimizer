"""
Unit tests for the backtesting engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.backtester import Backtester


@pytest.fixture
def sample_returns():
    np.random.seed(99)
    n_days = 1200
    n_assets = 5
    dates = pd.date_range('2015-01-01', periods=n_days, freq='B')
    data = np.random.randn(n_days, n_assets) * 0.01
    return pd.DataFrame(data, index=dates,
                        columns=['SPY', 'TLT', 'GLD', 'VNQ', 'EFA'])


class TestBacktester:
    def test_run_backtest_returns_dataframe(self, sample_returns):
        bt = Backtester(
            returns=sample_returns,
            allocators=['equal_weight', 'inverse_variance'],
            cov_estimators=['empirical'],
            train_window=252,
            test_window=63,
        )
        results = bt.run_backtest(verbose=False)
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_columns_match_strategies(self, sample_returns):
        bt = Backtester(
            returns=sample_returns,
            allocators=['equal_weight'],
            cov_estimators=['empirical'],
            train_window=252,
            test_window=63,
        )
        results = bt.run_backtest(verbose=False)
        assert 'equal_weight' in results.columns

    def test_compute_metrics_returns_dataframe(self, sample_returns):
        bt = Backtester(
            returns=sample_returns,
            allocators=['equal_weight'],
            cov_estimators=['empirical'],
            train_window=252,
            test_window=63,
        )
        results = bt.run_backtest(verbose=False)
        metrics = bt.compute_metrics(results)
        assert isinstance(metrics, pd.DataFrame)
        assert 'Sharpe Ratio' in metrics.columns
        assert 'Max Drawdown' in metrics.columns

    def test_no_nan_in_metrics(self, sample_returns):
        bt = Backtester(
            returns=sample_returns,
            allocators=['equal_weight', 'inverse_variance'],
            cov_estimators=['empirical'],
            train_window=252,
            test_window=63,
        )
        results = bt.run_backtest(verbose=False)
        metrics = bt.compute_metrics(results)
        # Spot-check that Sharpe ratio is a finite-looking string
        for col in metrics.index:
            assert metrics.loc[col, 'Sharpe Ratio'] != 'nan'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
