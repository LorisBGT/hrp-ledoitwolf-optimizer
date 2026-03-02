# hrp-ledoitwolf-optimizer

Portfolio optimization project combining Hierarchical Risk Parity with Ledoit-Wolf covariance shrinkage.

Markowitz is unstable out of sample because it inverts a noisy covariance matrix. HRP sidesteps that by using hierarchical clustering instead of inversion. Ledoit-Wolf fixes the covariance estimation noise at the source. Putting both together seemed worth testing properly — rolling backtest over 2008–2024, compared against several baselines.

---

## Structure

- `src/data_loader.py` — Yahoo Finance download, log returns, rolling train/test splits
- `src/covariance.py` — empirical covariance, Ledoit-Wolf, OAS, comparison helper
- `src/hrp.py` — HRP: distance matrix, Ward clustering, quasi-diagonalization, recursive bisection
- `src/allocators.py` — equal weight, inverse variance, Markowitz (long-only + short + leverage), Risk Parity
- `src/backtester.py` — walk-forward backtest engine, comprehensive performance metrics
- `src/analysis.py` — weight stability, rolling metrics, stress tests, robustness analysis
- `src/simulation.py` — block bootstrap for Sharpe distribution estimation
- `src/visualization.py` — dendrogram, correlation heatmaps, cumulative returns, drawdowns, rolling Sharpe
- `notebooks/` — 4 notebooks: data exploration, covariance analysis, HRP walkthrough, full backtest
- `tests/` — unit tests

---

## Setup

```bash
git clone https://github.com/LorisBGT/hrp-ledoitwolf-optimizer.git
cd hrp-ledoitwolf-optimizer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick example

```python
from src.data_loader import download_price_data, compute_returns
from src.covariance import ledoit_wolf_covariance
from src.hrp import HRP

prices = download_price_data(['SPY', 'TLT', 'GLD', 'EEM', 'VNQ'],
                             '2010-01-01', '2024-01-01')
returns = compute_returns(prices)

cov, alpha = ledoit_wolf_covariance(returns)
print(f'shrinkage coeff: {alpha:.3f}')

hrp = HRP()
weights = hrp.fit(returns, cov=cov)
print(weights.sort_values(ascending=False))
hrp.plot_dendrogram()
```

---

## Backtest

```python
from src.backtester import Backtester

bt = Backtester(
    returns=returns,
    allocators=['hrp', 'markowitz', 'inverse_variance', 'equal_weight', 'risk_parity'],
    cov_estimators=['empirical', 'ledoit_wolf'],
    train_window=3*252,
    test_window=252,
    transaction_cost=0.001
)

results = bt.run_backtest()
print(bt.compute_metrics(results))
print(bt.compute_weight_stability())

bt.plot_cumulative_returns(results)
bt.plot_rolling_sharpe(results, window=252)
bt.plot_drawdown(results)
```

---

## Advanced analysis

```python
from src.analysis import (
    plot_weight_heatmap,
    stress_test_subperiods,
    rolling_window_sensitivity
)

# Weight stability
for strat, weights_df in bt.weights_history_.items():
    plot_weight_heatmap(weights_df, title=f'{strat} — weights over time')

# Stress test by subperiod
periods = {
    'GFC': ('2007-10-01', '2009-03-31'),
    'COVID': ('2020-02-01', '2020-04-30'),
    'Rate shock': ('2022-01-01', '2022-12-31'),
}
print(stress_test_subperiods(results, periods))

# Sensitivity to training window
sens = rolling_window_sensitivity(
    returns, 'hrp', 'ledoit_wolf',
    train_windows=[252, 2*252, 3*252, 4*252]
)
print(sens)
```

---

## Bootstrap

```python
from src.simulation import compare_allocators_bootstrap, bootstrap_summary, plot_bootstrap_distributions
from src.hrp import HRP
from src.allocators import RiskParityAllocator, EqualWeightAllocator

allocator_fns = {
    'hrp_lw':       lambda r: HRP().fit(r, cov_estimator='ledoit_wolf'),
    'risk_parity':  lambda r: RiskParityAllocator('ledoit_wolf').fit(r),
    'equal_weight': lambda r: EqualWeightAllocator().fit(r),
}

bs = compare_allocators_bootstrap(returns, allocator_fns, n_bootstrap=500)
print(bootstrap_summary(bs))
plot_bootstrap_distributions(bs)
```

---

## Constraints (long-only vs leverage)

```python
from src.allocators import MarkowitzMinVarianceAllocator

# Long-only, fully invested (default)
long_only = MarkowitzMinVarianceAllocator('ledoit_wolf', allow_short=False)

# Allow short, max leverage 1.3
with_short = MarkowitzMinVarianceAllocator('ledoit_wolf', allow_short=True, max_leverage=1.3)
```

---

## Data

14 ETFs: SPY, QQQ, IWM, EFA, EEM, TLT, AGG, LQD, HYG, GLD, SLV, DBC, VNQ, TIP. Period 2008–2024, pulled from Yahoo Finance via `yfinance`.

Covers enough market regimes to make the backtest meaningful: GFC, 2020 crash, 2022 rate shock.

---

## Performance metrics

The backtester computes:

- Annualized return, volatility, Sharpe, Sortino
- Maximum drawdown, Calmar ratio
- Skewness, kurtosis, hit rate
- Weight stability: variance, turnover, Herfindahl concentration
- Rolling Sharpe over time
- Stress test performance by subperiod
- Sensitivity to training window length
- Bootstrap Sharpe distribution with confidence intervals

---

## Tests

```bash
pytest tests/ -v
```

---

## Notes

HRP is based on López de Prado (2016): the dendrogram leaf ordering replaces covariance matrix inversion with a hierarchical allocation. Ledoit-Wolf shrinkage (2004) handles estimation noise, pulling extreme eigenvalues toward the mean.

Markowitz with Ledoit-Wolf is included as a middle-ground — better covariance, but still inverts it. The point is that shrinkage alone isn't enough; the allocation structure matters.

Risk Parity (ERC, Maillard et al. 2010) is added as a structural benchmark: no inversion either, but targets equal risk contribution rather than hierarchical clustering.

Transaction costs are applied at each rebalance. Default is 10 bps (0.1%), configurable in the `Backtester` constructor.

---

## License

MIT
