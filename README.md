# hrp-ledoitwolf-optimizer

Portfolio optimization project I built to explore an alternative to classic Markowitz — specifically combining Hierarchical Risk Parity with Ledoit-Wolf covariance shrinkage.

The idea is simple: Markowitz is famously unstable out of sample because it inverts a noisy covariance matrix. HRP sidesteps that by using hierarchical clustering instead of inversion. Ledoit-Wolf fixes the covariance estimation noise at the source. Putting both together seemed worth testing properly, so I set up a rolling backtest over 2008–2024 and compared against several baselines.

---

## What's in here

- `src/data_loader.py` — downloads from Yahoo Finance, computes log returns, generates rolling train/test splits
- `src/covariance.py` — empirical covariance, Ledoit-Wolf, OAS, and a comparison helper
- `src/hrp.py` — full HRP implementation: distance matrix, Ward clustering, quasi-diagonalization, recursive bisection
- `src/allocators.py` — equal weight, inverse variance, Markowitz min-variance (CVXPY or scipy fallback)
- `src/backtester.py` — walk-forward backtest engine, performance metrics
- `src/visualization.py` — dendrogram, correlation heatmaps, cumulative returns, drawdowns, rolling Sharpe
- `notebooks/` — 4 notebooks: data exploration, covariance analysis, HRP walkthrough, full backtest results
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

## Running the backtest

```python
from src.backtester import Backtester

bt = Backtester(
    returns=returns,
    allocators=['hrp', 'markowitz', 'inverse_variance', 'equal_weight'],
    cov_estimators=['empirical', 'ledoit_wolf'],
    train_window=3*252,
    test_window=252,
)

results = bt.run_backtest()
print(bt.compute_metrics(results))
bt.plot_cumulative_returns(results)
```

---

## Data

14 ETFs: SPY, QQQ, IWM, EFA, EEM, TLT, AGG, LQD, HYG, GLD, SLV, DBC, VNQ, TIP. Period 2008–2024. All pulled from Yahoo Finance via `yfinance`, no API key needed.

Covers enough market regimes to make the backtest meaningful: GFC, 2020 crash, 2022 rate shock.

---

## Tests

```bash
pytest tests/ -v
```

---

## Notes

HRP is based on López de Prado (2016) — the core idea is using the dendrogram leaf ordering to replace covariance matrix inversion with a hierarchical allocation. Ledoit-Wolf shrinkage (2004) handles the estimation noise in the covariance itself, pulling extreme eigenvalues toward the mean.

Markowitz with Ledoit-Wolf is included as a middle-ground: better covariance, still inverts it. It shows that shrinkage alone isn't enough — the allocation structure matters.

---

## License

MIT
