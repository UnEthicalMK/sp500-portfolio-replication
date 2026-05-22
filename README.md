# S&P 500 Portfolio Replication — The Cloning Engine

A two-model system that tracks the S&P 500 using the minimum number
of stocks required to achieve acceptable tracking error (< 1% annualized).
Built with Lasso regression and a PyTorch sparse autoencoder.

## Results (2023 out-of-sample test year)

| Metric                    | Lasso Clone | Autoencoder | Naive Benchmark |
|---------------------------|-------------|-------------|-----------------|
| Tracking Error (ann.)     | X.XXX%      | X.XXX%      | X.XXX%          |
| Max Drawdown              | X.XX%       | X.XX%       | —               |
| Correlation with S&P 500  | 0.XXXX      | 0.XXXX      | —               |
| Stocks used               | XX          | XX          | 40              |

![Backtest comparison](results/backtest_comparison.png)
![Regularization path](results/regularization_path.png)

## Methodology

**Step 1 — Data:** 10 years of daily adjusted closing prices (2014–2023)
for the S&P 500 index and all 500 constituents, sourced via yfinance.
Daily log returns computed and winsorized at ±10%.

**Step 2 — Train/Val/Test split:** 2014–2021 training, 2022 validation
(alpha tuning only), 2023 test (sealed until final evaluation).

**Step 3 — Lasso model:** Regularization path sweep over 50 alpha values.
Objective: find the sparsest portfolio whose validation tracking error
stays under 1% annualized. positive=True enforces long-only weights.

**Step 4 — Autoencoder:** 500-input → 5-neuron bottleneck → sparse output.
Output layer masked to Lasso-selected stocks only. Trained on portfolio
return MSE loss, not plain reconstruction loss.

**Step 5 — Backtest:** Both models evaluated on unseen 2023 data.
Primary metric: annualized tracking error. Secondary: max drawdown,
correlation with index.

## Known Limitations

- Survivorship bias: constituent list reflects 2024 membership, not
  historical membership. Stocks that were delisted or removed between
  2014–2023 are excluded from the training universe.

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/sp500-portfolio-clone
cd sp500-portfolio-clone
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Requirements

Python 3.10+, see requirements.txt for full dependency list.