# Indian Equities ML Benchmark

A reproducible, leakage-aware baseline that predicts **next-day returns** across the NIFTY-50 universe. It is an educational research project inspired by the model families commonly used as fast, robust baselines in systematic trading research—not a production trading system or investment advice.

## Dataset

The project downloads the public Kaggle dataset [NIFTY-50 Stock Market Data (2000–2021)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data). Raw data is deliberately excluded from Git because it is third-party data; the download script makes the experiment repeatable.

## Models

| Family | Implementations |
| --- | --- |
| Linear | OLS, Ridge |
| Tree ensembles | Random Forest, Extra Trees |
| Gradient boosting | LightGBM |

All models use the same lagged, point-in-time features: 1/5/20-day returns, 20-day volatility, drawdown from the rolling high, volume z-score, and moving-average gap. The split is chronological (first 80% train, final 20% test); no random shuffling is used.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_kaggle.sh
.venv/bin/python src/train.py
```

Results are written to `reports/model_results.csv`, with dataset/split assumptions in `reports/run_metadata.json`.

## Evaluation

Besides RMSE and MAE, the report gives directional accuracy and a deliberately simple **hypothetical** net Sharpe proxy. It takes the sign of the return forecast and subtracts 5 bps whenever that sign changes. It omits slippage, market impact, liquidity constraints, borrow, delays, survivorship bias, and portfolio construction. Do not interpret it as an investable backtest or evidence of future returns.

## Research extensions

Use walk-forward validation, point-in-time constituents, realistic execution simulation, feature neutralization, cross-sectional ranking, and an out-of-sample post-2021 dataset before making any investment inference.
