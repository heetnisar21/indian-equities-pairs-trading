# Indian Equities Pairs Trading

A walk-forward, cost-aware statistical-arbitrage research framework for NSE equities. Built to demonstrate robust quant-research practice rather than optimize a headline backtest.

> Educational research only — not investment advice.

## Research design

- Sector-constrained Engle–Granger cointegration screen
- OLS hedge ratios and log-price spread construction
- Rolling, past-only formation windows with scheduled recalibration
- Z-score entries, exits, and stop rules
- One-day signal delay to prevent same-close look-ahead
- Gross-normalized legs, turnover costs, and net performance statistics
- Synthetic regression test for timing and finite-P&L checks

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python run_research.py
pytest -q
```

The example screens a liquid NSE banking universe using prices adjusted for splits and dividends, selects candidates through 2021, and reports only 2022 onward. Replace Yahoo Finance data and the placeholder 12 bps one-way turnover cost with licensed, broker-specific data before any live use.

## Interview narrative

“I built a market-neutral NSE statistical-arbitrage framework that estimates hedge ratios strictly out of sample and evaluates z-score signals net of turnover costs. I focused on look-ahead bias, data adjustment, multiple testing, borrow constraints, and regime instability before interpreting performance.”

## Next steps

Add point-in-time index membership, false-discovery control, liquidity/borrow filters, portfolio optimization, a realistic execution model, and paper-trading monitoring.
