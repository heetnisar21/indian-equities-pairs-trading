"""Data access with explicit adjustment and NSE ticker conventions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def nse_ticker(symbol: str) -> str:
    """Return the Yahoo Finance NSE ticker, accepting either form."""
    return symbol.upper() if symbol.upper().endswith(".NS") else f"{symbol.upper()}.NS"


def download_prices(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices and drop dates with incomplete observations.

    Adjusted prices incorporate splits and dividends. In production, replace Yahoo
    with a licensed NSE/vendor feed and retain its raw corporate-action records.
    """
    import yfinance as yf

    tickers = [nse_ticker(s) for s in symbols]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices.columns = [str(c).replace(".NS", "") for c in prices.columns]
    return prices.sort_index().dropna(how="any")

