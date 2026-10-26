"""Leakage-aware Engle--Granger pair selection and trading backtest."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class PairConfig:
    formation_days: int = 252
    trading_days: int = 21
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    cost_bps: float = 12.0  # one-way cost per unit of gross portfolio turnover


def _fit_spread(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS y = alpha + beta*x, fitted only on the supplied formation sample."""
    beta, alpha = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    return float(alpha), float(beta)


def discover_pairs(prices: pd.DataFrame, pvalue_cutoff: float = 0.05) -> pd.DataFrame:
    """Screen a same-sector universe using Engle--Granger cointegration.

    This is a candidate screen, not an economic justification. Select a universe
    ex-ante (e.g. liquid NIFTY Financial Services names) and apply multiple-testing
    control before allocating capital.
    """
    log_prices = np.log(prices.dropna())
    rows: list[dict[str, float | str]] = []
    for left, right in combinations(log_prices.columns, 2):
        stat, pvalue, _ = coint(log_prices[left], log_prices[right])
        if pvalue <= pvalue_cutoff:
            alpha, beta = _fit_spread(log_prices[left], log_prices[right])
            rows.append({"left": left, "right": right, "pvalue": pvalue, "t_stat": stat, "beta": beta, "alpha": alpha})
    return pd.DataFrame(rows).sort_values("pvalue", ignore_index=True) if rows else pd.DataFrame(
        columns=["left", "right", "pvalue", "t_stat", "beta", "alpha"]
    )


def backtest_pair(left: pd.Series, right: pd.Series, config: PairConfig = PairConfig()) -> pd.DataFrame:
    """Walk-forward backtest of one pair, returning daily positions and P&L.

    At each rebalance the hedge ratio and z-score moments use *only* the prior
    ``formation_days`` observations. Signals are shifted one day before P&L,
    preventing same-close execution look-ahead.
    """
    prices = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    logs = np.log(prices)
    out = pd.DataFrame(index=prices.index, columns=["beta", "zscore", "position"], dtype=float)
    # Piecewise-constant parameter estimates recalibrated at scheduled intervals.
    for trade_start in range(config.formation_days, len(logs), config.trading_days):
        train = logs.iloc[trade_start - config.formation_days:trade_start]
        alpha, beta = _fit_spread(train["left"], train["right"])
        spread_train = train["left"] - alpha - beta * train["right"]
        mu, sigma = spread_train.mean(), spread_train.std(ddof=1)
        trade_end = min(trade_start + config.trading_days, len(logs))
        spread = logs["left"].iloc[trade_start:trade_end] - alpha - beta * logs["right"].iloc[trade_start:trade_end]
        out.iloc[trade_start:trade_end, out.columns.get_loc("beta")] = beta
        out.iloc[trade_start:trade_end, out.columns.get_loc("zscore")] = (spread - mu) / sigma

    z = out["zscore"]
    position = pd.Series(0.0, index=out.index)
    state = 0.0
    for date, value in z.dropna().items():
        if state == 0 and value >= config.entry_z:
            state = -1.0  # short spread: short left, long beta units of right
        elif state == 0 and value <= -config.entry_z:
            state = 1.0
        elif state != 0 and (abs(value) <= config.exit_z or abs(value) >= config.stop_z):
            state = 0.0
        position.loc[date] = state
    out["position"] = position
    # The loop only writes trade dates. Clear values before the formation period explicitly.
    out.loc[out.index[: config.formation_days], "position"] = 0.0
    returns = prices.pct_change().fillna(0.0)
    # Gross-normalize the two legs, so a portfolio has one rupee of total exposure.
    beta = out["beta"].fillna(0.0)
    left_weight = out["position"] / (1 + beta.abs())
    right_weight = -out["position"] * beta / (1 + beta.abs())
    gross = left_weight.shift(1).fillna(0.0) * returns["left"] + right_weight.shift(1).fillna(0.0) * returns["right"]
    # This also charges re-hedging when beta changes between walk-forward windows.
    turnover = left_weight.diff().abs().fillna(0.0) + right_weight.diff().abs().fillna(0.0)
    costs = turnover * config.cost_bps / 10_000
    out["gross_return"], out["cost"], out["net_return"] = gross, costs, gross - costs
    out["equity"] = (1 + out["net_return"]).cumprod()
    return out


def performance_stats(daily_returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    """Compute annualized, net-of-cost performance statistics."""
    r = daily_returns.dropna()
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "annual_return": float(equity.iloc[-1] ** (annualization / len(r)) - 1),
        "annual_volatility": float(r.std(ddof=1) * np.sqrt(annualization)),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(annualization)) if r.std(ddof=1) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "observations": float(len(r)),
    }
