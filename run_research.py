"""Example: screen liquid banking names, then backtest the strongest candidate."""

from pathlib import Path

from indian_pairs.data import download_prices
from indian_pairs.strategy import PairConfig, backtest_pair, discover_pairs, performance_stats


UNIVERSE = ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"]


def main() -> None:
    prices = download_prices(UNIVERSE, start="2018-01-01", end="2025-01-01")
    formation = prices.loc[:"2021-12-31"]
    candidates = discover_pairs(formation)
    if candidates.empty:
        raise RuntimeError("No cointegrated pairs at this threshold; revise the pre-specified universe or period.")
    winner = candidates.iloc[0]
    result = backtest_pair(prices[winner.left], prices[winner.right], PairConfig())
    # Report only the period not used to select the pair.
    result = result.loc["2022-01-01":]
    print(f"Selected on formation sample: {winner.left}/{winner.right}, p={winner.pvalue:.4f}")
    print("Out-of-sample statistics:", performance_stats(result.net_return))
    Path("reports").mkdir(exist_ok=True)
    result.to_csv("reports/pair_backtest.csv")
    candidates.to_csv("reports/candidates.csv", index=False)


if __name__ == "__main__":
    main()
