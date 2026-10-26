"""Research tools for statistically disciplined NSE pairs trading."""

from .strategy import PairConfig, backtest_pair, discover_pairs

__all__ = ["PairConfig", "backtest_pair", "discover_pairs"]

