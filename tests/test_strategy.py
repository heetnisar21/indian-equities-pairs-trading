import numpy as np
import pandas as pd

from indian_pairs.strategy import PairConfig, backtest_pair, performance_stats


def test_backtest_is_finite_and_no_position_during_formation():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=180)
    x = pd.Series(100 * np.exp(rng.normal(0, 0.01, len(index)).cumsum()), index=index)
    y = pd.Series(x.to_numpy() * np.exp(rng.normal(0, 0.015, len(index))), index=index)
    cfg = PairConfig(formation_days=60, trading_days=10)
    result = backtest_pair(y, x, cfg)
    assert (result.position.iloc[:60] == 0).all()
    assert np.isfinite(result.net_return).all()
    assert {"sharpe", "max_drawdown"}.issubset(performance_stats(result.net_return))

