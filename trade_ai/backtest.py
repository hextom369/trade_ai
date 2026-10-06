"""Vectorised backtester with transaction costs and no look-ahead."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .metrics import performance_summary


@dataclass
class BacktestResult:
    returns: pd.Series          # strategy net returns per bar
    positions: pd.Series        # position held during each bar
    equity: pd.Series           # cumulative equity curve starting at 1.0
    benchmark: pd.Series        # buy-and-hold returns per bar
    stats: dict = field(default_factory=dict)
    benchmark_stats: dict = field(default_factory=dict)


def run_backtest(close: pd.Series, target_positions: pd.Series,
                 cost_bps: float = 5.0, slippage_bps: float = 2.0,
                 periods_per_year: int = 252) -> BacktestResult:
    """Simulate trading ``target_positions`` decided at each bar's close.

    The position chosen at the close of bar t is held over bar t+1, so the
    signal never sees the return it is paid on. Costs are charged on the
    absolute change in position, in basis points of notional.
    """
    target_positions = target_positions.reindex(close.index).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)
    held = target_positions.shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (cost_bps + slippage_bps) / 1e4
    net = held * asset_ret - cost
    equity = (1 + net).cumprod()
    return BacktestResult(
        returns=net.rename("strategy"),
        positions=held.rename("position"),
        equity=equity.rename("equity"),
        benchmark=asset_ret.rename("buy_hold"),
        stats=performance_summary(net, held, periods_per_year),
        benchmark_stats=performance_summary(asset_ret, pd.Series(1.0, index=close.index),
                                            periods_per_year),
    )


def trim_to_active(result: BacktestResult, proba: pd.Series,
                   periods_per_year: int = 252) -> BacktestResult:
    """Restrict a result to the out-of-sample period (first available prediction onward)."""
    first = proba.first_valid_index()
    if first is None:
        return result
    sl = slice(first, None)
    r, p, b = result.returns.loc[sl], result.positions.loc[sl], result.benchmark.loc[sl]
    return BacktestResult(
        returns=r, positions=p, equity=(1 + r).cumprod().rename("equity"), benchmark=b,
        stats=performance_summary(r, p, periods_per_year),
        benchmark_stats=performance_summary(b, pd.Series(1.0, index=b.index), periods_per_year),
    )

