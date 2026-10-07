"""Multi-symbol backtest on a volume-selected universe.

Each selected symbol gets an equal slice (1 / top_n) of equity and is traded by
the chosen strategy with the usual sizing; symbols outside the universe are flat.
Empty slots stay in cash, so exposure never exceeds the single-symbol limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .metrics import performance_summary
from .pipeline import PipelineConfig, rule_warmup, run_pipeline
from .rules import RuleConfig, rule_positions


@dataclass
class PortfolioResult:
    backtest: BacktestResult
    membership: pd.DataFrame        # bars x symbols, True when selected
    contributions: pd.DataFrame     # bars x symbols, net return contribution
    positions: pd.DataFrame         # bars x symbols, position held (fraction of equity)


def strategy_positions(df: pd.DataFrame, strategy: str, cfg: PipelineConfig,
                       rule_cfg: RuleConfig) -> pd.Series:
    if strategy == "ml":
        return run_pipeline(df, cfg).positions.fillna(0.0)
    return rule_positions(df, strategy, rule_cfg, cfg.strategy)


def run_portfolio_backtest(frames: dict[str, pd.DataFrame], membership: pd.DataFrame,
                           top_n: int, strategy: str = "ma_cross",
                           cfg: PipelineConfig | None = None,
                           rule_cfg: RuleConfig | None = None,
                           funding: dict[str, pd.Series] | None = None) -> PortfolioResult:
    cfg = cfg or PipelineConfig()
    rule_cfg = rule_cfg or RuleConfig()
    ppy = cfg.strategy.periods_per_year
    index = membership.index
    contrib, held = {}, {}
    for sym, df in frames.items():
        if sym not in membership or not membership[sym].any():
            continue
        target = strategy_positions(df, strategy, cfg, rule_cfg)
        target = target * membership[sym].reindex(df.index).fillna(False).astype(float) / top_n
        f = funding.get(sym) if funding else None
        bt = run_backtest(df["close"], target, cfg.cost_bps, cfg.slippage_bps, ppy, f)
        contrib[sym] = bt.returns.reindex(index).fillna(0.0)
        held[sym] = bt.positions.reindex(index).fillna(0.0)
    contributions = pd.DataFrame(contrib, index=index).fillna(0.0)
    positions = pd.DataFrame(held, index=index).fillna(0.0)

    # Score only once the universe exists and the strategy has warmed up.
    warmup = (cfg.walk_forward.min_train + cfg.walk_forward.horizon if strategy == "ml"
              else rule_warmup(strategy, rule_cfg, cfg))
    selected = membership.any(axis=1)
    first = selected[selected].index
    start_pos = min(index.get_loc(first[0]) + warmup, len(index) - 1) if len(first) else 0
    sl = slice(index[start_pos], None)

    r = contributions.sum(axis=1).loc[sl].rename("strategy")
    gross = positions.abs().sum(axis=1).loc[sl].rename("position")
    bench = pd.DataFrame({s: df["close"].pct_change() for s, df in frames.items()}) \
        .reindex(index).loc[sl]
    bench = bench.where(membership.shift(1).fillna(False).astype(bool).loc[sl]).mean(axis=1) \
        .fillna(0.0).rename("universe_equal_weight")
    bt = BacktestResult(
        returns=r, positions=gross, equity=(1 + r).cumprod().rename("equity"), benchmark=bench,
        stats=performance_summary(r, gross, ppy),
        benchmark_stats=performance_summary(bench, pd.Series(1.0, index=bench.index), ppy),
    )
    return PortfolioResult(bt, membership, contributions.loc[sl], positions.loc[sl])
