"""End-to-end research pipeline: features -> walk-forward model -> positions -> backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, run_backtest, trim_to_active
from .features import build_features, make_labels
from .model import WalkForwardConfig, make_model, walk_forward_predict
from .rules import RuleConfig, rule_positions
from .strategy import StrategyConfig, positions_from_proba, quality_gate


@dataclass
class PipelineConfig:
    label_threshold: float = 0.0
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    cost_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass
class PipelineResult:
    proba: pd.Series
    gate: pd.Series
    positions: pd.Series
    backtest: BacktestResult


def _prepare(df: pd.DataFrame, cfg: PipelineConfig):
    features = build_features(df)
    labels = make_labels(df, horizon=cfg.walk_forward.horizon, threshold=cfg.label_threshold)
    return features, labels


def _gate(proba: pd.Series, labels: pd.Series, cfg: PipelineConfig) -> pd.Series:
    if not cfg.strategy.quality_gate:
        return pd.Series(1.0, index=proba.index, name="gate")
    return quality_gate(proba, labels, cfg.walk_forward.horizon, cfg.strategy.gate_window)


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig | None = None,
                 funding: pd.Series | None = None) -> PipelineResult:
    """ML strategy. ``funding`` is the per-bar perp funding rate (see crypto_data.funding_per_bar)."""
    cfg = cfg or PipelineConfig()
    features, labels = _prepare(df, cfg)
    proba = walk_forward_predict(features, labels, cfg.walk_forward)
    gate = _gate(proba, labels, cfg)
    positions = (positions_from_proba(proba, df["close"], cfg.strategy) * gate).rename("position")
    ppy = cfg.strategy.periods_per_year
    bt = run_backtest(df["close"], positions, cfg.cost_bps, cfg.slippage_bps, ppy, funding)
    return PipelineResult(proba=proba, gate=gate, positions=positions,
                          backtest=trim_to_active(bt, proba, ppy))


def rule_warmup(rule: str, rule_cfg: RuleConfig, cfg: PipelineConfig) -> int:
    """Bars needed before the rule (and the vol-target estimate) is fully formed."""
    need = {"ma_cross": rule_cfg.slow, "breakout": rule_cfg.breakout_window + 1,
            "rsi_reversion": rule_cfg.rsi_window * 5}.get(rule, 0)
    return max(need, cfg.strategy.vol_window + 1)


def run_rule_pipeline(df: pd.DataFrame, rule: str, rule_cfg: RuleConfig | None = None,
                      cfg: PipelineConfig | None = None,
                      funding: pd.Series | None = None) -> PipelineResult:
    """Backtest a rule-based strategy with the same sizing, costs and funding as the ML one.

    Statistics cover the bars after the warm-up period only.
    """
    rule_cfg = rule_cfg or RuleConfig()
    cfg = cfg or PipelineConfig()
    ppy = cfg.strategy.periods_per_year
    positions = rule_positions(df, rule, rule_cfg, cfg.strategy)
    bt = run_backtest(df["close"], positions, cfg.cost_bps, cfg.slippage_bps, ppy, funding)
    active = pd.Series(float("nan"), index=df.index)
    active.iloc[min(rule_warmup(rule, rule_cfg, cfg), len(df) - 1):] = 1.0
    nan = pd.Series(float("nan"), index=df.index)
    return PipelineResult(proba=nan, gate=pd.Series(1.0, index=df.index, name="gate"),
                          positions=positions, backtest=trim_to_active(bt, active, ppy))


def latest_target(df: pd.DataFrame, strategy: str = "ml", cfg: PipelineConfig | None = None,
                  rule_cfg: RuleConfig | None = None) -> dict:
    """Target position (fraction of equity) to hold over the next bar, for any strategy."""
    cfg = cfg or PipelineConfig()
    if strategy == "ml":
        return {"strategy": "ml", **latest_signal(df, cfg)}
    pos = rule_positions(df, strategy, rule_cfg or RuleConfig(), cfg.strategy)
    last = df.index[-1]
    return {"strategy": strategy, "date": str(last), "close": float(df["close"].iloc[-1]),
            "target_position": float(pos.iloc[-1])}


def latest_signal(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> dict:
    """Target position for the next bar.

    The quality gate is computed from genuine walk-forward history; the final
    probability comes from a model refit on all labelled data.
    """
    cfg = cfg or PipelineConfig()
    wf = cfg.walk_forward
    features, labels = _prepare(df, cfg)
    proba = walk_forward_predict(features, labels, wf)
    gate = _gate(proba, labels, cfg)

    train_idx = labels.dropna().index
    if wf.train_window:
        train_idx = train_idx[-wf.train_window:]
    model = make_model(wf.model, wf.seed).fit(features.loc[train_idx].to_numpy(float),
                                              labels.loc[train_idx].astype(int).to_numpy())
    proba.iloc[-1] = model.predict_proba(features.iloc[[-1]].to_numpy(float))[0, 1]
    pos = positions_from_proba(proba, df["close"], cfg.strategy) * gate
    last = df.index[-1]
    return {
        "date": str(last.date()) if hasattr(last, "date") else str(last),
        "close": float(df["close"].iloc[-1]),
        "proba_up": float(proba.iloc[-1]),
        "quality_gate": float(gate.iloc[-1]),
        "target_position": float(pos.iloc[-1]),
    }
