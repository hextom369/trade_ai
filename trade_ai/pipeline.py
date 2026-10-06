"""End-to-end research pipeline: features -> walk-forward model -> positions -> backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, run_backtest, trim_to_active
from .features import build_features, make_labels
from .model import WalkForwardConfig, make_model, walk_forward_predict
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


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    features, labels = _prepare(df, cfg)
    proba = walk_forward_predict(features, labels, cfg.walk_forward)
    gate = _gate(proba, labels, cfg)
    positions = (positions_from_proba(proba, df["close"], cfg.strategy) * gate).rename("position")
    ppy = cfg.strategy.periods_per_year
    bt = run_backtest(df["close"], positions, cfg.cost_bps, cfg.slippage_bps, ppy)
    return PipelineResult(proba=proba, gate=gate, positions=positions,
                          backtest=trim_to_active(bt, proba, ppy))


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
