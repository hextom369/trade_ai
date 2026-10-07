"""Turn model probabilities into target positions with risk controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    entry_band: float = 0.04     # ignore |p - 0.5| below this (no-trade zone)
    full_edge: float = 0.15      # |p - 0.5| at which conviction reaches 100%
    long_only: bool = False
    target_vol: float = 0.15     # annualised volatility target; None/0 disables
    vol_window: int = 20
    max_leverage: float = 1.0
    smoothing: int = 3           # EMA span over raw signals to cut turnover; 1 = off
    quality_gate: bool = True    # scale exposure by the model's recent live hit rate
    gate_window: int = 120
    periods_per_year: int = 252
    rebalance_band: float = 0.0  # keep the current position unless the target moves more than this


def positions_from_proba(proba: pd.Series, close: pd.Series,
                         cfg: StrategyConfig | None = None) -> pd.Series:
    """Map P(up) to a position in [-max_leverage, max_leverage].

    1. Conviction = (p - 0.5) outside the no-trade band, rescaled so that
       ``full_edge`` maps to +/-1.
    2. Optionally EMA-smoothed to avoid flip-flopping.
    3. Scaled so the expected position volatility matches ``target_vol``.
    The position at row t is decided at the close of t and earns the return
    from t to t+1 (the backtester applies that shift).
    """
    cfg = cfg or StrategyConfig()
    edge = proba - 0.5
    band = cfg.entry_band
    span = max(cfg.full_edge - band, 1e-9)
    conviction = np.sign(edge) * ((edge.abs() - band) / span).clip(0, 1)
    conviction = conviction.fillna(0.0)
    return size_positions(conviction, close, cfg)


def size_positions(conviction: pd.Series, close: pd.Series,
                   cfg: StrategyConfig | None = None) -> pd.Series:
    """Turn a conviction in [-1, 1] into a position with smoothing and vol targeting.

    Shared by the ML strategy and the rule-based strategies.
    """
    cfg = cfg or StrategyConfig()
    conviction = conviction.reindex(close.index).fillna(0.0).clip(-1, 1)
    if cfg.long_only:
        conviction = conviction.clip(lower=0)
    if cfg.smoothing and cfg.smoothing > 1:
        conviction = conviction.ewm(span=cfg.smoothing, adjust=False).mean()

    if cfg.target_vol:
        realised = np.log(close).diff().rolling(cfg.vol_window).std() * np.sqrt(cfg.periods_per_year)
        scale = (cfg.target_vol / realised).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pos = conviction * scale
    else:
        pos = conviction
    pos = pos.clip(-cfg.max_leverage, cfg.max_leverage)
    if cfg.rebalance_band:
        pos = apply_rebalance_band(pos, cfg.rebalance_band)
    return pos.rename("position")


def apply_rebalance_band(target: pd.Series, band: float) -> pd.Series:
    """Hysteresis: only move to a new target when it differs from the held one by > ``band``.

    Mirrors the live trader, which skips orders smaller than its rebalance band;
    going flat (target 0) is always executed.
    """
    out = np.empty(len(target))
    held = 0.0
    for i, t in enumerate(target.to_numpy()):
        if (t == 0.0 and held != 0.0) or abs(t - held) > band:
            held = t
        out[i] = held
    return pd.Series(out, index=target.index)


def quality_gate(proba: pd.Series, labels: pd.Series, horizon: int,
                 window: int = 120, min_edge: float = 0.0, full_edge: float = 0.04) -> pd.Series:
    """Exposure multiplier in [0, 1] based on the model's recent live hit rate.

    A prediction made at t can only be scored once its label resolves at
    t + horizon, so scores are shifted by ``horizon`` before being used.
    Hit rate - 0.5 below ``min_edge`` gives 0 exposure, above ``full_edge`` gives 1.
    """
    valid = proba.notna() & labels.notna()
    hit = ((proba > 0.5).astype(float) == labels).astype(float).where(valid)
    rolling = hit.shift(horizon).rolling(window, min_periods=window // 2).mean()
    edge = rolling - 0.5
    gate = ((edge - min_edge) / max(full_edge - min_edge, 1e-9)).clip(0, 1)
    return gate.fillna(0.0).rename("gate")
