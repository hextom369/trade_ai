"""Rule-based strategies.

Each rule returns a conviction in [-1, 1] per bar using only data up to and
including that bar's close. ``rule_positions`` then applies the same sizing as
the ML strategy (smoothing, long-only, vol targeting, leverage cap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import rsi
from .strategy import StrategyConfig, size_positions

RULES = ("ma_cross", "breakout", "rsi_reversion")


@dataclass
class RuleConfig:
    fast: int = 20              # ma_cross: fast moving average
    slow: int = 100             # ma_cross: slow moving average
    breakout_window: int = 55   # breakout: Donchian entry channel
    exit_window: int = 20       # breakout: Donchian exit channel
    rsi_window: int = 14        # rsi_reversion
    rsi_low: float = 30.0       # rsi_reversion: buy below
    rsi_high: float = 70.0      # rsi_reversion: sell above
    rsi_exit: float = 50.0      # rsi_reversion: flatten when RSI crosses back here


def ma_cross(close: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """+1 when the fast MA is above the slow MA, -1 when below, 0 during warm-up."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return np.sign(f - s).fillna(0.0)


def _hold_until_exit(entry_long: pd.Series, entry_short: pd.Series,
                     exit_long: pd.Series, exit_short: pd.Series) -> pd.Series:
    """Stateful signal: enter on an entry condition, hold until the matching exit."""
    out = np.zeros(len(entry_long))
    state = 0.0
    el, es = entry_long.to_numpy(), entry_short.to_numpy()
    xl, xs = exit_long.to_numpy(), exit_short.to_numpy()
    for i in range(len(out)):
        if el[i]:
            state = 1.0
        elif es[i]:
            state = -1.0
        elif (state > 0 and xl[i]) or (state < 0 and xs[i]):
            state = 0.0
        out[i] = state
    return pd.Series(out, index=entry_long.index)


def breakout(df: pd.DataFrame, window: int = 55, exit_window: int = 20) -> pd.Series:
    """Donchian channel breakout (turtle style).

    Long when the close breaks the previous ``window``-bar high, short on a break
    of the previous low; exit on a break of the opposite ``exit_window`` channel.
    Channels use bars strictly before t, so the current bar is compared to the past.
    """
    close = df["close"]
    hi = df["high"].rolling(window).max().shift(1)
    lo = df["low"].rolling(window).min().shift(1)
    exit_lo = df["low"].rolling(exit_window).min().shift(1)
    exit_hi = df["high"].rolling(exit_window).max().shift(1)
    return _hold_until_exit(close > hi, close < lo, close < exit_lo, close > exit_hi)


def rsi_reversion(close: pd.Series, window: int = 14, low: float = 30.0,
                  high: float = 70.0, exit_level: float = 50.0) -> pd.Series:
    """Mean reversion: buy oversold, sell overbought, flatten when RSI returns to ``exit_level``."""
    r = rsi(close, window)
    return _hold_until_exit(r < low, r > high, r >= exit_level, r <= exit_level)


def rule_signal(df: pd.DataFrame, name: str, cfg: RuleConfig | None = None) -> pd.Series:
    cfg = cfg or RuleConfig()
    if name == "ma_cross":
        sig = ma_cross(df["close"], cfg.fast, cfg.slow)
    elif name == "breakout":
        sig = breakout(df, cfg.breakout_window, cfg.exit_window)
    elif name == "rsi_reversion":
        sig = rsi_reversion(df["close"], cfg.rsi_window, cfg.rsi_low, cfg.rsi_high, cfg.rsi_exit)
    else:
        raise ValueError(f"unknown rule: {name} (choose from {', '.join(RULES)})")
    return sig.rename("signal")


def rule_positions(df: pd.DataFrame, name: str, rule_cfg: RuleConfig | None = None,
                   strategy_cfg: StrategyConfig | None = None) -> pd.Series:
    """Target positions for a rule, sized like the ML strategy."""
    return size_positions(rule_signal(df, name, rule_cfg), df["close"], strategy_cfg)
