"""Intraday bar loading (CSV) and a synthetic generator for demos/tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import load_csv

DEFAULT_TZ = "America/New_York"


def to_local(df: pd.DataFrame, tz: str = DEFAULT_TZ, source_tz: str = "UTC") -> pd.DataFrame:
    """Return ``df`` with a tz-aware index in ``tz``; naive input is ``source_tz``."""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(source_tz)
    out = df.copy()
    out.index = idx.tz_convert(tz)
    out.index.name = "date"
    return out


def load_intraday_csv(path: str, tz: str = DEFAULT_TZ, source_tz: str = "UTC") -> pd.DataFrame:
    """Load 1-min/5-min OHLCV bars. Timestamps are bar *open* times."""
    return to_local(load_csv(path), tz, source_tz)


def synthetic_intraday(days: int = 250, seed: int = 0, start_price: float = 5000.0,
                       daily_range_pts: float = 50.0, trend_strength: float = 0.0,
                       tick_size: float = 0.25, start: str = "2024-01-02",
                       tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """1-minute RTH bars (09:30-16:00) for ``days`` business days.

    ``trend_strength`` > 0 makes the rest of the day drift in the direction of
    the first 15 minutes (the effect ORB tries to capture). 0 = random walk,
    on which no strategy should make money after costs.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    n = 390
    frames = []
    price = start_price
    for d in dates:
        day_scale = daily_range_pts * rng.lognormal(0.0, 0.3) / (np.sqrt(n) * 2.0)
        rets = rng.normal(0.0, day_scale, n)
        if trend_strength:
            direction = np.sign(rets[:15].sum()) or 1.0
            rets[15:] += trend_strength * day_scale * direction * 0.15
        close = price + np.cumsum(rets)
        open_ = np.concatenate([[price], close[:-1]])
        wick = np.abs(rng.normal(0, day_scale * 0.5, (2, n)))
        high = np.maximum(open_, close) + wick[0]
        low = np.minimum(open_, close) - wick[1]
        rnd = lambda x: np.round(x / tick_size) * tick_size  # noqa: E731
        idx = pd.date_range(pd.Timestamp(d.date()).replace(hour=9, minute=30), periods=n,
                            freq="1min", tz=tz)
        vol = rng.lognormal(7, 0.5, n) * (1 + np.abs(rets) / day_scale)
        frames.append(pd.DataFrame({"open": rnd(open_), "high": rnd(high), "low": rnd(low),
                                    "close": rnd(close), "volume": vol}, index=idx))
        price = close[-1] + rng.normal(0, day_scale * 3)
    out = pd.concat(frames)
    out.index.name = "date"
    return out
