"""OHLCV data loading.

All loaders return a DataFrame indexed by a sorted, unique DatetimeIndex with
lower-case columns: open, high, low, close, volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[REQUIRED_COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df[["open", "high", "low"]] = df[["open", "high", "low"]].apply(
        lambda s: s.fillna(df["close"])
    )
    return df


def load_csv(path: str, date_column: str | None = None) -> pd.DataFrame:
    """Load OHLCV from a CSV. The date column is auto-detected if not given."""
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    if date_column is None:
        for cand in ("date", "datetime", "timestamp", "time"):
            if cand in cols:
                date_column = cols[cand]
                break
    if date_column is None:
        raise ValueError("could not find a date column; pass date_column=")
    raw.index = pd.to_datetime(raw.pop(date_column))
    raw.index.name = "date"
    return _normalize(raw)


def load_yfinance(ticker: str, start: str | None = None, end: str | None = None,
                  interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV via yfinance (optional dependency: pip install trade-ai[data])."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is not installed: pip install 'trade-ai[data]'") from exc
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"no data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "date"
    return _normalize(raw)


def synthetic_ohlcv(n: int = 2000, seed: int = 0, drift: float = 0.0002,
                    vol: float = 0.01, regime_strength: float = 0.3,
                    start: str = "2015-01-01") -> pd.DataFrame:
    """Generate a random-walk price series with mild, regime-switching momentum.

    Useful for offline demos and tests. ``regime_strength`` controls how much
    yesterday's return leaks into today's (0 = pure random walk).
    """
    rng = np.random.default_rng(seed)
    regime = np.sign(np.sin(np.arange(n) / 90.0) + 1e-9)
    noise = rng.normal(0.0, vol, n)
    rets = np.empty(n)
    rets[0] = noise[0]
    for t in range(1, n):
        rets[t] = drift + regime_strength * regime[t] * rets[t - 1] + noise[t]
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(13, 0.4, n) * (1 + 20 * np.abs(rets))
    idx = pd.bdate_range(start=start, periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)
