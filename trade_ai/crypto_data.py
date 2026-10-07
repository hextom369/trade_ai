"""Crypto perpetual-futures data: OHLCV candles and funding rates.

Variational Omni prices its perps off the major venues, so public Binance
USDⓈ-M futures data is a good proxy for research and for driving signals.
Only closed candles are returned, so the last row is always a finished bar.

All functions take an optional ``http_get(url, params) -> json`` so they can be
used with a custom client (or a fake one in tests).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import _normalize

BINANCE_FUTURES = "https://fapi.binance.com"
HttpGet = Callable[[str, dict], Any]

_UNIT = {"m": "min", "h": "h", "d": "D", "w": "W"}


def http_get_json(url: str, params: dict | None = None, timeout: float = 15.0) -> Any:
    """Minimal JSON GET using the standard library (honours HTTPS_PROXY)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "trade_ai"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """'1m', '15m', '1h', '4h', '1d', '1w' -> Timedelta."""
    n, unit = int(interval[:-1]), interval[-1]
    if unit not in _UNIT:
        raise ValueError(f"unsupported interval: {interval}")
    return pd.Timedelta(n, _UNIT[unit])


def periods_per_year(interval: str) -> int:
    """Bars per year for a market that trades 24/7."""
    return int(round(pd.Timedelta(days=365) / interval_to_timedelta(interval)))


def _ms(ts) -> int:
    """Timestamp (naive = UTC) -> epoch milliseconds."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int(t.value // 1_000_000)


def load_binance_klines(symbol: str, interval: str = "1h", start: str | None = None,
                        end: str | None = None, limit: int = 1500,
                        http_get: HttpGet | None = None, now_ms: int | None = None,
                        base_url: str = BINANCE_FUTURES) -> pd.DataFrame:
    """Download closed perpetual-futures candles, paginating forward from ``start``.

    Without ``start`` the most recent ``limit`` candles are returned. The index is
    each bar's open time (UTC, tz-naive); the close of bar t is known at open of t+1.
    """
    http_get = http_get or http_get_json
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    params: dict = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    end_ms = _ms(end) if end else None
    if end_ms is not None:
        params["endTime"] = end_ms
    rows: list = []
    if start is None:
        rows = list(http_get(f"{base_url}/fapi/v1/klines", params))
    else:
        cursor = _ms(start)
        while True:
            batch = http_get(f"{base_url}/fapi/v1/klines", {**params, "startTime": cursor})
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if len(batch) < limit or (end_ms is not None and last_open >= end_ms):
                break
            cursor = last_open + 1
    rows = [r for r in rows if int(r[6]) < now_ms]  # drop the still-forming candle
    if not rows:
        raise ValueError(f"no candles returned for {symbol} {interval}")
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["open_time", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("open_time").astype("int64"), unit="ms")
    df.index.name = "date"
    return _normalize(df)


def load_binance_funding(symbol: str, start=None, end=None, limit: int = 1000,
                         http_get: HttpGet | None = None,
                         base_url: str = BINANCE_FUTURES) -> pd.Series:
    """Historical funding events: Series of rates indexed by settlement time (UTC)."""
    http_get = http_get or http_get_json
    params: dict = {"symbol": symbol.upper(), "limit": limit}
    if end is not None:
        params["endTime"] = _ms(end)
    rows: list = []
    cursor = _ms(start) if start is not None else None
    while True:
        p = dict(params)
        if cursor is not None:
            p["startTime"] = cursor
        batch = http_get(f"{base_url}/fapi/v1/fundingRate", p)
        if not batch:
            break
        rows.extend(batch)
        if cursor is None or len(batch) < limit:
            break
        cursor = int(batch[-1]["fundingTime"]) + 1
    if not rows:
        return pd.Series(dtype=float, name="funding")
    s = pd.Series([float(r["fundingRate"]) for r in rows],
                  index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms"),
                  name="funding")
    return s[~s.index.duplicated()].sort_index()


def funding_per_bar(events: pd.Series, bar_index: pd.DatetimeIndex,
                    bar_length: pd.Timedelta | None = None) -> pd.Series:
    """Sum funding events into the bar (indexed by open time) during which they settle.

    The position held over bar t is the one decided at the close of bar t-1, so
    an event at time T inside [open_t, open_{t+1}) is charged to bar t's position.
    """
    out = pd.Series(0.0, index=bar_index, name="funding")
    if bar_length is not None and len(bar_index):
        events = events[events.index < bar_index[-1] + bar_length]
    if events.empty or len(bar_index) == 0:
        return out
    pos = np.searchsorted(bar_index.values, events.index.values, side="right") - 1
    ok = pos >= 0
    sums = pd.Series(events.to_numpy()[ok]).groupby(pos[ok]).sum()
    out.iloc[sums.index.to_numpy()] = sums.to_numpy()
    return out
