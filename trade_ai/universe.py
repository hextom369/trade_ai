"""Volume-based symbol selection.

Which markets to trade is decided by traded value (quote volume = price x volume):
liquid markets have tighter spreads, less slippage and cleaner signals. Two ways
to rank:

* ``quote_volume`` - most traded value over the last ``window`` bars.
* ``surge``        - short-term traded value relative to its own longer average
                     (markets that are suddenly getting attention), among those
                     that still pass the liquidity floor.

The same ``rank_universe`` is used point-in-time in backtests (only volume known
at each rebalance bar) and on the latest bar when trading live.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .crypto_data import BINANCE_FUTURES, HttpGet, http_get_json

STABLECOINS = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "PYUSD", "EUR"}
RANK_BY = ("quote_volume", "surge")


def venue_ticker(symbol: str) -> str:
    """Binance perp symbol -> base ticker: BTCUSDT -> BTC, 1000PEPEUSDT -> PEPE."""
    base = symbol.upper().removesuffix("USDT")
    return re.sub(r"^(1000000|1000)(?=[A-Z])", "", base)


def binance_24h_quote_volume(http_get: HttpGet | None = None,
                             base_url: str = BINANCE_FUTURES) -> pd.Series:
    """Current 24h traded value (USDT) of every USDT-margined perp, largest first."""
    http_get = http_get or http_get_json
    rows = http_get(f"{base_url}/fapi/v1/ticker/24hr", {})
    vol = {r["symbol"]: float(r["quoteVolume"]) for r in rows
           if r["symbol"].endswith("USDT") and "_" not in r["symbol"]}
    return pd.Series(vol, name="quote_volume_24h", dtype=float).sort_values(ascending=False)


def prefilter(volumes: pd.Series, candidates: int, min_quote_volume: float = 0.0,
              allowed: set[str] | None = None) -> list[str]:
    """Top ``candidates`` symbols by 24h value, minus stablecoins / disallowed tickers."""
    out = []
    for sym, qv in volumes.sort_values(ascending=False).items():
        tick = venue_ticker(sym)
        if tick in STABLECOINS or qv < min_quote_volume:
            continue
        if allowed is not None and tick not in allowed:
            continue
        out.append(sym)
        if len(out) >= candidates:
            break
    return out


def quote_volume(df: pd.DataFrame) -> pd.Series:
    return df["close"] * df["volume"]


def volume_scores(frames: dict[str, pd.DataFrame], window: int = 168,
                  surge_window: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bar trailing traded value and surge ratio for every symbol (aligned frames).

    ``liquidity[t, s]`` = traded value over the ``window`` bars ending at t.
    ``surge[t, s]`` = average traded value of the last ``surge_window`` bars divided
    by the ``window`` average. Both only use bars up to and including t.
    """
    qv = pd.DataFrame({s: quote_volume(df) for s, df in frames.items()}).sort_index()
    liquidity = qv.rolling(window, min_periods=window).sum()
    short = qv.rolling(surge_window, min_periods=surge_window).mean()
    surge = short / (liquidity / window).replace(0, np.nan)
    return liquidity, surge


def rank_universe(liquidity: pd.Series, surge: pd.Series, top_n: int,
                  rank_by: str = "quote_volume", min_window_value: float = 0.0) -> list[str]:
    """Pick ``top_n`` symbols from one row of scores."""
    ok = liquidity.dropna()
    ok = ok[ok >= min_window_value]
    if rank_by == "quote_volume":
        ranked = ok.sort_values(ascending=False)
    elif rank_by == "surge":
        ranked = surge.reindex(ok.index).dropna().sort_values(ascending=False)
    else:
        raise ValueError(f"rank_by must be one of {RANK_BY}")
    return list(ranked.index[:top_n])


def point_in_time_universe(frames: dict[str, pd.DataFrame], top_n: int = 5,
                           window: int = 168, surge_window: int = 24,
                           rebalance_every: int = 24, rank_by: str = "quote_volume",
                           min_window_value: float = 0.0) -> pd.DataFrame:
    """Boolean membership (bars x symbols), re-ranked every ``rebalance_every`` bars.

    Selection at bar t uses volume up to t's close, and positions decided at t's
    close are only held from t+1 (the backtester's shift), so no future volume leaks.
    """
    liquidity, surge = volume_scores(frames, window, surge_window)
    members = pd.DataFrame(False, index=liquidity.index, columns=liquidity.columns)
    current: list[str] = []
    for i, t in enumerate(liquidity.index):
        if i % rebalance_every == 0:
            current = rank_universe(liquidity.loc[t], surge.loc[t], top_n, rank_by,
                                    min_window_value)
        if current:
            members.loc[t, current] = True
    # A symbol whose candle is missing at t cannot be traded at t.
    present = pd.DataFrame({s: df["close"].notna() for s, df in frames.items()})
    return members & present.reindex(members.index).fillna(False).astype(bool)
