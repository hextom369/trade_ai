"""Feature engineering and labelling.

Every feature at row ``t`` uses only data available at the close of bar ``t``.
Labels look forward and are therefore only used for training, never as inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a scale-free feature matrix (ratios / z-scores, not raw prices)."""
    close, volume = df["close"], df["volume"]
    logret = np.log(close).diff()
    f = pd.DataFrame(index=df.index)

    for w in (1, 2, 5, 10, 20, 60):
        f[f"ret_{w}"] = np.log(close / close.shift(w))
    for w in (10, 20, 60):
        f[f"vol_{w}"] = logret.rolling(w).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_60"]

    # Regime: positive lag-1 autocorrelation = trending, negative = mean reverting.
    lagged = logret.shift(1)
    for w in (20, 60):
        f[f"autocorr_{w}"] = logret.rolling(w).corr(lagged)
    f["ret1_x_autocorr"] = f["ret_1"] / f["vol_20"] * f["autocorr_60"]

    for w in (10, 20, 50, 200):
        f[f"dist_sma_{w}"] = close / close.rolling(w).mean() - 1
    f["sma_cross"] = close.rolling(20).mean() / close.rolling(50).mean() - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    f["rsi_14"] = rsi(close, 14) / 100 - 0.5
    f["rsi_5"] = rsi(close, 5) / 100 - 0.5

    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    f["bb_pos"] = (close - mid) / (2 * sd)

    f["atr_pct"] = atr(df) / close
    f["range_pct"] = (df["high"] - df["low"]) / close
    f["close_loc"] = ((close - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)).fillna(0.5)
    f["gap"] = np.log(df["open"] / close.shift(1))

    logvol = np.log1p(volume)
    f["volume_z"] = (logvol - logvol.rolling(20).mean()) / logvol.rolling(20).std()

    roll_max = close.rolling(60).max()
    roll_min = close.rolling(60).min()
    f["drawdown_60"] = close / roll_max - 1
    f["stoch_60"] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    if isinstance(df.index, pd.DatetimeIndex):
        f["dow"] = df.index.dayofweek / 4.0 - 0.5

    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0,
                vol_scaled: bool = True) -> pd.Series:
    """Binary label: 1 if the forward ``horizon``-bar log return beats the threshold.

    With ``vol_scaled`` the threshold is expressed in units of recent daily
    volatility * sqrt(horizon), so it adapts across calm and volatile regimes.
    Rows whose forward return is unknown get NaN.
    """
    close = df["close"]
    fwd = np.log(close.shift(-horizon) / close)
    thr = threshold
    if vol_scaled and threshold:
        thr = threshold * np.log(close).diff().rolling(20).std() * np.sqrt(horizon)
    label = (fwd > thr).astype(float)
    label[fwd.isna()] = np.nan
    return label
