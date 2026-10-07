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


# --- volume indicators -----------------------------------------------------

def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume of bar t divided by the average of the ``window`` bars before it."""
    return volume / volume.rolling(window).mean().shift(1).replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume: cumulative volume signed by the bar's direction."""
    return (np.sign(close.diff()).fillna(0.0) * volume).cumsum()


def obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Net signed volume over ``window`` bars as a share of total volume, in [-1, 1]."""
    return (obv(close, volume).diff(window) / volume.rolling(window).sum().replace(0, np.nan))


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-weighted average of the typical price over the last ``window`` bars."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return ((typical * df["volume"]).rolling(window).sum()
            / df["volume"].rolling(window).sum().replace(0, np.nan))


def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Where closes sit inside their bars, weighted by volume, in [-1, 1]."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng).fillna(0.0)
    return ((mfm * df["volume"]).rolling(window).sum()
            / df["volume"].rolling(window).sum().replace(0, np.nan))


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Volume-weighted RSI, 0..100."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    flow = typical * df["volume"]
    up = flow.where(typical.diff() > 0, 0.0).rolling(window).sum()
    down = flow.where(typical.diff() < 0, 0.0).rolling(window).sum()
    return (100 - 100 / (1 + up / down.replace(0, np.nan))).fillna(50.0)


def taker_buy_ratio(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Share of volume bought by takers (market buys) over ``window`` bars; 0.5 = balanced."""
    return (df["taker_buy_volume"].rolling(window).sum()
            / df["volume"].rolling(window).sum().replace(0, np.nan))


def _add_volume_features(f: pd.DataFrame, df: pd.DataFrame, logret: pd.Series,
                         logvol: pd.Series) -> None:
    """Volume and order-flow features (scale-free), added to ``f`` in place."""
    close, volume = df["close"], df["volume"]
    f["rel_vol_1"] = np.log(relative_volume(volume, 20))
    f["rel_vol_5_50"] = np.log(volume.rolling(5).mean() / volume.rolling(50).mean())
    f["obv_slope_10"] = obv_slope(close, volume, 10)
    f["obv_slope_40"] = obv_slope(close, volume, 40)
    for w in (20, 60):
        f[f"vwap_dist_{w}"] = close / rolling_vwap(df, w) - 1
    f["cmf_20"] = chaikin_money_flow(df, 20)
    f["mfi_14"] = money_flow_index(df, 14) / 100 - 0.5
    f["pv_corr_20"] = logret.rolling(20).corr(logvol.diff())
    # Big move on big volume is more informative than the same move on thin volume.
    f["ret1_x_relvol"] = f["ret_1"] / f["vol_20"] * f["rel_vol_1"].clip(-3, 3)
    if "taker_buy_volume" in df.columns:
        for w in (1, 5, 20):
            f[f"taker_buy_{w}"] = taker_buy_ratio(df, w) - 0.5


def build_features(df: pd.DataFrame, volume_features: bool = True) -> pd.DataFrame:
    """Build a scale-free feature matrix (ratios / z-scores, not raw prices).

    ``volume_features=False`` drops the volume / order-flow block, which makes it
    easy to measure how much volume actually adds on a given market.
    """
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
    if volume_features:
        _add_volume_features(f, df, logret, logvol)

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
