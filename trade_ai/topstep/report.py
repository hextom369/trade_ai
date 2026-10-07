"""Monthly performance report and Monte Carlo of the Topstep limits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import TopstepResult
from .rules import TopstepRules


def monthly_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["pnl", "days", "trade_days", "win_days", "best_day",
                                     "worst_day"])
    idx = pd.DatetimeIndex(daily.index)
    g = daily.assign(traded=daily["trades"] > 0,
                     won=daily["pnl"] > 0).groupby(idx.to_period("M"))
    return pd.DataFrame({
        "pnl": g["pnl"].sum(),
        "days": g["pnl"].size(),
        "trade_days": g["traded"].sum(),
        "win_days": g["won"].sum(),
        "best_day": g["pnl"].max(),
        "worst_day": g["pnl"].min(),
    })


def summarize(res: TopstepResult, monthly_target: float = 1_000.0) -> dict:
    t, d = res.trades, res.daily
    m = monthly_table(d)
    out: dict = {"trades": int(len(t)), "account_resets (MLL breaches)": res.resets}
    if len(t):
        wins = t["pnl"] > 0
        gross_win = t.loc[wins, "pnl"].sum()
        gross_loss = -t.loc[~wins, "pnl"].sum()
        out.update({
            "win_rate": float(wins.mean()),
            "avg_win": float(t.loc[wins, "pnl"].mean()) if wins.any() else 0.0,
            "avg_loss": float(t.loc[~wins, "pnl"].mean()) if (~wins).any() else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "expectancy_per_trade": float(t["pnl"].mean()),
            "avg_r": float(t["r"].mean()),
        })
    if len(d):
        cum = d["pnl"].cumsum()
        out.update({
            "total_pnl": float(cum.iloc[-1]),
            "max_drawdown": float((cum - cum.cummax()).min()),
            "worst_day": float(d["pnl"].min()),
            "dll_hits": int(sum(e["event"] == "DLL" for e in res.events)),
        })
    if len(m):
        out.update({
            "months": int(len(m)),
            "avg_month": float(m["pnl"].mean()),
            "median_month": float(m["pnl"].median()),
            "worst_month": float(m["pnl"].min()),
            "positive_months": float((m["pnl"] > 0).mean()),
            f"months_>=_{monthly_target:.0f}": float((m["pnl"] >= monthly_target).mean()),
        })
    return out


def monte_carlo(daily_pnl: pd.Series, rules: TopstepRules | None = None, n_paths: int = 5000,
                horizon_days: int = 21, monthly_target: float = 1_000.0, block: int = 5,
                seed: int = 0) -> dict:
    """Resample daily P&L (moving blocks) into ``horizon_days`` paths.

    Reports how often a month ends >= ``monthly_target`` and how often the
    trailing Maximum Loss Limit would be hit, starting from a fresh account.
    """
    rules = rules or TopstepRules()
    x = np.asarray(daily_pnl, float)
    if len(x) < block:
        raise ValueError("not enough days for Monte Carlo")
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon_days // block)
    starts = rng.integers(0, len(x) - block + 1, (n_paths, n_blocks))
    paths = x[starts[..., None] + np.arange(block)].reshape(n_paths, -1)[:, :horizon_days]
    bal = rules.account_size + np.cumsum(paths, axis=1)
    eod_high = np.maximum.accumulate(np.maximum(bal, rules.account_size), axis=1)
    floor_prev = np.minimum(np.concatenate([np.full((n_paths, 1), rules.account_size),
                                            eod_high[:, :-1]], axis=1) - rules.max_loss_limit,
                            rules.account_size)
    breached = (bal <= floor_prev).any(axis=1)
    final = bal[:, -1] - rules.account_size
    return {
        "p_month_>=_target": float(((final >= monthly_target) & ~breached).mean()),
        "p_month_positive": float((final > 0).mean()),
        "p_mll_breach": float(breached.mean()),
        "month_pnl_p10": float(np.percentile(final, 10)),
        "month_pnl_p50": float(np.percentile(final, 50)),
        "month_pnl_p90": float(np.percentile(final, 90)),
    }


def format_report(summary: dict) -> str:
    lines = []
    for k, v in summary.items():
        if isinstance(v, float):
            if k.startswith(("win_rate", "positive_months", "months_>=", "p_")):
                lines.append(f"  {k:32s} {v:9.1%}")
            else:
                lines.append(f"  {k:32s} {v:9.2f}")
        else:
            lines.append(f"  {k:32s} {v:>9}")
    return "\n".join(lines)
