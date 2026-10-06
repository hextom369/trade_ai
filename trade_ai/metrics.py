"""Performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def performance_summary(returns: pd.Series, positions: pd.Series,
                        periods_per_year: int = 252) -> dict:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    total = float((1 + r).prod() - 1)
    years = n / periods_per_year
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    std = r.std()
    downside = r[r < 0].std()
    sharpe = float(r.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    sortino = float(r.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0
    mdd = max_drawdown(r)
    active = r[positions.reindex(r.index).fillna(0).abs() > 1e-9]
    turnover = positions.reindex(r.index).fillna(0).diff().abs().sum() / years if years else 0.0
    return {
        "total_return": total,
        "cagr": float(cagr),
        "ann_vol": float(std * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "hit_rate": float((active > 0).mean()) if len(active) else 0.0,
        "exposure": float((positions.reindex(r.index).fillna(0).abs() > 1e-9).mean()),
        "ann_turnover": float(turnover),
        "bars": n,
    }


def format_summary(stats: dict) -> str:
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "hit_rate", "exposure"}
    lines = []
    for k, v in stats.items():
        if k in pct:
            lines.append(f"  {k:<14}{v:>10.2%}")
        elif isinstance(v, float):
            lines.append(f"  {k:<14}{v:>10.2f}")
        else:
            lines.append(f"  {k:<14}{v:>10}")
    return "\n".join(lines)
