"""Event-driven bar backtester with Topstep rules.

Execution model (deliberately pessimistic):
  * A signal on bar t is filled at bar t+1's open plus ``slippage_ticks``.
  * Stops fill at the stop price (or the open if the bar gaps through it)
    minus slippage. Targets fill at the limit price only when the bar trades
    ``limit_through_ticks`` beyond it.
  * If stop and target are both inside one bar, the stop is assumed first.
  * Unrealized P&L is checked against the Maximum Loss Limit and the Daily
    Loss Limit within every bar, like Topstep's real-time liquidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import pandas as pd

from .instruments import Instrument
from .risk import RiskConfig, RiskManager
from .rules import AccountState, TopstepRules
from .strategy import ORBStrategy


@dataclass
class _Position:
    side: int
    contracts: int
    entry: float
    stop: float
    target: float
    risk_pts: float
    entry_time: pd.Timestamp
    reason: str


@dataclass
class TopstepResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    events: list[dict] = field(default_factory=list)
    resets: int = 0
    rules: TopstepRules | None = None


def run_topstep_backtest(bars: pd.DataFrame, strategy: ORBStrategy, instrument: Instrument,
                         rules: TopstepRules | None = None, risk: RiskConfig | None = None,
                         limit_through_ticks: int = 1, reset_on_breach: bool = True,
                         ) -> TopstepResult:
    rules = rules or TopstepRules()
    risk_cfg = risk or RiskConfig()
    rm = RiskManager(risk_cfg, rules, instrument)
    state = AccountState(rules)
    tick = instrument.tick_size
    slip = risk_cfg.slippage_ticks * tick
    pv = instrument.point_value
    fee = instrument.fee_per_side
    exit_time: time = min(strategy.cfg.exit_time, rules.flat_by)

    idx = bars.index
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    trades: list[dict] = []
    daily: list[dict] = []
    events: list[dict] = []
    resets = 0
    dead = False
    pos: _Position | None = None
    pending = None
    cur_day: date | None = None
    trades_today = 0
    day_acc = 0.0  # realized P&L of the current day (survives account resets)

    def close(i: int, price: float, why: str) -> None:
        nonlocal pos, day_acc
        assert pos is not None
        gross = pos.side * (price - pos.entry) * pv * pos.contracts
        state.apply_pnl(gross - fee * pos.contracts)  # entry fee was charged at entry
        day_acc += gross - fee * pos.contracts
        net = gross - 2 * fee * pos.contracts
        rm.on_trade_closed(net)
        trades.append(dict(entry_time=pos.entry_time, exit_time=idx[i], side=pos.side,
                           contracts=pos.contracts, entry=pos.entry, exit=price, stop=pos.stop,
                           target=pos.target, pnl=net, r=net / (pos.risk_pts * pv * pos.contracts),
                           reason=pos.reason, exit_reason=why))
        pos = None

    def finish_day(last_i: int) -> None:
        nonlocal trades_today, day_acc
        if pos is not None:
            close(last_i, c[last_i] - pos.side * slip, "eod")
        blocked = rm.can_trade(state)[1]
        state.end_of_day(cur_day)
        daily.append(dict(date=pd.Timestamp(cur_day), pnl=day_acc, balance=state.balance,
                          mll_floor=state.mll_floor, trades=trades_today, blocked=blocked))
        trades_today = 0
        day_acc = 0.0

    for i in range(len(idx)):
        ts = idx[i]
        d = ts.date()
        if d != cur_day:
            if cur_day is not None:
                finish_day(i - 1)
            cur_day = d
            rm.new_day(d)
            pending = None
            if dead:
                rm.halt("account failed")

        # 1) fill a pending entry at this bar's open
        if pending is not None and pos is None:
            sig, pending = pending, None
            ok, _ = rm.can_trade(state)
            entry = o[i] + sig.side * slip
            stop = instrument.round_price(sig.stop)
            dist = sig.side * (entry - stop)
            if ok and dist > 0 and ts.time() < exit_time:
                n = rm.position_size(dist, state)
                if n > 0:
                    target = instrument.round_price(entry + sig.side * sig.target_r * dist)
                    pos = _Position(sig.side, n, entry, stop, target, dist, ts, sig.reason)
                    state.apply_pnl(-fee * n)
                    day_acc -= fee * n
                    rm.register_entry()
                    trades_today += 1

        # 2) manage the open position inside this bar
        if pos is not None:
            s = pos.side
            if ts.time() >= exit_time:
                close(i, o[i] - s * slip, "time")
            else:
                adverse = l[i] if s > 0 else h[i]
                favorable = h[i] if s > 0 else l[i]
                # price where equity (after exit fee) hits the nearest hard limit
                limits = [x for x in (state.mll_floor, state.dll_level()) if x is not None]
                threshold = max(limits)
                breach_px = pos.entry + s * (threshold - state.balance + fee * pos.contracts) / (
                    pv * pos.contracts)
                stop_hit = s * (adverse - pos.stop) <= 0
                breach_hit = s * (adverse - breach_px) <= 0 and s * (breach_px - pos.stop) > 0
                if breach_hit:
                    px = breach_px if s * (o[i] - breach_px) > 0 else o[i]
                    close(i, px - s * slip, state.breach(threshold) or "MLL")
                elif stop_hit:
                    px = pos.stop if s * (o[i] - pos.stop) > 0 else o[i]
                    close(i, px - s * slip, "stop")
                elif s * (favorable - pos.target) >= limit_through_ticks * tick - 1e-9:
                    close(i, pos.target, "target")
                elif (strategy.cfg.breakeven_r is not None
                      and s * (favorable - pos.entry) >= strategy.cfg.breakeven_r * pos.risk_pts):
                    pos.stop = pos.entry if s * (pos.entry - pos.stop) > 0 else pos.stop

            # a fill at a limit or a gap through the stop can breach a hard limit
            kind = state.breach(state.balance)
            if pos is None and kind and not rm.halted_reason:
                events.append(dict(time=ts, event=kind, balance=state.balance))
                if kind == "MLL":
                    if reset_on_breach:
                        resets += 1
                        state = AccountState(rules)
                    else:
                        dead = True
                    rm.halt("max loss limit")
                else:
                    rm.halt("daily loss limit")

        # 3) strategy sees the completed bar
        sig = strategy.on_bar(ts, o[i], h[i], l[i], c[i], v[i])
        if sig is not None and pos is None and pending is None and i + 1 < len(idx) \
                and idx[i + 1].date() == d:
            pending = sig

    if cur_day is not None:
        finish_day(len(idx) - 1)

    cols = ["entry_time", "exit_time", "side", "contracts", "entry", "exit", "stop", "target",
            "pnl", "r", "reason", "exit_reason"]
    daily_df = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame(
        columns=["pnl", "balance", "mll_floor", "trades", "blocked"])
    return TopstepResult(trades=pd.DataFrame(trades, columns=cols), daily=daily_df,
                         events=events, resets=resets, rules=rules)
