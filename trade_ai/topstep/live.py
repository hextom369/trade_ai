"""Live/paper trading loop for TopstepX.

Runs the same ORB strategy and risk manager as the backtester:
  * polls 1-minute bars, feeds each completed bar to the strategy once
  * on a fresh signal: size with RiskManager, market entry, then a
    protective stop and a take-profit limit (the bot cancels the remaining
    one when the position is flat = OCO emulation)
  * flattens before ``exit_time`` and whenever a personal loss stop is hit
  * keeps month-to-date P&L and the end-of-day balance high in a JSON file

``dry_run=True`` (the default) only logs the orders it would send. Run it
against a TopstepX *practice* account before using a Combine/funded account.
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import projectx as px
from .instruments import Instrument
from .risk import RiskConfig, RiskManager
from .rules import AccountState, TopstepRules
from .strategy import ORBStrategy

log = logging.getLogger("trade_ai.topstep.live")


@dataclass
class LiveConfig:
    symbol: str = "MES"
    account_name: str | None = None   # pick account by name; default: first tradable
    contract_id: str | None = None    # default: active front-month contract
    bar_minutes: int = 1
    poll_seconds: float = 5.0
    tz: str = "America/New_York"
    warmup_days: int = 30
    dry_run: bool = True
    mll_floor: float | None = None    # current MLL level shown in TopstepX (recommended)
    state_path: str = "topstep_state.json"


class LiveBot:
    def __init__(self, client: px.ProjectXClient, cfg: LiveConfig, strategy: ORBStrategy,
                 instrument: Instrument, rules: TopstepRules | None = None,
                 risk: RiskConfig | None = None):
        self.client = client
        self.cfg = cfg
        self.strategy = strategy
        self.inst = instrument
        self.rules = rules or TopstepRules()
        self.rm = RiskManager(risk or RiskConfig(), self.rules, instrument)
        self.state = AccountState(self.rules)
        self.account_id: int | None = None
        self.contract_id: str | None = cfg.contract_id
        self.last_bar: pd.Timestamp | None = None
        self.day: date | None = None
        self.bracket: dict[str, int] = {}

    # -- persistence -------------------------------------------------------
    def _load_state(self) -> dict:
        p = Path(self.cfg.state_path)
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_state(self) -> None:
        Path(self.cfg.state_path).write_text(json.dumps({
            "month": list(self.rm.month) if self.rm.month else None,
            "month_pnl": self.rm.month_pnl, "eod_high": self.state.eod_high,
            "day": self.day.isoformat() if self.day else None,
            "day_start_balance": self.state.day_start_balance,
        }, indent=2))

    # -- setup -------------------------------------------------------------
    def setup(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        accounts = [a for a in self.client.accounts() if a.get("canTrade", True)]
        if self.cfg.account_name:
            accounts = [a for a in accounts if a.get("name") == self.cfg.account_name]
        if not accounts:
            raise px.ProjectXError("no tradable account found")
        acct = accounts[0]
        self.account_id = int(acct["id"])
        if self.contract_id is None:
            self.contract_id = self.client.front_contract_id(self.cfg.symbol)
        balance = float(acct.get("balance", self.rules.account_size))
        self.state.balance = balance
        saved = self._load_state()
        self.state.eod_high = max(balance, float(saved.get("eod_high", balance)))
        if self.cfg.mll_floor is not None:
            self.state.eod_high = self.cfg.mll_floor + self.rules.max_loss_limit
        today = pd.Timestamp(now).tz_convert(self.cfg.tz).date()
        self.rm.new_day(today)
        if saved.get("month") == [today.year, today.month]:
            self.rm.month_pnl = float(saved.get("month_pnl", 0.0))
        self.state.day_start_balance = (float(saved["day_start_balance"])
                                        if saved.get("day") == today.isoformat() else balance)
        self.day = today
        log.info("account %s (%s) contract %s balance %.2f MLL floor %.2f month P&L %.2f",
                 acct.get("name"), self.account_id, self.contract_id, balance,
                 self.state.mll_floor, self.rm.month_pnl)
        hist = self._fetch_bars(now - timedelta(days=self.cfg.warmup_days * 1.5), now)
        for ts, row in hist.iterrows():  # warm up: decisions on old bars are ignored
            self.strategy.on_bar(ts, row.open, row.high, row.low, row.close, row.volume)
            self.last_bar = ts
        log.info("warm-up done: %d bars, avg daily range %s", len(hist),
                 self.strategy.avg_daily_range())

    def _fetch_bars(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = self.client.bars_range(self.contract_id, start, end, unit_number=self.cfg.bar_minutes)
        if df.empty:
            return df
        df.index = df.index.tz_convert(self.cfg.tz)
        done = df.index + pd.Timedelta(minutes=self.cfg.bar_minutes) <= pd.Timestamp(end)
        return df[done]

    # -- orders --------------------------------------------------------------
    def _send(self, what: str, fn, *args, **kw):
        if self.cfg.dry_run:
            log.info("[dry-run] %s", what)
            return -1
        log.info("%s", what)
        return fn(*args, **kw)

    def _position(self) -> dict | None:
        for p in self.client.open_positions(self.account_id):
            if p.get("contractId") == self.contract_id and p.get("size", 0):
                return p
        return None

    def _cancel_bracket(self) -> None:
        if self.cfg.dry_run:
            self.bracket.clear()
            return
        for o in self.client.open_orders(self.account_id):
            if o.get("contractId") == self.contract_id:
                self.client.cancel_order(self.account_id, int(o["id"]))
        self.bracket.clear()

    def flatten(self, why: str) -> None:
        log.warning("flatten: %s", why)
        self._cancel_bracket()
        self._send(f"close position ({why})", self.client.close_position,
                   self.account_id, self.contract_id)

    def _enter(self, side: int, stop: float, target_r: float, ref_price: float) -> None:
        stop = self.inst.round_price(stop)
        slip = self.rm.cfg.slippage_ticks * self.inst.tick_size
        dist = side * (ref_price + side * slip - stop)
        if dist <= 0:
            log.info("skip: price already beyond stop")
            return
        n = self.rm.position_size(dist, self.state)
        if n <= 0:
            log.info("skip: risk budget too small for stop distance %.2f", dist)
            return
        entry_side = px.SIDE_BUY if side > 0 else px.SIDE_SELL
        exit_side = px.SIDE_SELL if side > 0 else px.SIDE_BUY
        self._send(f"MARKET {'BUY' if side > 0 else 'SELL'} {n} {self.contract_id}",
                   self.client.place_order, self.account_id, self.contract_id,
                   px.ORDER_MARKET, entry_side, n, tag=None)
        self.rm.register_entry()
        entry = ref_price
        if not self.cfg.dry_run:
            for _ in range(10):
                pos = self._position()
                if pos:
                    entry = float(pos["averagePrice"])
                    break
                _time.sleep(0.5)
            else:
                log.error("entry not filled; cancelling")
                self._cancel_bracket()
                return
        target = self.inst.round_price(entry + side * target_r * abs(entry - stop))
        self.bracket["stop"] = self._send(
            f"STOP {n} @ {stop}", self.client.place_order, self.account_id, self.contract_id,
            px.ORDER_STOP, exit_side, n, stop_price=stop)
        self.bracket["target"] = self._send(
            f"LIMIT {n} @ {target}", self.client.place_order, self.account_id, self.contract_id,
            px.ORDER_LIMIT, exit_side, n, limit_price=target)
        log.info("entered %s %d @ %.2f stop %.2f target %.2f (month P&L %.2f)",
                 "long" if side > 0 else "short", n, entry, stop, target, self.rm.month_pnl)

    # -- main loop -------------------------------------------------------------
    def _sync_account(self) -> None:
        for a in self.client.accounts():
            if int(a["id"]) == self.account_id:
                new_bal = float(a.get("balance", self.state.balance))
                if new_bal != self.state.balance:
                    self.rm.on_trade_closed(new_bal - self.state.balance)
                self.state.balance = new_bal
                return

    def step(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        local = pd.Timestamp(now).tz_convert(self.cfg.tz)
        if local.date() != self.day:
            self.state.end_of_day(self.day)
            self.day = local.date()
            self.rm.new_day(self.day)
        self._sync_account()
        pos = None if self.cfg.dry_run else self._position()
        exit_time = min(self.strategy.cfg.exit_time, self.rules.flat_by)

        if pos is None and self.bracket:
            self._cancel_bracket()  # one leg filled -> remove the other
        if pos is not None:
            if local.time() >= exit_time:
                self.flatten("time exit")
            elif self.state.day_pnl <= -self.rm.cfg.daily_loss_stop:
                self.flatten("daily loss stop")

        start = (self.last_bar + pd.Timedelta(minutes=self.cfg.bar_minutes)
                 if self.last_bar is not None else local - pd.Timedelta(hours=8))
        new = self._fetch_bars(start.to_pydatetime(), now)
        if self.last_bar is not None:
            new = new[new.index > self.last_bar]
        for ts, row in new.iterrows():
            sig = self.strategy.on_bar(ts, row.open, row.high, row.low, row.close, row.volume)
            self.last_bar = ts
            fresh = ts + pd.Timedelta(minutes=2 * self.cfg.bar_minutes) >= local
            if sig is None or not fresh or pos is not None or local.time() >= exit_time:
                continue
            ok, why = self.rm.can_trade(self.state)
            if not ok:
                log.info("signal %s ignored: %s", sig.reason, why)
                continue
            self._enter(sig.side, sig.stop, sig.target_r, row.close)
        self._save_state()

    def run(self) -> None:  # pragma: no cover - network loop
        self.setup()
        log.info("bot running (dry_run=%s). Ctrl+C to stop.", self.cfg.dry_run)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("step failed; retrying")
            _time.sleep(self.cfg.poll_seconds)
