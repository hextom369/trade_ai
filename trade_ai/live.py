"""Live / paper trading loop with risk controls.

Each cycle, after a candle closes:
  1. load recent closed candles,
  2. compute the target position (fraction of equity) with the chosen strategy,
  3. run the safety checks (stale data, daily-loss kill switch, size caps),
  4. send the order needed to move from the current to the target position
     (or only log it in dry-run mode).
Every decision is appended to a JSONL log.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .brokers import Broker
from .crypto_data import interval_to_timedelta
from .pipeline import PipelineConfig, latest_target
from .rules import RuleConfig

log = logging.getLogger("trade_ai.live")


@dataclass
class RiskLimits:
    max_abs_position: float = 1.0      # hard cap on |target| as a fraction of equity
    max_notional: float | None = None  # hard cap on position notional (quote currency)
    min_trade_notional: float = 10.0   # skip orders smaller than this
    rebalance_band: float = 0.05       # skip if |target - current| < this fraction of equity
    max_daily_loss: float = 0.05       # flatten and halt for the UTC day below -5%


@dataclass
class TraderConfig:
    data_symbol: str                   # symbol for candles, e.g. BTCUSDT
    venue_symbol: str                  # symbol at the venue, e.g. BTC
    interval: str = "1h"
    strategy: str = "ma_cross"         # "ml" or a rule name
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    dry_run: bool = True
    state_dir: str = "trade_state"
    top_n: int = 1                     # equity slices when trading a universe


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


class Trader:
    def __init__(self, broker: Broker, cfg: TraderConfig,
                 now: Callable[[], pd.Timestamp] = _now):
        self.broker = broker
        self.cfg = cfg
        self.now = now
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.risk_path = os.path.join(cfg.state_dir, "risk.json")
        self.log_path = os.path.join(cfg.state_dir, "decisions.jsonl")
        self.risk_state = self._load_risk()

    # -- persistence -------------------------------------------------------
    def _load_risk(self) -> dict:
        if os.path.exists(self.risk_path):
            with open(self.risk_path) as f:
                return json.load(f)
        return {}

    def _save_risk(self) -> None:
        with open(self.risk_path, "w") as f:
            json.dump(self.risk_state, f)

    def _record(self, decision: dict) -> dict:
        decision = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **decision}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(decision, default=str) + "\n")
        log.info("%s", decision)
        return decision

    # -- one cycle ---------------------------------------------------------
    def _daily_loss_halt(self, equity: float) -> bool:
        day = str(self.now().date())
        if self.risk_state.get("day") != day:
            self.risk_state = {**self.risk_state, "day": day, "start_equity": equity,
                               "halted": False}
        start = self.risk_state["start_equity"]
        if not self.risk_state["halted"] and start > 0 and \
                equity < start * (1 - self.cfg.risk.max_daily_loss):
            self.risk_state["halted"] = True
        self._save_risk()
        return self.risk_state["halted"]

    def step(self, df: pd.DataFrame, symbol: str | None = None, weight: float = 1.0) -> dict:
        """Move ``symbol`` (default: the configured venue symbol) to its target.

        ``weight`` is the slice of equity this symbol may use (1 / number of symbols
        when trading a universe).
        """
        cfg, risk = self.cfg, self.cfg.risk
        sym = symbol or cfg.venue_symbol
        bar = interval_to_timedelta(cfg.interval)
        last_close_time = df.index[-1] + bar
        age = self.now() - last_close_time
        if age > bar:
            return self._record({"symbol": sym, "action": "skip",
                                 "reason": f"stale data: last bar closed {age} ago"})

        sig = latest_target(df, cfg.strategy, cfg.pipeline, cfg.rules)
        target_frac = max(-risk.max_abs_position, min(risk.max_abs_position, sig["target_position"]))
        target_frac *= weight

        price = self.broker.price(sym)
        equity = self.broker.equity()
        current = self.broker.position(sym)
        base = {"symbol": sym, "strategy": cfg.strategy, "weight": weight,
                "bar": str(df.index[-1]), "signal": sig, "price": price, "equity": equity,
                "position": current}

        if self._daily_loss_halt(equity):
            target_frac, base["halted"] = 0.0, True

        target_qty = target_frac * equity / price
        if risk.max_notional is not None:
            cap = risk.max_notional / price
            target_qty = max(-cap, min(cap, target_qty))
        delta = target_qty - current
        base.update(target_fraction=target_frac, target_qty=target_qty, order_qty=delta)

        notional = abs(delta) * price
        flattening = target_qty == 0 and current != 0
        return self._execute(sym, delta, notional, equity, flattening, base)

    def _execute(self, sym: str, delta: float, notional: float, equity: float,
                 flattening: bool, base: dict) -> dict:
        risk = self.cfg.risk
        if notional < risk.min_trade_notional or \
                (not flattening and equity > 0 and notional / equity < risk.rebalance_band):
            return self._record({**base, "action": "hold", "reason": "change below threshold"})
        if self.cfg.dry_run:
            return self._record({**base, "action": "dry_run"})
        fill = self.broker.market_order(sym, delta)
        return self._record({**base, "action": "order", "fill": asdict(fill)})

    def step_universe(self, frames: dict[str, pd.DataFrame]) -> list[dict]:
        """Trade every symbol in ``frames`` (venue symbol -> candles) with an equal slice
        of equity, and close positions in symbols that dropped out of the universe."""
        decisions = []
        previous = self.risk_state.get("universe", [])
        for sym in previous:
            if sym in frames:
                continue
            qty = self.broker.position(sym)
            if qty:
                price = self.broker.price(sym)
                base = {"symbol": sym, "reason": "left universe", "position": qty,
                        "target_qty": 0.0, "order_qty": -qty, "price": price}
                decisions.append(self._execute(sym, -qty, abs(qty) * price,
                                               self.broker.equity(), True, base))
        weight = 1.0 / max(self.cfg.top_n, len(frames), 1)
        for sym, df in frames.items():
            decisions.append(self.step(df, sym, weight))
        self.risk_state["universe"] = list(frames)
        self._save_risk()
        return decisions


def run_loop(trader: Trader, load: Callable[[], pd.DataFrame | dict[str, pd.DataFrame]],
             once: bool = False, delay_s: float = 5.0, max_errors: int = 5) -> None:
    """Run the trader right after every candle close, forever (or once).

    ``load`` returns one DataFrame (single symbol) or a dict of venue symbol ->
    DataFrame (volume-selected universe).
    """
    bar = interval_to_timedelta(trader.cfg.interval)
    errors = 0
    while True:
        try:
            data = load()
            if isinstance(data, dict):
                trader.step_universe(data)
            else:
                trader.step(data)
            errors = 0
        except NotImplementedError:
            raise
        except Exception as exc:  # network hiccups etc.: log, retry next bar
            errors += 1
            log.exception("cycle failed (%d/%d)", errors, max_errors)
            trader._record({"action": "error", "error": repr(exc)})
            if once or errors >= max_errors:
                raise
        if once:
            return
        now = trader.now()
        next_close = now.floor(bar) + bar
        time.sleep(max((next_close - now).total_seconds(), 0) + delay_s)
