"""Position sizing and personal risk limits that sit inside Topstep's limits.

The goal of ~$1,000/month on a 50K account is roughly $50 per trading day,
so the defaults risk little per trade and stop for the day early. The
Topstep limits ($1,000 DLL, $2,000 trailing MLL) are hard walls; the limits
here keep the bot well away from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .instruments import Instrument
from .rules import AccountState, TopstepRules


@dataclass
class RiskConfig:
    risk_per_trade: float = 150.0          # USD lost if the stop is hit (incl. fees/slippage)
    daily_loss_stop: float = 300.0         # stop for the day after losing this much
    daily_profit_target: float | None = 300.0  # stop for the day after making this much
    max_trades_per_day: int = 2
    monthly_target: float | None = 1_000.0
    # What to do once the month's profit reaches ``monthly_target``:
    # "stop" = no more trades this month, "reduce" = scale risk by
    # ``reduce_factor``, "continue" = ignore.
    after_monthly_target: str = "reduce"
    reduce_factor: float = 0.5
    # Never risk money that would bring equity closer than this to the MLL floor.
    mll_buffer: float = 300.0
    # Keep this much room above the Daily Loss Limit.
    dll_buffer: float = 100.0
    max_contracts: int | None = None
    slippage_ticks: float = 1.0

    def __post_init__(self) -> None:
        if self.after_monthly_target not in ("stop", "reduce", "continue"):
            raise ValueError("after_monthly_target must be stop, reduce or continue")


class RiskManager:
    def __init__(self, cfg: RiskConfig, rules: TopstepRules, instrument: Instrument):
        self.cfg = cfg
        self.rules = rules
        self.instrument = instrument
        self.day: date | None = None
        self.month: tuple[int, int] | None = None
        self.month_pnl = 0.0
        self.trades_today = 0
        self.halted_reason: str | None = None

    # -- lifecycle -------------------------------------------------------
    def new_day(self, day: date) -> None:
        month = (day.year, day.month)
        if month != self.month:
            self.month = month
            self.month_pnl = 0.0
        self.day = day
        self.trades_today = 0
        self.halted_reason = None

    def on_trade_closed(self, pnl: float) -> None:
        self.month_pnl += pnl

    def halt(self, reason: str) -> None:
        self.halted_reason = reason

    # -- checks ----------------------------------------------------------
    def month_factor(self) -> float:
        tgt = self.cfg.monthly_target
        if tgt is None or self.month_pnl < tgt:
            return 1.0
        return {"stop": 0.0, "reduce": self.cfg.reduce_factor, "continue": 1.0}[
            self.cfg.after_monthly_target]

    def can_trade(self, state: AccountState) -> tuple[bool, str]:
        c = self.cfg
        if self.halted_reason:
            return False, self.halted_reason
        if self.trades_today >= c.max_trades_per_day:
            return False, "max trades per day"
        if state.day_pnl <= -c.daily_loss_stop:
            return False, "daily loss stop"
        if c.daily_profit_target is not None and state.day_pnl >= c.daily_profit_target:
            return False, "daily profit target"
        if self.month_factor() == 0.0:
            return False, "monthly target reached"
        if state.balance - state.mll_floor - c.mll_buffer < 0.25 * c.risk_per_trade:
            # Kill switch: the strategy has lost most of the MLL cushion.
            return False, "too close to max loss limit"
        return True, ""

    def max_contracts(self) -> int:
        limit = int(round(self.rules.max_minis / self.instrument.mini_equivalent))
        if self.cfg.max_contracts is not None:
            limit = min(limit, self.cfg.max_contracts)
        return limit

    def risk_budget(self, state: AccountState) -> float:
        c = self.cfg
        budget = c.risk_per_trade * self.month_factor()
        budget = min(budget, c.daily_loss_stop + state.day_pnl)
        budget = min(budget, state.balance - state.mll_floor - c.mll_buffer)
        dll = state.dll_level()
        if dll is not None:
            budget = min(budget, state.balance - dll - c.dll_buffer)
        return max(budget, 0.0)

    def per_contract_risk(self, stop_distance: float) -> float:
        inst = self.instrument
        return (abs(stop_distance) * inst.point_value
                + 2 * inst.fee_per_side
                + 2 * self.cfg.slippage_ticks * inst.tick_value)

    def position_size(self, stop_distance: float, state: AccountState) -> int:
        """Contracts so that a stop-out loses at most the risk budget."""
        if stop_distance <= 0:
            return 0
        n = math.floor(self.risk_budget(state) / self.per_contract_risk(stop_distance))
        return max(0, min(n, self.max_contracts()))

    def register_entry(self) -> None:
        self.trades_today += 1
