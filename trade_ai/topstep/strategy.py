"""Intraday Opening Range Breakout (ORB) strategy.

The strategy is incremental: it sees one completed bar at a time through
``on_bar`` and keeps its own state, so the backtester and the live bot run
exactly the same code and a decision at bar t can only use bars <= t.

Rules (times are exchange-local, America/New_York by default):
  * Opening range = high/low of the first ``or_minutes`` of the session.
  * Skip the day if the range is too narrow or too wide relative to the
    average daily range of previous days (no news-spike or dead days).
  * Long when a bar closes above the range high (+ buffer) and above the
    session VWAP; short on the mirror image. Each direction at most once.
  * Stop at the range midpoint (or opposite side), target = ``target_r`` x
    risk, optional move-to-breakeven, and a hard time exit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta


@dataclass
class Signal:
    side: int              # +1 long, -1 short
    stop: float            # absolute stop price
    target_r: float        # take profit at target_r x (entry - stop) distance
    reason: str = ""


@dataclass
class ORBConfig:
    session_start: time = time(9, 30)
    or_minutes: int = 15
    entry_end: time = time(11, 30)
    exit_time: time = time(15, 50)
    stop_mode: str = "mid"          # "mid" or "range"
    target_r: float = 1.5
    breakeven_r: float | None = 1.0  # move stop to entry after this many R
    breakout_buffer_ticks: int = 1
    tick_size: float = 0.25
    atr_days: int = 14
    min_range_atr: float = 0.10     # skip if OR < 10% of avg daily range
    max_range_atr: float = 0.45     # skip if OR > 45% of avg daily range
    vwap_filter: bool = True
    allow_long: bool = True
    allow_short: bool = True
    max_signals_per_day: int = 2

    def __post_init__(self) -> None:
        if self.stop_mode not in ("mid", "range"):
            raise ValueError("stop_mode must be 'mid' or 'range'")


class ORBStrategy:
    def __init__(self, cfg: ORBConfig | None = None):
        self.cfg = cfg or ORBConfig()
        self.daily_ranges: deque[float] = deque(maxlen=self.cfg.atr_days)
        self.day: date | None = None
        self._reset_day()

    def _reset_day(self) -> None:
        self.or_high = float("-inf")
        self.or_low = float("inf")
        self.day_high = float("-inf")
        self.day_low = float("inf")
        self.pv = 0.0
        self.vol = 0.0
        self.taken: set[int] = set()
        self.signals_today = 0
        self.skip_reason: str | None = None

    @property
    def or_end(self) -> time:
        start = datetime.combine(date(2000, 1, 1), self.cfg.session_start)
        return (start + timedelta(minutes=self.cfg.or_minutes)).time()

    def new_day(self, day: date) -> None:
        if self.day is not None and self.day_high > self.day_low:
            self.daily_ranges.append(self.day_high - self.day_low)
        self.day = day
        self._reset_day()

    def avg_daily_range(self) -> float | None:
        if len(self.daily_ranges) < max(3, self.cfg.atr_days // 2):
            return None
        return sum(self.daily_ranges) / len(self.daily_ranges)

    def vwap(self) -> float | None:
        return self.pv / self.vol if self.vol > 0 else None

    def on_bar(self, ts: datetime, o: float, h: float, l: float, c: float,
               v: float) -> Signal | None:
        """Feed one completed bar (``ts`` = bar open time, local tz)."""
        cfg = self.cfg
        if self.day != ts.date():
            self.new_day(ts.date())
        t = ts.time()
        if t < cfg.session_start:
            return None
        self.day_high = max(self.day_high, h)
        self.day_low = min(self.day_low, l)
        typical = (h + l + c) / 3.0
        w = v if v > 0 else 1.0
        self.pv += typical * w
        self.vol += w

        if t < self.or_end:
            self.or_high = max(self.or_high, h)
            self.or_low = min(self.or_low, l)
            return None
        if self.or_high == float("-inf") or self.skip_reason:
            return None
        if t >= cfg.entry_end or self.signals_today >= cfg.max_signals_per_day:
            return None

        width = self.or_high - self.or_low
        adr = self.avg_daily_range()
        if adr is None:
            self.skip_reason = "warming up"
            return None
        if width < cfg.min_range_atr * adr or width > cfg.max_range_atr * adr or width <= 0:
            self.skip_reason = f"range {width:.2f} outside filter (ADR {adr:.2f})"
            return None

        buf = cfg.breakout_buffer_ticks * cfg.tick_size
        vwap = self.vwap()
        mid = (self.or_high + self.or_low) / 2.0
        if (cfg.allow_long and 1 not in self.taken and c > self.or_high + buf - 1e-12
                and (not cfg.vwap_filter or vwap is None or c > vwap)):
            stop = mid if cfg.stop_mode == "mid" else self.or_low
            return self._emit(1, stop, "ORB long")
        if (cfg.allow_short and -1 not in self.taken and c < self.or_low - buf + 1e-12
                and (not cfg.vwap_filter or vwap is None or c < vwap)):
            stop = mid if cfg.stop_mode == "mid" else self.or_high
            return self._emit(-1, stop, "ORB short")
        return None

    def _emit(self, side: int, stop: float, reason: str) -> Signal:
        self.taken.add(side)
        self.signals_today += 1
        return Signal(side=side, stop=stop, target_r=self.cfg.target_r, reason=reason)
