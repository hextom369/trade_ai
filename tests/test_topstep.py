from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from trade_ai.cli import main as cli_main
from trade_ai.topstep import (
    AccountState, ORBConfig, ORBStrategy, RiskConfig, RiskManager, Signal, TopstepRules,
    get_instrument, monte_carlo, monthly_table, run_topstep_backtest, summarize,
    synthetic_intraday,
)
from trade_ai.topstep import projectx as px
from trade_ai.topstep.live import LiveBot, LiveConfig

MES = get_instrument("MES")
TZ = "America/New_York"


# ---------------------------------------------------------------- rules
def test_mll_trails_eod_high_and_locks_at_start():
    s = AccountState(TopstepRules())
    assert s.mll_floor == 48_000
    s.apply_pnl(1_200)
    assert s.mll_floor == 48_000          # intraday gains do not move it
    s.end_of_day(date(2024, 1, 2))
    assert s.mll_floor == 49_200
    s.apply_pnl(-500)
    s.end_of_day(date(2024, 1, 3))
    assert s.mll_floor == 49_200          # never moves down
    s.apply_pnl(5_000)
    s.end_of_day(date(2024, 1, 4))
    assert s.mll_floor == 50_000          # locked at the starting balance


def test_breach_detection_and_consistency():
    s = AccountState(TopstepRules())
    assert s.breach(49_100) is None
    assert s.breach(49_000) == "DLL"
    assert s.breach(48_000) == "MLL"
    s.apply_pnl(2_000); s.end_of_day(date(2024, 1, 2))
    s.apply_pnl(1_000); s.end_of_day(date(2024, 1, 3))
    assert not s.consistency_ok()         # best day is 2/3 of profit
    s.apply_pnl(1_500); s.end_of_day(date(2024, 1, 4))
    assert s.consistency_ok() and s.target_reached()


# ---------------------------------------------------------------- risk
def test_position_size_respects_budget_and_contract_cap():
    rules = TopstepRules()
    rm = RiskManager(RiskConfig(risk_per_trade=150, slippage_ticks=1), rules, MES)
    rm.new_day(date(2024, 1, 2))
    s = AccountState(rules)
    # 4 pt stop: 4*5 + 2*0.5 fees + 2*1.25 slippage = 23.5 per contract -> 6
    assert rm.position_size(4.0, s) == 6
    big = RiskManager(RiskConfig(risk_per_trade=1_000, daily_loss_stop=1_000), rules, MES)
    big.new_day(date(2024, 1, 2))
    assert big.position_size(0.25, s) == 50  # capped at 50 micros (= 5 minis)
    assert rm.position_size(0.0, s) == 0
    es = RiskManager(RiskConfig(risk_per_trade=10_000), rules, get_instrument("ES"))
    es.new_day(date(2024, 1, 2))
    assert es.position_size(1.0, s) <= 5


def test_daily_and_monthly_limits():
    rules = TopstepRules()
    cfg = RiskConfig(daily_loss_stop=300, daily_profit_target=200, monthly_target=1_000,
                     after_monthly_target="stop")
    rm = RiskManager(cfg, rules, MES)
    s = AccountState(rules)
    rm.new_day(date(2024, 1, 2))
    assert rm.can_trade(s)[0]
    s.apply_pnl(-300)
    assert rm.can_trade(s) == (False, "daily loss stop")
    s.end_of_day(date(2024, 1, 2)); rm.new_day(date(2024, 1, 3))
    s.apply_pnl(250)
    assert rm.can_trade(s) == (False, "daily profit target")
    s.end_of_day(date(2024, 1, 3)); rm.new_day(date(2024, 1, 4))
    rm.on_trade_closed(1_000)
    assert rm.can_trade(s) == (False, "monthly target reached")
    rm.new_day(date(2024, 2, 1))            # new month resets it
    assert rm.can_trade(s)[0]
    rm.cfg.after_monthly_target = "reduce"
    rm.on_trade_closed(1_000)
    assert rm.risk_budget(s) == pytest.approx(cfg.risk_per_trade * cfg.reduce_factor)


def test_risk_budget_never_exceeds_mll_cushion():
    rules = TopstepRules()
    rm = RiskManager(RiskConfig(risk_per_trade=1_000, daily_loss_stop=5_000), rules, MES)
    rm.new_day(date(2024, 1, 2))
    s = AccountState(rules)
    s.apply_pnl(-1_500); s.end_of_day(date(2024, 1, 1))
    # cushion 500, buffer 300 -> at most 200 at risk
    assert rm.risk_budget(s) == pytest.approx(200)
    s.apply_pnl(-200)
    assert rm.can_trade(s) == (False, "too close to max loss limit")


# ---------------------------------------------------------------- strategy
def _signals(bars):
    strat = ORBStrategy()
    out = []
    for ts, r in bars.iterrows():
        sig = strat.on_bar(ts, r.open, r.high, r.low, r.close, r.volume)
        if sig:
            out.append((ts, sig.side, sig.stop))
    return out


def test_strategy_signals_do_not_look_ahead():
    bars = synthetic_intraday(days=40, seed=5, trend_strength=1.0)
    cut = bars.index[len(bars) * 3 // 4]
    tampered = bars.copy()
    tampered.loc[tampered.index > cut, ["open", "high", "low", "close"]] *= 1.1
    a = [s for s in _signals(bars) if s[0] <= cut]
    b = [s for s in _signals(tampered) if s[0] <= cut]
    assert a and a == b


def test_strategy_respects_entry_window_and_one_trade_per_side():
    bars = synthetic_intraday(days=60, seed=2, trend_strength=1.0)
    sigs = _signals(bars)
    assert sigs
    cfg = ORBConfig()
    for ts, _side, _stop in sigs:
        assert time(9, 45) <= ts.time() < cfg.entry_end
    per_day = pd.Series([(ts.date(), side) for ts, side, _ in sigs])
    assert not per_day.duplicated().any()


# ---------------------------------------------------------------- backtest
class _Scripted:
    """Emits given signals at given bar timestamps."""

    def __init__(self, signals, **cfg):
        self.cfg = ORBConfig(breakeven_r=None, **cfg)
        self.signals = signals

    def on_bar(self, ts, o, h, l, c, v):
        return self.signals.get(ts)


def _bars(rows, day="2024-01-02", start="10:00"):
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="1min", tz=TZ)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def test_backtest_fills_next_open_and_stop_wins_same_bar():
    bars = _bars([(100, 100, 100, 100),
                  (100, 101, 99, 100),      # entry at open 100 + 0.25 slip
                  (100, 104, 97, 100)])     # both stop (98) and target hit -> stop
    strat = _Scripted({bars.index[0]: Signal(1, 98.0, 1.5)})
    risk = RiskConfig(risk_per_trade=60, slippage_ticks=1, daily_profit_target=None)
    res = run_topstep_backtest(bars, strat, MES, risk=risk, limit_through_ticks=0)
    t = res.trades.iloc[0]
    assert t.entry == 100.25 and t.exit_reason == "stop" and t.exit == 97.75
    # 2.25 pt * $5 = 11.25 + fees 1 + slippage 2.5 = 14.75 per contract -> 4
    assert t.contracts == 4
    assert t.pnl == pytest.approx(-2.5 * 5 * 4 - 2 * 0.5 * 4)
    assert res.daily["pnl"].iloc[0] == pytest.approx(t.pnl)
    assert res.daily["balance"].iloc[0] == pytest.approx(50_000 + t.pnl)


def test_backtest_target_requires_trade_through_and_time_exit():
    bars = _bars([(100, 100, 100, 100),
                  (100, 100.5, 99.5, 100),   # entry 100.25, stop 99, target 102.125->102.25
                  (101, 102.25, 100.5, 102), # touches target only: no fill with 1 tick through
                  (102, 102.5, 101.5, 102)]) # trades through -> target fill
    strat = _Scripted({bars.index[0]: Signal(1, 99.0, 1.6)})
    res = run_topstep_backtest(bars, strat, MES, risk=RiskConfig(daily_profit_target=None))
    t = res.trades.iloc[0]
    assert t.exit_reason == "target" and t.exit == 102.25 and t.exit_time == bars.index[3]

    late = _bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100.5),
                  (100.5, 100.5, 100.5, 100.5)], start="15:47")
    strat = _Scripted({late.index[0]: Signal(1, 99.0, 5.0)})
    res = run_topstep_backtest(late, strat, MES, risk=RiskConfig(daily_profit_target=None))
    assert res.trades.iloc[0].exit_reason == "time"
    assert res.trades.iloc[0].exit_time.time() == time(15, 50)


def test_backtest_liquidates_at_dll_level():
    # Stop far below; no sizing buffer -> equity reaches the DLL before the stop.
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 60, 61),
                  (61, 62, 60, 61)])
    strat = _Scripted({bars.index[0]: Signal(1, 50.0, 1.0)})
    rules = TopstepRules(daily_loss_limit=500)
    risk = RiskConfig(risk_per_trade=1_000, daily_loss_stop=5_000, mll_buffer=0, dll_buffer=-1_000)
    res = run_topstep_backtest(bars, strat, MES, rules=rules, risk=risk)
    assert [e["event"] for e in res.events] == ["DLL"]
    assert res.trades.iloc[0].exit_reason == "DLL"
    assert res.daily["pnl"].iloc[0] == pytest.approx(-500, abs=30)


def test_gap_through_stop_breaching_mll_resets_account():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (40, 41, 39, 40),
                  (40, 41, 39, 40)])
    strat = _Scripted({bars.index[0]: Signal(1, 95.0, 1.0)})
    risk = RiskConfig(risk_per_trade=1_000, daily_loss_stop=1_000)
    res = run_topstep_backtest(bars, strat, MES, risk=risk)
    assert res.trades.iloc[0].exit_reason == "stop" and res.trades.iloc[0].exit == 39.75
    assert [e["event"] for e in res.events] == ["MLL"] and res.resets == 1
    assert res.daily["balance"].iloc[0] == 50_000   # fresh account after the reset


def test_random_walk_does_not_make_money():
    bars = synthetic_intraday(days=250, seed=11, trend_strength=0.0)
    res = run_topstep_backtest(bars, ORBStrategy(), MES)
    s = summarize(res)
    assert s["trades"] > 10
    assert s["expectancy_per_trade"] < 15
    assert res.trades["contracts"].max() <= 50


def test_reports():
    bars = synthetic_intraday(days=80, seed=4, trend_strength=1.0)
    res = run_topstep_backtest(bars, ORBStrategy(), MES)
    m = monthly_table(res.daily)
    assert m["pnl"].sum() == pytest.approx(res.daily["pnl"].sum())
    mc = monte_carlo(res.daily["pnl"], n_paths=500)
    assert 0 <= mc["p_month_>=_target"] <= mc["p_month_positive"] + 1e-9 <= 1 + 1e-9
    assert mc["p_mll_breach"] == pytest.approx(0.0, abs=1e-9) or mc["p_mll_breach"] <= 1


def test_cli_topstep_backtest(capsys):
    assert cli_main(["topstep", "backtest", "--days", "40", "--synthetic-trend", "1"]) == 0
    out = capsys.readouterr().out
    assert "Monthly P&L" in out and "synthetic" in out


# ---------------------------------------------------------------- ProjectX / live
class FakeAPI:
    def __init__(self, bars: pd.DataFrame | None = None):
        self.calls = []
        self.bars = bars if bars is not None else pd.DataFrame()
        self.position = None

    def __call__(self, url, body, headers):
        path = url.split("/api/", 1)[1]
        self.calls.append((path, body, headers))
        if path == "Auth/loginKey":
            return {"success": True, "token": "tok"}
        if path == "Account/search":
            return {"success": True, "accounts": [
                {"id": 7, "name": "50KTC", "balance": 50_000.0, "canTrade": True}]}
        if path == "Contract/search":
            return {"success": True, "contracts": [{"id": "CON.F.US.MES.Z24",
                                                    "activeContract": True}]}
        if path == "History/retrieveBars":
            s = pd.Timestamp(body["startTime"]); e = pd.Timestamp(body["endTime"])
            b = self.bars[(self.bars.index >= s) & (self.bars.index < e)]
            return {"success": True, "bars": [
                {"t": ts.isoformat(), "o": r.open, "h": r.high, "l": r.low, "c": r.close,
                 "v": r.volume} for ts, r in b.iterrows()]}
        if path == "Order/place":
            if body["type"] == px.ORDER_MARKET:
                self.position = {"contractId": body["contractId"], "size": body["size"],
                                 "type": 1 if body["side"] == 0 else 2, "averagePrice": 100.0}
            return {"success": True, "orderId": len(self.calls)}
        if path == "Position/searchOpen":
            return {"success": True, "positions": [self.position] if self.position else []}
        if path == "Order/searchOpen":
            return {"success": True, "orders": []}
        if path == "Order/bad":
            return {"success": False, "errorCode": 2, "errorMessage": "nope"}
        return {"success": True}


def test_projectx_client_requests():
    api = FakeAPI(bars=pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5],
                                     "volume": [10.0]},
                                    index=pd.DatetimeIndex(["2024-01-02 15:00"], tz="UTC")))
    c = px.ProjectXClient("me", "key", transport=api)
    assert c.accounts()[0]["id"] == 7
    assert api.calls[0][0] == "Auth/loginKey"
    assert api.calls[1][2]["Authorization"] == "Bearer tok"
    df = c.bars("X", datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None and df["close"].iloc[0] == 1.5
    assert api.calls[-1][1]["startTime"] == "2024-01-02T00:00:00.000Z"
    oid = c.place_order(7, "X", px.ORDER_STOP, px.SIDE_SELL, 3, stop_price=99.5)
    assert oid > 0 and api.calls[-1][1]["stopPrice"] == 99.5 and "limitPrice" not in api.calls[-1][1]
    with pytest.raises(px.ProjectXError, match="nope"):
        c._post("/api/Order/bad", {})
    with pytest.raises(px.ProjectXError):
        px.ProjectXClient("", "", transport=api).login()


def _live_bot(tmp_path, bars, dry_run):
    api = FakeAPI(bars.tz_convert("UTC"))
    client = px.ProjectXClient("me", "key", transport=api)
    cfg = LiveConfig(dry_run=dry_run, state_path=str(tmp_path / "state.json"), poll_seconds=0)
    bot = LiveBot(client, cfg, ORBStrategy(), MES)
    return api, bot


def test_live_bot_trades_same_signal_as_backtest(tmp_path):
    bars = synthetic_intraday(days=40, seed=5, trend_strength=1.0)
    sig_ts, side, stop = [s for s in _signals(bars)][-1]
    day = sig_ts.date()
    api, bot = _live_bot(tmp_path, bars, dry_run=False)
    bot.setup(now=pd.Timestamp(f"{day} 09:00", tz=TZ).to_pydatetime())
    # step through the morning minute by minute
    t = pd.Timestamp(f"{day} 09:31", tz=TZ)
    while t <= sig_ts + pd.Timedelta(minutes=2):
        bot.step(now=t.tz_convert("UTC").to_pydatetime())
        t += pd.Timedelta(minutes=1)
    orders = [b for p, b, _ in api.calls if p == "Order/place"]
    assert [o["type"] for o in orders] == [px.ORDER_MARKET, px.ORDER_STOP, px.ORDER_LIMIT]
    assert orders[0]["side"] == (px.SIDE_BUY if side > 0 else px.SIDE_SELL)
    assert orders[1]["stopPrice"] == MES.round_price(stop)
    assert orders[0]["size"] == orders[1]["size"] == orders[2]["size"] <= 50
    assert (tmp_path / "state.json").exists()


def test_live_bot_dry_run_sends_no_orders(tmp_path):
    bars = synthetic_intraday(days=40, seed=5, trend_strength=1.0)
    sig_ts = _signals(bars)[-1][0]
    api, bot = _live_bot(tmp_path, bars, dry_run=True)
    bot.setup(now=pd.Timestamp(f"{sig_ts.date()} 09:00", tz=TZ).to_pydatetime())
    for m in range(0, 150):
        now = pd.Timestamp(f"{sig_ts.date()} 09:31", tz=TZ) + pd.Timedelta(minutes=m)
        bot.step(now=now.tz_convert("UTC").to_pydatetime())
    assert not [p for p, _, _ in api.calls if p.startswith(("Order/place", "Position/close"))]
    assert bot.rm.trades_today == 1
