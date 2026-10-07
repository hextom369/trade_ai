"""``python -m trade_ai topstep ...`` commands.

    backtest   ORB + Topstep rules on CSV (or synthetic) intraday bars
    download   1-minute bars from TopstepX into a CSV
    live       run the bot (dry-run by default)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from .backtest import run_topstep_backtest
from .data import load_intraday_csv, synthetic_intraday
from .instruments import get_instrument
from .report import format_report, monte_carlo, monthly_table, summarize
from .risk import RiskConfig
from .rules import TopstepRules
from .strategy import ORBConfig, ORBStrategy


def _hm(s: str):
    return datetime.strptime(s, "%H:%M").time()


def _add_trading_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="MES", help="MES, MNQ, ES, NQ, ...")
    p.add_argument("--account", default="50k", choices=["50k", "100k", "150k"])
    p.add_argument("--no-dll", action="store_true", help="account has no Daily Loss Limit")
    p.add_argument("--fee", type=float, help="fee per contract per side (USD)")
    g = p.add_argument_group("risk")
    g.add_argument("--risk-per-trade", type=float, default=150.0)
    g.add_argument("--daily-loss-stop", type=float, default=300.0)
    g.add_argument("--daily-profit-target", type=float, default=300.0,
                   help="stop for the day after this profit (0 = off)")
    g.add_argument("--max-trades", type=int, default=2)
    g.add_argument("--monthly-target", type=float, default=1000.0)
    g.add_argument("--after-target", choices=["stop", "reduce", "continue"], default="reduce")
    g.add_argument("--max-contracts", type=int)
    g.add_argument("--slippage-ticks", type=float, default=1.0)
    s = p.add_argument_group("strategy (ORB)")
    s.add_argument("--or-minutes", type=int, default=15)
    s.add_argument("--entry-end", type=_hm, default=_hm("11:30"))
    s.add_argument("--exit-time", type=_hm, default=_hm("15:50"))
    s.add_argument("--stop-mode", choices=["mid", "range"], default="mid")
    s.add_argument("--target-r", type=float, default=1.5)
    s.add_argument("--breakeven-r", type=float, default=1.0, help="0 = off")
    s.add_argument("--min-range-adr", type=float, default=0.10)
    s.add_argument("--max-range-adr", type=float, default=0.45)
    s.add_argument("--no-vwap-filter", action="store_true")
    s.add_argument("--long-only", action="store_true")
    s.add_argument("--tz", default="America/New_York", help="exchange-local time zone")


def _build(args):
    inst = get_instrument(args.symbol)
    if args.fee is not None:
        inst = inst.with_fee(args.fee)
    rules = TopstepRules.combine(args.account)
    if args.no_dll:
        rules.daily_loss_limit = None
    risk = RiskConfig(risk_per_trade=args.risk_per_trade, daily_loss_stop=args.daily_loss_stop,
                      daily_profit_target=args.daily_profit_target or None,
                      max_trades_per_day=args.max_trades, monthly_target=args.monthly_target,
                      after_monthly_target=args.after_target, max_contracts=args.max_contracts,
                      slippage_ticks=args.slippage_ticks)
    strat = ORBStrategy(ORBConfig(
        or_minutes=args.or_minutes, entry_end=args.entry_end, exit_time=args.exit_time,
        stop_mode=args.stop_mode, target_r=args.target_r,
        breakeven_r=args.breakeven_r or None, tick_size=inst.tick_size,
        min_range_atr=args.min_range_adr, max_range_atr=args.max_range_adr,
        vwap_filter=not args.no_vwap_filter, allow_short=not args.long_only,
        max_signals_per_day=args.max_trades))
    return inst, rules, risk, strat


def _backtest(args) -> int:
    inst, rules, risk, strat = _build(args)
    if args.csv:
        bars = load_intraday_csv(args.csv, tz=args.tz, source_tz=args.source_tz)
    else:
        bars = synthetic_intraday(days=args.days, seed=args.seed,
                                  trend_strength=args.synthetic_trend, tz=args.tz)
        print("!! synthetic data: results say nothing about real markets\n")
    if args.start:
        bars = bars.loc[args.start:]
    if args.end:
        bars = bars.loc[:args.end]
    res = run_topstep_backtest(bars, strat, inst, rules, risk)
    print(f"{inst.symbol} {bars.index[0]} -> {bars.index[-1]}  ({args.account.upper()} rules)\n")
    print("Summary")
    print(format_report(summarize(res, args.monthly_target)))
    print("\nMonthly P&L")
    print(monthly_table(res.daily).round(2).to_string())
    if len(res.daily) >= 20:
        print("\nMonte Carlo (21 trading days, fresh account)")
        print(format_report(monte_carlo(res.daily["pnl"], rules,
                                        monthly_target=args.monthly_target)))
    if res.events:
        print("\nLimit events")
        for e in res.events:
            print(f"  {e['time']}  {e['event']}  balance {e['balance']:.2f}")
    if args.trades_out:
        res.trades.to_csv(args.trades_out, index=False)
        print(f"\nwrote {args.trades_out}")
    return 0


def _download(args) -> int:
    from .projectx import ProjectXClient

    client = ProjectXClient()
    cid = args.contract_id or client.front_contract_id(args.symbol)
    end = datetime.now(timezone.utc)
    df = client.bars_range(cid, end - timedelta(days=args.days), end, unit_number=args.minutes)
    df.to_csv(args.output)
    print(f"wrote {len(df)} bars of {cid} to {args.output} (UTC timestamps)")
    return 0


def _live(args) -> int:  # pragma: no cover - network
    from .live import LiveBot, LiveConfig
    from .projectx import ProjectXClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)])
    inst, rules, risk, strat = _build(args)
    cfg = LiveConfig(symbol=args.symbol, account_name=args.account_name,
                     contract_id=args.contract_id, tz=args.tz, dry_run=not args.live,
                     mll_floor=args.mll_floor, state_path=args.state)
    LiveBot(ProjectXClient(), cfg, strat, inst, rules, risk).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trade_ai topstep")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="backtest the ORB bot under Topstep rules")
    _add_trading_args(bt)
    bt.add_argument("--csv", help="intraday OHLCV CSV (bar open timestamps)")
    bt.add_argument("--source-tz", default="UTC", help="time zone of naive CSV timestamps")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--days", type=int, default=250, help="synthetic days")
    bt.add_argument("--seed", type=int, default=0)
    bt.add_argument("--synthetic-trend", type=float, default=0.0)
    bt.add_argument("--trades-out")

    dl = sub.add_parser("download", help="download 1-minute bars from TopstepX")
    dl.add_argument("--symbol", default="MES")
    dl.add_argument("--contract-id")
    dl.add_argument("--days", type=int, default=90)
    dl.add_argument("--minutes", type=int, default=1)
    dl.add_argument("--output", required=True)

    lv = sub.add_parser("live", help="run the bot on TopstepX (dry-run unless --live)")
    _add_trading_args(lv)
    lv.add_argument("--live", action="store_true", help="actually send orders")
    lv.add_argument("--account-name", help="TopstepX account name to trade")
    lv.add_argument("--contract-id")
    lv.add_argument("--mll-floor", type=float,
                    help="current Max Loss Limit level from TopstepX (recommended)")
    lv.add_argument("--state", default="topstep_state.json")
    lv.add_argument("--log", default="topstep_bot.log")

    args = parser.parse_args(argv)
    return {"backtest": _backtest, "download": _download, "live": _live}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
