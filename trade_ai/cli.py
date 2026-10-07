"""Command line interface.

Examples:
    python -m trade_ai backtest --synthetic
    python -m trade_ai backtest --csv prices.csv --long-only
    python -m trade_ai backtest --ticker SPY --start 2010-01-01
    python -m trade_ai signal --ticker 7203.T

Crypto perps (Binance futures candles + funding, Variational-ready):
    python -m trade_ai backtest --binance BTCUSDT --interval 1h --start 2024-01-01 --strategy breakout
    python -m trade_ai trade --binance BTCUSDT --venue-symbol BTC --strategy ma_cross --once

Volume-selected universe (trade the most traded markets):
    python -m trade_ai screen --top-n 5 --variational-only
    python -m trade_ai backtest --universe --candidates 30 --top-n 5 --start 2025-01-01 --strategy breakout
    python -m trade_ai trade --universe --top-n 5 --strategy vwap_obv --once
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

from .brokers import PaperBroker
from .crypto_data import (funding_per_bar, interval_to_timedelta, load_binance_funding,
                          load_binance_klines, periods_per_year)
from .data import load_csv, load_yfinance, synthetic_ohlcv
from .live import RiskLimits, Trader, TraderConfig, run_loop
from .metrics import format_summary
from .model import WalkForwardConfig
from .pipeline import PipelineConfig, latest_target, run_pipeline, run_rule_pipeline
from .portfolio import run_portfolio_backtest
from .report import paper_report
from .rules import RULES, RuleConfig
from .strategy import StrategyConfig
from .universe import (RANK_BY, binance_24h_quote_volume, point_in_time_universe, prefilter,
                       rank_universe, venue_ticker, volume_scores)
from .variational import VariationalBroker, VariationalPublicClient


def _add_common(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV file with date, open, high, low, close, volume")
    src.add_argument("--ticker", help="download via yfinance")
    src.add_argument("--synthetic", action="store_true", help="use generated demo data")
    src.add_argument("--binance", metavar="SYMBOL",
                     help="Binance USDT-M perpetual candles, e.g. BTCUSDT")
    src.add_argument("--universe", action="store_true",
                     help="trade the top --top-n Binance perps ranked by volume")
    _add_universe(p)
    p.add_argument("--interval", default="1h", help="candle interval for --binance (1m..1w)")
    p.add_argument("--market", choices=["futures", "spot"], default="futures",
                   help="Binance candles: USDT-M perps, or spot from the public data host "
                        "(reachable from US-hosted CI; no funding)")
    p.add_argument("--no-funding", action="store_true",
                   help="ignore perp funding in the backtest (--binance only)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--strategy", choices=["ml", *RULES], default="ml")
    p.add_argument("--fast", type=int, default=20, help="ma_cross fast MA")
    p.add_argument("--slow", type=int, default=100, help="ma_cross slow MA")
    p.add_argument("--breakout-window", type=int, default=55)
    p.add_argument("--exit-window", type=int, default=20)
    p.add_argument("--rsi-window", type=int, default=14)
    p.add_argument("--rsi-low", type=float, default=30.0)
    p.add_argument("--rsi-high", type=float, default=70.0)
    p.add_argument("--vol-window", type=int, default=20, help="volume_breakout avg volume window")
    p.add_argument("--vol-mult", type=float, default=2.0,
                   help="volume_breakout: required volume vs average")
    p.add_argument("--vwap-window", type=int, default=48, help="vwap_obv lookback")
    p.add_argument("--obv-min", type=float, default=0.1, help="vwap_obv min net volume share")
    p.add_argument("--taker-span", type=int, default=12, help="taker_flow EMA span")
    p.add_argument("--taker-band", type=float, default=0.02, help="taker_flow no-trade band")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--label-threshold", type=float, default=0.0,
                   help="label threshold in units of horizon volatility")
    p.add_argument("--no-volume-features", action="store_true",
                   help="ml: drop volume/order-flow features (to measure what volume adds)")
    p.add_argument("--model", choices=["ensemble", "hgb", "logreg"], default="ensemble")
    p.add_argument("--min-train", type=int, default=500)
    p.add_argument("--retrain-every", type=int, default=60)
    p.add_argument("--train-window", type=int, default=None)
    p.add_argument("--entry-band", type=float, default=0.04)
    p.add_argument("--full-edge", type=float, default=0.15)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--band", type=float, default=0.0,
                   help="backtest: ignore target changes smaller than this (like the live "
                        "--rebalance-band)")
    p.add_argument("--no-gate", action="store_true", help="disable the hit-rate quality gate")
    p.add_argument("--gate-window", type=int, default=120)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--periods-per-year", type=int, default=None,
                   help="default: 252, or 24/7 bars per year for --binance")


def _add_universe(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("volume-based symbol selection (--universe / screen)")
    g.add_argument("--symbols", help="comma-separated candidate symbols (default: current "
                   "top --candidates by 24h traded value)")
    g.add_argument("--candidates", type=int, default=30)
    g.add_argument("--top-n", type=int, default=5, help="symbols traded at once")
    g.add_argument("--rank-by", choices=RANK_BY, default="quote_volume",
                   help="quote_volume = most traded value; surge = sudden volume increase")
    g.add_argument("--volume-window", type=int, default=168,
                   help="bars of traded value used for ranking (168 = 7 days of 1h)")
    g.add_argument("--surge-window", type=int, default=24,
                   help="recent bars compared to the --volume-window average for surge")
    g.add_argument("--rebalance-every", type=int, default=24, help="re-rank every N bars")
    g.add_argument("--min-quote-volume", type=float, default=20e6,
                   help="minimum 24h traded value in USDT (liquidity floor)")
    g.add_argument("--variational-only", action="store_true",
                   help="only symbols listed on Variational Omni")


def _market(args) -> str:
    return getattr(args, "market", "futures")


def _candidates(args) -> list[str]:
    if args.symbols:
        return [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    allowed = None
    if args.variational_only:
        allowed = {str(x.get("ticker", "")).upper()
                   for x in VariationalPublicClient().listings()}
    return prefilter(binance_24h_quote_volume(), args.candidates, args.min_quote_volume, allowed)


def _window_floor(args) -> float:
    """--min-quote-volume (per 24h) expressed over --volume-window bars."""
    days = args.volume_window * interval_to_timedelta(args.interval) / pd.Timedelta(days=1)
    return args.min_quote_volume * days


def _load_frames(args, symbols, limit=None) -> dict:
    frames = {}
    for sym in symbols:
        try:
            if limit:
                frames[sym] = load_binance_klines(sym, args.interval, limit=limit,
                                                  market=_market(args))
            else:
                frames[sym] = load_binance_klines(sym, args.interval, args.start, args.end,
                                                  market=_market(args))
        except ValueError as exc:  # e.g. delisted / no candles in range
            logging.warning("skipping %s: %s", sym, exc)
    return frames


def _select_now(args, frames: dict) -> tuple[list[str], pd.DataFrame]:
    """Rank on the latest bar exactly as the backtest does at each rebalance."""
    liquidity, surge = volume_scores(frames, args.volume_window, args.surge_window)
    # No forward-fill: a symbol without a candle on the latest bar is not tradable.
    last_liq, last_surge = liquidity.iloc[-1], surge.iloc[-1]
    chosen = rank_universe(last_liq, last_surge, args.top_n, args.rank_by, _window_floor(args))
    table = pd.DataFrame({"venue": [venue_ticker(s) for s in last_liq.index],
                          "window_value_usd": last_liq, "surge": last_surge})
    table["selected"] = table.index.isin(chosen)
    key = "window_value_usd" if args.rank_by == "quote_volume" else "surge"
    return chosen, table.sort_values(key, ascending=False)


def _universe_backtest(args, cfg: PipelineConfig) -> int:
    frames = _load_frames(args, _candidates(args))
    if not frames:
        raise SystemExit("no candle data for any candidate symbol")
    bar = interval_to_timedelta(args.interval)
    funding = None
    if not args.no_funding and _market(args) == "futures":
        funding = {}
        for sym, df in frames.items():
            events = load_binance_funding(sym, start=df.index[0], end=df.index[-1] + bar)
            funding[sym] = funding_per_bar(events, df.index, bar)
    members = point_in_time_universe(frames, args.top_n, args.volume_window, args.surge_window,
                                     args.rebalance_every, args.rank_by, _window_floor(args))
    res = run_portfolio_backtest(frames, members, args.top_n, args.strategy, cfg, _rules(args),
                                 funding)
    b = res.backtest
    print(f"Universe: top {args.top_n} of {len(frames)} by {args.rank_by}, "
          f"re-ranked every {args.rebalance_every} bars")
    print(f"Period: {b.returns.index[0]} -> {b.returns.index[-1]}")
    print("\nStrategy")
    print(format_summary(b.stats))
    print("\nEqual-weight buy & hold of the selected symbols")
    print(format_summary(b.benchmark_stats))
    share = res.membership.loc[b.returns.index].mean().sort_values(ascending=False)
    contrib = res.contributions.sum().reindex(share.index)
    print("\nTime selected / summed return contribution")
    for sym in share.index[share > 0]:
        print(f"  {sym:<16}{share[sym]:>8.1%}{contrib[sym]:>10.2%}")
    if args.output:
        b.returns.to_frame().join([b.positions, b.equity, b.benchmark]).join(
            res.membership.loc[b.returns.index].add_prefix("in_")).to_csv(args.output)
        print(f"\nwrote {args.output}")
    return 0


def _screen(args) -> int:
    limit = args.volume_window + args.surge_window + 2
    frames = _load_frames(args, _candidates(args), limit=limit)
    chosen, table = _select_now(args, frames)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 120):
        print(table.to_string())
    print("\nselected:", ", ".join(chosen) or "(none)")
    return 0


def _load(args):
    if args.binance:
        df = load_binance_klines(args.binance, args.interval, args.start, args.end,
                                 market=_market(args))
    elif args.csv:
        df = load_csv(args.csv)
    elif args.ticker:
        df = load_yfinance(args.ticker, args.start, args.end)
    else:
        df = synthetic_ohlcv()
    if args.start:
        df = df.loc[args.start:]
    if args.end:
        df = df.loc[:args.end]
    return df


def _funding(args, df):
    if not args.binance or args.no_funding or _market(args) != "futures":
        return None
    events = load_binance_funding(args.binance, start=df.index[0],
                                  end=df.index[-1] + interval_to_timedelta(args.interval))
    return funding_per_bar(events, df.index, interval_to_timedelta(args.interval))


def _rules(args) -> RuleConfig:
    return RuleConfig(fast=args.fast, slow=args.slow, breakout_window=args.breakout_window,
                      exit_window=args.exit_window, rsi_window=args.rsi_window,
                      rsi_low=args.rsi_low, rsi_high=args.rsi_high,
                      vol_window=args.vol_window, vol_mult=args.vol_mult,
                      vwap_window=args.vwap_window, obv_min=args.obv_min,
                      taker_span=args.taker_span, taker_band=args.taker_band)


def _config(args) -> PipelineConfig:
    ppy = args.periods_per_year
    if ppy is None:
        ppy = periods_per_year(args.interval) if (args.binance or args.universe) else 252
    return PipelineConfig(
        label_threshold=args.label_threshold,
        volume_features=not args.no_volume_features,
        walk_forward=WalkForwardConfig(
            horizon=args.horizon, min_train=args.min_train, retrain_every=args.retrain_every,
            train_window=args.train_window, model=args.model,
        ),
        strategy=StrategyConfig(
            entry_band=args.entry_band, full_edge=args.full_edge, long_only=args.long_only, target_vol=args.target_vol,
            max_leverage=args.max_leverage, smoothing=args.smoothing,
            rebalance_band=args.band,
            quality_gate=not args.no_gate, gate_window=args.gate_window,
            periods_per_year=ppy,
        ),
        cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trade_ai")
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest", help="walk-forward backtest")
    _add_common(bt)
    bt.add_argument("--output", help="write per-bar results to this CSV")
    sig = sub.add_parser("signal", help="target position for the next bar")
    _add_common(sig)
    tr = sub.add_parser("trade", help="run the strategy live / on paper (dry run by default)")
    _add_common(tr)
    tr.add_argument("--venue-symbol", help="symbol at the venue (default: --binance without USDT)")
    tr.add_argument("--broker", choices=["paper", "variational"], default="paper")
    tr.add_argument("--paper-prices", choices=["candles", "variational"], default="candles",
                    help="paper broker reference price: last candle close or Variational mark")
    tr.add_argument("--paper-equity", type=float, default=10_000.0)
    tr.add_argument("--execute", action="store_true",
                    help="actually send orders (paper fills for --broker paper); default logs only")
    tr.add_argument("--once", action="store_true", help="run a single cycle and exit")
    tr.add_argument("--lookback", type=int, default=1500, help="candles loaded per cycle")
    tr.add_argument("--state-dir", default="trade_state")
    tr.add_argument("--max-position", type=float, default=1.0)
    tr.add_argument("--max-notional", type=float, default=None)
    tr.add_argument("--min-trade-notional", type=float, default=10.0)
    tr.add_argument("--rebalance-band", type=float, default=0.05)
    tr.add_argument("--max-daily-loss", type=float, default=0.05)
    tr.add_argument("--skip-processed-bars", action="store_true",
                    help="do nothing if the latest closed bar was already handled "
                         "(lets a scheduler run more often than the bar interval)")
    rp = sub.add_parser("report", help="summarize a paper/live trading state directory")
    rp.add_argument("--state-dir", default="trade_state")
    rp.add_argument("--paper-equity", type=float, default=10_000.0)
    sc = sub.add_parser("screen", help="rank Binance perps by volume and show the selection")
    sc.add_argument("--interval", default="1h")
    _add_universe(sc)
    args = parser.parse_args(argv)

    if args.command == "screen":
        return _screen(args)
    if args.command == "report":
        print(paper_report(args.state_dir, args.paper_equity), end="")
        return 0
    cfg = _config(args)
    if args.command == "trade":
        return _trade(args, cfg)
    if args.universe:
        if args.command == "backtest":
            return _universe_backtest(args, cfg)
        frames = _load_frames(args, _candidates(args), limit=max(args.volume_window + 2, 1500))
        chosen, _ = _select_now(args, frames)
        out = [latest_target(frames[s], args.strategy, cfg, _rules(args)) | {"symbol": s}
               for s in chosen]
        for o in out:
            o["target_position"] /= max(args.top_n, 1)  # equal slice of equity
        print(json.dumps(out, indent=2))
        return 0
    df = _load(args)
    if args.command == "backtest":
        funding = _funding(args, df)
        if args.strategy == "ml":
            res = run_pipeline(df, cfg, funding)
        else:
            res = run_rule_pipeline(df, args.strategy, _rules(args), cfg, funding)
        b = res.backtest
        print(f"Out-of-sample period: {b.returns.index[0]} -> {b.returns.index[-1]}")
        print("\nStrategy")
        print(format_summary(b.stats))
        print("\nBuy & hold")
        print(format_summary(b.benchmark_stats))
        if args.output:
            out = b.returns.to_frame().join([b.positions, b.equity, b.benchmark,
                                             res.proba.reindex(b.returns.index),
                                             res.gate.reindex(b.returns.index)])
            out.to_csv(args.output)
            print(f"\nwrote {args.output}")
    else:
        print(json.dumps(latest_target(df, args.strategy, cfg, _rules(args)), indent=2))
    return 0


def _universe_loader(args, latest: dict):
    """Loader for run_loop: re-rank every --rebalance-every bars, else reuse the selection."""
    bar = interval_to_timedelta(args.interval)
    path = os.path.join(args.state_dir, "universe_selection.json")
    limit = max(args.lookback, args.volume_window + args.surge_window + 2)

    def load():
        period = int(pd.Timestamp.now(tz="UTC").value // (bar * args.rebalance_every).value)
        saved = {}
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
        if saved.get("period") == period and saved.get("symbols"):
            frames = _load_frames(args, saved["symbols"], limit=limit)
        else:
            frames = _load_frames(args, _candidates(args), limit=limit)
            chosen, table = _select_now(args, frames)
            logging.info("universe re-ranked by %s:\n%s", args.rank_by, table.head(15))
            frames = {s: frames[s] for s in chosen}
            with open(path, "w") as f:
                json.dump({"period": period, "symbols": chosen}, f)
        out = {}
        for sym, df in frames.items():
            latest[venue_ticker(sym)] = float(df["close"].iloc[-1])
            out[venue_ticker(sym)] = df
        return out
    return load


def _trade(args, cfg: PipelineConfig) -> int:
    if not (args.binance or args.universe):
        raise SystemExit("trade needs --binance SYMBOL or --universe as its candle source")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.state_dir, exist_ok=True)
    latest: dict = {}
    if args.universe:
        venue, top_n = "", args.top_n
        load = _universe_loader(args, latest)
    else:
        venue, top_n = args.venue_symbol or venue_ticker(args.binance), 1

        def load():
            df = load_binance_klines(args.binance, args.interval, limit=args.lookback,
                                     market=_market(args))
            latest[venue] = float(df["close"].iloc[-1])
            return df

    if args.broker == "variational":
        broker = VariationalBroker()
    else:
        source = (VariationalPublicClient().mark_price if args.paper_prices == "variational"
                  else latest.__getitem__)
        broker = PaperBroker(initial_equity=args.paper_equity, slippage_bps=args.slippage_bps,
                             fee_bps=args.cost_bps, price_source=source,
                             state_path=os.path.join(args.state_dir, "paper_account.json"))
    trader = Trader(broker, TraderConfig(
        data_symbol=args.binance or "", venue_symbol=venue, interval=args.interval,
        strategy=args.strategy, pipeline=cfg, rules=_rules(args),
        risk=RiskLimits(max_abs_position=args.max_position, max_notional=args.max_notional,
                        min_trade_notional=args.min_trade_notional,
                        rebalance_band=args.rebalance_band, max_daily_loss=args.max_daily_loss),
        dry_run=not args.execute, state_dir=args.state_dir, top_n=top_n,
        skip_processed_bars=args.skip_processed_bars))
    try:
        run_loop(trader, load, once=args.once)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
