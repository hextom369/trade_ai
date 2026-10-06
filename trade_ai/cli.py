"""Command line interface.

Examples:
    python -m trade_ai backtest --synthetic
    python -m trade_ai backtest --csv prices.csv --long-only
    python -m trade_ai backtest --ticker SPY --start 2010-01-01
    python -m trade_ai signal --ticker 7203.T
"""

from __future__ import annotations

import argparse
import json

from .data import load_csv, load_yfinance, synthetic_ohlcv
from .metrics import format_summary
from .model import WalkForwardConfig
from .pipeline import PipelineConfig, latest_signal, run_pipeline
from .strategy import StrategyConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV file with date, open, high, low, close, volume")
    src.add_argument("--ticker", help="download via yfinance")
    src.add_argument("--synthetic", action="store_true", help="use generated demo data")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--label-threshold", type=float, default=0.0,
                   help="label threshold in units of horizon volatility")
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
    p.add_argument("--no-gate", action="store_true", help="disable the hit-rate quality gate")
    p.add_argument("--gate-window", type=int, default=120)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--periods-per-year", type=int, default=252)


def _load(args):
    if args.csv:
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


def _config(args) -> PipelineConfig:
    return PipelineConfig(
        label_threshold=args.label_threshold,
        walk_forward=WalkForwardConfig(
            horizon=args.horizon, min_train=args.min_train, retrain_every=args.retrain_every,
            train_window=args.train_window, model=args.model,
        ),
        strategy=StrategyConfig(
            entry_band=args.entry_band, full_edge=args.full_edge, long_only=args.long_only, target_vol=args.target_vol,
            max_leverage=args.max_leverage, smoothing=args.smoothing,
            quality_gate=not args.no_gate, gate_window=args.gate_window,
            periods_per_year=args.periods_per_year,
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
    args = parser.parse_args(argv)

    df = _load(args)
    cfg = _config(args)
    if args.command == "backtest":
        res = run_pipeline(df, cfg)
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
        print(json.dumps(latest_signal(df, cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
