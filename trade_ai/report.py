"""Markdown summary of a paper/live trading state directory."""

from __future__ import annotations

import json
import os


def _read_decisions(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def paper_report(state_dir: str, initial_equity: float = 10_000.0, last_n: int = 10) -> str:
    """Equity vs start and vs buy & hold, current positions, and the latest decisions."""
    decisions = _read_decisions(os.path.join(state_dir, "decisions.jsonl"))
    priced = [d for d in decisions if "equity" in d and "price" in d]
    acct_path = os.path.join(state_dir, "paper_account.json")
    account = {}
    if os.path.exists(acct_path):
        with open(acct_path) as f:
            account = json.load(f)

    lines = ["## Paper trading"]
    if not priced:
        lines.append("\nNo priced decisions yet.")
    else:
        first, last = priced[0], priced[-1]
        equity = last["equity"]
        lines += [
            "",
            f"- Period: {first.get('bar')} -> {last.get('bar')} ({len(priced)} cycles)",
            f"- Equity: **{equity:,.0f}** (start {initial_equity:,.0f}, "
            f"{equity / initial_equity - 1:+.2%})",
            f"- Buy & hold over the same period: {last['price'] / first['price'] - 1:+.2%} "
            f"({last.get('symbol', '')} {first['price']:,.2f} -> {last['price']:,.2f})",
            f"- Fills: {len(account.get('fills', []))}",
        ]
        held = {s: q for s, q in account.get("positions", {}).items() if q}
        lines.append(f"- Positions: {', '.join(f'{s} {q:+.6f}' for s, q in held.items()) or 'flat'}")
    errors = [d for d in decisions if d.get("action") in ("error", "skip")]
    if errors:
        lines.append(f"- Errors / skipped cycles: {len(errors)} (latest: "
                     f"{errors[-1].get('error') or errors[-1].get('reason')})")
    if decisions:
        lines += ["", "| time | symbol | action | target | position | price | equity |",
                  "|---|---|---|---|---|---|---|"]
        for d in decisions[-last_n:]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                d.get("time", ""), d.get("symbol", ""), d.get("action", ""),
                f"{d['target_fraction']:+.2f}" if "target_fraction" in d else "",
                f"{d['position']:+.6f}" if "position" in d else "",
                f"{d['price']:,.2f}" if "price" in d else "",
                f"{d['equity']:,.0f}" if "equity" in d else ""))
    return "\n".join(lines) + "\n"
