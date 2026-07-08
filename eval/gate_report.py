#!/usr/bin/env python3
"""P3 primary solver-quality artifact (plan §7): run the P2 Proof-Code Gate over the devset
and report, per category, PRECISION (must be 100% — any wrong PROVEN answer is a listed
failure) and RECALL (fraction answered for 0 tokens). Scored via eval.score.

Usage:  python -m eval.gate_report [--holdout]   (--holdout adds the adversarial variants)
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from agent.gate import solve as gate_solve
from eval.devset import load_devset
from eval.score import score_task


def run(include_holdout: bool) -> int:
    tasks = load_devset(include_holdout=include_holdout)
    totals: dict[str, int] = defaultdict(int)
    proven: dict[str, int] = defaultdict(int)
    wrong: list[tuple[str, str, object]] = []

    for t in tasks:
        cat = t["category"]
        totals[cat] += 1
        answer = gate_solve(cat, t["prompt"])
        if answer is None:
            continue
        proven[cat] += 1
        if not score_task(t, answer):
            wrong.append((t["id"], cat, answer))

    total_proven = sum(proven.values())
    total_tasks = sum(totals.values())
    precision = 0.0 if not total_proven else 100.0 * (total_proven - len(wrong)) / total_proven
    print(f"GATE_PRECISION {precision:.2f}% proven={total_proven} wrong={len(wrong)}")
    print(f"| category | proven | total | recall |")
    print(f"|----------|-------:|------:|-------:|")
    for cat in sorted(totals):
        print(f"| {cat} | {proven[cat]} | {totals[cat]} | {proven[cat] / totals[cat]:.0%} |")
    print(f"| **all** | {total_proven} | {total_tasks} | {total_proven / total_tasks:.0%} |")

    if wrong:
        print("\nOFFENDING PROVEN CASES (fix the solver's defer rule):")
        for tid, cat, ans in wrong[:10]:
            print(f"  {tid} [{cat}] -> {ans!r}")
        return 1
    print("\nPASS: gate precision = 100% (no wrong proven answers)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="include adversarial variants")
    args = ap.parse_args()
    return run(args.holdout)


if __name__ == "__main__":
    raise SystemExit(main())
