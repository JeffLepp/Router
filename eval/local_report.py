#!/usr/bin/env python3
"""P3 DG-5 artifact (plan §7): per enabled category, how many gate-DEFERRED tasks the Tier 1.5
local candidate ACCEPTS and how many of those are correct (accepted-local pass-rate). DG-5 keeps
a category ON only if accepted-local pass-rate >= that category's Fireworks pass-rate WITH a net
token saving.

Runs try_local over the devset with whatever local client is configured: the StubLocalLLM by
default (offline CI -> expect ~0 accepted, this only exercises the plumbing), or the real baked
GGUF when MODEL_GGUF/llama-server is up (the meaningful numbers -- same measurement as
scripts/bench_local.py). The Fireworks pass-rate column is **[own-key]** (needs eval/run_eval on
a real key).

Usage:  python -m eval.local_report [--holdout]
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

from agent.gate import solve as gate_solve
from agent.local_gate import try_local
from eval.devset import load_devset
from eval.score import score_task

_ENABLED = [
    "actual_qa", "math_reasoning", "sentiment_analysis",
    "named_entity_recognition", "code_debugging", "logic_puzzles",
]


def _local_cfg(category: str) -> dict:
    cats = {c: (c == category) for c in [
        "actual_qa", "math_reasoning", "sentiment_analysis", "summarization",
        "named_entity_recognition", "code_debugging", "logic_puzzles", "code_generation",
        "formatting"]}
    return {
        "enabled": True, "categories": cats, "self_consistency_k": 2, "temp": 0,
        "max_tokens": 64, "latency_cap_s": 5, "min_classifier_confidence": 0.5,
    }


async def _run(include_holdout: bool) -> int:
    tasks = load_devset(include_holdout=include_holdout)
    accepted: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    eligible: dict[str, int] = defaultdict(int)

    for t in tasks:
        cat = t["category"]
        if cat not in _ENABLED:
            continue
        if gate_solve(cat, t["prompt"]) is not None:
            continue  # gate proves it free -> not a local-tier win
        eligible[cat] += 1
        ans = await try_local(cat, t["prompt"], config=_local_cfg(cat),
                              classification_confidence=0.9)
        if ans is None:
            continue
        accepted[cat] += 1
        if score_task(t, ans):
            correct[cat] += 1

    print("| category | eligible | accepted | correct | accepted-local pass-rate | Fireworks pass-rate |")
    print("|----------|---------:|---------:|--------:|:------------------------:|:-------------------:|")
    wrong_total = 0
    for cat in _ENABLED:
        acc, cor, elig = accepted[cat], correct[cat], eligible[cat]
        wrong_total += acc - cor
        rate = f"{cor}/{acc} ({cor / acc:.0%})" if acc else "-"
        print(f"| {cat} | {elig} | {acc} | {cor} | {rate} | [own-key] |")
    print("\n> Accepted-local precision must be 100% (accepted == correct) before a category ships "
          "local=on. Fireworks pass-rate is the [own-key] `eval/run_eval` column; DG-5 keeps a "
          "category on only if local pass-rate >= Fireworks pass-rate with net token saving.")
    if wrong_total:
        print(f"\nWARN: {wrong_total} accepted-but-wrong local answer(s) -- tighten the accept gate.")
    return 1 if wrong_total else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    return asyncio.run(_run(args.holdout))


if __name__ == "__main__":
    raise SystemExit(main())
