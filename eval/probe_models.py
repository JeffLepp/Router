#!/usr/bin/env python3
"""P3 launch-day model probe (plan §7/§8) -- runs OUTSIDE the container, own key. [own-key]

Starts from the known allowed list, FAILS CLOSED if the official ALLOWED_MODELS differs (so a
silently-changed roster can't route to a model we never validated), runs the devset hard slice
against each allowed model, scores with eval.score, and emits the per-category tier map to bake
into agent/config.yaml's model routing.

Usage (launch day, with key):
  export FIREWORKS_API_KEY=...  FIREWORKS_BASE_URL=...  ALLOWED_MODELS=<official list>
  python -m eval.probe_models                # fails closed if ALLOWED_MODELS != known
  python -m eval.probe_models --force-rerank  # accept the new roster and re-rank

Offline self-check (no key, logic only):  python -m eval.probe_models --self-check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

from agent.remote import parse_allowed_models
from eval.devset import load_devset
from eval.score import score_task

KNOWN_MODELS = [
    "minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it",
    "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4",
]


def check_roster(allowed: list[str], force: bool) -> list[str]:
    """Fail closed: refuse to probe an unexpected roster unless --force-rerank."""
    if not allowed:
        raise SystemExit("ALLOWED_MODELS is empty -- set it to the official list.")
    if set(allowed) != set(KNOWN_MODELS) and not force:
        raise SystemExit(
            "ALLOWED_MODELS differs from the known validated list:\n"
            f"  known:    {sorted(KNOWN_MODELS)}\n  official: {sorted(allowed)}\n"
            "Re-run with --force-rerank to accept the new roster and re-probe."
        )
    return allowed


def tier_map(scores: dict[str, dict[str, tuple[int, int]]]) -> dict[str, str]:
    """scores[category][model] = (passed, total) -> best model per category (argmax pass-rate)."""
    out: dict[str, str] = {}
    for category, per_model in scores.items():
        best = max(per_model.items(), key=lambda kv: (kv[1][0] / kv[1][1] if kv[1][1] else 0, -len(kv[0])))
        out[category] = best[0]
    return out


async def _ask(model: str, prompt: str, max_tokens: int = 128) -> str:
    import httpx

    base = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base.rstrip('/v1')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
            json={
                "model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": "Answer with the smallest decisive payload. English only."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()


async def probe(allowed: list[str], hard_only: bool) -> dict[str, dict[str, tuple[int, int]]]:
    tasks = load_devset(include_holdout=True)
    if hard_only:
        tasks = [t for t in tasks if t.get("difficulty") == "hard" or t.get("holdout")]
    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for t in tasks:
        for model in allowed:
            try:
                ans = await _ask(model, t["prompt"])
                ok = score_task(t, ans)
            except Exception as exc:  # keep probing; a flaky model shouldn't abort the sweep
                print(f"  {model} failed on {t['id']}: {exc}", file=sys.stderr)
                ok = False
            cell = scores[t["category"]][model]
            cell[0] += int(ok)
            cell[1] += 1
    return {c: {m: (v[0], v[1]) for m, v in per.items()} for c, per in scores.items()}


async def _main_async(args) -> int:
    allowed = check_roster(parse_allowed_models(), args.force_rerank)
    scores = await probe(allowed, hard_only=not args.full)
    chosen = tier_map(scores)
    print("\n# Per-category tier map (bake into agent/config.yaml CATEGORY_MODEL_PATTERNS)\n")
    print("| category | best model | pass-rate |")
    print("|----------|------------|:---------:|")
    for cat in sorted(scores):
        p, tot = scores[cat][chosen[cat]]
        print(f"| {cat} | {chosen[cat]} | {p}/{tot} |")
    print("\ntier_map = " + json.dumps(chosen, indent=2))
    return 0


def _self_check() -> None:
    # fail-closed roster guard
    try:
        check_roster(["some-new-model"], force=False)
        raise AssertionError("should have failed closed on unexpected roster")
    except SystemExit:
        pass
    assert check_roster(KNOWN_MODELS, force=False) == KNOWN_MODELS
    assert check_roster(["x"], force=True) == ["x"]
    # argmax tier map
    scores = {
        "math_reasoning": {"minimax-m3": (9, 10), "gemma-4-31b-it": (6, 10)},
        "code_generation": {"kimi-k2p7-code": (8, 10), "minimax-m3": (5, 10)},
    }
    tm = tier_map(scores)
    assert tm == {"math_reasoning": "minimax-m3", "code_generation": "kimi-k2p7-code"}, tm
    print("PASS eval.probe_models self-check (fail-closed + tier-map argmax)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true", help="offline logic check, no key")
    ap.add_argument("--force-rerank", action="store_true", help="accept a roster != KNOWN_MODELS")
    ap.add_argument("--full", action="store_true", help="probe the whole devset, not just the hard slice")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return 0
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
