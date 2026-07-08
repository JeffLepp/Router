#!/usr/bin/env python3
"""P5 deliverable 2: launch-day dress rehearsal.

N synthetic tasks, Floor-C (local=off, m=1, B=high), 10-minute wall. Reports the
numbers plan §4/§8 want: gate-proven count, tokens spent + tokens the gate saved,
deferred-task completion time (wall), and the final ledger. Uses the mock Fireworks
(scripts/mock_fireworks.py) so it runs offline — the REAL sweep (P3, own key) is a
separate step; this rehearses orchestration + the gate, not model accuracy.

Usage:  python -m scripts.dress_rehearsal [--n 150] [--wall 600]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.classify import classify  # noqa: E402
from agent.gate import solve as gate_solve  # noqa: E402
from scripts.acceptance_p1 import (  # noqa: E402
    fake_tasks,
    free_port,
    metrics,
    run_agent,
    start_mock,
    validate_results,
    write_config,
)


async def _gate_proven(tasks: list[dict]) -> int:
    """Count tasks the free gate answers — same rule main.py uses (conf >= 0.6)."""
    proven = 0
    for t in tasks:
        c = await classify(t["prompt"], None)
        if c.confidence >= 0.6 and gate_solve(c.category, t["prompt"]) is not None:
            proven += 1
    return proven


def rehearse(n: int, wall: float) -> dict:
    tasks = fake_tasks(n)
    proven = asyncio.run(_gate_proven(tasks))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        server = start_mock(port)
        try:
            cfg = tmp / "floor_c.yaml"
            # Floor-C anchor: local off, every unproven task gets one minimal call.
            write_config(cfg, token_budget=0, mandatory_remote=1)
            ledger = tmp / "ledger.json"
            before = metrics(port)["requests"]
            t0 = time.monotonic()
            run_agent(tmp, tasks, cfg, f"http://127.0.0.1:{port}", ledger)
            wall_s = time.monotonic() - t0
            remote_calls = metrics(port)["requests"] - before
            rows = validate_results(tmp / "results.json")
            led = json.loads(ledger.read_text(encoding="utf-8"))
        finally:
            server.kill()
    tokens = led.get("total_tokens", 0)
    # The gate removed `proven` calls; at the run's own avg cost/call that is a
    # measured lower bound on tokens the gate saved vs a no-gate Floor-R baseline.
    avg = tokens / remote_calls if remote_calls else 0.0
    return {
        "n": len(rows),
        "gate_proven": proven,
        "deferred": len(rows) - proven,
        "remote_calls": remote_calls,
        "tokens_spent": tokens,
        "tokens_saved_est": round(proven * avg),
        "wall_s": round(wall_s, 2),
        "wall_budget_s": wall,
        "ledger": led,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--wall", type=float, default=600.0)
    args = ap.parse_args()
    r = rehearse(args.n, args.wall)

    print("== dress rehearsal (Floor-C, local=off, m=1, B=0; mock remote) ==")
    print(f"  tasks              {r['n']}")
    print(f"  gate-proven (free) {r['gate_proven']}  ({r['gate_proven'] / r['n']:.0%})")
    print(f"  deferred -> remote {r['deferred']}  (calls made: {r['remote_calls']})")
    print(f"  tokens spent       {r['tokens_spent']}")
    print(f"  tokens saved (est) {r['tokens_saved_est']}  (gate removed {r['gate_proven']} calls)")
    print(f"  wall / budget      {r['wall_s']}s / {r['wall_budget_s']}s")
    led = {k: v for k, v in r["ledger"].items() if k != "entries"}  # drop per-call rows
    print(f"  ledger             {json.dumps(led, sort_keys=True)}")

    # §4 targets: all tasks land in valid JSON, and wall stays under the 10-min budget.
    ok = r["n"] == args.n and r["wall_s"] < r["wall_budget_s"]
    print(f"\n{'PASS' if ok else 'FAIL'}: {r['n']}/{args.n} tasks, wall "
          f"{r['wall_s']}s {'<' if ok else '>='} {args.wall}s budget")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
