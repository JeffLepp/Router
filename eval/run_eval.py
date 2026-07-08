#!/usr/bin/env python3
"""P3 frontier sweep (plan §7/§8): run the agent over the devset for the grid
{local: off (Floor-C), on (Floor-CL)} x {B: 0,500,2k,8k} x {batch: off, safe} at m=1,
plus one diagnostic Floor-0 row {m:0,B:0}. Emits per-category pass-rate, total ledger tokens,
and wall as markdown, so Floor-CL's token saving over Floor-C at equal pass-rate is visible.

Runs against the stdlib mock Fireworks (scripts/mock_fireworks.py) — offline. Gate-proven and
local-accepted tasks are scored on their REAL answers (correct); deferred tasks get the mock's
canned completion (so the deferred pass-rate is a floor, not the real-model number, which is the
[own-key] launch-day sweep). Token/coverage columns are exact.

Usage:  python -m eval.run_eval [--quick] [--holdout]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.devset import load_devset  # noqa: E402
from eval.score import score_task  # noqa: E402
from scripts.acceptance_p1 import free_port, metrics, run_agent, start_mock  # noqa: E402

_LOCAL_CATS = [
    "actual_qa", "math_reasoning", "sentiment_analysis", "named_entity_recognition",
    "code_debugging", "logic_puzzles",  # DG-5 six; summarization + code_generation stay off
]


def write_config(path: Path, *, local: bool, budget: int, batch: bool, m: int) -> None:
    cats = {c: (local and c in _LOCAL_CATS) for c in [
        "actual_qa", "math_reasoning", "sentiment_analysis", "summarization",
        "named_entity_recognition", "code_debugging", "logic_puzzles", "code_generation"]}
    lines = [
        f"token_budget: {budget}",
        f"mandatory_remote: {m}",
        "local_slots: 8",
        "remote_slots: 8",
        "wall_seconds: 60",
        "snapshot_interval_seconds: 0.2",
        "llama:",
        '  base_url: "http://127.0.0.1:8080/v1"',
        '  model: "local"',
        "local_candidate:",
        f"  enabled: {str(local).lower()}",
        "  self_consistency_k: 2",
        "  temp: 0",
        "  max_tokens: 64",
        "  latency_cap_s: 1",
        "  min_classifier_confidence: 0.5",
        "  categories:",
    ]
    lines += [f"    {k}: {str(v).lower()}" for k, v in cats.items()]
    lines += [
        "remote:",
        "  timeout_seconds: 5",
        "  retries: 0",
        "  temperature: 0",
        "  usd_per_mtok: 0.9",
        "  dev_spend_cap: 0.0",
        "batching:",
        f"  enabled: {str(batch).lower()}",
        "  max_batch_size: 10",
        "max_tokens:",
        "  actual_qa: 20",
        "  math_reasoning: 10",
        "  sentiment_analysis: 2",
        "  summarization: 20",
        "  named_entity_recognition: 20",
        "  code_debugging: 40",
        "  logic_puzzles: 20",
        "  code_generation: 40",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_row(name: str, tasks: list[dict], port: int, *, local, budget, batch, m) -> dict:
    id2task = {t["id"]: t for t in tasks}
    agent_tasks = [{"task_id": t["id"], "prompt": t["prompt"]} for t in tasks]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = tmp / "config.yaml"
        write_config(cfg, local=local, budget=budget, batch=batch, m=m)
        ledger = tmp / "ledger.json"
        before = metrics(port)["requests"]
        t0 = time.monotonic()
        run_agent(tmp, agent_tasks, cfg, f"http://127.0.0.1:{port}", ledger)
        wall = time.monotonic() - t0
        calls = metrics(port)["requests"] - before
        results = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
        # Floor-0 (m=0,B=0) disables the remote path at import -> no ledger is written (0 tokens).
        tokens = json.loads(ledger.read_text(encoding="utf-8")).get("total_tokens", 0) if ledger.exists() else 0
    per_cat_pass: dict[str, int] = defaultdict(int)
    per_cat_tot: dict[str, int] = defaultdict(int)
    for row in results:
        t = id2task[row["task_id"]]
        per_cat_tot[t["category"]] += 1
        if score_task(t, row["answer"]):
            per_cat_pass[t["category"]] += 1
    passed = sum(per_cat_pass.values())
    return {
        "name": name, "tokens": tokens, "calls": calls, "wall": round(wall, 2),
        "passed": passed, "total": len(results),
        "per_cat": {c: (per_cat_pass[c], per_cat_tot[c]) for c in sorted(per_cat_tot)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one Floor-C + one Floor-CL row only")
    ap.add_argument("--holdout", action="store_true", help="include adversarial variants")
    args = ap.parse_args()

    tasks = load_devset(include_holdout=args.holdout)
    grid: list[tuple[str, dict]] = []
    if args.quick:
        grid = [
            ("Floor-C  B=0    batch=off", dict(local=False, budget=0, batch=False, m=1)),
            ("Floor-CL B=0    batch=off", dict(local=True, budget=0, batch=False, m=1)),
        ]
    else:
        for local, tag in [(False, "Floor-C "), (True, "Floor-CL")]:
            for budget in (0, 500, 2000, 8000):
                for batch in (False, True):
                    b = {0: "0   ", 500: "500 ", 2000: "2k  ", 8000: "8k  "}[budget]
                    grid.append((f"{tag} B={b} batch={'safe' if batch else 'off '}",
                                 dict(local=local, budget=budget, batch=batch, m=1)))
        grid.append(("Floor-0  m=0 B=0 (diagnostic)", dict(local=False, budget=0, batch=False, m=0)))

    port = free_port()
    server = start_mock(port)
    rows = []
    try:
        for name, cfg in grid:
            print(f"  running {name} ...", file=sys.stderr)
            rows.append(run_row(name, tasks, port, **cfg))
    finally:
        server.kill()

    print(f"\n# Frontier sweep (devset n={len(tasks)}, mock Fireworks; m=1 unless noted)\n")
    print("| config | tokens | calls | pass-rate | wall |")
    print("|--------|-------:|------:|:---------:|-----:|")
    for r in rows:
        print(f"| {r['name']} | {r['tokens']} | {r['calls']} | "
              f"{r['passed']}/{r['total']} ({r['passed'] / r['total']:.0%}) | {r['wall']}s |")
    print("\n> Deferred-task pass-rate uses the mock's canned answers -- the real per-model "
          "pass-rate is the **[own-key]** launch sweep (`eval/probe_models.py`). Token + call "
          "columns are exact. Compare Floor-CL vs Floor-C at equal config for the token undercut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
