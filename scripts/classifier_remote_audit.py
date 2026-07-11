#!/usr/bin/env python3
"""Measure the opt-in remote batch classifier without running answer calls."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from agent.config import AgentConfig
from agent.main import Task, TaskState, _run_remote_classifier


ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def _audit(tasks: list[dict], config: AgentConfig) -> tuple[dict, object | None]:
    states = [
        TaskState(
            Task(
                str(task.get("task_id") or task.get("id") or f"task_{index}"),
                str(task.get("prompt", "")),
            )
        )
        for index, task in enumerate(tasks)
    ]
    return await _run_remote_classifier(
        states,
        config,
        time.monotonic() + float(config.wall_seconds),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="agent/config.remote-classifier.yaml")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--out")
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    missing = [
        name
        for name in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit("missing required env: " + ", ".join(missing))

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    config = AgentConfig.from_path(ROOT / args.config)
    if not config.remote.get("classifier_enabled", False):
        raise SystemExit("remote classifier is disabled in the selected config")

    overrides, client = asyncio.run(_audit(tasks, config))
    expected = {
        str(task.get("task_id") or task.get("id") or f"task_{index}"): task.get("category")
        for index, task in enumerate(tasks)
    }
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    for task_id, gold in expected.items():
        predicted = overrides.get(task_id, ("MISSING", 0.0))[0]
        confusion[str(gold)][predicted] += 1
        correct += int(predicted == gold)

    ledger = client.ledger.as_dict() if client is not None else {}
    report = {
        "dataset": str(dataset_path),
        "correct": correct,
        "total": len(tasks),
        "accuracy": correct / len(tasks) if tasks else 0.0,
        "coverage": len(overrides) / len(tasks) if tasks else 0.0,
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
        "ledger": ledger,
        "diagnostics": list(getattr(client, "classifier_diagnostics", []) or []),
    }
    print(
        f"remote classifier: {correct}/{len(tasks)} ({report['accuracy']:.2%}), "
        f"coverage={len(overrides)}/{len(tasks)}"
    )
    print(
        f"requests={ledger.get('requests', 0)} tokens={ledger.get('total_tokens', 0)} "
        f"estimated_usd={ledger.get('estimated_usd', 0)}"
    )
    for category, counts in report["confusion"].items():
        print(f"{category}: {counts}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
