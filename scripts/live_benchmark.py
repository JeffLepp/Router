#!/usr/bin/env python3
"""Run an end-to-end router benchmark against mock or live Fireworks.

Default mode runs on the host. Use --container-image to exercise the built image.
The script never writes into the agent runtime; it creates a temporary input/output
workspace, runs the agent, scores the answers, and writes a benchmark report.

Examples:
  python -m scripts.live_benchmark --mock --limit 16
  python -m scripts.live_benchmark --holdout --score-judge
  python -m scripts.live_benchmark --container-image localhost:5000/floor-cl:test
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_yaml  # noqa: E402

JUDGE_METHODS = {"semantic_similarity", "contains_essential_points"}
ALL_CATEGORIES = [
    "actual_qa",
    "math_reasoning",
    "sentiment_analysis",
    "summarization",
    "named_entity_recognition",
    "code_debugging",
    "logic_puzzles",
    "code_generation",
]
LOCAL_CATEGORIES = {
    "actual_qa",
    "math_reasoning",
    "sentiment_analysis",
    "named_entity_recognition",
    "code_debugging",
    "logic_puzzles",
}


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.holdout:
        from eval.devset import load_devset

        tasks = load_devset(include_holdout=True)
    else:
        tasks = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        for task in tasks:
            task.setdefault("holdout", False)

    if args.category:
        wanted = set(args.category)
        tasks = [task for task in tasks if task.get("category") in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")
    return tasks


def _agent_rows(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id") or task.get("id") or f"task_{index}")
        rows.append({"task_id": task_id, "prompt": str(task.get("prompt", ""))})
    return rows


def _write_config(src: Path, dst: Path, args: argparse.Namespace) -> None:
    config = load_yaml(src)
    config["wall_seconds"] = float(args.wall_seconds)

    remote = dict(config.get("remote") or {})
    remote["timeout_seconds"] = float(args.remote_timeout)
    config["remote"] = remote

    local = dict(config.get("local_candidate") or {})
    local["enabled"] = args.floor == "floor-cl"
    categories = dict(local.get("categories") or {})
    for category in ALL_CATEGORIES:
        categories[category] = args.floor == "floor-cl" and category in LOCAL_CATEGORIES
    local["categories"] = categories
    config["local_candidate"] = local

    dst.write_text(_dump_yaml(config), encoding="utf-8")


def _dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_yaml(value, indent + 2).rstrip("\n"))
        elif isinstance(value, bool):
            lines.append(f"{pad}{key}: {'true' if value else 'false'}")
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{pad}{key}: "{escaped}"')
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines) + "\n"


def _env_for_run(args: argparse.Namespace, mock_base_url: str | None) -> dict[str, str]:
    env = os.environ.copy()
    _load_env_file(env, Path(args.env_file))
    if mock_base_url:
        env.update(
            {
                "FIREWORKS_BASE_URL": mock_base_url,
                "FIREWORKS_API_KEY": "mock-key",
                "ALLOWED_MODELS": "mock-8b-instruct",
                "AGENT_FORCE_STUB": "1",
            }
        )
        return env

    missing = [
        name
        for name in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        if not env.get(name)
    ]
    if missing:
        raise SystemExit(
            "missing required env for live Fireworks benchmark: "
            + ", ".join(missing)
            + "\nSet them in your shell; do not paste API keys into chat."
        )
    return env


def _load_env_file(env: dict[str, str], path: Path) -> None:
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value


def _run_host(
    work: Path,
    config: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    run_env = dict(env)
    run_env.update(
        {
            "PYTHONPATH": str(ROOT),
            "INPUT_PATH": str(work / "input" / "tasks.json"),
            "OUTPUT_PATH": str(work / "output" / "results.json"),
            "CONFIG_PATH": str(config),
            "AGENT_LEDGER_PATH": str(work / "output" / "ledger.json"),
            "AGENT_WALL_SECONDS": str(timeout),
        }
    )
    return subprocess.run(
        [PYTHON, "-m", "agent.main"],
        cwd=ROOT,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )


def _run_container(
    work: Path,
    config: Path,
    env: dict[str, str],
    image: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-e",
        "CONFIG_PATH=/cfg/config.yaml",
        "-e",
        "AGENT_LEDGER_PATH=/output/ledger.json",
        "-e",
        f"AGENT_WALL_SECONDS={timeout}",
        "-e",
        "FIREWORKS_BASE_URL",
        "-e",
        "ALLOWED_MODELS",
        "-v",
        f"{work / 'input'}:/input:ro",
        "-v",
        f"{work / 'output'}:/output",
        "-v",
        f"{config}:/cfg/config.yaml:ro",
    ]
    if env.get("FIREWORKS_API_KEY"):
        cmd.extend(["-e", "FIREWORKS_API_KEY"])
    if env.get("AGENT_FORCE_STUB"):
        cmd.extend(["-e", "AGENT_FORCE_STUB"])
    cmd.append(image)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout + 60,
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


async def _gate_proven_ids(tasks: list[dict[str, Any]]) -> set[str]:
    from agent.classify import classify
    from agent.gate import solve as gate_solve

    proven: set[str] = set()
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id") or task.get("id") or f"task_{index}")
        classification = await classify(str(task.get("prompt", "")), None)
        if classification.confidence < 0.6:
            continue
        if gate_solve(classification.category, str(task.get("prompt", ""))) is not None:
            proven.add(task_id)
    return proven


def _route_of(task_id: str, gate_ids: set[str], remote_ids: set[str]) -> str:
    if task_id in gate_ids:
        return "gate"
    if task_id in remote_ids:
        return "remote"
    return "local/other"


def _acc_rate(bucket: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, counts in sorted(bucket.items()):
        scored = counts["scored"]
        out[key] = {
            "passed": counts["passed"],
            "scored": scored,
            "accuracy": round(counts["passed"] / scored, 4) if scored else None,
        }
    return out


def _score(
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    ledger: dict[str, Any],
    gate_ids: set[str],
    score_judge: bool,
) -> dict[str, Any]:
    from eval.score import score_task

    by_result = {str(row.get("task_id")): str(row.get("answer", "")) for row in results}
    ledger_by_task = {str(e.get("task_id")): e for e in ledger.get("entries", [])}
    remote_ids = set(ledger_by_task)
    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    by_route: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "scored": 0})
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "scored": 0})
    tokens_by_category: dict[str, int] = defaultdict(int)

    total = passed = passed_judged = scored = unscored = missing = 0
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id") or task.get("id") or f"task_{index}")
        category = str(task.get("category", "unknown"))
        method = str(task.get("evaluation_method", ""))
        answer = by_result.get(task_id, "")
        entry = ledger_by_task.get(task_id)
        route = _route_of(task_id, gate_ids, remote_ids)
        model = str(entry.get("model") or route) if entry else route

        total += 1
        per_category[category]["total"] += 1
        if task_id in gate_ids:
            per_category[category]["gate"] += 1
        if task_id in remote_ids:
            per_category[category]["remote"] += 1
        if entry:
            tokens_by_category[category] += int(entry.get("total_tokens", 0))
        if task_id not in by_result:
            missing += 1
            per_category[category]["missing"] += 1

        if method in JUDGE_METHODS and not score_judge:
            unscored += 1
            per_category[category]["unscored"] += 1
            continue

        scored += 1
        per_category[category]["scored"] += 1
        by_route[route]["scored"] += 1
        by_model[model]["scored"] += 1
        try:
            ok = bool(score_task(task, answer))
        except Exception as exc:
            ok = False
            error = str(exc)
        else:
            error = ""
        # accuracy_judged: strict + eval-only judge rescue of Java/C code (own key, off-path).
        try:
            passed_judged += 1 if (ok or score_task(task, answer, judge=True)) else 0
        except Exception:
            passed_judged += 1 if ok else 0
        if ok:
            passed += 1
            per_category[category]["passed"] += 1
            by_route[route]["passed"] += 1
            by_model[model]["passed"] += 1
        else:
            failures.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "route": route,
                    "model": model,
                    "tokens": int(entry.get("total_tokens", 0)) if entry else 0,
                    "method": method,
                    "expected": str(task.get("expected_answer", "")),
                    "actual": answer,
                    "prompt": str(task.get("prompt", "")),
                    "answer": answer[:500],
                    "error": error,
                }
            )

    local_or_other = total - len(remote_ids) - len(gate_ids)
    return {
        "total": total,
        "scored": scored,
        "passed": passed,
        "unscored": unscored,
        "missing": missing,
        "accuracy": round(passed / scored, 4) if scored else None,
        "accuracy_strict": round(passed / scored, 4) if scored else None,
        "accuracy_judged": round(passed_judged / scored, 4) if scored else None,
        "gate_proven": len(gate_ids),
        "remote_called": len(remote_ids),
        "local_or_other": max(0, local_or_other),
        "per_category": {cat: dict(counts) for cat, counts in sorted(per_category.items())},
        "accuracy_by_route": _acc_rate(by_route),
        "accuracy_by_model": _acc_rate(by_model),
        "tokens_by_category": dict(sorted(tokens_by_category.items())),
        "failures": failures,
    }


def _print_report(report: dict[str, Any], max_failures: int) -> None:
    score = report["score"]
    ledger = report["ledger"]
    print("\n# Live Benchmark\n")
    print(f"runner      {report['runner']}")
    print(f"floor       {report['floor']}")
    print(f"mock        {report['mock']}")
    print(f"tasks       {score['total']} scored={score['scored']} unscored={score['unscored']}")
    if score["accuracy"] is None:
        print("accuracy    n/a")
    else:
        print(f"accuracy    strict {score['accuracy_strict']:.2%}  judged {score['accuracy_judged']:.2%}  ({score['passed']}/{score['scored']} strict)")
    print(f"sources     gate={score['gate_proven']} remote={score['remote_called']} local/other={score['local_or_other']}")
    print(f"tokens      total={ledger.get('total_tokens', 0)} prompt={ledger.get('prompt_tokens', 0)} completion={ledger.get('completion_tokens', 0)}")
    print(f"requests    {ledger.get('requests', 0)}")
    print(f"errors      remote={report['remote_errors']}")
    print(f"wall        {report['wall_s']:.2f}s")
    print(f"out_dir     {report['out_dir']}")

    print("\n| category | pass | scored | total | gate | remote | unscored |")
    print("|----------|-----:|-------:|------:|-----:|-------:|---------:|")
    for category, row in score["per_category"].items():
        print(
            f"| {category} | {row.get('passed', 0)} | {row.get('scored', 0)} | "
            f"{row.get('total', 0)} | {row.get('gate', 0)} | {row.get('remote', 0)} | "
            f"{row.get('unscored', 0)} |"
        )

    if score["failures"]:
        print(f"\nfirst {min(max_failures, len(score['failures']))} failures:")
        for failure in score["failures"][:max_failures]:
            print(
                f"- {failure['task_id']} [{failure['category']}/{failure['method']}]: "
                f"{failure['answer']!r}"
            )


FAILURE_CSV_FIELDS = [
    "task_id", "category", "route", "model", "tokens", "method", "expected", "actual", "prompt",
]


def _write_failures_csv(out_dir: Path, failures: list[dict[str, Any]]) -> None:
    with (out_dir / "failures.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in failures:
            writer.writerow({k: row.get(k, "") for k in FAILURE_CSV_FIELDS})


def _copy_artifacts(work: Path, out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.json",):
        shutil.copy2(work / "input" / name, out_dir / name)
    for name in ("results.json", "ledger.json"):
        src = work / "output" / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    _write_failures_csv(out_dir, report["score"]["failures"])
    # Keep benchmark.json failures slim; full expected/actual/prompt live in failures.csv.
    slim = {"task_id", "category", "method", "answer", "error"}
    report["score"]["failures"] = [
        {k: v for k, v in row.items() if k in slim} for row in report["score"]["failures"]
    ]
    (out_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "dataset.json"))
    parser.add_argument("--holdout", action="store_true", help="include eval/devset holdout variants")
    parser.add_argument("--category", action="append", choices=ALL_CATEGORIES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="use the local mock Fireworks server")
    parser.add_argument("--score-judge", action="store_true", help="score LLM-judge methods too")
    parser.add_argument("--floor", choices=("floor-c", "floor-cl"), default="floor-c")
    parser.add_argument("--config", default=str(ROOT / "agent" / "config.yaml"))
    parser.add_argument("--container-image", help="run the benchmark inside this image")
    parser.add_argument("--env-file", default=".env.local", help="local env file for Fireworks credentials")
    parser.add_argument("--out-dir", help="directory for results, ledger, and report")
    parser.add_argument("--wall-seconds", type=float, default=510.0)
    parser.add_argument("--remote-timeout", type=float, default=25.0)
    parser.add_argument("--max-failures", type=int, default=12)
    args = parser.parse_args()

    if args.mock and args.container_image:
        raise SystemExit("--mock is host-only; omit --container-image for mock smoke tests")

    tasks = _load_tasks(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        tempfile.mkdtemp(prefix="live_benchmark_")
    )
    work = Path(tempfile.mkdtemp(prefix="agent_run_"))
    (work / "input").mkdir()
    (work / "output").mkdir()
    (work / "input" / "tasks.json").write_text(
        json.dumps(_agent_rows(tasks), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = work / "config.yaml"
    _write_config(Path(args.config), config, args)

    mock_server = None
    mock_base_url = None
    if args.mock:
        from scripts.acceptance_p1 import free_port, start_mock

        port = free_port()
        mock_server = start_mock(port)
        mock_base_url = f"http://127.0.0.1:{port}"

    try:
        env = _env_for_run(args, mock_base_url)
        start = time.monotonic()
        if args.container_image:
            proc = _run_container(work, config, env, args.container_image, args.wall_seconds)
            runner = f"container:{args.container_image}"
        else:
            proc = _run_host(work, config, env, args.wall_seconds)
            runner = "host"
        wall_s = time.monotonic() - start
    finally:
        if mock_server is not None:
            mock_server.kill()

    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)

    results = _read_json(work / "output" / "results.json", [])
    ledger = _read_json(work / "output" / "ledger.json", {})
    gate_ids = asyncio.run(_gate_proven_ids(tasks))
    score = _score(tasks, results, ledger, gate_ids, args.score_judge)
    remote_errors = proc.stderr.count("remote_error ")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "floor": args.floor,
        "mock": bool(args.mock),
        "wall_s": round(wall_s, 3),
        "score_judge": bool(args.score_judge),
        "out_dir": str(out_dir),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "remote_errors": remote_errors,
        "ledger": ledger,
        "score": score,
    }
    _copy_artifacts(work, out_dir, report)
    _print_report(report, args.max_failures)
    return 0 if score["missing"] == 0 and remote_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
