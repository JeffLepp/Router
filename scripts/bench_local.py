#!/usr/bin/env python3
"""Summary-only Floor-CL candidate bakeoff.

The benchmark uses the real deterministic gate and local acceptance gate over the seven
unresolved public summaries plus five holdouts. A model qualifies only with zero accepted
errors, sufficient public recall, and the grader-style readiness/RSS/stage limits.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import solve as gate_solve  # noqa: E402
from agent.local_gate import try_local  # noqa: E402
from agent.local_llm import make_local_client  # noqa: E402
from eval.devset import load_devset  # noqa: E402
from eval.score import score_task  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(port: int, timeout: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return time.monotonic() - started
        except Exception:
            time.sleep(0.25)
    return -1.0


def _rss_bytes(pid: int) -> int:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


async def _monitor_rss(pid: int, stop: asyncio.Event, samples: list[int]) -> None:
    while not stop.is_set():
        samples.append(_rss_bytes(pid))
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            pass


def _applicable_summaries() -> list[dict[str, Any]]:
    rows = load_devset(include_holdout=True)
    return [
        row
        for row in rows
        if row["category"] == "summarization"
        and gate_solve("summarization", row["prompt"]) is None
    ]


def _local_config(task_timeout: float, stage_timeout: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "self_consistency_k": 1,
        "temp": 0,
        "max_tokens": 64,
        "latency_cap_s": task_timeout,
        "summary_queue_size": 1,
        "summary_retries": 1,
        "summary_task_timeout_s": task_timeout,
        "summary_stage_timeout_s": stage_timeout,
        "summary_max_tokens": {
            "one_sentence": 64,
            "three_sentences": 110,
            "structured_report": 150,
            "default": 150,
        },
        "categories": {"summarization": True},
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


async def _bench_model(
    name: str,
    gguf: str,
    *,
    server_bin: str,
    ctx_size: int,
    threads: int,
    startup_timeout: float,
    task_timeout: float,
    stage_timeout: float,
    rss_limit_gib: float,
    public_min: int,
    lfm_license_eligible: bool,
    only_ids: set[str],
) -> dict[str, Any]:
    port = _free_port()
    cmd = [
        server_bin,
        "--model",
        gguf,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--threads",
        str(threads),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "0",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return {"model": name, "error": f"server not found: {server_bin}"}
    try:
        cold = await asyncio.to_thread(_wait_health, port, startup_timeout)
        if cold < 0:
            return {"model": name, "error": "llama-server did not become healthy"}
        cfg = _local_config(task_timeout, stage_timeout)
        llama_cfg = {"base_url": f"http://127.0.0.1:{port}/v1", "model": "local"}
        client = await make_local_client(llama_cfg, cfg)
        tasks = _applicable_summaries()
        if only_ids:
            tasks = [task for task in tasks if task["id"] in only_ids]
        stage_started = time.monotonic()
        stage_deadline = stage_started + stage_timeout
        latencies: list[float] = []
        outputs: list[dict[str, Any]] = []
        rss_samples: list[int] = []
        monitor_stop = asyncio.Event()
        monitor = asyncio.create_task(_monitor_rss(proc.pid, monitor_stop, rss_samples))
        try:
            for index, task in enumerate(tasks):
                started = time.monotonic()
                diagnostics: dict[str, Any] = {}
                answer = await try_local(
                    "summarization",
                    task["prompt"],
                    config=cfg,
                    llama_config=llama_cfg,
                    client=client,
                    deadline=stage_deadline,
                    remaining_tasks=len(tasks) - index,
                    classification_confidence=1.0,
                    parallelism=1,
                    diagnostics=diagnostics,
                )
                elapsed = time.monotonic() - started
                latencies.append(elapsed)
                correct = bool(answer is not None and score_task(task, answer))
                outputs.append(
                    {
                        "id": task["id"],
                        "holdout": bool(task.get("holdout", False)),
                        "accepted": answer is not None,
                        "correct": correct,
                        "latency_s": round(elapsed, 3),
                        "answer": answer,
                        "candidate": diagnostics.get("candidate"),
                    }
                )
        finally:
            monitor_stop.set()
            await monitor

        stage_elapsed = time.monotonic() - stage_started
        accepted_rows = [row for row in outputs if row["accepted"]]
        wrong_rows = [row for row in accepted_rows if not row["correct"]]
        public_accepted = sum(
            row["accepted"] and not row["holdout"] for row in outputs
        )
        peak_rss = max(rss_samples or [0])
        license_eligible = lfm_license_eligible or "lfm" not in name.lower()
        qualified = (
            not wrong_rows
            and public_accepted >= public_min
            and cold <= startup_timeout
            and stage_elapsed <= stage_timeout
            and peak_rss <= rss_limit_gib * 1024**3
            and license_eligible
        )
        return {
            "model": name,
            "path": gguf,
            "gguf_bytes": Path(gguf).stat().st_size,
            "cold_s": round(cold, 3),
            "stage_s": round(stage_elapsed, 3),
            "p95_s": round(_p95(latencies), 3),
            "peak_rss_bytes": peak_rss,
            "applicable": len(tasks),
            "public_applicable": sum(not task.get("holdout", False) for task in tasks),
            "accepted": len(accepted_rows),
            "accepted_correct": len(accepted_rows) - len(wrong_rows),
            "public_accepted": public_accepted,
            "precision": 1.0 if not accepted_rows else (len(accepted_rows) - len(wrong_rows)) / len(accepted_rows),
            "license_eligible": license_eligible,
            "qualified": qualified,
            "outputs": outputs,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _parse_models(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return [("baked", os.environ.get("MODEL_GGUF", "/models/model.gguf"))]
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"invalid model pair: {item!r}")
        pairs.append((name.strip(), path.strip()))
    return pairs


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="name=path,name=path (default: baked $MODEL_GGUF)")
    parser.add_argument("--server-bin", default="llama-server")
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=45)
    parser.add_argument("--task-timeout", type=float, default=35)
    parser.add_argument("--stage-timeout", type=float, default=240)
    parser.add_argument("--rss-limit-gib", type=float, default=3.7)
    parser.add_argument("--code-batching-validated", action="store_true")
    parser.add_argument(
        "--lfm-license-eligible",
        action="store_true",
        help="confirm the submitting legal entity is eligible under LFM Open License v1.0",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--ids", help="comma-separated task ids for diagnostics")
    parser.add_argument("--require-qualifier", action="store_true")
    args = parser.parse_args()

    os.environ["EVAL_JUDGE_OFFLINE"] = "1"
    public_min = 5 if args.code_batching_validated else 6
    only_ids = {item.strip() for item in (args.ids or "").split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    for name, path in _parse_models(args.models):
        if not Path(path).is_file():
            rows.append({"model": name, "error": f"missing model: {path}"})
            continue
        print(f"benchmarking {name}: {path}", file=sys.stderr)
        rows.append(
            await _bench_model(
                name,
                path,
                server_bin=args.server_bin,
                ctx_size=args.ctx_size,
                threads=args.threads,
                startup_timeout=args.startup_timeout,
                task_timeout=args.task_timeout,
                stage_timeout=args.stage_timeout,
                rss_limit_gib=args.rss_limit_gib,
                public_min=public_min,
                lfm_license_eligible=args.lfm_license_eligible,
                only_ids=only_ids,
            )
        )

    qualified = [row for row in rows if row.get("qualified")]
    qualified.sort(
        key=lambda row: (
            -int(row["accepted_correct"]),
            float(row["p95_s"]),
            int(row["peak_rss_bytes"]),
            int(row["gguf_bytes"]),
        )
    )
    payload = {
        "criteria": {
            "applicable_required": 12,
            "precision_required": 1.0,
            "public_accepted_required": public_min,
            "startup_timeout_s": args.startup_timeout,
            "stage_timeout_s": args.stage_timeout,
            "rss_limit_gib": args.rss_limit_gib,
        },
        "selected": qualified[0]["model"] if qualified else None,
        "models": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1 if args.require_qualifier and not qualified else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
