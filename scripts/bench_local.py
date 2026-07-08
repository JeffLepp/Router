#!/usr/bin/env python3
"""P4 / DG-6 local model bakeoff.

For each candidate GGUF: boot llama-server, measure cold-start + tok/s, then run
dataset.json through the REAL pipeline (gate.solve -> local_gate.try_local with
every category enabled) and report the win metric: free answers the local tier
accepts, and how many of those are correct (accepted precision must be 100%).

Usage:
  python -m scripts.bench_local --models qwen=/models/qwen.gguf,llama=/models/llama.gguf
  # defaults to CPU-only, 512 ctx, two threads, and the baked $MODEL_GGUF.

Scoring reuses eval/score.py when present (P3); otherwise a minimal comparator.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.gate import solve as gate_solve  # noqa: E402
from agent.local_gate import try_local  # noqa: E402
from agent.local_llm import make_local_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _score(answer: str | None, expected, method: str) -> bool:
    """Prefer P3's scorer; fall back to a boring comparator."""
    try:
        from eval.score import score_one  # type: ignore

        return bool(score_one(answer, expected, method))
    except Exception:
        pass
    if answer is None:
        return expected is None
    a = str(answer).strip().lower()
    if expected is None:
        return a in {"", "null", "unanswerable", '{"answer":null,"valid":false}'}
    if isinstance(expected, (int, float)):
        try:
            return abs(float(a) - float(expected)) < 1e-6
        except ValueError:
            return False
    return a == str(expected).strip().lower()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_health(port: int, timeout: float = 60.0) -> float:
    """Return cold-start seconds, or -1 on failure."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return time.monotonic() - start
        except Exception:
            time.sleep(0.5)
    return -1.0


def _tok_per_s(port: int, model: str) -> float:
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "Count from 1 to 40."}],
         "max_tokens": 128, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    body = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dt = time.monotonic() - t0
    toks = body.get("usage", {}).get("completion_tokens", 0)
    return toks / dt if dt > 0 else 0.0


async def _bench_model(
    name: str,
    gguf: str,
    tasks: list[dict],
    *,
    ctx_size: int,
    threads: int,
    startup_timeout: float,
    n_gpu_layers: int,
) -> dict:
    port = _free_port()
    cmd = [
        "llama-server",
        "--model", gguf,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
    ]
    if n_gpu_layers:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return {"model": name, "error": "llama-server not found on PATH"}
    try:
        cold = _wait_health(port, timeout=startup_timeout)
        if cold < 0:
            return {"model": name, "error": "llama-server never healthy"}
        tps = _tok_per_s(port, "local")
        llama_cfg = {"base_url": f"http://127.0.0.1:{port}/v1", "model": "local"}
        local_cfg = {
            "enabled": True, "self_consistency_k": 2, "temp": 0, "max_tokens": 256,
            "latency_cap_s": 30,
            "categories": {c: True for c in _ALL_CATS},
        }
        client = await make_local_client(llama_cfg, local_cfg)
        accepted = correct = local_eligible = 0
        for t in tasks:
            cat = t["category"]
            if gate_solve(cat, t["prompt"]) is not None:
                continue  # gate proves it for free; not a local-tier win
            local_eligible += 1
            ans = await try_local(cat, t["prompt"], config=local_cfg,
                                  llama_config=llama_cfg, client=client)
            if ans is None:
                continue
            accepted += 1
            if _score(ans, t.get("expected_answer"), t.get("evaluation_method", "")):
                correct += 1
        precision = 100.0 * correct / accepted if accepted else 100.0
        return {"model": name, "cold_s": round(cold, 1), "tok_s": round(tps, 1),
                "eligible": local_eligible, "accepted": accepted, "correct": correct,
                "precision": round(precision, 1)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


_ALL_CATS = ["actual_qa", "math_reasoning", "sentiment_analysis", "summarization",
             "named_entity_recognition", "code_debugging", "logic_puzzles",
             "code_generation"]


def _parse_models(arg: str | None) -> list[tuple[str, str]]:
    if not arg:
        return [("baked", os.environ.get("MODEL_GGUF", "/models/model.gguf"))]
    out = []
    for pair in arg.split(","):
        name, _, path = pair.partition("=")
        out.append((name.strip(), path.strip()))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="name=path,name=path (default: $MODEL_GGUF)")
    ap.add_argument("--dataset", default=str(ROOT / "dataset.json"))
    ap.add_argument("--ctx-size", type=int, default=int(os.environ.get("LLAMA_CTX_SIZE", "512")))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LLAMA_THREADS", "2")))
    ap.add_argument(
        "--startup-timeout",
        type=float,
        default=float(os.environ.get("LLAMA_STARTUP_WAIT_S", "60")),
        help="seconds to wait for llama-server /health",
    )
    ap.add_argument(
        "--n-gpu-layers",
        type=int,
        default=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        help="optional GPU offload; default 0 keeps the bench CPU-only",
    )
    args = ap.parse_args()

    tasks = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = []
    for name, gguf in _parse_models(args.models):
        if not Path(gguf).exists():
            rows.append({"model": name, "error": f"missing {gguf}"})
            continue
        print(f"== benchmarking {name} ({gguf}) ==", file=sys.stderr)
        rows.append(await _bench_model(
            name,
            gguf,
            tasks,
            ctx_size=args.ctx_size,
            threads=args.threads,
            startup_timeout=args.startup_timeout,
            n_gpu_layers=args.n_gpu_layers,
        ))

    print("\n| model | cold s | tok/s | eligible | accepted | correct | precision % |")
    print("|-------|-------:|------:|---------:|---------:|--------:|------------:|")
    ok = True
    for r in rows:
        if "error" in r:
            print(f"| {r['model']} | - | - | - | - | - | {r['error']} |")
            continue
        print(f"| {r['model']} | {r['cold_s']} | {r['tok_s']} | {r['eligible']} | "
              f"{r['accepted']} | {r['correct']} | {r['precision']} |")
        if r["accepted"] and r["precision"] < 100.0:
            ok = False
    if not ok:
        print("\nWARN: a model accepted a WRONG answer (precision < 100%) — tighten "
              "its accept gate before baking.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
