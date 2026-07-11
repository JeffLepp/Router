#!/usr/bin/env python3
"""Make one tiny completion request per model without printing credentials."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from eval.probe_models import _ask
from scripts.live_benchmark import _load_env_file


ROOT = Path(__file__).resolve().parents[1]


async def _run(models: list[str]) -> int:
    failures = 0
    for model in models:
        try:
            answer = await _ask(model, "Reply with exactly OK.", max_tokens=8)
            status = "OK" if answer.strip().upper() == "OK" else "UNEXPECTED"
            print(f"{model}: {status} {answer[:40]!r}")
            failures += int(status != "OK")
        except Exception as exc:
            print(f"{model}: ERROR {type(exc).__name__}: {str(exc)[:180]}")
            failures += 1
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--env-file", default=".env.local")
    args = parser.parse_args()

    env = os.environ.copy()
    _load_env_file(env, ROOT / args.env_file)
    os.environ.update(env)
    missing = [
        name
        for name in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit("missing required env: " + ", ".join(missing))
    return asyncio.run(_run(args.models))


if __name__ == "__main__":
    raise SystemExit(main())
