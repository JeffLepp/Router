#!/usr/bin/env python3
"""Forecast an AMD submission from paired local benchmark reports.

Token forecasts use the candidate/baseline token ratio, which is more transferable than
the absolute 80-task total. Accuracy forecasts stay anchored to the latest official result;
local accuracy only contributes the candidate's delta because local absolute accuracy has
not tracked the hidden judge reliably.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(paths: list[str]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _totals(reports: list[dict[str, Any]]) -> tuple[int, int, int]:
    tokens = sum(int(report.get("ledger", {}).get("total_tokens", 0) or 0) for report in reports)
    scored = sum(int(report.get("score", {}).get("scored", 0) or 0) for report in reports)
    judged_passes = sum(
        round(float(report.get("score", {}).get("accuracy_judged", 0) or 0) * int(report.get("score", {}).get("scored", 0) or 0))
        for report in reports
    )
    return tokens, judged_passes, scored


def forecast(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    official_accuracy: float,
    official_tokens: int,
    hidden_tasks: int,
) -> dict[str, Any]:
    base_tokens, base_passes, base_scored = _totals(baseline)
    candidate_tokens, candidate_passes, candidate_scored = _totals(candidate)
    if base_tokens <= 0 or base_scored <= 0 or candidate_scored <= 0:
        raise ValueError("benchmark reports must contain nonzero tokens and scored tasks")

    token_ratio = candidate_tokens / base_tokens
    predicted_tokens = round(official_tokens * token_ratio)
    baseline_hidden_passes = round((official_accuracy / 100.0) * hidden_tasks)
    local_delta_rate = candidate_passes / candidate_scored - base_passes / base_scored
    predicted_passes = min(
        hidden_tasks,
        max(0, round(baseline_hidden_passes + local_delta_rate * hidden_tasks)),
    )
    gate_passes = math.ceil(0.80 * hidden_tasks)
    return {
        "baseline": {
            "local_tokens": base_tokens,
            "local_judged_passes": base_passes,
            "local_scored": base_scored,
            "official_accuracy_percent": official_accuracy,
            "official_tokens": official_tokens,
            "official_passes_estimate": baseline_hidden_passes,
        },
        "candidate": {
            "local_tokens": candidate_tokens,
            "local_judged_passes": candidate_passes,
            "local_scored": candidate_scored,
        },
        "forecast": {
            "token_ratio": round(token_ratio, 4),
            "official_tokens": predicted_tokens,
            "official_tokens_range_20pct": [
                round(predicted_tokens * 0.8),
                round(predicted_tokens * 1.2),
            ],
            "official_passes": predicted_passes,
            "official_accuracy_percent": round(100 * predicted_passes / hidden_tasks, 1),
            "accuracy_sensitivity_one_task": [
                round(100 * max(0, predicted_passes - 1) / hidden_tasks, 1),
                round(100 * min(hidden_tasks, predicted_passes + 1) / hidden_tasks, 1),
            ],
            "gate_passes_required": gate_passes,
            "gate_margin_tasks": predicted_passes - gate_passes,
        },
        "warning": (
            "This is a calibrated A/B forecast, not an independent hidden-accuracy estimate. "
            "Update submission_history.csv with every official result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True, help="baseline benchmark.json files")
    parser.add_argument("--candidate", nargs="+", required=True, help="candidate benchmark.json files")
    parser.add_argument("--official-accuracy", type=float, default=89.5)
    parser.add_argument("--official-tokens", type=int, default=14000)
    parser.add_argument("--hidden-tasks", type=int, default=19)
    args = parser.parse_args()
    result = forecast(
        _load(args.baseline),
        _load(args.candidate),
        args.official_accuracy,
        args.official_tokens,
        args.hidden_tasks,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
