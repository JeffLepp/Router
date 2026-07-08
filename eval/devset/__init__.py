"""Devset loader (plan §7): the 80 real dataset.json tasks (holdout=false) are the primary
devset; variants.json holds authored adversarial variants (holdout=true). Variants are NEVER
used as few-shot at runtime — this loader is eval-only.

Self-check:  python -m eval.devset
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent


def load_devset(include_holdout: bool = True) -> list[dict]:
    real = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    for t in real:
        t.setdefault("holdout", False)
    tasks = list(real)
    if include_holdout:
        tasks += json.loads((_HERE / "variants.json").read_text(encoding="utf-8"))
    return tasks


def _demo() -> None:
    from collections import Counter

    tasks = load_devset()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate ids in devset"
    cats = Counter(t["category"] for t in tasks)
    assert len(cats) == 8 and len(set(cats.values())) == 1, f"category imbalance: {cats}"
    holdout = sum(t["holdout"] for t in tasks)
    print(f"devset tasks={len(tasks)} categories={len(cats)} per_cat={cats.most_common(1)[0][1]} holdout={holdout}")
    assert holdout > 0 and all("holdout" in t for t in tasks)
    print("PASS eval.devset self-check")


if __name__ == "__main__":
    _demo()
