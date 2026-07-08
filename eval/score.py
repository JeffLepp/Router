#!/usr/bin/env python3
"""P3 deterministic scorer (plan §7). One function per evaluation_method seen in the
data — NOT a generic judge. Only semantic_similarity + contains_essential_points fall
through to eval/judge.py (LLM-judge, own key). Everything else is exact/verifiable here.

`score_one(answer, expected, method, task=None) -> bool` is the low-level entry (bench_local
imports it). `score_task(task, answer)` is the devset entry. `answer` may be the native gold
type (dict/None/str), a JSON string, or a plain gate/agent string — all normalized.

Self-check:  python -m eval.score   (gold-back == 100%, corrupted == 0%)
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Runnable tests per code task id (the dataset embeds tests in prose, not as a field).
# Same bank scripts/acceptance_p2.py uses; ids absent here fall back to code-string equality.
CODE_TESTS: dict[str, list[str]] = {
    "debug_001": ["assert square(4) == 16"],
    "debug_004": ["assert largest(8, 3) == 8", "assert largest(2, 9) == 9"],
    "debug_007": [
        "def sum_list(lst):\n    total = 0\n    for num in lst:\n        total += num\n    return total",
        "assert average([]) == 0",
        "assert average([2, 4, 6]) == 4",
    ],
    "gen_001": ["assert reverse_list([1, 2, 3]) == [3, 2, 1]"],
    "gen_004": ["assert count_words('Hi hi, there!') == {'hi': 2, 'there': 1}"],
    "gen_006": [
        "assert fibonacci(5) == [0, 1, 1, 2, 3]",
        "try:\n    fibonacci(-1)\n    raise AssertionError('missing ValueError')\nexcept ValueError:\n    pass",
    ],
    "gen_007": ["assert merge_sorted_lists([1, 3], [2, 4]) == [1, 2, 3, 4]"],
    "gen_009": ["assert safe_divide(8, 2) == 4", "assert safe_divide(8, 0) is None"],
}

_JUDGE_METHODS = {"semantic_similarity", "contains_essential_points"}


def _as_obj(x: Any) -> Any:
    """A JSON string becomes its object; anything else passes through."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def _norm_text(x: Any) -> str:
    return " ".join(str(x).strip().lower().split())


def _is_unanswerable(answer: Any) -> bool:
    obj = _as_obj(answer)
    if obj is None:
        return True
    if isinstance(obj, str):
        return obj.strip().lower() in {"", "null", "none", "unanswerable"}
    if isinstance(obj, dict):
        return obj.get("answer") is None and obj.get("valid") is False
    return False


def _number(text: Any) -> Decimal:
    s = str(text).strip()
    if "/" in s and not s.startswith("["):
        f = Fraction(s)
        return Decimal(f.numerator) / Decimal(f.denominator)
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        raise InvalidOperation(f"no number in {s!r}")
    raw = m.group(0)
    if "/" in raw:
        f = Fraction(raw)
        return Decimal(f.numerator) / Decimal(f.denominator)
    return Decimal(raw)


def _number_list(value: Any) -> list[Decimal]:
    data = _as_obj(value)
    if isinstance(data, str):
        data = json.loads(data)
    return sorted(Decimal(str(v)) for v in data)


def _sentiment_label(answer: Any) -> str | None:
    obj = _as_obj(answer)
    if isinstance(obj, dict):
        return obj.get("sentiment")
    low = str(answer).lower()
    for label in ("positive", "negative", "neutral", "mixed"):
        if label in low:
            return label
    return None


def _entities(answer: Any) -> set[tuple[str, str]]:
    obj = _as_obj(answer)
    if isinstance(obj, str):  # "text|TYPE" lines
        out = set()
        for line in obj.splitlines():
            if "|" in line:
                text, _, typ = line.partition("|")
                out.add((text.strip(), typ.strip().upper()))
        return out
    ents = obj["entities"] if isinstance(obj, dict) else obj
    return {(e["text"].strip(), e["type"].strip().upper()) for e in ents}


# ---- per-method scorers -----------------------------------------------------

def _exact(answer: Any, expected: Any) -> bool:
    if expected is None:
        return _is_unanswerable(answer)
    return _norm_text(answer) == _norm_text(expected)


def _numeric(answer: Any, expected: Any) -> bool:
    if expected is None:
        return _is_unanswerable(answer)
    try:
        return abs(_number(answer) - _number(expected)) <= Decimal("0.0001")
    except InvalidOperation:
        return False


def _numeric_list(answer: Any, expected: Any) -> bool:
    try:
        return _number_list(answer) == _number_list(expected)
    except (json.JSONDecodeError, InvalidOperation, ValueError, TypeError):
        return False


def _classification(answer: Any, expected: Any) -> bool:
    exp = _as_obj(expected)
    want = exp.get("sentiment") if isinstance(exp, dict) else exp
    return _sentiment_label(answer) == want


def _aspect(answer: Any, expected: Any) -> bool:
    if not _classification(answer, expected):
        return False
    obj, exp = _as_obj(answer), _as_obj(expected)
    exp_aspects = exp.get("aspects") if isinstance(exp, dict) else None
    if not exp_aspects:
        return True
    got = obj.get("aspects") if isinstance(obj, dict) else None
    if not isinstance(got, dict):
        return False
    return {k: _norm_text(v) for k, v in got.items()} == {
        k: _norm_text(v) for k, v in exp_aspects.items()
    }


def _entity_match(answer: Any, expected: Any) -> bool:
    try:
        return _entities(answer) == _entities(expected)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def _logic(answer: Any, expected: Any) -> bool:
    obj, exp = _as_obj(answer), _as_obj(expected)
    if not isinstance(obj, dict) or not isinstance(exp, dict):
        return False
    if obj.get("valid") is not exp.get("valid"):
        return False
    return _norm_text(obj.get("answer")) == _norm_text(exp.get("answer"))


def _code(answer: Any, expected: Any, task: dict | None) -> bool:
    # Prefer running the task's real tests; else fall back to normalized code equality.
    tid = (task or {}).get("id")
    if tid == "debug_009" or (isinstance(expected, str) and "test should expect" in str(answer).lower()):
        low = str(answer).lower()
        return "correct" in low and "6" in low
    if tid in CODE_TESTS:
        from agent.verify.code_v import run_inline_examples  # lazy: subprocess sandbox

        return run_inline_examples(str(answer), CODE_TESTS[tid], timeout=1.0)
    # ponytail: no test bank for this id -> string-equal the reference. Real launch scores
    # code via the task's own unit_tests; add ids to CODE_TESTS as devset grows.
    return _norm_text(answer) == _norm_text(expected)


def score_one(answer: Any, expected: Any, method: str, task: dict | None = None) -> bool:
    if method in _JUDGE_METHODS:
        from eval.judge import judge_summary  # lazy: only summarization needs it

        return judge_summary((task or {}).get("prompt", ""), answer, expected, method)
    if method == "unanswerable":
        return _is_unanswerable(answer)
    if method in {"numeric_exact_match", "numeric_tolerance"}:
        return _numeric(answer, expected)
    if method == "unordered_numeric_match":
        return _numeric_list(answer, expected)
    if method == "classification":
        return _classification(answer, expected)
    if method == "aspect_based_classification":
        return _aspect(answer, expected)
    if method == "entity_match":
        return _entity_match(answer, expected)
    if method in {"boolean_logic", "constraint_logic"}:
        return _logic(answer, expected)
    if method in {"unit_tests", "review_comment", "query_validation", "query_match", "schema_match"}:
        return _code(answer, expected, task)
    if method == "exact_match":
        return _exact(answer, expected)
    raise ValueError(f"unknown evaluation_method: {method}")


def score_task(task: dict, answer: Any) -> bool:
    return score_one(answer, task["expected_answer"], task["evaluation_method"], task)


def _demo() -> None:
    tasks = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    # 1) feeding each gold expected_answer back must score 100% (proves every scorer accepts gold)
    gold_pass = sum(score_task(t, t["expected_answer"]) for t in tasks)
    print(f"gold-back: {gold_pass}/{len(tasks)}")
    assert gold_pass == len(tasks), "a method scorer rejected its own gold"
    # 2) a corrupted answer must score 0 for every task
    corrupt_pass = sum(score_task(t, _corrupt(t["expected_answer"])) for t in tasks)
    print(f"corrupted: {corrupt_pass}/{len(tasks)} (want 0)")
    assert corrupt_pass == 0, "a corrupted answer was accepted"
    print("PASS eval.score self-check")


def _corrupt(expected: Any) -> Any:
    if expected is None:
        return "The answer is definitely 42."
    if isinstance(expected, dict):
        d = dict(expected)
        if "sentiment" in d:
            d["sentiment"] = "neutral" if d["sentiment"] != "neutral" else "positive"
        if "answer" in d:
            d["answer"] = "WRONG"
            d["valid"] = not d.get("valid", True)
        if "entities" in d:
            d["entities"] = [{"text": "Nobody", "type": "PERSON"}]
        if "aspects" in d:
            d["aspects"] = {k: "neutral" for k in d["aspects"]}
        if "summary" in d:
            d["summary"] = "Completely unrelated banana content." if isinstance(d["summary"], str) else ["banana"]
        return d
    return "totally-wrong-zzz-9999"


if __name__ == "__main__":
    _demo()
