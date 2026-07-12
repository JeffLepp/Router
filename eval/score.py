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
    # Phase 2 adversarial variant sets (eval/devset/variants2.json / variants3.json).
    "v2_debug_001": ["assert sum_to(5) == 15", "assert sum_to(1) == 1"],
    "v2_debug_002": ["assert count_vowels('Aeon') == 3", "assert count_vowels('SKY') == 0"],
    "v2_debug_004": ["assert append_item('a') == ['a']", "assert append_item('b') == ['b']"],
    "v2_debug_007": ["assert median([3, 1, 2]) == 2", "assert median([9, 4, 7, 1, 5]) == 5"],
    "v2_gen_001": ["assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]", "assert unique_sorted([]) == []"],
    "v2_gen_002": ["assert celsius_to_kelvin(0) == 273.15", "assert abs(celsius_to_kelvin(25) - 298.15) < 1e-9"],
    "v2_gen_004": ["assert char_frequency('Aa b') == {'a': 2, 'b': 1}"],
    "v2_gen_005": ["assert second_largest([4, 1, 4, 3]) == 3", "assert second_largest([7, 7]) is None"],
    "v2_gen_007": ["assert rle('aaabb') == 'a3b2'", "assert rle('') == ''", "assert rle('abc') == 'a1b1c1'"],
    "v3_debug_001": ["assert to_fahrenheit(0) == 32", "assert to_fahrenheit(100) == 212"],
    "v3_debug_002": ["assert find_negatives([1, -2, 3, -4]) == [-2, -4]", "assert find_negatives([]) == []"],
    "v3_debug_004": ["assert clamp(5, 1, 10) == 5", "assert clamp(-3, 1, 10) == 1", "assert clamp(99, 1, 10) == 10"],
    "v3_debug_007": ["assert longest_run([1, 1, 0, 1, 1, 1]) == 3", "assert longest_run([0, 0]) == 0", "assert longest_run([]) == 0"],
    "v3_gen_001": ["assert flatten([[1, 2], [3]]) == [1, 2, 3]", "assert flatten([]) == []"],
    "v3_gen_002": ["assert is_leap_year(2024)", "assert not is_leap_year(1900)", "assert is_leap_year(2000)", "assert not is_leap_year(2023)"],
    "v3_gen_004": ["assert word_lengths('hi there') == [2, 5]", "assert word_lengths('') == []"],
    "v3_gen_005": ["assert sum_of_squares(3) == 14", "assert sum_of_squares(1) == 1"],
    "v3_gen_007": [
        "assert most_common(['a', 'b', 'a']) == 'a'",
        "assert most_common([1, 2, 2, 1]) == 1",
        "assert most_common([]) is None",
    ],
    # Phase 3 gap-focused adversarial set (eval/devset/variants4.json).
    "v4_debug_001": ["assert up_to(5) == [1, 2, 3, 4, 5]", "assert up_to(1) == [1]"],
    "v4_debug_002": ["assert is_positive(5) == True", "assert is_positive(0) == False", "assert is_positive(-3) == False"],
    "v4_debug_003": ["assert add_item('a') == ['a']", "assert add_item('b') == ['b']"],
    "v4_debug_005": ["assert avg([2, 4, 6]) == 4", "assert avg([]) == 0"],
    "v4_debug_006": ["assert abs_diff(3, 8) == 5", "assert abs_diff(8, 3) == 5", "assert abs_diff(-2, -7) == 5"],
    "v4_gen_001": [
        "assert running_max([3, 1, 4, 1, 5, 9, 2]) == [3, 3, 4, 4, 5, 9, 9]",
        "assert running_max([]) == []",
        "assert running_max([-5, -2, -9]) == [-5, -2, -2]",
    ],
    "v4_gen_004": ["assert count_palindrome_words('noon is a racecar word') == 3", "assert count_palindrome_words('hello world') == 0"],
    "v4_gen_006": [
        "assert factorial(5) == 120",
        "assert factorial(0) == 1",
        "try:\n    factorial(-1)\n    raise AssertionError('missing ValueError')\nexcept ValueError:\n    pass",
    ],
    "v4_gen_007": [
        "assert interleave([1, 2, 3], [4, 5]) == [1, 4, 2, 5, 3]",
        "assert interleave([], [1, 2]) == [1, 2]",
        "assert interleave([1], []) == [1]",
    ],
    "v4_gen_009": ["assert safe_index([1, 2, 3], 1) == 2", "assert safe_index([1, 2, 3], 5) is None", "assert safe_index([], 0) is None"],
}

# JS tasks: real node execution (Node is on the scoring host). assert.* throws -> nonzero exit.
CODE_TESTS_JS: dict[str, list[str]] = {
    "debug_005": [
        "assert.deepStrictEqual(filterEven([1, 2, 3, 4, 5, 6]), [2, 4, 6]);",
        "assert.deepStrictEqual(filterEven([1, 3, 5]), []);",
    ],
    "gen_002": [
        "assert.strictEqual(sumArray([1, 2, 3, 4]), 10);",
        "assert.strictEqual(sumArray([]), 0);",
    ],
    "v2_debug_003": ["assert.strictEqual(total([1, 2, 3]), 6);", "assert.strictEqual(total([]), 0);"],
    "v2_debug_005": [
        "const orig = [3, 1, 2];",
        "assert.deepStrictEqual(sortedCopy(orig), [1, 2, 3]);",
        "assert.deepStrictEqual(orig, [3, 1, 2]);",
    ],
    "v2_gen_006": ["assert.strictEqual(titleCase('hello world'), 'Hello World');"],
    "v3_debug_003": ["assert.strictEqual(last([5, 6, 7]), 7);"],
    "v3_debug_005": ["assert.strictEqual(reverseWords('a b c'), 'c b a');"],
    "v3_gen_006": [
        "assert.deepStrictEqual(range(3), [0, 1, 2]);",
        "assert.deepStrictEqual(range(0), []);",
    ],
    "v4_debug_004": ["assert.strictEqual(sumFirstN(5), 15);", "assert.strictEqual(sumFirstN(1), 1);"],
    "v4_debug_007": [
        "assert.strictEqual(inRange(5), true);",
        "assert.strictEqual(inRange(15), false);",
        "assert.strictEqual(inRange(-3), false);",
    ],
    "v4_gen_002": [
        "assert.strictEqual(countUnique(['Cat', 'cat', 'Dog']), 2);",
        "assert.strictEqual(countUnique([]), 0);",
    ],
}

# Java/C: no toolchain on the scoring host, so they can't run. In judged mode the eval-only
# LLM judge (eval/judge.py, own key, never the competition path) rescores them; strict mode
# leaves them on string-equality (so they read as failures until judged).
JUDGE_CODE_IDS = {
    "debug_010", "gen_005", "gen_010",
    # Phase 2 variant-set Java/C/C++ tasks (no toolchain on the scoring host).
    "v2_debug_006", "v2_debug_010", "v2_gen_009", "v2_gen_010",
    "v3_debug_006", "v3_debug_010", "v3_gen_009", "v3_gen_010",
    # Phase 3 gap-focused set: SQL (phrasing varies) plus Java/C (no toolchain).
    "v4_debug_008", "v4_debug_009", "v4_debug_010",
    "v4_gen_003", "v4_gen_005", "v4_gen_008", "v4_gen_010",
}

_JUDGE_METHODS = {"semantic_similarity", "contains_essential_points"}


def _strip_code_fence(text: Any) -> str:
    s = str(text).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


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
    # Strict: gold-schema JSON only ({"entities":[...]} or a bare entity list). The old
    # "text|TYPE" line parsing accepted a format only WE emitted — the hidden grader
    # never saw it, so accepting it locally inflated practice scores (Phase 0).
    obj = _as_obj(answer)
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


def _code(answer: Any, expected: Any, task: dict | None, judge: bool = False) -> bool:
    # Prefer running the task's real tests; else judge (if allowed); else code equality.
    tid = (task or {}).get("id")
    if tid == "debug_009" or (isinstance(expected, str) and "test should expect" in str(answer).lower()):
        low = str(answer).lower()
        return "correct" in low and "6" in low
    code = _strip_code_fence(answer)
    if tid in CODE_TESTS_JS:
        from agent.verify.code_v import run_node_examples  # lazy: node subprocess

        return run_node_examples(code, CODE_TESTS_JS[tid], timeout=2.0)
    if tid in CODE_TESTS:
        from agent.verify.code_v import run_inline_examples  # lazy: subprocess sandbox

        return run_inline_examples(code, CODE_TESTS[tid], timeout=1.0)
    if judge and tid in JUDGE_CODE_IDS:
        from eval.judge import judge_code  # lazy, eval-only, own key

        return judge_code((task or {}).get("prompt", ""), code, expected)
    # ponytail: no test bank for this id -> string-equal the reference. Real launch scores
    # code via the task's own unit_tests; add ids to CODE_TESTS as devset grows.
    return _norm_text(code) == _norm_text(expected)


def score_one(answer: Any, expected: Any, method: str, task: dict | None = None, judge: bool = False) -> bool:
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
        return _code(answer, expected, task, judge=judge)
    if method == "exact_match":
        return _exact(answer, expected)
    raise ValueError(f"unknown evaluation_method: {method}")


def score_task(task: dict, answer: Any, judge: bool = False) -> bool:
    return score_one(answer, task["expected_answer"], task["evaluation_method"], task, judge=judge)


_GOLD_BACK_FILES = (
    "dataset.json",
    "eval/devset/variants2.json",
    "eval/devset/variants3.json",
    "eval/devset/variants4.json",
)


def _demo() -> None:
    import os

    os.environ.setdefault("EVAL_JUDGE_OFFLINE", "1")  # self-check must never spend tokens
    for rel in _GOLD_BACK_FILES:
        tasks = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        # 1) feeding each gold expected_answer back must score 100% (proves every scorer
        #    accepts gold, and actually RUNS the code golds against their test banks)
        gold_fail = [t["id"] for t in tasks if not score_task(t, t["expected_answer"])]
        print(f"{rel}: gold-back {len(tasks) - len(gold_fail)}/{len(tasks)}")
        assert not gold_fail, f"{rel}: scorer rejected its own gold: {gold_fail}"
        # 2) a corrupted answer must score 0 for every task
        corrupt_pass = [t["id"] for t in tasks if score_task(t, _corrupt(t["expected_answer"]))]
        print(f"{rel}: corrupted {len(corrupt_pass)}/{len(tasks)} (want 0)")
        assert not corrupt_pass, f"{rel}: corrupted answer accepted: {corrupt_pass}"
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
