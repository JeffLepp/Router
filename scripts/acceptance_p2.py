from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import solve as gate_solve
from agent.verify.code_v import run_inline_examples, run_python
from scripts.acceptance_p1 import (
    fake_tasks,
    free_port,
    metrics,
    run_agent,
    start_mock,
    validate_results,
    write_config,
)


def load_dataset() -> list[dict[str, Any]]:
    return json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))


def test_gate_precision_recall() -> None:
    rows = load_dataset()
    totals: dict[str, int] = defaultdict(int)
    answered: dict[str, int] = defaultdict(int)
    wrong: list[tuple[str, str, str]] = []

    for item in rows:
        category = item["category"]
        totals[category] += 1
        answer = gate_solve(category, item["prompt"])
        if answer is None:
            continue
        answered[category] += 1
        if not evaluate_item(item, answer):
            wrong.append((item["id"], category, answer))

    assert not wrong, "wrong proven answers: " + repr(wrong[:5])
    total_answered = sum(answered.values())
    print(f"GATE_PRECISION 100.00% answered={total_answered} wrong=0")
    for category in sorted(totals):
        recall = answered[category] / totals[category]
        print(f"GATE_RECALL {category} {answered[category]}/{totals[category]} {recall:.1%}")
    _assert_holdout_precision()


def _assert_holdout_precision() -> None:
    """Precision must hold on the ADVERSARIAL holdout too, or the 100% above is a comforting lie.
    Scored with the canonical eval.score, same as eval/gate_report.py --holdout."""
    from eval.devset import load_devset
    from eval.score import score_task

    holdout = [task for task in load_devset(include_holdout=True) if task.get("holdout")]
    wrong = []
    for task in holdout:
        answer = gate_solve(task["category"], task["prompt"])
        if answer is not None and not score_task(task, answer):
            wrong.append((task["id"], task["category"], answer))
    assert not wrong, "wrong proven answers on holdout: " + repr(wrong[:5])
    print(f"GATE_PRECISION_HOLDOUT 100.00% wrong=0 n_holdout={len(holdout)}")


def evaluate_item(item: dict[str, Any], answer: str) -> bool:
    method = item["evaluation_method"]
    expected = item["expected_answer"]
    if method == "unanswerable":
        return _is_unanswerable(answer)
    if method in {"numeric_exact_match", "numeric_tolerance"}:
        return _numeric_equal(answer, expected)
    if method == "unordered_numeric_match":
        return _number_list(answer) == _number_list(expected)
    if method in {"classification", "aspect_based_classification"}:
        return _sentiment_label(answer) == expected.get("sentiment")
    if method == "entity_match":
        return _entities(answer) == [
            (entity["text"], entity["type"]) for entity in expected["entities"]
        ]
    if method in {"boolean_logic", "constraint_logic"}:
        return _logic_match(answer, expected)
    if method in {"unit_tests", "review_comment", "query_validation", "query_match", "schema_match"}:
        return _code_match(item["id"], answer)
    return False


def _is_unanswerable(answer: str) -> bool:
    text = answer.strip()
    if text == "null":
        return True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return data is None or (
        isinstance(data, dict) and data.get("answer") is None and data.get("valid") is False
    )


def _numeric_equal(answer: str, expected: Any) -> bool:
    if expected is None:
        return _is_unanswerable(answer)
    return abs(_number_value(answer) - _number_value(str(expected))) <= Decimal("0.0001")


def _number_value(text: str) -> Decimal:
    text = text.strip()
    if "/" in text and not text.startswith("["):
        fraction = Fraction(text)
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if not match:
        raise ValueError(f"no number in {text!r}")
    raw = match.group(0)
    if "/" in raw:
        fraction = Fraction(raw)
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)
    return Decimal(raw)


def _number_list(value: Any) -> list[Decimal]:
    data = json.loads(value) if isinstance(value, str) else value
    if isinstance(data, str):
        data = json.loads(data)
    return sorted(Decimal(str(item)) for item in data)


def _sentiment_label(answer: str) -> str | None:
    try:
        data = json.loads(answer)
        if isinstance(data, dict):
            return data.get("sentiment")
    except json.JSONDecodeError:
        pass
    lowered = answer.lower()
    for label in ("positive", "negative", "neutral", "mixed"):
        if label in lowered:
            return label
    return None


def _entities(answer: str) -> list[tuple[str, str]]:
    data = json.loads(answer)
    return [(entity["text"], entity["type"]) for entity in data["entities"]]


def _logic_match(answer: str, expected: dict[str, Any]) -> bool:
    data = json.loads(answer)
    if data.get("valid") is not expected.get("valid"):
        return False
    return _normalize(data.get("answer")) == _normalize(expected.get("answer"))


def _normalize(value: Any) -> str:
    if value is None:
        return "<null>"
    return " ".join(str(value).lower().replace(".", "").split())


def _code_match(task_id: str, answer: str) -> bool:
    tests: dict[str, list[str]] = {
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
    if task_id == "debug_009":
        lowered = answer.lower()
        return "function is correct" in lowered and "expect 6" in lowered
    if task_id not in tests:
        return False
    return run_inline_examples(answer, tests[task_id], timeout=1.0)


def test_trap_defer_cases() -> None:
    traps = [
        ("sentiment_analysis", "The laptop is fast, but its battery life is terrible."),
        ("sentiment_analysis", "Wonderful support-I only had to wait three hours to get help."),
        ("sentiment_analysis", "The experience was not terrible."),
        ("named_entity_recognition", "Identify entities: 'The word Amazon can refer to a rainforest or a company.'"),
        ("logic_puzzles", "A is before B. Who is in the middle?"),
        ("math_reasoning", "How many apples did Joan keep after a complicated trade?"),
        ("code_debugging", "Debug Python:\n```python\nimport socket\n```"),
        ("code_debugging", "Debug Python:\n```python\nopen('x.txt', 'w').write('no')\n```"),
        ("code_debugging", "Debug Python:\n```python\nwhile True:\n    pass\n```"),
        ("code_generation", "Create a JavaScript function `sumArray` that accepts an array of numbers."),
    ]
    for category, prompt in traps:
        assert gate_solve(category, prompt) is None, (category, prompt)
    print(f"PASS trap defer cases ({len(traps)})")


def test_pipeline_remote_drop() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        server = start_mock(port)
        try:
            config = tmp / "config.yaml"
            write_config(config, token_budget=0, mandatory_remote=1)
            ledger = tmp / "ledger.json"
            tasks = fake_tasks(150)
            run_agent(tmp, tasks, config, f"http://127.0.0.1:{port}", ledger)
            rows = validate_results(tmp / "results.json")
            requests = metrics(port)["requests"]
            assert len(rows) == len(tasks)
            assert 0 < requests < len(tasks), requests
            print(
                f"PIPELINE_GATE tasks={len(tasks)} gate_proven={len(tasks) - requests} "
                f"remote_requests={requests}"
            )
        finally:
            server.kill()


def test_code_sandbox_guards() -> None:
    ok, output = run_python("while True:\n    pass\n", timeout=0.2)
    assert not ok and output == "timeout"
    assert gate_solve("code_debugging", "Debug Python:\n```python\nwhile True:\n    pass\n```") is None
    assert gate_solve("code_debugging", "Debug Python:\n```python\nimport socket\n```") is None
    assert gate_solve("code_debugging", "Debug Python:\n```python\nopen('x.txt', 'w').write('no')\n```") is None
    print("PASS code sandbox guards")


def main() -> None:
    test_gate_precision_recall()
    test_trap_defer_cases()
    test_pipeline_remote_drop()
    test_code_sandbox_guards()


if __name__ == "__main__":
    main()
