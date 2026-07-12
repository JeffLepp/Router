"""Release gate: the math_full gate profile must answer 0 math tasks wrongly.

Coverage may vary; precision may not. Gold `null`/None answers match the solver's
UNANSWERABLE sentinel. Exit 1 on any wrong answer.
Run: python -m scripts.math_audit
"""
import json
import sys
from collections import Counter

from agent.gate import solve
from agent.solvers.common import UNANSWERABLE

DATASETS = [
    "dataset.json",
    "eval/devset/variants.json",
    "eval/devset/variants2.json",
    "eval/devset/variants3.json",
]


def _matches(answer: str, gold) -> bool:
    if gold is None:
        return answer == UNANSWERABLE
    return answer.strip() == str(gold).strip()


def main() -> int:
    failed = False
    for path in DATASETS:
        tasks = [
            t for t in json.load(open(path, encoding="utf-8"))
            if t["category"] == "math_reasoning"
        ]
        stats = Counter()
        for t in tasks:
            answer = solve("math_reasoning", t["prompt"], profile="math_full")
            if answer is None:
                stats["defer"] += 1
            elif _matches(answer, t.get("expected_answer")):
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
                failed = True
                print(
                    f"WRONG {path} {t.get('id','?')}: "
                    f"gold={t.get('expected_answer')!r} pred={answer!r}"
                )
        n = len(tasks)
        cov = stats["correct"] + stats["wrong"]
        print(
            f"{path}: n={n} covered={cov} ({cov / n * 100:.0f}%) "
            f"correct={stats['correct']} wrong={stats['wrong']} defer={stats['defer']}"
        )
    print("FAIL" if failed else "PASS", "math audit")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
