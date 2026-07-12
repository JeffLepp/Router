"""Release gate: the prefilter must label 0 tasks wrongly on every labeled dataset.

Coverage may vary; precision may not. Exit 1 on any wrong label.
Run: python -m scripts.prefilter_audit
"""
import json
import sys
from collections import Counter

try:
    from agent.classify import prefilter_category
except ImportError:
    # ponytail: prefilter is a Phase 2/3 feature; on Phase 1 runtime this audit has nothing to check
    sys.exit("prefilter_category not present in agent.classify (Phase 1 runtime) - audit skipped")

DATASETS = [
    "dataset.json",
    "eval/devset/variants.json",
    "eval/devset/variants2.json",
    "eval/devset/variants3.json",
    "eval/devset/variants4.json",
    "eval/devset/stress.json",
]


def main() -> int:
    failed = False
    for path in DATASETS:
        tasks = json.load(open(path, encoding="utf-8"))
        stats = Counter()
        for t in tasks:
            pred = prefilter_category(t["prompt"])
            if pred is None:
                stats["defer"] += 1
            elif pred == t["category"]:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
                failed = True
                print(f"WRONG {path} {t.get('id','?')}: gold={t['category']} pred={pred}")
        n = len(tasks)
        cov = stats["correct"] + stats["wrong"]
        print(
            f"{path}: n={n} covered={cov} ({cov / n * 100:.0f}%) "
            f"correct={stats['correct']} wrong={stats['wrong']} defer={stats['defer']}"
        )
    print("FAIL" if failed else "PASS", "prefilter audit")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
