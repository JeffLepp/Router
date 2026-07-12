"""Release gate: per-task answer diff between a candidate run and a frozen baseline run.

Aggregate local accuracy is saturated and has failed to predict the hidden set
three times; the only trustworthy local signal is "which tasks changed answers
and did any go pass->fail". Any flip must be individually justified before submit.

Run: python -m scripts.flip_diff <baseline_out_dir> <candidate_out_dir>
Exit 1 on any pass->fail flip; prints all answer changes either way.
"""
import csv
import json
import sys
from pathlib import Path


def load_run(out_dir: str):
    d = Path(out_dir)
    answers = {r["task_id"]: r.get("answer") for r in json.loads((d / "results.json").read_text(encoding="utf-8"))}
    failed = set()
    fails_csv = d / "failures.csv"
    if fails_csv.exists():
        with fails_csv.open(encoding="utf-8", newline="") as f:
            failed = {row["task_id"] for row in csv.DictReader(f)}
    return answers, failed


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_ans, base_fail = load_run(sys.argv[1])
    cand_ans, cand_fail = load_run(sys.argv[2])

    common = sorted(set(base_ans) & set(cand_ans))
    missing = sorted(set(base_ans) ^ set(cand_ans))
    changed = [t for t in common if base_ans[t] != cand_ans[t]]
    pass_to_fail = sorted((set(cand_fail) - set(base_fail)) & set(common))
    fail_to_pass = sorted((set(base_fail) - set(cand_fail)) & set(common))

    for t in changed:
        status = "PASS->FAIL" if t in pass_to_fail else ("FAIL->PASS" if t in fail_to_pass else "same verdict")
        print(f"CHANGED {t} [{status}]")
        print(f"  base: {str(base_ans[t])[:200]}")
        print(f"  cand: {str(cand_ans[t])[:200]}")
    for t in missing:
        print(f"MISSING {t} (present in only one run)")

    print(f"\n{len(common)} common tasks; {len(changed)} answers changed; "
          f"{len(pass_to_fail)} pass->fail; {len(fail_to_pass)} fail->pass; {len(missing)} missing")
    if pass_to_fail or missing:
        print("GATE: FAIL")
        return 1
    print("GATE: PASS (justify each remaining change before submit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
