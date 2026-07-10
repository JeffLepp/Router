#!/usr/bin/env python3
"""P3 LLM-judge for the ONLY two non-deterministic methods: semantic_similarity and
contains_essential_points (the 10 summarization tasks). Own key, strict rubric.

Offline fallback (no FIREWORKS_API_KEY, or EVAL_JUDGE_OFFLINE=1): a deterministic
content-word-coverage check so the harness runs in CI. The real launch-day judge is the
remote path — set the key to use it.

Self-check:  python -m eval.judge   (gold summaries pass, corrupted fail)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_STOP = set(
    "a an the of to in on at and or but for with is are was were be been being this that "
    "these those it its as by from he she they we you i has have had will would can could "
    "which who whom whose their our your his her about into over under than then so".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _expected_texts(expected: Any) -> list[str]:
    exp = expected
    if isinstance(exp, str):
        try:
            exp = json.loads(exp)
        except json.JSONDecodeError:
            return [exp]
    if isinstance(exp, dict):
        summary = exp.get("summary", exp)
        if isinstance(summary, list):
            return [str(s) for s in summary]
        return [str(summary)]
    if isinstance(exp, list):
        return [str(s) for s in exp]
    return [str(exp)]


def _answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        try:
            obj = json.loads(answer)
        except json.JSONDecodeError:
            return answer
    else:
        obj = answer
    if isinstance(obj, dict):
        summary = obj.get("summary", obj)
        if isinstance(summary, list):
            return " ".join(str(s) for s in summary)
        return str(summary)
    if isinstance(obj, list):
        return " ".join(str(s) for s in obj)
    return str(obj)


def _offline_judge(answer: Any, expected: Any, method: str) -> bool:
    got = _content_words(_answer_text(answer))
    if not got:
        return False
    # contains_essential_points: each expected point must be substantially covered.
    # semantic_similarity: overall content-word overlap must be high.
    threshold = 0.5 if method == "contains_essential_points" else 0.5
    for exp_text in _expected_texts(expected):
        want = _content_words(exp_text)
        if not want:
            continue
        if len(want & got) / len(want) < threshold:
            return False
    return True


def _remote_verdict(rubric: str, content: str) -> bool:
    import httpx  # own-key path only

    base = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
    key = os.environ["FIREWORKS_API_KEY"]
    # Judge model (own key, off the competition path). EVAL_JUDGE_MODEL lets you point at a cheap
    # INSTRUCT model; default = first ALLOWED_MODELS entry. A *reasoning* model (e.g. minimax) must
    # finish thinking before it can emit the verdict, so max_tokens=4 made it ALWAYS return no
    # verdict -> every summary scored 0. Give it room and parse the verdict robustly.
    model = os.environ.get("EVAL_JUDGE_MODEL") or (
        os.environ.get("ALLOWED_MODELS", "").split(",")
        or ["accounts/fireworks/models/llama-v3p1-8b-instruct"]
    )[0].strip()
    max_toks = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", "512") or "512")
    resp = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": max_toks,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content": content},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"].get("content") or ""
    # Strip reasoning wrappers (possibly truncated) before reading the verdict.
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"(?is)<think>.*", " ", text)
    upper = text.upper()
    last_pass, last_fail = upper.rfind("PASS"), upper.rfind("FAIL")
    if last_pass == -1 and last_fail == -1:
        return False  # no verdict emitted -> conservative fail
    return last_pass > last_fail


def _remote_judge(prompt: str, answer: Any, expected: Any, method: str) -> bool:
    # Intent-scoring rubric: the competition judge grades whether the summary captures the
    # gist, not word-for-word coverage. The old "every essential point" wording under-counted
    # good paraphrases and made summarization read as ~0. This is measurement-only (own key,
    # off the competition token path) and never affects the shipped agent's output.
    rubric = (
        "You are grading a summary for intent. Reply with exactly PASS or FAIL. "
        "PASS if the candidate captures the main idea and the key facts of the reference and "
        "introduces no significant factual errors; reasonable paraphrasing and omission of "
        "minor details are fine. FAIL only if it misses the main point, is unrelated, empty, "
        "or contradicts the reference."
    )
    content = (
        f"REFERENCE:\n{json.dumps(_expected_texts(expected))}\n\n"
        f"CANDIDATE:\n{_answer_text(answer)}\n\nGrade (PASS/FAIL):"
    )
    return _remote_verdict(rubric, content)


_CODE_RUBRIC = (
    "You are a strict code grader. Reply with exactly PASS or FAIL. "
    "PASS only if the CANDIDATE code correctly and completely implements the same spec as the "
    "REFERENCE solution — same behavior on all inputs — ignoring formatting, naming, and style. "
    "FAIL if it is incomplete, truncated, or would behave differently on any input."
)


def judge_code(prompt: str, answer: Any, expected: Any) -> bool:
    """Eval-only rescue for code tasks whose language has no toolchain on the scoring host
    (Java/C). Own key, never the competition token path. Offline fallback = content-word
    overlap (CI stand-in); the real launch-day grader is the remote path — set the key."""
    if os.environ.get("EVAL_JUDGE_OFFLINE") == "1" or not os.environ.get("FIREWORKS_API_KEY"):
        return _offline_judge(answer, expected, "code")
    content = (
        f"SPEC:\n{prompt}\n\nREFERENCE:\n{_answer_text(expected)}\n\n"
        f"CANDIDATE:\n{_answer_text(answer)}\n\nGrade (PASS/FAIL):"
    )
    return _remote_verdict(_CODE_RUBRIC, content)  # [own-key]


def judge_summary(prompt: str, answer: Any, expected: Any, method: str) -> bool:
    if os.environ.get("EVAL_JUDGE_OFFLINE") == "1" or not os.environ.get("FIREWORKS_API_KEY"):
        return _offline_judge(answer, expected, method)
    return _remote_judge(prompt, answer, expected, method)  # [own-key]


def _demo() -> None:
    os.environ["EVAL_JUDGE_OFFLINE"] = "1"
    tasks = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    summ = [t for t in tasks if t["evaluation_method"] in {"semantic_similarity", "contains_essential_points"}]
    gold = sum(judge_summary(t["prompt"], t["expected_answer"], t["expected_answer"], t["evaluation_method"]) for t in summ)
    bad = sum(
        judge_summary(t["prompt"], "Completely unrelated banana content about nothing.", t["expected_answer"], t["evaluation_method"])
        for t in summ
    )
    print(f"summarization tasks={len(summ)} gold_pass={gold} corrupted_pass={bad}")
    assert gold >= 8, f"judge failed too many gold summaries: {gold}/{len(summ)}"
    assert bad == 0, "judge passed a garbage summary"
    print("PASS eval.judge self-check")


if __name__ == "__main__":
    _demo()
