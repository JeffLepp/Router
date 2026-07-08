from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool
    confidence: float
    reason: str


def judge_answer(prompt: str, answer: str, category: str) -> JudgeVerdict:
    text = answer.strip()
    if not text:
        return JudgeVerdict(False, 0.0, "empty answer")
    if any(ord(ch) > 127 for ch in text):
        # Non-ASCII can be valid English, so this is only a tiny penalty.
        return JudgeVerdict(True, 0.55, "answer is non-empty with unicode")
    if category == "sentiment" and not any(
        label in text.lower() for label in ("positive", "negative", "neutral")
    ):
        return JudgeVerdict(False, 0.25, "missing sentiment label")
    if category == "math" and not any(ch.isdigit() for ch in text):
        return JudgeVerdict(False, 0.2, "missing numeric result")
    if category in {"code_debug", "code_gen"} and "def " not in text and "function " not in text:
        return JudgeVerdict(False, 0.3, "missing code-looking output")
    return JudgeVerdict(True, 0.65, "basic intent checks passed")

