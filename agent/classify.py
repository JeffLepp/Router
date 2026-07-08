from __future__ import annotations

import re
from dataclasses import dataclass


CATEGORIES = (
    "actual_qa",
    "math_reasoning",
    "sentiment_analysis",
    "summarization",
    "named_entity_recognition",
    "code_debugging",
    "logic_puzzles",
    "code_generation",
)

LEGACY_CATEGORY_MAP = {
    "factual": "actual_qa",
    "math": "math_reasoning",
    "sentiment": "sentiment_analysis",
    "ner": "named_entity_recognition",
    "code_debug": "code_debugging",
    "logic": "logic_puzzles",
    "code_gen": "code_generation",
}


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float
    reason: str


_CODE_HINTS = re.compile(
    r"```|def\s+\w+\(|function\s+\w+\(|class\s+\w+|Traceback|bug|debug|fix|error",
    re.I,
)
_MATH_HINTS = re.compile(
    r"\b(calculate|compute|solve|percent|percentage|probability|rate|total|sum|"
    r"difference|product|ratio|projection|interest|average|square|divided|how many|"
    r"how much|quadratic|triangle|area)\b|[-+*/=]\s*\d|[\u00d7\u00f7]",
    re.I,
)
_SUMMARY_HINTS = re.compile(
    r"\b(summarize|summary|condense|tl;?dr|in \d+ words|bullet summary|"
    r"one sentence|executive summary)\b",
    re.I,
)
_SENTIMENT_HINTS = re.compile(
    r"\b(sentiment|positive|negative|neutral|classify.*review|label.*tone|"
    r"loved|love|terrible|waste of time|exceeded|disappoints)\b",
    re.I,
)
_NER_HINTS = re.compile(
    r"\b(named entit|extract entit|identify all people|identify organizations|"
    r"identify persons|extract customer|person|organization|location|date|MLS|NER\b)\b",
    re.I,
)
_LOGIC_HINTS = re.compile(
    r"\b(logic|deduce|deductive|constraint|puzzle|arrangement|who owns|which person|"
    r"truthful|liar|all conditions|sits|seat|houses|boxes|statements|can you conclude|"
    r"consistent at the same time|all [a-z]+ are|some [a-z]+|older than|"
    r"what day will it be)\b",
    re.I,
)
_CODE_GEN_HINTS = re.compile(
    r"\b(write|implement|create|design|return)\b.*\b(function|method|class|python|"
    r"javascript|java|sql|c\+\+|code|table)\b",
    re.I | re.S,
)


def canonical_category(category: str) -> str:
    mapped = LEGACY_CATEGORY_MAP.get(category, category)
    return mapped if mapped in CATEGORIES else "actual_qa"


async def classify(prompt: str, local_tiebreaker: object | None = None) -> Classification:
    text = prompt.strip()
    lower = text.lower()

    if _CODE_HINTS.search(text):
        if re.search(r"\b(bug|debug|fix|error|traceback|incorrect|broken)\b", lower):
            return Classification("code_debugging", 0.88, "code debug keywords")
        if _CODE_GEN_HINTS.search(text) or "write code" in lower:
            return Classification("code_generation", 0.82, "code generation keywords")
    if _SUMMARY_HINTS.search(text):
        return Classification("summarization", 0.9, "summarization keywords")
    if _NER_HINTS.search(text):
        return Classification("named_entity_recognition", 0.86, "entity extraction keywords")
    if _SENTIMENT_HINTS.search(text):
        return Classification("sentiment_analysis", 0.84, "sentiment keywords")
    if _LOGIC_HINTS.search(text):
        return Classification("logic_puzzles", 0.82, "logic keywords")
    if _MATH_HINTS.search(text):
        return Classification("math_reasoning", 0.8, "math keywords")
    if _CODE_GEN_HINTS.search(text):
        return Classification("code_generation", 0.75, "code generation keywords")

    # P1 keeps the optional local tiebreak hook physically off; low confidence routes general.
    return Classification("actual_qa", 0.55, "default factual")

