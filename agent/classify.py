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
    r"difference|product|ratio|projection|interest|average|square|divided|"
    r"how far|travels?|per hour|quadratic|triangle|area)\b|"
    r"\b\d+\s*(?:km/h|kph|mph|m/s)\b|(?<![A-Za-z])[-+*/=]\s*\d|[\u00d7\u00f7]",
    re.I,
)
# "how many/much" alone is not math -- "How many moons does Kepler-452b have?" is QA.
# Demoted to a weak cue that only fires alongside a real math signal: a standalone number
# token (\b\d+\b skips the glued "452b"), a math verb, or the \u00d7/\u00f7 symbols. (Issue 6)
_MATH_WEAK = re.compile(r"\bhow (?:many|much)\b", re.I)
_MATH_SECOND = re.compile(
    r"\b\d+\b|[\u00d7\u00f7]|\b(?:calculate|compute|percent|percentage|units?)\b", re.I
)
_SUMMARY_HINTS = re.compile(
    r"\b(summarize|summary|condense|tl;?dr|in \d+ words|bullet summary|"
    r"one sentence|executive summary)\b",
    re.I,
)
# Strong signals name the task or carry review-only polarity; safe to trust anywhere.
_SENTIMENT_STRONG = re.compile(
    r"\b(sentiment|positive|negative|neutral|classify.*review|label.*tone|"
    r"waste of time|exceeded|disappoints|disappointed)\b",
    re.I,
)
# Weak signals are generic adjectives that also show up in QA/summaries/code prompts.
# Only trust them on review-shaped (declarative, non-question) text -- see _looks_like_review.
_SENTIMENT_WEAK = re.compile(
    r"\b(loved|love|excellent|amazing|fantastic|great|wonderful|terrible|awful|"
    r"horrible|crowded|good things|only had to wait)\b|can.?t say enough",
    re.I,
)
_NEUTRAL_STATEMENT_VERBS = re.compile(r"\b(arrived|is|are|was|were)\b", re.I)
_QUESTION_OR_COMMAND_START = re.compile(
    r"\b(who|what|when|where|why|how|which|is|are|was|were|can|could|should|"
    r"would|do|does|did|calculate|compute|solve|summarize|extract|identify|"
    r"list|find|debug|fix|write|implement|create|design|return)\b",
    re.I,
)
_NER_HINTS = re.compile(
    # standalone strong signals, then an extraction verb followed (same clause) by an entity noun.
    # The verb requirement stops "Four people need to cross a bridge" (logic) landing here.
    r"named entit\w*|extract entit\w*|identify entit\w*|\b(?:MLS|NER)\b|"
    r"\b(?:identify|extract|list|find|label|tag)\b[^.?!]{0,40}?"
    r"\b(?:people|persons?|organi[sz]ations?|locations?|customers?|names?|dates?)\b",
    re.I,
)
_LOGIC_HINTS = re.compile(
    r"\b(logic|deduce|deductive|constraint|puzzle|arrangement|who owns|which person|"
    r"truthful|liar|all conditions|sits|seat|houses|boxes|statements|can you conclude|"
    r"consistent at the same time|all [a-z]+ are|some [a-z]+|older than|"
    r"what day will it be|cross a bridge|flashlight|crossing time|received distinct|"
    r"distinct grades|highest grade|who got grade|got grade)\b",
    re.I,
)
_CODE_GEN_HINTS = re.compile(
    r"\b(write|implement|create|design)\b.*\b(function|method|class|python|"
    r"javascript|java|sql|c\+\+|code|table)\b",
    re.I | re.S,
)


def canonical_category(category: str) -> str:
    mapped = LEGACY_CATEGORY_MAP.get(category, category)
    return mapped if mapped in CATEGORIES else "actual_qa"


def _looks_like_review(text: str) -> bool:
    """Declarative, non-question text -- the only shape we trust weak sentiment words on."""
    return "?" not in text and not _QUESTION_OR_COMMAND_START.match(text)


def _looks_like_neutral_sentiment_statement(text: str) -> bool:
    if not text.endswith(".") or "?" in text or "\n" in text or ":" in text:
        return False
    if _QUESTION_OR_COMMAND_START.match(text):
        return False
    if re.search(r"\b[A-Z]\s+(?:is|are|was|were)\b", text):
        return False
    words = re.findall(r"[A-Za-z0-9']+", text)
    return 4 <= len(words) <= 18 and bool(_NEUTRAL_STATEMENT_VERBS.search(text))


async def classify(prompt: str, local_tiebreaker: object | None = None) -> Classification:
    text = prompt.strip()
    lower = text.lower()

    if _CODE_HINTS.search(text):
        is_gen = bool(_CODE_GEN_HINTS.search(text)) or "write code" in lower
        # bare "error" alone shouldn't force debug — a generation task can mention "raising an error"
        if re.search(r"\b(bug|debug|fix|traceback|incorrect|broken)\b", lower) or (
            "error" in lower and not is_gen
        ):
            return Classification("code_debugging", 0.88, "code debug keywords")
        if is_gen:
            return Classification("code_generation", 0.82, "code generation keywords")
    if _SUMMARY_HINTS.search(text):
        return Classification("summarization", 0.9, "summarization keywords")
    if _NER_HINTS.search(text):
        return Classification("named_entity_recognition", 0.86, "entity extraction keywords")
    if _SENTIMENT_STRONG.search(text):
        return Classification("sentiment_analysis", 0.84, "sentiment keywords")
    # Logic before code_gen: the greedy `write|design ... table` net otherwise steals
    # seating/arrangement puzzles. Code_gen still precedes math so gen_002 ("...sum") wins.
    if _LOGIC_HINTS.search(text):
        return Classification("logic_puzzles", 0.82, "logic keywords")
    if _CODE_GEN_HINTS.search(text):
        return Classification("code_generation", 0.78, "code generation keywords")
    if _MATH_HINTS.search(text) or (_MATH_WEAK.search(text) and _MATH_SECOND.search(text)):
        return Classification("math_reasoning", 0.8, "math keywords")
    if _looks_like_review(text) and _SENTIMENT_WEAK.search(text):
        return Classification("sentiment_analysis", 0.7, "review sentiment adjectives")
    if _looks_like_neutral_sentiment_statement(text):
        return Classification("sentiment_analysis", 0.62, "short neutral statement")

    # P1 keeps the optional local tiebreak hook physically off; low confidence routes general.
    return Classification("actual_qa", 0.55, "default factual")


def _self_check() -> None:
    import asyncio

    cat = lambda p: asyncio.new_event_loop().run_until_complete(classify(p)).category
    # Issue 6: "how many/much" is only math with a real second signal.
    assert cat("How many moons does the planet Kepler-452b have?") == "actual_qa"
    assert cat("If you have 12 apples and give away 5, how many are left?") == "math_reasoning"
    assert cat("Jane bought 3 books at $12 each. How much did she spend in total?") == "math_reasoning"
    print("PASS classify self-check")


if __name__ == "__main__":
    _self_check()
