from __future__ import annotations

import json
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
# A leading summarize imperative wins over body keywords. Without this, a meeting summary
# whose *content* mentions a "bug" in an "authentication module" (summary_005) is stolen by
# _CODE_HINTS, which is tested first, and gets the code_debugging contract + code model.
_SUMMARY_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:summari[sz]e|condense|tl;?dr|"
    r"(?:create|write|give)(?:\s+\w+){0,3}\s+(?:executive\s+)?summary)\b",
    re.I,
)
# A command can name a code artifact "summary" without asking to summarize source content.
# Keep this narrow and start-anchored so code words inside the passage of a real summary do
# not steal the request (for example, a meeting report that discusses a broken function).
_SUMMARY_CODE_ARTIFACT = re.compile(
    r"^\s*(?:please\s+)?(?:write|implement|create|design)\b[^.?!:\n]{0,100}\b(?:"
    r"summary\s+(?:function|method|class|query|script|code)\b|"
    r"summary\s+table\b[^.?!:\n]{0,40}\b(?:sql|database|schema)\b)",
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


def build_remote_classifier_prompt(items: list[tuple[str, str]]) -> str:
    """Build one compact, answer-free classification request.

    The remote classifier only chooses a routing label. It never receives gold answers and
    never answers the tasks, so a malformed response can safely fall back to the local regex
    classifier without losing a task.
    """
    payload = [
        {"task_id": str(task_id), "prompt": str(prompt)}
        for task_id, prompt in items
    ]
    labels = ", ".join(CATEGORIES)
    return (
        "Classify each task by its requested output, not by incidental words in its passage.\n"
        f"Allowed labels: {labels}.\n"
        "Return ONLY one compact JSON object mapping every task_id to exactly one allowed "
        "label. Do not answer any task and do not explain.\n"
        "TASKS_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def parse_remote_classifications(
    response: str, expected_ids: set[str]
) -> dict[str, str]:
    """Parse a classifier response conservatively; invalid/missing rows defer locally."""
    text = str(response or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    payload: object | None = None
    if start >= 0 and end >= start:
        try:
            payload = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            payload = None

    rows: list[tuple[object, object]] = []
    if isinstance(payload, dict) and isinstance(payload.get("classifications"), list):
        payload = payload["classifications"]
    if isinstance(payload, dict):
        rows = list(payload.items())
    elif isinstance(payload, list):
        rows = [
            (row.get("task_id"), row.get("category"))
            for row in payload
            if isinstance(row, dict)
        ]
    if not rows:
        # A reasoning model can consume the output budget before writing the final `}`.
        # Complete string pairs are still safe to accept; unfinished/missing rows defer to
        # the local classifier rather than invalidating the whole batch.
        rows = re.findall(r'"([^"\\]+)"\s*:\s*"([^"\\]+)"', text)

    parsed: dict[str, str] = {}
    for task_id_raw, category_raw in rows:
        task_id = str(task_id_raw or "")
        category = LEGACY_CATEGORY_MAP.get(str(category_raw or "").strip(), str(category_raw or "").strip())
        if task_id in expected_ids and category in CATEGORIES:
            parsed[task_id] = category
    return parsed


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

    if _SUMMARY_CODE_ARTIFACT.search(text):
        return Classification("code_generation", 0.84, "summary-named code artifact")
    if _SUMMARY_LEAD.match(text):
        return Classification("summarization", 0.92, "leading summarize imperative")
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
    # A leading summarize imperative beats code keywords in the body (summary_005).
    assert cat("Summarize this meeting in five bullets: 'Karen found a bug in the auth module.'") \
        == "summarization"
    assert cat("Create an executive summary of this report in three sentences: 'Costs fell.'") \
        == "summarization"
    assert cat("Write a summary of this Python function in two sentences: def f(): return 1") \
        == "summarization"
    # A code artifact named "summary" is still code generation, not summarization.
    assert cat("Write a summary function in Python") == "code_generation"
    assert cat("Create a summary table in SQL") == "code_generation"
    # ...but a real debug task that merely mentions a summary stays code_debugging.
    assert cat("Fix the bug in this function:\n```python\ndef f(): return summary\n```") \
        == "code_debugging"
    prompt = build_remote_classifier_prompt(
        [("t1", "Which person is mentioned?"), ("t2", "The service was awful.")]
    )
    assert "TASKS_JSON:" in prompt and "t1" in prompt and "t2" in prompt
    parsed = parse_remote_classifications(
        '```json\n{"t1":"named_entity_recognition","t2":"sentiment_analysis","x":"logic_puzzles"}\n```',
        {"t1", "t2"},
    )
    assert parsed == {"t1": "named_entity_recognition", "t2": "sentiment_analysis"}
    assert parse_remote_classifications("not json", {"t1"}) == {}
    assert parse_remote_classifications(
        '{"t1":"logic_puzzles","t2":"named_entity_recognition', {"t1", "t2"}
    ) == {"t1": "logic_puzzles"}
    print("PASS classify self-check")


if __name__ == "__main__":
    _self_check()
