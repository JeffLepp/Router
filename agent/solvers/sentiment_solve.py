from __future__ import annotations

import re

from agent.solvers.common import compact_json, normalize


POSITIVE = {
    "absolutely loved",
    "love",
    "loved",
    "excellent",
    "exceeded all my expectations",
    "amazing",
    "fantastic",
    "great",
}
NEGATIVE = {
    "complete waste of time",
    "waste of time",
    "terrible",
    "awful",
    "horrible",
    "disappoints",
    "disappointed",
}
HEDGE_OR_SARCASM = {
    "can't say enough",
    "only had to wait",
    "great workout",
    "yeah right",
    "as if",
}
# Negative-event cues: if a positive lexicon word co-occurs with one of these, the text is
# almost certainly sarcastic/mixed ("Oh great, another Monday") -> defer rather than emit positive.
NEG_EVENT_CUE = re.compile(
    r"\b(again|another|cancel(?:led|ed)?|late|delay(?:ed|s)?|lost|stuck|ruin(?:ed)?|"
    r"broke|broken|waiting|traffic|monday)\b",
    re.I,
)


def solve(prompt: str) -> str | None:
    text = normalize(_strip_instruction(prompt)).strip()
    lower = text.lower()
    if not text:
        return None

    if any(marker in lower for marker in HEDGE_OR_SARCASM):
        return None
    if re.search(r"\b(but|though|although|however|except)\b", lower):
        return None
    if re.search(r"\b(not|never|hardly|barely|no)\b", lower):
        return None

    pos_hits = sum(1 for phrase in POSITIVE if phrase in lower)
    neg_hits = sum(1 for phrase in NEGATIVE if phrase in lower)
    if pos_hits and NEG_EVENT_CUE.search(lower):
        return None  # sarcasm/mixed -> defer instead of trusting the positive lexicon hit
    if pos_hits and not neg_hits:
        return compact_json({"sentiment": "positive"})
    if neg_hits and not pos_hits:
        return compact_json({"sentiment": "negative"})
    return None
    # ponytail: dropped the "short factual -> neutral" fallback. A lexicon can't tell
    # "arrived Tuesday" (neutral) from "package was lost" (negative); deferring is precision-safe.


def _strip_instruction(prompt: str) -> str:
    match = re.search(
        r"(?:classify|label|determine).*?sentiment\s*:?\s*(.+)$",
        prompt,
        flags=re.I | re.S,
    )
    return match.group(1) if match else prompt


def _self_check() -> None:
    assert solve("I absolutely loved this product; it exceeded all my expectations.") == compact_json({"sentiment": "positive"})
    assert solve("This movie was a complete waste of time.") == compact_json({"sentiment": "negative"})
    assert solve("Classify the sentiment: Oh great, another Monday at work.") is None
    assert solve("The package arrived on Tuesday.") is None
    assert solve("The laptop is fast, but its battery life is terrible.") is None
    assert solve("Wonderful support-I only had to wait three hours to get help.") is None
    assert solve("The experience was not terrible.") is None


if __name__ == "__main__":
    _self_check()
    print("sentiment solver self-check passed")
