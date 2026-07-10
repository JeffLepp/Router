from __future__ import annotations

import re

from agent.solvers.common import NUMBER_WORDS, normalize


_BULLET_REQUEST = re.compile(
    r"\b(?:in|into)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\s+bullets?\b",
    re.I,
)
_ACTION_LEAD = re.compile(r"\bAction\s+items\s*:\s*", re.I)
_ACTION_START = re.compile(r"^[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*\s+will\s+\S", re.I)


def solve(prompt: str) -> str | None:
    """Copy a fully structured meeting report without asking a model to paraphrase it.

    This intentionally supports only the benchmark-independent shape whose correctness can be
    checked mechanically: an explicit bullet count, a quoted/source passage, and an explicit
    ``Action items:`` clause made of named ``will`` assignments. Anything incomplete defers.
    """

    text = normalize(prompt).strip()
    count_match = _BULLET_REQUEST.search(text)
    action_match = _ACTION_LEAD.search(text)
    if not count_match or not action_match or "meeting" not in text[: action_match.start()].lower():
        return None
    requested = _parse_count(count_match.group(1))
    if requested is None or not 2 <= requested <= 12:
        return None

    source = _source_passage(text)
    if source is None:
        return None
    source_action = _ACTION_LEAD.search(source)
    if not source_action:
        return None
    narrative = source[: source_action.start()].strip()
    action_text = source[source_action.end() :].strip()
    narrative_sentences = _sentences(narrative)
    actions = _actions(action_text)
    if not narrative_sentences or not actions:
        return None

    # The explicit action sentence is one requested summary bullet. Refuse to invent, merge,
    # or omit narrative sentences merely to hit a requested count.
    if len(narrative_sentences) + 1 != requested:
        return None
    action_sentence = "Action items: " + _join_actions(actions)
    summary_bullets = narrative_sentences + [action_sentence]
    if len(summary_bullets) != requested:
        return None

    # Every action must survive byte-for-byte (apart from terminal punctuation) both in the
    # summary action bullet and in the separately requested action-item list.
    rendered = [*(f"- {sentence}" for sentence in summary_bullets), "", "Action items:"]
    rendered.extend(f"- {_terminal(action)}" for action in actions)
    answer = "\n".join(rendered)
    if any(_strip_terminal(action) not in answer for action in actions):
        return None
    return answer


def _parse_count(raw: str) -> int | None:
    lowered = raw.lower()
    if lowered in NUMBER_WORDS:
        return NUMBER_WORDS[lowered]
    try:
        return int(raw)
    except ValueError:
        return None


def _source_passage(prompt: str) -> str | None:
    """Unwrap a single outer quote without treating apostrophes as delimiters."""

    _instruction, sep, body = prompt.partition(":")
    if not sep:
        return None
    body = body.strip()
    if len(body) >= 2 and body[0] in {'"', "'"} and body[-1] == body[0]:
        body = body[1:-1].strip()
    if len(body) < 40:
        return None
    return body


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if any(not re.search(r"[.!?]$", part) for part in parts):
        return []
    return parts


def _actions(text: str) -> list[str]:
    cleaned = text.strip()
    if cleaned.endswith("'") or cleaned.endswith('"'):
        cleaned = cleaned[:-1].rstrip()
    cleaned = _strip_terminal(cleaned)
    parts = re.split(
        r",\s+and\s+(?=[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*\s+will\b)"
        r"|,\s+(?=[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*\s+will\b)"
        r"|\s+and\s+(?=[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*\s+will\b)",
        cleaned,
    )
    actions = [part.strip() for part in parts if part.strip()]
    if not actions or any(not _ACTION_START.match(action) for action in actions):
        return []
    assignees = [re.match(r"^([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*)\s+will\b", action, re.I).group(1).lower() for action in actions]  # type: ignore[union-attr]
    if len(set(assignees)) != len(assignees):
        return []
    return actions


def _join_actions(actions: list[str]) -> str:
    if len(actions) == 1:
        return _terminal(actions[0])
    if len(actions) == 2:
        return f"{_strip_terminal(actions[0])}, and {_terminal(actions[1])}"
    return ", ".join(_strip_terminal(action) for action in actions[:-1]) + ", and " + _terminal(actions[-1])


def _strip_terminal(text: str) -> str:
    return text.strip().rstrip(".!?")


def _terminal(text: str) -> str:
    return _strip_terminal(text) + "."


def _self_check() -> None:
    prompt = (
        "Summarize this meeting in three bullets and list all action items:\n\n"
        "'The team reviewed launch readiness. Mina found a blocker. "
        "Action items: Mina will fix the blocker, and Jo will rerun the tests.'"
    )
    answer = solve(prompt)
    assert answer is not None
    assert answer.split("\n\n", 1)[0].count("\n- ") + 1 == 3
    assert "Mina will fix the blocker" in answer and "Jo will rerun the tests" in answer
    assert solve("Summarize in three bullets: No action section here.") is None
    assert solve(prompt.replace("three bullets", "four bullets")) is None
    assert solve(prompt.replace("Jo will rerun", "Mina will rerun")) is None


if __name__ == "__main__":
    _self_check()
    print("summary solver self-check passed")
