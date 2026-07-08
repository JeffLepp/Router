from __future__ import annotations

import json
import re

from agent.solvers.common import compact_json
from agent.verify.format_v import valid_json_shape, within_word_limit


def solve(prompt: str) -> str | None:
    lower = prompt.lower()
    if "json" in lower and re.search(r"\b(repair|fix|valid)\b", lower):
        repaired = _repair_json(_extract_jsonish(prompt))
        if repaired is not None:
            return repaired
    if "word limit" in lower or "under" in lower:
        answer = _mechanical_length_answer(prompt)
        if answer is not None:
            return answer
    return None


def _extract_jsonish(prompt: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", prompt, flags=re.I | re.S)
    if fenced:
        return fenced[0].strip()
    match = re.search(r"(\{.*\}|\[.*\])", prompt, flags=re.S)
    return match.group(1).strip() if match else ""


def _repair_json(text: str) -> str | None:
    if not text:
        return None
    candidates = [text]
    quoted_keys = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', text)
    candidates.append(quoted_keys)
    candidates.append(re.sub(r",(\s*[}\]])", r"\1", quoted_keys))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        payload = compact_json(data)
        return payload if valid_json_shape(payload) else None
    return None


def _mechanical_length_answer(prompt: str) -> str | None:
    match = re.search(r"under\s+(\d+)\s+words?:\s*(.+)$", prompt, flags=re.I | re.S)
    if not match:
        return None
    limit = int(match.group(1))
    text = " ".join(match.group(2).split())
    words = text.split()
    if within_word_limit(text, limit):
        return text
    clipped = " ".join(words[:limit])
    return clipped if within_word_limit(clipped, limit) else None


def _self_check() -> None:
    assert solve("Fix this JSON: {name: \"Ada\", age: 37,}") == '{"name":"Ada","age":37}'
    assert solve("Return this under 3 words: alpha beta gamma delta") == "alpha beta gamma"
    assert solve("Fix this JSON: {bad") is None
    assert solve("Rewrite this creatively: hello") is None
    assert solve("Return this under 4 words: alpha beta") == "alpha beta"


if __name__ == "__main__":
    _self_check()
    print("format solver self-check passed")

