from __future__ import annotations

import json
import re
from typing import Any


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def within_word_limit(text: str, limit: int) -> bool:
    return word_count(text) <= limit


def valid_json_shape(text: str, required_keys: set[str] | None = None) -> bool:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return False
    if required_keys and isinstance(data, dict):
        return required_keys.issubset(data)
    if required_keys and isinstance(data, list):
        return all(isinstance(item, dict) and required_keys.issubset(item) for item in data)
    return True


def englishish(text: str) -> bool:
    if not text.strip():
        return False
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text)
    return letters == 0 or ascii_letters / max(1, letters) >= 0.75


def entities_are_substrings(source: str, entities: list[str]) -> bool:
    return all(entity in source for entity in entities)


def valid_entities_shape(data: Any) -> bool:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return False
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("type"), str)
        for item in data["entities"]
    )


def _self_check() -> None:
    assert word_count("one two-three") == 2
    assert within_word_limit("one two", 2)
    assert valid_json_shape('{"task_id":"t1","answer":"ok"}', {"task_id", "answer"})
    assert englishish("This is a concise English answer.")
    assert entities_are_substrings("Ada met AMD in Seattle.", ["Ada", "AMD"])
    assert valid_entities_shape('{"entities":[{"text":"Ada","type":"PERSON"}]}')


if __name__ == "__main__":
    _self_check()
    print("format_v self-check passed")
