from __future__ import annotations

import json
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any


UNANSWERABLE = "null"

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def normalize(text: str) -> str:
    replacements = {
        "\u00d7": "*",
        "\u00f7": "/",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b2": "^2",
        "Ã—": "*",
        "Ã·": "/",
        "âˆ’": "-",
        "â€”": "-",
        "â€“": "-",
        "â€™": "'",
        "Â²": "^2",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def extract_source_text(prompt: str) -> str:
    text = normalize(prompt)
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text, flags=re.S)
    if quoted:
        return next((first or second for first, second in quoted if first or second), "")
    match = re.search(r"\bfrom:\s*(.+)$", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_number(raw: str) -> Decimal:
    raw = raw.strip().lower().replace(",", "")
    if raw in NUMBER_WORDS:
        return Decimal(NUMBER_WORDS[raw])
    return Decimal(raw)


def format_decimal(value: Decimal | int | float | Fraction) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    dec = Decimal(str(value))
    if dec == dec.to_integral_value():
        return str(int(dec))
    rendered = format(dec.normalize(), "f")
    return rendered.rstrip("0").rstrip(".")


def sentence_is_short_factual(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text)
    return 3 <= len(words) <= 12 and text.strip().endswith(".")

