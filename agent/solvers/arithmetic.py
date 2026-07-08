from __future__ import annotations

import math
import re
from decimal import Decimal, DivisionByZero, InvalidOperation
from fractions import Fraction

from agent.solvers.common import UNANSWERABLE, format_decimal, normalize, parse_number
from agent.verify.math_v import safe_eval


_SAFE_EXPR_RE = re.compile(r"^[\d\s().+\-*/^]+$")
_NUM = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"


def solve(prompt: str) -> str | None:
    text = normalize(prompt)
    lower = text.lower()

    if _is_division_by_zero(lower):
        return UNANSWERABLE

    for handler in (
        _triangle_area,
        _quadratic_roots,
        _probability_without_replacement,
        _percent_of,
        _square_of,
        _give_away,
        _average,
        _direct_expression,
    ):
        answer = handler(text)
        if answer is not None:
            return answer
    return None


def _is_division_by_zero(lower: str) -> bool:
    return bool(
        re.search(r"\b(divided by|divide by|division by)\s+zero\b", lower)
        or re.search(r"[/]\s*0(?:\D|$)", lower)
    )


def _triangle_area(text: str) -> str | None:
    lower = text.lower()
    if "triangle" not in lower or "area" not in lower:
        return None
    match = re.search(
        rf"side lengths?\s+({_NUM})\s*(?:cm|m|in)?\s*,?\s+({_NUM})\s*(?:cm|m|in)?\s*,?\s*(?:and)?\s+({_NUM})",
        lower,
    )
    if not match:
        return None
    sides = [parse_number(part) for part in match.groups()]
    a, b, c = sorted(sides)
    if a + b <= c:
        return UNANSWERABLE
    s = (a + b + c) / Decimal(2)
    area = Decimal(str(math.sqrt(float(s * (s - a) * (s - b) * (s - c)))))
    return format_decimal(area.quantize(Decimal("0.0001")))


def _quadratic_roots(text: str) -> str | None:
    compact = re.sub(r"\s+", "", normalize(text).lower())
    compact = compact.replace("**2", "^2")
    match = re.search(r"x\^2([+-]\d*)x([+-]\d+)=0", compact)
    if not match:
        return None
    b_raw, c_raw = match.groups()
    b = _signed_coeff(b_raw)
    c = int(c_raw)
    disc = b * b - 4 * c
    if disc < 0:
        return UNANSWERABLE
    root = int(math.isqrt(disc))
    if root * root != disc:
        return None
    roots = sorted({Fraction(-b - root, 2), Fraction(-b + root, 2)})
    return "[" + ", ".join(format_decimal(root_value) for root_value in roots) + "]"


def _signed_coeff(raw: str) -> int:
    if raw == "+":
        return 1
    if raw == "-":
        return -1
    return int(raw)


def _probability_without_replacement(text: str) -> str | None:
    lower = text.lower()
    if "without replacement" not in lower or "probability" not in lower:
        return None
    color_counts = {
        color: int(count.replace(",", ""))
        for count, color in re.findall(rf"({_NUM})\s+(red|blue|green|yellow|black|white)", lower)
    }
    match = re.search(r"both(?:\s+are)?\s+(\w+)", lower)
    if not color_counts or not match:
        return None
    wanted = match.group(1)
    if wanted not in color_counts:
        return None
    total = sum(color_counts.values())
    count = color_counts[wanted]
    if total < 2 or count < 2:
        return "0"
    return format_decimal(Fraction(count * (count - 1), total * (total - 1)))


def _percent_of(text: str) -> str | None:
    lower = text.lower()
    match = re.search(rf"(?:what is\s+)?({_NUM})\s*(?:%|percent)\s+of\s+({_NUM})", lower)
    if not match:
        return None
    pct, base = (parse_number(part) for part in match.groups())
    return format_decimal(base * pct / Decimal(100))


def _square_of(text: str) -> str | None:
    match = re.search(rf"square of\s+({_NUM})", text, flags=re.I)
    if not match:
        return None
    value = parse_number(match.group(1))
    return format_decimal(value * value)


def _give_away(text: str) -> str | None:
    match = re.search(
        rf"have\s+({_NUM})\s+\w+.*?give away\s+({_NUM})",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return None
    start, removed = (parse_number(part) for part in match.groups())
    return format_decimal(start - removed)


def _average(text: str) -> str | None:
    if "average" not in text.lower():
        return None
    nums = [parse_number(part) for part in re.findall(_NUM, text)]
    if not nums:
        return None
    return format_decimal(sum(nums) / Decimal(len(nums)))


def _direct_expression(text: str) -> str | None:
    match = re.search(
        r"(?:what is|calculate|compute|solve)\s+(.+?)(?:\?|$)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return None
    expr = match.group(1).strip()
    expr = re.sub(r"\b(?:the|value of|answer to)\b", "", expr, flags=re.I).strip()
    expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr, flags=re.I)
    expr = expr.replace("^", "**")
    if not _SAFE_EXPR_RE.match(expr.replace("**", "^")):
        return None
    if not re.search(r"[+\-*/]", expr):
        return None
    try:
        value = safe_eval(expr)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError, ValueError, SyntaxError):
        return UNANSWERABLE if re.search(r"/\s*0(?:\D|$)", expr) else None
    return format_decimal(value)


def _self_check() -> None:
    assert solve("What is 25 x 4?") == "100"
    assert solve("What is the square of 9?") == "81"
    assert solve("Find all real solutions x to the quadratic equation x^2 - 5x + 6 = 0.") == "[2, 3]"
    assert solve("A jar contains 5 red, 7 blue, and 8 green marbles. If you randomly draw 2 marbles without replacement, what is the probability that both are blue?") == "21/190"
    assert solve("What is 10 divided by zero?") == UNANSWERABLE
    assert solve("A triangle has side lengths 2 cm, 3 cm, and 10 cm. What is its area?") == UNANSWERABLE
    assert solve("How many apples did Joan keep after a complicated trade?") is None


if __name__ == "__main__":
    _self_check()
    print("arithmetic solver self-check passed")
