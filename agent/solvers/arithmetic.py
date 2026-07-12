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
        _probability_two_draw,
        _percent_of,
        _power_phrase,
        _square_of,
        _op_chain,
        _give_away,
        _average,
        _direct_expression,
    ):
        answer = handler(text)
        if answer is not None:
            return answer
    return None


def solve_strict(prompt: str) -> str | None:
    """Solve only explicit arithmetic forms with no inferred real-world operation.

    These handlers read the exact operator from the prompt and recompute the result. Word
    problems, averages, probability prose, geometry, and invalid/unanswerable judgments are
    intentionally excluded because their interpretation can shift out of distribution.
    """
    text = normalize(prompt)
    for handler in (_percent_of, _square_of, _direct_expression):
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


_TWO_DRAW_CUE = re.compile(
    r"without replacement|without putting|without returning|at once|at the same time|"
    r"one after the other|simultaneously",
    re.I,
)


def _probability_two_draw(text: str) -> str | None:
    lower = text.lower()
    if "probability" not in lower or not _TWO_DRAW_CUE.search(lower):
        return None
    color_counts = {
        color: int(count.replace(",", ""))
        for count, color in re.findall(rf"({_NUM})\s+(red|blue|green|yellow|black|white)", lower)
    }
    match = re.search(r"both(?:\s+\w+)?\s+(?:are|will be|come out)\s+(\w+)", lower) or re.search(
        r"both\s+(\w+)", lower
    )
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


_POWER_WORDS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}


def _power_phrase(text: str) -> str | None:
    lower = text.lower()
    match = re.search(rf"({_NUM})\s+raised to the\s+(\w+)\s+power", lower)
    if match:
        base = parse_number(match.group(1))
        word = match.group(2)
        exp = _POWER_WORDS.get(word)
        if exp is None:
            digits = re.match(r"(\d+)(?:st|nd|rd|th)?$", word)
            if not digits:
                return None
            exp = int(digits.group(1))
        if exp > 12:
            return None
        return format_decimal(base**exp)
    match = re.search(rf"cube of\s+({_NUM})|({_NUM})\s+cubed", lower)
    if match:
        value = parse_number(match.group(1) or match.group(2))
        return format_decimal(value * value * value)
    match = re.search(rf"({_NUM})\s+squared", lower)
    if match:
        value = parse_number(match.group(1))
        return format_decimal(value * value)
    return None


_OP_STEP = re.compile(
    rf"(double|triple|halve)\s+(?:it|that|the result)|"
    rf"(add|subtract|plus|minus)\s+({_NUM})|"
    rf"(multiply|divide)\s+(?:it|that|the result)?\s*by\s+({_NUM})",
    re.I,
)


def _op_chain(text: str) -> str | None:
    lower = text.lower()
    start = re.search(rf"(?:take|start with|begin with)\s+(?:the number\s+)?({_NUM})", lower)
    if not start:
        return None
    value = parse_number(start.group(1))
    steps = list(_OP_STEP.finditer(lower[start.end():]))
    if not steps:
        return None
    used = {start.group(1)}
    for step in steps:
        unary, addsub, addsub_num, muldiv, muldiv_num = step.groups()
        if unary:
            value = value * 2 if unary == "double" else value * 3 if unary == "triple" else value / 2
        elif addsub:
            used.add(addsub_num)
            operand = parse_number(addsub_num)
            value = value + operand if addsub in {"add", "plus"} else value - operand
        else:
            used.add(muldiv_num)
            operand = parse_number(muldiv_num)
            if muldiv == "divide" and operand == 0:
                return UNANSWERABLE
            value = value * operand if muldiv == "multiply" else value / operand
    # every number in the prompt must belong to the chain, or the shape is not this template
    if set(re.findall(_NUM, lower)) - used:
        return None
    return format_decimal(value)


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
    # only bare number lists ("average of 2, 4, and 9"); "average speed/score of ..." prose
    # needs interpretation and must go remote
    match = re.search(
        rf"average of\s+((?:{_NUM})(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?:{_NUM}))+)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    nums = [parse_number(part) for part in re.findall(_NUM, match.group(1))]
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
    assert solve("What is 7 raised to the third power?") == "343"
    assert solve("What is 4 cubed?") == "64"
    assert solve("Take the number 13, double it, then subtract 5. What number results?") == "21"
    assert solve("Take 10, add 5, then divide by 3.") == "5"
    assert solve("Take 10, add 5, but also consider the 7 bonus points.") is None
    assert solve(
        "A drawer holds 4 black socks and 6 white socks. You pull out two socks one after the "
        "other without putting the first back. What is the probability that both socks are white?"
    ) == "1/3"
    assert solve(
        "A bag holds 5 green marbles and 3 yellow marbles. You grab two marbles at once. "
        "What is the probability that both are green?"
    ) == "5/14"
    assert solve("What is the average of 2, 4, and 9?") == "5"
    assert solve("The average speed over 100 km taking 2 hours plus a 1 hour break?") is None
    assert solve_strict("What is 25 x 4?") == "100"
    assert solve_strict("What is 15% of 200?") == "30"
    assert solve_strict("A car travels 60 km/h for 3.5 hours. How far does it travel?") is None
    assert solve_strict("What is the average of 2, 4, and 9?") is None


if __name__ == "__main__":
    _self_check()
    print("arithmetic solver self-check passed")
