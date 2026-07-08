from __future__ import annotations

import re
from decimal import Decimal

from agent.solvers.common import NUMBER_WORDS, format_decimal, normalize, parse_number


_COUNT = r"\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS)
_NUM = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"


def solve(prompt: str) -> str | None:
    text = normalize(prompt)
    for handler in (
        _rate_times_time,
        _item_total,
        _rectangle_area,
        _reverse_percent,
        _discount_price,
        _profit_loss,
        _unit_conversion,
    ):
        answer = handler(text)
        if answer is not None:
            return answer
    return None


def _rate_times_time(text: str) -> str | None:
    match = re.search(
        rf"travels?\s+({_NUM})\s*(?:km/h|kph|mph|meters per second|m/s).*?for\s+({_NUM})\s*(?:hours?|hrs?|h)\b",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return None
    rate, duration = (parse_number(part) for part in match.groups())
    return format_decimal(rate * duration)


def _item_total(text: str) -> str | None:
    if not re.search(r"\b(total|spend|spent|cost)\b", text, flags=re.I):
        return None
    parts = re.findall(
        rf"\b({_COUNT})\s+\w+\s+at\s+\$?({_NUM})\s+each",
        text,
        flags=re.I,
    )
    if not parts:
        return None
    total = Decimal(0)
    for count_raw, price_raw in parts:
        total += parse_number(count_raw) * parse_number(price_raw)
    return format_decimal(total)


def _rectangle_area(text: str) -> str | None:
    lower = text.lower()
    if "rectangle" not in lower or "area" not in lower:
        return None
    length = re.search(rf"length\s+of\s+({_NUM})", lower)
    width = re.search(rf"width\s+of\s+({_NUM})", lower)
    if not length or not width:
        return None
    return format_decimal(parse_number(length.group(1)) * parse_number(width.group(1)))


_ORIGINAL_CUE = re.compile(r"\b(originally|original|before)\b", re.I)


def _reverse_percent(text: str) -> str | None:
    lower = text.lower()
    if not _ORIGINAL_CUE.search(lower):
        return None
    # "$X after a Y% discount/increase" -> price then pct (the common phrasing)
    match = re.search(
        rf"\$?({_NUM})\s+after\s+(?:a\s+)?({_NUM})\s*(?:%|percent)\s+(discount|increase)",
        lower,
    )
    if match:
        final_raw, pct_raw, direction = match.groups()
    else:
        # legacy: "after a Y% discount ... is/was $X ... original"
        legacy = re.search(
            rf"after\s+(?:a\s+)?({_NUM})\s*(?:%|percent)\s+(discount|increase).*?(?:is|was)\s+\$?({_NUM})",
            lower,
            flags=re.S,
        )
        if not legacy:
            return None
        pct_raw, direction, final_raw = legacy.groups()
    pct = parse_number(pct_raw) / Decimal(100)
    final = parse_number(final_raw)
    divisor = Decimal(1) - pct if direction == "discount" else Decimal(1) + pct
    if divisor == 0:
        return None
    return format_decimal(final / divisor)


def _discount_price(text: str) -> str | None:
    lower = text.lower()
    if _ORIGINAL_CUE.search(lower):
        return None  # asks for the pre-discount price -> _reverse_percent territory, never compute forward
    match = re.search(
        rf"\$?({_NUM}).*?({_NUM})\s*(?:%|percent)\s+discount",
        lower,
    )
    if not match:
        return None
    base, pct = (parse_number(part) for part in match.groups())
    return format_decimal(base * (Decimal(1) - pct / Decimal(100)))


def _profit_loss(text: str) -> str | None:
    lower = text.lower()
    match = re.search(
        rf"bought.*?\$?({_NUM}).*?sold.*?\$?({_NUM}).*?(profit|loss)",
        lower,
    )
    if not match:
        return None
    bought, sold = (parse_number(part) for part in match.groups()[:2])
    return format_decimal(abs(sold - bought))


def _unit_conversion(text: str) -> str | None:
    lower = text.lower()
    match = re.search(
        rf"convert\s+({_NUM})\s+(kilometers?|km|meters?|m|centimeters?|cm)\s+to\s+(kilometers?|km|meters?|m|centimeters?|cm)",
        lower,
    )
    if not match:
        return None
    value_raw, source, target = match.groups()
    meters = parse_number(value_raw) * _to_meters(source)
    return format_decimal(meters / _to_meters(target))


def _to_meters(unit: str) -> Decimal:
    if unit in {"kilometer", "kilometers", "km"}:
        return Decimal(1000)
    if unit in {"centimeter", "centimeters", "cm"}:
        return Decimal("0.01")
    return Decimal(1)


def _self_check() -> None:
    assert solve("A car travels 60 km/h for 3.5 hours. How far does it travel?") == "210"
    assert solve("Jane bought 3 books at $12 each and two magazines at $5 each. How much did she spend in total?") == "46"
    assert solve("If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?") == "40"
    assert solve("A $100 jacket has a 20% discount. What is the sale price?") == "80"
    assert solve("Calculate the original price: a jacket costs $150 after a 25% discount.") == "200"
    assert solve("A shirt is $80 after a 20% discount. How much was the original price?") == "100"
    assert solve("Convert 2 kilometers to meters.") == "2000"
    assert solve("A shop sells a bundle with unknown taxes. What is the total?") is None


if __name__ == "__main__":
    _self_check()
    print("wordmath solver self-check passed")

