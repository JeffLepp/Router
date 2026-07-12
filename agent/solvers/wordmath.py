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
        _speed_from_distance,
        _item_total,
        _priced_items,
        _rectangle_area,
        _dimensions_area,
        _had_gave_found,
        _reverse_percent,
        _discount_price,
        _profit_loss,
        _unit_conversion,
    ):
        answer = handler(text)
        if answer is not None:
            return answer
    return None


def solve_strict(prompt: str) -> str | None:
    """Handle only explicit metric conversions with fixed, exact scale factors."""
    return _unit_conversion(normalize(prompt))


def _digits(text: str) -> set[str]:
    return set(re.findall(_NUM, text))


def _rate_times_time(text: str) -> str | None:
    match = re.search(
        rf"(?:travel|keep|maintain|ride|cycle|walk|run|drive|move|go|fly|pedal)s?\w*\s+"
        rf"(?:at\s+)?(?:a\s+)?(?:steady|constant)?\s*({_NUM})\s*"
        rf"(?:km/h|kph|mph|kilometers per hour|miles per hour|meters per second|m/s)"
        rf".*?for\s+({_NUM})\s*(?:hours?|hrs?|h)\b",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return None
    # a third number means a multi-leg problem this template does not model
    if _digits(text) - set(match.groups()):
        return None
    lower = text.lower()
    # the template computes distance; a speed question shares the same surface form
    if re.search(r"\baverage speed\b|\bhow fast\b|\bwhat speed\b", lower):
        return None
    # the answer unit is the rate's distance unit; a question asking for another unit
    # ("how many meters/miles") needs a conversion this template does not do
    asked = re.search(r"how many (\w+)", lower)
    km_rate = bool(re.search(r"km/h|kph|kilometers per hour", match.group(0), re.I))
    allowed = {"kilometers", "km"} if km_rate else {"miles"}
    if asked and asked.group(1) not in allowed:
        return None
    rate, duration = (parse_number(part) for part in match.groups())
    return format_decimal(rate * duration)


def _speed_from_distance(text: str) -> str | None:
    lower = text.lower()
    if not re.search(r"per hour|km/h|kph|mph|how fast|speed", lower):
        return None
    match = re.search(
        rf"(?:cover|travel|go|goe|drive|fly|run)s?\w*\s+({_NUM})\s*"
        rf"(?:km|kilometers?|miles?|meters?)\s+in\s+({_NUM})\s*(?:hours?|hrs?)\b",
        lower,
    )
    if not match:
        return None
    if _digits(lower) - set(match.groups()):
        return None
    # the template answers in distance-unit-per-hour; other asked units need conversion
    if re.search(r"m/s|meters per second", lower):
        return None
    if "mile" in lower and re.search(r"\bkm\b|kilometer", lower):
        return None
    distance, duration = (parse_number(part) for part in match.groups())
    if duration == 0:
        return None
    return format_decimal(distance / duration)


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
    # an unconsumed number (a bill paid with, a coupon, a fee) changes the true answer
    used = {value for pair in parts for value in pair if re.fullmatch(_NUM, value)}
    if _digits(text) - used:
        return None
    total = Decimal(0)
    for count_raw, price_raw in parts:
        total += parse_number(count_raw) * parse_number(price_raw)
    return format_decimal(total)


def _rectangle_area(text: str) -> str | None:
    lower = text.lower()
    if "rectangle" not in lower or "area" not in lower:
        return None
    # comparisons introduce a second shape whose area is the real question
    if re.search(r"\b(second|another|other|combined|twice|double|halved?|half)\b", lower):
        return None
    length = re.search(rf"length\s+of\s+({_NUM})", lower)
    width = re.search(rf"width\s+of\s+({_NUM})", lower)
    if not length or not width:
        return None
    if _digits(lower) - {length.group(1), width.group(1)}:
        return None
    return format_decimal(parse_number(length.group(1)) * parse_number(width.group(1)))


_DIM_UNIT = r"(?:meters?|metres?|m|centimeters?|cm|feet|foot|ft|inches|in|yards?|yd)"


def _dimensions_area(text: str) -> str | None:
    lower = text.lower()
    if "perimeter" in lower or not re.search(r"\bsquare\s+\w+|\barea\b", lower):
        return None
    match = re.search(
        rf"(?:measures?|is)\s+({_NUM})\s*{_DIM_UNIT}?\s*(?:by|x)\s+({_NUM})\s*{_DIM_UNIT}\b",
        lower,
    )
    if not match:
        return None
    if _digits(lower) - set(match.groups()):
        return None
    length, width = (parse_number(part) for part in match.groups())
    return format_decimal(length * width)


def _had_gave_found(text: str) -> str | None:
    lower = text.lower()
    if "how many" not in lower:
        return None
    # the template computes the subject's remaining count; the question must ask for it,
    # not for a transfer amount ("how many did her brother receive?")
    if not re.search(r"how many[^?]*\b(now|left|remain\w*|altogether|in all|in total|have)\b", lower):
        return None
    match = re.search(
        rf"(?:had|started with)\s+({_NUM})\s+\w+.*?"
        rf"(?:gave(?:\s+away)?|lost|used|spent|donated|sold)\s+({_NUM})\b.*?"
        rf"(?:found|bought|received|got|earned|picked up)\s+({_NUM})\b",
        lower,
        flags=re.S,
    )
    if not match:
        return None
    if _digits(lower) - set(match.groups()):
        return None
    start, removed, added = (parse_number(part) for part in match.groups())
    return format_decimal(start - removed + added)


def _priced_items(text: str) -> str | None:
    lower = text.lower()
    if not re.search(r"\b(altogether|in total|total|pay|spend|spent|cost)\b", lower):
        return None
    prices = {
        name: parse_number(price)
        for name, price in re.findall(
            rf"\b(\w+)\s+tickets?\s+(?:are|cost|is)\s+\$?({_NUM})", lower
        )
    }
    purchases = re.findall(rf"\b({_COUNT})\s+(\w+)\s+tickets?\b", lower)
    if not prices or not purchases:
        return None
    total = Decimal(0)
    seen = set(prices.values())
    for count_raw, name in purchases:
        if name not in prices:
            return None
        count = parse_number(count_raw)
        seen.add(count)
        total += count * prices[name]
    # every number in the prompt must be a price or a quantity we used
    if {parse_number(raw) for raw in _digits(lower)} - seen:
        return None
    return format_decimal(total)


_ORIGINAL_CUE = re.compile(r"\b(originally|original|before)\b", re.I)


def _reverse_percent(text: str) -> str | None:
    lower = text.lower()
    if not _ORIGINAL_CUE.search(lower):
        return None
    # a per-item or combined-total framing means the stated price is not the single
    # discounted price this template inverts
    if re.search(r"\b(each|apiece|together|combined|per\b)\b", lower):
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
    if re.search(r"\b(tax|fee|shipping|surcharge)\w*\b", lower):
        return None  # a post-discount add-on changes the final amount
    match = re.search(
        rf"\$?({_NUM}).*?({_NUM})\s*(?:%|percent)\s+discount",
        lower,
    )
    if not match:
        return None
    if _digits(lower) - set(match.groups()):
        return None
    base, pct = (parse_number(part) for part in match.groups())
    return format_decimal(base * (Decimal(1) - pct / Decimal(100)))


def _profit_loss(text: str) -> str | None:
    lower = text.lower()
    # the template computes the absolute difference, not a percentage or margin
    if re.search(r"%|percent|margin", lower):
        return None
    match = re.search(
        rf"bought.*?\$?({_NUM}).*?sold.*?\$?({_NUM}).*?(profit|loss)",
        lower,
    )
    if not match:
        return None
    # extra numbers mean quantities/fees this simple difference does not model
    if _digits(lower) - set(match.groups()[:2]):
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
    assert solve("A cyclist keeps a steady 14 km/h for 2.5 hours. How many kilometers does she cover?") == "35"
    assert solve("A train covers 240 km in 4 hours at a constant speed. How many kilometers per hour is it traveling?") == "60"
    assert solve("A garden bed measures 9 meters by 6 meters. How many square meters does it cover?") == "54"
    assert solve("A rectangular rug measures 6 meters by 7 meters. How many square meters of floor does it cover?") == "42"
    assert solve("A rectangle measures 6 meters by 7 meters and has a perimeter of 26. What is its area?") is None
    assert solve(
        "Nina had 45 stickers, gave 17 to her brother, and later found 8 more in a drawer. "
        "How many stickers does she have now?"
    ) == "36"
    assert solve(
        "Adult tickets are $18 and child tickets are $9. The Reyes family buys 2 adult tickets "
        "and 3 child tickets. How many dollars do they pay altogether?"
    ) == "63"
    assert solve("Tickets are $18 for the gala with a 10 percent fee. Buy 2 group tickets?") is None
    assert solve_strict("Convert 2 kilometers to meters.") == "2000"
    assert solve_strict("A car travels 60 km/h for 3 hours. How far does it travel?") is None


if __name__ == "__main__":
    _self_check()
    print("wordmath solver self-check passed")
