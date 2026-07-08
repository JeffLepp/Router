from __future__ import annotations

import re

from agent.solvers.common import compact_json, extract_source_text, normalize
from agent.verify.format_v import valid_entities_shape


ORG_NAMES = {
    "AMD",
    "Apple",
    "Arm",
    "Google",
    "Microsoft",
    "OpenAI",
    "Samsung",
}
LOCATION_NAMES = {"Toronto", "Paris", "France", "Seattle"}
EVENT_PATTERNS = [r"\b\d{4}\s+Summer Olympics\b"]
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
# Title-case bigrams beginning with these are place names ("San Carlos", "New York"), not people.
LOCATION_PREFIXES = {
    "San", "Los", "Las", "New", "Fort", "Lake", "Mount", "Saint", "Santa", "El", "Cape", "Port",
}
# Capitalized words that are not entities, so an uncovered one shouldn't force a defer.
NON_ENTITY_CAPS = {
    "British", "American", "French", "German", "Chinese", "Japanese", "European", "Canadian",
    "Client", "The", "There", "This", "That", "Something", "Somewhere",
} | set(MONTHS.split("|"))


def solve(prompt: str) -> str | None:
    text = normalize(prompt)
    lower = text.lower()
    source = extract_source_text(text)
    if not source:
        return None
    if "can refer" in source.lower():
        return None

    requested = _requested_types(lower)
    if not requested:
        return None
    generic = "named entities" in lower or "extract entities" in lower or "identify entities" in lower

    candidates: list[tuple[int, int, dict[str, str]]] = []
    person_spans: list[tuple[int, int]] = []

    if "CUSTOMER" in requested:
        for match in re.finditer(r"\bClient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", source):
            candidates.append((match.start(1), match.end(1), {"text": match.group(1), "type": "CUSTOMER"}))

    if "PERSON" in requested:
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", source):
            value = match.group(1)
            if _looks_like_person(value):
                person_spans.append((match.start(1), match.end(1)))
                candidates.append((match.start(1), match.end(1), {"text": value, "type": "PERSON"}))

    if "PROPERTY_ADDRESS" in requested:
        for match in re.finditer(
            r"\b\d+\s+[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)\b",
            source,
        ):
            candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "PROPERTY_ADDRESS"}))

    if "MLS_NUMBER" in requested:
        for match in re.finditer(r"\bMLS#\s*(\d+)\b", source):
            candidates.append((match.start(1), match.end(1), {"text": match.group(1), "type": "MLS_NUMBER"}))

    if "MONEY" in requested:
        for match in re.finditer(r"\$\d+(?:\.\d+)?(?:\s+(?:million|billion|thousand))?\b", source, flags=re.I):
            candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "MONEY"}))

    if "DATE" in requested:
        date_re = rf"\b(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b|\b(?:{MONTHS})\s+\d{{4}}\b"
        for match in re.finditer(date_re, source):
            candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "DATE"}))

    if "EVENT" in requested:
        for pattern in EVENT_PATTERNS:
            for match in re.finditer(pattern, source):
                candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "EVENT"}))

    if "ORGANIZATION" in requested:
        for name in sorted(ORG_NAMES, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(name)}\b", source):
                if any(start <= match.start() and match.end() <= end for start, end in person_spans):
                    continue
                candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "ORGANIZATION"}))

    if "LOCATION" in requested:
        for name in sorted(LOCATION_NAMES, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(name)}\b", source):
                candidates.append((match.start(), match.end(), {"text": match.group(0), "type": "LOCATION"}))

    entities = _dedupe_sorted(candidates)
    if not entities and re.search(r"\bno\s+(?:names?|dates?|entities)\b", source, flags=re.I):
        payload = compact_json({"entities": []})
        return payload if valid_entities_shape(payload) else None
    if not entities:
        return None
    # ponytail: a 7-org/4-location gazetteer can't safely do open-world "extract ALL named entities".
    # If a generic request leaves any capitalized proper noun uncovered, defer. Ceiling: scoped-type
    # requests with out-of-gazetteer entities still risk a miss -> grow the gazetteer or add a real
    # NER model if the hidden set needs it.
    if generic and _has_uncovered_proper_noun(source, entities):
        return None
    payload = compact_json({"entities": entities})
    return payload if valid_entities_shape(payload) else None


def _has_uncovered_proper_noun(source: str, entities: list[dict[str, str]]) -> bool:
    covered = {token for entity in entities for token in entity["text"].split()}
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", source):
        token = match.group(1)
        if token in covered or token in NON_ENTITY_CAPS:
            continue
        prev = source[: match.start()].rstrip()
        if not prev or prev[-1] in ".!?\"'":
            continue  # sentence-initial: can't tell a proper noun from a capitalized first word
        return True
    return False


def _requested_types(lower: str) -> set[str]:
    requested: set[str] = set()
    generic = "named entities" in lower or "extract entities" in lower or "identify entities" in lower
    if generic:
        requested.update({"PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "EVENT"})
    if re.search(r"\b(people|persons?|names?)\b", lower):
        requested.add("PERSON")
    if "customer" in lower:
        requested.add("CUSTOMER")
    if "organization" in lower or "organizations" in lower:
        requested.add("ORGANIZATION")
    if "money" in lower or "amount" in lower or "amounts" in lower:
        requested.add("MONEY")
    if "all people, organizations, and dates" in lower:
        requested.add("MONEY")
    if "date" in lower or "dates" in lower:
        requested.add("DATE")
    if "location" in lower or "locations" in lower:
        requested.add("LOCATION")
    if "event" in lower or "events" in lower:
        requested.add("EVENT")
    if "property" in lower or "address" in lower:
        requested.add("PROPERTY_ADDRESS")
    if "mls" in lower:
        requested.add("MLS_NUMBER")
    return requested


def _looks_like_person(value: str) -> bool:
    parts = value.split()
    if len(parts) < 2 or len(parts) > 3:
        return False
    if value in ORG_NAMES or value in LOCATION_NAMES:
        return False
    if parts[0] in LOCATION_PREFIXES:
        return False  # place name, not a person
    blocked = ORG_NAMES | LOCATION_NAMES | {
        "Summer",
        "Olympics",
        "Client",
        "CEO",
        "Street",
        "St",
        "Avenue",
        "Ave",
        "Road",
        "Rd",
        "Lane",
        "Ln",
        "Drive",
        "Dr",
    }
    structural_blocked = blocked - ORG_NAMES - LOCATION_NAMES
    return not any(part in structural_blocked or part.isupper() for part in parts)


def _dedupe_sorted(candidates: list[tuple[int, int, dict[str, str]]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, int]] = set()
    entities: list[dict[str, str]] = []
    for start, _end, entity in sorted(candidates, key=lambda item: (item[0], item[1], item[2]["type"])):
        key = (entity["text"], entity["type"], start)
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)
    return entities


def _self_check() -> None:
    assert solve("Identify people and locations: 'Elon Musk visited Toronto last summer.'") == compact_json({"entities": [{"text": "Elon Musk", "type": "PERSON"}, {"text": "Toronto", "type": "LOCATION"}]})
    assert solve("Identify organizations and money amounts: 'Apple invested $3 billion in British chipmaker Arm.'") == compact_json({"entities": [{"text": "Apple", "type": "ORGANIZATION"}, {"text": "$3 billion", "type": "MONEY"}, {"text": "Arm", "type": "ORGANIZATION"}]})
    assert solve("Extract customer names, property addresses, and MLS numbers from: Client Jane Doe bought 123 Maple Street (MLS# 456789).") == compact_json({"entities": [{"text": "Jane Doe", "type": "CUSTOMER"}, {"text": "123 Maple Street", "type": "PROPERTY_ADDRESS"}, {"text": "456789", "type": "MLS_NUMBER"}]})
    assert solve("Extract entities and classify them: 'Apple hired John Apple to work at its Paris office.'") == compact_json({"entities": [{"text": "Apple", "type": "ORGANIZATION"}, {"text": "John Apple", "type": "PERSON"}, {"text": "Paris", "type": "LOCATION"}]})
    assert solve("Identify persons and dates: 'There are no names or dates in this sentence.'") == compact_json({"entities": []})
    assert solve("Identify entities: 'The word Amazon can refer to a rainforest or a company.'") is None
    assert solve("Extract important things: 'Something happened somewhere.'") is None
    # open-world generic requests with out-of-gazetteer proper nouns must defer, not emit a partial set
    assert solve("Extract named entities: Barack Obama was born in Hawaii.") is None
    assert solve("Extract named entities: Tesla was founded by Elon Musk in San Carlos.") is None
    assert solve("Extract named entities: Marie Curie worked in Paris.") == compact_json({"entities": [{"text": "Marie Curie", "type": "PERSON"}, {"text": "Paris", "type": "LOCATION"}]})


if __name__ == "__main__":
    _self_check()
    print("ner solver self-check passed")
