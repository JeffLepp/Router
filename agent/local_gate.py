from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent.classify import canonical_category
from agent.config import AgentConfig
from agent.local_llm import LocalResult, make_local_client, normalize_sample
from agent.solvers.common import compact_json, normalize
from agent.solvers.ner_solve import (
    EVENT_PATTERNS,
    LOCATION_NAMES,
    MONTHS,
    ORG_NAMES,
    _looks_like_person,
)
from agent.solvers.sentiment_solve import solve as lexicon_sentiment
from agent.verify import code_v, format_v, logic_v, math_v


# Categories that skip local generation entirely (no reliable accept-gate). Summarization now
# has _accept_summary, so it is no longer skipped here — but it stays OFF via config until its
# quality/latency on the 1.5B is confirmed by measurement (config.track2.yaml summarization:false).
_SKIP_WITHOUT_GENERATION: set[str] = set()
_LABELS = {"positive", "negative", "neutral", "mixed"}
_ORDER_NAME = r"[A-Z][A-Za-z0-9_]*"
_MONEY_RE = re.compile(r"^\$\d[\d,]*(?:\.\d+)?(?:\s+(?:million|billion|thousand))?$", re.I)
_DATE_RE = re.compile(rf"^(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}$|^(?:{MONTHS})\s+\d{{4}}$")
_ADDRESS_RE = re.compile(
    r"^\d+\s+[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)$"
)


async def try_local(
    category: str,
    prompt: str,
    config: dict[str, Any] | None = None,
    llama_config: dict[str, Any] | None = None,
    client: Any | None = None,
    deadline: float | None = None,
    remaining_tasks: int = 1,
    now: Callable[[], float] | None = None,
    classification_confidence: float = 1.0,
    parallelism: int = 1,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Return a verified local answer, or None to defer to Fireworks."""

    if config is None:
        loaded = AgentConfig.from_path()
        config = loaded.local_candidate
        llama_config = loaded.llama if llama_config is None else llama_config
    cfg = dict(config or {})
    local_category = _local_category(category)

    if not _enabled_for(local_category, category, cfg):
        return None
    if local_category in _SKIP_WITHOUT_GENERATION:
        return None
    if _must_skip_before_generation(local_category, prompt):
        return None

    cap_key = "summary_task_timeout_s" if local_category == "summarization" else "latency_cap_s"
    cap = float(cfg.get(cap_key, cfg.get("latency_cap_s", 4)) or 4)
    if _deadline_tight(deadline, remaining_tasks, cap, now or time.monotonic, parallelism):
        return None

    if client is None:
        client = await make_local_client(llama_config or {}, cfg)

    k = max(1, int(cfg.get("self_consistency_k", 2) or 2))
    generation_k = k if _uses_self_consistency(local_category) else 1
    generation_started = time.monotonic()
    result = await _generate(
        client,
        _local_prompt(local_category, prompt),
        local_category,
        generation_k,
        cfg,
        cap,
    )
    if result is None:
        return None
    if diagnostics is not None:
        diagnostics["candidate"] = result.text

    if local_category == "math_reasoning":
        return _accept_math(prompt, result.text)
    if local_category in {"code_debugging", "code_generation"}:
        return _accept_code(prompt, result.text)
    if local_category == "formatting":
        return _accept_format(prompt, result.text)
    if local_category == "named_entity_recognition":
        return _accept_ner(prompt, result.text)
    if local_category == "logic_puzzles":
        return _accept_logic(prompt, result.samples)
    if local_category == "actual_qa":
        return _accept_factual(prompt, result.samples, cfg, classification_confidence, k)
    if local_category == "sentiment_analysis":
        return _accept_sentiment(prompt, result.samples, k)
    if local_category == "summarization":
        accepted = _accept_summary(prompt, result.text)
        retries = max(0, int(cfg.get("summary_retries", 0) or 0))
        if accepted is None and retries:
            remaining = cap - (time.monotonic() - generation_started)
            if remaining > 0.25:
                retry = await _generate(
                    client,
                    _summary_retry_prompt(prompt, result.text),
                    local_category,
                    1,
                    cfg,
                    remaining,
                )
                if retry is not None:
                    if diagnostics is not None:
                        diagnostics["candidate"] = retry.text
                        diagnostics["first_candidate"] = result.text
                    accepted = _accept_summary(prompt, retry.text)
        return accepted
    return None


def _local_category(category: str) -> str:
    lowered = category.strip().lower()
    if lowered in {"wordmath", "word_math", "word math"}:
        return "math_reasoning"
    if lowered in {"formatting", "format"}:
        return "formatting"
    if lowered == "long-context" or lowered == "long_context":
        return "summarization"
    return canonical_category(lowered)


def _enabled_for(local_category: str, raw_category: str, cfg: dict[str, Any]) -> bool:
    if not bool(cfg.get("enabled", False)):
        return False
    categories = cfg.get("categories", {}) or {}
    if not isinstance(categories, dict):
        return False
    keys = {
        local_category,
        raw_category,
        raw_category.strip().lower(),
        _legacy_key(local_category),
    }
    return any(bool(categories.get(key, False)) for key in keys)


def _legacy_key(category: str) -> str:
    return {
        "actual_qa": "factual",
        "math_reasoning": "math",
        "sentiment_analysis": "sentiment",
        "named_entity_recognition": "ner",
        "code_debugging": "code_debug",
        "logic_puzzles": "logic",
        "code_generation": "code_gen",
    }.get(category, category)


def _must_skip_before_generation(category: str, prompt: str) -> bool:
    if category in {"code_debugging", "code_generation"}:
        return not _extract_assertions(prompt)
    if category == "logic_puzzles":
        return not _extract_order_edges(prompt)
    if category == "summarization" and re.search(
        r"\b(?:without|do\s+not|don't)\s+(?:mention(?:ing)?\s+)?(?:its\s+)?main\s+conclusion\b",
        prompt,
        re.I,
    ):
        # A small local model cannot prove which proposition is the main conclusion. Accuracy
        # wins over one saved call, so this exclusion is enforced by deferring before generation.
        return True
    return False


def _deadline_tight(
    deadline: float | None,
    remaining_tasks: int,
    latency_cap_s: float,
    now: Callable[[], float],
    parallelism: int,
) -> bool:
    if deadline is None:
        return False
    remaining = deadline - now()
    if remaining <= 0:
        return True
    waves = max(1, (max(1, remaining_tasks) + max(1, parallelism) - 1) // max(1, parallelism))
    required = min(max(1.0, latency_cap_s * waves), latency_cap_s * 4) + 1.0
    return remaining <= required


def _uses_self_consistency(category: str) -> bool:
    return category in {"actual_qa", "sentiment_analysis", "logic_puzzles"}


async def _generate(
    client: Any,
    prompt: str,
    category: str,
    k: int,
    cfg: dict[str, Any],
    timeout: float,
) -> LocalResult | None:
    call = _call_generate(client, prompt, category, k, cfg, timeout)
    try:
        result = await asyncio.wait_for(call, timeout=timeout)
    except Exception:
        return None
    if result is None:
        return None
    if isinstance(result, str):
        return LocalResult(result, 0.0, [result])
    if isinstance(result, LocalResult):
        return result
    text = getattr(result, "text", "")
    samples = list(getattr(result, "samples", []) or ([text] if text else []))
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    return LocalResult(str(text), confidence, [str(item) for item in samples])


async def _call_generate(
    client: Any,
    prompt: str,
    category: str,
    k: int,
    cfg: dict[str, Any],
    timeout: float,
) -> Any:
    generate = client.generate
    try:
        maybe = generate(
            prompt,
            category=category,
            k=k,
            temperature=float(cfg.get("temp", 0) or 0),
            max_tokens=_local_max_tokens(category, cfg, prompt),
            timeout=timeout,
        )
    except TypeError:
        maybe = generate(prompt, category, k)
    if isinstance(maybe, Awaitable):
        return await maybe
    return maybe


def _local_prompt(category: str, prompt: str) -> str:
    instructions = {
        "actual_qa": "Answer with one short factual answer. No preamble.",
        "math_reasoning": "Return the final numeric answer only.",
        "sentiment_analysis": "Return exactly one label: positive, negative, neutral, or mixed.",
        "named_entity_recognition": 'Return JSON only: {"entities":[{"text":"...","type":"..."}]}.',
        "code_debugging": "Return corrected Python code only.",
        "code_generation": "Return Python code only.",
        "logic_puzzles": "Return the final assignment or ordering only.",
        "summarization": (
            "Rewrite the source concisely. Preserve every source name, number, date, and action "
            "assignee unless the task explicitly excludes it. Introduce no new facts. Obey the "
            "requested sentence or bullet count exactly. Output only the summary."
        ),
        "formatting": "Return only the requested formatted answer.",
    }
    return f"{instructions.get(category, 'Return the final answer only')}\nTASK:\n{prompt.strip()}"


def _local_max_tokens(category: str, cfg: dict[str, Any], prompt: str = "") -> int:
    if category == "summarization":
        caps = cfg.get("summary_max_tokens", {}) or {}
        if not isinstance(caps, dict):
            return int(caps or 150)
        lowered = prompt.lower()
        if re.search(r"\b(?:one|1)\s+sentence\b", lowered):
            key = "one_sentence"
        elif re.search(r"\b(?:three|3)\s+sentences?\b", lowered):
            key = "three_sentences"
        elif re.search(r"\b(?:bullets?|action\s+items?|structured)\b", lowered):
            key = "structured_report"
        else:
            key = "default"
        defaults = {
            "one_sentence": 64,
            "three_sentences": 110,
            "structured_report": 150,
            "default": 150,
        }
        return int(caps.get(key, defaults[key]) or defaults[key])
    return int(cfg.get("max_tokens", 64) or 64)


def _accept_math(prompt: str, candidate: str) -> str | None:
    expr = _extract_math_expression(prompt)
    if not expr:
        return None
    try:
        if math_v.recompute_matches(candidate, expr):
            return candidate.strip()
    except Exception:
        return None
    return None


def _extract_math_expression(prompt: str) -> str | None:
    text = prompt.replace("×", "*").replace("÷", "/")
    direct = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:\s*(?:[+*/-]|x)\s*[-+]?\d+(?:\.\d+)?)+",
        text,
        flags=re.I,
    )
    if direct:
        return direct[-1].replace("x", "*").replace("X", "*")
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+([-+]?\d+(?:\.\d+)?)", text, re.I)
    if match:
        pct, base = match.groups()
        return f"({pct}/100)*{base}"
    match = re.search(r"square of\s+([-+]?\d+(?:\.\d+)?)", text, re.I)
    if match:
        value = match.group(1)
        return f"{value}*{value}"
    if re.search(r"\b(sum|total|altogether)\b", text, re.I):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            return "+".join(nums)
    if "average" in text.lower():
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            return "(" + "+".join(nums) + f")/{len(nums)}"
    return None


def _accept_code(prompt: str, candidate: str) -> str | None:
    examples = _extract_assertions(prompt)
    if not examples:
        return None
    code = _extract_code(candidate)
    if not code_v.syntax_ok(code) or code_v.safety_error(code):
        return None
    return code if code_v.run_inline_examples(code, examples, timeout=1.0) else None


def _extract_assertions(prompt: str) -> list[str]:
    examples: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            examples.append(stripped)
    for block in re.findall(r"```(?:python|py)?\s*\n?(.*?)```", prompt, flags=re.I | re.S):
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert "):
                examples.append(stripped)
    return list(dict.fromkeys(examples))


def _extract_code(candidate: str) -> str:
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", candidate, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return candidate.strip()


def _accept_format(prompt: str, candidate: str) -> str | None:
    lowered = prompt.lower()
    if "json" in lowered and not format_v.valid_json_shape(candidate):
        return None
    match = re.search(r"(?:under|at most|no more than)\s+(\d+)\s+words?", lowered)
    if match and not format_v.within_word_limit(candidate, int(match.group(1))):
        return None
    return candidate.strip() if candidate.strip() and format_v.englishish(candidate) else None


def _accept_summary(prompt: str, candidate: str) -> str | None:
    """Accept only summaries whose important surface facts can be checked against source."""
    text = candidate.strip()
    if not text or not format_v.englishish(text):
        return None
    if "```" in text or re.search(r"\[[^\]]*\]", text):
        return None
    if re.match(r"^(?:summary|answer|response)\s*:", text, re.I):
        return None
    if re.search(
        r"\b(?:previous\s+draft|omitted\s+(?:source\s+)?facts?|all\s+sources?|"
        r"source\s+facts?|prior\s+drafts?)\b",
        text,
        re.I,
    ):
        return None
    lowered = prompt.lower()
    if re.search(
        r"\b(?:without|do\s+not|don't)\s+(?:mention(?:ing)?\s+)?(?:its\s+)?main\s+conclusion\b",
        lowered,
    ):
        return None
    match = re.search(
        r"(?:under|at most|no more than|within|fewer than|less than)\s+(\d+)\s+words?", lowered
    )
    if match and not format_v.within_word_limit(text, int(match.group(1))):
        return None
    words = format_v.word_count(text)
    if words < 3 or words > 200:
        return None
    source = _summary_source(prompt)
    if not source or format_v.word_count(source) < 5:
        return None

    if not _summary_shape_matches(prompt, text):
        return None
    if _raw_passage_copy(source, text):
        return None
    if not _source_sentences_covered(source, text):
        return None
    source_terms = _content_terms(source)
    candidate_terms = _content_terms(text)
    if candidate_terms and len(source_terms & candidate_terms) / len(candidate_terms) < 0.65:
        return None

    date_excluded = bool(
        re.search(r"\b(?:do\s+not|don't|without)\s+mention(?:ing)?\s+(?:the\s+)?date\b", lowered)
    )
    fact_source = source
    if date_excluded:
        if _date_spans(text):
            return None
        for date in _date_spans(source):
            fact_source = fact_source.replace(date, " ")

    source_numbers = _number_facts(fact_source)
    candidate_numbers = _number_facts(text)
    if not source_numbers.issubset(candidate_numbers):
        return None
    if not candidate_numbers.issubset(_number_facts(source)):
        return None

    source_names = _proper_name_tokens(fact_source)
    candidate_names = _proper_name_tokens(text, known=source_names)
    if not source_names.issubset(candidate_names):
        return None
    if not candidate_names.issubset(source_names):
        return None

    assignees = _summary_action_assignees(source)
    if any(not re.search(rf"\b{re.escape(name)}\b", text, re.I) for name in assignees):
        return None
    return text


def _summary_source(prompt: str) -> str | None:
    normalized = normalize(prompt).strip()
    _instruction, sep, body = normalized.partition(":")
    if not sep:
        return None
    body = body.strip()
    if len(body) >= 2 and body[0] in {'"', "'"} and body[-1] == body[0]:
        body = body[1:-1].strip()
    return body if body else None


def _summary_shape_matches(prompt: str, candidate: str) -> bool:
    lowered = prompt.lower()
    bullet_match = re.search(r"\b(?:in|into)\s+(\d+)\s+bullets?\b", lowered)
    if bullet_match:
        bullets = re.findall(r"(?m)^\s*[-*]\s+\S.+$", candidate)
        if len(bullets) != int(bullet_match.group(1)):
            return False
    elif re.search(r"\b(?:one|1)\s+sentence\b", lowered):
        if _sentence_count(candidate) != 1:
            return False
    elif re.search(r"\b(?:three|3)\s+sentences?\b", lowered):
        if _sentence_count(candidate) != 3:
            return False
    if "action items" in lowered:
        if not re.search(r"(?im)^\s*action\s+items\s*:\s*$", candidate):
            return False
    return True


def _sentence_count(text: str) -> int:
    if re.search(r"(?m)^\s*[-*]\s+", text):
        return 0
    pieces = re.findall(r"[^.!?]+[.!?](?=\s|$)|[^.!?]+$", text.strip())
    return len([piece for piece in pieces if re.search(r"\w", piece)])


def _number_facts(text: str) -> set[str]:
    return {
        match.replace(",", "").lower()
        for match in re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?", text)
    }


def _date_spans(text: str) -> list[str]:
    month = MONTHS
    return re.findall(
        rf"\b(?:{month})\s+\d{{1,2}}(?:,\s*|\s+)\d{{4}}\b|"
        rf"\b(?:{month})\s+\d{{4}}\b|\b\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}\b",
        text,
        re.I,
    )


_NAME_STOP = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "it",
    "summary",
    "action",
    "items",
    "report",
    "study",
    "researchers",
}


def _proper_name_tokens(text: str, known: set[str] | None = None) -> set[str]:
    known = known or set()
    names: set[str] = set()
    pattern = re.compile(r"\b(?:[A-Z]{3,}|[A-Z][a-z]{2,}|[A-Z][a-z]+[A-Z][A-Za-z]*)\b")
    for match in pattern.finditer(text):
        token = match.group(0)
        lowered = token.casefold()
        if lowered in _NAME_STOP:
            continue
        camel = bool(re.search(r"[a-z][A-Z]", token))
        acronym = token.isupper()
        prefix = text[: match.start()].rstrip()
        sentence_initial = not prefix or prefix[-1] in ".!?"
        if camel or acronym or not sentence_initial or lowered in known:
            names.add(lowered)
    return names


def _summary_action_assignees(source: str) -> set[str]:
    match = re.search(r"\baction\s+items\s*:\s*(.+)$", source, re.I | re.S)
    if not match:
        return set()
    return {
        name
        for name in re.findall(r"\b([A-Z][A-Za-z'-]*)\s+will\b", match.group(1))
    }


def _raw_passage_copy(source: str, candidate: str) -> bool:
    source_words = re.findall(r"[a-z0-9]+", source.lower())
    candidate_words = re.findall(r"[a-z0-9]+", candidate.lower())
    if not source_words or len(candidate_words) < int(len(source_words) * 0.85):
        return False
    source_set = set(source_words)
    overlap = sum(word in source_set for word in candidate_words) / len(candidate_words)
    return overlap >= 0.92


_SUMMARY_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "and", "or", "but", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this", "that", "it",
    "its", "as", "by", "from", "which", "who", "their", "has", "have", "had", "will",
    "would", "can", "could", "some", "other", "various",
}


def _fact_stem(word: str) -> str:
    value = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(value) > len(suffix) + 3 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _content_terms(text: str) -> set[str]:
    return {
        _fact_stem(word)
        for word in re.findall(r"[A-Za-z0-9]+", text)
        if word.lower() not in _SUMMARY_STOP and len(word) > 2
    }


def _source_sentences_covered(source: str, candidate: str) -> bool:
    got = _content_terms(candidate)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", source) if part.strip()]
    for sentence in sentences:
        wanted = _content_terms(sentence)
        if len(wanted) >= 4 and len(wanted & got) / len(wanted) < 0.4:
            return False
    return True


def _summary_retry_prompt(prompt: str, candidate: str) -> str:
    source = _summary_source(prompt) or ""
    fact_source = source
    forbidden: list[str] = []
    if re.search(
        r"\b(?:do\s+not|don't|without)\s+mention(?:ing)?\s+(?:the\s+)?date\b",
        prompt,
        re.I,
    ):
        forbidden = _date_spans(source)
        for date in forbidden:
            fact_source = fact_source.replace(date, " ")
    missing_numbers = sorted(_number_facts(fact_source) - _number_facts(candidate))
    source_names = _proper_name_tokens(fact_source)
    missing_names = sorted(source_names - _proper_name_tokens(candidate, known=source_names))
    missing_words = _missing_fact_words(fact_source, candidate)
    missing = ", ".join(missing_numbers + missing_names + missing_words)
    if not missing:
        missing = "the key facts from each source sentence"
    exclusion = ""
    if forbidden:
        exclusion = (
            " Remove the forbidden date completely—do not output any part of: "
            + ", ".join(forbidden)
            + "."
        )
    return (
        _local_prompt("summarization", prompt)
        + "\nYour previous draft was rejected. Rewrite it once and literally include these missing "
        + f"source facts: {missing}. Keep the required shape and introduce nothing new.{exclusion}\n"
        + f"PREVIOUS DRAFT:\n{candidate.strip()}"
    )


def _missing_fact_words(source: str, candidate: str) -> list[str]:
    candidate_terms = _content_terms(candidate)
    missing: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", source):
        lowered = word.lower()
        stem = _fact_stem(lowered)
        if lowered in _SUMMARY_STOP or len(lowered) <= 2 or stem in candidate_terms or stem in seen:
            continue
        seen.add(stem)
        missing.append(word)
        if len(missing) >= 24:
            break
    return missing


def _accept_ner(prompt: str, candidate: str) -> str | None:
    entities = _parse_entities(candidate)
    if entities is None:
        return None
    if not entities and not re.search(r"\bno\s+(?:entities|names|dates)\b", prompt, re.I):
        return None
    spans = [entity["text"] for entity in entities]
    if not format_v.entities_are_substrings(prompt, spans):
        return None
    # F6: every emitted (text, type) must be corroborated, or we can't trust the label.
    if not all(_ner_type_corroborated(e["text"], e["type"]) for e in entities):
        return None
    payload = compact_json({"entities": entities})
    return payload if format_v.valid_entities_shape(payload) else None


def _ner_type_corroborated(text: str, type_: str) -> bool:
    """Only trust a (text, type) pair the deterministic layer would also emit.

    Open-class types with no gazetteer/surface evidence are NOT corroborated -> defer.
    ponytail: with no real NER model, this ceilings local NER at gate-NER trust; that is
    the point. Flipping named_entity_recognition on adds ~0 recall but stays precision-safe.
    Upgrade path: a trusted NER model would replace this with its own confidence gate.
    """
    t = type_.strip().upper()
    if t in {"PERSON", "CUSTOMER"}:
        return _looks_like_person(text)
    if t == "ORGANIZATION":
        return text in ORG_NAMES
    if t == "LOCATION":
        return text in LOCATION_NAMES
    if t == "MONEY":
        return bool(_MONEY_RE.match(text))
    if t == "DATE":
        return bool(_DATE_RE.match(text))
    if t == "MLS_NUMBER":
        return text.isdigit()
    if t == "PROPERTY_ADDRESS":
        return bool(_ADDRESS_RE.match(text))
    if t == "EVENT":
        return any(re.fullmatch(pattern, text) for pattern in EVENT_PATTERNS)
    return False  # unknown type -> cannot corroborate -> defer


def _parse_entities(candidate: str) -> list[dict[str, str]] | None:
    text = candidate.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("entities"), list):
        entities = data["entities"]
        if all(isinstance(item, dict) and isinstance(item.get("text"), str) and isinstance(item.get("type"), str) for item in entities):
            return [{"text": item["text"], "type": item["type"]} for item in entities]
        return None
    if isinstance(data, list):
        if all(isinstance(item, dict) and isinstance(item.get("text"), str) and isinstance(item.get("type"), str) for item in data):
            return [{"text": item["text"], "type": item["type"]} for item in data]
        return None

    entities: list[dict[str, str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        name, kind = (part.strip() for part in line.split("|", 1))
        if name and kind:
            entities.append({"text": name, "type": kind})
    return entities if entities else None


def _accept_factual(
    prompt: str,
    samples: list[str],
    cfg: dict[str, Any],
    classification_confidence: float,
    expected_k: int,
) -> str | None:
    min_classifier = float(cfg.get("min_classifier_confidence", 0.5) or 0.5)
    if classification_confidence < min_classifier:
        return None
    nonempty = [sample.strip() for sample in samples if sample.strip()]
    if len(nonempty) < expected_k:
        return None
    if len({normalize_sample(sample) for sample in nonempty[:expected_k]}) != 1:
        return None
    candidate = nonempty[0]
    if not format_v.englishish(candidate):
        return None
    if format_v.word_count(candidate) > int(cfg.get("factual_max_words", 80) or 80):
        return None
    return candidate


def _accept_sentiment(prompt: str, samples: list[str], expected_k: int) -> str | None:
    labels = [_extract_label(sample) for sample in samples if sample.strip()]
    if not labels or labels[0] is None:
        return None
    candidate = labels[0]
    # Self-consistency is always required; the lexicon can only veto.
    if len(labels) < expected_k or any(label != candidate for label in labels[:expected_k]):
        return None
    lexicon = _lexicon_label(prompt)
    if lexicon is not None and candidate != lexicon:
        return None
    return compact_json({"sentiment": candidate})


def _extract_label(text: str) -> str | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("sentiment"), str):
            label = data["sentiment"].strip().lower()
            return label if label in _LABELS else None
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    hits = [label for label in _LABELS if re.search(rf"\b{label}\b", lowered)]
    return hits[0] if len(hits) == 1 else None


def _lexicon_label(prompt: str) -> str | None:
    answer = lexicon_sentiment(prompt)
    if not answer:
        return None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return None
    label = data.get("sentiment") if isinstance(data, dict) else None
    return label if isinstance(label, str) and label in _LABELS else None


def _accept_logic(prompt: str, samples: list[str]) -> str | None:
    edges = _extract_order_edges(prompt)
    if not edges or not logic_v.constraints_consistent(edges):
        return None
    nodes = sorted({node for edge in edges for node in edge})
    if len(nodes) > 7:
        return None
    solutions = []
    for perm in itertools.permutations(nodes):
        pos = {node: index for index, node in enumerate(perm)}
        if all(pos[before] < pos[after] for before, after in edges):
            solutions.append(perm)
            if len(solutions) > 1:
                return None
    if len(solutions) != 1:
        return None
    candidate_order = _order_from_answer(samples[0], nodes) if samples else None
    if candidate_order != solutions[0]:
        return None
    if len(samples) > 1 and logic_v.agreement_score(samples) < 1.0:
        return None
    return samples[0].strip()


def _extract_order_edges(prompt: str) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    patterns = [
        (rf"\b({_ORDER_NAME})\s+is\s+before\s+({_ORDER_NAME})\b", False),
        (rf"\b({_ORDER_NAME})\s+before\s+({_ORDER_NAME})\b", False),
        (rf"\b({_ORDER_NAME})\s+is\s+after\s+({_ORDER_NAME})\b", True),
        (rf"\b({_ORDER_NAME})\s+after\s+({_ORDER_NAME})\b", True),
        (rf"\b({_ORDER_NAME})\s+is\s+older\s+than\s+({_ORDER_NAME})\b", False),
        (rf"\b({_ORDER_NAME})\s*<\s*({_ORDER_NAME})\b", False),
    ]
    for pattern, reverse in patterns:
        for left, right in re.findall(pattern, prompt):
            edge = (right, left) if reverse else (left, right)
            if edge[0] != edge[1] and edge not in edges:
                edges.append(edge)
    return edges


def _order_from_answer(answer: str, nodes: list[str]) -> tuple[str, ...] | None:
    positions: list[tuple[int, str]] = []
    for node in nodes:
        match = re.search(rf"\b{re.escape(node)}\b", answer)
        if not match:
            return None
        positions.append((match.start(), node))
    ordered = tuple(node for _pos, node in sorted(positions))
    return ordered if len(set(ordered)) == len(nodes) else None


def _self_check() -> None:
    class StaticClient:
        def __init__(self, *answers: str) -> None:
            self.answers = list(answers)
            self.calls = 0

        async def generate(self, prompt: str, **kwargs: Any) -> LocalResult:
            self.calls += 1
            k = int(kwargs.get("k", 1))
            samples = self.answers[:k] or self.answers
            return LocalResult(samples[0], 0.9, samples)

    cfg = {"enabled": True, "categories": {"math_reasoning": True}, "latency_cap_s": 1}
    accepted = asyncio.run(
        try_local("math_reasoning", "Calculate 2 + 3.", cfg, client=StaticClient("5"))
    )
    assert accepted == "5"
    rejected = asyncio.run(
        try_local("math_reasoning", "Calculate 2 + 3.", cfg, client=StaticClient("6"))
    )
    assert rejected is None

    # summarization accept-gate: clean English summary passes; code/[brackets]/empty/over-limit defer
    assert _accept_summary(
        "Summarize in one sentence: A solar eclipse happens when the Moon blocks the Sun.",
        "At new moon, the Moon blocks the Sun in a solar eclipse.",
    )
    assert _accept_summary("Summarize", "```python\nprint(1)\n```") is None
    assert _accept_summary("Summarize", "See [placeholder] for details here.") is None
    assert _accept_summary("Summarize", "") is None
    assert _accept_summary("Summarize in under 5 words", "This summary is clearly far too long") is None
    assert _local_max_tokens("summarization", {}, "Summarize in one sentence: x") == 64
    assert _local_max_tokens("summarization", {}, "Create a report in three sentences: x") == 110
    assert _local_max_tokens("summarization", {}, "Summarize in five bullets: x") == 150
    assert _local_max_tokens("actual_qa", {"max_tokens": 48}) == 48


if __name__ == "__main__":
    _self_check()
    print("local_gate self-check passed")
