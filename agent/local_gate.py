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
from agent.solvers.common import compact_json
from agent.solvers.sentiment_solve import solve as lexicon_sentiment
from agent.verify import code_v, format_v, logic_v, math_v


_SKIP_WITHOUT_GENERATION = {"summarization"}
_LABELS = {"positive", "negative", "neutral", "mixed"}
_ORDER_NAME = r"[A-Z][A-Za-z0-9_]*"


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

    cap = float(cfg.get("latency_cap_s", 4) or 4)
    if _deadline_tight(deadline, remaining_tasks, cap, now or time.monotonic, parallelism):
        return None

    if client is None:
        client = await make_local_client(llama_config or {}, cfg)

    k = max(1, int(cfg.get("self_consistency_k", 2) or 2))
    generation_k = k if _uses_self_consistency(local_category) else 1
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
            max_tokens=int(cfg.get("max_tokens", 64) or 64),
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
        "formatting": "Return only the requested formatted answer.",
    }
    return f"{instructions.get(category, 'Return the final answer only')}\nTASK:\n{prompt.strip()}"


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


def _accept_ner(prompt: str, candidate: str) -> str | None:
    entities = _parse_entities(candidate)
    if entities is None:
        return None
    if not entities and not re.search(r"\bno\s+(?:entities|names|dates)\b", prompt, re.I):
        return None
    spans = [entity["text"] for entity in entities]
    if not format_v.entities_are_substrings(prompt, spans):
        return None
    payload = compact_json({"entities": entities})
    return payload if format_v.valid_entities_shape(payload) else None


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
    lexicon = _lexicon_label(prompt)
    if lexicon is not None:
        return compact_json({"sentiment": candidate}) if candidate == lexicon else None
    if len(labels) < expected_k or any(label != candidate for label in labels[:expected_k]):
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


if __name__ == "__main__":
    _self_check()
    print("local_gate self-check passed")
