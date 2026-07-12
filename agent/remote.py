from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request

from agent.classify import canonical_category


FAMILY_PRIOR = {
    "minimax": 0,
    "gemma": 1,
    "kimi": 2,
    "qwen": 3,
    "llama": 4,
    "mistral": 5,
    "mixtral": 6,
    "deepseek": 7,
    "gpt-oss": 8,
}
REASONING_RE = re.compile(r"\b(r1|reason(?:ing)?|thinking|think|qwq|gpt-oss)\b", re.I)
# Measured on Fireworks 2026-07-09 with a 1-token user message: minimax-m3 bills 119
# prompt tokens before we send a single word (a provider-side chat template), kimi-k2p7-code
# bills 24. That ~95-token surcharge is paid on EVERY call, so it dominates short-answer
# categories. Keep minimax only where world knowledge or nuance actually earns it back:
#   actual_qa   - kimi answered "unanswerable" for a knowable question (qa_007); minimax knew it.
#   sentiment   - minimax 4/5 vs kimi 3/5 on the non-gate tasks.
# The first two preferences are the only families we can exercise through the development
# proxy. The official judge also advertises Gemma models, but routing to an unmeasured model
# made the submitted image behave differently from every live gate. Keep Gemma as transport
# fallback only, after the validated Minimax/Kimi pair.
#
# Patterns are an ordered preference; unmatched families fall through to rank_models.
CATEGORY_MODEL_PATTERNS = {
    "code_debugging": (r"kimi.*code", r"minimax", r"gemma.*31.*it"),
    "code_generation": (r"kimi.*code", r"minimax", r"gemma.*31.*it"),
    "sentiment_analysis": (r"minimax", r"kimi.*code", r"gemma.*26.*it", r"gemma.*31.*it"),
    "math_reasoning": (r"minimax", r"kimi.*code", r"gemma.*31.*it", r"gemma.*26.*it"),
    "logic_puzzles": (r"kimi.*code", r"minimax", r"gemma.*31.*it"),
    "actual_qa": (r"minimax", r"kimi.*code", r"gemma.*31.*it"),
    "summarization": (r"kimi.*code", r"minimax", r"gemma.*31.*it"),
    "named_entity_recognition": (r"kimi.*code", r"minimax", r"gemma.*31.*it"),
}


@dataclass
class RemoteCall:
    task_id: str
    category: str
    prompt: str
    max_tokens: int
    flip_value: float = 1.0
    mandatory: bool = False
    reasoning_effort: str | None = None

    @property
    def estimated_tokens(self) -> int:
        prompt_tokens = max(1, len(self.prompt.split()))
        return prompt_tokens + self.max_tokens


@dataclass
class TokenLedger:
    usd_per_mtok: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        task_id: str,
        model: str,
        usage: dict[str, Any],
        estimated: int,
        *,
        category: str = "",
        stage: str = "answer",
        reasoning_effort: str = "",
        attempt: int = 0,
    ) -> None:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0) or estimated
        self.total_tokens += total
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.requests += 1
        self.entries.append(
            {
                "task_id": task_id,
                "model": model,
                "total_tokens": total,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "category": category,
                "stage": stage,
                "reasoning_effort": reasoning_effort or "none",
                "attempt": attempt,
            }
        )

    @property
    def estimated_usd(self) -> float:
        return (self.total_tokens / 1_000_000) * self.usd_per_mtok

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "requests": self.requests,
            "estimated_usd": round(self.estimated_usd, 8),
            "entries": self.entries,
        }


def parse_allowed_models(raw: str | None = None) -> list[str]:
    source = os.environ.get("ALLOWED_MODELS", "") if raw is None else raw
    return [part.strip() for part in source.split(",") if part.strip()]


def _param_count(model: str) -> float:
    # MoE IDs advertise active params (gemma-4-26b-a4b -> 4); that tracks
    # per-call cost better than total size.
    match = re.search(r"\ba(\d+(?:\.\d+)?)b\b", model, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"-(\d{1,3})(?:b|B|B-)", model)
    if match:
        return float(match.group(1))
    return 999.0


def rank_models(models: list[str]) -> list[str]:
    def key(model: str) -> tuple[int, int, float, str]:
        lower = model.lower()
        family = min(
            (score for name, score in FAMILY_PRIOR.items() if name in lower),
            default=50,
        )
        reasoning = 1 if REASONING_RE.search(model) else 0
        instruct_bonus = 0 if re.search(r"instruct|chat|it\b", lower) else 1
        return (reasoning, family + instruct_bonus, _param_count(model), model)

    return sorted(models, key=key)


def choose_default_model(models: list[str]) -> str:
    ranked = rank_models(models)
    if not ranked:
        raise RuntimeError("ALLOWED_MODELS is empty but remote calls are enabled")
    return ranked[0]


def choose_model_for_category(category: str, models: list[str]) -> str:
    category = canonical_category(category)
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty but remote calls are enabled")
    non_reasoning = [model for model in models if not REASONING_RE.search(model)]
    candidates = non_reasoning or models
    for pattern in CATEGORY_MODEL_PATTERNS.get(category, ()):
        regex = re.compile(pattern, re.I)
        matches = [model for model in candidates if regex.search(model)]
        if matches:
            return rank_models(matches)[0]
    return choose_default_model(candidates)


def choose_accuracy_model(category: str, models: list[str]) -> str:
    """Prefer reasoning-capable models for tasks where extra compute can change correctness.

    Other categories retain the measured family preferences, but unlike the token-first path
    they do not discard reasoning models before applying those preferences.
    """
    category = canonical_category(category)
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty but remote calls are enabled")
    if category in {"math_reasoning", "logic_puzzles", "code_debugging", "code_generation"}:
        reasoning = [model for model in models if REASONING_RE.search(model)]
        if reasoning:
            return rank_models(reasoning)[0]
    for pattern in CATEGORY_MODEL_PATTERNS.get(category, ()):
        matches = [model for model in models if re.search(pattern, model, re.I)]
        if matches:
            return rank_models(matches)[0]
    return choose_default_model(models)


def select_budgeted_calls(calls: list[RemoteCall], budget: int) -> list[RemoteCall]:
    if budget <= 0:
        return []
    ranked = sorted(
        calls,
        key=lambda call: (call.flip_value / max(1, call.estimated_tokens)),
        reverse=True,
    )
    selected: list[RemoteCall] = []
    spent = 0
    for call in ranked:
        estimate = call.estimated_tokens
        if spent + estimate > budget:
            continue
        selected.append(call)
        spent += estimate
    return selected


class RemoteClient:
    def __init__(
        self,
        timeout: float = 25.0,
        retries: int = 2,
        temperature: float = 0.0,
        accuracy_first: bool = False,
        reasoning_effort: str = "none",
        reasoning_effort_by_category: dict[str, str] | None = None,
        usd_per_mtok: float = 0.0,
        dev_spend_cap: float = 0.0,
    ) -> None:
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.models = parse_allowed_models()
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.accuracy_first = accuracy_first
        self.reasoning_effort = reasoning_effort
        self.reasoning_effort_by_category = {
            canonical_category(str(category)): str(effort)
            for category, effort in (reasoning_effort_by_category or {}).items()
        }
        self.dev_spend_cap = dev_spend_cap
        self.ledger = TokenLedger(usd_per_mtok=usd_per_mtok)
        # Models that rejected `reasoning_effort`; we stop sending it to them.
        self._no_reasoning_param: set[str] = set()
        self._dead_models: set[str] = set()
        if not self.base_url:
            raise RuntimeError("FIREWORKS_BASE_URL is required when remote calls are enabled")
        choose_default_model(self.models)

    def _usable_models(self) -> list[str]:
        alive = [model for model in self.models if model not in self._dead_models]
        return alive or self.models

    def _reasoning_effort(self, model: str, call: RemoteCall) -> str:
        if model in self._no_reasoning_param or call.task_id.startswith("classifier:"):
            return ""
        if call.reasoning_effort is not None:
            return call.reasoning_effort
        return resolve_reasoning_effort(
            call.category,
            call.prompt,
            self.reasoning_effort,
            self.reasoning_effort_by_category,
        )

    def _payload(self, model: str, call: RemoteCall) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer with the smallest decisive payload. English only.",
                },
                {"role": "user", "content": call.prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": call.max_tokens,
            "stop": ["\n\n\n"],
        }
        effort = self._reasoning_effort(model, call)
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    async def complete(self, call: RemoteCall) -> str:
        projected = self.ledger.estimated_usd + (
            call.estimated_tokens / 1_000_000
        ) * self.ledger.usd_per_mtok
        if self.dev_spend_cap > 0 and projected > self.dev_spend_cap:
            raise RuntimeError("remote dev spend cap reached")

        usable = self._usable_models()
        model = (
            choose_accuracy_model(call.category, usable)
            if self.accuracy_first
            else choose_model_for_category(call.category, usable)
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        deadline = time.monotonic() + self.timeout
        last_error: BaseException | None = None
        quality_fallback_used = False
        for attempt in range(self.retries + 1):
            remaining = max(0.5, deadline - time.monotonic())
            try:
                data = await asyncio.wait_for(
                    _post_json(
                        _completion_url(self.base_url),
                        self._payload(model, call),
                        headers,
                        remaining,
                    ),
                    timeout=remaining,
                )
                content = _message_content(data).strip()
                usage = dict(data.get("usage", {}) or {})
                actual_effort = self._reasoning_effort(model, call)
                self.ledger.add(
                    call.task_id,
                    model,
                    usage,
                    call.estimated_tokens,
                    category=canonical_category(call.category),
                    stage="classifier" if call.task_id.startswith("classifier:") else "answer",
                    reasoning_effort=actual_effort,
                    attempt=attempt,
                )
                # A blank or structurally truncated completion scores zero (code models can
                # burn most of the budget in hidden reasoning), so spend one retry on the
                # next validated model before returning it.
                incomplete = _code_payload_incomplete(call, content)
                fallback = _other_model(
                    model, self._usable_models(), call.category, self.accuracy_first
                )
                if (
                    (content and not incomplete)
                    or quality_fallback_used
                    or fallback is None
                    or attempt >= self.retries
                ):
                    return content
                quality_fallback_used = True
                model = fallback
            except BaseException as exc:
                last_error = exc
                if (
                    self._reasoning_effort(model, call)
                    and _param_rejection(exc)
                ):
                    self._no_reasoning_param.add(model)
                else:
                    if _model_unavailable(exc):
                        self._dead_models.add(model)
                    alternate = _other_model(
                        model,
                        self._usable_models(),
                        call.category,
                        self.accuracy_first,
                    )
                    if alternate is not None:
                        model = alternate
                if attempt >= self.retries:
                    break
                await asyncio.sleep(min(0.25 * (2**attempt), 1.0))
        raise RuntimeError(f"remote call failed: {last_error}")


_COMPLEX_SUMMARY_RE = re.compile(
    r"\b(?:action items?|follow[- ]?ups?|assigned tasks?|every follow|exactly|"
    r"leave out|omit|exclude|without (?:stating|mentioning|including)|do not include|"
    r"bullet points?|structured report|table)\b",
    re.I,
)
_COMPLEX_SENTIMENT_RE = re.compile(
    r"\b(?:sarcasm|sarcastic|irony|ironic|aspect|but|yet|though|however|despite|"
    r"five stars|what a bargain|truly impressive)\b",
    re.I,
)
_COMPLEX_NER_RE = re.compile(
    r"\b(?:events?|ambiguous|ambiguity|can (?:denote|refer to|mean|belong to)|"
    r"same name|proper nouns?|metonym|fictional|real person)\b",
    re.I,
)


def _adaptive_reasoning_effort(category: str, prompt: str, default: str) -> str:
    """Spend reasoning only where direct-output instructions contain real coupling.

    QA and NER are retrieval/extraction tasks, so hidden reasoning adds cost without a
    verification benefit. Sentiment keeps reasoning for mixed/aspect/sarcastic language.
    Summaries keep it for exclusion, exact-structure, and action-item constraints.
    """
    category = canonical_category(category)
    if category == "actual_qa":
        return "none"
    if category == "named_entity_recognition":
        return default if _COMPLEX_NER_RE.search(prompt) else "none"
    if category == "sentiment_analysis":
        return default if _COMPLEX_SENTIMENT_RE.search(prompt) else "none"
    if category == "summarization":
        return default if _COMPLEX_SUMMARY_RE.search(prompt) else "none"
    return default


def resolve_reasoning_effort(
    category: str,
    prompt: str,
    default: str,
    by_category: dict[str, str] | None = None,
) -> str:
    configured = {
        canonical_category(str(key)): str(value)
        for key, value in (by_category or {}).items()
    }.get(canonical_category(category))
    if configured == "adaptive":
        return _adaptive_reasoning_effort(category, prompt, default)
    if configured is not None:
        return configured
    return default


def _other_model(
    primary: str,
    models: list[str],
    category: str | None = None,
    accuracy_first: bool = False,
) -> str | None:
    remaining = [model for model in models if model != primary]
    if not remaining:
        return None
    if accuracy_first and category is not None:
        return choose_accuracy_model(category, remaining)
    return rank_models(remaining)[0]


_DIAGNOSIS_REQUEST_RE = re.compile(
    r"\b(?:review|diagnos(?:e|is)|comment|explain|identify)\b|"
    r"point out what is actually wrong",
    re.I,
)


def _code_payload_incomplete(call: RemoteCall, content: str) -> bool:
    """Catch only high-confidence truncation signals before accepting code output."""
    if call.category not in {"code_debugging", "code_generation"} or not content.strip():
        return not content.strip()
    # A diagnosis can legitimately quote one incomplete line from the source.
    if call.category == "code_debugging" and _DIAGNOSIS_REQUEST_RE.search(call.prompt):
        return False

    text = content.strip()
    if text.startswith("```") and text.count("```") < 2:
        return True
    fenced = re.search(r"```[A-Za-z0-9_+#.-]*\s*\n(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    lower_prompt = call.prompt.lower()
    if "python" in lower_prompt and re.search(r"^(?:async\s+)?def\s+|^class\s+", text, re.M):
        try:
            ast.parse(text)
        except SyntaxError:
            return True
    # C-family methods/classes require balanced blocks. This also catches the common
    # provider failure where output ends halfway through `return true`.
    if "{" in text and text.count("{") != text.count("}"):
        return True
    return False


def _param_rejection(exc: BaseException) -> bool:
    text = str(exc)
    return "reasoning_effort" in text or "invalid_request_error" in text or "HTTP 400" in text or "400 Bad Request" in text


def _model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        bool(re.search(r"\b(401|403|404)\b", text))
        or "model not found" in text
        or "does not exist" in text
        or "no access" in text
    )


def _message_content(data: dict[str, Any]) -> str:
    message = dict(data["choices"][0].get("message") or {})
    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content", "")
    return str(content or "")


def _completion_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


async def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    try:
        import httpx  # type: ignore

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return dict(response.json())
    except ModuleNotFoundError:
        return await asyncio.to_thread(_post_json_urllib, url, payload, headers, timeout)


def _post_json_urllib(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _self_check() -> None:
    models = [
        "minimax-m3",
        "kimi-k2p7-code",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it-nvfp4",
    ]
    assert choose_model_for_category("code_generation", models) == "kimi-k2p7-code"
    assert choose_model_for_category("code_debugging", models) == "kimi-k2p7-code"
    assert choose_model_for_category("math_reasoning", models) == "minimax-m3"
    assert choose_model_for_category("sentiment_analysis", models) == "minimax-m3"
    assert choose_model_for_category("named_entity_recognition", models) == "kimi-k2p7-code"
    # Knowledge stays on minimax; transformations take the low-overhead model.
    two = ["minimax-m3", "kimi-k2p7-code"]
    assert choose_model_for_category("actual_qa", two) == "minimax-m3"
    assert choose_model_for_category("sentiment_analysis", two) == "minimax-m3"
    assert choose_model_for_category("summarization", two) == "kimi-k2p7-code"
    assert choose_model_for_category("logic_puzzles", two) == "kimi-k2p7-code"
    assert choose_model_for_category("named_entity_recognition", two) == "kimi-k2p7-code"

    # The judge roster must not silently move primary traffic onto locally untestable Gemmas.
    official = [
        "minimax-m3",
        "kimi-k2p7-code",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it-nvfp4",
    ]
    expected = {
        "actual_qa": "minimax-m3",
        "math_reasoning": "minimax-m3",
        "sentiment_analysis": "minimax-m3",
        "summarization": "kimi-k2p7-code",
        "named_entity_recognition": "kimi-k2p7-code",
        "code_debugging": "kimi-k2p7-code",
        "logic_puzzles": "kimi-k2p7-code",
        "code_generation": "kimi-k2p7-code",
    }
    assert {name: choose_accuracy_model(name, official) for name in expected} == expected
    assert _param_count("gemma-4-26b-a4b-it") == 4.0
    assert _param_count("gemma-4-31b-it") == 31.0
    assert _param_count("mixtral-8x7b-instruct") == 7.0
    assert _param_count("minimax-m3") == 999.0
    # Active-param awareness ranks the a4b MoE cheapest among unknown families.
    ranked = rank_models(["foo-9b-it", "foo-26b-a4b-it"])
    assert ranked[0] == "foo-26b-a4b-it", ranked
    java_call = RemoteCall(
        task_id="code",
        category="code_generation",
        prompt="Write a Java method.",
        max_tokens=128,
    )
    assert _code_payload_incomplete(java_call, "public boolean ok() { return t")
    assert not _code_payload_incomplete(java_call, "public boolean ok() { return true; }")
    python_call = RemoteCall(
        task_id="py",
        category="code_generation",
        prompt="Write a Python function.",
        max_tokens=128,
    )
    assert _code_payload_incomplete(python_call, "def f(x):\n    return (")
    assert not _code_payload_incomplete(python_call, "def f(x):\n    return x")
    asyncio.run(_self_check_complete())


async def _self_check_complete() -> None:
    global _post_json
    os.environ.setdefault("FIREWORKS_BASE_URL", "http://stub.invalid/v1")
    os.environ.setdefault("ALLOWED_MODELS", "minimax-m3,gemma-4-26b-a4b-it")
    real = _post_json
    call = RemoteCall(task_id="t1", category="math_reasoning", prompt="2+2?", max_tokens=8)
    try:
        # 1. Blank completion falls back to a different model.
        models_called: list[str] = []

        async def blank_then_ok(url, payload, headers, timeout):
            models_called.append(payload["model"])
            content = "" if len(models_called) == 1 else "4"
            return {"choices": [{"message": {"content": content}}], "usage": {}}

        _post_json = blank_then_ok
        answer = await RemoteClient().complete(call)
        assert answer == "4", answer
        assert len(models_called) == 2 and models_called[0] != models_called[1], models_called

        # Structurally truncated code also rotates once instead of shipping a certain zero.
        code_models: list[str] = []

        async def truncated_then_complete(url, payload, headers, timeout):
            code_models.append(payload["model"])
            content = (
                "public boolean ok() { return t"
                if len(code_models) == 1
                else "public boolean ok() { return true; }"
            )
            return {"choices": [{"message": {"content": content}}], "usage": {}}

        _post_json = truncated_then_complete
        code_call = RemoteCall(
            task_id="code",
            category="code_generation",
            prompt="Write a Java method.",
            max_tokens=128,
        )
        answer = await RemoteClient().complete(code_call)
        assert answer.endswith("true; }"), answer
        assert len(code_models) == 2 and code_models[0] != code_models[1], code_models

        # 2. HTTP 400 on `reasoning_effort` retries without the param.
        efforts_sent: list[bool] = []

        async def reject_effort(url, payload, headers, timeout):
            efforts_sent.append("reasoning_effort" in payload)
            if "reasoning_effort" in payload:
                raise RuntimeError('HTTP 400: {"error":{"type":"invalid_request_error"}}')
            return {"choices": [{"message": {"content": "4"}}], "usage": {}}

        _post_json = reject_effort
        answer = await RemoteClient().complete(call)
        assert answer == "4", answer
        assert efforts_sent == [True, False], efforts_sent

        # 3. A request failure rotates to another model; hard failures are blacklisted.
        tried: list[str] = []

        async def dead_then_ok(url, payload, headers, timeout):
            tried.append(payload["model"])
            if len(tried) == 1:
                raise RuntimeError('HTTP 404: {"error":"Model not found"}')
            return {"choices": [{"message": {"content": "4"}}], "usage": {}}

        _post_json = dead_then_ok
        client = RemoteClient()
        answer = await client.complete(call)
        assert answer == "4" and len(tried) == 2 and tried[0] != tried[1], tried
        assert tried[0] in client._dead_models

        # 4. Transient errors rotate without blacklisting.
        transient_tried: list[str] = []

        async def transient_then_ok(url, payload, headers, timeout):
            transient_tried.append(payload["model"])
            if len(transient_tried) == 1:
                raise RuntimeError("HTTP 500: upstream overloaded")
            return {"choices": [{"message": {"content": "4"}}], "usage": {}}

        _post_json = transient_then_ok
        client = RemoteClient()
        answer = await client.complete(call)
        assert answer == "4" and transient_tried[0] != transient_tried[1]
        assert not client._dead_models
    finally:
        _post_json = real


if __name__ == "__main__":
    _self_check()
    print("remote self-check passed")
