from __future__ import annotations

import asyncio
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
REASONING_RE = re.compile(r"\b(r1|reason|thinking|think|qwq)\b", re.I)
NON_CHAT_RE = re.compile(
    r"\b(flux|image|audio|whisper|embed|tts|guard|moderat|rerank|ocr|video)\b",
    re.I,
)
MODEL_PREFIX = "accounts/fireworks/models/"
# Measured on Fireworks 2026-07-09 with a 1-token user message: minimax-m3 bills 119
# prompt tokens before we send a single word (a provider-side chat template), kimi-k2p7-code
# bills 24. That ~95-token surcharge is paid on EVERY call, so it dominates short-answer
# categories. Keep minimax only where world knowledge or nuance actually earns it back:
#   actual_qa   - kimi answered "unanswerable" for a knowable question (qa_007); minimax knew it.
#   sentiment   - minimax 4/5 vs kimi 3/5 on the non-gate tasks.
# Transformation categories (summarize/extract/deduce) read from the prompt, so they take the
# cheap model. Patterns are an ordered preference; unmatched families fall through to rank_models.
CATEGORY_MODEL_PATTERNS = {
    "code_debugging": (r"kimi.*code", r"gemma.*31.*it", r"minimax"),
    "code_generation": (r"kimi.*code", r"gemma.*31.*it", r"minimax"),
    "sentiment_analysis": (r"minimax", r"gemma.*26.*it", r"gemma.*31.*nvfp4"),
    "math_reasoning": (r"minimax", r"gemma.*31.*it", r"gemma.*26.*it"),
    "logic_puzzles": (r"minimax", r"gemma.*31.*it", r"kimi.*code"),
    "actual_qa": (r"minimax", r"gemma.*31.*it", r"gemma.*31.*nvfp4"),
    "summarization": (r"kimi.*code", r"gemma.*31.*it", r"minimax"),
    "named_entity_recognition": (r"gemma.*31.*it", r"kimi.*code", r"minimax"),
}


@dataclass
class RemoteCall:
    task_id: str
    category: str
    prompt: str
    max_tokens: int
    flip_value: float = 1.0
    mandatory: bool = False

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

    def add(self, task_id: str, model: str, usage: dict[str, Any], estimated: int) -> None:
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
    stripped = source.strip()
    if stripped.startswith("["):
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, list):
            return [str(model).strip() for model in decoded if str(model).strip()]
    return [part.strip() for part in source.split(",") if part.strip()]


def _chat_models(models: list[str]) -> list[str]:
    chat = [model for model in models if not NON_CHAT_RE.search(model)]
    return chat or models


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
    ranked = rank_models(_chat_models(models))
    if not ranked:
        raise RuntimeError("ALLOWED_MODELS is empty but remote calls are enabled")
    return ranked[0]


def choose_model_for_category(category: str, models: list[str]) -> str:
    category = canonical_category(category)
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty but remote calls are enabled")
    chat_models = _chat_models(models)
    non_reasoning = [model for model in chat_models if not REASONING_RE.search(model)]
    candidates = non_reasoning or chat_models
    for pattern in CATEGORY_MODEL_PATTERNS.get(category, ()):
        regex = re.compile(pattern, re.I)
        matches = [model for model in candidates if regex.search(model)]
        if matches:
            return rank_models(matches)[0]
    return choose_default_model(candidates)


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
        usd_per_mtok: float = 0.0,
        dev_spend_cap: float = 0.0,
    ) -> None:
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.models = parse_allowed_models()
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.dev_spend_cap = dev_spend_cap
        self.ledger = TokenLedger(usd_per_mtok=usd_per_mtok)
        # Models that rejected `reasoning_effort`; we stop sending it to them.
        self._no_reasoning_param: set[str] = set()
        if not self.base_url:
            raise RuntimeError("FIREWORKS_BASE_URL is required when remote calls are enabled")
        choose_default_model(self.models)

    def _payload(
        self, model: str, call: RemoteCall, max_tokens: int | None = None
    ) -> dict[str, Any]:
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
            "max_tokens": call.max_tokens if max_tokens is None else max_tokens,
        }
        if model not in self._no_reasoning_param:
            payload["reasoning_effort"] = "none"
        return payload

    async def complete(self, call: RemoteCall) -> str:
        projected = self.ledger.estimated_usd + (
            call.estimated_tokens / 1_000_000
        ) * self.ledger.usd_per_mtok
        if self.dev_spend_cap > 0 and projected > self.dev_spend_cap:
            raise RuntimeError("remote dev spend cap reached")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        deadline = time.monotonic() + self.timeout
        last_error: BaseException | None = None
        had_response = False

        for model in _ordered_models_for_category(call.category, self.models):
            model_had_response = False
            for serving_model in _serving_model_ids(model):
                transient_attempt = 0
                reasoning_retry_used = False
                length_retry_used = False
                max_tokens = call.max_tokens

                while time.monotonic() < deadline:
                    remaining = max(0.5, deadline - time.monotonic())
                    try:
                        data = await asyncio.wait_for(
                            _post_json(
                                _completion_url(self.base_url),
                                self._payload(serving_model, call, max_tokens),
                                headers,
                                remaining,
                            ),
                            timeout=remaining,
                        )
                        had_response = model_had_response = True
                        content = _message_content(data).strip()
                        usage = dict(data.get("usage", {}) or {})
                        self.ledger.add(
                            call.task_id, serving_model, usage, call.estimated_tokens
                        )

                        if _finish_reason(data) == "length" and not length_retry_used:
                            larger_cap = _retry_token_cap(max_tokens)
                            if larger_cap > max_tokens:
                                length_retry_used = True
                                max_tokens = larger_cap
                                continue
                        if content:
                            return content
                        last_error = RuntimeError("blank remote completion")
                        break
                    except BaseException as exc:
                        last_error = exc
                        if _param_rejection(exc) and not reasoning_retry_used:
                            reasoning_retry_used = True
                            self._no_reasoning_param.add(serving_model)
                            continue
                        # The harness can provide bare allowed IDs while the Fireworks
                        # serving API requires accounts/fireworks/models/<id>. A 404 is
                        # fast and unbilled, so try the canonical form, then another
                        # allowed model family.
                        if _not_found(exc):
                            break
                        if transient_attempt >= self.retries:
                            break
                        await asyncio.sleep(min(0.25 * (2**transient_attempt), 1.0))
                        transient_attempt += 1

                # Once an ID produced a response, a blank/length-limited answer should
                # fall back to another logical model, not retry the same model under a
                # second spelling.
                if model_had_response:
                    break

        if had_response:
            return ""
        raise RuntimeError(f"remote call failed: {last_error}")


def _ordered_models_for_category(category: str, models: list[str]) -> list[str]:
    candidates = _chat_models(models)
    primary = choose_model_for_category(category, candidates)
    return [primary, *(model for model in rank_models(candidates) if model != primary)]


def _serving_model_ids(model: str) -> tuple[str, ...]:
    if "/" in model:
        return (model,)
    return (model, MODEL_PREFIX + model)


def _param_rejection(exc: BaseException) -> bool:
    text = str(exc)
    return "reasoning_effort" in text or "invalid_request_error" in text or "HTTP 400" in text or "400 Bad Request" in text


def _not_found(exc: BaseException) -> bool:
    text = str(exc)
    return "HTTP 404" in text or "404 Not Found" in text


def _retry_token_cap(current: int) -> int:
    return min(max(current * 2, current + 128), 2400)


def _message_content(data: dict[str, Any]) -> str:
    message = dict(data["choices"][0].get("message") or {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        content = message.get("reasoning_content", "")
    return str(content or "")


def _finish_reason(data: dict[str, Any]) -> str:
    return str(data.get("choices", [{}])[0].get("finish_reason") or "").lower()


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
    assert choose_model_for_category("named_entity_recognition", models) == "gemma-4-31b-it"
    # Knowledge and hard reasoning stay on minimax; measured transformations use Kimi.
    two = ["minimax-m3", "kimi-k2p7-code"]
    assert choose_model_for_category("actual_qa", two) == "minimax-m3"
    assert choose_model_for_category("sentiment_analysis", two) == "minimax-m3"
    assert choose_model_for_category("summarization", two) == "kimi-k2p7-code"
    assert choose_model_for_category("logic_puzzles", two) == "minimax-m3"
    assert choose_model_for_category("named_entity_recognition", two) == "kimi-k2p7-code"
    assert _param_count("gemma-4-26b-a4b-it") == 4.0
    assert _param_count("gemma-4-31b-it") == 31.0
    assert _param_count("mixtral-8x7b-instruct") == 7.0
    assert _param_count("minimax-m3") == 999.0
    # Active-param awareness ranks the a4b MoE cheapest among unknown families.
    ranked = rank_models(["foo-9b-it", "foo-26b-a4b-it"])
    assert ranked[0] == "foo-26b-a4b-it", ranked
    assert parse_allowed_models('["minimax-m3", "kimi-k2p7-code"]') == two
    assert _serving_model_ids("minimax-m3") == (
        "minimax-m3",
        "accounts/fireworks/models/minimax-m3",
    )
    assert _serving_model_ids("accounts/team/models/custom") == (
        "accounts/team/models/custom",
    )
    assert choose_default_model(["flux-image", "minimax-m3"]) == "minimax-m3"
    asyncio.run(_self_check_complete())


async def _self_check_complete() -> None:
    global _post_json
    old_base = os.environ.get("FIREWORKS_BASE_URL")
    old_models = os.environ.get("ALLOWED_MODELS")
    os.environ["FIREWORKS_BASE_URL"] = "http://stub.invalid/v1"
    os.environ["ALLOWED_MODELS"] = "minimax-m3,gemma-4-26b-a4b-it"
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

        # 3. Bare allowed IDs can require the canonical Fireworks serving prefix.
        prefixed_models: list[str] = []

        async def require_prefix(url, payload, headers, timeout):
            prefixed_models.append(payload["model"])
            if not payload["model"].startswith(MODEL_PREFIX):
                raise RuntimeError("HTTP 404: model not found")
            return {"choices": [{"message": {"content": "4"}}], "usage": {}}

        _post_json = require_prefix
        answer = await RemoteClient().complete(call)
        assert answer == "4", answer
        assert prefixed_models == ["minimax-m3", MODEL_PREFIX + "minimax-m3"], prefixed_models

        # 4. A length-cut response is retried with a larger ceiling.
        caps: list[int] = []

        async def length_then_ok(url, payload, headers, timeout):
            caps.append(int(payload["max_tokens"]))
            finish = "length" if len(caps) == 1 else "stop"
            content = "partial" if len(caps) == 1 else "complete"
            return {
                "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {},
            }

        _post_json = length_then_ok
        answer = await RemoteClient().complete(call)
        assert answer == "complete", answer
        assert caps == [8, 136], caps
    finally:
        _post_json = real
        if old_base is None:
            os.environ.pop("FIREWORKS_BASE_URL", None)
        else:
            os.environ["FIREWORKS_BASE_URL"] = old_base
        if old_models is None:
            os.environ.pop("ALLOWED_MODELS", None)
        else:
            os.environ["ALLOWED_MODELS"] = old_models


if __name__ == "__main__":
    _self_check()
    print("remote self-check passed")
