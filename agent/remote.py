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
CATEGORY_MODEL_PATTERNS = {
    "code_debugging": (r"kimi.*code", r"gemma.*31.*it", r"minimax"),
    "code_generation": (r"kimi.*code", r"gemma.*31.*it", r"minimax"),
    "sentiment_analysis": (r"gemma.*26.*it", r"gemma.*31.*nvfp4", r"minimax"),
    "math_reasoning": (r"minimax", r"gemma.*31.*it", r"gemma.*26.*it"),
    "logic_puzzles": (r"minimax", r"gemma.*31.*it", r"gemma.*31.*nvfp4"),
    "actual_qa": (r"minimax", r"gemma.*31.*it", r"gemma.*31.*nvfp4"),
    "summarization": (r"minimax", r"gemma.*31.*it", r"gemma.*31.*nvfp4"),
    "named_entity_recognition": (r"gemma.*31.*it", r"minimax", r"gemma.*26.*it"),
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
    return [part.strip() for part in source.split(",") if part.strip()]


def _param_count(model: str) -> float:
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
        if not self.base_url:
            raise RuntimeError("FIREWORKS_BASE_URL is required when remote calls are enabled")
        choose_default_model(self.models)

    async def complete(self, call: RemoteCall) -> str:
        projected = self.ledger.estimated_usd + (
            call.estimated_tokens / 1_000_000
        ) * self.ledger.usd_per_mtok
        if self.dev_spend_cap > 0 and projected > self.dev_spend_cap:
            raise RuntimeError("remote dev spend cap reached")

        model = choose_model_for_category(call.category, self.models)
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
            "reasoning_effort": "none",
            "stop": ["\n\n\n"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        deadline = time.monotonic() + self.timeout
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            remaining = max(0.5, deadline - time.monotonic())
            try:
                data = await asyncio.wait_for(
                    _post_json(_completion_url(self.base_url), payload, headers, remaining),
                    timeout=remaining,
                )
                content = _message_content(data).strip()
                usage = dict(data.get("usage", {}) or {})
                self.ledger.add(call.task_id, model, usage, call.estimated_tokens)
                return content
            except BaseException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                await asyncio.sleep(min(0.25 * (2**attempt), 1.0))
        raise RuntimeError(f"remote call failed: {last_error}")


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
    assert choose_model_for_category("sentiment_analysis", models) == "gemma-4-26b-a4b-it"
    assert choose_model_for_category("named_entity_recognition", models) == "gemma-4-31b-it"


if __name__ == "__main__":
    _self_check()
    print("remote self-check passed")
