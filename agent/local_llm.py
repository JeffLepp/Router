from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib import request

from agent.classify import canonical_category


@dataclass
class LocalResult:
    text: str
    confidence: float = 0.0
    samples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.samples and self.text:
            self.samples = [self.text]


class StubLocalLLM:
    """Deterministic dev/CI stand-in for a local llama.cpp server."""

    def __init__(self, responses: list[str] | None = None, delay_s: float | None = None) -> None:
        env_delay = float(os.environ.get("STUB_DELAY_MS", "0") or 0) / 1000.0
        self.delay = env_delay if delay_s is None else delay_s
        self.responses = list(responses or [])
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        category: str = "actual_qa",
        k: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LocalResult | None:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.responses:
            samples = [self.responses.pop(0) for _ in range(max(1, k)) if self.responses]
            if not samples:
                return None
            return LocalResult(samples[0], 0.9, samples)

        task = prompt.rsplit("TASK:", 1)[-1].strip() if "TASK:" in prompt else prompt
        answer, confidence = self._answer(task, canonical_category(category))
        samples = [answer for _ in range(max(1, k))]
        return LocalResult(answer, confidence, samples)

    def _answer(self, task: str, category: str) -> tuple[str, float]:
        if category == "sentiment_analysis":
            return self._sentiment(task)
        if category == "math_reasoning":
            return self._math(task)
        if category == "summarization":
            return self._summary(task)
        if category == "named_entity_recognition":
            return self._ner(task)
        if category == "code_debugging":
            return self._code_debug(task)
        if category == "logic_puzzles":
            return ("The constraints point to the single consistent assignment.", 0.45)
        if category == "code_generation":
            return self._code_gen(task)
        return self._factual(task)

    def _factual(self, task: str) -> tuple[str, float]:
        lower = task.lower()
        if "gpu" in lower:
            return (
                "A GPU is a processor optimized for parallel computations, commonly used for graphics and AI workloads.",
                0.82,
            )
        if "cpu" in lower:
            return (
                "A CPU is a general-purpose processor that runs program instructions and coordinates system work.",
                0.82,
            )
        return (
            "The answer is a concise factual explanation of the concept and its main practical use.",
            0.75,
        )

    def _sentiment(self, task: str) -> tuple[str, float]:
        lower = task.lower()
        pos = sum(word in lower for word in ("great", "love", "excellent", "happy", "good"))
        neg = sum(word in lower for word in ("bad", "hate", "awful", "angry", "terrible"))
        if pos > neg:
            return ("positive", 0.8)
        if neg > pos:
            return ("negative", 0.8)
        return ("neutral", 0.65)

    def _math(self, task: str) -> tuple[str, float]:
        exprs = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)+", task)
        if exprs:
            try:
                value = _safe_arithmetic(exprs[-1])
                return (f"The final answer is {value}.", 0.75)
            except ValueError:
                pass
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", task)]
        if nums and re.search(r"\b(sum|total|altogether)\b", task, re.I):
            total = sum(nums)
            return (f"The final answer is {total:g}.", 0.65)
        return ("The final answer needs arithmetic from the prompt.", 0.35)

    def _summary(self, task: str) -> tuple[str, float]:
        forced = os.environ.get("STUB_SUMMARY_RESPONSE")
        if forced is not None:
            return (forced, 0.9)
        text = re.sub(r"\s+", " ", task).strip()
        text = re.sub(r"(?i)^summari[sz]e.*?:", "", text).strip()
        words = text.split()
        return (" ".join(words[: min(45, len(words))]), 0.55)

    def _ner(self, task: str) -> tuple[str, float]:
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b", task)
        entities: list[dict[str, str]] = []
        seen: set[str] = set()
        for name in names[:12]:
            if name in seen or name.lower() in {"extract", "named", "entities"}:
                continue
            seen.add(name)
            kind = "ORGANIZATION" if name.isupper() else "PERSON"
            entities.append({"text": name, "type": kind})
        return (json.dumps({"entities": entities}, separators=(",", ":")), 0.55)

    def _code_debug(self, task: str) -> tuple[str, float]:
        code = _extract_code(task)
        if code:
            return (code, 0.45)
        return ("The bug is in the implementation logic; return corrected code.", 0.35)

    def _code_gen(self, task: str) -> tuple[str, float]:
        if "return 42" in task.lower() or "returns 42" in task.lower():
            return ("def answer():\n    return 42", 0.75)
        return ("def solve(*args, **kwargs):\n    raise NotImplementedError('spec needed')", 0.35)


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str = "local",
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: float = 4.0,
        system_prompt: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max(1, int(max_tokens))
        self.timeout = max(0.1, float(timeout))
        self.system_prompt = system_prompt.strip()

    async def generate(
        self,
        prompt: str,
        category: str = "actual_qa",
        k: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LocalResult | None:
        samples: list[str] = []
        sample_count = max(1, int(k))
        base_temp = self.temperature if temperature is None else float(temperature)
        cap = self.timeout if timeout is None else max(0.1, float(timeout))
        token_cap = self.max_tokens if max_tokens is None else max(1, int(max_tokens))

        for index in range(sample_count):
            temp = base_temp if index == 0 else max(base_temp, 0.2)
            text = await self._complete(prompt, temp, token_cap, cap)
            if text is None:
                return None
            samples.append(text)
        confidence = 0.7 if len({normalize_sample(item) for item in samples}) == 1 else 0.55
        return LocalResult(samples[0], confidence, samples)

    async def _complete(
        self, prompt: str, temperature: float, max_tokens: int, timeout: float
    ) -> str | None:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            data = await asyncio.wait_for(
                _post_json(f"{self.base_url}/chat/completions", payload, timeout),
                timeout=timeout,
            )
            text = data["choices"][0]["message"]["content"]
        except Exception:
            return None
        return str(text).strip()


async def make_local_client(
    llama_config: dict[str, Any] | None = None,
    local_config: dict[str, Any] | None = None,
) -> StubLocalLLM | LlamaCppClient:
    if os.environ.get("AGENT_FORCE_STUB", "0").lower() in {"1", "true", "yes"}:
        return StubLocalLLM()
    llama_config = dict(llama_config or {})
    local_config = dict(local_config or {})
    return LlamaCppClient(
        base_url=str(llama_config.get("base_url", "http://127.0.0.1:8080/v1")),
        model=str(llama_config.get("model", "local")),
        temperature=float(local_config.get("temp", 0) or 0),
        max_tokens=int(local_config.get("max_tokens", 64) or 64),
        timeout=float(local_config.get("latency_cap_s", 4) or 4),
        system_prompt=os.environ.get(
            "LOCAL_SYSTEM_PROMPT", str(llama_config.get("system_prompt", ""))
        ),
    )


def normalize_sample(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    compact = re.sub(r"[^a-z0-9 .:_-]+", "", compact)
    return compact.strip(" .")


def _extract_code(task: str) -> str:
    if "```" in task:
        parts = task.split("```")
        if len(parts) >= 3:
            code = parts[1]
            first_newline = code.find("\n")
            if first_newline != -1 and re.fullmatch(
                r"[A-Za-z0-9_+#.-]+", code[:first_newline].strip()
            ):
                code = code[first_newline + 1 :]
            elif code.lower().startswith("python"):
                code = code[6:]
            return code.strip()
    return ""


def _safe_arithmetic(expr: str) -> float | int:
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("unsafe arithmetic")

    value = walk(ast.parse(expr, mode="eval"))
    return int(value) if value.is_integer() else round(value, 8)


async def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        import httpx  # type: ignore

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return dict(response.json())
    except ModuleNotFoundError:
        return await asyncio.to_thread(_post_json_urllib, url, payload, timeout)


def _post_json_urllib(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _self_check() -> None:
    stub = StubLocalLLM()
    result = asyncio.run(stub.generate("TASK: What is a GPU?", "actual_qa", k=2))
    assert result is not None
    assert len(result.samples) == 2
    assert normalize_sample("Hello,  world!") == "hello world"


if __name__ == "__main__":
    _self_check()
    print("local_llm self-check passed")
