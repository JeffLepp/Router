from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass
class LocalResult:
    text: str
    confidence: float


class StubLocalLLM:
    def __init__(self) -> None:
        self.delay = float(os.environ.get("STUB_DELAY_MS", "0") or 0) / 1000.0

    async def generate(self, prompt: str, category: str = "factual", k: int = 1) -> LocalResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        task = prompt.rsplit("TASK:", 1)[-1].strip() if "TASK:" in prompt else prompt
        answer, confidence = self._answer(task, category)
        return LocalResult(answer, confidence)

    def _answer(self, task: str, category: str) -> tuple[str, float]:
        if category == "sentiment":
            return self._sentiment(task)
        if category == "math":
            return self._math(task)
        if category == "summarization":
            return self._summary(task)
        if category == "ner":
            return self._ner(task)
        if category == "code_debug":
            return self._code_debug(task)
        if category == "logic":
            return ("The constraints point to the single consistent assignment.", 0.45)
        if category == "code_gen":
            return ("def solve(*args, **kwargs):\n    raise NotImplementedError('spec needed')", 0.35)
        return ("This asks for a factual explanation. The core answer should define the concept and state the key practical detail.", 0.45)

    def _sentiment(self, task: str) -> tuple[str, float]:
        lower = task.lower()
        pos = sum(word in lower for word in ("great", "love", "excellent", "happy", "good"))
        neg = sum(word in lower for word in ("bad", "hate", "awful", "angry", "terrible"))
        if pos > neg:
            return ("Sentiment: positive. The language is favorable.", 0.8)
        if neg > pos:
            return ("Sentiment: negative. The language is unfavorable.", 0.8)
        return ("Sentiment: neutral. The wording is mixed or informational.", 0.65)

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
        text = re.sub(r"\s+", " ", task).strip()
        text = re.sub(r"(?i)^summari[sz]e.*?:", "", text).strip()
        words = text.split()
        return (" ".join(words[: min(45, len(words))]), 0.55)

    def _ner(self, task: str) -> tuple[str, float]:
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", task)
        lines = []
        seen: set[str] = set()
        for name in names[:12]:
            if name in seen or name.lower() in {"extract", "named", "entities"}:
                continue
            seen.add(name)
            kind = "DATE" if re.search(r"\b\d{4}\b", name) else "OTHER"
            lines.append(f"{name}|{kind}")
        return ("\n".join(lines) if lines else "No named entities found.", 0.5)

    def _code_debug(self, task: str) -> tuple[str, float]:
        code = _extract_code(task)
        if code:
            return (code, 0.45)
        return ("The bug is in the implementation logic; return corrected code.", 0.35)


class LlamaCppClient:
    def __init__(self, base_url: str, model: str = "local", timeout: float = 25.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(self, prompt: str, category: str = "factual", k: int = 1) -> LocalResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2 if k == 1 else 0.5,
            "max_tokens": 512,
        }
        data = await _post_json(f"{self.base_url}/chat/completions", payload, self.timeout)
        text = data["choices"][0]["message"]["content"]
        return LocalResult(str(text).strip(), 0.55)


async def make_local_client(config: dict[str, Any]) -> StubLocalLLM | LlamaCppClient:
    if os.environ.get("AGENT_FORCE_STUB", "1") != "0":
        return StubLocalLLM()
    return LlamaCppClient(
        base_url=str(config.get("base_url", "http://127.0.0.1:8080/v1")),
        model=str(config.get("model", "local")),
    )


def _extract_code(task: str) -> str:
    if "```" in task:
        parts = task.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.lower().startswith("python"):
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

