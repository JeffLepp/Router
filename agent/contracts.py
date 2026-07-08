from __future__ import annotations

import re
from dataclasses import dataclass
from textwrap import shorten

from agent.classify import CATEGORIES, canonical_category


DEFAULT_MAX_TOKENS = {
    "actual_qa": 120,
    "math_reasoning": 10,
    "sentiment_analysis": 2,
    "summarization": 80,
    "named_entity_recognition": 60,
    "code_debugging": 300,
    "logic_puzzles": 40,
    "code_generation": 300,
}


@dataclass(frozen=True)
class Contract:
    category: str
    max_tokens: int
    remote_instruction: str

    def remote_prompt(self, task_prompt: str) -> str:
        kernel = compress_prompt(self.category, task_prompt)
        return f"{self.remote_instruction}\nKernel:\n{kernel}"

    def assemble(self, task_prompt: str, remote_payload: str) -> str:
        payload = remote_payload.strip()
        if not payload:
            return ""
        if self.category == "math_reasoning":
            return f"The final answer is {payload}."
        if self.category == "sentiment_analysis":
            return f"Sentiment: {payload}."
        if self.category == "named_entity_recognition":
            return f"Named entities:\n{payload}"
        if self.category == "logic_puzzles":
            return f"The answer is: {payload}."
        if self.category in {"code_debugging", "code_generation"}:
            return payload
        return payload


def _normalize_caps(max_tokens: dict[str, int] | None) -> dict[str, int]:
    caps = dict(DEFAULT_MAX_TOKENS)
    if not max_tokens:
        return caps
    for key, value in max_tokens.items():
        caps[canonical_category(key)] = int(value)
    return caps


def compress_prompt(category: str, task_prompt: str) -> str:
    category = canonical_category(category)
    text = re.sub(r"\s+", " ", task_prompt.strip())
    if category == "summarization":
        return _summary_kernel(text)
    if category in {"code_debugging", "code_generation"}:
        fenced = _extract_fenced_code(task_prompt)
        if fenced:
            return shorten(fenced, width=1600, placeholder=" ...")
    width = 1200 if category != "actual_qa" else 900
    return shorten(text, width=width, placeholder=" ...")


def _summary_kernel(text: str) -> str:
    constraint = text.split(":", 1)[0]
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text, flags=re.S)
    passage = " ".join(first or second for first, second in quoted).strip()
    if not passage:
        passage = text
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    keep = [sentence.strip() for sentence in sentences if sentence.strip()][:5]
    if len(sentences) > 5:
        keep.append(sentences[-1].strip())
    kernel = " ".join(keep)
    words = kernel.split()
    if len(words) > 140:
        kernel = " ".join(words[:140])
    return (
        f"Constraint: {shorten(constraint, width=180, placeholder=' ...')}\n"
        f"Extracted kernel (not raw passage): {kernel}\n"
        "Return the final summary only."
    )


def _extract_fenced_code(text: str) -> str:
    parts = text.split("```")
    if len(parts) < 3:
        return ""
    code = parts[1].strip()
    first_newline = code.find("\n")
    if first_newline != -1 and re.fullmatch(r"[A-Za-z0-9_+#.-]+", code[:first_newline].strip()):
        code = code[first_newline + 1 :]
    return code.strip()


def build_contracts(max_tokens: dict[str, int] | None = None) -> dict[str, Contract]:
    caps = _normalize_caps(max_tokens)
    return {
        "actual_qa": Contract(
            "actual_qa",
            caps["actual_qa"],
            "Return terse key-fact bullets only. No preamble.",
        ),
        "math_reasoning": Contract(
            "math_reasoning",
            caps["math_reasoning"],
            "Return the final number only.",
        ),
        "sentiment_analysis": Contract(
            "sentiment_analysis",
            caps["sentiment_analysis"],
            "Return exactly one label: positive, negative, neutral, or mixed.",
        ),
        "summarization": Contract(
            "summarization",
            caps["summarization"],
            "Summarize only the extracted kernel. Obey the stated format and word limit.",
        ),
        "named_entity_recognition": Contract(
            "named_entity_recognition",
            caps["named_entity_recognition"],
            "Return entity|TYPE lines only.",
        ),
        "code_debugging": Contract(
            "code_debugging",
            caps["code_debugging"],
            "Return corrected code only.",
        ),
        "logic_puzzles": Contract(
            "logic_puzzles",
            caps["logic_puzzles"],
            "Return final answer or assignment only.",
        ),
        "code_generation": Contract(
            "code_generation",
            caps["code_generation"],
            "Return code only.",
        ),
    }


def _self_check() -> None:
    contracts = build_contracts({"math": 8, "sentiment": 3})
    assert set(contracts) == set(CATEGORIES)
    assert contracts["math_reasoning"].max_tokens == 8
    assert "final number only" in contracts["math_reasoning"].remote_prompt("What is 2+2?")
    assert contracts["sentiment_analysis"].assemble("x", "positive") == "Sentiment: positive."
    assert "not raw passage" in contracts["summarization"].remote_prompt("Summarize: " + "word " * 300)


if __name__ == "__main__":
    _self_check()
    print("contracts self-check passed")
