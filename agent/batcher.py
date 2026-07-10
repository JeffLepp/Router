from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from agent.classify import canonical_category
from agent.contracts import build_contracts
from agent.remote import RemoteCall
from agent.verify.code_v import syntax_ok


SAFE_BATCH_CATEGORIES = {
    "actual_qa",
    "sentiment_analysis",
    "named_entity_recognition",
}
CODE_BATCH_CATEGORIES = {"code_debugging", "code_generation"}
_LABELS = {"positive", "negative", "neutral", "mixed"}

CompleteOne = Callable[[RemoteCall], Awaitable[str]]
CompleteBatch = Callable[[RemoteCall], Awaitable[str]]


@dataclass(frozen=True)
class BatchResult:
    payloads: dict[str, str]
    rerun: list[RemoteCall]
    saved_prompt_tokens: int


def is_batchable(call: RemoteCall, code_enabled: bool = False) -> bool:
    category = canonical_category(call.category)
    if category in CODE_BATCH_CATEGORIES:
        kernel = _kernel_from_prompt(call.prompt)
        return bool(code_enabled and "<<<" not in kernel and len(kernel.split()) <= 600)
    if category not in SAFE_BATCH_CATEGORIES:
        return False
    # Measure the KERNEL, not call.prompt: the latter still carries the per-task contract
    # instruction (~75 words), so every short QA task would trip an 80-word budget and
    # silently fall out of batching. The instruction is what batching amortizes away.
    if category == "actual_qa" and len(_kernel_from_prompt(call.prompt).split()) > 80:
        return False
    return True


def make_batches(
    calls: list[RemoteCall], max_batch_size: int = 10, code_enabled: bool = False
) -> list[list[RemoteCall]]:
    batches: list[list[RemoteCall]] = []
    current_by_category: dict[str, list[RemoteCall]] = {}
    for call in calls:
        category = canonical_category(call.category)
        if not is_batchable(call, code_enabled=code_enabled):
            batches.append([call])
            continue
        bucket = current_by_category.setdefault(category, [])
        bucket.append(call)
        if len(bucket) >= max_batch_size:
            batches.append(bucket[:])
            bucket.clear()
    for bucket in current_by_category.values():
        if bucket:
            batches.append(bucket[:])
    return batches


def build_batch_call(calls: list[RemoteCall]) -> RemoteCall:
    if not calls:
        raise ValueError("cannot batch zero calls")
    category = canonical_category(calls[0].category)
    if category in CODE_BATCH_CATEGORIES:
        lines = [
            f"<<<ITEM_{idx}>>>\n{_kernel_from_prompt(call.prompt)}\n<<<END_ITEM_{idx}>>>"
            for idx, call in enumerate(calls, start=1)
        ]
    else:
        lines = [
            f"{idx}) {_kernel_from_prompt(call.prompt)}"
            for idx, call in enumerate(calls, start=1)
        ]
    instruction = _batch_instruction(category, len(calls))
    prompt = f"{instruction}\n" + "\n".join(lines)
    max_tokens = min(sum(call.max_tokens for call in calls), _batch_cap(category, len(calls)))
    return RemoteCall(
        task_id="batch:" + ",".join(call.task_id for call in calls),
        category=category,
        prompt=prompt,
        max_tokens=max_tokens,
        flip_value=sum(call.flip_value for call in calls),
        mandatory=any(call.mandatory for call in calls),
    )


def parse_batch_response(calls: list[RemoteCall], response: str) -> BatchResult:
    by_index = {str(idx): call for idx, call in enumerate(calls, start=1)}
    payloads: dict[str, str] = {}
    bad_indexes: set[str] = set(by_index)
    category = canonical_category(calls[0].category) if calls else "actual_qa"
    if category in CODE_BATCH_CATEGORIES:
        for index, call in by_index.items():
            start = f"<<<ANSWER_{index}>>>"
            end = f"<<<END_ANSWER_{index}>>>"
            if response.count(start) != 1 or response.count(end) != 1:
                continue
            match = re.search(
                re.escape(start) + r"\s*\n?(.*?)\n?\s*" + re.escape(end),
                response,
                re.S,
            )
            if not match:
                continue
            payload = match.group(1).strip()
            if _code_payload_valid(call, payload):
                payloads[call.task_id] = payload
                bad_indexes.discard(index)
        rerun = [by_index[index] for index in sorted(bad_indexes, key=int)]
        saved = max(0, sum(_prompt_estimate(call) for call in calls) - _batch_prompt_estimate(calls))
        return BatchResult(payloads=payloads, rerun=rerun, saved_prompt_tokens=saved)
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\)\s*(.+?)\s*$", line)
        if not match or match.group(1) not in by_index:
            continue
        index, payload = match.groups()
        if _payload_valid(category, payload):
            payloads[by_index[index].task_id] = _normalize_batch_payload(category, payload)
            bad_indexes.discard(index)
    rerun = [by_index[index] for index in sorted(bad_indexes, key=int)]
    saved = max(0, sum(_prompt_estimate(call) for call in calls) - _batch_prompt_estimate(calls))
    return BatchResult(payloads=payloads, rerun=rerun, saved_prompt_tokens=saved)


async def _run_batch(
    batch: list[RemoteCall], complete_one: CompleteOne, complete_batch: CompleteBatch,
    code_enabled: bool = False,
) -> tuple[dict[str, str], int]:
    if len(batch) == 1 or not is_batchable(batch[0], code_enabled=code_enabled):
        return {batch[0].task_id: await complete_one(batch[0])}, 0

    response = await complete_batch(build_batch_call(batch))
    parsed = parse_batch_response(batch, response)
    payloads = dict(parsed.payloads)

    # Any line the model dropped, mis-numbered, or malformed falls back to its own call, so a
    # bad parse degrades to the un-batched cost instead of losing tasks. A blank batch response
    # (a timeout, say) reruns everything -- correct, just not cheap.
    reruns = await asyncio.gather(*(complete_one(call) for call in parsed.rerun))
    payloads.update(dict(zip((call.task_id for call in parsed.rerun), reruns)))

    spent_on_rerun = sum(_prompt_estimate(call) for call in parsed.rerun)
    return payloads, max(0, parsed.saved_prompt_tokens - spent_on_rerun)


async def complete_with_batching(
    calls: list[RemoteCall],
    complete_one: CompleteOne,
    complete_batch: CompleteBatch,
    max_batch_size: int = 10,
    code_enabled: bool = False,
) -> tuple[dict[str, str], int]:
    batches = make_batches(
        calls, max_batch_size=max_batch_size, code_enabled=code_enabled
    )
    results = await asyncio.gather(
        *(
            _run_batch(
                batch, complete_one, complete_batch, code_enabled=code_enabled
            )
            for batch in batches
        )
    )
    payloads: dict[str, str] = {}
    saved_tokens = 0
    for batch_payloads, saved in results:
        payloads.update(batch_payloads)
        saved_tokens += saved
    return payloads, saved_tokens


_LINE_FORMAT = {
    "sentiment_analysis": "N) <label or compact JSON, on ONE line>",
    "named_entity_recognition": "N) entity|TYPE; entity|TYPE",
}


def _batch_instruction(category: str, count: int) -> str:
    """The whole point of batching is to pay the per-category contract ONCE instead of
    `count` times (minimax bills ~119 template tokens per call on top of it). So carry the
    real contract text in, rather than a generic 'Return one line per item' that would drop
    the unanswerable rule, the comparison-sentence rule, and the sarcasm rule."""
    contract = build_contracts()[category].remote_instruction
    if category in CODE_BATCH_CATEGORIES:
        return (
            f"{contract}\n\n"
            f"Solve each of the {count} marked items independently in its requested language.\n"
            "For item N, put the complete multi-line answer between a line containing exactly "
            "<<<ANSWER_N>>> and a line containing exactly <<<END_ANSWER_N>>>.\n"
            "Emit every marker once, in order. No markdown fences, preamble, or text outside markers."
        )
    shape = _LINE_FORMAT.get(category, "N) <answer>")
    return (
        f"{contract}\n\n"
        f"Apply the rules above to each of the {count} numbered items below, independently.\n"
        f"Output exactly {count} numbered lines, one line per item, formatted `{shape}`.\n"
        f"No blank lines, no preamble, no commentary, no repeating the item."
    )


def _batch_cap(category: str, count: int) -> int:
    # A cap is a ceiling, not a spend. Sentiment aspect JSON runs ~40 tokens a line, and a
    # comparison-sentence QA answer ~15; starving the cap truncates the tail of the batch and
    # sends those items back for individual re-runs, which defeats the whole optimization.
    if category == "sentiment_analysis":
        return max(60 * count, 64)
    if category == "named_entity_recognition":
        return 30 * count
    if category in CODE_BATCH_CATEGORIES:
        return max(180 * count, 300)
    return max(30 * count, 40)


def _payload_valid(category: str, payload: str) -> bool:
    if category == "sentiment_analysis":
        text = payload.strip()
        if text.startswith("{"):  # aspect JSON — 3 of 5 sentiment tasks need this shape
            try:
                obj = json.loads(text)
            except ValueError:
                return False
            return isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in _LABELS
        return text.lower() in _LABELS
    if category == "named_entity_recognition":
        return bool(re.fullmatch(r"[^|]+(?:\|[A-Z_]+)(?:\s*;\s*[^|]+(?:\|[A-Z_]+))*", payload))
    return bool(payload.strip())


def _normalize_batch_payload(category: str, payload: str) -> str:
    """Convert category-specific one-line batch encodings to normal contract payloads."""
    if category == "named_entity_recognition":
        return "\n".join(part.strip() for part in payload.split(";") if part.strip())
    return payload


def _code_payload_valid(call: RemoteCall, payload: str) -> bool:
    if not payload or "<<<" in payload or "```" in payload:
        return False
    prompt = _kernel_from_prompt(call.prompt).lower()
    text = payload.strip()
    if "python" in prompt or "```py" in prompt:
        return syntax_ok(text)
    if "sql" in prompt:
        return bool(re.match(r"^(?:SELECT|CREATE|WITH)\b", text, re.I) and text.rstrip().endswith(";"))
    if "javascript" in prompt:
        return bool(re.search(r"\bfunction\b|=>|\b(?:let|const)\b", text))
    if "c++" in prompt or "cpp" in prompt:
        return bool(re.search(r"\bstd::|#include\s*<|\b(?:void|bool|int)\s+\w+\s*\(", text))
    if re.search(r"\bjava\b", prompt):
        return bool(re.search(r"\b(?:public|private|protected)\b", text))
    if re.search(r"\bin\s+c\b|```c\b", prompt):
        return bool(re.search(r"#include\s*<|\b(?:void|bool|int|char)\s+\w+\s*\(", text))
    return False


def _prompt_estimate(call: RemoteCall) -> int:
    return max(1, len(call.prompt.split()))


def _batch_prompt_estimate(calls: list[RemoteCall]) -> int:
    if not calls:
        return 0
    batch_call = build_batch_call(calls)
    return _prompt_estimate(batch_call)


def _kernel_from_prompt(prompt: str) -> str:
    marker = "Kernel:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _self_check() -> None:
    calls = [
        RemoteCall(f"t{i}", "sentiment_analysis", f"Prompt {i}", 2)
        for i in range(1, 4)
    ]
    parsed = parse_batch_response(calls, "1) positive\n2) ???\n3) negative")
    assert parsed.payloads == {"t1": "positive", "t3": "negative"}
    assert [call.task_id for call in parsed.rerun] == ["t2"]
    assert len(make_batches(calls, max_batch_size=2)) == 2
    assert is_batchable(calls[0])
    assert not is_batchable(RemoteCall("m", "math_reasoning", "2+2", 10))
    code_call = RemoteCall("c1", "code_generation", "Write Python code.", 100)
    assert not is_batchable(code_call)
    assert is_batchable(code_call, code_enabled=True)

    # A real QA call carries its contract instruction; batchability is judged on the kernel.
    from agent.contracts import build_contracts as _bc

    qa = _bc()["actual_qa"]
    question = "Who wrote the book 'Harry Potter and the Philosopher's Stone'?"
    real = RemoteCall("qa_001", "actual_qa", qa.remote_prompt(question), 40)
    assert len(real.prompt.split()) > 80, "instruction should dominate the raw prompt"
    assert len(_kernel_from_prompt(real.prompt).split()) < 20
    assert is_batchable(real), "short kernel -> must batch despite the long instruction"
    long_kernel = RemoteCall("qa_x", "actual_qa", qa.remote_prompt("word " * 90), 40)
    assert not is_batchable(long_kernel)

    # The batch prompt must carry the contract rules, not a generic 'one line per item'.
    batch_call = build_batch_call([real, RemoteCall("qa_009", "actual_qa", qa.remote_prompt("When did Atlantis join Canada?"), 40)])
    assert "unanswerable" in batch_call.prompt
    assert "Output exactly 2 numbered lines" in batch_call.prompt
    assert "Kernel:" not in batch_call.prompt  # per-task instruction stripped
    assert batch_call.prompt.count(question) == 1
    assert batch_call.prompt.count("Answer only.") == 1  # contract paid once, not twice

    # Sentiment aspect JSON is a valid batch line (3 of 5 sentiment tasks need it).
    aspect = '{"sentiment":"mixed","aspects":{"battery":"negative"}}'
    sent = [RemoteCall("s1", "sentiment_analysis", "p", 80), RemoteCall("s2", "sentiment_analysis", "p", 80)]
    got = parse_batch_response(sent, f"1) {aspect}\n2) neutral")
    assert got.payloads == {"s1": aspect, "s2": "neutral"}, got.payloads
    assert not got.rerun
    assert _batch_cap("sentiment_analysis", 5) >= 200  # aspect JSON needs headroom
    assert not _payload_valid("sentiment_analysis", '{"sentiment":"banana"}')

    # NER uses semicolons only to keep each batch member on one line. Restore the normal
    # newline-delimited contract payload before Contract.assemble and eval.score see it.
    ner = [RemoteCall("n1", "named_entity_recognition", "p", 60)]
    ner_got = parse_batch_response(ner, "1) Elon Musk|PERSON; Toronto|LOCATION")
    assert ner_got.payloads["n1"] == "Elon Musk|PERSON\nToronto|LOCATION"

    code_calls = [
        RemoteCall(f"c{i}", "code_generation", f"Write a Python function f{i}.", 100)
        for i in range(1, 4)
    ]
    framed = parse_batch_response(
        code_calls,
        "<<<ANSWER_1>>>\ndef f1():\n    return 1\n<<<END_ANSWER_1>>>\n"
        "<<<ANSWER_3>>>\ndef f3():\n    return 3\n<<<END_ANSWER_3>>>",
    )
    assert set(framed.payloads) == {"c1", "c3"}
    assert [call.task_id for call in framed.rerun] == ["c2"]

    # A blank batch response reruns every item rather than losing them.
    async def _blank(_call: RemoteCall) -> str:
        return ""

    async def _one(call: RemoteCall) -> str:
        return f"answer-{call.task_id}"

    payloads, _saved = asyncio.run(complete_with_batching(sent, _one, _blank, max_batch_size=10))
    assert payloads == {"s1": "answer-s1", "s2": "answer-s2"}, payloads


if __name__ == "__main__":
    _self_check()
    print("batcher self-check passed")
