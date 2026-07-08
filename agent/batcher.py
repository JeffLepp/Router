from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from agent.classify import canonical_category
from agent.remote import RemoteCall


SAFE_BATCH_CATEGORIES = {
    "actual_qa",
    "sentiment_analysis",
    "named_entity_recognition",
}

CompleteOne = Callable[[RemoteCall], Awaitable[str]]
CompleteBatch = Callable[[RemoteCall], Awaitable[str]]


@dataclass(frozen=True)
class BatchResult:
    payloads: dict[str, str]
    rerun: list[RemoteCall]
    saved_prompt_tokens: int


def is_batchable(call: RemoteCall) -> bool:
    category = canonical_category(call.category)
    if category not in SAFE_BATCH_CATEGORIES:
        return False
    if category == "actual_qa" and len(call.prompt.split()) > 80:
        return False
    return True


def make_batches(calls: list[RemoteCall], max_batch_size: int = 10) -> list[list[RemoteCall]]:
    batches: list[list[RemoteCall]] = []
    current_by_category: dict[str, list[RemoteCall]] = {}
    for call in calls:
        category = canonical_category(call.category)
        if not is_batchable(call):
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
    lines = [
        f"{idx}) {_kernel_from_prompt(call.prompt)}"
        for idx, call in enumerate(calls, start=1)
    ]
    instruction = _batch_instruction(category)
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
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\)\s*(.+?)\s*$", line)
        if not match or match.group(1) not in by_index:
            continue
        index, payload = match.groups()
        if _payload_valid(category, payload):
            payloads[by_index[index].task_id] = payload
            bad_indexes.discard(index)
    rerun = [by_index[index] for index in sorted(bad_indexes, key=int)]
    saved = max(0, sum(call.estimated_tokens for call in calls) - _batch_estimate(calls))
    return BatchResult(payloads=payloads, rerun=rerun, saved_prompt_tokens=saved)


async def complete_with_batching(
    calls: list[RemoteCall],
    complete_one: CompleteOne,
    complete_batch: CompleteBatch,
    max_batch_size: int = 10,
) -> tuple[dict[str, str], int]:
    payloads: dict[str, str] = {}
    saved_tokens = 0
    for batch in make_batches(calls, max_batch_size=max_batch_size):
        if len(batch) == 1 or not is_batchable(batch[0]):
            payloads[batch[0].task_id] = await complete_one(batch[0])
            continue
        batch_call = build_batch_call(batch)
        response = await complete_batch(batch_call)
        parsed = parse_batch_response(batch, response)
        payloads.update(parsed.payloads)
        saved_tokens += parsed.saved_prompt_tokens
        for failed in parsed.rerun:
            payloads[failed.task_id] = await complete_one(failed)
    return payloads, saved_tokens


def _batch_instruction(category: str) -> str:
    if category == "sentiment_analysis":
        return "Return one line per item as: 1) positive|negative|neutral|mixed"
    if category == "named_entity_recognition":
        return "Return one line per item as: 1) entity|TYPE"
    return "Return one line per item as: 1) answer"


def _batch_cap(category: str, count: int) -> int:
    if category == "sentiment_analysis":
        return max(2 * count, 4)
    if category == "named_entity_recognition":
        return 30 * count
    return 20 * count


def _payload_valid(category: str, payload: str) -> bool:
    if category == "sentiment_analysis":
        return payload.lower() in {"positive", "negative", "neutral", "mixed"}
    if category == "named_entity_recognition":
        return bool(re.fullmatch(r"[^|]+(?:\|[A-Z_]+)(?:\s*;\s*[^|]+(?:\|[A-Z_]+))*", payload))
    return bool(payload.strip())


def _batch_estimate(calls: list[RemoteCall]) -> int:
    if not calls:
        return 0
    batch_call = build_batch_call(calls)
    return len(batch_call.prompt.split()) + batch_call.max_tokens


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


if __name__ == "__main__":
    _self_check()
    print("batcher self-check passed")
