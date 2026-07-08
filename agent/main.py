from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.classify import classify
from agent.config import AgentConfig
from agent.contracts import build_contracts
from agent.gate import solve as gate_solve


@dataclass
class Task:
    task_id: str
    prompt: str


@dataclass
class TaskState:
    task: Task
    category: str = "actual_qa"
    answer: str = ""
    confidence: float = 0.0
    remote_prompt: str = ""
    remote_max_tokens: int = 0
    source: str = "deferred"


class SnapshotWriter:
    def __init__(self, output_path: Path, states: dict[str, TaskState]) -> None:
        self.output_path = output_path
        self.states = states
        self.lock = asyncio.Lock()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self) -> None:
        async with self.lock:
            self.write_sync()

    def write_sync(self) -> None:
        rows = [
            {"task_id": state.task.task_id, "answer": state.answer}
            for state in self.states.values()
        ]
        tmp_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(rows, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp_path, self.output_path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.025)


def _paths() -> tuple[Path, Path, Path | None]:
    input_path = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
    output_path = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))
    config_raw = os.environ.get("CONFIG_PATH")
    return input_path, output_path, Path(config_raw) if config_raw else None


def _load_tasks(path: Path) -> list[Task]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw or "[]")
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        data = []
    if not isinstance(data, list):
        data = []
    deduped: dict[str, Task] = {}
    order: list[str] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or item.get("id") or f"task_{index}")
        prompt = str(item.get("prompt", ""))
        if task_id in deduped:
            order.remove(task_id)
        order.append(task_id)
        deduped[task_id] = Task(task_id=task_id, prompt=prompt)
    return [deduped[task_id] for task_id in order]


def _write_import_audit() -> None:
    audit_path = os.environ.get("AGENT_IMPORT_AUDIT_PATH")
    if not audit_path:
        return
    payload = {
        "remote_imported": "agent.remote" in sys.modules,
        "remote_enabled_env": bool(os.environ.get("FIREWORKS_BASE_URL")),
    }
    Path(audit_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def _snapshot_loop(writer: SnapshotWriter, interval: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            await writer.write()


async def _prepare_one(
    state: TaskState,
    contracts: dict[str, Any],
    writer: SnapshotWriter,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        classification = await classify(state.task.prompt, None)
        state.category = classification.category
        state.confidence = classification.confidence
        answer = None
        if classification.confidence >= 0.6:
            answer = gate_solve(state.category, state.task.prompt)
        if answer is not None:
            state.answer = answer
            state.source = "gate"
            state.confidence = 1.0
        else:
            contract = contracts[state.category]
            state.remote_prompt = contract.remote_prompt(state.task.prompt)
            state.remote_max_tokens = contract.max_tokens
            state.source = "deferred"
    await writer.write()


async def _run_remote(
    states: list[TaskState],
    config: AgentConfig,
    contracts: dict[str, Any],
    writer: SnapshotWriter,
    deadline: float,
) -> Any | None:
    deferred = [state for state in states if not state.answer]
    if not deferred or not config.remote_enabled:
        return None

    remote_module = importlib.import_module("agent.remote")
    client = remote_module.RemoteClient(
        timeout=float(config.remote.get("timeout_seconds", 25)),
        retries=int(config.remote.get("retries", 2)),
        temperature=float(config.remote.get("temperature", 0)),
        usd_per_mtok=float(config.remote.get("usd_per_mtok", 0) or 0),
        dev_spend_cap=float(config.remote.get("dev_spend_cap", 0) or 0),
    )
    calls = [
        remote_module.RemoteCall(
            task_id=state.task.task_id,
            category=state.category,
            prompt=state.remote_prompt,
            max_tokens=state.remote_max_tokens,
            flip_value=max(0.05, 1.0 - state.confidence),
            mandatory=config.mandatory_remote > 0,
        )
        for state in deferred
    ]
    selected = list(calls) if config.mandatory_remote > 0 else []
    if config.mandatory_remote <= 0 and config.token_budget > 0:
        selected.extend(remote_module.select_budgeted_calls(calls, config.token_budget))
    by_id = {state.task.task_id: state for state in states}
    semaphore = asyncio.Semaphore(config.remote_slots)

    async def complete_one(call: Any) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("global wall clock exhausted")
        timeout = min(float(config.remote.get("timeout_seconds", 25)), max(0.5, remaining))
        return await asyncio.wait_for(client.complete(call), timeout=timeout)

    async def write_payload(task_id: str, payload: str) -> None:
        state = by_id[task_id]
        state.answer = contracts[state.category].assemble(state.task.prompt, payload)
        state.confidence = max(state.confidence, 0.75)
        state.source = "remote"
        await writer.write()

    async def one(call: Any) -> None:
        async with semaphore:
            try:
                payload = await complete_one(call)
                await write_payload(call.task_id, payload)
            except Exception as exc:
                state = by_id[call.task_id]
                state.answer = state.answer or ""
                state.confidence = min(state.confidence, 0.4)
                print(f"remote_error task_id={call.task_id} error={exc}", file=sys.stderr)
                await writer.write()

    batching = dict(config.batching or {})
    if batching.get("enabled", False):
        batcher = importlib.import_module("agent.batcher")
        payloads, saved = await batcher.complete_with_batching(
            selected,
            complete_one=complete_one,
            complete_batch=complete_one,
            max_batch_size=int(batching.get("max_batch_size", 10) or 10),
        )
        for task_id, payload in payloads.items():
            await write_payload(task_id, payload)
        print(f"BATCH_LEDGER saved_estimated_tokens={saved}")
    else:
        await asyncio.gather(*(one(call) for call in selected))
    return client


async def run_agent() -> int:
    input_path, output_path, config_path = _paths()
    config = AgentConfig.from_path(config_path)
    if "AGENT_WALL_SECONDS" in os.environ:
        config.wall_seconds = float(os.environ["AGENT_WALL_SECONDS"])
    if "AGENT_SNAPSHOT_INTERVAL" in os.environ:
        config.snapshot_interval_seconds = float(os.environ["AGENT_SNAPSHOT_INTERVAL"])

    tasks = _load_tasks(input_path)
    states = {task.task_id: TaskState(task=task) for task in tasks}
    writer = SnapshotWriter(output_path, states)
    writer.write_sync()

    stop = asyncio.Event()

    def flush_and_stop(signum: int, frame: Any) -> None:
        writer.write_sync()
        stop.set()

    try:
        signal.signal(signal.SIGTERM, flush_and_stop)
    except (ValueError, AttributeError):
        pass

    snapshot_task = asyncio.create_task(
        _snapshot_loop(writer, config.snapshot_interval_seconds, stop)
    )
    start = time.monotonic()
    deadline = start + config.wall_seconds
    contracts = build_contracts(config.max_tokens)
    local_sem = asyncio.Semaphore(config.local_slots)
    state_list = list(states.values())

    try:
        await asyncio.gather(
            *(_prepare_one(state, contracts, writer, local_sem) for state in state_list)
        )
        remote_client = await _run_remote(state_list, config, contracts, writer, deadline)
        await writer.write()
        if remote_client is not None:
            ledger = remote_client.ledger.as_dict()
            print("TOKEN_LEDGER " + json.dumps(ledger, sort_keys=True))
            ledger_path = os.environ.get("AGENT_LEDGER_PATH")
            if ledger_path:
                Path(ledger_path).write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except Exception as exc:
        for state in state_list:
            if not state.answer:
                state.answer = ""
        await writer.write()
        print(f"agent_error {exc}", file=sys.stderr)
    finally:
        stop.set()
        snapshot_task.cancel()
        try:
            await snapshot_task
        except asyncio.CancelledError:
            pass
        writer.write_sync()
        _write_import_audit()
    return 0


def main() -> None:
    try:
        code = asyncio.run(run_agent())
    except BaseException as exc:
        print(f"fatal_agent_error {exc}", file=sys.stderr)
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
