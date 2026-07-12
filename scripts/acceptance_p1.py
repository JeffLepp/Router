from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_config(path: Path, token_budget: int = 0, mandatory_remote: int = 0) -> None:
    path.write_text(
        "\n".join(
            [
                f"token_budget: {token_budget}",
                f"mandatory_remote: {mandatory_remote}",
                "local_slots: 8",
                "remote_slots: 8",
                "wall_seconds: 30",
                "snapshot_interval_seconds: 0.2",
                "remote:",
                "  timeout_seconds: 5",
                "  retries: 0",
                "  temperature: 0",
                "  usd_per_mtok: 0.9",
                "  dev_spend_cap: 0.0",
                "batching:",
                "  enabled: false",
                "  max_batch_size: 10",
                "max_tokens:",
                "  actual_qa: 20",
                "  math_reasoning: 10",
                "  sentiment_analysis: 2",
                "  summarization: 20",
                "  named_entity_recognition: 20",
                "  code_debugging: 40",
                "  logic_puzzles: 20",
                "  code_generation: 40",
                "",
            ]
        ),
        encoding="utf-8",
    )


def fake_tasks(n: int) -> list[dict[str, str]]:
    prompts = [
        "Classify the sentiment: I love this excellent result.",
        "Calculate 20 + 22.",
        "Summarize this passage in one sentence: AMD builds fast chips for AI workloads.",
        "Extract named entities: Ada Lovelace visited Seattle with AMD on July 7, 2026.",
        "Debug this Python code: ```python\ndef add(a,b):\n    return a-b\n```",
        "Solve this logic puzzle with all conditions satisfied: A is before B.",
        "Write a Python function that returns 42.",
        "What is a GPU?",
    ]
    return [
        {"task_id": f"t{i:03d}", "prompt": prompts[i % len(prompts)]}
        for i in range(n)
    ]


def run_agent(
    tmp: Path,
    tasks: list[dict[str, str]],
    config: Path,
    base_url: str | None = None,
    ledger_path: Path | None = None,
    audit_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
    bom: bool = False,
) -> subprocess.CompletedProcess[str]:
    input_path = tmp / "tasks.json"
    output_path = tmp / "results.json"
    data = json.dumps(tasks, ensure_ascii=False)
    if bom:
        input_path.write_text("\ufeff" + data, encoding="utf-8")
    else:
        input_path.write_text(data, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "INPUT_PATH": str(input_path),
            "OUTPUT_PATH": str(output_path),
            "CONFIG_PATH": str(config),
            "AGENT_FORCE_STUB": "1",
            "ALLOWED_MODELS": "mock-8b-instruct",
        }
    )
    env.pop("FIREWORKS_API_KEY", None)
    if base_url:
        env["FIREWORKS_BASE_URL"] = base_url
        env["FIREWORKS_API_KEY"] = "test-key"
    else:
        env["FIREWORKS_BASE_URL"] = "http://203.0.113.1:9"
    if ledger_path:
        env["AGENT_LEDGER_PATH"] = str(ledger_path)
    if audit_path:
        env["AGENT_IMPORT_AUDIT_PATH"] = str(audit_path)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [PYTHON, "-m", "agent.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert proc.returncode == 0, proc.stderr
    validate_results(output_path)
    return proc


def validate_results(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for _ in range(40):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            break
        except (PermissionError, json.JSONDecodeError, FileNotFoundError) as exc:
            last_error = exc
            time.sleep(0.025)
    else:
        raise AssertionError(f"could not read valid results.json: {last_error}")
    assert isinstance(rows, list)
    for row in rows:
        assert set(row) == {"task_id", "answer"}
        assert isinstance(row["task_id"], str)
        assert isinstance(row["answer"], str)
    return rows


def start_mock(port: int, host: str = "127.0.0.1") -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen(
        [
            PYTHON,
            str(ROOT / "scripts" / "mock_fireworks.py"),
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            metrics(port)
            return proc
        except Exception:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError("mock server did not start")


def metrics(port: int) -> dict[str, int]:
    with request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def test_synthetic_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        server = start_mock(port)
        try:
            config = tmp / "config.yaml"
            write_config(config, token_budget=500, mandatory_remote=0)
            ledger = tmp / "ledger.json"
            run_agent(tmp, fake_tasks(150), config, f"http://127.0.0.1:{port}", ledger)
            rows = validate_results(tmp / "results.json")
            assert len(rows) == 150
            assert json.loads(ledger.read_text(encoding="utf-8"))["total_tokens"] <= 500
        finally:
            server.kill()


def test_kill_valid_json() -> None:
    for idx in range(3):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            config = tmp / "config.yaml"
            write_config(config, token_budget=0, mandatory_remote=0)
            input_path = tmp / "tasks.json"
            output_path = tmp / "results.json"
            tasks = fake_tasks(60)
            input_path.write_text(json.dumps(tasks), encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "INPUT_PATH": str(input_path),
                    "OUTPUT_PATH": str(output_path),
                    "CONFIG_PATH": str(config),
                    "AGENT_FORCE_STUB": "1",
                    "STUB_DELAY_MS": "20",
                    "AGENT_SNAPSHOT_INTERVAL": "0.05",
                }
            )
            proc = subprocess.Popen([PYTHON, "-m", "agent.main"], cwd=ROOT, env=env)
            deadline = time.time() + 5
            while time.time() < deadline:
                if output_path.exists():
                    validate_results(output_path)
                    break
                time.sleep(0.02)
            assert output_path.exists(), "initial snapshot was not written before kill test"
            time.sleep(0.08 + random.random() * 0.18)
            proc.kill()
            proc.wait(timeout=5)
            rows = validate_results(output_path)
            assert len(rows) == 60, f"kill run {idx} dropped task ids"


def test_zero_remote_and_mandatory() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        server = start_mock(port)
        try:
            config = tmp / "zero.yaml"
            write_config(config, token_budget=0, mandatory_remote=0)
            audit = tmp / "import_audit.json"
            run_agent(tmp, fake_tasks(12), config, audit_path=audit)
            assert metrics(port)["requests"] == 0
            assert json.loads(audit.read_text(encoding="utf-8"))["remote_imported"] is False

            config2 = tmp / "mandatory.yaml"
            write_config(config2, token_budget=0, mandatory_remote=1)
            before = metrics(port)["requests"]
            run_agent(tmp, fake_tasks(12), config2, f"http://127.0.0.1:{port}")
            routed = metrics(port)["requests"] - before
            assert 0 < routed < 12, f"expected real gate to reduce remote calls, got {routed}"
        finally:
            server.kill()


def test_input_hardening() -> None:
    cases = [
        [],
        [{"task_id": "one", "prompt": "What is a GPU?", "unknown": 1}],
        [{"task_id": "long", "prompt": "Summarize: " + ("word " * 32000)}],
        [{"task_id": "unicode", "prompt": "Summarize: Cafe \u2603 AMD"}],
        [
            {"task_id": "dup", "prompt": "first"},
            {"task_id": "dup", "prompt": "What is a GPU?"},
        ],
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        config = tmp / "config.yaml"
        write_config(config, token_budget=0, mandatory_remote=0)
        for idx, tasks in enumerate(cases):
            case_dir = tmp / f"case{idx}"
            case_dir.mkdir()
            run_agent(case_dir, tasks, config, bom=(idx == 3))
            rows = validate_results(case_dir / "results.json")
            if idx == 4:
                assert len(rows) == 1 and rows[0]["task_id"] == "dup"


def test_verifiers() -> None:
    modules = [
        "agent.verify.math_v",
        "agent.verify.code_v",
        "agent.verify.format_v",
        "agent.verify.logic_v",
        "agent.gate",
        "agent.contracts",
        "agent.local_llm",
        "agent.local_gate",
        "agent.batcher",
        "agent.remote",
        "agent.solvers.arithmetic",
        "agent.solvers.wordmath",
        "agent.solvers.logic_solve",
        "agent.solvers.ner_solve",
        "agent.solvers.sentiment_solve",
        "agent.solvers.format_solve",
        "agent.solvers.code_solve",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for module in modules:
        proc = subprocess.run(
            [PYTHON, "-m", module],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr
        print(proc.stdout.strip())


def test_model_preferences() -> None:
    from agent.remote import choose_accuracy_model, choose_model_for_category

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
    assert {category: choose_accuracy_model(category, models) for category in expected} == expected
    rigorous = models + ["gpt-reasoning-70b"]
    assert choose_accuracy_model("logic_puzzles", rigorous) == "gpt-reasoning-70b"
    assert choose_accuracy_model("math_reasoning", rigorous) == "gpt-reasoning-70b"
    assert choose_accuracy_model("actual_qa", rigorous) == "minimax-m3"


def test_batcher_malformed_line() -> None:
    import asyncio

    from agent.batcher import complete_with_batching
    from agent.contracts import build_contracts
    from agent.remote import RemoteCall

    contract = build_contracts()["sentiment_analysis"]
    calls = [
        RemoteCall("t1", "sentiment_analysis", contract.remote_prompt("I love it."), contract.max_tokens),
        RemoteCall("t2", "sentiment_analysis", contract.remote_prompt("It exists."), contract.max_tokens),
        RemoteCall("t3", "sentiment_analysis", contract.remote_prompt("I hate it."), contract.max_tokens),
    ]
    rerun: list[str] = []

    async def one(call: RemoteCall) -> str:
        rerun.append(call.task_id)
        return "neutral"

    async def batch(call: RemoteCall) -> str:
        return "1) positive\n2) ???\n3) negative"

    payloads, saved = asyncio.run(
        complete_with_batching(calls, complete_one=one, complete_batch=batch)
    )
    assert payloads == {"t1": "positive", "t3": "negative", "t2": "neutral"}
    assert rerun == ["t2"]
    assert saved > 0
    print(f"BATCH_TEST saved_estimated_tokens={saved} rerun={rerun}")


def test_batcher_preserves_ner_entities() -> None:
    from agent.batcher import parse_batch_response
    from agent.remote import RemoteCall

    calls = [
        RemoteCall("n1", "named_entity_recognition", "p1", 60),
        RemoteCall("n2", "named_entity_recognition", "p2", 60),
    ]
    parsed = parse_batch_response(
        calls,
        "1) Elon Musk|PERSON; Toronto|LOCATION\n"
        "2) Apple|ORGANIZATION; $3 billion|MONEY; Arm|ORGANIZATION",
    )
    assert not parsed.rerun
    assert parsed.payloads["n1"] == "Elon Musk|PERSON\nToronto|LOCATION"
    # This legacy batch format is intentionally preservation-only. The strict scorer no
    # longer accepts it; the NER gold-schema migration is tested as a separate experiment.


def test_mock_recognizes_batch_protocol() -> None:
    from agent.batcher import build_batch_call, parse_batch_response
    from agent.contracts import build_contracts
    from agent.remote import RemoteCall
    from scripts.mock_fireworks import Handler

    contracts = build_contracts()
    cases = (
        ("actual_qa", 10, "What is a GPU?"),
        ("sentiment_analysis", 3, "I love it."),
        ("named_entity_recognition", 3, "Extract entities: AMD opened an office in Seattle."),
    )
    for category, count, prompt in cases:
        contract = contracts[category]
        calls = [
            RemoteCall(
                f"{category}_{index}",
                category,
                contract.remote_prompt(prompt),
                contract.max_tokens,
            )
            for index in range(1, count + 1)
        ]
        batch_call = build_batch_call(calls)
        assert "one line per item" in batch_call.prompt.lower()
        response = Handler._content(None, batch_call.prompt, batch_call.max_tokens)
        parsed = parse_batch_response(calls, response)
        assert not parsed.rerun, (category, response)
        assert set(parsed.payloads) == {call.task_id for call in calls}
        assert f"{count})" in response


def test_benchmark_accumulates_batch_rerun_tokens() -> None:
    from scripts.live_benchmark import _ledger_by_task

    entries = [
        {
            "task_id": "batch:a,b",
            "model": "batch-model",
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
        },
        {
            "task_id": "b",
            "model": "retry-model",
            "total_tokens": 30,
            "prompt_tokens": 20,
            "completion_tokens": 10,
        },
    ]
    by_task = _ledger_by_task(entries)
    assert by_task["a"]["total_tokens"] == 50
    assert by_task["b"]["total_tokens"] == 80
    assert by_task["b"]["model"] == "retry-model"
    assert sum(entry["total_tokens"] for entry in by_task.values()) == 130

    uneven = _ledger_by_task(
        [{"task_id": "batch:a,b,c", "total_tokens": 101, "prompt_tokens": 80, "completion_tokens": 21}]
    )
    assert sum(entry["total_tokens"] for entry in uneven.values()) == 101
    assert sum(entry["prompt_tokens"] for entry in uneven.values()) == 80
    assert sum(entry["completion_tokens"] for entry in uneven.values()) == 21


def test_code_batch_partial_rerun() -> None:
    import asyncio

    from agent.batcher import build_batch_call, complete_with_batching, parse_batch_response
    from agent.contracts import build_contracts
    from agent.remote import RemoteCall

    contract = build_contracts()["code_generation"]
    calls = [
        RemoteCall(
            f"c{index}",
            "code_generation",
            contract.remote_prompt(f"Write a Python function f{index} that returns {index}."),
            contract.max_tokens,
        )
        for index in range(1, 4)
    ]
    batch = build_batch_call(calls)
    assert "<<<ITEM_1>>>" in batch.prompt and "<<<ANSWER_N>>>" in batch.prompt
    partial = (
        "<<<ANSWER_1>>>\ndef f1():\n    return 1\n<<<END_ANSWER_1>>>\n"
        "<<<ANSWER_3>>>\ndef f3():\n    return 3\n<<<END_ANSWER_3>>>"
    )
    parsed = parse_batch_response(calls, partial)
    assert set(parsed.payloads) == {"c1", "c3"}
    assert [call.task_id for call in parsed.rerun] == ["c2"]

    wrong_language = parse_batch_response(
        calls[:1],
        "<<<ANSWER_1>>>\nfunction f1() { return 1; }\n<<<END_ANSWER_1>>>",
    )
    assert not wrong_language.payloads and [call.task_id for call in wrong_language.rerun] == ["c1"]

    rerun: list[str] = []

    async def complete_batch(_call: RemoteCall) -> str:
        return partial

    async def complete_one(call: RemoteCall) -> str:
        rerun.append(call.task_id)
        return "def f2():\n    return 2"

    payloads, _saved = asyncio.run(
        complete_with_batching(
            calls,
            complete_one=complete_one,
            complete_batch=complete_batch,
            code_enabled=True,
        )
    )
    assert set(payloads) == {"c1", "c2", "c3"}
    assert rerun == ["c2"]


def test_default_config_is_accuracy_first() -> None:
    from agent.config import AgentConfig

    config = AgentConfig.from_path()
    assert config.token_budget == 0
    assert config.mandatory_remote == 1
    assert config.remote_enabled is True
    assert config.gate_enabled is True
    assert config.gate_profile == "strict"
    assert config.batching.get("enabled") is False
    assert config.remote.get("accuracy_first") is True
    assert config.remote.get("classifier_enabled") is True

    experiment = AgentConfig.from_path(ROOT / "agent" / "config.remote-classifier.yaml")
    assert experiment.remote.get("classifier_enabled") is True


def test_remote_batch_classifier_overrides_routes() -> None:
    import asyncio

    from agent.config import AgentConfig
    from agent.contracts import build_contracts
    from agent.main import (
        Task,
        TaskState,
        _prepare_deterministic_one,
        _run_remote_classifier,
    )

    class FakeClient:
        def __init__(self) -> None:
            self.reasoning_effort = "high"
            self.calls = 0

        async def complete(self, call: object) -> str:
            self.calls += 1
            assert self.reasoning_effort == ""
            prompt = getattr(call, "prompt")
            rows = json.loads(prompt.split("TASKS_JSON:", 1)[1].strip())
            mapping = {
                row["task_id"]: (
                    "named_entity_recognition"
                    if row["task_id"] == "ner"
                    else "logic_puzzles"
                )
                for row in rows
            }
            return json.dumps(mapping)

    states = [
        TaskState(Task("ner", "Which people and places occur in the passage?")),
        TaskState(Task("logic", "Who must be first under these conditions?")),
    ]
    config = AgentConfig(
        mandatory_remote=1,
        remote={
            "classifier_enabled": True,
            "classifier_batch_size": 1,
            "classifier_max_tokens": 64,
            "classifier_confidence": 0.99,
            "timeout_seconds": 2,
        },
    )
    client = FakeClient()
    overrides, returned = asyncio.run(
        _run_remote_classifier(states, config, time.monotonic() + 5, client)
    )
    assert returned is client and client.calls == 2
    assert client.reasoning_effort == "high"
    assert overrides == {
        "ner": ("named_entity_recognition", 0.99),
        "logic": ("logic_puzzles", 0.99),
    }

    class Writer:
        async def write(self) -> None:
            return None

    prepared = states[0]
    asyncio.run(
        _prepare_deterministic_one(
            prepared,
            config,
            build_contracts(),
            Writer(),  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            overrides["ner"],
        )
    )
    assert prepared.category == "named_entity_recognition"
    assert prepared.source == "deferred" and prepared.remote_prompt


def test_accuracy_profile_keeps_only_strict_arithmetic() -> None:
    import asyncio

    from agent.config import AgentConfig
    from agent.contracts import build_contracts
    from agent.main import Task, TaskState, _prepare_deterministic_one

    class Writer:
        async def write(self) -> None:
            return None

    async def prepare(prompt: str, profile: str) -> TaskState:
        state = TaskState(Task("task", prompt))
        await _prepare_deterministic_one(
            state,
            AgentConfig(gate_enabled=True, gate_profile=profile),
            build_contracts(),
            Writer(),  # type: ignore[arg-type]
            asyncio.Semaphore(1),
        )
        return state

    arithmetic = asyncio.run(prepare("What is 2 + 2?", "strict"))
    assert arithmetic.answer == "4" and arithmetic.source == "gate"
    sentiment = asyncio.run(prepare("Classify the sentiment: I loved it.", "strict"))
    assert sentiment.answer == "" and sentiment.source == "deferred" and sentiment.remote_prompt
    legacy = asyncio.run(prepare("Classify the sentiment: I loved it.", "legacy"))
    assert legacy.answer and legacy.source == "gate"


def main() -> None:
    tests = [
        ("synthetic run + B ledger", test_synthetic_run),
        ("kill valid json", test_kill_valid_json),
        ("zero remote + mandatory m=1", test_zero_remote_and_mandatory),
        ("input hardening", test_input_hardening),
        ("verifier self-checks", test_verifiers),
        ("model preferences", test_model_preferences),
        ("batcher malformed line", test_batcher_malformed_line),
        ("batcher NER entities", test_batcher_preserves_ner_entities),
        ("mock batch protocol", test_mock_recognizes_batch_protocol),
        ("batch rerun token accounting", test_benchmark_accumulates_batch_rerun_tokens),
        ("code batch partial rerun", test_code_batch_partial_rerun),
        ("default config is accuracy-first", test_default_config_is_accuracy_first),
        ("remote batch classifier overrides routes", test_remote_batch_classifier_overrides_routes),
        ("accuracy profile keeps strict arithmetic", test_accuracy_profile_keeps_only_strict_arithmetic),
    ]
    for name, func in tests:
        start = time.time()
        func()
        print(f"PASS {name} ({time.time() - start:.2f}s)")


if __name__ == "__main__":
    main()
