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


def start_mock(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen(
        [PYTHON, str(ROOT / "scripts" / "mock_fireworks.py"), "--port", str(port)],
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
    from agent.remote import choose_model_for_category

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


def test_batcher_malformed_line() -> None:
    import asyncio

    from agent.batcher import complete_with_batching
    from agent.remote import RemoteCall

    instruction = "Return exactly one label: positive, negative, neutral, or mixed."
    calls = [
        RemoteCall("t1", "sentiment_analysis", f"{instruction}\nKernel: I love it.", 2),
        RemoteCall("t2", "sentiment_analysis", f"{instruction}\nKernel: It exists.", 2),
        RemoteCall("t3", "sentiment_analysis", f"{instruction}\nKernel: I hate it.", 2),
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


def test_default_config_is_floor_c() -> None:
    from agent.config import AgentConfig

    config = AgentConfig.from_path()
    assert config.token_budget == 0
    assert config.mandatory_remote == 1
    assert config.remote_enabled is True


def main() -> None:
    tests = [
        ("synthetic run + B ledger", test_synthetic_run),
        ("kill valid json", test_kill_valid_json),
        ("zero remote + mandatory m=1", test_zero_remote_and_mandatory),
        ("input hardening", test_input_hardening),
        ("verifier self-checks", test_verifiers),
        ("model preferences", test_model_preferences),
        ("batcher malformed line", test_batcher_malformed_line),
        ("default config is Floor-C", test_default_config_is_floor_c),
    ]
    for name, func in tests:
        start = time.time()
        func()
        print(f"PASS {name} ({time.time() - start:.2f}s)")


if __name__ == "__main__":
    main()
