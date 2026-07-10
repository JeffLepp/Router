from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.local_gate import try_local
from agent.local_llm import LocalResult
from scripts.acceptance_p1 import (
    fake_tasks,
    free_port,
    metrics,
    run_agent,
    start_mock,
    validate_results,
)


class ScriptedClient:
    def __init__(self, *answers: str, delay_s: float = 0.0) -> None:
        self.answers = list(answers)
        self.delay_s = delay_s
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> LocalResult:
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        k = int(kwargs.get("k", 1) or 1)
        samples = (self.answers[:k] or self.answers[:1]) if self.answers else [""]
        return LocalResult(samples[0], 0.9, samples)


def local_config(*enabled_categories: str, enabled: bool = True, cap: float = 1.0) -> dict[str, Any]:
    categories = {
        "actual_qa": False,
        "math_reasoning": False,
        "sentiment_analysis": False,
        "summarization": False,
        "named_entity_recognition": False,
        "code_debugging": False,
        "logic_puzzles": False,
        "code_generation": False,
        "formatting": False,
    }
    for category in enabled_categories:
        categories[category] = True
    return {
        "enabled": enabled,
        "categories": categories,
        "self_consistency_k": 2,
        "temp": 0,
        "max_tokens": 64,
        "latency_cap_s": cap,
        "summary_max_tokens": {
            "one_sentence": 64,
            "three_sentences": 110,
            "structured_report": 150,
            "default": 150,
        },
        "summary_retries": 1,
        "min_classifier_confidence": 0.5,
    }


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_accept_reject_set() -> None:
    cases = [
        (
            "math accept direct",
            "math_reasoning",
            "Calculate 8 + 7.",
            ["15"],
            True,
        ),
        (
            "math reject wrong",
            "math_reasoning",
            "Calculate 8 + 7.",
            ["16"],
            False,
        ),
        (
            "math accept percent",
            "math_reasoning",
            "What is 20 percent of 50?",
            ["10"],
            True,
        ),
        (
            "ner accept json",
            "named_entity_recognition",
            "Extract entities: Ada Lovelace joined AMD in Seattle.",
            ['{"entities":[{"text":"Ada Lovelace","type":"PERSON"},{"text":"AMD","type":"ORGANIZATION"}]}'],
            True,
        ),
        (
            "ner accept lines",
            "named_entity_recognition",
            "Extract people: Ada Lovelace visited Seattle.",
            ["Ada Lovelace|PERSON"],
            True,
        ),
        (
            "ner reject hallucinated span",
            "named_entity_recognition",
            "Extract people: Ada Lovelace visited Seattle.",
            ['{"entities":[{"text":"Grace Hopper","type":"PERSON"}]}'],
            False,
        ),
        (
            "ner reject wrong type",
            "named_entity_recognition",
            "Extract entities: Ada Lovelace joined AMD in Seattle.",
            ['{"entities":[{"text":"Seattle","type":"PERSON"}]}'],
            False,
        ),
        (
            "sentiment accept lexicon",
            "sentiment_analysis",
            "Classify sentiment: I love this excellent result.",
            ["positive", "positive"],
            True,
        ),
        (
            "sentiment reject lexicon disagreement",
            "sentiment_analysis",
            "Classify sentiment: I love this excellent result.",
            ["negative", "negative"],
            False,
        ),
        (
            "sentiment reject lexicon-overrides-disagreement",
            "sentiment_analysis",
            "Classify sentiment: I love this excellent result.",
            ["positive", "negative"],
            False,
        ),
        (
            "sentiment reject self-consistency",
            "sentiment_analysis",
            "Classify sentiment: The rollout was fine overall.",
            ["positive", "neutral"],
            False,
        ),
        (
            "factual accept consistency",
            "actual_qa",
            "What is a GPU?",
            [
                "A GPU is a processor optimized for parallel computation.",
                "A GPU is a processor optimized for parallel computation.",
            ],
            True,
        ),
        (
            "factual reject disagreement",
            "actual_qa",
            "What is a GPU?",
            ["A GPU is a graphics processor.", "A GPU is a storage device."],
            False,
        ),
        (
            "code debug accept tests",
            "code_debugging",
            "Debug Python:\n```python\ndef add(a, b):\n    return a - b\n```\nassert add(2, 3) == 5",
            ["def add(a, b):\n    return a + b"],
            True,
        ),
        (
            "code debug reject failing tests",
            "code_debugging",
            "Debug Python:\n```python\ndef add(a, b):\n    return a - b\n```\nassert add(2, 3) == 5",
            ["def add(a, b):\n    return a - b"],
            False,
        ),
        (
            "logic accept unique order",
            "logic_puzzles",
            "A is before B. B is before C. Return the full order.",
            ["A, B, C", "A, B, C"],
            True,
        ),
        (
            "logic reject wrong order",
            "logic_puzzles",
            "A is before B. B is before C. Return the full order.",
            ["C, B, A", "C, B, A"],
            False,
        ),
        (
            "formatting accept json",
            "formatting",
            "Return JSON with the answer.",
            ['{"answer":"ok"}'],
            True,
        ),
        (
            "summary accept preserved facts",
            "summarization",
            "Summarize in one sentence: The Apollo mission carried 3 astronauts to the Moon "
            "and returned them safely to Earth after completing its goals.",
            ["Apollo carried 3 astronauts to the Moon, completed its goals, and returned safely to Earth."],
            True,
        ),
        (
            "summary reject missing number",
            "summarization",
            "Summarize in one sentence: The Apollo mission carried 3 astronauts to the Moon "
            "and returned them safely to Earth.",
            ["Apollo carried astronauts to the Moon and returned them safely to Earth."],
            False,
        ),
        (
            "summary reject hallucinated name",
            "summarization",
            "Summarize in one sentence: The Apollo mission carried 3 astronauts to the Moon "
            "and returned them safely to Earth.",
            ["NASA's Apollo carried 3 astronauts to the Moon and returned them safely to Earth."],
            False,
        ),
        (
            "summary accept date exclusion",
            "summarization",
            "Summarize but do not mention the date: On March 14, 2026, SpaceX launched "
            "Starship with 4 astronauts to lunar orbit and returned them to Earth.",
            ["SpaceX launched Starship with 4 astronauts to lunar orbit and returned them to Earth."],
            True,
        ),
        (
            "summary reject excluded date",
            "summarization",
            "Summarize but do not mention the date: On March 14, 2026, SpaceX launched "
            "Starship with 4 astronauts to lunar orbit and returned them to Earth.",
            ["On March 14, 2026, SpaceX launched Starship with 4 astronauts to lunar orbit and returned them to Earth."],
            False,
        ),
    ]
    accepted = 0
    rejected = 0
    for name, category, prompt, answers, should_accept in cases:
        client = ScriptedClient(*answers)
        result = run(
            try_local(
                category,
                prompt,
                config=local_config(category),
                client=client,
                classification_confidence=0.9,
            )
        )
        if should_accept:
            assert result is not None, name
            accepted += 1
        else:
            assert result is None, name
            rejected += 1
    print(f"LOCAL_GATE_PRECISION 100.00% accepted={accepted} rejected={rejected}")


def test_disabled_and_no_generation_paths() -> None:
    disabled_client = ScriptedClient("A GPU is a processor.", "A GPU is a processor.")
    disabled = run(
        try_local(
            "actual_qa",
            "What is a GPU?",
            config=local_config("actual_qa", enabled=False),
            client=disabled_client,
        )
    )
    assert disabled is None
    assert disabled_client.calls == 0

    no_tests_client = ScriptedClient("def answer():\n    return 42")
    no_tests = run(
        try_local(
            "code_generation",
            "Write Python code that returns 42.",
            config=local_config("code_generation"),
            client=no_tests_client,
        )
    )
    assert no_tests is None
    assert no_tests_client.calls == 0
    conclusion_client = ScriptedClient("A fluent but unsafe conclusion summary.")
    conclusion = run(
        try_local(
            "summarization",
            "Summarize without mentioning its main conclusion: The council approved a project. "
            "It may create jobs.",
            config=local_config("summarization"),
            client=conclusion_client,
        )
    )
    assert conclusion is None
    assert conclusion_client.calls == 0
    print("PASS disabled and code-without-tests no-generation paths")


def test_deadline_guard() -> None:
    client = ScriptedClient("A GPU is a processor.", "A GPU is a processor.")
    result = run(
        try_local(
            "actual_qa",
            "What is a GPU?",
            config=local_config("actual_qa", cap=4),
            client=client,
            deadline=100.0,
            remaining_tasks=10,
            now=lambda: 99.0,
            classification_confidence=0.9,
        )
    )
    assert result is None
    assert client.calls == 0
    print("PASS deadline guard skipped generation")


def test_latency_cap() -> None:
    client = ScriptedClient(
        "A GPU is a processor.",
        "A GPU is a processor.",
        delay_s=1.0,
    )
    start = time.monotonic()
    result = run(
        try_local(
            "actual_qa",
            "What is a GPU?",
            config=local_config("actual_qa", cap=0.05),
            client=client,
            classification_confidence=0.9,
        )
    )
    elapsed = time.monotonic() - start
    assert result is None
    assert elapsed < 0.5
    print(f"PASS latency cap elapsed={elapsed:.2f}s")


def test_summary_output_caps() -> None:
    one = ScriptedClient("Green plants make food with sunlight and give off oxygen.")
    assert run(
        try_local(
            "summarization",
            "Summarize in one sentence: Green plants use sunlight to make food and release oxygen.",
            config=local_config("summarization"),
            client=one,
        )
    )
    assert one.kwargs[0]["max_tokens"] == 64

    three = ScriptedClient(
        "Commuting fell because of remote work. Cities saw more cycling. More bike lanes are needed by planners."
    )
    assert run(
        try_local(
            "summarization",
            "Create a summary in three sentences: Remote work reduced commuting. Cycling increased "
            "across cities. Planners need more bike lanes.",
            config=local_config("summarization"),
            client=three,
        )
    )
    assert three.kwargs[0]["max_tokens"] == 110
    print("PASS summary shape-specific output caps")


def write_pipeline_config(path: Path, local_enabled: bool) -> None:
    categories = {
        "actual_qa": False,
        "math_reasoning": False,
        "sentiment_analysis": False,
        "summarization": local_enabled,
        "named_entity_recognition": False,
        "code_debugging": False,
        "logic_puzzles": False,
        "code_generation": False,
    }
    lines = [
        "token_budget: 0",
        "mandatory_remote: 1",
        "local_slots: 8",
        "remote_slots: 8",
        "wall_seconds: 30",
        "snapshot_interval_seconds: 0.2",
        "llama:",
        '  base_url: "http://127.0.0.1:8080/v1"',
        '  model: "local"',
        "local_candidate:",
        f"  enabled: {str(local_enabled).lower()}",
        "  self_consistency_k: 2",
        "  temp: 0",
        "  max_tokens: 64",
        "  latency_cap_s: 1",
        "  summary_queue_size: 1",
        "  summary_retries: 1",
        "  summary_stage_timeout_s: 10",
        "  summary_max_tokens:",
        "    one_sentence: 64",
        "    three_sentences: 110",
        "    structured_report: 150",
        "    default: 150",
        "  min_classifier_confidence: 0.5",
        "  categories:",
    ]
    lines.extend(f"    {key}: {str(value).lower()}" for key, value in categories.items())
    lines.extend(
        [
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
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_pipeline_summary_queue_token_drop() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        server = start_mock(port)
        try:
            summary_prompt = (
                "Summarize in one sentence: The Apollo mission carried 3 astronauts to the Moon "
                "and returned them safely to Earth after completing its goals."
            )
            tasks = [
                {"task_id": f"s{i:03d}", "prompt": summary_prompt}
                if i % 2 == 0
                else {"task_id": f"q{i:03d}", "prompt": "What is a GPU?"}
                for i in range(60)
            ]
            baseline_config = tmp / "baseline.yaml"
            local_config_path = tmp / "local.yaml"
            (tmp / "baseline").mkdir()
            (tmp / "local").mkdir()
            write_pipeline_config(baseline_config, local_enabled=False)
            write_pipeline_config(local_config_path, local_enabled=True)

            baseline_ledger = tmp / "baseline-ledger.json"
            before = metrics(port)["requests"]
            run_agent(tmp / "baseline", tasks, baseline_config, f"http://127.0.0.1:{port}", baseline_ledger)
            baseline_requests = metrics(port)["requests"] - before
            baseline_tokens = json.loads(baseline_ledger.read_text(encoding="utf-8"))["total_tokens"]

            local_ledger = tmp / "local-ledger.json"
            before = metrics(port)["requests"]
            run_agent(
                tmp / "local",
                tasks,
                local_config_path,
                f"http://127.0.0.1:{port}",
                local_ledger,
                extra_env={
                    "STUB_SUMMARY_RESPONSE": (
                        "Apollo carried 3 astronauts to the Moon, completed its goals, and "
                        "returned safely to Earth."
                    )
                },
            )
            local_requests = metrics(port)["requests"] - before
            local_tokens = json.loads(local_ledger.read_text(encoding="utf-8"))["total_tokens"]
            rows = validate_results(tmp / "local" / "results.json")

            assert len(rows) == len(tasks)
            assert 0 < local_requests < baseline_requests, (local_requests, baseline_requests)
            assert local_tokens < baseline_tokens, (local_tokens, baseline_tokens)
            print(
                "PIPELINE_SUMMARY_QUEUE "
                f"tasks={len(tasks)} baseline_remote={baseline_requests} local_remote={local_requests} "
                f"baseline_tokens={baseline_tokens} local_tokens={local_tokens}"
            )
        finally:
            server.kill()


def main() -> None:
    tests = [
        ("local gate accept/reject set", test_accept_reject_set),
        ("disabled + no generation", test_disabled_and_no_generation_paths),
        ("deadline guard", test_deadline_guard),
        ("latency cap", test_latency_cap),
        ("summary output caps", test_summary_output_caps),
        ("pipeline summary queue token drop", test_pipeline_summary_queue_token_drop),
    ]
    for name, func in tests:
        start = time.time()
        func()
        print(f"PASS {name} ({time.time() - start:.2f}s)")


if __name__ == "__main__":
    main()
