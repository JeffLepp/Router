from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Counter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.total_tokens = 0

    def add(self, total: int) -> None:
        with self.lock:
            self.requests += 1
            self.total_tokens += total

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {"requests": self.requests, "total_tokens": self.total_tokens}


COUNTER = Counter()


class Handler(BaseHTTPRequestHandler):
    server_version = "MockFireworks/0.1"

    def do_GET(self) -> None:
        if self.path == "/metrics":
            self._send(COUNTER.snapshot())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")
        messages = payload.get("messages", [])
        user = " ".join(str(msg.get("content", "")) for msg in messages if msg.get("role") == "user")
        max_tokens = int(payload.get("max_tokens", 16) or 16)
        content = self._content(user, max_tokens)
        prompt_tokens = max(1, len(user.split()))
        completion_tokens = max(1, len(content.split()))
        total = prompt_tokens + completion_tokens
        COUNTER.add(total)
        self._send(
            {
                "id": "mock",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total,
                },
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _content(self, user: str, max_tokens: int) -> str:
        lower = user.lower()
        if "tasks_json:" in lower and "allowed labels:" in lower:
            try:
                from agent.classify import classify

                rows = json.loads(user.split("TASKS_JSON:", 1)[1].strip())
                mapped = {
                    str(row["task_id"]): asyncio.run(
                        classify(str(row.get("prompt", "")), None)
                    ).category
                    for row in rows
                    if isinstance(row, dict) and row.get("task_id") is not None
                }
                return json.dumps(mapped, separators=(",", ":"))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return "{}"
        numbered = []
        for line in user.splitlines():
            match = re.match(r"^\s*(\d+)\)\s+", line)
            if match:
                numbered.append(match.group(1))
        if numbered and "one line per item" in lower:
            if (
                "positive|negative|neutral" in lower
                or "one label: positive, negative, neutral" in lower
            ):
                return "\n".join(f"{idx}) positive" for idx in numbered)
            if "entity|type" in lower:
                return "\n".join(f"{idx}) AMD|ORGANIZATION" for idx in numbered)
            return "\n".join(f"{idx}) key fact" for idx in numbered)
        if "one label" in lower or "sentiment" in lower:
            return "positive"
        if "final number only" in lower or "math" in lower:
            return "42"
        if '"entities"' in lower and "exact span" in lower:
            return '{"entities":[{"text":"AMD","type":"ORGANIZATION"}]}'
        if "entity|type" in lower:
            return "AMD|ORG"
        if "corrected code" in lower or "return code only" in lower:
            return "def solve():\n    return 42"
        if "assignment" in lower:
            return "A=1, B=2"
        words = "key fact"
        return " ".join([words] * max(1, min(max_tokens, 8)))

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock_fireworks listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
