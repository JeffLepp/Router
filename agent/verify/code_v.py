from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path


BANNED_IMPORT_ROOTS = {
    "http",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
BANNED_CALLS = {"open", "__import__", "eval", "exec", "compile", "input"}
BANNED_ATTRS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
}


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def syntax_check(code: str, language: str = "python") -> bool:
    if language.lower() in {"python", "py"}:
        return syntax_ok(code)
    return bool(code.strip())


def safety_error(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    return f"banned import: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in BANNED_IMPORT_ROOTS:
                return f"banned import: {root}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                return f"banned call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in BANNED_ATTRS:
                    return f"banned call: {pair[0]}.{pair[1]}"
    return None


def run_python(code: str, timeout: float = 2.0) -> tuple[bool, str]:
    safety = safety_error(code)
    if safety:
        return False, safety
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, output


def run_inline_examples(code: str, examples: list[str], timeout: float = 2.0) -> bool:
    harness = code + "\n\n" + "\n".join(examples) + "\n"
    ok, _ = run_python(harness, timeout=timeout)
    return ok


def run_node(code: str, timeout: float = 2.0) -> tuple[bool, str]:
    # ponytail: no JS AST sandbox (unlike run_python). Eval-only — scoring our own
    # benchmark answers on the host, never the competition path. Harden if that changes.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.js"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["node", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return False, "node not found"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, output


def run_node_examples(code: str, examples: list[str], timeout: float = 2.0) -> bool:
    # `require('assert')`, not `require('node:assert')`: the node: scheme prefix only exists from
    # Node 14.18/16 on, and Ubuntu 22.04's packaged node is 12 -- there the harness threw
    # "Cannot find module 'node:assert'" and every JS task failed before its code ever ran.
    # The unprefixed form resolves on every version, including 22.
    harness = "const assert = require('assert');\n" + code + "\n\n" + "\n".join(examples) + "\n"
    ok, _ = run_node(harness, timeout=timeout)
    return ok


def _self_check() -> None:
    assert syntax_ok("def add(a, b):\n    return a + b\n")
    assert not syntax_ok("def nope(:\n")
    ok, _ = run_python("print('ok')")
    assert ok
    assert run_inline_examples("def add(a, b):\n    return a + b", ["assert add(2, 3) == 5"])
    assert not run_inline_examples("def add(a, b):\n    return a - b", ["assert add(2, 3) == 5"])
    import shutil
    if shutil.which("node"):
        assert run_node_examples("function add(a,b){return a+b;}", ["assert.strictEqual(add(2,3),5);"])
        assert not run_node_examples("function add(a,b){return a-b;}", ["assert.strictEqual(add(2,3),5);"])
    ok, output = run_python("while True:\n    pass\n", timeout=0.2)
    assert not ok and output == "timeout"
    ok, output = run_python("import socket\n")
    assert not ok and "banned import" in output
    ok, output = run_python("open('x.txt', 'w').write('no')\n")
    assert not ok and "banned call" in output


if __name__ == "__main__":
    _self_check()
    print("code_v self-check passed")
