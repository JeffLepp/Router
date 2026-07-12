from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _coerce_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _minimal_yaml(text)


@dataclass
class AgentConfig:
    token_budget: int = 0
    mandatory_remote: int = 1
    gate_enabled: bool = True
    gate_profile: str = "legacy"
    local_slots: int = 4
    remote_slots: int = 8
    wall_seconds: float = 510.0
    snapshot_interval_seconds: float = 5.0
    model_paths: dict[str, str] = field(default_factory=dict)
    llama: dict[str, Any] = field(default_factory=dict)
    local_candidate: dict[str, Any] = field(default_factory=dict)
    batching: dict[str, Any] = field(default_factory=dict)
    remote: dict[str, Any] = field(default_factory=dict)
    max_tokens: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "AgentConfig":
        data = load_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)
        return cls(
            token_budget=int(data.get("token_budget", 0) or 0),
            mandatory_remote=int(data.get("mandatory_remote", 1)),
            gate_enabled=bool(data.get("gate_enabled", True)),
            gate_profile=str(data.get("gate_profile", "legacy") or "legacy"),
            local_slots=int(data.get("local_slots", 4) or 4),
            remote_slots=int(data.get("remote_slots", 8) or 8),
            wall_seconds=float(data.get("wall_seconds", 510) or 510),
            snapshot_interval_seconds=float(
                data.get("snapshot_interval_seconds", 5) or 5
            ),
            model_paths=dict(data.get("model_paths", {}) or {}),
            llama=dict(data.get("llama", {}) or {}),
            local_candidate=dict(data.get("local_candidate", {}) or {}),
            batching=dict(data.get("batching", {}) or {}),
            remote=dict(data.get("remote", {}) or {}),
            max_tokens={k: int(v) for k, v in (data.get("max_tokens", {}) or {}).items()},
        )

    @property
    def remote_enabled(self) -> bool:
        return self.mandatory_remote > 0 or self.token_budget > 0
