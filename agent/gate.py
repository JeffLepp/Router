from __future__ import annotations

from collections.abc import Callable

from agent.classify import CATEGORIES, canonical_category


Solver = Callable[[str], str | None]


def _defer(prompt: str) -> str | None:
    return None


SOLVERS: dict[str, Solver] = {category: _defer for category in CATEGORIES}


def solve(category: str, prompt: str) -> str | None:
    """P1 pass-through stub: P2 replaces registry entries with proven solvers."""
    solver = SOLVERS.get(canonical_category(category), _defer)
    return solver(prompt)


def _self_check() -> None:
    for category in CATEGORIES:
        assert solve(category, "What is 2 + 2?") is None
    assert solve("math", "legacy alias") is None
    assert set(SOLVERS) == set(CATEGORIES)


if __name__ == "__main__":
    _self_check()
    print("gate stub self-check passed")
