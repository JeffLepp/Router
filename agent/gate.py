from __future__ import annotations

from collections.abc import Callable

from agent.classify import CATEGORIES, canonical_category
from agent.solvers import (
    arithmetic,
    code_solve,
    logic_solve,
    ner_solve,
    sentiment_solve,
    summary_solve,
    wordmath,
)


Solver = Callable[[str], str | None]


def _defer(prompt: str) -> str | None:
    return None


def _first(*solvers: Solver) -> Solver:
    def solve(prompt: str) -> str | None:
        for solver in solvers:
            answer = solver(prompt)
            if answer is not None:
                return answer
        return None

    return solve


SOLVERS: dict[str, Solver] = {
    "actual_qa": _defer,
    "math_reasoning": _first(arithmetic.solve, wordmath.solve),
    "sentiment_analysis": sentiment_solve.solve,
    "summarization": summary_solve.solve,
    "named_entity_recognition": ner_solve.solve,
    "code_debugging": code_solve.solve,
    "logic_puzzles": logic_solve.solve,
    "code_generation": code_solve.solve,
}


def solve(category: str, prompt: str) -> str | None:
    """Return a mechanically proven answer, or None to escalate."""
    solver = SOLVERS.get(canonical_category(category), _defer)
    return solver(prompt)


def _self_check() -> None:
    assert solve("math_reasoning", "What is 2 + 2?") == "4"
    assert solve("math", "What is 2 + 2?") == "4"
    assert solve("sentiment_analysis", "I absolutely loved it.") is not None
    assert solve("summarization", "Summarize this paragraph.") is None
    assert set(SOLVERS) == set(CATEGORIES)


if __name__ == "__main__":
    _self_check()
    print("gate solver self-check passed")
