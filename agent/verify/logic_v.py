from __future__ import annotations

from collections import Counter


def normalize_assignment(answer: str) -> str:
    return " ".join(answer.lower().replace("\n", " ").split())


def agreement_score(answers: list[str]) -> float:
    if not answers:
        return 0.0
    normalized = [normalize_assignment(answer) for answer in answers if answer.strip()]
    if not normalized:
        return 0.0
    count = Counter(normalized).most_common(1)[0][1]
    return count / len(answers)


def majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized = [(normalize_assignment(answer), answer) for answer in answers]
    winner = Counter(item[0] for item in normalized).most_common(1)[0][0]
    for norm, original in normalized:
        if norm == winner:
            return original
    return answers[0]


def constraints_consistent(edges: list[tuple[str, str]]) -> bool:
    graph: dict[str, set[str]] = {}
    for before, after in edges:
        graph.setdefault(before, set()).add(after)
        graph.setdefault(after, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for child in graph[node]:
            if not visit(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in graph)


def _self_check() -> None:
    assert normalize_assignment("A = 1\nB = 2") == "a = 1 b = 2"
    assert agreement_score(["x", "x", "y"]) == 2 / 3
    assert agreement_score([]) == 0.0
    assert majority_answer(["one", "two", "one"]) == "one"
    assert majority_answer([]) == ""
    assert constraints_consistent([("A", "B"), ("B", "C")])
    assert not constraints_consistent([("A", "B"), ("B", "A")])


if __name__ == "__main__":
    _self_check()
    print("logic_v self-check passed")
