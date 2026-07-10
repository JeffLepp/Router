from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Hashable


Value = Hashable
Assignment = dict[str, Value]
Predicate = Callable[[Assignment], bool]


@dataclass(frozen=True)
class Constraint:
    scope: tuple[str, ...]
    predicate: Predicate

    def accepts(self, assignment: Assignment) -> bool:
        return not all(name in assignment for name in self.scope) or self.predicate(assignment)


@dataclass
class FiniteDomainProblem:
    """Tiny dependency-free CSP enumerator for the small logic prompts in this track."""

    variables: dict[str, tuple[Value, ...]] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)

    def add_variable(self, name: str, domain: Iterable[Value]) -> None:
        values = tuple(dict.fromkeys(domain))
        if not name or not values or name in self.variables:
            raise ValueError(f"invalid CSP variable {name!r}")
        self.variables[name] = values

    def add_constraint(self, scope: Sequence[str], predicate: Predicate) -> None:
        names = tuple(scope)
        if not names or any(name not in self.variables for name in names):
            raise ValueError("constraint references an unknown variable")
        self.constraints.append(Constraint(names, predicate))

    def add_all_different(self, names: Sequence[str]) -> None:
        scope = tuple(names)

        def different(assignment: Assignment) -> bool:
            values = [assignment[name] for name in scope]
            return len(values) == len(set(values))

        self.add_constraint(scope, different)

    def solutions(self, limit: int = 10000) -> list[Assignment]:
        if limit < 1:
            return []
        order = sorted(self.variables, key=lambda name: (len(self.variables[name]), name))
        out: list[Assignment] = []
        assignment: Assignment = {}

        def search(index: int) -> None:
            if len(out) >= limit:
                return
            if index == len(order):
                out.append(dict(assignment))
                return
            name = order[index]
            for value in self.variables[name]:
                assignment[name] = value
                if all(constraint.accepts(assignment) for constraint in self.constraints):
                    search(index + 1)
                assignment.pop(name, None)

        search(0)
        return out


def _self_check() -> None:
    problem = FiniteDomainProblem()
    for name in ("A", "B", "C"):
        problem.add_variable(name, range(3))
    problem.add_all_different(("A", "B", "C"))
    problem.add_constraint(("A", "B"), lambda a: int(a["A"]) < int(a["B"]))
    problem.add_constraint(("B", "C"), lambda a: int(a["B"]) + 1 == int(a["C"]))
    assert problem.solutions() == [{"A": 0, "B": 1, "C": 2}]


if __name__ == "__main__":
    _self_check()
    print("CSP self-check passed")
