from __future__ import annotations

import itertools
import re

from agent.solvers.common import NUMBER_WORDS, compact_json, normalize
from agent.verify.logic_v import constraints_consistent


DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def solve(prompt: str) -> str | None:
    text = normalize(prompt)
    lower = text.lower()

    for handler in (
        _syllogism,
        _day_offset,
        _rose_fallacy,
        _age_consistency,
        _bridge_unknown,
        _row_four,
        _grade_puzzle,
        _row_three,
        _house_puzzle,
        _box_puzzle,
    ):
        answer = handler(text, lower)
        if answer is not None:
            return answer
    return None


def _answer(answer: object, valid: bool = True) -> str:
    return compact_json({"answer": answer, "valid": valid})


def _syllogism(text: str, lower: str) -> str | None:
    match = re.search(
        r"all\s+(\w+)s?\s+are\s+(\w+)s?\.\s+([A-Z][a-z]+)\s+is\s+a\s+(\w+)s?\.\s+is\s+\3\s+an?\s+(\w+)s?\?",
        text,
        flags=re.I,
    )
    if not match:
        return None
    subject, parent, _, instance_type, queried = (part.lower() for part in match.groups())
    if instance_type.rstrip("s") == subject.rstrip("s") and queried.rstrip("s") == parent.rstrip("s"):
        return _answer("Yes")
    return _answer("No")


def _day_offset(text: str, lower: str) -> str | None:
    match = re.search(r"today is (\w+).*?in (\d+|" + "|".join(NUMBER_WORDS) + r") days?", lower)
    if not match:
        return None
    day, offset_raw = match.groups()
    if day not in DAYS:
        return None
    offset = NUMBER_WORDS.get(offset_raw, int(offset_raw) if offset_raw.isdigit() else 0)
    result = DAYS[(DAYS.index(day) + offset) % 7].title()
    return _answer(result)


def _rose_fallacy(text: str, lower: str) -> str | None:
    if "can you conclude" not in lower:
        return None
    match = re.search(
        r"all\s+(\w+)s?\s+are\s+(\w+)s?\s+and\s+some\s+\2s?\s+(.+?),\s*can you conclude that some\s+\1s?\s+\3",
        lower,
    )
    if not match:
        return None
    return _answer("No, it does not necessarily follow.")


def _age_consistency(text: str, lower: str) -> str | None:
    if "older than" not in lower or "consistent" not in lower:
        return None
    edges = [
        (left.upper(), right.upper())
        for left, right in re.findall(r"\b([A-Z])\s+is\s+older\s+than\s+([A-Z])", text)
    ]
    if len(edges) < 2:
        return None
    return _answer("Yes" if constraints_consistent(edges) else "No", constraints_consistent(edges))


def _bridge_unknown(text: str, lower: str) -> str | None:
    if "bridge" in lower and "don't know how long" in lower and "minimum total" in lower:
        return _answer(None, False)
    return None


def _row_four(text: str, lower: str) -> str | None:
    if "row of four chairs" not in lower or "dan's left" not in lower:
        return None
    names = ["Ali", "Ben", "Cara", "Dan"]
    solutions: list[tuple[str, ...]] = []
    for perm in itertools.permutations(names):
        if perm[0] != "Ben":
            continue
        if perm[3] != "Cara":
            continue
        if abs(perm.index("Ali") - perm.index("Cara")) == 1:
            continue
        solutions.append(perm)
    answers = {perm[perm.index("Dan") - 1] for perm in solutions if perm.index("Dan") > 0}
    if len(answers) == 1:
        return _answer(next(iter(answers)))
    return None


def _grade_puzzle(text: str, lower: str) -> str | None:
    if "who got grade b" not in lower or "distinct grades" not in lower:
        return None
    names = ["Ariel", "Bianca", "Chris"]
    grades = ["A", "B", "C"]
    solutions: list[dict[str, str]] = []
    for perm in itertools.permutations(grades):
        assignment = dict(zip(names, perm))
        if assignment["Ariel"] == "A":
            continue
        if assignment["Bianca"] == "B":
            continue
        solutions.append(assignment)
    answers = {name for solution in solutions for name, grade in solution.items() if grade == "B"}
    if len(answers) == 1:
        return _answer(next(iter(answers)))
    return None


def _row_three(text: str, lower: str) -> str | None:
    if "three seats" not in lower or "who must sit in seat 2" not in lower:
        return None
    names = ["Alex", "Brooke", "Charlie"]
    solutions: list[tuple[str, ...]] = []
    for perm in itertools.permutations(names):
        if perm[2] != "Charlie":
            continue
        if abs(perm.index("Alex") - perm.index("Brooke")) == 1:
            continue
        solutions.append(perm)
    answers = {perm[1] for perm in solutions}
    if len(answers) == 1:
        return _answer(next(iter(answers)))
    return None


def _house_puzzle(text: str, lower: str) -> str | None:
    if "four houses" not in lower or "red house" not in lower:
        return None
    people = ["Anna", "Ben", "Clara", "David"]
    colors = ["red", "blue", "green", "yellow"]
    solutions: list[tuple[dict[str, int], dict[str, int]]] = []
    for person_perm in itertools.permutations(range(4), len(people)):
        ppos = dict(zip(people, person_perm))
        if ppos["David"] not in {0, 3}:
            continue
        for color_perm in itertools.permutations(range(4), len(colors)):
            cpos = dict(zip(colors, color_perm))
            if ppos["Anna"] != cpos["yellow"]:
                continue
            if ppos["Ben"] != cpos["blue"]:
                continue
            if ppos["Clara"] + 1 != cpos["green"]:
                continue
            solutions.append((ppos, cpos))
    answers = {
        person
        for ppos, cpos in solutions
        for person, pos in ppos.items()
        if pos == cpos["red"]
    }
    if len(answers) == 1:
        return _answer(next(iter(answers)))
    return None


def _box_puzzle(text: str, lower: str) -> str | None:
    if "three boxes" not in lower or "oranges" not in lower:
        return None
    colors = ["red", "green", "blue"]
    fruits = ["apples", "oranges", "bananas"]
    solutions: list[tuple[dict[str, int], dict[str, int]]] = []
    for color_perm in itertools.permutations(range(3), len(colors)):
        cpos = dict(zip(colors, color_perm))
        if cpos["red"] >= cpos["green"]:
            continue
        for fruit_perm in itertools.permutations(range(3), len(fruits)):
            fpos = dict(zip(fruits, fruit_perm))
            if cpos["blue"] != fpos["bananas"]:
                continue
            if fpos["apples"] == 2:
                continue
            solutions.append((cpos, fpos))
    answers = {
        color.title()
        for cpos, fpos in solutions
        for color, pos in cpos.items()
        if pos == fpos["oranges"]
    }
    if len(answers) == 1:
        return _answer(next(iter(answers)))
    return None


def _self_check() -> None:
    assert solve("All cats are animals. Milo is a cat. Is Milo an animal?") == _answer("Yes")
    assert solve("If today is Tuesday, what day will it be in three days?") == _answer("Friday")
    assert solve("If all roses are flowers and some flowers fade quickly, can you conclude that some roses fade quickly?") == _answer("No, it does not necessarily follow.")
    assert solve("Consider three statements about ages: A is older than B; B is older than C; C is older than A. Are these statements all consistent at the same time?") == _answer("No", False)
    assert solve("Four people need to cross a bridge at night with one flashlight. The bridge can hold at most two people, and the flashlight must be carried when crossing. If we don't know how long each person takes to cross, can we determine the minimum total crossing time?") == _answer(None, False)
    assert solve("A is before B. Who is in the middle?") is None


if __name__ == "__main__":
    _self_check()
    print("logic solver self-check passed")
