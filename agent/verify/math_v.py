from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_final_number(answer: str) -> Decimal | None:
    matches = NUMBER_RE.findall(answer)
    if not matches:
        return None
    return Decimal(matches[-1].replace(",", ""))


def safe_eval(expr: str) -> Decimal:
    try:
        import sympy  # type: ignore

        value = sympy.N(sympy.sympify(expr))
        return Decimal(str(value))
    except ModuleNotFoundError:
        return Decimal(str(_ast_eval(expr)))


def recompute_matches(answer: str, expr: str, tolerance: Decimal = Decimal("0.0001")) -> bool:
    found = extract_final_number(answer)
    if found is None:
        return False
    expected = safe_eval(expr)
    return abs(found - expected) <= tolerance


def _ast_eval(expr: str) -> float:
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("unsafe expression")

    return walk(ast.parse(expr, mode="eval"))


def _self_check() -> None:
    assert extract_final_number("The answer is 42.") == Decimal("42")
    assert extract_final_number("First 1, then 2.5") == Decimal("2.5")
    assert recompute_matches("Final: 9", "4+5")
    assert recompute_matches("Final: 6", "2*3")
    assert not recompute_matches("Final: 7", "2*3")


if __name__ == "__main__":
    _self_check()
    print("math_v self-check passed")

