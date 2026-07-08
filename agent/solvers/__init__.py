"""Deterministic prove-or-defer solvers for the zero-token gate."""

from __future__ import annotations

from collections.abc import Callable


Solver = Callable[[str], str | None]

