"""Deterministic execution of FinQA programs under the official evaluator's value rules.

The official script executes with Python floats, rounds a numeric final result to five
decimals and compares it for exact equality with ``exe_ans``. This executor reproduces
those rules on top of the parser. It never calls ``eval`` or ``exec``; every step is one of
the fixed operations below, so a model can choose a calculation but cannot run code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .program import parse_program

ROUNDING_DECIMALS = 5
Value = float | str  # str only for "yes" / "no" from greater()


@dataclass(frozen=True)
class ExecutionStep:
    index: int
    operation: str
    arguments: tuple[str, str]
    operands: tuple[float, ...]
    result: Value


@dataclass(frozen=True)
class ExecutionResult:
    value: Value | None
    steps: tuple[ExecutionStep, ...] = field(default=())
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None


def to_number(token: str) -> float | None:
    """Convert a program or table token as the official evaluator does."""
    token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        pass
    if "%" in token:
        try:
            return float(token.replace("%", "")) / 100.0
        except ValueError:
            return None
    if token.startswith("const_"):
        constant = token.removeprefix("const_")
        return -1.0 if constant == "m1" else _float_or_none(constant)
    return None


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _row_values(row: Sequence[object]) -> list[float] | None:
    values = []
    for cell in row:
        number = to_number(str(cell).replace("$", "").strip().split("(")[0].strip())
        if number is None:
            return None
        values.append(number)
    return values


_ARITHMETIC = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a**b,
    "greater": lambda a, b: "yes" if a > b else "no",
}
_TABLE = {
    "table_max": max,
    "table_min": min,
    "table_sum": sum,
    "table_average": lambda values: sum(values) / len(values),
}


def execute_program(program: str, table: Sequence[Sequence[object]]) -> ExecutionResult:
    """Execute a program against its table; errors are returned, never raised."""
    parsed = parse_program(program)
    if not parsed.is_valid:
        return ExecutionResult(None, error=f"parse error: {parsed.error}")
    rows = {str(row[0]): row[1:] for row in table if row}  # a later duplicate label wins
    results: list[Value] = []
    steps: list[ExecutionStep] = []
    for index, step in enumerate(parsed.steps):
        if step.operation in _TABLE:
            label = step.arguments[0]
            values = _row_values(rows[label]) if label in rows else None
            if not values:
                return ExecutionResult(None, tuple(steps), f"no numeric table row {label!r}")
            result: Value = _TABLE[step.operation](values)
            steps.append(
                ExecutionStep(index, step.operation, step.arguments, tuple(values), result)
            )
            results.append(result)
            continue
        operands: list[float] = []
        for argument in step.arguments:
            if argument.startswith("#"):
                previous = results[int(argument[1:])]
                if isinstance(previous, str):
                    return ExecutionResult(None, tuple(steps), f"{argument} is not a number")
                operands.append(previous)
            else:
                number = to_number(argument)
                if number is None:
                    return ExecutionResult(None, tuple(steps), f"non-numeric argument {argument!r}")
                operands.append(number)
        try:
            result = _ARITHMETIC[step.operation](*operands)
        except (ZeroDivisionError, OverflowError) as error:
            return ExecutionResult(None, tuple(steps), f"arithmetic error: {error}")
        if isinstance(result, complex):
            return ExecutionResult(None, tuple(steps), "complex result")
        steps.append(ExecutionStep(index, step.operation, step.arguments, tuple(operands), result))
        results.append(result)
    final = results[-1]
    value = final if isinstance(final, str) else round(final, ROUNDING_DECIMALS)
    return ExecutionResult(value, tuple(steps))
