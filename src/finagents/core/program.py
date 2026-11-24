"""The FinQA program language: parse, never eval.

A program is a comma-separated list of calls, e.g.
``subtract(1200, 1000), divide(#0, 1000)``: ``#0`` is the result of step 0, ``const_100``
is the constant 100. Table operations take a row label: ``table_max(net revenue, none)``.
Only the operations observed in the official FinQA release are accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARITHMETIC = frozenset({"add", "subtract", "multiply", "divide", "exp", "greater"})
TABLE = frozenset({"table_average", "table_max", "table_min", "table_sum"})
OPERATIONS = ARITHMETIC | TABLE

_CALL = re.compile(r"^(?P<op>[a-z_]+)\((?P<args>.*)\)$")
NUMERIC_LITERAL = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)%?")
_CONSTANT = re.compile(r"const_(?:m)?\d+")
_REFERENCE = re.compile(r"#\d+")


@dataclass(frozen=True)
class Step:
    operation: str
    arguments: tuple[str, str]
    kinds: tuple[str, str]  # numeric_literal | named_constant | intermediate_reference | ...


@dataclass(frozen=True)
class ParsedProgram:
    steps: tuple[Step, ...]
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None

    @property
    def literals(self) -> list[str]:
        return [
            argument
            for step in self.steps
            for argument, kind in zip(step.arguments, step.kinds, strict=True)
            if kind == "numeric_literal"
        ]


def argument_kind(argument: str) -> str:
    if NUMERIC_LITERAL.fullmatch(argument):
        return "numeric_literal"
    if _CONSTANT.fullmatch(argument):
        return "named_constant"
    if _REFERENCE.fullmatch(argument):
        return "intermediate_reference"
    return "unsupported"


def _split_calls(program: str) -> list[str] | None:
    """Split top-level calls on commas outside parentheses; None if unbalanced."""
    parts, depth, start = [], 0, 0
    for position, character in enumerate(program):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            parts.append(program[start:position].strip())
            start = position + 1
    if depth != 0:
        return None
    parts.append(program[start:].strip())
    return parts


def parse_program(program: str) -> ParsedProgram:
    """Parse a program into steps, or return the first syntax error."""
    if not isinstance(program, str) or not program.strip():
        return ParsedProgram((), "program must be a non-empty string")
    calls = _split_calls(program)
    if calls is None:
        return ParsedProgram((), "unbalanced parentheses")
    steps: list[Step] = []
    for index, call in enumerate(calls):
        match = _CALL.fullmatch(call)
        if match is None:
            return ParsedProgram((), f"invalid call syntax: {call!r}")
        operation = match["op"]
        if operation not in OPERATIONS:
            return ParsedProgram((), f"unsupported operation: {operation!r}")
        arguments = tuple(argument.strip() for argument in match["args"].split(","))
        if len(arguments) != 2 or not all(arguments):
            return ParsedProgram((), f"expected two arguments in {call!r}")
        if operation in TABLE and arguments[1] == "none":
            kinds = ("table_label", "none")
        else:
            kinds = tuple(argument_kind(argument) for argument in arguments)
            if "unsupported" in kinds:
                return ParsedProgram((), f"unsupported argument in {call!r}")
            for argument, kind in zip(arguments, kinds, strict=True):
                if kind == "intermediate_reference" and int(argument[1:]) >= index:
                    return ParsedProgram((), f"{argument} refers to a later step in {call!r}")
        steps.append(Step(operation, arguments, kinds))  # type: ignore[arg-type]
    return ParsedProgram(tuple(steps))
