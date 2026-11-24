"""Parsing financial values ("$ 1,234.5", "( 12 )", "4.3%") and their units, without eval.

Values are parsed to ``Decimal`` exactly as written. Nothing is rescaled: "in millions" is
recorded as the unit, never multiplied out, matching how FinQA programs use raw numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_SPAN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)%?(?![\w.])")

_VALUE = re.compile(
    r"^\s*(?P<open>\()?\s*(?P<currency>[$€£])?\s*(?P<sign>[+-])?\s*"
    r"(?P<number>(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))\s*(?P<percent>%?)\s*(?P<close>\))?\s*$"
)
# FinQA cells often repeat a value in parentheses: "$ -4063 ( 4063 )", "32.26% ( 32.26 % )".
_ECHOED_VALUE = re.compile(r"(?<=\d|%)\s*\(\s*[\d.,]+\s*%?\s*\)\s*$")
_SCALE = re.compile(r"\b(thousands?|millions?|billions?)\b", re.I)
_UNAVAILABLE = frozenset({"", "-", "—", "–", "n/a", "na", "none", "not available"})


@dataclass(frozen=True)
class ParsedValue:
    value: Decimal
    is_percent: bool
    currency: str | None


def clean_cell(raw: str) -> str:
    """Drop a trailing parenthesised echo of the cell's own value."""
    return _ECHOED_VALUE.sub("", raw).strip()


def parse_value(raw: str) -> ParsedValue | None:
    """Parse one table cell or number span; ``None`` for years, dates, text and blanks."""
    text = clean_cell(raw)
    if text.lower() in _UNAVAILABLE or YEAR.fullmatch(text):
        return None
    match = _VALUE.fullmatch(text)
    if match is None or bool(match["open"]) != bool(match["close"]):
        return None
    try:
        value = Decimal(match["number"].replace(",", ""))
    except InvalidOperation:
        return None
    if match["open"] or match["sign"] == "-":
        value = -value  # accounting parentheses or an explicit minus
    return ParsedValue(value, bool(match["percent"]), match["currency"])


def literal(value: Decimal) -> str:
    """Render a value as a FinQA program literal, e.g. ``-1970`` or ``14.7``."""
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def unit_of(*contexts: str, is_percent: bool = False, currency: str | None = None) -> str:
    """A short unit label from the cell and its context: "%", "$ millions", "millions", ..."""
    if is_percent:
        return "%"
    scale = next(
        (m.group(1).lower().rstrip("s") + "s" for c in contexts if (m := _SCALE.search(c))), ""
    )
    symbol = currency or ("$" if any("$" in context for context in contexts) else "")
    return " ".join(part for part in (symbol, scale) if part) or "number"


def units_compatible(left: str, right: str) -> bool:
    """Two units can be combined in one calculation: percents only with percents."""
    if (left == "%") != (right == "%"):
        return False
    if left == "number" or right == "number":
        return True
    scale = lambda unit: unit.replace("$", "").strip()  # noqa: E731
    return not scale(left) or not scale(right) or scale(left) == scale(right)
