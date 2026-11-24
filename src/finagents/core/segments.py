"""Splitting a report into retrievable segments and reading the numbers in them.

A segment is one table row (``table_n``) or one sentence (``text_n``, counted over
``pre_text + post_text``): the same keys FinQA uses for gold evidence, so retrieval can be
scored against it. A value mention is one number in a segment with what describes it: its
row label and column for a table cell, the nearby words for a sentence, plus period and unit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .values import NUMBER_SPAN, YEAR, literal, parse_value, unit_of

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
STOPWORDS = frozenset(
    """a an and are as at be been by during for from had has have in into is it its of on or
    over than that the their them then there these this those to was were what when which
    while who will with within how much many did does do""".split()
)
_TEXT_WINDOW = 8  # words on each side of a number that describe it


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens without stopwords; commas inside numbers dropped."""
    return [t for t in _TOKEN.findall(text.lower().replace(",", "")) if t not in STOPWORDS]


@dataclass(frozen=True)
class Segment:
    key: str
    source: str  # "table" | "text"
    text: str
    cells: tuple[str, ...] = ()
    header: tuple[str, ...] = ()

    @property
    def retrieval_text(self) -> str:
        """Table rows are rendered with their column headers so years and units are visible."""
        if self.source != "table" or self.key == "table_0" or not self.header:
            return self.text
        pairs = [f"{h}: {c}" for h, c in zip(self.header[1:], self.cells[1:], strict=False)]
        return f"{self.cells[0]} | " + " ; ".join(pairs)


@dataclass(frozen=True)
class Mention:
    segment_key: str
    source: str
    literal: str
    value: Decimal
    label: str
    column: str
    periods: tuple[str, ...]
    unit: str
    is_percent: bool


def build_segments(payload: Mapping[str, Any]) -> list[Segment]:
    """All table rows (header included) and sentences of an inference payload."""
    table = payload["table"]
    header = tuple(str(cell) for cell in table[0]) if table else ()
    segments = [
        Segment(f"table_{i}", "table", " | ".join(map(str, row)), tuple(map(str, row)), header)
        for i, row in enumerate(table)
    ]
    sentences = list(payload["pre_text"]) + list(payload["post_text"])
    segments += [Segment(f"text_{i}", "text", str(s)) for i, s in enumerate(sentences)]
    return segments


def table_mentions(segment: Segment, table_context: str = "") -> list[Mention]:
    if segment.source != "table" or len(segment.cells) < 2:
        return []
    label = segment.cells[0]
    # In key-value tables the first row holds data, so its cells have no column header.
    columns = segment.header[1:] if segment.key != "table_0" else ("",) * (len(segment.cells) - 1)
    mentions = []
    for column, raw in zip(columns, segment.cells[1:], strict=False):
        parsed = parse_value(raw)
        if parsed is None:
            continue
        mentions.append(
            Mention(
                segment_key=segment.key,
                source="table",
                literal=literal(parsed.value),
                value=parsed.value,
                label=label,
                column=column,
                periods=tuple(dict.fromkeys(YEAR.findall(f"{column} {label}"))),
                unit=unit_of(
                    raw,
                    column,
                    label,
                    table_context,
                    is_percent=parsed.is_percent,
                    currency=parsed.currency,
                ),
                is_percent=parsed.is_percent,
            )
        )
    return mentions


def text_mentions(segment: Segment) -> list[Mention]:
    if segment.source != "text":
        return []
    text = segment.text
    years = [(match.start(), match.group()) for match in YEAR.finditer(text)]
    mentions = []
    for match in NUMBER_SPAN.finditer(text):
        raw = match.group()
        after_text = text[match.end() :]
        is_percent = raw.endswith("%") or after_text.lstrip().startswith("%")
        parsed = parse_value(raw.rstrip("%") + ("%" if is_percent else ""))
        if parsed is None:
            continue
        before = text[: match.start()].split()[-_TEXT_WINDOW:]
        after = after_text.split()[:_TEXT_WINDOW]
        window = " ".join([*before, *after])
        nearest = sorted(years, key=lambda year: abs(year[0] - match.start()))
        mentions.append(
            Mention(
                segment_key=segment.key,
                source="text",
                literal=literal(parsed.value),
                value=parsed.value,
                label=window,
                column="",
                periods=(nearest[0][1],) if nearest else (),
                unit=unit_of(
                    raw,
                    " ".join(after[:3]),
                    is_percent=is_percent,
                    currency=parsed.currency or ("$" if before and before[-1] == "$" else None),
                ),
                is_percent=is_percent,
            )
        )
    return mentions


def extract_mentions(segments: Iterable[Segment]) -> list[Mention]:
    segments = list(segments)
    header = next((s for s in segments if s.key == "table_0"), None)
    table_context = header.text if header else ""
    mentions: list[Mention] = []
    for segment in segments:
        if segment.source == "table":
            mentions += table_mentions(segment, table_context)
        else:
            mentions += text_mentions(segment)
    return mentions


def label_match(terms: Iterable[str], mention: Mention) -> float:
    """How well a mention's label covers the query terms, in [0, 1].

    Row-label hits count fully, column hits half; long labels that match by accident (a whole
    sentence against a two-word row label) are discounted.
    """
    terms = list(terms)
    if not terms:
        return 0.0
    label_tokens = set(tokenize(mention.label))
    column_tokens = set(tokenize(mention.column)) - label_tokens
    hits = sum(1.0 if t in label_tokens else 0.5 if t in column_tokens else 0.0 for t in terms)
    coverage = hits / len(terms)
    precision = hits / max(len(label_tokens), 1)
    return round(0.75 * coverage + 0.25 * min(precision, 1.0), 6)
