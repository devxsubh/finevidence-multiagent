"""Agent 2 - Query Analyst: turn the question into a structured task.

Rules decide the operation from the question's wording (written from train questions only),
the years named in it, and the search terms for the metric. With an LLM enabled, Claude
proposes the same ``QueryAnalysis``; it is accepted only if it is consistent with the
question (periods must appear in it, the metric must have search terms), else the rules win.
"""

from __future__ import annotations

import re

from finagents.core.segments import tokenize
from finagents.core.values import YEAR
from finagents.state import AgentState, Operation, QueryAnalysis

from .base import Agent

# Checked in order; the first match decides. Words like "maximum" or "average" often name
# an entity on train ("minimum lease payments"), so those rules come after change and ratio.
_RULES: tuple[tuple[Operation, re.Pattern[str]], ...] = (
    (Operation.PERCENTAGE_CHANGE, re.compile(
        r"\b(growth rate|rate of growth|percent(age)? (change|increase|decrease|decline|growth)"
        r"|% (change|increase|decrease)|change in percent(age)?|(increase|decrease|decline|"
        r"change|growth) in percent(age)?|percentage growth|cagr|grow by what percent"
        r"|by what percent(age)?|by how much (did|has|have|was|were|is|does))\b")),
    (Operation.COMPARISON, re.compile(
        r"^(is|was|were|are|did|does|do)\b.*\b(greater|more|higher|less|lower|exceed)")),
    (Operation.CHANGE, re.compile(
        r"\b(change|difference|variation|increase|decrease|decline|growth|grew|"
        r"reduction|fluctuation|how much (more|less|higher|lower))\b")),
    (Operation.RATIO, re.compile(
        r"\b(percent(age)?|portion|proportion|ratio|share|fraction|rate|margin|%|what part)\b")),
    (Operation.AVERAGE, re.compile(r"\b(average|mean)\b")),
    (Operation.ROW_MAX, re.compile(r"\b(maximum|highest|largest|greatest|peak)\b")),
    (Operation.ROW_MIN, re.compile(r"\b(minimum|lowest|smallest)\b")),
    (Operation.SUM, re.compile(r"\b(total|combined|sum|together|aggregate|in all)\b")),
)  # fmt: skip

_TASK_WORDS = frozenset(
    """percent percentage change changes difference variation increase increased decrease
    decreased decline growth grew rate ratio portion proportion share fraction total combined
    sum average mean maximum minimum highest lowest largest smallest greatest between
    compared value amount year years period annual approximate approximately based
    respectively millions million billions billion thousands thousands dollars usd""".split()
)
_RATIO_SPLITS = (
    re.compile(r"\bratio of (?P<part>.+?) to (?P<whole>.+)"),
    re.compile(
        r"(?P<part>.+?)\b(?:as|is|was) an? (?:percent(?:age)?|portion|share|proportion"
        r"|fraction) of (?P<whole>.+)"
    ),
    re.compile(
        r"\b(?:percent(?:age)?|portion|proportion|share|fraction|%) of (?:the )?"
        r"(?P<whole>.+?) (?:is|was|were|are|came|comes|is from|was from|related to"
        r"|relates to|attributable to|consisted of|made up of|represented by|due to)"
        r" (?P<part>.+)"
    ),
)
_REQUIRED = {
    Operation.PERCENTAGE_CHANGE: 2, Operation.CHANGE: 2, Operation.COMPARISON: 2,
    Operation.RATIO: 2, Operation.SUM: 2, Operation.AVERAGE: 2,
    Operation.ROW_MAX: 1, Operation.ROW_MIN: 1, Operation.UNSUPPORTED: 0,
}  # fmt: skip


def classify(question: str) -> Operation:
    text = question.lower()
    return next((op for op, pattern in _RULES if pattern.search(text)), Operation.UNSUPPORTED)


def entity_terms(text: str) -> list[str]:
    """Content words of the question without task words, years or bare numbers."""
    terms = [
        t for t in tokenize(text)
        if t not in _TASK_WORDS and not YEAR.fullmatch(t) and not t[0].isdigit()
    ]  # fmt: skip
    return list(dict.fromkeys(terms))


def ratio_terms(question: str) -> tuple[list[str], list[str]]:
    text = question.lower()
    for pattern in _RATIO_SPLITS:
        match = pattern.search(text)
        if match:
            part, whole = entity_terms(match["part"]), entity_terms(match["whole"])
            if part and whole:
                return part, whole
    return [], []


def analyze_with_rules(question: str) -> QueryAnalysis:
    operation = classify(question)
    terms = entity_terms(question)
    part, whole = ratio_terms(question) if operation is Operation.RATIO else ([], [])
    periods = sorted(set(YEAR.findall(question)))
    return QueryAnalysis(
        metric=" ".join(terms) or "the asked quantity",
        entity_terms=terms,
        periods=periods,
        operation=operation,
        part_terms=part,
        whole_terms=whole,
        required_evidence=max(
            _REQUIRED[operation],
            len(periods) if operation in {Operation.SUM, Operation.AVERAGE} else 0,
        ),
        rationale=f"rule match: {operation.value}; periods {periods or 'none named'}",
    )


SYSTEM_PROMPT = """You analyze questions about company financial reports.
Return the calculation the question needs, choosing exactly one operation:
percentage_change ((new-old)/old), change (new-old), ratio (part/whole), sum, average,
row_max / row_min (largest / smallest value in one table row), comparison (is A greater
than B), or unsupported (anything else, e.g. several chained calculations).
metric: the quantity asked about, as words from the question. entity_terms: 1-6 lowercase
search words for it (no years, no calculation words). periods: only years written in the
question, oldest first. For ratios also give part_terms and whole_terms.
required_evidence: how many numbers the calculation needs. Do not answer the question."""


class QueryAnalyst(Agent):
    name = "query_analyst"
    responsibility = "Understand the question: metric, periods and operation"
    tools = ("rule_parser", "llm_structured_output")

    def run(self, state: AgentState):
        question = state["payload"]["question"]
        rules = analyze_with_rules(question)
        analysis, note = rules, "rules"
        if self.llm:
            proposed = self.llm.structured(
                system=SYSTEM_PROMPT, prompt=f"Question: {question}", schema=QueryAnalysis
            )
            accepted, why = self._validate(proposed, question)
            if accepted:
                analysis = accepted.model_copy(update={"backend": self.llm_backend()})
                note = self.llm_backend()
            else:
                note = f"rules (LLM output rejected: {why or self.llm.last_error})"
        summary = (
            f"{analysis.operation.value} of '{analysis.metric}'"
            f" for {', '.join(analysis.periods) or 'no named period'}"
        )
        return (
            {"analysis": analysis, "research_round": 1},
            summary,
            {
                "analysis": analysis.model_dump(mode="json"),
                "backend": note,
            },
        )

    @staticmethod
    def _validate(proposed: QueryAnalysis | None, question: str):
        if proposed is None:
            return None, None
        question_years = set(YEAR.findall(question))
        periods = sorted(p for p in proposed.periods if p in question_years)
        terms = [t for t in (t.lower().strip() for t in proposed.entity_terms) if t]
        if not terms and proposed.operation is not Operation.UNSUPPORTED:
            return None, "no search terms"
        fixed = proposed.model_copy(update={"periods": periods, "entity_terms": terms})
        return fixed, None
