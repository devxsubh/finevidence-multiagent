"""Agent 5 - Reasoning Agent, and the Program Validator gate that follows it.

The Reasoning Agent writes calculation programs over evidence ids, never over numbers:
``subtract(E2, E1), divide(#0, E1)``. Ids are then replaced by the audited values. The
offline backend binds the task's slots (old/new period, part/whole, one value per period)
to the best-matching audited values; with an LLM enabled, Claude picks the operation and
the evidence ids first and the templates follow as alternatives.

The Program Validator is a deterministic gate: the program must parse, use only supported
operations, reference earlier steps only, and every number in it must be an audited value
or a named constant. An invalid program never reaches the calculator.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from finagents.core.program import parse_program
from finagents.core.segments import label_match
from finagents.core.values import YEAR
from finagents.state import (
    AgentState,
    CandidateProgram,
    EvidenceValue,
    Operation,
    QueryAnalysis,
    ValidationReport,
)

from .base import Agent

_CHANGE = {Operation.PERCENTAGE_CHANGE, Operation.CHANGE, Operation.COMPARISON}
_TOTAL = re.compile(r"\btotal\b", re.I)
_UNSAFE_LABEL = re.compile(r"[(),]")
_ID = re.compile(r"\bE\d+\b")
# A bare number used as an argument, e.g. "subtract(1200, E1)"; ids, constants and #n are fine.
_WRITTEN_NUMBER = re.compile(r"[(,]\s*-?(?:\d+(?:\.\d+)?|\.\d+)%?\s*(?=[,)])")
Scored = tuple[EvidenceValue, float]
MAX_GROUPS = 6


def _period(value: EvidenceValue) -> str:
    return value.periods[0] if value.periods else (value.column or "unlabelled period")


def _candidate(
    template: str,
    values: Sequence[EvidenceValue],
    score: float,
    why: str,
    proposer: str = "template",
) -> CandidateProgram:
    by_id = {v.id: v for v in values}
    program = _ID.sub(lambda m: by_id[m.group()].literal, template)
    return CandidateProgram(
        template=template,
        program=program,
        evidence_ids=list(dict.fromkeys(v.id for v in values)),
        score=round(score, 6),
        rationale=why,
        proposer=proposer,
    )


class TemplateReasoner:
    """Plan-driven slot filling over audited values (offline backend)."""

    def propose(
        self,
        analysis: QueryAnalysis,
        evidence: Sequence[EvidenceValue],
        table: Sequence[Sequence[str]],
        question: str,
    ) -> list[CandidateProgram]:
        if analysis.operation is Operation.UNSUPPORTED or not evidence:
            return []
        scored = [(v, v.relevance) for v in evidence]
        op = analysis.operation
        if op in _CHANGE:
            candidates = self._two_period(analysis, scored, question)
        elif op is Operation.RATIO:
            candidates = self._ratio(analysis, scored)
        elif op in {Operation.SUM, Operation.AVERAGE}:
            candidates = self._aggregate(analysis, scored, table)
        else:
            candidates = self._row_extreme(analysis, scored, table)
        unique: dict[str, CandidateProgram] = {}
        for c in sorted(candidates, key=lambda c: -c.score):
            unique.setdefault(c.program, c)
        return list(unique.values())

    @staticmethod
    def _groups(scored: Sequence[Scored]) -> list[list[Scored]]:
        """Values grouped by segment (one table row or sentence), best group first."""
        groups: dict[str, list[Scored]] = {}
        for value, score in scored:
            groups.setdefault(value.segment_key, []).append((value, score))
        return sorted(groups.values(), key=lambda g: -max(s for _, s in g))[:MAX_GROUPS]

    def _two_period(self, analysis: QueryAnalysis, scored, question: str):
        order = list(dict.fromkeys(YEAR.findall(question)))
        sources = [(_period_pairs(g, analysis.periods), 0.0) for g in self._groups(scored)]
        # The two periods may sit in different rows ("2015 revenue", "2016 revenue").
        sources.append((_cross_segment_pairs(scored, analysis.periods), -0.1))
        bonus = 0.2 if analysis.periods else 0.0
        out = []
        for pairs, adjustment in sources:
            for (old, old_s), (new, new_s) in pairs:
                if analysis.operation is Operation.COMPARISON:
                    first, second = (
                        (old, new)
                        if order and old.periods and old.periods[0] == order[0]
                        else (new, old)
                    )
                    template = f"greater({first.id}, {second.id})"
                elif analysis.operation is Operation.CHANGE:
                    template = f"subtract({new.id}, {old.id})"
                else:
                    template = f"subtract({new.id}, {old.id}), divide(#0, {old.id})"
                out.append(
                    _candidate(
                        template,
                        [old, new],
                        (old_s + new_s) / 2 + bonus + adjustment,
                        f"{analysis.operation.value} of '{new.label}' {_period(old)} -> {_period(new)}",
                    )
                )
        return out

    def _ratio(self, analysis: QueryAnalysis, scored):
        period = analysis.periods[-1] if analysis.periods else None
        pool = [(v, s) for v, s in scored if period is None or period in v.periods or not v.periods]
        if analysis.part_terms:
            bonus = {id(v): s - label_match(analysis.entity_terms, v) for v, s in pool}  # type: ignore[arg-type]
            parts_pool = [(v, label_match(analysis.part_terms, v) + bonus[id(v)]) for v, _ in pool]  # type: ignore[arg-type]
            whole_pool = [(v, label_match(analysis.whole_terms, v) + bonus[id(v)]) for v, _ in pool]  # type: ignore[arg-type]
        else:
            parts_pool = whole_pool = pool
        weight = 1.0 if analysis.whole_terms else 0.5
        out = []
        for part, part_s in sorted(parts_pool, key=lambda x: -x[1])[:MAX_GROUPS]:
            for whole, whole_s in whole_pool:
                if whole is part or whole.value == 0:
                    continue
                same_column = whole.column == part.column and whole.source == part.source
                if not (same_column or whole.segment_key == part.segment_key):
                    continue
                structure = (
                    0.35
                    if same_column and _TOTAL.search(whole.label) and not _TOTAL.search(part.label)
                    else 0.0
                )
                if whole.segment_key == part.segment_key and not same_column:
                    structure -= 0.15
                out.append(
                    _candidate(
                        f"divide({part.id}, {whole.id})",
                        [part, whole],
                        (part_s + weight * whole_s + structure) / (1 + weight),
                        f"ratio of '{part.label}' to '{whole.label}' {_period(part)}",
                    )
                )
        return out

    def _aggregate(self, analysis: QueryAnalysis, scored, table):
        groups = [(g, 0.0) for g in self._groups(scored)]
        if len(analysis.periods) >= 2:
            groups.append((_best_per_period(scored, analysis.periods), -0.1))
        out = []
        for group, adjustment in groups:
            values = _values_for_periods(group, analysis.periods)
            if len(values) < 2:
                continue
            ids = [v.id for v, _ in values]
            steps = [f"add({ids[0]}, {ids[1]})"] + [
                f"add(#{i}, {x})" for i, x in enumerate(ids[2:])
            ]
            template = ", ".join(steps)
            if analysis.operation is Operation.AVERAGE:
                template += f", divide(#{len(ids) - 2}, const_{len(ids)})"
            score = (
                sum(s for _, s in values) / len(values)
                + (0.2 if analysis.periods else 0.0)
                + adjustment
            )
            out.append(
                _candidate(
                    template,
                    [v for v, _ in values],
                    score,
                    f"{analysis.operation.value} of '{values[0][0].label}' over "
                    f"{', '.join(_period(v) for v, _ in values)}",
                )
            )
            if (
                analysis.operation is Operation.AVERAGE
                and not analysis.periods
                and _is_period_series(group)
            ):
                out += self._row_extreme(analysis, group, table, operation="table_average")
        return out

    def _row_extreme(self, analysis: QueryAnalysis, scored, table, operation: str | None = None):
        operation = operation or (
            "table_max" if analysis.operation is Operation.ROW_MAX else "table_min"
        )
        rows = {str(row[0]) for row in table if row}
        out = []
        for group in self._groups(scored):
            value, score = max(group, key=lambda x: x[1])
            if (
                value.source != "table"
                or _UNSAFE_LABEL.search(value.label)
                or value.label not in rows
            ):
                continue
            out.append(
                CandidateProgram(
                    template=f"{operation}({value.label}, none)",
                    program=f"{operation}({value.label}, none)",
                    evidence_ids=[v.id for v, _ in group],
                    score=round(score - (0.05 if operation == "table_average" else 0.0), 6),
                    rationale=f"{operation} over table row '{value.label}'",
                    proposer="template",
                )
            )
        return out


def _period_pairs(group: Sequence[Scored], periods: Sequence[str]) -> list[tuple[Scored, Scored]]:
    """(old, new) pairs from one row: named periods first, else adjacent dated columns."""
    if len(periods) >= 2:
        old = [x for x in group if periods[0] in x[0].periods]
        new = [x for x in group if periods[-1] in x[0].periods]
        return list(itertools.product(old[:1], new[:1]))
    dated = sorted((x for x in group if x[0].periods), key=lambda x: x[0].periods[0])
    if len(periods) == 1:
        named = [x for x in dated if x[0].periods[0] == periods[0]]
        earlier = [x for x in dated if x[0].periods[0] < periods[0]]
        return [(earlier[-1], named[0])] if named and earlier else []
    if len(dated) >= 2 and dated[-1][0].periods[0] != dated[-2][0].periods[0]:
        return [(dated[-2], dated[-1])]
    if len(group) >= 2 and group[0][0].source == "table":
        return [(group[1], group[0])]  # undated columns: FinQA lists the latest period first
    return []


def _cross_segment_pairs(
    scored: Sequence[Scored], periods: Sequence[str]
) -> list[tuple[Scored, Scored]]:
    if len(periods) >= 2:
        old_p, new_p = periods[0], periods[-1]
    elif len(periods) == 1:
        new_p, old_p = periods[0], str(int(periods[0]) - 1)
    else:
        return []

    def best(period: str) -> Scored | None:
        matches = [x for x in scored if period in x[0].periods]
        return max(matches, key=lambda x: x[1]) if matches else None

    old, new = best(old_p), best(new_p)
    if old is None or new is None or old[0].segment_key == new[0].segment_key:
        return []
    return [(old, new)]


def _best_per_period(scored: Sequence[Scored], periods: Sequence[str]) -> list[Scored]:
    chosen = []
    for period in periods:
        matches = [x for x in scored if period in x[0].periods]
        if matches:
            chosen.append(max(matches, key=lambda x: x[1]))
    return chosen


def _values_for_periods(group: Sequence[Scored], periods: Sequence[str]) -> list[Scored]:
    if len(periods) >= 2:
        chosen = []
        for period in periods:
            matches = [x for x in group if period in x[0].periods]
            if not matches:
                return []
            chosen.append(matches[0])
        return chosen
    return list(group) if _is_period_series(group) else []


def _is_period_series(group: Sequence[Scored]) -> bool:
    """One table row of a single measure over distinct periods (safe to sum or average)."""
    if len(group) < 2 or group[0][0].source != "table":
        return False
    periods = [x[0].periods[0] if x[0].periods else None for x in group]
    return None not in periods and len(set(periods)) == len(periods)


class ProgramPlan(BaseModel):
    """What the LLM may return: a program over evidence ids, constants and step references."""

    template: str = Field(description="e.g. 'subtract(E2, E1), divide(#0, E1)'")
    rationale: str


REASONING_PROMPT = """You write calculation programs for financial questions.
Use ONLY these operations: add, subtract, multiply, divide, exp, greater (each takes two
arguments), table_max/table_min/table_sum/table_average(<row label>, none).
Arguments may be evidence ids (E1, E2, ...), constants (const_100, const_2, const_m1) or
earlier step results (#0 is the first step). Never write a number yourself.
FinQA convention: a percentage change is subtract(new, old), divide(#0, old) - do not
multiply by 100. Pick evidence whose label, column and period match the question."""


class ReasoningAgent(Agent):
    name = "reasoning"
    responsibility = "Construct the calculation program over evidence ids"
    tools = ("program_templates", "llm_structured_output", "finqa_program_schema")

    def __init__(self, config, llm=None) -> None:
        super().__init__(config, llm)
        self.templates = TemplateReasoner()

    def run(self, state: AgentState):
        analysis, audit, payload = state["analysis"], state["audit"], state["payload"]
        candidates = self.templates.propose(
            analysis, audit.evidence, payload["table"], payload["question"]
        )
        note = "templates"
        if self.llm and audit.evidence:
            llm_candidate, note = self._llm_candidate(
                analysis, audit.evidence, payload["question"], candidates
            )
            if llm_candidate:
                candidates = [llm_candidate] + [
                    c for c in candidates if c.program != llm_candidate.program
                ]
        candidates = candidates[: self.config.max_candidates]
        summary = (
            f"{len(candidates)} candidate program(s); best: {candidates[0].template}"
            if candidates
            else "no candidate program fits the audited evidence"
        )
        return (
            {"candidates": candidates, "candidate_index": 0},
            summary,
            {
                "backend": note,
                "candidates": [c.model_dump() for c in candidates[:5]],
            },
        )

    def _llm_candidate(self, analysis, evidence, question, templates):
        listing = "\n".join(
            f"{v.id}: {v.label} | column: {v.column or '-'} | periods: {','.join(v.periods) or '-'}"
            f" | unit: {v.unit} | value: {v.literal}"
            for v in evidence[:40]
        )
        prompt = (
            f"Question: {question}\nOperation: {analysis.operation.value}\n"
            f"Metric: {analysis.metric}\nPeriods: {analysis.periods}\n\nEvidence:\n{listing}"
        )
        plan = self.llm.structured(system=REASONING_PROMPT, prompt=prompt, schema=ProgramPlan)
        if plan is None:
            return None, f"templates (LLM unavailable: {self.llm.last_error})"
        ids = _ID.findall(plan.template)
        by_id = {v.id: v for v in evidence}
        if (
            not ids
            or any(i not in by_id for i in ids)
            or not parse_program(_ID.sub("1", plan.template)).is_valid
        ):
            return None, "templates (LLM program rejected: unknown ids or invalid syntax)"
        if _WRITTEN_NUMBER.search(plan.template):
            return None, "templates (LLM program rejected: it wrote a number itself)"
        used = [by_id[i] for i in dict.fromkeys(ids)]
        best_template = max((c.score for c in templates), default=0.5)
        score = max(
            sum(v.relevance for v in used) / len(used) + (0.2 if analysis.periods else 0.0),
            best_template,
        )
        return _candidate(
            plan.template, used, score, plan.rationale, proposer=self.llm_backend()
        ), self.llm_backend()


class ProgramValidator(Agent):
    name = "program_validator"
    responsibility = "Reject malformed or ungrounded programs before execution"
    tools = ("program_parser",)

    def run(self, state: AgentState):
        candidate = state["candidates"][state["candidate_index"]]
        audit = state["audit"]
        errors = []
        parsed = parse_program(candidate.program)
        if not parsed.is_valid:
            errors.append(parsed.error or "invalid program")
        allowed = {v.literal for v in audit.evidence if v.id in candidate.evidence_ids}
        ungrounded = [lit for lit in parsed.literals if lit not in allowed]
        if ungrounded:
            errors.append(f"numbers not taken from the cited evidence: {ungrounded}")
        unknown = [
            i for i in _ID.findall(candidate.template) if i not in {v.id for v in audit.evidence}
        ]
        if unknown:
            errors.append(f"unknown evidence ids {unknown}")
        report = ValidationReport(valid=not errors, errors=errors)
        summary = f"candidate {state['candidate_index'] + 1}: " + (
            "valid" if report.valid else "; ".join(errors)
        )
        return {"validation": report}, summary, {"program": candidate.program}
