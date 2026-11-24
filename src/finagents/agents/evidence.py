"""Agents 3 and 4 - Evidence Researcher and Evidence Auditor.

The Researcher searches the report and reads the numbers in what it finds. Round 1 searches
with the question; each retry the orchestrator asks for widens the search (more segments)
and adds sub-queries for the metric in each named period.

The Auditor decides which of those numbers later agents may use. It checks that each value
describes the metric, comes from the right period, has a known unit and can be traced back
to the report text. Only audited values get an evidence id (E1, E2, ...).
"""

from __future__ import annotations

from collections.abc import Sequence

from finagents.core.segments import Mention, build_segments, extract_mentions, label_match
from finagents.core.values import parse_value
from finagents.state import (
    AgentState,
    AuditReport,
    Check,
    EvidenceValue,
    Operation,
    RetrievedSegment,
)

from .base import Agent

_CHANGE_OPS = {Operation.PERCENTAGE_CHANGE, Operation.CHANGE, Operation.COMPARISON}
MAX_AUDITED_VALUES = 60


def research_queries(analysis, round_number: int) -> list[str]:
    """Round 1 uses the question; later rounds use metric + period sub-queries."""
    entity = " ".join(analysis.entity_terms)
    queries = [f"{entity} {period}" for period in analysis.periods] or [entity]
    if round_number > 2 and analysis.operation is Operation.RATIO:
        queries.append(f"total {entity}")
    return [q for q in queries if q.strip()]


class EvidenceResearcher(Agent):
    name = "evidence_researcher"
    responsibility = "Search report tables and narrative for the metric"
    tools = ("search_evidence", "extract_values")

    def __init__(self, config, llm=None, *, retriever) -> None:
        super().__init__(config, llm)
        self.retriever = retriever

    def run(self, state: AgentState):
        payload, analysis = state["payload"], state["analysis"]
        round_number = state.get("research_round", 1)
        k = self.config.initial_k + self.config.k_step * (round_number - 1)
        segments = build_segments(payload)
        queries = [payload["question"]] + (
            research_queries(analysis, round_number) if round_number > 1 else []
        )
        best: dict[str, RetrievedSegment] = {s.key: s for s in state.get("retrieved", [])}
        for query in queries:
            for item in self.retriever.rank(segments, query)[:k]:
                previous = best.get(item.segment.key)
                if previous is None or item.rank < previous.rank:
                    best[item.segment.key] = RetrievedSegment(
                        key=item.segment.key,
                        source=item.segment.source,  # type: ignore[arg-type]
                        text=item.segment.text,
                        score=round(item.score, 6),
                        rank=item.rank,
                    )
        retrieved = sorted(best.values(), key=lambda s: (s.rank, s.key))
        keys = {s.key for s in retrieved}
        mentions = extract_mentions(s for s in segments if s.key in keys or s.key == "table_0")
        mentions = [m for m in mentions if m.segment_key in keys]
        rank = {s.key: s.rank for s in retrieved}
        found = [self._value(m, analysis, rank[m.segment_key]) for m in mentions]
        new = len(keys) - len(state.get("retrieved", []))
        summary = (
            f"round {round_number}: {len(retrieved)} segments ({new} new, k={k}, "
            f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'}), {len(found)} values"
        )
        return (
            {"retrieved": retrieved, "found": found},
            summary,
            {
                "retriever": self.retriever.name,
                "queries": queries,
                "k": k,
                "top": [s.key for s in retrieved[:10]],
            },
        )

    @staticmethod
    def _value(mention: Mention, analysis, rank: int) -> EvidenceValue:
        terms = analysis.entity_terms
        relevance = label_match(terms, mention) + 0.3 / rank
        return EvidenceValue(
            id="",
            segment_key=mention.segment_key,
            source=mention.source,  # type: ignore[arg-type]
            label=mention.label,
            column=mention.column,
            periods=list(mention.periods),
            literal=mention.literal,
            value=float(mention.value),
            unit=mention.unit,
            relevance=round(relevance, 6),
        )


class EvidenceAuditor(Agent):
    name = "evidence_auditor"
    responsibility = "Validate values, units, reporting periods and source relevance"
    tools = ("relevance_check", "period_check", "unit_check", "traceability_check")

    def run(self, state: AgentState):
        analysis = state["analysis"]
        found: list[EvidenceValue] = state.get("found", [])
        texts = {s.key: s.text for s in state.get("retrieved", [])}
        checks: list[Check] = []

        traceable = [v for v in found if _traceable(v, texts.get(v.segment_key, ""))]
        checks.append(
            Check(
                name="traceable",
                passed=len(traceable) == len(found),
                detail=f"{len(traceable)}/{len(found)} values re-found in their source text",
            )
        )

        terms = [*analysis.entity_terms, *analysis.part_terms, *analysis.whole_terms]
        relevant = [v for v in traceable if _label_relevance(terms, v) >= self.config.min_relevance]
        # Keep the strongest values even below the threshold: row labels are often terse.
        ranked = sorted(traceable, key=lambda v: -v.relevance)
        kept_ids = {id(v) for v in relevant} | {id(v) for v in ranked[:12]}
        kept = sorted((v for v in traceable if id(v) in kept_ids), key=lambda v: -v.relevance)
        kept = kept[:MAX_AUDITED_VALUES]
        checks.append(
            Check(
                name="relevant",
                passed=bool(relevant),
                detail=f"{len(relevant)} values describe '{analysis.metric}'",
            )
        )

        covered = {p for v in kept for p in v.periods}
        missing = [p for p in analysis.periods if p not in covered]
        period_ok = not missing or analysis.operation in {Operation.ROW_MAX, Operation.ROW_MIN}
        checks.append(
            Check(
                name="periods_covered",
                passed=period_ok,
                detail="all named periods found" if not missing else f"no value for {missing}",
            )
        )

        units = sorted({v.unit for v in kept})
        mixed = "%" in units and len(units) > 1
        checks.append(
            Check(
                name="units_known",
                passed=True,
                hard=False,
                detail=f"units: {', '.join(units) or 'none'}"
                + (" (percent and amounts mixed)" if mixed else ""),
            )
        )

        enough = len(kept) >= max(analysis.required_evidence, 1)
        checks.append(
            Check(
                name="enough_evidence",
                passed=enough,
                detail=f"{len(kept)} usable values, {analysis.required_evidence} needed",
            )
        )

        evidence = [v.model_copy(update={"id": f"E{i}"}) for i, v in enumerate(kept, start=1)]
        problems = [c for c in checks if c.hard and not c.passed and c.name != "traceable"]
        # A missing period with otherwise usable evidence is worth another search round,
        # but the values found so far still go to the Reasoning Agent if no round is left.
        status = "valid" if enough and period_ok else "insufficient"
        reason = "; ".join(c.detail for c in problems)
        report = AuditReport(
            status=status,
            evidence=evidence,
            checks=checks,  # type: ignore[arg-type]
            rejected=len(found) - len(kept),
            reason=reason,
        )
        summary = f"{status}: {len(evidence)} audited values, {report.rejected} rejected" + (
            f" ({reason})" if reason else ""
        )
        return {"audit": report}, summary, {"checks": [c.model_dump() for c in checks]}


def _label_relevance(terms: Sequence[str], value: EvidenceValue) -> float:
    return label_match(terms, value)  # type: ignore[arg-type]


def _traceable(value: EvidenceValue, text: str) -> bool:
    """The value's literal can be parsed back out of its own segment text."""
    target = abs(float(value.literal))
    for token in text.replace("|", " ").split():
        parsed = parse_value(token.rstrip(",;")) or parse_value(token.rstrip(",;."))
        if parsed is not None and abs(abs(float(parsed.value)) - target) < 1e-9:
            return True
    return False
