"""Agents 6 and 7 - Calculation Agent and Verification Agent.

The Calculation Agent runs the validated program on the deterministic executor (official
FinQA value rules, no eval). The Verification Agent then checks the result independently:
it recomputes every step in exact decimal arithmetic, re-finds every operand in the report
text, checks the program matches the requested operation, and applies the abstention policy
(confidence, a close rival with a different answer, conflicting evidence).
"""

from __future__ import annotations

import operator
from decimal import Decimal, DivisionByZero, InvalidOperation

from finagents.core.executor import execute_program, to_number
from finagents.core.program import parse_program
from finagents.core.segments import build_segments
from finagents.core.values import parse_value
from finagents.state import (
    AgentState,
    Calculation,
    CalculationStep,
    CandidateProgram,
    Check,
    EvidenceValue,
    Operation,
    Verification,
)

from .base import Agent

MAX_ABS_PERCENT_CHANGE = 10.0  # beyond +/-1000% a wrong operand is far likelier than reality
_SHARE_WORDS = ("portion", "proportion", "percentage of", "percent of", "share of", "% of")
_DECIMAL_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "exp": operator.pow,
}
_REQUIRED_OPS = {
    Operation.PERCENTAGE_CHANGE: ({"subtract", "divide"},),
    Operation.CHANGE: ({"subtract"},),
    Operation.RATIO: ({"divide"},),
    Operation.SUM: ({"add"}, {"table_sum"}),
    Operation.AVERAGE: ({"divide"}, {"table_average"}),
    Operation.ROW_MAX: ({"table_max"},),
    Operation.ROW_MIN: ({"table_min"},),
    Operation.COMPARISON: ({"greater"},),
}


class CalculationAgent(Agent):
    name = "calculation"
    responsibility = "Execute the program with deterministic arithmetic"
    tools = ("restricted_executor",)

    def run(self, state: AgentState):
        candidate = state["candidates"][state["candidate_index"]]
        result = execute_program(candidate.program, state["payload"]["table"])
        calculation = Calculation(
            value=result.value,
            steps=[
                CalculationStep(
                    index=s.index,
                    operation=s.operation,
                    arguments=list(s.arguments),
                    operands=list(s.operands),
                    result=s.result,
                )
                for s in result.steps
            ],
            error=result.error,
        )
        summary = (
            f"{candidate.program} = {result.value}"
            if result.is_valid
            else f"failed: {result.error}"
        )
        return {"calculation": calculation}, summary, {"program": candidate.program}


class VerificationAgent(Agent):
    name = "verification"
    responsibility = "Independently verify operands, calculation and evidence"
    tools = ("decimal_recompute", "evidence_checker", "plausibility_rules", "restricted_executor")

    def run(self, state: AgentState):
        candidate = state["candidates"][state["candidate_index"]]
        calculation, analysis, audit = state["calculation"], state["analysis"], state["audit"]
        payload = state["payload"]
        evidence = {v.id: v for v in audit.evidence}
        checks = hard_checks(candidate, calculation, analysis, payload, evidence)
        checks += soft_checks(
            candidate, analysis, evidence, audit.evidence, self.config.strict_units
        )
        confidence = self.confidence(candidate, checks)
        verification = self.decide(state, candidate, checks, confidence)
        summary = (
            ("verified" if verification.verified else f"not verified ({verification.failure})")
            + f", confidence {confidence:.2f}"
            + (f": {verification.reason}" if verification.reason else "")
        )
        attempt = {
            "program": candidate.program,
            "template": candidate.template,
            "value": calculation.value,
            "confidence": confidence,
            "verified": verification.verified,
            "failure": verification.failure,
            "candidate_index": state["candidate_index"],
            "round": state.get("research_round", 1),
        }
        return (
            {"verification": verification, "attempts": [attempt]},
            summary,
            {"checks": [c.model_dump() for c in checks]},
        )

    def confidence(self, candidate: CandidateProgram, checks: list[Check]) -> float:
        support = max(0.0, candidate.score)
        if any(c.name == "periods_match" and not c.passed for c in checks):
            support *= self.config.period_mismatch_penalty
        return round(min(1.0, support), 4)

    def decide(self, state, candidate, checks, confidence) -> Verification:
        config = self.config
        failed_hard = [c for c in checks if c.hard and not c.passed]
        if not config.verify:  # ablation: accept anything that executed
            ok = state["calculation"].value is not None
            return Verification(
                verified=ok,
                confidence=confidence,
                checks=checks,
                failure="none" if ok else "checks",
                reason="" if ok else "did not execute",
            )
        if failed_hard:
            return Verification(
                verified=False,
                confidence=confidence,
                checks=checks,
                failure="checks",
                reason="; ".join(f"{c.name}: {c.detail}" for c in failed_hard),
            )
        if not config.abstain:
            return Verification(verified=True, confidence=confidence, checks=checks)
        conflict = next(
            (c for c in checks if c.name == "no_conflicting_evidence" and not c.passed), None
        )
        if conflict and config.abstain_on_conflict:
            return Verification(
                verified=False,
                confidence=confidence,
                checks=checks,
                failure="conflict",
                reason=f"conflicting evidence: {conflict.detail}",
            )
        if confidence < config.min_confidence:
            return Verification(
                verified=False,
                confidence=confidence,
                checks=checks,
                failure="low_confidence",
                reason=f"confidence {confidence:.2f} below {config.min_confidence:.2f}",
            )
        rival = self._rival(state, candidate, confidence)
        if rival:
            return Verification(
                verified=False,
                confidence=confidence,
                checks=checks,
                failure="ambiguous",
                reason=rival,
            )
        return Verification(verified=True, confidence=confidence, checks=checks)

    def _rival(self, state, best: CandidateProgram, confidence: float) -> str | None:
        """A different answer from another valid candidate with nearly the same support."""
        best_value = state["calculation"].value
        evidence = {v.id: v for v in state["audit"].evidence}
        for other in state["candidates"]:
            if other.program == best.program:
                continue
            result = execute_program(other.program, state["payload"]["table"])
            if result.value is None or result.value == best_value:
                continue
            calc = Calculation(value=result.value, steps=[], error=None)
            other_checks = hard_checks(
                other, calc, state["analysis"], state["payload"], evidence, recompute=False
            )
            other_checks += soft_checks(
                other,
                state["analysis"],
                evidence,
                state["audit"].evidence,
                self.config.strict_units,
            )
            if any(c.hard and not c.passed for c in other_checks):
                continue
            if confidence - self.confidence(other, other_checks) < self.config.ambiguity_margin:
                return (
                    f"ambiguous: {best.program} = {best_value} and {other.program} = {result.value}"
                    " are equally supported"
                )
        return None


def hard_checks(
    candidate, calculation: Calculation, analysis, payload, evidence, *, recompute=True
) -> list[Check]:
    checks = [
        Check(
            name="executes",
            passed=calculation.value is not None,
            detail="program executed" if calculation.value is not None else f"{calculation.error}",
        )
    ]
    checks.append(_grounding(candidate, payload, evidence))
    if recompute and calculation.value is not None:
        checks.append(_recompute(candidate, calculation, payload))
    checks.append(_operation(candidate, analysis))
    if calculation.value is not None:
        checks.append(_plausible(analysis, calculation.value, payload["question"]))
    return checks


def soft_checks(
    candidate, analysis, evidence, all_evidence, strict_units: bool = True
) -> list[Check]:
    used = [evidence[i] for i in candidate.evidence_ids if i in evidence]
    return [
        _periods(analysis, used, candidate),
        _units(candidate, used, strict_units),
        _conflicts(used, all_evidence),
    ]


def _grounding(candidate: CandidateProgram, payload, evidence) -> Check:
    """Every number in the program is an audited value re-found in its original segment."""
    texts = {s.key: s.text for s in build_segments(payload)}
    cited = [evidence[i] for i in candidate.evidence_ids if i in evidence]
    literals = parse_program(candidate.program).literals
    missing = []
    for lit in literals:
        sources = [v for v in cited if v.literal == lit]
        if not any(_in_text(lit, texts.get(v.segment_key, "")) for v in sources):
            missing.append(lit)
    if missing:
        return Check(
            name="operands_grounded", passed=False, detail=f"not found in cited sources: {missing}"
        )
    return Check(
        name="operands_grounded",
        passed=True,
        detail=f"{len(literals)} operand(s) traced to {sorted({v.segment_key for v in cited})}",
    )


def _in_text(lit: str, text: str) -> bool:
    target = abs(Decimal(lit))
    for token in text.replace("|", " ").split():
        parsed = parse_value(token.rstrip(",;")) or parse_value(token.rstrip(",;."))
        if parsed is not None and abs(parsed.value) == target:
            return True
    return False


def _recompute(candidate: CandidateProgram, calculation: Calculation, payload) -> Check:
    """Recompute arithmetic steps in exact decimals and compare with the float executor."""
    results: list[Decimal | str] = []
    rows = {str(r[0]): r[1:] for r in payload["table"] if r}
    try:
        for step in parse_program(candidate.program).steps:
            if step.operation.startswith("table_"):
                cells = [
                    Decimal(str(to_number(str(c).replace("$", "").strip().split("(")[0].strip())))
                    for c in rows[step.arguments[0]]
                ]
                total = sum(cells, Decimal(0))
                value = {
                    "table_sum": total,
                    "table_average": total / len(cells),
                    "table_max": max(cells),
                    "table_min": min(cells),
                }[step.operation]
                results.append(value)
                continue
            a, b = (
                results[int(x[1:])] if x.startswith("#") else Decimal(str(to_number(x)))
                for x in step.arguments
            )
            if step.operation == "greater":
                results.append("yes" if a > b else "no")
                continue
            results.append(_DECIMAL_OPS[step.operation](a, b))
    except (DivisionByZero, InvalidOperation, KeyError, TypeError, ValueError) as error:
        return Check(
            name="recomputed", passed=False, detail=f"independent recomputation failed: {error}"
        )
    exact, executed = results[-1], calculation.value
    if isinstance(exact, str) or isinstance(executed, str):
        ok = exact == executed
    else:
        ok = abs(float(exact) - float(executed)) <= max(1e-5, abs(float(exact)) * 1e-9)
    return Check(
        name="recomputed",
        passed=ok,
        detail=f"exact decimal result {exact if isinstance(exact, str) else round(float(exact), 6)}"
        + (" matches" if ok else f" differs from {executed}"),
    )


def _operation(candidate: CandidateProgram, analysis) -> Check:
    ops = {step.operation for step in parse_program(candidate.program).steps}
    options = _REQUIRED_OPS.get(analysis.operation, ())
    ok = any(required <= ops for required in options)
    return Check(
        name="matches_operation",
        passed=ok,
        detail=f"uses {sorted(ops)} for {analysis.operation.value}",
    )


def _plausible(analysis, value, question: str) -> Check:
    op = analysis.operation
    if isinstance(value, str):
        return Check(
            name="plausible",
            passed=op is Operation.COMPARISON,
            detail=f"yes/no result for {op.value}",
        )
    if op is Operation.COMPARISON:
        return Check(name="plausible", passed=False, detail="comparison produced a number")
    if op is Operation.PERCENTAGE_CHANGE and abs(value) > MAX_ABS_PERCENT_CHANGE:
        return Check(
            name="plausible", passed=False, detail=f"percent change {value:.4g} exceeds +/-1000%"
        )
    if (
        op is Operation.RATIO
        and not 0 <= abs(value) <= 1
        and any(w in question.lower() for w in _SHARE_WORDS)
    ):
        return Check(
            name="plausible", passed=False, detail=f"a share of a whole cannot be {value:.4g}"
        )
    if op is Operation.CHANGE and value == 0:
        return Check(
            name="plausible", passed=False, detail="a change of exactly zero: same value twice"
        )
    return Check(name="plausible", passed=True, detail=f"{value:.6g} is plausible for {op.value}")


def _periods(analysis, used: list[EvidenceValue], candidate) -> Check:
    if not analysis.periods or not used or candidate.template.startswith("table_"):
        return Check(
            name="periods_match", passed=True, hard=False, detail="no period constraint to check"
        )
    have = {p for v in used for p in v.periods}
    missing = [p for p in analysis.periods if p not in have]
    if analysis.operation is Operation.RATIO:
        missing = [] if analysis.periods[-1] in have or not have else missing
    if missing:
        return Check(
            name="periods_match", passed=False, hard=False, detail=f"no operand from {missing}"
        )
    return Check(
        name="periods_match", passed=True, hard=False, detail="operands cover the named periods"
    )


def _units(candidate, used: list[EvidenceValue], strict: bool) -> Check:
    additive = any(
        s.operation in {"add", "subtract"} for s in parse_program(candidate.program).steps
    )
    percent = {v.unit == "%" for v in used}
    if additive and len(percent) > 1:
        return Check(
            name="units_consistent",
            passed=False,
            hard=strict,
            detail="adds or subtracts a percentage and an amount",
        )
    return Check(
        name="units_consistent",
        passed=True,
        hard=False,
        detail=f"units {sorted({v.unit for v in used})}",
    )


def _conflicts(used: list[EvidenceValue], all_evidence: list[EvidenceValue]) -> Check:
    """Another table row reports a different value for the same label, column and period."""
    problems = []
    for value in used:
        if value.source != "table" or not value.periods:
            continue
        rivals = [
            o
            for o in all_evidence
            if o.source == "table"
            and o.segment_key != value.segment_key
            and o.label.strip().lower() == value.label.strip().lower()
            and o.column.strip().lower() == value.column.strip().lower()
            and o.periods == value.periods
            and abs(o.value) != abs(value.value)
        ]
        if rivals:
            problems.append(
                f"'{value.label}' {value.periods}: {value.literal} ({value.segment_key}) vs "
                + ", ".join(f"{o.literal} ({o.segment_key})" for o in rivals)
            )
    if problems:
        return Check(
            name="no_conflicting_evidence", passed=False, hard=False, detail="; ".join(problems)
        )
    return Check(
        name="no_conflicting_evidence", passed=True, hard=False, detail="no conflicting values"
    )
