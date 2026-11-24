"""Agent 8 - Response Agent: the final answer and its audit trail.

It formats the verified value, cites every evidence segment the program used, writes the
calculation step by step and as one formula, and reports confidence. When the orchestrator
abstained it explains why and shows what was withheld. With an LLM enabled, Claude may
rephrase the one-sentence explanation, but only if every number in its sentence already
appears in the evidence or the answer; otherwise the template sentence is kept.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from finagents.core.values import NUMBER_SPAN
from finagents.state import (
    AgentState,
    CandidateProgram,
    Citation,
    EvidenceValue,
    FinalAnswer,
    Operation,
    QueryAnalysis,
)

from .base import Agent

_SYMBOL = {"add": "+", "subtract": "−", "multiply": "×", "divide": "/", "exp": "^"}


def display_value(analysis: QueryAnalysis, question: str, value) -> str | None:
    """FinQA stores percentages as ratios (0.098); show them as 9.80%."""
    if value is None or isinstance(value, str):
        return value
    asks_percent = "percent" in question.lower() or "%" in question
    if analysis.operation is Operation.PERCENTAGE_CHANGE or (
        analysis.operation is Operation.RATIO and asks_percent
    ):
        return f"{value * 100:.2f}%"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def formula(candidate: CandidateProgram, steps) -> str:
    """Nested human-readable formula, e.g. ``(1200 − 1000) / 1000``."""
    parts: list[str] = []
    for step in steps:
        if step.operation.startswith("table_"):
            parts.append(f"{step.operation.removeprefix('table_')}(row '{step.arguments[0]}')")
            continue
        a, b = (
            parts[int(x[1:])]
            if x.startswith("#")
            else x.replace("const_m1", "-1").replace("const_", "")
            for x in step.arguments
        )
        if step.operation == "greater":
            parts.append(f"{a} > {b}")
        else:
            parts.append(f"({a} {_SYMBOL[step.operation]} {b})")
    text = parts[-1] if parts else candidate.program
    return text[1:-1] if text.startswith("(") and text.endswith(")") else text


def calculation_lines(steps) -> list[str]:
    lines = []
    for step in steps:
        result = step.result if isinstance(step.result, str) else f"{step.result:,.6g}"
        lines.append(
            f"Step {step.index + 1}: {step.operation}({', '.join(step.arguments)}) = {result}"
        )
    return lines


def template_sentence(analysis: QueryAnalysis, display: str, used: list[EvidenceValue]) -> str:
    def describe(v: EvidenceValue) -> str:
        period = ", ".join(v.periods) or v.column or "the report"
        return f"{v.literal} ({period})"

    metric = analysis.metric
    op = analysis.operation
    if op in {Operation.PERCENTAGE_CHANGE, Operation.CHANGE} and len(used) >= 2:
        verb = "changed by" if op is Operation.CHANGE else "changed by"
        return f"{metric.capitalize()} {verb} {display}, from {describe(used[0])} to {describe(used[1])}."
    if op is Operation.RATIO and len(used) >= 2:
        return f"The ratio of {used[0].label} to {used[1].label} is {display}."
    if op is Operation.COMPARISON:
        return f"Answer: {display} — comparing {', '.join(describe(v) for v in used)}."
    return f"The {op.value.replace('_', ' ')} of {metric} is {display}."


class Sentence(BaseModel):
    sentence: str


class ResponseAgent(Agent):
    name = "response"
    responsibility = "Generate the final answer and audit trail"
    tools = ("citation_formatter", "llm_structured_output")

    def run(self, state: AgentState):
        payload, analysis = state["payload"], state["analysis"]
        reason = state.get("abstain_reason")
        verification = state.get("verification")
        candidates = state.get("candidates") or []
        index = state.get("candidate_index", 0)
        candidate = candidates[index] if candidates and index < len(candidates) else None
        calculation = state.get("calculation")
        audit = state.get("audit")
        evidence = {v.id: v for v in audit.evidence} if audit else {}
        used = [evidence[i] for i in candidate.evidence_ids if i in evidence] if candidate else []
        texts = {s.key: s.text for s in state.get("retrieved", [])}
        citations = [
            Citation(
                segment_key=key,
                source="table" if key.startswith("table") else "text",
                text=texts.get(key, ""),
                values=[v for v in used if v.segment_key == key],
            )
            for key in dict.fromkeys(v.segment_key for v in used)
        ]
        value = calculation.value if calculation else None
        attempts = state.get("attempts", [])
        alternatives = [
            f"{a['program']} = {a['value']}"
            for a in attempts
            if candidate is None or a["program"] != candidate.program
        ][:3]
        steps = calculation.steps if calculation else []
        answered = reason is None and candidate is not None and value is not None
        display = display_value(analysis, payload["question"], value) if answered else None
        explanation = ""
        note = "template"
        if answered:
            explanation = template_sentence(analysis, display, used)
            if self.llm:
                explanation, note = self._phrase(payload["question"], display, used, explanation)
        else:
            explanation = f"No answer: {reason or 'no verified calculation'}."
        final = FinalAnswer(
            status="answered" if answered else "abstained",
            value=value if answered else None,
            display=display,
            explanation=explanation,
            citations=citations,
            calculation=calculation_lines(steps),
            formula=(f"{formula(candidate, steps)} = {display}" if answered and steps else None),
            program=candidate.program if candidate else None,
            confidence=verification.confidence if verification else 0.0,
            abstention_reason=reason,
            checks=verification.checks if verification else [],
            alternatives=alternatives,
            withheld_value=None if answered else value,
        )
        summary = f"answered {display}" if answered else f"abstained: {reason}"
        return {"final": final}, summary, {"backend": note}

    def _phrase(self, question, display, used, fallback):
        listing = "; ".join(f"{v.label} {','.join(v.periods)}: {v.literal} {v.unit}" for v in used)
        result = self.llm.structured(
            system="Write one plain sentence answering the question with the given answer and "
            "evidence. Use only numbers that appear below. No advice.",
            prompt=f"Question: {question}\nAnswer: {display}\nEvidence: {listing}",
            schema=Sentence,
        )
        if result is None:
            return fallback, f"template (LLM unavailable: {self.llm.last_error})"
        allowed = {
            n.replace(",", "") for n in NUMBER_SPAN.findall(f"{display} {listing} {question}")
        }
        allowed |= {re.sub(r"%$", "", n) for n in allowed}
        used_numbers = {
            n.replace(",", "").rstrip("%") for n in NUMBER_SPAN.findall(result.sentence)
        }
        if used_numbers - {n.rstrip("%") for n in allowed}:
            return fallback, "template (LLM sentence used numbers not in the evidence)"
        return result.sentence, self.llm_backend()
