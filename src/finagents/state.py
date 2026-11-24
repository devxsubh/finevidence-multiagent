"""Typed state shared by the agents, and the structured messages they exchange.

Every agent reads the fields it needs from ``AgentState`` and returns only the fields it
owns. All messages are Pydantic models, so each hand-off is validated and every run can be
serialized to JSON for the API, the frontend and the evaluation logs.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Operation(StrEnum):
    PERCENTAGE_CHANGE = "percentage_change"
    CHANGE = "change"
    RATIO = "ratio"
    SUM = "sum"
    AVERAGE = "average"
    ROW_MAX = "row_max"
    ROW_MIN = "row_min"
    COMPARISON = "comparison"
    UNSUPPORTED = "unsupported"


# 1. Query Analyst -> structured task
class QueryAnalysis(BaseModel):
    metric: str = Field(description="the financial quantity asked about, e.g. 'net revenue'")
    entity_terms: list[str] = Field(description="search terms for the metric")
    periods: list[str] = Field(description="years named in the question, oldest first")
    operation: Operation
    part_terms: list[str] = Field(default_factory=list, description="ratio numerator terms")
    whole_terms: list[str] = Field(default_factory=list, description="ratio denominator terms")
    required_evidence: int = Field(description="how many values the calculation needs")
    rationale: str = ""
    backend: str = "rules"


# 2. Evidence Researcher -> retrieved segments
class RetrievedSegment(BaseModel):
    key: str
    source: Literal["table", "text"]
    text: str
    score: float
    rank: int


# 3. Evidence Auditor -> validated values
class EvidenceValue(BaseModel):
    id: str  # E1, E2, ... : the only way later agents may refer to a number
    segment_key: str
    source: Literal["table", "text"]
    label: str
    column: str
    periods: list[str]
    literal: str
    value: float
    unit: str
    relevance: float


class Check(BaseModel):
    name: str
    passed: bool
    detail: str
    hard: bool = True  # a failed hard check rejects; a failed soft check lowers confidence


class AuditReport(BaseModel):
    status: Literal["valid", "insufficient"]
    evidence: list[EvidenceValue]
    checks: list[Check]
    rejected: int = 0
    reason: str = ""


# 4. Reasoning Agent -> candidate programs over evidence ids
class CandidateProgram(BaseModel):
    template: str = Field(description="program over evidence ids, e.g. subtract(E2, E1)")
    program: str = Field(description="the same program with ids replaced by their values")
    evidence_ids: list[str]
    score: float
    rationale: str
    proposer: str


class ValidationReport(BaseModel):
    valid: bool
    errors: list[str]


class CalculationStep(BaseModel):
    index: int
    operation: str
    arguments: list[str]
    operands: list[float]
    result: float | str


class Calculation(BaseModel):
    value: float | str | None
    steps: list[CalculationStep]
    error: str | None = None


class Verification(BaseModel):
    verified: bool
    confidence: float
    checks: list[Check]
    reason: str = ""
    # Why it failed, so the orchestrator can pick the next step:
    # checks -> try the next candidate; low_confidence / ambiguous -> search more;
    # conflict -> abstain now.
    failure: Literal["none", "checks", "low_confidence", "ambiguous", "conflict"] = "none"


class Citation(BaseModel):
    segment_key: str
    source: str
    text: str
    values: list[EvidenceValue]


class FinalAnswer(BaseModel):
    status: Literal["answered", "abstained"]
    value: float | str | None
    display: str | None
    explanation: str
    citations: list[Citation]
    calculation: list[str]
    formula: str | None
    program: str | None
    confidence: float
    abstention_reason: str | None
    checks: list[Check]
    alternatives: list[str]
    withheld_value: float | str | None = None


class AgentStep(BaseModel):
    """One entry of the human-readable execution trace."""

    agent: str
    summary: str
    tools: list[str] = Field(default_factory=list)
    route: str | None = None  # the orchestrator's next decision after this step
    milliseconds: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    payload: dict[str, Any]  # allowlisted question + report; nothing else
    analysis: QueryAnalysis
    research_round: int
    retrieved: list[RetrievedSegment]
    found: list[EvidenceValue]  # values read by the researcher, before the audit
    audit: AuditReport
    candidates: list[CandidateProgram]
    candidate_index: int
    validation: ValidationReport
    calculation: Calculation
    verification: Verification
    attempts: Annotated[list[dict[str, Any]], operator.add]  # every candidate tried
    abstain_reason: str | None
    route: str  # the orchestrator's last routing decision
    final: FinalAnswer
    trace: Annotated[list[AgentStep], operator.add]
