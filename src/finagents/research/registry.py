"""The hypotheses the research team may test, and what each one changes.

Agents choose and run experiments on their own, but only from this list: every hypothesis
is a bounded change to the system's settings, already implemented and tested. That keeps
the loop from rewriting the benchmark or the metric to make itself look better.

``protected`` hypotheses need a human's approval before they run (see ``approvals.py``):
they add an external dependency or cost, or touch the evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Hypothesis:
    id: str
    weakness: str  # which diagnosed weakness it addresses
    statement: str
    change: dict[str, Any] = field(default_factory=dict)  # relative ("+0.1") or absolute values
    objective: str = "execution_accuracy"  # the metric it should improve
    guard: str = ""  # a metric that must not get meaningfully worse
    protected: bool = False
    reason_protected: str = ""


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "H1-confidence-up",
        "wrong_answers",
        "Raising the confidence threshold removes wrong answers faster than correct ones.",
        {"min_confidence": "+0.1"},
        objective="wrong_answer_rate",
        guard="execution_accuracy",
    ),
    Hypothesis(
        "H2-confidence-down",
        "over_abstention",
        "Lowering the confidence threshold recovers correct answers the agent withheld.",
        {"min_confidence": "-0.1"},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
    ),
    Hypothesis(
        "H3-wider-search",
        "insufficient_evidence",
        "Retrieving more segments per round finds evidence the first search missed.",
        {"initial_k": "+3"},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
    ),
    Hypothesis(
        "H4-more-rounds",
        "insufficient_evidence",
        "A fourth research round resolves questions still unanswered after three.",
        {"max_research_rounds": "+1"},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
    ),
    Hypothesis(
        "H5-stricter-relevance",
        "wrong_answers",
        "Auditing out weakly related values stops the reasoner binding the wrong row.",
        {"min_relevance": "+0.15"},
        objective="wrong_answer_rate",
        guard="execution_accuracy",
    ),
    Hypothesis(
        "H6-wider-ambiguity-margin",
        "wrong_answers",
        "Abstaining when a rival answer is within 0.05 support avoids coin-flip answers.",
        {"ambiguity_margin": 0.05},
        objective="wrong_answer_rate",
        guard="execution_accuracy",
    ),
    Hypothesis(
        "H7-softer-period-penalty",
        "over_abstention",
        "A milder penalty for operands from unnamed periods keeps answers that were right.",
        {"period_mismatch_penalty": 0.8},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
    ),
    Hypothesis(
        "H8-relaxed-units",
        "no_program",
        "Allowing percent/amount arithmetic recovers programs the unit check rejected.",
        {"strict_units": False},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
    ),
    Hypothesis(
        "H9-llm-reasoning",
        "weak_operation",
        "Letting Claude choose operations and evidence fixes question shapes the templates miss.",
        {"use_llm": True, "llm_agents": ["query_analyst", "reasoning"]},
        objective="execution_accuracy",
        guard="wrong_answer_rate",
        protected=True,
        reason_protected="adds an external LLM dependency with per-question API cost",
    ),
)

# Settings an experiment may change. Anything else (split, metric, data) is off limits.
TUNABLE = {
    "min_confidence",
    "initial_k",
    "max_research_rounds",
    "min_relevance",
    "ambiguity_margin",
    "period_mismatch_penalty",
    "strict_units",
    "use_llm",
    "llm_agents",
}

PROTECTED_ACTIONS = {
    "evaluate_test": "the official test split is evaluated once, at the end, by a human decision",
    "promote_default": "changing the default system configuration",
    "change_protocol": "changing the benchmark, metric or split protocol",
}


def apply_change(settings: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """New settings with the hypothesis applied; relative changes are strings like '+0.1'."""
    updated = dict(settings)
    for key, value in change.items():
        if key not in TUNABLE:
            raise ValueError(f"{key!r} is not a tunable setting")
        if isinstance(value, str) and value[:1] in "+-":
            current = updated[key]
            updated[key] = (
                round(current + type(current)(float(value)), 6)
                if isinstance(current, float)
                else current + int(float(value))
            )
        else:
            updated[key] = value
    return updated


def by_id(hypothesis_id: str) -> Hypothesis:
    return next(h for h in HYPOTHESES if h.id == hypothesis_id)
