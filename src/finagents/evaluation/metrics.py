"""Scoring under the official FinQA convention, plus the metrics abstention needs.

* Execution accuracy: answered and the value equals ``exe_ans`` after five-decimal rounding.
  Every question stays in the denominator; an abstention counts as not correct.
* Coverage: share of questions answered. Selective accuracy: accuracy on answered questions.
* Wrong-answer rate: share of all questions answered wrongly (the costly error for analysts).
* Cites all gold: an answered question whose citations include every gold evidence segment.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from finagents.core.executor import ROUNDING_DECIMALS
from finagents.core.records import gold


def is_correct(value: Any, reference: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), ROUNDING_DECIMALS) == reference
    return value == reference


ABSTAIN_KINDS = (
    ("unsupported", "unsupported question"),
    ("insufficient_evidence", "insufficient evidence"),
    ("no_program", "no calculation fits"),
    ("conflict", "conflicting evidence"),
    ("ambiguous", "ambiguous"),
    ("low_confidence", "confidence"),
    ("failed_checks", "passed verification"),
)


def abstain_kind(reason: str | None) -> str | None:
    if not reason:
        return None
    return next((kind for kind, marker in ABSTAIN_KINDS if marker in reason), "other")


def score(record: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reference = gold(record)
    answer = result["answer"]
    answered = answer["status"] == "answered"
    cited = {c["segment_key"] for c in answer["citations"]}
    gold_keys = set(reference["gold_inds"])
    return {
        "id": result["id"],
        "operation": result["analysis"]["operation"],
        "answered": answered,
        "correct": answered and is_correct(answer["value"], reference["exe_ans"]),
        "cites_all_gold": answered and bool(gold_keys) and gold_keys <= cited,
        "withheld_correct": not answered
        and is_correct(answer.get("withheld_value"), reference["exe_ans"]),
        "abstain_kind": None if answered else abstain_kind(answer.get("abstention_reason")),
        "rounds": result["research_rounds"],
        "candidates_tried": result["candidates_tried"],
        "milliseconds": result["milliseconds"],
    }


def bootstrap_interval(
    values: Sequence[float], *, samples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on paired outcomes."""
    right_only = sum(r and not left_ for left_, r in zip(left, right, strict=True))
    left_only = sum(left_ and not r for left_, r in zip(left, right, strict=True))
    n, k = right_only + left_only, min(right_only, left_only)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"right_only": right_only, "left_only": left_only, "p_value": p}


def _rate(rows, key) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    answered = [r for r in rows if r["answered"]]
    correct = [float(r["correct"]) for r in rows]
    low, high = bootstrap_interval(correct) if rows else (0.0, 0.0)
    by_op: dict[str, list] = defaultdict(list)
    for r in rows:
        by_op[r["operation"]].append(r)
    return {
        "questions": n,
        "execution_accuracy": _rate(rows, "correct"),
        "execution_accuracy_ci95": [round(low, 4), round(high, 4)],
        "coverage": round(len(answered) / n, 4) if n else 0.0,
        "selective_accuracy": _rate(answered, "correct"),
        "wrong_answer_rate": round(sum(r["answered"] and not r["correct"] for r in rows) / n, 4)
        if n
        else 0.0,
        "cites_all_gold": _rate(answered, "cites_all_gold"),
        "withheld_but_correct": sum(r["withheld_correct"] for r in rows),
        "abstentions": {
            kind: sum(r["abstain_kind"] == kind for r in rows)
            for kind, _ in ABSTAIN_KINDS
            if any(r["abstain_kind"] == kind for r in rows)
        },
        "mean_rounds": round(sum(r["rounds"] for r in rows) / n, 3) if n else 0.0,
        "mean_milliseconds": round(sum(r["milliseconds"] for r in rows) / n, 2) if n else 0.0,
        "by_operation": {
            op: {
                "questions": len(items),
                "execution_accuracy": _rate(items, "correct"),
                "coverage": round(sum(r["answered"] for r in items) / len(items), 4),
                "wrong_answer_rate": round(
                    sum(r["answered"] and not r["correct"] for r in items) / len(items), 4
                ),
            }
            for op, items in sorted(by_op.items())
        },
    }
