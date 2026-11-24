"""Loading FinQA records and building the only input the agents may see.

A FinQA record also carries its answer, program and gold evidence. Agents must never read
those, so every agent works from ``inference_input(record)``: an allowlisted copy holding
only the question and the report (table, pre_text, post_text).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = ("train", "dev")  # the official test split is not used (see docs/evaluation.md)

INFERENCE_FIELDS = frozenset({"id", "question", "table", "pre_text", "post_text"})


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load one FinQA split as a list of records."""
    if split not in SPLITS:
        raise ValueError(f"unknown or protected split {split!r}; allowed: {SPLITS}")
    payload = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(r, Mapping) for r in payload):
        raise ValueError(f"{split}.json must be a JSON list of objects")
    return payload


@lru_cache(maxsize=4)
def cached_split(split: str) -> tuple[dict[str, Any], ...]:
    """Load a split once per process (the API and evaluations reuse it)."""
    return tuple(load_split(split))


def inference_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """The allowlisted fields of a record; accepts raw FinQA records or user reports."""
    question = record["qa"]["question"] if "qa" in record else record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record needs a non-empty question")
    payload = {
        "id": str(record.get("id", "user-report")),
        "question": question.strip(),
        "table": [[str(cell) for cell in row] for row in record.get("table", [])],
        "pre_text": [str(text) for text in record.get("pre_text", [])],
        "post_text": [str(text) for text in record.get("post_text", [])],
    }
    assert_inference_safe(payload)
    return payload


def assert_inference_safe(payload: Mapping[str, Any]) -> None:
    """Raise if a payload carries any field outside the allowlist (e.g. ``qa`` or ``answer``)."""
    extra = set(payload) - INFERENCE_FIELDS
    if extra:
        raise ValueError(f"inference payload contains disallowed fields: {sorted(extra)}")


def gold(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reference answer fields, used only by evaluation and the training of the reranker."""
    qa = record["qa"]
    return {
        "exe_ans": qa.get("exe_ans"),
        "answer": qa.get("answer"),
        "program": qa.get("program"),
        "gold_inds": dict(qa.get("gold_inds") or {}),
    }
