"""Run systems over a split under identical conditions and save every answer.

Each run writes ``runs/<config>_<split>.jsonl`` (one full result per question: answer,
citations, calculation and agent trace) and a summary, so every reported number can be
re-derived from the saved artifacts.
"""

from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finagents.agents import Orchestrator
from finagents.config import SystemConfig, load_config
from finagents.core.records import PROJECT_ROOT, gold, inference_input, load_split
from finagents.retrieval import build_retriever

from .metrics import mcnemar, score, summarize

RUNS = PROJECT_ROOT / "runs"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


# Claude Opus 5.5 list prices, USD per million tokens (input, output), for cost reporting.
PRICES = {"claude-opus-5-5": (4.0, 20.0)}


def llm_usage(system: Orchestrator) -> dict[str, Any] | None:
    llm = system.llm
    if llm is None or not hasattr(llm, "calls"):
        return None
    usage = {
        "model": llm.model,
        "calls": llm.calls,
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens,
    }
    if llm.model in PRICES:
        price_in, price_out = PRICES[llm.model]
        cost = (llm.input_tokens * price_in + llm.output_tokens * price_out) / 1e6
        usage["estimated_cost_usd"] = round(cost, 4)
    return usage


def evaluate(
    config: SystemConfig,
    records: Sequence[dict],
    *,
    split: str,
    tag: str | None = None,
    save: bool = True,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Answer every record (in parallel threads if ``workers`` > 1; order is preserved)."""
    system = Orchestrator(config)
    results: list[dict] = [{}] * len(records)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(system.run, record): i for i, record in enumerate(records)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if progress and (done % 25 == 0 or done == len(records)):
                progress(done, len(records))
    rows = [score(record, result) for record, result in zip(records, results, strict=True)]
    summary = {
        "config": config.name,
        "split": split,
        "commit": _commit(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": config.model_dump(mode="json"),
        "llm_usage": llm_usage(system),
        **summarize(rows),
    }
    if save:
        name = tag or f"{config.name}_{split}"
        RUNS.mkdir(exist_ok=True)
        with (RUNS / f"{name}.jsonl").open("w") as handle:
            for result, row in zip(results, rows, strict=True):
                handle.write(json.dumps({**result, "score": row}) + "\n")
        (RUNS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    summary["rows"] = rows
    return summary


def sample_indices(total: int, size: int, seed: int = 42) -> list[int]:
    """A reproducible random sample of question indices, in dataset order."""
    return sorted(random.Random(seed).sample(range(total), size))


def compare(
    names: Sequence[str],
    split: str = "dev",
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
    workers: int = 1,
    log=print,
) -> dict:
    """Evaluate several configs on the same questions; paired tests against the first.

    ``sample`` picks a seeded random subset; its outputs get a ``_sample<N>`` suffix so they
    never overwrite the full-split results.
    """
    records = load_split(split)
    indices = list(range(len(records)))
    suffix = ""
    if sample:
        indices = sample_indices(len(records), sample, seed)
        suffix = f"_sample{sample}"
    elif limit:
        indices = indices[:limit]
        suffix = f"_first{limit}"
    records = [records[i] for i in indices]
    summaries, rows = {}, {}
    for name in names:
        log(f"evaluating {name} on {split}{suffix} ({len(records)} questions)")
        tag = f"{name.replace('/', '_')}_{split}{suffix}"
        summary = evaluate(
            load_config(name),
            records,
            split=split,
            tag=tag,
            workers=workers,
            progress=lambda i, n: log(f"  {i}/{n}"),
        )
        rows[name] = summary.pop("rows")
        summaries[name] = summary
    baseline = names[0]
    base = rows[baseline]
    for name in names[1:]:
        summaries[name]["vs_baseline"] = {
            "correct": mcnemar([r["correct"] for r in base], [r["correct"] for r in rows[name]]),
            # right_only here = wrong answers the baseline gave and this system avoided
            "wrong_answers_avoided": mcnemar(
                [not (r["answered"] and not r["correct"]) for r in base],
                [not (r["answered"] and not r["correct"]) for r in rows[name]],
            ),
        }
    comparison = {
        "split": split,
        "questions": len(records),
        "sample_seed": seed if sample else None,
        "indices": indices if (sample or limit) else None,
        "baseline": baseline,
        "systems": summaries,
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"comparison_{split}{suffix}.json").write_text(json.dumps(comparison, indent=2))
    return comparison


def retrieval_recall(
    split: str = "dev", ks: Sequence[int] = (3, 5, 10), limit: int | None = None
) -> dict:
    """All-gold recall@k of each retriever, ranking with the question alone."""
    from finagents.core.segments import build_segments

    records = load_split(split)[:limit] if limit else load_split(split)
    report = {}
    for name in ("bm25", "tfidf", "learned"):
        retriever = build_retriever(name)
        hits = {k: 0 for k in ks}
        total = 0
        for record in records:
            payload = inference_input(record)
            gold_keys = set(gold(record)["gold_inds"])
            if not gold_keys:
                continue
            total += 1
            order = [
                r.segment.key for r in retriever.rank(build_segments(payload), payload["question"])
            ]
            for k in ks:
                hits[k] += gold_keys <= set(order[:k])
        report[name] = {f"recall@{k}": round(hits[k] / total, 4) for k in ks}
    report["questions"] = total
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"retrieval_{split}.json").write_text(json.dumps(report, indent=2))
    return report


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())
