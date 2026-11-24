"""HTTP API for the frontend: ask questions, browse examples, read results, run research.

Start with ``uv run finagents serve`` (http://localhost:8000, docs at /docs).
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finagents.agents import AGENT_ROSTER, Orchestrator
from finagents.config import CONFIG_DIR, load_config
from finagents.core.records import PROJECT_ROOT, cached_split, gold, inference_input
from finagents.llm import llm_status
from finagents.research import approvals
from finagents.research.registry import HYPOTHESES
from finagents.retrieval import ARTIFACTS

RUNS = PROJECT_ROOT / "runs"
app = FastAPI(title="FinAgents API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=8)
def system(config: str) -> Orchestrator:
    return Orchestrator(load_config(config))


def _configs() -> list[str]:
    """Top-level configs plus research-loop candidates (``candidates/<name>``)."""
    top = sorted(p.stem for p in CONFIG_DIR.glob("*.toml"))
    return top + sorted(f"candidates/{p.stem}" for p in (CONFIG_DIR / "candidates").glob("*.toml"))


def _read_json(path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": llm_status(),
        "reranker_trained": (ARTIFACTS / "reranker.json").exists(),
        "configs": _configs(),
    }


@app.get("/api/agents")
def agents() -> list[dict[str, Any]]:
    return [{"name": n, "responsibility": r, "tools": t} for n, r, t in AGENT_ROSTER]


@app.get("/api/configs")
def configs() -> list[dict[str, Any]]:
    return [
        {"name": n, **{k: v for k, v in load_config(n).model_dump().items() if k != "name"}}
        for n in _configs()
    ]


@app.get("/api/examples")
def examples(
    split: Literal["train", "dev"] = "dev",
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    q: str = "",
) -> dict[str, Any]:
    records = cached_split(split)
    indexed = [(i, r) for i, r in enumerate(records) if q.lower() in r["qa"]["question"].lower()]
    page = indexed[offset : offset + limit]
    return {
        "total": len(indexed),
        "items": [{"index": i, "id": r["id"], "question": r["qa"]["question"]} for i, r in page],
    }


@app.get("/api/examples/{split}/{index}")
def example(split: Literal["train", "dev"], index: int) -> dict[str, Any]:
    records = cached_split(split)
    if not 0 <= index < len(records):
        raise HTTPException(404, "no such example")
    return {"index": index, **inference_input(records[index])}


class ReportInput(BaseModel):
    question: str = Field(min_length=3)
    table: list[list[str]] = Field(default_factory=list)
    pre_text: list[str] = Field(default_factory=list)
    post_text: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    split: Literal["train", "dev"] | None = None
    index: int | None = None
    record: ReportInput | None = None
    question: str | None = Field(None, description="ask this instead of the stored question")
    config: str = "full"


def _prepare(request: AskRequest) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The record to answer and, for a dataset question asked verbatim, its reference."""
    if request.config not in _configs():
        raise HTTPException(400, f"unknown config {request.config!r}")
    if request.record is not None:
        record: dict[str, Any] = {"id": "user-report", **request.record.model_dump()}
        reference = None
    elif request.split and request.index is not None:
        records = cached_split(request.split)
        if not 0 <= request.index < len(records):
            raise HTTPException(404, "no such example")
        record = records[request.index]
        reference = gold(record)  # returned for comparison only; agents never see it
    else:
        raise HTTPException(400, "send either split + index or a record")
    if not record.get("table") and not record.get("pre_text") and not record.get("post_text"):
        raise HTTPException(400, "the report needs a table or text")
    if request.question and request.question.strip():
        asked = request.question.strip()
        if "qa" in record and asked.lower() != record["qa"]["question"].strip().lower():
            reference = None  # a new question about a dataset report has no gold answer
        record = {**inference_input(record), "question": asked}
    return record, reference


def _finish(result: dict[str, Any], record: dict[str, Any], reference: dict[str, Any] | None):
    result["report"] = inference_input(record)
    if reference:
        result["reference"] = {
            "exe_ans": reference["exe_ans"],
            "answer": reference["answer"],
            "program": reference["program"],
            "gold_inds": list(reference["gold_inds"]),
        }
    return result


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    record, reference = _prepare(request)
    return _finish(system(request.config).run(record), record, reference)


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: one ``step`` event per agent as it finishes, then ``result``."""
    record, reference = _prepare(request)

    def events():
        try:
            for kind, item in system(request.config).stream(record):
                if kind == "result":
                    item = _finish(item, record, reference)
                yield f"event: {kind}\ndata: {json.dumps(item)}\n\n"
        except Exception as error:  # reported in the chat instead of a broken stream
            detail = json.dumps({"detail": f"{type(error).__name__}: {error}"})
            yield f"event: error\ndata: {detail}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/evaluation")
def evaluation(split: str = "dev") -> dict[str, Any]:
    comparison = _read_json(RUNS / f"comparison_{split}.json")
    retrieval = _read_json(RUNS / f"retrieval_{split}.json")
    if comparison is None:
        raise HTTPException(
            404, f"no evaluation yet: run `uv run finagents evaluate --split {split}`"
        )
    return {"comparison": comparison, "retrieval": retrieval}


# Research lab ------------------------------------------------------------------------------

_research: dict[str, Any] = {"running": False, "last": None, "error": None, "log": []}
_lock = threading.Lock()


class ResearchRequest(BaseModel):
    config: str = "full"
    iterations: int = Field(3, ge=1, le=9)


@app.get("/api/research/hypotheses")
def hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "id": h.id,
            "weakness": h.weakness,
            "statement": h.statement,
            "change": h.change,
            "objective": h.objective,
            "guard": h.guard,
            "protected": h.protected,
            "approval": approvals.status("run_hypothesis", h.id),
        }
        for h in HYPOTHESES
    ]


@app.post("/api/research/run")
def start_research(request: ResearchRequest) -> dict[str, Any]:
    from finagents.research.loop import ResearchLoop

    with _lock:
        if _research["running"]:
            raise HTTPException(409, "a research session is already running")
        _research.update(running=True, error=None, log=[])

    def work() -> None:
        try:
            loop = ResearchLoop(
                request.config, max_iterations=request.iterations, log=_research["log"].append
            )
            _research["last"] = loop.run()
        except Exception as error:  # surfaced to the UI; the loop itself never raises by design
            _research["error"] = f"{type(error).__name__}: {error}"
        finally:
            _research["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.get("/api/research/status")
def research_status() -> dict[str, Any]:
    return {
        "running": _research["running"],
        "error": _research["error"],
        "log": _research["log"][-20:],
        "last": _research["last"],
    }


@app.get("/api/research/sessions")
def research_sessions() -> list[dict[str, Any]]:
    directory = RUNS / "research"
    sessions = []
    for path in sorted(directory.glob("R*.json"), reverse=True) if directory.exists() else []:
        entries = json.loads(path.read_text())
        sessions.append({"session": path.stem, "entries": entries})
    return sessions


@app.get("/api/research/approvals")
def research_approvals() -> list[dict[str, Any]]:
    return approvals.list_requests()


class Decision(BaseModel):
    decision: Literal["approved", "rejected"]


@app.post("/api/research/approvals/{request_id}")
def decide(request_id: str, body: Decision) -> dict[str, Any]:
    try:
        return approvals.decide(request_id, body.decision)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
