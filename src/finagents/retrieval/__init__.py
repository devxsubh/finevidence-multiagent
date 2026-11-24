"""Evidence retrieval: BM25 and TF-IDF baselines, and the learned (neural) reranker."""

from __future__ import annotations

from pathlib import Path

from finagents.core.records import PROJECT_ROOT

from .learned import LearnedReranker
from .lexical import Bm25Retriever, Ranked, TfidfModel, TfidfRetriever

ARTIFACTS = PROJECT_ROOT / "artifacts"


def build_retriever(name: str, artifacts: Path = ARTIFACTS):
    """``bm25`` needs nothing; ``tfidf`` and ``learned`` need ``finagents train`` artifacts."""
    if name == "bm25":
        return Bm25Retriever()
    if name in {"tfidf", "learned"} and not (artifacts / "tfidf.json").exists():
        raise FileNotFoundError(f"{artifacts / 'tfidf.json'} missing: run `uv run finagents train`")
    if name == "tfidf":
        return TfidfRetriever(TfidfModel.load(artifacts / "tfidf.json"))
    if name == "learned":
        return LearnedReranker.load(artifacts)
    raise ValueError(f"unknown retriever {name!r}")


__all__ = ["Bm25Retriever", "LearnedReranker", "Ranked", "TfidfRetriever", "build_retriever"]
