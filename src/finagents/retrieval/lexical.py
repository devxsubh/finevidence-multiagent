"""Lexical retrievers: BM25 (no training) and TF-IDF (IDF fitted on the train split).

Both rank the segments of one report for one query. They are the baselines the learned
reranker is compared against, and they produce the reranker's main input features.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from finagents.core.segments import Segment, tokenize


@dataclass(frozen=True)
class Ranked:
    segment: Segment
    score: float
    rank: int  # 1-based


def ranked(segments: Sequence[Segment], scores: Sequence[float]) -> list[Ranked]:
    order = sorted(range(len(segments)), key=lambda i: (-scores[i], i))
    return [Ranked(segments[i], float(scores[i]), r) for r, i in enumerate(order, start=1)]


def bm25_scores(segments: Sequence[Segment], query: str, k1: float = 1.2, b: float = 0.75):
    """Okapi BM25 with term statistics from this report's own segments only."""
    documents = [tokenize(segment.retrieval_text) for segment in segments]
    terms = set(tokenize(query))
    count = len(documents)
    average = sum(map(len, documents)) / max(count, 1) or 1.0
    frequency = Counter(term for document in documents for term in set(document))
    scores = []
    for document in documents:
        tf = Counter(document)
        score = 0.0
        for term in terms & tf.keys():
            idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
            norm = tf[term] + k1 * (1 - b + b * len(document) / average)
            score += idf * tf[term] * (k1 + 1) / norm
        scores.append(score)
    return scores


class Bm25Retriever:
    name = "bm25"

    def rank(self, segments: Sequence[Segment], query: str) -> list[Ranked]:
        return ranked(segments, bm25_scores(segments, query))


class TfidfModel:
    """Inverse document frequencies over train-split segments, stored as plain JSON."""

    def __init__(self, idf: dict[str, float], default_idf: float) -> None:
        self.idf = idf
        self.default_idf = default_idf

    @classmethod
    def fit(cls, segment_texts: Sequence[str]) -> TfidfModel:
        frequency = Counter(term for text in segment_texts for term in set(tokenize(text)))
        count = len(segment_texts)
        idf = {term: math.log((1 + count) / (1 + df)) + 1 for term, df in frequency.items()}
        return cls(idf, math.log(1 + count) + 1)

    def vector(self, text: str) -> dict[str, float]:
        tf = Counter(tokenize(text))
        weights = {
            term: (1 + math.log(n)) * self.idf.get(term, self.default_idf) for term, n in tf.items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {term: w / norm for term, w in weights.items()}

    def scores(self, segments: Sequence[Segment], query: str) -> list[float]:
        q = self.vector(query)
        return [
            sum(weight * q.get(term, 0.0) for term, weight in self.vector(s.retrieval_text).items())
            for s in segments
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"idf": self.idf, "default_idf": self.default_idf}))

    @classmethod
    def load(cls, path: Path) -> TfidfModel:
        data = json.loads(path.read_text())
        return cls(data["idf"], data["default_idf"])


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, model: TfidfModel) -> None:
        self.model = model

    def rank(self, segments: Sequence[Segment], query: str) -> list[Ranked]:
        return ranked(segments, self.model.scores(segments, query))
