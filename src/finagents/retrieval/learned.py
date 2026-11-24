"""Learned evidence reranker: a small neural network trained on train-split gold evidence.

For each (question, segment) pair it combines lexical scores with structural signals: does
the segment mention the question's years, is it a table row, does its row label match the
question, how many numbers it holds. The label is whether the segment is in FinQA's
``gold_inds``. It trains in about a minute on CPU and is stored as plain JSON weights
(no pickle), so loading it never executes code.

Model choice (MLP vs logistic regression) is made on a held-out slice of *train*; dev is
used only to report the final recall.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from finagents.core.records import gold, inference_input
from finagents.core.segments import Segment, build_segments, tokenize
from finagents.core.values import NUMBER_SPAN, YEAR

from .lexical import Ranked, TfidfModel, bm25_scores, ranked

FEATURES = (
    "bm25_norm", "bm25_rr", "tfidf", "tfidf_rr", "overlap", "label_overlap",
    "year_coverage", "has_year", "is_table", "is_header", "numbers", "length", "position",
)  # fmt: skip


def _reciprocal_ranks(scores: Sequence[float]) -> list[float]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rr = [0.0] * len(scores)
    for rank, index in enumerate(order, start=1):
        rr[index] = 1.0 / rank
    return rr


def featurize(segments: Sequence[Segment], query: str, tfidf: TfidfModel) -> np.ndarray:
    """One feature row per segment; every feature is computed from inference inputs only."""
    bm25 = bm25_scores(segments, query)
    top = max(bm25) or 1.0
    tf = tfidf.scores(segments, query)
    q_terms = set(tokenize(query))
    q_years = set(YEAR.findall(query))
    bm25_rr, tf_rr = _reciprocal_ranks(bm25), _reciprocal_ranks(tf)
    rows = []
    for i, segment in enumerate(segments):
        text = segment.retrieval_text
        terms = set(tokenize(text))
        label_terms = set(tokenize(segment.cells[0] if segment.cells else text[:60]))
        years = set(YEAR.findall(text))
        rows.append(
            [
                bm25[i] / top,
                bm25_rr[i],
                tf[i],
                tf_rr[i],
                len(q_terms & terms) / max(len(q_terms), 1),
                len(q_terms & label_terms) / max(len(label_terms), 1),
                len(q_years & years) / max(len(q_years), 1) if q_years else 0.0,
                1.0 if q_years & years else 0.0,
                1.0 if segment.source == "table" else 0.0,
                1.0 if segment.key == "table_0" else 0.0,
                math.log1p(len(NUMBER_SPAN.findall(segment.text))),
                math.log1p(len(terms)),
                i / max(len(segments) - 1, 1),
            ]
        )
    return np.asarray(rows, dtype=float)


class LearnedReranker:
    """Inference for the trained reranker: standardize, then logistic or one-hidden-layer MLP."""

    name = "learned"

    def __init__(self, weights: dict, tfidf: TfidfModel) -> None:
        self.weights = weights
        self.tfidf = tfidf
        self.mean = np.asarray(weights["mean"])
        self.scale = np.asarray(weights["scale"])

    def score(self, segments: Sequence[Segment], query: str) -> np.ndarray:
        x = (featurize(segments, query, self.tfidf) - self.mean) / self.scale
        if self.weights["kind"] == "mlp":
            hidden = np.maximum(
                0.0, x @ np.asarray(self.weights["w1"]) + np.asarray(self.weights["b1"])
            )
            logits = hidden @ np.asarray(self.weights["w2"]) + np.asarray(self.weights["b2"])
        else:
            logits = x @ np.asarray(self.weights["w"]) + self.weights["b"]
        return np.asarray(logits).reshape(-1)

    def rank(self, segments: Sequence[Segment], query: str) -> list[Ranked]:
        return ranked(segments, self.score(segments, query).tolist())

    @classmethod
    def load(cls, directory: Path) -> LearnedReranker:
        weights = json.loads((directory / "reranker.json").read_text())
        return cls(weights, TfidfModel.load(directory / "tfidf.json"))


def _dataset(records, tfidf: TfidfModel):
    features, labels, groups = [], [], []
    for group, record in enumerate(records):
        payload = inference_input(record)
        segments = build_segments(payload)
        gold_keys = set(gold(record)["gold_inds"])
        features.append(featurize(segments, payload["question"], tfidf))
        labels += [1 if s.key in gold_keys else 0 for s in segments]
        groups += [group] * len(segments)
    return np.vstack(features), np.asarray(labels), np.asarray(groups)


def _recall_at(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Share of questions whose gold segments are all in the top k."""
    hits = total = 0
    for group in np.unique(groups):
        mask = groups == group
        if labels[mask].sum() == 0:
            continue
        top = np.argsort(-scores[mask], kind="stable")[:k]
        total += 1
        hits += labels[mask][top].sum() == labels[mask].sum()
    return hits / max(total, 1)


def train_reranker(train_records, directory: Path, *, seed: int = 42, log=print) -> dict:
    """Fit TF-IDF and the reranker on train only; select the model on a train holdout."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    records = list(train_records)
    random.Random(seed).shuffle(records)
    holdout_size = len(records) // 10
    fit_records, holdout = records[holdout_size:], records[:holdout_size]

    texts = [s.retrieval_text for r in fit_records for s in build_segments(inference_input(r))]
    tfidf = TfidfModel.fit(texts)
    x_fit, y_fit, _ = _dataset(fit_records, tfidf)
    x_hold, y_hold, g_hold = _dataset(holdout, tfidf)
    scaler = StandardScaler().fit(x_fit)
    xs_fit, xs_hold = scaler.transform(x_fit), scaler.transform(x_hold)

    candidates = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }
    report = {"fit_records": len(fit_records), "holdout_records": len(holdout), "models": {}}
    best_name, best_recall = None, -1.0
    for name, model in candidates.items():
        model.fit(xs_fit, y_fit)
        scores = (
            model.decision_function(xs_hold)
            if name == "logistic"
            else model.predict_proba(xs_hold)[:, 1]
        )
        recall = {k: round(float(_recall_at(scores, y_hold, g_hold, k)), 4) for k in (3, 5, 10)}
        report["models"][name] = recall
        log(f"  {name}: train-holdout all-gold recall {recall}")
        if recall[5] > best_recall:
            best_name, best_recall = name, recall[5]
    model = candidates[best_name]
    weights = {
        "kind": best_name,
        "features": list(FEATURES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    if best_name == "mlp":
        weights |= {
            "w1": model.coefs_[0].tolist(),
            "b1": model.intercepts_[0].tolist(),
            "w2": model.coefs_[1].tolist(),
            "b2": model.intercepts_[1].tolist(),
        }
    else:
        weights |= {"w": model.coef_[0].tolist(), "b": float(model.intercept_[0])}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reranker.json").write_text(json.dumps(weights))
    tfidf.save(directory / "tfidf.json")
    report["selected"] = best_name
    (directory / "reranker_report.json").write_text(json.dumps(report, indent=2))
    return report
