"""The research agent team that experiments on FinAgents itself.

Seven agents, each with one job:

1. Research Analyst    - reads the verified literature notes and the current results; names weaknesses
2. Dataset Agent       - profiles the train split: question types, evidence patterns, program lengths
3. Experiment Designer - turns a weakness into a hypothesis, a control and an acceptance criterion
4. ML Engineering      - implements the variant (a validated config) and saves accepted candidates
5. Evaluation Agent    - runs control and variant on the same train questions and compares them
6. Research Critic     - checks leakage, statistical significance and a second train slice
7. Documentation Agent - appends every iteration to docs/research-log.md and runs/research/

Selection uses train questions only. Dev is never used inside the loop; the official test
split needs human approval.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from finagents.agents.query_analyst import classify
from finagents.config import CONFIG_DIR, SystemConfig
from finagents.core.program import parse_program
from finagents.core.records import PROJECT_ROOT, gold, load_split
from finagents.evaluation.metrics import mcnemar
from finagents.evaluation.runner import evaluate

from . import approvals
from .registry import HYPOTHESES, TUNABLE, Hypothesis, apply_change

SELECTION = slice(1000, 1600)  # train questions used to choose
CONFIRMATION = slice(2000, 2600)  # a second train slice the critic checks for over-fitting
ALPHA = 0.05
GUARD_TOLERANCE = 0.01
RESEARCH_LOG = PROJECT_ROOT / "docs" / "research-log.md"
CANDIDATES = CONFIG_DIR / "candidates"
LITERATURE = PROJECT_ROOT / "docs" / "background" / "research-notes.md"


def _higher_is_better(metric: str) -> bool:
    return metric != "wrong_answer_rate"


class ResearchAnalyst:
    name = "research_analyst"

    def literature(self) -> list[str]:
        """Verified facts from the project's sourced notes (no claims are generated)."""
        if not LITERATURE.exists():
            return []
        facts = re.findall(r"\*\*Verified source fact[^*]*\*\*\s*(.+)", LITERATURE.read_text())
        return [fact.strip() for fact in facts]

    def weaknesses(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Weaknesses ranked by how many questions they affect."""
        n = summary["questions"]
        found = [
            {
                "weakness": "wrong_answers",
                "affected": round(summary["wrong_answer_rate"] * n),
                "evidence": f"wrong-answer rate {summary['wrong_answer_rate']:.1%}",
            },
            {
                "weakness": "over_abstention",
                "affected": summary["withheld_but_correct"],
                "evidence": f"{summary['withheld_but_correct']} correct answers withheld",
            },
        ]
        abstentions = summary.get("abstentions", {})
        for kind in ("insufficient_evidence", "no_program"):
            if abstentions.get(kind):
                found.append(
                    {
                        "weakness": kind,
                        "affected": abstentions[kind],
                        "evidence": f"{abstentions[kind]} abstentions: {kind.replace('_', ' ')}",
                    }
                )
        ops = [
            (op, s)
            for op, s in summary["by_operation"].items()
            if s["questions"] >= 20 and op != "unsupported"
        ]
        if ops:
            op, stats = min(ops, key=lambda item: item[1]["execution_accuracy"])
            found.append(
                {
                    "weakness": "weak_operation",
                    "affected": round((1 - stats["execution_accuracy"]) * stats["questions"]),
                    "evidence": f"{op}: {stats['execution_accuracy']:.1%} accuracy on {stats['questions']} questions",
                }
            )
        return sorted(found, key=lambda w: -w["affected"])


class DatasetAgent:
    name = "dataset_agent"

    def profile(self, records) -> dict[str, Any]:
        ops = Counter(classify(r["qa"]["question"]).value for r in records)
        sources = Counter()
        facts = Counter()
        steps = Counter()
        for r in records:
            keys = gold(r)["gold_inds"]
            facts[min(len(keys), 4)] += 1
            kinds = {k.split("_")[0] for k in keys}
            sources["table+text" if len(kinds) > 1 else next(iter(kinds), "none")] += 1
            steps[min(len(parse_program(gold(r)["program"]).steps), 3)] += 1
        n = len(records)
        share = lambda c: {str(k): round(v / n, 4) for k, v in sorted(c.items())}  # noqa: E731
        return {
            "split": "train",
            "questions": n,
            "operation_types": share(ops),
            "evidence_source": share(sources),
            "gold_facts (4 = 4+)": share(facts),
            "program_steps (3 = 3+)": share(steps),
        }


class ExperimentDesigner:
    name = "experiment_designer"

    def choose(self, weaknesses, tried: set[str]) -> tuple[Hypothesis | None, str]:
        for weakness in weaknesses:
            for hypothesis in HYPOTHESES:
                if hypothesis.weakness == weakness["weakness"] and hypothesis.id not in tried:
                    return (
                        hypothesis,
                        f"addresses '{weakness['weakness']}' ({weakness['evidence']})",
                    )
        return None, "no untested hypothesis matches the remaining weaknesses"

    def design(self, hypothesis: Hypothesis, baseline: dict[str, Any]) -> dict[str, Any]:
        direction = "increase" if _higher_is_better(hypothesis.objective) else "decrease"
        return {
            "hypothesis": hypothesis.id,
            "statement": hypothesis.statement,
            "control": baseline["name"],
            "change": hypothesis.change,
            "selection": f"train[{SELECTION.start}:{SELECTION.stop}]",
            "confirmation": f"train[{CONFIRMATION.start}:{CONFIRMATION.stop}]",
            "acceptance": (
                f"{hypothesis.objective} must {direction} with exact McNemar p < {ALPHA} on the "
                f"selection slice, move the same way on the confirmation slice, and "
                f"{hypothesis.guard or 'nothing'} must not worsen by more than {GUARD_TOLERANCE:.0%}"
            ),
        }


class MLEngineer:
    name = "ml_engineer"

    def implement(
        self, baseline: dict[str, Any], hypothesis: Hypothesis, session: str
    ) -> SystemConfig:
        settings = apply_change(baseline, hypothesis.change)
        settings["name"] = f"{session}-{hypothesis.id}"
        settings["description"] = f"{hypothesis.statement} (variant of {baseline['name']})"
        return SystemConfig(**settings)  # validates every value

    def save_candidate(self, config: SystemConfig) -> Path:
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        path = CANDIDATES / f"{config.name}.toml"
        lines = [
            "# Candidate accepted by the research loop. Promote it only with human approval.\n"
        ]
        for key, value in config.model_dump(mode="json").items():
            lines.append(f"{key} = {json.dumps(value)}\n")
        path.write_text("".join(lines))
        tomllib.loads(path.read_text())  # the saved file must load back
        return path


class EvaluationAgent:
    name = "evaluation_agent"

    def run(self, config: SystemConfig, records) -> tuple[list[dict], dict]:
        summary = evaluate(config, records, split="train", save=False)
        rows = summary.pop("rows")
        return rows, summary

    @staticmethod
    def delta(control: dict, variant: dict) -> dict[str, float]:
        keys = (
            "execution_accuracy",
            "coverage",
            "selective_accuracy",
            "wrong_answer_rate",
            "cites_all_gold",
        )
        return {k: round(variant[k] - control[k], 4) for k in keys}


class ResearchCritic:
    name = "research_critic"

    def review(
        self,
        hypothesis: Hypothesis,
        experiment: dict,
        control_rows,
        variant_rows,
        control_summary,
        variant_summary,
        confirm_control,
        confirm_variant,
    ) -> dict[str, Any]:
        problems, notes = [], []
        # Leakage and protocol
        if set(hypothesis.change) - TUNABLE:
            problems.append(
                f"changes non-tunable settings {sorted(set(hypothesis.change) - TUNABLE)}"
            )
        if "train" not in experiment["selection"] or "train" not in experiment["confirmation"]:
            problems.append("experiment did not run on train")
        notes.append(
            "leakage check: selection and confirmation are train slices; dev and test untouched"
        )
        # Significance on the objective
        objective, guard = hypothesis.objective, hypothesis.guard
        if objective == "wrong_answer_rate":
            outcome = lambda rows: [not (r["answered"] and not r["correct"]) for r in rows]  # noqa: E731
        else:
            outcome = lambda rows: [r["correct"] for r in rows]  # noqa: E731
        test = mcnemar(outcome(control_rows), outcome(variant_rows))
        improved = test["right_only"] > test["left_only"]
        if not improved or test["p_value"] >= ALPHA:
            problems.append(
                f"{objective}: {test['right_only']} better vs {test['left_only']} worse, "
                f"p = {test['p_value']:.3g} (needs improvement with p < {ALPHA})"
            )
        else:
            notes.append(
                f"{objective} improved: {test['right_only']} vs {test['left_only']}, p = {test['p_value']:.3g}"
            )
        # Guard metric
        if guard:
            change = variant_summary[guard] - control_summary[guard]
            worse = -change if _higher_is_better(guard) else change
            if worse > GUARD_TOLERANCE:
                problems.append(f"guard {guard} worsened by {worse:.1%}")
        # Confirmation slice: same direction, guard holds
        confirm_change = confirm_variant[objective] - confirm_control[objective]
        same_direction = confirm_change > 0 if _higher_is_better(objective) else confirm_change < 0
        if not same_direction:
            problems.append(
                f"confirmation slice: {objective} moved {confirm_change:+.1%} (wrong direction)"
            )
        else:
            notes.append(f"confirmation slice: {objective} {confirm_change:+.1%}")
        return {"accepted": not problems, "problems": problems, "notes": notes, "mcnemar": test}

    def analyze_failure(self, verdict: dict, delta: dict) -> str:
        if not verdict["problems"]:
            return ""
        biggest = max(delta, key=lambda k: abs(delta[k]))
        return (
            f"Rejected: {verdict['problems'][0]}. Largest movement: {biggest} {delta[biggest]:+.1%}. "
            "The change is kept out of the candidate set; the next iteration tries another hypothesis."
        )


class DocumentationAgent:
    name = "documentation_agent"

    def record(self, session: str, entry: dict[str, Any]) -> None:
        directory = PROJECT_ROOT / "runs" / "research"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{session}.json"
        entries = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps([*entries, entry], indent=2))
        RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not RESEARCH_LOG.exists():
            RESEARCH_LOG.write_text(
                "# Research log\n\nWritten by the Documentation Agent. Every entry is one "
                "iteration of the improvement loop, on train questions only.\n"
            )
        with RESEARCH_LOG.open("a") as log:
            log.write(self._markdown(session, entry))

    @staticmethod
    def _markdown(session: str, e: dict[str, Any]) -> str:
        lines = [
            f"\n## {session} · iteration {e['iteration']} · {e.get('hypothesis') or 'no hypothesis'}\n",
            f"- **Weakness:** {e.get('weakness', '-')}\n",
            f"- **Outcome:** {e['outcome']}\n",
        ]
        if e.get("statement"):
            lines.append(f"- **Hypothesis:** {e['statement']}\n")
        if e.get("acceptance"):
            lines.append(f"- **Acceptance:** {e['acceptance']}\n")
        if e.get("delta"):
            lines.append(
                "- **Change vs control (selection):** "
                + ", ".join(f"{k} {v:+.1%}" for k, v in e["delta"].items())
                + "\n"
            )
        for note in e.get("notes", []) + e.get("problems", []):
            lines.append(f"  - {note}\n")
        if e.get("candidate"):
            lines.append(f"- **Candidate saved:** `{e['candidate']}`\n")
        return "".join(lines)


def selection_records():
    train = load_split("train")
    return train[SELECTION], train[CONFIRMATION]


def approved(hypothesis: Hypothesis) -> bool:
    return approvals.status("run_hypothesis", hypothesis.id) == "approved"
