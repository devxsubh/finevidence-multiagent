"""The autonomous improvement loop, as a LangGraph workflow.

    analyse performance -> identify weakness -> generate hypothesis -> design experiment
      -> implement changes -> run experiments -> evaluate results -> research critic review
      -> improvement valid?  yes: save candidate  |  no: analyse failure
      -> document -> next iteration (until the budget or the hypotheses run out)

A protected hypothesis is not run: the loop files an approval request, documents it and
moves to the next weakness. An accepted candidate becomes the control for later iterations
in the same session; it does not replace the default system without a human decision.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from finagents.config import SystemConfig, load_config

from . import approvals
from .registry import by_id
from .team import (
    DatasetAgent,
    DocumentationAgent,
    EvaluationAgent,
    ExperimentDesigner,
    MLEngineer,
    ResearchAnalyst,
    ResearchCritic,
    approved,
    selection_records,
)


class ResearchState(TypedDict, total=False):
    session: str
    iteration: int
    max_iterations: int
    control: dict[str, Any]
    control_rows: list
    control_summary: dict
    confirm_control: dict
    weaknesses: list
    hypothesis: str | None
    tried: list[str]
    experiment: dict
    variant: dict
    variant_rows: list
    variant_summary: dict
    confirm_variant: dict
    delta: dict
    verdict: dict
    entry: dict
    candidates: list[str]
    trace: Annotated[list[dict], operator.add]


def _step(agent: str, summary: str, **details) -> dict:
    return {"agent": agent, "summary": summary, "details": details}


class ResearchLoop:
    def __init__(self, base_config: str = "full", *, max_iterations: int = 3, log=print) -> None:
        self.base_config = load_config(base_config)
        self.max_iterations = max_iterations
        self.log = log
        self.analyst, self.dataset = ResearchAnalyst(), DatasetAgent()
        self.designer, self.engineer = ExperimentDesigner(), MLEngineer()
        self.evaluator, self.critic, self.writer = (
            EvaluationAgent(),
            ResearchCritic(),
            DocumentationAgent(),
        )
        self.selection, self.confirmation = selection_records()
        self.graph = self._build()

    def _build(self):
        g = StateGraph(ResearchState)
        for name in (
            "analyze_performance",
            "identify_weakness",
            "generate_hypothesis",
            "request_approval",
            "design_experiment",
            "implement_changes",
            "run_experiments",
            "evaluate_results",
            "critic_review",
            "save_candidate",
            "analyze_failure",
            "document",
        ):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "analyze_performance")
        g.add_edge("analyze_performance", "identify_weakness")
        g.add_edge("identify_weakness", "generate_hypothesis")
        g.add_conditional_edges(
            "generate_hypothesis",
            self._after_hypothesis,
            ["design_experiment", "request_approval", "document"],
        )
        g.add_edge("request_approval", "document")
        for a, b in (
            ("design_experiment", "implement_changes"),
            ("implement_changes", "run_experiments"),
            ("run_experiments", "evaluate_results"),
            ("evaluate_results", "critic_review"),
        ):
            g.add_edge(a, b)
        g.add_conditional_edges(
            "critic_review",
            lambda s: "save_candidate" if s["verdict"]["accepted"] else "analyze_failure",
            ["save_candidate", "analyze_failure"],
        )
        g.add_edge("save_candidate", "document")
        g.add_edge("analyze_failure", "document")
        g.add_conditional_edges("document", self._after_document, ["analyze_performance", END])
        return g.compile()

    # Nodes ---------------------------------------------------------------------------------

    def analyze_performance(self, s: ResearchState):
        steps = []
        if s.get("iteration", 0) == 0:
            facts = self.analyst.literature()
            steps.append(
                _step(
                    "research_analyst",
                    f"read {len(facts)} verified facts from the literature notes",
                    facts=facts,
                )
            )
            profile = self.dataset.profile(self.selection)
            steps.append(
                _step(
                    "dataset_agent",
                    f"profiled {profile['questions']} train questions",
                    profile=profile,
                )
            )
        if "control_rows" not in s:
            control = SystemConfig(**s["control"])
            self.log(
                f"[research] evaluating control {control.name} on train selection + confirmation"
            )
            rows, summary = self.evaluator.run(control, self.selection)
            _, confirm = self.evaluator.run(control, self.confirmation)
            steps.append(
                _step(
                    "evaluation_agent",
                    f"control {control.name}: exec acc {summary['execution_accuracy']:.1%}, "
                    f"wrong-answer rate {summary['wrong_answer_rate']:.1%}, coverage {summary['coverage']:.1%}",
                    metrics={k: v for k, v in summary.items() if k != "settings"},
                )
            )
            return {
                "control_rows": rows,
                "control_summary": summary,
                "confirm_control": confirm,
                "iteration": s.get("iteration", 0) + 1,
                "trace": steps,
            }
        return {"iteration": s.get("iteration", 0) + 1, "trace": steps}

    def identify_weakness(self, s: ResearchState):
        weaknesses = self.analyst.weaknesses(s["control_summary"])
        top = weaknesses[0] if weaknesses else None
        return {
            "weaknesses": weaknesses,
            "trace": [
                _step(
                    "research_analyst",
                    f"largest weakness: {top['weakness']} ({top['evidence']})"
                    if top
                    else "no weakness found",
                    weaknesses=weaknesses,
                )
            ],
        }

    def generate_hypothesis(self, s: ResearchState):
        hypothesis, why = self.designer.choose(s["weaknesses"], set(s.get("tried", [])))
        hid = hypothesis.id if hypothesis else None
        summary = f"{hid}: {hypothesis.statement} ({why})" if hypothesis else why
        return {
            "hypothesis": hid,
            "tried": [*s.get("tried", []), *([hid] if hid else [])],
            "trace": [_step("experiment_designer", summary)],
        }

    def request_approval(self, s: ResearchState):
        h = by_id(s["hypothesis"])
        item = approvals.request("run_hypothesis", h.id, h.reason_protected)
        entry = {
            "iteration": s["iteration"],
            "hypothesis": h.id,
            "statement": h.statement,
            "weakness": h.weakness,
            "outcome": f"awaiting human approval ({item['id']}): {h.reason_protected}",
        }
        return {
            "entry": entry,
            "trace": [
                _step(
                    "experiment_designer",
                    f"{h.id} is protected ({h.reason_protected}); approval request {item['id']} filed, not run",
                )
            ],
        }

    def design_experiment(self, s: ResearchState):
        experiment = self.designer.design(by_id(s["hypothesis"]), s["control"])
        return {
            "experiment": experiment,
            "trace": [_step("experiment_designer", experiment["acceptance"], **experiment)],
        }

    def implement_changes(self, s: ResearchState):
        config = self.engineer.implement(s["control"], by_id(s["hypothesis"]), s["session"])
        changed = {
            k: v
            for k, v in config.model_dump().items()
            if s["control"].get(k) != v and k not in {"name", "description"}
        }
        return {
            "variant": config.model_dump(mode="json"),
            "trace": [
                _step("ml_engineer", f"built variant {config.name}: {changed}", changed=changed)
            ],
        }

    def run_experiments(self, s: ResearchState):
        variant = SystemConfig(**s["variant"])
        self.log(f"[research] evaluating variant {variant.name}")
        rows, summary = self.evaluator.run(variant, self.selection)
        _, confirm = self.evaluator.run(variant, self.confirmation)
        return {
            "variant_rows": rows,
            "variant_summary": summary,
            "confirm_variant": confirm,
            "trace": [
                _step(
                    "evaluation_agent",
                    f"ran control and variant on {len(rows)} selection and "
                    f"{len(self.confirmation)} confirmation train questions",
                )
            ],
        }

    def evaluate_results(self, s: ResearchState):
        delta = self.evaluator.delta(s["control_summary"], s["variant_summary"])
        return {
            "delta": delta,
            "trace": [
                _step(
                    "evaluation_agent",
                    "variant vs control: " + ", ".join(f"{k} {v:+.1%}" for k, v in delta.items()),
                    delta=delta,
                )
            ],
        }

    def critic_review(self, s: ResearchState):
        verdict = self.critic.review(
            by_id(s["hypothesis"]),
            s["experiment"],
            s["control_rows"],
            s["variant_rows"],
            s["control_summary"],
            s["variant_summary"],
            s["confirm_control"],
            s["confirm_variant"],
        )
        summary = (
            "accepted" if verdict["accepted"] else "rejected: " + "; ".join(verdict["problems"])
        )
        return {"verdict": verdict, "trace": [_step("research_critic", summary, **verdict)]}

    def save_candidate(self, s: ResearchState):
        path = self.engineer.save_candidate(SystemConfig(**s["variant"]))
        entry = self._entry(
            s, f"accepted; candidate saved to {path.name} and used as the new control"
        )
        entry["candidate"] = str(path.relative_to(path.parents[2]))
        return {
            "entry": entry,
            "candidates": [*s.get("candidates", []), entry["candidate"]],
            "control": s["variant"],
            "control_rows": s["variant_rows"],
            "control_summary": s["variant_summary"],
            "confirm_control": s["confirm_variant"],
            "trace": [_step("ml_engineer", f"saved candidate {path.name}; it is now the control")],
        }

    def analyze_failure(self, s: ResearchState):
        analysis = self.critic.analyze_failure(s["verdict"], s["delta"])
        return {"entry": self._entry(s, "rejected"), "trace": [_step("research_critic", analysis)]}

    def document(self, s: ResearchState):
        entry = s.get("entry") or {
            "iteration": s["iteration"],
            "outcome": "stopped: no untested hypothesis left",
        }
        self.writer.record(s["session"], entry)
        return {
            "entry": {},
            "trace": [
                _step(
                    "documentation_agent", f"logged iteration {s['iteration']}: {entry['outcome']}"
                )
            ],
        }

    # Routing -------------------------------------------------------------------------------

    def _after_hypothesis(self, s: ResearchState) -> str:
        if not s.get("hypothesis"):
            return "document"
        h = by_id(s["hypothesis"])
        return "request_approval" if h.protected and not approved(h) else "design_experiment"

    def _after_document(self, s: ResearchState):
        if not s.get("hypothesis") or s["iteration"] >= s["max_iterations"]:
            return END
        return "analyze_performance"

    def _entry(self, s: ResearchState, outcome: str) -> dict:
        h = by_id(s["hypothesis"])
        return {
            "iteration": s["iteration"],
            "hypothesis": h.id,
            "statement": h.statement,
            "weakness": h.weakness,
            "acceptance": s["experiment"]["acceptance"],
            "delta": s["delta"],
            "outcome": outcome,
            "notes": s["verdict"]["notes"],
            "problems": s["verdict"]["problems"],
        }

    def run(self) -> dict[str, Any]:
        session = "R" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        state = self.graph.invoke(
            {
                "session": session,
                "iteration": 0,
                "max_iterations": self.max_iterations,
                "control": self.base_config.model_dump(mode="json"),
                "tried": [],
                "candidates": [],
            },
            {"recursion_limit": 500},
        )
        return {
            "session": session,
            "iterations": state["iteration"],
            "candidates": state.get("candidates", []),
            "final_control": state["control"]["name"],
            "trace": state["trace"],
        }
