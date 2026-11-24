"""Agent 1 - Master Orchestrator: the LangGraph workflow that coordinates the other agents.

The orchestrator owns every transition. After each agent finishes it reads the shared
state, decides the next step, and records that decision in the trace:

    question -> orchestrator -> query analyst -> evidence researcher -> evidence auditor
       auditor:      valid -> reasoning | insufficient -> research again (<= max rounds) | abstain
       reasoning:    candidates -> program validator | none -> research again | abstain
       validator:    valid -> calculation | invalid -> next candidate | research again | abstain
       calculation:  ok -> verification | error -> next candidate | ...
       verification: verified -> response | failed checks -> next candidate
                     low confidence / ambiguous -> research again | conflict -> abstain
    abstain -> response -> answer + evidence + calculation, or a reasoned abstention

Retry limits are the research-round budget and the candidate list, so every run ends.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from finagents.config import SystemConfig
from finagents.core.records import inference_input
from finagents.llm import LLMClient, get_llm
from finagents.retrieval import build_retriever
from finagents.state import AgentState, AgentStep, FinalAnswer, Operation

from .base import Agent
from .compute import CalculationAgent, VerificationAgent
from .evidence import EvidenceAuditor, EvidenceResearcher
from .query_analyst import QueryAnalyst
from .reasoning import ProgramValidator, ReasoningAgent
from .response import ResponseAgent

AGENT_ROSTER = [
    (
        "orchestrator",
        "Coordinate the workflow, route between agents, enforce retry limits",
        ["langgraph", "state_manager"],
    ),
    (QueryAnalyst.name, QueryAnalyst.responsibility, list(QueryAnalyst.tools)),
    (EvidenceResearcher.name, EvidenceResearcher.responsibility, list(EvidenceResearcher.tools)),
    (EvidenceAuditor.name, EvidenceAuditor.responsibility, list(EvidenceAuditor.tools)),
    (ReasoningAgent.name, ReasoningAgent.responsibility, list(ReasoningAgent.tools)),
    (ProgramValidator.name, ProgramValidator.responsibility, list(ProgramValidator.tools)),
    (CalculationAgent.name, CalculationAgent.responsibility, list(CalculationAgent.tools)),
    (VerificationAgent.name, VerificationAgent.responsibility, list(VerificationAgent.tools)),
    (ResponseAgent.name, ResponseAgent.responsibility, list(ResponseAgent.tools)),
]


class Orchestrator:
    """Builds the agents and the graph; ``run(record)`` answers one question."""

    def __init__(
        self, config: SystemConfig, *, llm: LLMClient | None = None, retriever=None
    ) -> None:
        self.config = config
        self.llm = (
            llm if llm is not None else (get_llm(config.llm_model) if config.use_llm else None)
        )
        self.retriever = retriever or build_retriever(config.retriever)
        self.agents: dict[str, Agent] = {
            "query_analyst": QueryAnalyst(config, self.llm),
            "evidence_researcher": EvidenceResearcher(config, self.llm, retriever=self.retriever),
            "evidence_auditor": EvidenceAuditor(config, self.llm),
            "reasoning": ReasoningAgent(config, self.llm),
            "program_validator": ProgramValidator(config, self.llm),
            "calculation": CalculationAgent(config, self.llm),
            "verification": VerificationAgent(config, self.llm),
            "response": ResponseAgent(config, self.llm),
        }
        self.graph = self._build()

    # Graph -------------------------------------------------------------------------------

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("orchestrator", self._start)
        graph.add_node("abstain", self._abstain)
        for name, agent in self.agents.items():
            graph.add_node(name, self._managed(name, agent))
        graph.add_edge(START, "orchestrator")
        graph.add_edge("orchestrator", "query_analyst")
        routed = [
            "query_analyst",
            "evidence_researcher",
            "evidence_auditor",
            "reasoning",
            "program_validator",
            "calculation",
            "verification",
        ]
        destinations = [
            "evidence_researcher",
            "evidence_auditor",
            "reasoning",
            "program_validator",
            "calculation",
            "verification",
            "response",
            "abstain",
        ]
        for name in routed:
            graph.add_conditional_edges(name, lambda state: state["route"], destinations)
        graph.add_edge("abstain", "response")
        graph.add_edge("response", END)
        return graph.compile()

    def _managed(self, name: str, agent: Agent) -> Callable[[AgentState], dict[str, Any]]:
        """Run an agent, then let the orchestrator decide the next step and update state."""

        def node(state: AgentState) -> dict[str, Any]:
            update = agent(state)
            if name == "response":
                return update
            route, changes = self.decide(name, {**state, **update})
            update.update(changes)
            update["route"] = route
            update["trace"][-1].route = route
            return update

        return node

    def _start(self, state: AgentState) -> dict[str, Any]:
        budget = (
            f"budget: {self.config.max_research_rounds} research round(s), "
            f"{self.config.max_candidates} candidate program(s)"
        )
        llm = f"LLM: {self.llm.name}" if self.llm else "LLM: off (deterministic backends)"
        step = AgentStep(
            agent="orchestrator",
            summary=f"plan: analyse -> research -> audit -> reason -> "
            f"validate -> calculate -> verify -> respond; {budget}; {llm}",
            tools=["langgraph", "state_manager"],
            route="query_analyst",
            details={"config": self.config.name, "retriever": self.retriever.name},
        )
        return {"trace": [step], "attempts": [], "research_round": 1, "abstain_reason": None}

    def _abstain(self, state: AgentState) -> dict[str, Any]:
        step = AgentStep(
            agent="orchestrator",
            summary=f"abstain: {state.get('abstain_reason')}",
            tools=["state_manager"],
            route="response",
        )
        return {"trace": [step]}

    # Routing policy ------------------------------------------------------------------------

    def decide(self, after: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The next node and any state changes the orchestrator makes (retry counters)."""
        if after == "query_analyst":
            if state["analysis"].operation is Operation.UNSUPPORTED:
                return "abstain", {
                    "abstain_reason": "unsupported question: no supported calculation shape"
                }
            return "evidence_researcher", {}
        if after == "evidence_researcher":
            return "evidence_auditor", {}
        if after == "evidence_auditor":
            audit = state["audit"]
            if audit.status == "valid":
                return "reasoning", {}
            if self._can_research(state):
                return "evidence_researcher", {"research_round": state["research_round"] + 1}
            if audit.evidence and len(audit.evidence) >= max(
                state["analysis"].required_evidence, 1
            ):
                return "reasoning", {}  # budget spent: reason over what was found
            return "abstain", {"abstain_reason": f"insufficient evidence: {audit.reason}"}
        if after == "reasoning":
            if state["candidates"]:
                return "program_validator", {}
            return self._research_or_abstain(state, "no calculation fits the audited evidence")
        if after == "program_validator":
            if state["validation"].valid:
                return "calculation", {}
            return self._next_candidate(state, "no valid program")
        if after == "calculation":
            if state["calculation"].value is not None:
                return "verification", {}
            return self._next_candidate(state, "no program executed")
        if after == "verification":
            verification = state["verification"]
            if verification.verified:
                return "response", {}
            if verification.failure == "checks":
                return self._next_candidate(
                    state, f"no candidate passed verification ({verification.reason})"
                )
            if verification.failure == "conflict":
                return "abstain", {"abstain_reason": verification.reason}
            return self._research_or_abstain(state, verification.reason)
        raise ValueError(f"no routing rule after {after!r}")

    def _can_research(self, state) -> bool:
        return state.get("research_round", 1) < self.config.max_research_rounds

    def _next_candidate(self, state, reason: str) -> tuple[str, dict[str, Any]]:
        index = state["candidate_index"] + 1
        if index < len(state["candidates"]):
            return "program_validator", {"candidate_index": index}
        return self._research_or_abstain(state, reason)

    def _research_or_abstain(self, state, reason: str) -> tuple[str, dict[str, Any]]:
        if self._can_research(state):
            return "evidence_researcher", {"research_round": state["research_round"] + 1}
        if not self.config.abstain:
            fallback = self._best_executed(state)
            if fallback is not None:
                return "response", fallback
        return "abstain", {"abstain_reason": reason}

    def _best_executed(self, state) -> dict[str, Any] | None:
        """No-abstention ablation: answer with the best candidate that executed."""
        from finagents.core.executor import execute_program

        for index, candidate in enumerate(state.get("candidates", [])):
            result = execute_program(candidate.program, state["payload"]["table"])
            if result.value is not None:
                calc = self.agents["calculation"]({**state, "candidate_index": index})[
                    "calculation"
                ]
                return {"candidate_index": index, "calculation": calc}
        return None

    # Entry point ---------------------------------------------------------------------------

    def run(self, record: dict[str, Any]) -> dict[str, Any]:
        """Answer one FinQA record or user report. Returns the answer, trace and timing."""
        result: dict[str, Any] = {}
        for kind, item in self.stream(record):
            if kind == "result":
                result = item
        return result

    def stream(self, record: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``("step", trace_entry)`` as each agent finishes, then ``("result", result)``."""
        started = time.perf_counter()
        payload = inference_input(record)
        state: dict[str, Any] = {}
        sent = 0
        for state in self.graph.stream(
            {"payload": payload}, {"recursion_limit": 200}, stream_mode="values"
        ):
            for step in state.get("trace", [])[sent:]:
                yield "step", step.model_dump(mode="json")
            sent = len(state.get("trace", []))
        yield "result", self._result(payload, state, started)

    def _result(
        self, payload: dict[str, Any], state: dict[str, Any], started: float
    ) -> dict[str, Any]:
        final: FinalAnswer = state["final"]
        return {
            "id": payload["id"],
            "question": payload["question"],
            "config": self.config.name,
            "answer": final.model_dump(mode="json"),
            "analysis": state["analysis"].model_dump(mode="json"),
            "trace": [step.model_dump(mode="json") for step in state["trace"]],
            "retrieved": [{"key": s.key, "rank": s.rank} for s in state.get("retrieved", [])],
            "evidence": [
                v.model_dump(mode="json")
                for v in (state["audit"].evidence if "audit" in state else [])
            ],
            "research_rounds": state.get("research_round", 1),
            "candidates_tried": len(state.get("attempts", [])),
            "milliseconds": round((time.perf_counter() - started) * 1000, 2),
        }
