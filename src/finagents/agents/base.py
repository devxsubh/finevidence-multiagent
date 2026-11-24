"""What every agent shares: a name, a fixed tool list, and a timed trace entry per run."""

from __future__ import annotations

import time
from typing import Any

from finagents.config import SystemConfig
from finagents.llm import LLMClient
from finagents.state import AgentState, AgentStep


class Agent:
    """Subclasses set ``name``, ``responsibility`` and ``tools`` and implement ``run``.

    ``run`` returns ``(update, summary, details)``. The agent may only return the state
    fields it owns; ``__call__`` adds the trace entry.
    """

    name = "agent"
    responsibility = ""
    tools: tuple[str, ...] = ()

    def __init__(self, config: SystemConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm if config.use_llm and self.name in config.llm_agents else None

    def run(self, state: AgentState) -> tuple[dict[str, Any], str, dict[str, Any]]:
        raise NotImplementedError

    def __call__(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        update, summary, details = self.run(state)
        step = AgentStep(
            agent=self.name,
            summary=summary,
            tools=list(self.tools),
            milliseconds=round((time.perf_counter() - started) * 1000, 2),
            details=details,
        )
        return {**update, "trace": [step]}

    def llm_backend(self) -> str:
        return f"llm:{self.llm.name}" if self.llm else "rules"
