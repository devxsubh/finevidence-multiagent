"""System configuration: one versioned TOML file per variant (see ``configs/``)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from finagents.core.records import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


class SystemConfig(BaseModel):
    name: str = "full"
    description: str = ""
    # Retrieval
    retriever: Literal["bm25", "tfidf", "learned"] = "learned"
    initial_k: int = Field(5, ge=1)
    k_step: int = Field(5, ge=0)
    max_research_rounds: int = Field(3, ge=1)
    # Audit
    min_relevance: float = Field(0.25, ge=0, le=1)
    strict_units: bool = True
    # Reasoning and verification
    max_candidates: int = Field(12, ge=1)
    verify: bool = True
    # Abstention policy
    abstain: bool = True
    min_confidence: float = Field(0.6, ge=0, le=1.5)
    ambiguity_margin: float = Field(0.01, ge=0)
    abstain_on_conflict: bool = True
    period_mismatch_penalty: float = Field(0.6, ge=0, le=1)
    # LLM use: an agent uses the LLM only if this is on and a provider is available.
    use_llm: bool = False
    llm_agents: list[Literal["query_analyst", "reasoning", "response"]] = Field(
        default_factory=lambda: ["query_analyst", "reasoning", "response"]
    )
    llm_model: str = "claude-opus-5-5"


def load_config(name_or_path: str | Path = "full") -> SystemConfig:
    """Load ``configs/<name>.toml`` or a path to a TOML file."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.toml"
    return SystemConfig(**tomllib.loads(path.read_text()))
