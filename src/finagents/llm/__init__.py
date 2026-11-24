"""Model-agnostic LLM interface.

Agents ask for one thing: a structured answer validated against a Pydantic schema. Any
provider that can do that fits ``LLMClient``. The Claude backend is used when the
``anthropic`` package is installed and credentials are available; otherwise ``get_llm``
returns ``None`` and every agent uses its deterministic offline backend.

LLMs decide *what* to do (which operation, which evidence ids). They never supply numbers:
agents substitute values from audited evidence and the executor computes the result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    name: str
    last_error: str | None

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T | None:
        """A validated instance of ``schema``, or ``None`` if the model could not provide one."""
        ...


ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read ``KEY=value`` lines from the project's .env into the environment.

    Variables already set in the environment win. Values are never logged.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def credentials_available() -> bool:
    """True if the Anthropic SDK can find credentials (API key, auth token or a CLI profile)."""
    load_env_file()
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def llm_status() -> dict[str, object]:
    try:
        import anthropic  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    return {"provider": "anthropic", "installed": installed, "credentials": credentials_available()}


def get_llm(model: str = "claude-opus-5-5") -> LLMClient | None:
    status = llm_status()
    if not (status["installed"] and status["credentials"]):
        return None
    from .claude import ClaudeClient

    return ClaudeClient(model=model)
