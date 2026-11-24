"""Claude backend for ``LLMClient``, using structured outputs (``messages.parse``)."""

from __future__ import annotations

import threading
from typing import TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ClaudeClient:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", *, timeout: float = 120.0) -> None:
        self.model = model
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=2)
        self._local = threading.local()  # last_error is per thread: questions run in parallel
        # Running token totals, so a run's API cost can be reported.
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()  # evaluation runs questions in parallel threads

    @property
    def last_error(self) -> str | None:
        return getattr(self._local, "error", None)

    @last_error.setter
    def last_error(self, value: str | None) -> None:
        self._local.error = value

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T | None:
        self.last_error = None
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,  # room for adaptive thinking before the short JSON answer
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except anthropic.AuthenticationError:
            self.last_error = "authentication failed"
        except anthropic.PermissionDeniedError:
            self.last_error = "API key lacks permission"
        except anthropic.RateLimitError:
            self.last_error = "rate limited"
        except anthropic.APIStatusError as error:
            self.last_error = f"API error {error.status_code}"
        except anthropic.APIConnectionError:
            self.last_error = "connection error"
        else:
            with self._lock:
                self.calls += 1
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens
            if response.stop_reason == "refusal":
                self.last_error = "model declined the request"
                return None
            if response.stop_reason == "max_tokens":
                self.last_error = "response truncated"
                return None
            return response.parsed_output
        return None
