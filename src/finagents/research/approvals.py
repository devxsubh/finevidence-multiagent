"""Human approval gate for protected research actions.

The research loop never performs a protected action on its own. It files a request in
``runs/research/approvals.json`` and moves on. A person approves or rejects it with
``finagents research approve <id>`` (or from the frontend's Research Lab); an approved
hypothesis runs in the next research session.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from finagents.core.records import PROJECT_ROOT

APPROVALS = PROJECT_ROOT / "runs" / "research" / "approvals.json"


def _load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def _save(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def list_requests(path: Path = APPROVALS) -> list[dict[str, Any]]:
    return _load(path)


def request(action: str, subject: str, reason: str, path: Path = APPROVALS) -> dict[str, Any]:
    """File a request once; an existing pending or decided request for the same subject is reused."""
    items = _load(path)
    existing = next((i for i in items if i["action"] == action and i["subject"] == subject), None)
    if existing:
        return existing
    item = {
        "id": uuid.uuid4().hex[:8],
        "action": action,
        "subject": subject,
        "reason": reason,
        "status": "pending",
        "requested": datetime.now(UTC).isoformat(timespec="seconds"),
        "decided": None,
    }
    _save([*items, item], path)
    return item


def decide(
    request_id: str, decision: Literal["approved", "rejected"], path: Path = APPROVALS
) -> dict[str, Any]:
    items = _load(path)
    for item in items:
        if item["id"] == request_id:
            item["status"] = decision
            item["decided"] = datetime.now(UTC).isoformat(timespec="seconds")
            _save(items, path)
            return item
    raise KeyError(f"no approval request {request_id!r}")


def status(action: str, subject: str, path: Path = APPROVALS) -> str | None:
    item = next((i for i in _load(path) if i["action"] == action and i["subject"] == subject), None)
    return item["status"] if item else None
