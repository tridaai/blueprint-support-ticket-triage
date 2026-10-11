"""The agent roster: who can take which queue, and what they're good at.

Assignment *suggestions* (assignment.py) score the agents on this
roster. The roster is bundled data — synthetic people, like the
sample tickets — loaded from ``data/agent_roster.json`` and
validated at load: a queue the router cannot produce, or an entry
missing its fields, fails loudly here rather than quietly warping
suggestions later. A deployment replaces the file with its real
roster; the shape is the contract.
"""

from __future__ import annotations

import json
from importlib import resources

from .composition import QUEUES


def load_roster() -> list[dict]:
    """The bundled agent roster, validated."""
    text = (
        resources.files("support_triage")
        .joinpath("data/agent_roster.json")
        .read_text(encoding="utf-8")
    )
    return validate_roster(json.loads(text))


def validate_roster(raw) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("agent roster must be a non-empty list")
    roster: list[dict] = []
    seen_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"roster entry must be an object, got {entry!r}")
        agent_id = str(entry.get("id", "")).strip()
        name = str(entry.get("name", "")).strip()
        queues = entry.get("queues")
        skills = entry.get("skills")
        if not agent_id or not name:
            raise ValueError(f"roster entry needs an id and a name: {entry!r}")
        if agent_id in seen_ids:
            raise ValueError(f"duplicate roster agent id {agent_id!r}")
        if not isinstance(queues, list) or not queues:
            raise ValueError(f"roster agent {agent_id!r} covers no queues")
        unknown = [q for q in queues if q not in QUEUES]
        if unknown:
            raise ValueError(
                f"roster agent {agent_id!r} covers unknown queues {unknown}; "
                f"the router produces {list(QUEUES)}"
            )
        if not isinstance(skills, list):
            raise ValueError(f"roster agent {agent_id!r} skills must be a list")
        seen_ids.add(agent_id)
        roster.append(
            {
                "id": agent_id,
                "name": name,
                "queues": [str(q) for q in queues],
                "skills": [str(s) for s in skills],
            }
        )
    return roster
