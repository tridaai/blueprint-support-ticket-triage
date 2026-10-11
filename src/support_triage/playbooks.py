"""Category playbooks: the authored handling steps for a ticket's category.

A playbook is guidance for the human working the ticket — the
checklist a senior agent would tape to the monitor. It is bundled
data (``data/playbooks.json``), authored per category like the
roster is authored per agent, and loaded with the same discipline:
validated at load against the categories the decision port can
actually produce, so a typo fails loudly here instead of silently
leaving a category unplaybooked.

What a playbook is NOT: a workflow. Nothing executes its steps,
nothing blocks on them, and a fully unchecked playbook changes
nothing about a run — its state, its gate, its calibration. The
check-offs on a record (``playbook_checks`` / ``playbook_na`` on
``TicketResult``) are the agent's own bookkeeping, kept as an
append-only overlay in the corrections style: who ticked what,
when, and — when the set is marked not-applicable — why.

Categories with no authored playbook say so plainly. That honesty
is deliberate: an absent playbook is a gap in the authored data,
and dressing a generic list up as category guidance would hide it.
"""

from __future__ import annotations

import json
from importlib import resources

from .questions import CATEGORIES
from .schemas import TicketResult


def load_playbooks() -> dict[str, dict]:
    """The bundled playbooks, validated: ``{category: {steps: [...]}}``."""
    text = (
        resources.files("support_triage")
        .joinpath("data/playbooks.json")
        .read_text(encoding="utf-8")
    )
    return validate_playbooks(json.loads(text))


def validate_playbooks(raw) -> dict[str, dict]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("playbooks must be a non-empty object keyed by category")
    playbooks: dict[str, dict] = {}
    for category, entry in raw.items():
        if category not in CATEGORIES:
            raise ValueError(
                f"playbook authored for unknown category {category!r}; "
                f"the decision port produces {list(CATEGORIES)}"
            )
        if not isinstance(entry, dict) or not isinstance(entry.get("steps"), list):
            raise ValueError(f"playbook for {category!r} needs a steps list")
        steps: list[dict] = []
        seen_ids: set[str] = set()
        for step in entry["steps"]:
            if not isinstance(step, dict):
                raise ValueError(f"playbook step must be an object, got {step!r}")
            step_id = str(step.get("id", "")).strip()
            text = str(step.get("text", "")).strip()
            if not step_id or not text:
                raise ValueError(
                    f"playbook step in {category!r} needs an id and text: {step!r}"
                )
            if step_id in seen_ids:
                raise ValueError(f"duplicate playbook step id {step_id!r} in {category!r}")
            seen_ids.add(step_id)
            steps.append(
                {
                    "id": step_id,
                    "text": text,
                    "reference": str(step.get("reference", "")).strip(),
                }
            )
        if not steps:
            raise ValueError(f"playbook for {category!r} has no steps")
        playbooks[category] = {"steps": steps}
    return playbooks


def playbook_for(category: str, playbooks: dict[str, dict] | None = None) -> dict | None:
    """The authored playbook for a category, or ``None`` — plainly none."""
    books = playbooks if playbooks is not None else load_playbooks()
    return books.get(category)


def step_ids(playbook: dict) -> list[str]:
    return [step["id"] for step in playbook["steps"]]


def active_na_mark(record: TicketResult):
    """The live not-applicable mark on a run, if one stands (the
    last entry decides, as with the duplicate mark)."""
    if record.playbook_na and record.playbook_na[-1].active:
        return record.playbook_na[-1]
    return None


def playbook_view(record: TicketResult, playbooks: dict[str, dict] | None = None) -> dict:
    """The playbook as the workbench presents it: the authored steps
    for the run's category with each step's effective check state
    resolved from the record's overlay, and the not-applicable
    state. A pure read — the record is never touched here."""
    playbook = playbook_for(record.category, playbooks)
    if playbook is None:
        return {
            "category": record.category,
            "available": False,
            "steps": [],
            "not_applicable": None,
            "note": (
                f"no playbook authored for category '{record.category}' — "
                "the gap is in the authored data, not in this ticket"
            ),
        }
    last_action: dict[str, dict] = {}
    for check in record.playbook_checks:
        last_action[check.step_id] = {
            "checked": check.action == "checked",
            "by": check.by,
            "at": check.at,
        }
    steps = []
    for step in playbook["steps"]:
        state = last_action.get(step["id"], {"checked": False, "by": "", "at": ""})
        steps.append({**step, **state})
    na = active_na_mark(record)
    return {
        "category": record.category,
        "available": True,
        "steps": steps,
        "checked_count": sum(1 for step in steps if step["checked"]),
        "not_applicable": (
            {"by": na.by, "reason": na.reason, "at": na.at} if na is not None else None
        ),
        "note": "guidance for the agent — nothing executes these steps or blocks on them",
    }
