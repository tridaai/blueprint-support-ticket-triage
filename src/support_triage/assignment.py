"""Assignment suggestions: who should probably take this ticket?

Advisory, end to end. For a ticket waiting at the gate this module
ranks the agents on the roster (roster.py) who cover the ticket's
queue and prints its working — the suggestion is worth exactly as
much as the reasons under it. Nothing here assigns anything: there
is no write path in this module, and assignment in the system
remains a human's act — the gate's ``POST /tickets/{id}/assign``
note, or the breach watch's reasoned reassignment (watch.py).
:func:`current_assignee` below is the one reading rule both write
paths answer to.

The score is a weighted sum of three factors, each computed in code
from records the system already keeps:

- **track record** (0.50) — of the outcomes this agent recorded in
  the ticket's category (``recorded_by`` on the outcome, matched to
  the roster by id or name), the share they marked resolved. An
  agent with no judged outcomes in the category scores a neutral
  0.5 — no history is not a demerit, and the reason says so.
- **open load** (0.30) — ``1 / (1 + open)`` where *open* counts the
  tickets still ``awaiting_agent`` whose current assignee, by
  :func:`current_assignee`, is this agent. Fewer in hand, higher score.
- **skills** (0.20) — the share of the ticket's needs the agent's
  skill tags cover, where needs are the ticket's category, product
  area, and raised safety flags.

Ranking is deterministic: score, then lighter load, then agent id.
"""

from __future__ import annotations

import re

from .schemas import TicketResult

WEIGHTS = {"track_record": 0.50, "open_load": 0.30, "skills": 0.20}

_ASSIGNED_RE = re.compile(r"\[assigned to ([^\]]+)\]")


def _normalise(who: str) -> str:
    return " ".join(str(who).split()).strip().lower()


def _matches(who: str, agent: dict) -> bool:
    value = _normalise(who)
    return bool(value) and value in {_normalise(agent["id"]), _normalise(agent["name"])}


def _last_assignee(record: TicketResult) -> str:
    marks = _ASSIGNED_RE.findall(record.decision_reason or "")
    return marks[-1] if marks else ""


def current_assignee(record: TicketResult) -> dict | None:
    """Whose run this is, by the one reading rule every surface uses.

    Two write paths exist, and this resolver is where they are
    reconciled: the breach watch's reassignment appends a
    structured :class:`~support_triage.schemas.AssignmentEntry`
    (assignee, lead, reason, time) to ``record.assignments``; the
    older gate idiom appends a ``[assigned to …]`` mark to
    ``decision_reason``. The structured trail is the later idiom
    and wins when it has entries; otherwise the latest mark is
    the answer; otherwise the run is unassigned (``None``).

    ``source`` says which path answered — ``"reassignment"`` (the
    entry's lead and time ride along) or ``"note"`` (a gate mark:
    the mark itself records neither who wrote it nor when, and
    the resolver says so rather than borrowing ``decision_by``,
    which belongs to the decision, not the assignment).
    """
    if record.assignments:
        entry = record.assignments[-1]
        return {
            "agent": entry.assignee,
            "source": "reassignment",
            "by": entry.by,
            "at": entry.at,
            "reason": entry.reason,
        }
    marked = _last_assignee(record)
    if marked:
        return {"agent": marked, "source": "note", "by": "", "at": "", "reason": ""}
    return None


def open_load(records: list[TicketResult], agent: dict) -> int:
    """Tickets still at the gate whose current assignee — by the
    one reading rule (:func:`current_assignee`) — is this agent."""
    return sum(
        1
        for record in records
        if record.status == "awaiting_agent"
        and (current := current_assignee(record)) is not None
        and _matches(current["agent"], agent)
    )


def _track_record(records: list[TicketResult], agent: dict, category: str) -> tuple[int, int]:
    """(judged, resolved): outcomes this agent recorded in a category."""
    judged = resolved = 0
    for record in records:
        outcome = record.outcome
        if outcome is None or not outcome.final_category:
            continue
        if outcome.final_category != category:
            continue
        if not _matches(outcome.recorded_by, agent):
            continue
        judged += 1
        if outcome.resolved:
            resolved += 1
    return judged, resolved


def _needs(record: TicketResult) -> list[str]:
    needs: list[str] = []
    for value in (record.category, record.product_area):
        if value and value not in needs:
            needs.append(value)
    for flag, raised in (record.composition.flags or {}).items():
        if raised and flag not in needs:
            needs.append(flag)
    return needs


def _candidate(
    record: TicketResult, records: list[TicketResult], agent: dict, needs: list[str]
) -> dict:
    judged, resolved = _track_record(records, agent, record.category)
    if judged:
        track = resolved / judged
        track_reason = (
            f"resolved {resolved} of {judged} recorded "
            f"'{record.category}' outcomes ({round(100 * resolved / judged)}%)"
        )
    else:
        track = 0.5
        track_reason = (
            f"no recorded outcomes in '{record.category}' yet — neutral track record"
        )

    load_open = open_load(records, agent)
    load = 1.0 / (1 + load_open)

    matched = [need for need in needs if need in agent["skills"]]
    skills = len(matched) / len(needs) if needs else 1.0
    if matched:
        skills_reason = f"skills match: {', '.join(matched)}"
    else:
        skills_reason = "no skill overlap with this ticket's category, area or flags"

    score = round(
        WEIGHTS["track_record"] * track
        + WEIGHTS["open_load"] * load
        + WEIGHTS["skills"] * skills,
        4,
    )
    return {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "score": score,
        "factors": {
            "track_record": round(track, 4),
            "open_load": round(load, 4),
            "skills": round(skills, 4),
        },
        "open_tickets": load_open,
        "judged_outcomes": judged,
        "resolved_outcomes": resolved,
        "reasons": [
            f"covers queue {record.composition.queue}",
            track_reason,
            f"{load_open} open ticket(s) currently assigned",
            skills_reason,
        ],
    }


def suggest_assignment(
    record: TicketResult, records: list[TicketResult], roster: list[dict]
) -> dict:
    """The suggestion payload for one ticket (pure; no writes)."""
    queue = record.composition.queue
    base = {
        "ticket_id": record.ticket_id,
        "queue": queue,
        "category": record.category,
        "status": record.status,
        "weights": dict(WEIGHTS),
    }
    if record.status != "awaiting_agent":
        return {
            **base,
            "suggestion": None,
            "runners_up": [],
            "candidates": 0,
            "note": (
                f"ticket is {record.status}, not waiting at the gate — "
                "suggestions are made for tickets awaiting an agent"
            ),
        }
    eligible = [agent for agent in roster if queue in agent["queues"]]
    if not eligible:
        return {
            **base,
            "suggestion": None,
            "runners_up": [],
            "candidates": 0,
            "note": f"no roster agent covers queue '{queue}' — assign manually",
        }
    needs = _needs(record)
    ranked = [_candidate(record, records, agent, needs) for agent in eligible]
    ranked.sort(key=lambda c: (-c["score"], c["open_tickets"], c["agent_id"]))
    return {
        **base,
        "suggestion": ranked[0],
        "runners_up": ranked[1:3],
        "candidates": len(ranked),
        "note": "suggestion — a human assigns; nothing was assigned",
    }
