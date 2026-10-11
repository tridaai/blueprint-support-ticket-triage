"""The agent's workbench: what triage hands over, the record keeps helping.

The pipeline ends at ``AWAITING_AGENT`` — and until this round the
record mostly stopped helping there. The similar-tickets list named
neighbours and whether they had outcomes; nobody joined the two
into what the agent actually asks: *how did tickets like this one
end, and how long did they take?* The workbench is that join, plus
the two things an agent does with a record open: work the
category's playbook (playbooks.py), and draft their internal
working note.

Three rules hold across all of it:

- **Knowledge is computed, never quoted from thin air.** The
  aggregate reads only similar runs WITH recorded outcomes.
  Similar runs without one stay in the similar list but never
  enter the figures, and the accounting says how many were left
  out and why. Below :data:`KNOWLEDGE_MIN_SAMPLES` resolved
  neighbours the rate and median are withheld (``None``) with a
  note saying so — the small-sample honesty the drift watch uses,
  applied to advice an agent might act on.
- **Time-to-resolution is arithmetic over recorded stamps.** An
  outcome carries no duration field; what is recorded is the
  ticket's ``received_at`` and the outcome's ``recorded_at``
  (stamped when the agent closes the ticket out). The elapsed
  intake-to-outcome hours are computed in code from those two
  stamps and named exactly that — never dressed up as a measured
  handling time the records don't contain.
- **The note draft is a different surface from the customer draft.**
  The template first-response (templates.py, ``DraftInfo``) is a
  customer-facing draft for one safe slice. The working note is
  internal, composed deterministically from the record (the
  composition's own routing reason verbatim, the raised verify
  answers as what the customer is blocked on, duplicate and thread
  context, the SLA clock), labelled INTERNAL, and saved back by the
  agent under their name. Neither surface can reach the other, and
  nothing here is ever sent anywhere.

The mutations live on the service; everything in this module is a
pure read or composition over records.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from . import corrections
from .schemas import TicketResult
from .sla import parse_timestamp

# Below this many outcome-bearing neighbours, the knowledge surface
# reports the count and refuses the rate: two resolved neighbours
# are an anecdote, not a resolution rate.
KNOWLEDGE_MIN_SAMPLES = 3

# The verify threshold the eval pack uses to call an answer raised.
FLAG_THRESHOLD = 0.5

# What a raised verify answer means for the customer, in the note's
# words — composed by code from the answer key, never generated.
_BLOCKED_ON = {
    "needs_engineering": "waiting on engineering input",
    "mentions_legal": "flagged for legal/compliance review",
    "is_outage_report": "reported against an outage — outage context applies",
    "contains_pii": "ticket contains PII — handle per the privacy policy",
    "is_abusive": "abusive language flagged — proceed per the conduct policy",
}


class WorkbenchError(ValueError):
    """A workbench action that cannot stand (the CorrectionError idiom).

    ``http_status`` is the API's mapping: 409 for a state conflict
    (already checked, already marked), 422 for an action invalid on
    its face (unknown step, missing actor, empty note).
    """

    http_status = 422


class WorkbenchConflict(WorkbenchError):
    http_status = 409


def hours_from_intake(record: TicketResult) -> float | None:
    """Elapsed hours from the ticket's intake stamp to its outcome
    stamp, or ``None`` when either stamp is missing/unparseable —
    the honest absence, never a guess."""
    if record.outcome is None:
        return None
    received = parse_timestamp(record.received_at)
    recorded = parse_timestamp(record.outcome.recorded_at)
    if received is None or recorded is None:
        return None
    return round((recorded - received).total_seconds() / 3600.0, 2)


def knowledge_entry(record: TicketResult, similarity: float) -> dict:
    """One similar run WITH its outcome, as the workbench shows it."""
    outcome = record.outcome
    assert outcome is not None  # callers filter first
    return {
        "ticket_id": record.ticket_id,
        "similarity": round(float(similarity), 4),
        "subject": record.subject,
        "resolved": outcome.resolved,
        "final_category": outcome.final_category,
        "final_queue": outcome.final_queue,
        "resolution_summary": outcome.resolution_summary,
        "recorded_by": outcome.recorded_by,
        "recorded_at": outcome.recorded_at,
        "hours_from_intake": hours_from_intake(record),
    }


def build_knowledge(similar_items: list[dict], records_by_id: dict[str, TicketResult]) -> dict:
    """Join similar runs to their outcomes and aggregate in code.

    ``similar_items`` is the similar-tickets list (the existing
    machinery, unchanged); only neighbours whose record carries an
    outcome enter the knowledge set. The aggregate's accounting
    names the ones left out — a neighbour with no outcome is
    evidence about similarity, not about resolution.
    """
    known: list[dict] = []
    without_outcome = 0
    for item in similar_items:
        record = records_by_id.get(item["ticket_id"])
        if record is None:
            continue
        if record.outcome is None:
            without_outcome += 1
            continue
        known.append(knowledge_entry(record, item["similarity"]))
    n = len(known)
    resolved = sum(1 for entry in known if entry["resolved"])
    hours = [entry["hours_from_intake"] for entry in known if entry["hours_from_intake"] is not None]
    enough = n >= KNOWLEDGE_MIN_SAMPLES
    if n == 0:
        note = "no similar ticket has a recorded outcome yet — nothing to learn from so far"
    elif not enough:
        note = (
            f"only {n} similar ticket(s) with recorded outcomes — too few to quote "
            f"a resolution rate or a median time (needs {KNOWLEDGE_MIN_SAMPLES})"
        )
    else:
        note = f"figures over the {n} similar ticket(s) with recorded outcomes"
    if without_outcome:
        note += (
            f"; {without_outcome} similar ticket(s) have no recorded outcome — "
            "listed as similar, excluded from these figures"
        )
    return {
        "entries": known,
        "aggregate": {
            "similar_total": len(known) + without_outcome,
            "with_outcome": n,
            "without_outcome": without_outcome,
            "resolved_count": resolved,
            "resolved_rate": round(resolved / n, 4) if enough else None,
            "median_hours_from_intake": (
                round(statistics.median(hours), 2) if enough and hours else None
            ),
            "note": note,
        },
    }


def build_working_note(
    record: TicketResult,
    *,
    grouping: dict,
    chain_size: int,
    sla: dict | None,
    now: datetime | None = None,
) -> str:
    """The deterministic internal working-note draft for one run.

    Composed from the record alone: where the ticket sits and why
    (the composition's own routing reason, verbatim), what the
    customer is blocked on (the raised verify answers, in code's
    words), duplicate and thread context, and the SLA clock state.
    Deterministic by construction — the same record composes the
    same draft, so what an agent edits is stable under them.
    """
    composition = record.composition
    lines = [
        "INTERNAL WORKING NOTE — DRAFT",
        "Composed by code from the decision record. Edit freely; saving "
        "keeps your version under your name. Internal only — never sent "
        "to the customer, and not the customer template draft.",
        "",
        f"Ticket: {record.ticket_id} — {record.subject or '(no subject)'}",
        f"Category: {record.category} · Queue: {composition.queue} · "
        f"Priority: {composition.priority_band} ({composition.priority_score:.1f})",
        f"Routing reason: {composition.routing_reason}",
    ]
    if composition.override:
        lines.append(f"Routing override: {composition.override}")
    if composition.reopen_floor_applied:
        lines.append(
            f"Reopen floor applied: band raised from {composition.band_before_floor} "
            f"to {composition.priority_band}"
        )
    raised = [
        answer
        for answer in record.answers
        if answer.kind == "verify"
        and isinstance(answer.value, (int, float))
        and float(answer.value) >= FLAG_THRESHOLD
    ]
    lines.append("")
    if raised:
        lines.append("What the customer is blocked on / what needs care:")
        for answer in raised:
            meaning = _BLOCKED_ON.get(answer.key, answer.key)
            lines.append(f"- {meaning} ({answer.key} raised, p={float(answer.value):.2f})")
    else:
        lines.append("No verify flags raised — nothing structural is blocking the customer.")
    if record.guardrails.human_review_reasons:
        lines.append("Human review was required:")
        for reason in record.guardrails.human_review_reasons:
            lines.append(f"- {reason}")
    lines.append("")
    if record.duplicate_of:
        mark = corrections.active_duplicate_mark(record)
        who = f" by {mark.marked_by}" if mark is not None and mark.marked_by else ""
        lines.append(
            f"Duplicate: confirmed duplicate of {record.duplicate_of}{who} — "
            "the surviving run carries the conversation."
        )
    elif record.duplicate.is_duplicate:
        lines.append(
            f"Duplicate: machine-flagged probable re-send of "
            f"{record.duplicate.similar_ticket_id} "
            f"(similarity {record.duplicate.similarity:.3f}) — unconfirmed."
        )
    else:
        lines.append("Duplicate: none flagged.")
    thread_id = grouping.get("thread_id", "")
    if thread_id:
        thread_line = f"Thread: {thread_id} ({chain_size} run(s) in the chain/thread view)"
        if grouping.get("corrected"):
            thread_line += f" — {grouping.get('lineage', 'grouping corrected by an operator')}"
        lines.append(thread_line)
    if record.reopen is not None:
        lines.append(
            f"Reopen: #{record.reopen.reopen_count} of {record.reopen.root_ticket_id} "
            f"(limit {record.reopen.reopen_limit})"
            + (" — limit hit, human review forced." if record.reopen.limit_hit else ".")
        )
    lines.append("")
    if sla is not None:
        lines.append(
            f"SLA clock: {sla['state']} — target {sla['target_hours']:g}h for the "
            f"{composition.priority_band} band, due {sla['due_at']}."
        )
    elif record.status != "awaiting_agent":
        lines.append("SLA clock: not running — the run has left the gate.")
    else:
        lines.append("SLA clock: none — the record carries no parseable intake stamp.")
    return "\n".join(lines) + "\n"
