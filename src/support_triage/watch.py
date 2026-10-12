"""The breach watch: the open queue, ranked by the clock.

Everything else in this system answers questions about one ticket,
or counts the queue's states (the overview tiles say how many runs
have breached; the queue table sorts by priority and chips the
breached ones). Neither answers the duty lead's question at 09:40:
*which three runs breach next, in what order, and whose are they?*
This module is that view — a pure read model over the stored runs,
recomputed on every read, that writes nothing.

**The watch reads the code clock, never the model's risk.** Two
things carry the letters SLA here (sla.py keeps them apart): the
SLA *clock* — arithmetic, the run's own priority-band target minus
its age — and ``sla_breach_risk``, a model judgement that feeds
the priority formula. The watch ranks by the clock alone. A row
also *shows* the model's risk value, labelled as the model's
judgement, so a lead can see the two disagree — a calm ticket the
model fears, an outage the model loves that is about to breach —
but the number never sorts, bands, or filters anything. The suite
pins a pair of runs the two orderings place in opposite order,
and the watch follows the clock.

**Bands** (the constants below, printed on every watch payload,
so the rule travels with its output). Bands are absolute
wall-clock distance to breach — the thing a lead scans — measured
against each run's OWN target, never a global clock:

- ``breached`` — the target has run out (seconds to breach < 0);
- ``due_within_1h`` — at most one hour left. One hour is the
  critical band's whole target: anything inside it is inside one
  critical ticket's entire lifetime;
- ``due_within_4h`` — at most four hours left, the high band's
  target;
- ``later`` — more than four hours left.

Bands are NOT the clock's states, and the watch says so on its
face: ``due_soon`` is a *fraction* of each band's own target (the
last 25%), so a low ticket (72h target) is due soon with 18 hours
left and still ``later`` here, while a critical ticket is
``due_within_1h`` from the minute it arrives. Both readings are
honest; they answer different questions, and each row carries
both — the clock's state and the watch band — so neither has to
impersonate the other.

**Ranking.** Breached first (most overdue first), then nearest
breach — one sort on seconds-to-breach, ascending, does both.
Ties (same received instant, same band target) break by the
composition priority score, highest first — the queue's own
ordering, reused — then by ticket id, so the order is total and
deterministic.

**Population.** Exactly the runs with a running clock: status
``awaiting_agent`` (the store's ``OPEN_STATUSES`` — the same
population the queue's open view shows) and a parseable
``received_at``. Open runs with no clock are counted by name
(``unclocked_open``) and never ranked on a guessed clock; decided
runs have no clock and are not the watch's business.

**Assignees and suggestions.** Each row carries the run's current
assignee by the one reading rule
(:func:`support_triage.assignment.current_assignee` — the watch's
reassignment trail first, the gate's ``[assigned to …]`` note
second) and the assignment module's top suggestion for the run,
advisory as always. The lead's one act from this view is the
reassignment (service.reassign_run): a named lead, a reason, and
a new assignee, appended to the run's assignment trail. It moves
the run between people and nothing else — the clock, the
composition, and the status are untouched, and nothing anywhere
reassigns automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import assignment
from .schemas import TicketResult
from .sla import format_timestamp, sla_clock

# The band thresholds, in seconds to breach. Authored here, in
# one place, and printed on every watch payload. 1h is the
# critical band's whole SLA target; 4h is the high band's (the
# targets themselves live in composition.SLA_TARGET_HOURS — the
# watch measures against each run's own, and borrows only these
# two landmarks for its absolute bands).
DUE_WITHIN_1H_SECONDS = 3600
DUE_WITHIN_4H_SECONDS = 4 * 3600

WATCH_BANDS = ("breached", "due_within_1h", "due_within_4h", "later")

CLOCK_NOTE = (
    "ranked by the code SLA clock (each run's own priority-band "
    "target minus its age) — the model's sla_breach_risk judgement "
    "is shown per row but never sorts, bands, or filters the watch"
)


def band_for_seconds(seconds_to_breach: float) -> str:
    """The watch band for a signed distance to breach."""
    if seconds_to_breach < 0:
        return "breached"
    if seconds_to_breach <= DUE_WITHIN_1H_SECONDS:
        return "due_within_1h"
    if seconds_to_breach <= DUE_WITHIN_4H_SECONDS:
        return "due_within_4h"
    return "later"


def band_constants() -> dict:
    """The authored band rule, as data — the API and the console
    print it beside the rows it produced."""
    return {
        "breached": "seconds to breach < 0 (the run's own target has run out)",
        "due_within_1h_seconds": DUE_WITHIN_1H_SECONDS,
        "due_within_4h_seconds": DUE_WITHIN_4H_SECONDS,
        "later": "seconds to breach above the 4h threshold",
    }


def _model_risk(record: TicketResult) -> float | None:
    answer = record.answer("sla_breach_risk")
    if answer is None or answer.value is None:
        return None
    return float(answer.value)


def _row(record: TicketResult, clock: dict, records, roster) -> dict:
    assignee = assignment.current_assignee(record)
    suggestion = assignment.suggest_assignment(record, records, roster)
    top = suggestion.get("suggestion")
    return {
        "ticket_id": record.ticket_id,
        "subject": record.subject,
        "sender_name": record.sender_name,
        "queue": record.composition.queue,
        "category": record.category,
        "product_area": record.product_area,
        "priority_band": record.composition.priority_band,
        "priority_score": record.composition.priority_score,
        "received_at": record.received_at,
        "target_hours": clock["target_hours"],
        "due_at": clock["due_at"],
        "seconds_to_breach": clock["seconds_to_breach"],
        "sla_state": clock["state"],
        "watch_band": band_for_seconds(clock["seconds_to_breach"]),
        # The model's judgement, carried for the lead's eye only.
        # It is read from the stored answers and plays no part in
        # the band above or the order of the rows.
        "sla_breach_risk": _model_risk(record),
        "assignee": assignee["agent"] if assignee else "",
        "assignee_source": assignee["source"] if assignee else "",
        "assigned_by": assignee["by"] if assignee else "",
        "suggested_agent": top["agent_name"] if top else "",
        "suggested_agent_id": top["agent_id"] if top else "",
        "suggestion_note": "" if top else suggestion.get("note", ""),
    }


def build_watch(
    records: list[TicketResult],
    roster: list[dict],
    now: datetime | None = None,
) -> dict:
    """The ranked breach watch over the stored runs (pure).

    Every function here takes ``now`` explicitly, the sla.py
    idiom, so tests and the golden pack inject the clock instead
    of racing the wall.
    """
    now = now or datetime.now(timezone.utc)
    rows: list[dict] = []
    unclocked_open = 0
    for record in records:
        if record.status != "awaiting_agent":
            continue
        clock = sla_clock(record, now)
        if clock is None:
            # Open, but no clock to rank by — counted, never
            # placed on a guessed one.
            unclocked_open += 1
            continue
        rows.append(_row(record, clock, records, roster))
    rows.sort(
        key=lambda row: (
            row["seconds_to_breach"],
            -row["priority_score"],
            row["ticket_id"],
        )
    )
    counts = {band: 0 for band in WATCH_BANDS}
    for row in rows:
        counts[row["watch_band"]] += 1
    return {
        "as_of": format_timestamp(now),
        "clock": "code_sla_clock",
        "clock_note": CLOCK_NOTE,
        "bands": band_constants(),
        "counts": counts,
        "open_runs": len(rows) + unclocked_open,
        "unclocked_open": unclocked_open,
        "rows": rows,
    }


def watch_block_for(record: TicketResult, now: datetime | None = None) -> dict | None:
    """One run's watch band at a moment — the audit export's block.

    The export is a document about one run, so the block carries
    the band and the clock figures it came from, labelled with
    the moment they were computed (the export's own time). The
    ranking is a property of the whole queue and is not restated
    here. ``None`` when the run has no running clock — the export
    is then exactly what it was before the watch existed.
    """
    now = now or datetime.now(timezone.utc)
    clock = sla_clock(record, now)
    if clock is None:
        return None
    return {
        "band": band_for_seconds(clock["seconds_to_breach"]),
        "sla_state": clock["state"],
        "seconds_to_breach": clock["seconds_to_breach"],
        "target_hours": clock["target_hours"],
        "due_at": clock["due_at"],
        "computed_at": format_timestamp(now),
        "note": (
            "the run's breach-watch band, computed at export time "
            "from the code SLA clock — a point-in-time reading, "
            "not part of the stored record"
        ),
    }
