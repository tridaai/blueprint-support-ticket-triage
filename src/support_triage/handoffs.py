"""The shift handoff: one signed page the next lead accepts.

Support runs around the clock, and the desk's state at any moment
already has honest views — the breach watch (watch.py) ranks the
open queue by the clock, incident clustering (incidents.py) names
the events and the proposals waiting on a lead, the queue view
carries every open run's SLA state, QA review (reviews.py) knows
what grading is still owed. But at shift change those four views
are four places, and the handover between two leads happens in
whichever of them someone remembers to open. Knowledge that lives
in four views and one conversation is folklore with good tooling.

This module is the page the conversation is about. A handoff is
**composed in code from the desk's own views and nothing else**:

- the breach watch section is the watch's own payload — its band
  counts verbatim and its top rows under an authored cap
  (:data:`TOP_WATCH_ROWS`, printed on the payload) — so the page
  can never rank differently from the watch it cites;
- the incidents section reads the stored open incidents (member
  counts from the incident views, which read members live) and
  the live proposal scan (how many proposals wait, the first one
  named);
- the queues section groups the queue view's own rows
  (``store.queue_summary``) by queue and counts the SLA clock
  states those rows already carry — the clock is computed once,
  in sla.py, and read here, never recomputed;
- the QA section reads the review records: per sample, drawn runs
  minus verdicted runs is the grading still owed — the backlog —
  summed over samples.

No section introduces arithmetic of its own beyond counting what
a source view produced; the suite pins the composed page equal
to the source views at the same instant, field by field. The one
human-authored part is the outgoing lead's note: prose the
machine does not write, summarise, or grade.

**The handshake is the record.** A handoff is created by a named
outgoing lead, addressed to a named incoming lead — two different
people, or there is no handover to sign. It stands
``outstanding`` until the ADDRESSED lead accepts it: acceptance
by its author is refused by name (a page you wrote is not a page
you received), and so is acceptance by anyone else (the page is
addressed, not public). The acceptor may accept with exceptions —
free text naming what they dispute — which rides the record
beside the figures and changes none of them: a dispute about the
watch's top row is a sentence on the page, not an edit to the
watch. An acceptance is testimony, not a draft: accepting twice
is a conflict. A newer handoff supersedes an unaccepted older
one — the older record stays, its status lapses to
``superseded``, and it can no longer be accepted; at most one
handoff is outstanding at any moment.

Handoffs are desk-level records. They live in their own store
table, never on a run's record: no run changes, in any field,
because a handoff was created, accepted, or superseded (the
suite pins run records byte-identical across the whole
lifecycle). The run audit export does not carry them — an export
is a document about one run, and a shift page is not part of a
run's story. And the handoff publishes nothing: no CLI, no
events, no webhooks — the bus subscribers contracted for run
events (the watch's precedent), and a lead opening the console
tab is the reader this page is written for.
"""

from __future__ import annotations

from .schemas import ShiftHandoff

# How many of the watch's ranked rows the page carries. A handoff
# is a page a lead reads at a boundary, not the watch itself: the
# counts give the whole queue, the top rows give the names that
# cannot wait. Authored here and printed on every payload.
TOP_WATCH_ROWS = 5


class HandoffError(ValueError):
    """A handoff action that cannot stand.

    ``http_status`` is the API's mapping: 409 for a state conflict
    (accepting a handoff already accepted, or one superseded by a
    newer page), 422 for an action invalid on its face (a missing
    or wrong actor — the author accepting their own page, a
    bystander accepting a page addressed to someone else).
    """

    http_status = 422


class HandoffConflict(HandoffError):
    http_status = 409


def handoff_constants() -> dict:
    """The authored composition rule, as data — the payload and
    the console print it beside the page it produced."""
    return {"top_watch_rows": TOP_WATCH_ROWS}


def _watch_section(watch: dict) -> dict:
    rows = watch["rows"]
    return {
        "source": (
            "the breach watch (watch.py) — open runs ranked by the "
            "code SLA clock; counts and rows are the watch's own"
        ),
        "counts": dict(watch["counts"]),
        "open_runs": watch["open_runs"],
        "unclocked_open": watch["unclocked_open"],
        "rows_total": len(rows),
        "top_rows": [
            {
                "ticket_id": row["ticket_id"],
                "subject": row["subject"],
                "queue": row["queue"],
                "priority_band": row["priority_band"],
                "watch_band": row["watch_band"],
                "seconds_to_breach": row["seconds_to_breach"],
                "due_at": row["due_at"],
                "assignee": row["assignee"],
            }
            for row in rows[:TOP_WATCH_ROWS]
        ],
    }


def _incidents_section(cluster_payload: dict, incident_views: list[dict]) -> dict:
    proposals = cluster_payload["proposals"]
    top = proposals[0] if proposals else None
    return {
        "source": (
            "incident clustering (incidents.py) — confirmed incidents "
            "and the live proposal scan"
        ),
        "open": [
            {
                "incident_id": view["incident"]["incident_id"],
                "title": view["incident"]["title"],
                "member_count": view["member_count"],
                "open_members": view["open_members"],
            }
            for view in incident_views
        ],
        "open_count": len(incident_views),
        "proposals_pending": len(proposals),
        "top_proposal": (
            {
                "proposal_id": top["proposal_id"],
                "suggested_title": top["suggested_title"],
                "member_count": top["signals"]["member_count"],
                "product_area": top["product_area"],
                "category": top["category"],
            }
            if top is not None
            else None
        ),
    }


def _queues_section(queue_rows: list[dict]) -> dict:
    """Per-queue pressure, grouped from the queue view's own rows.

    ``queue_rows`` is ``store.queue_summary``'s output: one row
    per open run, each carrying the SLA clock state sla.py
    computed for it. Grouping and counting those states is the
    whole of the arithmetic here — no clock is recomputed, so
    this section cannot disagree with the queue it summarises.
    A row with no running clock counts under ``unclocked``.
    """
    by_queue: dict[str, dict] = {}
    for row in queue_rows:
        entry = by_queue.setdefault(
            row["queue"],
            {
                "queue": row["queue"],
                "open": 0,
                "breached": 0,
                "due_soon": 0,
                "ok": 0,
                "unclocked": 0,
            },
        )
        entry["open"] += 1
        state = row.get("sla_state")
        if state in ("breached", "due_soon", "ok"):
            entry[state] += 1
        else:
            entry["unclocked"] += 1
    rows = sorted(
        by_queue.values(),
        key=lambda entry: (-entry["breached"], -entry["due_soon"], entry["queue"]),
    )
    return {
        "source": (
            "the queue view (store.queue_summary) — every open run "
            "with the SLA clock state the queue itself shows"
        ),
        "rows": rows,
        "open_total": sum(entry["open"] for entry in rows),
    }


def _qa_section(samples: list, verdicts: list) -> dict:
    """The QA review backlog: grading drawn and not yet graded.

    Per sample, the backlog is the sample's own accounting —
    runs it drew minus runs carrying a verdict in it (the same
    ``verdicted_runs`` the sample's own view reports). A run
    drawn into two samples is owed twice: it is two gradings.
    """
    per_sample = []
    for sample in samples:
        verdicted = len(
            {
                verdict.ticket_id
                for verdict in verdicts
                if verdict.sample_id == sample.sample_id
            }
        )
        drawn = len(sample.run_ids)
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "drawn": drawn,
                "verdicted": verdicted,
                "backlog": drawn - verdicted,
            }
        )
    per_sample.sort(key=lambda entry: entry["sample_id"])
    return {
        "source": (
            "QA review (reviews.py) — samples drawn and the verdicts "
            "recorded against them"
        ),
        "samples": len(per_sample),
        "drawn_runs": sum(entry["drawn"] for entry in per_sample),
        "verdicted_runs": sum(entry["verdicted"] for entry in per_sample),
        "backlog": sum(entry["backlog"] for entry in per_sample),
        "per_sample": per_sample,
    }


def compose_sections(
    *,
    watch: dict,
    cluster_payload: dict,
    incident_views: list[dict],
    queue_rows: list[dict],
    samples: list,
    verdicts: list,
) -> dict:
    """The handoff's figures, composed from the source views.

    Pure: every input is a source view's own output (the watch
    payload, the proposal scan, the open incidents' views, the
    queue rows, the review records), and the output is those
    outputs selected and counted — the composition the module
    docstring promises, and the suite pins against the views
    recomputed independently at the same instant.
    """
    return {
        "constants": handoff_constants(),
        "watch": _watch_section(watch),
        "incidents": _incidents_section(cluster_payload, incident_views),
        "queues": _queues_section(queue_rows),
        "qa_review": _qa_section(samples, verdicts),
    }


def handoff_view(handoff: ShiftHandoff) -> dict:
    """One handoff as the surfaces serve it: the stored record,
    plus the one derived word a reader scans for — whether this
    page is still waiting on its handshake."""
    view = handoff.model_dump()
    view["outstanding"] = handoff.status == "outstanding"
    return view
