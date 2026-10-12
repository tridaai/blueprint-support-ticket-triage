"""SLA attainment: the period report the desk reports upward.

The breach watch (watch.py) works the live queue: what breaches
next, and whose is it. This module answers the other question,
the one asked at the end of the month with a slide deck open:
*last month, what share of the tickets we decided met their SLA —
per queue, per priority band — and how long did decisions take?*

**Population: decisions, not arrivals.** A run belongs to the
period its decision landed in (``decided_at`` in [start, end)),
whenever the ticket arrived. Attainment measures the desk's
completed work in the window; keying on arrival would grade this
month's team on last month's queue and leave a ticket decided
late in no period at all. The bands and targets are the SLA
clock's own (composition.SLA_TARGET_HOURS), and a decision's
outcome is the clock's arithmetic evaluated at the decision
moment — :func:`support_triage.sla.sla_reading_at`, the one
computation behind the live clock and the watch — never a second
formula: a decision is **breached** when the clock would have
read breached at the instant it landed, **met** otherwise.

**Honest denominators.** Only judged decisions divide: a decided
run whose ``received_at`` is missing or unparseable had no clock
start on record, so its outcome cannot be judged — it is counted
by name as UNJUDGEABLE in its cell, never silently met and never
dropped, and it stays out of the percentage. A queue × band cell
with no judged decisions reports no percentage at all — an empty
cell is empty, never 100%. Runs still open are not in the decided
population at all; they appear as their own snapshot,
**open at period end**, so a hard month cannot be laundered by
leaving tickets undecided.

**The data boundary, stated on the face.** The decision moment
was not always recorded: decisions predate the ``decided_at``
stamp (see schemas.TicketResult), and a record from that era
keeps every fact of its decision except when it happened. Those
runs belong to no period — the report counts them on its face
(``boundary.decided_without_stamp``) so the size of the blind
spot is itself a reported number, and nothing back-fills a
plausible date for them.

**Open at period end** is a snapshot, reconstructed from the
records as they stand: a run counts when it arrived before the
period ended (parseable ``received_at`` before ``end``) and had
not been decided before the period ended — still
``awaiting_agent`` today, or decided at/after ``end``. Its state
is the clock's reading at ``end`` itself. Read at the period end
(the live use), the snapshot's population is exactly the breach
watch's at that instant — same runs, same bands — and the suite
pins that agreement; the two views share the clock and neither
forks its counting. One definitional seam: the watch is a live
view and ranks whatever waits now, while the snapshot counts
only what had *arrived* by the end — a run received after ``end``
was not open at period end and is not in the snapshot. Open runs
with no parseable arrival are counted as
``unclocked_open_at_end``, the watch's own treatment.

A pure read model: recomputed from the stored runs on every
read, writing nothing.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from .composition import SLA_TARGET_HOURS
from .schemas import TicketResult
from .sla import format_timestamp, parse_timestamp, sla_reading_at

DECIDED_STATUSES = ("triaged", "rejected")

BANDS = tuple(SLA_TARGET_HOURS)

PERIOD_RULE = "[start, end) — start inclusive, end exclusive"

POPULATION_RULE = (
    "population: runs whose decision landed in the period "
    "(decided_at in [start, end)), whenever they arrived — "
    "attainment measures the desk's completed work in the "
    "window. Still-open runs are never counted as met or "
    "breached; they appear as the open-at-end snapshot."
)

UNJUDGEABLE_REASON = (
    "no parseable received_at on record — the SLA clock never "
    "started, so the decision's outcome cannot be judged"
)


class _Cell:
    """Counts and decision times for one slice of the population."""

    def __init__(self) -> None:
        self.judged = 0
        self.met = 0
        self.breached = 0
        self.unjudgeable = 0
        self.elapsed: list[float] = []

    def add(self, other: "_Cell") -> None:
        self.judged += other.judged
        self.met += other.met
        self.breached += other.breached
        self.unjudgeable += other.unjudgeable
        self.elapsed.extend(other.elapsed)

    def as_dict(self) -> dict:
        out = {
            "decided": self.judged + self.unjudgeable,
            "judged": self.judged,
            "met": self.met,
            "breached": self.breached,
            "unjudgeable": self.unjudgeable,
            "attainment_pct": None,
            "median_seconds": None,
            "worst_seconds": None,
        }
        if self.judged:
            out["attainment_pct"] = round(100.0 * self.met / self.judged, 1)
            out["median_seconds"] = round(statistics.median(self.elapsed), 1)
            out["worst_seconds"] = round(max(self.elapsed), 1)
        return out


def _open_row(open_rows: dict, queue: str, band: str) -> dict:
    return open_rows.setdefault(
        (queue, band),
        {"open": 0, "states": {"ok": 0, "due_soon": 0, "breached": 0}, "ticket_ids": []},
    )


def build_attainment(
    records: list[TicketResult], start: datetime, end: datetime
) -> dict:
    """The attainment report for [start, end) over the stored runs."""
    cells: dict[tuple[str, str], _Cell] = {}
    unjudgeable: list[dict] = []
    decided_without_stamp = 0
    open_rows: dict[tuple[str, str], dict] = {}
    unclocked_open_at_end = 0

    for record in records:
        queue = record.composition.queue
        band = record.composition.priority_band
        decided_at = parse_timestamp(record.decided_at) if record.decided_at else None
        if record.status in DECIDED_STATUSES:
            if decided_at is None:
                # Decided, but its moment was never recorded (or
                # cannot be read): it belongs to no period. Counted
                # on the face, placed nowhere.
                decided_without_stamp += 1
            elif start <= decided_at < end:
                cell = cells.setdefault((queue, band), _Cell())
                reading = sla_reading_at(record, decided_at)
                if reading is None:
                    cell.unjudgeable += 1
                    unjudgeable.append(
                        {"ticket_id": record.ticket_id, "reason": UNJUDGEABLE_REASON}
                    )
                else:
                    cell.judged += 1
                    if reading["state"] == "breached":
                        cell.breached += 1
                    else:
                        cell.met += 1
                    cell.elapsed.append(reading["age_seconds"])
            if decided_at is not None and decided_at >= end:
                # Not yet decided when the period ended: the run
                # stands in the open-at-end snapshot.
                received = parse_timestamp(record.received_at)
                if received is not None and received < end:
                    reading = sla_reading_at(record, end)
                    row = _open_row(open_rows, queue, band)
                    row["open"] += 1
                    row["states"][reading["state"]] += 1
                    row["ticket_ids"].append(record.ticket_id)
        elif record.status == "awaiting_agent":
            received = parse_timestamp(record.received_at)
            if received is None:
                unclocked_open_at_end += 1
            elif received < end:
                reading = sla_reading_at(record, end)
                row = _open_row(open_rows, queue, band)
                row["open"] += 1
                row["states"][reading["state"]] += 1
                row["ticket_ids"].append(record.ticket_id)

    queues = sorted({q for q, _ in cells} | {q for q, _ in open_rows})
    queue_blocks = []
    grand = _Cell()
    for queue in queues:
        band_cells: dict[str, dict] = {}
        queue_total = _Cell()
        for band in BANDS:
            cell = cells.get((queue, band), _Cell())
            band_cells[band] = cell.as_dict()
            queue_total.add(cell)
        queue_blocks.append(
            {"queue": queue, "bands": band_cells, "totals": queue_total.as_dict()}
        )
        grand.add(queue_total)

    open_block_rows = []
    for queue in queues:
        for band in BANDS:
            row = open_rows.get((queue, band))
            if row is None:
                continue
            open_block_rows.append(
                {
                    "queue": queue,
                    "band": band,
                    "open": row["open"],
                    "states": row["states"],
                    "ticket_ids": sorted(row["ticket_ids"]),
                }
            )

    return {
        "period": {
            "start": format_timestamp(start),
            "end": format_timestamp(end),
            "rule": PERIOD_RULE,
        },
        "population_rule": POPULATION_RULE,
        "targets_hours": dict(SLA_TARGET_HOURS),
        "queues": queue_blocks,
        "totals": grand.as_dict(),
        "boundary": {
            "decided_without_stamp": decided_without_stamp,
            "note": (
                "decided runs whose moment was never recorded (or "
                "cannot be read) belong to no period: they appear "
                "in no cell, and this count is the size of that "
                "blind spot. Nothing back-fills a date for them."
            ),
        },
        "unjudgeable": unjudgeable,
        "open_at_end": {
            "rows": open_block_rows,
            "total": sum(row["open"] for row in open_block_rows),
            "unclocked_open_at_end": unclocked_open_at_end,
            "note": (
                "runs that arrived before the period ended and had "
                "not been decided before it ended, with the clock's "
                "state at the period end — reported, never counted "
                "as met or breached"
            ),
        },
    }
