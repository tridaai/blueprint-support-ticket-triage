"""Operator corrections: the machine groups and guesses; humans overrule.

Two machine judgements shipped as labels, with no way for the agent
who actually knows to say so on the record:

- **thread grouping** (threads.py) — deterministic, and sometimes
  wrong in both directions: two different issues whose words overlap
  land in one thread; one conversation split across phrasings lands
  in two.
- **the duplicate flag** (``DuplicateInfo``) — an embedding
  similarity over a threshold. A guess, labelled as one.

This module is the correction layer over both. Its rules:

- **Overlay, never rewrite.** A correction appends to the record
  (``thread_corrections`` / ``duplicate_marks``); the computed
  ``thread_id`` and the intake flag stay exactly as the machine left
  them, so the machine's judgement and the human's overrule are both
  on the record, with the human's name and reason. Views resolve the
  **effective** thread through the last correction; everything that
  reads decisions (composition, guardrails, replay, calibration of
  the run itself) is untouched — a correction changes where a run
  *sits*, never what was *decided* on it.
- **Chains move whole.** A reopen chain is one conversation by
  construction, so attach and detach apply to every run in the
  chain. A correction can never split a chain.
- **Same customer or nothing.** Grouping never crosses customers,
  and neither do corrections: attach and merge require the chains
  involved to share one customer key (threads.customer_key of the
  stored intake), and a confirmed duplicate must be the same
  customer asking again — a mark across customers is refused.
- **A confirmed duplicate leaves the samples, not the counts.** An
  active mark excludes the run from calibration sampling and the
  drift window (livecal.py / drift.py read the mark through here):
  it is not a second opinion about the backend. It still counts in
  volume stats — it was a ticket, an agent still handled it.

The service (service.py) owns the mutations and the store writes;
everything in this module is a pure read over records.
"""

from __future__ import annotations

from . import threads
from .schemas import DuplicateMark, TicketResult


class CorrectionError(ValueError):
    """A correction the operator asked for that cannot stand.

    ``http_status`` is the API's mapping: 409 for a state conflict
    (already marked, already its own thread), 422 for a correction
    that is invalid on its face (cross-customer, self-reference,
    missing actor or reason).
    """

    http_status = 422


class CorrectionConflict(CorrectionError):
    http_status = 409


def computed_thread_id(record: TicketResult) -> str:
    """The thread intake computed for this run — never corrected away."""
    return threads.effective_thread_id(record.thread_id, record.ticket_id)


def effective_thread_id(record: TicketResult) -> str:
    """The thread views show: the last correction's target when the
    operator has overruled the grouping, the computed id otherwise."""
    if record.thread_corrections:
        return record.thread_corrections[-1].to_thread_id
    return computed_thread_id(record)


def grouping_view(record: TicketResult) -> dict:
    """The grouping as views present it: effective id, computed id,
    and the correction lineage when an operator has overruled it."""
    corrections = [c.model_dump() for c in record.thread_corrections]
    view = {
        "thread_id": effective_thread_id(record),
        "computed_thread_id": computed_thread_id(record),
        "corrected": bool(corrections),
        "corrections": corrections,
        "lineage": "",
    }
    if corrections:
        last = corrections[-1]
        view["lineage"] = (
            f"grouping: corrected by {last['by'] or 'operator'} — {last['reason']}"
            if last["reason"]
            else f"grouping: corrected by {last['by'] or 'operator'}"
        )
    return view


def chain_root_id(record: TicketResult) -> str:
    """The root of the reopen chain this run belongs to (itself,
    for a run that is not a reopen)."""
    if record.reopen is not None and record.reopen.root_ticket_id:
        return record.reopen.root_ticket_id
    return record.ticket_id


def chain_of(records: list[TicketResult], record: TicketResult) -> list[TicketResult]:
    """Every stored run in the same reopen chain as ``record``."""
    root = chain_root_id(record)
    return [r for r in records if chain_root_id(r) == root]


def active_duplicate_mark(record: TicketResult) -> DuplicateMark | None:
    """The live confirmed-duplicate mark on a run, if one stands."""
    for mark in reversed(record.duplicate_marks):
        if mark.active:
            return mark
    return None


def is_confirmed_duplicate(record: TicketResult) -> bool:
    return active_duplicate_mark(record) is not None


def duplicate_exclusions(records: list[TicketResult]) -> list[dict]:
    """The sampling accounting: runs an active mark keeps out of
    calibration and the drift window, each with its mark's lineage."""
    excluded = []
    for record in records:
        mark = active_duplicate_mark(record)
        if mark is not None:
            excluded.append(
                {
                    "ticket_id": record.ticket_id,
                    "duplicate_of": mark.duplicate_of,
                    "marked_by": mark.marked_by,
                    "reason": mark.reason,
                    "marked_at": mark.marked_at,
                }
            )
    return excluded
