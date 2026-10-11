"""The customer dossier: the triage knows who is writing.

Every round before this one answered per-ticket. But the agent
holding a ticket asks a different question first — *have they
written before, and how did it go?* — and until now nothing in the
system could answer it. The dossier is that answer, computed in
code from the stored records:

- **One customer, one key.** The key is the one the thread
  machinery already trusts (:func:`threads.customer_key`): the
  normalised sender email when intake carried one, else the
  normalised sender name under a ``name:`` prefix, else *unknown*
  — and unknown customers have no dossier, because a dossier over
  "we don't know who this is" would be a dossier over strangers.
  The limits are stated, not hidden: two people sharing one inbox
  share a dossier; one person writing from two addresses has two.
  That is what email-keyed support history honestly is.
- **A pure projection.** The dossier reads records; it never
  writes, merges, or re-routes anything. Threads appear under
  their EFFECTIVE ids (corrections.py): a ticket an operator
  detached from a thread does not sit in that thread's row here,
  and its own row says the grouping was corrected.
- **Flags are facts, not opinions.** Two history flags are
  computed from the records under definitions that live in this
  module and nowhere else (:data:`REPEAT_ISSUE_MIN_DECISIONS` /
  :data:`REPEAT_ISSUE_WINDOW_DAYS`,
  :data:`FREQUENT_CONTACT_MIN_TICKETS` /
  :data:`FREQUENT_CONTACT_WINDOW_DAYS`). Each flag carries its
  evidence — the ticket ids and the window — so an agent can
  check the claim in one click. The flags are shown to humans
  and offered to the decision call as context; they NEVER enter
  composition: priority and routing stay the formula's, and a
  repeat customer's ticket composes exactly as a first-timer's
  with the same answers.
- **Small samples are labelled, not dressed up.** The resolved
  rate and the median intake-to-outcome hours are withheld below
  the same floor the workbench uses
  (:data:`workbench.KNOWLEDGE_MIN_SAMPLES`), with a note saying
  why — two outcomes are an anecdote, not a rate.
- **The decision call sees counts, never text.** At triage the
  service composes a compact history block from the customer's
  PRIOR records and offers it through the decision port's
  ``context`` argument, under the reopen idiom: fenced history
  for the LLM/Jev adapters, accepted and deliberately unused by
  the offline heuristics, and what happened is recorded on the
  run (``CustomerHistoryInfo.context_use``). The block carries
  counts, categories, flags, and the last outcome — NEVER another
  ticket's subject or body: history is code-composed arithmetic,
  so a prior ticket's words can neither leak into a stranger's
  prompt nor smuggle instructions into this one.

The service owns the record/input reads (it already resolves a
stored run's customer key for the correction layer); everything
in this module is a pure read or composition over what it passes.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from . import corrections, threads
from .schemas import HistoryFlag, TicketInput, TicketResult
from .sla import parse_timestamp
from .workbench import KNOWLEDGE_MIN_SAMPLES, hours_from_intake

# -- the flag definitions (one place, authored) ----------------------------
#
# repeat_issue: the same category DECIDED this many times inside
# the window. "Decided" is the store's own word (reviews.py): the
# run left the gate as triaged or rejected. Awaiting-agent runs
# don't count — the desk hasn't decided them yet.
REPEAT_ISSUE_MIN_DECISIONS = 3
REPEAT_ISSUE_WINDOW_DAYS = 90
# frequent_contact: this many tickets of ANY status inside the
# window — volume is a fact about contact, not about decisions.
FREQUENT_CONTACT_MIN_TICKETS = 5
FREQUENT_CONTACT_WINDOW_DAYS = 30

DECIDED_STATUSES = ("triaged", "rejected")

KEYING_NOTE = (
    "tickets are grouped by the normalised sender email when intake "
    "carried one, else by the normalised sender name — the same key "
    "the conversation threads use. Two people sharing one inbox share "
    "a dossier; one person writing from two addresses has two."
)


def identity_kind(customer_key: str) -> str:
    """What kind of key this is: ``email`` | ``name`` | ``unknown``."""
    if not customer_key:
        return "unknown"
    return "name" if customer_key.startswith("name:") else "email"


def customer_records(
    records: list[TicketResult],
    customer_keys: dict[str, str],
    customer_key: str,
) -> list[TicketResult]:
    """One customer's records, chronological (intake stamp, then id).

    ``customer_keys`` maps ticket id → the key the service resolved
    for that stored run (the record itself carries no email — the
    key comes from the stored intake, exactly as the correction
    layer resolves it).
    """
    mine = [r for r in records if customer_keys.get(r.ticket_id) == customer_key]

    def sort_key(record: TicketResult):
        parsed = parse_timestamp(record.received_at)
        return (parsed is None, parsed or datetime.min.replace(tzinfo=timezone.utc), record.ticket_id)

    return sorted(mine, key=sort_key)


def _window_ids(
    records: list[TicketResult], anchor: datetime, days: int
) -> list[TicketResult]:
    """The records whose intake stamp falls in [anchor - days, anchor].

    A record with no parseable stamp cannot be placed in a window;
    it still counts in the dossier's totals — it is simply not
    evidence for a time-windowed claim.
    """
    start = anchor - timedelta(days=days)
    inside = []
    for record in records:
        parsed = parse_timestamp(record.received_at)
        if parsed is not None and start <= parsed <= anchor:
            inside.append(record)
    return inside


def compute_flags(
    mine: list[TicketResult], *, anchor: datetime | None
) -> list[HistoryFlag]:
    """The history flags over one customer's records, evidence attached.

    Windows are measured back from ``anchor`` (the dossier's as-of
    moment: the newest ticket's stamp for the read model, the new
    ticket's own stamp at triage time). With no anchor — a customer
    whose records carry no parseable stamp at all — no windowed
    claim can be made and no flag fires.
    """
    if anchor is None:
        return []
    flags: list[HistoryFlag] = []
    decided = [r for r in mine if r.status in DECIDED_STATUSES]
    in_window = _window_ids(decided, anchor, REPEAT_ISSUE_WINDOW_DAYS)
    by_category: dict[str, list[TicketResult]] = {}
    for record in in_window:
        by_category.setdefault(record.category, []).append(record)
    for category in sorted(by_category, key=lambda c: (-len(by_category[c]), c)):
        group = by_category[category]
        if len(group) < REPEAT_ISSUE_MIN_DECISIONS:
            continue
        ids = [r.ticket_id for r in group]
        flags.append(
            HistoryFlag(
                flag="repeat_issue",
                category=category,
                count=len(group),
                window_days=REPEAT_ISSUE_WINDOW_DAYS,
                ticket_ids=ids,
                detail=(
                    f"category {category} decided {len(group)} times in the "
                    f"last {REPEAT_ISSUE_WINDOW_DAYS} days ({', '.join(ids)})"
                ),
            )
        )
    recent = _window_ids(mine, anchor, FREQUENT_CONTACT_WINDOW_DAYS)
    if len(recent) >= FREQUENT_CONTACT_MIN_TICKETS:
        ids = [r.ticket_id for r in recent]
        flags.append(
            HistoryFlag(
                flag="frequent_contact",
                count=len(recent),
                window_days=FREQUENT_CONTACT_WINDOW_DAYS,
                ticket_ids=ids,
                detail=(
                    f"{len(recent)} tickets in the last "
                    f"{FREQUENT_CONTACT_WINDOW_DAYS} days ({', '.join(ids)})"
                ),
            )
        )
    return flags


def outcome_figures(mine: list[TicketResult]) -> dict:
    """Resolved rate + median hours over the outcome-bearing records,
    under the repo's small-sample honesty (the workbench floor)."""
    with_outcome = [r for r in mine if r.outcome is not None]
    n = len(with_outcome)
    resolved = sum(1 for r in with_outcome if r.outcome is not None and r.outcome.resolved)
    hours = [h for h in (hours_from_intake(r) for r in with_outcome) if h is not None]
    enough = n >= KNOWLEDGE_MIN_SAMPLES
    if n == 0:
        note = "no recorded outcomes for this customer yet — nothing to quote"
    elif not enough:
        note = (
            f"only {n} ticket(s) with recorded outcomes — too few to quote "
            f"a resolved rate or a median time (needs {KNOWLEDGE_MIN_SAMPLES})"
        )
    else:
        note = f"figures over the {n} ticket(s) with recorded outcomes"
    return {
        "with_outcome": n,
        "resolved_count": resolved,
        "resolved_rate": round(resolved / n, 4) if enough else None,
        "median_hours_from_intake": (
            round(statistics.median(hours), 2) if enough and hours else None
        ),
        "note": note,
    }


def last_outcome(mine: list[TicketResult]) -> dict | None:
    """The most recently RECORDED outcome (by its own stamp), or None."""
    bearing = [r for r in mine if r.outcome is not None]
    if not bearing:
        return None

    def sort_key(record: TicketResult):
        outcome = record.outcome
        assert outcome is not None
        recorded = parse_timestamp(outcome.recorded_at)
        received = parse_timestamp(record.received_at)
        return (
            recorded or datetime.min.replace(tzinfo=timezone.utc),
            received or datetime.min.replace(tzinfo=timezone.utc),
            record.ticket_id,
        )

    record = max(bearing, key=sort_key)
    outcome = record.outcome
    assert outcome is not None
    return {
        "ticket_id": record.ticket_id,
        "final_category": outcome.final_category,
        "final_queue": outcome.final_queue,
        "resolved": outcome.resolved,
        "recorded_by": outcome.recorded_by,
        "recorded_at": outcome.recorded_at,
    }


def summarise_history(
    customer_key: str,
    mine: list[TicketResult],
    *,
    anchor: datetime | None,
) -> dict:
    """The compact history summary: what the record carries and the
    decision-call block is composed from. ``mine`` is the customer's
    records BEFORE the new ticket (at triage it isn't stored yet)."""
    figures = outcome_figures(mine)
    last = last_outcome(mine)
    return {
        "customer_key": customer_key,
        "prior_tickets": len(mine),
        "open": sum(1 for r in mine if r.status == "awaiting_agent"),
        "decided": sum(1 for r in mine if r.status in DECIDED_STATUSES),
        "reopened": sum(1 for r in mine if r.reopen is not None),
        "outcomes_recorded": figures["with_outcome"],
        "resolved_count": figures["resolved_count"],
        "resolved_rate": figures["resolved_rate"],
        "last_outcome": last,
        "flags": compute_flags(mine, anchor=anchor),
    }


def compose_history_block(summary: dict) -> str:
    """The history block offered to the decision call, as text.

    Counts, categories, flags, the last outcome — composed by code
    from the summary. No subject, no body, no other ticket's words:
    there is nothing in here a prior ticket's text could have
    influenced beyond the arithmetic of its own record.
    """
    key = summary["customer_key"]
    who = key[len("name:"):] if key.startswith("name:") else key
    lines = [
        f"Customer history for {who} (computed in code from the stored "
        "records — counts and categories only, no other ticket's text):",
        f"- {summary['prior_tickets']} earlier ticket(s): "
        f"{summary['decided']} decided, {summary['open']} awaiting an agent, "
        f"{summary['reopened']} of them reopened runs.",
    ]
    recorded = summary["outcomes_recorded"]
    if recorded == 0:
        lines.append("- Recorded outcomes: none yet.")
    elif summary["resolved_rate"] is None:
        lines.append(
            f"- Recorded outcomes: {recorded} ({summary['resolved_count']} "
            "resolved — too few to quote a rate)."
        )
    else:
        lines.append(
            f"- Recorded outcomes: {recorded} ({summary['resolved_count']} "
            f"resolved — rate {summary['resolved_rate']:.2f})."
        )
    last = summary["last_outcome"]
    if last is not None:
        lines.append(
            f"- Most recent outcome: {last['ticket_id']}, final category "
            f"{last['final_category'] or '—'}, "
            f"{'resolved' if last['resolved'] else 'not resolved'}."
        )
    if summary["flags"]:
        lines.append(
            "- History flags: "
            + "; ".join(flag.detail for flag in summary["flags"])
            + "."
        )
    else:
        lines.append("- History flags: none.")
    return "\n".join(lines)


def build_dossier(
    customer_key: str,
    records: list[TicketResult],
    customer_keys: dict[str, str],
    inputs: dict[str, TicketInput | None],
    *,
    now: datetime | None = None,
) -> dict:
    """One customer's dossier: every ticket, the counts, the outcome
    history, their threads, and the flags — a pure projection.

    The window anchor is the customer's newest parseable intake
    stamp (the dossier is a statement about the record as it
    stands), falling back to ``now`` when no stamp parses.
    """
    mine = customer_records(records, customer_keys, customer_key)
    stamps = [parse_timestamp(r.received_at) for r in mine]
    stamps = [s for s in stamps if s is not None]
    anchor = max(stamps) if stamps else now
    figures = outcome_figures(mine)
    flags = compute_flags(mine, anchor=anchor)

    rows = []
    for record in mine:
        ticket_input = inputs.get(record.ticket_id)
        row = threads.run_row(record, ticket_input)
        # The operator overlay, exactly as the thread views resolve
        # it: the effective thread is where the run sits NOW.
        row["thread_id"] = corrections.effective_thread_id(record)
        row["grouping"] = corrections.grouping_view(record)
        row["duplicate_of"] = record.duplicate_of
        row["hours_from_intake"] = hours_from_intake(record)
        rows.append(row)
    thread_summaries = threads.thread_summaries(rows)

    latest_input = None
    for record in reversed(mine):
        if inputs.get(record.ticket_id) is not None:
            latest_input = inputs[record.ticket_id]
            break
    email = ""
    tier = ""
    if latest_input is not None:
        email = latest_input.sender_email
        tier = latest_input.account_tier
    kind = identity_kind(customer_key)
    return {
        "customer_key": customer_key,
        "identity": {
            "kind": kind,
            "email": email,
            "name": mine[-1].sender_name if mine else "",
            "account_tier": tier,
        },
        "keying_note": KEYING_NOTE,
        "as_of": anchor.strftime("%Y-%m-%dT%H:%M:%SZ") if anchor is not None else "",
        "counts": {
            "total": len(mine),
            "open": sum(1 for r in mine if r.status == "awaiting_agent"),
            "decided": sum(1 for r in mine if r.status in DECIDED_STATUSES),
            "reopened": sum(1 for r in mine if r.reopen is not None),
            "with_outcome": figures["with_outcome"],
            "threads": len(thread_summaries),
        },
        "tickets": rows,
        "threads": thread_summaries,
        "outcomes": figures,
        "last_outcome": last_outcome(mine),
        "flags": [flag.model_dump() for flag in flags],
    }
