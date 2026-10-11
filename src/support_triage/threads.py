"""Conversation threads: runs that belong to one conversation.

A customer rarely writes once. They write, get an answer, write
again about the same thing — and each of those contacts is its own
run with its own record. This module groups the runs into
**threads**: one customer, one issue, one conversation.

The grouping is code, computed at intake, and deliberately modest:

- **same customer** — the normalised sender email when the intake
  carried one, else the normalised sender name. Different customers
  never share a thread, however similar their words.
- **same issue** — the token containment of the subject+body
  token sets (shared tokens over the smaller set) at or over
  :data:`THREAD_OVERLAP_THRESHOLD`. A ticket from the same customer
  about something else scores low and starts its own thread — the
  near-miss is the normal case, not an error.
- **same conversation, not same history** — the earlier run must sit
  within :data:`THREAD_WINDOW_DAYS` before the new one.

A thread id is ``thr-<root ticket id>``: the id of the earliest run
in the thread. A reopen run joins its prior run's thread by
construction (the service passes it), never by re-derivation.

Grouping is a **label**. It assigns an id at intake and reads group
the records afterwards; it never merges, edits, or re-routes a run.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .sla import parse_timestamp

THREAD_WINDOW_DAYS = 30
THREAD_OVERLAP_THRESHOLD = 0.35

_WORD = re.compile(r"[a-z0-9']+")

# Filler that every ticket shares and no issue owns — left in, it
# would make strangers look related. Content words carry the issue.
_STOPWORDS = {
    "the", "and", "for", "you", "your", "yours", "our", "ours", "this",
    "that", "with", "from", "have", "has", "had", "are", "was", "were",
    "will", "would", "can", "could", "should", "please", "thanks",
    "thank", "hello", "hi", "hey", "just", "get", "got", "but", "not",
    "all", "any", "when", "what", "how", "why", "its", "it's", "in",
    "on", "of", "to", "a", "an", "is", "i", "we", "my", "me", "am",
    "it", "isnt", "isn't", "dont", "don't", "cant", "can't", "wont",
    "won't", "there", "here", "about", "into", "over", "after",
    "before", "again", "still", "now", "then", "than", "too", "very",
}


def tokens_for(subject: str, body: str) -> set[str]:
    """The issue's vocabulary: content tokens of subject + body."""
    words = _WORD.findall(f"{subject or ''}\n{body or ''}".lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def customer_key(sender_email: str = "", sender_name: str = "") -> str:
    """Who a ticket is from, normalised: email first, name as fallback.

    An empty key means "we don't know who this is" — unknown
    customers never thread with anyone, not even each other.
    """
    email = (sender_email or "").strip().lower()
    if email:
        return email
    name = " ".join((sender_name or "").strip().lower().split())
    return f"name:{name}" if name else ""


def overlap(a: set[str], b: set[str]) -> float:
    """Token containment of two token sets: shared tokens over the
    SMALLER set's size (the overlap coefficient; 0.0 when either set
    is empty).

    Containment, not Jaccard: a follow-up is usually much shorter
    than the ticket it continues, and Jaccard would score a genuine
    follow-up that repeats the original's key phrases at ~0.2 just
    for being brief. Containment asks the question thread grouping
    actually cares about — how much of the shorter text's vocabulary
    already appeared in the other text.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def effective_thread_id(thread_id: str, ticket_id: str) -> str:
    """A run's thread id, deriving the singleton for records that
    predate threads (or were never assigned one): a run with no
    stored id is its own thread's root."""
    return thread_id if thread_id else f"thr-{ticket_id}"


def assign_thread(
    *,
    ticket_id: str,
    customer: str,
    subject: str,
    body: str,
    received_at: datetime | None,
    candidates: list[dict],
) -> dict:
    """Decide the thread for one ticket at intake.

    ``candidates`` are the earlier runs as plain dicts (``ticket_id``,
    ``customer``, ``subject``, ``body``, ``received_at`` as a
    datetime or None, ``thread_id`` as stored). The best candidate —
    same customer, inside the window, overlap at or over the
    threshold; ties broken toward the most recent — lends its
    thread. Anything less starts a new thread rooted at this ticket.
    Pure and deterministic: same inputs, same thread, every time.
    """
    own = {"thread_id": f"thr-{ticket_id}", "matched_ticket_id": None, "overlap": 0.0}
    if not customer or received_at is None:
        return own
    new_tokens = tokens_for(subject, body)
    if not new_tokens:
        return own
    best: tuple[float, datetime, str, dict] | None = None
    for candidate in candidates:
        if candidate.get("customer") != customer:
            continue
        candidate_at = candidate.get("received_at")
        if candidate_at is None:
            continue
        age = received_at - candidate_at
        if age < timedelta(0) or age > timedelta(days=THREAD_WINDOW_DAYS):
            continue
        score = overlap(new_tokens, tokens_for(candidate.get("subject", ""), candidate.get("body", "")))
        if score < THREAD_OVERLAP_THRESHOLD:
            continue
        ranking = (score, candidate_at, candidate["ticket_id"])
        if best is None or ranking > best[:3]:
            best = (score, candidate_at, candidate["ticket_id"], candidate)
    if best is None:
        return own
    score, _at, _tid, candidate = best
    return {
        "thread_id": effective_thread_id(candidate.get("thread_id", ""), candidate["ticket_id"]),
        "matched_ticket_id": candidate["ticket_id"],
        "overlap": round(score, 4),
    }


def run_row(record, ticket_input) -> dict:
    """One run as a thread-view row (service-facing assembly)."""
    outcome = None
    if record.outcome is not None:
        outcome = {
            "final_category": record.outcome.final_category,
            "final_queue": record.outcome.final_queue,
            "resolved": record.outcome.resolved,
            "recorded_by": record.outcome.recorded_by,
            "recorded_at": record.outcome.recorded_at,
        }
    return {
        "ticket_id": record.ticket_id,
        "thread_id": effective_thread_id(record.thread_id, record.ticket_id),
        "subject": record.subject,
        "sender_name": record.sender_name,
        "sender_email": ticket_input.sender_email if ticket_input is not None else "",
        "channel": record.channel,
        "received_at": record.received_at,
        "status": record.status,
        "category": record.category,
        "queue": record.composition.queue,
        "priority_band": record.composition.priority_band,
        "is_reopen": record.reopen is not None,
        "prior_run_id": record.reopen.prior_run_id if record.reopen else "",
        "reopen_count": record.reopen.reopen_count if record.reopen else 0,
        "outcome": outcome,
    }


def _chronological(rows: list[dict]) -> list[dict]:
    def key(row: dict):
        parsed = parse_timestamp(row.get("received_at", ""))
        return (parsed is None, parsed or datetime.min, row["ticket_id"])

    return sorted(rows, key=key)


def thread_summaries(rows: list[dict]) -> list[dict]:
    """Group run rows into thread summaries, latest activity first."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["thread_id"], []).append(row)
    summaries: list[dict] = []
    for thread_id, members in groups.items():
        members = _chronological(members)
        latest_outcome = None
        for row in reversed(members):
            if row["outcome"] is not None:
                latest_outcome = row["outcome"]
                break
        root = members[0]
        summaries.append(
            {
                "thread_id": thread_id,
                "customer_name": root["sender_name"],
                "customer_email": root["sender_email"],
                "subject": root["subject"],
                "run_count": len(members),
                "ticket_ids": [row["ticket_id"] for row in members],
                "open_runs": sum(1 for row in members if row["status"] == "awaiting_agent"),
                "reopened": any(row["is_reopen"] for row in members),
                # An operator has overruled the computed grouping for
                # at least one run here (corrections.py) — the rows
                # carry the lineage; the summary just flies the flag.
                "corrected": any(
                    (row.get("grouping") or {}).get("corrected") for row in members
                ),
                "latest_outcome": latest_outcome,
                "first_received_at": root["received_at"],
                "latest_received_at": members[-1]["received_at"],
            }
        )
    summaries.sort(key=lambda s: (s["latest_received_at"], s["thread_id"]), reverse=True)
    return summaries


def thread_chain(rows: list[dict], thread_id: str) -> dict:
    """One thread, whole: the runs in order, each with its outcome."""
    members = _chronological([row for row in rows if row["thread_id"] == thread_id])
    return {"thread_id": thread_id, "run_count": len(members), "runs": members}
