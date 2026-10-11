"""Conversation threads (round 6): grouping that only labels.

The pins:

- same customer + subject/body token overlap at or above the policy
  threshold + inside the 30-day window → one thread;
- a near-miss (same customer, unrelated text) starts a NEW thread —
  grouping never reaches across topics;
- a different customer with identical text never joins;
- a reopen joins its prior run's thread by construction;
- thread reads never mutate a run (byte-identical records);
- legacy records with no stored thread_id read as singleton threads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api, threads
from support_triage.schemas import AgentOutcome
from support_triage.sla import parse_timestamp

from helpers import fresh_service, make_ticket


BILLING_BODY = (
    "I was charged twice for the April subscription — two identical payments of $49 "
    "left my card. Please refund one of them."
)

FOLLOW_UP_BODY = (
    "Following up on my earlier message: I was charged twice for the April "
    "subscription, and the refund for one of the two payments still has not "
    "arrived on my card."
)

HOW_TO_BODY = (
    "How do I invite a new team member to our workspace and set their user role? "
    "I can't find the team members page in the account settings."
)


def _ticket(ticket_id, subject, body, email="priya@example.com", when="2026-10-01T09:00:00Z", name="Priya Nair"):
    return make_ticket(
        ticket_id=ticket_id,
        subject=subject,
        body=body,
        sender_name=name,
        sender_email=email,
        received_at=when,
    )


def _pair_service():
    """Two related billing tickets from one customer, two days apart."""
    service = fresh_service()
    service.run_ticket(_ticket("T-1", "Charged twice for the April subscription", BILLING_BODY))
    service.run_ticket(
        _ticket(
            "T-2",
            "Still waiting on the April double-charge refund",
            FOLLOW_UP_BODY,
            when="2026-10-03T09:00:00Z",
        )
    )
    return service


# --- the deterministic function, unit-level -------------------------------


def _decide(ticket, candidates):
    return threads.assign_thread(
        ticket_id=ticket.ticket_id,
        customer=threads.customer_key(ticket.sender_email, ticket.sender_name),
        subject=ticket.subject,
        body=ticket.body,
        received_at=parse_timestamp(ticket.received_at),
        candidates=candidates,
    )


def _candidate_of(ticket):
    return {
        "ticket_id": ticket.ticket_id,
        "customer": threads.customer_key(ticket.sender_email, ticket.sender_name),
        "subject": ticket.subject,
        "body": ticket.body,
        "received_at": parse_timestamp(ticket.received_at),
        "thread_id": "",
    }


def test_assign_thread_joins_on_overlap_and_rejects_the_rest():
    first = _ticket("A-1", "Charged twice for the April subscription", BILLING_BODY)
    candidates = [_candidate_of(first)]
    related = _ticket("A-2", "Still waiting on the April double-charge refund", FOLLOW_UP_BODY,
                      when="2026-10-05T09:00:00Z")
    unrelated = _ticket("A-3", "How do I add a team member?", HOW_TO_BODY, when="2026-10-05T09:00:00Z")
    other_customer = _ticket("A-4", "Charged twice for the April subscription", BILLING_BODY,
                             email="someone.else@example.com", name="Other Person",
                             when="2026-10-05T09:00:00Z")
    too_late = _ticket("A-5", "Still waiting on the April double-charge refund", FOLLOW_UP_BODY,
                       when="2026-12-15T09:00:00Z")

    joined = _decide(related, candidates)
    assert joined["thread_id"] == "thr-A-1"
    assert joined["matched_ticket_id"] == "A-1"
    assert joined["overlap"] >= threads.THREAD_OVERLAP_THRESHOLD
    # The near-miss: same customer, different issue — a new thread.
    assert _decide(unrelated, candidates)["thread_id"] == "thr-A-3"
    # Same words, different customer — never a thread.
    assert _decide(other_customer, candidates)["thread_id"] == "thr-A-4"
    # Same everything, 75 days later — outside the window, a new thread.
    assert _decide(too_late, candidates)["thread_id"] == "thr-A-5"


def test_overlap_and_customer_key_units():
    tokens = threads.tokens_for
    assert threads.overlap(tokens("charged twice", "april"), tokens("charged twice", "april")) == 1.0
    assert threads.overlap(tokens("charged twice", ""), tokens("add team", "member")) == 0.0
    # Containment: the shorter text fully inside the longer scores 1.0.
    assert threads.overlap(tokens("charged twice", ""), tokens("charged twice", "april refund")) == 1.0
    assert threads.overlap(set(), {"charged"}) == 0.0
    assert threads.customer_key("  Priya@Example.COM ", "Ignored") == "priya@example.com"
    assert threads.customer_key("", "  Priya   Nair ") == "name:priya nair"
    assert threads.customer_key("", "") == ""
    assert threads.effective_thread_id("", "A-9") == "thr-A-9"
    assert threads.effective_thread_id("thr-A-1", "A-2") == "thr-A-1"


# --- service level ----------------------------------------------------------


def test_related_tickets_share_a_thread_and_reads_do_not_mutate():
    service = _pair_service()
    first = service.get_record("T-1")
    second = service.get_record("T-2")
    assert first.thread_id == "thr-T-1"
    assert second.thread_id == "thr-T-1"

    before = {tid: service.get_record(tid).model_dump_json() for tid in ("T-1", "T-2")}
    listing = service.list_threads()
    (thread,) = listing["threads"]
    assert thread["thread_id"] == "thr-T-1"
    assert thread["run_count"] == 2
    assert thread["ticket_ids"] == ["T-1", "T-2"]
    assert thread["reopened"] is False
    assert thread["open_runs"] == 2
    assert thread["latest_outcome"] is None

    detail = service.get_thread("thr-T-1")
    assert [run["ticket_id"] for run in detail["runs"]] == ["T-1", "T-2"]
    assert service.run_thread("T-2")["thread_id"] == "thr-T-1"
    # Grouping only labels: both records are byte-identical after reads.
    assert {tid: service.get_record(tid).model_dump_json() for tid in before} == before


def test_thread_view_carries_outcomes_and_reopen_state():
    service = _pair_service()
    service.record_outcome(
        "T-1",
        AgentOutcome(final_category="billing", final_queue="billing_support",
                     resolved=False, recorded_by="agent-1"),
    )
    reopened = service.reopen_run("T-1", message="The refund is still missing.", actor="agent-1")
    detail = service.get_thread("thr-T-1")
    assert detail["run_count"] == 3
    assert detail["reopened"] is True
    outcomes = {run["ticket_id"]: run["outcome"] for run in detail["runs"]}
    assert outcomes["T-1"]["resolved"] is False
    assert outcomes[reopened.result.ticket_id] is None
    assert detail["latest_outcome"]["resolved"] is False


def test_near_miss_stays_separate_at_service_level():
    service = fresh_service()
    service.run_ticket(_ticket("N-1", "Charged twice for the April subscription", BILLING_BODY))
    service.run_ticket(_ticket("N-2", "How do I add a team member?", HOW_TO_BODY,
                                when="2026-10-02T09:00:00Z"))
    listing = service.list_threads()
    assert len(listing["threads"]) == 2
    assert {t["thread_id"] for t in listing["threads"]} == {"thr-N-1", "thr-N-2"}


def test_unknown_thread_and_run_thread_are_404_shaped():
    service = fresh_service()
    with pytest.raises(KeyError):
        service.get_thread("thr-NOPE")
    with pytest.raises(KeyError):
        service.run_thread("NOPE")


# --- the HTTP surface ---------------------------------------------------------


@pytest.fixture()
def client():
    service = fresh_service()
    api.set_service(service)
    yield TestClient(api.app), service
    api.set_service(None)


def test_api_thread_views(client):
    http, service = client
    service.run_ticket(_ticket("W-1", "Charged twice for the April subscription", BILLING_BODY))
    service.run_ticket(_ticket("W-2", "Still waiting on the April double-charge refund",
                                FOLLOW_UP_BODY, when="2026-10-03T09:00:00Z"))

    listing = http.get("/api/threads")
    assert listing.status_code == 200
    (thread,) = listing.json()["threads"]
    assert thread["thread_id"] == "thr-W-1" and thread["run_count"] == 2

    detail = http.get("/api/threads/thr-W-1")
    assert detail.status_code == 200
    assert [run["ticket_id"] for run in detail.json()["runs"]] == ["W-1", "W-2"]

    by_run = http.get("/api/runs/W-2/thread")
    assert by_run.status_code == 200
    assert by_run.json()["thread_id"] == "thr-W-1"

    assert http.get("/api/threads/thr-NOPE").status_code == 404
    assert http.get("/api/runs/NOPE/thread").status_code == 404
