"""Assignment suggestions: roster/load/outcome fixtures, hand-scored.

The ranking fixture is worked out in the comments before it is
asserted. Weights: track record 0.50, open load 0.30, skills 0.20.
Needs for the target ticket are [billing, checkout] (category, area;
no flags raised).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.assignment import suggest_assignment
from support_triage.composition import QUEUES
from support_triage.roster import load_roster, validate_roster
from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    TicketResult,
)

from helpers import fresh_service

ROSTER = [
    {"id": "pro", "name": "Pat Pro", "queues": ["billing_support"],
     "skills": ["billing", "checkout"]},
    {"id": "new", "name": "Nia New", "queues": ["billing_support"],
     "skills": ["billing"]},
    {"id": "busy", "name": "Bob Busy", "queues": ["billing_support"],
     "skills": ["billing", "checkout", "account_admin"]},
    {"id": "tech", "name": "Tess Tech", "queues": ["technical_support"],
     "skills": ["technical_issue"]},
]


def record(ticket_id, **overrides) -> TicketResult:
    base = dict(
        ticket_id=ticket_id,
        status="awaiting_agent",
        category="billing",
        product_area="checkout",
        composition=CompositionInfo(queue="billing_support"),
    )
    base.update(overrides)
    return TicketResult(**base)


def judged(ticket_id, by, resolved, category="billing") -> TicketResult:
    return record(
        ticket_id,
        status="triaged",
        outcome=AgentOutcome(
            final_category=category, final_queue="billing_support",
            resolved=resolved, recorded_by=by,
            recorded_at="2026-10-01T00:00:00Z",
        ),
    )


def fixture_records():
    """Target + the history the ranking is computed over.

    - pro:  4 judged billing outcomes, 3 resolved → track 0.75;
            0 open → load 1.0; skills 2/2 → 1.0.
            score = .5·.75 + .3·1.0 + .2·1.0 = 0.875
    - busy: 2 judged billing outcomes, 2 resolved → track 1.0;
            3 open (two by name, one by id) → load 0.25;
            skills 2/2 → 1.0.
            score = .5·1.0 + .3·.25 + .2·1.0 = 0.775
    - new:  no judged outcomes → neutral 0.5; 0 open → load 1.0;
            skills 1/2 → 0.5.
            score = .5·.5 + .3·1.0 + .2·.5 = 0.65
    Expected order: pro, busy, new. (busy has the perfect track
    record and still loses — to load. That's the point of the sum.)
    """
    records = [record("TGT-1")]
    records += [judged(f"P-{i}", "pro", resolved=i < 3) for i in range(4)]
    records += [judged(f"B-{i}", "busy", resolved=True) for i in range(2)]
    records += [
        record("L-1", decision_reason="[assigned to Bob Busy]"),
        record("L-2", decision_reason="note [assigned to Bob Busy]"),
        record("L-3", decision_reason="[assigned to busy]"),
        # a decided ticket naming pro is NOT open load:
        record("L-4", status="triaged", decision_reason="[assigned to Pat Pro]"),
        # another category's outcomes do not count for anyone here:
        judged("O-1", "pro", resolved=False, category="complaint"),
    ]
    return records


def test_ranking_matches_the_hand_scored_fixture():
    records = fixture_records()
    payload = suggest_assignment(records[0], records, ROSTER)
    assert payload["candidates"] == 3  # tech is not on this queue
    assert payload["suggestion"]["agent_id"] == "pro"
    assert payload["suggestion"]["score"] == 0.875
    assert [c["agent_id"] for c in payload["runners_up"]] == ["busy", "new"]
    assert payload["runners_up"][0]["score"] == 0.775
    assert payload["runners_up"][1]["score"] == 0.65


def test_reasons_are_printed_per_factor():
    records = fixture_records()
    payload = suggest_assignment(records[0], records, ROSTER)
    pro_reasons = payload["suggestion"]["reasons"]
    assert "covers queue billing_support" in pro_reasons
    assert "resolved 3 of 4 recorded 'billing' outcomes (75%)" in pro_reasons
    assert "0 open ticket(s) currently assigned" in pro_reasons
    assert "skills match: billing, checkout" in pro_reasons

    new = payload["runners_up"][1]
    assert new["factors"]["track_record"] == 0.5
    assert any("no recorded outcomes in 'billing' yet" in r for r in new["reasons"])

    busy = payload["runners_up"][0]
    assert busy["open_tickets"] == 3
    assert busy["factors"]["open_load"] == 0.25
    assert any("3 open ticket(s) currently assigned" in r for r in busy["reasons"])


def test_uncovered_queue_returns_an_honest_no_suggestion():
    target = record(
        "TGT-2",
        category="complaint",
        composition=CompositionInfo(queue="compliance_review"),
    )
    payload = suggest_assignment(target, [target], ROSTER)
    assert payload["suggestion"] is None
    assert payload["runners_up"] == []
    assert "no roster agent covers queue 'compliance_review'" in payload["note"]


def test_decided_ticket_gets_no_suggestion():
    target = record("TGT-3", status="triaged")
    payload = suggest_assignment(target, [target], ROSTER)
    assert payload["suggestion"] is None
    assert "not waiting at the gate" in payload["note"]


# ---------------------------------------------------------------------------
# the bundled roster
# ---------------------------------------------------------------------------


def test_bundled_roster_covers_every_queue_the_router_produces():
    roster = load_roster()
    covered = {queue for agent in roster for queue in agent["queues"]}
    assert covered == set(QUEUES)


def test_roster_validation_rejects_an_unknown_queue():
    with pytest.raises(ValueError, match="unknown queues"):
        validate_roster(
            [{"id": "x", "name": "X", "queues": ["moon_support"], "skills": []}]
        )


# ---------------------------------------------------------------------------
# the endpoint — and the invariant: it assigns nothing
# ---------------------------------------------------------------------------


@pytest.fixture()
def assignment_client():
    service = fresh_service()
    service.store.save(record("TGT-1"))
    api.set_service(service)
    with TestClient(api.app) as client:
        yield client, service
    api.set_service(None)


def test_assignment_endpoint_suggests_and_writes_nothing(assignment_client):
    client, service = assignment_client
    before = service.store.get("TGT-1")
    body = client.get("/api/runs/TGT-1/assignment").json()
    assert body["suggestion"] is not None
    assert body["suggestion"]["agent_id"] in {a["id"] for a in load_roster()}
    assert "a human assigns" in body["note"]
    assert body["suggestion"]["reasons"]  # the working is printed
    after = service.store.get("TGT-1")
    assert after.status == "awaiting_agent"
    assert after.decision_reason == before.decision_reason
    assert after.decision == before.decision


def test_assignment_endpoint_unknown_ticket_is_404(assignment_client):
    client, _service = assignment_client
    assert client.get("/api/runs/NOPE/assignment").status_code == 404
