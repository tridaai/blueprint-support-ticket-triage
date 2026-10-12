"""The shift handoff: one composed page, and the handshake over it.

Pins, by layer:

- the composition — every section of the page equals its source
  view at the same instant, recomputed independently here: the
  watch's counts and ranked rows, the proposal scan, the open
  incidents' member counts, the queue rows' own SLA states
  grouped per queue, and the review backlog as drawn minus
  verdicted per sample. No forked arithmetic: the page selects
  and counts what the views produced;
- the hand-derived figures — a staged world whose watch bands,
  queue splits, and backlog are computed by hand in this file's
  comments and asserted exactly, so a staging drift names itself;
- the cap — the watch section carries the top rows under the
  authored cap, printed on the payload, with the full ranked
  total beside them;
- the handshake — creation names two different leads or is
  refused; acceptance belongs to the addressed lead alone
  (self-acceptance and third parties refused by name); accepting
  with exceptions stores the dispute verbatim and edits no
  figure; a settled page refuses a second acceptance; creating
  over an outstanding page supersedes it, and the lapsed page
  can no longer be accepted;
- the firewall — run records byte-identical across create,
  accept, and supersede: a handoff is a desk-level record and a
  shift change moves no triage;
- the surfaces — the API composes, lists with the outstanding
  page singled out, accepts, and maps its refusals (422 unnamed
  or wrong actor, 409 settled or superseded, 404 unknown).

All clocks are injected; no test races the wall.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.handoffs import (
    TOP_WATCH_ROWS,
    HandoffConflict,
    HandoffError,
)
from support_triage.schemas import (
    CompositionInfo,
    Incident,
    IncidentMember,
    ReviewSample,
    ReviewVerdict,
    TicketInput,
    TicketResult,
)
from support_triage.sla import format_timestamp

from helpers import fresh_service

NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
EARLIER = datetime(2026, 6, 2, 10, 0, 0, tzinfo=timezone.utc)

FROM = "L. Lead"
TO = "M. Lead"


def _record(
    ticket_id,
    *,
    band="low",
    queue="general_support",
    received_at="2026-06-02T11:00:00Z",
    status="awaiting_agent",
    category="technical_issue",
    product_area="dashboard",
    subject="",
    score=40.0,
):
    return TicketResult(
        ticket_id=ticket_id,
        status=status,
        subject=subject or f"subject {ticket_id}",
        sender_name="Test Sender",
        received_at=received_at,
        category=category,
        product_area=product_area,
        answers=[],
        composition=CompositionInfo(
            priority_band=band, priority_score=score, queue=queue
        ),
    )


# The staged world, hand-derived at NOW (2026-06-02T12:00Z).
# Band targets: critical 1h, high 4h, medium 24h, low 72h.
#
# - H-BREACH  high, received 05:00 → 7h old, 3h past its target:
#   breached (−10800s), queue incident_response.
# - H-SOON    low, received May 30 12:30 → 71.5h old, 1800s left:
#   watch due_within_1h; clock state due_soon (last quarter of
#   72h is 18h), queue billing_support.
# - H-DUE     medium, received Jun 1 14:00 → 22h old, 2h left:
#   watch due_within_4h; clock due_soon (last quarter of 24h is
#   6h), queue technical_support.
# - H-C1..C4  medium billing/checkout, received 08:00–09:50 → ~20h
#   left: watch later, clock ok, queue billing_support — and one
#   cluster: same placement, one window, shared words (the saved
#   inputs below carry the double-charge text).
# - H-M1/M2   low, received 09:00 → later/ok, queue
#   account_support; members of the stored open incident.
# - H-L1      low, received 11:00 → later/ok, technical_support.
# - H-L2      low, received 10:00 → later/ok, general_support.
# - H-DEC1/2  decided (triaged / rejected): no clock, not open;
#   the QA sample draws exactly these two, one verdicted.
#
# Placements are all distinct (area, category) buckets outside
# the cluster, and subjects carry real, disjoint words — a record
# with no saved input is tokenised on its subject alone, and
# placeholder subjects would share enough tokens to fake a
# second cluster the staging never intended.
#
# Watch: 11 rows — breached 1, due_within_1h 1, due_within_4h 1,
# later 8. Ranked: H-BREACH, H-SOON, H-DUE, then the cluster by
# remaining time (C1 72000s … C4 77400s), then M1/M2, L2, L1.
# Top 5 under the cap: BREACH, SOON, DUE, C1, C2.
# Queues: incident_response 1 (breached 1); billing_support 5
# (due_soon 1 — H-SOON — ok 4); technical_support 2 (due_soon 1,
# ok 1); account_support 2 (ok 2); general_support 1 (ok 1).
# QA: one sample, drawn 2, verdicted 1, backlog 1.
# Incidents: one open (2 members, both open); one live proposal
# (the cluster, 4 members).

_CLUSTER_BODIES = {
    "H-C1": "I was charged twice at checkout for order 5521. Two identical payments of 59 dollars left my card. The checkout page froze once but I clicked pay a single time. Please refund the duplicate payment.",
    "H-C2": "Checkout charged my card twice for the same order. Two payments of 59 dollars show on my card statement. I only placed one order. Please refund one payment.",
    "H-C3": "There are two charges on my card for a single checkout order. The payment page froze during checkout and now I was charged twice. Refund the second charge please.",
    "H-C4": "The checkout took payment twice from my card this morning. One order, two payments of 59 dollars. Please refund the extra payment as soon as possible.",
}
_CLUSTER_SUBJECTS = {
    "H-C1": "Charged twice at checkout",
    "H-C2": "Double payment at checkout",
    "H-C3": "Two charges for one checkout order",
    "H-C4": "Checkout took the payment twice",
}
_CLUSTER_RECEIVED = {
    "H-C1": "2026-06-02T08:00:00Z",
    "H-C2": "2026-06-02T08:40:00Z",
    "H-C3": "2026-06-02T09:15:00Z",
    "H-C4": "2026-06-02T09:50:00Z",
}


def build_world(service):
    store = service.store
    store.save(
        _record(
            "H-BREACH", band="high", queue="incident_response",
            received_at="2026-06-02T05:00:00Z", score=90.0,
            subject="Production outage — everything is down",
        )
    )
    store.save(
        _record(
            "H-SOON", band="low", queue="billing_support",
            received_at="2026-05-30T12:30:00Z", category="billing",
            product_area="invoicing",
            subject="Charged twice for the March subscription",
        )
    )
    store.save(
        _record(
            "H-DUE", band="medium", queue="technical_support",
            received_at="2026-06-01T14:00:00Z", product_area="reports",
            subject="Nightly export job stuck at ninety percent",
        )
    )
    for ticket_id, body in _CLUSTER_BODIES.items():
        record = _record(
            ticket_id, band="medium", queue="billing_support",
            received_at=_CLUSTER_RECEIVED[ticket_id],
            category="billing", product_area="checkout",
        )
        record.subject = _CLUSTER_SUBJECTS[ticket_id]
        store.save(record)
        store.save_input(
            TicketInput(
                ticket_id=ticket_id,
                subject=_CLUSTER_SUBJECTS[ticket_id],
                body=body,
                sender_name="Test Sender",
                sender_email=f"{ticket_id.lower()}@example.com",
                channel="email",
                account_tier="pro",
            )
        )
    store.save(
        _record(
            "H-M1", queue="account_support", received_at="2026-06-02T09:00:00Z",
            category="account_access", product_area="account",
            subject="Password reset email never arrives",
        )
    )
    store.save(
        _record(
            "H-M2", queue="account_support", received_at="2026-06-02T09:00:00Z",
            category="account_access", product_area="account",
            subject="Locked out after changing my password",
        )
    )
    store.save(
        _record(
            "H-L1", queue="technical_support", category="how_to",
            subject="How do I pin a chart to the home page?",
        )
    )
    store.save(
        _record(
            "H-L2", queue="general_support", received_at="2026-06-02T10:00:00Z",
            product_area="notifications",
            subject="Webhook deliveries stopped on Tuesday",
        )
    )
    store.save(
        _record("H-DEC1", status="triaged", queue="billing_support",
                category="billing", product_area="invoicing",
                subject="Invoice for April shows unfamiliar seats")
    )
    store.save(
        _record("H-DEC2", status="rejected", queue="technical_support",
                product_area="reports",
                subject="Quarterly report totals look wrong")
    )
    store.save_incident(
        Incident(
            incident_id="INC-0001",
            title="Possible incident: account access in account (2 reports since 09:00 UTC)",
            product_area="account",
            category="account_access",
            status="open",
            anchor_ticket_id="H-M1",
            members=[
                IncidentMember(ticket_id="H-M1", added_by=FROM, origin="proposal"),
                IncidentMember(ticket_id="H-M2", added_by=FROM, origin="proposal"),
            ],
            opened_by=FROM,
            opened_at="2026-06-02T09:30:00Z",
        )
    )
    store.save_review_sample(
        ReviewSample(
            sample_id="QAS-0001",
            seed=7,
            size_requested=2,
            created_by="qa-lead",
            created_at="2026-06-02T08:00:00Z",
            run_ids=["H-DEC1", "H-DEC2"],
        )
    )
    store.save_review_verdict(
        ReviewVerdict(
            sample_id="QAS-0001",
            ticket_id="H-DEC1",
            reviewer="qa-lead",
            families={
                "category": "agree",
                "priority_drivers": "agree",
                "routing": "agree",
                "verify": "agree",
            },
            overall="agree",
            recorded_at="2026-06-02T09:00:00Z",
        )
    )
    return service


def _snapshot(service) -> dict:
    return {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }


def _create(service, **overrides):
    kwargs = {"from_lead": FROM, "to_lead": TO, "note": "", "now": NOW}
    kwargs.update(overrides)
    return service.create_handoff(**kwargs)


# ---------------------------------------------------------------------------
# the composition equals the source views
# ---------------------------------------------------------------------------


def test_sections_equal_the_source_views_at_the_same_instant():
    service = build_world(fresh_service())
    handoff = _create(service, note="the outage is the vendor's; watch the cluster")
    sections = handoff.sections

    # The watch section is the watch's own payload, selected.
    watch = service.breach_watch(NOW)
    assert sections["watch"]["counts"] == watch["counts"]
    assert sections["watch"]["open_runs"] == watch["open_runs"]
    assert sections["watch"]["unclocked_open"] == watch["unclocked_open"]
    assert sections["watch"]["rows_total"] == len(watch["rows"])
    top = sections["watch"]["top_rows"]
    assert len(top) == TOP_WATCH_ROWS
    for carried, row in zip(top, watch["rows"]):
        for key in (
            "ticket_id", "subject", "queue", "priority_band",
            "watch_band", "seconds_to_breach", "due_at", "assignee",
        ):
            assert carried[key] == row[key]

    # The incidents section is the scan + the incident views.
    scan = service.cluster_proposals()
    assert sections["incidents"]["proposals_pending"] == len(scan["proposals"])
    assert (
        sections["incidents"]["top_proposal"]["proposal_id"]
        == scan["proposals"][0]["proposal_id"]
    )
    views = service.list_incidents()
    open_views = [v for v in views if v["incident"]["status"] == "open"]
    assert sections["incidents"]["open_count"] == len(open_views)
    carried = sections["incidents"]["open"][0]
    source = open_views[0]
    assert carried["incident_id"] == source["incident"]["incident_id"]
    assert carried["member_count"] == source["member_count"]
    assert carried["open_members"] == source["open_members"]

    # The queues section groups the queue view's own rows.
    rows = service.store.queue_summary(now=NOW)
    expected: dict[str, dict] = {}
    for row in rows:
        entry = expected.setdefault(
            row["queue"],
            {"queue": row["queue"], "open": 0, "breached": 0,
             "due_soon": 0, "ok": 0, "unclocked": 0},
        )
        entry["open"] += 1
        entry[row["sla_state"] or "unclocked"] += 1
    assert {r["queue"]: r for r in sections["queues"]["rows"]} == expected
    assert sections["queues"]["open_total"] == len(rows)

    # The QA section is the samples' own accounting.
    view = service.get_review_sample("QAS-0001")
    per_sample = sections["qa_review"]["per_sample"]
    assert per_sample == [
        {
            "sample_id": "QAS-0001",
            "drawn": len(view["sample"]["run_ids"]),
            "verdicted": view["verdicted_runs"],
            "backlog": len(view["sample"]["run_ids"]) - view["verdicted_runs"],
        }
    ]
    assert sections["qa_review"]["backlog"] == 1


def test_hand_derived_figures():
    service = build_world(fresh_service())
    sections = _create(service).sections

    assert sections["watch"]["counts"] == {
        "breached": 1,
        "due_within_1h": 1,
        "due_within_4h": 1,
        "later": 8,
    }
    assert sections["watch"]["open_runs"] == 11
    assert [row["ticket_id"] for row in sections["watch"]["top_rows"]] == [
        "H-BREACH", "H-SOON", "H-DUE", "H-C1", "H-C2",
    ]
    assert sections["watch"]["top_rows"][0]["seconds_to_breach"] == -10800.0

    queues = {row["queue"]: row for row in sections["queues"]["rows"]}
    assert queues["incident_response"] == {
        "queue": "incident_response", "open": 1, "breached": 1,
        "due_soon": 0, "ok": 0, "unclocked": 0,
    }
    assert queues["billing_support"]["open"] == 5
    assert queues["billing_support"]["due_soon"] == 1  # H-SOON's last quarter
    assert queues["billing_support"]["ok"] == 4
    assert queues["technical_support"]["due_soon"] == 1  # H-DUE
    assert sections["queues"]["open_total"] == 11
    # Pressure order: the breached queue first, then by due-soon.
    assert [row["queue"] for row in sections["queues"]["rows"]] == [
        "incident_response",
        "billing_support",
        "technical_support",
        "account_support",
        "general_support",
    ]

    assert sections["incidents"]["open"] == [
        {
            "incident_id": "INC-0001",
            "title": "Possible incident: account access in account (2 reports since 09:00 UTC)",
            "member_count": 2,
            "open_members": 2,
        }
    ]
    assert sections["incidents"]["proposals_pending"] == 1
    assert sections["incidents"]["top_proposal"]["member_count"] == 4

    assert sections["qa_review"]["samples"] == 1
    assert sections["qa_review"]["drawn_runs"] == 2
    assert sections["qa_review"]["verdicted_runs"] == 1
    assert sections["qa_review"]["backlog"] == 1

    assert sections["constants"] == {"top_watch_rows": TOP_WATCH_ROWS}


def test_as_of_states_the_boundary_the_figures_were_composed_at():
    service = build_world(fresh_service())
    handoff = _create(service, as_of=EARLIER)
    # Composed at 10:00, stamped as created at NOW: the boundary
    # is the as_of, and the figures are the 10:00 figures —
    # H-SOON had 2.5h left then, a due_within_4h row, not 1h.
    assert handoff.as_of == format_timestamp(EARLIER)
    assert handoff.created_at == format_timestamp(NOW)
    assert handoff.sections["watch"]["counts"] == {
        "breached": 1,
        "due_within_1h": 0,
        "due_within_4h": 2,
        "later": 8,
    }


def test_an_empty_desk_composes_an_honest_page():
    service = fresh_service()
    handoff = _create(service)
    assert handoff.sections["watch"]["rows_total"] == 0
    assert handoff.sections["watch"]["top_rows"] == []
    assert handoff.sections["incidents"]["open"] == []
    assert handoff.sections["incidents"]["top_proposal"] is None
    assert handoff.sections["queues"]["rows"] == []
    assert handoff.sections["qa_review"]["backlog"] == 0


# ---------------------------------------------------------------------------
# the handshake
# ---------------------------------------------------------------------------


def test_creation_names_two_different_leads():
    service = build_world(fresh_service())
    with pytest.raises(HandoffError, match="from_lead is required"):
        _create(service, from_lead="  ")
    with pytest.raises(HandoffError, match="to_lead is required"):
        _create(service, to_lead="")
    with pytest.raises(HandoffError, match="cannot hand a shift to themselves"):
        _create(service, to_lead=FROM)


def test_acceptance_belongs_to_the_addressed_lead():
    service = build_world(fresh_service())
    handoff = _create(service)
    with pytest.raises(HandoffError, match="self-acceptance"):
        service.accept_handoff(handoff.handoff_id, actor=FROM, now=NOW)
    with pytest.raises(HandoffError, match="is addressed to"):
        service.accept_handoff(handoff.handoff_id, actor="P. Bystander", now=NOW)
    with pytest.raises(HandoffError, match="actor is required"):
        service.accept_handoff(handoff.handoff_id, actor=" ", now=NOW)
    accepted = service.accept_handoff(handoff.handoff_id, actor=TO, now=NOW)
    assert accepted.status == "accepted"
    assert accepted.acceptance == "accepted"
    assert accepted.exceptions == ""
    assert accepted.accepted_by == TO
    assert accepted.accepted_at == format_timestamp(NOW)
    assert service.outstanding_handoff() is None
    # An acceptance is testimony: it cannot be signed twice.
    with pytest.raises(HandoffConflict, match="already accepted"):
        service.accept_handoff(handoff.handoff_id, actor=TO, now=NOW)


def test_acceptance_with_exceptions_edits_no_figure():
    service = build_world(fresh_service())
    handoff = _create(service, note="vendor outage; cluster looks real")
    before = handoff.model_dump()
    accepted = service.accept_handoff(
        handoff.handoff_id,
        actor=TO,
        exceptions="the cluster is two coincidences, not one event",
        now=NOW,
    )
    assert accepted.acceptance == "accepted_with_exceptions"
    assert accepted.exceptions == "the cluster is two coincidences, not one event"
    # The dispute rides beside the figures; the figures are the
    # composed ones, untouched, and so is everything else the
    # author wrote.
    assert accepted.sections == before["sections"]
    assert accepted.note == before["note"]
    assert accepted.from_lead == before["from_lead"]


def test_a_newer_handoff_supersedes_an_unaccepted_one():
    service = build_world(fresh_service())
    first = _create(service)
    second = _create(service, note="updated page")
    stored_first = service.get_handoff(first.handoff_id)
    assert stored_first["status"] == "superseded"
    assert stored_first["outstanding"] is False
    outstanding = service.outstanding_handoff()
    assert outstanding["handoff_id"] == second.handoff_id
    # At most one page ever waits on its handshake.
    assert sum(1 for v in service.list_handoffs() if v["outstanding"]) == 1
    # The lapsed page can no longer be accepted.
    with pytest.raises(HandoffConflict, match="superseded"):
        service.accept_handoff(first.handoff_id, actor=TO, now=NOW)
    # The accepted page is history: a third handoff supersedes
    # only the outstanding one, never the settled record.
    service.accept_handoff(second.handoff_id, actor=TO, now=NOW)
    third = _create(service)
    assert service.get_handoff(second.handoff_id)["status"] == "accepted"
    assert service.outstanding_handoff()["handoff_id"] == third.handoff_id


def test_unknown_handoff_is_a_key_error():
    service = build_world(fresh_service())
    with pytest.raises(KeyError):
        service.accept_handoff("HOF-9999", actor=TO, now=NOW)
    with pytest.raises(KeyError):
        service.get_handoff("HOF-9999")


def test_records_are_byte_identical_across_the_lifecycle():
    service = build_world(fresh_service())
    before = _snapshot(service)
    first = _create(service, note="page one")
    assert _snapshot(service) == before
    service.accept_handoff(
        first.handoff_id, actor=TO, exceptions="one doubt", now=NOW
    )
    assert _snapshot(service) == before
    second = _create(service)
    _create(service)  # supersedes the second
    assert service.get_handoff(second.handoff_id)["status"] == "superseded"
    assert _snapshot(service) == before


def test_the_note_is_the_authors_own_and_rides_the_record():
    service = build_world(fresh_service())
    handoff = _create(service, note="  watch the checkout cluster  ")
    assert handoff.note == "watch the checkout cluster"
    assert handoff.status == "outstanding"
    assert handoff.handoff_id == "HOF-0001"
    assert handoff.as_of == format_timestamp(NOW)


# ---------------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------------


def _client(service):
    api.set_service(service)
    return TestClient(api.app)


def test_api_composes_lists_and_accepts():
    service = build_world(fresh_service())
    try:
        with _client(service) as client:
            created = client.post(
                "/api/handoffs",
                json={"from_lead": FROM, "to_lead": TO, "note": "page"},
            )
            assert created.status_code == 201
            body = created.json()
            assert body["handoff_id"] == "HOF-0001"
            assert body["status"] == "outstanding"
            assert body["sections"]["qa_review"]["backlog"] == 1

            listed = client.get("/api/handoffs")
            assert listed.status_code == 200
            payload = listed.json()
            assert payload["outstanding"]["handoff_id"] == "HOF-0001"
            assert [h["handoff_id"] for h in payload["handoffs"]] == ["HOF-0001"]
            assert payload["handoffs"][0]["outstanding"] is True

            refused = client.post(
                "/api/handoffs/HOF-0001/accept", json={"actor": FROM}
            )
            assert refused.status_code == 422
            assert "self-acceptance" in refused.json()["detail"]

            accepted = client.post(
                "/api/handoffs/HOF-0001/accept",
                json={"actor": TO, "exceptions": "top row disputed"},
            )
            assert accepted.status_code == 200
            assert accepted.json()["acceptance"] == "accepted_with_exceptions"
            assert accepted.json()["sections"] == body["sections"]

            settled = client.post(
                "/api/handoffs/HOF-0001/accept", json={"actor": TO}
            )
            assert settled.status_code == 409

            one = client.get("/api/handoffs/HOF-0001")
            assert one.status_code == 200
            assert one.json()["outstanding"] is False
            assert client.get("/api/handoffs/HOF-9999").status_code == 404
            assert (
                client.post(
                    "/api/handoffs/HOF-9999/accept", json={"actor": TO}
                ).status_code
                == 404
            )
    finally:
        api.set_service(None)


def test_api_refuses_an_unnamed_or_single_lead_handoff():
    service = build_world(fresh_service())
    try:
        with _client(service) as client:
            missing = client.post(
                "/api/handoffs", json={"from_lead": "", "to_lead": TO}
            )
            assert missing.status_code == 422
            assert "from_lead is required" in missing.json()["detail"]
            same = client.post(
                "/api/handoffs", json={"from_lead": FROM, "to_lead": FROM}
            )
            assert same.status_code == 422
            assert "cannot hand a shift to themselves" in same.json()["detail"]
            bad_as_of = client.post(
                "/api/handoffs",
                json={"from_lead": FROM, "to_lead": TO, "as_of": "not-a-time"},
            )
            assert bad_as_of.status_code == 422
    finally:
        api.set_service(None)


def test_console_serves_the_handoff_tab():
    service = build_world(fresh_service())
    try:
        with _client(service) as client:
            html = client.get("/").text
            assert 'id="tab-handoff"' in html
            assert 'id="view-handoff"' in html
            assert "Shift handoff" in html
    finally:
        api.set_service(None)
