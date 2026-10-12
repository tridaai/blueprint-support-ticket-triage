"""SLA attainment (attainment.py): the period report, judged honestly.

The staged world's arithmetic is hand-derived in the comments;
the report must reproduce it exactly. The world also pins the
round's honesty boundaries: decisions predate the decided_at
stamp and belong to no period (counted on the face), a decided
run with no clock start is unjudgeable (counted, never met,
never in the percentage), and still-open runs are a snapshot —
never silent successes.
"""

from __future__ import annotations

from datetime import timezone

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.attainment import build_attainment
from support_triage.schemas import CompositionInfo, TicketResult
from support_triage.sla import parse_timestamp
from support_triage.watch import build_watch

from helpers import fresh_service, make_ticket

START = parse_timestamp("2026-06-01T00:00:00Z")
END = parse_timestamp("2026-07-01T00:00:00Z")


def staged(ticket_id, queue, band, *, status="triaged", received="", decided=""):
    return TicketResult(
        ticket_id=ticket_id,
        status=status,
        received_at=received,
        decision="approved" if status == "triaged" else "",
        decided_at=decided,
        composition=CompositionInfo(queue=queue, priority_band=band),
    )


def world():
    """The staged month of June 2026 (targets: critical 1h,
    high 4h, medium 24h, low 72h)."""
    return [
        # billing / critical: A1 arrived before the period and was
        # decided 0.75h after arrival, inside it — the decision is
        # the population key, so it counts, met. A2: 2.5h — breached.
        staged("A1", "billing_support", "critical",
               received="2026-05-31T23:30:00Z", decided="2026-06-01T00:15:00Z"),
        staged("A2", "billing_support", "critical",
               received="2026-06-02T10:00:00Z", decided="2026-06-02T12:30:00Z"),
        # billing / low: 24h met, 120h breached.
        staged("B1", "billing_support", "low",
               received="2026-06-05T00:00:00Z", decided="2026-06-06T00:00:00Z"),
        staged("B2", "billing_support", "low",
               received="2026-06-05T00:00:00Z", decided="2026-06-10T00:00:00Z"),
        # incident / high: 2h — met.
        staged("C1", "incident_response", "high",
               received="2026-06-03T08:00:00Z", decided="2026-06-03T10:00:00Z"),
        # incident / high, decided but never stamped (the pre-stamp
        # era): belongs to no period, counted on the face.
        staged("C2", "incident_response", "high",
               received="2026-06-03T08:00:00Z", decided=""),
        # incident / medium, decided in period with no clock start:
        # unjudgeable.
        staged("D1", "incident_response", "medium",
               received="", decided="2026-06-04T09:00:00Z"),
        # billing / high, decided in May: invisible to June.
        staged("E1", "billing_support", "high",
               received="2026-05-19T00:00:00Z", decided="2026-05-20T00:00:00Z"),
        # incident / critical, decided in July: not June's decided
        # population, but open at June's end (age at end 134h vs a
        # 1h target — breached in the snapshot).
        staged("F1", "incident_response", "critical",
               received="2026-06-25T10:00:00Z", decided="2026-07-03T10:00:00Z"),
        # still open: G1 medium arrived Jun 15 (age at end 372h —
        # breached), G2 low arrived Jun 30 noon (age 12h — ok).
        staged("G1", "billing_support", "medium", status="awaiting_agent",
               received="2026-06-15T12:00:00Z"),
        staged("G2", "billing_support", "low", status="awaiting_agent",
               received="2026-06-30T12:00:00Z"),
        # arrived after the period ended: not open at June's end.
        staged("H1", "general_support", "low", status="awaiting_agent",
               received="2026-07-02T00:00:00Z"),
        # open with no parseable arrival: unclocked, counted not placed.
        staged("U1", "technical_support", "high", status="awaiting_agent",
               received="not-a-date"),
    ]


def queue_block(report, queue):
    return next(block for block in report["queues"] if block["queue"] == queue)


def test_hand_derived_cells():
    report = build_attainment(world(), START, END)

    billing = queue_block(report, "billing_support")
    critical = billing["bands"]["critical"]
    assert critical == {
        "decided": 2, "judged": 2, "met": 1, "breached": 1, "unjudgeable": 0,
        "attainment_pct": 50.0, "median_seconds": 5850.0, "worst_seconds": 9000.0,
    }
    low = billing["bands"]["low"]
    assert low["decided"] == 2 and low["met"] == 1 and low["breached"] == 1
    assert low["attainment_pct"] == 50.0
    assert low["median_seconds"] == 259200.0 and low["worst_seconds"] == 432000.0
    # E1 was decided in May: billing/high is an empty cell, and an
    # empty cell reports no percentage — never 100%.
    high = billing["bands"]["high"]
    assert high["decided"] == 0 and high["attainment_pct"] is None
    assert high["median_seconds"] is None and high["worst_seconds"] is None
    totals = billing["totals"]
    assert totals["decided"] == 4 and totals["met"] == 2 and totals["breached"] == 2
    assert totals["attainment_pct"] == 50.0
    assert totals["median_seconds"] == 47700.0 and totals["worst_seconds"] == 432000.0

    incident = queue_block(report, "incident_response")
    assert incident["bands"]["high"]["met"] == 1
    assert incident["bands"]["high"]["attainment_pct"] == 100.0
    medium = incident["bands"]["medium"]
    assert medium["decided"] == 1 and medium["judged"] == 0
    assert medium["unjudgeable"] == 1 and medium["attainment_pct"] is None
    # The unjudgeable run counts in the queue's decided but stays
    # out of its percentage's denominator.
    assert incident["totals"]["decided"] == 2 and incident["totals"]["judged"] == 1
    assert incident["totals"]["attainment_pct"] == 100.0

    grand = report["totals"]
    assert grand["decided"] == 6 and grand["judged"] == 5
    assert grand["met"] == 3 and grand["breached"] == 2
    assert grand["unjudgeable"] == 1 and grand["attainment_pct"] == 60.0
    assert grand["median_seconds"] == 9000.0 and grand["worst_seconds"] == 432000.0


def test_boundaries_are_named_not_filled():
    report = build_attainment(world(), START, END)
    # The pre-stamp decision (C2) is counted on the face and
    # appears in no cell: incident/high shows only C1.
    assert report["boundary"]["decided_without_stamp"] == 1
    assert queue_block(report, "incident_response")["bands"]["high"]["decided"] == 1
    # The unjudgeable decision is named, with its reason.
    assert [u["ticket_id"] for u in report["unjudgeable"]] == ["D1"]
    assert "received_at" in report["unjudgeable"][0]["reason"]
    # The face states its rules.
    assert report["period"]["rule"].startswith("[start, end)")
    assert "decided_at" in report["population_rule"]
    assert report["targets_hours"] == {
        "critical": 1.0, "high": 4.0, "medium": 24.0, "low": 72.0
    }


def test_open_at_end_snapshot():
    report = build_attainment(world(), START, END)
    rows = {(r["queue"], r["band"]): r for r in report["open_at_end"]["rows"]}
    # F1 was decided in July, so at June's end it stood open —
    # and long past its 1h target.
    f1 = rows[("incident_response", "critical")]
    assert f1["open"] == 1 and f1["ticket_ids"] == ["F1"]
    assert f1["states"] == {"ok": 0, "due_soon": 0, "breached": 1}
    g1 = rows[("billing_support", "medium")]
    assert g1["ticket_ids"] == ["G1"] and g1["states"]["breached"] == 1
    g2 = rows[("billing_support", "low")]
    assert g2["ticket_ids"] == ["G2"] and g2["states"]["ok"] == 1
    # H1 arrived after the period ended: nowhere in the snapshot,
    # and its queue earns no block from it.
    assert ("general_support", "low") not in rows
    assert all(b["queue"] != "general_support" for b in report["queues"])
    assert report["open_at_end"]["total"] == 3
    assert report["open_at_end"]["unclocked_open_at_end"] == 1
    # Snapshot runs are not in the decided population.
    assert report["totals"]["decided"] == 6


def test_open_snapshot_agrees_with_the_breach_watch_at_period_end():
    """The composition pin: read at the period end, the snapshot
    and the watch are the same population — same runs, same bands.
    Staged with F1 still awaiting (its decision lands later)."""
    records = [r for r in world() if r.ticket_id != "H1"]  # H1 arrives after the end
    for record in records:
        if record.ticket_id == "F1":
            record.status = "awaiting_agent"
            record.decided_at = ""
    report = build_attainment(records, START, END)
    watch = build_watch(records, [], END)

    snapshot_ids = {}
    for row in report["open_at_end"]["rows"]:
        snapshot_ids.setdefault(row["band"], set()).update(row["ticket_ids"])
    watch_ids = {}
    for row in watch["rows"]:
        watch_ids.setdefault(row["priority_band"], set()).add(row["ticket_id"])
    assert snapshot_ids == watch_ids
    assert report["open_at_end"]["unclocked_open_at_end"] == watch["unclocked_open"]
    # And the snapshot's clock states are the watch rows' states
    # (each staged row holds one run, so its state is unambiguous).
    snapshot_states = {}
    for row in report["open_at_end"]["rows"]:
        (state,) = [s for s, n in row["states"].items() if n]
        for ticket_id in row["ticket_ids"]:
            snapshot_states[ticket_id] = state
    for row in watch["rows"]:
        assert snapshot_states[row["ticket_id"]] == row["sla_state"]


def test_period_bounds_are_half_open():
    records = [
        staged("AT-START", "billing_support", "low",
               received="2026-05-31T00:00:00Z", decided="2026-06-01T00:00:00Z"),
        staged("AT-END", "billing_support", "low",
               received="2026-06-30T00:00:00Z", decided="2026-07-01T00:00:00Z"),
    ]
    june = build_attainment(records, START, END)
    assert queue_block(june, "billing_support")["bands"]["low"]["decided"] == 1
    july = build_attainment(
        records, END, parse_timestamp("2026-08-01T00:00:00Z")
    )
    assert queue_block(july, "billing_support")["bands"]["low"]["decided"] == 1


def test_decision_at_exact_target_is_met_not_breached():
    # The clock reads breached only once the target has run out
    # (remaining < 0); a decision landing exactly on it is met.
    records = [
        staged("EDGE", "billing_support", "critical",
               received="2026-06-02T10:00:00Z", decided="2026-06-02T11:00:00Z"),
    ]
    cell = queue_block(build_attainment(records, START, END), "billing_support")
    assert cell["bands"]["critical"]["met"] == 1
    assert cell["bands"]["critical"]["breached"] == 0


def test_empty_period_reports_emptiness():
    report = build_attainment([], START, END)
    assert report["queues"] == []
    assert report["totals"]["decided"] == 0
    assert report["totals"]["attainment_pct"] is None
    assert report["open_at_end"]["total"] == 0


# ---------------------------------------------------------------------------
# the stamp: decisions record their moment from this round on
# ---------------------------------------------------------------------------


def test_record_decision_stamps_the_moment_on_the_resume_path():
    service = fresh_service()
    service.run_ticket(make_ticket())
    moment = parse_timestamp("2026-06-02T12:00:00Z")
    record = service.record_decision("T-1", "approved", by="agent-1", now=moment)
    assert record.decided_at == "2026-06-02T12:00:00Z"
    assert service.get_record("T-1").decided_at == "2026-06-02T12:00:00Z"


def test_record_decision_stamps_the_moment_on_the_direct_path():
    # A stored record with no live checkpoint (the restart case):
    # the service applies the decision itself and stamps it too.
    service = fresh_service()
    service.store.save(
        staged("D-1", "billing_support", "low", status="awaiting_agent",
               received="2026-06-02T10:00:00Z")
    )
    moment = parse_timestamp("2026-06-02T12:00:00Z")
    record = service.record_decision("D-1", "rejected", by="agent-2", now=moment)
    assert record.status == "rejected"
    assert record.decided_at == "2026-06-02T12:00:00Z"


def test_record_decision_stamps_wall_clock_by_default():
    service = fresh_service()
    service.run_ticket(make_ticket())
    record = service.record_decision("T-1", "approved", by="agent-1")
    stamped = parse_timestamp(record.decided_at)
    assert stamped is not None and stamped.tzinfo is not None


def test_stamped_decisions_flow_into_the_report():
    service = fresh_service()
    service.run_ticket(make_ticket(received_at="2026-06-02T10:00:00Z"))
    service.record_decision(
        "T-1", "approved", by="agent-1",
        now=parse_timestamp("2026-06-02T10:30:00Z"),
    )
    report = service.sla_attainment(START, END)
    billing = queue_block(report, "billing_support")
    judged = sum(b["judged"] for b in billing["bands"].values())
    assert judged == 1
    assert report["boundary"]["decided_without_stamp"] == 0


# ---------------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def test_attainment_endpoint(client):
    response = client.get(
        "/api/attainment",
        params={"start": "2026-06-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["period"]["start"] == "2026-06-01T00:00:00Z"
    assert body["totals"]["decided"] == 0


def test_attainment_endpoint_refuses_bad_periods(client):
    assert client.get("/api/attainment").status_code == 422
    bad_start = client.get(
        "/api/attainment",
        params={"start": "june", "end": "2026-07-01T00:00:00Z"},
    )
    assert bad_start.status_code == 422 and "start" in bad_start.json()["detail"]
    backwards = client.get(
        "/api/attainment",
        params={"start": "2026-07-01T00:00:00Z", "end": "2026-06-01T00:00:00Z"},
    )
    assert backwards.status_code == 422
    assert "end must be after start" in backwards.json()["detail"]
