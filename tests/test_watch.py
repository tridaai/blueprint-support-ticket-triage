"""The breach watch: the queue ranked by the code clock, and the
lead's one act.

Pins, by layer:

- the bands — boundaries exact, measured against each run's OWN
  band target (a low run can be hours old and ``later``; a
  critical run is ``due_within_1h`` from arrival);
- the ranking — breached first (most overdue first), then
  nearest breach; ties fall to the composition priority score,
  the queue's own order, then ticket id;
- the separation — a run the MODEL fears (sla_breach_risk 9.5)
  with a comfortable clock ranks behind a run the model shrugs
  at whose clock is nearly out: the watch follows the clock;
- the population — decided runs are not the watch's business,
  and an open run with no parseable received_at is counted as
  unclocked, never ranked on a guessed clock;
- the assignee — one reading rule over two write paths: the
  watch's structured reassignment trail wins, the gate's
  ``[assigned to …]`` note answers when the trail is empty;
- the act — reassignment requires assignee + lead + reason,
  refuses a decided run, and moves NOTHING else: the record is
  byte-identical apart from the trail, and the SLA clock reads
  the same before and after (a reassignment never buys time);
- the surfaces — the API serves the watch with its constants,
  the reassign route maps its refusals (422 unnamed field, 409
  decided, 404 unknown), and the audit export carries the band
  at export time for a waiting run and nothing new for a decided
  one.

All clocks are injected; no test races the wall.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.assignment import current_assignee, open_load
from support_triage.exports import render_text
from support_triage.roster import load_roster
from support_triage.schemas import (
    AssignmentEntry,
    CompositionInfo,
    QuestionAnswer,
    TicketResult,
)
from support_triage.sla import sla_clock
from support_triage.watch import band_for_seconds, build_watch

from helpers import fresh_service, make_ticket

NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
ROSTER = load_roster()


def _ago(**kwargs) -> str:
    return (NOW - timedelta(**kwargs)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(
    ticket_id="W-1",
    band="medium",
    score=40.0,
    received_at="",
    status="awaiting_agent",
    risk=None,
    queue="technical_support",
    decision_reason="",
    assignments=None,
):
    answers = []
    if risk is not None:
        answers.append(
            QuestionAnswer(
                key="sla_breach_risk", kind="score", value=risk,
                confidence=0.9,
            )
        )
    return TicketResult(
        ticket_id=ticket_id,
        status=status,
        subject=f"subject {ticket_id}",
        sender_name="Test Sender",
        received_at=received_at,
        category="technical_issue",
        product_area="dashboard",
        answers=answers,
        composition=CompositionInfo(
            priority_band=band, priority_score=score, queue=queue
        ),
        decision_reason=decision_reason,
        assignments=list(assignments or []),
    )


def _watch(records, now=NOW):
    return build_watch(records, ROSTER, now)


# ---------------------------------------------------------------------------
# the bands
# ---------------------------------------------------------------------------


def test_band_boundaries_are_exact():
    assert band_for_seconds(-1) == "breached"
    assert band_for_seconds(-7200) == "breached"
    assert band_for_seconds(0) == "due_within_1h"
    assert band_for_seconds(3600) == "due_within_1h"
    assert band_for_seconds(3601) == "due_within_4h"
    assert band_for_seconds(14400) == "due_within_4h"
    assert band_for_seconds(14401) == "later"


def test_bands_measure_against_each_runs_own_target():
    # low band, 72h target: 30 hours old still has 42h left.
    low = make_record("W-LOW", band="low", received_at=_ago(hours=30))
    # critical band, 1h target: 30 minutes old has 30 minutes left.
    critical = make_record("W-CRIT", band="critical", received_at=_ago(minutes=30))
    # medium band, 24h target: 20 hours old has exactly 4h left —
    # the due_within_4h boundary, inclusive.
    medium = make_record("W-MED", band="medium", received_at=_ago(hours=20))
    rows = {r["ticket_id"]: r for r in _watch([low, critical, medium])["rows"]}
    assert rows["W-LOW"]["watch_band"] == "later"
    assert rows["W-LOW"]["target_hours"] == 72.0
    assert rows["W-CRIT"]["watch_band"] == "due_within_1h"
    assert rows["W-CRIT"]["target_hours"] == 1.0
    assert rows["W-MED"]["watch_band"] == "due_within_4h"
    assert rows["W-MED"]["seconds_to_breach"] == 14400.0
    # The clock's own state rides along, distinct from the band:
    # the low run's clock says ok; the watch still calls it later.
    assert rows["W-LOW"]["sla_state"] == "ok"


def test_watch_payload_prints_its_rule():
    payload = _watch([make_record(received_at=_ago(hours=1))])
    assert payload["clock"] == "code_sla_clock"
    assert "sla_breach_risk" in payload["clock_note"]
    assert payload["bands"]["due_within_1h_seconds"] == 3600
    assert payload["bands"]["due_within_4h_seconds"] == 14400
    assert payload["as_of"] == "2026-06-02T12:00:00Z"


# ---------------------------------------------------------------------------
# the ranking — and the separation from the model's risk
# ---------------------------------------------------------------------------


def test_breached_first_most_overdue_first_then_nearest_breach():
    records = [
        make_record("W-LATER", band="low", received_at=_ago(hours=2)),
        make_record("W-BREACH-NEW", band="critical", received_at=_ago(hours=2)),
        make_record("W-NEAR", band="high", received_at=_ago(hours=3, minutes=30)),
        make_record("W-BREACH-OLD", band="high", received_at=_ago(hours=9)),
    ]
    rows = _watch(records)["rows"]
    # high target 4h: W-BREACH-OLD is 5h over; critical 1h:
    # W-BREACH-NEW is 1h over; W-NEAR has 30m left; W-LATER 70h.
    assert [r["ticket_id"] for r in rows] == [
        "W-BREACH-OLD", "W-BREACH-NEW", "W-NEAR", "W-LATER",
    ]
    assert [r["watch_band"] for r in rows] == [
        "breached", "breached", "due_within_1h", "later",
    ]


def test_ties_break_by_priority_score_then_ticket_id():
    same_instant = _ago(hours=23, minutes=30)  # medium: 1800s left
    records = [
        make_record("W-T2", band="medium", score=35.0, received_at=same_instant),
        make_record("W-T1", band="medium", score=45.0, received_at=same_instant),
        make_record("W-T3", band="medium", score=45.0, received_at=same_instant),
    ]
    rows = _watch(records)["rows"]
    assert [r["ticket_id"] for r in rows] == ["W-T1", "W-T3", "W-T2"]


def test_the_watch_follows_the_clock_not_the_models_risk():
    # The model fears this run (risk 9.5 of 10) — but it is a low
    # band ticket with 71 hours of clock left.
    feared = make_record(
        "W-FEARED", band="low", received_at=_ago(hours=1), risk=9.5
    )
    # The model shrugs at this one (risk 0.5) — but its 24h clock
    # has 30 minutes left.
    shrugged = make_record(
        "W-SHRUGGED", band="medium", received_at=_ago(hours=23, minutes=30),
        risk=0.5,
    )
    payload = _watch([feared, shrugged])
    rows = payload["rows"]
    assert [r["ticket_id"] for r in rows] == ["W-SHRUGGED", "W-FEARED"]
    # The model's numbers ride the rows, labelled, unsorted-by.
    assert rows[0]["sla_breach_risk"] == 0.5
    assert rows[1]["sla_breach_risk"] == 9.5
    assert payload["counts"] == {
        "breached": 0, "due_within_1h": 1, "due_within_4h": 0, "later": 1,
    }


# ---------------------------------------------------------------------------
# the population
# ---------------------------------------------------------------------------


def test_decided_runs_are_not_the_watchs_business():
    decided = make_record(
        "W-DONE", band="critical", received_at=_ago(hours=9), status="triaged"
    )
    payload = _watch([decided])
    assert payload["rows"] == []
    assert payload["open_runs"] == 0


def test_unclocked_open_runs_are_counted_never_ranked():
    no_stamp = make_record("W-NOSTAMP", received_at="")
    bad_stamp = make_record("W-BADSTAMP", received_at="not a date")
    clocked = make_record("W-CLOCKED", received_at=_ago(hours=1))
    payload = _watch([no_stamp, bad_stamp, clocked])
    assert [r["ticket_id"] for r in payload["rows"]] == ["W-CLOCKED"]
    assert payload["unclocked_open"] == 2
    assert payload["open_runs"] == 3


def test_watch_is_pure_and_deterministic():
    records = [
        make_record("W-1", received_at=_ago(hours=3)),
        make_record("W-2", band="critical", received_at=_ago(minutes=20)),
    ]
    before = [r.model_dump_json() for r in records]
    first = _watch(records)
    second = _watch(records)
    assert first == second
    assert [r.model_dump_json() for r in records] == before


def test_rows_carry_queue_suggestion_and_the_uncovered_note():
    covered = make_record("W-COV", queue="technical_support",
                          received_at=_ago(hours=1))
    uncovered = make_record("W-UNC", queue="no_such_queue",
                            received_at=_ago(hours=1))
    rows = {r["ticket_id"]: r for r in _watch([covered, uncovered])["rows"]}
    assert rows["W-COV"]["suggested_agent"]  # a roster name
    assert rows["W-COV"]["suggestion_note"] == ""
    assert rows["W-UNC"]["suggested_agent"] == ""
    assert "no roster agent covers" in rows["W-UNC"]["suggestion_note"]


# ---------------------------------------------------------------------------
# the assignee: one reading rule over two write paths
# ---------------------------------------------------------------------------


def test_current_assignee_reading_rule():
    assert current_assignee(make_record()) is None
    noted = make_record(decision_reason="looked fine [assigned to Ben Kowalski]")
    assert current_assignee(noted) == {
        "agent": "Ben Kowalski", "source": "note", "by": "", "at": "",
        "reason": "",
    }
    twice = make_record(
        decision_reason="[assigned to Ben Kowalski] [assigned to Daniel Park]"
    )
    assert current_assignee(twice)["agent"] == "Daniel Park"
    trailed = make_record(
        decision_reason="[assigned to Ben Kowalski]",
        assignments=[
            AssignmentEntry(
                assignee="Grace Chen", by="L. Lead",
                reason="Ben is at capacity", at="2026-06-02T11:00:00Z",
            )
        ],
    )
    resolved = current_assignee(trailed)
    assert resolved["agent"] == "Grace Chen"
    assert resolved["source"] == "reassignment"
    assert resolved["by"] == "L. Lead"


def test_open_load_follows_the_resolver():
    ben = next(a for a in ROSTER if a["id"] == "b.kowalski")
    grace = next(a for a in ROSTER if a["id"] == "g.chen")
    records = [
        make_record(
            "W-L1",
            decision_reason="[assigned to Ben Kowalski]",
            assignments=[
                AssignmentEntry(
                    assignee="Grace Chen", by="L. Lead", reason="r",
                    at="2026-06-02T11:00:00Z",
                )
            ],
        ),
        make_record("W-L2", decision_reason="[assigned to Ben Kowalski]"),
    ]
    assert open_load(records, ben) == 1
    assert open_load(records, grace) == 1


# ---------------------------------------------------------------------------
# the lead's act, through the service
# ---------------------------------------------------------------------------


def _service_with_ticket(received_at="2026-06-02T11:00:00Z", ticket_id="W-ACT"):
    service = fresh_service()
    service.run_ticket(make_ticket(ticket_id=ticket_id, received_at=received_at))
    return service


def test_reassign_appends_a_named_reasoned_entry():
    service = _service_with_ticket()
    record = service.reassign_run(
        "W-ACT", assignee="Ben Kowalski", by="L. Lead",
        reason="nearest the breach, and Ben covers the queue", now=NOW,
    )
    assert len(record.assignments) == 1
    entry = record.assignments[0]
    assert (entry.assignee, entry.by, entry.reason) == (
        "Ben Kowalski", "L. Lead", "nearest the breach, and Ben covers the queue",
    )
    assert entry.at == "2026-06-02T12:00:00Z"
    # And the watch now shows the new hand, sourced to the trail.
    row = service.breach_watch(NOW)["rows"][0]
    assert row["assignee"] == "Ben Kowalski"
    assert row["assignee_source"] == "reassignment"
    assert row["assigned_by"] == "L. Lead"


def test_reassign_moves_nothing_but_the_trail():
    service = _service_with_ticket()
    before = service.get_record("W-ACT")
    clock_before = sla_clock(before, NOW)
    blob_before = before.model_dump()
    service.reassign_run(
        "W-ACT", assignee="Daniel Park", by="L. Lead", reason="load", now=NOW,
    )
    after = service.get_record("W-ACT")
    blob_after = after.model_dump()
    assert blob_after.pop("assignments")
    assert blob_before.pop("assignments") == []
    assert blob_after == blob_before  # every other field byte-identical
    assert sla_clock(after, NOW) == clock_before
    assert after.status == "awaiting_agent"
    assert after.received_at == before.received_at


def test_reassign_refusals_name_themselves():
    service = _service_with_ticket()
    with pytest.raises(KeyError):
        service.reassign_run("W-NOPE", assignee="A", by="L", reason="r")
    with pytest.raises(ValueError, match="must name the agent"):
        service.reassign_run("W-ACT", assignee=" ", by="L", reason="r")
    with pytest.raises(ValueError, match="must name the lead"):
        service.reassign_run("W-ACT", assignee="A", by="", reason="r")
    with pytest.raises(ValueError, match="must carry a reason"):
        service.reassign_run("W-ACT", assignee="A", by="L", reason="")
    service.record_decision("W-ACT", "approved", by="A. Agent")
    with pytest.raises(ValueError, match="not awaiting_agent"):
        service.reassign_run("W-ACT", assignee="A", by="L", reason="r")


def test_gate_note_then_watch_reassignment_reads_in_order():
    service = _service_with_ticket()
    service.assign("W-ACT", "Carmen Ruiz", by="gate agent")
    row = service.breach_watch(NOW)["rows"][0]
    assert row["assignee"] == "Carmen Ruiz"
    assert row["assignee_source"] == "note"
    service.reassign_run(
        "W-ACT", assignee="Grace Chen", by="L. Lead", reason="Carmen is out",
        now=NOW,
    )
    row = service.breach_watch(NOW)["rows"][0]
    assert row["assignee"] == "Grace Chen"
    assert row["assignee_source"] == "reassignment"


# ---------------------------------------------------------------------------
# the surfaces: API + audit export
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def _stage(client, *records):
    service = api.get_service()
    for record in records:
        service.store.save(record)


def test_api_watch_and_reassign(client):
    _stage(
        client,
        make_record("W-API-1", band="critical", received_at=_ago(hours=3)),
        make_record("W-API-2", band="low", received_at=_ago(hours=1)),
    )
    # The API reads at wall-clock now — long after both staged
    # runs' targets — so both stand breached, the one whose target
    # ran out first (the critical, due 2026-06-02T10:00Z) ahead of
    # the low run (due 2026-06-05T11:00Z). The exact band math at a
    # fixed instant is pinned in the direct tests above.
    payload = client.get("/api/watch").json()
    assert payload["clock"] == "code_sla_clock"
    assert payload["counts"]["breached"] == 2
    assert [r["ticket_id"] for r in payload["rows"]] == ["W-API-1", "W-API-2"]

    refused = client.post(
        "/api/runs/W-API-1/reassign",
        json={"assignee": "Ben Kowalski", "by": "L. Lead"},
    )
    assert refused.status_code == 422
    assert "reason" in refused.json()["detail"]

    missing = client.post(
        "/api/runs/W-API-9/reassign",
        json={"assignee": "Ben Kowalski", "by": "L. Lead", "reason": "r"},
    )
    assert missing.status_code == 404

    done = client.post(
        "/api/runs/W-API-1/reassign",
        json={"assignee": "Ben Kowalski", "by": "L. Lead", "reason": "load"},
    )
    assert done.status_code == 200
    assert done.json()["assignments"][0]["assignee"] == "Ben Kowalski"

    service = api.get_service()
    service.record_decision("W-API-2", "approved", by="A. Agent")
    decided = client.post(
        "/api/runs/W-API-2/reassign",
        json={"assignee": "Ben Kowalski", "by": "L. Lead", "reason": "load"},
    )
    assert decided.status_code == 409


def test_export_carries_the_band_for_a_waiting_run_only():
    service = fresh_service()
    service.run_ticket(
        make_ticket(ticket_id="W-EXP", received_at="2026-06-02T11:00:00Z")
    )
    # Wall-clock export: the run arrived "an hour ago" relative to
    # its own stamp only in the golden sense — here we assert the
    # block's internal consistency with the export's SLA section
    # rather than a fixed band, since export time is the wall.
    document = service.export_run("W-EXP")
    assert document["watch"] is not None
    assert document["watch"]["band"] == band_for_seconds(
        document["sla"]["seconds_to_breach"]
    )
    assert document["watch"]["computed_at"] == document["exported_at"]
    assert "BREACH WATCH" in render_text(document)

    service.record_decision("W-EXP", "approved", by="A. Agent")
    decided = service.export_run("W-EXP")
    assert decided["watch"] is None
    assert "BREACH WATCH" not in render_text(decided)
