"""SLA clocks: deterministic arithmetic, injected clocks, no wall races.

The clock is the model's opposite number: band target minus age,
nothing probabilistic. These tests pin the state boundaries
(ok / due_soon / breached) with hand-computed numbers, and check the
clock's fields ride the queue rows and the overview stats.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from support_triage.composition import policy_snapshot
from support_triage.schemas import CompositionInfo, TicketResult
from support_triage.sla import parse_timestamp, sla_clock
from support_triage.stats import overview_stats
from support_triage.store import InMemoryStore

NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


def _ago(**kwargs) -> str:
    return (NOW - timedelta(**kwargs)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(band="critical", received_at="", status="awaiting_agent", ticket_id="SLA-1"):
    return TicketResult(
        ticket_id=ticket_id,
        status=status,
        received_at=received_at,
        composition=CompositionInfo(priority_band=band, queue="technical_support"),
    )


def test_clock_ok_with_most_of_the_target_remaining():
    # critical target is 1h; 30 minutes in, half the clock remains.
    clock = sla_clock(make_record(received_at=_ago(minutes=30)), NOW)
    assert clock["state"] == "ok"
    assert clock["target_hours"] == 1.0
    assert clock["age_seconds"] == 1800.0
    assert clock["seconds_to_breach"] == 1800.0


def test_clock_due_soon_in_the_last_quarter():
    # 50 minutes into a 1h target: 600s left, inside the last 25%.
    clock = sla_clock(make_record(received_at=_ago(minutes=50)), NOW)
    assert clock["state"] == "due_soon"
    assert clock["seconds_to_breach"] == 600.0


def test_clock_due_soon_boundary_is_inclusive():
    # Exactly 25% remaining (45 minutes into 1h) is due_soon, not ok.
    clock = sla_clock(make_record(received_at=_ago(minutes=45)), NOW)
    assert clock["state"] == "due_soon"


def test_clock_breached_once_the_target_runs_out():
    clock = sla_clock(make_record(received_at=_ago(hours=2)), NOW)
    assert clock["state"] == "breached"
    assert clock["seconds_to_breach"] == -3600.0


def test_clock_scales_with_the_band_target():
    # low target is 72h; 60h in leaves 12h — the last quarter (18h).
    clock = sla_clock(make_record(band="low", received_at=_ago(hours=60)), NOW)
    assert clock["state"] == "due_soon"
    assert clock["target_hours"] == 72.0
    # high target is 4h; 1h in is comfortably ok.
    clock = sla_clock(make_record(band="high", received_at=_ago(hours=1)), NOW)
    assert clock["state"] == "ok"


def test_no_clock_for_decided_tickets():
    record = make_record(received_at=_ago(hours=5), status="triaged")
    assert sla_clock(record, NOW) is None


def test_no_clock_without_a_parseable_received_at():
    assert sla_clock(make_record(received_at=""), NOW) is None
    assert sla_clock(make_record(received_at="not a date"), NOW) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("2026-03-04T09:14:00Z") is not None


def test_queue_rows_carry_the_clock():
    store = InMemoryStore()
    store.save(make_record(received_at=_ago(hours=2), ticket_id="SLA-OLD"))
    store.save(make_record(received_at=_ago(minutes=10), ticket_id="SLA-NEW"))
    rows = {row["ticket_id"]: row for row in store.queue_summary(now=NOW)}
    assert rows["SLA-OLD"]["sla_state"] == "breached"
    assert rows["SLA-OLD"]["seconds_to_breach"] == -3600.0
    assert rows["SLA-NEW"]["sla_state"] == "ok"
    assert rows["SLA-NEW"]["sla_target_hours"] == 1.0
    assert rows["SLA-NEW"]["age_seconds"] == 600.0


def test_overview_stats_count_clock_states():
    records = [
        make_record(received_at=_ago(hours=2), ticket_id="A"),
        make_record(received_at=_ago(minutes=50), ticket_id="B"),
        make_record(received_at=_ago(minutes=10), ticket_id="C"),
        make_record(received_at=_ago(hours=9), ticket_id="D", status="triaged"),
    ]
    stats = overview_stats(records, now=NOW)
    assert stats["sla"] == {"breached": 1, "due_soon": 1, "ok": 1}


def test_policy_snapshot_exposes_sla_targets():
    sla = policy_snapshot()["sla"]
    assert sla["target_hours"] == {"critical": 1.0, "high": 4.0, "medium": 24.0, "low": 72.0}
    assert sla["due_soon_fraction"] == 0.25


def test_a_fresh_run_starts_its_clock_at_intake():
    from helpers import fresh_service, make_ticket

    service = fresh_service()
    run = service.run_ticket(make_ticket(ticket_id="SLA-LIVE", received_at=""))
    # No received_at on the ticket: the run stamps its own start time,
    # so the clock is running and the (low-band) ticket is ok.
    assert run.result.received_at
    clock = sla_clock(run.result)  # wall clock, moments after intake
    assert clock is not None and clock["state"] == "ok"
