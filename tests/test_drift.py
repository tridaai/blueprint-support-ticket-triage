"""Calibration drift watch: seeded outcome streams, hand-computed windows.

The window math below is worked out in the comments before it is
asserted: a verify answer of p against truth False contributes
p² to the Brier sum, against truth True it contributes (1 − p)².
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.drift import drift_report
from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    QuestionAnswer,
    TicketResult,
)

from helpers import fresh_service


def verify_answer(key, p_true) -> QuestionAnswer:
    return QuestionAnswer(
        key=key, kind="verify", value=p_true,
        probabilities={"true": p_true, "false": 1 - p_true},
        confidence=max(p_true, 1 - p_true),
    )


def outcome_record(ticket_id, answers, recorded_at, **outcome_kwargs) -> TicketResult:
    return TicketResult(
        ticket_id=ticket_id,
        status="triaged",
        category="billing",
        answers=answers,
        composition=CompositionInfo(queue="billing_support"),
        outcome=AgentOutcome(recorded_at=recorded_at, **outcome_kwargs),
    )


def verify_stream(count, p_true, truth, key="is_outage_report", start_minute=0):
    """`count` records, each judging one verify question the same way."""
    records = []
    for i in range(count):
        minute = start_minute + i
        records.append(
            outcome_record(
                f"S-{start_minute}-{i}",
                [verify_answer(key, p_true)],
                f"2026-10-01T00:{minute:02d}:00Z",
                verify_truth={key: truth},
            )
        )
    return records


# ---------------------------------------------------------------------------
# the stable stream: window == baseline, nothing flags
# ---------------------------------------------------------------------------


def test_stable_stream_flags_nothing():
    # p=0.1 vs truth False → per-case Brier 0.01; baseline is 0.01 too.
    records = verify_stream(30, 0.1, False)
    report = drift_report(records, {"is_outage_report": {"kind": "verify", "brier": 0.01}})
    row = report["questions"]["is_outage_report"]
    assert report["available"] is True
    assert row["window_brier"] == 0.01
    assert row["live_brier"] == 0.01
    assert row["window_n"] == 30
    assert row["enough_samples"] is True
    assert row["delta_vs_eval"] == 0.0
    assert row["drift"] is False
    assert report["any_drift"] is False


def test_degrading_stream_flags_the_right_question_only():
    # Same records judge two questions: pii stays calibrated
    # (0.05 vs False → 0.0025), outage collapses (0.9 vs False → 0.81).
    records = []
    for i in range(30):
        records.append(
            outcome_record(
                f"D-{i}",
                [verify_answer("contains_pii", 0.05), verify_answer("is_outage_report", 0.9)],
                f"2026-10-01T00:{i:02d}:00Z",
                verify_truth={"contains_pii": False, "is_outage_report": False},
            )
        )
    baseline = {
        "contains_pii": {"kind": "verify", "brier": 0.0017},
        "is_outage_report": {"kind": "verify", "brier": 0.01},
    }
    report = drift_report(records, baseline)
    assert report["questions"]["is_outage_report"]["window_brier"] == 0.81
    assert report["questions"]["is_outage_report"]["delta_vs_eval"] == 0.8
    assert report["questions"]["is_outage_report"]["drift"] is True
    assert report["questions"]["contains_pii"]["drift"] is False
    assert report["any_drift"] is True


def test_below_minimum_window_never_flags():
    # Terrible calibration, but only 10 labelled outcomes: the gate
    # is the sample count, not the score.
    records = verify_stream(10, 0.9, False)
    report = drift_report(records, {"is_outage_report": {"kind": "verify", "brier": 0.01}})
    row = report["questions"]["is_outage_report"]
    assert row["window_brier"] == 0.81
    assert row["window_n"] == 10
    assert row["enough_samples"] is False
    assert row["drift"] is False
    assert report["any_drift"] is False


def test_window_is_the_last_n_by_recorded_time_not_insertion_order():
    # 60 outcomes: minutes 0–29 good (0.1 vs False → 0.01),
    # minutes 30–59 bad (0.9 vs False → 0.81). Window of 50 takes
    # minutes 10–59: 20 good + 30 bad → (20·0.01 + 30·0.81)/50 = 0.49.
    # All-time: (30·0.01 + 30·0.81)/60 = 0.41.
    records = verify_stream(30, 0.1, False) + verify_stream(30, 0.9, False, start_minute=30)
    # Deliberately hand the report the stream in reverse: ordering
    # must come from recorded_at, never from list position.
    report = drift_report(list(reversed(records)), {"is_outage_report": {"kind": "verify", "brier": 0.01}})
    row = report["questions"]["is_outage_report"]
    assert report["window"]["outcomes_total"] == 60
    assert report["window"]["outcomes_in_window"] == 50
    assert row["window_n"] == 50
    assert row["live_n"] == 60
    assert row["window_brier"] == 0.49
    assert row["live_brier"] == 0.41
    assert row["drift"] is True


def test_category_drift_uses_the_multiclass_brier():
    # Dist {billing 0.8, technical_issue 0.2}, truth technical_issue:
    # (0.8−0)² + (0.2−1)² = 0.64 + 0.64 = 1.28 per case.
    records = []
    for i in range(25):
        records.append(
            outcome_record(
                f"C-{i}",
                [QuestionAnswer(
                    key="category", kind="choice", value="billing",
                    probabilities={"billing": 0.8, "technical_issue": 0.2},
                    confidence=0.8,
                )],
                f"2026-10-01T00:{i:02d}:00Z",
                final_category="technical_issue",
            )
        )
    report = drift_report(records, {"category": {"kind": "choice", "brier": 0.083}})
    row = report["questions"]["category"]
    assert row["kind"] == "choice"
    assert row["window_brier"] == 1.28
    assert row["window_n"] == 25
    assert row["drift"] is True


# ---------------------------------------------------------------------------
# honest states
# ---------------------------------------------------------------------------


def test_no_outcomes_is_an_honest_empty_state():
    records = [TicketResult(ticket_id="X-1", status="awaiting_agent")]
    report = drift_report(records, None)
    assert report["available"] is False
    assert "no agent outcomes recorded yet" in report["note"]
    assert report["any_drift"] is False
    assert report["questions"] == {}


def test_outcomes_without_an_eval_baseline_cannot_flag():
    records = verify_stream(30, 0.9, False)
    report = drift_report(records, None)
    row = report["questions"]["is_outage_report"]
    assert report["available"] is True
    assert row["eval_brier"] is None
    assert row["delta_vs_eval"] is None
    assert row["drift"] is False
    assert "no eval baseline" in report["note"]


def test_threshold_and_window_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("DRIFT_BRIER_THRESHOLD", "0.9")
    records = verify_stream(30, 0.9, False)  # delta 0.80 < 0.90 now
    report = drift_report(records, {"is_outage_report": {"kind": "verify", "brier": 0.01}})
    assert report["window"]["brier_threshold"] == 0.9
    assert report["questions"]["is_outage_report"]["drift"] is False


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def drift_client():
    service = fresh_service()
    api.set_service(service)
    with TestClient(api.app) as client:
        yield client, service
    api.set_service(None)


def test_drift_endpoint_reports_the_honest_empty_state(drift_client):
    client, _service = drift_client
    body = client.get("/api/stats/drift").json()
    assert body["available"] is False
    assert body["window"]["size"] == 50
    assert body["window"]["min_samples"] == 20
    assert body["window"]["brier_threshold"] == 0.05


def test_drift_endpoint_pairs_live_window_with_the_eval_baseline(drift_client):
    client, service = drift_client
    service.store.save(
        outcome_record(
            "E-1",
            [verify_answer("is_outage_report", 0.9)],
            "2026-10-01T00:00:00Z",
            verify_truth={"is_outage_report": False},
        )
    )
    body = client.get("/api/stats/drift").json()
    assert body["available"] is True
    row = body["questions"]["is_outage_report"]
    # the packaged eval baseline rides along (is_outage_report is in it)
    assert row["eval_brier"] is not None
    assert row["window_brier"] == 0.81
    assert row["window_n"] == 1
    assert row["enough_samples"] is False
    assert row["drift"] is False
