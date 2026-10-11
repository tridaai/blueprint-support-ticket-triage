"""Dashboards API: stats endpoints, policy, queue filters, outcomes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.schemas import (
    CompositionInfo,
    GuardrailInfo,
    QuestionAnswer,
    RunTelemetry,
    TicketResult,
)

from helpers import fresh_service

FLAGS_OFF = {
    "contains_pii": False,
    "is_abusive": False,
    "mentions_legal": False,
    "is_outage_report": False,
    "needs_engineering": False,
}


def seed_record(service, ticket_id, **overrides) -> TicketResult:
    base = dict(
        ticket_id=ticket_id,
        status="awaiting_agent",
        subject="Refund for a duplicate charge",
        sender_name="Priya Nair",
        category="billing",
        product_area="checkout",
        answers=[
            QuestionAnswer(
                key="category", kind="choice", value="billing",
                probabilities={"billing": 0.9, "other": 0.1}, confidence=0.9,
            ),
            QuestionAnswer(
                key="contains_pii", kind="verify", value=0.05,
                probabilities={"true": 0.05, "false": 0.95}, confidence=0.95,
            ),
        ],
        composition=CompositionInfo(
            priority_score=60.0, priority_band="high",
            queue="billing_support", flags=dict(FLAGS_OFF),
        ),
        guardrails=GuardrailInfo(checks=[{"name": "completeness", "passed": True, "detail": "ok"}]),
        telemetry=RunTelemetry(backend="offline", decision_latency_ms=12.0, question_count=11),
    )
    base.update(overrides)
    record = TicketResult(**base)
    service.store.save(record)
    return record


@pytest.fixture()
def seeded():
    service = fresh_service()
    seed_record(service, "D-1")
    seed_record(
        service, "D-2",
        subject="API returns 500 on every call",
        sender_name="Marco Ruiz",
        category="technical_issue",
        composition=CompositionInfo(
            priority_score=95.0, priority_band="critical",
            queue="incident_response",
            flags=dict(FLAGS_OFF, is_outage_report=True),
        ),
        telemetry=RunTelemetry(backend="llm", decision_latency_ms=300.0, question_count=11),
    )
    seed_record(
        service, "D-3",
        status="triaged",
        subject="Please delete my account data",
        sender_name="Ana Souza",
        category="account_access",
        composition=CompositionInfo(
            priority_score=40.0, priority_band="medium",
            queue="compliance_review",
            flags=dict(FLAGS_OFF, contains_pii=True, mentions_legal=True),
        ),
    )
    api.set_service(service)
    with TestClient(api.app) as client:
        yield client, service
    api.set_service(None)


@pytest.fixture()
def client(seeded):
    return seeded[0]


# ---------------------------------------------------------------------------
# stats + policy endpoints
# ---------------------------------------------------------------------------


def test_stats_overview_endpoint(client):
    body = client.get("/api/stats/overview").json()
    assert body["total_runs"] == 3
    assert body["by_queue"]["incident_response"] == 1
    assert body["priority_bands"]["critical"] == 1
    assert body["safety_flags"]["contains_pii"] == 1
    assert body["safety_flags"]["is_outage_report"] == 1
    assert body["outcomes_recorded"] == 0


def test_stats_performance_endpoint(client):
    body = client.get("/api/stats/performance").json()
    assert body["total_runs"] == 3
    assert body["by_backend"]["llm"]["decision_latency_ms"]["mean"] == 300.0
    assert body["by_backend"]["offline"]["runs"] == 2


def test_stats_calibration_endpoint_pairs_eval_with_live(client):
    body = client.get("/api/stats/calibration").json()
    assert "eval" in body and "live" in body
    # no outcomes recorded yet: the live section says so, honestly
    assert body["live"]["available"] is False
    assert "no agent outcomes recorded yet" in body["live"]["note"]


def test_policy_endpoint_exposes_the_composition_config(client):
    body = client.get("/api/policy").json()
    assert body["priority"]["weights"]["urgency"] == 0.30
    assert body["priority"]["outage_boost"] == 15.0
    assert body["routing"]["category_queues"]["billing"] == "billing_support"
    assert body["routing"]["overrides"][0]["flag"] == "mentions_legal"
    assert body["guardrails"]["category_min_confidence"] == 0.55
    assert body["autonomy"]["note"].startswith("advisory only")


# ---------------------------------------------------------------------------
# queue filters + search
# ---------------------------------------------------------------------------


def test_queue_unfiltered_is_open_only(client):
    rows = client.get("/queue").json()["tickets"]
    assert [r["ticket_id"] for r in rows] == ["D-2", "D-1"]  # priority order


def test_queue_status_filter_searches_decided_tickets(client):
    rows = client.get("/queue", params={"status": "triaged"}).json()["tickets"]
    assert [r["ticket_id"] for r in rows] == ["D-3"]


def test_queue_filters_by_queue_band_flag_backend(client):
    get = lambda **kw: client.get("/queue", params=kw).json()["tickets"]  # noqa: E731
    assert [r["ticket_id"] for r in get(queue="incident_response")] == ["D-2"]
    assert [r["ticket_id"] for r in get(priority_band="medium", status="triaged")] == ["D-3"]
    assert [r["ticket_id"] for r in get(flag="is_outage_report")] == ["D-2"]
    assert [r["ticket_id"] for r in get(flag="contains_pii", status="triaged")] == ["D-3"]
    assert [r["ticket_id"] for r in get(backend="llm")] == ["D-2"]
    assert get(queue="billing_support", priority_band="critical") == []


def test_queue_text_search_covers_subject_and_sender_case_insensitively(client):
    get = lambda **kw: client.get("/queue", params=kw).json()["tickets"]  # noqa: E731
    assert [r["ticket_id"] for r in get(q="duplicate charge")] == ["D-1"]
    assert [r["ticket_id"] for r in get(q="MARCO")] == ["D-2"]
    assert [r["ticket_id"] for r in get(q="ana", status="triaged")] == ["D-3"]
    assert get(q="no such words") == []


def test_queue_rejects_an_unknown_flag(client):
    response = client.get("/queue", params={"flag": "was_rude"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# outcome endpoint -> live calibration loop
# ---------------------------------------------------------------------------


def test_outcome_endpoint_records_normalised_truth(client):
    response = client.post(
        "/api/runs/D-1/outcome",
        json={
            "final_category": "billing",
            "final_queue": "billing_support",
            "resolved": True,
            "by": "agent-9",
            "verify_truth": {"was_pii": False},
        },
    )
    assert response.status_code == 200
    outcome = response.json()["outcome"]
    assert outcome["verify_truth"] == {"contains_pii": False}
    assert outcome["recorded_by"] == "agent-9"
    assert outcome["recorded_at"]

    live = client.get("/api/stats/calibration").json()["live"]
    assert live["available"] is True
    assert live["runs_with_outcomes"] == 1
    assert live["category_accuracy"] == 1.0
    assert live["queue_accuracy"] == 1.0
    assert live["resolution_rate"] == 1.0
    # predicted P(pii) was 0.05, truth False -> Brier 0.0025
    assert live["per_question"]["contains_pii"]["brier"] == pytest.approx(0.0025)


def test_outcome_endpoint_validation(client):
    unknown = client.post("/api/runs/NOPE/outcome", json={"resolved": True})
    assert unknown.status_code == 404
    bad_category = client.post(
        "/api/runs/D-1/outcome", json={"final_category": "vibes"}
    )
    assert bad_category.status_code == 422
    bad_truth = client.post(
        "/api/runs/D-1/outcome", json={"verify_truth": {"was_rude": True}}
    )
    assert bad_truth.status_code == 422
    bad_score = client.post(
        "/api/runs/D-1/outcome", json={"score_truth": {"urgency": 42}}
    )
    assert bad_score.status_code == 422


def test_dashboards_view_is_in_the_console(client):
    html = client.get("/").text
    assert "Dashboards" in html
    assert 'id="view-dashboards"' in html
    assert "Policy what-if" in html
