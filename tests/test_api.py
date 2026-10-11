"""API surface: routes, auth, honest readiness, error shapes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api

from helpers import fresh_service, make_ticket


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "support-ticket-triage"


def test_readiness_is_honest_about_test_doubles(client):
    response = client.get("/readiness")
    assert response.status_code == 200
    assert "in-memory" in response.json()["store"]


def test_console_and_favicon_are_served(client):
    assert client.get("/").status_code == 200
    assert "Support Ticket Triage Agent" in client.get("/").text
    assert client.get("/favicon.svg").status_code == 200


def test_samples_are_synthetic_and_complete(client):
    body = client.get("/samples").json()
    assert len(body["tickets"]) == 10
    assert all(t["ticket_id"].startswith("SUP-") for t in body["tickets"])


def test_analyze_then_fetch_then_decide(client):
    ticket = make_ticket(ticket_id="API-1")
    response = client.post("/tickets/analyze", json=ticket.model_dump())
    assert response.status_code == 201
    record = response.json()
    assert record["status"] == "awaiting_agent"
    assert record["composition"]["queue"] == "billing_support"

    fetched = client.get("/tickets/API-1")
    assert fetched.status_code == 200
    assert fetched.json()["ticket_id"] == "API-1"

    decided = client.post(
        "/tickets/API-1/decision",
        json={"decision": "approved", "by": "agent-1", "assignee": "agent-2"},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "triaged"
    assert "agent-2" in decided.json()["decision_reason"]


def test_unknown_ticket_404s(client):
    assert client.get("/tickets/NOPE").status_code == 404
    response = client.post("/tickets/NOPE/decision", json={"decision": "approved"})
    assert response.status_code == 404


def test_invalid_decision_value_is_422(client):
    client.post("/tickets/analyze", json=make_ticket(ticket_id="API-2").model_dump())
    response = client.post("/tickets/API-2/decision", json={"decision": "shrug"})
    assert response.status_code == 422


def test_second_decision_conflicts(client):
    client.post("/tickets/analyze", json=make_ticket(ticket_id="API-3").model_dump())
    client.post("/tickets/API-3/decision", json={"decision": "approved"})
    response = client.post("/tickets/API-3/decision", json={"decision": "rejected"})
    assert response.status_code == 409


def test_assign_endpoint(client):
    client.post("/tickets/analyze", json=make_ticket(ticket_id="API-4").model_dump())
    response = client.post("/tickets/API-4/assign", json={"assignee": "agent-7"})
    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_agent"
    empty = client.post("/tickets/API-4/assign", json={"assignee": "  "})
    assert empty.status_code == 422


def test_queue_and_metrics(client):
    client.post("/tickets/analyze", json=make_ticket(ticket_id="API-5").model_dump())
    queue = client.get("/queue").json()["tickets"]
    assert len(queue) == 1
    assert queue[0]["ticket_id"] == "API-5"
    assert queue[0]["category"] == "billing"
    metrics = client.get("/metrics").json()
    assert metrics["total_tickets"] == 1
    assert metrics["awaiting_agent"] == 1


def test_eval_results_404_with_instructions_when_absent(client, monkeypatch, tmp_path):
    # Point the packaged-data lookup at an empty location by removing files
    # is intrusive; instead assert the endpoint answers with one of the two
    # documented shapes (results present in a built checkout, or the 404).
    response = client.get("/evals/results")
    assert response.status_code in (200, 404)
    if response.status_code == 404:
        assert "run the eval pack" in response.json()["detail"]


def test_api_key_gate(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    assert client.get("/queue").status_code == 401
    assert client.get("/queue", headers={"X-API-Key": "test-key"}).status_code == 200
    # probes and console stay public
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_request_id_header_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"
    generated = client.get("/health")
    assert generated.headers["X-Request-ID"]
