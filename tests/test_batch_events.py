"""Batch intake, the lifecycle event bus + SSE stream, gate webhooks.

Batch: a loop over single runs — per-ticket failures stay in their
own entry. Events: the bus is fed by the same service paths the API
uses, and the SSE endpoint is exercised by really subscribing and
really running a ticket, reading the stream with a timeout guard.
Webhooks: the transport is stubbed (no network), delivery is joined.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from support_triage import api, webhooks
from support_triage.service import BATCH_MAX_TICKETS

from helpers import fresh_service, make_ticket


@pytest.fixture()
def service():
    return fresh_service()


@pytest.fixture()
def client(service):
    api.set_service(service)
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


# ---------------------------------------------------------------------------
# batch intake
# ---------------------------------------------------------------------------


def test_batch_triages_each_ticket_and_summarises(client):
    payload = {
        "tickets": [
            make_ticket(ticket_id="B-1").model_dump(),
            make_ticket(
                ticket_id="B-2",
                subject="How do I export my dashboard to PDF?",
                body="Hi, quick question — how do I export my dashboard to PDF? No rush. Thanks!",
            ).model_dump(),
        ]
    }
    response = client.post("/api/batches", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["counts"] == {"total": 2, "succeeded": 2, "failed": 0}
    by_id = {entry["ticket_id"]: entry for entry in body["results"]}
    assert by_id["B-1"]["queue"] == "billing_support"
    assert by_id["B-1"]["status"] == "awaiting_agent"
    assert by_id["B-2"]["queue"] == "general_support"
    # Both tickets really queued:
    assert client.get("/queue").json()["count"] == 2


def test_batch_accepts_a_bare_list(client):
    response = client.post("/api/batches", json=[make_ticket(ticket_id="B-3").model_dump()])
    assert response.status_code == 201
    assert response.json()["counts"]["succeeded"] == 1


def test_batch_reports_a_bad_ticket_without_sinking_the_rest(client):
    payload = [
        make_ticket(ticket_id="B-4").model_dump(),
        {"subject": "no ticket id at all"},  # fails TicketInput validation
        make_ticket(ticket_id="B-5").model_dump(),
    ]
    response = client.post("/api/batches", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["counts"] == {"total": 3, "succeeded": 2, "failed": 1}
    errors = [e for e in body["results"] if e["status"] == "error"]
    assert len(errors) == 1 and "invalid ticket" in errors[0]["error"]


def test_batch_reports_a_failing_run_per_ticket(service):
    from support_triage.errors import ProviderError

    class ExplodingEngine:
        name = "exploding"

        def decide(self, state, questions, context=""):
            raise ProviderError("backend is down")

    service.engine = ExplodingEngine()
    outcome = service.run_batch([make_ticket(ticket_id="B-6").model_dump()])
    assert outcome["counts"] == {"total": 1, "succeeded": 0, "failed": 1}
    assert "backend is down" in outcome["results"][0]["error"]


def test_batch_over_the_cap_is_rejected_whole(client):
    payload = [make_ticket(ticket_id=f"B-CAP-{i}").model_dump() for i in range(BATCH_MAX_TICKETS + 1)]
    response = client.post("/api/batches", json=payload)
    assert response.status_code == 422
    assert str(BATCH_MAX_TICKETS) in response.json()["detail"]
    assert client.get("/queue").json()["count"] == 0


def test_batch_rejects_a_non_list_body(client):
    response = client.post("/api/batches", json={"nope": 1})
    assert response.status_code == 422


def test_batch_completed_event_is_published(service):
    subscriber = service.bus.subscribe()
    service.run_batch([make_ticket(ticket_id="B-7").model_dump()])
    events = []
    while not subscriber.empty():
        events.append(subscriber.get_nowait())
    types = [e["type"] for e in events]
    assert "run_completed" in types
    batch_events = [e for e in events if e["type"] == "batch_completed"]
    assert batch_events and batch_events[0]["detail"]["counts"]["succeeded"] == 1


# ---------------------------------------------------------------------------
# event bus + SSE
# ---------------------------------------------------------------------------


def test_bus_fans_out_and_unsubscribes(service):
    first, second = service.bus.subscribe(), service.bus.subscribe()
    service._publish("run_completed", "T-1", {"status": "awaiting_agent"})
    assert first.get_nowait()["ticket_id"] == "T-1"
    assert second.get_nowait()["ticket_id"] == "T-1"
    service.bus.unsubscribe(first)
    service._publish("run_completed", "T-2", {"status": "awaiting_agent"})
    assert first.empty()
    assert second.get_nowait()["ticket_id"] == "T-2"


def test_run_and_outcome_publish_lifecycle_events(service):
    subscriber = service.bus.subscribe()
    service.run_ticket(make_ticket(ticket_id="EV-1"))
    from support_triage.schemas import AgentOutcome

    service.record_outcome("EV-1", AgentOutcome(final_category="billing", resolved=True))
    seen = {}
    while not subscriber.empty():
        event = subscriber.get_nowait()
        seen.setdefault(event["type"], event)
    assert seen["run_completed"]["detail"]["queue"] == "billing_support"
    assert seen["outcome_recorded"]["detail"]["resolved"] is True


def test_guardrail_blocked_is_published_for_an_indecisive_run(service):
    # The canned answers make category decisive; an indecisive stub
    # forces the low-confidence guardrail to fail its check.
    from helpers import valid_answers
    from support_triage.decisions import build_decision_set
    from support_triage.questions import triage_questions

    class IndecisiveEngine:
        name = "indecisive"

        def decide(self, state, questions, context=""):
            raw = valid_answers()
            options = [o for o in raw["category"]["probabilities"]]
            raw["category"] = {"probabilities": {o: 1.0 / len(options) for o in options}}
            return build_decision_set(
                questions, raw, backend=self.name, model="stub", latency_ms=1.0
            )

    service.engine = IndecisiveEngine()
    subscriber = service.bus.subscribe()
    run = service.run_ticket(make_ticket(ticket_id="EV-2"))
    assert not run.result.guardrails.passed
    types = []
    while not subscriber.empty():
        types.append(subscriber.get_nowait()["type"])
    assert "run_completed" in types and "guardrail_blocked" in types


def test_sse_stream_delivers_a_run_completed_event(service):
    """The SSE endpoint against a real (loopback) uvicorn server.

    TestClient's portal transport stalls on an unbounded stream
    behind the middleware stack, so this test uses the production
    transport instead — same process, ephemeral port, no external
    network. The ticket is triaged over HTTP once the subscription
    has registered on the bus.
    """
    import httpx
    import uvicorn

    api.set_service(service)
    config = uvicorn.Config(api.app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        # trust_env=False: this sandbox exports proxy variables that
        # are meaningless (and unparseable) for a loopback server.
        web = httpx.Client(base_url=base, trust_env=False, timeout=10)
        assert web.get("/health").status_code == 200

        def trigger():
            deadline = time.time() + 5
            while service.bus.subscriber_count == 0 and time.time() < deadline:
                time.sleep(0.02)
            web.post("/tickets/analyze", json=make_ticket(ticket_id="SSE-1").model_dump())

        worker = threading.Thread(target=trigger, daemon=True)
        worker.start()
        received: list[str] = []
        with web.stream("GET", "/api/events") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            for line in response.iter_lines():
                received.append(line)
                if '"ticket_id": "SSE-1"' in line:  # the data line of the event
                    break
        worker.join(timeout=5)
        text = "\n".join(received)
        assert "event: run_completed" in text
        assert '"ticket_id": "SSE-1"' in text
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        api.set_service(None)


# ---------------------------------------------------------------------------
# gate webhooks
# ---------------------------------------------------------------------------


def test_webhook_is_a_noop_when_unconfigured(service, monkeypatch):
    calls = []
    monkeypatch.setattr(webhooks, "_post", lambda *a: calls.append(a))
    assert webhooks.notify("run_completed", {"ticket_id": "W-0"}) is None
    assert calls == []


def test_webhook_posts_the_compact_event_with_secret(service, monkeypatch):
    monkeypatch.setenv("TRIAGE_WEBHOOK_URL", "http://receiver.invalid/hook")
    monkeypatch.setenv("TRIAGE_WEBHOOK_SECRET", "s3cret")
    calls = []

    def stub_post(url, headers, body, timeout):
        calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})

    monkeypatch.setattr(webhooks, "_post", stub_post)
    # Drive it through the real service path: a completed run.
    subscriber = service.bus.subscribe()  # keep the bus honest too
    service.run_ticket(make_ticket(ticket_id="W-1"))
    # Delivery is on a daemon thread; poll briefly for the stub call.
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "http://receiver.invalid/hook"
    assert call["headers"]["X-Triage-Webhook-Secret"] == "s3cret"
    body = call["body"]
    assert body["event"] == "run_completed"
    assert body["ticket_id"] == "W-1"
    assert body["queue"] == "billing_support"
    assert body["status"] == "awaiting_agent"
    assert body["guardrails"]["passed"] is True
    assert body["at"]


def test_webhook_failure_is_logged_and_never_raises(monkeypatch):
    monkeypatch.setenv("TRIAGE_WEBHOOK_URL", "http://receiver.invalid/hook")

    def broken_post(url, headers, body, timeout):
        raise ConnectionError("receiver is down")

    monkeypatch.setattr(webhooks, "_post", broken_post)
    thread = webhooks.notify("run_completed", {"ticket_id": "W-2"})
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()  # delivered (and failed) without raising
