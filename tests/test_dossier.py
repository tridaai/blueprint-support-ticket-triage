"""The customer dossier: who is writing, computed in code.

Pins, in order:

- the keying rule — the thread machinery's key (normalised email,
  name fallback, unknown customers have no dossier), stated limits
  and all;
- the dossier as a pure projection: counts, outcome figures under
  the small-sample floor, threads under the EFFECTIVE grouping (a
  detach moves the ticket), and byte-identical records across
  computing it;
- the two flags with their exact evidence — repeat_issue and
  frequent_contact firing, and silence below the thresholds;
- the decision call seeing the dossier deterministically: the
  history block per backend (fenced for the LLM, the context field
  for Jev, accepted-and-unused offline with the record saying so),
  the no-other-ticket's-text pin, and the composition-invariance
  pin — a repeat customer's ticket composes exactly as a
  first-timer's with the same answers, because history never
  enters the formula.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_triage import api, dossier
from support_triage.backends import (
    JevDecisionEngine,
    LLMDecisionEngine,
    OfflineDecisionEngine,
)
from support_triage.exports import render_text
from support_triage.schemas import AgentOutcome, TicketInput, TicketResult
from support_triage.service import TriageService

from helpers import (
    fake_jev_transport,
    fresh_service,
    make_ticket,
    valid_answers,
)

PRIYA = "priya@example.com"

_BILLING_BODIES = [
    "I was charged twice for the March subscription — two identical "
    "payments of $49 left my card. Please refund one of them.",
    "My April invoice shows a seat charge for a colleague who left in "
    "March. Please correct the invoice and refund the seat charge.",
    "The renewal charge went through at the old price even though we "
    "downgraded last month. Please refund the difference.",
]


def _run_billing(service: TriageService, ticket_id: str, day: str, body: str):
    return service.run_ticket(
        make_ticket(
            ticket_id=ticket_id,
            subject=f"Billing question {ticket_id}",
            body=body,
            received_at=f"2026-09-{day}T10:00:00Z",
        )
    ).result


def _decide_and_close(service: TriageService, ticket_id: str, resolved: bool = True):
    service.record_decision(ticket_id, "approved", by="agent-1")
    service.record_outcome(
        ticket_id,
        AgentOutcome(
            final_category="billing",
            final_queue="billing_support",
            resolved=resolved,
            recorded_by="agent-1",
        ),
    )


def _priya_history(service: TriageService) -> None:
    """Three decided billing tickets with outcomes, inside 90 days."""
    for index, (ticket_id, day) in enumerate(
        [("DOS-1", "02"), ("DOS-2", "09"), ("DOS-3", "16")]
    ):
        _run_billing(service, ticket_id, day, _BILLING_BODIES[index])
        _decide_and_close(service, ticket_id, resolved=index != 1)


# --- the keying rule -------------------------------------------------------


def test_key_is_the_normalised_email_and_the_dossier_finds_it():
    service = fresh_service()
    service.run_ticket(
        make_ticket(ticket_id="KEY-1", sender_email="  Priya@Example.COM ")
    )
    record = service.get_record("KEY-1")
    assert record.history.customer_key == PRIYA
    found = service.customer_dossier(PRIYA)
    assert found["identity"]["kind"] == "email"
    assert found["identity"]["email"] == PRIYA
    assert found["counts"]["total"] == 1


def test_name_fallback_key_when_intake_carries_no_email():
    service = fresh_service()
    service.run_ticket(
        make_ticket(ticket_id="KEY-2", sender_email="", sender_name="Sam Ortiz")
    )
    record = service.get_record("KEY-2")
    assert record.history.customer_key == "name:sam ortiz"
    found = service.dossier_for_run("KEY-2")
    assert found["identity"]["kind"] == "name"
    assert found["customer_key"] == "name:sam ortiz"


def test_unknown_customer_has_no_dossier_and_no_history():
    service = fresh_service()
    run = service.run_ticket(
        make_ticket(ticket_id="KEY-3", sender_email="", sender_name="")
    )
    assert run.result.history.customer_key == ""
    assert run.result.history.context_provided is False
    with pytest.raises(KeyError):
        service.customer_dossier("")
    with pytest.raises(KeyError):
        service.dossier_for_run("KEY-3")
    with pytest.raises(KeyError):
        service.customer_dossier("nobody@example.com")


# --- the dossier read model -------------------------------------------------


def test_dossier_counts_outcomes_and_last_outcome():
    service = fresh_service()
    _priya_history(service)
    run = _run_billing(service, "DOS-4", "20", "One more billing question about my card.")
    found = service.customer_dossier(PRIYA)
    assert found["counts"] == {
        "total": 4,
        "open": 1,
        "decided": 3,
        "reopened": 0,
        "with_outcome": 3,
        "threads": found["counts"]["threads"],
    }
    assert found["outcomes"]["with_outcome"] == 3
    assert found["outcomes"]["resolved_count"] == 2
    assert found["outcomes"]["resolved_rate"] == round(2 / 3, 4)
    assert found["outcomes"]["median_hours_from_intake"] is not None
    assert found["last_outcome"]["ticket_id"] == "DOS-3"
    assert [t["ticket_id"] for t in found["tickets"]] == [
        "DOS-1",
        "DOS-2",
        "DOS-3",
        "DOS-4",
    ]
    assert run.history.prior_tickets == 3


def test_dossier_is_a_pure_projection():
    service = fresh_service()
    _priya_history(service)
    before = json.dumps(
        [r.model_dump() for r in service.store.all_records()], sort_keys=True
    )
    service.customer_dossier(PRIYA)
    service.dossier_for_run("DOS-2")
    after = json.dumps(
        [r.model_dump() for r in service.store.all_records()], sort_keys=True
    )
    assert before == after


def test_small_sample_honesty_withholds_the_rate():
    service = fresh_service()
    _run_billing(service, "DOS-S1", "02", _BILLING_BODIES[0])
    _decide_and_close(service, "DOS-S1")
    _run_billing(service, "DOS-S2", "09", _BILLING_BODIES[1])
    found = service.customer_dossier(PRIYA)
    assert found["outcomes"]["with_outcome"] == 1
    assert found["outcomes"]["resolved_rate"] is None
    assert found["outcomes"]["median_hours_from_intake"] is None
    assert "too few to quote" in found["outcomes"]["note"]


def test_outcome_figures_median_is_intake_to_outcome_arithmetic():
    def record(ticket_id: str, received: str, recorded: str, resolved: bool):
        return TicketResult(
            ticket_id=ticket_id,
            received_at=received,
            outcome=AgentOutcome(
                final_category="billing",
                resolved=resolved,
                recorded_by="agent-1",
                recorded_at=recorded,
            ),
        )

    mine = [
        record("M-1", "2026-09-01T00:00:00Z", "2026-09-01T10:00:00Z", True),
        record("M-2", "2026-09-02T00:00:00Z", "2026-09-03T04:00:00Z", True),
        record("M-3", "2026-09-03T00:00:00Z", "2026-09-03T06:00:00Z", False),
    ]
    figures = dossier.outcome_figures(mine)
    assert figures["resolved_rate"] == round(2 / 3, 4)
    assert figures["median_hours_from_intake"] == 10.0  # median of 10, 28, 6


# --- the flags ---------------------------------------------------------------


def test_repeat_issue_flag_fires_with_exact_evidence():
    service = fresh_service()
    _priya_history(service)
    found = service.customer_dossier(PRIYA)
    assert len(found["flags"]) == 1
    flag = found["flags"][0]
    assert flag["flag"] == "repeat_issue"
    assert flag["category"] == "billing"
    assert flag["count"] == 3
    assert flag["window_days"] == dossier.REPEAT_ISSUE_WINDOW_DAYS
    assert flag["ticket_ids"] == ["DOS-1", "DOS-2", "DOS-3"]


def test_repeat_issue_needs_decisions_not_just_contacts():
    service = fresh_service()
    for index, ticket_id in enumerate(["DOS-U1", "DOS-U2", "DOS-U3"]):
        _run_billing(service, ticket_id, f"0{2 + index}", _BILLING_BODIES[index])
    # Three billing tickets, none decided — no flag.
    assert service.customer_dossier(PRIYA)["flags"] == []


def test_frequent_contact_flag_fires_on_volume_alone():
    service = fresh_service()
    bodies = [
        ("FQ-1", "Charged twice", _BILLING_BODIES[0]),
        ("FQ-2", "App crashes", "The app crashes every time I open the export tab."),
        ("FQ-3", "Adding a teammate", "How do I add a teammate to the workspace?"),
        ("FQ-4", "Locked out", "I am locked out of my account after the password change."),
        ("FQ-5", "Invoice copy", "Please send a copy of the May invoice for our records."),
    ]
    for index, (ticket_id, subject, body) in enumerate(bodies):
        service.run_ticket(
            make_ticket(
                ticket_id=ticket_id,
                subject=subject,
                body=body,
                sender_name="Fred Osei",
                sender_email="fred@example.com",
                received_at=f"2026-09-0{index + 1}T09:00:00Z",
            )
        )
    service.record_decision("FQ-1", "approved", by="agent-1")
    service.record_decision("FQ-2", "approved", by="agent-1")
    found = service.customer_dossier("fred@example.com")
    assert [f["flag"] for f in found["flags"]] == ["frequent_contact"]
    flag = found["flags"][0]
    assert flag["count"] == 5
    assert flag["window_days"] == dossier.FREQUENT_CONTACT_WINDOW_DAYS
    assert flag["ticket_ids"] == ["FQ-1", "FQ-2", "FQ-3", "FQ-4", "FQ-5"]
    assert found["counts"]["total"] == 5


def test_below_threshold_customer_has_no_flags():
    service = fresh_service()
    _run_billing(service, "DOS-B1", "02", _BILLING_BODIES[0])
    _decide_and_close(service, "DOS-B1")
    _run_billing(service, "DOS-B2", "09", _BILLING_BODIES[1])
    _decide_and_close(service, "DOS-B2")
    found = service.customer_dossier(PRIYA)
    assert found["flags"] == []
    assert found["counts"]["decided"] == 2


def test_old_decisions_fall_outside_the_repeat_window():
    service = fresh_service()
    # Three decided billing tickets, but spread over five months:
    # no 90-day window anchored at the newest holds all three.
    for ticket_id, received in [
        ("DOS-O1", "2026-04-01T10:00:00Z"),
        ("DOS-O2", "2026-06-15T10:00:00Z"),
        ("DOS-O3", "2026-09-01T10:00:00Z"),
    ]:
        service.run_ticket(
            make_ticket(ticket_id=ticket_id, body=_BILLING_BODIES[0], received_at=received)
        )
        _decide_and_close(service, ticket_id)
    assert service.customer_dossier(PRIYA)["flags"] == []


# --- the dossier under a correction overlay ---------------------------------


def test_detach_moves_the_ticket_in_the_dossier():
    service = fresh_service()
    body = (
        "I was charged twice for the subscription and one payment must "
        "be refunded to my card."
    )
    service.run_ticket(
        make_ticket(
            ticket_id="GRP-1",
            subject="Charged twice",
            body=body,
            sender_name="Sam Ortiz",
            sender_email="sam@example.com",
            received_at="2026-09-01T10:00:00Z",
        )
    )
    service.run_ticket(
        make_ticket(
            ticket_id="GRP-2",
            subject="Charged twice again",
            body=body + " This is the second month in a row.",
            sender_name="Sam Ortiz",
            sender_email="sam@example.com",
            received_at="2026-09-03T10:00:00Z",
        )
    )
    before = service.customer_dossier("sam@example.com")
    assert before["counts"]["threads"] == 1
    assert before["threads"][0]["ticket_ids"] == ["GRP-1", "GRP-2"]

    service.detach_run("GRP-2", actor="agent-1", reason="a separate month's charge")

    after = service.customer_dossier("sam@example.com")
    assert after["counts"]["threads"] == 2
    by_thread = {t["thread_id"]: t["ticket_ids"] for t in after["threads"]}
    assert by_thread["thr-GRP-1"] == ["GRP-1"]
    # A detached chain sits under its operator thread id (corrections.py).
    assert by_thread["thr-op-GRP-2"] == ["GRP-2"]
    moved = next(t for t in after["tickets"] if t["ticket_id"] == "GRP-2")
    assert moved["thread_id"] == "thr-op-GRP-2"
    assert moved["grouping"]["corrected"] is True
    stayed = next(t for t in after["tickets"] if t["ticket_id"] == "GRP-1")
    assert stayed["thread_id"] == "thr-GRP-1"


# --- the decision call sees the dossier --------------------------------------


def test_first_contact_offers_no_history():
    service = fresh_service()
    run = service.run_ticket(make_ticket(ticket_id="DOS-F0"))
    history = run.result.history
    assert history.context_provided is False
    assert history.context_use == ""
    assert history.prior_tickets == 0
    assert history.flags == []


def test_history_lands_on_the_record_with_flags_and_last_outcome():
    service = fresh_service()
    _priya_history(service)
    run = _run_billing(service, "DOS-4", "20", "One more billing question about my card.")
    history = run.history
    assert history.customer_key == PRIYA
    assert history.context_provided is True
    assert history.context_use == "ignored_offline"  # the offline backend, stated
    assert history.prior_tickets == 3
    assert history.decided == 3
    assert history.outcomes_recorded == 3
    assert history.resolved_rate == round(2 / 3, 4)
    assert history.last_outcome_ticket_id == "DOS-3"
    assert history.last_outcome_category == "billing"
    assert history.last_outcome_resolved is True
    assert [f.flag for f in history.flags] == ["repeat_issue"]
    assert history.flags[0].ticket_ids == ["DOS-1", "DOS-2", "DOS-3"]


def test_llm_prompt_carries_the_block_fenced_and_text_free():
    captured: dict = {}

    def complete(system, user, max_tokens=2200):
        captured["user"] = user
        return json.dumps({"answers": valid_answers()})

    service = fresh_service(
        LLMDecisionEngine("openai", complete=complete, model="gpt-5-mini")
    )
    _priya_history(service)
    captured.clear()
    run = service.run_ticket(
        make_ticket(
            ticket_id="DOS-LLM",
            body="A new billing question about this month's card statement.",
            received_at="2026-09-20T10:00:00Z",
        )
    )
    user = captured["user"]
    assert "Customer history for priya@example.com" in user
    assert "category billing decided 3 times in the last 90 days" in user
    assert "Most recent outcome: DOS-3" in user
    # Fenced as history, outside the untrusted ticket fence.
    assert user.index("<<<END UNTRUSTED>>>") < user.index("Customer history for")
    # No other ticket's words anywhere in the prompt: the distinctive
    # phrases of DOS-2's and DOS-3's bodies never appear.
    assert "seat charge for a colleague" not in user
    assert "downgraded last month" not in user
    assert run.result.history.context_use == "history_block"


def test_jev_request_carries_the_block_as_its_prior_history_field(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-not-real")
    transport = fake_jev_transport()
    service = fresh_service(JevDecisionEngine(transport=transport))
    _priya_history(service)
    first = transport.calls[0]["body"]
    assert isinstance(first["state"], str)  # DOS-1 was a first contact
    run = service.run_ticket(
        make_ticket(
            ticket_id="DOS-JEV",
            body="A new billing question about this month's card statement.",
            received_at="2026-09-20T10:00:00Z",
        )
    )
    state = transport.calls[-1]["body"]["state"]
    assert "A new billing question" in state["ticket"]
    assert state["prior_history"].startswith("<<<PRIOR HANDLING HISTORY")
    assert "Customer history for priya@example.com" in state["prior_history"]
    assert "seat charge for a colleague" not in state["prior_history"]
    assert "Customer history" not in state["ticket"]  # never folded into the ticket
    assert run.result.history.context_use == "history_block"


def test_offline_answers_do_not_depend_on_history():
    engine = OfflineDecisionEngine()
    state = "Subject: Charged twice\nChannel: email\nA double charge and a refund."
    from support_triage.questions import triage_questions

    bare = engine.decide(state, triage_questions())
    with_history = engine.decide(
        state, triage_questions(), context="Customer history for x: flags everywhere."
    )
    assert bare.results == with_history.results


def test_flags_never_change_composition():
    """Same answers in, same composition out — first-timer vs a
    customer whose flags are flying. The LLM double answers
    identically for both; only the offered context differs."""

    def complete(system, user, max_tokens=2200):
        return json.dumps({"answers": valid_answers()})

    engine = LLMDecisionEngine("openai", complete=complete, model="gpt-5-mini")

    first_timer = fresh_service(engine)
    run_first = first_timer.run_ticket(
        make_ticket(
            ticket_id="CMP-A",
            sender_name="Pat A",
            sender_email="first@example.com",
            received_at="2026-09-20T10:00:00Z",
        )
    ).result

    veteran_service = fresh_service(engine)
    for index, ticket_id in enumerate(["CMP-P1", "CMP-P2", "CMP-P3"]):
        veteran_service.run_ticket(
            make_ticket(
                ticket_id=ticket_id,
                subject=f"Billing question {ticket_id}",
                body=_BILLING_BODIES[index],
                sender_name="Pat A",
                sender_email="veteran@example.com",
                received_at=f"2026-09-{2 + index * 7:02d}T10:00:00Z",
            )
        )
        _decide_and_close(veteran_service, ticket_id)
    run_veteran = veteran_service.run_ticket(
        make_ticket(
            ticket_id="CMP-B",
            sender_name="Pat A",
            sender_email="veteran@example.com",
            received_at="2026-09-20T10:00:00Z",
        )
    ).result

    assert run_first.history.flags == []
    assert [f.flag for f in run_veteran.history.flags] == ["repeat_issue"]
    assert run_veteran.composition.model_dump() == run_first.composition.model_dump()
    assert [a.model_dump() for a in run_veteran.answers] == [
        a.model_dump() for a in run_first.answers
    ]
    assert run_veteran.category == run_first.category


def test_reopen_run_also_carries_customer_history():
    service = fresh_service()
    _run_billing(service, "DOS-R1", "02", _BILLING_BODIES[0])
    service.record_decision("DOS-R1", "approved", by="agent-1")
    run = service.reopen_run(
        "DOS-R1", message="It happened again this month.", actor="agent-1"
    )
    assert run.result.reopen is not None
    assert run.result.history.context_provided is True
    assert run.result.history.prior_tickets == 1


# --- surfaces ------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def test_api_dossier_by_run_and_by_key(client):
    service = api.get_service()
    _priya_history(service)
    by_run = client.get("/api/runs/DOS-2/dossier")
    assert by_run.status_code == 200
    assert by_run.json()["counts"]["total"] == 3
    assert by_run.json()["flags"][0]["flag"] == "repeat_issue"
    by_key = client.get("/api/customers/dossier", params={"key": PRIYA})
    assert by_key.status_code == 200
    assert by_key.json()["customer_key"] == PRIYA


def test_api_dossier_unknowns_are_404(client):
    assert (
        client.get("/api/customers/dossier", params={"key": "ghost@example.com"}).status_code
        == 404
    )
    assert client.get("/api/customers/dossier").status_code == 404
    assert client.get("/api/runs/NOPE/dossier").status_code == 404


def test_console_serves_the_dossier_section(client):
    page = client.get("/").text
    assert "Customer dossier" in page
    assert "loadDossier" in page


def test_export_carries_the_customer_history():
    service = fresh_service()
    _priya_history(service)
    _run_billing(service, "DOS-4", "20", "One more billing question about my card.")
    document = service.export_run("DOS-4")
    history = document["customer_history"]
    assert history["customer_key"] == PRIYA
    assert history["context"]["provided"] is True
    assert history["context"]["use"] == "ignored_offline"
    assert history["prior_tickets"] == 3
    assert history["flags"][0]["flag"] == "repeat_issue"
    assert history["last_outcome"]["ticket_id"] == "DOS-3"
    text = render_text(document)
    assert "CUSTOMER HISTORY" in text
    assert "category billing decided 3 times in the last 90 days" in text
