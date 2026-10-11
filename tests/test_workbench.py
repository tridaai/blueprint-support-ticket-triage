"""The agent's workbench: knowledge, playbooks, working notes.

Knowledge: seeded neighbours with authored intake ages — the
aggregate math is pinned (median over intake-to-outcome hours,
resolved rate, the similar-without-outcome exclusion accounted,
the rate withheld below the small-sample floor).

Playbook + note: overlay actions land on the record with their
actor; refusals carry the workbench taxonomy; and the pins that
matter — the run's triage (status, composition, decision, the
customer template draft) and the calibration samples are byte-
identical before and after every workbench action.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from support_triage import api, workbench
from support_triage.decisions import build_decision_set
from support_triage.exports import build_export, render_text
from support_triage.livecal import live_samples
from support_triage.questions import triage_questions
from support_triage.schemas import AgentOutcome

from helpers import fresh_service, make_ticket, valid_answers

BODY = (
    "Hi, I was charged twice for the March subscription — two identical "
    "payments of $49 left my card. Please refund one of them. Thanks."
)


def _hours_ago(hours: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(hours=hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_neighbour(service, ticket_id: str, *, age_hours=None, outcome=None, body=BODY):
    overrides = {"ticket_id": ticket_id, "body": body}
    if age_hours is not None:
        overrides["received_at"] = _hours_ago(age_hours)
    service.run_ticket(make_ticket(**overrides))
    if outcome is not None:
        service.record_outcome(ticket_id, outcome)


def _outcome(resolved=True, summary="", by="A. Agent"):
    return AgentOutcome(
        final_category="billing",
        final_queue="billing_support",
        resolved=resolved,
        recorded_by=by,
        resolution_summary=summary,
    )


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
# resolution knowledge
# ---------------------------------------------------------------------------


def test_knowledge_aggregate_math_and_exclusion(service):
    _seed_neighbour(service, "K-1", age_hours=10, outcome=_outcome(True, "Refunded one charge."))
    _seed_neighbour(service, "K-2", age_hours=20, outcome=_outcome(True, "Credited the account."))
    _seed_neighbour(service, "K-3", age_hours=30, outcome=_outcome(False, "Duplicate was a renewal."))
    _seed_neighbour(service, "K-4")  # similar, no outcome — excluded, accounted
    service.run_ticket(make_ticket(ticket_id="K-9", body=BODY))

    payload = service.workbench_knowledge("K-9")
    agg = payload["aggregate"]
    assert agg["with_outcome"] == 3
    assert agg["without_outcome"] == 1
    assert agg["similar_total"] == 4
    assert agg["resolved_count"] == 2
    assert agg["resolved_rate"] == round(2 / 3, 4)
    assert agg["median_hours_from_intake"] == pytest.approx(20.0, abs=0.1)
    assert "excluded from these figures" in agg["note"]
    entries = {e["ticket_id"]: e for e in payload["entries"]}
    assert set(entries) == {"K-1", "K-2", "K-3"}
    assert entries["K-1"]["resolution_summary"] == "Refunded one charge."
    assert entries["K-1"]["hours_from_intake"] == pytest.approx(10.0, abs=0.1)
    assert entries["K-3"]["resolved"] is False


def test_knowledge_withholds_rate_below_the_floor(service):
    _seed_neighbour(service, "S-1", age_hours=5, outcome=_outcome(True))
    _seed_neighbour(service, "S-2", age_hours=6, outcome=_outcome(True))
    service.run_ticket(make_ticket(ticket_id="S-9", body=BODY))

    agg = service.workbench_knowledge("S-9")["aggregate"]
    assert agg["with_outcome"] == 2
    assert agg["resolved_rate"] is None
    assert agg["median_hours_from_intake"] is None
    assert "too few to quote" in agg["note"]


def test_knowledge_no_outcomes_yet(service):
    _seed_neighbour(service, "N-1")
    service.run_ticket(make_ticket(ticket_id="N-9", body=BODY))
    payload = service.workbench_knowledge("N-9")
    assert payload["entries"] == []
    agg = payload["aggregate"]
    assert agg["with_outcome"] == 0 and agg["without_outcome"] == 1
    assert "no similar ticket has a recorded outcome" in agg["note"]


def test_knowledge_hours_none_without_intake_stamp(service):
    _seed_neighbour(service, "H-1", outcome=_outcome(True))
    # A legacy record whose intake stamp is gone: the elapsed time
    # is honestly absent, never reconstructed.
    legacy = service.get_record("H-1")
    legacy.received_at = ""
    service.store.save(legacy)
    service.run_ticket(make_ticket(ticket_id="H-9", body=BODY))
    payload = service.workbench_knowledge("H-9")
    assert payload["entries"][0]["hours_from_intake"] is None


# ---------------------------------------------------------------------------
# playbook overlay
# ---------------------------------------------------------------------------


def _billing_run(service, ticket_id="P-1"):
    service.run_ticket(make_ticket(ticket_id=ticket_id, body=BODY))
    return service.get_record(ticket_id)


def test_playbook_view_and_check_off(service):
    _billing_run(service)
    view = service.workbench_playbook("P-1")
    assert view["available"] is True and view["category"] == "billing"
    assert view["checked_count"] == 0

    record = service.check_playbook_step("P-1", "b1", actor="A. Agent")
    assert record.playbook_checks[-1].action == "checked"
    assert record.playbook_checks[-1].by == "A. Agent"
    view = service.workbench_playbook("P-1")
    step = next(s for s in view["steps"] if s["id"] == "b1")
    assert step["checked"] is True and step["by"] == "A. Agent"

    service.check_playbook_step("P-1", "b1", actor="A. Agent", checked=False)
    view = service.workbench_playbook("P-1")
    step = next(s for s in view["steps"] if s["id"] == "b1")
    assert step["checked"] is False
    # the trail keeps both halves
    assert [c.action for c in service.get_record("P-1").playbook_checks] == [
        "checked",
        "unchecked",
    ]


def test_playbook_refusals(service):
    _billing_run(service)
    with pytest.raises(workbench.WorkbenchError):
        service.check_playbook_step("P-1", "b1", actor="  ")
    with pytest.raises(workbench.WorkbenchError):
        service.check_playbook_step("P-1", "nope", actor="A. Agent")
    service.check_playbook_step("P-1", "b1", actor="A. Agent")
    with pytest.raises(workbench.WorkbenchConflict) as excinfo:
        service.check_playbook_step("P-1", "b1", actor="A. Agent")
    assert excinfo.value.http_status == 409
    with pytest.raises(KeyError):
        service.check_playbook_step("MISSING", "b1", actor="A. Agent")


def test_playbook_not_applicable_blocks_checks_until_lifted(service):
    _billing_run(service)
    service.mark_playbook_na("P-1", actor="A. Agent", reason="enterprise contract — handled by the account team")
    view = service.workbench_playbook("P-1")
    assert view["not_applicable"]["reason"].startswith("enterprise contract")
    with pytest.raises(workbench.WorkbenchConflict):
        service.check_playbook_step("P-1", "b1", actor="A. Agent")
    with pytest.raises(workbench.WorkbenchConflict):
        service.mark_playbook_na("P-1", actor="A. Agent", reason="again")
    service.lift_playbook_na("P-1", actor="A. Agent")
    assert service.workbench_playbook("P-1")["not_applicable"] is None
    service.check_playbook_step("P-1", "b1", actor="A. Agent")  # works again


def test_category_without_playbook_says_so_plainly(service):
    service.run_ticket(
        make_ticket(
            ticket_id="F-1",
            subject="Please add dark mode to the dashboard",
            body="A dark mode option in the dashboard would really help our night-shift team. Please add it.",
        )
    )
    record = service.get_record("F-1")
    if record.category != "feature_request":
        pytest.skip(f"offline backend routed this as {record.category}, not feature_request")
    view = service.workbench_playbook("F-1")
    assert view["available"] is False
    assert "no playbook authored" in view["note"]
    with pytest.raises(workbench.WorkbenchError):
        service.check_playbook_step("F-1", "b1", actor="A. Agent")


# ---------------------------------------------------------------------------
# the working note
# ---------------------------------------------------------------------------


class _EngineeringEngine:
    """Fixed answers: a technical issue that needs engineering."""

    name = "stub"
    model = "stub-v1"

    def decide(self, state, questions, context=""):
        raw = valid_answers(
            category={"probabilities": {
                "technical_issue": 0.9, "billing": 0.02, "account_access": 0.02,
                "how_to": 0.015, "bug_report": 0.015, "feature_request": 0.01,
                "complaint": 0.01, "other": 0.01,
            }},
            needs_engineering={"probabilities": {"true": 0.9, "false": 0.1}},
        )
        return build_decision_set(
            triage_questions(), raw, backend=self.name, model=self.model, latency_ms=1.0
        )


def test_note_draft_carries_routing_reason_verbatim_and_blockers():
    service = fresh_service(engine=_EngineeringEngine())
    result = service.run_ticket(
        make_ticket(ticket_id="W-1", subject="API returns 500 on export", body="Export fails.")
    ).result
    draft = service.working_note_draft("W-1")["draft"]
    assert "INTERNAL WORKING NOTE" in draft
    assert result.composition.routing_reason in draft
    assert "waiting on engineering input" in draft
    assert "needs_engineering" in draft
    assert "never sent" in draft


def test_saved_note_lands_on_the_record(service):
    _billing_run(service)
    record = service.save_working_note("P-1", text="Called the customer; refund raised.", actor="A. Agent")
    assert record.working_note.text == "Called the customer; refund raised."
    assert record.working_note.saved_by == "A. Agent"
    payload = service.working_note_draft("P-1")
    assert payload["saved"]["text"] == "Called the customer; refund raised."
    with pytest.raises(workbench.WorkbenchError):
        service.save_working_note("P-1", text="   ", actor="A. Agent")
    with pytest.raises(workbench.WorkbenchError):
        service.save_working_note("P-1", text="x", actor="")


# ---------------------------------------------------------------------------
# the pins: triage + calibration untouched; export carries the blocks
# ---------------------------------------------------------------------------


def test_workbench_actions_leave_triage_and_sampling_untouched(service):
    _seed_neighbour(service, "U-1", age_hours=4, outcome=_outcome(True))
    _billing_run(service, "U-9")
    record = service.get_record("U-9")
    before = {
        "status": record.status,
        "composition": record.composition.model_dump(),
        "decision": record.decision,
        "draft": record.draft.model_dump() if record.draft else None,
        "answers": [a.model_dump() for a in record.answers],
    }
    samples_before = live_samples(service.store.all_records())

    service.check_playbook_step("U-9", "b2", actor="A. Agent")
    service.mark_playbook_na("U-9", actor="A. Agent", reason="second ticket — same handling")
    service.save_working_note("U-9", text="Internal: watching for the customer's reply.", actor="A. Agent")
    service.workbench("U-9")

    after = service.get_record("U-9")
    assert after.status == before["status"]
    assert after.composition.model_dump() == before["composition"]
    assert after.decision == before["decision"]
    assert (after.draft.model_dump() if after.draft else None) == before["draft"]
    assert [a.model_dump() for a in after.answers] == before["answers"]
    assert live_samples(service.store.all_records()) == samples_before


def test_customer_template_draft_untouched_by_workbench(service):
    # A how_to ticket on the offline backend earns the template draft
    # (the one customer-facing draft surface). Workbench actions
    # must not alter it — different surface, different name.
    result = service.run_ticket(
        make_ticket(
            ticket_id="D-1",
            subject="How do I authenticate API requests?",
            body="How do I authenticate API requests with a token? The quickstart mentions headers but not the token format.",
        )
    ).result
    if result.draft is None:
        pytest.skip("offline backend did not grant a template draft for this ticket")
    draft_before = result.draft.model_dump()
    service.check_playbook_step("D-1", "h1", actor="A. Agent")
    service.save_working_note("D-1", text="Internal note.", actor="A. Agent")
    after = service.get_record("D-1")
    assert after.draft is not None and after.draft.model_dump() == draft_before
    assert after.working_note is not None  # the surfaces coexist, separately


def test_export_carries_playbook_and_saved_note(service):
    _billing_run(service)
    service.check_playbook_step("P-1", "b1", actor="A. Agent")
    service.save_working_note("P-1", text="Refund raised, awaiting settlement.", actor="A. Agent")
    document = build_export(service.get_record("P-1"), exported_at="2026-10-11T00:00:00Z")
    assert document["playbook"]["checked_count"] == 1
    assert document["working_note"]["text"] == "Refund raised, awaiting settlement."
    text = render_text(document)
    assert "PLAYBOOK PROGRESS" in text
    assert "[x] b1:" in text
    assert "INTERNAL WORKING NOTE" in text
    assert "Refund raised, awaiting settlement." in text


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_workbench_api_round_trip(client, service):
    _seed_neighbour(service, "A-1", age_hours=8, outcome=_outcome(True, "Sorted."))
    _billing_run(service, "A-9")
    payload = client.get("/api/runs/A-9/workbench").json()
    assert payload["knowledge"]["aggregate"]["with_outcome"] == 1
    assert payload["playbook"]["available"] is True
    assert "INTERNAL WORKING NOTE" in payload["working_note"]["draft"]

    missing_actor = client.post(
        "/api/runs/A-9/workbench/playbook/check", json={"step_id": "b1", "actor": ""}
    )
    assert missing_actor.status_code == 422
    checked = client.post(
        "/api/runs/A-9/workbench/playbook/check", json={"step_id": "b1", "actor": "A. Agent"}
    )
    assert checked.status_code == 200
    conflict = client.post(
        "/api/runs/A-9/workbench/playbook/check", json={"step_id": "b1", "actor": "A. Agent"}
    )
    assert conflict.status_code == 409
    saved = client.post(
        "/api/runs/A-9/workbench/note", json={"text": "Internal note.", "actor": "A. Agent"}
    )
    assert saved.status_code == 200
    assert saved.json()["working_note"]["text"] == "Internal note."
    assert client.get("/api/runs/NOPE/workbench").status_code == 404


def test_outcome_api_accepts_resolution_summary(client, service):
    _billing_run(service, "A-5")
    response = client.post(
        "/api/runs/A-5/outcome",
        json={
            "final_category": "billing",
            "final_queue": "billing_support",
            "resolved": True,
            "by": "A. Agent",
            "resolution_summary": "Refunded the duplicate charge.",
        },
    )
    assert response.status_code == 200
    assert response.json()["outcome"]["resolution_summary"] == "Refunded the duplicate charge."
