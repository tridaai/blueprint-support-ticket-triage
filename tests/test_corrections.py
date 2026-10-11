"""Operator corrections: overrule the grouping, confirm the duplicate.

The pins:

- corrections are an OVERLAY: the computed thread_id and every
  decision field on a record are byte-identical after attach /
  detach / merge / mark — only the correction entries are added;
- attach and detach move a whole reopen chain (a chain is never
  split), and a reopen created after a correction joins the
  corrected thread;
- refusals: detach when alone, attach to the run's own thread,
  merge into itself (409-shaped conflicts); cross-customer attach /
  merge / duplicate marks (422-shaped); unknown ids are KeyErrors;
- a confirmed duplicate is excluded from live calibration samples
  and the drift stream — with the exclusion accounted in both
  payloads — while volume stats still count the run;
- un-marking restores sampling and keeps both halves of the audit
  trail; a second mark while one stands is a 409 naming the first;
- replay identity holds over corrected runs (moved == 0, and the
  report equals the pre-correction report).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api, corrections
from support_triage.schemas import AgentOutcome

from helpers import fresh_service, make_ticket


BILLING_BODY = (
    "I was charged twice for the April subscription — two identical payments of $49 "
    "left my card. Please refund one of them."
)

FOLLOW_UP_BODY = (
    "Following up on my earlier message: I was charged twice for the April "
    "subscription, and the refund for one of the two payments still has not "
    "arrived on my card."
)

HOW_TO_BODY = (
    "How do I invite a new team member to our workspace and set their user role? "
    "I can't find the team members page in the account settings."
)


def _ticket(ticket_id, subject, body, email="priya@example.com",
            when="2026-10-01T09:00:00Z", name="Priya Nair"):
    return make_ticket(
        ticket_id=ticket_id, subject=subject, body=body,
        sender_name=name, sender_email=email, received_at=when,
    )


def _two_threads_service():
    """A billing thread (two runs) and a separate how-to thread."""
    service = fresh_service()
    service.run_ticket(_ticket("T-1", "Charged twice for the April subscription", BILLING_BODY))
    service.run_ticket(_ticket("T-2", "Still waiting on the April double-charge refund",
                               FOLLOW_UP_BODY, when="2026-10-03T09:00:00Z"))
    service.run_ticket(_ticket("U-1", "How do I add a team member?", HOW_TO_BODY,
                               when="2026-10-03T10:00:00Z"))
    return service


def _decision_dump(record) -> str:
    """The record with the correction overlay removed — what must
    never change when an operator corrects a grouping or a mark."""
    payload = record.model_dump()
    payload.pop("thread_corrections", None)
    payload.pop("duplicate_marks", None)
    payload.pop("duplicate_of", None)
    return repr(sorted(payload.items()))


# --- thread corrections -----------------------------------------------------


def test_attach_moves_the_chain_and_preserves_the_computed_grouping():
    service = _two_threads_service()
    before = {tid: _decision_dump(service.get_record(tid)) for tid in ("T-1", "T-2", "U-1")}

    view = service.attach_run_to_thread(
        "T-2", "thr-U-1", actor="agent-1", reason="same conversation after all"
    )
    assert view["thread_id"] == "thr-U-1"
    assert {row["ticket_id"] for row in view["runs"]} == {"T-2", "U-1"}

    listing = {t["thread_id"]: t for t in service.list_threads()["threads"]}
    assert set(listing) == {"thr-T-1", "thr-U-1"}
    assert listing["thr-U-1"]["corrected"] is True
    assert listing["thr-T-1"]["corrected"] is False

    moved = service.get_record("T-2")
    # The computed grouping is preserved underneath the overlay.
    assert moved.thread_id == "thr-T-1"
    (entry,) = moved.thread_corrections
    assert entry.action == "attach"
    assert entry.from_thread_id == "thr-T-1"
    assert entry.to_thread_id == "thr-U-1"
    assert entry.by == "agent-1" and entry.reason == "same conversation after all"
    # Attach is chain-scoped: T-1 is a separate chain that merely
    # shared the computed thread, and it did NOT move.
    assert service.get_record("T-1").thread_corrections == []
    assert service.run_thread("T-1")["thread_id"] == "thr-T-1"
    # …and nothing about any decision changed, on any record.
    after = {tid: _decision_dump(service.get_record(tid)) for tid in before}
    assert after == before
    # The corrected row carries the lineage, computed id visible.
    row = next(r for r in view["runs"] if r["ticket_id"] == "T-2")
    assert row["grouping"]["corrected"] is True
    assert row["grouping"]["computed_thread_id"] == "thr-T-1"
    assert "corrected by agent-1" in row["grouping"]["lineage"]


def test_detach_moves_the_whole_reopen_chain_never_part_of_it():
    service = fresh_service()
    service.run_ticket(_ticket("C-1", "Charged twice for the April subscription", BILLING_BODY))
    service.record_outcome(
        "C-1",
        AgentOutcome(final_category="billing", final_queue="billing_support",
                     resolved=False, recorded_by="agent-1"),
    )
    child = service.reopen_run("C-1", message="The refund is still missing.", actor="agent-1").result
    # A second chain lands in the same computed thread.
    service.run_ticket(_ticket("C-9", "Still waiting on the April double-charge refund",
                               FOLLOW_UP_BODY, when="2026-10-05T09:00:00Z"))
    assert service.run_thread("C-1")["run_count"] == 3

    # Detach addressed at the reopen (a chain member) moves the chain.
    view = service.detach_run(child.ticket_id, actor="agent-2",
                              reason="a different card and a different charge")
    assert view["thread_id"] == f"thr-op-C-1"
    assert {row["ticket_id"] for row in view["runs"]} == {"C-1", child.ticket_id}
    # The other chain stayed behind; neither record's computed id moved.
    assert service.run_thread("C-9")["thread_id"] == "thr-C-1"
    assert service.get_record(child.ticket_id).thread_id == "thr-C-1"
    assert service.get_record("C-1").thread_corrections[-1].action == "detach"


def test_reopen_after_a_correction_joins_the_corrected_thread():
    service = _two_threads_service()
    service.detach_run("T-1", actor="agent-1", reason="separate card, separate dispute")
    service.record_outcome(
        "T-1",
        AgentOutcome(final_category="billing", final_queue="billing_support",
                     resolved=False, recorded_by="agent-1"),
    )
    child = service.reopen_run("T-1", message="Still no refund.", actor="agent-1").result
    assert service.run_thread(child.ticket_id)["thread_id"] == "thr-op-T-1"
    assert child.thread_id == "thr-op-T-1"


def test_merge_folds_the_source_thread_into_the_target():
    service = _two_threads_service()
    view = service.merge_threads(
        "thr-U-1", "thr-T-1", actor="agent-1", reason="one conversation, two phrasings"
    )
    assert view["thread_id"] == "thr-T-1"
    assert {row["ticket_id"] for row in view["runs"]} == {"T-1", "T-2", "U-1"}
    moved = service.get_record("U-1")
    assert moved.thread_id == "thr-U-1"  # computed, preserved
    assert moved.thread_corrections[-1].action == "merge"
    assert service.list_threads()["threads"][0]["corrected"] is True


def test_thread_correction_refusals():
    service = _two_threads_service()
    # Detach a chain that is alone in its thread: nothing to correct.
    with pytest.raises(corrections.CorrectionConflict):
        service.detach_run("U-1", actor="a", reason="r")
    # Attach to the thread the run is already in.
    with pytest.raises(corrections.CorrectionConflict):
        service.attach_run_to_thread("T-1", "thr-T-1", actor="a", reason="r")
    # Merge a thread into itself.
    with pytest.raises(corrections.CorrectionConflict):
        service.merge_threads("thr-T-1", "thr-T-1", actor="a", reason="r")
    # Unknown ids are KeyErrors (the API's 404).
    with pytest.raises(KeyError):
        service.attach_run_to_thread("T-1", "thr-NOPE", actor="a", reason="r")
    with pytest.raises(KeyError):
        service.attach_run_to_thread("NOPE", "thr-T-1", actor="a", reason="r")
    with pytest.raises(KeyError):
        service.merge_threads("thr-NOPE", "thr-T-1", actor="a", reason="r")
    # Actor and reason are not optional.
    with pytest.raises(corrections.CorrectionError):
        service.detach_run("T-1", actor="", reason="r")
    with pytest.raises(corrections.CorrectionError):
        service.merge_threads("thr-U-1", "thr-T-1", actor="a", reason="  ")


def test_corrections_never_cross_customers():
    service = _two_threads_service()
    service.run_ticket(_ticket("X-1", "Charged twice for the April subscription", BILLING_BODY,
                               email="oscar@example.com", name="Oscar Reyes",
                               when="2026-10-04T09:00:00Z"))
    with pytest.raises(corrections.CorrectionError):
        service.attach_run_to_thread("X-1", "thr-T-1", actor="a", reason="r")
    with pytest.raises(corrections.CorrectionError):
        service.merge_threads("thr-X-1", "thr-T-1", actor="a", reason="r")
    # Nothing moved.
    assert service.run_thread("X-1")["thread_id"] == "thr-X-1"


def test_replay_identity_holds_over_corrected_runs():
    service = _two_threads_service()
    baseline = service.policy_replay()
    assert baseline["moved"] == 0
    service.attach_run_to_thread("U-1", "thr-T-1", actor="agent-1", reason="same conversation")
    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="resent the same ticket")
    after = service.policy_replay()
    assert after["moved"] == 0
    assert after["moved_runs"] == baseline["moved_runs"]
    assert after["runs_replayed"] == baseline["runs_replayed"]


# --- confirmed duplicates -----------------------------------------------------


def _outcome(category="billing", queue="billing_support", resolved=True):
    return AgentOutcome(final_category=category, final_queue=queue,
                        resolved=resolved, recorded_by="agent-1",
                        verify_truth={"contains_pii": False})


def test_mark_duplicate_lands_on_the_duplicate_only():
    service = _two_threads_service()
    survivor_before = service.get_record("T-1").model_dump_json()
    record = service.mark_duplicate(
        "T-2", "T-1", actor="agent-1", reason="customer re-sent the same ticket"
    )
    assert record.duplicate_of == "T-1"
    (mark,) = record.duplicate_marks
    assert mark.active and mark.marked_by == "agent-1"
    # The surviving run is byte-identical.
    assert service.get_record("T-1").model_dump_json() == survivor_before
    # The decision on the duplicate is untouched.
    assert _decision_dump(service.get_record("T-2"))


def test_double_mark_is_a_conflict_naming_the_first_mark():
    service = _two_threads_service()
    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="resent")
    with pytest.raises(corrections.CorrectionConflict) as caught:
        service.mark_duplicate("T-2", "U-1", actor="agent-2", reason="actually this one")
    assert "already marked as a duplicate of T-1" in str(caught.value)
    assert caught.value.http_status == 409


def test_mark_refusals_cross_customer_self_and_unknown():
    service = _two_threads_service()
    service.run_ticket(_ticket("X-1", "Charged twice", BILLING_BODY,
                               email="oscar@example.com", name="Oscar Reyes",
                               when="2026-10-04T09:00:00Z"))
    with pytest.raises(corrections.CorrectionError):
        service.mark_duplicate("X-1", "T-1", actor="a", reason="r")  # different customer
    with pytest.raises(corrections.CorrectionError):
        service.mark_duplicate("T-1", "T-1", actor="a", reason="r")  # itself
    with pytest.raises(KeyError):
        service.mark_duplicate("T-1", "NOPE", actor="a", reason="r")
    with pytest.raises(KeyError):
        service.mark_duplicate("NOPE", "T-1", actor="a", reason="r")


def test_confirmed_duplicate_leaves_the_samples_not_the_volume():
    service = _two_threads_service()
    service.record_outcome("T-1", _outcome())
    service.record_outcome("T-2", _outcome())
    live_before = service.live_calibration()
    assert live_before["cases"] == 2 and live_before["runs_sampled"] == 2

    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="resent the same ticket")
    live = service.live_calibration()
    # Out of the samples, with the accounting on the payload…
    assert live["cases"] == 1 and live["runs_sampled"] == 1
    assert live["runs_with_outcomes"] == 2  # still a labelled run
    (excluded,) = live["duplicates_excluded"]
    assert excluded["ticket_id"] == "T-2" and excluded["duplicate_of"] == "T-1"
    assert "excluded from sampling" in live["note"]
    # …and out of the drift stream, accounted there too.
    drift = service.stats_drift(None)
    assert drift["window"]["outcomes_total"] == 1
    assert drift["excluded_duplicates"][0]["ticket_id"] == "T-2"
    # …but volume still counts it: it was a ticket.
    overview = service.stats_overview()
    assert overview["total_runs"] == 3
    assert overview["confirmed_duplicates"] == 1
    assert service.metrics()["total_tickets"] == 3


def test_unmark_restores_sampling_and_keeps_the_audit_trail():
    service = _two_threads_service()
    service.record_outcome("T-1", _outcome())
    service.record_outcome("T-2", _outcome())
    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="resent")
    record = service.unmark_duplicate(
        "T-2", actor="agent-2", reason="a follow-up, not a resend"
    )
    assert record.duplicate_of == ""
    (mark,) = record.duplicate_marks
    assert not mark.active
    assert mark.unmarked_by == "agent-2" and mark.unmark_reason == "a follow-up, not a resend"
    assert mark.marked_by == "agent-1"  # both halves survive
    live = service.live_calibration()
    assert live["cases"] == 2 and live["duplicates_excluded"] == []
    # Lifting a mark that does not stand is a conflict.
    with pytest.raises(corrections.CorrectionConflict):
        service.unmark_duplicate("T-2", actor="a", reason="r")
    # And the run can be marked again afterwards.
    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="confirmed twice over")
    assert service.get_record("T-2").duplicate_of == "T-1"


# --- the export ----------------------------------------------------------------


def test_export_carries_the_corrections_in_both_renderings():
    from support_triage.exports import build_export, render_text

    service = _two_threads_service()
    service.attach_run_to_thread("U-1", "thr-T-1", actor="agent-1",
                                 reason="same workspace dispute")
    service.mark_duplicate("T-2", "T-1", actor="agent-1", reason="resent the same ticket")

    moved = build_export(service.get_record("U-1"), exported_at="2026-10-11T00:00:00Z")
    assert moved["grouping"]["corrected"] is True
    assert moved["grouping"]["computed_thread_id"] == "thr-U-1"
    assert moved["grouping"]["thread_id"] == "thr-T-1"

    dup = build_export(service.get_record("T-2"), exported_at="2026-10-11T00:00:00Z")
    assert dup["confirmed_duplicate"]["duplicate_of"] == "T-1"
    assert dup["confirmed_duplicate"]["active"] is True

    text = render_text(dup)
    assert "CONFIRMED DUPLICATE (operator)" in text
    assert "duplicate of T-1 by agent-1" in text
    assert "excluded from calibration sampling" in text
    moved_text = render_text(moved)
    assert "GROUPING (operator-corrected)" in moved_text
    assert "thr-U-1" in moved_text and "same workspace dispute" in moved_text
    # An uncorrected, unmarked run grows no new blocks.
    plain = build_export(service.get_record("T-1"), exported_at="2026-10-11T00:00:00Z")
    assert plain["grouping"]["corrected"] is False
    assert plain["confirmed_duplicate"] is None
    assert "CONFIRMED DUPLICATE" not in render_text(plain)


# --- the HTTP surface -----------------------------------------------------------


@pytest.fixture()
def client():
    service = fresh_service()
    api.set_service(service)
    yield TestClient(api.app), service
    api.set_service(None)


def _seed_http(service):
    service.run_ticket(_ticket("H-1", "Charged twice for the April subscription", BILLING_BODY))
    service.run_ticket(_ticket("H-2", "Still waiting on the April double-charge refund",
                               FOLLOW_UP_BODY, when="2026-10-03T09:00:00Z"))
    service.run_ticket(_ticket("H-3", "How do I add a team member?", HOW_TO_BODY,
                               when="2026-10-03T10:00:00Z"))


def test_api_thread_corrections(client):
    http, service = client
    _seed_http(service)

    attached = http.post(
        "/api/runs/H-3/thread/attach",
        json={"thread_id": "thr-H-1", "actor": "agent-1", "reason": "same conversation"},
    )
    assert attached.status_code == 200
    assert attached.json()["thread_id"] == "thr-H-1"

    detached = http.post(
        "/api/runs/H-3/thread/detach", json={"actor": "agent-1", "reason": "separate after all"}
    )
    assert detached.status_code == 200
    assert detached.json()["thread_id"] == "thr-op-H-3"

    merged = http.post(
        "/api/threads/merge",
        json={"source_thread_id": "thr-op-H-3", "target_thread_id": "thr-H-1",
              "actor": "agent-1", "reason": "and back again"},
    )
    assert merged.status_code == 200
    assert merged.json()["run_count"] == 3

    # Refusals map onto statuses: 422 no reason, 404 unknown thread/run.
    assert http.post("/api/runs/H-3/thread/detach",
                     json={"actor": "a", "reason": ""}).status_code == 422
    assert http.post("/api/runs/H-3/thread/attach",
                     json={"thread_id": "thr-NOPE", "actor": "a", "reason": "r"}).status_code == 404
    assert http.post("/api/runs/NOPE/thread/detach",
                     json={"actor": "a", "reason": "r"}).status_code == 404
    # Detach of a chain alone in the merged thread is fine (others share
    # it) — but merging a thread into itself is a 409.
    assert http.post("/api/threads/merge",
                     json={"source_thread_id": "thr-H-1", "target_thread_id": "thr-H-1",
                           "actor": "a", "reason": "r"}).status_code == 409


def test_api_duplicate_mark_and_unmark(client):
    http, service = client
    _seed_http(service)

    marked = http.post(
        "/api/runs/H-2/duplicate",
        json={"duplicate_of": "H-1", "actor": "agent-1", "reason": "resent"},
    )
    assert marked.status_code == 200
    assert marked.json()["duplicate_of"] == "H-1"

    again = http.post(
        "/api/runs/H-2/duplicate",
        json={"duplicate_of": "H-1", "actor": "agent-1", "reason": "resent"},
    )
    assert again.status_code == 409
    assert "already marked as a duplicate of H-1" in again.json()["detail"]

    assert http.post("/api/runs/H-2/duplicate",
                     json={"duplicate_of": "H-1", "actor": "", "reason": "r"}).status_code == 422
    assert http.post("/api/runs/NOPE/duplicate",
                     json={"duplicate_of": "H-1", "actor": "a", "reason": "r"}).status_code == 404

    lifted = http.post(
        "/api/runs/H-2/duplicate/unmark",
        json={"actor": "agent-2", "reason": "a follow-up, not a resend"},
    )
    assert lifted.status_code == 200
    assert lifted.json()["duplicate_of"] == ""
    assert lifted.json()["duplicate_marks"][0]["unmarked_by"] == "agent-2"
