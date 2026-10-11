"""Outcome capture + live calibration: hand-computed fixtures."""

from __future__ import annotations

import pytest

from support_triage.livecal import (
    live_calibration,
    normalise_score_truth,
    normalise_verify_truth,
    validate_final_category,
)
from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    QuestionAnswer,
    TicketResult,
)


def record_with_answers(ticket_id, answers, **overrides) -> TicketResult:
    base = dict(
        ticket_id=ticket_id,
        status="triaged",
        category="billing",
        answers=answers,
        composition=CompositionInfo(queue="billing_support"),
    )
    base.update(overrides)
    return TicketResult(**base)


def verify_answer(key, p_true) -> QuestionAnswer:
    return QuestionAnswer(
        key=key, kind="verify", value=p_true,
        probabilities={"true": p_true, "false": 1 - p_true},
        confidence=max(p_true, 1 - p_true),
    )


def category_answer(dist) -> QuestionAnswer:
    top = max(dist, key=lambda k: dist[k])
    return QuestionAnswer(
        key="category", kind="choice", value=top,
        probabilities=dist, confidence=dist[top],
    )


# ---------------------------------------------------------------------------
# truth normalisation
# ---------------------------------------------------------------------------


def test_verify_truth_aliases_normalise_to_question_keys():
    assert normalise_verify_truth({"was_pii": True, "was_outage": False}) == {
        "contains_pii": True,
        "is_outage_report": False,
    }
    assert normalise_verify_truth({"contains_pii": True}) == {"contains_pii": True}
    assert normalise_verify_truth({}) == {}


def test_verify_truth_unknown_key_is_an_error_not_a_drop():
    with pytest.raises(ValueError, match="unknown verify truth key"):
        normalise_verify_truth({"was_rude": True})


def test_score_truth_validation():
    assert normalise_score_truth({"urgency": 8}) == {"urgency": 8}
    with pytest.raises(ValueError, match="unknown score truth key"):
        normalise_score_truth({"vibes": 3})
    with pytest.raises(ValueError, match="0..10"):
        normalise_score_truth({"urgency": 11})


def test_final_category_validation():
    assert validate_final_category("billing") == "billing"
    assert validate_final_category("") == ""
    with pytest.raises(ValueError, match="not one of"):
        validate_final_category("vibes")


# ---------------------------------------------------------------------------
# live calibration math
# ---------------------------------------------------------------------------


def test_no_outcomes_is_an_honest_empty_state():
    report = live_calibration([])
    assert report["available"] is False
    assert "no agent outcomes recorded yet" in report["note"]
    assert report["runs_with_outcomes"] == 0
    assert report["overall_brier"] is None
    assert report["category_accuracy"] is None

    bare = record_with_answers("L-0", [verify_answer("contains_pii", 0.9)])
    report = live_calibration([bare])
    assert report["available"] is False
    assert report["runs_total"] == 1


def test_verify_brier_and_bins_are_hand_computed():
    records = [
        record_with_answers(
            "L-1", [verify_answer("contains_pii", 0.8)],
            outcome=AgentOutcome(verify_truth={"contains_pii": True}),
        ),
        record_with_answers(
            "L-2", [verify_answer("contains_pii", 0.3)],
            outcome=AgentOutcome(verify_truth={"contains_pii": False}),
        ),
    ]
    report = live_calibration(records)
    assert report["available"] is True
    assert report["runs_with_outcomes"] == 2
    entry = report["per_question"]["contains_pii"]
    assert entry["kind"] == "verify" and entry["n"] == 2
    # (0.8-1)^2 = 0.04 and (0.3-0)^2 = 0.09 -> mean 0.065
    assert entry["brier"] == pytest.approx(0.065)
    bins = {b["bin"]: b for b in entry["bins"]}
    assert bins["0.8-0.9"]["n"] == 1
    assert bins["0.8-0.9"]["mean_predicted"] == 0.8
    assert bins["0.8-0.9"]["empirical"] == 1.0
    assert bins["0.3-0.4"]["empirical"] == 0.0


def test_category_one_vs_rest_bins_and_accuracy():
    records = [
        record_with_answers(
            "L-3", [category_answer({"billing": 0.6, "other": 0.4})],
            outcome=AgentOutcome(final_category="billing", final_queue="billing_support"),
        ),
        record_with_answers(
            "L-4", [category_answer({"billing": 0.7, "other": 0.3})],
            outcome=AgentOutcome(final_category="other", final_queue="general_support"),
        ),
    ]
    report = live_calibration(records)
    entry = report["per_question"]["category"]
    # multiclass Brier: case1 (0.6-1)^2+(0.4)^2 = 0.32; case2 (0.7)^2+(0.3-1)^2 = 0.98
    assert entry["brier"] == pytest.approx(0.65)
    bins = {b["bin"]: b for b in entry["bins"]}
    assert bins["0.6-0.7"]["empirical"] == 1.0  # chosen value was right
    assert bins["0.7-0.8"]["empirical"] == 0.0  # chosen value was wrong
    assert report["category_accuracy"] == 0.5
    assert report["queue_accuracy"] == 0.5


def test_score_high_event_calibration():
    dist = {"9": 0.5, "5": 0.5}
    records = [
        record_with_answers(
            "L-5",
            [QuestionAnswer(key="urgency", kind="score", value=7.0,
                            probabilities=dist, confidence=0.5)],
            outcome=AgentOutcome(score_truth={"urgency": 9}, resolved=True),
        )
    ]
    report = live_calibration(records)
    entry = report["per_question"]["urgency"]
    assert entry["kind"] == "score"
    # P(>=7) = 0.5, truth high -> (0.5-1)^2 = 0.25; expected 7.0 vs 9 -> MAE 2.0
    assert entry["high_event_brier"] == pytest.approx(0.25)
    assert entry["mae"] == pytest.approx(2.0)
    assert report["resolution_rate"] == 1.0


def test_partial_truths_only_score_what_was_judged():
    records = [
        record_with_answers(
            "L-6",
            [verify_answer("contains_pii", 0.9), verify_answer("is_abusive", 0.9),
             category_answer({"billing": 1.0})],
            outcome=AgentOutcome(verify_truth={"contains_pii": True}),
        )
    ]
    report = live_calibration(records)
    assert set(report["per_question"]) == {"contains_pii"}
    # final_category left blank: no accuracy claim either way
    assert report["category_accuracy"] is None
    assert report["queue_accuracy"] is None


def test_service_records_and_replaces_outcomes():
    from helpers import fresh_service, make_ticket

    service = fresh_service()
    service.run_ticket(make_ticket(ticket_id="OC-1"))
    outcome = AgentOutcome(final_category="billing", resolved=False)
    record = service.record_outcome("OC-1", outcome)
    assert record.outcome is not None
    assert record.outcome.recorded_at  # stamped by the service
    corrected = AgentOutcome(final_category="how_to", resolved=True)
    record = service.record_outcome("OC-1", corrected)
    assert record.outcome.final_category == "how_to"
    assert service.get_record("OC-1").outcome.resolved is True
    with pytest.raises(KeyError):
        service.record_outcome("NOPE", outcome)
