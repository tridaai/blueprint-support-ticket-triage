"""Composition and guardrails: the math and the rules, pinned down."""

from __future__ import annotations

import pytest

from support_triage.composition import (
    PRIORITY_WEIGHTS,
    compose,
    priority_band,
    recommend_autonomy,
)
from support_triage.decisions import build_decision_set
from support_triage.guardrails import run_guardrails
from support_triage.questions import triage_questions

from helpers import score_dist, valid_answers


def decide_with(**overrides):
    return build_decision_set(triage_questions(), valid_answers(**overrides))


def test_weights_sum_to_one():
    assert sum(PRIORITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_priority_formula_uses_expected_scores_and_verify_probability():
    decisions = decide_with(
        urgency={"probabilities": score_dist(10, 1.0)},
        sla_breach_risk={"probabilities": score_dist(0, 1.0)},
        churn_risk={"probabilities": score_dist(0, 1.0)},
        sentiment={"probabilities": score_dist(10, 1.0)},
        is_abusive={"probabilities": {"true": 0.0, "false": 1.0}},
    )
    composed = compose(decisions)
    # urgency alone at max: 0.30 * 100
    assert composed.priority_score == pytest.approx(30.0)
    assert composed.contributions["urgency"] == pytest.approx(30.0)
    assert composed.outage_boost_applied is False


def test_outage_boost_is_added_and_capped():
    decisions = decide_with(
        urgency={"probabilities": score_dist(10, 1.0)},
        sla_breach_risk={"probabilities": score_dist(10, 1.0)},
        churn_risk={"probabilities": score_dist(10, 1.0)},
        sentiment={"probabilities": score_dist(0, 1.0)},
        is_outage_report={"probabilities": {"true": 0.9, "false": 0.1}},
    )
    composed = compose(decisions)
    assert composed.outage_boost_applied is True
    assert composed.priority_score == 100.0  # 30+25+20+15+~0, +15 boost, capped


def test_bands():
    assert priority_band(85) == "critical"
    assert priority_band(80) == "critical"
    assert priority_band(60) == "high"
    assert priority_band(40) == "medium"
    assert priority_band(10) == "low"


def test_routing_matrix_defaults():
    composed = compose(decide_with())
    assert composed.queue == "billing_support"
    assert composed.override == ""


def test_api_area_refines_technical_to_developer_support():
    answers = valid_answers()
    answers["category"] = {"probabilities": {"technical_issue": 0.8, **{o: 0.2 / 7 for o in ("billing", "account_access", "how_to", "bug_report", "feature_request", "complaint", "other")}}}
    answers["product_area"] = {"probabilities": {"api": 0.8, **{o: 0.2 / 6 for o in ("checkout", "mobile_app", "dashboard", "integrations", "account_admin", "other")}}}
    composed = compose(build_decision_set(triage_questions(), answers))
    assert composed.queue == "developer_support"


def test_legal_override_beats_outage_override():
    decisions = decide_with(
        mentions_legal={"probabilities": {"true": 0.9, "false": 0.1}},
        is_outage_report={"probabilities": {"true": 0.95, "false": 0.05}},
    )
    composed = compose(decisions)
    assert composed.queue == "compliance_review"
    assert composed.override == "legal"


def test_outage_override_beats_engineering_override():
    decisions = decide_with(
        is_outage_report={"probabilities": {"true": 0.9, "false": 0.1}},
        needs_engineering={"probabilities": {"true": 0.9, "false": 0.1}},
    )
    composed = compose(decisions)
    assert composed.queue == "incident_response"
    assert composed.override == "outage"


def test_engineering_override():
    decisions = decide_with(
        needs_engineering={"probabilities": {"true": 0.8, "false": 0.2}},
    )
    composed = compose(decisions)
    assert composed.queue == "engineering_escalation"


def test_autonomy_only_for_confident_flag_free_how_to():
    answers = valid_answers()
    answers["category"] = {"probabilities": {"how_to": 0.9, **{o: 0.1 / 7 for o in ("billing", "technical_issue", "account_access", "bug_report", "feature_request", "complaint", "other")}}}
    decisions = build_decision_set(triage_questions(), answers)
    composed = compose(decisions)
    rec = recommend_autonomy(
        decisions, composed, guardrails_passed=True,
        requires_human_review=False, is_duplicate=False,
    )
    assert rec.eligible is True
    blocked = recommend_autonomy(
        decisions, composed, guardrails_passed=True,
        requires_human_review=False, is_duplicate=True,
    )
    assert blocked.eligible is False
    assert any("duplicate" in b for b in blocked.blockers)


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------

KEYS = [q.key for q in triage_questions()]


def test_guardrails_pass_on_a_decisive_clean_set():
    outcome = run_guardrails(decide_with(), KEYS)
    assert outcome.passed
    assert outcome.requires_human_review is False


def test_low_category_confidence_forces_human_review():
    answers = valid_answers()
    answers["category"] = {
        "probabilities": {o: 1 / 8 for o in
                          ("billing", "technical_issue", "account_access", "how_to",
                           "bug_report", "feature_request", "complaint", "other")}
    }
    outcome = run_guardrails(build_decision_set(triage_questions(), answers), KEYS)
    assert outcome.requires_human_review is True
    assert any("category confidence" in r for r in outcome.human_review_reasons)


def test_undecided_verify_forces_human_review():
    decisions = decide_with(is_abusive={"probabilities": {"true": 0.5, "false": 0.5}})
    outcome = run_guardrails(decisions, KEYS)
    assert outcome.requires_human_review is True
    assert any("is_abusive" in r for r in outcome.human_review_reasons)


def test_decisive_verify_does_not_force_review():
    decisions = decide_with(is_abusive={"probabilities": {"true": 0.7, "false": 0.3}})
    outcome = run_guardrails(decisions, KEYS)
    assert outcome.requires_human_review is False


def test_safety_flags_are_recorded_and_named():
    decisions = decide_with(contains_pii={"probabilities": {"true": 0.95, "false": 0.05}})
    outcome = run_guardrails(decisions, KEYS)
    safety = next(c for c in outcome.checks if c.name == "safety_flags")
    assert safety.passed is True  # flags are information, not malfunction
    assert "contains_pii" in safety.detail
