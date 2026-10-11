"""The decision port: validation is code, and it is strict."""

from __future__ import annotations

import pytest

from support_triage.decisions import (
    ChoiceQuestion,
    DecisionValidationError,
    ScoreQuestion,
    VerifyQuestion,
    build_decision_set,
)
from support_triage.questions import triage_questions

from helpers import score_dist, valid_answers


def test_choice_value_is_argmax_and_confidence_is_its_probability():
    questions = triage_questions()
    result = build_decision_set(questions, valid_answers())
    category = result.get("category")
    assert category.value == "billing"
    assert category.confidence == pytest.approx(0.7)
    assert result.choice("category") == "billing"


def test_score_value_is_expected_value_not_modal():
    questions = triage_questions()
    answers = valid_answers(urgency={"probabilities": score_dist(8, 0.5)})
    result = build_decision_set(questions, answers)
    # distribution: 0.5 on 8, 0.05 on each other score -> 4 + 0.05 * 47 = 6.35
    assert result.score("urgency") == pytest.approx(6.35, abs=0.05)
    assert result.get("urgency").confidence == pytest.approx(0.5)


def test_verify_value_is_probability_of_true():
    questions = triage_questions()
    answers = valid_answers(contains_pii={"probabilities": {"true": 0.9, "false": 0.1}})
    result = build_decision_set(questions, answers)
    assert result.probability("contains_pii") == pytest.approx(0.9)
    assert result.confidence("contains_pii") == pytest.approx(0.9)


def test_verify_accepts_bare_probability_form():
    questions = triage_questions()
    answers = valid_answers(mentions_legal={"probabilities": {"probability": 0.8}})
    result = build_decision_set(questions, answers)
    assert result.probability("mentions_legal") == pytest.approx(0.8)
    assert result.get("mentions_legal").probabilities["false"] == pytest.approx(0.2)


def test_missing_question_is_rejected():
    questions = triage_questions()
    answers = valid_answers()
    del answers["urgency"]
    with pytest.raises(DecisionValidationError, match="missing"):
        build_decision_set(questions, answers)


def test_unknown_question_is_rejected():
    questions = triage_questions()
    answers = valid_answers()
    answers["mood"] = {"probabilities": {"true": 0.5, "false": 0.5}}
    with pytest.raises(DecisionValidationError, match="unexpected"):
        build_decision_set(questions, answers)


def test_distribution_must_sum_to_one_within_tolerance():
    questions = triage_questions()
    answers = valid_answers()
    answers["category"]["probabilities"]["billing"] = 0.4  # sums to 0.7
    with pytest.raises(DecisionValidationError, match="sum"):
        build_decision_set(questions, answers)


def test_small_sum_drift_is_renormalised_not_rejected():
    questions = triage_questions()
    answers = valid_answers()
    dist = answers["category"]["probabilities"]
    dist["billing"] = 0.71  # sums to 1.01 — inside tolerance
    result = build_decision_set(questions, answers)
    assert sum(result.get("category").probabilities.values()) == pytest.approx(1.0)


def test_probability_outside_unit_interval_is_rejected():
    questions = triage_questions()
    answers = valid_answers()
    answers["is_abusive"] = {"probabilities": {"true": 1.4, "false": -0.4}}
    with pytest.raises(DecisionValidationError, match="outside"):
        build_decision_set(questions, answers)


def test_choice_distribution_must_cover_exactly_the_options():
    questions = triage_questions()
    answers = valid_answers()
    del answers["category"]["probabilities"]["other"]
    with pytest.raises(DecisionValidationError, match="keys mismatch"):
        build_decision_set(questions, answers)


def test_score_distribution_must_cover_the_whole_scale():
    questions = triage_questions()
    answers = valid_answers()
    answers["urgency"] = {"probabilities": {"0": 0.5, "10": 0.5}}
    with pytest.raises(DecisionValidationError, match="keys mismatch"):
        build_decision_set(questions, answers)


def test_question_primitives_carry_their_contract():
    choice = ChoiceQuestion(key="c", prompt="pick", options=("a", "b"))
    score = ScoreQuestion(key="s", prompt="rate", rubric="0 none, 10 max")
    verify = VerifyQuestion(key="v", prompt="true?", statement="It is so.")
    assert choice.kind == "choice" and score.kind == "score" and verify.kind == "verify"
    assert score.minimum == 0 and score.maximum == 10


def test_triage_set_shape_is_eleven_questions():
    questions = triage_questions()
    kinds = [q.kind for q in questions]
    assert len(questions) == 11
    assert kinds.count("choice") == 2
    assert kinds.count("score") == 4
    assert kinds.count("verify") == 5
    keys = [q.key for q in questions]
    assert keys == [
        "category", "product_area", "urgency", "sentiment", "churn_risk",
        "sla_breach_risk", "contains_pii", "is_abusive", "mentions_legal",
        "is_outage_report", "needs_engineering",
    ]
