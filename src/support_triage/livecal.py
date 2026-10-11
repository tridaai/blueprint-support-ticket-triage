"""Live calibration: score yesterday's probabilities against outcomes.

The eval pack (``evals/run_calibration.py``) measures calibration
against authored golden truths. This module measures the same thing
against the ground truth that matters in production: what the agent
who handled the ticket finally found. An agent records that truth via
``POST /api/runs/{id}/outcome`` (final category and queue, whether it
resolved, and — where the agent can judge — per-question truth for
the verify and score questions). Records carrying an outcome become
labelled samples for :func:`support_triage.calibration.calibration_report`,
fed in exactly the eval runner's sample shape and binned by exactly
the eval runner's code, so live and eval figures are comparable.

Truth keys accept agent-facing aliases (``was_pii``) as well as the
question keys themselves (``contains_pii``); unknown keys are an
error, never silently dropped — a truth nobody reads is worse than
no truth.
"""

from __future__ import annotations

from .calibration import calibration_report
from .questions import CATEGORIES
from .schemas import TicketResult

VERIFY_KEYS = (
    "contains_pii",
    "is_abusive",
    "mentions_legal",
    "is_outage_report",
    "needs_engineering",
)

SCORE_KEYS = ("urgency", "sentiment", "churn_risk", "sla_breach_risk")

VERIFY_TRUTH_ALIASES = {
    "was_pii": "contains_pii",
    "contains_pii": "contains_pii",
    "had_pii": "contains_pii",
    "was_abusive": "is_abusive",
    "is_abusive": "is_abusive",
    "was_legal": "mentions_legal",
    "was_legal_matter": "mentions_legal",
    "mentions_legal": "mentions_legal",
    "raised_legal": "mentions_legal",
    "was_outage": "is_outage_report",
    "was_outage_report": "is_outage_report",
    "is_outage_report": "is_outage_report",
    "needed_engineering": "needs_engineering",
    "needs_engineering": "needs_engineering",
    "required_engineering": "needs_engineering",
}


def normalise_verify_truth(raw: dict) -> dict[str, bool]:
    """Map agent-facing truth keys onto question keys (strict)."""
    normalised: dict[str, bool] = {}
    for key, value in (raw or {}).items():
        target = VERIFY_TRUTH_ALIASES.get(str(key))
        if target is None:
            raise ValueError(
                f"unknown verify truth key {key!r}; expected one of "
                f"{sorted(set(VERIFY_TRUTH_ALIASES))}"
            )
        normalised[target] = bool(value)
    return normalised


def normalise_score_truth(raw: dict) -> dict[str, int]:
    """Validate score truths: known question keys, integers 0..10."""
    normalised: dict[str, int] = {}
    for key, value in (raw or {}).items():
        if key not in SCORE_KEYS:
            raise ValueError(
                f"unknown score truth key {key!r}; expected one of {list(SCORE_KEYS)}"
            )
        score = int(value)
        if not 0 <= score <= 10:
            raise ValueError(f"score truth for {key!r} must be 0..10, got {value!r}")
        normalised[key] = score
    return normalised


def validate_final_category(category: str) -> str:
    category = (category or "").strip()
    if category and category not in CATEGORIES:
        raise ValueError(
            f"final_category {category!r} is not one of {list(CATEGORIES)}"
        )
    return category


def live_samples(records: list[TicketResult]) -> list[dict]:
    """Calibration samples (eval-runner shape) from records with outcomes."""
    samples: list[dict] = []
    for record in records:
        outcome = record.outcome
        if outcome is None:
            continue
        sample: dict = {"verify": {}, "choice": {}, "score": {}}
        if outcome.final_category:
            answer = record.answer("category")
            if answer is not None and answer.probabilities:
                sample["choice"]["category"] = (
                    dict(answer.probabilities),
                    outcome.final_category,
                )
        for key, truth in outcome.verify_truth.items():
            answer = record.answer(key)
            if answer is not None:
                sample["verify"][key] = (float(answer.value), bool(truth))
        for key, truth in outcome.score_truth.items():
            answer = record.answer(key)
            if answer is not None and answer.probabilities:
                sample["score"][key] = (dict(answer.probabilities), int(truth))
        if sample["verify"] or sample["choice"] or sample["score"]:
            samples.append(sample)
    return samples


def live_calibration(records: list[TicketResult]) -> dict:
    """Live calibration report over records that carry an agent outcome."""
    with_outcomes = [r for r in records if r.outcome is not None]
    samples = live_samples(records)

    category_judged = [r for r in with_outcomes if r.outcome and r.outcome.final_category]
    queue_judged = [r for r in with_outcomes if r.outcome and r.outcome.final_queue]
    payload: dict = {
        "available": bool(samples),
        "runs_total": len(records),
        "runs_with_outcomes": len(with_outcomes),
        "note": (
            "live calibration over agent-recorded outcomes"
            if samples
            else "no agent outcomes recorded yet — record one via "
            "POST /api/runs/{ticket_id}/outcome and live calibration appears here"
        ),
        "category_accuracy": (
            round(
                sum(1 for r in category_judged if r.category == r.outcome.final_category)
                / len(category_judged),
                4,
            )
            if category_judged
            else None
        ),
        "queue_accuracy": (
            round(
                sum(
                    1
                    for r in queue_judged
                    if r.composition.queue == r.outcome.final_queue
                )
                / len(queue_judged),
                4,
            )
            if queue_judged
            else None
        ),
        "resolution_rate": (
            round(
                sum(1 for r in with_outcomes if r.outcome and r.outcome.resolved)
                / len(with_outcomes),
                4,
            )
            if with_outcomes
            else None
        ),
    }
    if samples:
        payload.update(calibration_report(samples))
    else:
        payload.update(
            {"cases": 0, "overall_brier": None, "worst_bin_gap": None,
             "worst_bin": None, "per_question": {}}
        )
    return payload
