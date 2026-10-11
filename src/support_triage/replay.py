"""Policy impact replay: what would a different policy have done?

The console's what-if simulator sketches a hypothetical over the
open queue, in the browser, approximately. This is the grown-up
version, and it makes three sterner promises:

- **Every stored run is replayed** — decided, rejected and waiting
  alike — not just the tickets still open.
- **The real composition code runs.** The stored question answers
  (the decision layer's output, distributions and all) are rebuilt
  into a :class:`~support_triage.decisions.DecisionSet` and fed
  through :func:`support_triage.composition.compose` and
  :func:`support_triage.guardrails.run_guardrails` with the
  hypothetical constants. No approximation, and no second
  implementation of the policy to drift from the first.
- **No backend is ever called.** The recorded answers are the
  inputs; re-deciding history is neither possible (the model may
  have changed) nor the question. The question a policy change asks
  is what *code* would have done with the same answers.

And the invariant the whole feature stands on: **a replay writes
nothing.** Records are read, hypothetical consequences are computed
in memory, and the report is the only output — every stored record
is byte-identical afterwards (pinned by test, in the same spirit
as the assignment suggestion's purity test).

A run *moves* when its queue, priority band, or guardrail outcome
(passed / human-review forcing) would differ under the hypothetical.
Score changes inside a band are reported per run and counted, but a
score alone does not make a run "moved" — the three consequence
dimensions are what an operator acts on.
"""

from __future__ import annotations

from . import composition as comp
from . import guardrails as guard
from .decisions import DecisionSet, QuestionResult
from .questions import triage_questions
from .schemas import TicketResult

WEIGHT_SUM_TOLERANCE = 0.01


def validate_hypothetical(
    *,
    category_min_confidence: float | None = None,
    weights: dict | None = None,
    outage_boost: bool | None = None,
) -> dict:
    """Normalise a hypothetical policy change into the policy in
    force for a replay, or raise ``ValueError`` saying what's wrong.

    Every field is optional; an omitted field keeps the live value,
    so an empty hypothetical replays the current policy (and must
    therefore move nothing — the suite pins that identity).
    """
    if category_min_confidence is None:
        threshold = guard.CATEGORY_MIN_CONFIDENCE
    else:
        try:
            threshold = float(category_min_confidence)
        except (TypeError, ValueError):
            raise ValueError("category_min_confidence must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("category_min_confidence must be between 0 and 1")

    if weights is None:
        effective_weights = dict(comp.PRIORITY_WEIGHTS)
    else:
        if not isinstance(weights, dict):
            raise ValueError("weights must be an object keyed by formula term")
        expected = set(comp.PRIORITY_WEIGHTS)
        got = set(weights)
        if got != expected:
            raise ValueError(
                f"weights must name exactly the formula terms {sorted(expected)} "
                f"(missing={sorted(expected - got)}, unexpected={sorted(got - expected)})"
            )
        effective_weights = {}
        for term in comp.PRIORITY_WEIGHTS:
            try:
                value = float(weights[term])
            except (TypeError, ValueError):
                raise ValueError(f"weight for {term!r} must be a number")
            if value < 0:
                raise ValueError(f"weight for {term!r} must not be negative")
            effective_weights[term] = value
        total = sum(effective_weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"weights must sum to 1 (± {WEIGHT_SUM_TOLERANCE}); they sum to {total:.4f}"
            )

    boost_enabled = True if outage_boost is None else bool(outage_boost)
    return {
        "category_min_confidence": threshold,
        "weights": effective_weights,
        "outage_boost_enabled": boost_enabled,
        "outage_boost_points": comp.OUTAGE_BOOST if boost_enabled else 0.0,
    }


def current_policy() -> dict:
    """The live policy, in the same shape as a validated hypothetical."""
    return {
        "category_min_confidence": guard.CATEGORY_MIN_CONFIDENCE,
        "weights": dict(comp.PRIORITY_WEIGHTS),
        "outage_boost_enabled": True,
        "outage_boost_points": comp.OUTAGE_BOOST,
    }


def decision_set_from_record(record: TicketResult) -> DecisionSet:
    """Rebuild the decision set a stored record was composed from.

    The record's answers carry kind, value, full distribution and
    confidence — everything composition and the guardrails read.
    """
    telemetry = record.telemetry
    return DecisionSet(
        results=[
            QuestionResult(
                key=answer.key,
                kind=answer.kind,
                value=answer.value,
                probabilities=dict(answer.probabilities),
                confidence=answer.confidence,
            )
            for answer in record.answers
        ],
        backend=telemetry.backend if telemetry else "",
        model=telemetry.model if telemetry else "",
    )


def _consequence(*, queue: str, band: str, score: float, passed: bool, review: bool) -> dict:
    return {
        "queue": queue,
        "priority_band": band,
        "priority_score": score,
        "guardrails_passed": passed,
        "requires_human_review": review,
    }


def _stored_consequence(record: TicketResult) -> dict:
    return _consequence(
        queue=record.composition.queue,
        band=record.composition.priority_band,
        score=record.composition.priority_score,
        passed=record.guardrails.passed,
        review=record.guardrails.requires_human_review,
    )


def replay_policy(records: list[TicketResult], hypothetical: dict) -> dict:
    """Replay every stored run's answers under a hypothetical policy.

    ``hypothetical`` is the output of :func:`validate_hypothetical`.
    Pure: reads records, returns a report, writes nothing.
    """
    question_keys = [q.key for q in triage_questions()]
    expected_keys = set(question_keys)
    moved_runs: list[dict] = []
    skipped: list[dict] = []
    by_queue_pair: dict[str, int] = {}
    by_band_pair: dict[str, int] = {}
    by_guardrail_flip: dict[str, int] = {}
    score_changed = 0
    replayed = 0

    for record in records:
        if {a.key for a in record.answers} != expected_keys:
            # A record without the full answer set (never produced by
            # the pipeline; possible in a hand-built store) cannot be
            # replayed honestly — name it, don't guess at it.
            skipped.append(
                {
                    "ticket_id": record.ticket_id,
                    "reason": "stored record does not carry the full question set",
                }
            )
            continue
        replayed += 1
        decisions = decision_set_from_record(record)
        composed = comp.compose(
            decisions,
            weights=hypothetical["weights"],
            outage_boost=hypothetical["outage_boost_points"],
        )
        verdict = guard.run_guardrails(
            decisions,
            question_keys,
            category_min_confidence=hypothetical["category_min_confidence"],
        )
        before = _stored_consequence(record)
        after = _consequence(
            queue=composed.queue,
            band=composed.priority_band,
            score=composed.priority_score,
            passed=verdict.passed,
            review=verdict.requires_human_review,
        )

        changes: list[str] = []
        if before["queue"] != after["queue"]:
            changes.append("queue")
            pair = f"{before['queue']} → {after['queue']}"
            by_queue_pair[pair] = by_queue_pair.get(pair, 0) + 1
        if before["priority_band"] != after["priority_band"]:
            changes.append("priority_band")
            pair = f"{before['priority_band']} → {after['priority_band']}"
            by_band_pair[pair] = by_band_pair.get(pair, 0) + 1
        if before["priority_score"] != after["priority_score"]:
            score_changed += 1
        guardrail_flip = (
            before["guardrails_passed"] != after["guardrails_passed"]
            or before["requires_human_review"] != after["requires_human_review"]
        )
        if guardrail_flip:
            changes.append("guardrails")
            if before["guardrails_passed"] != after["guardrails_passed"]:
                key = (
                    f"guardrails {'passed' if before['guardrails_passed'] else 'failed'}"
                    f" → {'passed' if after['guardrails_passed'] else 'failed'}"
                )
                by_guardrail_flip[key] = by_guardrail_flip.get(key, 0) + 1
            if before["requires_human_review"] != after["requires_human_review"]:
                key = (
                    f"human review {'forced' if before['requires_human_review'] else 'not forced'}"
                    f" → {'forced' if after['requires_human_review'] else 'not forced'}"
                )
                by_guardrail_flip[key] = by_guardrail_flip.get(key, 0) + 1
        if changes:
            moved_runs.append(
                {
                    "ticket_id": record.ticket_id,
                    "status": record.status,
                    "changes": changes,
                    "before": before,
                    "after": after,
                }
            )

    moved_runs.sort(key=lambda row: row["ticket_id"])
    return {
        "hypothetical": hypothetical,
        "current_policy": current_policy(),
        "runs_replayed": replayed,
        "runs_skipped": len(skipped),
        "skipped": skipped,
        "moved": len(moved_runs),
        "unchanged": replayed - len(moved_runs),
        "score_changed": score_changed,
        "by_queue_pair": dict(sorted(by_queue_pair.items())),
        "by_band_pair": dict(sorted(by_band_pair.items())),
        "by_guardrail_flip": dict(sorted(by_guardrail_flip.items())),
        "moved_runs": moved_runs,
        "note": (
            "replay only — the stored answers were re-composed in code under the "
            "hypothetical policy; no backend was called and no record was changed"
        ),
    }
