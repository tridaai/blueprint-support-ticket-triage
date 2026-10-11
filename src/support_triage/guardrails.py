"""Guardrails: deterministic checks over the decision layer.

Prompts ask; guardrails decide. These checks run in code over the
validated decision set, and their outcomes are recorded on the run:

- **completeness** — every question in the set has exactly one answer
  (the port validation already guarantees the shape; this check makes
  the guarantee a named, recorded verdict like its siblings).
- **distribution_sanity** — every distribution sums to 1 within the
  port tolerance and every probability sits in [0, 1]. Again enforced
  upstream; recorded here so the record shows it was checked.
- **low_confidence** — the run's honesty rule. A *gating* question
  whose answer is indecisive cannot be composed into a confident
  consequence: the run is forced to HUMAN_REVIEW instead of the
  pipeline guessing. For choice gating (category) indecision is a top
  probability under the threshold; for verify gating it is a
  probability inside the undecided band around 0.5.
- **safety_flags** — raised safety verifies (PII, abuse, legal,
  outage) are named on the record and block the autonomy
  recommendation and any template draft. The check itself passes —
  the flags are information, not malfunction — but a ticket carrying
  one can never be auto-anything.

A failed guardrail does not fail the run: it marks the record, forces
human review, and the agent decides with the failure in front of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .decisions import SUM_TOLERANCE, DecisionSet

# Gating thresholds: below/inside these, the answer is "we don't know"
# and a human must look. Retuned per deployment, in one place.
CATEGORY_MIN_CONFIDENCE = 0.55
VERIFY_DECISIVE_LOW = 0.35  # p <= this: confidently false
VERIFY_DECISIVE_HIGH = 0.65  # p >= this: confidently true
GATING_VERIFY = ("contains_pii", "is_abusive", "mentions_legal", "is_outage_report")
SAFETY_VERIFY = ("contains_pii", "is_abusive", "mentions_legal", "is_outage_report")


@dataclass
class GuardrailCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GuardrailOutcome:
    checks: list[GuardrailCheck] = field(default_factory=list)
    requires_human_review: bool = False
    human_review_reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "requires_human_review": self.requires_human_review,
            "human_review_reasons": self.human_review_reasons,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def run_guardrails(
    decisions: DecisionSet,
    question_keys: list[str],
    *,
    category_min_confidence: float | None = None,
) -> GuardrailOutcome:
    """Run every check over one decision set.

    ``category_min_confidence`` overrides the forcing threshold for
    this call only (the impact replay, replay.py); the default is
    the live policy constant and every existing caller passes
    neither argument nor thought to it.
    """
    min_confidence = (
        CATEGORY_MIN_CONFIDENCE if category_min_confidence is None else category_min_confidence
    )
    checks: list[GuardrailCheck] = []
    review_reasons: list[str] = []

    # completeness
    answered = [r.key for r in decisions.results]
    complete = sorted(answered) == sorted(question_keys) and len(answered) == len(set(answered))
    checks.append(
        GuardrailCheck(
            "completeness",
            complete,
            f"{len(answered)}/{len(question_keys)} questions answered exactly once"
            if complete
            else f"answered keys {sorted(answered)} != question keys {sorted(question_keys)}",
        )
    )

    # distribution sanity
    problems: list[str] = []
    for result in decisions.results:
        total = sum(result.probabilities.values())
        if abs(total - 1.0) > SUM_TOLERANCE:
            problems.append(f"{result.key} sums to {total:.3f}")
        for option, p in result.probabilities.items():
            if not (0.0 <= p <= 1.0):
                problems.append(f"{result.key}.{option} = {p} outside [0, 1]")
    checks.append(
        GuardrailCheck(
            "distribution_sanity",
            not problems,
            "all distributions sum to 1 ± tolerance, probabilities in [0, 1]"
            if not problems
            else "; ".join(problems),
        )
    )

    # low confidence (the honesty rule)
    indecisive: list[str] = []
    category_confidence = decisions.confidence("category")
    if category_confidence < min_confidence:
        indecisive.append(
            f"category confidence {category_confidence:.2f} < {min_confidence}"
        )
    for key in GATING_VERIFY:
        p = decisions.probability(key)
        if VERIFY_DECISIVE_LOW < p < VERIFY_DECISIVE_HIGH:
            indecisive.append(f"{key} probability {p:.2f} is in the undecided band")
    if indecisive:
        review_reasons.extend(f"low confidence: {item}" for item in indecisive)
    checks.append(
        GuardrailCheck(
            "low_confidence",
            not indecisive,
            "all gating questions decisive" if not indecisive else "; ".join(indecisive),
        )
    )

    # safety flags
    raised = [key for key in SAFETY_VERIFY if decisions.probability(key) >= 0.5]
    checks.append(
        GuardrailCheck(
            "safety_flags",
            True,  # flags are information; the check records them
            "no safety flags raised"
            if not raised
            else "safety flags raised: "
            + ", ".join(f"{key} (p={decisions.probability(key):.2f})" for key in raised)
            + " — autonomy and template draft blocked",
        )
    )

    return GuardrailOutcome(
        checks=checks,
        requires_human_review=bool(review_reasons),
        human_review_reasons=review_reasons,
    )
