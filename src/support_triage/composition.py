"""Composition: where decisions become consequences — in code.

The decision model returns distributions. Everything a business would
argue about happens here, deterministically, where it can be unit
tested and read aloud:

- **Priority score** — one weighted formula over the scored answers
  (weights below, summing to 1), plus a fixed outage boost. The full
  breakdown (each term's contribution) is recorded on the run, so an
  agent can always answer "why is this a 73?".
- **Routing** — a matrix from (category, product area) to a queue,
  with three hard overrides that fire on verify probabilities:
  legal → compliance, outage → incident, engineering → escalation.
  Override precedence is legal, then outage, then engineering: a
  legal matter is a legal matter even mid-outage; an outage is
  coordinated by the incident queue even when engineering fixes it.
- **Autonomy recommendation** — which tickets the *shape* of the
  answers marks as safe for straight-through handling (simple how-to,
  high confidence, no flags, no duplicate). Computed, recorded, and
  **never acted on**: every run ends at the agent gate regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import env_int
from .decisions import DecisionSet

# --- priority formula ---------------------------------------------------
# Weights over the normalised (0..1) scored answers. They sum to 1.0;
# changing them is a product decision, so they live in exactly one
# place and every run record carries the weights it was scored under.
PRIORITY_WEIGHTS = {
    "urgency": 0.30,
    "sla_breach_risk": 0.25,
    "churn_risk": 0.20,
    "anger": 0.15,  # derived: 10 - sentiment
    "abuse": 0.10,  # the is_abusive probability — abusive tickets need a human fast, not slow
}
OUTAGE_BOOST = 15.0  # added when P(outage) >= FLAG_THRESHOLD
FLAG_THRESHOLD = 0.5  # a verify answer counts as "raised" at this probability

# Priority bands (score ranges are part of the eval contract).
BANDS = ((80.0, "critical"), (55.0, "high"), (30.0, "medium"), (0.0, "low"))

# --- reopen policy --------------------------------------------------------
# A reopened ticket has already failed the customer once, so its
# composed band is floored one step above the band its answers alone
# earn (see compose's ``reopen`` argument) — the floor is recorded on
# the run like the outage boost, never applied invisibly. And a chain
# may only be reopened so many times through the normal path: the
# reopen that reaches REOPEN_LIMIT is forced to human review by the
# guardrails (guardrails.py) instead of being routed by composition
# alone. The limit is a deployment knob (env REOPEN_LIMIT); the
# default and the floor live here with the other policy constants.
REOPEN_LIMIT_DEFAULT = 2


def reopen_limit_in_force() -> int:
    """The chain reopen limit for runs composed now (env-overridable)."""
    return env_int("REOPEN_LIMIT", REOPEN_LIMIT_DEFAULT)


# --- SLA clock targets ----------------------------------------------------
# Deterministic response/resolution targets per priority band, in
# hours, read by the SLA clock (sla.py). These are NOT the model's
# sla_breach_risk answer: that is a judgement (a distribution over a
# rubric) that feeds the priority formula above. The clock is plain
# arithmetic over the ticket's age — the two never mix.
SLA_TARGET_HOURS = {"critical": 1.0, "high": 4.0, "medium": 24.0, "low": 72.0}
# A ticket is "due soon" once its remaining time falls to this
# fraction of its target (the last quarter of the clock).
SLA_DUE_SOON_FRACTION = 0.25


def priority_band(score: float) -> str:
    for floor, name in BANDS:
        if score >= floor:
            return name
    return "low"


# --- routing --------------------------------------------------------------
QUEUES = (
    "billing_support",
    "technical_support",
    "developer_support",
    "account_support",
    "general_support",
    "product_feedback",
    "customer_success",
    "compliance_review",
    "incident_response",
    "engineering_escalation",
)

_CATEGORY_QUEUE = {
    "billing": "billing_support",
    "technical_issue": "technical_support",
    "account_access": "account_support",
    "how_to": "general_support",
    "bug_report": "technical_support",
    "feature_request": "product_feedback",
    "complaint": "customer_success",
    "other": "general_support",
}

# (category, product_area) cells that refine the category default.
_AREA_REFINEMENTS = {
    ("technical_issue", "api"): "developer_support",
    ("bug_report", "api"): "developer_support",
}


@dataclass
class Composition:
    """The deterministic reading of one decision set."""

    priority_score: float
    priority_band: str
    contributions: dict[str, float]  # formula term -> points contributed
    outage_boost_applied: bool
    queue: str
    routing_reason: str
    override: str = ""  # "" | legal | outage | engineering
    flags: dict[str, bool] = field(default_factory=dict)  # verify key -> raised?
    # The reopen floor, when it ran: the band was raised one step
    # from ``band_before_floor``. Recorded on the run either way.
    reopen_floor_applied: bool = False
    band_before_floor: str = ""
    # The weights this composition was actually scored under — the
    # module constants for a live run, a hypothetical set for a replay
    # (replay.py). Recorded on the run either way.
    weights: dict[str, float] = field(default_factory=lambda: dict(PRIORITY_WEIGHTS))

    def as_dict(self) -> dict:
        return {
            "priority_score": self.priority_score,
            "priority_band": self.priority_band,
            "contributions": self.contributions,
            "outage_boost_applied": self.outage_boost_applied,
            "reopen_floor_applied": self.reopen_floor_applied,
            "band_before_floor": self.band_before_floor,
            "queue": self.queue,
            "routing_reason": self.routing_reason,
            "override": self.override,
            "flags": self.flags,
            "weights": self.weights,
        }


def _raise_one_band(band: str) -> str:
    """One step up the band ladder (critical stays critical)."""
    ladder = [name for _floor, name in reversed(BANDS)]  # low → critical
    index = ladder.index(band) if band in ladder else 0
    return ladder[min(index + 1, len(ladder) - 1)]


def compose(
    decisions: DecisionSet,
    *,
    weights: dict[str, float] | None = None,
    outage_boost: float | None = None,
    reopen: bool = False,
) -> Composition:
    """Priority + routing for one validated decision set.

    ``weights`` / ``outage_boost`` override the module policy for
    this call only (the impact replay, replay.py); the defaults are
    the live policy and every existing caller passes neither.

    ``reopen`` applies the reopen priority floor: the band is raised
    one step from the band the score earns (the score itself is
    untouched — the floor is a band rule, recorded on the result).
    The replay passes it for stored reopen runs so history recomposes
    exactly as it was composed.
    """
    effective_weights = dict(weights) if weights is not None else PRIORITY_WEIGHTS
    boost_points = OUTAGE_BOOST if outage_boost is None else outage_boost
    urgency = decisions.score("urgency") / 10.0
    sla = decisions.score("sla_breach_risk") / 10.0
    churn = decisions.score("churn_risk") / 10.0
    anger = (10.0 - decisions.score("sentiment")) / 10.0
    abuse = decisions.probability("is_abusive")

    term_values = {
        "urgency": urgency,
        "sla_breach_risk": sla,
        "churn_risk": churn,
        "anger": anger,
        "abuse": abuse,
    }
    contributions = {
        term: round(effective_weights[term] * term_values[term] * 100, 2)
        for term in PRIORITY_WEIGHTS  # canonical term order
    }
    score = sum(contributions.values())
    outage = decisions.probability("is_outage_report") >= FLAG_THRESHOLD
    if outage:
        score = min(100.0, score + boost_points)
    score = round(score, 1)

    flags = {
        key: decisions.probability(key) >= FLAG_THRESHOLD
        for key in (
            "contains_pii",
            "is_abusive",
            "mentions_legal",
            "is_outage_report",
            "needs_engineering",
        )
    }

    category = decisions.choice("category")
    area = decisions.choice("product_area")
    queue = _AREA_REFINEMENTS.get((category, area), _CATEGORY_QUEUE.get(category, "general_support"))
    reason = f"matrix: {category} × {area} → {queue}"
    override = ""
    if flags["mentions_legal"]:
        queue, override = "compliance_review", "legal"
        reason = f"override: mentions_legal (p={decisions.probability('mentions_legal'):.2f}) → compliance_review"
    elif flags["is_outage_report"]:
        queue, override = "incident_response", "outage"
        reason = f"override: is_outage_report (p={decisions.probability('is_outage_report'):.2f}) → incident_response"
    elif flags["needs_engineering"]:
        queue, override = "engineering_escalation", "engineering"
        reason = f"override: needs_engineering (p={decisions.probability('needs_engineering'):.2f}) → engineering_escalation"

    band = priority_band(score)
    band_before_floor = ""
    if reopen:
        band_before_floor = band
        band = _raise_one_band(band)

    return Composition(
        priority_score=score,
        priority_band=band,
        contributions=contributions,
        outage_boost_applied=outage,
        reopen_floor_applied=reopen,
        band_before_floor=band_before_floor,
        queue=queue,
        routing_reason=reason,
        override=override,
        flags=flags,
        weights=dict(effective_weights),
    )


# --- autonomy recommendation ----------------------------------------------
# Categories a deployment would even consider for straight-through
# handling: information requests with a documented answer. Everything
# else always needs a human's judgement in this blueprint's reading.
AUTO_RESOLVABLE_CATEGORIES = {"how_to"}
AUTO_MIN_CATEGORY_CONFIDENCE = 0.80


@dataclass
class AutonomyRecommendation:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"eligible": self.eligible, "reasons": self.reasons, "blockers": self.blockers}


def recommend_autonomy(
    decisions: DecisionSet,
    composition: Composition,
    *,
    guardrails_passed: bool,
    requires_human_review: bool,
    is_duplicate: bool,
    is_reopen: bool = False,
) -> AutonomyRecommendation:
    """Would this ticket be a straight-through candidate? (Advisory only.)

    The recommendation is computed on every run and never acted on:
    the graph has no node that resolves a ticket.
    """
    reasons: list[str] = []
    blockers: list[str] = []
    if is_reopen:
        blockers.append("reopened ticket — a follow-up is handled by a human")
    category = decisions.choice("category")
    if category in AUTO_RESOLVABLE_CATEGORIES:
        reasons.append(f"category {category} is in the auto-resolvable class")
    else:
        blockers.append(f"category {category} always needs an agent")
    confidence = decisions.confidence("category")
    if confidence >= AUTO_MIN_CATEGORY_CONFIDENCE:
        reasons.append(f"category confidence {confidence:.2f} ≥ {AUTO_MIN_CATEGORY_CONFIDENCE}")
    else:
        blockers.append(f"category confidence {confidence:.2f} < {AUTO_MIN_CATEGORY_CONFIDENCE}")
    for key, raised in composition.flags.items():
        if raised:
            blockers.append(f"flag raised: {key}")
    if not guardrails_passed:
        blockers.append("guardrails did not pass")
    if requires_human_review:
        blockers.append("a gating question was indecisive (human review forced)")
    if is_duplicate:
        blockers.append("probable duplicate of an earlier ticket")
    return AutonomyRecommendation(
        eligible=not blockers, reasons=reasons, blockers=blockers
    )


# --- the policy, as data ---------------------------------------------------


def policy_snapshot() -> dict:
    """The composition policy as one read-only document.

    Everything a reviewer or the console's what-if panel needs to see
    the rules the runs were composed under: the priority formula and
    its weights, the band floors, the routing matrix with its
    refinements and overrides, the guardrail forcing thresholds, and
    the autonomy bar. Derived from the constants above (and the
    guardrail module's) — never a second copy that could drift.
    """
    from . import threads
    from .guardrails import (
        CATEGORY_MIN_CONFIDENCE,
        GATING_VERIFY,
        VERIFY_DECISIVE_HIGH,
        VERIFY_DECISIVE_LOW,
    )
    from .questions import CATEGORIES, PRODUCT_AREAS

    return {
        "priority": {
            "weights": dict(PRIORITY_WEIGHTS),
            "outage_boost": OUTAGE_BOOST,
            "flag_threshold": FLAG_THRESHOLD,
            "bands": [
                {"band": name, "min_score": floor} for floor, name in BANDS
            ],
        },
        "routing": {
            "queues": list(QUEUES),
            "categories": list(CATEGORIES),
            "product_areas": list(PRODUCT_AREAS),
            "category_queues": dict(_CATEGORY_QUEUE),
            "area_refinements": {
                f"{category} × {area}": queue
                for (category, area), queue in sorted(_AREA_REFINEMENTS.items())
            },
            "overrides": [
                {"flag": "mentions_legal", "queue": "compliance_review", "precedence": 1},
                {"flag": "is_outage_report", "queue": "incident_response", "precedence": 2},
                {"flag": "needs_engineering", "queue": "engineering_escalation", "precedence": 3},
            ],
        },
        "guardrails": {
            "category_min_confidence": CATEGORY_MIN_CONFIDENCE,
            "verify_decisive_low": VERIFY_DECISIVE_LOW,
            "verify_decisive_high": VERIFY_DECISIVE_HIGH,
            "gating_verify": list(GATING_VERIFY),
        },
        "autonomy": {
            "auto_resolvable_categories": sorted(AUTO_RESOLVABLE_CATEGORIES),
            "min_category_confidence": AUTO_MIN_CATEGORY_CONFIDENCE,
            "note": "advisory only — computed on every run, never acted on",
        },
        "sla": {
            "target_hours": dict(SLA_TARGET_HOURS),
            "due_soon_fraction": SLA_DUE_SOON_FRACTION,
            "note": (
                "deterministic clock targets per priority band (sla.py); "
                "distinct from the model's sla_breach_risk judgement"
            ),
        },
        "reopen": {
            "limit": reopen_limit_in_force(),
            "limit_default": REOPEN_LIMIT_DEFAULT,
            "limit_env": "REOPEN_LIMIT",
            "priority_floor": "one band above the band the answers earn",
            "template_drafts": "never offered on a reopen run (code, not a prompt)",
            "limit_behaviour": (
                "the reopen that reaches the limit is forced to human "
                "review by the reopen_limit guardrail check"
            ),
            "note": (
                "a reopen is a new run linked to the old (the old record "
                "is never mutated); the prior run's answers and outcome "
                "ride as labelled history context for the decision call"
            ),
        },
        "threads": {
            "window_days": threads.THREAD_WINDOW_DAYS,
            "overlap_threshold": threads.THREAD_OVERLAP_THRESHOLD,
            "method": (
                "same customer (normalised email, else name) + token "
                "containment of subject/body tokens (shared over the "
                "smaller set) at or over the threshold, within the "
                "window — a label assigned at intake; grouping merges "
                "and alters nothing"
            ),
        },
    }
