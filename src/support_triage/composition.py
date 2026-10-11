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

    def as_dict(self) -> dict:
        return {
            "priority_score": self.priority_score,
            "priority_band": self.priority_band,
            "contributions": self.contributions,
            "outage_boost_applied": self.outage_boost_applied,
            "queue": self.queue,
            "routing_reason": self.routing_reason,
            "override": self.override,
            "flags": self.flags,
            "weights": PRIORITY_WEIGHTS,
        }


def compose(decisions: DecisionSet) -> Composition:
    """Priority + routing for one validated decision set."""
    urgency = decisions.score("urgency") / 10.0
    sla = decisions.score("sla_breach_risk") / 10.0
    churn = decisions.score("churn_risk") / 10.0
    anger = (10.0 - decisions.score("sentiment")) / 10.0
    abuse = decisions.probability("is_abusive")

    contributions = {
        "urgency": round(PRIORITY_WEIGHTS["urgency"] * urgency * 100, 2),
        "sla_breach_risk": round(PRIORITY_WEIGHTS["sla_breach_risk"] * sla * 100, 2),
        "churn_risk": round(PRIORITY_WEIGHTS["churn_risk"] * churn * 100, 2),
        "anger": round(PRIORITY_WEIGHTS["anger"] * anger * 100, 2),
        "abuse": round(PRIORITY_WEIGHTS["abuse"] * abuse * 100, 2),
    }
    score = sum(contributions.values())
    outage = decisions.probability("is_outage_report") >= FLAG_THRESHOLD
    if outage:
        score = min(100.0, score + OUTAGE_BOOST)
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

    return Composition(
        priority_score=score,
        priority_band=priority_band(score),
        contributions=contributions,
        outage_boost_applied=outage,
        queue=queue,
        routing_reason=reason,
        override=override,
        flags=flags,
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
) -> AutonomyRecommendation:
    """Would this ticket be a straight-through candidate? (Advisory only.)

    The recommendation is computed on every run and never acted on:
    the graph has no node that resolves a ticket.
    """
    reasons: list[str] = []
    blockers: list[str] = []
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
