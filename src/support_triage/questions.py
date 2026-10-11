"""The triage question set: the decomposed decisions one ticket needs.

The set is fixed by the blueprint (a deployment retunes rubrics and
thresholds, not the vocabulary). Twelve questions, one decision call:

- two **choice** questions place the ticket (category, product area);
- four **score** questions grade it (urgency, sentiment, churn risk,
  SLA breach risk) on rubrics whose ends are written down, so a 7
  means the same thing whichever backend produced it;
- six **verify** questions flag what code must never miss (PII, abuse,
  legal mentions, outage reports, engineering need).

The composition layer (``composition.py``) reads exactly these keys.
"""

from __future__ import annotations

from .decisions import ChoiceQuestion, Question, ScoreQuestion, VerifyQuestion

CATEGORIES = (
    "billing",
    "technical_issue",
    "account_access",
    "how_to",
    "bug_report",
    "feature_request",
    "complaint",
    "other",
)

PRODUCT_AREAS = (
    "checkout",
    "mobile_app",
    "api",
    "dashboard",
    "integrations",
    "account_admin",
    "other",
)


def triage_questions() -> list[Question]:
    """The full question set, in evaluation order."""
    return [
        ChoiceQuestion(
            key="category",
            prompt="What kind of support ticket is this, primarily?",
            options=CATEGORIES,
        ),
        ChoiceQuestion(
            key="product_area",
            prompt="Which product area does the ticket concern, primarily?",
            options=PRODUCT_AREAS,
        ),
        ScoreQuestion(
            key="urgency",
            prompt="How urgent is this for the customer?",
            rubric=(
                "0 = no time pressure at all; 5 = work is slowed and a workaround "
                "exists; 8 = work is blocked; 10 = the customer is fully down or "
                "losing money by the minute."
            ),
        ),
        ScoreQuestion(
            key="sentiment",
            prompt="How positive is the customer's tone?",
            rubric=(
                "0 = furious or threatening; 3 = clearly frustrated; 5 = neutral, "
                "businesslike; 8 = polite and patient; 10 = explicitly grateful."
            ),
        ),
        ScoreQuestion(
            key="churn_risk",
            prompt="How likely is this customer to churn over this issue?",
            rubric=(
                "0 = no churn signal; 5 = dissatisfaction with some loyalty "
                "signals; 8 = evaluating alternatives or mentioning cancel; "
                "10 = has stated they are leaving or cancelling over this."
            ),
        ),
        ScoreQuestion(
            key="sla_breach_risk",
            prompt="How likely is this ticket to breach its first-response or resolution SLA if queued normally?",
            rubric=(
                "0 = trivially within SLA; 5 = needs same-day attention; "
                "8 = likely breach without priority handling; 10 = already "
                "effectively breached or minutes from it."
            ),
        ),
        VerifyQuestion(
            key="contains_pii",
            prompt="Does the ticket text contain personal data beyond a name and work email?",
            statement=(
                "The ticket contains personal data such as a password, payment "
                "card number, government ID, home address, or private phone number."
            ),
        ),
        VerifyQuestion(
            key="is_abusive",
            prompt="Is the ticket abusive toward support staff?",
            statement=(
                "The ticket contains profanity directed at staff, personal "
                "insults, threats, or harassment."
            ),
        ),
        VerifyQuestion(
            key="mentions_legal",
            prompt="Does the ticket raise a legal or regulatory matter?",
            statement=(
                "The ticket mentions lawyers, litigation, regulators, GDPR or "
                "privacy-law rights, chargebacks through a bank, or a formal dispute."
            ),
        ),
        VerifyQuestion(
            key="is_outage_report",
            prompt="Is the customer reporting a service outage or total failure?",
            statement=(
                "The customer reports the service is down, unavailable, or "
                "failing for everyone, not just a personal issue."
            ),
        ),
        VerifyQuestion(
            key="needs_engineering",
            prompt="Does resolving this require engineering (a code change, data fix, or infrastructure work)?",
            statement=(
                "Resolution requires engineering work beyond what support can "
                "do with existing tools and documentation."
            ),
        ),
    ]
