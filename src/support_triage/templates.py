"""Template first-response drafts — deterministic, and deliberately rare.

The decision model never writes prose. For exactly one slice of
tickets — a clear **how_to** question, confidently classified, with no
safety flags, no duplicate link, and guardrails green — code can offer
the agent a template first response: a pointer to the documented
answer path for the product area, in fixed wording. The agent reads
it, edits or discards it, and sends it themselves; the system sends
nothing, ever.

Every other ticket gets no draft at all. That asymmetry is the point:
the template exists to show where straight-through handling *would*
start, under the autonomy recommendation — not to draft the queue.
"""

from __future__ import annotations

from .composition import Composition
from .decisions import DecisionSet
from .guardrails import GuardrailOutcome
from .schemas import DraftInfo

# Product area -> where the documented answer lives. A deployment
# points these at its own help centre; the wording is a template, not
# generated text.
_AREA_GUIDES = {
    "checkout": "the checkout guide (Billing & checkout → Completing a purchase)",
    "mobile_app": "the mobile app guide (Mobile app → Getting started)",
    "api": "the API reference (Developers → API quickstart)",
    "dashboard": "the dashboard guide (Using the dashboard → Overview)",
    "integrations": "the integrations guide (Settings → Integrations)",
    "account_admin": "the account administration guide (Settings → Account)",
    "other": "the help centre",
}

_TEMPLATE = (
    "Hi {name},\n\n"
    "Thanks for getting in touch. For this, the steps in {guide} cover "
    "exactly what you're describing — the short version is to open that "
    "section and follow the numbered steps; each one takes under a minute.\n\n"
    "If anything there doesn't match what you see on your screen, reply "
    "to this message with a screenshot and we'll take it from there.\n\n"
    "— Support (template draft, sent by an agent)"
)


def build_template_draft(
    decisions: DecisionSet,
    composition: Composition,
    guardrails: GuardrailOutcome,
    *,
    sender_name: str = "",
    is_duplicate: bool = False,
    is_reopen: bool = False,
) -> DraftInfo | None:
    """A draft for the safest slice only; ``None`` everywhere else."""
    if is_reopen:
        # No second template, ever: a customer whose issue already
        # failed once does not get the canned path again. Enforced
        # here — the one place drafts are built — not by asking a
        # model or an operator to remember.
        return None
    if decisions.choice("category") != "how_to":
        return None
    if decisions.confidence("category") < 0.80:
        return None
    if any(composition.flags.values()):
        return None
    if not guardrails.passed or guardrails.requires_human_review:
        return None
    if is_duplicate:
        return None
    area = decisions.choice("product_area")
    guide = _AREA_GUIDES.get(area, _AREA_GUIDES["other"])
    name = (sender_name or "").split()[0] if sender_name.strip() else "there"
    return DraftInfo(
        text=_TEMPLATE.format(name=name, guide=guide),
        template=f"how_to_first_response[{area}]",
    )
