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

Which wording stands behind the family's names is governed
(template_governance.py): the bundled text here is the default,
and a governed record — authored, edited, promoted, retired by
named people — replaces it for its name, state included. The
state check lives in this module, at the one place drafts are
built; a template that is not live drafts nothing, and the run's
record says the step was skipped for state.
"""

from __future__ import annotations

from .composition import Composition
from .decisions import DecisionSet
from .guardrails import GuardrailOutcome
from .schemas import DraftInfo

# The version the bundled wording stands at. A governed record
# (template_governance.py) carries its own version, bumped on every
# edit; drafts written from the bundled wording are stamped with
# this one, so "version 1" always names the words as shipped.
BUNDLED_VERSION = 1


def template_name(area: str) -> str:
    """The name a draft for ``area`` carries on its run — the one
    template family, keyed by product area."""
    return f"how_to_first_response[{area}]"


def template_areas() -> list[str]:
    """The product areas the bundled family covers."""
    return list(_AREA_GUIDES)


def bundled_text() -> str:
    """The bundled draft body (the wording as shipped)."""
    return _TEMPLATE


def bundled_guide(area: str) -> str:
    """The bundled guide pointer for ``area``."""
    return _AREA_GUIDES.get(area, _AREA_GUIDES["other"])


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


def _slice_gates(
    decisions: DecisionSet,
    composition: Composition,
    guardrails: GuardrailOutcome,
    *,
    is_duplicate: bool = False,
    is_reopen: bool = False,
) -> bool:
    """The safest-slice gates, in one place: both the draft builder
    and the governance block below judge by exactly these, so the
    run's explanation of a skipped draft can never drift from the
    rule that skipped it."""
    if is_reopen:
        # No second template, ever: a customer whose issue already
        # failed once does not get the canned path again. Enforced
        # here — the one place drafts are built — not by asking a
        # model or an operator to remember.
        return False
    if decisions.choice("category") != "how_to":
        return False
    if decisions.confidence("category") < 0.80:
        return False
    if any(composition.flags.values()):
        return False
    if not guardrails.passed or guardrails.requires_human_review:
        return False
    if is_duplicate:
        return False
    return True


def _governed_row(
    governed_templates: dict | None, area: str
) -> dict | None:
    """The effective registry row for ``area``'s template, when the
    caller supplied the registry (the service does, on every run).
    ``None`` — a graph driven without the service — means the
    bundled wording stands, exactly as before governance existed.
    A supplied registry that somehow lacks the name falls back to
    the bundled row for the area rather than inventing a state."""
    if governed_templates is None:
        return None
    return governed_templates.get(template_name(area))


def build_template_draft(
    decisions: DecisionSet,
    composition: Composition,
    guardrails: GuardrailOutcome,
    *,
    sender_name: str = "",
    is_duplicate: bool = False,
    is_reopen: bool = False,
    governed_templates: dict | None = None,
) -> DraftInfo | None:
    """A draft for the safest slice only; ``None`` everywhere else.

    Governance (template_governance.py) is consulted at this one
    place: when the registry row for the area's template is not
    LIVE — a draft under review, a retired template — no draft is
    built, however clean the slice. A live governed row supplies
    the words and the version stamped onto the draft; the bundled
    wording serves where no record stands.
    """
    if not _slice_gates(
        decisions,
        composition,
        guardrails,
        is_duplicate=is_duplicate,
        is_reopen=is_reopen,
    ):
        return None
    area = decisions.choice("product_area")
    row = _governed_row(governed_templates, area)
    if row is not None and row.get("state") != "live":
        return None
    if row is not None:
        text = str(row.get("text") or _TEMPLATE)
        guide = str(row.get("guide") or _AREA_GUIDES["other"])
        version = int(row.get("version") or BUNDLED_VERSION)
    else:
        text = _TEMPLATE
        guide = _AREA_GUIDES.get(area, _AREA_GUIDES["other"])
        version = BUNDLED_VERSION
    name = (sender_name or "").split()[0] if sender_name.strip() else "there"
    return DraftInfo(
        text=text.format(name=name, guide=guide),
        template=template_name(area),
        template_version=version,
    )


def template_state_block(
    decisions: DecisionSet,
    composition: Composition,
    guardrails: GuardrailOutcome,
    *,
    is_duplicate: bool = False,
    is_reopen: bool = False,
    governed_templates: dict | None = None,
) -> tuple[str, str] | None:
    """Why governance withheld a draft this run would otherwise
    have earned: ``(template name, state)`` when every slice gate
    passes but the template's registry row is not live — else
    ``None``. The compose step asks so the run's record can say,
    in the vocabulary it already uses for the reopen suppression,
    that the template step was skipped for state — the record
    shows the skip; this function is the only judge of it, using
    the same gates the builder does.
    """
    if not _slice_gates(
        decisions,
        composition,
        guardrails,
        is_duplicate=is_duplicate,
        is_reopen=is_reopen,
    ):
        return None
    area = decisions.choice("product_area")
    row = _governed_row(governed_templates, area)
    if row is None or row.get("state") == "live":
        return None
    return template_name(area), str(row.get("state") or "draft")
