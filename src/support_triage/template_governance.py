"""Template governance: a template earns its way live, and can lose the job.

The template first-response draft (templates.py) is the only text
this system ever composes toward a customer. It ships as bundled
code — one body, keyed by product area — which means the words an
agent may send were, until this module, approved by nobody in
particular and changeable only by editing source. Effectiveness
(template_effectiveness.py) grades the drafts from outcomes, and
its flag advises the template's owners — but a flagged template
stayed live forever, because no act existed that could take it
off the shelf, and no record existed of who had ever said its
words were fit to serve.

This module is that record and those acts.

**The registry is bundled data plus a governed overlay** — the
roster's pattern (roster.py: bundled data, loaded and validated),
extended with the one thing a lifecycle needs: state someone can
change. The bundled family stands as shipped — every template
live at version 1 with the bundled wording — and a stored
``TemplateRecord`` exists only for a template a human has taken
in hand. From the moment a record exists, the record IS the
template: its state, its words, its version. The merge is this
module's ``registry``; selection (templates.py) reads the same
effective view the console shows, so what the desk sees is what
the pipeline obeys.

**The lifecycle is draft → live → retired, and every transition
is an act with a name on it.**

- *Authoring* writes new words for a template name — a named
  editor, a stated reason — and the record starts in **draft**:
  under governance, a template's words are never live before
  someone approves them.
- *Editing* changes a record's words: the version bumps, the
  editor becomes the version's author, and a live template
  returns to draft. The words an agent sends never change
  silently underneath the approval that let them serve.
- *Promotion* (draft → live) takes a named approver who is NOT
  the version's author — self-promotion is refused by name, the
  shift handoff's handshake rule — and a stated basis, recorded
  on the template's history.
- *Retirement* (→ retired) takes a named editor and a reason.
  Retired is terminal: the record stays as history, and no act
  un-retires it — new words for the name are a new authoring only
  where no record stands, never a rewrite of a retired one.

History is append-only: every act appends a ``TemplateAct``; no
act edits or removes an earlier one.

**Retirement by the numbers is a proposal, never an act.** The
watch below reads the effectiveness report — round 11's figures,
reused verbatim, never recomputed — and flags a live template
``retirement_due`` when the report's own ``needs_review`` flag
stands on at least :data:`RETIREMENT_MIN_USES` outcome-bearing
uses: the review margin was exceeded on a sample twice the size
of the floor a rate needs to be quotable at all. The bar is set
higher than the flag's because retirement has consequences a
flag does not. But a due template KEEPS SERVING until a named
editor retires it. The reasons are the flag's own: a reopen rate
is computed from outcomes that can have causes outside the
template, and the effectiveness module's founding discipline is
that a number advises a human who reads the reopened tickets —
it does not reach into the pipeline. A proposal that acted on
its own would let one bad month silence a template with no one
having read a single ticket. (Rounds 12–13 hold the same line:
cluster proposals never merge tickets; the breach watch never
reassigns anyone.)

**What governance never touches.** When a template may be used
at all — the safest slice, the never-send boundary — is
templates.py's business and is unchanged: governance decides
WHICH wording stands behind the family's names, never whether
the slice exists. And governing a template writes nothing to any
run's record: drafts already composed keep the version they were
stamped with, which is how the audit export can say truthfully
which words a run was offered.
"""

from __future__ import annotations

from . import template_effectiveness, templates
from .schemas import TemplateRecord

# A live template is proposed for retirement when the
# effectiveness report's needs_review flag stands on at least
# this many outcome-bearing uses — twice the floor a rate needs
# to be quotable (template_effectiveness.EFFECTIVENESS_MIN_USES).
# Authored here, printed on the governance payload; a flag over
# five tickets is a prompt to read them, not to retire over them.
RETIREMENT_MIN_USES = 10

RETIREMENT_DUE = "retirement_due"

STATES = ("draft", "live", "retired")


class TemplateGovernanceError(ValueError):
    """A governance action that cannot stand.

    ``http_status`` is the API's mapping: 409 for a state
    conflict (promoting a template that is not in draft, editing
    or re-authoring over a retired record), 422 for an action
    invalid on its face (a missing actor, a missing reason,
    self-promotion, an unknown template name).
    """

    http_status = 422


class TemplateGovernanceConflict(TemplateGovernanceError):
    http_status = 409


def governance_constants() -> dict:
    """The authored rules of the lifecycle, as data — the payload
    and the console print them beside the registry they govern."""
    return {
        "retirement_min_uses": RETIREMENT_MIN_USES,
        "effectiveness_min_uses": template_effectiveness.EFFECTIVENESS_MIN_USES,
        "reopen_review_margin": template_effectiveness.REOPEN_REVIEW_MARGIN,
        "states": list(STATES),
    }


def bundled_view(area: str) -> dict:
    """One bundled template as the registry shows it before anyone
    governs it: live, version 1, the wording as shipped."""
    return {
        "name": templates.template_name(area),
        "area": area,
        "text": templates.bundled_text(),
        "guide": templates.bundled_guide(area),
        "state": "live",
        "version": templates.BUNDLED_VERSION,
        "authored_by": "",
        "source": "bundled",
        "created_at": "",
        "updated_at": "",
        "history": [],
    }


def registry(records: list[TemplateRecord]) -> list[dict]:
    """The effective registry: the bundled family, with every
    governed record standing in place of its bundled default.

    One row per template name, sorted by name — the view the
    console, the API, and template selection all read, so the
    state a reader sees is the state the pipeline obeys.
    """
    governed = {record.name: record for record in records}
    rows: list[dict] = []
    for area in templates.template_areas():
        name = templates.template_name(area)
        record = governed.pop(name, None)
        if record is None:
            rows.append(bundled_view(area))
        else:
            row = record.model_dump()
            row["source"] = "governed"
            rows.append(row)
    # A governed record whose name the bundled family does not
    # know cannot arise through the service (authoring validates
    # the name), but the registry never silently drops a record:
    # it stands in the list, where a reader can see it.
    for record in governed.values():
        row = record.model_dump()
        row["source"] = "governed"
        rows.append(row)
    rows.sort(key=lambda row: row["name"])
    return rows


def effectiveness_row(report: dict, name: str) -> dict | None:
    """The template's effectiveness row, verbatim from the report
    (round 11's figures) — matched by name across categories
    (a template's uses are how_to runs by construction, but the
    report is the report: if a name somehow stands in two
    categories, the row with more uses is the template's record).
    """
    rows = [row for row in report["templates"] if row["template"] == name]
    if not rows:
        return None
    return max(rows, key=lambda row: row["uses"])


def retirement_due(row: dict, report: dict) -> tuple[bool, str]:
    """Whether the numbers propose this template's retirement.

    ``row`` is one registry row. Due requires the row to be a
    LIVE template whose effectiveness row carries round 11's
    ``needs_review`` flag on at least RETIREMENT_MIN_USES
    outcome-bearing uses. Returns the verdict and the sentence
    that explains it — or, when not due, why not (not live, no
    measured uses yet, flag not standing, sample too small):
    the console prints the sentence either way, so "not due" is
    a reading, not an absence.
    """
    if row["state"] != "live":
        return False, f"state is {row['state']} — only a live template can fall due"
    figures = effectiveness_row(report, row["name"])
    if figures is None:
        return False, "no outcome-bearing uses measured yet — nothing to judge it by"
    if figures["flag"] != template_effectiveness.NEEDS_REVIEW:
        return False, (
            f"the effectiveness flag does not stand over "
            f"{figures['uses']} outcome-bearing use(s)"
        )
    if figures["uses"] < RETIREMENT_MIN_USES:
        return False, (
            f"flagged needs_review, but over {figures['uses']} "
            f"outcome-bearing use(s) — below the {RETIREMENT_MIN_USES}-use "
            "floor for proposing retirement; read the tickets first"
        )
    return True, (
        f"reopen rate {figures['reopen_rate']:.2f} beats the "
        f"{figures['category']} baseline {figures['baseline_reopen_rate']:.2f} "
        f"by more than the {template_effectiveness.REOPEN_REVIEW_MARGIN:.2f} "
        f"margin over {figures['uses']} outcome-bearing uses — retirement "
        "is proposed; the template keeps serving until a named editor "
        "retires it"
    )


def template_view(row: dict, report: dict) -> dict:
    """One registry row as the surfaces serve it: the record, its
    effectiveness figures (round 11's row, verbatim, or None), and
    the retirement watch's verdict with its sentence."""
    due, detail = retirement_due(row, report)
    view = dict(row)
    view["effectiveness"] = effectiveness_row(report, row["name"])
    view["retirement_due"] = due
    view["retirement_detail"] = detail
    return view
