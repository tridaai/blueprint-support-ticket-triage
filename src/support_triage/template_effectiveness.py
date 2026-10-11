"""Template effectiveness: do the drafts earn their place?

The template draft (templates.py) is the only text this system ever
composes toward a customer — and even it is never sent by the
system; an agent reads it, edits or discards it, and sends it
themselves. Templates are deliberately rare (one slice: a confident
how_to, no flags, guardrails green, never a reopen, never a
duplicate). Rare is not the same as good. Whether a template helps
or quietly hurts is knowable from the outcomes of the tickets that
carried it — did they resolve, did the customer come back (a
reopen), did the agent end up escalating by hand? This module is
that measurement, computed in code over the stored records:

- **The link already rides the record.** A run that earned a draft
  carries the template's name on it (``DraftInfo.template``, e.g.
  ``how_to_first_response[api]`` — the one template family, bound
  to the how_to category and keyed by product area). Nothing new
  is recorded; this module reads what composition already wrote.
  A record whose draft carries NO name — the shape a pre-link
  record would have — is counted as *unknown template*: it joins
  the totals and its category's baseline, but no per-template rate
  is ever attributed to a name nobody recorded. History is never
  backfilled with a guess.
- **The universe is outcome-bearing runs.** A template's record is
  judged only where the ticket's later life is known: a recorded
  outcome (``AgentOutcome``). Runs without one say nothing about
  the template yet and are not counted as uses.
- **The rates, defined from the outcome vocabulary and nothing
  else.** *Resolved*: the outcome's ``resolved`` flag. *Reopened*:
  a later run's reopen chain roots at this run
  (``ReopenInfo.root_ticket_id``) — the customer came back on the
  same chain, however the follow-up was worded. *Escalated*: the
  outcome's ``final_queue`` is one of the human escalation queues
  (:data:`ESCALATION_QUEUES`) that triage had NOT routed the run
  to — the agent who handled it moved it up by hand. Rates are
  plain counts over uses, rounded like the other read models'.
- **Small samples are labelled, not dressed up.** Below
  :data:`EFFECTIVENESS_MIN_USES` outcome-bearing uses a template's
  rates are withheld (``None``) — the counts stand, the percentages
  wait, in the repo's standing idiom (the QA report's verdict
  floor, the workbench's sample floor).
- **The flag is advice to the template's owners, and nothing
  more.** A template whose reopen rate beats its own category's
  baseline reopen rate by more than
  :data:`REOPEN_REVIEW_MARGIN` is flagged ``needs_review`` — when
  both rates are quotable. The flag changes nothing: no template
  is suppressed, no composition input moves. Composition never
  reads this module (the dossier firewall's sibling): the same
  ticket composes byte-identically the day before and the day
  after its template is flagged, and the golden set pins that.
- **Confirmed duplicates are not second opinions.** A run under an
  active operator duplicate mark is excluded from the rates (the
  calibration-sampling precedent: a duplicate is the same customer
  asking again, not independent evidence about a template) and
  counted in the totals, named, so the accounting reconciles.

Everything here is a pure read over the records it is handed. The
report's thresholds are authored data in this module and printed
on the report itself.
"""

from __future__ import annotations

from . import corrections
from .schemas import TicketResult

# Below this many outcome-bearing uses, a rate is withheld (counts
# still shown) — the QA report's floor size, for the same reason:
# a percentage over a handful of tickets is a claim the data
# cannot carry.
EFFECTIVENESS_MIN_USES = 5

# A template is flagged for its owners' review when its reopen
# rate exceeds its category's baseline reopen rate by MORE than
# this margin (a rate difference, not a ratio). Authored policy,
# printed on the report; changing it is a product decision.
REOPEN_REVIEW_MARGIN = 0.25

# The queues that mean "a human took this up a level": the routing
# matrix's two escalation destinations. An outcome landing in one
# of these — when triage itself routed elsewhere — is the agent's
# escalation, recorded in the outcome's final_queue.
ESCALATION_QUEUES = ("compliance_review", "engineering_escalation")

NEEDS_REVIEW = "needs_review"


def template_name(record: TicketResult) -> str | None:
    """The template a run's draft used, as recorded on the run.

    ``None`` when the run carried no draft at all; ``""`` when it
    carried a draft whose template name was never recorded (the
    pre-link shape) — an honest unknown, never a guess.
    """
    if record.draft is None:
        return None
    return record.draft.template or ""


def _rate(count: int, total: int) -> float | None:
    """A quotable rate, or ``None`` below the floor (counts stand)."""
    if total < EFFECTIVENESS_MIN_USES:
        return None
    return round(count / total, 4)


def _figures(runs: list[TicketResult], reopened_roots: set[str]) -> dict:
    """The counts and rates over one slice of outcome-bearing runs."""
    uses = len(runs)
    resolved = sum(1 for r in runs if r.outcome is not None and r.outcome.resolved)
    reopened = sum(1 for r in runs if r.ticket_id in reopened_roots)
    escalated = sum(
        1
        for r in runs
        if r.outcome is not None
        and r.outcome.final_queue in ESCALATION_QUEUES
        and r.outcome.final_queue != r.composition.queue
    )
    return {
        "uses": uses,
        "resolved": resolved,
        "resolution_rate": _rate(resolved, uses),
        "reopened": reopened,
        "reopen_rate": _rate(reopened, uses),
        "escalated": escalated,
        "escalation_rate": _rate(escalated, uses),
    }


def effectiveness_report(records: list[TicketResult]) -> dict:
    """Per-template effectiveness over the outcome-bearing runs.

    Pure: reads the records, writes nothing. See the module
    docstring for the definitions; every threshold the report
    applies is printed on it.
    """
    reopened_roots = {
        r.reopen.root_ticket_id for r in records if r.reopen is not None
    }
    bearing = [r for r in records if r.outcome is not None]
    measured: list[TicketResult] = []
    excluded_duplicates = 0
    for record in bearing:
        if corrections.active_duplicate_mark(record) is not None:
            excluded_duplicates += 1
        else:
            measured.append(record)

    # Category baselines: every measured outcome-bearing run in
    # the category, templated or not — the water the template
    # swims in. The unknown-template runs belong here: their
    # category is recorded even when their template is not.
    by_category: dict[str, list[TicketResult]] = {}
    for record in measured:
        by_category.setdefault(record.category, []).append(record)
    categories = [
        {"category": category, **_figures(runs, reopened_roots)}
        for category, runs in sorted(by_category.items())
    ]
    baseline_by_category = {row["category"]: row for row in categories}

    by_template: dict[tuple[str, str], list[TicketResult]] = {}
    with_template = 0
    unknown_template = 0
    no_template = 0
    for record in measured:
        name = template_name(record)
        if name is None:
            no_template += 1
        elif name == "":
            unknown_template += 1
        else:
            with_template += 1
            by_template.setdefault((name, record.category), []).append(record)

    templates = []
    for (name, category), runs in sorted(by_template.items()):
        row: dict = {"template": name, "category": category}
        row.update(_figures(runs, reopened_roots))
        baseline = baseline_by_category[category]
        row["baseline_reopen_rate"] = baseline["reopen_rate"]
        row["flag"] = ""
        row["flag_detail"] = ""
        if row["reopen_rate"] is None:
            row["note"] = (
                f"rates withheld: {row['uses']} outcome-bearing use(s) is "
                f"below the {EFFECTIVENESS_MIN_USES}-use floor for quoting "
                "a rate — the counts stand, the percentages wait"
            )
        else:
            row["note"] = (
                f"figures over the {row['uses']} outcome-bearing use(s) "
                "of this template"
            )
            if (
                baseline["reopen_rate"] is not None
                and row["reopen_rate"] - baseline["reopen_rate"]
                > REOPEN_REVIEW_MARGIN
            ):
                row["flag"] = NEEDS_REVIEW
                row["flag_detail"] = (
                    f"reopen rate {row['reopen_rate']:.2f} beats the "
                    f"{category} baseline {baseline['reopen_rate']:.2f} by "
                    f"more than the {REOPEN_REVIEW_MARGIN:.2f} review "
                    "margin — the template's owners should read the "
                    "reopened tickets; nothing changes automatically"
                )
        templates.append(row)

    flagged = sum(1 for row in templates if row["flag"] == NEEDS_REVIEW)
    if not bearing:
        note = "no recorded outcomes yet — a template is judged only where a ticket's later life is known"
    elif not templates:
        note = (
            f"{len(measured)} outcome-bearing run(s) measured and none "
            "carried a named template — nothing to grade yet"
        )
    else:
        note = (
            f"{len(templates)} template(s) over {with_template} "
            "outcome-bearing use(s)"
            + (
                f"; {flagged} flagged needs_review"
                if flagged
                else "; none flagged needs_review"
            )
            + " — a flag advises the template's owners and changes nothing"
        )

    return {
        "templates": templates,
        "categories": categories,
        "totals": {
            "outcome_bearing": len(bearing),
            "measured": len(measured),
            "excluded_confirmed_duplicates": excluded_duplicates,
            "with_template": with_template,
            "unknown_template": unknown_template,
            "no_template": no_template,
        },
        "min_uses_for_rate": EFFECTIVENESS_MIN_USES,
        "reopen_review_margin": REOPEN_REVIEW_MARGIN,
        "escalation_queues": list(ESCALATION_QUEUES),
        "note": note,
    }


def template_stats_for_run(report: dict, record: TicketResult) -> dict:
    """One run's slice of the report — the audit export's block.

    The export shows the run beside the track record of the
    template it carried (the report row, verbatim), or says
    plainly that the run carried none — or one whose name was
    never recorded, in which case no track record can be
    attributed and the block says that instead of borrowing
    another template's numbers.
    """
    name = template_name(record)
    if name is None:
        return {
            "template": "",
            "state": "no_template",
            "note": "this run carried no template draft — nothing to measure",
        }
    if name == "":
        return {
            "template": "",
            "state": "unknown_template",
            "note": (
                "this run's draft carries no template name (the "
                "pre-link record shape) — it counts in the totals, "
                "and no template's track record is attributed to it"
            ),
        }
    for row in report["templates"]:
        if row["template"] == name and row["category"] == record.category:
            return {"template": name, "state": "measured", **row}
    return {
        "template": name,
        "state": "unmeasured",
        "note": (
            "no outcome-bearing uses of this template are measured "
            "yet — a template is judged only where a ticket's later "
            "life is known"
        ),
    }
