"""Template effectiveness: the drafts, graded by what happened next.

Pins, in order:

- the rates' definitions, from the outcome vocabulary only:
  resolved is the outcome's flag, reopened is a chain rooting at
  the run, escalated is the outcome's final queue landing in a
  human escalation queue triage had not routed to;
- the small-sample floor: counts always shown, rates withheld
  below EFFECTIVENESS_MIN_USES outcome-bearing uses;
- the needs_review flag: a reopen rate beating the category
  baseline by MORE than the margin (the exact margin is not
  enough), and only when both rates are quotable;
- the accounting: an unnamed draft (the pre-link shape) counts in
  the totals and the category baseline but is attributed to no
  template; a run without an outcome is not a use; a confirmed
  duplicate is excluded from the rates and named in the totals;
- the firewall: the report is a pure read, and a ticket composes
  byte-identically whether or not its template stands flagged —
  the flag suppresses nothing, moves nothing;
- the surfaces: the service method, the stats endpoint, and the
  audit export's template block (the run beside its template's
  track record).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_triage import api, template_effectiveness as te
from support_triage.exports import render_text
from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    DraftInfo,
    DuplicateMark,
    ReopenInfo,
    TicketInput,
    TicketResult,
)

from helpers import fresh_service

DASH = "how_to_first_response[dashboard]"
API_T = "how_to_first_response[api]"


def outcome_run(
    ticket_id: str,
    *,
    category: str = "how_to",
    template: str | None = API_T,
    resolved: bool = True,
    final_queue: str = "general_support",
    queue: str = "general_support",
    with_outcome: bool = True,
    **overrides,
) -> TicketResult:
    base = dict(
        ticket_id=ticket_id,
        status="triaged",
        category=category,
        composition=CompositionInfo(queue=queue),
        draft=(
            DraftInfo(text="draft", template=template)
            if template is not None
            else None
        ),
        outcome=(
            AgentOutcome(
                final_category=category,
                final_queue=final_queue,
                resolved=resolved,
                recorded_by="agent",
            )
            if with_outcome
            else None
        ),
    )
    base.update(overrides)
    return TicketResult(**base)


def reopen_of(root: str, n: int = 1) -> TicketResult:
    return TicketResult(
        ticket_id=f"{root}-R{n}",
        status="awaiting_agent",
        category="how_to",
        reopen=ReopenInfo(
            prior_run_id=root, root_ticket_id=root, reopen_count=n
        ),
    )


def row_for(report: dict, template: str) -> dict:
    return next(r for r in report["templates"] if r["template"] == template)


# ---------------------------------------------------------------------------
# the rates, from the outcome vocabulary
# ---------------------------------------------------------------------------


def test_rates_are_counts_over_outcome_bearing_uses():
    records = [
        outcome_run("T-1", resolved=True),
        outcome_run("T-2", resolved=True),
        outcome_run("T-3", resolved=False),
        outcome_run("T-4", resolved=True, final_queue="engineering_escalation"),
        outcome_run("T-5", resolved=False),
        outcome_run("T-6", resolved=True),
        reopen_of("T-2"),
        reopen_of("T-5"),
        reopen_of("T-5", n=2),  # a chain of two still counts the run once
    ]
    report = te.effectiveness_report(records)
    row = row_for(report, API_T)
    assert row["uses"] == 6
    assert row["resolved"] == 4 and row["resolution_rate"] == 0.6667
    assert row["reopened"] == 2 and row["reopen_rate"] == 0.3333
    assert row["escalated"] == 1 and row["escalation_rate"] == 0.1667
    assert report["totals"]["outcome_bearing"] == 6


def test_escalation_must_be_the_agents_move_not_triages():
    # Triage itself routed this one to engineering; the outcome
    # agreeing is not an escalation by the agent's hand.
    records = [
        outcome_run(
            f"E-{i}",
            final_queue="engineering_escalation",
            queue="engineering_escalation",
        )
        for i in range(5)
    ]
    row = row_for(te.effectiveness_report(records), API_T)
    assert row["escalated"] == 0 and row["escalation_rate"] == 0.0


def test_a_run_without_an_outcome_is_not_a_use():
    records = [outcome_run("N-1", with_outcome=False)]
    report = te.effectiveness_report(records)
    assert report["templates"] == []
    assert report["totals"]["outcome_bearing"] == 0
    assert "no recorded outcomes yet" in report["note"]


# ---------------------------------------------------------------------------
# the floor and the flag
# ---------------------------------------------------------------------------


def test_rates_withheld_below_the_floor_counts_stand():
    records = [outcome_run(f"W-{i}", resolved=(i % 2 == 0)) for i in range(4)]
    report = te.effectiveness_report(records)
    row = row_for(report, API_T)
    assert row["uses"] == 4 and row["resolved"] == 2
    assert row["resolution_rate"] is None
    assert row["reopen_rate"] is None and row["escalation_rate"] is None
    assert row["flag"] == ""
    assert "withheld" in row["note"]


def test_flag_fires_when_reopens_beat_the_baseline_by_the_margin():
    records = [outcome_run(f"F-{i}", resolved=False) for i in range(5)]
    records += [reopen_of(f"F-{i}") for i in range(4)]  # template: 4/5 reopened
    records += [
        outcome_run(f"B-{i}", template=None, resolved=True) for i in range(5)
    ]  # no-template how_to runs: the rest of the baseline, none reopened
    report = te.effectiveness_report(records)
    row = row_for(report, API_T)
    baseline = next(
        c for c in report["categories"] if c["category"] == "how_to"
    )
    assert baseline["uses"] == 10 and baseline["reopen_rate"] == 0.4
    assert row["reopen_rate"] == 0.8 and row["baseline_reopen_rate"] == 0.4
    assert row["flag"] == te.NEEDS_REVIEW
    assert "nothing changes automatically" in row["flag_detail"]
    assert report["reopen_review_margin"] == te.REOPEN_REVIEW_MARGIN


def test_exactly_the_margin_does_not_flag():
    # Template 3/5 reopened (0.6); category baseline 7/20 (0.35):
    # the difference is exactly the margin — "more than" is the rule.
    records = [outcome_run(f"M-{i}", resolved=False) for i in range(5)]
    records += [reopen_of(f"M-{i}") for i in range(3)]
    records += [
        outcome_run(f"O-{i}", template=None, resolved=True) for i in range(15)
    ]
    records += [reopen_of(f"O-{i}") for i in range(4)]
    row = row_for(te.effectiveness_report(records), API_T)
    assert row["reopen_rate"] == 0.6 and row["baseline_reopen_rate"] == 0.35
    assert row["flag"] == ""


def test_no_flag_when_the_baseline_itself_is_unquotable():
    # The template's own slice is quotable but the category has so
    # few OTHER runs that... the baseline includes the template's
    # runs, so build the unquotable case directly: a category whose
    # total outcome-bearing uses sit below the floor cannot happen
    # for a quotable template — instead pin the withheld-template
    # side: below the floor there is no rate to beat anything with.
    records = [outcome_run(f"U-{i}", resolved=False) for i in range(4)]
    records += [reopen_of(f"U-{i}") for i in range(4)]  # all reopened
    row = row_for(te.effectiveness_report(records), API_T)
    assert row["reopen_rate"] is None and row["flag"] == ""


# ---------------------------------------------------------------------------
# the accounting: unknown, none, duplicates
# ---------------------------------------------------------------------------


def test_unnamed_draft_counts_in_totals_and_baseline_never_attributed():
    records = [outcome_run(f"K-{i}", template="") for i in range(5)]
    records += [reopen_of("K-0")]
    report = te.effectiveness_report(records)
    assert report["templates"] == []  # no name, no row
    assert report["totals"]["unknown_template"] == 5
    baseline = report["categories"][0]
    assert baseline["category"] == "how_to" and baseline["uses"] == 5
    assert baseline["reopened"] == 1


def test_no_template_runs_count_as_their_own_bucket():
    records = [outcome_run(f"Z-{i}", template=None) for i in range(3)]
    totals = te.effectiveness_report(records)["totals"]
    assert totals["no_template"] == 3 and totals["with_template"] == 0
    assert totals["measured"] == 3


def test_confirmed_duplicate_is_excluded_from_rates_and_named():
    records = [outcome_run(f"D-{i}") for i in range(5)]
    duplicate = outcome_run(
        "D-dup",
        resolved=False,
        duplicate_of="D-0",
        duplicate_marks=[
            DuplicateMark(
                duplicate_of="D-0",
                marked_by="operator",
                reason="same customer, same ask",
                marked_at="2026-10-01T10:00:00Z",
            )
        ],
    )
    report = te.effectiveness_report(records + [duplicate, reopen_of("D-dup")])
    row = row_for(report, API_T)
    assert row["uses"] == 5  # the duplicate's outcome is not evidence
    totals = report["totals"]
    assert totals["outcome_bearing"] == 6
    assert totals["excluded_confirmed_duplicates"] == 1
    assert totals["measured"] == 5


def test_report_is_a_pure_read():
    records = [outcome_run(f"P-{i}") for i in range(5)] + [reopen_of("P-1")]
    before = json.dumps(
        [r.model_dump() for r in records], sort_keys=True, default=str
    )
    te.effectiveness_report(records)
    after = json.dumps(
        [r.model_dump() for r in records], sort_keys=True, default=str
    )
    assert before == after


def test_template_name_reads_the_record_as_it_stands():
    assert te.template_name(outcome_run("A", template=None)) is None
    assert te.template_name(outcome_run("B", template="")) == ""
    assert te.template_name(outcome_run("C")) == API_T


def test_stats_for_run_states():
    records = [outcome_run(f"S-{i}") for i in range(5)]
    report = te.effectiveness_report(records)
    measured = te.template_stats_for_run(report, outcome_run("S-9"))
    assert measured["state"] == "measured" and measured["uses"] == 5
    none_stats = te.template_stats_for_run(
        report, outcome_run("S-10", template=None)
    )
    assert none_stats["state"] == "no_template"
    unknown_stats = te.template_stats_for_run(
        report, outcome_run("S-11", template="")
    )
    assert unknown_stats["state"] == "unknown_template"
    unmeasured = te.template_stats_for_run(
        report, outcome_run("S-12", template=DASH)
    )
    assert unmeasured["state"] == "unmeasured"


# ---------------------------------------------------------------------------
# the firewall, at the service: a flag changes no composition
# ---------------------------------------------------------------------------

_PROBE = TicketInput(
    ticket_id="PROBE-1",
    subject="How do I change the dashboard date range?",
    body=(
        "How do I change the date range for the chart on the overview "
        "page? I want the dashboard to show the last quarter by "
        "default. Thanks."
    ),
    sender_name="Probe Pat",
    sender_email="probe@example.com",
    channel="email",
)

_SEED = [
    ("FW-D1", "How do I add a chart to the dashboard?",
     "Hi, how do I add a new chart widget to my main dashboard? I would like to pin the weekly revenue chart at the top. Thanks!"),
    ("FW-D2", "How do I schedule a dashboard report?",
     "Hello, how do I schedule the dashboard report so it is emailed to my team every Monday morning? Please let me know the steps."),
    ("FW-D3", "How can I filter the analytics page by region?",
     "Hi there, how can I filter the analytics page so the dashboard only shows the European region? Thanks for the guide."),
    ("FW-D4", "How do I share my dashboard with a colleague?",
     "How do I share my dashboard view with a colleague on another team? She only needs to see the overview page, not edit it."),
    ("FW-D5", "How do I rename a dashboard?",
     "Hi, how do I rename a dashboard I created last month? The chart titles are fine, just the dashboard name itself. No rush."),
    ("FW-A1", "How do I create an API token?",
     "Hi, how do I create a new API token for our integration? I checked the developers section but could not find the token page."),
    ("FW-A2", "How to set up a webhook endpoint?",
     "Hello, how to register a webhook endpoint so our server gets notified when an order changes? A guide would be great."),
    ("FW-A3", "How can I paginate API results?",
     "How can I paginate through API results with curl? The endpoint returns fifty records and I need all of them. Thanks."),
    ("FW-A4", "Where can I find the API response codes?",
     "Where can I find what the API response codes mean? Our SDK shows a code I do not recognise and I want the reference."),
    ("FW-A5", "How do I test the API in a sandbox?",
     "Hi, how do I test the API in a sandbox before going live? Is it possible to point the SDK at a test endpoint?"),
]


def _seed_flagged_world(service) -> dict:
    """Five dashboard-template tickets, every one reopened; five
    api-template tickets, all resolved clean. The dashboard
    template must stand flagged when this returns."""
    for ticket_id, subject, body in _SEED:
        result = service.run_ticket(
            TicketInput(
                ticket_id=ticket_id,
                subject=subject,
                body=body,
                sender_name=f"User {ticket_id}",
                sender_email=f"{ticket_id.lower()}@example.com",
                channel="email",
            )
        ).result
        assert result.draft is not None, f"premise: {ticket_id} earned no draft"
        service.record_outcome(
            ticket_id,
            AgentOutcome(
                final_category="how_to",
                final_queue="general_support",
                resolved=not ticket_id.startswith("FW-D"),
                recorded_by="test-agent",
            ),
        )
        if ticket_id.startswith("FW-D"):
            service.reopen_run(
                ticket_id,
                message="I tried that and it is still not what I see — can you help?",
                actor="test-agent",
            )
    report = service.template_effectiveness()
    row = row_for(report, DASH)
    assert row["flag"] == te.NEEDS_REVIEW, f"premise: no flag in {report}"
    return report


def test_flag_changes_no_composition_and_suppresses_no_draft():
    clean = fresh_service()
    before = clean.run_ticket(_PROBE).result

    flagged = fresh_service()
    _seed_flagged_world(flagged)
    after = flagged.run_ticket(_PROBE).result

    assert after.composition.model_dump() == before.composition.model_dump()
    assert [a.model_dump() for a in after.answers] == [
        a.model_dump() for a in before.answers
    ]
    # The flagged template is still offered, word for word: the
    # flag advises the owners, it does not gag the draft.
    assert after.draft is not None and before.draft is not None
    assert after.draft.template == before.draft.template == DASH


def test_service_report_and_export_carry_the_flag():
    service = fresh_service()
    _seed_flagged_world(service)
    document = service.export_run("FW-D1")
    stats = document["template_effectiveness"]
    assert stats["state"] == "measured" and stats["template"] == DASH
    assert stats["flag"] == te.NEEDS_REVIEW
    text = render_text(document)
    assert "TEMPLATE EFFECTIVENESS" in text and "needs_review" in text

    clean_document = service.export_run("FW-A1")
    assert clean_document["template_effectiveness"]["flag"] == ""


def test_export_of_a_run_without_a_draft_says_so():
    service = fresh_service()
    service.run_ticket(
        TicketInput(
            ticket_id="PLAIN-1",
            subject="Charged twice",
            body="I was charged twice for the March subscription — please refund one.",
            sender_name="Priya Nair",
            sender_email="priya@example.com",
            channel="email",
        )
    )
    document = service.export_run("PLAIN-1")
    assert document["template_effectiveness"]["state"] == "no_template"


# ---------------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def test_stats_templates_endpoint_on_an_empty_store(client):
    response = client.get("/api/stats/templates")
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["outcome_bearing"] == 0
    assert body["min_uses_for_rate"] == te.EFFECTIVENESS_MIN_USES
    assert body["reopen_review_margin"] == te.REOPEN_REVIEW_MARGIN
