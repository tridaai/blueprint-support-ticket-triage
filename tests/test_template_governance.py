"""Template governance: a template earns its way live, and can lose the job.

Pins, in order:

- the registry's defaults: with no governed record a template
  stands bundled — live, version 1, the wording as shipped — and
  the authored constants ride the payload;
- authoring: a named editor with a stated basis starts a record
  in DRAFT at version 1; a draft template drafts nothing, and the
  run's trace says the step was skipped for state;
- the refusals, by name: no actor, no basis, an unknown area,
  words the draft builder cannot fill, authoring over a governed
  or retired name;
- promotion: a second pair of eyes — the version's author is
  refused (self-promotion), a missing basis is refused, a live or
  retired template cannot be promoted — and after it, the
  authored words serve at the record's version;
- editing: the version bumps, the editor becomes the author, a
  live template returns to draft (and stops drafting) until
  re-promoted; a no-change edit and an edit of a retired
  template are refused; editing a bundled template governs it;
- retirement: a named editor with a reason; the template drafts
  nothing from then on; retiring twice is a conflict; governing
  never writes to a run's record;
- the retirement watch: round 11's figures, reused — due when
  the needs_review flag stands on at least RETIREMENT_MIN_USES
  uses, not due below that floor however loud the flag, and a
  due template is still LIVE (a proposal is not an act);
- the surfaces: the service views, the API (201/409/422/404),
  and the audit export's version stamp.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_triage import api, template_governance as tg
from support_triage.exports import render_text
from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    DraftInfo,
    ReopenInfo,
    TicketInput,
    TicketResult,
)

from helpers import fresh_service

API_T = "how_to_first_response[api]"
DASH_T = "how_to_first_response[dashboard]"
CHECKOUT_T = "how_to_first_response[checkout]"

AUTHORED_TEXT = (
    "Hi {name},\n\n"
    "Thanks for writing in. The steps in {guide} cover exactly "
    "this — work through them in order.\n\n"
    "— Support (template draft, sent by an agent)"
)
AUTHORED_GUIDE = "the API handbook (Developers → API handbook)"

_API_TEXTS = [
    (
        "How do I create an API token?",
        "Hi, how do I create a new API token for our integration? "
        "I checked the developers section but could not find the "
        "token page.",
    ),
    (
        "How to set up a webhook endpoint?",
        "Hello, how to register a webhook endpoint so our server "
        "gets notified when an order changes? A guide would be great.",
    ),
    (
        "How can I paginate API results?",
        "How can I paginate through API results with curl? The "
        "endpoint returns fifty records and I need all of them. Thanks.",
    ),
]

_RUNS = {"n": 0}


def _run_api_ticket(service):
    _RUNS["n"] += 1
    n = _RUNS["n"]
    subject, body = _API_TEXTS[(n - 1) % len(_API_TEXTS)]
    ticket = TicketInput(
        ticket_id=f"GOV-T{n}",
        subject=subject,
        body=body,
        sender_name="Gov Tester",
        sender_email=f"gov{n}@example.com",
        channel="email",
    )
    return service.run_ticket(ticket).result


def _skip_lines(result) -> list[str]:
    return [
        detail
        for step in result.trace
        for detail in step.details
        if "template draft: skipped" in detail
    ]


def _author_api(service, actor="T. Editor"):
    return service.author_template(
        area="api",
        text=AUTHORED_TEXT,
        guide=AUTHORED_GUIDE,
        actor=actor,
        reason="the handbook is where API answers live now",
    )


# ---------------------------------------------------------------------------
# the registry's defaults
# ---------------------------------------------------------------------------


def test_registry_stands_bundled_until_governed():
    service = fresh_service()
    payload = service.list_templates()
    rows = {row["name"]: row for row in payload["templates"]}
    assert len(rows) == 7
    for row in rows.values():
        assert row["state"] == "live"
        assert row["version"] == 1
        assert row["source"] == "bundled"
        assert row["retirement_due"] is False
    assert payload["constants"]["retirement_min_uses"] == tg.RETIREMENT_MIN_USES
    assert payload["constants"]["states"] == ["draft", "live", "retired"]


def test_bundled_template_drafts_as_always():
    service = fresh_service()
    result = _run_api_ticket(service)
    assert result.draft is not None
    assert result.draft.template == API_T
    assert result.draft.template_version == 1
    assert "API reference" in result.draft.text  # the bundled guide
    assert _skip_lines(result) == []


# ---------------------------------------------------------------------------
# authoring
# ---------------------------------------------------------------------------


def test_authoring_starts_a_draft_and_the_draft_does_not_draft():
    service = fresh_service()
    record = _author_api(service)
    assert record.state == "draft" and record.version == 1
    assert record.authored_by == "T. Editor"
    assert [act.action for act in record.history] == ["authored"]

    result = _run_api_ticket(service)
    assert result.draft is None
    assert _skip_lines(result) == [
        f"template draft: skipped — template {API_T} is draft "
        "(not live); no draft offered"
    ]


def test_authoring_refusals_are_named():
    service = fresh_service()
    with pytest.raises(tg.TemplateGovernanceError, match="actor is required"):
        service.author_template(
            area="api", text=AUTHORED_TEXT, guide=AUTHORED_GUIDE,
            actor="", reason="basis",
        )
    with pytest.raises(tg.TemplateGovernanceError, match="reason is required"):
        service.author_template(
            area="api", text=AUTHORED_TEXT, guide=AUTHORED_GUIDE,
            actor="T. Editor", reason="",
        )
    with pytest.raises(tg.TemplateGovernanceError, match="unknown product area"):
        service.author_template(
            area="teleport", text=AUTHORED_TEXT, guide=AUTHORED_GUIDE,
            actor="T. Editor", reason="basis",
        )
    with pytest.raises(tg.TemplateGovernanceError, match="must carry"):
        service.author_template(
            area="api", text="Hi {name}, no pointer here.",
            guide=AUTHORED_GUIDE, actor="T. Editor", reason="basis",
        )
    with pytest.raises(tg.TemplateGovernanceError, match="cannot be filled"):
        service.author_template(
            area="api", text="Hi {name}, see {guide} and {ticket_price}.",
            guide=AUTHORED_GUIDE, actor="T. Editor", reason="basis",
        )


def test_authoring_over_a_governed_name_is_a_conflict():
    service = fresh_service()
    _author_api(service)
    with pytest.raises(tg.TemplateGovernanceConflict, match="already stands governed"):
        _author_api(service)


# ---------------------------------------------------------------------------
# promotion
# ---------------------------------------------------------------------------


def test_self_promotion_is_refused_and_a_second_pair_of_eyes_promotes():
    service = fresh_service()
    _author_api(service)
    with pytest.raises(tg.TemplateGovernanceError, match="self-promotion"):
        service.promote_template(API_T, actor="T. Editor", reason="looks fine")
    with pytest.raises(tg.TemplateGovernanceError, match="reason is required"):
        service.promote_template(API_T, actor="R. Approver", reason="")

    record = service.promote_template(
        API_T, actor="R. Approver", reason="read against the handbook; wording is accurate"
    )
    assert record.state == "live" and record.version == 1
    assert [act.action for act in record.history] == ["authored", "promoted"]

    result = _run_api_ticket(service)
    assert result.draft is not None
    assert "API handbook" in result.draft.text  # the authored words serve
    assert result.draft.template_version == 1
    assert _skip_lines(result) == []


def test_promoting_what_is_not_a_draft_is_a_conflict():
    service = fresh_service()
    with pytest.raises(tg.TemplateGovernanceConflict, match="already live"):
        service.promote_template(API_T, actor="R. Approver", reason="basis")
    _author_api(service)
    service.retire_template(API_T, actor="T. Editor", reason="pulled")
    with pytest.raises(tg.TemplateGovernanceConflict, match="retired"):
        service.promote_template(API_T, actor="R. Approver", reason="basis")


# ---------------------------------------------------------------------------
# editing
# ---------------------------------------------------------------------------


def test_editing_a_live_template_returns_it_to_draft_at_a_new_version():
    service = fresh_service()
    _author_api(service)
    service.promote_template(API_T, actor="R. Approver", reason="checked")

    record = service.edit_template(
        API_T,
        actor="T. Editor",
        guide="the API handbook, 2nd edition (Developers → API handbook)",
        reason="the handbook was revised",
    )
    assert record.state == "draft" and record.version == 2
    assert record.authored_by == "T. Editor"
    assert [act.action for act in record.history] == [
        "authored", "promoted", "edited",
    ]

    # The edited words do not serve until re-promoted.
    assert _run_api_ticket(service).draft is None
    # And the editor who wrote v2 cannot promote their own words.
    with pytest.raises(tg.TemplateGovernanceError, match="self-promotion"):
        service.promote_template(API_T, actor="T. Editor", reason="ship it")
    service.promote_template(API_T, actor="R. Approver", reason="re-checked")
    result = _run_api_ticket(service)
    assert result.draft is not None and result.draft.template_version == 2
    assert "2nd edition" in result.draft.text


def test_editing_a_bundled_template_governs_it():
    service = fresh_service()
    record = service.edit_template(
        CHECKOUT_T, actor="T. Editor", guide="the checkout handbook",
        reason="the guide moved",
    )
    assert record.state == "draft" and record.version == 2
    view = service.get_template(CHECKOUT_T)
    assert view["source"] == "governed" and view["state"] == "draft"


def test_edit_refusals_are_named():
    service = fresh_service()
    with pytest.raises(KeyError, match="unknown template"):
        service.edit_template("how_to_first_response[nope]", actor="T", guide="g")
    with pytest.raises(tg.TemplateGovernanceError, match="changes nothing"):
        service.edit_template(API_T, actor="T. Editor")  # no words change
    service.retire_template(API_T, actor="T. Editor", reason="pulled")
    with pytest.raises(tg.TemplateGovernanceConflict, match="are history"):
        service.edit_template(API_T, actor="T. Editor", guide="elsewhere")


# ---------------------------------------------------------------------------
# retirement
# ---------------------------------------------------------------------------


def test_retirement_takes_the_template_off_the_shelf():
    service = fresh_service()
    with pytest.raises(tg.TemplateGovernanceError, match="reason is required"):
        service.retire_template(API_T, actor="T. Editor", reason="")
    record = service.retire_template(
        API_T, actor="T. Editor", reason="the pointer has not landed for a month"
    )
    assert record.state == "retired"
    assert [act.action for act in record.history] == ["retired"]

    result = _run_api_ticket(service)
    assert result.draft is None
    assert _skip_lines(result) == [
        f"template draft: skipped — template {API_T} is retired "
        "(not live); no draft offered"
    ]
    with pytest.raises(tg.TemplateGovernanceConflict, match="already retired"):
        service.retire_template(API_T, actor="T. Editor", reason="again")


def test_governance_never_writes_to_a_run_record():
    service = fresh_service()
    result = _run_api_ticket(service)
    before = service.get_record(result.ticket_id).model_dump_json()
    _author_api(service)
    service.promote_template(API_T, actor="R. Approver", reason="checked")
    service.edit_template(API_T, actor="T. Editor", guide="elsewhere", reason="moved")
    service.retire_template(DASH_T, actor="T. Editor", reason="pulled")
    after = service.get_record(result.ticket_id).model_dump_json()
    assert before == after


# ---------------------------------------------------------------------------
# the retirement watch: round 11's figures, reused
# ---------------------------------------------------------------------------


def _fabricated_use(ticket_id: str, template: str) -> TicketResult:
    return TicketResult(
        ticket_id=ticket_id,
        status="triaged",
        category="how_to",
        composition=CompositionInfo(queue="general_support"),
        draft=DraftInfo(text="draft", template=template),
        outcome=AgentOutcome(
            final_category="how_to",
            final_queue="general_support",
            resolved=True,
            recorded_by="agent",
        ),
    )


def _fabricated_reopen(root: str) -> TicketResult:
    return TicketResult(
        ticket_id=f"{root}-R1",
        status="awaiting_agent",
        category="how_to",
        reopen=ReopenInfo(prior_run_id=root, root_ticket_id=root, reopen_count=1),
    )


def _world(service, *, api_uses: int, api_reopened: int, dash_uses: int):
    for i in range(api_uses):
        service.store.save(_fabricated_use(f"W-A{i}", API_T))
        if i < api_reopened:
            service.store.save(_fabricated_reopen(f"W-A{i}"))
    for i in range(dash_uses):
        service.store.save(_fabricated_use(f"W-D{i}", DASH_T))


def test_retirement_due_when_the_flag_stands_over_enough_uses():
    service = fresh_service()
    # api: 10 uses, 8 reopened (0.80); dashboard: 10 clean uses.
    # Baseline 8/20 = 0.40; 0.80 - 0.40 = 0.40 > the 0.25 margin.
    _world(service, api_uses=10, api_reopened=8, dash_uses=10)
    view = service.get_template(API_T)
    assert view["effectiveness"]["flag"] == "needs_review"
    assert view["effectiveness"]["uses"] == 10
    assert view["retirement_due"] is True
    assert "keeps serving until a named editor retires it" in view["retirement_detail"]
    # Due is a proposal: the record still says live.
    assert view["state"] == "live"
    dash = service.get_template(DASH_T)
    assert dash["retirement_due"] is False


def test_flagged_below_the_retirement_floor_is_not_due():
    service = fresh_service()
    # api: 5 uses, all reopened (1.00); dashboard: 5 clean uses.
    # Baseline 5/10 = 0.50; the flag stands (1.00 - 0.50 > 0.25),
    # but five uses are below the retirement floor of ten.
    _world(service, api_uses=5, api_reopened=5, dash_uses=5)
    view = service.get_template(API_T)
    assert view["effectiveness"]["flag"] == "needs_review"
    assert view["retirement_due"] is False
    assert "below the 10-use floor" in view["retirement_detail"]


def test_no_uses_no_verdict_but_a_sentence():
    service = fresh_service()
    view = service.get_template(API_T)
    assert view["retirement_due"] is False
    assert "no outcome-bearing uses" in view["retirement_detail"]


# ---------------------------------------------------------------------------
# the export stamp
# ---------------------------------------------------------------------------


def test_export_names_the_version_the_draft_was_written_under():
    service = fresh_service()
    result = _run_api_ticket(service)
    document = service.export_run(result.ticket_id)
    block = document["template_effectiveness"]
    assert block["template_version"] == 1
    assert block["template_state_at_draft"] == "live"
    assert "version 1 of the template, live at the time" in render_text(document)


# ---------------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def test_templates_endpoint_lists_the_registry(client):
    response = client.get("/api/templates")
    assert response.status_code == 200
    body = response.json()
    assert len(body["templates"]) == 7
    assert body["constants"]["retirement_min_uses"] == tg.RETIREMENT_MIN_USES


def test_governance_acts_over_http(client):
    created = client.post(
        "/api/templates",
        json={
            "area": "api", "text": AUTHORED_TEXT, "guide": AUTHORED_GUIDE,
            "actor": "T. Editor", "reason": "the handbook moved",
        },
    )
    assert created.status_code == 201
    assert created.json()["state"] == "draft"

    again = client.post(
        "/api/templates",
        json={
            "area": "api", "text": AUTHORED_TEXT, "guide": AUTHORED_GUIDE,
            "actor": "T. Editor", "reason": "twice",
        },
    )
    assert again.status_code == 409

    selfish = client.post(
        f"/api/templates/{API_T}/promote",
        json={"actor": "T. Editor", "reason": "mine"},
    )
    assert selfish.status_code == 422
    assert "self-promotion" in selfish.json()["detail"]

    promoted = client.post(
        f"/api/templates/{API_T}/promote",
        json={"actor": "R. Approver", "reason": "read and checked"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["state"] == "live"

    retired = client.post(
        f"/api/templates/{API_T}/retire",
        json={"actor": "T. Editor", "reason": "not landing"},
    )
    assert retired.status_code == 200
    assert retired.json()["state"] == "retired"


def test_unknown_template_is_a_404(client):
    assert client.get("/api/templates/how_to_first_response[nope]").status_code == 404
    response = client.post(
        "/api/templates/how_to_first_response[nope]/retire",
        json={"actor": "T. Editor", "reason": "x"},
    )
    assert response.status_code == 404
