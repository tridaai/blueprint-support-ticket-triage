"""Incident clustering (incidents.py): proposals in code, a lead's
confirmation as an overlay, membership that moves nothing.

Pins, by layer:

- the proposal rule — same product area AND category, arrivals
  inside the window of the group's anchor, token containment
  against the anchor at the threshold, the member floor; each
  clause of the rule killed in turn (window edge, overlap edge,
  split bucket, below floor) and the group falls;
- the exclusions — reopen runs, confirmed duplicates, members of
  an open incident, unplaced runs: out of the evidence, counted
  by name;
- the never-re-nag watermark — a dismissed member set is
  suppressed exactly; a set that gains a member proposes again
  and names the dismissal it outgrew;
- the lifecycle — confirm (stored incident, members by origin),
  additions offered and decided per ticket, closure as a named
  act; refusals name their reasons (stale proposal, closed
  incident, non-candidate, missing actor or reason);
- the overlay promise — member runs' records byte-identical
  across confirm, add, decline and close; composition of a new
  ticket in an incident world identical to a fresh one (the
  round-9 firewall, restated for membership);
- the surfaces — queue badge, run-view context line, the audit
  export's incident block (present for a member, absent — and
  the export otherwise unchanged — for everyone else), and the
  API routes with their status codes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api, incidents
from support_triage.schemas import (
    ClusterDismissal,
    DuplicateMark,
    Incident,
    IncidentMember,
    ReopenInfo,
    TicketInput,
    TicketResult,
)

from helpers import fresh_service, make_ticket

# ---------------------------------------------------------------------------
# staged texts (the golden G67 checkout world, proven against the
# offline backend: billing + checkout, anchor overlaps 0.59–0.69)
# ---------------------------------------------------------------------------

D_TICKETS = [
    ("T-D1", "2026-06-02T08:00:00Z", "Mia Ford", "mia@example.com",
     "Charged twice at checkout",
     "I was charged twice at checkout for order 5521. Two identical payments of 59 dollars left my card. The checkout page froze once but I clicked pay a single time. Please refund the duplicate payment."),
    ("T-D2", "2026-06-02T08:40:00Z", "Noa Lind", "noa@example.com",
     "Double payment at checkout",
     "Checkout charged my card twice for the same order. Two payments of 59 dollars show on my card statement. I only placed one order. Please refund one payment."),
    ("T-D3", "2026-06-02T09:15:00Z", "Omar Sy", "omar@example.com",
     "Two charges for one checkout order",
     "There are two charges on my card for a single checkout order. The payment page froze during checkout and now I was charged twice. Refund the second charge please."),
    ("T-D4", "2026-06-02T09:50:00Z", "Pam Reid", "pam@example.com",
     "Checkout took the payment twice",
     "The checkout took payment twice from my card this morning. One order, two payments of 59 dollars. Please refund the extra payment as soon as possible."),
]
D5 = ("T-D5", "2026-06-02T10:30:00Z", "Raj Iyer", "raj@example.com",
      "Charged twice again at checkout",
      "Checkout charged me twice again this morning — two payments of 59 dollars on my card for one order. The page froze at payment. Please refund the duplicate.")
D6 = ("T-D6", "2026-06-03T06:00:00Z", "Tom Beck", "tom@example.com",
      "Charged twice at checkout yesterday",
      "I was charged twice at checkout yesterday for one order. Two payments of 59 dollars on my card. Please refund the duplicate payment.")


def run_d(service, spec) -> TicketResult:
    tid, at, name, email, subject, body = spec
    return service.run_ticket(
        TicketInput(
            ticket_id=tid, subject=subject, body=body, sender_name=name,
            sender_email=email, channel="email", account_tier="pro",
            received_at=at,
        )
    ).result


def stage_checkout_world(service) -> list[TicketResult]:
    return [run_d(service, spec) for spec in D_TICKETS]


def only_proposal(service) -> dict:
    proposals = service.cluster_proposals()["proposals"]
    assert len(proposals) == 1, proposals
    return proposals[0]


def record_blobs(service) -> dict[str, str]:
    return {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }


# ---------------------------------------------------------------------------
# the pure proposer: the rule, clause by clause
# ---------------------------------------------------------------------------


def fabricated(ticket_id, at, *, subject="", body="", area="dashboard",
               category="technical_issue", **kwargs) -> TicketResult:
    return TicketResult(
        ticket_id=ticket_id, subject=subject, received_at=at,
        product_area=area, category=category, **kwargs,
    )


def propose(records, texts=None, customers=None, **kwargs):
    texts = texts or {r.ticket_id: f"{r.subject}\n" for r in records}
    customers = customers or {r.ticket_id: f"c-{r.ticket_id}" for r in records}
    kwargs.setdefault("incidents", [])
    kwargs.setdefault("dismissals", [])
    return incidents.propose_clusters(records, texts, customers, **kwargs)


# Phonetic-alphabet texts: token sets of exactly ten content words,
# so overlaps are arithmetic, not vibes.
ANCHOR_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
SOLID = [
    "alpha bravo charlie delta echo kilo lima mike november oscar",
    "alpha bravo charlie delta foxtrot papa quebec romeo sierra tango",
    "alpha bravo charlie echo golf uniform victor whiskey xray yankee",
]
AT_THRESHOLD = "alpha bravo charlie zulu apple banana cherry dragon eagle falcon"
BELOW_THRESHOLD = "alpha bravo zulu apple banana cherry dragon eagle falcon grape"


def word_records(extra=()):
    records = [
        fabricated("W-1", "2026-06-01T08:00:00Z", subject=ANCHOR_WORDS),
        fabricated("W-2", "2026-06-01T08:30:00Z", subject=SOLID[0]),
        fabricated("W-3", "2026-06-01T09:00:00Z", subject=SOLID[1]),
        fabricated("W-4", "2026-06-01T09:30:00Z", subject=SOLID[2]),
        *extra,
    ]
    texts = {r.ticket_id: r.subject for r in records}
    return records, texts


def test_proposal_rule_holds_and_prints_its_constants():
    records, texts = word_records()
    payload = propose(records, texts)
    assert payload["constants"] == {
        "window_hours": 6, "min_members": 4, "overlap_threshold": 0.30,
    }
    assert len(payload["proposals"]) == 1
    proposal = payload["proposals"][0]
    assert [m["ticket_id"] for m in proposal["members"]] == ["W-1", "W-2", "W-3", "W-4"]
    assert proposal["anchor_ticket_id"] == "W-1"
    assert proposal["signals"]["member_count"] == 4
    assert proposal["signals"]["distinct_customers"] == 4
    assert proposal["proposal_id"] == incidents.proposal_id_for(
        ["W-4", "W-2", "W-1", "W-3"]
    )
    assert "dashboard" in proposal["suggested_title"]


def test_overlap_threshold_is_a_cliff_at_the_authored_value():
    # Exactly 0.30 containment against the anchor (3 of 10 shared):
    # the member joins. At 0.20 the group still forms without it.
    records, texts = word_records(
        extra=[fabricated("W-5", "2026-06-01T10:00:00Z", subject=AT_THRESHOLD)]
    )
    texts["W-5"] = AT_THRESHOLD
    payload = propose(records, texts)
    members = [m["ticket_id"] for m in payload["proposals"][0]["members"]]
    assert members == ["W-1", "W-2", "W-3", "W-4", "W-5"]

    records, texts = word_records(
        extra=[fabricated("W-5", "2026-06-01T10:00:00Z", subject=BELOW_THRESHOLD)]
    )
    texts["W-5"] = BELOW_THRESHOLD
    payload = propose(records, texts)
    members = [m["ticket_id"] for m in payload["proposals"][0]["members"]]
    assert members == ["W-1", "W-2", "W-3", "W-4"]


def test_window_edge_is_inclusive_and_then_closed():
    # The fourth member exactly at anchor + window joins; one
    # minute later the group is three and nothing proposes.
    records, texts = word_records()
    records[3] = fabricated("W-4", "2026-06-01T14:00:00Z", subject=SOLID[2])
    payload = propose(records, texts)
    assert len(payload["proposals"]) == 1

    records, texts = word_records()
    records[3] = fabricated("W-4", "2026-06-01T14:01:00Z", subject=SOLID[2])
    payload = propose(records, texts)
    assert payload["proposals"] == []


def test_split_bucket_never_proposes():
    records, texts = word_records()
    records[3] = fabricated(
        "W-4", "2026-06-01T09:30:00Z", subject=SOLID[2], category="how_to"
    )
    payload = propose(records, texts)
    assert payload["proposals"] == []


def test_below_floor_never_proposes():
    records, texts = word_records()
    payload = propose(records[:3], texts)
    assert payload["proposals"] == []
    assert payload["scanned_runs"] == 3


def test_proposer_is_deterministic():
    records, texts = word_records()
    first = propose(records, texts)
    second = propose(records, texts)
    assert first == second


def test_exclusions_are_out_of_the_evidence_and_counted():
    records, texts = word_records()
    reopen = fabricated(
        "W-5", "2026-06-01T10:00:00Z", subject=SOLID[0],
        reopen=ReopenInfo(prior_run_id="W-1", root_ticket_id="W-1", reopen_count=1),
    )
    duplicate = fabricated(
        "W-6", "2026-06-01T10:05:00Z", subject=SOLID[1],
        duplicate_marks=[
            DuplicateMark(duplicate_of="W-2", marked_by="op", reason="same ask",
                          marked_at="2026-06-01T10:06:00Z")
        ],
        duplicate_of="W-2",
    )
    unplaced = fabricated("W-7", "2026-06-01T10:10:00Z", subject=SOLID[2], category="")
    records += [reopen, duplicate, unplaced]
    texts.update({r.ticket_id: r.subject for r in (reopen, duplicate, unplaced)})
    payload = propose(records, texts)
    members = [m["ticket_id"] for m in payload["proposals"][0]["members"]]
    assert members == ["W-1", "W-2", "W-3", "W-4"]
    assert payload["excluded"] == {
        "reopen_runs": 1, "confirmed_duplicates": 1,
        "open_incident_members": 0, "unplaced": 1,
    }


def test_open_incident_members_are_not_reproposed():
    records, texts = word_records()
    incident = Incident(
        incident_id="INC-0001", title="t", product_area="dashboard",
        category="technical_issue", status="open", anchor_ticket_id="W-1",
        members=[
            IncidentMember(ticket_id=tid, added_by="lead", origin="proposal")
            for tid in ("W-1", "W-2", "W-3", "W-4")
        ],
    )
    payload = propose(records, texts, incidents=[incident])
    assert payload["proposals"] == []
    assert payload["excluded"]["open_incident_members"] == 4


def test_dismissed_set_is_suppressed_and_growth_reproposes():
    records, texts = word_records()
    dismissal = ClusterDismissal(
        member_ticket_ids=["W-1", "W-2", "W-3", "W-4"],
        product_area="dashboard", category="technical_issue",
        dismissed_by="lead", reason="one customer's load test",
        dismissed_at="2026-06-01T12:00:00Z",
    )
    payload = propose(records, texts, dismissals=[dismissal])
    assert payload["proposals"] == []
    assert payload["suppressed_dismissed"] == 1

    grown, grown_texts = word_records(
        extra=[fabricated("W-5", "2026-06-01T10:00:00Z", subject=SOLID[0])]
    )
    grown_texts["W-5"] = SOLID[0]
    payload = propose(grown, grown_texts, dismissals=[dismissal])
    assert len(payload["proposals"]) == 1
    outgrows = payload["proposals"][0]["outgrows_dismissal"]
    assert outgrows["dismissed_by"] == "lead"
    assert outgrows["dismissed_member_count"] == 4


# ---------------------------------------------------------------------------
# the lifecycle, through the service
# ---------------------------------------------------------------------------


def test_confirm_creates_the_incident_and_touches_no_run():
    service = fresh_service()
    stage_checkout_world(service)
    before = record_blobs(service)
    proposal = only_proposal(service)
    incident = service.confirm_cluster(proposal["proposal_id"], actor="L. Lead")
    assert incident.incident_id == "INC-0001"
    assert incident.status == "open"
    assert incident.opened_by == "L. Lead"
    assert incident.title == proposal["suggested_title"]
    assert incident.member_ids() == [spec[0] for spec in D_TICKETS]
    assert all(m.origin == "proposal" for m in incident.members)
    assert incident.signals["member_count"] == 4
    assert record_blobs(service) == before
    # And the confirmed set leaves the proposal list.
    assert service.cluster_proposals()["proposals"] == []


def test_confirm_can_rename_and_requires_an_actor():
    service = fresh_service()
    stage_checkout_world(service)
    proposal = only_proposal(service)
    with pytest.raises(incidents.IncidentError, match="actor is required"):
        service.confirm_cluster(proposal["proposal_id"], actor=" ")
    incident = service.confirm_cluster(
        proposal["proposal_id"], actor="L. Lead", title="Gateway double-charge"
    )
    assert incident.title == "Gateway double-charge"


def test_confirming_a_stale_proposal_is_a_conflict():
    service = fresh_service()
    stage_checkout_world(service)
    with pytest.raises(incidents.IncidentConflict, match="no longer stands"):
        service.confirm_cluster("clu-doesnotexist", actor="L. Lead")


def test_dismissal_requires_actor_and_reason_and_suppresses():
    service = fresh_service()
    stage_checkout_world(service)
    proposal = only_proposal(service)
    with pytest.raises(incidents.IncidentError, match="actor is required"):
        service.dismiss_cluster(proposal["proposal_id"], actor="", reason="x")
    with pytest.raises(incidents.IncidentError, match="reason is required"):
        service.dismiss_cluster(proposal["proposal_id"], actor="L. Lead", reason=" ")
    dismissal = service.dismiss_cluster(
        proposal["proposal_id"], actor="L. Lead", reason="One customer's retry storm"
    )
    assert dismissal.member_ticket_ids == sorted(spec[0] for spec in D_TICKETS)
    payload = service.cluster_proposals()
    assert payload["proposals"] == []
    assert payload["suppressed_dismissed"] == 1
    assert payload["dismissed_sets"] == 1


def test_additions_add_decline_close_and_the_records_never_move():
    service = fresh_service()
    stage_checkout_world(service)
    proposal = only_proposal(service)
    service.confirm_cluster(proposal["proposal_id"], actor="L. Lead")
    before = record_blobs(service)

    run_d(service, D5)
    candidates = service.incident_additions("INC-0001")
    assert [c["ticket_id"] for c in candidates] == ["T-D5"]

    # A next-day lookalike is outside the window: not a candidate,
    # and the refusal names it.
    run_d(service, D6)
    assert [c["ticket_id"] for c in service.incident_additions("INC-0001")] == ["T-D5"]
    with pytest.raises(incidents.IncidentError, match="not a candidate"):
        service.add_incident_member("INC-0001", "T-D6", actor="L. Lead")

    view = service.add_incident_member("INC-0001", "T-D5", actor="L. Lead")
    members = {m["ticket_id"]: m["origin"] for m in view["members"]}
    assert members["T-D5"] == "addition"
    assert members["T-D1"] == "proposal"
    after_add = record_blobs(service)
    for ticket_id, blob in before.items():
        assert after_add[ticket_id] == blob

    closed = service.close_incident(
        "INC-0001", actor="L. Lead", reason="Gateway fix deployed"
    )
    assert closed.status == "closed"
    assert closed.closed_by == "L. Lead"
    # Closure decided nothing about any member.
    for ticket_id, blob in before.items():
        assert record_blobs(service)[ticket_id] == blob
    member = service.get_record("T-D1")
    assert member.status == "awaiting_agent" and member.decision == ""
    with pytest.raises(incidents.IncidentConflict, match="already closed"):
        service.close_incident("INC-0001", actor="L. Lead", reason="again")
    with pytest.raises(incidents.IncidentConflict, match="closed"):
        service.add_incident_member("INC-0001", "T-D6", actor="L. Lead")


def test_declined_addition_is_never_reoffered():
    service = fresh_service()
    stage_checkout_world(service)
    service.confirm_cluster(only_proposal(service)["proposal_id"], actor="L. Lead")
    run_d(service, D5)
    assert [c["ticket_id"] for c in service.incident_additions("INC-0001")] == ["T-D5"]
    service.decline_incident_addition(
        "INC-0001", "T-D5", actor="L. Lead", reason="Different order path"
    )
    assert service.incident_additions("INC-0001") == []
    with pytest.raises(incidents.IncidentError, match="already declined"):
        service.add_incident_member("INC-0001", "T-D5", actor="L. Lead")


def test_membership_does_not_change_what_a_new_ticket_composes():
    # The firewall, restated for incidents (round 9's dossier
    # precedent): a probe ticket triaged in a world with an open
    # incident composes byte-identically to the same ticket in a
    # fresh service — membership annotates views, never answers.
    service = fresh_service()
    stage_checkout_world(service)
    service.confirm_cluster(only_proposal(service)["proposal_id"], actor="L. Lead")
    probe = make_ticket(
        ticket_id="PROBE-1",
        subject="Charged twice at checkout",
        body=("I was charged twice at checkout for order 9917. Two payments "
              "of 59 dollars left my card. Please refund the duplicate payment."),
        sender_name="Probe Person", sender_email="probe@example.com",
        received_at="2026-06-02T11:30:00Z",
    )
    here = service.run_ticket(probe).result
    fresh = fresh_service().run_ticket(probe).result
    assert here.composition.model_dump() == fresh.composition.model_dump()
    assert [a.model_dump() for a in here.answers] == [
        a.model_dump() for a in fresh.answers
    ]
    here_draft = here.draft.text if here.draft is not None else None
    fresh_draft = fresh.draft.text if fresh.draft is not None else None
    assert here_draft == fresh_draft


# ---------------------------------------------------------------------------
# the surfaces: queue badge, run view, export, API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    api.set_service(fresh_service())
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def _api_stage(client) -> str:
    service = api.get_service()
    stage_checkout_world(service)
    return service.cluster_proposals()["proposals"][0]["proposal_id"]


def test_api_proposals_confirm_and_incident_page(client):
    proposal_id = _api_stage(client)
    payload = client.get("/api/incidents/proposals").json()
    assert payload["constants"]["min_members"] == 4
    assert [p["proposal_id"] for p in payload["proposals"]] == [proposal_id]

    refused = client.post(
        f"/api/incidents/proposals/{proposal_id}/confirm", json={"actor": ""}
    )
    assert refused.status_code == 422

    created = client.post(
        f"/api/incidents/proposals/{proposal_id}/confirm",
        json={"actor": "L. Lead"},
    )
    assert created.status_code == 201
    assert created.json()["incident_id"] == "INC-0001"

    page = client.get("/api/incidents/INC-0001").json()
    assert page["member_count"] == 4
    assert page["open_members"] == 4
    assert page["addition_candidates"] == []
    assert page["context_line"].startswith("incident: INC-0001 · ")

    listing = client.get("/api/incidents").json()
    assert [v["incident"]["incident_id"] for v in listing["incidents"]] == ["INC-0001"]
    assert client.get("/api/incidents/INC-9999").status_code == 404


def test_api_dismiss_and_close_status_codes(client):
    proposal_id = _api_stage(client)
    no_reason = client.post(
        f"/api/incidents/proposals/{proposal_id}/dismiss",
        json={"actor": "L. Lead", "reason": ""},
    )
    assert no_reason.status_code == 422
    dismissed = client.post(
        f"/api/incidents/proposals/{proposal_id}/dismiss",
        json={"actor": "L. Lead", "reason": "Not one event"},
    )
    assert dismissed.status_code == 201
    assert dismissed.json()["dismissed_by"] == "L. Lead"
    # The dismissed proposal no longer stands to be confirmed.
    stale = client.post(
        f"/api/incidents/proposals/{proposal_id}/confirm", json={"actor": "L. Lead"}
    )
    assert stale.status_code == 409


def test_queue_badge_and_run_view_context_line(client):
    proposal_id = _api_stage(client)
    client.post(
        f"/api/incidents/proposals/{proposal_id}/confirm", json={"actor": "L. Lead"}
    )
    rows = client.get("/queue", params={"status": "awaiting_agent"}).json()["tickets"]
    by_id = {row["ticket_id"]: row for row in rows}
    assert by_id["T-D1"]["incident"]["incident_id"] == "INC-0001"
    assert by_id["T-D1"]["incident"]["member_count"] == 4

    view = client.get("/tickets/T-D1").json()
    assert view["incident_context"].startswith("incident: INC-0001 · ")
    assert view["incidents"][0]["origin"] == "proposal"
    # The stored record inside the view is the record, unchanged:
    assert view["status"] == "awaiting_agent"
    assert "incident" not in view["composition"]


def test_export_carries_the_incident_block_for_members_only(client):
    service = api.get_service()
    proposal_id = _api_stage(client)
    client.post(
        f"/api/incidents/proposals/{proposal_id}/confirm", json={"actor": "L. Lead"}
    )
    member_doc = client.get("/api/runs/T-D1/export").json()
    assert member_doc["incident"]["memberships"][0]["incident_id"] == "INC-0001"
    text = client.get("/api/runs/T-D1/export", params={"format": "text"}).text
    assert "INCIDENT (this run was worked as part of an event" in text
    assert "INC-0001" in text

    run_d(service, D6)  # never a member of anything
    outsider_doc = client.get("/api/runs/T-D6/export").json()
    assert outsider_doc["incident"] is None
    outsider_text = client.get(
        "/api/runs/T-D6/export", params={"format": "text"}
    ).text
    assert "INCIDENT (this run" not in outsider_text


def test_incident_tab_is_in_the_console(client):
    html = client.get("/").text
    assert 'id="tab-incidents"' in html
    assert 'id="view-incidents"' in html
