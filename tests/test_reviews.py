"""QA review: humans grade the triage, on a sample that can't be gamed.

The pins:

- the population is DECIDED runs (the store's own word: triaged /
  rejected) — an awaiting run never draws; a confirmed duplicate
  never draws and is named in the sample's accounting;
- the draw is stratified by (queue, priority band) with
  proportional largest-remainder allocation, seeded, and
  reproducible: the same seeded world redraws the same run ids;
- previously-reviewed runs sit out by default and return when the
  exclusion is switched off — accounted either way;
- a verdict grades every family (the port's shape: category,
  priority_drivers, routing, verify) plus an overall; partial or
  unknown families are refused, a run the sample never drew is
  refused, and a second verdict from the same reviewer is a 409 —
  another reviewer may still grade the run;
- the agreement report's math: per-family rates, the per-backend
  split, the disagreement list — and rates withheld, counts shown,
  below the verdict floor;
- grading is inert: the run's record, the live calibration
  samples, and the live calibration payload are byte-identical
  across verdict recording. The report reads; it tunes nothing.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from support_triage import api, reviews
from support_triage.livecal import live_calibration, live_samples
from support_triage.schemas import AgentOutcome

from helpers import fresh_service, make_ticket

BILLING = (
    "I was charged twice for the {month} subscription — two identical "
    "payments of $49 left my card. Please refund one of them."
)
CRASH = "The mobile app crashes every time I open it since the update. I cannot work."
HOW_TO = "Hi, how do I export my reports to CSV? Thanks."
OUTAGE = (
    "Your service is completely down for our whole team, nobody can "
    "log in. This is an outage."
)

ALL_AGREE = {
    "category": "agree",
    "priority_drivers": "agree",
    "routing": "agree",
    "verify": "agree",
}


def _run(service, ticket_id, subject, body, name, email, decide=True):
    result = service.run_ticket(
        make_ticket(
            ticket_id=ticket_id, subject=subject, body=body,
            sender_name=name, sender_email=email,
        )
    ).result
    if decide:
        service.record_decision(ticket_id, "approved", by="agent-1")
    return result


def _world(service):
    """Eight decided runs over four strata (4/2/1/1) — the draw world.

    billing_support/low ×4, technical_support/medium ×2,
    general_support/low ×1, incident_response/medium ×1.
    """
    _run(service, "QA-B1", "Charged twice", BILLING.format(month="March"), "Priya Nair", "priya@example.com")
    _run(service, "QA-B2", "Charged twice", BILLING.format(month="April"), "Tom Becker", "tom@example.com")
    _run(service, "QA-B3", "Double charge", BILLING.format(month="May"), "Ruth Adeyemi", "ruth@example.com")
    _run(service, "QA-B4", "Charged twice", BILLING.format(month="June"), "Omar Haddad", "omar@example.com")
    _run(service, "QA-T1", "App crashes on launch", CRASH, "Sam Ortiz", "sam@example.com")
    _run(service, "QA-T2", "App crashes on launch", CRASH, "Kim Ray", "kim@example.com")
    _run(service, "QA-H1", "How do I export?", HOW_TO, "Jo Lee", "jo@example.com")
    _run(service, "QA-O1", "Service is down", OUTAGE, "Ana Ruiz", "ana@example.com")
    return service


@pytest.fixture()
def service():
    return _world(fresh_service())


@pytest.fixture()
def client(service):
    api.set_service(service)
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


# -- the draw ---------------------------------------------------------------


def test_draw_is_stratified_seeded_and_reproducible(service):
    sample = service.create_review_sample(size=5, seed=11, created_by="qa-lead")
    # The exact draw for this world + seed (pinned, not approximate):
    # proportional allocation 2/1/1/1 over the 4/2/1/1 strata.
    assert sample.run_ids == ["QA-B1", "QA-B2", "QA-H1", "QA-O1", "QA-T1"]
    strata = {(row["queue"], row["priority_band"]): row for row in sample.strata}
    assert strata[("billing_support", "low")]["eligible"] == 4
    assert strata[("billing_support", "low")]["sampled"] == 2
    assert strata[("technical_support", "medium")]["eligible"] == 2
    assert strata[("technical_support", "medium")]["sampled"] == 1
    assert sample.population == {
        "decided_total": 8,
        "excluded_duplicates": [],
        "excluded_previously_reviewed": [],
        "eligible": 8,
    }
    # Same store, same seed, same size → the same sample.
    redraw = service.create_review_sample(size=5, seed=11, created_by="qa-lead")
    assert redraw.run_ids == sample.run_ids
    # A different seed is a different shuffle of the big stratum.
    other = service.create_review_sample(size=5, seed=99, created_by="qa-lead")
    assert sorted(other.run_ids) != sorted(sample.run_ids) or other.run_ids != sample.run_ids


def test_draw_population_excludes_awaiting_and_confirmed_duplicates(service):
    _run(service, "QA-W1", "Charged twice", BILLING.format(month="July"),
         "Priya Nair", "priya@example.com", decide=False)  # stays awaiting
    _run(service, "QA-D1", "Charged twice", BILLING.format(month="March"),
         "Priya Nair", "priya@example.com")
    service.mark_duplicate("QA-D1", "QA-B1", actor="op-1", reason="same customer re-sent it")
    sample = service.create_review_sample(size=50, seed=1, created_by="qa-lead")
    assert "QA-W1" not in sample.run_ids  # not decided — not in the population at all
    assert "QA-D1" not in sample.run_ids  # confirmed duplicate — excluded, named
    assert sample.population["decided_total"] == 9
    assert sample.population["excluded_duplicates"] == ["QA-D1"]
    assert sample.population["eligible"] == 8
    assert len(sample.run_ids) == 8
    assert "ran past the eligible population" in sample.note


def test_draw_refusals(service):
    with pytest.raises(reviews.ReviewError, match="created_by is required"):
        service.create_review_sample(size=3, seed=1, created_by=" ")
    with pytest.raises(reviews.ReviewError, match="positive integer"):
        service.create_review_sample(size=0, seed=1, created_by="qa-lead")
    empty = fresh_service()
    with pytest.raises(reviews.ReviewError, match="no eligible runs"):
        empty.create_review_sample(size=3, seed=1, created_by="qa-lead")


def test_previously_reviewed_exclusion_and_its_off_switch(service):
    first = service.create_review_sample(size=5, seed=11, created_by="qa-lead")
    graded = first.run_ids[0]
    service.record_review_verdict(
        first.sample_id, ticket_id=graded, reviewer="rev-1",
        families=dict(ALL_AGREE), overall="agree",
    )
    second = service.create_review_sample(size=5, seed=11, created_by="qa-lead")
    assert graded not in second.run_ids
    assert second.population["excluded_previously_reviewed"] == [graded]
    # The flag off: the graded run is eligible again, accounted as such.
    third = service.create_review_sample(
        size=5, seed=11, created_by="qa-lead", exclude_previously_reviewed=False
    )
    assert third.population["excluded_previously_reviewed"] == []
    assert third.population["eligible"] == 8
    assert graded in third.run_ids


# -- verdicts ---------------------------------------------------------------


def _sample_with_verdicts(service, verdict_specs):
    sample = service.create_review_sample(size=8, seed=3, created_by="qa-lead")
    assert len(sample.run_ids) == 8
    for ticket_id, reviewer, families, overall, note in verdict_specs:
        service.record_review_verdict(
            sample.sample_id, ticket_id=ticket_id, reviewer=reviewer,
            families=families, overall=overall, note=note,
        )
    return sample


def test_verdict_refusals(service):
    sample = service.create_review_sample(size=4, seed=5, created_by="qa-lead")
    target = sample.run_ids[0]
    with pytest.raises(KeyError, match="unknown review sample"):
        service.record_review_verdict(
            "QAS-9999", ticket_id=target, reviewer="rev-1",
            families=dict(ALL_AGREE), overall="agree",
        )
    with pytest.raises(reviews.ReviewError, match="not in sample"):
        service.record_review_verdict(
            sample.sample_id, ticket_id="QA-B4" if "QA-B4" not in sample.run_ids else "QA-T2",
            reviewer="rev-1", families=dict(ALL_AGREE), overall="agree",
        )
    with pytest.raises(reviews.ReviewError, match="grades every family"):
        service.record_review_verdict(
            sample.sample_id, ticket_id=target, reviewer="rev-1",
            families={"category": "agree"}, overall="agree",
        )
    with pytest.raises(reviews.ReviewError, match="unknown review families"):
        service.record_review_verdict(
            sample.sample_id, ticket_id=target, reviewer="rev-1",
            families={**ALL_AGREE, "tone": "agree"}, overall="agree",
        )
    with pytest.raises(reviews.ReviewError, match="overall verdict must be"):
        service.record_review_verdict(
            sample.sample_id, ticket_id=target, reviewer="rev-1",
            families=dict(ALL_AGREE), overall="mostly fine",
        )
    with pytest.raises(reviews.ReviewError, match="reviewer is required"):
        service.record_review_verdict(
            sample.sample_id, ticket_id=target, reviewer=" ",
            families=dict(ALL_AGREE), overall="agree",
        )
    service.record_review_verdict(
        sample.sample_id, ticket_id=target, reviewer="rev-1",
        families=dict(ALL_AGREE), overall="agree",
    )
    with pytest.raises(reviews.ReviewConflict, match="already graded"):
        service.record_review_verdict(
            sample.sample_id, ticket_id=target, reviewer="rev-1",
            families=dict(ALL_AGREE), overall="disagree",
        )
    # Another reviewer may still grade the same run.
    service.record_review_verdict(
        sample.sample_id, ticket_id=target, reviewer="rev-2",
        families=dict(ALL_AGREE), overall="agree",
    )


# -- the report -------------------------------------------------------------

SIX_VERDICTS = [
    # (ticket, reviewer, families, overall, note) — authored so the
    # family math is exact: category 5/6, drivers 4/6, routing 3/6,
    # verify 6/6, overall 4/6.
    ("QA-B1", "rev-1", dict(ALL_AGREE), "agree", ""),
    ("QA-B2", "rev-1", {**ALL_AGREE, "routing": "disagree"}, "disagree", "billing queue, not general"),
    ("QA-B3", "rev-1", {**ALL_AGREE, "priority_drivers": "disagree", "routing": "disagree"}, "disagree", "urgency read too low"),
    ("QA-B4", "rev-2", dict(ALL_AGREE), "agree", ""),
    ("QA-T1", "rev-2", {**ALL_AGREE, "category": "disagree", "priority_drivers": "disagree"}, "agree", "answers off, destination right"),
    ("QA-H1", "rev-2", {**ALL_AGREE, "routing": "disagree"}, "agree", "should have stayed in general"),
]


def test_report_math_and_disagreement_list(service):
    sample = _sample_with_verdicts(service, SIX_VERDICTS)
    report = service.review_report([sample.sample_id])
    assert report["verdicts"] == 6
    assert report["verdicted_runs"] == 6
    assert report["overall_agree"] == 4
    assert report["overall_rate"] == pytest.approx(4 / 6, abs=1e-4)
    assert report["families"]["category"] == {
        "verdicts": 6, "agree": 5, "disagree": 1, "rate": pytest.approx(5 / 6, abs=1e-4),
    }
    assert report["families"]["priority_drivers"]["agree"] == 4
    assert report["families"]["routing"]["agree"] == 3
    assert report["families"]["verify"]["agree"] == 6
    assert report["families"]["verify"]["rate"] == 1.0
    # Per-backend split: every run in this world ran on the offline backend.
    assert set(report["by_backend"]) == {"offline"}
    assert report["by_backend"]["offline"]["verdicts"] == 6
    assert report["by_backend"]["offline"]["overall_rate"] == pytest.approx(4 / 6, abs=1e-4)
    rows = {(r["ticket_id"], r["family"]) for r in report["disagreements"]}
    assert rows == {
        ("QA-B2", "routing"), ("QA-B2", "overall"),
        ("QA-B3", "priority_drivers"), ("QA-B3", "routing"), ("QA-B3", "overall"),
        ("QA-T1", "category"), ("QA-T1", "priority_drivers"),
        ("QA-H1", "routing"),
    }
    notes = {r["ticket_id"]: r["note"] for r in report["disagreements"] if r["family"] == "routing"}
    assert notes["QA-B2"] == "billing queue, not general"


def test_report_withholds_rates_below_the_floor(service):
    sample = service.create_review_sample(size=4, seed=5, created_by="qa-lead")
    for ticket_id in sample.run_ids[:2]:
        service.record_review_verdict(
            sample.sample_id, ticket_id=ticket_id, reviewer="rev-1",
            families=dict(ALL_AGREE), overall="agree",
        )
    report = service.review_report([sample.sample_id])
    assert report["verdicts"] == 2
    assert report["overall_agree"] == 2
    assert report["overall_rate"] is None
    assert all(row["rate"] is None for row in report["families"].values())
    assert "withheld" in report["note"]
    # Per-backend tallies withhold on the same floor.
    assert report["by_backend"]["offline"]["overall_rate"] is None
    assert report["by_backend"]["offline"]["verdicts"] == 2


def test_report_over_no_samples_is_an_honest_empty(service):
    report = service.review_report()
    assert report["verdicts"] == 0
    assert report["overall_rate"] is None
    assert "no verdicts recorded yet" in report["note"]
    with pytest.raises(KeyError, match="unknown review sample"):
        service.review_report(["QAS-9999"])


# -- grading is inert ---------------------------------------------------------


def test_verdicts_leave_the_record_and_calibration_byte_identical(service):
    # Give two runs outcomes so the calibration samples are non-empty
    # and the byte-identical pin is a real comparison, not a vacuity.
    for ticket_id, category in (("QA-B1", "billing"), ("QA-T1", "technical_issue")):
        service.record_outcome(
            ticket_id,
            AgentOutcome(final_category=category, final_queue="", resolved=True,
                         recorded_by="agent-1"),
        )
    records_before = {
        r.ticket_id: r.model_dump_json() for r in service.store.all_records()
    }
    samples_before = json.dumps(live_samples(service.store.all_records()), sort_keys=True, default=str)
    calibration_before = json.dumps(live_calibration(service.store.all_records()), sort_keys=True, default=str)

    sample = service.create_review_sample(size=5, seed=11, created_by="qa-lead")
    for ticket_id in sample.run_ids:
        service.record_review_verdict(
            sample.sample_id, ticket_id=ticket_id, reviewer="rev-1",
            families={**ALL_AGREE, "routing": "disagree"}, overall="disagree",
            note="graded",
        )

    records_after = {r.ticket_id: r.model_dump_json() for r in service.store.all_records()}
    assert records_after == records_before
    assert json.dumps(live_samples(service.store.all_records()), sort_keys=True, default=str) == samples_before
    assert json.dumps(live_calibration(service.store.all_records()), sort_keys=True, default=str) == calibration_before


def test_export_carries_the_review_block(service):
    sample = service.create_review_sample(size=8, seed=3, created_by="qa-lead")
    target = sample.run_ids[0]
    before = service.export_run(target)
    assert before["qa_review"] is None
    service.record_review_verdict(
        sample.sample_id, ticket_id=target, reviewer="rev-1",
        families={**ALL_AGREE, "verify": "disagree"}, overall="disagree",
        note="PII was in the body and the answer missed it",
    )
    document = service.export_run(target)
    block = document["qa_review"]
    assert block is not None and len(block["verdicts"]) == 1
    entry = block["verdicts"][0]
    assert entry["sample_id"] == sample.sample_id
    assert entry["families"]["verify"] == "disagree"
    from support_triage.exports import render_text

    text = render_text(document)
    assert "QA REVIEW" in text
    assert "PII was in the body" in text


# -- API round trip -----------------------------------------------------------


def test_review_api_round_trip(client):
    created = client.post(
        "/api/reviews/samples",
        json={"size": 5, "seed": 11, "created_by": "qa-lead"},
    )
    assert created.status_code == 201
    sample = created.json()
    assert sample["run_ids"] == ["QA-B1", "QA-B2", "QA-H1", "QA-O1", "QA-T1"]

    listed = client.get("/api/reviews/samples")
    assert [s["sample_id"] for s in listed.json()["samples"]] == [sample["sample_id"]]

    verdict = client.post(
        f"/api/reviews/samples/{sample['sample_id']}/verdicts",
        json={
            "ticket_id": "QA-B1", "reviewer": "rev-1",
            "families": dict(ALL_AGREE), "overall": "agree", "note": "",
        },
    )
    assert verdict.status_code == 201
    repeat = client.post(
        f"/api/reviews/samples/{sample['sample_id']}/verdicts",
        json={
            "ticket_id": "QA-B1", "reviewer": "rev-1",
            "families": dict(ALL_AGREE), "overall": "agree", "note": "",
        },
    )
    assert repeat.status_code == 409

    detail = client.get(f"/api/reviews/samples/{sample['sample_id']}")
    assert detail.json()["verdicted_runs"] == 1
    report = client.get("/api/reviews/report", params={"samples": sample["sample_id"]})
    assert report.status_code == 200
    assert report.json()["verdicts"] == 1
    assert client.get("/api/reviews/report", params={"samples": "QAS-9999"}).status_code == 404


def test_review_api_refusals(client):
    assert client.post(
        "/api/reviews/samples", json={"size": 3, "seed": 1}
    ).status_code == 422  # no created_by
    assert client.post(
        "/api/reviews/samples", json={"size": 0, "seed": 1, "created_by": "qa"}
    ).status_code == 422  # bad size
    assert client.post(
        "/api/reviews/samples/QAS-9999/verdicts",
        json={"ticket_id": "QA-B1", "reviewer": "rev-1",
              "families": dict(ALL_AGREE), "overall": "agree"},
    ).status_code == 404
