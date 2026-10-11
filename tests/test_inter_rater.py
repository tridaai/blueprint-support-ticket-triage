"""Inter-rater reliability: do the reviewers agree with each other?

Round 8 measured reviewer-vs-agent. This round measures the QA
program's other leg. The pins:

- the overlap is part of the seeded draw: two named reviewers and
  a count (or fraction) pick runs out of the drawn sample with the
  draw's own RNG stream, so the assignment reproduces exactly and
  rides the sample record; half-stated overlaps are refused by
  name (one reviewer, the same name twice, count AND fraction, a
  count past the draw, a fraction that rounds to nothing);
- assignment binds ONLY on the overlap runs: a third reviewer
  grading one is refused by name as a conflict, while the same
  reviewer grades a non-overlap run freely — the open discipline
  round 8 pinned, with the assignment as a floor on the pair;
- the kappa math is checked from first principles: the expected
  values are recomputed in this file from the authored verdict
  table with exact fractions, and pinned against closed forms;
- the degenerate family is honest: both raters agree on every
  paired run → raw rate 1.0, kappa None, the reason named (no
  variance) — never a fabricated 1.0;
- kappa is withheld below the ten-pair floor while the raw rate
  still shows, and partial pairs (one grade landed) are accounted,
  not silently dropped;
- paired disagreements list both verdicts and both notes side by
  side — the calibration meeting's agenda;
- grading stays inert: records, live samples, and the live
  calibration payload are byte-identical across paired grading.
"""

from __future__ import annotations

import json
from fractions import Fraction

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
FAMILIES = ("category", "priority_drivers", "routing", "verify")


def _run(service, ticket_id, subject, body, name, email):
    service.run_ticket(
        make_ticket(
            ticket_id=ticket_id, subject=subject, body=body,
            sender_name=name, sender_email=email,
        )
    )
    service.record_decision(ticket_id, "approved", by="agent-1")


def _world8(service):
    """The round-8 draw world: eight decided runs over four strata."""
    _run(service, "QA-B1", "Charged twice", BILLING.format(month="March"), "Priya Nair", "priya@example.com")
    _run(service, "QA-B2", "Charged twice", BILLING.format(month="April"), "Tom Becker", "tom@example.com")
    _run(service, "QA-B3", "Double charge", BILLING.format(month="May"), "Ruth Adeyemi", "ruth@example.com")
    _run(service, "QA-B4", "Charged twice", BILLING.format(month="June"), "Omar Haddad", "omar@example.com")
    _run(service, "QA-T1", "App crashes on launch", CRASH, "Sam Ortiz", "sam@example.com")
    _run(service, "QA-T2", "App crashes on launch", CRASH, "Kim Ray", "kim@example.com")
    _run(service, "QA-H1", "How do I export?", HOW_TO, "Jo Lee", "jo@example.com")
    _run(service, "QA-O1", "Service is down", OUTAGE, "Ana Ruiz", "ana@example.com")
    return service


def _world12(service):
    """Twelve decided runs — enough for a full overlap over the floor."""
    senders = [
        ("Priya Nair", "priya@example.com"), ("Tom Becker", "tom@example.com"),
        ("Ruth Adeyemi", "ruth@example.com"), ("Omar Haddad", "omar@example.com"),
        ("Ines Prado", "ines@example.com"), ("Karl Weiss", "karl@example.com"),
    ]
    months = ["March", "April", "May", "June", "July", "August"]
    for i, ((name, email), month) in enumerate(zip(senders, months), start=1):
        _run(service, f"IRR-{i}", "Charged twice", BILLING.format(month=month), name, email)
    _run(service, "IRR-7", "App crashes on launch", CRASH, "Sam Ortiz", "sam@example.com")
    _run(service, "IRR-8", "App crashes on launch", CRASH, "Kim Ray", "kim@example.com")
    _run(service, "IRR-9", "App crashes on launch", CRASH, "Nadia Fox", "nadia@example.com")
    _run(service, "IRR-10", "How do I export?", HOW_TO, "Jo Lee", "jo@example.com")
    _run(service, "IRR-11", "How do I export?", "Hi, how do I export my invoices to PDF? Thanks.", "Ben Osei", "ben@example.com")
    _run(service, "IRR-12", "Service is down", OUTAGE, "Ana Ruiz", "ana@example.com")
    return service


@pytest.fixture()
def service():
    return _world8(fresh_service())


@pytest.fixture()
def service12():
    return _world12(fresh_service())


@pytest.fixture()
def client(service):
    api.set_service(service)
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_service(None)


def _grade(service, sample_id, ticket_id, reviewer, disagree=(), overall="agree", note=""):
    families = {
        family: ("disagree" if family in disagree else "agree") for family in FAMILIES
    }
    return service.record_review_verdict(
        sample_id, ticket_id=ticket_id, reviewer=reviewer,
        families=families, overall=overall, note=note,
    )


# -- the overlap draw ---------------------------------------------------------


def test_overlap_draw_is_part_of_the_seeded_sample(service):
    plain = service.create_review_sample(size=8, seed=3, created_by="qa-lead")
    assert plain.overlap == {}
    sample = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=3,
    )
    # The overlap never changes the draw itself.
    assert sample.run_ids == plain.run_ids
    # The exact assignment for this world + seed (pinned): the draw's
    # own stream picks QA-B1, QA-B3, QA-B2 out of the eight.
    assert sample.overlap == {
        "reviewers": ["rev-a", "rev-b"],
        "requested": {"count": 3},
        "count": 3,
        "run_ids": ["QA-B1", "QA-B3", "QA-B2"],
    }
    assert "rev-a and rev-b double-grade 3" in sample.note
    # A redraw with the same inputs reproduces the assignment exactly.
    redraw = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=3,
    )
    assert redraw.overlap == sample.overlap
    # The fraction form: half of eight is four runs, same stream.
    by_fraction = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_fraction=0.5,
    )
    assert by_fraction.overlap["requested"] == {"fraction": 0.5}
    assert by_fraction.overlap["count"] == 4
    assert by_fraction.overlap["run_ids"] == ["QA-B1", "QA-B3", "QA-B2", "QA-T1"]


def test_overlap_request_refusals(service):
    def draw(**kwargs):
        return service.create_review_sample(size=8, seed=3, created_by="qa-lead", **kwargs)

    with pytest.raises(reviews.ReviewError, match="names its two reviewers"):
        draw(overlap_fraction=0.5)
    with pytest.raises(reviews.ReviewError, match="exactly two reviewers"):
        draw(overlap_reviewers=["rev-a"], overlap_count=2)
    with pytest.raises(reviews.ReviewError, match="exactly two reviewers"):
        draw(overlap_reviewers=["rev-a", "rev-b", "rev-c"], overlap_count=2)
    with pytest.raises(reviews.ReviewError, match="must be named"):
        draw(overlap_reviewers=["rev-a", " "], overlap_count=2)
    with pytest.raises(reviews.ReviewError, match="two different people"):
        draw(overlap_reviewers=["rev-a", "rev-a"], overlap_count=2)
    with pytest.raises(reviews.ReviewError, match="exactly one of"):
        draw(overlap_reviewers=["rev-a", "rev-b"])
    with pytest.raises(reviews.ReviewError, match="exactly one of"):
        draw(overlap_reviewers=["rev-a", "rev-b"], overlap_count=2, overlap_fraction=0.5)
    with pytest.raises(reviews.ReviewError, match="positive integer"):
        draw(overlap_reviewers=["rev-a", "rev-b"], overlap_count=0)
    with pytest.raises(reviews.ReviewError, match="exceeds the 8 runs"):
        draw(overlap_reviewers=["rev-a", "rev-b"], overlap_count=9)
    with pytest.raises(reviews.ReviewError, match=r"\(0, 1\]"):
        draw(overlap_reviewers=["rev-a", "rev-b"], overlap_fraction=0)
    with pytest.raises(reviews.ReviewError, match=r"\(0, 1\]"):
        draw(overlap_reviewers=["rev-a", "rev-b"], overlap_fraction=1.5)
    with pytest.raises(reviews.ReviewError, match="rounds to no runs"):
        service.create_review_sample(
            size=2, seed=3, created_by="qa-lead",
            overlap_reviewers=["rev-a", "rev-b"], overlap_fraction=0.1,
        )


# -- assignment enforcement ---------------------------------------------------


def test_assignment_binds_only_on_overlap_runs(service):
    sample = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=3,
    )
    overlap_run = sample.overlap["run_ids"][0]  # QA-B1
    open_run = next(t for t in sample.run_ids if t not in sample.overlap["run_ids"])

    # A third reviewer on an overlap run: refused, and the refusal
    # names the pair they are not part of.
    with pytest.raises(reviews.ReviewConflict, match="not assigned") as excinfo:
        _grade(service, sample.sample_id, overlap_run, "rev-c")
    assert "rev-a and rev-b" in str(excinfo.value)

    # The pair grades it, each exactly once — the standing rule
    # still applies inside the assignment.
    _grade(service, sample.sample_id, overlap_run, "rev-a")
    _grade(service, sample.sample_id, overlap_run, "rev-b")
    with pytest.raises(reviews.ReviewConflict, match="already graded"):
        _grade(service, sample.sample_id, overlap_run, "rev-a")

    # The line: off the overlap set, grading stays open — the same
    # third reviewer, and an assigned reviewer, both grade freely.
    _grade(service, sample.sample_id, open_run, "rev-c")
    _grade(service, sample.sample_id, open_run, "rev-a")


# -- the kappa math -------------------------------------------------------------

# The authored verdict table for the twelve-run world. Each entry
# names the runs (1-based) where that reviewer disagrees.
A_DISAGREE = {
    "category": {11, 12},
    "priority_drivers": {7, 8, 9, 10, 11, 12},
    "routing": {3, 6, 9, 12},
    "verify": set(),
}
B_DISAGREE = {
    "category": {10, 11, 12},
    "priority_drivers": {6, 8, 9, 10, 11, 12},
    "routing": {3, 6},
    "verify": set(),
}
A_OVERALL_DISAGREE = {10, 11, 12}
B_OVERALL_DISAGREE = {11, 12}

NOTES = {
    ("IRR-6", "rev-b"): "urgency is higher than the drivers read it",
    ("IRR-7", "rev-a"): "churn risk underweighted for a locked-out team",
    ("IRR-9", "rev-a"): "an outage pattern belongs with incident response",
    ("IRR-10", "rev-a"): "destination right, the reading was not",
    ("IRR-10", "rev-b"): "reads as account access, not billing",
    ("IRR-12", "rev-a"): "the outage queue, not general support",
}


def _words(disagree_sets, family):
    return [
        "disagree" if i in disagree_sets[family] else "agree" for i in range(1, 13)
    ]


def _expected_row(a_words, b_words):
    """The pair stats, recomputed from first principles — exact
    fractions over the two raters' marginals, written here from the
    definition of Cohen's kappa, not read off the implementation."""
    n = len(a_words)
    both_agree = sum(1 for a, b in zip(a_words, b_words) if a == b == "agree")
    both_disagree = sum(1 for a, b in zip(a_words, b_words) if a == b == "disagree")
    observed = Fraction(both_agree + both_disagree, n)
    a_agree = sum(1 for a in a_words if a == "agree")
    b_agree = sum(1 for b in b_words if b == "agree")
    expected = Fraction(
        a_agree * b_agree + (n - a_agree) * (n - b_agree), n * n
    )
    kappa = None if expected == 1 else round(float((observed - expected) / (1 - expected)), 4)
    return {
        "pairs": n,
        "both_agree": both_agree,
        "both_disagree": both_disagree,
        "split": n - both_agree - both_disagree,
        "agreement_rate": round(float(observed), 4),
        "kappa": kappa,
    }


def _grade_the_twelve(service):
    sample = service.create_review_sample(
        size=12, seed=7, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=12,
    )
    assert sample.overlap["run_ids"] == sample.run_ids  # the whole draw is paired
    for i in range(1, 13):
        ticket_id = f"IRR-{i}"
        _grade(
            service, sample.sample_id, ticket_id, "rev-a",
            disagree={f for f in FAMILIES if i in A_DISAGREE[f]},
            overall="disagree" if i in A_OVERALL_DISAGREE else "agree",
            note=NOTES.get((ticket_id, "rev-a"), ""),
        )
        _grade(
            service, sample.sample_id, ticket_id, "rev-b",
            disagree={f for f in FAMILIES if i in B_DISAGREE[f]},
            overall="disagree" if i in B_OVERALL_DISAGREE else "agree",
            note=NOTES.get((ticket_id, "rev-b"), ""),
        )
    return sample


def test_kappa_math_matches_first_principles(service12):
    sample = _grade_the_twelve(service12)
    block = service12.review_report([sample.sample_id])["inter_rater"]
    assert block["reviewers"] == ["rev-a", "rev-b"]
    assert block["pairs_assigned"] == 12
    assert block["pairs_complete"] == 12
    assert block["pairs_partial"] == 0
    assert block["min_pairs_for_kappa"] == reviews.KAPPA_MIN_PAIRS

    for family in FAMILIES:
        expected = _expected_row(_words(A_DISAGREE, family), _words(B_DISAGREE, family))
        row = block["families"][family]
        for key, wanted in expected.items():
            assert row[key] == wanted, f"{family}.{key}: {row[key]} != {wanted}"
    overall = _expected_row(
        ["disagree" if i in A_OVERALL_DISAGREE else "agree" for i in range(1, 13)],
        ["disagree" if i in B_OVERALL_DISAGREE else "agree" for i in range(1, 13)],
    )
    for key, wanted in overall.items():
        assert block["overall"][key] == wanted, f"overall.{key}"

    # The closed forms, so the table above cannot drift silently:
    # category κ = (11/12 − 2/3)/(1/3) = 0.75; drivers κ = 2/3;
    # routing κ = 4/7; overall κ = 0.75.
    assert block["families"]["category"]["kappa"] == 0.75
    assert block["families"]["priority_drivers"]["kappa"] == 0.6667
    assert block["families"]["routing"]["kappa"] == 0.5714
    assert block["overall"]["kappa"] == 0.75

    # The degenerate family: both raters agreed on verify for all
    # twelve pairs — the raw rate is the whole story, kappa is
    # undefined and says why instead of printing a flattering 1.0.
    verify = block["families"]["verify"]
    assert verify["agreement_rate"] == 1.0
    assert verify["kappa"] is None
    assert "no variance" in verify["kappa_note"]
    assert "nothing to correct for" in verify["kappa_note"]


def test_paired_disagreements_carry_both_sides(service12):
    sample = _grade_the_twelve(service12)
    block = service12.review_report([sample.sample_id])["inter_rater"]
    rows = {(r["ticket_id"], r["family"]): r for r in block["disagreements"]}
    # From the table: category splits on IRR-10; drivers on IRR-6
    # and IRR-7; routing on IRR-9 and IRR-12; overall on IRR-10.
    assert set(rows) == {
        ("IRR-10", "category"), ("IRR-6", "priority_drivers"),
        ("IRR-7", "priority_drivers"), ("IRR-9", "routing"),
        ("IRR-12", "routing"), ("IRR-10", "overall"),
    }
    row = rows[("IRR-9", "routing")]
    assert row["reviewer_a"] == "rev-a" and row["verdict_a"] == "disagree"
    assert row["note_a"] == "an outage pattern belongs with incident response"
    assert row["reviewer_b"] == "rev-b" and row["verdict_b"] == "agree"
    assert row["note_b"] == ""


def test_kappa_withheld_below_the_pair_floor(service):
    sample = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=4,
    )
    assert sample.overlap["run_ids"] == ["QA-B1", "QA-B3", "QA-B2", "QA-T1"]
    # Both grade three of the four (with one routing split); the
    # fourth gets only rev-a — a partial pair, accounted, not paired.
    for ticket_id in ("QA-B1", "QA-B3"):
        _grade(service, sample.sample_id, ticket_id, "rev-a")
        _grade(service, sample.sample_id, ticket_id, "rev-b")
    _grade(service, sample.sample_id, "QA-B2", "rev-a")
    _grade(service, sample.sample_id, "QA-B2", "rev-b", disagree={"routing"},
           note="billing queue, not general")
    _grade(service, sample.sample_id, "QA-T1", "rev-a")

    block = service.review_report([sample.sample_id])["inter_rater"]
    assert block["pairs_assigned"] == 4
    assert block["pairs_complete"] == 3
    assert block["pairs_partial"] == 1
    for family in FAMILIES:
        row = block["families"][family]
        assert row["pairs"] == 3
        assert row["kappa"] is None
        assert "withheld" in row["kappa_note"]
    # The raw rates still show: category unanimous, routing 2 of 3.
    assert block["families"]["category"]["agreement_rate"] == 1.0
    assert block["families"]["routing"]["agreement_rate"] == 0.6667
    assert block["families"]["routing"]["split"] == 1


def test_inter_rater_block_without_overlap_is_an_honest_empty(service):
    sample = service.create_review_sample(size=4, seed=5, created_by="qa-lead")
    _grade(service, sample.sample_id, sample.run_ids[0], "rev-1")
    block = service.review_report([sample.sample_id])["inter_rater"]
    assert block["pairs_assigned"] == 0
    assert block["pairs_complete"] == 0
    assert block["reviewers"] == []
    assert block["families"]["category"]["pairs"] == 0
    assert block["families"]["category"]["agreement_rate"] is None
    assert block["families"]["category"]["kappa"] is None
    assert "no overlap sets in scope" in block["note"]
    assert block["disagreements"] == []


# -- inertness ------------------------------------------------------------------


def test_inter_rater_grading_is_inert(service):
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

    sample = service.create_review_sample(
        size=8, seed=3, created_by="qa-lead",
        overlap_reviewers=["rev-a", "rev-b"], overlap_count=4,
    )
    for ticket_id in sample.overlap["run_ids"]:
        _grade(service, sample.sample_id, ticket_id, "rev-a", disagree={"routing"})
        _grade(service, sample.sample_id, ticket_id, "rev-b")

    records_after = {r.ticket_id: r.model_dump_json() for r in service.store.all_records()}
    assert records_after == records_before
    assert json.dumps(live_samples(service.store.all_records()), sort_keys=True, default=str) == samples_before
    assert json.dumps(live_calibration(service.store.all_records()), sort_keys=True, default=str) == calibration_before


# -- API round trip ---------------------------------------------------------------


def test_inter_rater_api_round_trip(client):
    created = client.post(
        "/api/reviews/samples",
        json={
            "size": 8, "seed": 3, "created_by": "qa-lead",
            "overlap_reviewers": ["rev-a", "rev-b"], "overlap_count": 3,
        },
    )
    assert created.status_code == 201
    sample = created.json()
    assert sample["overlap"]["run_ids"] == ["QA-B1", "QA-B3", "QA-B2"]
    overlap_run = sample["overlap"]["run_ids"][0]

    refused = client.post(
        f"/api/reviews/samples/{sample['sample_id']}/verdicts",
        json={"ticket_id": overlap_run, "reviewer": "rev-c",
              "families": dict(ALL_AGREE), "overall": "agree"},
    )
    assert refused.status_code == 409
    assert "not assigned" in refused.json()["detail"]

    accepted = client.post(
        f"/api/reviews/samples/{sample['sample_id']}/verdicts",
        json={"ticket_id": overlap_run, "reviewer": "rev-a",
              "families": dict(ALL_AGREE), "overall": "agree"},
    )
    assert accepted.status_code == 201

    report = client.get("/api/reviews/report", params={"samples": sample["sample_id"]})
    assert report.status_code == 200
    block = report.json()["inter_rater"]
    assert block["pairs_assigned"] == 3
    assert block["pairs_complete"] == 0
    assert block["pairs_partial"] == 1
    assert block["reviewers"] == ["rev-a", "rev-b"]
