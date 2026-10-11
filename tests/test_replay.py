"""Policy impact replay: stored answers, hypothetical constants.

The engineered runs, worked out before they are asserted (default
answers: every score's expected value 5.0, P(abusive) 0.1, category
billing at the stated confidence):

- terms under the live weights: urgency .30·.5, sla .25·.5,
  churn .20·.5, anger .15·.5, abuse .10·.1 → 15 + 12.5 + 10 + 7.5
  + 1 = 46.0 → band medium, queue billing_support (matrix).
- the OUTAGE run adds P(is_outage_report) 0.9: boost +15 → 61.0 →
  band high, and the outage override routes it to
  incident_response. Boost off: 46.0, medium — but the queue
  stands, because the override answers the flag, not the boost.
- the LOWCONF run has category confidence 0.45 < 0.55: the
  low_confidence guardrail fails and review is forced. A threshold
  of 0.40 un-forces it; nothing else about it changes.
- the MIDCONF run has category confidence 0.60: decisive live, but
  a threshold of 0.75 forces it — and the 0.70-confidence runs
  (PLAIN, OUTAGE) flip with it, which the aggregate counts show.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from support_triage import api
from support_triage.decisions import build_decision_set
from support_triage.questions import CATEGORIES
from support_triage.schemas import TicketResult

from helpers import fresh_service, make_ticket, valid_answers


def category_dist(top_p: float) -> dict:
    rest = [c for c in CATEGORIES if c != "billing"]
    return {
        "probabilities": {
            "billing": top_p,
            **{c: (1.0 - top_p) / len(rest) for c in rest},
        }
    }


class ScriptedEngine:
    """Answers by subject marker, so one service can hold runs of
    several engineered shapes."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, by_marker: dict):
        self.by_marker = by_marker

    def decide(self, state, questions, context=""):
        overrides = {}
        for marker, candidate in self.by_marker.items():
            if marker in state:
                overrides = candidate
        return build_decision_set(
            questions,
            valid_answers(**overrides),
            backend="scripted",
            model="scripted-1",
        )


def replay_service() -> object:
    engine = ScriptedEngine(
        {
            "OUTAGE-MARKER": {
                "is_outage_report": {"probabilities": {"true": 0.9, "false": 0.1}}
            },
            "LOWCONF-MARKER": {"category": category_dist(0.45)},
            "MIDCONF-MARKER": {"category": category_dist(0.60)},
        }
    )
    service = fresh_service(engine=engine)
    service.run_ticket(make_ticket(ticket_id="R-PLAIN", subject="Plain billing question"))
    service.run_ticket(make_ticket(ticket_id="R-OUTAGE", subject="OUTAGE-MARKER everything is down"))
    service.run_ticket(make_ticket(ticket_id="R-LOWCONF", subject="LOWCONF-MARKER unsure what this is"))
    service.run_ticket(make_ticket(ticket_id="R-MIDCONF", subject="MIDCONF-MARKER probably billing"))
    return service


def moved_by_id(report: dict) -> dict:
    return {row["ticket_id"]: row for row in report["moved_runs"]}


# ---------------------------------------------------------------------------
# the identity: replaying the live policy moves nothing
# ---------------------------------------------------------------------------


def test_identity_replay_moves_nothing():
    service = replay_service()
    report = service.policy_replay()
    assert report["runs_replayed"] == 4
    assert report["runs_skipped"] == 0
    assert report["moved"] == 0
    assert report["unchanged"] == 4
    assert report["moved_runs"] == []
    assert report["by_queue_pair"] == {}
    assert report["score_changed"] == 0


def test_report_carries_the_policy_in_force():
    service = replay_service()
    report = service.policy_replay(category_min_confidence=0.7, outage_boost=False)
    assert report["hypothetical"]["category_min_confidence"] == 0.7
    assert report["hypothetical"]["outage_boost_enabled"] is False
    assert report["hypothetical"]["outage_boost_points"] == 0.0
    assert report["current_policy"]["category_min_confidence"] == 0.55
    assert report["current_policy"]["outage_boost_points"] == 15.0
    assert "no record was changed" in report["note"]


# ---------------------------------------------------------------------------
# the three knobs, one at a time
# ---------------------------------------------------------------------------


def test_turning_the_outage_boost_off_moves_only_the_outage_run():
    service = replay_service()
    report = service.policy_replay(outage_boost=False)
    assert report["moved"] == 1
    row = moved_by_id(report)["R-OUTAGE"]
    assert row["changes"] == ["priority_band"]
    assert row["before"]["priority_band"] == "high"
    assert row["before"]["priority_score"] == 61.0
    assert row["after"]["priority_band"] == "medium"
    assert row["after"]["priority_score"] == 46.0
    # the queue stands: the outage override answers the raised flag
    # (P ≥ 0.5), not the boost — the toggle only takes points away
    assert row["before"]["queue"] == row["after"]["queue"] == "incident_response"
    assert report["by_queue_pair"] == {}
    assert report["by_band_pair"] == {"high → medium": 1}
    assert report["by_guardrail_flip"] == {}


def test_lowering_the_threshold_unforces_the_low_confidence_run():
    service = replay_service()
    report = service.policy_replay(category_min_confidence=0.40)
    assert report["moved"] == 1
    row = moved_by_id(report)["R-LOWCONF"]
    assert row["changes"] == ["guardrails"]
    assert row["before"]["guardrails_passed"] is False
    assert row["before"]["requires_human_review"] is True
    assert row["after"]["guardrails_passed"] is True
    assert row["after"]["requires_human_review"] is False
    # un-forcing is not a reroute: queue and band stand
    assert row["before"]["queue"] == row["after"]["queue"] == "billing_support"
    assert row["before"]["priority_band"] == row["after"]["priority_band"]
    assert report["by_guardrail_flip"] == {
        "guardrails failed → passed": 1,
        "human review forced → not forced": 1,
    }


def test_raising_the_threshold_forces_the_mid_confidence_run():
    service = replay_service()
    report = service.policy_replay(category_min_confidence=0.75)
    row = moved_by_id(report)["R-MIDCONF"]
    assert row["changes"] == ["guardrails"]
    assert row["before"]["guardrails_passed"] is True
    assert row["after"]["guardrails_passed"] is False
    assert row["after"]["requires_human_review"] is True
    # the 0.70-confidence runs (PLAIN, OUTAGE) sit under 0.75 too:
    # three runs flip, and the aggregate says so
    assert report["by_guardrail_flip"] == {
        "guardrails passed → failed": 3,
        "human review not forced → forced": 3,
    }


def test_weights_can_move_a_band():
    service = replay_service()
    weights = {
        "urgency": 0.1,
        "sla_breach_risk": 0.1,
        "churn_risk": 0.1,
        "anger": 0.1,
        "abuse": 0.6,
    }
    report = service.policy_replay(weights=weights)
    row = moved_by_id(report)["R-PLAIN"]
    # 5 + 5 + 5 + 5 + 6 = 26.0 → low
    assert row["after"]["priority_score"] == 26.0
    assert row["changes"] == ["priority_band"]
    assert report["by_band_pair"]["medium → low"] >= 1


def test_a_score_change_inside_the_band_is_counted_not_moved():
    service = replay_service()
    weights = {
        "urgency": 0.20,
        "sla_breach_risk": 0.25,
        "churn_risk": 0.20,
        "anger": 0.15,
        "abuse": 0.20,
    }
    report = service.policy_replay(weights=weights)
    # R-PLAIN: 10 + 12.5 + 10 + 7.5 + 2 = 42.0 — different score,
    # same band, same queue, same guardrails: not a move.
    assert "R-PLAIN" not in moved_by_id(report)
    assert report["score_changed"] >= 1


# ---------------------------------------------------------------------------
# the invariants: no backend call, no write, honest skips
# ---------------------------------------------------------------------------


def test_replay_never_calls_a_backend():
    service = replay_service()

    class ExplodingEngine:
        name = "exploding"

        def decide(self, state, questions, context=""):  # noqa: ARG002
            raise AssertionError("replay must never call a decision backend")

    service.engine = ExplodingEngine()
    report = service.policy_replay(outage_boost=False)
    assert report["moved"] == 1  # the replay ran on stored answers alone


def test_a_record_without_answers_is_skipped_and_named():
    service = replay_service()
    service.store.save(TicketResult(ticket_id="R-BARE"))
    report = service.policy_replay(outage_boost=False)
    assert report["runs_replayed"] == 4
    assert report["runs_skipped"] == 1
    assert report["skipped"][0]["ticket_id"] == "R-BARE"
    assert report["moved"] == 1


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hypothetical",
    [
        {"category_min_confidence": 1.5},
        {"category_min_confidence": -0.1},
        {"weights": {"urgency": 1.0}},  # missing terms
        {"weights": {"urgency": 0.3, "sla_breach_risk": 0.25, "churn_risk": 0.2,
                      "anger": 0.15, "abuse": 0.1, "surprise": 0.0}},  # extra term
        {"weights": {"urgency": 0.3, "sla_breach_risk": 0.25, "churn_risk": 0.2,
                      "anger": 0.15, "abuse": 0.05}},  # sums to 0.95
        {"weights": {"urgency": -0.3, "sla_breach_risk": 0.55, "churn_risk": 0.3,
                      "anger": 0.25, "abuse": 0.2}},  # negative term
    ],
)
def test_invalid_hypotheticals_raise(hypothetical):
    service = replay_service()
    with pytest.raises(ValueError):
        service.policy_replay(**hypothetical)


# ---------------------------------------------------------------------------
# the endpoint — and the purity pin
# ---------------------------------------------------------------------------


@pytest.fixture()
def replay_client():
    service = replay_service()
    api.set_service(service)
    with TestClient(api.app) as client:
        yield client, service
    api.set_service(None)


def test_replay_endpoint_reports_moves(replay_client):
    client, _service = replay_client
    response = client.post("/api/policy/replay", json={"outage_boost": False})
    assert response.status_code == 200
    body = response.json()
    assert body["runs_replayed"] == 4
    assert body["moved"] == 1
    assert body["moved_runs"][0]["ticket_id"] == "R-OUTAGE"
    assert body["hypothetical"]["outage_boost_enabled"] is False


def test_replay_endpoint_rejects_a_bad_hypothetical(replay_client):
    client, _service = replay_client
    response = client.post(
        "/api/policy/replay", json={"weights": {"urgency": 0.9}}
    )
    assert response.status_code == 422
    assert "weights" in response.json()["detail"]


def test_replay_leaves_every_record_byte_identical(replay_client):
    client, service = replay_client
    before = {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }
    # a hypothetical that moves runs on all three dimensions at once
    response = client.post(
        "/api/policy/replay",
        json={"outage_boost": False, "category_min_confidence": 0.75},
    )
    assert response.status_code == 200
    assert response.json()["moved"] >= 2  # the replay really did work
    after = {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }
    assert after == before
