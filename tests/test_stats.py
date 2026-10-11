"""Stats layer: exact aggregates over hand-built records."""

from __future__ import annotations

import pytest

from support_triage.schemas import (
    AgentOutcome,
    CompositionInfo,
    GuardrailInfo,
    QuestionAnswer,
    RunTelemetry,
    TicketResult,
)
from support_triage.stats import overview_stats, performance_stats, percentile

ALL_FLAGS_OFF = {
    "contains_pii": False,
    "is_abusive": False,
    "mentions_legal": False,
    "is_outage_report": False,
    "needs_engineering": False,
}


def make_record(ticket_id="S-1", **overrides) -> TicketResult:
    base = dict(
        ticket_id=ticket_id,
        status="awaiting_agent",
        subject="Something broke",
        sender_name="Sam Sender",
        category="billing",
        product_area="checkout",
        answers=[
            QuestionAnswer(
                key="category", kind="choice", value="billing",
                probabilities={"billing": 0.9, "other": 0.1}, confidence=0.9,
            )
        ],
        composition=CompositionInfo(
            priority_score=60.0, priority_band="high",
            queue="billing_support", flags=dict(ALL_FLAGS_OFF),
        ),
        guardrails=GuardrailInfo(
            passed=True, requires_human_review=False,
            checks=[{"name": "completeness", "passed": True, "detail": "ok"}],
        ),
        telemetry=RunTelemetry(backend="offline", decision_latency_ms=10.0, question_count=11),
    )
    base.update(overrides)
    return TicketResult(**base)


def test_percentile_interpolates():
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 38.5
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.0) == 10.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 1.0) == 40.0


def test_overview_counts_and_rates_are_exact():
    flagged = dict(ALL_FLAGS_OFF, contains_pii=True)
    outage = dict(ALL_FLAGS_OFF, is_outage_report=True)
    failed_checks = [
        {"name": "completeness", "passed": True, "detail": "ok"},
        {"name": "low_confidence", "passed": False, "detail": "category confidence 0.40 < 0.55"},
    ]
    records = [
        make_record(
            "S-1",
            composition=CompositionInfo(
                priority_score=60.0, priority_band="high",
                queue="billing_support", flags=flagged,
            ),
            guardrails=GuardrailInfo(
                passed=False, requires_human_review=True,
                human_review_reasons=["low confidence: category"], checks=failed_checks,
            ),
        ),
        make_record(
            "S-2",
            category="bug_report",
            composition=CompositionInfo(
                priority_score=12.0, priority_band="low",
                queue="technical_support", flags=dict(ALL_FLAGS_OFF),
            ),
            draft={"text": "hi", "template": "how_to"},
            autonomy={"eligible": True, "reasons": ["simple"], "blockers": []},
        ),
        make_record(
            "S-3",
            status="triaged",
            composition=CompositionInfo(
                priority_score=91.0, priority_band="critical",
                queue="billing_support", flags=outage,
            ),
            guardrails=GuardrailInfo(
                passed=False, requires_human_review=True, checks=failed_checks,
            ),
            outcome=AgentOutcome(final_category="billing", resolved=True),
        ),
    ]
    stats = overview_stats(records)
    assert stats["total_runs"] == 3
    assert stats["by_status"] == {"awaiting_agent": 2, "triaged": 1}
    assert stats["by_queue"] == {"billing_support": 2, "technical_support": 1}
    assert stats["by_category"] == {"billing": 2, "bug_report": 1}
    assert stats["priority_bands"] == {"critical": 1, "high": 1, "medium": 0, "low": 1}
    assert stats["safety_flags"]["contains_pii"] == 1
    assert stats["safety_flags"]["is_outage_report"] == 1
    assert stats["safety_flags"]["is_abusive"] == 0
    assert stats["guardrail_failures"] == {"low_confidence": 2}
    assert stats["rates"]["low_confidence_forcing"] == {"count": 2, "rate": 0.6667}
    assert stats["rates"]["template_draft"] == {"count": 1, "rate": 0.3333}
    assert stats["rates"]["autonomy_recommended"] == {"count": 1, "rate": 0.3333}
    assert stats["outcomes_recorded"] == 1


def test_overview_of_nothing_is_zeroed_not_broken():
    stats = overview_stats([])
    assert stats["total_runs"] == 0
    assert stats["rates"]["low_confidence_forcing"] == {"count": 0, "rate": 0.0}
    assert stats["priority_bands"] == {"critical": 0, "high": 0, "medium": 0, "low": 0}


def test_performance_percentiles_tokens_cost_and_errors():
    records = [
        make_record(
            f"P-{i}",
            telemetry=RunTelemetry(
                backend="llm", model="m", decision_latency_ms=latency,
                question_count=11, input_tokens=100, output_tokens=50,
                estimated_cost_usd=0.01,
            ),
        )
        for i, latency in enumerate([10.0, 20.0, 30.0, 40.0])
    ]
    records.append(make_record("P-x", telemetry=None))
    errors = [
        {"backend": "llm", "error": "boom", "ticket_id": "P-e1"},
        {"backend": "llm", "error": "boom again", "ticket_id": "P-e2"},
        {"backend": "jev", "error": "no key", "ticket_id": "P-e3"},
    ]
    stats = performance_stats(records, errors)
    assert stats["total_runs"] == 5
    assert stats["total_errors"] == 3
    llm = stats["by_backend"]["llm"]
    assert llm["runs"] == 4 and llm["decision_calls"] == 4 and llm["questions"] == 44
    assert llm["decision_latency_ms"] == {"count": 4, "mean": 25.0, "p50": 25.0, "p95": 38.5}
    assert llm["input_tokens"] == 400 and llm["output_tokens"] == 200
    assert llm["estimated_cost_usd"] == pytest.approx(0.04)
    assert llm["errors"] == 2
    # a backend with errors but no successful runs still appears
    assert stats["by_backend"]["jev"]["runs"] == 0
    assert stats["by_backend"]["jev"]["errors"] == 1
    # a record without telemetry lands in the unknown bucket
    assert stats["by_backend"]["unknown"]["runs"] == 1


def test_failed_run_is_logged_as_error_telemetry():
    from support_triage.errors import ProviderError

    from helpers import fresh_service

    class ExplodingEngine:
        name = "exploding"

        def decide(self, state, questions, context=""):  # noqa: ARG002
            raise ProviderError("provider is down")

    service = fresh_service(engine=ExplodingEngine())
    from helpers import make_ticket

    with pytest.raises(ProviderError):
        service.run_ticket(make_ticket(ticket_id="ERR-1"))
    errors = service.store.all_errors()
    assert len(errors) == 1
    assert errors[0]["backend"] == "exploding"
    assert errors[0]["ticket_id"] == "ERR-1"
    stats = service.stats_performance()
    assert stats["by_backend"]["exploding"]["errors"] == 1
    assert stats["total_runs"] == 0
