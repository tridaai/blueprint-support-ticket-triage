"""Persistence doubles, calibration math, configuration, infrastructure."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from support_triage.calibration import (
    brier_binary,
    brier_multiclass,
    calibration_report,
    reliability_bins,
    score_event_probability,
)
from support_triage.embeddings import HashingEmbedder, InMemoryVectorIndex, cosine
from support_triage.objectstore import InMemoryObjectStore
from support_triage.schemas import TicketResult
from support_triage.store import InMemoryStore

from helpers import fresh_service, make_ticket

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# stores and indexes (test doubles behave like the real contracts)
# ---------------------------------------------------------------------------


def _record(ticket_id="T-1", **kwargs) -> TicketResult:
    service = fresh_service()
    return service.run_ticket(make_ticket(ticket_id=ticket_id, **kwargs)).result


def test_inmemory_store_roundtrip_and_isolation():
    store = InMemoryStore()
    record = _record()
    store.save(record)
    fetched = store.get("T-1")
    assert fetched is not None and fetched.ticket_id == "T-1"
    fetched.status = "triaged"  # mutating the copy must not leak back
    assert store.get("T-1").status == "awaiting_agent"
    assert store.get("MISSING") is None
    assert len(store.all_records()) == 1


def test_inmemory_object_store_roundtrip():
    store = InMemoryObjectStore()
    assert store.get("k") is None
    assert store.exists("k") is False
    store.put("k", b"payload")
    assert store.get("k") == b"payload"
    assert store.exists("k") is True


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder()
    a = embedder.embed("the dashboard export is broken")
    b = embedder.embed("the dashboard export is broken")
    assert a == b
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, embedder.embed("refund for a double charge please")) < 0.5


def test_inmemory_index_finds_the_nearest_neighbour():
    embedder = HashingEmbedder()
    index = InMemoryVectorIndex()
    index.add("A", embedder.embed("dashboard export uses semicolons in the csv file"), "m")
    index.add("B", embedder.embed("please refund my double charge from march"), "m")
    nearest = index.nearest(embedder.embed("dashboard export uses semicolons in the csv"), limit=1)
    assert nearest[0][0] == "A"
    assert nearest[0][1] > 0.8
    excluded = index.nearest(embedder.embed("dashboard export uses semicolons in the csv file"), exclude_id="A", limit=1)
    assert excluded[0][0] == "B"


def test_queue_summary_orders_by_priority_descending():
    store = InMemoryStore()
    low = _record("T-LOW")
    store.save(low)
    high_service = fresh_service()
    outage = next(
        t for t in __import__("support_triage.samples", fromlist=["load_sample_tickets"]).load_sample_tickets()
        if t.ticket_id == "SUP-1002"
    )
    store.save(high_service.run_ticket(outage).result)
    summary = store.queue_summary()
    assert summary[0]["priority_score"] >= summary[-1]["priority_score"]
    assert summary[0]["ticket_id"] == "SUP-1002"


# ---------------------------------------------------------------------------
# calibration math
# ---------------------------------------------------------------------------


def test_brier_binary_extremes():
    assert brier_binary([(1.0, 1.0), (0.0, 0.0)]) == 0.0
    assert brier_binary([(1.0, 0.0)]) == 1.0
    assert brier_binary([(0.5, 1.0), (0.5, 0.0)]) == pytest.approx(0.25)


def test_brier_multiclass():
    assert brier_multiclass([({"a": 1.0, "b": 0.0}, "a")]) == 0.0
    assert brier_multiclass([({"a": 0.5, "b": 0.5}, "a")]) == pytest.approx(0.5)


def test_reliability_bins_group_and_measure():
    pairs = [(0.05, 0.0)] * 9 + [(0.95, 1.0)]
    bins = reliability_bins(pairs)
    low = bins[0]
    assert low["n"] == 9 and low["empirical"] == 0.0
    assert bins[9]["n"] == 1 and bins[9]["empirical"] == 1.0
    empty = bins[5]
    assert empty["n"] == 0 and empty["gap"] is None


def test_score_event_probability():
    dist = {str(s): 0.05 for s in range(11)}
    dist["9"] = 0.5  # total mass is now exactly 1.0
    assert score_event_probability(dist, 7) == pytest.approx(0.65)
    assert score_event_probability(dist, 10) == pytest.approx(0.05)


def test_calibration_report_shapes_and_gaps():
    samples = [
        {"verify": {"contains_pii": (0.9, True)}, "choice": {}, "score": {}},
        {"verify": {"contains_pii": (0.1, False)}, "choice": {}, "score": {}},
    ]
    report = calibration_report(samples)
    assert report["cases"] == 2
    entry = report["per_question"]["contains_pii"]
    assert entry["kind"] == "verify"
    assert entry["brier"] == pytest.approx(0.01)
    # two tiny bins never gate: worst gap stays 0 at the default min n
    assert report["worst_bin_gap"] == 0.0


def test_calibration_report_flags_a_badly_miscalibrated_bin():
    samples = [
        {"verify": {"is_abusive": (0.95, False)}, "choice": {}, "score": {}}
        for _ in range(10)
    ]
    report = calibration_report(samples)
    assert report["worst_bin_gap"] > 0.9
    assert report["worst_bin"]["question"] == "is_abusive"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_env_helpers(monkeypatch):
    from support_triage.config import env_float, env_int, env_str

    monkeypatch.setenv("X_STR", "value")
    monkeypatch.setenv("X_FLOAT", "2.5")
    monkeypatch.setenv("X_INT", "7")
    monkeypatch.setenv("X_BAD", "not-a-number")
    assert env_str("X_STR") == "value"
    assert env_str("X_MISSING", "fallback") == "fallback"
    assert env_float("X_FLOAT", 1.0) == 2.5
    assert env_float("X_BAD", 1.0) == 1.0
    assert env_int("X_INT", 1) == 7
    assert env_int("X_BAD", 3) == 3


def test_dotenv_parser(tmp_path):
    from support_triage.config import _parse_line

    assert _parse_line("KEY=value") == ("KEY", "value")
    assert _parse_line('export KEY="quoted value"') == ("KEY", "quoted value")
    assert _parse_line("# comment") is None
    assert _parse_line("no-equals-here") is None
    assert _parse_line("  SPACED = yes ") == ("SPACED", "yes")


# ---------------------------------------------------------------------------
# infrastructure files (the production spine is part of the contract)
# ---------------------------------------------------------------------------


def test_migrations_are_numbered_and_cover_both_tables():
    files = sorted((REPO_ROOT / "migrations").glob("[0-9]*.sql"))
    assert [f.name for f in files] == [
        "0001_ticket_records.sql",
        "0002_ticket_embeddings.sql",
        "0003_outcomes_and_run_errors.sql",
        "0004_review_samples.sql",
    ]
    text = "".join(f.read_text() for f in files)
    assert "ticket_records" in text and "ticket_embeddings" in text
    assert "outcome" in text and "run_errors" in text
    assert "review_samples" in text and "review_verdicts" in text
    assert "CREATE EXTENSION" in text and "vector" in text


def test_compose_runs_the_production_topology():
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "pgvector/pgvector:pg16" in compose
    assert "seaweedfs" in compose.lower()
    assert "http://minio:8333" in compose
    assert "DATABASE_URL" in compose
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "USER app" in dockerfile or "USER 10001" in dockerfile or "useradd" in dockerfile
    assert "/health" in dockerfile


def test_constraints_are_frozen_and_pip_safe():
    constraints = (REPO_ROOT / "constraints.txt").read_text()
    assert "-e ." not in constraints
    assert "langgraph==0.6.11" in constraints
    assert "fastapi==0.143.0" in constraints


def test_checkpointer_selection(monkeypatch):
    from support_triage.checkpoints import default_checkpointer

    monkeypatch.setenv("CHECKPOINTS", "off")
    assert default_checkpointer() is None
    monkeypatch.delenv("CHECKPOINTS")
    saver = default_checkpointer()
    assert saver is not None  # MemorySaver without DATABASE_URL
