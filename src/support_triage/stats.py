"""Aggregate statistics over the run store: the dashboards' data layer.

Pure functions over stored decision records (plus the store's
run-error log), shared by every backend — the PostgreSQL store and
the in-memory test double answer identically because both feed this
module the same record list. Nothing here calls a model or mutates
state; it counts what the pipeline already recorded:

- :func:`overview_stats` — queue health and automation posture:
  counts by status / queue / category, the priority-band spread,
  safety-flag counts, guardrail failures by check, and the three
  rates an operator actually argues about (how often low confidence
  forced a human, how often a template draft was offered, how often
  the autonomy recommendation said yes — remembering it is never
  acted on).
- :func:`performance_stats` — per decision backend: run counts,
  decision-call counts, latency percentiles over the recorded
  decision latency, token totals and estimated cost, and the count
  of failed runs the store's error log holds for that backend.
"""

from __future__ import annotations

from datetime import datetime

from . import corrections
from .composition import BANDS
from .schemas import TicketResult
from .sla import sla_clock

SAFETY_FLAG_KEYS = (
    "contains_pii",
    "is_abusive",
    "mentions_legal",
    "is_outage_report",
    "needs_engineering",
)

BAND_ORDER = [name for _floor, name in BANDS]


def percentile(values: list[float], q: float) -> float:
    """The ``q``-quantile (0..1) with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * fraction, 2)


def _count_by(records: list[TicketResult], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = key(record)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rate(count: int, total: int) -> dict:
    return {"count": count, "rate": round(count / total, 4) if total else 0.0}


def overview_stats(records: list[TicketResult], now: datetime | None = None) -> dict:
    """Queue health + automation posture over every stored run.

    The ``sla`` block counts the deterministic SLA clock (sla.py)
    over open tickets whose clock is running — breached and due-soon
    are the two numbers a queue lead watches. ``now`` is injectable
    for tests; the API passes nothing and gets the wall clock.
    """
    total = len(records)
    bands = {name: 0 for name in BAND_ORDER}
    for record in records:
        band = record.composition.priority_band
        bands[band] = bands.get(band, 0) + 1

    flags = {key: 0 for key in SAFETY_FLAG_KEYS}
    for record in records:
        for key in SAFETY_FLAG_KEYS:
            if record.composition.flags.get(key):
                flags[key] += 1

    guardrail_failures: dict[str, int] = {}
    for record in records:
        for check in record.guardrails.checks:
            if not check.get("passed", True):
                name = str(check.get("name", "unknown"))
                guardrail_failures[name] = guardrail_failures.get(name, 0) + 1

    sla_counts = {"breached": 0, "due_soon": 0, "ok": 0}
    for record in records:
        clock = sla_clock(record, now)
        if clock is not None:
            sla_counts[clock["state"]] += 1

    return {
        "total_runs": total,
        "by_status": _count_by(records, lambda r: r.status),
        "by_queue": _count_by(records, lambda r: r.composition.queue),
        "by_category": _count_by(records, lambda r: r.category),
        "priority_bands": bands,
        "sla": sla_counts,
        "safety_flags": flags,
        "guardrail_failures": dict(sorted(guardrail_failures.items())),
        "rates": {
            # requires_human_review is set by exactly one thing — the
            # low-confidence guardrail — so this IS the forcing rate.
            "low_confidence_forcing": _rate(
                sum(1 for r in records if r.guardrails.requires_human_review), total
            ),
            "template_draft": _rate(sum(1 for r in records if r.draft is not None), total),
            "autonomy_recommended": _rate(
                sum(1 for r in records if r.autonomy.eligible), total
            ),
        },
        "outcomes_recorded": sum(1 for r in records if r.outcome is not None),
        # Volume counts every run, confirmed duplicates included —
        # they were tickets and agents handled them. The mark only
        # removes them from calibration sampling (livecal.py); the
        # count here is how an operator sees that population.
        "confirmed_duplicates": sum(
            1 for r in records if corrections.is_confirmed_duplicate(r)
        ),
    }


def performance_stats(records: list[TicketResult], errors: list[dict]) -> dict:
    """Per-backend decision performance from run telemetry + the error log."""
    by_backend: dict[str, dict] = {}

    def bucket(name: str) -> dict:
        return by_backend.setdefault(
            name,
            {
                "runs": 0,
                "decision_calls": 0,
                "questions": 0,
                "_latencies": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "runs_with_usage": 0,
                "_cost": 0.0,
                "errors": 0,
            },
        )

    for record in records:
        telemetry = record.telemetry
        name = telemetry.backend if telemetry and telemetry.backend else "unknown"
        entry = bucket(name)
        entry["runs"] += 1
        entry["decision_calls"] += 1  # one decision call per run, by design
        if telemetry is None:
            continue
        entry["questions"] += telemetry.question_count
        if telemetry.decision_latency_ms:
            entry["_latencies"].append(float(telemetry.decision_latency_ms))
        if telemetry.input_tokens is not None or telemetry.output_tokens is not None:
            entry["runs_with_usage"] += 1
            entry["input_tokens"] += telemetry.input_tokens or 0
            entry["output_tokens"] += telemetry.output_tokens or 0
        if telemetry.estimated_cost_usd is not None:
            entry["_cost"] += telemetry.estimated_cost_usd

    for error in errors:
        bucket(str(error.get("backend") or "unknown"))["errors"] += 1

    rendered: dict[str, dict] = {}
    for name, entry in sorted(by_backend.items()):
        latencies = entry.pop("_latencies")
        cost = entry.pop("_cost")
        entry["decision_latency_ms"] = {
            "count": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        }
        entry["estimated_cost_usd"] = round(cost, 6)
        rendered[name] = entry

    return {
        "total_runs": len(records),
        "total_errors": len(errors),
        "by_backend": rendered,
    }
