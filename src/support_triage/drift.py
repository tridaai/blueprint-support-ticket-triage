"""Calibration drift watch: is the live model getting *worse*?

Live calibration (livecal.py) answers "how calibrated are we?" over
every outcome ever recorded. That average hides the failure mode
operators actually fear: the engine was fine in March and has been
quietly degrading since — a provider model changed, the ticket mix
shifted, a queue's tickets got harder. This module watches for
exactly that. It replays the labelled outcome stream in recorded
order, scores the most recent window of outcomes with the same
:func:`support_triage.calibration.calibration_report` the eval pack
and live calibration use, and compares each question's windowed
Brier against its **eval baseline** (the eval pack's per-question
Brier — the number the engine shipped with):

    drift  ⟺  window_n ≥ DRIFT_MIN_SAMPLES
              and window_brier − eval_brier ≥ DRIFT_BRIER_THRESHOLD

The minimum-sample gate matters as much as the threshold: a window
with six outcomes in it flags noise, not drift, so a question below
the gate is reported as warming up and can never flag. The window
counts *labelled* outcomes — records whose recorded outcome produced
at least one calibration sample — in ``recorded_at`` order.

Thresholds are deployment config (environment-overridable, read at
call time): ``DRIFT_WINDOW`` (50), ``DRIFT_MIN_SAMPLES`` (20),
``DRIFT_BRIER_THRESHOLD`` (0.05). The payload carries the values in
force so the dashboard never shows a flag without its ruler.
"""

from __future__ import annotations

from .calibration import calibration_report
from .config import env_float, env_int
from .livecal import live_samples
from .schemas import TicketResult

DRIFT_WINDOW = 50
DRIFT_MIN_SAMPLES = 20
DRIFT_BRIER_THRESHOLD = 0.05


def _labelled_stream(records: list[TicketResult]) -> list[dict]:
    """Calibration samples in recorded order, one entry per record
    whose outcome produced at least one sample."""
    with_outcomes = [r for r in records if r.outcome is not None]
    with_outcomes.sort(key=lambda r: (r.outcome.recorded_at or "", r.ticket_id))
    stream: list[dict] = []
    for record in with_outcomes:
        stream.extend(live_samples([record]))
    return stream


def drift_report(
    records: list[TicketResult],
    eval_per_question: dict | None,
) -> dict:
    """Windowed-vs-baseline calibration drift over the outcome stream.

    ``eval_per_question`` is the eval pack's ``per_question`` mapping
    (question key → ``{"kind": ..., "brier": ...}``), or ``None`` when
    no eval run exists — without a baseline nothing can be called
    drift, and the payload says so instead of guessing.
    """
    window_size = env_int("DRIFT_WINDOW", DRIFT_WINDOW)
    min_samples = env_int("DRIFT_MIN_SAMPLES", DRIFT_MIN_SAMPLES)
    threshold = env_float("DRIFT_BRIER_THRESHOLD", DRIFT_BRIER_THRESHOLD)

    stream = _labelled_stream(records)
    windowed = stream[-window_size:] if window_size > 0 else []
    live = calibration_report(stream)["per_question"] if stream else {}
    recent = calibration_report(windowed)["per_question"] if windowed else {}
    baseline = eval_per_question or {}

    keys = sorted(set(baseline) | set(live) | set(recent))
    questions: dict[str, dict] = {}
    any_drift = False
    for key in keys:
        eval_row = baseline.get(key) or {}
        live_row = live.get(key) or {}
        win_row = recent.get(key) or {}
        eval_brier = eval_row.get("brier")
        win_brier = win_row.get("brier")
        win_n = int(win_row.get("n", 0))
        enough = win_n >= min_samples
        delta = (
            round(win_brier - eval_brier, 4)
            if win_brier is not None and eval_brier is not None
            else None
        )
        drift = bool(
            enough and delta is not None and delta >= threshold
        )
        any_drift = any_drift or drift
        questions[key] = {
            "kind": eval_row.get("kind") or live_row.get("kind") or win_row.get("kind"),
            "eval_brier": eval_brier,
            "live_brier": live_row.get("brier"),
            "live_n": int(live_row.get("n", 0)),
            "window_brier": win_brier,
            "window_n": win_n,
            "delta_vs_eval": delta,
            "enough_samples": enough,
            "drift": drift,
        }

    if not stream:
        note = (
            "no agent outcomes recorded yet — record outcomes and the "
            "drift watch starts comparing the live window to the eval baseline"
        )
    elif eval_per_question is None:
        note = (
            "no eval baseline yet — run the eval pack (make evals); "
            "without the baseline Brier, degradation has nothing to be measured against"
        )
    elif any_drift:
        note = "one or more questions degraded past the threshold in the recent window"
    else:
        note = "no question is past the drift threshold in the recent window"

    return {
        "available": bool(stream),
        "note": note,
        "window": {
            "size": window_size,
            "min_samples": min_samples,
            "brier_threshold": threshold,
            "outcomes_in_window": len(windowed),
            "outcomes_total": len(stream),
        },
        "any_drift": any_drift,
        "questions": questions,
    }
