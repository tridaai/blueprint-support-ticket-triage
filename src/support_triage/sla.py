"""SLA clocks: the deterministic side of "will this breach?".

Two different things carry the letters SLA in this blueprint, and
they are kept apart on purpose — in naming, in code, and on the
record:

- ``sla_breach_risk`` — a *model judgement*. One of the eleven typed
  questions: a probability distribution over a rubric, produced by
  whichever decision backend ran, feeding the priority formula.
- the **SLA clock** (this module) — *arithmetic*. Each priority band
  carries a target in
  :data:`support_triage.composition.SLA_TARGET_HOURS`; the clock
  compares the ticket's age (now − ``received_at``) with its band's
  target and reports one of three states. No model is consulted and
  no probability is involved.

States, for an open ticket (``awaiting_agent``):

- ``ok`` — more than the last quarter of the target remains;
- ``due_soon`` — the remaining time has fallen to
  :data:`~support_triage.composition.SLA_DUE_SOON_FRACTION` (25%) of
  the target or less, but has not run out (a critical ticket, 1h
  target, is due soon with 15 minutes left; a low ticket, 72h, with
  18 hours left);
- ``breached`` — the target has run out.

Only open tickets have a running clock: once an agent decides, the
record keeps its timestamps but :func:`sla_clock` returns ``None``.
A record whose ``received_at`` is missing or unparseable likewise
has no clock — the honest answer is no clock, never a guessed one.
Every function here takes ``now`` explicitly so tests inject the
clock instead of racing the wall.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .composition import SLA_DUE_SOON_FRACTION, SLA_TARGET_HOURS
from .schemas import TicketResult

SLA_STATES = ("ok", "due_soon", "breached")


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` suffix welcome) to an aware
    datetime; ``None`` when the value is empty or unparseable."""
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_timestamp(moment: datetime) -> str:
    """Render an aware datetime in the record's ISO-Z shape."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reading(received: datetime, band: str, moment: datetime) -> dict | None:
    """The clock's arithmetic at one moment, for one arrival + band.

    The single computation behind every SLA reading in the system:
    the live clock (:func:`sla_clock`), the breach watch, and the
    attainment report's decision-moment judgements all read through
    here, so a state named anywhere is the same arithmetic
    everywhere.
    """
    target_hours = SLA_TARGET_HOURS.get(band)
    if target_hours is None:  # pragma: no cover - bands are a closed set
        return None
    target_seconds = target_hours * 3600.0
    age_seconds = (moment - received).total_seconds()
    remaining = target_seconds - age_seconds
    if remaining < 0:
        state = "breached"
    elif remaining <= target_seconds * SLA_DUE_SOON_FRACTION:
        state = "due_soon"
    else:
        state = "ok"
    from datetime import timedelta

    return {
        "state": state,
        "target_hours": target_hours,
        "age_seconds": round(age_seconds, 1),
        "seconds_to_breach": round(remaining, 1),
        "due_at": format_timestamp(received + timedelta(seconds=target_seconds)),
    }


def sla_reading_at(record: TicketResult, moment: datetime) -> dict | None:
    """The clock's reading for a record at a stated moment.

    :func:`sla_clock` answers for a run whose clock is still
    running; this is the same arithmetic evaluated at a moment the
    caller names — the attainment report judges a decision by the
    reading at its ``decided_at`` — whatever the record's status
    now. ``None`` when the record has no parseable ``received_at``:
    no clock ever started on record, and no reading is guessed.
    """
    received = parse_timestamp(record.received_at)
    if received is None:
        return None
    return _reading(received, record.composition.priority_band, moment)


def sla_clock(record: TicketResult, now: datetime | None = None) -> dict | None:
    """The SLA clock for one record, or ``None`` when it isn't running.

    Returns the state, the band's target, the ticket's age, and the
    seconds left before breach (negative once breached).
    """
    if record.status != "awaiting_agent":
        return None
    received = parse_timestamp(record.received_at)
    if received is None:
        return None
    now = now or datetime.now(timezone.utc)
    return _reading(received, record.composition.priority_band, now)
