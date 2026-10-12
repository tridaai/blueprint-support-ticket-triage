"""Triage-record storage: PostgreSQL in production, in-memory for tests.

The store is where decision records live after a run and where the
agent's decision is recorded. It implements the
:class:`~support_triage.ports.Store` port.

- :class:`PostgresStore` — one short-lived connection per call
  (simple, correct under the API's concurrency model); applies the
  SQL migrations from ``migrations/`` on construction; stores the full
  record as JSONB with promoted status/queue/priority columns for
  queue queries.
- :class:`InMemoryStore` — a thread-safe dict. A TEST DOUBLE: with it,
  records vanish on restart, by design, and the API's /readiness says
  so honestly.

:func:`default_store` picks PostgreSQL when ``DATABASE_URL`` is set,
the double otherwise. Queue projections (open tickets, per-queue
counts) are shared logic over ``all_records`` so both backends answer
identically.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime

from .config import env_str, load_dotenv
from .schemas import TicketInput, TicketResult
from .sla import sla_clock

OPEN_STATUSES = ("awaiting_agent",)


class StoreBase:
    """Queue projections shared by both store backends."""

    def save(self, record: TicketResult) -> None:  # pragma: no cover - port stub
        raise NotImplementedError

    def get(self, ticket_id: str) -> TicketResult | None:  # pragma: no cover
        raise NotImplementedError

    def all_records(self) -> list[TicketResult]:  # pragma: no cover
        raise NotImplementedError

    def save_input(self, ticket: TicketInput) -> None:
        """Persist the raw intake alongside its record (default: no-op
        for stores that archive via the object store instead)."""

    def get_input(self, ticket_id: str) -> TicketInput | None:
        return None

    # -- run-error telemetry ------------------------------------------------
    # Failed runs produce no record, so their telemetry lives here: a
    # small append-only log the performance stats count per backend.
    # The default implementation is process-local (right for the test
    # doubles); PostgresStore persists it in the run_errors table.

    def save_error(self, entry: dict) -> None:
        self.__dict__.setdefault("_error_log", []).append(dict(entry))

    def all_errors(self) -> list[dict]:
        return [dict(e) for e in self.__dict__.get("_error_log", [])]

    # -- QA review records (reviews.py) -----------------------------------
    # Samples and verdicts are stored records of their own — never
    # part of a run's record (a graded triage stays byte-identical).
    # Same arrangement as the error log: a process-local default
    # (right for the test doubles), PostgresStore persists them in
    # the review tables (migration 0004).

    def save_review_sample(self, sample) -> None:
        log = self.__dict__.setdefault("_review_samples", {})
        log[sample.sample_id] = sample.model_copy(deep=True)

    def all_review_samples(self) -> list:
        return [
            sample.model_copy(deep=True)
            for sample in self.__dict__.get("_review_samples", {}).values()
        ]

    def save_review_verdict(self, verdict) -> None:
        self.__dict__.setdefault("_review_verdicts", []).append(
            verdict.model_copy(deep=True)
        )

    def all_review_verdicts(self) -> list:
        return [
            verdict.model_copy(deep=True)
            for verdict in self.__dict__.get("_review_verdicts", [])
        ]

    # -- incident records (incidents.py) ----------------------------------
    # Incidents and cluster dismissals are stored records of their
    # own — the human acts over the computed proposals. Membership
    # lives here, never on a run's record (a member's triage stays
    # byte-identical). Same arrangement as the review records:
    # process-local defaults for the doubles, PostgresStore
    # persists them in the incident tables (migration 0005).

    def save_incident(self, incident) -> None:
        log = self.__dict__.setdefault("_incidents", {})
        log[incident.incident_id] = incident.model_copy(deep=True)

    def all_incidents(self) -> list:
        return [
            incident.model_copy(deep=True)
            for incident in self.__dict__.get("_incidents", {}).values()
        ]

    def save_cluster_dismissal(self, dismissal) -> None:
        self.__dict__.setdefault("_cluster_dismissals", []).append(
            dismissal.model_copy(deep=True)
        )

    def all_cluster_dismissals(self) -> list:
        return [
            dismissal.model_copy(deep=True)
            for dismissal in self.__dict__.get("_cluster_dismissals", [])
        ]

    # -- shift handoff records (handoffs.py) -------------------------------
    # A handoff is a stored record of its own — one composed page
    # per shift change, with the handshake over it. It is never
    # part of a run's record (a shift change must not move a
    # triage). Same arrangement as the incident records:
    # process-local defaults for the doubles, PostgresStore
    # persists them in the handoffs table (migration 0006).

    def save_handoff(self, handoff) -> None:
        log = self.__dict__.setdefault("_handoffs", {})
        log[handoff.handoff_id] = handoff.model_copy(deep=True)

    def all_handoffs(self) -> list:
        return [
            handoff.model_copy(deep=True)
            for handoff in self.__dict__.get("_handoffs", {}).values()
        ]

    def queue_summary(
        self, filters: dict | None = None, now: datetime | None = None
    ) -> list[dict]:
        """Ticket rows for the console queue, highest priority first.

        Unfiltered, this is the open queue (round-1 behaviour: only
        tickets awaiting an agent). ``filters`` may carry ``queue``,
        ``status``, ``priority_band``, ``flag`` (a verify key that
        must be raised), ``backend`` and ``q`` (case-insensitive text
        over subject and sender); naming a ``status`` searches across
        decided tickets too.

        Every row also carries the deterministic SLA clock (sla.py)
        for open tickets: ``sla_state`` (ok | due_soon | breached),
        ``sla_target_hours``, ``age_seconds`` and
        ``seconds_to_breach`` — all ``None`` when the clock isn't
        running (decided ticket, or no parseable received_at). The
        sort order is unchanged: priority score, highest first;
        ``now`` is injectable so tests never race the wall clock.
        """
        filters = filters or {}
        records = list(self.all_records())
        if filters.get("status"):
            records = [r for r in records if r.status == filters["status"]]
        else:
            records = [r for r in records if r.status in OPEN_STATUSES]
        if filters.get("queue"):
            records = [r for r in records if r.composition.queue == filters["queue"]]
        if filters.get("priority_band"):
            records = [
                r for r in records if r.composition.priority_band == filters["priority_band"]
            ]
        if filters.get("flag"):
            records = [r for r in records if r.composition.flags.get(filters["flag"])]
        if filters.get("backend"):
            records = [
                r
                for r in records
                if (r.telemetry.backend if r.telemetry else "") == filters["backend"]
            ]
        if filters.get("q"):
            needle = str(filters["q"]).lower()
            records = [
                r
                for r in records
                if needle in r.subject.lower() or needle in r.sender_name.lower()
            ]
        records.sort(key=lambda r: r.composition.priority_score, reverse=True)
        rows = []
        for r in records:
            clock = sla_clock(r, now)
            rows.append(
                {
                    "ticket_id": r.ticket_id,
                    "subject": r.subject,
                    "sender_name": r.sender_name,
                    "channel": r.channel,
                    "received_at": r.received_at,
                    "category": r.category,
                    "product_area": r.product_area,
                    "priority_score": r.composition.priority_score,
                    "priority_band": r.composition.priority_band,
                    "queue": r.composition.queue,
                    "status": r.status,
                    "backend": r.telemetry.backend if r.telemetry else "",
                    "flags": dict(r.composition.flags),
                    "decision_confidence": round(
                        max((a.confidence for a in r.answers), default=0.0), 3
                    ),
                    "category_confidence": round(
                        next((a.confidence for a in r.answers if a.key == "category"), 0.0), 3
                    ),
                    "requires_human_review": r.guardrails.requires_human_review,
                    "is_duplicate": r.duplicate.is_duplicate,
                    "sla_state": clock["state"] if clock else None,
                    "sla_target_hours": clock["target_hours"] if clock else None,
                    "age_seconds": clock["age_seconds"] if clock else None,
                    "seconds_to_breach": clock["seconds_to_breach"] if clock else None,
                    "sla_due_at": clock["due_at"] if clock else None,
                }
            )
        return rows

    def metrics(self) -> dict:
        records = self.all_records()
        queues: dict[str, int] = {}
        backends: dict[str, int] = {}
        for record in records:
            queues[record.composition.queue] = queues.get(record.composition.queue, 0) + 1
            backend = record.telemetry.backend if record.telemetry else "unknown"
            backends[backend] = backends.get(backend, 0) + 1
        return {
            "total_tickets": len(records),
            "awaiting_agent": sum(1 for r in records if r.status == "awaiting_agent"),
            "triaged": sum(1 for r in records if r.status == "triaged"),
            "rejected": sum(1 for r in records if r.status == "rejected"),
            "human_review_forced": sum(
                1 for r in records if r.guardrails.requires_human_review
            ),
            "duplicates_flagged": sum(1 for r in records if r.duplicate.is_duplicate),
            "drafts_offered": sum(1 for r in records if r.draft is not None),
            "by_queue": queues,
            "by_backend": backends,
        }


class InMemoryStore(StoreBase):
    """Thread-safe in-memory store (tests / offline API smoke runs)."""

    def __init__(self) -> None:
        self._records: dict[str, TicketResult] = {}
        self._inputs: dict[str, TicketInput] = {}
        self._lock = threading.Lock()

    def save(self, record: TicketResult) -> None:
        with self._lock:
            self._records[record.ticket_id] = record

    def get(self, ticket_id: str) -> TicketResult | None:
        with self._lock:
            record = self._records.get(ticket_id)
            return record.model_copy(deep=True) if record else None

    def all_records(self) -> list[TicketResult]:
        with self._lock:
            return [r.model_copy(deep=True) for r in self._records.values()]

    def save_input(self, ticket: TicketInput) -> None:
        with self._lock:
            self._inputs[ticket.ticket_id] = ticket

    def get_input(self, ticket_id: str) -> TicketInput | None:
        with self._lock:
            ticket = self._inputs.get(ticket_id)
            return ticket.model_copy(deep=True) if ticket else None


class PostgresStore(StoreBase):
    """PostgreSQL-backed store (production). Schema in ``migrations/``."""

    def __init__(self, database_url: str) -> None:
        import psycopg  # local import: the double must work without the driver

        self._url = database_url
        self._psycopg = psycopg
        with self._connect() as conn:
            from .migrations import run_migrations

            run_migrations(conn)

    def _connect(self):
        return self._psycopg.connect(self._url)

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone()[0] == 1
        except Exception:
            return False

    def save(self, record: TicketResult) -> None:
        payload = json.dumps(record.model_dump())
        outcome = (
            json.dumps(record.outcome.model_dump()) if record.outcome is not None else None
        )
        decided = record.status in ("triaged", "rejected")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ticket_records (ticket_id, status, queue, priority, decided, payload, outcome, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
                    ON CONFLICT (ticket_id) DO UPDATE
                    SET status = EXCLUDED.status,
                        queue = EXCLUDED.queue,
                        priority = EXCLUDED.priority,
                        decided = EXCLUDED.decided,
                        payload = EXCLUDED.payload,
                        outcome = EXCLUDED.outcome,
                        updated_at = now()
                    """,
                    (
                        record.ticket_id,
                        record.status,
                        record.composition.queue,
                        record.composition.priority_score,
                        decided,
                        payload,
                        outcome,
                    ),
                )
            conn.commit()

    def save_error(self, entry: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO run_errors (ticket_id, backend, error) VALUES (%s, %s, %s)",
                    (
                        str(entry.get("ticket_id", "")),
                        str(entry.get("backend", "")),
                        str(entry.get("error", "")),
                    ),
                )
            conn.commit()

    def all_errors(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ticket_id, backend, error, at FROM run_errors ORDER BY id"
                )
                rows = cur.fetchall()
        return [
            {
                "ticket_id": row[0],
                "backend": row[1],
                "error": row[2],
                "at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
            }
            for row in rows
        ]

    # -- QA review records: persisted in their own tables (0004) ----------

    def save_review_sample(self, sample) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO review_samples (sample_id, payload)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (sample_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (sample.sample_id, json.dumps(sample.model_dump())),
                )
            conn.commit()

    def all_review_samples(self) -> list:
        from .schemas import ReviewSample

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM review_samples ORDER BY sample_id")
                rows = cur.fetchall()
        return [ReviewSample.model_validate(row[0]) for row in rows]

    def save_review_verdict(self, verdict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO review_verdicts (sample_id, ticket_id, reviewer, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (
                        verdict.sample_id,
                        verdict.ticket_id,
                        verdict.reviewer,
                        json.dumps(verdict.model_dump()),
                    ),
                )
            conn.commit()

    def all_review_verdicts(self) -> list:
        from .schemas import ReviewVerdict

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM review_verdicts ORDER BY id")
                rows = cur.fetchall()
        return [ReviewVerdict.model_validate(row[0]) for row in rows]

    # -- incident records: persisted in their own tables (0005) -----------

    def save_incident(self, incident) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incidents (incident_id, payload)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (incident_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (incident.incident_id, json.dumps(incident.model_dump())),
                )
            conn.commit()

    def all_incidents(self) -> list:
        from .schemas import Incident

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM incidents ORDER BY incident_id")
                rows = cur.fetchall()
        return [Incident.model_validate(row[0]) for row in rows]

    def save_cluster_dismissal(self, dismissal) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_dismissals (fingerprint, payload)
                    VALUES (%s, %s::jsonb)
                    """,
                    (
                        ",".join(dismissal.member_ticket_ids),
                        json.dumps(dismissal.model_dump()),
                    ),
                )
            conn.commit()

    def all_cluster_dismissals(self) -> list:
        from .schemas import ClusterDismissal

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM cluster_dismissals ORDER BY id")
                rows = cur.fetchall()
        return [ClusterDismissal.model_validate(row[0]) for row in rows]

    # -- shift handoff records: persisted in their own table (0006) --------

    def save_handoff(self, handoff) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO handoffs (handoff_id, payload)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (handoff_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (handoff.handoff_id, json.dumps(handoff.model_dump())),
                )
            conn.commit()

    def all_handoffs(self) -> list:
        from .schemas import ShiftHandoff

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM handoffs ORDER BY handoff_id")
                rows = cur.fetchall()
        return [ShiftHandoff.model_validate(row[0]) for row in rows]

    def get(self, ticket_id: str) -> TicketResult | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM ticket_records WHERE ticket_id = %s",
                    (ticket_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return TicketResult.model_validate(row[0])

    def all_records(self) -> list[TicketResult]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM ticket_records ORDER BY updated_at")
                rows = cur.fetchall()
        return [TicketResult.model_validate(row[0]) for row in rows]

    # Raw inputs live in the object store in production (see service);
    # the record payload carries everything the queue needs.


def default_store() -> StoreBase:
    """PostgreSQL when DATABASE_URL is set; the in-memory double otherwise."""
    load_dotenv()
    url = env_str("DATABASE_URL", "")
    if url:
        return PostgresStore(url)
    return InMemoryStore()
