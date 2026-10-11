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

from .config import env_str, load_dotenv
from .schemas import TicketInput, TicketResult

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

    def queue_summary(self) -> list[dict]:
        """Open tickets (awaiting an agent), newest priority first."""
        records = [r for r in self.all_records() if r.status in OPEN_STATUSES]
        records.sort(key=lambda r: r.composition.priority_score, reverse=True)
        return [
            {
                "ticket_id": r.ticket_id,
                "subject": r.subject,
                "sender_name": r.sender_name,
                "channel": r.channel,
                "category": r.category,
                "product_area": r.product_area,
                "priority_score": r.composition.priority_score,
                "priority_band": r.composition.priority_band,
                "queue": r.composition.queue,
                "status": r.status,
                "decision_confidence": round(
                    max((a.confidence for a in r.answers), default=0.0), 3
                ),
                "category_confidence": round(
                    next((a.confidence for a in r.answers if a.key == "category"), 0.0), 3
                ),
                "requires_human_review": r.guardrails.requires_human_review,
                "is_duplicate": r.duplicate.is_duplicate,
            }
            for r in records
        ]

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
        decided = record.status in ("triaged", "rejected")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ticket_records (ticket_id, status, queue, priority, decided, payload, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (ticket_id) DO UPDATE
                    SET status = EXCLUDED.status,
                        queue = EXCLUDED.queue,
                        priority = EXCLUDED.priority,
                        decided = EXCLUDED.decided,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    """,
                    (
                        record.ticket_id,
                        record.status,
                        record.composition.queue,
                        record.composition.priority_score,
                        decided,
                        payload,
                    ),
                )
            conn.commit()

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
