"""Service layer: runs the graph, owns persistence and the gate.

The surfaces (API, CLI, demo, evals) talk to :class:`TriageService`,
never to the graph directly. The service:

- starts a run (graph → agent gate), saves the decision record, and
  archives the raw ticket payload to the object store;
- records the agent's decision — by resuming the graph when its
  checkpoint is live, or by applying the decision directly to the
  stored record when it is not (API restart, checkpoints off). The
  record is the source of truth either way, and the two paths produce
  the same record;
- answers queue/metrics/record reads from the store.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from langgraph.types import Command
from pydantic import ValidationError

from . import webhooks
from .events import CollectingSink, EventBus, RunEvent
from .graph import AgentRun, build_graph
from .ports import Checkpointer, DecisionEngine, Embedder, EventSink, ObjectStore, Store, VectorIndex
from .schemas import AgentOutcome, TicketInput, TicketResult
from .sla import format_timestamp

# A batch is an intake convenience, not a bulk-import channel: the
# cap keeps one request from monopolising the service. Tickets past
# the cap are rejected as a whole (a loud 422), never silently cut.
BATCH_MAX_TICKETS = 50

# The floor for "similar past tickets" (service.similar_tickets):
# well below the duplicate threshold (0.85) — a neighbour worth
# learning from, not a re-send. Env-tunable per deployment.
SIMILARITY_FLOOR = 0.30

# Lifecycle event types on the service bus (and the SSE stream):
# run_completed, guardrail_blocked, outcome_recorded, batch_completed.
_WEBHOOK_EVENTS = ("run_completed", "guardrail_blocked")


class TriageService:
    def __init__(
        self,
        engine: DecisionEngine,
        embedder: Embedder,
        index: VectorIndex,
        store: Store,
        objects: ObjectStore,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder
        self.index = index
        self.store = store
        self.objects = objects
        self._checkpointer = checkpointer
        self._checkpointer_ready = checkpointer is not None
        # Lifecycle fan-out: SSE subscribers (and anything in-process)
        # hear about completed runs, guardrail blocks, recorded
        # outcomes, and finished batches as they happen.
        self.bus = EventBus()

    # -- lifecycle events ---------------------------------------------------

    def _publish(self, event_type: str, ticket_id: str, detail: dict) -> None:
        """Publish one lifecycle event to the bus, and fire the gate
        webhook for the two event types it covers. Neither channel can
        fail the caller's work: the bus never blocks, the webhook is
        fire-and-forget."""
        event = {
            "type": event_type,
            "ticket_id": ticket_id,
            "at": format_timestamp(datetime.now(timezone.utc)),
            "detail": detail,
        }
        self.bus.publish(event)
        if event_type in _WEBHOOK_EVENTS:
            webhooks.notify(
                event_type,
                {
                    "ticket_id": ticket_id,
                    "queue": detail.get("queue", ""),
                    "priority_band": detail.get("priority_band", ""),
                    "status": detail.get("status", ""),
                    "guardrails": detail.get("guardrails", {}),
                    "at": event["at"],
                },
            )

    @staticmethod
    def _guardrail_summary(result: TicketResult) -> dict:
        return {
            "passed": result.guardrails.passed,
            "requires_human_review": result.guardrails.requires_human_review,
            "failed_checks": [
                str(check.get("name", "unknown"))
                for check in result.guardrails.checks
                if not check.get("passed", True)
            ],
        }

    def _get_checkpointer(self):
        """One checkpointer per service, created lazily and reused —
        the in-memory saver's state lives in the instance, so a fresh
        saver per call would lose every paused gate."""
        if not self._checkpointer_ready:
            from .checkpoints import default_checkpointer

            self._checkpointer = default_checkpointer()
            self._checkpointer_ready = True
        return self._checkpointer

    # -- runs -------------------------------------------------------------

    def run_ticket(self, ticket: TicketInput, sink: EventSink | None = None) -> AgentRun:
        """Run one ticket to the agent gate and persist the record."""
        sink = sink or CollectingSink()
        graph = build_graph(
            self.engine, self.embedder, self.index, self._get_checkpointer(), sink
        )
        config = {"configurable": {"thread_id": ticket.ticket_id}}
        sink.emit(RunEvent(type="run_started", ticket_id=ticket.ticket_id))
        try:
            final = graph.invoke(
                {"ticket": ticket.model_dump(), "started_at": time.time(), "trace": []},
                config,
            )
        except Exception as exc:
            sink.emit(
                RunEvent(type="run_failed", ticket_id=ticket.ticket_id, detail={"error": str(exc)})
            )
            try:
                self.store.save_error(
                    {
                        "ticket_id": ticket.ticket_id,
                        "backend": getattr(self.engine, "name", ""),
                        "error": str(exc),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            except Exception:
                pass  # telemetry must never mask the run's real error
            raise
        awaiting = "__interrupt__" in final
        result = TicketResult.model_validate(final["result"])
        self.store.save(result)
        self.store.save_input(ticket)
        self._archive(ticket)
        sink.emit(
            RunEvent(
                type="run_completed",
                ticket_id=ticket.ticket_id,
                detail={"status": result.status},
            )
        )
        detail = {
            "status": result.status,
            "queue": result.composition.queue,
            "priority_band": result.composition.priority_band,
            "guardrails": self._guardrail_summary(result),
        }
        self._publish("run_completed", ticket.ticket_id, detail)
        if not result.guardrails.passed:
            # A failed check is a blocked decision path (no draft, no
            # autonomy path): a lifecycle fact of its own, on the bus
            # and on the webhook, not a footnote of run_completed.
            self._publish("guardrail_blocked", ticket.ticket_id, detail)
        return AgentRun(result=result, awaiting_agent=awaiting)

    def run_batch(self, items: list) -> dict:
        """Triage a list of raw ticket payloads, one by one.

        Batch intake is a loop over :meth:`run_ticket`, nothing
        more: every ticket gets the full pipeline and its own gate.
        A ticket that fails — malformed payload, dead provider —
        is reported in its own entry and never sinks the batch;
        a batch past :data:`BATCH_MAX_TICKETS` is rejected whole.
        """
        if len(items) > BATCH_MAX_TICKETS:
            raise ValueError(
                f"batch of {len(items)} exceeds the cap of {BATCH_MAX_TICKETS} tickets"
            )
        results: list[dict] = []
        succeeded = 0
        for item in items:
            ticket_id = str(item.get("ticket_id", "")) if isinstance(item, dict) else ""
            try:
                ticket = TicketInput.model_validate(item)
            except ValidationError as exc:
                results.append(
                    {"ticket_id": ticket_id, "status": "error", "error": f"invalid ticket: {exc}"}
                )
                continue
            try:
                run = self.run_ticket(ticket)
            except Exception as exc:
                results.append(
                    {"ticket_id": ticket.ticket_id, "status": "error", "error": str(exc)}
                )
                continue
            succeeded += 1
            results.append(
                {
                    "ticket_id": run.result.ticket_id,
                    "status": run.result.status,
                    "queue": run.result.composition.queue,
                    "priority_band": run.result.composition.priority_band,
                    "priority_score": run.result.composition.priority_score,
                }
            )
        counts = {
            "total": len(items),
            "succeeded": succeeded,
            "failed": len(items) - succeeded,
        }
        self._publish("batch_completed", "", {"counts": counts})
        return {"results": results, "counts": counts}

    def record_decision(
        self,
        ticket_id: str,
        decision: str,
        by: str = "",
        reason: str = "",
        assignee: str = "",
    ) -> TicketResult:
        """Record the agent's decision for a ticket waiting at the gate."""
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        if record.status != "awaiting_agent":
            raise ValueError(
                f"ticket {ticket_id} is {record.status}, not awaiting_agent — "
                "a decision has already been recorded"
            )
        payload = {
            "decision": decision,
            "by": by,
            "reason": reason,
            "assignee": assignee,
        }
        resumed = self._try_resume(ticket_id, payload)
        if resumed is not None:
            self.store.save(resumed)
            return resumed
        # No live checkpoint (restart / checkpoints off): apply the same
        # transition directly to the stored record.
        record.decision = decision
        record.decision_by = by
        note = reason
        if assignee:
            note = f"{note} [assigned to {assignee}]".strip()
        record.decision_reason = note
        record.status = "triaged" if decision == "approved" else "rejected"
        self.store.save(record)
        return record

    def assign(self, ticket_id: str, assignee: str, by: str = "") -> TicketResult:
        """Assign a waiting ticket to an agent without deciding it."""
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        note = f"[assigned to {assignee}]"
        record.decision_reason = f"{record.decision_reason} {note}".strip()
        if by:
            record.decision_by = record.decision_by or by
        self.store.save(record)
        return record

    def _try_resume(self, ticket_id: str, payload: dict) -> TicketResult | None:
        try:
            graph = build_graph(
                self.engine,
                self.embedder,
                self.index,
                self._get_checkpointer(),
                CollectingSink(),
            )
            config = {"configurable": {"thread_id": ticket_id}}
            snapshot = graph.get_state(config)
            if not snapshot or not snapshot.next:
                return None
            final = graph.invoke(Command(resume=payload), config)
            if "result" not in final:
                return None
            return TicketResult.model_validate(final["result"])
        except Exception:
            return None

    # -- comparison + similar (experiments and neighbours, not runs) -------

    def compare(self, ticket: TicketInput, backend_names: list[str] | None = None) -> dict:
        """Run one ticket through several decision backends (compare.py).

        An experiment, not a triage: nothing is queued, gated, or
        added to the duplicate index. The comparison record is
        archived to the object store, apart from the run records.
        """
        from .compare import compare_ticket

        payload = compare_ticket(ticket, backend_names)
        try:
            self.objects.put(
                f"comparisons/{payload['comparison_id']}.json",
                json.dumps(payload, indent=2).encode("utf-8"),
                "application/json",
            )
        except Exception:
            pass  # the caller has the payload; the archive is a copy
        return payload

    def get_comparison(self, comparison_id: str) -> dict | None:
        """Read back an archived comparison record, if it exists."""
        try:
            raw = self.objects.get(f"comparisons/{comparison_id}.json")
        except Exception:
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def similar_tickets(self, ticket_id: str, limit: int = 5) -> dict:
        """Past tickets nearest to this one, with their outcomes.

        Uses the same embedder + vector index as the duplicate
        check (the port both the in-memory index and pgvector
        implement), but reads further down the neighbour list and
        keeps everything over the similarity floor — where the
        duplicate check asks "is this the same ticket again?", this
        asks "what happened to tickets like this one?". A neighbour's
        recorded outcome (final category/queue, resolved) rides
        along; no outcome is shown as ``None``, never invented.
        """
        from .config import env_float

        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        floor = env_float("SIMILARITY_FLOOR", SIMILARITY_FLOOR)
        vector = self.embedder.embed(self._ticket_text(ticket_id, record))
        neighbours = self.index.nearest(
            vector, exclude_id=ticket_id, limit=max(limit * 3, limit)
        )
        items: list[dict] = []
        for other_id, similarity in neighbours:
            if float(similarity) < floor:
                continue
            other = self.store.get(other_id)
            if other is None:
                continue
            outcome = None
            if other.outcome is not None:
                outcome = {
                    "final_category": other.outcome.final_category,
                    "final_queue": other.outcome.final_queue,
                    "resolved": other.outcome.resolved,
                    "recorded_by": other.outcome.recorded_by,
                }
            items.append(
                {
                    "ticket_id": other.ticket_id,
                    "similarity": round(float(similarity), 4),
                    "subject": other.subject,
                    "category": other.category,
                    "queue": other.composition.queue,
                    "priority_band": other.composition.priority_band,
                    "status": other.status,
                    "outcome_recorded": other.outcome is not None,
                    "outcome": outcome,
                }
            )
            if len(items) >= limit:
                break
        return {"ticket_id": ticket_id, "floor": floor, "similar": items}

    def _ticket_text(self, ticket_id: str, record: TicketResult) -> str:
        """The text a ticket is embedded by: its stored input when
        the store kept it, the archived payload in production, and
        the record's subject as the last resort."""
        stored = self.store.get_input(ticket_id)
        if stored is not None:
            return f"{stored.subject}\n{stored.body}"
        try:
            raw = self.objects.get(f"tickets/{ticket_id}.json")
        except Exception:
            raw = None
        if raw:
            try:
                payload = json.loads(raw)
                return f"{payload.get('subject', '')}\n{payload.get('body', '')}"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return record.subject

    # -- outcomes + stats ----------------------------------------------------

    def record_outcome(self, ticket_id: str, outcome: AgentOutcome) -> TicketResult:
        """Attach the agent's ground truth to a handled ticket's record.

        The outcome is the label live calibration reads; recording one
        never changes the triage itself. Re-recording replaces the
        previous outcome (agents correct themselves too).
        """
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        outcome.recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.outcome = outcome
        self.store.save(record)
        self._publish(
            "outcome_recorded",
            ticket_id,
            {
                "final_category": outcome.final_category,
                "final_queue": outcome.final_queue,
                "resolved": outcome.resolved,
            },
        )
        return record

    def stats_overview(self) -> dict:
        from .stats import overview_stats

        return overview_stats(self.store.all_records())

    def stats_performance(self) -> dict:
        from .stats import performance_stats

        return performance_stats(self.store.all_records(), self.store.all_errors())

    def live_calibration(self) -> dict:
        from .livecal import live_calibration

        return live_calibration(self.store.all_records())

    # -- reads --------------------------------------------------------------

    def get_record(self, ticket_id: str) -> TicketResult | None:
        return self.store.get(ticket_id)

    def queue(self, filters: dict | None = None) -> list[dict]:
        return self.store.queue_summary(filters)

    def metrics(self) -> dict:
        return self.store.metrics()

    # -- internals ----------------------------------------------------------

    def _archive(self, ticket: TicketInput) -> None:
        """Archive the raw intake payload to the object store (best effort:
        an archive failure must not lose a completed triage)."""
        try:
            payload = json.dumps(ticket.model_dump(), indent=2).encode("utf-8")
            self.objects.put(f"tickets/{ticket.ticket_id}.json", payload, "application/json")
        except Exception:
            pass
