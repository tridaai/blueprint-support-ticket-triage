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

from langgraph.types import Command

from .events import CollectingSink, RunEvent
from .graph import AgentRun, build_graph
from .ports import Checkpointer, DecisionEngine, Embedder, EventSink, ObjectStore, Store, VectorIndex
from .schemas import TicketInput, TicketResult


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
        return AgentRun(result=result, awaiting_agent=awaiting)

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

    # -- reads --------------------------------------------------------------

    def get_record(self, ticket_id: str) -> TicketResult | None:
        return self.store.get(ticket_id)

    def queue(self) -> list[dict]:
        return self.store.queue_summary()

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
