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
- answers queue/metrics/record reads from the store;
- reopens a handled ticket: the customer's follow-up becomes a new,
  linked run (the prior record is never mutated) carrying the prior
  run's record as decision context, and assigns every fresh run its
  conversation thread at intake (threads.py);
- records operator corrections (corrections.py): thread attach /
  detach / merge and confirmed-duplicate marks — append-only
  overlays on the records that change where a run sits (and whether
  it counts as a calibration sample), never what was decided on it;
- runs QA review (reviews.py): draws the seeded, stratified sample
  over decided runs, stores reviewers' per-family verdicts on the
  review record (never on the run's record), and serves the
  agreement report — the layer that grades the triage itself, where
  calibration only ever graded outcomes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from langgraph.types import Command
from pydantic import ValidationError

from . import corrections, playbooks, threads, webhooks, workbench
from .composition import reopen_limit_in_force
from .events import CollectingSink, EventBus, RunEvent
from .graph import AgentRun, build_graph
from .ports import Checkpointer, DecisionEngine, Embedder, EventSink, ObjectStore, Store, VectorIndex
from .schemas import (
    AgentOutcome,
    DuplicateMark,
    ReviewSample,
    ReviewVerdict,
    ThreadCorrection,
    TicketInput,
    TicketResult,
)
from .sla import format_timestamp, parse_timestamp

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

    def run_ticket(
        self,
        ticket: TicketInput,
        sink: EventSink | None = None,
        reopen: dict | None = None,
    ) -> AgentRun:
        """Run one ticket to the agent gate and persist the record.

        ``reopen`` is set only by :meth:`reopen_run`: the chain
        links, the limit in force, the prior run's record as context,
        and the thread the new run joins by construction. A normal
        run is assigned its conversation thread here, at intake,
        from the stored runs (threads.py).
        """
        sink = sink or CollectingSink()
        graph = build_graph(
            self.engine, self.embedder, self.index, self._get_checkpointer(), sink
        )
        config = {"configurable": {"thread_id": ticket.ticket_id}}
        if reopen is not None:
            thread_id = reopen.get("thread_id", "")
            prior_context = reopen.get("context", "")
        else:
            thread_id = self._assign_thread(ticket)
            prior_context = ""
        sink.emit(RunEvent(type="run_started", ticket_id=ticket.ticket_id))
        try:
            final = graph.invoke(
                {
                    "ticket": ticket.model_dump(),
                    "started_at": time.time(),
                    "trace": [],
                    "reopen": reopen or {},
                    "prior_context": prior_context,
                    "thread_id": thread_id,
                },
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

    def stats_drift(self, eval_per_question: dict | None) -> dict:
        from .drift import drift_report

        return drift_report(self.store.all_records(), eval_per_question)

    def assignment_suggestion(self, ticket_id: str) -> dict:
        """Who should probably take this ticket (assignment.py).

        A read end to end: the roster is loaded, the records are
        scored, a suggestion comes back. Assigning remains the
        human's POST, exactly as before.
        """
        from .assignment import suggest_assignment
        from .roster import load_roster

        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        return suggest_assignment(record, self.store.all_records(), load_roster())

    # -- reopens + threads ---------------------------------------------------

    def reopen_run(
        self,
        prior_ticket_id: str,
        *,
        message: str,
        actor: str,
        subject: str | None = None,
    ) -> AgentRun:
        """Reopen a handled ticket with the customer's follow-up.

        The follow-up becomes a NEW run, linked to the old one
        (``ReopenInfo`` on the new record); the prior record is
        never mutated. The new run re-enters the same pipeline with
        the code-level reopen consequences the graph applies: the
        prior run's record offered to the decision call as
        labelled history context, no template draft, the one-band
        priority floor, and the reopen-limit guardrail. The sender,
        channel and tier carry over from the prior intake; the
        follow-up message is the new run's body, screened at intake
        exactly like any other ticket text.
        """
        record = self.store.get(prior_ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {prior_ticket_id!r}")
        if not (message or "").strip():
            raise ValueError("a reopen needs the customer's follow-up message")
        prior_input = self._stored_input(prior_ticket_id)
        previous_count = record.reopen.reopen_count if record.reopen else 0
        count = previous_count + 1
        root = record.reopen.root_ticket_id if record.reopen else record.ticket_id
        base_id = f"{root}-R{count}"
        new_id = base_id
        suffix = 2
        while self.store.get(new_id) is not None:
            # Reopening a run that already has a reopen child (a
            # branch, not the chain's head): never overwrite the
            # existing run — take the next free id on the same base.
            new_id = f"{base_id}-{suffix}"
            suffix += 1
        now = format_timestamp(datetime.now(timezone.utc))
        new_ticket = TicketInput(
            ticket_id=new_id,
            subject=subject
            if subject is not None
            else (prior_input.subject if prior_input is not None else record.subject),
            body=message,
            sender_name=(
                prior_input.sender_name if prior_input is not None else record.sender_name
            ),
            sender_email=prior_input.sender_email if prior_input is not None else "",
            channel=prior_input.channel if prior_input is not None else record.channel,
            account_tier=prior_input.account_tier if prior_input is not None else "",
            received_at=now,
            previous_ticket_ids=[prior_ticket_id],
        )
        reopen = {
            "prior_run_id": prior_ticket_id,
            "root_ticket_id": root,
            "reopen_count": count,
            "reopen_limit": reopen_limit_in_force(),
            "reopened_by": actor,
            "reopened_at": now,
            "context": self._prior_context(record),
            "context_use": (
                "ignored_offline"
                if getattr(self.engine, "name", "") == "offline"
                else "history_block"
            ),
            # The chain joins the prior run's EFFECTIVE thread: if an
            # operator corrected the chain's grouping, the reopen
            # follows the correction, not the superseded computation.
            "thread_id": corrections.effective_thread_id(record),
        }
        return self.run_ticket(new_ticket, reopen=reopen)

    @staticmethod
    def _prior_context(record: TicketResult) -> str:
        """The prior run's record as context text for a reopen.

        Composed by code from the stored record — the answers with
        their confidences, the routing, the agent's decision and
        the recorded outcome. Facts only; the backends fence it as
        historical record.
        """

        def fmt(value) -> str:
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        lines = [
            f"Prior run {record.ticket_id}: category {record.category}, "
            f"product area {record.product_area}, routed to "
            f"{record.composition.queue} at priority "
            f"{record.composition.priority_band} (score {record.composition.priority_score}).",
            "Recorded answers: "
            + "; ".join(
                f"{a.key}={fmt(a.value)} (confidence {a.confidence:.2f})"
                for a in record.answers
            )
            + ".",
        ]
        if record.decision:
            lines.append(
                f"Agent decision on the prior run: {record.decision}"
                + (f" by {record.decision_by}" if record.decision_by else "")
                + "."
            )
        if record.outcome is not None:
            outcome = record.outcome
            lines.append(
                f"Recorded outcome: final category {outcome.final_category or '—'}, "
                f"final queue {outcome.final_queue or '—'}, "
                f"resolved: {'yes' if outcome.resolved else 'no'}."
            )
            if outcome.verify_truth:
                lines.append(
                    "Verify truth: "
                    + "; ".join(
                        f"{k}={v}" for k, v in sorted(outcome.verify_truth.items())
                    )
                    + "."
                )
            if outcome.score_truth:
                lines.append(
                    "Score truth: "
                    + "; ".join(
                        f"{k}={v}" for k, v in sorted(outcome.score_truth.items())
                    )
                    + "."
                )
        else:
            lines.append("No outcome was recorded for the prior run.")
        return "\n".join(lines)

    def _stored_input(self, ticket_id: str) -> TicketInput | None:
        """The raw intake for a stored run: the store's copy when it
        kept one, else the archived object-store payload, else None."""
        stored = self.store.get_input(ticket_id)
        if stored is not None:
            return stored
        try:
            raw = self.objects.get(f"tickets/{ticket_id}.json")
        except Exception:
            raw = None
        if raw:
            try:
                return TicketInput.model_validate(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
                pass
        return None

    def _assign_thread(self, ticket: TicketInput) -> str:
        """Thread assignment for a fresh run, over the stored runs."""
        candidates = []
        for record in self.store.all_records():
            stored = self._stored_input(record.ticket_id)
            candidates.append(
                {
                    "ticket_id": record.ticket_id,
                    "customer": threads.customer_key(
                        stored.sender_email if stored is not None else "",
                        record.sender_name,
                    ),
                    "subject": record.subject,
                    "body": stored.body if stored is not None else "",
                    "received_at": parse_timestamp(record.received_at),
                    "thread_id": record.thread_id,
                }
            )
        received = parse_timestamp(ticket.received_at) or datetime.now(timezone.utc)
        return threads.assign_thread(
            ticket_id=ticket.ticket_id,
            customer=threads.customer_key(ticket.sender_email, ticket.sender_name),
            subject=ticket.subject,
            body=ticket.body,
            received_at=received,
            candidates=candidates,
        )["thread_id"]

    def _thread_rows(self) -> list[dict]:
        rows = []
        for record in self.store.all_records():
            row = threads.run_row(record, self._stored_input(record.ticket_id))
            # The operator overlay (corrections.py): views group by
            # the EFFECTIVE thread and carry the grouping lineage;
            # the computed id stays visible beside it.
            row["thread_id"] = corrections.effective_thread_id(record)
            row["grouping"] = corrections.grouping_view(record)
            row["duplicate_of"] = record.duplicate_of
            rows.append(row)
        return rows

    def list_threads(self) -> dict:
        """Every conversation thread over the stored runs (threads.py)."""
        return {"threads": threads.thread_summaries(self._thread_rows())}

    def get_thread(self, thread_id: str) -> dict:
        """One thread whole: its summary state (open/reopened, latest
        outcome) plus the runs in order, each with its outcome."""
        rows = self._thread_rows()
        chain = threads.thread_chain(rows, thread_id)
        if chain["run_count"] == 0:
            raise KeyError(f"unknown thread {thread_id!r}")
        summary = next(
            s for s in threads.thread_summaries(rows) if s["thread_id"] == thread_id
        )
        return {**summary, **chain}

    def run_thread(self, ticket_id: str) -> dict:
        """The thread a run belongs to (its chain, with outcomes)."""
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        return self.get_thread(corrections.effective_thread_id(record))

    # -- operator corrections (threads + confirmed duplicates) --------------

    @staticmethod
    def _require_actor_reason(actor: str, reason: str) -> tuple[str, str]:
        actor = (actor or "").strip()
        reason = (reason or "").strip()
        if not actor:
            raise corrections.CorrectionError(
                "actor is required — a correction names who made it"
            )
        if not reason:
            raise corrections.CorrectionError(
                "reason is required — a correction states why the machine was wrong"
            )
        return actor, reason

    def _customer_key_for(self, record: TicketResult) -> str:
        stored = self._stored_input(record.ticket_id)
        return threads.customer_key(
            stored.sender_email if stored is not None else "",
            record.sender_name,
        )

    def _apply_thread_correction(
        self,
        chain: list[TicketResult],
        *,
        action: str,
        to_thread_id: str,
        actor: str,
        reason: str,
    ) -> None:
        """Append one correction entry to every run in a chain and
        save — the computed thread_id on each record is untouched."""
        now = format_timestamp(datetime.now(timezone.utc))
        for member in chain:
            member.thread_corrections.append(
                ThreadCorrection(
                    action=action,
                    from_thread_id=corrections.effective_thread_id(member),
                    to_thread_id=to_thread_id,
                    by=actor,
                    reason=reason,
                    at=now,
                )
            )
            self.store.save(member)

    def attach_run_to_thread(
        self, ticket_id: str, target_thread_id: str, *, actor: str, reason: str
    ) -> dict:
        """Move a run's whole reopen chain into another thread.

        The operator's word over the intake computation: the chain
        (never part of it) joins ``target_thread_id``, which must be
        an existing thread of the SAME customer — grouping never
        crosses customers, and neither do corrections. Returns the
        run's corrected thread view.
        """
        actor, reason = self._require_actor_reason(actor, reason)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        records = self.store.all_records()
        target_members = [
            r for r in records if corrections.effective_thread_id(r) == target_thread_id
        ]
        if not target_members:
            raise KeyError(f"unknown thread {target_thread_id!r}")
        current = corrections.effective_thread_id(record)
        if current == target_thread_id:
            raise corrections.CorrectionConflict(
                f"run {ticket_id} is already in thread {target_thread_id}"
            )
        chain = corrections.chain_of(records, record)
        customer = self._customer_key_for(record)
        for member in target_members:
            if self._customer_key_for(member) != customer:
                raise corrections.CorrectionError(
                    f"thread {target_thread_id} belongs to a different customer — "
                    "a correction cannot group two customers' runs"
                )
        self._apply_thread_correction(
            chain, action="attach", to_thread_id=target_thread_id,
            actor=actor, reason=reason,
        )
        return self.run_thread(ticket_id)

    def detach_run(self, ticket_id: str, *, actor: str, reason: str) -> dict:
        """Move a run's whole reopen chain out into its own thread.

        The detached chain gets an operator thread id
        (``thr-op-<chain root>``) so it can never collide with a
        computed ``thr-<ticket>`` id. Refused when the chain is
        already alone in its thread — there is nothing to correct.
        Returns the new thread's view.
        """
        actor, reason = self._require_actor_reason(actor, reason)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        records = self.store.all_records()
        chain = corrections.chain_of(records, record)
        current = corrections.effective_thread_id(record)
        others = [
            r
            for r in records
            if corrections.effective_thread_id(r) == current
            and corrections.chain_root_id(r) != corrections.chain_root_id(record)
        ]
        if not others:
            raise corrections.CorrectionConflict(
                f"run {ticket_id} is already alone in thread {current} — "
                "there is no grouping to correct"
            )
        new_thread_id = f"thr-op-{corrections.chain_root_id(record)}"
        self._apply_thread_correction(
            chain, action="detach", to_thread_id=new_thread_id,
            actor=actor, reason=reason,
        )
        return self.run_thread(ticket_id)

    def merge_threads(
        self, source_thread_id: str, target_thread_id: str, *, actor: str, reason: str
    ) -> dict:
        """Fold one thread into another: every run in the source
        thread is corrected into the target, which survives. Both
        threads must exist and belong to one customer. Returns the
        surviving thread's view."""
        actor, reason = self._require_actor_reason(actor, reason)
        if source_thread_id == target_thread_id:
            raise corrections.CorrectionConflict(
                f"thread {source_thread_id} cannot merge into itself"
            )
        records = self.store.all_records()
        source_members = [
            r for r in records if corrections.effective_thread_id(r) == source_thread_id
        ]
        if not source_members:
            raise KeyError(f"unknown thread {source_thread_id!r}")
        target_members = [
            r for r in records if corrections.effective_thread_id(r) == target_thread_id
        ]
        if not target_members:
            raise KeyError(f"unknown thread {target_thread_id!r}")
        customers = {self._customer_key_for(r) for r in source_members + target_members}
        if len(customers) > 1:
            raise corrections.CorrectionError(
                f"threads {source_thread_id} and {target_thread_id} belong to "
                "different customers — a correction cannot merge them"
            )
        now = format_timestamp(datetime.now(timezone.utc))
        for member in source_members:
            member.thread_corrections.append(
                ThreadCorrection(
                    action="merge",
                    from_thread_id=source_thread_id,
                    to_thread_id=target_thread_id,
                    by=actor,
                    reason=reason,
                    at=now,
                )
            )
            self.store.save(member)
        return self.get_thread(target_thread_id)

    def mark_duplicate(
        self, ticket_id: str, duplicate_of: str, *, actor: str, reason: str
    ) -> TicketResult:
        """Confirm, on the record, that one run duplicates another.

        The operator's confirmation of (or correction to) the intake
        duplicate guess. The mark lands on the duplicate's record
        only — the surviving run is untouched — and takes effect
        immediately: the run leaves the calibration samples and the
        drift window, with the exclusion accounted in both payloads.
        A run with a live mark cannot be marked again; a mark across
        customers is refused (a duplicate is the same customer
        asking again).
        """
        actor, reason = self._require_actor_reason(actor, reason)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        survivor = self.store.get(duplicate_of)
        if survivor is None:
            raise KeyError(f"unknown ticket {duplicate_of!r}")
        if duplicate_of == ticket_id:
            raise corrections.CorrectionError(
                "a run cannot be marked as a duplicate of itself"
            )
        existing = corrections.active_duplicate_mark(record)
        if existing is not None:
            raise corrections.CorrectionConflict(
                f"run {ticket_id} is already marked as a duplicate of "
                f"{existing.duplicate_of} (by {existing.marked_by or 'operator'}: "
                f"{existing.reason}) — unmark it before marking it again"
            )
        own_key = self._customer_key_for(record)
        other_key = self._customer_key_for(survivor)
        if not own_key or own_key != other_key:
            raise corrections.CorrectionError(
                f"runs {ticket_id} and {duplicate_of} are not from the same "
                "customer — a confirmed duplicate is the same customer asking "
                "the same thing again"
            )
        record.duplicate_marks.append(
            DuplicateMark(
                duplicate_of=duplicate_of,
                marked_by=actor,
                reason=reason,
                marked_at=format_timestamp(datetime.now(timezone.utc)),
            )
        )
        record.duplicate_of = duplicate_of
        self.store.save(record)
        return record

    def unmark_duplicate(
        self, ticket_id: str, *, actor: str, reason: str
    ) -> TicketResult:
        """Lift a confirmed-duplicate mark (also audited).

        The mark entry stays on the record with its un-mark half
        filled in; the run re-enters the calibration samples and the
        drift window from the next read.
        """
        actor, reason = self._require_actor_reason(actor, reason)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        mark = corrections.active_duplicate_mark(record)
        if mark is None:
            raise corrections.CorrectionConflict(
                f"run {ticket_id} has no confirmed-duplicate mark to lift"
            )
        mark.unmarked_by = actor
        mark.unmark_reason = reason
        mark.unmarked_at = format_timestamp(datetime.now(timezone.utc))
        record.duplicate_of = ""
        self.store.save(record)
        return record

    # -- the agent's workbench (knowledge, playbook, working note) ---------

    @staticmethod
    def _require_actor(actor: str) -> str:
        actor = (actor or "").strip()
        if not actor:
            raise workbench.WorkbenchError(
                "actor is required — a workbench action names who made it"
            )
        return actor

    def workbench_knowledge(self, ticket_id: str, limit: int = 10) -> dict:
        """How similar tickets actually ended (workbench.py).

        The similar-tickets machinery, joined to outcomes in code:
        neighbours with outcomes become knowledge entries with an
        aggregate over them; neighbours without stay counted as
        excluded, never silently dropped or silently included.
        """
        similar = self.similar_tickets(ticket_id, limit=limit)
        records = {r.ticket_id: r for r in self.store.all_records()}
        return {
            "ticket_id": ticket_id,
            **workbench.build_knowledge(similar["similar"], records),
        }

    def workbench_playbook(self, ticket_id: str) -> dict:
        """The run's category playbook with its check state."""
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        return {"ticket_id": ticket_id, **playbooks.playbook_view(record)}

    def working_note_draft(self, ticket_id: str) -> dict:
        """The deterministic internal note draft for one run.

        A composition, not a write: the draft exists only in this
        response until an agent saves their edited version through
        :meth:`save_working_note`.
        """
        from .sla import sla_clock

        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        thread = self.run_thread(ticket_id)
        return {
            "ticket_id": ticket_id,
            "draft": workbench.build_working_note(
                record,
                grouping=corrections.grouping_view(record),
                chain_size=thread.get("run_count", 1) or 1,
                sla=sla_clock(record),
            ),
            "saved": record.working_note.model_dump() if record.working_note else None,
            "note": (
                "internal draft composed by code from the record — "
                "not the customer template draft, and never sent"
            ),
        }

    def workbench(self, ticket_id: str) -> dict:
        """The whole workbench for one run: knowledge, playbook, note."""
        if self.store.get(ticket_id) is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        return {
            "ticket_id": ticket_id,
            "knowledge": self.workbench_knowledge(ticket_id),
            "playbook": self.workbench_playbook(ticket_id),
            "working_note": self.working_note_draft(ticket_id),
        }

    def _playbook_for_record(self, record: TicketResult) -> dict:
        book = playbooks.playbook_for(record.category)
        if book is None:
            raise workbench.WorkbenchError(
                f"no playbook authored for category '{record.category}' — "
                "there are no steps to check off"
            )
        return book

    def check_playbook_step(
        self, ticket_id: str, step_id: str, *, actor: str, checked: bool = True
    ) -> TicketResult:
        """Check (or uncheck) one playbook step on the run's record.

        Bookkeeping for the human: an append-only overlay entry.
        The triage — status, composition, decision, calibration —
        is untouched by construction: this method reads the record,
        appends one entry, and saves it back.
        """
        from .schemas import PlaybookCheck

        actor = self._require_actor(actor)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        book = self._playbook_for_record(record)
        if step_id not in playbooks.step_ids(book):
            raise workbench.WorkbenchError(
                f"unknown playbook step {step_id!r} for category '{record.category}'"
            )
        if playbooks.active_na_mark(record) is not None:
            raise workbench.WorkbenchConflict(
                f"run {ticket_id}'s playbook is marked not-applicable — "
                "lift the mark before checking steps"
            )
        view = playbooks.playbook_view(record)
        state = next(s for s in view["steps"] if s["id"] == step_id)
        if state["checked"] == checked:
            raise workbench.WorkbenchConflict(
                f"step {step_id} is already {'checked' if checked else 'unchecked'} "
                f"on run {ticket_id}"
            )
        record.playbook_checks.append(
            PlaybookCheck(
                step_id=step_id,
                action="checked" if checked else "unchecked",
                by=actor,
                at=format_timestamp(datetime.now(timezone.utc)),
            )
        )
        self.store.save(record)
        return record

    def mark_playbook_na(self, ticket_id: str, *, actor: str, reason: str) -> TicketResult:
        """Mark the run's playbook not-applicable, with the reason."""
        from .schemas import PlaybookNaMark

        actor = self._require_actor(actor)
        reason = (reason or "").strip()
        if not reason:
            raise workbench.WorkbenchError(
                "reason is required — a not-applicable mark states why the "
                "playbook does not fit this ticket"
            )
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        self._playbook_for_record(record)
        if playbooks.active_na_mark(record) is not None:
            raise workbench.WorkbenchConflict(
                f"run {ticket_id}'s playbook is already marked not-applicable"
            )
        record.playbook_na.append(
            PlaybookNaMark(
                action="marked",
                by=actor,
                reason=reason,
                at=format_timestamp(datetime.now(timezone.utc)),
            )
        )
        self.store.save(record)
        return record

    def lift_playbook_na(self, ticket_id: str, *, actor: str) -> TicketResult:
        """Lift a not-applicable mark (the mark entry stays, audited)."""
        from .schemas import PlaybookNaMark

        actor = self._require_actor(actor)
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        if playbooks.active_na_mark(record) is None:
            raise workbench.WorkbenchConflict(
                f"run {ticket_id}'s playbook has no not-applicable mark to lift"
            )
        record.playbook_na.append(
            PlaybookNaMark(
                action="lifted",
                by=actor,
                at=format_timestamp(datetime.now(timezone.utc)),
            )
        )
        self.store.save(record)
        return record

    def save_working_note(self, ticket_id: str, *, text: str, actor: str) -> TicketResult:
        """Save the agent's edited internal working note on the record.

        The saved note is the agent's text under the agent's name;
        saving replaces any earlier saved note (the export carries
        the live one). The triage itself is untouched, exactly as
        with the playbook overlay.
        """
        from .schemas import WorkingNote

        actor = self._require_actor(actor)
        text = (text or "").strip()
        if not text:
            raise workbench.WorkbenchError("the note text is empty — nothing to save")
        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        record.working_note = WorkingNote(
            text=text,
            saved_by=actor,
            saved_at=format_timestamp(datetime.now(timezone.utc)),
        )
        self.store.save(record)
        return record

    # -- QA review (samples, verdicts, agreement report) -------------------

    def create_review_sample(
        self,
        *,
        size: int,
        seed: int = 0,
        created_by: str = "",
        exclude_previously_reviewed: bool = True,
    ) -> ReviewSample:
        """Draw a QA review sample over the decided runs (reviews.py).

        Stratified by queue and priority band, seeded, reproducible;
        confirmed duplicates never draw, and previously-reviewed
        runs sit out unless the caller switches the exclusion off.
        The sample is stored as its own record — the draw, its
        strata, and every exclusion by name — so the round can be
        audited and re-run exactly. No run record is touched.
        """
        from . import reviews

        created_by = (created_by or "").strip()
        if not created_by:
            raise reviews.ReviewError(
                "created_by is required — a sample names who drew it"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise reviews.ReviewError(
                f"sample size must be a positive integer, got {size!r}"
            )
        records = self.store.all_records()
        run_ids, population, strata = reviews.draw_sample(
            records,
            self.store.all_review_verdicts(),
            size=size,
            seed=int(seed),
            exclude_previously_reviewed=bool(exclude_previously_reviewed),
        )
        if not run_ids:
            raise reviews.ReviewError(
                "no eligible runs to sample — the population is decided runs "
                "that are not confirmed duplicates"
                + (
                    " or previously reviewed"
                    if exclude_previously_reviewed
                    else ""
                )
                + f" (decided total: {population['decided_total']})"
            )
        existing = self.store.all_review_samples()
        sample_id = f"QAS-{len(existing) + 1:04d}"
        note = (
            f"drew {len(run_ids)} of {population['eligible']} eligible decided "
            f"runs (requested {size})"
        )
        if len(run_ids) < size:
            note += " — the request ran past the eligible population, so the sample is the population"
        sample = ReviewSample(
            sample_id=sample_id,
            seed=int(seed),
            size_requested=size,
            created_by=created_by,
            created_at=format_timestamp(datetime.now(timezone.utc)),
            exclude_previously_reviewed=bool(exclude_previously_reviewed),
            population=population,
            strata=strata,
            run_ids=run_ids,
            note=note,
        )
        self.store.save_review_sample(sample)
        self._publish(
            "review_sample_created",
            "",
            {"sample_id": sample_id, "runs": len(run_ids), "seed": sample.seed},
        )
        return sample

    def list_review_samples(self) -> list[ReviewSample]:
        return self.store.all_review_samples()

    def get_review_sample(self, sample_id: str) -> dict:
        """One sample with the verdicts recorded against it so far."""
        sample = self._review_sample_or_raise(sample_id)
        verdicts = [
            verdict.model_dump()
            for verdict in self.store.all_review_verdicts()
            if verdict.sample_id == sample_id
        ]
        return {
            "sample": sample.model_dump(),
            "verdicts": verdicts,
            "verdicted_runs": len({v["ticket_id"] for v in verdicts}),
        }

    def _review_sample_or_raise(self, sample_id: str) -> ReviewSample:
        for sample in self.store.all_review_samples():
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(f"unknown review sample {sample_id!r}")

    def record_review_verdict(
        self,
        sample_id: str,
        *,
        ticket_id: str,
        reviewer: str,
        families: dict,
        overall: str,
        note: str = "",
    ) -> ReviewVerdict:
        """Record one reviewer's grading of one sampled run.

        The verdict lives on the review record tied to the sample;
        the run's triage record is never opened for writing here —
        grading changes nothing about what was decided (the suite
        pins the record, the calibration samples, and the live
        calibration payload byte-identical across a verdict).
        """
        from . import reviews

        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise reviews.ReviewError(
                "reviewer is required — a verdict names who graded it"
            )
        sample = self._review_sample_or_raise(sample_id)
        if ticket_id not in sample.run_ids:
            raise reviews.ReviewError(
                f"run {ticket_id!r} is not in sample {sample_id} — a verdict "
                "grades a run the sample drew, never one it did not"
            )
        cleaned = reviews.validate_verdict_families(families)
        overall_clean = reviews.validate_overall(overall)
        for existing in self.store.all_review_verdicts():
            if (
                existing.sample_id == sample_id
                and existing.ticket_id == ticket_id
                and existing.reviewer == reviewer
            ):
                raise reviews.ReviewConflict(
                    f"{reviewer} already graded run {ticket_id} in sample "
                    f"{sample_id} (overall: {existing.overall}) — a verdict "
                    "is final; another reviewer may still grade the run"
                )
        verdict = ReviewVerdict(
            sample_id=sample_id,
            ticket_id=ticket_id,
            reviewer=reviewer,
            families=cleaned,
            overall=overall_clean,
            note=(note or "").strip(),
            recorded_at=format_timestamp(datetime.now(timezone.utc)),
        )
        self.store.save_review_verdict(verdict)
        self._publish(
            "review_verdict_recorded",
            ticket_id,
            {"sample_id": sample_id, "reviewer": reviewer, "overall": overall_clean},
        )
        return verdict

    def review_report(self, sample_ids: list[str] | None = None) -> dict:
        """The agreement report over one sample, a set, or all of
        them (reviews.py) — a read model; it recomputes nothing
        about any run and changes nothing anywhere."""
        from . import reviews

        all_samples = self.store.all_review_samples()
        if sample_ids:
            wanted = set(sample_ids)
            samples = [s for s in all_samples if s.sample_id in wanted]
            missing = wanted - {s.sample_id for s in samples}
            if missing:
                raise KeyError(f"unknown review sample(s) {sorted(missing)}")
        else:
            samples = all_samples
        records = {r.ticket_id: r for r in self.store.all_records()}
        return reviews.agreement_report(
            samples, self.store.all_review_verdicts(), records
        )

    def review_entries_for(self, ticket_id: str) -> list[dict]:
        """Every verdict recorded on one run, across all samples —
        the audit export's review block for the run."""
        from . import reviews

        return reviews.review_entries_for(ticket_id, self.store.all_review_verdicts())

    # -- policy replay + audit export (reads over the records) --------------

    def policy_replay(
        self,
        *,
        category_min_confidence: float | None = None,
        weights: dict | None = None,
        outage_boost: bool | None = None,
    ) -> dict:
        """What a hypothetical policy would have done to history
        (replay.py): the stored answers re-composed in code, the
        moved runs reported. No backend is called and nothing is
        written — validation errors raise ``ValueError``."""
        from .replay import replay_policy, validate_hypothetical

        hypothetical = validate_hypothetical(
            category_min_confidence=category_min_confidence,
            weights=weights,
            outage_boost=outage_boost,
        )
        return replay_policy(self.store.all_records(), hypothetical)

    def export_run(self, ticket_id: str, eval_per_question: dict | None = None) -> dict:
        """One run as a portable audit document (exports.py).

        Composed from the stored record at read time; the drift
        section carries the watch's current verdict on the questions
        this run answered (``eval_per_question`` is the eval pack's
        baseline, exactly as for the drift endpoint — without it the
        section says so rather than guessing).
        """
        from .drift import drift_report
        from .exports import build_export
        from .sla import sla_clock

        record = self.store.get(ticket_id)
        if record is None:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        report = drift_report(self.store.all_records(), eval_per_question)
        questions: dict = {}
        flagged: list[str] = []
        for answer in record.answers:
            row = report["questions"].get(answer.key)
            if row is None:
                continue
            questions[answer.key] = row
            if row.get("drift"):
                flagged.append(answer.key)
        drift = {
            "note": report["note"],
            "any_drift": report["any_drift"],
            "flagged_questions": flagged,
            "questions": questions,
        }
        entries = self.review_entries_for(ticket_id)
        return build_export(
            record,
            exported_at=format_timestamp(datetime.now(timezone.utc)),
            ticket_input=self.store.get_input(ticket_id),
            sla=sla_clock(record),
            drift=drift,
            review={"verdicts": entries} if entries else None,
        )

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
