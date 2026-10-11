"""The triage pipeline as a LangGraph state graph.

Shape (one pass, then a human gate)::

    intake → screen → duplicate → decide → compose → agent_gate ⏸ → record_decision

- ``intake`` normalises the ticket (whitespace, defaults).
- ``screen`` runs the prompt-injection screen over subject and body;
  flagged sentences are removed from what any model reads, and the
  surviving text is fenced as untrusted data.
- ``duplicate`` embeds the ticket and asks the vector index for the
  nearest recent ticket; similarity at or over the threshold (in
  code) links the ticket as a probable duplicate.
- ``decide`` makes the ONE decision call: the full question set
  against the screened state, answered with probability
  distributions by whichever engine sits behind the port.
- ``compose`` turns distributions into consequences in code —
  priority formula, routing matrix + overrides, guardrails, autonomy
  recommendation, the rare template draft — and assembles the
  decision record. On a reopen run (service-set state), compose also
  applies the reopen consequences: the decision call receives the
  prior run's record as port context, the priority band is floored
  one step up, the template draft is suppressed in code, and the
  reopen-limit guardrail can force human review. On every run the
  decision call also receives the customer-history block the
  service composed from the customer's prior records (dossier.py)
  as port context, and compose records the history summary on the
  run — context for the call and the human, never an input to the
  formula.
- ``agent_gate`` pauses the graph (LangGraph ``interrupt``) with the
  record saved and status ``awaiting_agent``. Nothing has been sent
  to anyone; nothing ever is, by this system.
- ``record_decision`` applies the agent's approve/reject on resume.

Every node is timed and traced, and structured events flow to the
run's :class:`~support_triage.ports.EventSink`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import composition as comp
from . import guardrails as guard
from . import screening
from .config import env_float
from .decisions import DecisionSet
from .events import NullSink, RunEvent
from .ports import DecisionEngine, Embedder, EventSink, VectorIndex
from .questions import triage_questions
from .sla import format_timestamp
from .schemas import (
    AutonomyInfo,
    CompositionInfo,
    CustomerHistoryInfo,
    DraftInfo,
    DuplicateInfo,
    GuardrailInfo,
    HistoryFlag,
    QuestionAnswer,
    ReopenInfo,
    RunTelemetry,
    TicketInput,
    TicketResult,
    TraceStep,
)
from .templates import build_template_draft

DUPLICATE_THRESHOLD_DEFAULT = 0.85

_NODE_TITLES = {
    "intake": "Intake — ticket normalised",
    "screen": "Untrusted-text screen — injection patterns flagged and fenced",
    "duplicate": "Duplicate check — nearest recent ticket compared in code",
    "decide": "Decision call — eleven typed questions, one call",
    "compose": "Composition — priority, routing, guardrails (all in code)",
    "agent_gate": "Agent gate — the record waits for a human",
    "record_decision": "Agent decision recorded",
}


class TriageState(TypedDict, total=False):
    ticket: dict
    state_text: str
    injection_flags: list[str]
    duplicate: dict
    decisions: dict
    decision_meta: dict
    result: dict
    decision: dict
    trace: list[dict]
    started_at: float
    # Reopen runs only (set by the service): the chain links, the
    # limit in force, and the prior run's record as context text for
    # the decide node. Absent/empty on a first-pass run.
    reopen: dict
    prior_context: str
    # Customer history (set by the service on every run, dossier.py):
    # the history block composed from the customer's prior records —
    # offered to the decide node alongside any reopen context — and
    # the summary dict the compose node records on the run. Absent
    # when the graph is driven without the service.
    history_context: str
    history: dict
    # The conversation thread this run belongs to (threads.py),
    # assigned by the service at intake.
    thread_id: str


@dataclass
class AgentRun:
    """The outcome of starting a run: the record + whether it waits."""

    result: TicketResult
    awaiting_agent: bool


def _node(state: TriageState, name: str, sink: EventSink, fn) -> dict:
    """Run one node with timing, tracing, and events (the house wrapper)."""
    ticket_id = state["ticket"]["ticket_id"]
    sink.emit(RunEvent(type="node_started", ticket_id=ticket_id, node=name))
    started = time.perf_counter()
    update = fn(state)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    details = update.pop("_details", []) if isinstance(update, dict) else []
    step = TraceStep(
        name=name, title=_NODE_TITLES.get(name, name), details=details, duration_ms=duration_ms
    )
    trace = list(state.get("trace", []))
    trace.append(step.model_dump())
    update["trace"] = trace
    if isinstance(update.get("result"), dict):
        # The record carries its own trace: keep it in step with the
        # graph state's, including the step that produced it.
        merged = dict(update["result"])
        merged["trace"] = trace
        update["result"] = merged
    sink.emit(
        RunEvent(type="node_finished", ticket_id=ticket_id, node=name, duration_ms=duration_ms)
    )
    return update


def build_graph(
    engine: DecisionEngine,
    embedder: Embedder,
    index: VectorIndex,
    checkpointer,
    sink: EventSink | None = None,
):
    """Compile the triage graph with its collaborators bound."""
    sink = sink or NullSink()
    questions = triage_questions()
    question_keys = [q.key for q in questions]

    def intake(state: TriageState) -> dict:
        ticket = TicketInput.model_validate(state["ticket"])
        ticket.subject = " ".join(ticket.subject.split())
        ticket.body = ticket.body.strip()
        return {
            "ticket": ticket.model_dump(),
            "_details": [
                f"subject: {ticket.subject[:80] or '(none)'}",
                f"channel: {ticket.channel}; sender: {ticket.sender_name or ticket.sender_email or 'unknown'}",
            ],
        }

    def screen(state: TriageState) -> dict:
        ticket = TicketInput.model_validate(state["ticket"])
        flags = screening.screen_text("subject", ticket.subject) + screening.screen_text(
            "body", ticket.body
        )
        clean_body = screening.sanitize_text(ticket.body)
        header = [
            f"Subject: {ticket.subject}",
            f"Channel: {ticket.channel}",
        ]
        if ticket.account_tier:
            header.append(f"Account tier: {ticket.account_tier}")
        state_text = "\n".join(header) + "\n" + screening.fence(clean_body)
        return {
            "injection_flags": flags,
            "state_text": state_text,
            "_details": (
                [f"{len(flags)} injection pattern(s) flagged and removed from model state"]
                if flags
                else ["no injection patterns found"]
            ),
        }

    def duplicate(state: TriageState) -> dict:
        ticket = TicketInput.model_validate(state["ticket"])
        threshold = env_float("DUPLICATE_SIMILARITY_THRESHOLD", DUPLICATE_THRESHOLD_DEFAULT)
        text = f"{ticket.subject}\n{ticket.body}"
        try:
            vector = embedder.embed(text)
            neighbours = index.nearest(vector, exclude_id=ticket.ticket_id, limit=1)
            index.add(ticket.ticket_id, vector, embedder.model)
        except Exception:
            # A failed duplicate check degrades to "no link found" —
            # it must never block triage. (Unlike a decision failure:
            # a missing duplicate link is recoverable by an agent.)
            neighbours = []
        info = DuplicateInfo(threshold=threshold, embedder=embedder.model)
        if neighbours:
            other_id, similarity = neighbours[0]
            info.similar_ticket_id = other_id
            info.similarity = round(float(similarity), 4)
            info.is_duplicate = float(similarity) >= threshold
        if info.is_duplicate:
            sink.emit(
                RunEvent(
                    type="duplicate_flagged",
                    ticket_id=ticket.ticket_id,
                    detail={"similar_ticket_id": info.similar_ticket_id, "similarity": info.similarity},
                )
            )
        return {
            "duplicate": info.model_dump(),
            "_details": [
                f"nearest: {info.similar_ticket_id or 'none'} "
                f"(similarity {info.similarity:.3f}, threshold {threshold})"
                + (" — probable duplicate" if info.is_duplicate else "")
            ],
        }

    def decide(state: TriageState) -> dict:
        ticket = TicketInput.model_validate(state["ticket"])
        # The port's context argument carries history, never new
        # customer text: on a reopen, the prior run's record; on
        # every run, the customer-history block composed in code
        # from the customer's prior records (dossier.py — counts
        # and categories only). Both ride together when both exist.
        # Fenced history for the LLM/Jev adapters, ignored by the
        # offline heuristics; state_text itself is still only the
        # new (screened) customer message.
        context = "\n\n".join(
            part
            for part in (
                state.get("prior_context", ""),
                state.get("history_context", ""),
            )
            if part
        )
        decisions = engine.decide(state["state_text"], questions, context=context)
        sink.emit(
            RunEvent(
                type="decision_made",
                ticket_id=ticket.ticket_id,
                detail={
                    "backend": decisions.backend,
                    "model": decisions.model,
                    "latency_ms": decisions.latency_ms,
                    "question_count": len(questions),
                },
            )
        )
        return {
            "decisions": decisions.as_dict(),
            "decision_meta": {
                "backend": decisions.backend,
                "model": decisions.model,
                "latency_ms": decisions.latency_ms,
                "usage": decisions.usage,
            },
            "_details": [
                f"backend: {decisions.backend} ({decisions.model})",
                f"decision latency: {decisions.latency_ms} ms for {len(questions)} questions",
                f"category: {decisions.choice('category')} (confidence {decisions.confidence('category'):.2f})",
            ],
        }

    def compose(state: TriageState) -> dict:
        ticket = TicketInput.model_validate(state["ticket"])
        meta = state["decision_meta"]
        decisions = DecisionSet(
            results=[
                _result_from_dict(key, value)
                for key, value in state["decisions"].items()
            ],
            backend=meta["backend"],
            model=meta["model"],
            latency_ms=meta["latency_ms"],
            usage=meta.get("usage", {}),
        )
        reopen = state.get("reopen") or {}
        is_reopen = bool(reopen)
        composed = comp.compose(decisions, reopen=is_reopen)
        verdict = guard.run_guardrails(
            decisions,
            question_keys,
            reopen=(int(reopen["reopen_count"]), int(reopen["reopen_limit"]))
            if is_reopen
            else None,
        )
        duplicate = DuplicateInfo.model_validate(state["duplicate"])
        autonomy = comp.recommend_autonomy(
            decisions,
            composed,
            guardrails_passed=verdict.passed,
            requires_human_review=verdict.requires_human_review,
            is_duplicate=duplicate.is_duplicate,
            is_reopen=is_reopen,
        )
        draft = build_template_draft(
            decisions,
            composed,
            verdict,
            sender_name=ticket.sender_name,
            is_duplicate=duplicate.is_duplicate,
            is_reopen=is_reopen,
        )
        usage = meta.get("usage", {})
        telemetry = RunTelemetry(
            backend=meta["backend"],
            model=meta["model"],
            decision_latency_ms=meta["latency_ms"],
            latency_seconds=round(time.time() - state.get("started_at", time.time()), 3),
            question_count=len(questions),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            estimated_cost_usd=usage.get("estimated_cost_usd"),
        )
        received_at = ticket.received_at or format_timestamp(
            datetime.fromtimestamp(state.get("started_at", time.time()), timezone.utc)
        )
        # Customer history (dossier.py): the summary the service
        # computed over the customer's prior records, plus what this
        # backend did with the block — the reopen idiom, stated on
        # the record either way. History informs; composition below
        # never reads it.
        history_state = state.get("history") or {}
        history_offered = bool(state.get("history_context"))
        last_outcome = history_state.get("last_outcome") or {}
        history_info = CustomerHistoryInfo(
            customer_key=str(history_state.get("customer_key", "")),
            context_provided=history_offered,
            context_use=(
                ""
                if not history_offered
                else "ignored_offline"
                if meta["backend"] == "offline"
                else "history_block"
            ),
            prior_tickets=int(history_state.get("prior_tickets", 0)),
            decided=int(history_state.get("decided", 0)),
            reopened=int(history_state.get("reopened", 0)),
            outcomes_recorded=int(history_state.get("outcomes_recorded", 0)),
            resolved_rate=history_state.get("resolved_rate"),
            last_outcome_ticket_id=str(last_outcome.get("ticket_id", "")),
            last_outcome_category=str(last_outcome.get("final_category", "")),
            last_outcome_resolved=(
                bool(last_outcome["resolved"]) if last_outcome else None
            ),
            flags=[
                HistoryFlag.model_validate(flag)
                for flag in history_state.get("flags", [])
            ],
        )
        result = TicketResult(
            ticket_id=ticket.ticket_id,
            status="awaiting_agent",
            subject=ticket.subject,
            sender_name=ticket.sender_name,
            channel=ticket.channel,
            received_at=received_at,
            category=decisions.choice("category"),
            product_area=decisions.choice("product_area"),
            answers=[
                QuestionAnswer(
                    key=r.key,
                    kind=r.kind,
                    value=r.value,
                    probabilities=r.probabilities,
                    confidence=r.confidence,
                )
                for r in decisions.results
            ],
            composition=CompositionInfo(**composed.as_dict()),
            duplicate=duplicate,
            injection_flags=state.get("injection_flags", []),
            guardrails=GuardrailInfo(**verdict.as_dict()),
            autonomy=AutonomyInfo(**autonomy.as_dict()),
            draft=draft,
            reopen=(
                ReopenInfo(
                    prior_run_id=str(reopen["prior_run_id"]),
                    root_ticket_id=str(reopen["root_ticket_id"]),
                    reopen_count=int(reopen["reopen_count"]),
                    reopen_limit=int(reopen["reopen_limit"]),
                    limit_hit=int(reopen["reopen_count"]) >= int(reopen["reopen_limit"]),
                    reopened_by=str(reopen.get("reopened_by", "")),
                    reopened_at=str(reopen.get("reopened_at", "")),
                    context_provided=bool(state.get("prior_context")),
                    context_use=str(reopen.get("context_use", "")),
                    template_suppressed=True,
                )
                if is_reopen
                else None
            ),
            thread_id=state.get("thread_id", ""),
            history=history_info,
            external_action_taken=False,
            telemetry=telemetry,
        )
        sink.emit(
            RunEvent(
                type="guardrail_verdict",
                ticket_id=ticket.ticket_id,
                detail={
                    "passed": verdict.passed,
                    "requires_human_review": verdict.requires_human_review,
                },
            )
        )
        details = [
            f"priority: {composed.priority_score} ({composed.priority_band})",
            f"queue: {composed.queue} — {composed.routing_reason}",
            f"guardrails: {'passed' if verdict.passed else 'FAILED'}"
            + ("; human review forced" if verdict.requires_human_review else ""),
            f"autonomy recommendation: {'eligible' if autonomy.eligible else 'not eligible'}"
            " (advisory only — never acted on)",
            "status: awaiting_agent — nothing sent to the customer",
        ]
        if history_state:
            if history_info.context_provided:
                flag_names = ", ".join(
                    flag.flag for flag in history_info.flags
                )
                details.append(
                    f"customer history: {history_info.prior_tickets} prior "
                    f"ticket(s), {history_info.decided} decided"
                    + (f"; flags: {flag_names}" if flag_names else "; no flags")
                    + f"; context: {history_info.context_use} "
                    "(history informs — it never enters composition)"
                )
            else:
                details.append(
                    "customer history: first recorded contact — no prior "
                    "tickets, no history context offered"
                )
        if is_reopen:
            details.append(
                f"reopen #{reopen['reopen_count']} of {reopen['prior_run_id']} "
                f"(chain root {reopen['root_ticket_id']}, limit {reopen['reopen_limit']})"
            )
            details.append(
                f"reopen floor: band raised {composed.band_before_floor} → "
                f"{composed.priority_band}"
                if composed.band_before_floor != composed.priority_band
                else "reopen floor: band already critical — floor applied, no raise possible"
            )
            details.append("template draft: suppressed (reopen runs never get the canned draft)")
        return {
            "result": result.model_dump(),
            "_details": details,
        }

    def agent_gate(state: TriageState) -> dict:
        outcome = interrupt(
            {
                "ticket_id": state["ticket"]["ticket_id"],
                "status": "awaiting_agent",
                "prompt": "Approve the triage (optionally assign), or reject it.",
            }
        )
        decision = outcome if isinstance(outcome, dict) else {"decision": str(outcome)}
        return {"decision": decision, "_details": ["agent decision received"]}

    def record_decision(state: TriageState) -> dict:
        result = TicketResult.model_validate(state["result"])
        decision = state.get("decision", {})
        verdict = str(decision.get("decision", "")).strip().lower()
        result.decision = verdict
        result.decision_by = str(decision.get("by", ""))
        result.decision_reason = str(decision.get("reason", ""))
        assignee = str(decision.get("assignee", "")).strip()
        if assignee:
            result.decision_reason = (
                f"{result.decision_reason} [assigned to {assignee}]".strip()
            )
        result.status = "triaged" if verdict == "approved" else "rejected"
        return {
            "result": result.model_dump(),
            "_details": [
                f"decision: {verdict} by {result.decision_by or 'agent'}",
                f"status: {result.status}",
            ],
        }

    builder = StateGraph(TriageState)
    builder.add_node("intake", lambda s: _node(s, "intake", sink, intake))
    builder.add_node("screen", lambda s: _node(s, "screen", sink, screen))
    builder.add_node("duplicate", lambda s: _node(s, "duplicate", sink, duplicate))
    builder.add_node("decide", lambda s: _node(s, "decide", sink, decide))
    builder.add_node("compose", lambda s: _node(s, "compose", sink, compose))
    builder.add_node("agent_gate", lambda s: _node(s, "agent_gate", sink, agent_gate))
    builder.add_node("record_decision", lambda s: _node(s, "record_decision", sink, record_decision))
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "screen")
    builder.add_edge("screen", "duplicate")
    builder.add_edge("duplicate", "decide")
    builder.add_edge("decide", "compose")
    builder.add_edge("compose", "agent_gate")
    builder.add_edge("agent_gate", "record_decision")
    builder.add_edge("record_decision", END)
    return builder.compile(checkpointer=checkpointer)


def _result_from_dict(key: str, data: dict):
    from .decisions import QuestionResult

    return QuestionResult(
        key=key,
        kind=data["kind"],
        value=data["value"],
        probabilities=data["probabilities"],
        confidence=data["confidence"],
    )
