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
  decision record.
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
    DraftInfo,
    DuplicateInfo,
    GuardrailInfo,
    QuestionAnswer,
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
        decisions = engine.decide(state["state_text"], questions)
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
        composed = comp.compose(decisions)
        verdict = guard.run_guardrails(decisions, question_keys)
        duplicate = DuplicateInfo.model_validate(state["duplicate"])
        autonomy = comp.recommend_autonomy(
            decisions,
            composed,
            guardrails_passed=verdict.passed,
            requires_human_review=verdict.requires_human_review,
            is_duplicate=duplicate.is_duplicate,
        )
        draft = build_template_draft(
            decisions,
            composed,
            verdict,
            sender_name=ticket.sender_name,
            is_duplicate=duplicate.is_duplicate,
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
        return {
            "result": result.model_dump(),
            "_details": [
                f"priority: {composed.priority_score} ({composed.priority_band})",
                f"queue: {composed.queue} — {composed.routing_reason}",
                f"guardrails: {'passed' if verdict.passed else 'FAILED'}"
                + ("; human review forced" if verdict.requires_human_review else ""),
                f"autonomy recommendation: {'eligible' if autonomy.eligible else 'not eligible'}"
                " (advisory only — never acted on)",
                "status: awaiting_agent — nothing sent to the customer",
            ],
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
