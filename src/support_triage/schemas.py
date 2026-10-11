"""Data model for the triage pipeline.

Pydantic models shared by the graph, the service, the API, and the
record store. ``TicketResult`` is the full decision record — the thing
an agent reads at the gate — and the store persists it whole (JSONB
in PostgreSQL, a dict in the test double), so what the console shows
is exactly what was decided, distributions and all.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """One support ticket at intake. Synthetic data only in this repo."""

    ticket_id: str
    subject: str = ""
    body: str = ""
    sender_name: str = ""
    sender_email: str = ""
    channel: str = "email"  # email | chat | phone | social | api
    account_tier: str = ""  # free | pro | enterprise — context, never a decision input to hide behind
    received_at: str = ""  # ISO datetime
    previous_ticket_ids: list[str] = Field(default_factory=list)


class QuestionAnswer(BaseModel):
    """One question's validated answer, as recorded on the run."""

    key: str
    kind: str  # choice | score | verify
    value: Any = None  # str (choice) | float (score expected / verify P(true))
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


class DuplicateInfo(BaseModel):
    is_duplicate: bool = False
    similar_ticket_id: str = ""
    similarity: float = 0.0
    threshold: float = 0.0
    embedder: str = ""


class CompositionInfo(BaseModel):
    priority_score: float = 0.0
    priority_band: str = "low"
    contributions: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    outage_boost_applied: bool = False
    queue: str = "general_support"
    routing_reason: str = ""
    override: str = ""
    flags: dict[str, bool] = Field(default_factory=dict)


class GuardrailInfo(BaseModel):
    passed: bool = True
    requires_human_review: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)


class AutonomyInfo(BaseModel):
    eligible: bool = False
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class DraftInfo(BaseModel):
    """The optional template first-response draft. A draft for the
    agent to send or discard — this system never sends it."""

    text: str = ""
    template: str = ""
    note: str = "Template draft — written by code, not a model. An agent sends it; the system does not."


class RunTelemetry(BaseModel):
    backend: str = ""
    model: str = ""
    decision_latency_ms: float = 0.0
    latency_seconds: float = 0.0
    question_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class TraceStep(BaseModel):
    name: str
    title: str
    details: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class TicketResult(BaseModel):
    """The complete decision record for one ticket."""

    ticket_id: str
    status: str = "awaiting_agent"  # awaiting_agent | triaged | rejected
    subject: str = ""
    sender_name: str = ""
    channel: str = ""
    category: str = ""
    product_area: str = ""
    answers: list[QuestionAnswer] = Field(default_factory=list)
    composition: CompositionInfo = Field(default_factory=CompositionInfo)
    duplicate: DuplicateInfo = Field(default_factory=DuplicateInfo)
    injection_flags: list[str] = Field(default_factory=list)
    guardrails: GuardrailInfo = Field(default_factory=GuardrailInfo)
    autonomy: AutonomyInfo = Field(default_factory=AutonomyInfo)
    draft: Optional[DraftInfo] = None
    decision: str = ""  # agent decision once made: approved | rejected
    decision_by: str = ""
    decision_reason: str = ""
    external_action_taken: bool = False  # invariant: always False
    trace: list[TraceStep] = Field(default_factory=list)
    telemetry: Optional[RunTelemetry] = None

    def answer(self, key: str) -> QuestionAnswer | None:
        for answer in self.answers:
            if answer.key == key:
                return answer
        return None
