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
    # The reopen priority floor (composition.py): on a reopen run the
    # band is raised one step — the floor is part of composition, so
    # it is recorded like the outage boost, with the band the answers
    # alone would have earned kept beside it.
    reopen_floor_applied: bool = False
    band_before_floor: str = ""
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


class AgentOutcome(BaseModel):
    """The agent's final ground truth for a handled ticket.

    Recorded after the fact (``POST /api/runs/{id}/outcome``) by the
    agent who actually handled the ticket: where it really belonged,
    whether it resolved, and — where the agent can judge — the truth
    of the verify/score questions. This is the label set live
    calibration scores the recorded probabilities against; truths are
    keyed by question key (the API accepts agent-facing aliases and
    normalises them before they land here).
    """

    final_category: str = ""
    final_queue: str = ""
    resolved: bool = False
    recorded_by: str = ""
    recorded_at: str = ""  # ISO datetime, stamped by the service
    # The agent's own words for how it ended — free text, optional.
    # The workbench (workbench.py) shows it to the next agent facing
    # a similar ticket; it plays no part in calibration.
    resolution_summary: str = ""
    verify_truth: dict[str, bool] = Field(default_factory=dict)
    score_truth: dict[str, int] = Field(default_factory=dict)


class ReopenInfo(BaseModel):
    """This run is a reopen of an earlier run — the chain links.

    A reopen is a NEW run, never an edit of the old one: the prior
    run's record is untouched, and this block is the whole link —
    which run it follows, where the chain started, how many reopens
    the chain has had (this one included), the limit in force when
    it ran and whether it hit it, and who logged the customer's
    follow-up. ``context_use`` says, honestly, what the decision
    backend did with the prior run's record: the LLM/Jev adapters
    receive it as a fenced history block; the deterministic offline
    backend ignores it by design (its answers come from the new
    message alone), and the record says so rather than implying the
    history was read.
    """

    prior_run_id: str = ""
    root_ticket_id: str = ""
    reopen_count: int = 0  # this run's position in the chain (first reopen = 1)
    reopen_limit: int = 0  # the REOPEN_LIMIT in force when this run ran
    limit_hit: bool = False  # reopen_count reached the limit → human review
    reopened_by: str = ""  # the agent who logged the follow-up
    reopened_at: str = ""  # ISO datetime, stamped by the service
    context_provided: bool = False  # the prior run's record was offered as context
    context_use: str = ""  # history_block | ignored_offline
    template_suppressed: bool = True  # a reopen never gets the canned draft


class ThreadCorrection(BaseModel):
    """One operator correction to a run's computed thread grouping.

    Corrections are an OVERLAY (corrections.py): the computed
    ``thread_id`` on the record is never rewritten; a correction
    appends an entry here saying where the run's chain belongs
    instead, who moved it, and why. The effective thread a view
    shows is the last correction's ``to_thread_id`` — or the
    computed id when the list is empty. ``from_thread_id`` keeps the
    lineage honest: what the machine had computed (or a prior
    correction had set) is on the record, not lost. Corrections are
    applied to a whole reopen chain at once — a chain is never split.
    """

    action: str = ""  # attach | detach | merge
    from_thread_id: str = ""
    to_thread_id: str = ""
    by: str = ""  # the operator who made the correction
    reason: str = ""
    at: str = ""  # ISO datetime, stamped by the service


class DuplicateMark(BaseModel):
    """An operator's confirmed duplicate judgement on one run.

    The intake duplicate check (``DuplicateInfo``) is a machine
    guess — a similarity over a threshold. This is the human's word:
    this run IS the same customer asking the same thing again as run
    ``duplicate_of``. The mark lives on the duplicate's own record
    (the surviving run is untouched), and an un-mark fills the
    ``unmarked_*`` fields rather than deleting the entry — the audit
    trail keeps both halves. While a mark is active the run is
    excluded from calibration sampling and the drift window (a
    duplicate is not a second opinion about the backend) but still
    counts in volume stats (it WAS a ticket).
    """

    duplicate_of: str = ""
    marked_by: str = ""
    reason: str = ""
    marked_at: str = ""  # ISO datetime, stamped by the service
    unmarked_by: str = ""
    unmarked_at: str = ""
    unmark_reason: str = ""

    @property
    def active(self) -> bool:
        return not self.unmarked_at


class PlaybookCheck(BaseModel):
    """One check-off action on the run's category playbook.

    An append-only overlay (the corrections precedent): checking a
    step appends ``action="checked"``, unchecking appends
    ``"unchecked"`` — the effective state of a step is its LAST
    entry, and the whole trail stays on the record. Playbook
    progress is guidance bookkeeping for the human working the
    ticket: it never feeds composition, guardrails, calibration, or
    the run's state.
    """

    step_id: str = ""
    action: str = ""  # checked | unchecked
    by: str = ""  # the agent who acted
    at: str = ""  # ISO datetime, stamped by the service


class PlaybookNaMark(BaseModel):
    """The playbook set marked not-applicable for this run, or lifted.

    Append-only like the checks: ``action="marked"`` carries the
    agent's reason; ``"lifted"`` restores the steps. The effective
    state is the last entry. A marked set refuses step checks until
    it is lifted — working a playbook you have declared irrelevant
    is exactly the box-ticking the mark exists to prevent.
    """

    action: str = ""  # marked | lifted
    by: str = ""
    reason: str = ""  # required on "marked"; the lift may restate it
    at: str = ""  # ISO datetime, stamped by the service

    @property
    def active(self) -> bool:
        return self.action == "marked"


class WorkingNote(BaseModel):
    """The agent's saved internal working note for this run.

    Starts life as the deterministic draft :mod:`workbench`
    composes from the record; the agent edits it and saves it back.
    Internal, always: it is not the customer template draft
    (``DraftInfo``), it is never sent anywhere, and saving it
    changes nothing about the triage — it is the agent's own
    handwriting on the file.
    """

    text: str = ""
    saved_by: str = ""
    saved_at: str = ""  # ISO datetime, stamped by the service


class ReviewSample(BaseModel):
    """One QA review round: a seeded, stratified draw over decided runs.

    Calibration (livecal.py) measures outcomes — did the ticket
    resolve. QA review measures the DECISIONS: were the answers and
    the routing right, as judged by a human reviewer who was not
    handed a cherry-picked pile. The draw is made by
    :mod:`reviews` — stratified by queue and priority band, seeded,
    and reproducible — and the sample is a stored record in its own
    right (seed, strata counts, run ids, exclusions with names), so
    a review round can be audited and re-run exactly. A sample is
    NOT part of any run's record: reviewing never edits a triage.
    """

    sample_id: str
    seed: int = 0
    size_requested: int = 0
    created_by: str = ""
    created_at: str = ""  # ISO datetime, stamped by the service
    exclude_previously_reviewed: bool = True
    # The population the draw was made over, accounted in full:
    # decided runs in the store, the confirmed duplicates excluded
    # (named), the previously-reviewed runs excluded (named, when
    # the flag was on), and what was left eligible.
    population: dict = Field(default_factory=dict)
    # One row per (queue, band) stratum: eligible vs sampled counts.
    strata: list[dict] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)  # the draw, in draw order
    note: str = ""


class ReviewVerdict(BaseModel):
    """One reviewer's judgement of one sampled run.

    Append-only on the review record (reviews.py): a reviewer
    grades the run's triage per question family (agree / disagree)
    plus an overall verdict and a free-text note. Verdicts live
    with the sample, never on the run's own record — the triage a
    reviewer graded stays byte-identical under the grading. One
    verdict per (sample, run, reviewer): a second by the same
    reviewer is a conflict, not an edit.
    """

    sample_id: str = ""
    ticket_id: str = ""
    reviewer: str = ""
    families: dict[str, str] = Field(default_factory=dict)  # family -> agree | disagree
    overall: str = ""  # agree | disagree
    note: str = ""
    recorded_at: str = ""  # ISO datetime, stamped by the service


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
    received_at: str = ""  # the ticket's intake timestamp — the SLA clock's start
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
    outcome: Optional[AgentOutcome] = None  # agent's ground truth, once recorded
    reopen: Optional[ReopenInfo] = None  # set when this run reopens an earlier one
    # Conversation thread (threads.py): runs from one customer about
    # one issue share an id — "thr-<root ticket id>". A label,
    # assigned at intake; it merges and alters nothing.
    thread_id: str = ""
    # Operator corrections to that grouping (corrections.py): an
    # append-only overlay — thread_id above stays exactly what intake
    # computed, and views resolve the effective thread through the
    # last correction. Empty means the machine's grouping stands.
    thread_corrections: list[ThreadCorrection] = Field(default_factory=list)
    # Operator-confirmed duplicate (corrections.py): the audit trail
    # of mark/un-mark actions, plus the live annotation — the ticket
    # id this run is a confirmed duplicate of, "" when no mark is
    # active. The intake guess stays in `duplicate` above; this is
    # the human's confirmation of it (or of a pair the machine missed).
    duplicate_marks: list[DuplicateMark] = Field(default_factory=list)
    duplicate_of: str = ""
    # The agent's workbench (workbench.py): playbook check-offs and
    # not-applicable marks for the run's category, and the saved
    # internal working note. All overlays on the record like the
    # corrections above — the decision record itself is untouched.
    playbook_checks: list[PlaybookCheck] = Field(default_factory=list)
    playbook_na: list[PlaybookNaMark] = Field(default_factory=list)
    working_note: Optional[WorkingNote] = None
    external_action_taken: bool = False  # invariant: always False
    trace: list[TraceStep] = Field(default_factory=list)
    telemetry: Optional[RunTelemetry] = None

    def answer(self, key: str) -> QuestionAnswer | None:
        for answer in self.answers:
            if answer.key == key:
                return answer
        return None
