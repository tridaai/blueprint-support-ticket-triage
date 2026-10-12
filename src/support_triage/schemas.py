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
    # The governed version of the template this draft was written
    # under (template_governance.py): 1 for the bundled wording as
    # shipped, bumped on every governed edit. 0 is the honest
    # unrecorded shape — a draft composed before template versions
    # existed, or by a graph driven without the service — never a
    # guess at a version nobody stamped.
    template_version: int = 0
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


class HistoryFlag(BaseModel):
    """One computed history fact about the customer (dossier.py).

    A flag is arithmetic over the stored records, not a model
    opinion: the category decided three times in ninety days, the
    five contacts in thirty. It carries its evidence — the ticket
    ids and the window — so the claim can be checked against the
    dossier in one click. Flags are shown to agents and offered to
    the decision call as context; they never enter composition.
    """

    flag: str = ""  # repeat_issue | frequent_contact
    category: str = ""  # the repeated category (repeat_issue only)
    count: int = 0
    window_days: int = 0
    ticket_ids: list[str] = Field(default_factory=list)
    detail: str = ""


class CustomerHistoryInfo(BaseModel):
    """The customer history a run was triaged WITH (dossier.py).

    Computed by the service from the customer's PRIOR stored
    records, before this run existed, and recorded here so the
    record shows what history the decision call was offered: the
    counts, the flags, the last outcome, and — honestly, under the
    reopen idiom — what the backend did with the block
    (``context_use``: ``history_block`` when the LLM/Jev adapters
    received it fenced, ``ignored_offline`` when the deterministic
    backend accepted and deliberately unused it). The block itself
    is counts and categories composed by code — never another
    ticket's text. A first recorded contact carries
    ``context_provided=False``: there was no history to offer.
    """

    customer_key: str = ""  # threads.customer_key of this run's intake
    context_provided: bool = False  # prior records existed and a block was offered
    context_use: str = ""  # history_block | ignored_offline | "" when not provided
    prior_tickets: int = 0
    decided: int = 0
    reopened: int = 0
    outcomes_recorded: int = 0
    resolved_rate: Optional[float] = None  # withheld below the small-sample floor
    last_outcome_ticket_id: str = ""
    last_outcome_category: str = ""
    last_outcome_resolved: Optional[bool] = None
    flags: list[HistoryFlag] = Field(default_factory=list)


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
    # The overlap set (round 10), empty when the draw requested
    # none: {reviewers: [a, b], requested: {count|fraction},
    # count: k, run_ids: [...]} — the runs both named reviewers
    # double-grade, picked by the draw's own seeded stream
    # (reviews.draw_sample), so the pairing reproduces with the
    # sample and the report can measure reviewer-vs-reviewer
    # agreement over exactly the runs the record assigns.
    overlap: dict = Field(default_factory=dict)
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


class IncidentMember(BaseModel):
    """One run's membership of an incident (incidents.py).

    Membership lives on the INCIDENT's record, never on the run's:
    the member run's ``TicketResult`` is byte-identical whether or
    not it belongs to an incident, exactly as a QA verdict lives
    with its sample and never on the graded run. ``origin`` says
    how the run joined: ``proposal`` (it was in the cluster a lead
    confirmed) or ``addition`` (it arrived later and a lead added
    it from the incident's candidate list).
    """

    ticket_id: str = ""
    added_by: str = ""  # the lead who confirmed / added it
    added_at: str = ""  # ISO datetime, stamped by the service
    origin: str = ""  # proposal | addition


class IncidentDecline(BaseModel):
    """A lead's refusal of one addition candidate for an incident.

    Append-only on the incident record: a declined ticket is never
    proposed as an addition to THIS incident again (the watermark
    discipline — a grouping a human refused does not re-nag), while
    remaining fully eligible for any other incident or cluster.
    """

    ticket_id: str = ""
    by: str = ""
    reason: str = ""
    at: str = ""  # ISO datetime, stamped by the service


class Incident(BaseModel):
    """One confirmed incident: many customers, one underlying event.

    A cluster proposal (incidents.py) is a computed view; THIS is
    the stored record a named lead creates by confirming one. The
    incident is a lens and a working list over its member runs —
    it merges nothing, re-routes nothing, and closing it is a
    named act that leaves every member's own status, decision and
    outcome exactly where its own flow put them. ``signals``
    carries the proposal's evidence as it stood at confirmation,
    so what the lead saw when they said "yes, one event" stays on
    the record after the queue has moved on.
    """

    incident_id: str
    title: str = ""  # composed in code at proposal time; a lead may rename at confirmation
    product_area: str = ""
    category: str = ""
    status: str = "open"  # open | closed
    anchor_ticket_id: str = ""  # the earliest member — the text additions are matched against
    proposal_id: str = ""  # the deterministic fingerprint of the confirmed proposal
    signals: dict = Field(default_factory=dict)
    members: list[IncidentMember] = Field(default_factory=list)
    declined: list[IncidentDecline] = Field(default_factory=list)
    opened_by: str = ""
    opened_at: str = ""  # ISO datetime, stamped by the service
    closed_by: str = ""
    closed_at: str = ""
    close_reason: str = ""

    def member_ids(self) -> list[str]:
        return [member.ticket_id for member in self.members]


class ClusterDismissal(BaseModel):
    """A lead's dismissal of one cluster proposal, with the reason.

    The never-re-nag watermark: a proposal whose member set is
    exactly a dismissed set is suppressed on every later scan. A
    set that GAINS a member is a different set and may propose
    again — growth is new evidence, and the new proposal names the
    dismissal it outgrew. Dismissals live in their own store
    records, like incidents: they are human acts, and human acts
    are stored, audited, and never recomputed away.
    """

    member_ticket_ids: list[str] = Field(default_factory=list)  # sorted
    product_area: str = ""
    category: str = ""
    dismissed_by: str = ""
    reason: str = ""
    dismissed_at: str = ""  # ISO datetime, stamped by the service

    @property
    def member_set(self) -> frozenset:
        return frozenset(self.member_ticket_ids)


class ShiftHandoff(BaseModel):
    """One shift handoff between two named leads (handoffs.py).

    The desk's state at a shift boundary, composed in code from
    the desk's own views — the breach watch's band counts and top
    rows, the open incidents and pending cluster proposals, the
    per-queue SLA pressure, the QA review backlog — plus the one
    part a human authors: the outgoing lead's note. The composed
    figures live in ``sections`` exactly as the source views
    computed them at ``as_of``; accepting the handoff never edits
    them, and the acceptor's disputed items ride in
    ``exceptions`` as text beside the figures, not over them.

    A handoff is a stored record of its own, like a review sample
    or an incident — it is not part of any run's record, and its
    existence changes nothing about any run. Its lifecycle is the
    handshake: ``outstanding`` from creation until the addressed
    incoming lead accepts it (``accepted``, plainly or with
    exceptions), or a newer handoff supersedes it
    (``superseded`` — the record stays; only its claim to be the
    current page lapses).
    """

    handoff_id: str
    from_lead: str = ""  # the outgoing lead who composed it
    to_lead: str = ""  # the incoming lead it is addressed to
    as_of: str = ""  # the stated shift boundary the figures were composed at
    created_at: str = ""  # ISO datetime, stamped by the service
    note: str = ""  # the outgoing lead's free text — the only human-authored part
    sections: dict = Field(default_factory=dict)  # composed figures, by section
    status: str = "outstanding"  # outstanding | accepted | superseded
    accepted_by: str = ""
    accepted_at: str = ""  # ISO datetime, stamped by the service
    acceptance: str = ""  # accepted | accepted_with_exceptions
    exceptions: str = ""  # the acceptor's disputed items, verbatim


class TemplateAct(BaseModel):
    """One entry in a template's append-only history
    (template_governance.py): who did what to the template, when,
    and why. Acts are never edited or removed — a correction is a
    later act, exactly like an operator correction on a run."""

    action: str = ""  # authored | edited | promoted | retired
    actor: str = ""
    at: str = ""  # ISO datetime, stamped by the service
    reason: str = ""
    version: int = 0  # the record's version once this act stands


class TemplateRecord(BaseModel):
    """One governed template (template_governance.py).

    The template family is bundled code (templates.py): one body,
    keyed by product area, named ``how_to_first_response[<area>]``
    on every run that carries it. A record exists only for a
    template someone has governed — authored over the bundled
    wording, edited, promoted, retired — and from that moment the
    record IS the template: its state decides whether selection
    may use it, its text and guide are the words a draft is
    written from, and its ``version`` (bumped on every text edit)
    is stamped onto each draft it produces. A template with no
    record stands at the bundled default: live, version 1, the
    wording as shipped.

    The lifecycle is ``draft → live → retired``. Only a live
    template is ever selected. Editing the words of a live
    template returns it to draft — the words an agent sends never
    change silently underneath the approval that let them serve.
    Retired is terminal: the record stays as history, and its
    history (``TemplateAct`` entries) is append-only.
    """

    name: str  # how_to_first_response[<area>] — the name runs carry
    area: str = ""  # the product area the name is keyed by
    text: str = ""  # the draft body ({name} / {guide} placeholders)
    guide: str = ""  # where the documented answer lives, for the area
    state: str = "draft"  # draft | live | retired
    version: int = 1  # bumped on every text edit; acts don't bump it
    authored_by: str = ""  # who authored the current version's words
    created_at: str = ""  # ISO datetime, stamped by the service
    updated_at: str = ""  # ISO datetime, stamped by the service
    history: list[TemplateAct] = Field(default_factory=list)


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


class AssignmentEntry(BaseModel):
    """One reassignment of a waiting run, made from the breach watch.

    An append-only overlay (the corrections precedent): the run's
    current assignee is the LAST entry here — or, when the list is
    empty, the latest ``[assigned to …]`` mark the gate wrote into
    ``decision_reason`` (assignment.py's resolver reads both, in
    that order, so there is one answer to "whose is this?"). An
    entry is a lead's named act: who takes the run now, which lead
    moved it, and why — all three required, or the act is refused.
    Reassignment moves the run between people; it never touches
    the run's clock (``received_at`` is intake's, and the SLA
    target hangs off it), its composition, or its status.
    """

    assignee: str = ""
    by: str = ""  # the lead who made the reassignment
    reason: str = ""
    at: str = ""  # ISO datetime, stamped by the service


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
    # When the decision landed, stamped by the service at decision
    # time (attainment.py reads it). Before the stamp existed the
    # record kept every fact of the decision except its moment, so
    # decisions from that era carry "" here and belong to no
    # attainment period — the report counts them as its boundary
    # rather than guessing when they happened.
    decided_at: str = ""
    outcome: Optional[AgentOutcome] = None  # agent's ground truth, once recorded
    reopen: Optional[ReopenInfo] = None  # set when this run reopens an earlier one
    # Customer history (dossier.py): what the customer's prior
    # records said when this run was triaged, and what the decision
    # call did with the history block. Context for the human and
    # the model — never an input to composition.
    history: CustomerHistoryInfo = Field(default_factory=CustomerHistoryInfo)
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
    # Reassignments from the breach watch (watch.py): a lead's
    # named, reasoned acts moving the waiting run to another agent.
    # Append-only overlay; the effective assignee is the last entry
    # (assignment.current_assignee resolves it against the older
    # "[assigned to …]" note idiom). Moving people never moves the
    # run's clock, composition, or status.
    assignments: list[AssignmentEntry] = Field(default_factory=list)
    external_action_taken: bool = False  # invariant: always False
    trace: list[TraceStep] = Field(default_factory=list)
    telemetry: Optional[RunTelemetry] = None

    def answer(self, key: str) -> QuestionAnswer | None:
        for answer in self.answers:
            if answer.key == key:
                return answer
        return None
