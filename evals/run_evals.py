"""Golden routing eval: known tickets in, routing truth checked.

Seventy synthetic tickets with authored expectations — category,
queue, priority band, and the five safety/engineering verify flags.
(36 through round 2; round 3 added multi-issue tickets, three more
injection phrasings, a calm ticket that must stay low, a repeat
outage follow-up, and an abuse-plus-legal combination; round 6 added
the reopen cases — a reopened how_to that must not get a template
draft, a reopen whose band rises by the priority floor, a chain
that hits the reopen limit and goes to human review — and a thread
near-miss that must stay in its own thread; the operator-correction
round added the correction cases — a computed thread pair an
operator splits, a duplicate an operator confirms into the
calibration accounting, and the correction refusals; the workbench
round added the workbench cases — resolution knowledge over seeded
neighbours with outcomes, the small-sample withholding, the
playbook overlay actions and refusals, and the working-note draft
carrying the routing reason verbatim; the QA-review round added the
review cases — a seeded world of decided runs whose stratified
sample is asserted exactly, verdicts graded against the agreement
report's math including the withheld rate below the verdict floor,
and the records and calibration payload pinned byte-identical
across grading; the customer-dossier round added the dossier
cases — a staged customer history whose history flags fire with
exact evidence both at triage (on the new run's record) and in
the dossier read model, a below-threshold customer with neither
flag and the resolved rate honestly withheld, and an operator
detach that moves the ticket between the dossier's thread rows,
the dossier itself pinned as a pure projection; the inter-rater
round added the paired-grading cases — an overlap draw asserted
exactly with its assignment, a third reviewer refused off the
overlap set while the open line holds off it, and a twelve-run
double-graded world whose per-family Cohen's kappas are asserted
to the decimal, including the no-variance family whose kappa is
honestly null; the template-effectiveness round added the
template cases — a staged world of template-earning tickets whose
per-template resolution / reopen / escalation rates are asserted
from hand-counted outcomes (one template flagged needs_review,
one clean, one below the use floor with its rates withheld, and a
pre-link record with an unnamed draft counted in the totals but
attributed to no template), and a firewall case pinning that a
flagged template's tickets compose byte-identically to a fresh
service and still earn their draft; the breach-watch round added
the watch cases — a staged queue at a fixed instant whose ranked
order and bands are hand-derived from each run's own SLA target,
a plain run outranking a feared outage purely on the clock (the
model's sla_breach_risk shown, never sorted by), the gate-note
assignee read off the watch, and the lead's reassignment asserted
as an overlay: the record byte-identical apart from its
assignment trail, the clock unmoved, the refusals named; the
shift-handoff round added the handoff case — a staged desk at a
fixed boundary (a confirmed cluster, a breached outage, a QA
sample with grading owed) whose composed page is asserted
section by section against hand-derived figures, then the
handshake played out: self-acceptance refused, acceptance with
exceptions storing the dispute and editing no figure, a newer
page superseding an unaccepted one, and every run record
byte-identical across the whole lifecycle; the template-
governance round added the governance case — a template's full
lifecycle played against staged runs: authored words standing
in draft and drafting nothing, self-promotion refused by name,
a second pair of eyes letting the words serve, the effectiveness
figures decaying past the review margin over the retirement
floor until the watch proposes retirement, the due template
still serving (a proposal is not an act), a named retirement
ending its drafting, an edit on a live template dropping it
back to draft at a bumped version until re-promoted — and no
run record written by any of it.)
Each case runs the full pipeline (screen → duplicate → one decision
call → composition → guardrails) through a fresh service, on the
deterministic offline backend by default, so the gate is hermetic:
``uv run python evals/run_evals.py`` exits non-zero below the gate.

These expectations encode intended behaviour of the pipeline as
designed — a regression gate, not an accuracy claim about any model.

Results are written to ``evals/results.json`` and to the packaged
``src/support_triage/data/eval_results.json`` the console reads.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_triage import corrections  # noqa: E402
from support_triage.backends import OfflineDecisionEngine  # noqa: E402
from support_triage.embeddings import HashingEmbedder, InMemoryVectorIndex  # noqa: E402
from support_triage.livecal import live_calibration, live_samples  # noqa: E402
from support_triage.objectstore import InMemoryObjectStore  # noqa: E402
from support_triage.schemas import AgentOutcome, TicketInput  # noqa: E402
from support_triage.service import TriageService  # noqa: E402
from support_triage.stats import overview_stats  # noqa: E402
from support_triage.store import InMemoryStore  # noqa: E402
from support_triage.threads import effective_thread_id  # noqa: E402

GATE = 0.95
VERIFY_KEYS = ("contains_pii", "is_abusive", "mentions_legal", "is_outage_report", "needs_engineering")
FLAG_THRESHOLD = 0.5


def load_cases() -> list[dict]:
    path = Path(__file__).resolve().parent / "golden.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fresh_service(engine=None) -> TriageService:
    return TriageService(
        engine=engine or OfflineDecisionEngine(),
        embedder=HashingEmbedder(),
        index=InMemoryVectorIndex(),
        store=InMemoryStore(),
        objects=InMemoryObjectStore(),
    )


def run_case(service: TriageService, case: dict):
    """Run one golden case (plus its `previous` ticket, when present).

    Returns ``(result, chain_failures)``: reopen cases run a chain —
    the ticket, its recorded outcome when the case carries one, then
    each follow-up in ``reopens`` — and the returned result is the
    FINAL reopened run, which is what ``expected`` describes. Chain
    premises that only the chain can judge (the original run really
    earned its template draft, so the reopen's missing draft means
    suppression) come back as chain failures.
    """
    if case.get("previous"):
        service.run_ticket(TicketInput.model_validate(case["previous"]))
    result = service.run_ticket(TicketInput.model_validate(case["ticket"])).result
    chain_failures: list[str] = []
    if case.get("expect_original_draft") is True and result.draft is None:
        chain_failures.append(
            "premise broken: the original run earned no template draft, "
            "so the reopen's missing draft proves nothing"
        )
    if case.get("expect_original_draft") is False and result.draft is not None:
        chain_failures.append("the original run unexpectedly earned a template draft")
    if case.get("outcome"):
        outcome = case["outcome"]
        service.record_outcome(
            result.ticket_id,
            AgentOutcome(
                final_category=outcome.get("final_category", ""),
                final_queue=outcome.get("final_queue", ""),
                resolved=bool(outcome.get("resolved", False)),
                recorded_by=outcome.get("by", "eval-agent"),
            ),
        )
    current = result
    for spec in case.get("reopens", []):
        current = service.reopen_run(
            current.ticket_id,
            message=spec["message"],
            actor=spec.get("actor", "eval-agent"),
        ).result
    return current, chain_failures


def check_thread_case(service: TriageService, case: dict) -> list[str]:
    """A thread case: two tickets from one customer, one relation.

    ``expected`` describes the first ticket (checked as usual); the
    second ticket in ``thread_pair`` is judged only on the thread
    relation — same thread, or (the near-miss) firmly separate.
    """
    failures: list[str] = []
    first = service.run_ticket(TicketInput.model_validate(case["ticket"])).result
    failures += [f"first ticket: {failure}" for failure in check_case(case, first)]
    second = service.run_ticket(
        TicketInput.model_validate(case["thread_pair"]["ticket"])
    ).result
    first_thread = effective_thread_id(first.thread_id, first.ticket_id)
    second_thread = effective_thread_id(second.thread_id, second.ticket_id)
    relation = case.get("expect_thread_relation", "separate")
    if relation == "separate" and first_thread == second_thread:
        failures.append(
            f"near-miss tickets share thread {first_thread} — they must stay separate"
        )
    if relation == "same" and first_thread != second_thread:
        failures.append(
            f"related tickets split across threads {first_thread} / {second_thread}"
        )
    return failures


def _apply_operator_action(service: TriageService, action: dict) -> list[str]:
    """One declared operator action; `expect_error` names the refusal
    a correct implementation must raise (matched as a substring)."""
    kind = action["type"]
    actor = action.get("actor", "eval-agent")
    reason = action.get("reason", "eval correction")
    expect_error = action.get("expect_error")
    try:
        if kind == "attach":
            service.attach_run_to_thread(action["ticket"], action["thread_id"],
                                         actor=actor, reason=reason)
        elif kind == "detach":
            service.detach_run(action["ticket"], actor=actor, reason=reason)
        elif kind == "merge":
            service.merge_threads(action["source"], action["target"],
                                  actor=actor, reason=reason)
        elif kind == "mark_duplicate":
            service.mark_duplicate(action["ticket"], action["of"],
                                   actor=actor, reason=reason)
        elif kind == "unmark_duplicate":
            service.unmark_duplicate(action["ticket"], actor=actor, reason=reason)
        else:
            return [f"unknown operator action {kind!r}"]
    except Exception as exc:  # a refusal is a ValueError/KeyError by design
        if expect_error and expect_error in str(exc):
            return []
        return [f"operator action {kind} raised unexpectedly: {exc}"]
    if expect_error:
        return [
            f"operator action {kind} should have been refused "
            f"({expect_error!r}) but succeeded"
        ]
    return []


def check_operator_case(service: TriageService, case: dict) -> list[str]:
    """An operator-correction case (corrections.py).

    Runs the case's tickets (``previous``, ``ticket``, ``thread_pair``,
    ``extra_tickets``), records any ``outcomes``, applies the
    ``operator_actions`` in order — each either succeeding or raising
    its declared refusal — then judges what the case declares: the
    thread relation before/after, the grouping lineage on corrected
    records (computed id preserved underneath), and the calibration
    accounting for confirmed duplicates (excluded from the samples,
    still counted in volume).
    """
    failures: list[str] = []
    runs = 0
    if case.get("previous"):
        service.run_ticket(TicketInput.model_validate(case["previous"]))
        runs += 1
    result = service.run_ticket(TicketInput.model_validate(case["ticket"])).result
    runs += 1
    failures += check_case(case, result)
    if case.get("thread_pair"):
        service.run_ticket(TicketInput.model_validate(case["thread_pair"]["ticket"]))
        runs += 1
    for extra in case.get("extra_tickets", []):
        service.run_ticket(TicketInput.model_validate(extra))
        runs += 1

    def relation(a: str, b: str) -> str:
        same = service.run_thread(a)["thread_id"] == service.run_thread(b)["thread_id"]
        return "same" if same else "separate"

    before = case.get("expect_thread_relation_before")
    if before and relation(before["a"], before["b"]) != before["relation"]:
        failures.append(
            f"premise broken: tickets {before['a']} / {before['b']} are not "
            f"{before['relation']} before the correction"
        )
    for ticket_id, outcome in (case.get("outcomes") or {}).items():
        service.record_outcome(
            ticket_id,
            AgentOutcome(
                final_category=outcome.get("final_category", ""),
                final_queue=outcome.get("final_queue", ""),
                resolved=bool(outcome.get("resolved", False)),
                recorded_by=outcome.get("by", "eval-agent"),
            ),
        )
    for action in case.get("operator_actions", []):
        failures += _apply_operator_action(service, action)
    after = case.get("expect_thread_relation_after")
    if after and relation(after["a"], after["b"]) != after["relation"]:
        failures.append(
            f"tickets {after['a']} / {after['b']} are not {after['relation']} "
            "after the correction"
        )
    for ticket_id in case.get("expect_grouping_lineage", []):
        record = service.get_record(ticket_id)
        view = corrections.grouping_view(record)
        if not view["corrected"]:
            failures.append(f"{ticket_id} carries no correction lineage")
        elif view["computed_thread_id"] != effective_thread_id(
            record.thread_id, record.ticket_id
        ):
            failures.append(f"{ticket_id}'s computed thread id was rewritten")
    for ticket_id in case.get("expect_duplicate_accounting", []):
        live = live_calibration(service.store.all_records())
        excluded = [entry["ticket_id"] for entry in live["duplicates_excluded"]]
        if ticket_id not in excluded:
            failures.append(
                f"{ticket_id} is not accounted as an excluded duplicate "
                "in live calibration"
            )
        overview = overview_stats(service.store.all_records())
        if overview["total_runs"] != runs:
            failures.append(
                f"volume stats count {overview['total_runs']} runs, expected {runs} "
                "(a confirmed duplicate still counts — it was a ticket)"
            )
    return failures


def _record_eval_outcome(service: TriageService, ticket_id: str, outcome: dict) -> None:
    service.record_outcome(
        ticket_id,
        AgentOutcome(
            final_category=outcome.get("final_category", ""),
            final_queue=outcome.get("final_queue", ""),
            resolved=bool(outcome.get("resolved", False)),
            recorded_by=outcome.get("by", "eval-agent"),
            resolution_summary=outcome.get("resolution_summary", ""),
        ),
    )


def _apply_workbench_action(service: TriageService, ticket_id: str, action: dict) -> list[str]:
    """One declared workbench action; `expect_error` names the
    refusal a correct implementation must raise (substring)."""
    kind = action["type"]
    actor = action.get("actor", "eval-agent")
    expect_error = action.get("expect_error")
    try:
        if kind == "check_step":
            service.check_playbook_step(ticket_id, action["step_id"], actor=actor)
        elif kind == "uncheck_step":
            service.check_playbook_step(ticket_id, action["step_id"], actor=actor, checked=False)
        elif kind == "mark_na":
            service.mark_playbook_na(
                ticket_id, actor=actor, reason=action.get("reason", "eval mark")
            )
        elif kind == "lift_na":
            service.lift_playbook_na(ticket_id, actor=actor)
        elif kind == "save_note":
            service.save_working_note(ticket_id, text=action["text"], actor=actor)
        else:
            return [f"unknown workbench action {kind!r}"]
    except Exception as exc:  # a refusal is a ValueError/KeyError by design
        if expect_error and expect_error in str(exc):
            return []
        return [f"workbench action {kind} raised unexpectedly: {exc}"]
    if expect_error:
        return [
            f"workbench action {kind} should have been refused "
            f"({expect_error!r}) but succeeded"
        ]
    return []


def check_workbench_case(service: TriageService, case: dict) -> list[str]:
    """A workbench case (workbench.py).

    Seeds the case's ``neighbours`` first — similar tickets, some
    with recorded outcomes at authored intake ages (``age_hours``
    becomes a received_at that many hours before now, so the
    intake-to-outcome elapsed is deterministic) — plus any
    ``extra_tickets``; then runs the main ticket through the normal
    pipeline checks and judges the workbench: the knowledge
    aggregate over the neighbours, the declared workbench actions
    (each succeeding or raising its refusal), the playbook state
    they leave, and the saved note.
    """
    failures: list[str] = []
    for spec in case.get("neighbours", []):
        ticket = dict(spec["ticket"])
        if "age_hours" in spec:
            moment = datetime.now(timezone.utc) - timedelta(hours=spec["age_hours"])
            ticket["received_at"] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
        service.run_ticket(TicketInput.model_validate(ticket))
        if spec.get("outcome"):
            _record_eval_outcome(service, ticket["ticket_id"], spec["outcome"])
    for extra in case.get("extra_tickets", []):
        service.run_ticket(TicketInput.model_validate(extra))

    result, chain_failures = run_case(service, case)
    failures += chain_failures + check_case(case, result)

    exp = case.get("expect_workbench") or {}
    if exp:
        wb = service.workbench(result.ticket_id)
        agg = wb["knowledge"]["aggregate"]
        for key in ("with_outcome", "without_outcome", "resolved_count", "similar_total"):
            if key in exp and agg[key] != exp[key]:
                failures.append(f"knowledge {key}: expected {exp[key]}, got {agg[key]}")
        if "resolved_rate" in exp:
            got = agg["resolved_rate"]
            if exp["resolved_rate"] is None and got is not None:
                failures.append(f"knowledge rate should be withheld, got {got}")
            elif exp["resolved_rate"] is not None and (
                got is None or abs(got - exp["resolved_rate"]) > 1e-4
            ):
                failures.append(
                    f"knowledge rate: expected {exp['resolved_rate']}, got {got}"
                )
        if "median_hours" in exp:
            got = agg["median_hours_from_intake"]
            if exp["median_hours"] is None and got is not None:
                failures.append(f"knowledge median should be withheld, got {got}")
            elif exp["median_hours"] is not None and (
                got is None or abs(got - exp["median_hours"]) > 0.25
            ):
                failures.append(
                    f"knowledge median: expected ~{exp['median_hours']}h, got {got}"
                )
        if exp.get("note_contains") and exp["note_contains"] not in agg["note"]:
            failures.append(f"knowledge note does not say {exp['note_contains']!r}: {agg['note']}")
        if exp.get("note_draft_contains_routing_reason"):
            if result.composition.routing_reason not in wb["working_note"]["draft"]:
                failures.append("the working-note draft lacks the routing reason verbatim")
    for action in case.get("workbench_actions", []):
        failures += _apply_workbench_action(service, result.ticket_id, action)
    playbook_exp = case.get("expect_playbook")
    if playbook_exp is not None:
        view = service.workbench_playbook(result.ticket_id)
        if view["available"] != playbook_exp.get("available", True):
            failures.append(
                f"playbook availability: expected {playbook_exp.get('available', True)}, "
                f"got {view['available']}"
            )
        checked = sorted(s["id"] for s in view["steps"] if s["checked"])
        if checked != sorted(playbook_exp.get("checked", [])):
            failures.append(
                f"playbook checked steps: expected {sorted(playbook_exp.get('checked', []))}, "
                f"got {checked}"
            )
        if (view["not_applicable"] is None) == bool(playbook_exp.get("not_applicable")):
            failures.append("playbook not-applicable state does not match the expectation")
    if case.get("expect_no_playbook_ticket"):
        view = service.workbench_playbook(case["expect_no_playbook_ticket"])
        if view["available"]:
            failures.append(
                f"{case['expect_no_playbook_ticket']} unexpectedly has a playbook"
            )
    if case.get("expect_saved_note"):
        record = service.get_record(result.ticket_id)
        if record.working_note is None or record.working_note.text != case["expect_saved_note"]:
            failures.append("the saved working note is not on the record as declared")
    return failures


def check_case(case: dict, result) -> list[str]:
    expected = case["expected"]
    failures: list[str] = []
    if result.category != expected["category"]:
        failures.append(f"category {result.category} != expected {expected['category']}")
    queue = result.composition.queue
    if queue != expected["queue"]:
        failures.append(f"queue {queue} != expected {expected['queue']}")
    band = result.composition.priority_band
    if band != expected["priority_band"]:
        failures.append(
            f"priority band {band} (score {result.composition.priority_score}) "
            f"!= expected {expected['priority_band']}"
        )
    for key, truth in expected["verify"].items():
        answer = result.answer(key)
        raised = answer is not None and float(answer.value) >= FLAG_THRESHOLD
        if raised != truth:
            failures.append(f"{key} raised={raised} (p={float(answer.value):.2f}) != expected {truth}")
    if case.get("expect_duplicate") and not result.duplicate.is_duplicate:
        failures.append(
            f"expected a duplicate link (similarity {result.duplicate.similarity:.3f})"
        )
    if case.get("expect_injection_flags") and not result.injection_flags:
        failures.append("expected injection flags from the screen")
    expect_reopen = case.get("expect_reopen")
    if expect_reopen:
        info = result.reopen
        if info is None:
            failures.append("expected a reopen run, but the result carries no reopen link")
        else:
            if info.reopen_count != expect_reopen["count"]:
                failures.append(
                    f"reopen count {info.reopen_count} != expected {expect_reopen['count']}"
                )
            if expect_reopen.get("floor_applied") and not result.composition.reopen_floor_applied:
                failures.append("the reopen priority floor was not applied")
            if expect_reopen.get("limit_hit"):
                if not info.limit_hit:
                    failures.append("the chain was expected to hit the reopen limit")
                if not result.guardrails.requires_human_review:
                    failures.append("a limit-hit reopen must be forced to human review")
                if not any(
                    "reopen limit" in reason
                    for reason in result.guardrails.human_review_reasons
                ):
                    failures.append("the human-review reasons do not name the reopen limit")
        if expect_reopen.get("no_draft") and result.draft is not None:
            failures.append("a reopen run received a template draft — reopens never do")
    if result.status != "awaiting_agent" or result.external_action_taken:
        failures.append(f"run did not end clean at the gate (status {result.status})")
    return failures


def check_review_case(service: TriageService, case: dict) -> list[str]:
    """QA-review cases (rounds 8 + 10): grading the triage itself.

    Seeds a world of runs, decides them at the gate (the review
    population is decided runs — the store's own word), optionally
    marks a confirmed duplicate, then draws the sample and asserts
    the draw itself: the exact run ids for the case's seed, the
    population accounting (decided total, exclusions by name), and
    a redraw that must reproduce it. Declared verdicts are recorded
    (``expect_error`` names the refusal a correct implementation
    must raise), and the agreement report is graded on its math:
    per-family rates, the per-backend split, the disagreement
    count, the withheld rate below the floor. A case may also state
    an ``overlap`` on the sample — the two reviewers, a count or
    fraction, the exact assigned run ids — which the redraw must
    reproduce too, and grade the report's ``inter_rater`` block:
    pair accounting, per-family raw agreement and Cohen's kappa
    (``null`` where undefined, with the note naming why), and the
    paired disagreement count. When the case asks for the pins,
    the run records and the live calibration payload are
    snapshotted before the verdicts and must be byte-identical
    after — grading a triage changes nothing about it.
    """
    failures: list[str] = []
    for entry in case.get("review_seed", []):
        ticket = TicketInput.model_validate(entry["ticket"])
        service.run_ticket(ticket)
        if entry.get("outcome"):
            _record_eval_outcome(service, ticket.ticket_id, entry["outcome"])
        if entry.get("decide", True):
            service.record_decision(ticket.ticket_id, "approved", by="eval-agent")
    for entry in case.get("review_seed", []):
        if entry.get("duplicate_of"):
            service.mark_duplicate(
                entry["ticket"]["ticket_id"],
                entry["duplicate_of"],
                actor="eval-operator",
                reason="the customer re-sent the same ticket",
            )

    spec = case["qa_sample"]
    overlap_spec = spec.get("overlap") or {}
    overlap_kwargs = {}
    if overlap_spec:
        overlap_kwargs = {
            "overlap_reviewers": overlap_spec["reviewers"],
            "overlap_count": overlap_spec.get("count"),
            "overlap_fraction": overlap_spec.get("fraction"),
        }
    sample = service.create_review_sample(
        size=spec["size"],
        seed=spec.get("seed", 0),
        created_by="eval-qa",
        exclude_previously_reviewed=spec.get("exclude_previously_reviewed", True),
        **overlap_kwargs,
    )
    expect_ids = spec.get("expect_run_ids")
    if expect_ids is not None and sample.run_ids != expect_ids:
        failures.append(
            f"sampler drew {sample.run_ids}, expected exactly {expect_ids} "
            "(same store + seed + size must reproduce the same sample)"
        )
    if overlap_spec:
        if sample.overlap.get("reviewers") != overlap_spec["reviewers"]:
            failures.append(
                f"sample overlap reviewers = {sample.overlap.get('reviewers')}, "
                f"expected {overlap_spec['reviewers']}"
            )
        expect_overlap = overlap_spec.get("expect_run_ids")
        if expect_overlap is not None and sample.overlap.get("run_ids") != expect_overlap:
            failures.append(
                f"sampler assigned the overlap to {sample.overlap.get('run_ids')}, "
                f"expected exactly {expect_overlap} (the assignment is part of "
                "the seeded draw and must reproduce with it)"
            )
    for key, wanted in spec.get("expect_population", {}).items():
        if sample.population.get(key) != wanted:
            failures.append(
                f"sample population[{key!r}] = {sample.population.get(key)}, expected {wanted}"
            )
    if spec.get("expect_redraw_same"):
        redraw = service.create_review_sample(
            size=spec["size"], seed=spec.get("seed", 0), created_by="eval-qa",
            exclude_previously_reviewed=spec.get("exclude_previously_reviewed", True),
            **overlap_kwargs,
        )
        if redraw.run_ids != sample.run_ids:
            failures.append(
                f"redraw with the same seed gave {redraw.run_ids}, expected {sample.run_ids}"
            )
        if overlap_spec and redraw.overlap.get("run_ids") != sample.overlap.get("run_ids"):
            failures.append(
                f"redraw assigned the overlap to {redraw.overlap.get('run_ids')}, "
                f"expected {sample.overlap.get('run_ids')}"
            )

    snapshot = None
    if case.get("qa_pins"):
        records = service.store.all_records()
        snapshot = (
            json.dumps([r.model_dump() for r in records], sort_keys=True, default=str),
            json.dumps(live_samples(records), sort_keys=True, default=str),
            json.dumps(live_calibration(records), sort_keys=True, default=str),
        )

    for verdict_spec in case.get("qa_verdicts", []):
        expect_error = verdict_spec.get("expect_error")
        try:
            service.record_review_verdict(
                sample.sample_id,
                ticket_id=verdict_spec["ticket"],
                reviewer=verdict_spec.get("reviewer", "eval-reviewer"),
                families=verdict_spec.get("families", {}),
                overall=verdict_spec.get("overall", ""),
                note=verdict_spec.get("note", ""),
            )
        except Exception as exc:  # noqa: BLE001 — a refusal is data here
            if expect_error and expect_error in str(exc):
                continue
            failures.append(
                f"verdict on {verdict_spec['ticket']} raised {exc!r}, expected success"
                if not expect_error
                else f"verdict on {verdict_spec['ticket']} raised {exc!r}, "
                f"expected a refusal naming {expect_error!r}"
            )
            continue
        if expect_error:
            failures.append(
                f"verdict on {verdict_spec['ticket']} was accepted, "
                f"expected a refusal naming {expect_error!r}"
            )

    if case.get("qa_pins") and snapshot is not None:
        records = service.store.all_records()
        after = (
            json.dumps([r.model_dump() for r in records], sort_keys=True, default=str),
            json.dumps(live_samples(records), sort_keys=True, default=str),
            json.dumps(live_calibration(records), sort_keys=True, default=str),
        )
        labels = ("run records", "live calibration samples", "live calibration payload")
        for label, before_blob, after_blob in zip(labels, snapshot, after):
            if before_blob != after_blob:
                failures.append(
                    f"recording verdicts changed the {label} — grading must be inert"
                )

    report_spec = case.get("qa_expect_report")
    if report_spec:
        report = service.review_report([sample.sample_id])
        for key in ("verdicts", "overall_agree", "sampled_runs", "verdicted_runs"):
            if key in report_spec and report[key] != report_spec[key]:
                failures.append(
                    f"report {key} = {report[key]}, expected {report_spec[key]}"
                )
        if "overall_rate" in report_spec and report["overall_rate"] != report_spec["overall_rate"]:
            failures.append(
                f"report overall rate = {report['overall_rate']}, "
                f"expected {report_spec['overall_rate']}"
            )
        for family, wanted in report_spec.get("families", {}).items():
            row = report["families"][family]
            agree, rate = wanted
            if row["agree"] != agree or row["rate"] != rate:
                failures.append(
                    f"report family {family} = agree {row['agree']}, rate {row['rate']}; "
                    f"expected agree {agree}, rate {rate}"
                )
        if "disagreement_rows" in report_spec and len(report["disagreements"]) != report_spec["disagreement_rows"]:
            failures.append(
                f"report lists {len(report['disagreements'])} disagreement rows, "
                f"expected {report_spec['disagreement_rows']}"
            )
        if "by_backend" in report_spec and sorted(report["by_backend"]) != report_spec["by_backend"]:
            failures.append(
                f"report backends = {sorted(report['by_backend'])}, "
                f"expected {report_spec['by_backend']}"
            )
        if "note_contains" in report_spec and report_spec["note_contains"] not in report["note"]:
            failures.append(
                f"report note {report['note']!r} does not name {report_spec['note_contains']!r}"
            )
        ir_spec = report_spec.get("inter_rater")
        if ir_spec:
            block = report["inter_rater"]
            for key in ("pairs_assigned", "pairs_complete", "pairs_partial"):
                if key in ir_spec and block[key] != ir_spec[key]:
                    failures.append(
                        f"inter-rater {key} = {block[key]}, expected {ir_spec[key]}"
                    )
            for family, wanted in ir_spec.get("families", {}).items():
                row = block["families"][family]
                rate, kappa = wanted
                if row["agreement_rate"] != rate or row["kappa"] != kappa:
                    failures.append(
                        f"inter-rater family {family} = rate {row['agreement_rate']}, "
                        f"kappa {row['kappa']}; expected rate {rate}, kappa {kappa}"
                    )
            if "overall" in ir_spec:
                rate, kappa = ir_spec["overall"]
                row = block["overall"]
                if row["agreement_rate"] != rate or row["kappa"] != kappa:
                    failures.append(
                        f"inter-rater overall = rate {row['agreement_rate']}, "
                        f"kappa {row['kappa']}; expected rate {rate}, kappa {kappa}"
                    )
            if "pair_disagreement_rows" in ir_spec and len(block["disagreements"]) != ir_spec["pair_disagreement_rows"]:
                failures.append(
                    f"inter-rater lists {len(block['disagreements'])} paired "
                    f"disagreement rows, expected {ir_spec['pair_disagreement_rows']}"
                )
            for family, needle in ir_spec.get("kappa_note_contains", {}).items():
                if needle not in block["families"][family]["kappa_note"]:
                    failures.append(
                        f"inter-rater family {family} kappa note "
                        f"{block['families'][family]['kappa_note']!r} does not "
                        f"name {needle!r}"
                    )

    for more in case.get("qa_more_samples", []):
        follow = service.create_review_sample(
            size=more["size"],
            seed=more.get("seed", 0),
            created_by="eval-qa",
            exclude_previously_reviewed=more.get("exclude_previously_reviewed", True),
        )
        for key, wanted in more.get("expect_population", {}).items():
            if follow.population.get(key) != wanted:
                failures.append(
                    f"follow-up sample population[{key!r}] = {follow.population.get(key)}, "
                    f"expected {wanted}"
                )
        for absent in more.get("expect_absent", []):
            if absent in follow.run_ids:
                failures.append(f"follow-up sample drew {absent}, which it must exclude")
    return failures


def _check_dossier_view(service: TriageService, spec: dict, label: str) -> list[str]:
    """Judge one dossier expectation block — and pin the projection's
    purity: the stored records are byte-identical after computing it."""
    failures: list[str] = []
    before = json.dumps(
        [r.model_dump() for r in service.store.all_records()], sort_keys=True, default=str
    )
    view = service.customer_dossier(spec["key"])
    after = json.dumps(
        [r.model_dump() for r in service.store.all_records()], sort_keys=True, default=str
    )
    if before != after:
        failures.append(f"{label}: computing the dossier changed the stored records")
    for key, wanted in (spec.get("counts") or {}).items():
        if view["counts"].get(key) != wanted:
            failures.append(
                f"{label}: dossier counts[{key!r}] = {view['counts'].get(key)}, "
                f"expected {wanted}"
            )
    if "ticket_ids" in spec:
        got = [row["ticket_id"] for row in view["tickets"]]
        if got != spec["ticket_ids"]:
            failures.append(f"{label}: dossier tickets {got}, expected {spec['ticket_ids']}")
    flags = [
        (f["flag"], f["category"], f["count"], f["ticket_ids"]) for f in view["flags"]
    ]
    wanted_flags = [
        (f["flag"], f.get("category", ""), f["count"], f["ticket_ids"])
        for f in spec.get("flags", [])
    ]
    if "flags" in spec and flags != wanted_flags:
        failures.append(f"{label}: dossier flags {flags}, expected {wanted_flags}")
    for key, wanted in (spec.get("outcomes") or {}).items():
        got = view["outcomes"].get(key)
        if got != wanted:
            failures.append(f"{label}: dossier outcomes[{key!r}] = {got}, expected {wanted}")
    if "threads" in spec and view["counts"]["threads"] != spec["threads"]:
        failures.append(
            f"{label}: dossier threads = {view['counts']['threads']}, "
            f"expected {spec['threads']}"
        )
    for thread_id, members in (spec.get("thread_members") or {}).items():
        summary = next(
            (s for s in view["threads"] if s["thread_id"] == thread_id), None
        )
        if summary is None or summary["ticket_ids"] != members:
            failures.append(
                f"{label}: dossier thread {thread_id} holds "
                f"{summary['ticket_ids'] if summary else None}, expected {members}"
            )
    for ticket_id in spec.get("corrected_tickets", []):
        row = next((t for t in view["tickets"] if t["ticket_id"] == ticket_id), None)
        if row is None or not (row.get("grouping") or {}).get("corrected"):
            failures.append(
                f"{label}: dossier row {ticket_id} does not carry the corrected grouping"
            )
    return failures


def check_dossier_case(service: TriageService, case: dict) -> list[str]:
    """A customer-dossier case (dossier.py).

    Seeds a staged customer history (tickets, gate decisions,
    recorded outcomes) and judges the dossier over it; applies any
    operator actions; then runs the case's new ticket and judges
    what its record carries — the history summary, the flags with
    their exact evidence, and the composed history block, which
    must include the declared lines and NEVER a phrase from
    another ticket's body — and finally judges the dossier over
    the whole record, new ticket included.
    """
    failures: list[str] = []
    for entry in case.get("dossier_seed", []):
        ticket = TicketInput.model_validate(entry["ticket"])
        service.run_ticket(ticket)
        if entry.get("decide", True):
            service.record_decision(ticket.ticket_id, "approved", by="eval-agent")
        if entry.get("outcome"):
            _record_eval_outcome(service, ticket.ticket_id, entry["outcome"])
    if case.get("expect_dossier_before"):
        failures += _check_dossier_view(
            service, case["expect_dossier_before"], "dossier before"
        )
    for action in case.get("operator_actions", []):
        failures += _apply_operator_action(service, action)

    new_spec = case.get("dossier_new_ticket")
    if new_spec:
        new_ticket = TicketInput.model_validate(new_spec)
        # The block the run is about to be offered, composed over the
        # same priors the run itself will see.
        block = service._customer_history(new_ticket)["context"]
        for phrase in case.get("history_block_includes", []):
            if phrase not in block:
                failures.append(f"history block does not carry {phrase!r}:\n{block}")
        for phrase in case.get("history_block_excludes", []):
            if phrase in block:
                failures.append(
                    f"history block leaks another ticket's text: {phrase!r} is in it"
                )
        result = service.run_ticket(new_ticket).result
        if result.status != "awaiting_agent" or result.external_action_taken:
            failures.append(f"run did not end clean at the gate (status {result.status})")
        exp = case.get("expect_history") or {}
        history = result.history
        for key in (
            "context_provided",
            "context_use",
            "prior_tickets",
            "decided",
            "reopened",
            "outcomes_recorded",
            "last_outcome_ticket_id",
        ):
            if key in exp and getattr(history, key) != exp[key]:
                failures.append(
                    f"history {key} = {getattr(history, key)}, expected {exp[key]}"
                )
        if "resolved_rate" in exp and history.resolved_rate != exp["resolved_rate"]:
            failures.append(
                f"history resolved_rate = {history.resolved_rate}, "
                f"expected {exp['resolved_rate']} (None = honestly withheld)"
            )
        got_flags = [
            (f.flag, f.category, f.count, f.ticket_ids) for f in history.flags
        ]
        wanted_flags = [
            (f["flag"], f.get("category", ""), f["count"], f["ticket_ids"])
            for f in exp.get("flags", [])
        ]
        if "flags" in exp and got_flags != wanted_flags:
            failures.append(f"history flags {got_flags}, expected {wanted_flags}")
    # The dossier expectation describes the customer's whole record
    # as it stands at the end of the case — new ticket included.
    if case.get("expect_dossier"):
        failures += _check_dossier_view(service, case["expect_dossier"], "dossier")
    return failures


def check_template_case(service: TriageService, case: dict) -> list[str]:
    """A template-effectiveness case (template_effectiveness.py).

    Seeds a staged world of template-earning tickets — each seed's
    earned template is asserted as a premise (a seed that earns no
    draft, or the wrong area's, proves nothing) — decides them,
    records their outcomes, reopens the declared ones, and then
    un-names one stored draft (``legacy_template_ticket``) to stage
    the pre-link record shape. Judges the effectiveness report
    against hand-counted expectations: per-template rows, the
    category baseline, the totals accounting, and the flagged set.
    The report is pinned as a pure read (the stored records are
    byte-identical across computing it), and a ``firewall_probe``
    ticket must compose byte-identically in this flagged world and
    in a fresh service where the flag does not stand.
    """
    failures: list[str] = []
    for entry in case.get("template_seed", []):
        ticket = TicketInput.model_validate(entry["ticket"])
        result = service.run_ticket(ticket).result
        got = result.draft.template if result.draft is not None else None
        if got != entry.get("expect_template"):
            failures.append(
                f"{ticket.ticket_id}: premise broken — earned template "
                f"{got!r}, expected {entry.get('expect_template')!r}"
            )
        if entry.get("decide", True):
            service.record_decision(ticket.ticket_id, "approved", by="eval-agent")
        if entry.get("outcome"):
            _record_eval_outcome(service, ticket.ticket_id, entry["outcome"])
        for spec in entry.get("reopens", []):
            service.reopen_run(
                ticket.ticket_id,
                message=spec["message"],
                actor=spec.get("actor", "eval-agent"),
            )
    legacy = case.get("legacy_template_ticket")
    if legacy:
        record = service.get_record(legacy)
        if record is None or record.draft is None:
            failures.append(
                f"legacy premise broken: {legacy} has no stored draft to un-name"
            )
        else:
            # The pre-link record shape: a draft whose template name
            # was never recorded. Staged by blanking the stored
            # record — history is not backfilled, here or in code.
            record.draft.template = ""
            service.store.save(record)

    exp = case.get("expect_templates") or {}
    if exp:
        before = json.dumps(
            [r.model_dump() for r in service.store.all_records()],
            sort_keys=True,
            default=str,
        )
        report = service.template_effectiveness()
        after = json.dumps(
            [r.model_dump() for r in service.store.all_records()],
            sort_keys=True,
            default=str,
        )
        if before != after:
            failures.append("computing the effectiveness report changed the stored records")
        for key, wanted in (exp.get("totals") or {}).items():
            if report["totals"].get(key) != wanted:
                failures.append(
                    f"totals[{key!r}] = {report['totals'].get(key)}, expected {wanted}"
                )
        rows = {row["template"]: row for row in report["templates"]}
        for name, wanted_row in (exp.get("rows") or {}).items():
            row = rows.get(name)
            if row is None:
                failures.append(f"no effectiveness row for template {name!r}")
                continue
            for key, wanted in wanted_row.items():
                if row.get(key) != wanted:
                    failures.append(
                        f"row {name!r}[{key!r}] = {row.get(key)}, expected {wanted}"
                    )
        for name in exp.get("absent_rows", []):
            if name in rows:
                failures.append(f"template {name!r} unexpectedly has a row")
        categories = {row["category"]: row for row in report["categories"]}
        for category, wanted_row in (exp.get("categories") or {}).items():
            row = categories.get(category)
            if row is None:
                failures.append(f"no baseline row for category {category!r}")
                continue
            for key, wanted in wanted_row.items():
                if row.get(key) != wanted:
                    failures.append(
                        f"baseline {category!r}[{key!r}] = {row.get(key)}, "
                        f"expected {wanted}"
                    )
        if "flagged" in exp:
            flagged = sorted(
                row["template"] for row in report["templates"] if row["flag"]
            )
            if flagged != sorted(exp["flagged"]):
                failures.append(f"flagged templates {flagged}, expected {exp['flagged']}")

    probe = case.get("firewall_probe")
    if probe:
        spec = TicketInput.model_validate(probe)
        here = service.run_ticket(spec).result
        fresh = fresh_service().run_ticket(
            TicketInput.model_validate(probe)
        ).result
        if here.composition.model_dump() != fresh.composition.model_dump():
            failures.append(
                "firewall broken: the probe composed differently in the "
                "flagged world than in a fresh service"
            )
        if [a.model_dump() for a in here.answers] != [
            a.model_dump() for a in fresh.answers
        ]:
            failures.append("firewall broken: the probe's answers differ")
        here_draft = here.draft.template if here.draft is not None else None
        fresh_draft = fresh.draft.template if fresh.draft is not None else None
        if here_draft != fresh_draft:
            failures.append(
                f"firewall broken: probe draft {here_draft!r} under the flag "
                f"vs {fresh_draft!r} without it — a flag must suppress nothing"
            )

    for exp_export in case.get("expect_export", []):
        document = service.export_run(exp_export["ticket_id"])
        stats = document["template_effectiveness"] or {}
        for key, wanted in exp_export.items():
            if key == "ticket_id":
                continue
            if stats.get(key) != wanted:
                failures.append(
                    f"export {exp_export['ticket_id']}: template block "
                    f"[{key!r}] = {stats.get(key)}, expected {wanted}"
                )
    return failures


def check_incident_case(service: TriageService, case: dict) -> list[str]:
    """An incident-clustering case (incidents.py).

    Seeds a staged world of tickets — each seed's decided
    placement is asserted as a premise (a seed that lands in the
    wrong bucket proves nothing) — then applies the case's
    ``incident_actions`` in order: scans judged on the exact
    proposal member sets, confirmations and dismissals by member
    set, late arrivals run in mid-case, addition lists, per-ticket
    add/decline decisions, and closure. After EVERY action the
    stored run records are checked byte-identical against the
    snapshot (extended whenever a new ticket is run in): the
    whole lifecycle is an overlay, and this is where that promise
    is kept or broken.
    """
    failures: list[str] = []

    def run_seed(entry: dict) -> None:
        ticket = TicketInput.model_validate(entry["ticket"])
        result = service.run_ticket(ticket).result
        placement = entry.get("expect_placement") or {}
        for field in ("category", "product_area"):
            wanted = placement.get(field)
            if wanted and getattr(result, field) != wanted:
                failures.append(
                    f"{ticket.ticket_id}: premise broken — decided "
                    f"{field} {getattr(result, field)!r}, expected {wanted!r}; "
                    "the clustering judgement would prove nothing"
                )

    for entry in case.get("incident_seed", []):
        run_seed(entry)

    snapshot = {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }

    def records_unchanged(label: str) -> None:
        current = {
            record.ticket_id: record.model_dump_json()
            for record in service.store.all_records()
        }
        for ticket_id, before in snapshot.items():
            if current.get(ticket_id) != before:
                failures.append(
                    f"{label}: run {ticket_id}'s record changed — incident "
                    "acts are an overlay and must never write to a run"
                )
        snapshot.update(
            {tid: blob for tid, blob in current.items() if tid not in snapshot}
        )

    def proposal_for(members: list[str]) -> dict | None:
        wanted = sorted(members)
        for proposal in service.cluster_proposals()["proposals"]:
            if sorted(m["ticket_id"] for m in proposal["members"]) == wanted:
                return proposal
        return None

    for index, action in enumerate(case.get("incident_actions", [])):
        kind = action["type"]
        label = f"action {index} ({kind})"
        expect_error = action.get("expect_error")
        try:
            if kind == "scan":
                payload = service.cluster_proposals()
                got = [
                    [m["ticket_id"] for m in proposal["members"]]
                    for proposal in payload["proposals"]
                ]
                wanted = [entry["members"] for entry in action["expect_proposals"]]
                if got != wanted:
                    failures.append(
                        f"{label}: proposals {got}, expected {wanted}"
                    )
                for entry in action["expect_proposals"]:
                    proposal = proposal_for(entry["members"])
                    if proposal is None:
                        continue
                    if "outgrows_dismissal" in entry:
                        outgrows = proposal["outgrows_dismissal"] is not None
                        if outgrows != entry["outgrows_dismissal"]:
                            failures.append(
                                f"{label}: proposal {entry['members']} "
                                f"outgrows_dismissal is {outgrows}, "
                                f"expected {entry['outgrows_dismissal']}"
                            )
                    signals = proposal["signals"]
                    for key, wanted_value in (entry.get("signals") or {}).items():
                        if signals.get(key) != wanted_value:
                            failures.append(
                                f"{label}: signal {key!r} = "
                                f"{signals.get(key)}, expected {wanted_value}"
                            )
                if "expect_suppressed" in action:
                    if payload["suppressed_dismissed"] != action["expect_suppressed"]:
                        failures.append(
                            f"{label}: suppressed_dismissed = "
                            f"{payload['suppressed_dismissed']}, expected "
                            f"{action['expect_suppressed']}"
                        )
            elif kind == "run":
                run_seed(action)
            elif kind == "confirm":
                proposal = proposal_for(action["members"])
                if proposal is None:
                    failures.append(
                        f"{label}: no live proposal for members "
                        f"{action['members']} to confirm"
                    )
                else:
                    incident = service.confirm_cluster(
                        proposal["proposal_id"],
                        actor=action.get("actor", "eval-lead"),
                        title=action.get("title", ""),
                    )
                    if action.get("expect_incident") and (
                        incident.incident_id != action["expect_incident"]
                    ):
                        failures.append(
                            f"{label}: confirmed as {incident.incident_id}, "
                            f"expected {action['expect_incident']}"
                        )
                    origins = {m.ticket_id: m.origin for m in incident.members}
                    if origins != {tid: "proposal" for tid in action["members"]}:
                        failures.append(
                            f"{label}: member origins {origins}, expected all "
                            "'proposal' over the confirmed set"
                        )
            elif kind == "dismiss":
                proposal = proposal_for(action["members"])
                if proposal is None:
                    failures.append(
                        f"{label}: no live proposal for members "
                        f"{action['members']} to dismiss"
                    )
                else:
                    dismissal = service.dismiss_cluster(
                        proposal["proposal_id"],
                        actor=action.get("actor", "eval-lead"),
                        reason=action.get("reason", "eval dismissal"),
                    )
                    if dismissal.member_ticket_ids != sorted(action["members"]):
                        failures.append(
                            f"{label}: dismissal stored "
                            f"{dismissal.member_ticket_ids}, expected "
                            f"{sorted(action['members'])}"
                        )
            elif kind == "expect_additions":
                got = [
                    c["ticket_id"]
                    for c in service.incident_additions(action["incident"])
                ]
                for ticket_id in action.get("contains", []):
                    if ticket_id not in got:
                        failures.append(
                            f"{label}: additions {got} missing {ticket_id}"
                        )
                for ticket_id in action.get("excludes", []):
                    if ticket_id in got:
                        failures.append(
                            f"{label}: additions {got} must exclude {ticket_id}"
                        )
            elif kind == "add":
                service.add_incident_member(
                    action["incident"],
                    action["ticket"],
                    actor=action.get("actor", "eval-lead"),
                )
            elif kind == "decline":
                service.decline_incident_addition(
                    action["incident"],
                    action["ticket"],
                    actor=action.get("actor", "eval-lead"),
                    reason=action.get("reason", ""),
                )
            elif kind == "close":
                service.close_incident(
                    action["incident"],
                    actor=action.get("actor", "eval-lead"),
                    reason=action.get("reason", "eval closure"),
                )
            elif kind == "expect_incident":
                view = service.get_incident(action["incident"])
                record = view["incident"]
                if record["status"] != action["status"]:
                    failures.append(
                        f"{label}: incident status {record['status']}, "
                        f"expected {action['status']}"
                    )
                member_ids = [m["ticket_id"] for m in view["members"]]
                if member_ids != action["members"]:
                    failures.append(
                        f"{label}: members {member_ids}, expected "
                        f"{action['members']}"
                    )
                for ticket_id, origin in (action.get("origins") or {}).items():
                    member = next(
                        (m for m in view["members"] if m["ticket_id"] == ticket_id),
                        None,
                    )
                    if member is None or member["origin"] != origin:
                        failures.append(
                            f"{label}: member {ticket_id} origin "
                            f"{member and member['origin']}, expected {origin!r}"
                        )
            else:
                failures.append(f"{label}: unknown incident action {kind!r}")
        except Exception as exc:  # a refusal is an IncidentError by design
            if not (expect_error and expect_error in str(exc)):
                failures.append(f"{label}: raised unexpectedly: {exc}")
        else:
            if expect_error:
                failures.append(
                    f"{label}: should have been refused ({expect_error!r}) "
                    "but succeeded"
                )
        records_unchanged(label)
    return failures


def check_watch_case(service: TriageService, case: dict) -> list[str]:
    """A breach-watch case (watch.py).

    Seeds a staged queue whose received_at stamps are authored,
    then reads the watch at the case's fixed instant: the ranked
    order, the bands and the counts are hand-derived from each
    run's own SLA target (band premise asserted per seed, so a
    staging drift names itself), the model's risk is asserted to
    rank OPPOSITE the clock on the planted pair, and the lead's
    acts play out under the overlay discipline — a reassignment
    may change nothing on any record but the target's assignment
    trail, and no run's clock reading may move.
    """
    from support_triage.sla import format_timestamp, parse_timestamp, sla_clock

    failures: list[str] = []
    now = parse_timestamp(case["watch_now"])
    if now is None:
        return [f"watch_now {case['watch_now']!r} does not parse"]

    for entry in case.get("watch_seed", []):
        ticket = TicketInput.model_validate(entry["ticket"])
        result = service.run_ticket(ticket).result
        wanted_band = entry.get("expect_band")
        if wanted_band and result.composition.priority_band != wanted_band:
            failures.append(
                f"{ticket.ticket_id}: premise broken — decided band "
                f"{result.composition.priority_band!r}, expected "
                f"{wanted_band!r}; the watch judgement would prove nothing"
            )

    def snapshot() -> dict:
        return {
            record.ticket_id: record.model_dump()
            for record in service.store.all_records()
        }

    def clocks() -> dict:
        return {
            record.ticket_id: sla_clock(record, now)
            for record in service.store.all_records()
        }

    state = {"records": snapshot(), "clocks": clocks()}

    def records_intact(label: str, *, trail_for: str | None = None) -> None:
        current = snapshot()
        for ticket_id, before in state["records"].items():
            after = current.get(ticket_id)
            if after is None:
                failures.append(f"{label}: run {ticket_id} vanished")
                continue
            if ticket_id == trail_for:
                after = {**after, "assignments": before["assignments"]}
            if after != before:
                failures.append(
                    f"{label}: run {ticket_id}'s record changed — the "
                    "watch reads and the reassignment overlays; neither "
                    "rewrites a run"
                )
        state["records"] = current

    def clocks_unmoved(label: str) -> None:
        current = clocks()
        if current != state["clocks"]:
            failures.append(
                f"{label}: an SLA clock reading moved — a reassignment "
                "never buys a run more time"
            )
        state["clocks"] = current

    for index, action in enumerate(case.get("watch_actions", [])):
        kind = action["type"]
        label = f"action {index} ({kind})"
        expect_error = action.get("expect_error")
        try:
            if kind == "assign":
                service.assign(
                    action["ticket"], action["assignee"],
                    by=action.get("by", "eval-gate"),
                )
                state["records"] = snapshot()
                state["clocks"] = clocks()
            elif kind == "decide":
                service.record_decision(
                    action["ticket"], action.get("decision", "approved"),
                    by=action.get("by", "eval-agent"),
                )
                state["records"] = snapshot()
                state["clocks"] = clocks()
            elif kind == "reassign":
                record = service.reassign_run(
                    action["ticket"],
                    assignee=action.get("assignee", ""),
                    by=action.get("by", ""),
                    reason=action.get("reason", ""),
                    now=now,
                )
                entry = record.assignments[-1] if record.assignments else None
                if entry is None:
                    failures.append(f"{label}: no assignment entry was appended")
                else:
                    if entry.assignee != action.get("assignee"):
                        failures.append(
                            f"{label}: trail names {entry.assignee!r}, "
                            f"expected {action.get('assignee')!r}"
                        )
                    if entry.by != action.get("by"):
                        failures.append(
                            f"{label}: trail lead {entry.by!r}, "
                            f"expected {action.get('by')!r}"
                        )
                    if entry.reason != action.get("reason"):
                        failures.append(
                            f"{label}: trail reason {entry.reason!r}, "
                            f"expected {action.get('reason')!r}"
                        )
                    if entry.at != format_timestamp(now):
                        failures.append(
                            f"{label}: trail stamped {entry.at!r}, "
                            f"expected the watch instant {format_timestamp(now)!r}"
                        )
                records_intact(label, trail_for=action["ticket"])
                clocks_unmoved(label)
            elif kind == "watch":
                payload = service.breach_watch(now)
                rows = payload["rows"]
                got_order = [row["ticket_id"] for row in rows]
                if "expect_order" in action and got_order != action["expect_order"]:
                    failures.append(
                        f"{label}: order {got_order}, expected "
                        f"{action['expect_order']}"
                    )
                by_id = {row["ticket_id"]: row for row in rows}
                for ticket_id, band in (action.get("expect_bands") or {}).items():
                    row = by_id.get(ticket_id)
                    if row is None:
                        failures.append(f"{label}: {ticket_id} missing from the watch")
                    elif row["watch_band"] != band:
                        failures.append(
                            f"{label}: {ticket_id} band {row['watch_band']!r}, "
                            f"expected {band!r}"
                        )
                for key, wanted in (action.get("expect_counts") or {}).items():
                    if payload["counts"].get(key) != wanted:
                        failures.append(
                            f"{label}: count {key} = {payload['counts'].get(key)}, "
                            f"expected {wanted}"
                        )
                if "expect_open" in action and payload["open_runs"] != action["expect_open"]:
                    failures.append(
                        f"{label}: open_runs {payload['open_runs']}, "
                        f"expected {action['expect_open']}"
                    )
                for ticket_id, who in (action.get("expect_assignee") or {}).items():
                    row = by_id.get(ticket_id)
                    if row is None:
                        failures.append(f"{label}: {ticket_id} missing from the watch")
                        continue
                    if row["assignee"] != who["agent"]:
                        failures.append(
                            f"{label}: {ticket_id} assignee {row['assignee']!r}, "
                            f"expected {who['agent']!r}"
                        )
                    if row["assignee_source"] != who["source"]:
                        failures.append(
                            f"{label}: {ticket_id} assignee source "
                            f"{row['assignee_source']!r}, expected {who['source']!r}"
                        )
                pair = action.get("risk_ranks_opposite")
                if pair:
                    first, second = pair
                    one, two = by_id.get(first), by_id.get(second)
                    if one is None or two is None:
                        failures.append(f"{label}: risk pair {pair} not both on the watch")
                    else:
                        if not one["sla_breach_risk"] < two["sla_breach_risk"]:
                            failures.append(
                                f"{label}: premise broken — the model's risk "
                                f"does not disagree with the clock on {pair} "
                                f"({one['sla_breach_risk']} vs "
                                f"{two['sla_breach_risk']})"
                            )
                        if got_order.index(first) > got_order.index(second):
                            failures.append(
                                f"{label}: {first} ranks behind {second} — "
                                "the watch followed the model's risk, "
                                "not the clock"
                            )
                records_intact(label)
                clocks_unmoved(label)
            else:
                failures.append(f"{label}: unknown watch action {kind!r}")
        except Exception as exc:  # a refusal is a ValueError/KeyError by design
            if not (expect_error and expect_error in str(exc)):
                failures.append(f"{label}: raised unexpectedly: {exc}")
            else:
                records_intact(label)
                clocks_unmoved(label)
        else:
            if expect_error:
                failures.append(
                    f"{label}: should have been refused ({expect_error!r}) "
                    "but succeeded"
                )
    return failures


def check_handoff_case(service: TriageService, case: dict) -> list[str]:
    """A shift-handoff case (handoffs.py).

    Seeds a staged desk — each seed's decided band and placement
    asserted as a premise — then plays the case's
    ``handoff_actions`` at the case's fixed boundary instant:
    decisions and a cluster confirmation stage the world, a QA
    sample is drawn and partly graded, and handoffs are composed
    and judged section by section against the hand-derived
    figures the case carries. The handshake plays out in full:
    self-acceptance refused, a third party refused, acceptance
    with exceptions storing the dispute verbatim while the
    composed sections stay byte-identical, a newer page
    superseding an unaccepted one (which then refuses
    acceptance), and a final plain acceptance. After every action
    that is not itself a run-writing act, the stored run records
    are checked byte-identical against the snapshot: a handoff is
    a desk-level record, and a shift change moves no triage.
    """
    from support_triage.sla import format_timestamp, parse_timestamp

    failures: list[str] = []
    now = parse_timestamp(case["handoff_now"])
    if now is None:
        return [f"handoff_now {case['handoff_now']!r} does not parse"]

    for entry in case.get("handoff_seed", []):
        ticket = TicketInput.model_validate(entry["ticket"])
        result = service.run_ticket(ticket).result
        wanted_band = entry.get("expect_band")
        if wanted_band and result.composition.priority_band != wanted_band:
            failures.append(
                f"{ticket.ticket_id}: premise broken — decided band "
                f"{result.composition.priority_band!r}, expected "
                f"{wanted_band!r}; the handoff figures would prove nothing"
            )
        placement = entry.get("expect_placement") or {}
        for field in ("category", "product_area"):
            wanted = placement.get(field)
            if wanted and getattr(result, field) != wanted:
                failures.append(
                    f"{ticket.ticket_id}: premise broken — decided "
                    f"{field} {getattr(result, field)!r}, expected {wanted!r}"
                )

    snapshot = {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }

    def records_unchanged(label: str) -> None:
        current = {
            record.ticket_id: record.model_dump_json()
            for record in service.store.all_records()
        }
        for ticket_id, before in snapshot.items():
            if current.get(ticket_id) != before:
                failures.append(
                    f"{label}: run {ticket_id}'s record changed — a "
                    "handoff is a desk-level record and must never "
                    "write to a run"
                )
        snapshot.update(
            {tid: blob for tid, blob in current.items() if tid not in snapshot}
        )

    def proposal_for(members: list[str]) -> dict | None:
        wanted = sorted(members)
        for proposal in service.cluster_proposals()["proposals"]:
            if sorted(m["ticket_id"] for m in proposal["members"]) == wanted:
                return proposal
        return None

    def check_sections(label: str, handoff, expected: dict) -> None:
        sections = handoff.sections
        watch = sections["watch"]
        if watch["counts"] != expected["watch_counts"]:
            failures.append(
                f"{label}: watch counts {watch['counts']}, expected "
                f"{expected['watch_counts']}"
            )
        # The composition pin, in golden form: the page's counts
        # are the watch's own, recomputed at the same instant.
        live = service.breach_watch(now)
        if watch["counts"] != live["counts"]:
            failures.append(
                f"{label}: the page's watch counts {watch['counts']} "
                f"disagree with the watch itself ({live['counts']})"
            )
        if watch["open_runs"] != expected["watch_open"]:
            failures.append(
                f"{label}: watch open_runs {watch['open_runs']}, "
                f"expected {expected['watch_open']}"
            )
        top = [row["ticket_id"] for row in watch["top_rows"]]
        if top != expected["watch_top"]:
            failures.append(
                f"{label}: watch top rows {top}, expected {expected['watch_top']}"
            )
        if sections["queues"]["rows"] != expected["queues"]:
            failures.append(
                f"{label}: queue rows {sections['queues']['rows']}, "
                f"expected {expected['queues']}"
            )
        incidents = sections["incidents"]
        for key, wanted in expected["incidents"].items():
            if key in ("member_count", "open_members"):
                got = incidents["open"][0][key] if incidents["open"] else None
            else:
                got = incidents.get(key)
            if got != wanted:
                failures.append(
                    f"{label}: incidents {key} = {got}, expected {wanted}"
                )
        qa = sections["qa_review"]
        for key, wanted in expected["qa"].items():
            if qa.get(key) != wanted:
                failures.append(
                    f"{label}: qa {key} = {qa.get(key)}, expected {wanted}"
                )

    for index, action in enumerate(case.get("handoff_actions", [])):
        kind = action["type"]
        label = f"action {index} ({kind})"
        expect_error = action.get("expect_error")
        try:
            if kind == "decide":
                service.record_decision(
                    action["ticket"], action.get("decision", "approved"),
                    by=action.get("by", "eval-agent"),
                )
                snapshot.update(
                    {
                        record.ticket_id: record.model_dump_json()
                        for record in service.store.all_records()
                    }
                )
            elif kind == "confirm_cluster":
                proposal = proposal_for(action["members"])
                if proposal is None:
                    failures.append(
                        f"{label}: no live proposal for members "
                        f"{action['members']} to confirm"
                    )
                else:
                    incident = service.confirm_cluster(
                        proposal["proposal_id"],
                        actor=action.get("actor", "eval-lead"),
                    )
                    if action.get("expect_incident") and (
                        incident.incident_id != action["expect_incident"]
                    ):
                        failures.append(
                            f"{label}: confirmed as {incident.incident_id}, "
                            f"expected {action['expect_incident']}"
                        )
                records_unchanged(label)
            elif kind == "sample":
                sample = service.create_review_sample(
                    size=action["size"],
                    seed=action.get("seed", 0),
                    created_by=action.get("by", "qa-lead"),
                )
                if action.get("expect_sample") and (
                    sample.sample_id != action["expect_sample"]
                ):
                    failures.append(
                        f"{label}: sample {sample.sample_id}, expected "
                        f"{action['expect_sample']}"
                    )
                if "expect_drawn" in action and (
                    len(sample.run_ids) != action["expect_drawn"]
                ):
                    failures.append(
                        f"{label}: drew {len(sample.run_ids)} runs, "
                        f"expected {action['expect_drawn']}"
                    )
                records_unchanged(label)
            elif kind == "verdict":
                service.record_review_verdict(
                    action["sample"],
                    ticket_id=action["ticket"],
                    reviewer=action.get("reviewer", "qa-lead"),
                    families={
                        "category": "agree",
                        "priority_drivers": "agree",
                        "routing": "agree",
                        "verify": "agree",
                    },
                    overall="agree",
                )
                records_unchanged(label)
            elif kind == "create":
                handoff = service.create_handoff(
                    from_lead=action.get("from_lead", ""),
                    to_lead=action.get("to_lead", ""),
                    note=action.get("note", ""),
                    now=now,
                )
                if action.get("expect_handoff") and (
                    handoff.handoff_id != action["expect_handoff"]
                ):
                    failures.append(
                        f"{label}: created {handoff.handoff_id}, expected "
                        f"{action['expect_handoff']}"
                    )
                if handoff.as_of != format_timestamp(now):
                    failures.append(
                        f"{label}: as_of {handoff.as_of!r}, expected the "
                        f"boundary {format_timestamp(now)!r}"
                    )
                if handoff.status != "outstanding":
                    failures.append(
                        f"{label}: new handoff is {handoff.status}, "
                        "expected outstanding"
                    )
                if "expect_sections" in action:
                    check_sections(label, handoff, action["expect_sections"])
                if "expect_supersedes" in action:
                    older = service.get_handoff(action["expect_supersedes"])
                    if older["status"] != "superseded":
                        failures.append(
                            f"{label}: {action['expect_supersedes']} is "
                            f"{older['status']}, expected superseded"
                        )
                records_unchanged(label)
            elif kind == "accept":
                before = service.get_handoff(action["handoff"])
                accepted = service.accept_handoff(
                    action["handoff"],
                    actor=action.get("actor", ""),
                    exceptions=action.get("exceptions", ""),
                    now=now,
                )
                if accepted.status != "accepted":
                    failures.append(
                        f"{label}: handoff is {accepted.status} after "
                        "acceptance, expected accepted"
                    )
                if "expect_acceptance" in action and (
                    accepted.acceptance != action["expect_acceptance"]
                ):
                    failures.append(
                        f"{label}: acceptance {accepted.acceptance!r}, "
                        f"expected {action['expect_acceptance']!r}"
                    )
                if accepted.exceptions != action.get("exceptions", ""):
                    failures.append(
                        f"{label}: exceptions {accepted.exceptions!r}, "
                        f"expected {action.get('exceptions', '')!r}"
                    )
                if accepted.sections != before["sections"]:
                    failures.append(
                        f"{label}: the composed sections changed under "
                        "acceptance — exceptions ride beside the figures, "
                        "never over them"
                    )
                if accepted.accepted_at != format_timestamp(now):
                    failures.append(
                        f"{label}: accepted_at {accepted.accepted_at!r}, "
                        f"expected the boundary {format_timestamp(now)!r}"
                    )
                records_unchanged(label)
            elif kind == "expect_handoff":
                view = service.get_handoff(action["handoff"])
                if view["status"] != action["status"]:
                    failures.append(
                        f"{label}: {action['handoff']} is {view['status']}, "
                        f"expected {action['status']}"
                    )
                records_unchanged(label)
            else:
                failures.append(f"{label}: unknown handoff action {kind!r}")
        except Exception as exc:  # a refusal is a HandoffError by design
            if not (expect_error and expect_error in str(exc)):
                failures.append(f"{label}: raised unexpectedly: {exc}")
            else:
                records_unchanged(label)
        else:
            if expect_error:
                failures.append(
                    f"{label}: should have been refused ({expect_error!r}) "
                    "but succeeded"
                )
    return failures


_MISSING = object()


def check_template_governance_case(service, case: dict) -> list[str]:
    """Round 15 — template governance: the lifecycle, end to end.

    ``template_gov_seed`` runs premise tickets under the bundled
    defaults (each entry asserts the template it earns), and the
    governed records in the store are snapshotted — no governance
    act that follows may write to a run's record; the snapshot is
    re-checked at the end. ``template_gov_actions`` then walks
    the lifecycle in order: an ``author`` creates words in draft,
    ``run`` entries assert what selection did with the template
    in that state (a draft or retired template earns no draft,
    and the trace names the skip and the state; ``expect_skip``
    is the state), ``promote``/``edit``/``retire`` perform the
    acts — any of them may carry ``expect_error``, a fragment the
    refusal must name (self-promotion is the standing example) —
    and ``expect_view`` asserts the registry's public view of a
    template: its state, version, effectiveness flag, measured
    uses, and the retirement watch's verdict. Run entries may
    carry ``outcome`` (recorded as in the template case) and
    ``reopen`` (the run is reopened once) — the raw material the
    retirement watch reads, through round 11's own figures.
    """
    failures: list[str] = []

    def run_expect(entry: dict, where: str):
        ticket_in = TicketInput.model_validate(entry["ticket"])
        try:
            result = service.run_ticket(ticket_in).result
        except Exception as exc:  # noqa: BLE001 - a failed run is a failure
            failures.append(f"{where}: run failed: {exc}")
            return None
        want = entry.get("expect_template", _MISSING)
        if want is not _MISSING:
            got = result.draft.template if result.draft else None
            if got != want:
                failures.append(
                    f"{where}: expected template {want!r}, got {got!r}"
                )
        if entry.get("expect_template_version") is not None:
            got_version = (
                result.draft.template_version if result.draft else None
            )
            if got_version != entry["expect_template_version"]:
                failures.append(
                    f"{where}: expected template version "
                    f"{entry['expect_template_version']}, got {got_version}"
                )
        if entry.get("expect_text_contains"):
            text = result.draft.text if result.draft else ""
            if entry["expect_text_contains"] not in text:
                failures.append(
                    f"{where}: draft should carry "
                    f"{entry['expect_text_contains']!r}; got {text[:120]!r}"
                )
        if entry.get("expect_skip"):
            name = entry["expect_template_name"]
            line = (
                f"template draft: skipped — template {name} is "
                f"{entry['expect_skip']} (not live); no draft offered"
            )
            details = [
                d for step in result.trace for d in step.details
            ]
            if line not in details:
                failures.append(
                    f"{where}: trace should name the state skip "
                    f"{line!r}; details were {details}"
                )
        if entry.get("outcome"):
            try:
                _record_eval_outcome(
                    service, result.ticket_id, entry["outcome"]
                )
            except Exception as exc:  # noqa: BLE001 - asserted as a failure
                failures.append(f"{where}: outcome failed: {exc}")
        if entry.get("reopen"):
            try:
                service.reopen_run(
                    result.ticket_id,
                    message=(
                        "I followed the steps and it still does not "
                        "work — please look again."
                    ),
                    actor="eval-agent",
                )
            except Exception as exc:  # noqa: BLE001 - a failed act is a failure
                failures.append(f"{where}: reopen failed: {exc}")
        return result

    for entry in case.get("template_gov_seed", []):
        run_expect(entry, f"seed {entry['ticket']['ticket_id']}")

    snapshot = {
        record.ticket_id: record.model_dump_json()
        for record in service.store.all_records()
    }

    acts = {
        "author": lambda a: service.author_template(
            area=a["area"], text=a["text"], guide=a["guide"],
            actor=a["actor"], reason=a.get("reason", ""),
        ),
        "edit": lambda a: service.edit_template(
            a["name"], actor=a["actor"], text=a.get("text"),
            guide=a.get("guide"), reason=a.get("reason", ""),
        ),
        "promote": lambda a: service.promote_template(
            a["name"], actor=a["actor"], reason=a.get("reason", "")
        ),
        "retire": lambda a: service.retire_template(
            a["name"], actor=a["actor"], reason=a.get("reason", "")
        ),
    }
    for index, action in enumerate(case.get("template_gov_actions", [])):
        kind = action["type"]
        where = f"action {index} ({kind})"
        if kind == "run":
            run_expect(action, where)
            continue
        if kind == "expect_view":
            views = {
                row["name"]: row
                for row in service.list_templates()["templates"]
            }
            view = views.get(action["name"])
            if view is None:
                failures.append(f"{where}: no view for {action['name']!r}")
                continue
            for field in ("state", "version", "source", "retirement_due"):
                if field in action and view[field] != action[field]:
                    failures.append(
                        f"{where}: view {field} should be "
                        f"{action[field]!r}, got {view[field]!r}"
                    )
            eff = view.get("effectiveness") or {}
            if "flag" in action and eff.get("flag", "") != action["flag"]:
                failures.append(
                    f"{where}: effectiveness flag should be "
                    f"{action['flag']!r}, got {eff.get('flag')!r}"
                )
            if "uses" in action and eff.get("uses") != action["uses"]:
                failures.append(
                    f"{where}: effectiveness uses should be "
                    f"{action['uses']}, got {eff.get('uses')}"
                )
            continue
        if kind not in acts:
            failures.append(f"{where}: unknown action type {kind!r}")
            continue
        try:
            record = acts[kind](action)
        except Exception as exc:  # noqa: BLE001 - refusals are asserted on
            if action.get("expect_error"):
                if action["expect_error"] not in str(exc):
                    failures.append(
                        f"{where}: refusal should name "
                        f"{action['expect_error']!r}; got {exc}"
                    )
            else:
                failures.append(f"{where}: act failed: {exc}")
            continue
        if action.get("expect_error"):
            failures.append(
                f"{where}: expected refusal naming "
                f"{action['expect_error']!r}; the act succeeded"
            )
            continue
        for field in ("state", "version"):
            if field in action.get("expect_record", {}):
                got = getattr(record, field)
                want = action["expect_record"][field]
                if got != want:
                    failures.append(
                        f"{where}: record {field} should be {want!r}, "
                        f"got {got!r}"
                    )

    for ticket_id, frozen in snapshot.items():
        current = service.get_record(ticket_id).model_dump_json()
        if current != frozen:
            failures.append(
                f"governance wrote to run record {ticket_id} — records "
                "are the past and stay byte-identical under every act"
            )
    return failures


def main() -> int:
    cases = load_cases()
    service = fresh_service()
    passed = 0
    failures_by_case: dict[str, list[str]] = {}
    for case in cases:
        # A fresh service per case keeps the duplicate index honest:
        # only a case's own `previous` ticket may link to it.
        if "template_gov_seed" in case:
            failures = check_template_governance_case(fresh_service(), case)
        elif "template_seed" in case:
            failures = check_template_case(fresh_service(), case)
        elif "handoff_seed" in case:
            failures = check_handoff_case(fresh_service(), case)
        elif "watch_seed" in case:
            failures = check_watch_case(fresh_service(), case)
        elif "incident_seed" in case:
            failures = check_incident_case(fresh_service(), case)
        elif "qa_sample" in case:
            failures = check_review_case(fresh_service(), case)
        elif "dossier_seed" in case or "dossier_new_ticket" in case:
            failures = check_dossier_case(fresh_service(), case)
        elif "operator_actions" in case:
            failures = check_operator_case(fresh_service(), case)
        elif "neighbours" in case or "workbench_actions" in case:
            failures = check_workbench_case(fresh_service(), case)
        elif "thread_pair" in case:
            failures = check_thread_case(fresh_service(), case)
        else:
            result, chain_failures = run_case(fresh_service(), case)
            failures = chain_failures + check_case(case, result)
        if failures:
            failures_by_case[case["id"]] = failures
        else:
            passed += 1
    score = passed / len(cases) if cases else 0.0
    payload = {
        "backend": "offline",
        "total": len(cases),
        "passed": passed,
        "score": round(score, 4),
        "gate": GATE,
        "failures": failures_by_case,
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src" / "support_triage" / "data" / "eval_results.json"
    )
    packaged.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"golden routing eval (backend=offline): {passed}/{len(cases)} passed ({score:.1%})")
    for case_id, failures in failures_by_case.items():
        for failure in failures:
            print(f"  FAIL {case_id}: {failure}")
    if score < GATE:
        print(f"GATE FAILED: score {score:.3f} < {GATE}")
        return 1
    print(f"gate: PASS (>= {GATE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
