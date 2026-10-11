"""Golden routing eval: known tickets in, routing truth checked.

Fifty-one synthetic tickets with authored expectations — category,
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
calibration accounting, and the correction refusals.)
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_triage import corrections  # noqa: E402
from support_triage.backends import OfflineDecisionEngine  # noqa: E402
from support_triage.embeddings import HashingEmbedder, InMemoryVectorIndex  # noqa: E402
from support_triage.livecal import live_calibration  # noqa: E402
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


def main() -> int:
    cases = load_cases()
    service = fresh_service()
    passed = 0
    failures_by_case: dict[str, list[str]] = {}
    for case in cases:
        # A fresh service per case keeps the duplicate index honest:
        # only a case's own `previous` ticket may link to it.
        if "operator_actions" in case:
            failures = check_operator_case(fresh_service(), case)
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
