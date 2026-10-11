"""Run audit export: one run, as one portable document.

A decision record lives in the store as JSON, and the console shows
it well — but an audit, an incident review, or a customer's "show me
exactly why this ticket went there" does not happen in the console.
This module renders one stored run as a single self-contained
document, in JSON or in plain text, that can leave the system and
still answer every question about the run:

- what came in (intake, with the injection screen's verdict — the
  ticket text is untrusted, and the export says what the screen did
  to it);
- what the decision layer answered (all eleven questions, values,
  confidences, full distributions);
- how code composed those answers (the priority formula with its
  weights, each term's input and points, the score arithmetic
  checked; the routing matrix cell and any override that fired);
- what the guardrails said, check by check;
- where the SLA clock stands, what the drift watch currently says
  about the questions this run used, and — when an agent recorded
  one — the outcome that judged the run;
- for a reopen run, the chain context: which run it follows, the
  count against the reopen limit, what the decision backend did
  with the prior run's record, and the priority floor it applied;
- the operator overlay (corrections.py): the thread grouping as
  computed and as corrected (with each correction's actor and
  reason), and any confirmed-duplicate mark — the machine's
  duplicate guess and the human's confirmation are separate blocks,
  because they are separate judgements.

Everything here is composed in code from the stored record. No
model is consulted and no prose is generated: the text rendering is
a formatting of the same dict the JSON returns, so the two cannot
disagree (the suite pins the key facts in both).
"""

from __future__ import annotations

from . import composition as comp
from . import corrections
from .schemas import TicketResult

# The formula terms and the answer each one reads, for the trace.
# (anger and abuse are derived terms — sentiment inverted, the
# is_abusive probability itself — spelled out so the math can be
# followed without opening composition.py.)
_TERM_BASIS = {
    "urgency": "urgency score ÷ 10",
    "sla_breach_risk": "sla_breach_risk score ÷ 10",
    "churn_risk": "churn_risk score ÷ 10",
    "anger": "(10 − sentiment score) ÷ 10",
    "abuse": "P(is_abusive)",
}


def _term_inputs(record: TicketResult) -> dict[str, float]:
    def value(key: str) -> float:
        answer = record.answer(key)
        return float(answer.value) if answer is not None else 0.0

    return {
        "urgency": value("urgency") / 10.0,
        "sla_breach_risk": value("sla_breach_risk") / 10.0,
        "churn_risk": value("churn_risk") / 10.0,
        "anger": (10.0 - value("sentiment")) / 10.0,
        "abuse": value("is_abusive"),
    }


def build_export(
    record: TicketResult,
    *,
    exported_at: str,
    ticket_input=None,
    sla: dict | None = None,
    drift: dict | None = None,
) -> dict:
    """Assemble the audit document for one stored run (pure)."""
    composition = record.composition
    inputs = _term_inputs(record)
    terms = []
    for term, weight in composition.weights.items():
        terms.append(
            {
                "term": term,
                "basis": _TERM_BASIS.get(term, ""),
                "input": round(inputs.get(term, 0.0), 4),
                "weight": weight,
                "points": composition.contributions.get(term, 0.0),
            }
        )
    terms_total = round(sum(t["points"] for t in terms), 2)
    boost_points = comp.OUTAGE_BOOST if composition.outage_boost_applied else 0.0

    intake = {
        "ticket_id": record.ticket_id,
        "subject": record.subject,
        "sender_name": record.sender_name,
        "channel": record.channel,
        "received_at": record.received_at,
        "status": record.status,
        "thread_id": record.thread_id,
    }
    if ticket_input is not None:
        intake.update(
            {
                "sender_email": ticket_input.sender_email,
                "account_tier": ticket_input.account_tier,
                "body": ticket_input.body,
                "previous_ticket_ids": list(ticket_input.previous_ticket_ids),
            }
        )

    if record.injection_flags:
        screen_note = (
            "patterns flagged above were removed from the text before "
            "any model read it; the surviving text was fenced as untrusted data"
        )
    else:
        screen_note = "no injection patterns flagged; the text was still fenced as untrusted data"

    reopen = None
    if record.reopen is not None:
        info = record.reopen
        reopen = {
            "prior_run_id": info.prior_run_id,
            "root_ticket_id": info.root_ticket_id,
            "reopen_count": info.reopen_count,
            "reopen_limit": info.reopen_limit,
            "limit_hit": info.limit_hit,
            "reopened_by": info.reopened_by,
            "reopened_at": info.reopened_at,
            "context": {
                "provided": info.context_provided,
                "use": info.context_use,
                "note": (
                    "the prior run's answers and outcome were offered to the "
                    "decision call as fenced history"
                    if info.context_use == "history_block"
                    else "the deterministic offline backend does not read "
                    "history — its answers come from the new message alone, "
                    "and the record says so"
                    if info.context_use == "ignored_offline"
                    else "no context was provided"
                ),
            },
            "template_draft_suppressed": info.template_suppressed,
            "priority_floor": {
                "applied": composition.reopen_floor_applied,
                "band_before": composition.band_before_floor,
                "band_after": composition.priority_band,
            },
        }

    grouping = corrections.grouping_view(record)
    active_mark = corrections.active_duplicate_mark(record)
    confirmed_duplicate = None
    if record.duplicate_marks:
        confirmed_duplicate = {
            "duplicate_of": record.duplicate_of,
            "active": active_mark is not None,
            "marks": [mark.model_dump() for mark in record.duplicate_marks],
            "sampling": (
                "excluded from calibration sampling and the drift window "
                "while the mark stands; still counted in volume stats"
                if active_mark is not None
                else "mark lifted — the run samples normally again"
            ),
        }

    document = {
        "kind": "run_audit_export",
        "export_version": 1,
        "ticket_id": record.ticket_id,
        "exported_at": exported_at,
        "status": record.status,
        "external_action_taken": record.external_action_taken,
        "intake": intake,
        "grouping": grouping,
        "confirmed_duplicate": confirmed_duplicate,
        "reopen": reopen,
        "injection_screen": {
            "flagged": bool(record.injection_flags),
            "flags": list(record.injection_flags),
            "note": screen_note,
        },
        "duplicate": record.duplicate.model_dump(),
        "answers": [
            {
                "key": answer.key,
                "kind": answer.kind,
                "value": answer.value,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
            }
            for answer in record.answers
        ],
        "composition": {
            "priority_score": composition.priority_score,
            "priority_band": composition.priority_band,
            "terms": terms,
            "terms_total": terms_total,
            "outage_boost_applied": composition.outage_boost_applied,
            "outage_boost_points": boost_points,
            "reopen_floor_applied": composition.reopen_floor_applied,
            "band_before_floor": composition.band_before_floor,
            "score_check": round(terms_total + boost_points, 1)
            == round(composition.priority_score, 1),
            "routing": {
                "category": record.category,
                "product_area": record.product_area,
                "matrix_cell": f"{record.category} × {record.product_area}",
                "queue": composition.queue,
                "override": composition.override,
                "reason": composition.routing_reason,
            },
        },
        "guardrails": {
            "passed": record.guardrails.passed,
            "requires_human_review": record.guardrails.requires_human_review,
            "human_review_reasons": list(record.guardrails.human_review_reasons),
            "checks": [dict(check) for check in record.guardrails.checks],
        },
        "autonomy": record.autonomy.model_dump(),
        "sla": sla,
        "drift": drift,
        "agent_decision": {
            "decision": record.decision,
            "by": record.decision_by,
            "reason": record.decision_reason,
        }
        if record.decision
        else None,
        "outcome": record.outcome.model_dump() if record.outcome is not None else None,
        "draft": record.draft.model_dump() if record.draft is not None else None,
        "telemetry": record.telemetry.model_dump() if record.telemetry is not None else None,
        "trace": [step.model_dump() for step in record.trace],
        "note": (
            "composed by code from the stored decision record — "
            "no model text; nothing in this document was sent to the customer"
        ),
    }
    return document


def _fmt(value, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_text(document: dict) -> str:
    """Render the export dict as plain text (the handoff idiom:
    one assembly, two renderings, so they cannot drift apart)."""
    intake = document["intake"]
    screen = document["injection_screen"]
    lines: list[str] = []
    lines.append(f"RUN AUDIT EXPORT — {document['ticket_id']}")
    lines.append(
        f"Exported: {document['exported_at']} · Status: {document['status']} · "
        f"External action taken: {'yes' if document['external_action_taken'] else 'no (never)'}"
    )
    lines.append("")
    lines.append("INTAKE")
    lines.append(f"Subject: {intake['subject']}")
    sender = intake["sender_name"] or "unknown"
    if intake.get("sender_email"):
        sender += f" <{intake['sender_email']}>"
    lines.append(f"Sender: {sender} · Channel: {intake['channel']} · Received: {intake['received_at']}")
    if intake.get("account_tier"):
        lines.append(f"Account tier: {intake['account_tier']}")
    if intake.get("body"):
        lines.append("Body:")
        lines.append(intake["body"])
    lines.append("")
    lines.append("INJECTION SCREEN")
    if screen["flagged"]:
        lines.append("Flagged patterns (removed before any model read the ticket):")
        for flag in screen["flags"]:
            lines.append(f"  - {flag}")
    else:
        lines.append("No injection patterns flagged.")
    lines.append(f"Note: {screen['note']}")
    reopen = document.get("reopen")
    if reopen:
        ctx = reopen["context"]
        floor = reopen["priority_floor"]
        lines.append("")
        lines.append("REOPEN")
        lines.append(
            f"Reopen #{reopen['reopen_count']} of prior run {reopen['prior_run_id']} "
            f"(chain root {reopen['root_ticket_id']}, limit {reopen['reopen_limit']}"
            + (", LIMIT HIT — forced to human review" if reopen["limit_hit"] else "")
            + f"), logged by {reopen['reopened_by'] or 'agent'} at {reopen['reopened_at']}"
        )
        lines.append(
            "Context: "
            + (
                f"prior run's record offered as fenced history ({ctx['note']})"
                if ctx["use"] == "history_block"
                else ctx["note"]
            )
        )
        lines.append(
            "Template draft: suppressed — a reopen never gets the canned draft"
            if reopen["template_draft_suppressed"]
            else "Template draft: not suppressed"
        )
        lines.append(
            f"Priority floor: band {floor['band_before']} → {floor['band_after']}"
            if floor["band_before"] != floor["band_after"]
            else "Priority floor: applied (band already at the floor's step or above)"
        )
        if intake.get("thread_id"):
            lines.append(f"Thread: {intake['thread_id']}")

    grouping = document.get("grouping") or {}
    if grouping.get("corrected"):
        lines.append("")
        lines.append("GROUPING (operator-corrected)")
        lines.append(
            f"Thread as grouped at intake: {grouping['computed_thread_id']} · "
            f"thread as corrected: {grouping['thread_id']}"
        )
        for entry in grouping["corrections"]:
            lines.append(
                f"  {entry['action']}: {entry['from_thread_id']} → {entry['to_thread_id']} "
                f"by {entry['by'] or 'operator'} at {entry['at']} — {entry['reason']}"
            )
        lines.append(
            "The computed grouping is preserved above; the correction is an "
            "overlay and changed no decision on any run."
        )

    duplicate = document["duplicate"]
    lines.append("")
    lines.append("DUPLICATE CHECK")
    if duplicate["is_duplicate"]:
        lines.append(
            f"Probable duplicate of {duplicate['similar_ticket_id']} "
            f"(similarity {_fmt(duplicate['similarity'], 3)} ≥ threshold {_fmt(duplicate['threshold'], 2)})"
        )
    else:
        lines.append(
            f"Not a duplicate (nearest similarity {_fmt(duplicate['similarity'], 3)}, "
            f"threshold {_fmt(duplicate['threshold'], 2)})"
        )

    confirmed = document.get("confirmed_duplicate")
    if confirmed:
        lines.append("")
        lines.append("CONFIRMED DUPLICATE (operator)")
        for mark in confirmed["marks"]:
            lines.append(
                f"Marked as a duplicate of {mark['duplicate_of']} by "
                f"{mark['marked_by'] or 'operator'} at {mark['marked_at']} — {mark['reason']}"
            )
            if mark["unmarked_at"]:
                lines.append(
                    f"  mark lifted by {mark['unmarked_by'] or 'operator'} at "
                    f"{mark['unmarked_at']} — {mark['unmark_reason']}"
                )
        lines.append(f"Sampling: {confirmed['sampling']}")

    telemetry = document["telemetry"] or {}
    lines.append("")
    lines.append(
        "DECISION ANSWERS"
        + (
            f" (backend {telemetry.get('backend', '')}, model {telemetry.get('model', '')}, "
            f"{telemetry.get('question_count', len(document['answers']))} questions, one call)"
            if telemetry
            else ""
        )
    )
    for answer in document["answers"]:
        value = answer["value"]
        shown = _fmt(value) if isinstance(value, float) else str(value)
        lines.append(
            f"{answer['key']} ({answer['kind']}): {shown} — confidence {_fmt(answer['confidence'])}"
        )
        dist = " · ".join(
            f"{option} {_fmt(p)}"
            for option, p in sorted(
                answer["probabilities"].items(), key=lambda kv: -kv[1]
            )
        )
        if dist:
            lines.append(f"  {dist}")

    composed = document["composition"]
    routing = composed["routing"]
    lines.append("")
    lines.append("COMPOSITION (in code)")
    lines.append(
        f"Priority: {_fmt(composed['priority_score'], 1)} ({composed['priority_band']})"
    )
    for term in composed["terms"]:
        lines.append(
            f"  {term['term']}: {term['basis']} = {_fmt(term['input'], 4)} "
            f"× weight {_fmt(term['weight'])} × 100 = {_fmt(term['points'])} points"
        )
    if composed["outage_boost_applied"]:
        lines.append(
            f"  outage boost: +{_fmt(composed['outage_boost_points'], 1)} "
            "(P(is_outage_report) ≥ 0.50)"
        )
    if composed.get("reopen_floor_applied"):
        lines.append(
            f"  reopen floor: band {composed['band_before_floor']} → "
            f"{composed['priority_band']} (reopen runs are floored one band up)"
        )
    lines.append(
        f"  total: {_fmt(composed['terms_total'])} + {_fmt(composed['outage_boost_points'], 1)} "
        f"= {_fmt(composed['priority_score'], 1)}"
        + ("" if composed["score_check"] else "  [DOES NOT RECONCILE]")
    )
    lines.append(
        f"Routing: matrix cell {routing['matrix_cell']} → {routing['queue']}"
    )
    if routing["override"]:
        lines.append(f"Override fired: {routing['override']}")
    lines.append(f"Reason: {routing['reason']}")

    guardrails = document["guardrails"]
    lines.append("")
    lines.append("GUARDRAILS")
    lines.append(
        f"Overall: {'passed' if guardrails['passed'] else 'FAILED'} · "
        f"human review: {'forced' if guardrails['requires_human_review'] else 'not forced'}"
    )
    for reason in guardrails["human_review_reasons"]:
        lines.append(f"  review reason: {reason}")
    for check in guardrails["checks"]:
        lines.append(
            f"  {'PASS' if check.get('passed') else 'FAIL'} {check.get('name', '')} — {check.get('detail', '')}"
        )

    autonomy = document["autonomy"]
    lines.append("")
    lines.append("AUTONOMY RECOMMENDATION (advisory only — never acted on)")
    lines.append(f"Eligible: {'yes' if autonomy['eligible'] else 'no'}")
    for reason in autonomy["reasons"]:
        lines.append(f"  for: {reason}")
    for blocker in autonomy["blockers"]:
        lines.append(f"  blocker: {blocker}")

    lines.append("")
    lines.append("SLA CLOCK (deterministic, at export time)")
    sla = document["sla"]
    if sla is None:
        lines.append(f"No running clock — the ticket is {document['status']}, not awaiting an agent.")
    else:
        lines.append(
            f"State: {sla['state']} · target {_fmt(sla['target_hours'], 1)}h for band "
            f"{composed['priority_band']} · due at {sla['due_at']} · "
            f"{_fmt(sla['seconds_to_breach'], 0)}s to breach"
        )

    lines.append("")
    lines.append("DRIFT CONTEXT")
    drift = document["drift"]
    if not drift:
        lines.append("Not assessed for this export.")
    else:
        lines.append(drift["note"])
        if drift["flagged_questions"]:
            for key in drift["flagged_questions"]:
                row = drift["questions"].get(key) or {}
                lines.append(
                    f"  FLAGGED {key}: window Brier {_fmt(row.get('window_brier'), 3)} "
                    f"vs eval baseline {_fmt(row.get('eval_brier'), 3)} "
                    f"(window n={row.get('window_n')})"
                )
        else:
            lines.append("No question this run answered is currently flagged for drift.")

    lines.append("")
    lines.append("AGENT DECISION")
    decision = document["agent_decision"]
    if decision is None:
        lines.append("None recorded — the run has not been decided at the gate.")
    else:
        lines.append(
            f"{decision['decision']} by {decision['by'] or 'agent'}"
            + (f" — {decision['reason']}" if decision["reason"] else "")
        )

    lines.append("")
    lines.append("OUTCOME (agent ground truth)")
    outcome = document["outcome"]
    if outcome is None:
        lines.append("No outcome recorded yet.")
    else:
        lines.append(
            f"Final: {outcome['final_category'] or '—'} → {outcome['final_queue'] or '—'} · "
            f"{'resolved' if outcome['resolved'] else 'not resolved'} · "
            f"recorded by {outcome['recorded_by'] or 'agent'} at {outcome['recorded_at']}"
        )
        if outcome["verify_truth"]:
            lines.append(
                "  verify truth: "
                + " · ".join(f"{k}={v}" for k, v in sorted(outcome["verify_truth"].items()))
            )
        if outcome["score_truth"]:
            lines.append(
                "  score truth: "
                + " · ".join(f"{k}={v}" for k, v in sorted(outcome["score_truth"].items()))
            )

    if document["draft"]:
        lines.append("")
        lines.append("TEMPLATE DRAFT (for the agent — never sent by the system)")
        lines.append(document["draft"]["text"])

    lines.append("")
    lines.append("TRACE")
    for step in document["trace"]:
        lines.append(f"- {step['title']} ({_fmt(step['duration_ms'])} ms)")
        for detail in step["details"]:
            lines.append(f"    {detail}")

    lines.append("")
    lines.append(f"Note: {document['note']}")
    return "\n".join(lines) + "\n"
