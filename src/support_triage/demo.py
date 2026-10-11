"""Demo: run one bundled ticket through the pipeline, traced.

Default mode needs no keys and no network — the offline decision
backend, in-memory stores — and prints the run as a narrated trace:
input, screening, duplicate check, the eleven answers with their
distributions, the composition math, the guardrail verdicts, and the
final waiting state. ``--index N`` picks another sample ticket;
``--decision approved`` also records an agent's approval at the gate.

With ``DECISION_BACKEND`` set to a real backend (jev / llm / a
provider alias) the same trace shows the real engine's answers.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_dotenv, silence_langchain_deprecation_warnings

silence_langchain_deprecation_warnings()

from .events import PrintingSink  # noqa: E402
from .samples import load_sample_tickets  # noqa: E402
from .service import TriageService  # noqa: E402
from .wiring import build_service  # noqa: E402


def _bar(probability: float, width: int = 24) -> str:
    filled = int(round(probability * width))
    return "█" * filled + "░" * (width - filled)


def describe_backend(service: TriageService) -> str:
    engine = service.engine
    if engine.name == "offline":
        return (
            "DECISION BACKEND: offline heuristics (DECISION_BACKEND=offline) — "
            "deterministic keyword/rubric rules, the labelled plumbing backend. "
            "Set DECISION_BACKEND=jev or =llm in .env for a real decision model."
        )
    return f"DECISION BACKEND: {engine.name} (model: {getattr(engine, 'model', '') or getattr(engine, 'provider', '')})"


def print_run(service: TriageService, ticket, run) -> None:
    result = run.result
    print(describe_backend(service))
    print()
    print("[1] INPUT")
    print(f"    Ticket: {ticket.ticket_id} · {ticket.channel} · from {ticket.sender_name}")
    print(f"    Subject: {ticket.subject}")
    print(f"    Body: {ticket.body[:160]}")
    print()
    print("[2] SCREEN + DUPLICATE CHECK")
    if result.injection_flags:
        for flag in result.injection_flags:
            print(f"    injection flagged: {flag}")
    else:
        print("    injection: none found")
    duplicate = result.duplicate
    if duplicate.is_duplicate:
        print(
            f"    duplicate: probable re-send of {duplicate.similar_ticket_id} "
            f"(similarity {duplicate.similarity:.3f} ≥ {duplicate.threshold})"
        )
    else:
        print(f"    duplicate: no (nearest similarity {duplicate.similarity:.3f})")
    print()
    print("[3] DECISIONS — eleven typed answers, distributions not prose")
    for answer in result.answers:
        if answer.kind == "choice":
            top = sorted(answer.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
            rendered = "  ".join(f"{k} {_bar(p, 12)} {p:.2f}" for k, p in top)
            print(f"    {answer.key:<16} → {answer.value:<16} {rendered}")
        elif answer.kind == "score":
            print(
                f"    {answer.key:<16} → expected {float(answer.value):.1f}/10 "
                f"(confidence {answer.confidence:.2f})"
            )
        else:
            print(
                f"    {answer.key:<16} → P(true) {float(answer.value):.2f} "
                f"{_bar(float(answer.value), 16)}"
            )
    telemetry = result.telemetry
    if telemetry:
        print(
            f"    backend {telemetry.backend} · decision latency "
            f"{telemetry.decision_latency_ms} ms · {telemetry.question_count} questions in one call"
        )
    print()
    print("[4] COMPOSITION — priority and routing, computed in code")
    comp = result.composition
    terms = " + ".join(
        f"{name} {value}" for name, value in comp.contributions.items()
    )
    print(f"    priority = {terms}" + (" + outage boost 15" if comp.outage_boost_applied else ""))
    print(f"    priority score: {comp.priority_score} ({comp.priority_band})")
    print(f"    queue: {comp.queue}")
    print(f"    routing: {comp.routing_reason}")
    print()
    print("[5] GUARDRAILS")
    for check in result.guardrails.checks:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"    [{mark}] {check['name']}: {check['detail']}")
    if result.guardrails.requires_human_review:
        print(f"    human review forced: {'; '.join(result.guardrails.human_review_reasons)}")
    print()
    print("[6] AUTONOMY + DRAFT (advisory only)")
    autonomy = result.autonomy
    print(f"    auto-resolvable class: {'yes' if autonomy.eligible else 'no'}")
    for blocker in autonomy.blockers:
        print(f"      - {blocker}")
    if result.draft:
        print(f"    template draft offered ({result.draft.template}) — an agent may send it; the system does not.")
    else:
        print("    template draft: none (this ticket is outside the safe slice)")
    print()
    print("[7] FINAL STATE")
    print(f"    status: {result.status}")
    print("    The record waits at the agent gate. Nothing was sent to the customer.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support-triage-demo")
    parser.add_argument("--index", type=int, default=0, help="sample ticket index (default 0)")
    parser.add_argument("--decision", choices=["approved", "rejected"], default=None)
    parser.add_argument("--by", default="demo-agent")
    args = parser.parse_args(argv)

    load_dotenv()
    tickets = load_sample_tickets()
    if not (0 <= args.index < len(tickets)):
        print(f"error: --index must be between 0 and {len(tickets) - 1}")
        return 1
    ticket = tickets[args.index]
    service = build_service()
    try:
        run = service.run_ticket(ticket, sink=PrintingSink())
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print()
    print_run(service, ticket, run)
    if args.decision:
        record = service.record_decision(ticket.ticket_id, args.decision, by=args.by)
        print()
        print(f"    agent decision recorded: {args.decision} by {args.by} → status {record.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
