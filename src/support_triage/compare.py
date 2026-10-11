"""Backend comparison: one ticket, several decision engines, side by side.

This is the System One showcase. The same ticket state is put to
each selected backend behind the port — offline always, plus
whichever of jev / llm / a provider alias is configured — and each
backend's answers are composed by the *same* code (priority,
routing, guardrails). What differs between columns is only the
decision model; what the business does with the answers cannot
drift between backends, because it is not the backends' business.

A comparison is an experiment, not a triage: it does NOT run the
graph, does NOT queue a run, and does NOT touch the duplicate index
or the agent gate. The screened ticket text is the only pipeline
step shared with a real run — the injection screen is not optional,
even for an experiment. The record is archived to the object store
under ``comparisons/{id}.json``, deliberately apart from the queue's
records (no queue table, no migration).

The diff is computed against the first backend that answered (the
reference column): category and product-area flips, queue and band
changes, verify flags that flip raised-ness at the composition
threshold, and review-forcing differences — the disagreements an
operator would actually argue about.
"""

from __future__ import annotations

import uuid

from . import composition as comp
from . import guardrails as guard
from . import screening
from .backends import get_decision_engine
from .decisions import DecisionSet
from .errors import ProviderError
from .questions import triage_questions
from .schemas import TicketInput

# The candidate backends a comparison may select from. "offline" is
# always constructible; the others fail loudly (ProviderError) when
# unconfigured, and the comparison reports that per column instead of
# sinking the whole experiment.
CANDIDATE_BACKENDS = ("offline", "jev", "llm", "openai", "anthropic", "ollama")


def available_backends() -> list[dict]:
    """Which candidate backends can be constructed right now.

    Construction is attempted (cheap for every adapter: clients are
    built lazily per call) so an unconfigured backend reports its
    honest reason here rather than at selection time.
    """
    available = []
    for name in CANDIDATE_BACKENDS:
        try:
            engine = get_decision_engine(name)
        except Exception as exc:
            available.append({"name": name, "available": False, "reason": str(exc)})
            continue
        available.append({"name": name, "available": True, "model": getattr(engine, "model", "")})
    return available


def screen_state_text(ticket: TicketInput) -> str:
    """The state text a decision call reads — the graph's screen step.

    Flagged sentences are removed and the surviving text fenced,
    exactly as in a real run, so a comparison measures the same
    inputs a triage would decide on.
    """
    clean_body = screening.sanitize_text(ticket.body)
    header = [f"Subject: {ticket.subject}", f"Channel: {ticket.channel}"]
    if ticket.account_tier:
        header.append(f"Account tier: {ticket.account_tier}")
    return "\n".join(header) + "\n" + screening.fence(clean_body)


def _run_backend(engine, state_text: str, questions) -> dict:
    decisions: DecisionSet = engine.decide(state_text, questions)
    composed = comp.compose(decisions)
    verdict = guard.run_guardrails(decisions, [q.key for q in questions])
    answers = decisions.as_dict()
    return {
        "backend": decisions.backend or getattr(engine, "name", ""),
        "model": decisions.model,
        "latency_ms": decisions.latency_ms,
        "category": decisions.choice("category"),
        "product_area": decisions.choice("product_area"),
        "answers": {
            key: {
                "kind": entry["kind"],
                "value": entry["value"],
                "confidence": entry["confidence"],
            }
            for key, entry in answers.items()
        },
        "priority_score": composed.priority_score,
        "priority_band": composed.priority_band,
        "queue": composed.queue,
        "override": composed.override,
        "flags": dict(composed.flags),
        "guardrails": {
            "passed": verdict.passed,
            "requires_human_review": verdict.requires_human_review,
            "failed_checks": [
                check.name for check in verdict.checks if not check.passed
            ],
        },
        "usage": decisions.usage,
    }


def _diff(reference: dict, other: dict) -> list[dict]:
    """Where one backend's column disagrees with the reference."""
    disagreements: list[dict] = []

    def note(field: str, ref_value, other_value):
        disagreements.append(
            {"backend": other["backend"], "field": field, "reference": ref_value, "value": other_value}
        )

    for field in ("category", "product_area", "queue", "priority_band"):
        if other[field] != reference[field]:
            note(field, reference[field], other[field])
    if abs(other["priority_score"] - reference["priority_score"]) > 0.05:
        note("priority_score", reference["priority_score"], other["priority_score"])
    if other["guardrails"]["requires_human_review"] != reference["guardrails"]["requires_human_review"]:
        note(
            "requires_human_review",
            reference["guardrails"]["requires_human_review"],
            other["guardrails"]["requires_human_review"],
        )
    for key, raised in other["flags"].items():
        if reference["flags"].get(key) != raised:
            note(f"flag:{key}", reference["flags"].get(key), raised)
    return disagreements


def compare_ticket(ticket: TicketInput, backend_names: list[str] | None = None) -> dict:
    """Run one ticket's decision+composition under several backends.

    ``backend_names=None`` selects every candidate that constructs;
    an explicit list is honoured as given (an unconfigured name
    becomes an error column, reported — not hidden). The first
    backend that answers is the diff's reference.
    """
    questions = triage_questions()
    state_text = screen_state_text(ticket)
    if backend_names is None:
        names = [entry["name"] for entry in available_backends() if entry["available"]]
    else:
        names = list(backend_names)
    columns: list[dict] = []
    for name in names:
        try:
            engine = get_decision_engine(name)
        except Exception as exc:
            columns.append({"backend": name, "error": f"not configured: {exc}"})
            continue
        try:
            columns.append(_run_backend(engine, state_text, questions))
        except ProviderError as exc:
            columns.append({"backend": name, "error": str(exc)})
        except Exception as exc:  # an adapter bug must not sink the comparison
            columns.append({"backend": name, "error": f"{type(exc).__name__}: {exc}"})
    answered = [c for c in columns if "error" not in c]
    reference = answered[0] if answered else None
    disagreements: list[dict] = []
    if reference is not None:
        for column in answered[1:]:
            disagreements.extend(_diff(reference, column))
    return {
        "comparison_id": uuid.uuid4().hex[:12],
        "ticket_id": ticket.ticket_id,
        "subject": ticket.subject,
        "reference_backend": reference["backend"] if reference else None,
        "backends": columns,
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
    }
