"""Shared builders for the suite: tickets, canned answers, fake transports."""

from __future__ import annotations

import json

from support_triage.decisions import (
    ChoiceQuestion,
    ScoreQuestion,
    VerifyQuestion,
)
from support_triage.questions import triage_questions
from support_triage.schemas import TicketInput
from support_triage.service import TriageService
from support_triage.store import InMemoryStore
from support_triage.objectstore import InMemoryObjectStore
from support_triage.embeddings import HashingEmbedder, InMemoryVectorIndex


def make_ticket(**overrides) -> TicketInput:
    base = {
        "ticket_id": "T-1",
        "subject": "Charged twice for the March subscription",
        "body": (
            "Hi, I was charged twice for the March subscription — two identical "
            "payments of $49 left my card. Please refund one of them. Thanks."
        ),
        "sender_name": "Priya Nair",
        "sender_email": "priya@example.com",
        "channel": "email",
        "account_tier": "pro",
    }
    base.update(overrides)
    return TicketInput(**base)


def valid_answers(**overrides) -> dict:
    """A raw answer dict that passes port validation for the triage set.

    Defaults: category billing (0.7), area other, mid scores, all
    verifies unlikely. Override per key with a probabilities dict.
    """
    answers: dict = {}
    for question in triage_questions():
        if isinstance(question, ChoiceQuestion):
            top = "billing" if question.key == "category" else "other"
            rest = [o for o in question.options if o != top]
            dist = {top: 0.7}
            for option in rest:
                dist[option] = 0.3 / len(rest)
            answers[question.key] = {"probabilities": dist}
        elif isinstance(question, ScoreQuestion):
            dist = {str(s): 0.02 for s in range(question.minimum, question.maximum + 1)}
            dist["5"] = 0.8
            answers[question.key] = {"probabilities": dist}
        elif isinstance(question, VerifyQuestion):
            answers[question.key] = {"probabilities": {"true": 0.1, "false": 0.9}}
    answers.update(overrides)
    return answers


def score_dist(peak: int, peak_mass: float = 0.8) -> dict[str, float]:
    dist = {str(s): (1.0 - peak_mass) / 10 for s in range(11)}
    dist[str(peak)] = peak_mass
    return dist


def fake_complete_factory(payload=None, fail_first_with: str | None = None):
    """An injectable LLM `complete` callable.

    Returns the canned payload as JSON text; when `fail_first_with` is
    given, the first call returns that string instead (retry tests).
    """
    state = {"calls": 0}

    def complete(system: str, user: str, max_tokens: int = 2200) -> str:
        state["calls"] += 1
        if fail_first_with is not None and state["calls"] == 1:
            return fail_first_with
        return json.dumps(payload if payload is not None else {"answers": valid_answers()})

    complete.state = state
    return complete


def fake_jev_transport(status: int = 200, payload: dict | None = None):
    """An injectable Jev transport recording the request it received."""
    calls: list[dict] = []

    def transport(method, url, headers, body, timeout):
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        return status, payload if payload is not None else {
            "answers": valid_answers(),
            "model": "jev-test",
        }

    transport.calls = calls
    return transport


def fresh_service(engine=None) -> TriageService:
    from support_triage.backends import OfflineDecisionEngine

    return TriageService(
        engine=engine or OfflineDecisionEngine(),
        embedder=HashingEmbedder(),
        index=InMemoryVectorIndex(),
        store=InMemoryStore(),
        objects=InMemoryObjectStore(),
    )
