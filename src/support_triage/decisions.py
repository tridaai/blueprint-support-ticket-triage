"""The decision port's vocabulary: questions in, calibrated answers out.

This blueprint is built on a *System One* split (after Kahneman's fast
system, and the model class TypeSafe AI's Jev popularised): the model
never writes prose for the pipeline. It answers a fixed set of typed
questions about a piece of state, and every answer is a **probability
distribution**, not a string. Code — never the model — composes those
distributions into a priority score, a queue, and a set of guardrail
verdicts.

Three question primitives cover the triage question set:

- :class:`ChoiceQuestion` — pick one of a fixed option list. The answer
  is a probability per option; the value is the most likely option.
- :class:`ScoreQuestion` — rate on an integer rubric (0–10 here). The
  answer is a probability per score; the value is the distribution's
  expected score.
- :class:`VerifyQuestion` — is a statement true? The answer is a single
  probability of true, carried as a two-way distribution.

All questions in a set are evaluated in ONE decision call against the
same state, so a set of twelve questions costs one round trip — that,
plus calibration, is the entire point of the shape.

Validation lives here, not in the backends: :func:`build_decision_set`
takes whatever raw distributions a backend produced and either returns
a typed :class:`DecisionSet` or raises :class:`DecisionValidationError`.
A backend cannot hand the pipeline a malformed answer quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# A distribution must sum to 1 within this tolerance. System One models
# return exact distributions; LLM structured output drifts a little,
# and code renormalises only inside this band — beyond it the answer
# is rejected, not repaired into existence.
SUM_TOLERANCE = 0.02


class DecisionValidationError(ValueError):
    """A backend's raw answer failed validation (see build_decision_set)."""


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one of ``options``. ``key`` names the answer in the set."""

    key: str
    prompt: str
    options: tuple[str, ...]
    kind: str = field(default="choice", init=False)


@dataclass(frozen=True)
class ScoreQuestion:
    """Rate on an integer rubric from ``minimum`` to ``maximum``.

    ``rubric`` tells the decider what the ends of the scale mean — it is
    part of the question, identical for every backend, so a score means
    the same thing whichever engine produced it.
    """

    key: str
    prompt: str
    minimum: int = 0
    maximum: int = 10
    rubric: str = ""
    kind: str = field(default="score", init=False)


@dataclass(frozen=True)
class VerifyQuestion:
    """Is ``statement`` true of the state? Answer: P(true)."""

    key: str
    prompt: str
    statement: str = ""
    kind: str = field(default="verify", init=False)


Question = Union[ChoiceQuestion, ScoreQuestion, VerifyQuestion]


@dataclass
class QuestionResult:
    """One validated answer.

    ``probabilities`` is the full distribution (option → p for choice,
    score-as-string → p for score, ``{"true": p, "false": 1-p}`` for
    verify). ``value`` and ``confidence`` are *derived from the
    distribution in code*, never taken from the backend's word:

    - choice: value = argmax option; confidence = its probability.
    - score: value = expected score; confidence = the modal bucket's
      probability (how concentrated the answer is).
    - verify: value = P(true); confidence = max(p, 1 − p) — how
      decided the answer is, in either direction.
    """

    key: str
    kind: str
    value: object  # str for choice, float for score / verify
    probabilities: dict[str, float]
    confidence: float

    def expected_score(self) -> float:
        """Expected value of a score distribution (score results only)."""
        return sum(float(k) * p for k, p in self.probabilities.items())


@dataclass
class DecisionSet:
    """The validated answers to one question set, from one backend call."""

    results: list[QuestionResult]
    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)  # tokens etc., when the backend reports them

    def get(self, key: str) -> QuestionResult:
        for result in self.results:
            if result.key == key:
                return result
        raise KeyError(f"no answer for question {key!r} in this decision set")

    def choice(self, key: str) -> str:
        return str(self.get(key).value)

    def score(self, key: str) -> float:
        return float(self.get(key).value)

    def probability(self, key: str) -> float:
        """P(true) for a verify question."""
        return float(self.get(key).value)

    def confidence(self, key: str) -> float:
        return self.get(key).confidence

    def as_dict(self) -> dict:
        return {
            result.key: {
                "kind": result.kind,
                "value": result.value,
                "probabilities": result.probabilities,
                "confidence": result.confidence,
            }
            for result in self.results
        }


def _check_distribution(key: str, dist: dict, expected_keys: set[str]) -> dict[str, float]:
    if not isinstance(dist, dict) or not dist:
        raise DecisionValidationError(f"{key}: probabilities must be a non-empty object")
    keys = {str(k) for k in dist.keys()}
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise DecisionValidationError(
            f"{key}: distribution keys mismatch (missing={missing}, unexpected={extra})"
        )
    cleaned: dict[str, float] = {}
    for raw_key, raw_p in dist.items():
        try:
            p = float(raw_p)
        except (TypeError, ValueError):
            raise DecisionValidationError(f"{key}: probability for {raw_key!r} is not a number")
        if not (0.0 <= p <= 1.0):
            raise DecisionValidationError(f"{key}: probability {p} for {raw_key!r} is outside [0, 1]")
        cleaned[str(raw_key)] = p
    total = sum(cleaned.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise DecisionValidationError(
            f"{key}: probabilities sum to {total:.4f}, outside 1 ± {SUM_TOLERANCE}"
        )
    # Renormalise inside the tolerance band so downstream math is exact.
    return {k: p / total for k, p in cleaned.items()}


def build_decision_set(
    questions: list[Question],
    raw: dict[str, dict],
    *,
    backend: str = "",
    model: str = "",
    latency_ms: float = 0.0,
    usage: dict | None = None,
) -> DecisionSet:
    """Validate raw per-question distributions into a DecisionSet.

    ``raw`` maps question key → {"probabilities": {...}}. Every question
    must be answered exactly once (unknown keys are an error too — a
    backend inventing questions is as broken as one skipping them).
    """
    if not isinstance(raw, dict):
        raise DecisionValidationError("decision answer must be an object keyed by question key")
    expected = {q.key for q in questions}
    got = set(raw.keys())
    if got != expected:
        raise DecisionValidationError(
            f"answer keys mismatch (missing={sorted(expected - got)}, unexpected={sorted(got - expected)})"
        )
    results: list[QuestionResult] = []
    for question in questions:
        answer = raw[question.key]
        if not isinstance(answer, dict) or "probabilities" not in answer:
            raise DecisionValidationError(f"{question.key}: answer carries no probabilities object")
        dist = answer["probabilities"]
        if isinstance(question, ChoiceQuestion):
            cleaned = _check_distribution(question.key, dist, set(question.options))
            value = max(cleaned, key=lambda k: cleaned[k])
            confidence = cleaned[value]
        elif isinstance(question, ScoreQuestion):
            expected_keys = {str(s) for s in range(question.minimum, question.maximum + 1)}
            cleaned = _check_distribution(question.key, dist, expected_keys)
            value = sum(float(k) * p for k, p in cleaned.items())
            confidence = max(cleaned.values())
        elif isinstance(question, VerifyQuestion):
            # Accept either a two-way distribution or a bare probability.
            if set(map(str, dist.keys())) == {"probability"}:
                try:
                    p_true = float(dist["probability"])
                except (TypeError, ValueError):
                    raise DecisionValidationError(f"{question.key}: probability is not a number")
                if not (0.0 <= p_true <= 1.0):
                    raise DecisionValidationError(f"{question.key}: probability {p_true} outside [0, 1]")
                cleaned = {"true": p_true, "false": 1.0 - p_true}
            else:
                cleaned = _check_distribution(question.key, dist, {"true", "false"})
            value = cleaned["true"]
            confidence = max(cleaned["true"], cleaned["false"])
        else:  # pragma: no cover - defensive
            raise DecisionValidationError(f"{question.key}: unknown question type")
        results.append(
            QuestionResult(
                key=question.key,
                kind=question.kind,
                value=value,
                probabilities=cleaned,
                confidence=round(confidence, 4),
            )
        )
    return DecisionSet(
        results=results,
        backend=backend,
        model=model,
        latency_ms=latency_ms,
        usage=usage or {},
    )
