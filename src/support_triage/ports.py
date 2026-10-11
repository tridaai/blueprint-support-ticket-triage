"""Ports: the seams of the agent, declared in one place.

The pipeline depends on replaceable collaborators, each a
``typing.Protocol`` (structural — an implementation qualifies by
having the methods, not by inheriting):

- :class:`DecisionEngine` — the System One decision model. Answers a
  set of typed questions about a state with calibrated probability
  distributions. Implementations live in ``backends.py``: the TypeSafe
  (Jev) adapter, the LLM structured-output adapter, and the
  deterministic offline heuristics.
- :class:`Embedder` — text → vector, for the duplicate check.
  Implementations live in ``embeddings.py``.
- :class:`VectorIndex` — recent-ticket vectors, nearest-neighbour
  search (pgvector in production, in-memory as a test double).
- :class:`Store` — where triage records and agent decisions live
  (PostgreSQL in production, in-memory as a test double).
  Implementations live in ``store.py``.
- :class:`ObjectStore` — raw ticket payloads archived behind object
  keys (S3-compatible in production). Implementations live in
  ``objectstore.py``.
- :class:`EventSink` — where structured run events go.
  Implementations live in ``events.py``.
- :class:`Checkpointer` — LangGraph graph-state persistence behind
  the agent gate (PostgreSQL saver in production, in-memory saver as
  a test double).

``wiring.py`` is the composition root that builds the concrete
collaborators from configuration; the rest of the codebase programs
against these protocols. This module is deliberately a leaf: it
imports nothing from the package at runtime, so anything may import
it without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence

if TYPE_CHECKING:  # annotation-only imports (no runtime cycles)
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
    from langchain_core.runnables import RunnableConfig

    from .decisions import DecisionSet, Question
    from .events import RunEvent
    from .store import TriageRecord


class DecisionEngine(Protocol):
    """The decision-model seam: one call, a whole question set.

    ``decide`` receives the (screened, fenced) state text and the
    question list, and returns a validated :class:`DecisionSet`.
    Implementations must raise ``ProviderError`` on any failure —
    never return a partial or silently-degraded answer.

    ``context`` is the reopen seam: on a reopen run it carries the
    prior run's recorded answers and outcome, as historical record.
    Each backend declares what it does with it — the LLM and Jev
    adapters pass it as a fenced history block, separate from the
    new customer text; the offline heuristics ignore it (their
    answers are a function of the ticket's own words). What a
    backend did is recorded on the run (``ReopenInfo.context_use``);
    no backend may silently pretend the history was read.
    """

    name: str

    def decide(
        self, state: str, questions: list["Question"], context: str = ""
    ) -> "DecisionSet": ...


class Embedder(Protocol):
    """The embedding seam (duplicate detection only)."""

    name: str
    model: str

    def embed(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    """Recent-ticket vectors with nearest-neighbour search."""

    def add(self, ticket_id: str, vector: list[float], model: str) -> None: ...

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 1
    ) -> list[tuple[str, float]]: ...


class Store(Protocol):
    """The record seam: triage results in, decisions and history out."""

    def save(self, record: "TriageRecord") -> None: ...

    def get(self, ticket_id: str) -> "TriageRecord | None": ...

    def all_records(self) -> list["TriageRecord"]: ...

    def save_error(self, entry: dict) -> None: ...

    def all_errors(self) -> list[dict]: ...


class ObjectStore(Protocol):
    """The document seam: raw ticket payloads live behind object keys,
    never local paths (S3-compatible in production; an in-memory
    double for tests). Implementations in ``objectstore.py``."""

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str: ...

    def get(self, key: str) -> bytes | None: ...

    def exists(self, key: str) -> bool: ...


class EventSink(Protocol):
    """The run-events seam: one ``emit`` per structured event."""

    def emit(self, event: "RunEvent") -> None: ...


class Checkpointer(Protocol):
    """The graph-state seam behind the agent gate.

    Mirrors the sync surface of LangGraph's ``BaseCheckpointSaver``;
    the shipped selections (Postgres saver / memory saver) satisfy it
    by construction.
    """

    def get_tuple(self, config: "RunnableConfig") -> "CheckpointTuple | None": ...

    def list(
        self,
        config: "RunnableConfig | None",
        *,
        filter: dict[str, Any] | None = None,
        before: "RunnableConfig | None" = None,
        limit: int | None = None,
    ) -> Iterator["CheckpointTuple"]: ...

    def put(
        self,
        config: "RunnableConfig",
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: dict[str, Any],
    ) -> "RunnableConfig": ...

    def put_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None: ...

    def delete_thread(self, thread_id: str) -> None: ...
