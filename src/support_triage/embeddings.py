"""Embeddings + vector index for the duplicate check.

A re-sent ticket ("any update on my issue??" twenty minutes after the
original) should land on the same agent's desk, linked — not open a
parallel thread. The check is deliberately ordinary code around the
port: embed the ticket, ask the index for the nearest recent ticket,
compare cosine similarity against a threshold in code.

Two embedders:

- :class:`ProviderEmbedder` — any OpenAI-compatible ``/embeddings``
  endpoint (OpenAI, NVIDIA NIM, a self-hosted server), used when
  ``OPENAI_API_KEY`` is set. Vectors record their model so a model
  switch re-embeds instead of mixing spaces.
- :class:`HashingEmbedder` — a deterministic bag-of-words stand-in
  (feature hashing over word unigrams+bigrams). It is NOT a semantic
  model and the docs say so; it exists so the offline demo and the
  hermetic tests exercise the same duplicate machinery, and it does
  catch near-verbatim re-sends, which is the duplicate shape that
  matters most at intake.

Two indexes behind the :class:`~support_triage.ports.VectorIndex`
port: pgvector (production) and in-memory (test double).
"""

from __future__ import annotations

import hashlib
import math
import re
import threading

from .config import env_str, load_dotenv
from .errors import translate_provider_error

EMBEDDING_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN.findall((text or "").lower())
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class HashingEmbedder:
    """Deterministic feature-hashing embedder — offline stand-in."""

    name = "hashing"
    model = "hashing-bow-256"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class ProviderEmbedder:
    """Embeddings via an OpenAI-compatible endpoint (the `llm` extra)."""

    name = "provider"

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        api_key = env_str("OPENAI_API_KEY")
        if not api_key:
            from .errors import not_configured_error

            raise not_configured_error("provider embeddings", "OPENAI_API_KEY")
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "provider embeddings need the openai SDK, which is not installed. "
                "Install the optional llm extra: uv sync --extra llm "
                "(pip fallback: pip install -e \".[llm]\"), then re-run."
            ) from exc
        self.model = model or env_str("EMBEDDING_MODEL", "text-embedding-3-small")
        self._base_url = env_str("OPENAI_BASE_URL")
        try:
            self._client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as exc:  # client construction (bad URL/proxy env)
            raise translate_provider_error(exc, backend="openai", base_url=self._base_url) from exc

    def embed(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings.create(model=self.model, input=text[:8000])
        except Exception as exc:
            raise translate_provider_error(exc, backend="openai", base_url=self._base_url) from exc
        return list(response.data[0].embedding)


def get_embedder() -> HashingEmbedder | ProviderEmbedder:
    """The embedder for this process: provider when a key is configured,
    the hashing stand-in otherwise (offline demo, tests)."""
    load_dotenv()
    if env_str("EMBEDDER", "").strip().lower() == "provider" or (
        env_str("OPENAI_API_KEY") and env_str("EMBEDDER", "").strip().lower() != "hashing"
    ):
        return ProviderEmbedder()
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorIndex:
    """Process-local vector index — test double only."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], str]] = {}
        self._lock = threading.Lock()

    def add(self, ticket_id: str, vector: list[float], model: str) -> None:
        with self._lock:
            self._vectors[ticket_id] = (vector, model)

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 1
    ) -> list[tuple[str, float]]:
        with self._lock:
            items = list(self._vectors.items())
        scored = [
            (ticket_id, cosine(vector, stored))
            for ticket_id, (stored, _model) in items
            if ticket_id != exclude_id
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


class PgVectorIndex:
    """pgvector-backed index (production). Schema in
    ``migrations/0002_ticket_embeddings.sql``, applied on construction.
    Similarity is computed in SQL with the ``<=>`` cosine operator."""

    def __init__(self, database_url: str) -> None:
        import psycopg  # local import: the double must work without the driver

        self._url = database_url
        self._psycopg = psycopg
        with self._connect() as conn:
            from .migrations import run_migrations

            run_migrations(conn)

    def _connect(self):
        return self._psycopg.connect(self._url)

    @staticmethod
    def _literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    def add(self, ticket_id: str, vector: list[float], model: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ticket_embeddings (ticket_id, model, embedding, updated_at)
                    VALUES (%s, %s, %s::vector, now())
                    ON CONFLICT (ticket_id) DO UPDATE
                    SET model = EXCLUDED.model, embedding = EXCLUDED.embedding, updated_at = now()
                    """,
                    (ticket_id, model, self._literal(vector)),
                )
            conn.commit()

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 1
    ) -> list[tuple[str, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ticket_id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM ticket_embeddings
                    WHERE ticket_id <> %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (self._literal(vector), exclude_id or "", self._literal(vector), limit),
                )
                rows = cur.fetchall()
        return [(row[0], float(row[1])) for row in rows]


def get_vector_index() -> InMemoryVectorIndex | PgVectorIndex:
    """The index for this process: pgvector when ``DATABASE_URL`` is set,
    the in-memory double otherwise."""
    load_dotenv()
    url = env_str("DATABASE_URL", "")
    if url:
        return PgVectorIndex(url)
    return InMemoryVectorIndex()
