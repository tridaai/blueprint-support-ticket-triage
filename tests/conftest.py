"""Pytest fixtures: hermetic tests, whatever the developer's shell holds.

The suite must behave identically on a bare machine and on one with
real keys in the environment (or a populated .env): an API key leaking
in from the shell would flip backend selection and turn unit tests
into live calls. So an autouse fixture scrubs every variable the
package reads (by exact name, by provider prefix, and by suffix)
before each test, and restores them afterwards.
"""

from __future__ import annotations

import os

import pytest

_EXACT = {
    "DECISION_BACKEND",
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_MODEL",
    "LLM_PROVIDER",
    "LLM_TIMEOUT_SECONDS",
    "DECISION_TIMEOUT_SECONDS",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDER",
    "DATABASE_URL",
    "CHECKPOINTS",
    "MIGRATIONS_DIR",
    "API_KEY",
    "S3_BUCKET",
    "S3_ENDPOINT_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "DUPLICATE_SIMILARITY_THRESHOLD",
    "SIMILARITY_FLOOR",
    "DRIFT_WINDOW",
    "DRIFT_MIN_SAMPLES",
    "DRIFT_BRIER_THRESHOLD",
    "TRIAGE_WEBHOOK_URL",
    "TRIAGE_WEBHOOK_SECRET",
    "EVAL_LIVE",
}

_PREFIXES = ("OPENAI_", "ANTHROPIC_", "OLLAMA_", "TYPESAFE_")
_SUFFIXES = ("_API_KEY", "_MODEL", "_BASE_URL")


def _is_agent_env(name: str) -> bool:
    return (
        name in _EXACT
        or name.startswith(_PREFIXES)
        or name.endswith(_SUFFIXES)
    )


@pytest.fixture(autouse=True)
def scrub_env(monkeypatch):
    saved = {k: v for k, v in os.environ.items() if _is_agent_env(k)}
    for key in list(os.environ):
        if _is_agent_env(key):
            monkeypatch.delenv(key, raising=False)
    yield
    for key, value in saved.items():
        os.environ[key] = value


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Neutralise .env loading: tests configure via monkeypatched env only."""
    import support_triage.config as config

    monkeypatch.setattr(config, "load_dotenv", lambda path=None: None)
    # Modules that imported the symbol directly get the same treatment.
    import support_triage.backends as backends
    import support_triage.checkpoints as checkpoints
    import support_triage.embeddings as embeddings
    import support_triage.objectstore as objectstore
    import support_triage.store as store
    import support_triage.wiring as wiring  # noqa: F401

    for module in (backends, checkpoints, embeddings, objectstore, store):
        if hasattr(module, "load_dotenv"):
            monkeypatch.setattr(module, "load_dotenv", lambda path=None: None)
