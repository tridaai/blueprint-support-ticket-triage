"""Composition root: build a TriageService from configuration.

Every concrete collaborator is chosen here — decision engine
(DECISION_BACKEND), embedder, vector index, record store, object
store — and injected into the service. Nothing else in the package
reads backend-selection environment variables.
"""

from __future__ import annotations

from .backends import get_decision_engine
from .embeddings import get_embedder, get_vector_index
from .objectstore import default_object_store
from .service import TriageService
from .store import default_store


def build_service() -> TriageService:
    """The service for this process, wired from the environment."""
    return TriageService(
        engine=get_decision_engine(),
        embedder=get_embedder(),
        index=get_vector_index(),
        store=default_store(),
        objects=default_object_store(),
    )
