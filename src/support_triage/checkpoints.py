"""Checkpoint storage for the agent gate (LangGraph saver selection).

The graph pauses at the agent gate with ``interrupt()``; the paused
state has to live somewhere durable in production. Selection mirrors
the record store:

- ``DATABASE_URL`` set → :class:`PostgresSaver` over a small
  connection pool, schema applied by ``saver.setup()`` (its own
  checkpoint tables, separate from the record migrations).
- No ``DATABASE_URL`` → :class:`MemorySaver`, the in-process double.
- ``CHECKPOINTS=off`` → ``None``: runs execute straight through and a
  decision can only be recorded for a run in the same call (the
  service's fallback path); useful for batch eval runs.
"""

from __future__ import annotations

from .config import env_str, load_dotenv
from .ports import Checkpointer


def default_checkpointer() -> Checkpointer | None:
    load_dotenv()
    if env_str("CHECKPOINTS", "").strip().lower() == "off":
        return None
    url = env_str("DATABASE_URL", "")
    if not url:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=url,
        max_size=4,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def close_checkpointer(checkpointer: Checkpointer | None) -> None:
    """Best-effort pool cleanup for processes that own the saver."""
    conn = getattr(checkpointer, "conn", None)
    if conn is not None and hasattr(conn, "close"):
        try:
            conn.close()
        except Exception:
            pass
