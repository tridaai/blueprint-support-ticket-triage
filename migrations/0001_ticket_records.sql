-- Triage records, one row per ticket run.
--
-- The full decision record (answers with distributions, composition,
-- guardrails, telemetry) is stored as one JSONB payload so the record
-- the agent reviewed is exactly the record on file; a row is re-saved
-- in place when the agent decides, so the payload on disk always
-- carries the current status. promoted columns (status, queue,
-- priority) exist for queue queries and metrics without unpacking
-- JSON on every read.
CREATE TABLE IF NOT EXISTS ticket_records (
    ticket_id   TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'awaiting_agent',
    queue       TEXT NOT NULL DEFAULT '',
    priority    DOUBLE PRECISION NOT NULL DEFAULT 0,
    decided     BOOLEAN NOT NULL DEFAULT FALSE,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticket_records_status ON ticket_records (status);
CREATE INDEX IF NOT EXISTS idx_ticket_records_queue ON ticket_records (queue);
