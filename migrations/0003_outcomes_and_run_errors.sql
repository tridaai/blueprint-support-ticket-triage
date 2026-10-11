-- Agent outcomes + run-error telemetry (round 2).
--
-- The agent's ground truth for a handled ticket (final category and
-- queue, resolved?, per-question truths) rides inside the record
-- payload like everything else; the promoted `outcome` column exists
-- so outcome coverage — "how many handled tickets have we labelled?"
-- — is a cheap SQL question instead of a payload unpack. `run_errors`
-- is the append-only log of failed runs: a failed run produces no
-- record, and the performance dashboard still has to count it.
ALTER TABLE ticket_records ADD COLUMN IF NOT EXISTS outcome JSONB;

CREATE TABLE IF NOT EXISTS run_errors (
    id         BIGSERIAL PRIMARY KEY,
    ticket_id  TEXT NOT NULL DEFAULT '',
    backend    TEXT NOT NULL DEFAULT '',
    error      TEXT NOT NULL DEFAULT '',
    at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_errors_backend ON run_errors (backend);
