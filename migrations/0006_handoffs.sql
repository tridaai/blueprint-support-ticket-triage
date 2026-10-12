-- Shift handoff records (round 14).
--
-- A handoff is one stored document per shift change: the desk's
-- figures composed at a stated boundary (in the payload, section
-- by section, exactly as the source views computed them) and the
-- handshake over them — who wrote it, who it is addressed to, and
-- whether that lead accepted it, plainly or with exceptions. It
-- lives in its own table for the reason every human act here
-- does: a handoff is desk-level, never part of a run's record, so
-- no run's triage can change because a shift changed. Superseded
-- handoffs stay as rows — the payload's status lapses, the record
-- does not.
CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
