-- QA review records (round 8): samples + verdicts.
--
-- A review sample is a seeded, stratified draw over decided runs,
-- and a verdict is one reviewer's per-family grading of one sampled
-- run. Neither is part of a run's record: the triage a reviewer
-- graded must stay byte-identical under the grading, so samples
-- and verdicts live in their own tables, keyed back to runs by
-- ticket id. Verdicts are append-only (one row per verdict, never
-- updated); samples are written once at creation.
CREATE TABLE IF NOT EXISTS review_samples (
    sample_id   TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_verdicts (
    id          BIGSERIAL PRIMARY KEY,
    sample_id   TEXT NOT NULL,
    ticket_id   TEXT NOT NULL,
    reviewer    TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_verdicts_sample ON review_verdicts (sample_id);
CREATE INDEX IF NOT EXISTS idx_review_verdicts_ticket ON review_verdicts (ticket_id);
