-- Incident records (round 12): incidents + cluster dismissals.
--
-- A cluster proposal is a computed view over the run records and
-- is never stored; what a lead DID about one is. Confirming a
-- proposal creates an incident row — membership lives on the
-- incident's payload, never on a member run's record, so a
-- member's triage stays byte-identical under its membership
-- (the same discipline as the review tables in 0004). Dismissals
-- are append-only: the member-set fingerprint a lead refused,
-- kept so the exact set is never proposed again, while a set
-- that gains a member may propose as the new evidence it is.
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cluster_dismissals (
    id          BIGSERIAL PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    payload     JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cluster_dismissals_fingerprint ON cluster_dismissals (fingerprint);
