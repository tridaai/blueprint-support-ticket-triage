-- Governed template records (template_governance.py).
--
-- One row per template a human has governed — authored over the
-- bundled wording, edited, promoted, or retired. The template
-- family itself is bundled code (one body keyed by product area);
-- a template with no row here stands at the bundled default
-- (live, version 1, the wording as shipped), so this table holds
-- only the overlay: the names whose lifecycle someone has taken
-- in hand. The record carries its state, its current words, its
-- version, and its append-only act history in the payload — the
-- governance of a template is desk-level state, never part of a
-- run's record, so no run's triage changes because a template
-- was promoted or retired after the fact.
CREATE TABLE IF NOT EXISTS templates (
    name       TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
