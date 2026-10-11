# Architecture — Support Ticket Triage Agent

## Provenance and stance

This blueprint is Trida AI's original work: the design, the code, the
question set, the rubrics, and every ticket in the repository were
written for this blueprint. The *concept* it explores — models that
return typed, calibrated decisions instead of text — is credited where
it is due: System 1 / System 2 is Kahneman's framing, and TypeSafe
AI's Jev is the product that made "decision models" a concrete API
shape worth designing against. Patterns were learned from public
sources; no code, prose, or data was copied from any of them. The
tickets are synthetic, including the card number (a public test
number) in the PII sample.

## Decisions, not drafts

The four sibling blueprints in this family end at a human gate over a
*draft* — a letter, a packet, a review. This one ends at a gate over a
*record of decisions*, and the difference is the design:

- A generative model is good at language and unreliable at
  commitment. Asked "how urgent is this, 0–10?" in prose, it answers
  with a number-shaped vibe. Asked for a **probability distribution
  over the rubric**, its answer becomes measurable: the calibration
  pack can score whether 0.8 means 0.8.
- Decisions decompose. "Triage this ticket" is one vague judgement;
  eleven small questions (category? area? urgency? does it contain
  PII?) are each checkable, each calibratable, and each reusable —
  the same Verify answer that routes a legal ticket also blocks its
  autonomy recommendation.
- Consequences stay in code. Priority arithmetic, the routing
  matrix, the overrides, and the guardrails are deterministic
  functions of the distributions. Two runs of the same ticket get the
  same priority; an auditor can recompute any record by hand from
  the distributions printed on it.

The model is therefore used the way System One works in Kahneman's
account: fast pattern judgements, in parallel, at low cost — while
the slow, accountable system (code, then a human) owns everything
that follows from them.

## The pipeline

```
intake → screen → duplicate → decide → compose → agent gate ⏸ → record decision
```

- **intake** normalises the ticket. Nothing is inferred yet.
- **screen** treats the ticket as untrusted text (it is
  customer-authored): injection pattern families run over subject and
  body; flagged sentences are removed from model state and the rest
  is fenced. Screening removes the instruction, never the ticket —
  a customer *asking* for a refund is a billing ticket; a customer
  *ordering the system* to refund them is flagged and still triaged.
- **duplicate** embeds the ticket (hashing stand-in offline; any
  OpenAI-compatible embeddings endpoint in production) and compares
  cosine similarity against recent tickets in pgvector. The threshold
  comparison is code; a link is recorded on the record, and the
  original ticket's agent can be handed the thread.
- **decide** is the single decision call: the full question set
  against the screened state. One call, eleven answers — the cost
  shape is the point of the port (a System One call is priced and
  timed like a classifier, not like eleven chat turns).
- **compose** reads the distributions deterministically:
  `priority = 100 × (0.30·urgency + 0.25·sla_risk + 0.20·churn +
  0.15·anger + 0.10·abuse_p)` over normalised scores, +15 when
  P(outage) ≥ 0.5, capped at 100; routing by matrix with override
  precedence legal → outage → engineering; autonomy recommendation;
  the template draft decision.
- **agent gate** (LangGraph `interrupt`) persists the record as
  `awaiting_agent`. Approve/reject/assign arrive via API or CLI and
  are recorded with the agent's name and note.

### Guardrails (code, not prompts)

Four checks run over every decision set, and all four are on the
record: **completeness** (every question answered exactly once),
**distribution sanity** (sums and ranges), **low confidence** (the
honesty rule: an indecisive gating answer — flat category, or a
safety Verify inside the undecided band — forces HUMAN_REVIEW rather
than letting composition guess), and **safety flags** (raised PII /
abuse / legal / outage verifies are named, and block autonomy and
drafts unconditionally). The port validation in `decisions.py` is the
first two checks' enforcement; the guardrail layer records them as
verdicts alongside the checks only it can make.

### The template draft, and why it is rare

Exactly one slice earns a draft: category how_to at confidence
≥ 0.80, zero safety flags, guardrails green, not a duplicate. The
draft is a fixed template parameterised by the product area's guide —
code writing text, not a model writing text. It exists to mark the
boundary a deployment would automate first *if* it chose to, under
the autonomy recommendation. This system never sends it; an agent
does, or doesn't.

## The backends, honestly

- **Jev (TypeSafe)** is the design's native engine and the one this
  blueprint cannot yet run live: early access. The adapter implements
  the documented primitives contract and its failure behaviour is part
  of the contract — missing key, refused call, and malformed answer
  are all loud `ProviderError`s. A triage record that silently came
  from a different engine than it claims would be a false record, so
  there is no fallback path.
- **The LLM adapter** exists so the port is exercisable today: one
  structured-output call, temperature 0, strict schema demanded,
  code-side validation, one bounded retry with the failure named.
  Its probabilities are *elicited*, not trained — the calibration
  pack in live mode (`EVAL_LIVE=1`) measures exactly how much that
  matters per question, reported rather than gated.
- **The offline rules** are a sparring partner with a labelled
  jersey. Their role is to make the whole machine — composition,
  guardrails, evals, console — testable with zero keys, and to give
  the calibration pack a deterministic baseline. The pack caught
  real defects in them during development (flat choice distributions,
  a score shape whose truncated mean drifted, fence boilerplate
  tripping the keyword rules), which is what a baseline is for.

## Production topology

```
            ┌────────────────────────────┐
 tickets →  │  API (stateless, FastAPI)  │ → S3 bucket: raw payloads
            │  LangGraph pipeline        │
            └────────────┬───────────────┘
                         │
              PostgreSQL (one database):
              records · checkpoints · pgvector
```

- The app container holds no state: records, graph checkpoints, and
  embeddings live in PostgreSQL; raw ticket payloads in the S3
  bucket. Migrations are numbered SQL files applied at startup.
- `/readiness` distinguishes "PostgreSQL unreachable" (503) from
  "running on test doubles" (200, stated plainly). A deployment that
  loses its database fails readiness instead of quietly going
  in-memory — the doubles exist for tests and demos, and the code
  paths that select them say so.
- Telemetry is per-run and on the record: backend + model, decision
  latency, question count, token usage and estimated cost where the
  backend reports them, guardrail outcomes. `/metrics` aggregates
  across records for the queue-level view.

## Stats, outcomes, and live calibration

The dashboards layer is deliberately boring: pure aggregation
functions (`stats.py`) over the store's record list, so PostgreSQL
and the in-memory double produce identical dashboards from identical
records. Failed runs are the one gap in the record list — an
exception leaves no record — so the store keeps a small append-only
error log (process-local for the doubles, the `run_errors` table in
PostgreSQL) and the service writes to it on the failure path, after
emitting the `run_failed` event and before re-raising. Telemetry
failure never masks the run's real error.

An agent's decision at the gate says "I accept this triage"; an
**outcome** says "here is what was actually true". Outcomes are a
separate, later write (`POST /api/runs/{id}/outcome`) because they
are only knowable after the work: final category and queue, whether
it resolved, and per-question truth for the questions an agent can
judge. They are stored on the record (and promoted to a column by
migration 0003) and never feed back into the triage itself — the
pipeline's record of what it believed stays untouched, which is
what makes scoring it honest.

Live calibration (`livecal.py`) converts outcome-carrying records
into the eval runner's sample shape and calls the same
`calibration_report`: category scored one-vs-rest on the chosen
value through the confidence bins (plus multiclass Brier over the
full distribution), verify questions as binary Brier + reliability
bins, score questions through the high-event (≥ 7) Brier and
expected-score MAE. Eval and live figures are therefore directly
comparable — same bins, same code, different truths: authored
golden labels before launch, the operation's own outcomes after.
The endpoint reports `available: false` with an explanatory note
until the first outcome exists; a dashboard that invented a live
figure from no labels would be lying in exactly the way this
blueprint exists to avoid.

The policy endpoint (`GET /api/policy`) is the composition module's
constants rendered as data — one source of truth, read by the
console's policy panel and what-if simulator. The simulator runs
client-side over fetched records and persists nothing: it answers
"what would this threshold do to today's queue?" without giving a
hypothetical the power to touch a real record.

## Drift watch and assignment suggestions

Two round-4 additions sit on top of the outcome stream, and both
are reads over the same records — neither writes anything.

The **drift watch** (`drift.py`) replays the labelled outcomes in
`recorded_at` order (ties broken by ticket id, so the stream is
deterministic), takes the last `DRIFT_WINDOW` of them, and runs the
window and the full stream through the same `calibration_report`.
Each question's row carries the eval baseline Brier, the all-time
live Brier, the windowed Brier, and both sample counts; the flag is
`window_n ≥ DRIFT_MIN_SAMPLES and window_brier − eval_brier ≥
DRIFT_BRIER_THRESHOLD`. The sample floor is the design point: Brier
over a handful of outcomes is noise, so the watch refuses to flag
what it cannot yet measure — the row reports as warming up instead.
Without an eval baseline there is nothing to degrade *from*, and
the payload says that too. Thresholds are environment config read
at call time, in the same idiom as `SIMILARITY_FLOOR`, and the
values in force travel in the payload.

**Assignment suggestions** (`assignment.py`, roster in `roster.py`
+ `data/agent_roster.json`) rank the roster agents covering a
waiting ticket's queue by a weighted sum — track record 0.50
(resolved share of the outcomes that agent recorded in the
ticket's category, matched by roster id or name; no history scores
a neutral 0.5), open load 0.30 (`1/(1+open)`, where open counts
gate-waiting records whose latest `[assigned to …]` note names the
agent — the note idiom is the only assignment record the system
keeps, so the suggestion reads the same source the humans write),
skills 0.20 (share of the ticket's needs — category, product area,
raised flags — covered by the agent's tags). Every candidate
carries its per-factor reasons as strings; the console prints them
verbatim. The feature's invariant is structural: the module has no
store handle and no write path, the endpoint is a GET, and the
ticket's record is byte-identical after a suggestion is served.

## Policy replay and audit export

Round 5 adds two reads over the same records, both built so the
hypothetical can never touch the real.

**Policy impact replay** (`replay.py`) starts from a small,
deliberate refactor: `compose()` and `run_guardrails()` take the
policy constants they read — priority weights, outage boost,
category forcing threshold — as keyword-only overrides whose
defaults are the live values, so every existing caller is
unchanged and the replay runs the *same* code the pipeline runs,
not a re-implementation of it. A replay rebuilds each record's
answers into a `DecisionSet` (the record carries kind, value,
distribution and confidence — everything composition reads) and
compares the recomputed consequence against the stored one. A run
*moves* on the three dimensions an operator acts on — queue,
priority band, guardrail outcome; a score change inside a band is
counted and shown, but is not a move. The invariants are
structural, as with assignment suggestions: the replay module
receives a list of records, never a store handle, and the service
passes `all_records()` — there is no write path to misuse, a test
pins the records byte-identical across an endpoint replay, and a
second test swaps in an engine that raises if called, proving the
decision layer is never re-asked. Records without the full answer
set (never produced by the pipeline; possible in a hand-built
store) are skipped and named in the report rather than replayed
on partial evidence. Invalid hypotheticals — weights that don't
name exactly the five terms, don't sum to 1, or go negative; a
threshold outside [0, 1] — are 422s with the reason spelled out.

**Run audit export** (`exports.py`) renders one run as a portable
document. The assembly is one dict, built in code from the stored
record (plus the stored intake payload, the SLA clock at export
time, and the drift watch's current per-question rows for the
questions this run answered, assembled by the service from the
same `drift_report` the dashboard reads); the JSON endpoint
returns it and `?format=text` renders it, so the two formats
cannot disagree — a test asserts the key facts in both. The
composition section is a trace, not a summary: each formula term
carries its basis (which answer, normalised how), its input, its
weight and its points, and a `score_check` boolean reconciles the
term total plus boost against the recorded score — an export that
doesn't add up says so on its face instead of rounding quietly.

## Events, comparisons, and SLA clocks

**The event bus** (`events.py: EventBus`, owned by the service) is
the round-3 addition underneath everything live. The service
publishes four lifecycle events as plain dicts — `run_completed`
(a run reached the gate), `guardrail_blocked` (its guardrails
failed a check — a blocked draft/autonomy path is a lifecycle fact,
not a footnote of completion), `outcome_recorded`, and
`batch_completed`. Publishing never blocks: subscriber queues are
capped and drop their oldest events, because a stalled pipeline is
worse than a missed UI refresh. `GET /api/events` is a
`StreamingResponse` over one subscription (dependency-free SSE; the
console falls back to polling when the stream cannot be held).
Gate webhooks (`webhooks.py`) ride the same two gate events:
fire-and-forget POSTs on a daemon thread with a 3-second timeout,
failures logged and forgotten. Neither channel is load-bearing —
the record in the store is the truth, and both channels exist to
save a human a refresh.

**Batch intake** is intentionally not a new pipeline: it is a loop
over `run_ticket` in the service, with validation and provider
failures captured per ticket. The cap (50) is a whole-batch 422 —
silently truncating an intake would lose tickets without a trace,
which is the one failure shape this blueprint never chooses.

**Comparisons** (`compare.py`) share exactly one pipeline step with
a run — the injection screen — and deliberately skip the rest of
the graph: no duplicate check (a comparison must not teach the
index), no gate, no record. Each backend's raw answers go through
the port validation and the same `compose` + guardrails code, so a
column difference is attributable to the decision model alone.
Records are archived to the object store under `comparisons/` —
they are experiments with a different lifecycle from run records,
and putting them in the queue's table (or a new migration) would
blur exactly that line.

**Similar tickets** reuse the duplicate check's port (`Embedder` +
`VectorIndex`, so pgvector and the in-memory double behave alike)
with a different question and a lower floor (0.30 vs the duplicate
threshold's 0.85): not "is this the same ticket?" but "what
happened to tickets like this?" The answer joins each neighbour's
record to its outcome when one exists — the loop the outcome
endpoints opened, closed in the other direction.

**SLA clocks vs SLA risk** is the naming discipline the code
enforces. `sla_breach_risk` is a model judgement — a distribution,
an input to the priority formula, calibrated like every other
question. The SLA *clock* (`sla.py`) is arithmetic: the band's
target from `SLA_TARGET_HOURS` minus the ticket's age since
`received_at`, reported as ok / due_soon (last quarter of the
target) / breached. The clock takes `now` as a parameter everywhere
so its tests are exact; it never consults the model, and the
model's risk answer never moves a target.

## What the agent will never do

- **Send anything to a customer.** No node in the graph has that
  capability; the template draft is text on a screen for an agent.
- **Act on its autonomy recommendation.** It is computed, recorded,
  and displayed — and the graph has no edge that consumes it.
- **Guess through indecision.** A gating answer inside the undecided
  band forces human review; the pipeline would rather stop than be
  confidently wrong on a safety question.
- **Degrade silently.** A failed decision call fails the run with a
  translated, actionable error. There is no partial record and no
  substitute backend.
- **Treat ticket text as instructions.** Screened, fenced, and —
  where flagged — removed, before any engine sees it.
