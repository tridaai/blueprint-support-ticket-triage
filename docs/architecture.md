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

## Reopens and conversation threads

**The reopen** (`service.reopen_run`, graph reopen state) is how a
run's story continues. The follow-up message becomes a new ticket
— `<root>-R<count>` ids, suffixed rather than overwritten when a
branch collides — whose intake fields carry over from the prior
run's stored input (store copy, else the object-store archive,
else the record's own fields). The prior run is never written to
again; the link (`ReopenInfo` on the new record: prior id, chain
root, count, the limit in force, who logged it) rides the record
JSON, so no migration was needed — the same precedent as outcomes.

Three consequences are enforced where they can't be prompted away:

- **Context through the port.** `DecisionEngine.decide` takes an
  optional `context` — the prior run's record composed by code
  (answers with confidences, routing, the agent's decision, the
  recorded outcome with its truths). The adapters decide what the
  channel honestly supports: the LLM adapter appends it to the
  prompt as a fenced prior-history block (`screening.fence_history`)
  *after* the untrusted fence closes; the Jev adapter sends it as
  a separate `context` field, never folded into the ticket state;
  the offline backend ignores it, deterministically, and
  `ReopenInfo.context_use` records which behaviour the run got.
  The injection screen runs on the follow-up message in the intake
  node, exactly as on a first pass — history is context, the new
  message is still untrusted customer text.
- **No second template.** `build_template_draft` refuses reopen
  runs as its first check, and the autonomy recommender gains the
  matching blocker, so no call site can re-enable the canned path
  for a customer whose issue already failed once.
- **Floor and limit in composition/guardrails.** `compose(...,
  reopen=True)` raises the composed band one step after the score
  is fixed (`band_before_floor` is recorded; the export and the
  console render it as its own line, like the outage boost).
  `run_guardrails` gains a `reopen_limit` check: a run whose count
  reaches the limit in force (`REOPEN_LIMIT`, default 2, read like
  the other policy constants) fails that check and is forced to
  human review with the reason named. Replay passes each record's
  stored count/limit through, so the identity property (an empty
  hypothetical moves nothing) holds over reopen runs too.

**Threads** (`threads.py`) group runs into conversations with one
deterministic function at intake. The customer key is the
normalised sender email, falling back to the normalised name;
unknown customers never thread. Similarity is token containment
of the subject+body content tokens — shared tokens over the
smaller set — chosen over Jaccard because a follow-up is usually
much shorter than the ticket it continues and Jaccard scores
brevity as unrelatedness. At or over 0.35, with the earlier run
inside 30 days, the best candidate (ties toward the most recent)
lends its thread id; the thread id is the root's, `thr-<ticket
id>`, and lives on the record JSON. A reopen skips derivation and
joins its prior run's thread directly. The read side
(`thread_summaries` / `thread_chain`) assembles summaries and
chains from the records at read time: run counts, open runs,
whether the thread was ever reopened, the latest outcome — pure
reads that leave every record byte-identical.

## Operator corrections

The grouping above and the intake duplicate flag are machine
judgements shipped as labels. **Corrections** (`corrections.py`,
service methods `attach_run_to_thread` / `detach_run` /
`merge_threads` / `mark_duplicate` / `unmark_duplicate`) are the
operator layer over them, stored as append-only entries on the
record JSON — the same no-migration precedent as outcomes and
reopens.

**Storage decision: per-record overlay entries, not a corrections
table.** A thread correction is a fact about the runs it moves,
and the record is already the unit every read assembles from;
riding the record means the Postgres payload and the in-memory
double answer identically with no join and no new migration, and
the audit export — built from the record alone — carries the full
lineage for free. The cost is that a merge writes one entry per
source record; merges are rare, human-paced events, so the write
amplification is irrelevant next to the read simplicity.

**Chain semantics: corrections move chains, never runs.** Attach
and detach are addressed at a run but applied to every run in its
reopen chain (the chain is identified by `ReopenInfo.root_ticket_id`,
or the run itself). The alternative — refusing to detach a chain
member — would leave the common case (the follow-up proves the
chain is mis-grouped) uncorrectable without detaching the root and
hoping the operator finds it. Moving the whole chain keeps the one
invariant the grouping itself promises: a chain is one
conversation, so it sits in exactly one thread, corrected or not.
A detach names its target `thr-op-<chain root>` — operator threads
are visibly operator-made and can never collide with a computed
`thr-<ticket id>`. Reopens created after a correction join the
chain's *effective* thread, so a corrected chain stays corrected
as it grows.

**What corrections may never touch.** The decision: answers,
composition, guardrails, draft, outcome are byte-identical before
and after any correction (pinned by test over a decision-only dump
of the record). Replay reads answers only, so identity replay over
corrected runs moves nothing — also pinned. The one read whose
numbers change is sampling: `livecal.live_samples` skips records
with an active duplicate mark, and the drift stream inherits the
exclusion because it is built from that function. Both payloads
carry the `duplicates_excluded` accounting (ticket, survivor,
marker, reason) so the missing sample is a stated fact, not a
silent gap. Volume reads (`overview_stats`, store metrics) take no
notice of marks — the run was a ticket and stays counted; the
overview does carry `confirmed_duplicates` so the population is
visible as a population.

**Refusal taxonomy.** `CorrectionError` carries its HTTP status:
409 for state conflicts (double mark, detach-when-alone, attach to
the current thread, merge into itself, un-mark with no live mark),
422 for corrections invalid on their face (missing actor or reason,
self-duplicate, any cross-customer attach/merge/mark — the
customer key is taken from the stored intake, the same key intake
grouping uses, so a correction can never join what the machine was
forbidden to). Unknown runs and threads stay `KeyError` → 404, the
house style.

## The agent's workbench

`workbench.py` + `playbooks.py` are the post-gate layer: everything
they do happens after triage, to a record the pipeline has finished
with, and none of it feeds back. The separation is structural, not
promissory — the golden pack pins the run's status, composition,
answers, and calibration samples byte-identical across workbench
actions, and `livecal.live_samples` reads outcomes and answers only,
so workbench state cannot enter a sample by construction.

**Resolution knowledge** is a join, not a new model: the
similar-tickets machinery (same embedder, same index, same floor)
produces the neighbour list exactly as before, and the workbench
joins it to stored outcomes in code. The honesty rules are the
drift watch's, applied to advice: an aggregate over fewer than
`KNOWLEDGE_MIN_SAMPLES` (3) outcome-bearing neighbours withholds
its rate and median rather than quoting them; neighbours without
outcomes are counted as excluded in the payload's accounting, so
the reader can see the shape of what was left out. Time-to-
resolution deserves its sentence: outcomes carry no duration
field, so the figure is elapsed hours between the two stamps the
records do carry — intake (`received_at`) and outcome
(`recorded_at`, stamped when the agent closes the ticket out) —
computed in `hours_from_intake` and named exactly that on every
surface. A record missing either stamp yields `None`, never a
reconstruction.

**Playbooks** are authored data (`data/playbooks.json`, validated
at load against the decision port's category set, the roster
pattern). Progress is an append-only overlay on the record
(`playbook_checks`, `playbook_na`) with the corrections'
semantics: the effective state is the last entry, the trail is
never rewritten, mutations demand an actor (and a reason for the
not-applicable mark), and conflicts are 409s. The one hard rule
beyond bookkeeping: a set marked not-applicable refuses step
checks until the mark is lifted — the mark exists to stop
box-ticking, so working the steps underneath it is a conflict,
not a shortcut.

**The working note** is composed in `build_working_note` from the
record alone — routing reason verbatim (pinned by golden case
G55), raised verify answers mapped to code-authored "blocked on"
lines, duplicate/thread context read through `corrections`, and
the deterministic SLA clock. Determinism is the point: the draft
is stable under the agent while they edit it, and the saved note
is a separate stored object (`working_note` on the record) that
no read recomputes. It is deliberately a different surface from
the template first-response draft — internal vs customer-facing,
saved-under-a-name vs offered-for-one-slice — and the export
carries them as separate blocks so an audit never confuses them.

## QA review

`reviews.py` is the layer that grades decisions instead of
outcomes. Everything else called "calibration" in this repo —
the eval pack, `livecal.py`, the drift watch — scores recorded
probabilities against what eventually happened to the ticket.
Whether the *answers* were right is a different question, and it
belongs to a human reviewer; the design problem is the sample,
because a reviewer who picks (or is handed) the interesting
tickets grades a fiction.

**The sampler** (`draw_sample`, pure) therefore carries the
audit weight. Population: decided runs — the store's promoted
definition, `status` in `triaged` / `rejected` — because the gate
has ruled and the record is final. Strata are
`(composition.queue, composition.priority_band)` — the
composition's own `BANDS` — with proportional allocation by
largest remainder, capped per stratum, surplus seats re-offered
in the same remainder order. Inside a stratum, candidates are
sorted by ticket id and shuffled with `random.Random(seed)`;
strata are drawn in sorted key order. The draw is thus a pure
function of (records, seed, size, exclusion flag) — the golden
pack pins the exact run ids for a seeded world, and a redraw in
the same case must reproduce them. Exclusions are structural,
not editorial: a confirmed duplicate never draws (read through
`corrections.is_confirmed_duplicate`, the same predicate the
calibration exclusion uses), and runs carrying a verdict in an
earlier sample are excluded by default behind an explicit
request flag. Both exclusions are named, with ids, in the
sample record's `population` block; the sample itself is a
stored record (`ReviewSample`: seed, strata rows, run ids,
creator, stamp) in its own tables (migration 0004), never part
of any run's record.

**Verdicts** decompose along the port's real shape. The
families are grounded in the `DecisionSet` structure, not
invented for the UI: `category` grades the two choice answers,
`priority_drivers` the four score answers, `verify` the five
verify answers, and `routing` grades composition's outputs —
queue and band — as their own family, because agreement with
every answer does not imply agreement with the destination.
Validation is strict (every family present, values in
{agree, disagree}) and verdicts are append-only with one
verdict per (sample, run, reviewer): a repeat is a 409 conflict
naming the standing verdict, in the corrections idiom — grades
are testimony, and a second reviewer remains free to disagree.

**The report** (`agreement_report`, pure) aggregates over a
sample set: per-family tallies, an overall tally, a per-backend
split keyed off each verdicted run's `telemetry.backend` (the
judgement graded is the backend's answers, so the split follows
the producer), and a disagreement list sorted for audit reading.
Rates pass through one withholding rule — `None` below five
verdicts, counts always present — the same small-sample honesty
as the workbench floor and the drift minimum, sized for a review
round. The report is a read model in the strong sense: nothing
in the repo consumes it — no weight, threshold, or prompt reads
an agreement rate — and the golden pack pins the property that
matters: run records, `live_samples` output, and the
`live_calibration` payload are byte-identical before and after
verdicts are recorded. Storage follows the run-errors precedent
(base-class process-local default, Postgres override), so the
InMemory doubles stay honest and the migration carries the
production tables.

## The customer dossier

`dossier.py` is the per-customer read over everything the rounds
above stored per-ticket. Three design points carry it.

**The key is borrowed, not invented.** Customer identity was
already decided by the thread machinery (`threads.customer_key`:
normalised email, else `name:`-prefixed normalised name, else the
empty key that groups with no one), and the correction layer
already resolves a stored run's key from its stored intake
(`service._customer_key_for`, since `TicketResult` deliberately
carries no email). The dossier reuses both, so a customer is the
same customer in every surface. The known failure modes are
documented on the dossier itself rather than engineered away:
shared inboxes merge histories, multiple addresses split them.
That is the honest semantics of email-keyed support history, and
pretending to resolve identity beyond it would be a claim the
records cannot back.

**The flags are definitions, not vibes.** `repeat_issue` and
`frequent_contact` are computed over the records with thresholds
and windows that live in `dossier.py` as named constants — the
module docstring and the README state them, and each flag carries
its ticket-id evidence. Windows anchor at the newest relevant
stamp (the new ticket's own stamp at triage, the customer's
latest at read time), so the same records always produce the
same flags. Two asymmetries are deliberate: `repeat_issue`
counts *decided* runs (the reviews population — the desk has
actually ruled on them) in the record's triage category, while
`frequent_contact` counts contacts of any status, because volume
is a fact about contact, not about decisions. Records with no
parseable intake stamp still count in totals but can never be
window evidence.

**The decision-call seam is the reopen seam.** The history block
rides the port's existing `context` argument — the graph joins
it after any reopen prior-context — so each backend treats it
exactly as it treats reopen history: the LLM adapter fences it as
`PRIOR HANDLING HISTORY`, Jev sends it as the request's `context`
field, the offline heuristics accept and ignore it. What differs
is the content discipline: the reopen context is one run's record,
while the history block is *arithmetic over many records* —
counts, categories, flag details, the last outcome — composed by
`summarise_history` / `compose_history_block` with no slot a
ticket's text could flow into. A test pins that a prior ticket's
distinctive body phrases appear nowhere in a later prompt. The
run records the summary plus `context_use` in
`CustomerHistoryInfo`, parallel to `ReopenInfo`, and the audit
export carries the same block. Finally, the composition firewall:
`compose` reads the history dict only to *record* it. The golden
composition-invariance test (identical canned answers for a
first-timer and a flagged repeat customer → byte-identical
composition and answers) is the property that keeps "context"
from quietly becoming "input".

The dossier view itself is a pure projection assembled from
`all_records()` plus stored inputs — no new storage, no
migration — grouping thread rows exactly as the thread views do
(`threads.run_row` + the corrections overlay), so a detach is
visible here the moment it lands. Its outcome figures reuse the
workbench's `hours_from_intake` and its `KNOWLEDGE_MIN_SAMPLES`
floor: one floor for "too few to quote", wherever the repo quotes
a rate over outcomes.

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
