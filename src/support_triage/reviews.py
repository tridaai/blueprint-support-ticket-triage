"""QA review: humans grade the triage itself, on a sample that can't be gamed.

Everything else in this repo that carries the word "calibration"
measures OUTCOMES — did the ticket resolve, was the agent's ground
truth kind to the recorded probabilities. Nobody measured whether
the DECISIONS were right: whether this ticket really was a billing
issue, whether those priority drivers were honestly read, whether
the queue was the right destination. That judgement belongs to a
human reviewer — and ad-hoc reviewing cherry-picks, because a
reviewer handed (or choosing) the pile will reach for the
interesting tickets and grade a fiction.

So the sample is drawn by code, and the draw is the auditable part:

- **Population: decided runs.** "Decided" is the store's own word
  (store.py's promoted column): ``status`` in ``triaged`` /
  ``rejected`` — the run reached the gate and an agent ruled on the
  triage, so the record being graded is final. Awaiting runs are
  still in flight; grading them grades a draft.
- **Stratified by queue and band.** The strata are
  ``(composition.queue, composition.priority_band)`` — the
  composition's own bands (composition.BANDS) — so the draw covers
  every destination and every priority level in proportion, not
  just the loud queues. Allocation is proportional with
  largest-remainder rounding; the sample record carries every
  stratum's eligible and sampled counts, so coverage is checkable.
- **Seeded and reproducible.** Candidates inside a stratum are
  sorted by ticket id, shuffled with ``random.Random(seed)``, and
  drawn in stratum order. Same store, same seed, same size, same
  exclusion flag → the same sample, byte for byte. The seed rides
  on the sample record.
- **Exclusions, named.** A confirmed duplicate (corrections.py)
  never samples — the same discipline as calibration: it is not a
  second opinion about the backend. Runs already reviewed — a
  verdict exists for them in an earlier sample — are excluded by
  default (``exclude_previously_reviewed``), so a reviewer does
  not unknowingly re-grade old work; the request can switch the
  exclusion off, and either way the sample record names who was
  excluded and why.

Verdicts decompose the way the decision port does. The families a
reviewer grades are the port's own shape:

- ``category`` — the choice answers (``category``, ``product_area``);
- ``priority_drivers`` — the score answers (``urgency``,
  ``sentiment``, ``churn_risk``, ``sla_breach_risk``);
- ``routing`` — what composition made of the answers: the queue and
  the priority band. A reviewer can agree with every answer and
  still dispute the destination, so routing is graded on its own;
- ``verify`` — the five verify answers (PII, abuse, legal, outage,
  engineering).

Each family gets agree / disagree, plus an overall verdict and a
note. Verdicts are append-only records tied to the sample; a run's
triage record is never edited by being graded (the suite pins it
byte-identical). One verdict per (sample, run, reviewer): a second
verdict from the same reviewer is refused as a conflict — grades
are testimony, not drafts.

The agreement report is a READ model over the verdicts: per-family
agreement, a per-backend split (the backend is the run record's
telemetry), overall agreement, and the disagreement list. Rates
are withheld below :data:`AGREEMENT_MIN_VERDICTS` verdicts — the
counts always show — because three agreeing verdicts are not a
rate, they are an anecdote (the small-sample honesty the drift
watch and the workbench use). And the report changes nothing: no
weights, no thresholds, no prompts read it. What a disagreement
means is a conversation for the humans; this module's job is to
make sure the conversation is about a fair sample.
"""

from __future__ import annotations

import random

from . import corrections
from .schemas import ReviewSample, ReviewVerdict, TicketResult

# The review families, in the order a reviewer meets them. Each
# carries the question keys it grades (routing grades composition's
# outputs, which are not questions — its key list is empty and the
# surfaces say what it covers instead).
FAMILIES: dict[str, dict] = {
    "category": {
        "label": "Category",
        "covers": "the choice answers: category and product area",
        "question_keys": ["category", "product_area"],
    },
    "priority_drivers": {
        "label": "Priority drivers",
        "covers": "the score answers: urgency, sentiment, churn risk, SLA breach risk",
        "question_keys": ["urgency", "sentiment", "churn_risk", "sla_breach_risk"],
    },
    "routing": {
        "label": "Routing",
        "covers": "the composed destination: queue and priority band",
        "question_keys": [],
    },
    "verify": {
        "label": "Verify answers",
        "covers": (
            "the verify answers: PII, abuse, legal mentions, "
            "outage report, engineering need"
        ),
        "question_keys": [
            "contains_pii",
            "is_abusive",
            "mentions_legal",
            "is_outage_report",
            "needs_engineering",
        ],
    },
}

FAMILY_KEYS = tuple(FAMILIES)

VERDICTS = ("agree", "disagree")

# Below this many verdicts a rate is withheld (counts still shown):
# an agreement rate over a handful of grades is an anecdote wearing
# a percentage sign — the workbench's withholding rule, sized for
# a review round.
AGREEMENT_MIN_VERDICTS = 5

# "Decided" is the store's own definition (the promoted column in
# store.py): the agent ruled on the triage at the gate.
DECIDED_STATUSES = ("triaged", "rejected")


class ReviewError(ValueError):
    """A review action that cannot stand (the CorrectionError idiom).

    ``http_status`` is the API's mapping: 409 for a state conflict
    (a second verdict from the same reviewer), 422 for a request
    invalid on its face (bad size, unknown family, a verdict for a
    run the sample never drew).
    """

    http_status = 422


class ReviewConflict(ReviewError):
    http_status = 409


def is_decided(record: TicketResult) -> bool:
    """The review population's membership test (see module docstring)."""
    return record.status in DECIDED_STATUSES


def previously_reviewed_ids(verdicts: list[ReviewVerdict]) -> set[str]:
    """Runs that already carry a verdict in some earlier sample."""
    return {verdict.ticket_id for verdict in verdicts}


def draw_sample(
    records: list[TicketResult],
    prior_verdicts: list[ReviewVerdict],
    *,
    size: int,
    seed: int,
    exclude_previously_reviewed: bool = True,
) -> tuple[list[str], dict, list[dict]]:
    """The stratified, seeded draw. Pure: returns
    ``(run_ids, population_accounting, strata_rows)``.

    Strata are (queue, band) pairs over decided, non-duplicate,
    (optionally) never-reviewed runs. Allocation is proportional
    to stratum size by largest remainder, capped at each stratum's
    eligible count, with surplus seats re-offered in the same
    remainder order until the request is filled or the population
    is exhausted — the sample record says which happened.
    """
    decided = [r for r in records if is_decided(r)]
    duplicates = [r for r in decided if corrections.is_confirmed_duplicate(r)]
    reviewed = previously_reviewed_ids(prior_verdicts)
    eligible = [r for r in decided if not corrections.is_confirmed_duplicate(r)]
    excluded_reviewed: list[str] = []
    if exclude_previously_reviewed:
        excluded_reviewed = sorted(
            r.ticket_id for r in eligible if r.ticket_id in reviewed
        )
        eligible = [r for r in eligible if r.ticket_id not in reviewed]

    strata_map: dict[tuple[str, str], list[str]] = {}
    for record in eligible:
        key = (record.composition.queue, record.composition.priority_band)
        strata_map.setdefault(key, []).append(record.ticket_id)
    for ids in strata_map.values():
        ids.sort()

    total = len(eligible)
    wanted = min(size, total)
    allocation: dict[tuple[str, str], int] = {key: 0 for key in strata_map}
    if total and wanted:
        raw = {key: wanted * len(ids) / total for key, ids in strata_map.items()}
        allocation = {
            key: min(int(raw[key]), len(strata_map[key])) for key in strata_map
        }
        seats = wanted - sum(allocation.values())
        # Largest remainder first; the stratum key breaks ties so the
        # order — and therefore the draw — never depends on dict order.
        order = sorted(
            strata_map,
            key=lambda key: (-(raw[key] - int(raw[key])), key),
        )
        while seats > 0:
            progressed = False
            for key in order:
                if seats == 0:
                    break
                if allocation[key] < len(strata_map[key]):
                    allocation[key] += 1
                    seats -= 1
                    progressed = True
            if not progressed:
                break

    rng = random.Random(seed)
    run_ids: list[str] = []
    strata_rows: list[dict] = []
    for key in sorted(strata_map):
        ids = list(strata_map[key])
        rng.shuffle(ids)
        take = ids[: allocation[key]]
        run_ids.extend(take)
        strata_rows.append(
            {
                "queue": key[0],
                "priority_band": key[1],
                "eligible": len(strata_map[key]),
                "sampled": len(take),
                "run_ids": take,
            }
        )

    population = {
        "decided_total": len(decided),
        "excluded_duplicates": sorted(r.ticket_id for r in duplicates),
        "excluded_previously_reviewed": excluded_reviewed,
        "eligible": total,
    }
    return run_ids, population, strata_rows


def validate_verdict_families(families: dict) -> dict[str, str]:
    """A verdict grades every family, each agree / disagree — strict.

    A partial verdict is not a verdict: the decomposition is the
    point, and a reviewer who cannot judge a family says so in the
    note with an ``agree`` they stand behind, not by omitting it.
    Unknown families and unknown verdict words are errors, never
    silently dropped.
    """
    families = families or {}
    unknown = sorted(set(families) - set(FAMILY_KEYS))
    if unknown:
        raise ReviewError(
            f"unknown review families {unknown}; the families are {list(FAMILY_KEYS)}"
        )
    missing = [key for key in FAMILY_KEYS if key not in families]
    if missing:
        raise ReviewError(
            f"a verdict grades every family; missing {missing}"
        )
    cleaned: dict[str, str] = {}
    for key in FAMILY_KEYS:
        value = str(families[key]).strip().lower()
        if value not in VERDICTS:
            raise ReviewError(
                f"family {key!r} verdict must be one of {list(VERDICTS)}, got {families[key]!r}"
            )
        cleaned[key] = value
    return cleaned


def validate_overall(overall: str) -> str:
    value = (overall or "").strip().lower()
    if value not in VERDICTS:
        raise ReviewError(
            f"overall verdict must be one of {list(VERDICTS)}, got {overall!r}"
        )
    return value


def _rate(agree: int, n: int) -> float | None:
    """The agreement rate, or None when the sample is too small to
    quote one (the count still shows — see AGREEMENT_MIN_VERDICTS)."""
    if n < AGREEMENT_MIN_VERDICTS:
        return None
    return round(agree / n, 4)


def _tally(verdicts: list[ReviewVerdict]) -> dict:
    """Counts + rates over a verdict set: overall and per family."""
    n = len(verdicts)
    overall_agree = sum(1 for v in verdicts if v.overall == "agree")
    families: dict[str, dict] = {}
    for key in FAMILY_KEYS:
        fam_n = sum(1 for v in verdicts if key in v.families)
        fam_agree = sum(1 for v in verdicts if v.families.get(key) == "agree")
        families[key] = {
            "verdicts": fam_n,
            "agree": fam_agree,
            "disagree": fam_n - fam_agree,
            "rate": _rate(fam_agree, fam_n),
        }
    return {
        "verdicts": n,
        "overall_agree": overall_agree,
        "overall_disagree": n - overall_agree,
        "overall_rate": _rate(overall_agree, n),
        "families": families,
    }


def agreement_report(
    samples: list[ReviewSample],
    verdicts: list[ReviewVerdict],
    records_by_id: dict[str, TicketResult],
) -> dict:
    """The agreement read model over a set of samples' verdicts.

    Pure and read-only: nothing here feeds a weight, a threshold, or
    a prompt. The per-backend split reads the backend off each
    verdicted run's record (telemetry) — the judgement being graded
    is the backend's answers, so the split is by the backend that
    produced them.
    """
    sample_ids = [s.sample_id for s in samples]
    in_scope = [v for v in verdicts if v.sample_id in set(sample_ids)]
    tally = _tally(in_scope)

    backends: dict[str, list[ReviewVerdict]] = {}
    for verdict in in_scope:
        record = records_by_id.get(verdict.ticket_id)
        backend = (
            record.telemetry.backend
            if record is not None and record.telemetry is not None
            else ""
        ) or "unknown"
        backends.setdefault(backend, []).append(verdict)
    by_backend = {
        backend: _tally(group) for backend, group in sorted(backends.items())
    }

    disagreements: list[dict] = []
    for verdict in in_scope:
        for key in FAMILY_KEYS:
            if verdict.families.get(key) == "disagree":
                disagreements.append(
                    {
                        "ticket_id": verdict.ticket_id,
                        "sample_id": verdict.sample_id,
                        "family": key,
                        "reviewer": verdict.reviewer,
                        "note": verdict.note,
                    }
                )
        if verdict.overall == "disagree":
            disagreements.append(
                {
                    "ticket_id": verdict.ticket_id,
                    "sample_id": verdict.sample_id,
                    "family": "overall",
                    "reviewer": verdict.reviewer,
                    "note": verdict.note,
                }
            )
    disagreements.sort(key=lambda row: (row["ticket_id"], row["family"], row["reviewer"]))

    sampled_runs = sum(len(s.run_ids) for s in samples)
    verdicted_runs = len({v.ticket_id for v in in_scope})
    withheld = (
        tally["overall_rate"] is None and tally["verdicts"] > 0
    )
    if not in_scope:
        note = "no verdicts recorded yet — draw a sample and grade it"
    elif withheld:
        note = (
            f"rates withheld: {tally['verdicts']} verdict(s) is below the "
            f"{AGREEMENT_MIN_VERDICTS}-verdict floor for quoting an agreement "
            "rate — the counts stand, the percentages wait"
        )
    else:
        note = f"agreement over {tally['verdicts']} verdict(s) on {verdicted_runs} run(s)"

    return {
        "samples": sample_ids,
        "sampled_runs": sampled_runs,
        "verdicted_runs": verdicted_runs,
        "verdicts": tally["verdicts"],
        "overall_agree": tally["overall_agree"],
        "overall_disagree": tally["overall_disagree"],
        "overall_rate": tally["overall_rate"],
        "families": tally["families"],
        "by_backend": by_backend,
        "disagreements": disagreements,
        "min_verdicts_for_rate": AGREEMENT_MIN_VERDICTS,
        "note": note,
    }


def review_entries_for(
    ticket_id: str, verdicts: list[ReviewVerdict]
) -> list[dict]:
    """Every verdict ever recorded on one run, oldest first — the
    audit export's review block for that run."""
    entries = [
        verdict.model_dump() for verdict in verdicts if verdict.ticket_id == ticket_id
    ]
    entries.sort(key=lambda entry: (entry["recorded_at"], entry["sample_id"]))
    return entries
