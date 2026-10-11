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

Round 10 adds the second agreement question: do the REVIEWERS
agree with EACH OTHER? Reviewer-vs-agent agreement is noise if
two trained reviewers grade the same triage differently — the
standard itself is then unclear, and no agent score built on it
means anything. So a sample can carry an **overlap set**: a stated
count (or fraction) of its drawn runs assigned to two named
reviewers, both of whom grade them. The assignment is part of
the seeded draw — the same RNG stream that shuffled the strata
picks the overlap runs — so it reproduces exactly, and it rides
on the sample record with both names. On overlap runs the
assignment binds: only the two named reviewers may grade them
(the service refuses anyone else, by name), because the paired
measurement is only auditable if the pair is the pair. On every
other run grading stays as open as it always was — any named
reviewer — because the assignment is a floor of responsibility
for the overlap set, not a cage on the whole round.

Over the completed pairs the report computes, per family and
overall, the raw agreement rate and Cohen's kappa — the
chance-corrected figure, from the two raters' own marginals,
in exact fractions. Kappa is withheld below
:data:`KAPPA_MIN_PAIRS` paired runs (the raw rate still shows),
and it is honestly *undefined* — ``None``, with the reason
named — when both raters' marginals have no variance (each said
the same word on every paired run): chance agreement is then
total, there is nothing to correct for, and printing 1.0 would
fabricate a statistic the data does not contain. Paired
disagreements are listed with both verdicts and both notes side
by side: that list is the calibration meeting's agenda, because
the families where raters split are where the standard is
ambiguous.
"""

from __future__ import annotations

import random
from fractions import Fraction

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

# Below this many COMPLETED PAIRS a kappa is withheld (the raw
# agreement rate still shows): kappa divides by (1 − expected
# agreement), so on a handful of pairs a single split swings it
# wildly. Ten pairs is the smallest set on which a chance-corrected
# figure says anything a calibration meeting should hear.
KAPPA_MIN_PAIRS = 10

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


def validate_overlap(
    overlap_reviewers,
    overlap_fraction,
    overlap_count,
    drawn: int,
) -> tuple[list[str], int, dict] | None:
    """Validate an overlap request against the drawn sample size.

    Returns ``(reviewers, k, requested)`` — the two reviewer
    names (stripped), how many of the drawn runs they double-grade,
    and the request as stated (for the sample record) — or ``None``
    when no overlap was requested at all. Anything half-stated is
    refused by name: an overlap is an audit assignment, and an
    assignment nobody can read back exactly is worse than none.
    """
    if overlap_reviewers is None and overlap_fraction is None and overlap_count is None:
        return None
    if not overlap_reviewers:
        raise ReviewError(
            "an overlap names its two reviewers — a fraction or count "
            "without the pair does not assign anyone"
        )
    reviewers = [str(name).strip() for name in overlap_reviewers]
    if len(reviewers) != 2:
        raise ReviewError(
            f"an overlap names exactly two reviewers, got {len(reviewers)}: "
            f"{list(overlap_reviewers)}"
        )
    if any(not name for name in reviewers):
        raise ReviewError("overlap reviewers must be named — an empty name assigns nobody")
    if reviewers[0] == reviewers[1]:
        raise ReviewError(
            f"the overlap reviewers must be two different people, got "
            f"{reviewers[0]!r} twice — a reviewer cannot pair with themselves"
        )
    if (overlap_fraction is None) == (overlap_count is None):
        raise ReviewError(
            "state the overlap as a fraction of the sample or as a count "
            "of runs — exactly one of overlap_fraction / overlap_count"
        )
    if overlap_count is not None:
        if (
            not isinstance(overlap_count, int)
            or isinstance(overlap_count, bool)
            or overlap_count < 1
        ):
            raise ReviewError(
                f"overlap count must be a positive integer, got {overlap_count!r}"
            )
        if overlap_count > drawn:
            raise ReviewError(
                f"overlap count {overlap_count} exceeds the {drawn} runs the "
                "sample drew — the pair cannot double-grade runs that are not there"
            )
        return reviewers, overlap_count, {"count": overlap_count}
    fraction = overlap_fraction
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0 < fraction <= 1
    ):
        raise ReviewError(
            f"overlap fraction must be in (0, 1], got {fraction!r}"
        )
    k = int(fraction * drawn + 0.5)
    if k < 1:
        raise ReviewError(
            f"overlap fraction {fraction} over {drawn} drawn runs rounds to "
            "no runs at all — state a count instead"
        )
    return reviewers, k, {"fraction": fraction}


def draw_sample(
    records: list[TicketResult],
    prior_verdicts: list[ReviewVerdict],
    *,
    size: int,
    seed: int,
    exclude_previously_reviewed: bool = True,
    overlap_reviewers=None,
    overlap_fraction=None,
    overlap_count=None,
) -> tuple[list[str], dict, list[dict], dict]:
    """The stratified, seeded draw. Pure: returns
    ``(run_ids, population_accounting, strata_rows, overlap)``.

    Strata are (queue, band) pairs over decided, non-duplicate,
    (optionally) never-reviewed runs. Allocation is proportional
    to stratum size by largest remainder, capped at each stratum's
    eligible count, with surplus seats re-offered in the same
    remainder order until the request is filled or the population
    is exhausted — the sample record says which happened.

    When an overlap is requested (see :func:`validate_overlap`),
    the SAME RNG stream that shuffled the strata goes on to pick
    the overlap runs out of the drawn ids, so the assignment is a
    pure function of the same inputs as the draw itself — same
    store, seed, size, flags and overlap request → the same pair
    grading the same runs. The returned ``overlap`` dict is empty
    when no overlap was requested; the draw without one consumes
    no extra randomness and is byte-identical to the pre-overlap
    sampler.
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

    overlap: dict = {}
    validated = validate_overlap(
        overlap_reviewers, overlap_fraction, overlap_count, len(run_ids)
    )
    if validated is not None:
        reviewers, k, requested = validated
        picks = list(run_ids)
        rng.shuffle(picks)
        chosen = set(picks[:k])
        overlap = {
            "reviewers": reviewers,
            "requested": requested,
            "count": k,
            # In draw order, so the record reads against run_ids.
            "run_ids": [ticket_id for ticket_id in run_ids if ticket_id in chosen],
        }
    return run_ids, population, strata_rows, overlap


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


def _pair_stats(pairs: list[tuple[str, str]]) -> dict:
    """Raw agreement + Cohen's kappa over paired verdicts.

    ``pairs`` is a list of ``(reviewer_a's word, reviewer_b's word)``
    over runs BOTH assigned reviewers graded, for one family (or
    the overall verdicts). Kappa is computed from the two raters'
    marginals in exact fractions — ``(observed − expected) /
    (1 − expected)`` — and is ``None``, with the reason in
    ``kappa_note``, when it is withheld (below the pair floor) or
    undefined (no variance: expected agreement is total, so the
    correction has no denominator — the raw rate is the whole
    story and a printed 1.0 would be a fabrication).
    """
    n = len(pairs)
    both_agree = sum(1 for a, b in pairs if a == "agree" and b == "agree")
    both_disagree = sum(1 for a, b in pairs if a == "disagree" and b == "disagree")
    split = n - both_agree - both_disagree
    row = {
        "pairs": n,
        "both_agree": both_agree,
        "both_disagree": both_disagree,
        "split": split,
        "agreement_rate": round((both_agree + both_disagree) / n, 4) if n else None,
        "kappa": None,
        "kappa_note": "",
    }
    if n == 0:
        row["kappa_note"] = (
            "no paired verdicts yet — no run has both assigned reviewers' grades"
        )
        return row
    if n < KAPPA_MIN_PAIRS:
        row["kappa_note"] = (
            f"kappa withheld: {n} paired verdict(s) is below the "
            f"{KAPPA_MIN_PAIRS}-pair floor — the raw agreement rate stands, "
            "the chance-corrected figure waits for a bigger overlap"
        )
        return row
    a_agree = sum(1 for a, _ in pairs if a == "agree")
    b_agree = sum(1 for _, b in pairs if b == "agree")
    if a_agree == b_agree and a_agree in (0, n):
        word = "agree" if a_agree == n else "disagree"
        row["kappa_note"] = (
            f"kappa undefined: no variance — both reviewers said {word!r} on "
            "every paired run, so chance agreement is total and there is "
            "nothing to correct for; the raw rate is the whole story"
        )
        return row
    observed = Fraction(both_agree + both_disagree, n)
    expected = Fraction(
        a_agree * b_agree + (n - a_agree) * (n - b_agree), n * n
    )
    row["kappa"] = round(float((observed - expected) / (1 - expected)), 4)
    return row


def _inter_rater(
    samples: list[ReviewSample], in_scope: list[ReviewVerdict]
) -> dict:
    """The reviewer-vs-reviewer block: overlap pairs, kappa, and
    the paired disagreement list (see the module docstring).

    Pairs are formed strictly inside one sample, between the two
    reviewers that sample's overlap assigned — verdicts never pair
    across samples, and a run only pairs when BOTH assigned
    reviewers have graded it there. A pair grades every family at
    once (verdicts are whole), so the pair count is one number;
    each family's stats are computed over the same pairs.
    """
    by_key = {
        (v.sample_id, v.ticket_id, v.reviewer): v for v in in_scope
    }
    completed: list[tuple[str, str, str, ReviewVerdict, str, ReviewVerdict]] = []
    pairs_assigned = 0
    pairs_partial = 0
    reviewers_seen: set[str] = set()
    for sample in samples:
        overlap = sample.overlap or {}
        run_ids = overlap.get("run_ids") or []
        reviewers = overlap.get("reviewers") or []
        if not run_ids or len(reviewers) != 2:
            continue
        reviewer_a, reviewer_b = reviewers
        reviewers_seen.update(reviewers)
        pairs_assigned += len(run_ids)
        for ticket_id in run_ids:
            verdict_a = by_key.get((sample.sample_id, ticket_id, reviewer_a))
            verdict_b = by_key.get((sample.sample_id, ticket_id, reviewer_b))
            if verdict_a is not None and verdict_b is not None:
                completed.append(
                    (sample.sample_id, ticket_id, reviewer_a, verdict_a,
                     reviewer_b, verdict_b)
                )
            elif verdict_a is not None or verdict_b is not None:
                pairs_partial += 1

    def pairs_for(key: str) -> list[tuple[str, str]]:
        if key == "overall":
            return [(va.overall, vb.overall) for _, _, _, va, _, vb in completed]
        return [
            (va.families.get(key, ""), vb.families.get(key, ""))
            for _, _, _, va, _, vb in completed
        ]

    pair_disagreements: list[dict] = []
    for sample_id, ticket_id, reviewer_a, verdict_a, reviewer_b, verdict_b in completed:
        for key in (*FAMILY_KEYS, "overall"):
            word_a = verdict_a.overall if key == "overall" else verdict_a.families.get(key)
            word_b = verdict_b.overall if key == "overall" else verdict_b.families.get(key)
            if word_a != word_b:
                pair_disagreements.append(
                    {
                        "ticket_id": ticket_id,
                        "sample_id": sample_id,
                        "family": key,
                        "reviewer_a": reviewer_a,
                        "verdict_a": word_a,
                        "note_a": verdict_a.note,
                        "reviewer_b": reviewer_b,
                        "verdict_b": word_b,
                        "note_b": verdict_b.note,
                    }
                )
    pair_disagreements.sort(key=lambda row: (row["ticket_id"], row["family"]))

    if not pairs_assigned:
        note = (
            "no overlap sets in scope — draw a sample with two named "
            "reviewers to measure whether the reviewers agree with each other"
        )
    elif not completed:
        note = (
            f"{pairs_assigned} overlap run(s) assigned and none fully graded "
            "yet — a pair counts when both assigned reviewers have graded the run"
        )
    else:
        note = (
            f"reviewer-vs-reviewer over {len(completed)} completed pair(s) "
            f"of {pairs_assigned} assigned"
            + (f"; {pairs_partial} run(s) still wait for one grade" if pairs_partial else "")
        )

    return {
        "reviewers": sorted(reviewers_seen),
        "pairs_assigned": pairs_assigned,
        "pairs_complete": len(completed),
        "pairs_partial": pairs_partial,
        "families": {key: _pair_stats(pairs_for(key)) for key in FAMILY_KEYS},
        "overall": _pair_stats(pairs_for("overall")),
        "disagreements": pair_disagreements,
        "min_pairs_for_kappa": KAPPA_MIN_PAIRS,
        "note": note,
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
        "inter_rater": _inter_rater(samples, in_scope),
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
