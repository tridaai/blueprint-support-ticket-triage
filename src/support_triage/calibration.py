"""Calibration math: do the probabilities mean what they say?

A decision engine that says 0.8 should be right about 80% of the
time. This module scores that claim over labelled cases:

- **Verify questions** — binary Brier score, ``mean((p − y)²)``, plus
  reliability bins: bucket predictions into ten bins and compare each
  bin's mean prediction with its empirical frequency.
- **Choice questions** — multiclass Brier over the full distribution,
  and confidence bins: bucket the top probability, compare with the
  bucket's accuracy.
- **Score questions** — Brier over the eleven score buckets against
  the one-hot truth, plus calibration of the "high" event
  (P(score ≥ 7) vs how often the truth was ≥ 7).

Pure functions over plain data — the eval runner feeds them backend
answers and authored truths; tests feed them by hand.
"""

from __future__ import annotations

BIN_EDGES = [i / 10 for i in range(11)]


def _bin_index(p: float) -> int:
    return min(int(p * 10), 9)


def reliability_bins(pairs: list[tuple[float, float]]) -> list[dict]:
    """Bucket (predicted, outcome) pairs; outcome is 0.0/1.0 for binary
    events or an accuracy flag for confidence bins."""
    bins: list[dict] = []
    for i in range(10):
        members = [(p, y) for p, y in pairs if _bin_index(p) == i]
        if not members:
            bins.append(
                {"bin": f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i + 1]:.1f}", "n": 0,
                 "mean_predicted": None, "empirical": None, "gap": None}
            )
            continue
        mean_predicted = sum(p for p, _ in members) / len(members)
        empirical = sum(y for _, y in members) / len(members)
        bins.append(
            {
                "bin": f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i + 1]:.1f}",
                "n": len(members),
                "mean_predicted": round(mean_predicted, 4),
                "empirical": round(empirical, 4),
                "gap": round(abs(mean_predicted - empirical), 4),
            }
        )
    return bins


def brier_binary(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def brier_multiclass(distributions: list[tuple[dict[str, float], str]]) -> float:
    """Mean over cases of Σ_k (p_k − 1[k = truth])²."""
    if not distributions:
        return 0.0
    total = 0.0
    for dist, truth in distributions:
        total += sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in dist.items())
    return total / len(distributions)


def score_event_probability(dist: dict[str, float], threshold: int = 7) -> float:
    """P(score ≥ threshold) from a score distribution."""
    return sum(p for k, p in dist.items() if int(k) >= threshold)


def calibration_report(samples: list[dict], bin_min_n: int = 8) -> dict:
    """Aggregate calibration over labelled samples.

    Each sample: ``{"verify": {key: (p, truth_bool)}, "choice":
    {key: (distribution, truth)}, "score": {key: (distribution,
    truth_int)}}`` — any subset of sections.

    ``bin_min_n`` is the smallest bin that counts toward the worst-bin
    figure: below ~8 members a bin's empirical rate is noise-dominated
    (a 5-member bin's rate moves in 0.2 steps), so small bins are
    reported but never gate.
    """
    verify_pairs: dict[str, list[tuple[float, float]]] = {}
    choice_pairs: dict[str, list[tuple[float, float]]] = {}
    choice_dists: dict[str, list[tuple[dict[str, float], str]]] = {}
    score_dists: dict[str, list[tuple[dict[str, float], int]]] = {}
    score_events: dict[str, list[tuple[float, float]]] = {}

    for sample in samples:
        for key, (p, truth) in sample.get("verify", {}).items():
            verify_pairs.setdefault(key, []).append((p, 1.0 if truth else 0.0))
        for key, (dist, truth) in sample.get("choice", {}).items():
            top = max(dist.values()) if dist else 0.0
            predicted = max(dist, key=lambda k: dist[k]) if dist else ""
            choice_pairs.setdefault(key, []).append((top, 1.0 if predicted == truth else 0.0))
            choice_dists.setdefault(key, []).append((dist, truth))
        for key, (dist, truth) in sample.get("score", {}).items():
            score_dists.setdefault(key, []).append((dist, truth))
            score_events.setdefault(key, []).append(
                (score_event_probability(dist), 1.0 if truth >= 7 else 0.0)
            )

    per_question: dict[str, dict] = {}
    all_briers: list[float] = []
    worst_bin_gap = 0.0
    worst_bin: dict | None = None

    def _note_bins(question: str, bins: list[dict]) -> None:
        nonlocal worst_bin_gap, worst_bin
        for entry in bins:
            if entry["n"] >= bin_min_n and entry["gap"] is not None and entry["gap"] > worst_bin_gap:
                worst_bin_gap = entry["gap"]
                worst_bin = {"question": question, **entry}

    for key, pairs in sorted(verify_pairs.items()):
        brier = brier_binary(pairs)
        bins = reliability_bins(pairs)
        _note_bins(key, bins)
        per_question[key] = {"kind": "verify", "n": len(pairs), "brier": round(brier, 4), "bins": bins}
        all_briers.append(brier)
    for key, dists in sorted(choice_dists.items()):
        brier = brier_multiclass([(d, t) for d, t in dists])
        bins = reliability_bins(choice_pairs[key])
        _note_bins(key, bins)
        per_question[key] = {"kind": "choice", "n": len(dists), "brier": round(brier, 4), "bins": bins}
        all_briers.append(brier)
    for key, dists in sorted(score_dists.items()):
        brier = brier_multiclass(
            [({str(s): p for s, p in d.items()}, str(t)) for d, t in dists]
        )
        event_brier = brier_binary(score_events[key])
        mae = sum(
            abs(sum(float(s) * p for s, p in d.items()) - t) for d, t in dists
        ) / len(dists)
        per_question[key] = {
            "kind": "score",
            "n": len(dists),
            "brier": round(brier, 4),
            "high_event_brier": round(event_brier, 4),
            "mae": round(mae, 4),
            "bins": reliability_bins(score_events[key]),
        }
        _note_bins(key, per_question[key]["bins"])
        all_briers.append(brier)

    return {
        "cases": len(samples),
        "overall_brier": round(sum(all_briers) / len(all_briers), 4) if all_briers else 0.0,
        "worst_bin_gap": round(worst_bin_gap, 4),
        "worst_bin": worst_bin,
        "per_question": per_question,
    }
