"""Calibration eval: do the probabilities mean what they say?

Runs the golden set and scores the returned distributions against
the authored truths (``support_triage.calibration``): reliability
bins and Brier scores per question plus summary figures.

Two modes:

- **Offline (default)** — the deterministic rules backend. Metrics
  are REGRESSION GATES, per question kind, because the kinds have
  different units and a single blended number would gate nothing
  meaningful:

  - verify: binary Brier ≤ 0.10 per question;
  - category (choice): multiclass Brier ≤ 0.25;
  - scores: "high event" (score ≥ 7) binary Brier ≤ 0.10 and
    expected-score MAE ≤ 1.75 per question — an eleven-bucket
    one-hot Brier mostly measures honest ±1 uncertainty, so it is
    printed for the record but not gated;
  - reliability bins: no bin with n ≥ 8 may sit more than 0.35 from
    its empirical rate, across all binned questions (smaller bins are
    printed but noise-dominated, so they never gate);

  Exit non-zero on any gate failure.

- **Live (``EVAL_LIVE=1``)** — the engine named by DECISION_BACKEND
  (llm / jev / a provider alias) answers instead. Metrics are printed
  and saved, never gated: a live model's calibration is a measurement
  to read, and the report says so.

Output: ``evals/calibration_results.json`` + the packaged copy the
console reads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_evals import fresh_service, load_cases, run_case  # noqa: E402

from support_triage.calibration import calibration_report  # noqa: E402

GATES = {
    "verify_brier_max": 0.10,
    "choice_brier_max": 0.25,
    "score_event_brier_max": 0.10,
    "score_mae_max": 1.75,
    "bin_gap_max": 0.35,
    "bin_min_n": 8,
}


def collect_samples(engine=None) -> tuple[list[dict], str]:
    cases = load_cases()
    samples: list[dict] = []
    backend_name = "offline"
    for case in cases:
        service = fresh_service(engine)
        result = run_case(service, case)
        backend_name = result.telemetry.backend if result.telemetry else backend_name
        expected = case["expected"]
        sample: dict = {"verify": {}, "choice": {}, "score": {}}
        for key, truth in expected["verify"].items():
            answer = result.answer(key)
            sample["verify"][key] = (float(answer.value), bool(truth))
        category = result.answer("category")
        sample["choice"]["category"] = (category.probabilities, expected["category"])
        for key, truth in expected["scores"].items():
            answer = result.answer(key)
            sample["score"][key] = (answer.probabilities, int(truth))
        samples.append(sample)
    return samples, backend_name


def print_report(report: dict, backend: str, live: bool) -> None:
    mode = "LIVE — reported, not gated" if live else "offline — regression gates apply"
    print(f"calibration (backend={backend}, {mode}) over {report['cases']} cases")
    print(f"  overall Brier: {report['overall_brier']}  (lower is better; 0.25 ≈ base-rate guessing)")
    print(f"  worst reliability-bin gap (n≥8): {report['worst_bin_gap']}"
          + (f"  [{report['worst_bin']['question']} bin {report['worst_bin']['bin']}: "
             f"predicted {report['worst_bin']['mean_predicted']} vs empirical {report['worst_bin']['empirical']}]"
             if report["worst_bin"] else ""))
    for key, entry in report["per_question"].items():
        extra = ""
        if entry["kind"] == "score":
            extra = f", high-event Brier {entry['high_event_brier']}, MAE {entry['mae']}"
        print(f"  {key:<18} {entry['kind']:<6} n={entry['n']:<3} Brier {entry['brier']}{extra}")
    print("  reliability bins (verify + category confidence + score high-event):")
    for key, entry in report["per_question"].items():
        filled = [b for b in entry["bins"] if b["n"]]
        if not filled:
            continue
        rendered = " · ".join(
            f"{b['bin']}: n={b['n']} pred {b['mean_predicted']} vs emp {b['empirical']}"
            for b in filled
        )
        print(f"    {key:<18} {rendered}")


def main() -> int:
    live = os.environ.get("EVAL_LIVE") == "1"
    engine = None
    if live:
        from support_triage.backends import get_decision_engine

        engine = get_decision_engine()
        if engine.name == "offline":
            print("EVAL_LIVE=1 but DECISION_BACKEND resolves to offline — set DECISION_BACKEND=llm or =jev")
            return 1
    samples, backend = collect_samples(engine)
    report = calibration_report(samples, bin_min_n=GATES["bin_min_n"])
    report["backend"] = backend
    report["live"] = live
    report["gates"] = GATES

    out = Path(__file__).resolve().parent / "calibration_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src" / "support_triage" / "data" / "calibration_results.json"
    )
    packaged.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report, backend, live)
    if live:
        return 0
    failed: list[str] = []
    for key, entry in report["per_question"].items():
        if entry["kind"] == "verify" and entry["brier"] > GATES["verify_brier_max"]:
            failed.append(f"{key} verify Brier {entry['brier']} > {GATES['verify_brier_max']}")
        if entry["kind"] == "choice" and entry["brier"] > GATES["choice_brier_max"]:
            failed.append(f"{key} choice Brier {entry['brier']} > {GATES['choice_brier_max']}")
        if entry["kind"] == "score":
            if entry["high_event_brier"] > GATES["score_event_brier_max"]:
                failed.append(
                    f"{key} high-event Brier {entry['high_event_brier']} > {GATES['score_event_brier_max']}"
                )
            if entry["mae"] > GATES["score_mae_max"]:
                failed.append(f"{key} MAE {entry['mae']} > {GATES['score_mae_max']}")
    if report["worst_bin_gap"] > GATES["bin_gap_max"]:
        failed.append(f"worst bin gap {report['worst_bin_gap']} > {GATES['bin_gap_max']}")
    if failed:
        print("GATE FAILED: " + "; ".join(failed))
        return 1
    print("gate: PASS (per-kind thresholds — see module docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
