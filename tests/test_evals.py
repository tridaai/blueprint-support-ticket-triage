"""The eval packs, run in-process: both gates must hold in the suite too."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EVALS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EVALS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EVALS_DIR))
    return module


def test_golden_cases_all_pass_the_pipeline_checks():
    runner = _load("run_evals")
    cases = runner.load_cases()
    assert len(cases) == 69
    failures = {}
    for case in cases:
        if "template_seed" in case:
            case_failures = runner.check_template_case(runner.fresh_service(), case)
        elif "handoff_seed" in case:
            case_failures = runner.check_handoff_case(runner.fresh_service(), case)
        elif "watch_seed" in case:
            case_failures = runner.check_watch_case(runner.fresh_service(), case)
        elif "incident_seed" in case:
            case_failures = runner.check_incident_case(runner.fresh_service(), case)
        elif "qa_sample" in case:
            case_failures = runner.check_review_case(runner.fresh_service(), case)
        elif "dossier_seed" in case or "dossier_new_ticket" in case:
            case_failures = runner.check_dossier_case(runner.fresh_service(), case)
        elif "operator_actions" in case:
            case_failures = runner.check_operator_case(runner.fresh_service(), case)
        elif "neighbours" in case or "workbench_actions" in case:
            case_failures = runner.check_workbench_case(runner.fresh_service(), case)
        elif "thread_pair" in case:
            case_failures = runner.check_thread_case(runner.fresh_service(), case)
        else:
            result, chain_failures = runner.run_case(runner.fresh_service(), case)
            case_failures = chain_failures + runner.check_case(case, result)
        if case_failures:
            failures[case["id"]] = case_failures
    assert failures == {}


def test_calibration_gates_hold_for_the_offline_backend():
    runner = _load("run_calibration")
    samples, backend = runner.collect_samples()
    from support_triage.calibration import calibration_report

    report = calibration_report(samples, bin_min_n=runner.GATES["bin_min_n"])
    assert backend == "offline"
    for key, entry in report["per_question"].items():
        if entry["kind"] == "verify":
            assert entry["brier"] <= runner.GATES["verify_brier_max"], key
        if entry["kind"] == "choice":
            assert entry["brier"] <= runner.GATES["choice_brier_max"], key
        if entry["kind"] == "score":
            assert entry["high_event_brier"] <= runner.GATES["score_event_brier_max"], key
            assert entry["mae"] <= runner.GATES["score_mae_max"], key
    assert report["worst_bin_gap"] <= runner.GATES["bin_gap_max"]
