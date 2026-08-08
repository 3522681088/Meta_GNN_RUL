"""Experiment A14_1: independent model-seed and endpoint confirmation of A14.

This is a pre-registered protocol replication of A14, not a new warning-model
search.  It keeps exactly the A14 features, equal score weights, selection NASA
75th-percentile risk-event definition, four score-quantile candidates, and
success criteria.  The separation is:

* Threshold fitting: model seeds 100, 101, 102; selection endpoint seeds
  9201--9205.
* Confirmation: held-out model seeds 103, 104; confirmation endpoint seeds
  9301--9305.

No numeric threshold from A14 is reused.  A14_1 independently recalibrates
the *same pre-specified rule* on its fitting subset, then applies it unchanged
to the held-out subset.  A10's own A9 blend remains selection-only for every
model seed, as it was in the source-ablation experiment.

This is CPU analysis only: it reuses A10 endpoint outputs and A12 coverage
metadata, trains no predictor, changes no prediction, and never accesses the
official C-MAPSS test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA10_source_domain_ablation_robustness as a10  # noqa: E402
from scripts import experimentA14_source_availability_risk_warning_validation as a14  # noqa: E402


SCRIPT_VERSION = "experimentA14_1_independent_seed_endpoint_risk_warning_confirmation_v1"
EXPERIMENT_ID = "experimentA14_1"
DEFAULT_OUTPUT = "outputs/experimentA14_1_independent_seed_endpoint_risk_warning_confirmation"
DEFAULT_A10_OUTPUT = a10.DEFAULT_OUTPUT
DEFAULT_A12_OUTPUT = "outputs/experimentA12_coverage_aware_source_weighting_training_only"
DEFAULT_A13_OUTPUT = "outputs/experimentA13_selection_gated_source_policy_confirmation"
DEFAULT_A14_OUTPUT = a14.DEFAULT_OUTPUT

FIT_MODEL_SEEDS = (100, 101, 102)
CONFIRMATION_MODEL_SEEDS = (103, 104)
SELECTION_ENDPOINT_SEEDS = (9201, 9202, 9203, 9204, 9205)
CONFIRMATION_ENDPOINT_SEEDS = (9301, 9302, 9303, 9304, 9305)

QUESTION = (
    "Does the A14 selection-locked source-availability warning, recalibrated "
    "only with model seeds 100-102 and new selection endpoints, reproduce "
    "cross-condition NASA/absolute-error risk stratification on held-out model "
    "seeds 103-104 and new confirmation endpoints?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A14_1 independent risk-warning confirmation")
    parser.add_argument("--output-dir")
    parser.add_argument("--a10-output-dir")
    parser.add_argument("--a12-output-dir")
    parser.add_argument("--a13-output-dir")
    parser.add_argument("--a14-output-dir")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> Path:
    return Path(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, value: Any) -> None:
    a1.atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def root_paths(output: Path) -> dict[str, Path]:
    p = EXPERIMENT_ID
    return {
        "manifest": output / f"{p}_manifest.json",
        "dry": output / f"{p}_dry_run.json",
        "integrity": output / f"{p}_reference_input_integrity.json",
        "causality": output / f"{p}_risk_warning_causality_audit.json",
        "coverage": output / f"{p}_source_coverage_features.csv",
        "parameters": output / f"{p}_selection_warning_parameters.csv",
        "selection_curve": output / f"{p}_selection_warning_threshold_grid.csv",
        "selection": output / f"{p}_selection_risk_predictions.csv",
        "confirmation": output / f"{p}_confirmation_risk_predictions.csv",
        "pairs": output / f"{p}_confirmation_warning_pair_metrics.csv",
        "conditions": output / f"{p}_source_ablation_warning_summary.csv",
        "stages": output / f"{p}_stage_warning_summary.csv",
        "domains": output / f"{p}_target_domain_warning_summary.csv",
        "decision": output / f"{p}_confirmation_decision.json",
    }


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    formal = not args.quick
    return {
        "experiment_id": EXPERIMENT_ID,
        "output_dir": str(resolved(args.output_dir, DEFAULT_OUTPUT)),
        "a10_output_dir": str(resolved(args.a10_output_dir, DEFAULT_A10_OUTPUT)),
        "a12_output_dir": str(resolved(args.a12_output_dir, DEFAULT_A12_OUTPUT)),
        "a13_output_dir": str(resolved(args.a13_output_dir, DEFAULT_A13_OUTPUT)),
        "a14_output_dir": str(resolved(args.a14_output_dir, DEFAULT_A14_OUTPUT)),
        "fit_model_seeds": list(FIT_MODEL_SEEDS if formal else (100,)),
        "confirmation_model_seeds": list(CONFIRMATION_MODEL_SEEDS if formal else (101,)),
        "selection_endpoint_seeds": list(SELECTION_ENDPOINT_SEEDS if formal else (9201,)),
        "confirmation_endpoint_seeds": list(CONFIRMATION_ENDPOINT_SEEDS if formal else (9301,)),
        "reference_prediction": "A10_uniform_source_A9_blend",
        "coverage_feature": "mean_distance + 0.5 * distance_spread",
        "disagreement_feature": "abs(prediction_baseline - prediction_cycle_age)",
        "fixed_score_weights": a14.RISK_SCORE_WEIGHTS,
        "warning_score_quantile_grid": list(a14.WARNING_SCORE_QUANTILES),
        "selection_risk_event_quantile": a14.SELECTION_RISK_EVENT_QUANTILE,
        "minimum_warning_rate": a14.MIN_WARNING_RATE,
        "maximum_warning_rate": a14.MAX_WARNING_RATE,
        "minimum_passing_source_ablation_conditions": a14.MIN_CONDITION_PASSES,
        "minimum_passing_target_domains": a14.MIN_TARGET_DOMAIN_PASSES,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "quick_mode": bool(args.quick),
    }


def expected_counts(experiment: dict[str, Any]) -> dict[str, int]:
    domains = list(a10.DOMAINS) if not experiment["quick_mode"] else ["FD004"]
    condition_count = len(domains) * 3
    return {
        "source_conditions": condition_count,
        "selection_parameter_sets": condition_count,
        "confirmation_pairs": (
            condition_count
            * len(experiment["confirmation_model_seeds"])
            * len(a10.TARGET_SPLIT_SEEDS if not experiment["quick_mode"] else [6401])
            * len(a10.ROLE_PARTITIONS if not experiment["quick_mode"] else [1])
            * len(experiment["confirmation_endpoint_seeds"])
        ),
    }


def select_raw(frame: pd.DataFrame, seeds: list[int], quick: bool) -> pd.DataFrame:
    output = frame[frame["model_seed"].astype(int).isin(list(map(int, seeds)))].copy()
    if quick:
        output = output[
            (output["target_domain"] == "FD004")
            & (output["target_split_seed"].astype(int) == 6401)
        ].copy()
    if output.empty:
        raise RuntimeError(f"A14_1 has no A10 records for model seeds {seeds}")
    return output


def evaluation_config(experiment: dict[str, Any], *, seeds: list[int]) -> dict[str, Any]:
    return {
        "domains": list(a10.DOMAINS) if not experiment["quick_mode"] else ["FD004"],
        "model_seeds": list(map(int, seeds)),
        "target_split_seeds": list(a10.TARGET_SPLIT_SEEDS) if not experiment["quick_mode"] else [6401],
        "role_partitions": list(a10.ROLE_PARTITIONS) if not experiment["quick_mode"] else [1],
        "selection_endpoint_seeds": list(map(int, experiment["selection_endpoint_seeds"])),
        "confirmation_endpoint_seeds": list(map(int, experiment["confirmation_endpoint_seeds"])),
        "high_rul_threshold": 60.0,
        "alpha_grid": list(a10.ALPHA_GRID),
        "prediction_gate_threshold": a10.GATE_THRESHOLD,
        "selection_safety_margin_pct": 3.0,
        "stage_noninferiority_margin_pct": 3.0,
    }


def validate_inputs(experiment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # A14 validates that A10/A12/A13 were completed formal training-only
    # experiments.  Keep it formal even for A14_1 --quick so raw inputs are
    # never mistaken for quick-mode source artifacts.
    a14_inputs, prior = a14.validate_inputs({
        "a10_output_dir": experiment["a10_output_dir"],
        "a12_output_dir": experiment["a12_output_dir"],
        "a13_output_dir": experiment["a13_output_dir"],
        "quick_mode": False,
    })
    a14_directory = Path(experiment["a14_output_dir"])
    a14_manifest_path = a14_directory / "experimentA14_manifest.json"
    a14_decision_path = a14_directory / "experimentA14_confirmation_decision.json"
    a14_manifest, a14_decision = read_json(a14_manifest_path), read_json(a14_decision_path)
    a14.assert_training_only(a14_manifest, a14_decision, "A14")
    if not a14_decision.get("passed"):
        raise RuntimeError("A14_1 requires the completed, passed A14 direction")
    a14_inputs["integrity"].update({
        "a14_output_dir": str(a14_directory),
        "a14_manifest_hash": a1.file_sha256(a14_manifest_path),
        "a14_decision_hash": a1.file_sha256(a14_decision_path),
        "a14_passed": True,
        "a14_official_test_files_accessed": False,
        "a14_official_test_forward_run": False,
    })
    return a14_inputs, prior


def validate_existing_manifest(paths: dict[str, Path], manifest: dict[str, Any], resume: bool) -> None:
    if not paths["manifest"].is_file():
        return
    previous = read_json(paths["manifest"])
    for key in ("experiment_config", "input_integrity", "registered_primary_question"):
        if previous.get(key) != manifest.get(key):
            raise RuntimeError(f"existing A14_1 output is incompatible at {key}; use a new output directory")
    if previous.get("script_hash") != manifest["script_hash"] and not resume:
        raise RuntimeError("A14_1 script changed; use --resume only after reviewing the change")
    if previous.get("script_hash") != manifest["script_hash"]:
        manifest["resumed_from_script_hash"] = previous.get("script_hash")


def main() -> None:
    args = parse_args()
    experiment = load_config(args)
    if set(experiment["fit_model_seeds"]) & set(experiment["confirmation_model_seeds"]):
        raise RuntimeError("A14_1 threshold-fit and confirmation model-seed sets must be disjoint")
    if set(experiment["selection_endpoint_seeds"]) & set(experiment["confirmation_endpoint_seeds"]):
        raise RuntimeError("A14_1 selection and confirmation endpoint-seed sets must be disjoint")
    output = Path(experiment["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    inputs, prior = validate_inputs(experiment)
    expected = expected_counts(experiment)
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "mode": "analysis_only_independent_model_seed_and_endpoint_confirmation",
        "new_predictor_training": False,
        "gpu_workers": 0,
        "threshold_fit_model_seeds": experiment["fit_model_seeds"],
        "confirmation_model_seeds": experiment["confirmation_model_seeds"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "model_seed_sets_disjoint": True,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "expected_source_ablation_conditions": expected["source_conditions"],
        "expected_selection_parameter_sets": expected["selection_parameter_sets"],
        "expected_confirmation_pairs": expected["confirmation_pairs"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "experiment_config": experiment,
        "input_integrity": inputs["integrity"],
        "registered_primary_question": QUESTION,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    validate_existing_manifest(paths, manifest, args.resume)
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["dry"], dry)
    atomic_json(paths["integrity"], inputs["integrity"])
    atomic_json(paths["causality"], {
        "experiment_id": EXPERIMENT_ID,
        "replicates": "experimentA14",
        "analysis_mode": "selection_locked_warning_independent_model_seed_and_endpoint_confirmation",
        "risk_features_unchanged_from_A14": True,
        "risk_score_weights_unchanged_from_A14": a14.RISK_SCORE_WEIGHTS,
        "risk_event_quantile_unchanged_from_A14": a14.SELECTION_RISK_EVENT_QUANTILE,
        "warning_score_quantile_grid_unchanged_from_A14": list(a14.WARNING_SCORE_QUANTILES),
        "a14_numeric_thresholds_reused": False,
        "threshold_fit_model_seeds": experiment["fit_model_seeds"],
        "confirmation_model_seeds": experiment["confirmation_model_seeds"],
        "model_seed_sets_disjoint": True,
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "risk_score_uses_target_labels": False,
        "selection_labels_used_only_for": ["NASA risk-event cutoff", "warning score-threshold choice"],
        "confirmation_labels_used_for_warning_choice": False,
        "true_RUL_used_as_warning_feature": False,
        "future_windows_used": False,
        "prediction_values_changed": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return

    print("[A14_1] reconstructing threshold-fit A10 predictions (model seeds 100-102)...", flush=True)
    fit_raw = select_raw(inputs["a10_endpoints"], experiment["fit_model_seeds"], bool(experiment["quick_mode"]))
    fit = a10.crossfit(fit_raw, inputs["protocol"], evaluation_config(experiment, seeds=experiment["fit_model_seeds"]))
    print("[A14_1] reconstructing held-out confirmation A10 predictions (model seeds 103-104)...", flush=True)
    confirmation_raw = select_raw(inputs["a10_endpoints"], experiment["confirmation_model_seeds"], bool(experiment["quick_mode"]))
    heldout = a10.crossfit(confirmation_raw, inputs["protocol"], evaluation_config(experiment, seeds=experiment["confirmation_model_seeds"]))

    coverage = a14.coverage_features(inputs["a12_parameters"])
    if experiment["quick_mode"]:
        coverage = coverage[
            (coverage["target_domain"] == "FD004")
            & (coverage["target_split_seed"].astype(int) == 6401)
        ].copy()
    selection = a14.attach_coverage(fit["selection_prediction"], coverage)
    confirmation = a14.attach_coverage(heldout["confirmation_prediction"], coverage)
    actual_pairs = int(confirmation.groupby(a10.PAIR_KEYS).ngroups)
    if actual_pairs != expected["confirmation_pairs"]:
        raise RuntimeError(f"A14_1 confirmation pair count incomplete: {actual_pairs} != {expected['confirmation_pairs']}")
    print("[A14_1] locking unchanged A14 warning protocol on the fit subset...", flush=True)
    parameters, selection_grid, selection = a14.choose_warning_parameters(selection, experiment)
    if len(parameters) != expected["selection_parameter_sets"]:
        raise RuntimeError("A14_1 warning parameter-set count is incomplete")
    confirmation = a14.apply_locked_warning(confirmation, parameters)
    pairs = a14.pair_warning_metrics(confirmation)
    print(f"[A14_1] evaluating {len(confirmation)} held-out confirmation records across {actual_pairs} paired cells...", flush=True)
    conditions, stages, domains = a14.confirmation_summaries(confirmation, pairs, experiment)
    complete = bool(
        len(parameters) == expected["selection_parameter_sets"]
        and actual_pairs == expected["confirmation_pairs"]
        and len(conditions) == expected["source_conditions"]
    )
    condition_passes = int(conditions["condition_warning_supported"].astype(bool).sum())
    domain_passes = int(domains["target_domain_warning_supported"].astype(bool).sum())
    passed = bool(
        complete
        and condition_passes >= int(experiment["minimum_passing_source_ablation_conditions"])
        and domain_passes >= int(experiment["minimum_passing_target_domains"])
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "new_predictor_training": False,
        "replication_of": "experimentA14",
        "reference_prediction": "A10 uniform-source A9 blend",
        "threshold_fit_model_seeds": experiment["fit_model_seeds"],
        "confirmation_model_seeds": experiment["confirmation_model_seeds"],
        "model_seed_sets_disjoint": True,
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "expected_source_ablation_conditions": expected["source_conditions"],
        "completed_source_ablation_conditions": int(len(conditions)),
        "expected_selection_parameter_sets": expected["selection_parameter_sets"],
        "completed_selection_parameter_sets": int(len(parameters)),
        "expected_confirmation_pairs": expected["confirmation_pairs"],
        "completed_confirmation_pairs": actual_pairs,
        "warning_rate_mean": float(conditions["warning_rate"].mean()),
        "warning_rate_range": [float(conditions["warning_rate"].min()), float(conditions["warning_rate"].max())],
        "passing_source_ablation_conditions": condition_passes,
        "minimum_passing_source_ablation_conditions": int(experiment["minimum_passing_source_ablation_conditions"]),
        "passing_target_domains": domain_passes,
        "minimum_passing_target_domains": int(experiment["minimum_passing_target_domains"]),
        "mean_nasa_warning_to_clear_ratio": float(conditions["nasa_warning_to_clear_ratio"].mean()),
        "mean_absolute_error_warning_to_clear_ratio": float(conditions["absolute_error_warning_to_clear_ratio"].mean()),
        "mean_score_nasa_spearman": float(conditions["score_nasa_spearman"].mean()),
        "mean_balanced_accuracy": float(conditions["balanced_accuracy"].mean()),
        "passed": passed if not args.quick else complete,
        "reason": (
            "A14_1 independently confirmed A14 source-availability risk stratification on held-out model seeds and endpoints"
            if passed else
            "A14_1 completed, but the A14 warning protocol did not reproduce every registered cross-condition criterion on held-out model seeds and endpoints"
        ),
        "next_action": (
            "report_replicated_source_availability_warning_scope" if passed else
            "stop_source_availability_warning_extension_and_report_A14_as_preliminary"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for name, frame in (
        ("coverage", coverage),
        ("parameters", parameters),
        ("selection_curve", selection_grid),
        ("selection", selection),
        ("confirmation", confirmation),
        ("pairs", pairs),
        ("conditions", conditions),
        ("stages", stages),
        ("domains", domains),
    ):
        a1.atomic_write_text(paths[name], frame.to_csv(index=False))
    atomic_json(paths["decision"], decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
