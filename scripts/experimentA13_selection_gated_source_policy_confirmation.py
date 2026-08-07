"""Experiment A13: selection-gated source-policy confirmation.

A12 found that continuous coverage-aware two-source pretraining was not a
safe universal replacement for the locked uniform-source A9 blend.  A13 does
not retune its temperature, clipping limits, or neural predictor.  Instead it
uses *only A10/A12 already-completed training-only endpoint outputs*.

For each target/held-out-source/model-seed/target-split/role cell, independent
selection engines decide whether the A12 coverage-weighted branch may be used.
It must improve selection RMSE and be non-inferior (3%) for NASA/RMSE in both
true-RUL stages.  Otherwise A13 deterministically falls back to the uniform
A10 branch.  Disjoint confirmation engines evaluate the locked decisions.

No official C-MAPSS test file is read.  This is an analysis/policy-confirmation
experiment, so it does not launch workers or consume GPUs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA10_source_domain_ablation_robustness as a10  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402


SCRIPT_VERSION = "experimentA13_selection_gated_source_policy_confirmation_v1"
EXPERIMENT_ID = "experimentA13"
DEFAULT_OUTPUT = "outputs/experimentA13_selection_gated_source_policy_confirmation"
DEFAULT_A10_OUTPUT = a10.DEFAULT_OUTPUT
DEFAULT_A12_OUTPUT = "outputs/experimentA12_coverage_aware_source_weighting_training_only"
REFERENCE = "uniform_source_a9_blend"
WEIGHTED = "coverage_aware_source_weighted_a9_blend"
POLICY = "selection_gated_source_policy"
MARGIN = 0.03
MIN_WEIGHTED_SELECTION_RATE = 0.20
MIN_WEIGHTED_CONDITIONS = 2
MIN_ROBUST_CONDITIONS = 9
QUESTION = (
    "Does a selection-only gate use coverage-aware source weighting only in "
    "cells with selection RMSE improvement and stage safety, while preserving "
    "source-ablation robustness against the locked uniform-source A9 reference?"
)

POLICY_KEYS = [
    "target_domain", "heldout_source_domain", "model_seed",
    "target_split_seed", "role_partition",
]
PAIR_KEYS = a10.PAIR_KEYS
PRED_KEYS = a9.PRED_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A13 selection-gated source policy")
    parser.add_argument("--output-dir")
    parser.add_argument("--a10-output-dir")
    parser.add_argument("--a12-output-dir")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> Path:
    return Path(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty input is missing: {path}")
    return pd.read_csv(path)


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def root_paths(output: Path) -> dict[str, Path]:
    prefix = EXPERIMENT_ID
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "dry": output / f"{prefix}_dry_run.json",
        "input_integrity": output / f"{prefix}_reference_input_integrity.json",
        "causality": output / f"{prefix}_policy_causality_audit.json",
        "gate": output / f"{prefix}_selection_gate_decisions.csv",
        "selection_pairs": output / f"{prefix}_selection_paired_weighted_vs_uniform.csv",
        "confirmation": output / f"{prefix}_confirmation_endpoint_predictions.csv",
        "paired": output / f"{prefix}_paired_policy_vs_uniform.csv",
        "high": output / f"{prefix}_high_rul_paired_policy_vs_uniform.csv",
        "low": output / f"{prefix}_low_rul_paired_policy_vs_uniform.csv",
        "summary": output / f"{prefix}_comparison_summary.csv",
        "ablation": output / f"{prefix}_source_ablation_summary.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def input_paths(root: Path, experiment_id: str) -> dict[str, Path]:
    return {
        "manifest": root / f"{experiment_id}_manifest.json",
        "decision": root / f"{experiment_id}_confirmation_decision.json",
        "protocol": root / f"{experiment_id}_protocol.json",
        "endpoint": root / f"{experiment_id}_pool_endpoint_predictions.csv",
    }


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "a10_output_dir": str(resolved(args.a10_output_dir, DEFAULT_A10_OUTPUT)),
        "a12_output_dir": str(resolved(args.a12_output_dir, DEFAULT_A12_OUTPUT)),
        "output_dir": str(resolved(args.output_dir, DEFAULT_OUTPUT)),
        "selection_gate_rule": (
            "weighted iff selection full RMSE relative delta < 0 and each "
            "true-stage NASA/RMSE relative delta <= 0.03; otherwise uniform"
        ),
        "stage_noninferiority_margin": MARGIN,
        "minimum_weighted_selection_rate": MIN_WEIGHTED_SELECTION_RATE,
        "minimum_weighted_conditions": MIN_WEIGHTED_CONDITIONS,
        "minimum_passing_ablation_conditions": MIN_ROBUST_CONDITIONS,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "quick_mode": bool(args.quick),
    }
    return experiment


def assert_training_only(manifest: dict[str, Any], decision: dict[str, Any], name: str) -> None:
    if not decision.get("complete") or decision.get("quick_mode"):
        raise RuntimeError(f"A13 requires the completed formal {name} output")
    for payload in (manifest, decision):
        if payload.get("official_test_files_accessed") or payload.get("official_test_forward_run"):
            raise RuntimeError(f"{name} is contaminated by official-test access")


def expected_counts(protocol: dict[str, Any], experiment: dict[str, Any]) -> dict[str, int]:
    domains = list(protocol)
    model_seeds = a10.MODEL_SEEDS
    splits = a10.TARGET_SPLIT_SEEDS
    roles = a10.ROLE_PARTITIONS
    confirmation = a10.CONFIRMATION_ENDPOINT_SEEDS
    selection = a10.SELECTION_ENDPOINT_SEEDS
    if experiment["quick_mode"]:
        domains, model_seeds, splits, roles = ["FD004"], (100,), (6401,), (1,)
        confirmation, selection = (9101,), (9001,)
    return {
        "policy_decisions": len(domains) * 3 * len(model_seeds) * len(splits) * len(roles),
        "confirmation_pairs": len(domains) * 3 * len(model_seeds) * len(splits) * len(roles) * len(confirmation),
        "selection_pairs": len(domains) * 3 * len(model_seeds) * len(splits) * len(roles) * len(selection),
    }


def filter_formal(frame: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    if not experiment["quick_mode"]:
        return frame.copy()
    return frame[
        (frame.target_domain == "FD004")
        & (frame.model_seed == 100)
        & (frame.target_split_seed == 6401)
    ].copy()


def validate_inputs(experiment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    a10_paths = input_paths(Path(experiment["a10_output_dir"]), "experimentA10")
    a12_paths = input_paths(Path(experiment["a12_output_dir"]), "experimentA12")
    missing = [str(path) for path in (*a10_paths.values(), *a12_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("A13 requires completed A10 and A12 files:\n" + "\n".join(missing))
    a10_manifest, a10_decision = read_json(a10_paths["manifest"]), read_json(a10_paths["decision"])
    a12_manifest, a12_decision = read_json(a12_paths["manifest"]), read_json(a12_paths["decision"])
    assert_training_only(a10_manifest, a10_decision, "A10")
    assert_training_only(a12_manifest, a12_decision, "A12")
    protocol = read_json(a10_paths["protocol"])
    a12_protocol = read_json(a12_paths["protocol"])
    if protocol != a12_protocol:
        raise RuntimeError("A10 and A12 protocols differ; A13 cannot pair them")
    expected = expected_counts(protocol, experiment)
    a10_endpoint = filter_formal(read_csv(a10_paths["endpoint"]), experiment)
    a12_endpoint = filter_formal(read_csv(a12_paths["endpoint"]), experiment)
    # role_partition is created later by A10 crossfit; raw pool endpoints are
    # intentionally role-agnostic at this point.
    required = set([
        "target_domain", "heldout_source_domain", "model_seed",
        "target_split_seed", "representation", *PRED_KEYS,
    ])
    for name, frame in (("A10 endpoint", a10_endpoint), ("A12 endpoint", a12_endpoint)):
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{name} lacks required columns: {sorted(required - set(frame.columns))}")
        if frame.official_test_files_accessed.astype(bool).any() or frame.official_test_forward_run.astype(bool).any():
            raise RuntimeError(f"official-test contamination found in {name}")
    integrity = {
        "a10_output_dir": str(Path(experiment["a10_output_dir"])),
        "a12_output_dir": str(Path(experiment["a12_output_dir"])),
        "a10_manifest_hash": a1.file_sha256(a10_paths["manifest"]),
        "a10_decision_hash": a1.file_sha256(a10_paths["decision"]),
        "a10_protocol_hash": a1.file_sha256(a10_paths["protocol"]),
        "a10_endpoint_hash": a1.file_sha256(a10_paths["endpoint"]),
        "a12_manifest_hash": a1.file_sha256(a12_paths["manifest"]),
        "a12_decision_hash": a1.file_sha256(a12_paths["decision"]),
        "a12_protocol_hash": a1.file_sha256(a12_paths["protocol"]),
        "a12_endpoint_hash": a1.file_sha256(a12_paths["endpoint"]),
        "a10_expected_training_cells": a10_decision.get("expected_training_cells"),
        "a10_completed_training_cells": a10_decision.get("completed_training_cells"),
        "a12_expected_training_cells": a12_decision.get("expected_training_cells"),
        "a12_completed_training_cells": a12_decision.get("completed_training_cells"),
        "a10_official_test_files_accessed": False,
        "a12_official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    payload = {
        "protocol": protocol,
        "a10_endpoint": a10_endpoint,
        "a12_endpoint": a12_endpoint,
        "expected": expected,
        "integrity": integrity,
    }
    return payload, a10_decision, a12_decision


def evaluation_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    config = {
        "domains": list(a10.DOMAINS),
        "model_seeds": list(a10.MODEL_SEEDS),
        "target_split_seeds": list(a10.TARGET_SPLIT_SEEDS),
        "role_partitions": list(a10.ROLE_PARTITIONS),
        "selection_endpoint_seeds": list(a10.SELECTION_ENDPOINT_SEEDS),
        "confirmation_endpoint_seeds": list(a10.CONFIRMATION_ENDPOINT_SEEDS),
        "high_rul_threshold": 60.0,
        "alpha_grid": list(a10.ALPHA_GRID),
        "prediction_gate_threshold": a10.GATE_THRESHOLD,
        "selection_safety_margin_pct": 3.0,
        "stage_noninferiority_margin_pct": 3.0,
        "bootstrap_repetitions": int(experiment["bootstrap_repetitions"]),
    }
    if experiment["quick_mode"]:
        config.update({
            "domains": ["FD004"], "model_seeds": [100], "target_split_seeds": [6401],
            "role_partitions": [1], "selection_endpoint_seeds": [9001],
            "confirmation_endpoint_seeds": [9101], "bootstrap_repetitions": min(100, int(experiment["bootstrap_repetitions"])),
        })
    return config


def merge_branch_predictions(uniform: pd.DataFrame, weighted: pd.DataFrame, role: str) -> pd.DataFrame:
    keys = POLICY_KEYS + ["endpoint_seed"] + PRED_KEYS
    needed = keys + ["prediction_blend"]
    left = uniform[needed].rename(columns={"prediction_blend": "prediction_uniform"})
    right = weighted[needed].rename(columns={"prediction_blend": "prediction_weighted"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"A13 {role} prediction alignment failed")
    return merged


def branch_risk(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return a9.risk(frame, column)


def relative_risk(frame: pd.DataFrame, stage: str) -> dict[str, float]:
    if stage == "high":
        selected = frame[frame.label.astype(float) > 60.0]
    elif stage == "low":
        selected = frame[frame.label.astype(float) <= 60.0]
    else:
        selected = frame
    if selected.empty:
        raise RuntimeError(f"A13 selection gate lacks {stage} records")
    ref, cand = branch_risk(selected, "prediction_uniform"), branch_risk(selected, "prediction_weighted")
    return {
        "nasa_relative_delta": a9.rel(cand, ref, "nasa_score"),
        "rmse_relative_delta": a9.rel(cand, ref, "rmse"),
        "nasa_score_delta": cand["nasa_score"] - ref["nasa_score"],
        "rmse_delta": cand["rmse"] - ref["rmse"],
    }


def gate_decisions(selection: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    rows = []
    margin = float(experiment["stage_noninferiority_margin"])
    for values, frame in selection.groupby(POLICY_KEYS, sort=True):
        full, high, low = relative_risk(frame, "full"), relative_risk(frame, "high"), relative_risk(frame, "low")
        rmse_improved = full["rmse_relative_delta"] < 0.0
        stage_safe = max(
            high["nasa_relative_delta"], high["rmse_relative_delta"],
            low["nasa_relative_delta"], low["rmse_relative_delta"],
        ) <= margin
        use_weighted = bool(rmse_improved and stage_safe)
        row = dict(zip(POLICY_KEYS, values))
        row.update({
            "selection_record_count": int(len(frame)),
            "selection_endpoint_seed_count": int(frame.endpoint_seed.nunique()),
            "selection_full_nasa_relative_delta": full["nasa_relative_delta"],
            "selection_full_rmse_relative_delta": full["rmse_relative_delta"],
            "selection_high_rul_nasa_relative_delta": high["nasa_relative_delta"],
            "selection_high_rul_rmse_relative_delta": high["rmse_relative_delta"],
            "selection_low_mid_rul_nasa_relative_delta": low["nasa_relative_delta"],
            "selection_low_mid_rul_rmse_relative_delta": low["rmse_relative_delta"],
            "selection_rmse_improved": bool(rmse_improved),
            "selection_stage_noninferior": bool(stage_safe),
            "use_coverage_weighted": use_weighted,
            "selected_policy": WEIGHTED if use_weighted else REFERENCE,
            "decision_reason": "selection_rmse_improved_and_stage_safe" if use_weighted else "uniform_fallback",
            "uses_selection_labels_only": True,
            "uses_confirmation_labels": False,
            "uses_official_test": False,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(POLICY_KEYS).reset_index(drop=True)


def apply_policy(confirmation: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    output = confirmation.merge(decisions[POLICY_KEYS + ["use_coverage_weighted", "selected_policy", "decision_reason"]], on=POLICY_KEYS, how="left", validate="many_to_one")
    if output.use_coverage_weighted.isna().any():
        raise RuntimeError("A13 confirmation has no corresponding selection decision")
    output["prediction_policy"] = np.where(output.use_coverage_weighted.astype(bool), output.prediction_weighted, output.prediction_uniform)
    return output


def policy_pairs(predictions: pd.DataFrame, expected: int) -> pd.DataFrame:
    rows = []
    for values, frame in predictions.groupby(PAIR_KEYS, sort=True):
        reference, candidate = a9.risk(frame, "prediction_uniform"), a9.risk(frame, "prediction_policy")
        row = dict(zip(PAIR_KEYS, values))
        row.update({
            "candidate": POLICY,
            "reference": REFERENCE,
            "use_coverage_weighted": bool(frame.use_coverage_weighted.iloc[0]),
            "selected_policy": str(frame.selected_policy.iloc[0]),
        })
        for metric in a10.a8.METRICS:
            row[f"{metric}_{REFERENCE}"] = reference[metric]
            row[f"{metric}_{POLICY}"] = candidate[metric]
            row[f"{metric}_delta_candidate_minus_baseline"] = candidate[metric] - reference[metric]
        row["nasa_relative_delta"] = a9.rel(candidate, reference, "nasa_score")
        row["rmse_relative_delta"] = a9.rel(candidate, reference, "rmse")
        row["candidate_nasa_win"] = candidate["nasa_score"] < reference["nasa_score"]
        row["candidate_rmse_win"] = candidate["rmse"] < reference["rmse"]
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS).reset_index(drop=True)
    if len(output) != expected:
        raise RuntimeError(f"A13 confirmation pairs incomplete: {len(output)} != {expected}")
    return output


def stage_pairs(predictions: pd.DataFrame, high: bool, expected: int) -> pd.DataFrame:
    selected = predictions[predictions.label.astype(float) > 60.0].copy() if high else predictions[predictions.label.astype(float) <= 60.0].copy()
    rows = []
    for values, frame in selected.groupby(PAIR_KEYS, sort=True):
        reference, candidate = a9.risk(frame, "prediction_uniform"), a9.risk(frame, "prediction_policy")
        row = dict(zip(PAIR_KEYS, values))
        row.update({
            "rul_stage": "high_rul_gt60" if high else "low_or_mid_rul_le60",
            "rul_threshold": 60.0,
            "stage_engine_count": int(frame.unit.nunique()),
            "candidate": POLICY,
            "reference": REFERENCE,
            "use_coverage_weighted": bool(frame.use_coverage_weighted.iloc[0]),
        })
        for metric in a10.a8.METRICS:
            row[f"{metric}_{REFERENCE}"] = reference[metric]
            row[f"{metric}_{POLICY}"] = candidate[metric]
            row[f"{metric}_delta_candidate_minus_baseline"] = candidate[metric] - reference[metric]
        row["nasa_relative_delta"] = a9.rel(candidate, reference, "nasa_score")
        row["rmse_relative_delta"] = a9.rel(candidate, reference, "rmse")
        row["candidate_nasa_win"] = candidate["nasa_score"] < reference["nasa_score"]
        row["candidate_rmse_win"] = candidate["rmse"] < reference["rmse"]
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS).reset_index(drop=True)
    if len(output) != expected:
        raise RuntimeError(f"A13 true-stage pairs incomplete: {len(output)} != {expected}")
    return output


def ablation_summary(full: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    rows = []
    repetitions = int(experiment["bootstrap_repetitions"])
    for (target, heldout), frame in full.groupby(["target_domain", "heldout_source_domain"], sort=True):
        h = high[(high.target_domain == target) & (high.heldout_source_domain == heldout)]
        l = low[(low.target_domain == target) & (low.heldout_source_domain == heldout)]
        fci = a10.bootstrap(frame, "rmse_relative_delta", repetitions, stable_seed(EXPERIMENT_ID, target, heldout, "full"))
        hn = a10.bootstrap(h, "nasa_relative_delta", repetitions, stable_seed(EXPERIMENT_ID, target, heldout, "high_nasa"))
        hr = a10.bootstrap(h, "rmse_relative_delta", repetitions, stable_seed(EXPERIMENT_ID, target, heldout, "high_rmse"))
        ln = a10.bootstrap(l, "nasa_relative_delta", repetitions, stable_seed(EXPERIMENT_ID, target, heldout, "low_nasa"))
        lr = a10.bootstrap(l, "rmse_relative_delta", repetitions, stable_seed(EXPERIMENT_ID, target, heldout, "low_rmse"))
        passed = fci[1] < 0.0 and max(hn[1], hr[1], ln[1], lr[1]) <= MARGIN
        rows.append({
            "target_domain": target,
            "heldout_source_domain": heldout,
            "n_pairs": int(len(frame)),
            "weighted_pair_rate": float(frame.use_coverage_weighted.astype(bool).mean()),
            "full_rmse_improvement_pct": float(-100.0 * frame.rmse_relative_delta.mean()),
            "full_rmse_ci95": json.dumps(fci),
            "high_nasa_ci95": json.dumps(hn),
            "high_rmse_ci95": json.dumps(hr),
            "low_nasa_ci95": json.dumps(ln),
            "low_rmse_ci95": json.dumps(lr),
            "robust_condition_passed": bool(passed),
        })
    return pd.DataFrame(rows).sort_values(["target_domain", "heldout_source_domain"]).reset_index(drop=True)


def validate_existing_manifest(paths: dict[str, Path], manifest: dict[str, Any], resume: bool) -> None:
    if not paths["manifest"].is_file():
        return
    previous = read_json(paths["manifest"])
    for key in ("experiment_config", "input_integrity", "registered_primary_question"):
        if previous.get(key) != manifest.get(key):
            raise RuntimeError(f"existing A13 output is incompatible at {key}; use a new output directory")
    if previous.get("script_hash") != manifest["script_hash"] and not resume:
        raise RuntimeError("A13 script changed; use --resume only after reviewing the change")
    if previous.get("script_hash") != manifest["script_hash"]:
        manifest["resumed_from_script_hash"] = previous.get("script_hash")


def main() -> None:
    args = parse_args()
    experiment = load_config(args)
    output = Path(experiment["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    inputs, a10_decision, a12_decision = validate_inputs(experiment)
    expected = inputs["expected"]
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "mode": "analysis_only_reuses_A10_A12_training_outputs",
        "new_predictor_training": False,
        "gpu_workers": 0,
        "expected_policy_decisions": expected["policy_decisions"],
        "expected_selection_pairs": expected["selection_pairs"],
        "expected_confirmation_pairs": expected["confirmation_pairs"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
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
    atomic_json(paths["input_integrity"], inputs["integrity"])
    atomic_json(paths["causality"], {
        "experiment_id": EXPERIMENT_ID,
        "policy_mode": "selection_only_gate_then_locked_confirmation",
        "reuses_pretrained_A10_A12_models": True,
        "selection_labels_used_to_choose_policy": True,
        "confirmation_labels_used_to_choose_policy": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "gate_rule": experiment["selection_gate_rule"],
    })
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return

    evaluation = evaluation_experiment(experiment)
    uniform_evaluated = a10.crossfit(inputs["a10_endpoint"], inputs["protocol"], evaluation)
    weighted_evaluated = a10.crossfit(inputs["a12_endpoint"], inputs["protocol"], evaluation)
    selection = merge_branch_predictions(uniform_evaluated["selection_prediction"], weighted_evaluated["selection_prediction"], "selection")
    decisions = gate_decisions(selection, experiment)
    if len(decisions) != expected["policy_decisions"]:
        raise RuntimeError(f"A13 gate decisions incomplete: {len(decisions)} != {expected['policy_decisions']}")
    confirmation = merge_branch_predictions(uniform_evaluated["confirmation_prediction"], weighted_evaluated["confirmation_prediction"], "confirmation")
    confirmation = apply_policy(confirmation, decisions)
    pairs = policy_pairs(confirmation, expected["confirmation_pairs"])
    high = stage_pairs(confirmation, True, expected["confirmation_pairs"])
    low = stage_pairs(confirmation, False, expected["confirmation_pairs"])
    comparison = pd.concat([
        a10.summary(pairs, evaluation, "full_endpoint_policy_vs_uniform"),
        a10.summary(high, evaluation, "high_rul_policy_vs_uniform"),
        a10.summary(low, evaluation, "low_rul_policy_vs_uniform"),
    ], ignore_index=True)
    ablation = ablation_summary(pairs, high, low, experiment)
    overall = comparison.query("comparison == 'full_endpoint_policy_vs_uniform' and scope == 'ALL'").iloc[0]
    high_all = a10.stage_summary(high, evaluation, "high_rul_gt60")
    low_all = a10.stage_summary(low, evaluation, "low_or_mid_rul_le60")
    full_ok = float(overall.rmse_relative_boot_ci95_high) < 0.0
    high_ok = high_all["nasa_relative_ci95"][1] <= MARGIN and high_all["rmse_relative_ci95"][1] <= MARGIN
    low_ok = low_all["nasa_relative_ci95"][1] <= MARGIN and low_all["rmse_relative_ci95"][1] <= MARGIN
    weighted_rate = float(decisions.use_coverage_weighted.astype(bool).mean())
    condition_usage = decisions.groupby(["target_domain", "heldout_source_domain"]).use_coverage_weighted.mean()
    weighted_conditions = int((condition_usage >= MIN_WEIGHTED_SELECTION_RATE).sum())
    robust_conditions = int(ablation.robust_condition_passed.astype(bool).sum())
    complete = len(pairs) == expected["confirmation_pairs"] and len(decisions) == expected["policy_decisions"]
    passed = bool(
        complete
        and full_ok
        and high_ok
        and low_ok
        and weighted_rate >= MIN_WEIGHTED_SELECTION_RATE
        and weighted_conditions >= MIN_WEIGHTED_CONDITIONS
        and robust_conditions >= MIN_ROBUST_CONDITIONS
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "new_predictor_training": False,
        "reference": REFERENCE,
        "candidate": POLICY,
        "weighted_branch": WEIGHTED,
        "expected_policy_decisions": expected["policy_decisions"],
        "completed_policy_decisions": int(len(decisions)),
        "expected_confirmation_pairs": expected["confirmation_pairs"],
        "completed_confirmation_pairs": int(len(pairs)),
        "weighted_selection_rate": weighted_rate,
        "minimum_weighted_selection_rate": MIN_WEIGHTED_SELECTION_RATE,
        "weighted_source_ablation_conditions": weighted_conditions,
        "minimum_weighted_source_ablation_conditions": MIN_WEIGHTED_CONDITIONS,
        "source_ablation_conditions": 12 if not args.quick else 3,
        "passing_ablation_conditions": robust_conditions,
        "minimum_passing_ablation_conditions": MIN_ROBUST_CONDITIONS,
        "full_endpoint_result": {
            "nasa_improvement_pct": float(overall.nasa_improvement_pct),
            "nasa_relative_ci95": [float(overall.nasa_relative_boot_ci95_low), float(overall.nasa_relative_boot_ci95_high)],
            "rmse_improvement_pct": float(-overall.rmse_degradation_pct),
            "rmse_relative_ci95": [float(overall.rmse_relative_boot_ci95_low), float(overall.rmse_relative_boot_ci95_high)],
            "strict_rmse_improvement": bool(full_ok),
        },
        "high_rul_safety_result": {**high_all, "noninferiority_passed": bool(high_ok)},
        "low_rul_safety_result": {**low_all, "noninferiority_passed": bool(low_ok)},
        "passed": passed if not args.quick else complete,
        "reason": (
            "A13 confirmed a non-trivial, selection-gated source policy under the registered source-ablation protocol"
            if passed else
            "A13 completed, but the selection-gated source policy did not meet every registered efficacy, safety, usage, or robustness criterion"
        ),
        "next_action": "report_A13_robustness_extension" if passed else "stop_source_coverage_remediation_direction_and_report_A9_scope_boundary",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for name, frame in (
        ("gate", decisions),
        ("selection_pairs", selection),
        ("confirmation", confirmation),
        ("paired", pairs),
        ("high", high),
        ("low", low),
        ("summary", comparison),
        ("ablation", ablation),
    ):
        a1.atomic_write_text(paths[name], frame.to_csv(index=False))
    atomic_json(paths["decision"], decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
