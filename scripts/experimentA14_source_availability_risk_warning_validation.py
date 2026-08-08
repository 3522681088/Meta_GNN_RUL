"""Experiment A14: source-availability risk-warning validation.

This is deliberately NOT another prediction-repair experiment.  A11 linked
source coverage and baseline/cycle-age disagreement to source-ablation
failures, while A12/A13 showed that coverage-aware weighting (even when
selection-gated) was not a safe universal replacement for the locked A9
policy.  A14 therefore asks a narrower deployment question:

Can target-adaptation coverage information plus disagreement between the two
already-deployed A9 component predictors identify endpoint records whose A10
uniform-source A9 blend has elevated NASA risk / absolute error?

The script reuses completed A10 endpoint predictions and A12 coverage-distance
metadata.  It does not train a model, change a prediction, access official
C-MAPSS test files, or use confirmation labels to choose a warning threshold.

For every target-domain / held-out-source condition, a fixed equal-weight
score is constructed from:
  (1) coverage_risk_index = mean source coverage distance
                             + 0.5 * source-distance spread;
  (2) |prediction_baseline - prediction_cycle_age|.
Selection endpoint records set robust score scaling, the NASA-risk event
cut-off, and one warning threshold from a pre-registered four-quantile grid.
Confirmation endpoint records evaluate the locked warning rule.
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


SCRIPT_VERSION = "experimentA14_source_availability_risk_warning_validation_v1"
EXPERIMENT_ID = "experimentA14"
DEFAULT_OUTPUT = "outputs/experimentA14_source_availability_risk_warning_validation"
DEFAULT_A10_OUTPUT = a10.DEFAULT_OUTPUT
DEFAULT_A12_OUTPUT = "outputs/experimentA12_coverage_aware_source_weighting_training_only"
DEFAULT_A13_OUTPUT = "outputs/experimentA13_selection_gated_source_policy_confirmation"

QUESTION = (
    "Can a selection-locked source-availability warning score, constructed "
    "only from target-adaptation coverage risk and baseline/cycle-age "
    "prediction disagreement, identify elevated A10 uniform-source A9 blend "
    "endpoint NASA risk and absolute error under two-source ablation?"
)

CONDITION_KEYS = ["target_domain", "heldout_source_domain"]
PAIR_KEYS = a10.PAIR_KEYS
SOURCE_KEYS = ["target_domain", "heldout_source_domain", "model_seed", "target_split_seed"]
RISK_SCORE_WEIGHTS = {"coverage_risk_index_z": 1.0, "prediction_disagreement_z": 1.0}
WARNING_SCORE_QUANTILES = (0.50, 0.60, 0.70, 0.80)
SELECTION_RISK_EVENT_QUANTILE = 0.75
MIN_WARNING_RATE = 0.10
MAX_WARNING_RATE = 0.50
MIN_CONDITION_PASSES = 9
MIN_TARGET_DOMAIN_PASSES = 3
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A14 source-availability risk warning")
    parser.add_argument("--output-dir")
    parser.add_argument("--a10-output-dir")
    parser.add_argument("--a12-output-dir")
    parser.add_argument("--a13-output-dir")
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty input is missing: {path}")
    return pd.read_csv(path)


def stable_seed(*parts: Any) -> int:
    text = ":".join(map(str, parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


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


def input_paths(directory: Path, experiment_id: str) -> dict[str, Path]:
    paths = {
        "manifest": directory / f"{experiment_id}_manifest.json",
        "decision": directory / f"{experiment_id}_confirmation_decision.json",
    }
    if experiment_id in {"experimentA10", "experimentA12"}:
        paths.update({
            "protocol": directory / f"{experiment_id}_protocol.json",
            "endpoint": directory / f"{experiment_id}_pool_endpoint_predictions.csv",
        })
    if experiment_id == "experimentA12":
        paths["source_parameters"] = directory / f"{experiment_id}_source_weight_parameters.csv"
    return paths


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "output_dir": str(resolved(args.output_dir, DEFAULT_OUTPUT)),
        "a10_output_dir": str(resolved(args.a10_output_dir, DEFAULT_A10_OUTPUT)),
        "a12_output_dir": str(resolved(args.a12_output_dir, DEFAULT_A12_OUTPUT)),
        "a13_output_dir": str(resolved(args.a13_output_dir, DEFAULT_A13_OUTPUT)),
        "reference_prediction": "A10_uniform_source_A9_blend",
        "coverage_feature": "mean_distance + 0.5 * distance_spread",
        "disagreement_feature": "abs(prediction_baseline - prediction_cycle_age)",
        "fixed_score_weights": RISK_SCORE_WEIGHTS,
        "warning_score_quantile_grid": list(WARNING_SCORE_QUANTILES),
        "selection_risk_event_quantile": SELECTION_RISK_EVENT_QUANTILE,
        "minimum_warning_rate": MIN_WARNING_RATE,
        "maximum_warning_rate": MAX_WARNING_RATE,
        "minimum_passing_source_ablation_conditions": MIN_CONDITION_PASSES,
        "minimum_passing_target_domains": MIN_TARGET_DOMAIN_PASSES,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "quick_mode": bool(args.quick),
    }


def assert_training_only(manifest: dict[str, Any], decision: dict[str, Any], name: str) -> None:
    if not decision.get("complete"):
        raise RuntimeError(f"A14 requires completed {name} outputs")
    if decision.get("quick_mode"):
        raise RuntimeError(f"A14 cannot use quick-mode {name} outputs")
    for payload in (manifest, decision):
        if payload.get("official_test_files_accessed") or payload.get("official_test_forward_run"):
            raise RuntimeError(f"{name} is contaminated by official-test access")


def filter_formal(frame: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    if not experiment["quick_mode"]:
        return frame.copy()
    return frame[
        (frame["target_domain"] == "FD004")
        & (frame["model_seed"] == 100)
        & (frame["target_split_seed"] == 6401)
    ].copy()


def expected_counts(experiment: dict[str, Any]) -> dict[str, int]:
    domains = list(a10.DOMAINS)
    model_seeds = list(a10.MODEL_SEEDS)
    splits = list(a10.TARGET_SPLIT_SEEDS)
    roles = list(a10.ROLE_PARTITIONS)
    endpoints = list(a10.CONFIRMATION_ENDPOINT_SEEDS)
    if experiment["quick_mode"]:
        domains, model_seeds, splits, roles, endpoints = ["FD004"], [100], [6401], [1], [9101]
    condition_count = len(domains) * 3
    return {
        "source_conditions": condition_count,
        "selection_parameter_sets": condition_count,
        "confirmation_pairs": condition_count * len(model_seeds) * len(splits) * len(roles) * len(endpoints),
    }


def validate_inputs(experiment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    a10_paths = input_paths(Path(experiment["a10_output_dir"]), "experimentA10")
    a12_paths = input_paths(Path(experiment["a12_output_dir"]), "experimentA12")
    a13_paths = input_paths(Path(experiment["a13_output_dir"]), "experimentA13")
    all_paths = [*a10_paths.values(), *a12_paths.values(), *a13_paths.values()]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("A14 requires completed A10/A12/A13 files:\n" + "\n".join(missing))

    a10_manifest, a10_decision = read_json(a10_paths["manifest"]), read_json(a10_paths["decision"])
    a12_manifest, a12_decision = read_json(a12_paths["manifest"]), read_json(a12_paths["decision"])
    a13_manifest, a13_decision = read_json(a13_paths["manifest"]), read_json(a13_paths["decision"])
    assert_training_only(a10_manifest, a10_decision, "A10")
    assert_training_only(a12_manifest, a12_decision, "A12")
    assert_training_only(a13_manifest, a13_decision, "A13")
    protocol = read_json(a10_paths["protocol"])
    if protocol != read_json(a12_paths["protocol"]):
        raise RuntimeError("A10 and A12 protocols differ; coverage metadata cannot be paired safely")

    endpoints = filter_formal(read_csv(a10_paths["endpoint"]), experiment)
    parameters = filter_formal(read_csv(a12_paths["source_parameters"]), experiment)
    required_endpoint = {
        "target_domain", "heldout_source_domain", "model_seed", "target_split_seed",
        "representation", *a9.PRED_KEYS, "official_test_files_accessed", "official_test_forward_run",
    }
    required_parameters = {
        *SOURCE_KEYS, "source_domain", "coverage_distance", "source_weight",
        "uses_target_labels", "uses_selection_units", "uses_confirmation_units", "uses_official_test",
    }
    if not required_endpoint.issubset(endpoints.columns):
        raise RuntimeError(f"A10 endpoint lacks columns: {sorted(required_endpoint - set(endpoints.columns))}")
    if not required_parameters.issubset(parameters.columns):
        raise RuntimeError(f"A12 source parameters lack columns: {sorted(required_parameters - set(parameters.columns))}")
    if endpoints["official_test_files_accessed"].astype(bool).any() or endpoints["official_test_forward_run"].astype(bool).any():
        raise RuntimeError("official-test contamination in A10 endpoint records")
    for column in ("uses_target_labels", "uses_selection_units", "uses_confirmation_units", "uses_official_test"):
        if parameters[column].astype(bool).any():
            raise RuntimeError(f"A12 source-coverage metadata uses forbidden information: {column}")

    integrity = {
        "a10_output_dir": str(Path(experiment["a10_output_dir"])),
        "a12_output_dir": str(Path(experiment["a12_output_dir"])),
        "a13_output_dir": str(Path(experiment["a13_output_dir"])),
        "a10_manifest_hash": a1.file_sha256(a10_paths["manifest"]),
        "a10_decision_hash": a1.file_sha256(a10_paths["decision"]),
        "a10_protocol_hash": a1.file_sha256(a10_paths["protocol"]),
        "a10_endpoint_hash": a1.file_sha256(a10_paths["endpoint"]),
        "a12_manifest_hash": a1.file_sha256(a12_paths["manifest"]),
        "a12_decision_hash": a1.file_sha256(a12_paths["decision"]),
        "a12_protocol_hash": a1.file_sha256(a12_paths["protocol"]),
        "a12_source_parameters_hash": a1.file_sha256(a12_paths["source_parameters"]),
        "a13_manifest_hash": a1.file_sha256(a13_paths["manifest"]),
        "a13_decision_hash": a1.file_sha256(a13_paths["decision"]),
        "a10_expected_training_cells": a10_decision.get("expected_training_cells"),
        "a10_completed_training_cells": a10_decision.get("completed_training_cells"),
        "a12_expected_training_cells": a12_decision.get("expected_training_cells"),
        "a12_completed_training_cells": a12_decision.get("completed_training_cells"),
        "a13_expected_confirmation_pairs": a13_decision.get("expected_confirmation_pairs"),
        "a13_completed_confirmation_pairs": a13_decision.get("completed_confirmation_pairs"),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return {
        "protocol": protocol,
        "a10_endpoints": endpoints,
        "a12_parameters": parameters,
        "integrity": integrity,
    }, {"a10": a10_decision, "a12": a12_decision, "a13": a13_decision}


def evaluation_config(experiment: dict[str, Any]) -> dict[str, Any]:
    cfg = {
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
    }
    if experiment["quick_mode"]:
        cfg.update({
            "domains": ["FD004"], "model_seeds": [100], "target_split_seeds": [6401],
            "role_partitions": [1], "selection_endpoint_seeds": [9001],
            "confirmation_endpoint_seeds": [9101],
        })
    return cfg


def coverage_features(parameters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, frame in parameters.groupby(SOURCE_KEYS, sort=True):
        if len(frame) != 2 or frame["source_domain"].nunique() != 2:
            raise RuntimeError(f"A14 expects exactly two active A12 source rows for {values}")
        distances = frame["coverage_distance"].astype(float).to_numpy()
        weights = frame["source_weight"].astype(float).to_numpy()
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"A12 source weights do not sum to one for {values}")
        row = dict(zip(SOURCE_KEYS, values))
        row.update({
            "active_source_count": 2,
            "coverage_distance_mean": float(distances.mean()),
            "coverage_distance_max": float(distances.max()),
            "coverage_distance_min": float(distances.min()),
            "coverage_distance_spread": float(distances.max() - distances.min()),
            "coverage_risk_index": float(distances.mean() + 0.5 * (distances.max() - distances.min())),
            "source_weight_imbalance": float(abs(weights[0] - weights[1])),
            "source_weight_min": float(weights.min()),
            "source_weight_max": float(weights.max()),
            "coverage_uses_target_adaptation_inputs": True,
            "coverage_uses_target_labels": False,
            "coverage_uses_selection_units": False,
            "coverage_uses_confirmation_units": False,
            "coverage_uses_official_test": False,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(SOURCE_KEYS).reset_index(drop=True)


def attach_coverage(predictions: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    output = predictions.merge(coverage, on=SOURCE_KEYS, how="left", validate="many_to_one")
    if output["coverage_risk_index"].isna().any():
        raise RuntimeError("A14 could not match A12 coverage metadata to A10 prediction records")
    output["prediction_disagreement_abs"] = (
        output["prediction_baseline"].astype(float) - output["prediction_cycle_age"].astype(float)
    ).abs()
    output["blend_error"] = output["prediction_blend"].astype(float) - output["label"].astype(float)
    output["absolute_error"] = output["blend_error"].abs()
    error = output["blend_error"].to_numpy(dtype=float)
    output["nasa_contribution"] = np.where(error < 0.0, np.exp(np.clip(-error / 13.0, -50.0, 50.0)) - 1.0, np.exp(np.clip(error / 10.0, -50.0, 50.0)) - 1.0)
    return output


def robust_location_scale(values: pd.Series) -> tuple[float, float]:
    median = float(values.median())
    q25, q75 = np.quantile(values.astype(float).to_numpy(), [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(values.astype(float).std(ddof=0))
    return median, (scale if np.isfinite(scale) and scale > EPS else 1.0)


def with_score(frame: pd.DataFrame, params: pd.Series | dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    coverage_median, coverage_scale = float(params["coverage_median"]), float(params["coverage_scale"])
    disagreement_median, disagreement_scale = float(params["disagreement_median"]), float(params["disagreement_scale"])
    out["coverage_risk_index_z"] = np.clip((out["coverage_risk_index"].astype(float) - coverage_median) / coverage_scale, -10.0, 10.0)
    out["prediction_disagreement_z"] = np.clip((out["prediction_disagreement_abs"].astype(float) - disagreement_median) / disagreement_scale, -10.0, 10.0)
    out["risk_score"] = (
        RISK_SCORE_WEIGHTS["coverage_risk_index_z"] * out["coverage_risk_index_z"]
        + RISK_SCORE_WEIGHTS["prediction_disagreement_z"] * out["prediction_disagreement_z"]
    )
    return out


def classification_metrics(frame: pd.DataFrame, warning: pd.Series, event: pd.Series) -> dict[str, float]:
    warning = warning.astype(bool).to_numpy()
    event = event.astype(bool).to_numpy()
    tp = int(np.logical_and(warning, event).sum())
    fp = int(np.logical_and(warning, ~event).sum())
    tn = int(np.logical_and(~warning, ~event).sum())
    fn = int(np.logical_and(~warning, event).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "warning_rate": float(warning.mean()),
        "event_rate": float(event.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "youden_j": float(sensitivity + specificity - 1.0),
        "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else 0.0,
    }


def choose_warning_parameters(selection: pd.DataFrame, experiment: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_rows, grid_rows, scored_parts = [], [], []
    for values, frame in selection.groupby(CONDITION_KEYS, sort=True):
        target, heldout = values
        coverage_median, coverage_scale = robust_location_scale(frame["coverage_risk_index"])
        disagreement_median, disagreement_scale = robust_location_scale(frame["prediction_disagreement_abs"])
        base = {
            "target_domain": target,
            "heldout_source_domain": heldout,
            "coverage_median": coverage_median,
            "coverage_scale": coverage_scale,
            "disagreement_median": disagreement_median,
            "disagreement_scale": disagreement_scale,
        }
        scored = with_score(frame, base)
        event_cutoff = float(np.quantile(scored["nasa_contribution"].to_numpy(dtype=float), SELECTION_RISK_EVENT_QUANTILE))
        event = scored["nasa_contribution"].astype(float) >= event_cutoff
        rows = []
        for quantile in WARNING_SCORE_QUANTILES:
            threshold = float(np.quantile(scored["risk_score"].to_numpy(dtype=float), quantile))
            warning = scored["risk_score"].astype(float) >= threshold
            metrics = classification_metrics(scored, warning, event)
            feasible_rate = MIN_WARNING_RATE <= metrics["warning_rate"] <= MAX_WARNING_RATE
            rows.append({
                **base,
                "score_quantile": float(quantile),
                "score_threshold": threshold,
                "selection_nasa_risk_cutoff": event_cutoff,
                "selection_record_count": int(len(scored)),
                "selection_uses_labels_only": True,
                "selection_score_uses_labels": False,
                "selection_threshold_feasible_rate": bool(feasible_rate),
                **metrics,
            })
        grid = pd.DataFrame(rows)
        feasible = grid[grid["selection_threshold_feasible_rate"]].copy()
        if feasible.empty:
            raise RuntimeError(f"A14 no warning-rate-feasible threshold for {target}/{heldout}")
        chosen = feasible.sort_values(
            ["youden_j", "balanced_accuracy", "sensitivity", "score_quantile"],
            ascending=[False, False, False, False], kind="mergesort",
        ).iloc[0].to_dict()
        chosen.update({
            "selected_by": "max_selection_youden_j_then_balanced_accuracy_then_sensitivity_then_higher_quantile",
            "warning_rule": "risk_score >= locked_score_threshold",
            "risk_event_definition": "A10 uniform-source A9 blend NASA contribution >= selection 75th percentile",
            "uses_confirmation_labels": False,
            "uses_official_test": False,
        })
        parameter_rows.append(chosen)
        grid_rows.append(grid.assign(selected=grid["score_quantile"] == float(chosen["score_quantile"])))
        scored["selection_nasa_risk_cutoff"] = event_cutoff
        scored["risk_event"] = event.astype(bool)
        scored["warning"] = scored["risk_score"] >= float(chosen["score_threshold"])
        scored["locked_score_threshold"] = float(chosen["score_threshold"])
        scored["selected_score_quantile"] = float(chosen["score_quantile"])
        scored_parts.append(scored)
    return (
        pd.DataFrame(parameter_rows).sort_values(CONDITION_KEYS).reset_index(drop=True),
        pd.concat(grid_rows, ignore_index=True).sort_values(CONDITION_KEYS + ["score_quantile"]).reset_index(drop=True),
        pd.concat(scored_parts, ignore_index=True),
    )


def apply_locked_warning(confirmation: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    columns = CONDITION_KEYS + [
        "coverage_median", "coverage_scale", "disagreement_median", "disagreement_scale",
        "score_threshold", "selection_nasa_risk_cutoff", "score_quantile",
    ]
    output = confirmation.merge(parameters[columns], on=CONDITION_KEYS, how="left", validate="many_to_one")
    if output["score_threshold"].isna().any():
        raise RuntimeError("A14 confirmation records lack locked selection warning parameters")
    parts = []
    for _, frame in output.groupby(CONDITION_KEYS, sort=False):
        parts.append(with_score(frame, frame.iloc[0]))
    output = pd.concat(parts, ignore_index=True)
    output["warning"] = output["risk_score"].astype(float) >= output["score_threshold"].astype(float)
    # This variable is an evaluation label only.  It is generated after the
    # warning score/threshold have been locked and is never fed back to them.
    output["risk_event"] = output["nasa_contribution"].astype(float) >= output["selection_nasa_risk_cutoff"].astype(float)
    output["uses_confirmation_labels_for_warning"] = False
    output["uses_official_test"] = False
    return output


def pair_warning_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, part in frame.groupby(PAIR_KEYS, sort=True):
        flagged = part[part["warning"].astype(bool)]
        clear = part[~part["warning"].astype(bool)]
        row = dict(zip(PAIR_KEYS, values))
        row.update({
            "warning_record_count": int(len(flagged)),
            "clear_record_count": int(len(clear)),
            "warning_rate": float(part["warning"].astype(bool).mean()),
            "nasa_risk_event_rate": float(part["risk_event"].astype(bool).mean()),
        })
        for prefix, metric in (("nasa", "nasa_contribution"), ("absolute_error", "absolute_error")):
            row[f"warning_{prefix}_mean"] = float(flagged[metric].mean()) if not flagged.empty else np.nan
            row[f"clear_{prefix}_mean"] = float(clear[metric].mean()) if not clear.empty else np.nan
            if not flagged.empty and not clear.empty:
                row[f"{prefix}_warning_to_clear_ratio"] = float(flagged[metric].mean() / max(clear[metric].mean(), EPS))
            else:
                row[f"{prefix}_warning_to_clear_ratio"] = np.nan
        row["score_nasa_spearman"] = float(part["risk_score"].rank().corr(part["nasa_contribution"].rank(), method="pearson"))
        row["score_absolute_error_spearman"] = float(part["risk_score"].rank().corr(part["absolute_error"].rank(), method="pearson"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(PAIR_KEYS).reset_index(drop=True)


def paired_ratio_ci(pairs: pd.DataFrame, numerator: str, denominator: str, repetitions: int, seed: int) -> tuple[float, float]:
    values = pairs[[numerator, denominator]].dropna().astype(float).to_numpy()
    values = values[(values[:, 0] >= 0.0) & (values[:, 1] >= 0.0)]
    if len(values) < 20:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    batch = min(250, repetitions)
    for start in range(0, repetitions, batch):
        count = min(batch, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        chosen = values[indices]
        numerator_mean = chosen[:, :, 0].mean(axis=1)
        denominator_mean = chosen[:, :, 1].mean(axis=1)
        samples[start:start + count] = numerator_mean / np.maximum(denominator_mean, EPS)
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def warning_summary(frame: pd.DataFrame, pairs: pd.DataFrame, target: str, heldout: str, repetitions: int, suffix: str) -> dict[str, Any]:
    warning = frame["warning"].astype(bool)
    event = frame["risk_event"].astype(bool)
    cls = classification_metrics(frame, warning, event)
    flagged, clear = frame[warning], frame[~warning]
    nasa_ci = paired_ratio_ci(
        pairs, "warning_nasa_mean", "clear_nasa_mean", repetitions,
        stable_seed(EXPERIMENT_ID, target, heldout, suffix, "nasa"),
    )
    absolute_ci = paired_ratio_ci(
        pairs, "warning_absolute_error_mean", "clear_absolute_error_mean", repetitions,
        stable_seed(EXPERIMENT_ID, target, heldout, suffix, "absolute"),
    )
    nasa_ratio = float(flagged["nasa_contribution"].mean() / max(clear["nasa_contribution"].mean(), EPS)) if not flagged.empty and not clear.empty else float("nan")
    absolute_ratio = float(flagged["absolute_error"].mean() / max(clear["absolute_error"].mean(), EPS)) if not flagged.empty and not clear.empty else float("nan")
    return {
        "target_domain": target,
        "heldout_source_domain": heldout,
        "scope": suffix,
        "n_records": int(len(frame)),
        "warning_n_records": int(len(flagged)),
        "clear_n_records": int(len(clear)),
        "warning_nasa_mean": float(flagged["nasa_contribution"].mean()) if not flagged.empty else float("nan"),
        "clear_nasa_mean": float(clear["nasa_contribution"].mean()) if not clear.empty else float("nan"),
        "nasa_warning_to_clear_ratio": nasa_ratio,
        "nasa_ratio_ci95_low": nasa_ci[0],
        "nasa_ratio_ci95_high": nasa_ci[1],
        "warning_absolute_error_mean": float(flagged["absolute_error"].mean()) if not flagged.empty else float("nan"),
        "clear_absolute_error_mean": float(clear["absolute_error"].mean()) if not clear.empty else float("nan"),
        "absolute_error_warning_to_clear_ratio": absolute_ratio,
        "absolute_error_ratio_ci95_low": absolute_ci[0],
        "absolute_error_ratio_ci95_high": absolute_ci[1],
        "score_nasa_spearman": float(frame["risk_score"].rank().corr(frame["nasa_contribution"].rank(), method="pearson")),
        "score_absolute_error_spearman": float(frame["risk_score"].rank().corr(frame["absolute_error"].rank(), method="pearson")),
        **cls,
    }


def confirmation_summaries(confirmation: pd.DataFrame, pairs: pd.DataFrame, experiment: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_rows, stage_rows = [], []
    reps = int(experiment["bootstrap_repetitions"])
    for (target, heldout), frame in confirmation.groupby(CONDITION_KEYS, sort=True):
        pair_frame = pairs[(pairs["target_domain"] == target) & (pairs["heldout_source_domain"] == heldout)]
        main = warning_summary(frame, pair_frame, target, heldout, reps, "all")
        main["condition_warning_supported"] = bool(
            np.isfinite(main["nasa_ratio_ci95_low"])
            and main["nasa_ratio_ci95_low"] > 1.0
            and main["absolute_error_ratio_ci95_low"] > 1.0
            and main["balanced_accuracy"] > 0.5
        )
        condition_rows.append(main)
        for stage_name, stage in (
            ("high_rul_gt60", frame[frame["label"].astype(float) > 60.0]),
            ("low_or_mid_rul_le60", frame[frame["label"].astype(float) <= 60.0]),
        ):
            stage_keys = stage[PAIR_KEYS].drop_duplicates()
            stage_pairs = pair_frame.merge(stage_keys, on=PAIR_KEYS, how="inner")
            stage_row = warning_summary(stage, stage_pairs, target, heldout, reps, stage_name)
            stage_row["true_rul_threshold"] = 60.0
            stage_rows.append(stage_row)
    conditions = pd.DataFrame(condition_rows).sort_values(CONDITION_KEYS).reset_index(drop=True)
    stages = pd.DataFrame(stage_rows).sort_values(CONDITION_KEYS + ["scope"]).reset_index(drop=True)
    domains = (
        conditions.groupby("target_domain", as_index=False)
        .agg(
            source_ablation_conditions=("heldout_source_domain", "count"),
            passing_conditions=("condition_warning_supported", "sum"),
            mean_warning_rate=("warning_rate", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_nasa_warning_to_clear_ratio=("nasa_warning_to_clear_ratio", "mean"),
            mean_absolute_error_warning_to_clear_ratio=("absolute_error_warning_to_clear_ratio", "mean"),
            mean_score_nasa_spearman=("score_nasa_spearman", "mean"),
        )
    )
    domains["target_domain_warning_supported"] = domains["passing_conditions"].astype(int) >= 2
    return conditions, stages, domains.sort_values("target_domain").reset_index(drop=True)


def validate_existing_manifest(paths: dict[str, Path], manifest: dict[str, Any], resume: bool) -> None:
    if not paths["manifest"].is_file():
        return
    previous = read_json(paths["manifest"])
    for key in ("experiment_config", "input_integrity", "registered_primary_question"):
        if previous.get(key) != manifest.get(key):
            raise RuntimeError(f"existing A14 output is incompatible at {key}; use a new output directory")
    if previous.get("script_hash") != manifest["script_hash"] and not resume:
        raise RuntimeError("A14 script changed; use --resume only after reviewing the change")
    if previous.get("script_hash") != manifest["script_hash"]:
        manifest["resumed_from_script_hash"] = previous.get("script_hash")


def main() -> None:
    args = parse_args()
    experiment = load_config(args)
    output = Path(experiment["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    inputs, prior_decisions = validate_inputs(experiment)
    expected = expected_counts(experiment)
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "mode": "analysis_only_reuses_A10_predictions_and_A12_coverage_metadata",
        "new_predictor_training": False,
        "gpu_workers": 0,
        "expected_source_ablation_conditions": expected["source_conditions"],
        "expected_selection_parameter_sets": expected["selection_parameter_sets"],
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
    atomic_json(paths["integrity"], inputs["integrity"])
    atomic_json(paths["causality"], {
        "experiment_id": EXPERIMENT_ID,
        "analysis_mode": "selection_locked_risk_warning_not_prediction_repair",
        "reference_prediction": "A10 uniform-source A9 blend",
        "coverage_feature_source": "A12 source_weight_parameters fitted only from target adaptation-engine inputs",
        "prediction_disagreement_source": "A10 baseline and causal-cycle-age component predictions",
        "risk_score_uses_target_labels": False,
        "risk_score_uses_selection_labels": False,
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

    print("[A14] reconstructing A10 selection/confirmation endpoint predictions...", flush=True)
    evaluated = a10.crossfit(inputs["a10_endpoints"], inputs["protocol"], evaluation_config(experiment))
    coverage = coverage_features(inputs["a12_parameters"])
    selection = attach_coverage(evaluated["selection_prediction"], coverage)
    confirmation = attach_coverage(evaluated["confirmation_prediction"], coverage)
    actual_pairs = int(confirmation.groupby(PAIR_KEYS).ngroups)
    if actual_pairs != expected["confirmation_pairs"]:
        raise RuntimeError(f"A14 confirmation pair count is incomplete: {actual_pairs} != {expected['confirmation_pairs']}")
    if coverage.groupby(CONDITION_KEYS).ngroups != expected["source_conditions"]:
        raise RuntimeError("A14 source coverage conditions are incomplete")
    print("[A14] locking source-availability warning thresholds on selection endpoints...", flush=True)
    parameters, selection_grid, selection = choose_warning_parameters(selection, experiment)
    if len(parameters) != expected["selection_parameter_sets"]:
        raise RuntimeError("A14 warning parameter sets are incomplete")
    confirmation = apply_locked_warning(confirmation, parameters)
    pairs = pair_warning_metrics(confirmation)
    print(f"[A14] evaluating {len(confirmation)} confirmation records across {actual_pairs} paired cells...", flush=True)
    conditions, stages, domains = confirmation_summaries(confirmation, pairs, experiment)
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
        "reference_prediction": "A10 uniform-source A9 blend",
        "expected_source_ablation_conditions": expected["source_conditions"],
        "completed_source_ablation_conditions": int(len(conditions)),
        "expected_selection_parameter_sets": expected["selection_parameter_sets"],
        "completed_selection_parameter_sets": int(len(parameters)),
        "expected_confirmation_pairs": expected["confirmation_pairs"],
        "completed_confirmation_pairs": actual_pairs,
        "selection_confirmation_endpoint_seeds_disjoint": True,
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
            "A14 confirmed that the selection-locked source-availability warning consistently stratifies endpoint risk"
            if passed else
            "A14 completed, but the selection-locked warning did not meet the registered cross-condition risk-stratification criteria"
        ),
        "next_action": (
            "report_source_availability_warning_scope" if passed else
            "end_source_availability_extension_and_report_A9_scope_boundary"
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
