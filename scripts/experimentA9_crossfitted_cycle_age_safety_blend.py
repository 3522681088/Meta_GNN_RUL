"""Experiment A9: cross-fitted bounded blend of baseline and causal cycle-age.

The A8/A8_1 representation is useful but has high-RUL positive-bias risk.  A9
keeps their training protocol unchanged and selects, on selection engines only,
a bounded prediction blend:
    baseline + alpha * (cycle_age - baseline)
with independent high/lower predicted-RUL alphas from a fixed grid.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402

SCRIPT_VERSION = "experimentA9_crossfitted_cycle_age_safety_blend_v1"
EXPERIMENT_ID = "experimentA9"
DEFAULT_OUTPUT = "outputs/experimentA9_crossfitted_cycle_age_safety_blend"
MODEL_SEEDS = [100, 101, 102, 103, 104]
TARGET_SPLIT_SEEDS = [6401, 6402, 6403, 6404, 6405]
ROLE_PARTITIONS = [1, 2, 3, 4, 5]
SELECTION_ENDPOINT_SEEDS = [8801, 8802, 8803, 8804, 8805]
CONFIRMATION_ENDPOINT_SEEDS = [8901, 8902, 8903, 8904, 8905]
ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
GATE_THRESHOLD = 60.0
MARGIN = 0.03
BASE, AGE, BLEND = "baseline_sensor_settings", "causal_cycle_age", "crossfitted_safety_blend"
PAIR_KEYS = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"]
FIXED_KEYS = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_fraction"]
PRED_KEYS = ["unit", "endpoint_fraction", "unit_window_index", "label"]
QUESTION = "Does a selection-only bounded baseline/cycle-age blend retain causal-age gains while restoring true-high-RUL NASA/RMSE safety?"


def configure_a8() -> None:
    # A8 workers resolve these globals at call time.  Pointing __file__ here
    # makes all child workers execute this A9 wrapper rather than A8.
    a8.__file__ = str(Path(__file__).resolve())
    a8.SCRIPT_VERSION = SCRIPT_VERSION
    a8.EXPERIMENT_ID = EXPERIMENT_ID
    a8.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    a8.MODEL_SEEDS = MODEL_SEEDS.copy()
    a8.TARGET_SPLIT_SEEDS = TARGET_SPLIT_SEEDS.copy()
    a8.ROLE_PARTITIONS = ROLE_PARTITIONS.copy()
    a8.SELECTION_ENDPOINT_SEEDS = SELECTION_ENDPOINT_SEEDS.copy()
    a8.CONFIRMATION_ENDPOINT_SEEDS = CONFIRMATION_ENDPOINT_SEEDS.copy()


configure_a8()
_A8_LOAD_CONFIG = a8.load_config


def load_config(args: Any) -> tuple[dict, dict]:
    base, experiment = _A8_LOAD_CONFIG(args)
    experiment = deepcopy(experiment)
    experiment.update({
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": "crossfitted_cycle_age_safety_blend",
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
        "alpha_grid": list(ALPHA_GRID),
        "prediction_gate_threshold": GATE_THRESHOLD,
        "selection_safety_margin_pct": 100 * MARGIN,
        "selection_confirmation_endpoint_seeds_disjoint": True,
    })
    if args.quick:
        experiment.update({
            "domains": ["FD004"], "model_seeds": [100], "target_split_seeds": [6401],
            "role_partitions": [1], "selection_endpoint_seeds": [8801],
            "confirmation_endpoint_seeds": [8901],
        })
    return base, experiment


def risk(frame: pd.DataFrame, column: str) -> dict[str, float]:
    x = pd.DataFrame({"label": frame["label"].astype(float), "prediction": frame[column].astype(float)})
    x["error"] = x["prediction"] - x["label"]
    return a8.a4.endpoint_risk_metrics(x)


def rel(candidate: dict[str, float], baseline: dict[str, float], metric: str) -> float:
    if float(baseline[metric]) <= 0:
        raise RuntimeError(f"non-positive baseline {metric}")
    return float((candidate[metric] - baseline[metric]) / baseline[metric])


def select_paired(frame: pd.DataFrame, units: list[int], *, assignment=None, fraction=None) -> pd.DataFrame:
    parts = []
    for representation in (BASE, AGE):
        part = a8.a21.endpoint_subset(
            frame[frame["representation"] == representation], units,
            assignment=assignment, fraction=fraction,
        )
        parts.append(part[PRED_KEYS + ["representation", "prediction"]])
    wide = pd.concat(parts, ignore_index=True).pivot(index=PRED_KEYS, columns="representation", values="prediction").reset_index()
    if BASE not in wide or AGE not in wide or len(wide) != len(units):
        raise RuntimeError("A9 baseline/cycle-age endpoint alignment failed")
    wide = wide.rename(columns={BASE: "prediction_baseline", AGE: "prediction_cycle_age"})
    wide["prediction_mean_for_gate"] = (wide["prediction_baseline"] + wide["prediction_cycle_age"]) / 2
    wide["prediction_gate"] = np.where(wide["prediction_mean_for_gate"] >= GATE_THRESHOLD, "high_pred_rul_ge60", "lower_pred_rul_lt60")
    return wide


def blend(frame: pd.DataFrame, alpha_high: float, alpha_low: float) -> pd.DataFrame:
    out = frame.copy()
    out["alpha_high"], out["alpha_low"] = float(alpha_high), float(alpha_low)
    out["blend_alpha"] = np.where(out["prediction_gate"] == "high_pred_rul_ge60", alpha_high, alpha_low).astype(float)
    out["prediction_blend"] = out["prediction_baseline"] + out["blend_alpha"] * (out["prediction_cycle_age"] - out["prediction_baseline"])
    lo = np.minimum(out["prediction_baseline"], out["prediction_cycle_age"])
    hi = np.maximum(out["prediction_baseline"], out["prediction_cycle_age"])
    if not bool(((out["prediction_blend"] >= lo - 1e-8) & (out["prediction_blend"] <= hi + 1e-8)).all()):
        raise AssertionError("blend was not a convex combination")
    return out


def choose_alpha(selection: pd.DataFrame, common: dict[str, Any], experiment: dict) -> tuple[dict[str, Any], pd.DataFrame]:
    high_mask = selection["label"] > float(experiment["high_rul_threshold"])
    if not high_mask.any() or high_mask.all():
        raise RuntimeError("selection data lacks a true-RUL stage")
    rows = []
    for ah, al in product(experiment["alpha_grid"], repeat=2):
        applied = blend(selection, ah, al)
        base_all, cand_all = risk(applied, "prediction_baseline"), risk(applied, "prediction_blend")
        base_high, cand_high = risk(applied[high_mask], "prediction_baseline"), risk(applied[high_mask], "prediction_blend")
        base_low, cand_low = risk(applied[~high_mask], "prediction_baseline"), risk(applied[~high_mask], "prediction_blend")
        h_nasa, h_rmse = rel(cand_high, base_high, "nasa_score"), rel(cand_high, base_high, "rmse")
        l_nasa, l_rmse = rel(cand_low, base_low, "nasa_score"), rel(cand_low, base_low, "rmse")
        rows.append({**common, "alpha_high": ah, "alpha_low": al,
            "full_nasa_relative_delta": rel(cand_all, base_all, "nasa_score"),
            "full_rmse_relative_delta": rel(cand_all, base_all, "rmse"),
            "high_rul_nasa_relative_delta": h_nasa, "high_rul_rmse_relative_delta": h_rmse,
            "low_mid_rul_nasa_relative_delta": l_nasa, "low_mid_rul_rmse_relative_delta": l_rmse,
            "selection_safety_feasible": max(h_nasa, h_rmse, l_nasa, l_rmse) <= MARGIN,
            "selection_gate_high_rate": float((applied["prediction_gate"] == "high_pred_rul_ge60").mean()),
            "selection_n_records": int(len(selection)), "selection_uses_labels_only": True,
        })
    grid = pd.DataFrame(rows)
    feasible = grid[grid["selection_safety_feasible"]].copy()
    if feasible.empty:
        chosen = grid.query("alpha_high == 0 and alpha_low == 0").iloc[0]
    else:
        chosen = feasible.sort_values(["full_nasa_relative_delta", "full_rmse_relative_delta", "alpha_high", "alpha_low"], kind="mergesort").iloc[0]
    params = {**common, "alpha_high": float(chosen.alpha_high), "alpha_low": float(chosen.alpha_low),
        "selection_safety_feasible": bool(chosen.selection_safety_feasible),
        "fallback_to_baseline": bool(chosen.alpha_high == 0 and chosen.alpha_low == 0),
        "selection_full_nasa_relative_delta": float(chosen.full_nasa_relative_delta),
        "selection_full_rmse_relative_delta": float(chosen.full_rmse_relative_delta),
        "selection_high_rul_nasa_relative_delta": float(chosen.high_rul_nasa_relative_delta),
        "selection_high_rul_rmse_relative_delta": float(chosen.high_rul_rmse_relative_delta),
        "selection_low_mid_rul_nasa_relative_delta": float(chosen.low_mid_rul_nasa_relative_delta),
        "selection_low_mid_rul_rmse_relative_delta": float(chosen.low_mid_rul_rmse_relative_delta),
        "selection_gate_high_rate": float(chosen.selection_gate_high_rate),
        "confirmation_used_for_alpha_selection": False}
    return params, grid


def evaluation_rows(frame: pd.DataFrame, common: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, column in ((BASE, "prediction_baseline"), (AGE, "prediction_cycle_age"), (BLEND, "prediction_blend")):
        rows.append({**common, "variant": variant, **risk(frame, column),
            "evaluation_engine_count": int(frame.unit.nunique()),
            "prediction_gate_high_rate": float((frame.prediction_gate == "high_pred_rul_ge60").mean()),
            "official_test_files_accessed": False, "official_test_forward_run": False})
    return rows


def crossfit(endpoints: pd.DataFrame, protocols: dict[str, dict], experiment: dict) -> dict[str, pd.DataFrame]:
    selection_parts, confirmation_parts, grid_parts = [], [], []
    parameters, selection_rows, confirmation_rows, fixed_rows = [], [], [], []
    for (domain, mseed, split_seed), source in endpoints.groupby(["target_domain", "model_seed", "target_split_seed"]):
        split = protocols[str(domain)]["role_splits"][str(int(split_seed))]
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]
            selection_units = list(map(int, roles["selection_units"]))
            confirmation_units = list(map(int, roles["confirmation_units"]))
            if set(selection_units) & set(confirmation_units):
                raise AssertionError("selection/confirmation engine overlap")
            common = {"target_domain": str(domain), "model_seed": int(mseed), "target_split_seed": int(split_seed), "role_partition": int(partition)}
            selected_parts = []
            for endpoint_seed in experiment["selection_endpoint_seeds"]:
                assignment = a8.a21.balanced_assignment(selection_units, str(domain), int(split_seed), int(partition), int(endpoint_seed), "selection")
                selected_parts.append(select_paired(source, selection_units, assignment=assignment).assign(**common, endpoint_seed=int(endpoint_seed), evaluation_role="selection"))
            selection = pd.concat(selected_parts, ignore_index=True)
            param, grid = choose_alpha(selection, common, experiment)
            parameters.append(param); grid_parts.append(grid)
            selection = blend(selection, param["alpha_high"], param["alpha_low"])
            selection_parts.append(selection)
            for endpoint_seed in experiment["selection_endpoint_seeds"]:
                chunk = selection[selection.endpoint_seed == int(endpoint_seed)]
                selection_rows.extend(evaluation_rows(chunk, {**common, "endpoint_seed": int(endpoint_seed), "evaluation_role": "selection", "evaluation_protocol": "balanced_endpoint", "alpha_high": param["alpha_high"], "alpha_low": param["alpha_low"]}))
            for endpoint_seed in experiment["confirmation_endpoint_seeds"]:
                assignment = a8.a21.balanced_assignment(confirmation_units, str(domain), int(split_seed), int(partition), int(endpoint_seed), "confirmation")
                applied = blend(select_paired(source, confirmation_units, assignment=assignment), param["alpha_high"], param["alpha_low"]).assign(**common, endpoint_seed=int(endpoint_seed), evaluation_role="confirmation")
                confirmation_parts.append(applied)
                confirmation_rows.extend(evaluation_rows(applied, {**common, "endpoint_seed": int(endpoint_seed), "evaluation_role": "confirmation", "evaluation_protocol": "balanced_endpoint", "alpha_high": param["alpha_high"], "alpha_low": param["alpha_low"]}))
            for fraction in experiment["endpoint_fractions"]:
                applied = blend(select_paired(source, confirmation_units, fraction=float(fraction)), param["alpha_high"], param["alpha_low"])
                fixed_rows.extend(evaluation_rows(applied, {**common, "endpoint_fraction": float(fraction), "evaluation_role": "confirmation", "evaluation_protocol": f"fixed_endpoint_{int(round(100 * float(fraction))):03d}", "alpha_high": param["alpha_high"], "alpha_low": param["alpha_low"]}))
    return {"selection_predictions": pd.concat(selection_parts, ignore_index=True), "confirmation_predictions": pd.concat(confirmation_parts, ignore_index=True), "selection_run": pd.DataFrame(selection_rows), "confirmation_run": pd.DataFrame(confirmation_rows), "fixed_run": pd.DataFrame(fixed_rows), "parameters": pd.DataFrame(parameters), "grid": pd.concat(grid_parts, ignore_index=True)}


def pair(results: pd.DataFrame, candidate: str, keys: list[str]) -> pd.DataFrame:
    pivot = results.pivot(index=keys, columns="variant", values=a8.METRICS).reset_index()
    pivot.columns = ["_".join(str(x) for x in c if str(x)) if isinstance(c, tuple) else c for c in pivot.columns]
    out = pivot[keys].copy()
    for metric in a8.METRICS:
        out[f"{metric}_{BASE}"] = pivot[f"{metric}_{BASE}"].astype(float)
        out[f"{metric}_{candidate}"] = pivot[f"{metric}_{candidate}"].astype(float)
        out[f"{metric}_delta_candidate_minus_baseline"] = out[f"{metric}_{candidate}"] - out[f"{metric}_{BASE}"]
    out["candidate"] = candidate
    out["nasa_relative_delta"] = out["nasa_score_delta_candidate_minus_baseline"] / out[f"nasa_score_{BASE}"]
    out["rmse_relative_delta"] = out["rmse_delta_candidate_minus_baseline"] / out[f"rmse_{BASE}"]
    out["candidate_nasa_win"] = out["nasa_score_delta_candidate_minus_baseline"] < 0
    out["candidate_rmse_win"] = out["rmse_delta_candidate_minus_baseline"] < 0
    return out.sort_values(keys)


def stage_pair(predictions: pd.DataFrame, high: bool, experiment: dict) -> pd.DataFrame:
    stage = predictions[predictions.label > float(experiment["high_rul_threshold"])].copy() if high else predictions[predictions.label <= float(experiment["high_rul_threshold"])].copy()
    rows = []
    for values, frame in stage.groupby(PAIR_KEYS):
        base, cand = risk(frame, "prediction_baseline"), risk(frame, "prediction_blend")
        row = dict(zip(PAIR_KEYS, values)); row.update({"rul_stage": "high_rul_gt60" if high else "low_or_mid_rul_le60", "rul_threshold": float(experiment["high_rul_threshold"]), "stage_engine_count": int(frame.unit.nunique()), "candidate": BLEND})
        for metric in a8.METRICS:
            row[f"{metric}_{BASE}"] = base[metric]; row[f"{metric}_{BLEND}"] = cand[metric]; row[f"{metric}_delta_candidate_minus_baseline"] = cand[metric] - base[metric]
        row["nasa_relative_delta"] = rel(cand, base, "nasa_score"); row["rmse_relative_delta"] = rel(cand, base, "rmse")
        row["candidate_nasa_win"] = cand["nasa_score"] < base["nasa_score"]; row["candidate_rmse_win"] = cand["rmse"] < base["rmse"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(PAIR_KEYS)
    expected = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    if len(out) != expected: raise RuntimeError("incomplete A9 true-stage confirmation pairs")
    return out


def augment_a9(args: Any, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"]); paths = a8.root_paths(output)
    endpoints = a8.load_csv(paths["endpoint_predictions"])
    protocols = a8.read_json(paths["protocol"])
    evaluated = crossfit(endpoints, protocols, experiment)
    confirmation = evaluated["confirmation_run"].sort_values(PAIR_KEYS + ["variant"])
    blend_pairs, age_pairs = pair(confirmation, BLEND, PAIR_KEYS), pair(confirmation, AGE, PAIR_KEYS)
    fixed_pairs = pair(evaluated["fixed_run"], BLEND, FIXED_KEYS)
    high_pairs, low_pairs = stage_pair(evaluated["confirmation_predictions"], True, experiment), stage_pair(evaluated["confirmation_predictions"], False, experiment)
    comparisons = pd.concat([a8.comparison_summary(blend_pairs, experiment, "full_endpoint_blend_vs_baseline"), a8.comparison_summary(age_pairs, experiment, "full_endpoint_age_vs_baseline"), a8.comparison_summary(high_pairs, experiment, "high_rul_blend_vs_baseline"), a8.comparison_summary(low_pairs, experiment, "low_rul_blend_vs_baseline")], ignore_index=True)
    high, low = a8.stage_summary(high_pairs, experiment, "high_rul_blend_vs_baseline"), a8.stage_summary(low_pairs, experiment, "low_rul_blend_vs_baseline")
    overall = comparisons.query("comparison == 'full_endpoint_blend_vs_baseline' and scope == 'ALL'").iloc[0]
    margin = float(experiment["stage_noninferiority_margin_pct"]) / 100.0
    full_ok = float(overall.nasa_relative_boot_ci95_high) < 0 or float(overall.rmse_relative_boot_ci95_high) < 0
    high_ok = high["nasa_relative_ci95"][1] <= margin and high["rmse_relative_ci95"][1] <= margin
    low_ok = low["nasa_relative_ci95"][1] <= margin and low["rmse_relative_ci95"][1] <= margin
    expected_pairs = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    complete = endpoints.cell_id.nunique() == 2 * len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) and len(blend_pairs) == expected_pairs and len(confirmation) == expected_pairs * 3
    decision = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "complete": bool(complete), "quick_mode": bool(experiment["quick_mode"]), "expected_training_cells": 2 * len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]), "completed_training_cells": int(endpoints.cell_id.nunique()), "expected_confirmation_records": expected_pairs * 3, "completed_confirmation_records": int(len(confirmation)), "expected_primary_pairs": expected_pairs, "completed_primary_pairs": int(len(blend_pairs)), "alpha_grid": list(ALPHA_GRID), "prediction_gate_threshold": GATE_THRESHOLD, "selection_confirmation_endpoint_seeds_disjoint": True, "official_test_files_accessed": False, "official_test_forward_run": False, "blend_parameter_summary": {"n_parameter_sets": int(len(evaluated["parameters"])), "alpha_high_mean": float(evaluated["parameters"].alpha_high.mean()), "alpha_low_mean": float(evaluated["parameters"].alpha_low.mean()), "fallback_to_baseline_rate": float(evaluated["parameters"].fallback_to_baseline.astype(bool).mean())}, "full_endpoint_result": {"nasa_improvement_pct": float(overall.nasa_improvement_pct), "nasa_relative_ci95": [float(overall.nasa_relative_boot_ci95_low), float(overall.nasa_relative_boot_ci95_high)], "rmse_degradation_pct": float(overall.rmse_degradation_pct), "rmse_relative_ci95": [float(overall.rmse_relative_boot_ci95_low), float(overall.rmse_relative_boot_ci95_high)], "at_least_one_metric_strictly_improved": bool(full_ok)}, "high_rul_safety_result": {**high, "noninferiority_passed": bool(high_ok)}, "low_rul_safety_result": {**low, "noninferiority_passed": bool(low_ok)}, "passed": bool(complete and full_ok and high_ok and low_ok) if not experiment["quick_mode"] else bool(complete), "reason": "A9 confirmed bounded blend efficacy and stage safety" if complete and full_ok and high_ok and low_ok else "A9 completed, but the cross-fitted blend did not meet every registered criterion", "next_action": "lock_A9_blend_then_official_confirmation" if complete and full_ok and high_ok and low_ok else "stop_cycle_age_blend_direction_and_reassess_experimentA10"}
    extras = {"blend_selection_grid": evaluated["grid"], "blend_parameters": evaluated["parameters"], "blend_selection_predictions": evaluated["selection_predictions"], "blend_confirmation_predictions": evaluated["confirmation_predictions"], "blend_selection_run_level": evaluated["selection_run"], "blend_confirmation_run_level": confirmation, "paired_blend_vs_baseline": blend_pairs, "paired_age_vs_baseline": age_pairs, "fixed_endpoint_paired_blend_vs_baseline": fixed_pairs, "high_rul_paired_blend_vs_baseline": high_pairs, "low_rul_paired_blend_vs_baseline": low_pairs, "comparison_summary": comparisons}
    for stem, frame in extras.items(): a8.a1.atomic_write_text(output / f"{EXPERIMENT_ID}_{stem}.csv", frame.to_csv(index=False))
    a8.a1.atomic_write_text(paths["comparison"], comparisons.to_csv(index=False))
    a8.atomic_json(paths["decision"], decision)
    manifest = a8.read_json(paths["manifest"]); manifest.update({"registered_primary_question": QUESTION, "blend_policy": {"formula": "baseline + alpha * (cycle_age - baseline)", "alpha_grid": list(ALPHA_GRID), "prediction_gate": "mean(current predictions) >= 60", "gate_uses_true_rul": False, "selection_labels_used_only_for_alpha_selection": True, "confirmation_used_for_alpha_selection": False, "safety_margin_pct": 100 * MARGIN}}); a8.atomic_json(paths["manifest"], manifest)
    a8.atomic_json(output / f"{EXPERIMENT_ID}_blend_causality_audit.json", {"experiment_id": EXPERIMENT_ID, "gate_uses_true_rul": False, "blend_uses_only_current_predictions": True, "selection_labels_used_only_to_choose_alpha": True, "confirmation_used_for_alpha_selection": False, "official_test_files_accessed": False, "official_test_forward_run": False})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = a8.parse_args(); base, experiment = load_config(args); a8.validate_config(base, experiment)
    if args.worker_domain is not None:
        if args.worker_seed is None: raise ValueError("--worker-domain requires --worker-seed")
        a8.worker_main(args, base, experiment); return
    a8.parent_main(args, base, experiment)
    if not args.dry_run: augment_a9(args, base, experiment)


if __name__ == "__main__": main()
