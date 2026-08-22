#!/usr/bin/env python3
"""A24.4 frozen four-method descriptive synthesis.

This script combines the already-frozen A23.4 ordinary-transfer predictions
with the already-frozen A24.3 meta-learning predictions.  It never trains or
adapts a predictor and it deliberately does not turn the post-confirmation
synthesis into a new efficacy claim.

Expected methods
----------------
A23.4: scratch_k, pretrain_finetune_k
A24.3: meta_no_graph_k, meta_gnn_k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ID = "experimentA24_4"
SCRIPT_VERSION = "experimentA24_4_meta_vs_ordinary_transfer_synthesis_v1"

A23_FILES = {
    "manifest": "experimentA23_4_manifest.json",
    "decision": "experimentA23_4_confirmation_decision.json",
    "predictions": "experimentA23_4_causal_anchor_predictions.csv",
}
A24_FILES = {
    "manifest": "experimentA24_3_manifest.json",
    "decision": "experimentA24_3_confirmation_decision.json",
    "predictions": "experimentA24_3_causal_anchor_predictions.csv",
}

METHODS = (
    "scratch_k",
    "pretrain_finetune_k",
    "meta_no_graph_k",
    "meta_gnn_k",
)
COMPARISONS = (
    ("meta_no_graph_k", "scratch_k"),
    ("meta_no_graph_k", "pretrain_finetune_k"),
    ("meta_gnn_k", "scratch_k"),
    ("meta_gnn_k", "pretrain_finetune_k"),
)
PAIR_KEYS = (
    "target_domain",
    "model_seed",
    "support_split_seed",
    "shot",
    "engine_id",
    "prefix_label",
    "registered_rul_anchor",
)
CONTEXT_COLUMNS = ("true_rul", "rul_stage")
VALUE_COLUMNS = (
    "prediction",
    "error",
    "absolute_error",
    "squared_error",
    "nasa_score_component",
)
CELL_KEYS = (
    "target_domain",
    "model_seed",
    "support_split_seed",
    "shot",
    "prefix_label",
    "registered_rul_anchor",
    "rul_stage",
)
PRIMARY_SHOT = 5
REGISTERED_ANCHORS = (15.0, 45.0, 90.0)


class ContractError(RuntimeError):
    """Raised when a frozen-input or pairing contract is violated."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen descriptive synthesis of A23.4 and A24.3 outputs."
    )
    parser.add_argument("--a23-4-output-dir", type=Path, required=True)
    parser.add_argument("--a24-3-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=244000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Return the existing completed decision instead of overwriting it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a completed A24.4 result after all input checks pass.",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions < 200:
        parser.error("--bootstrap-repetitions must be at least 200")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{label} is missing required columns: {missing}")


def normalize_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    require(not normalized.isna().any(), f"{label} contains non-boolean values")
    return normalized.astype(bool)


def verify_manifest_artifact(root: Path, manifest: Mapping[str, Any], name: str) -> str:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), f"manifest has no artifact mapping: {root}")
    expected = artifacts.get(name)
    require(isinstance(expected, str) and len(expected) == 64,
            f"manifest has no valid SHA-256 for {name}")
    path = root / name
    require(path.is_file(), f"required artifact does not exist: {path}")
    observed = sha256_file(path)
    require(observed == expected,
            f"SHA-256 mismatch for {path}: expected={expected}, observed={observed}")
    return observed


def validate_decision(decision: Mapping[str, Any], expected_id: str, path: Path) -> None:
    require(decision.get("experiment_id") == expected_id,
            f"unexpected experiment_id in {path}")
    require(decision.get("complete") is True, f"input is not complete: {path}")
    # passed=False is a valid frozen scientific result and must not block synthesis.
    for field in ("new_predictor_training", "official_test_files_accessed",
                  "official_test_forward_run"):
        require(decision.get(field) is False, f"{path}: expected {field}=false")


def validate_cross_contracts(a23_manifest: Mapping[str, Any],
                             a24_manifest: Mapping[str, Any]) -> dict[str, Any]:
    a23_inputs = a23_manifest.get("inputs", {})
    a24_inputs = a24_manifest.get("inputs", {})
    a23_protocol = a23_inputs.get("protocol_hashes")
    a24_protocol = a24_inputs.get("a23_protocol_hashes")
    require(isinstance(a23_protocol, dict) and isinstance(a24_protocol, dict),
            "A23 protocol hash mappings are absent")
    require(a23_protocol == a24_protocol,
            "A23.4 and A24.3 were not evaluated under identical A23 protocol hashes")
    require(a23_inputs.get("config_sha256") == a24_inputs.get("config_sha256"),
            "A23.4 and A24.3 config hashes differ")
    return {
        "a23_protocol_hashes": a23_protocol,
        "config_sha256": a23_inputs.get("config_sha256"),
    }


@dataclass(frozen=True)
class InputBundle:
    a23_manifest: dict[str, Any]
    a24_manifest: dict[str, Any]
    a23_decision: dict[str, Any]
    a24_decision: dict[str, Any]
    hashes: dict[str, str]
    shared_contract: dict[str, Any]


def preflight_inputs(a23_root: Path, a24_root: Path) -> InputBundle:
    a23_root = a23_root.expanduser().resolve()
    a24_root = a24_root.expanduser().resolve()
    require(a23_root.is_dir(), f"A23.4 output directory not found: {a23_root}")
    require(a24_root.is_dir(), f"A24.3 output directory not found: {a24_root}")

    a23_manifest_path = a23_root / A23_FILES["manifest"]
    a24_manifest_path = a24_root / A24_FILES["manifest"]
    a23_decision_path = a23_root / A23_FILES["decision"]
    a24_decision_path = a24_root / A24_FILES["decision"]
    a23_manifest = load_json(a23_manifest_path)
    a24_manifest = load_json(a24_manifest_path)
    a23_decision = load_json(a23_decision_path)
    a24_decision = load_json(a24_decision_path)

    require(a23_manifest.get("experiment_id") == "experimentA23_4",
            "A23.4 manifest experiment_id mismatch")
    require(a24_manifest.get("experiment_id") == "experimentA24_3",
            "A24.3 manifest experiment_id mismatch")
    validate_decision(a23_decision, "experimentA23_4", a23_decision_path)
    validate_decision(a24_decision, "experimentA24_3", a24_decision_path)

    hashes = {
        "a23_manifest_sha256": sha256_file(a23_manifest_path),
        "a24_manifest_sha256": sha256_file(a24_manifest_path),
        "a23_decision_sha256": verify_manifest_artifact(
            a23_root, a23_manifest, A23_FILES["decision"]
        ),
        "a24_decision_sha256": verify_manifest_artifact(
            a24_root, a24_manifest, A24_FILES["decision"]
        ),
        "a23_predictions_sha256": verify_manifest_artifact(
            a23_root, a23_manifest, A23_FILES["predictions"]
        ),
        "a24_predictions_sha256": verify_manifest_artifact(
            a24_root, a24_manifest, A24_FILES["predictions"]
        ),
    }
    shared_contract = validate_cross_contracts(a23_manifest, a24_manifest)
    return InputBundle(
        a23_manifest=a23_manifest,
        a24_manifest=a24_manifest,
        a23_decision=a23_decision,
        a24_decision=a24_decision,
        hashes=hashes,
        shared_contract=shared_contract,
    )


def load_prediction_file(path: Path, method_column: str,
                         allowed_methods: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(PAIR_KEYS) | set(CONTEXT_COLUMNS) | set(VALUE_COLUMNS) | {
        method_column,
        "input_uses_future_cycles",
        "confirmation_used_for_training",
        "new_predictor_training",
        "official_test_files_accessed",
        "official_test_forward_run",
    }
    require_columns(frame, required, str(path))
    frame = frame.rename(columns={method_column: "method"})
    frame = frame.loc[frame["method"].isin(allowed_methods)].copy()
    require(set(frame["method"].unique()) == allowed_methods,
            f"{path} does not contain exactly the required methods {sorted(allowed_methods)}")

    for field in (
        "input_uses_future_cycles",
        "confirmation_used_for_training",
        "new_predictor_training",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        values = normalize_bool(frame[field], f"{path}:{field}")
        require(not values.any(), f"{path}: expected every {field}=false")

    numeric = (
        "model_seed", "support_split_seed", "shot", "engine_id",
        "registered_rul_anchor", "true_rul", "prediction", "error",
        "absolute_error", "squared_error", "nasa_score_component",
    )
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        require(np.isfinite(frame[field].to_numpy(dtype=float)).all(),
                f"{path}:{field} contains NaN or infinite values")
    require((frame["squared_error"] >= 0).all(), f"{path}: negative squared error")
    require((frame["nasa_score_component"] >= 0).all(),
            f"{path}: negative NASA score component")
    require(np.allclose(frame["prediction"] - frame["true_rul"], frame["error"],
                        rtol=1e-6, atol=1e-5),
            f"{path}: prediction - true_rul does not equal error")
    require(np.allclose(frame["error"].abs(), frame["absolute_error"],
                        rtol=1e-6, atol=1e-5),
            f"{path}: absolute_error is inconsistent")
    require(np.allclose(frame["error"].pow(2), frame["squared_error"],
                        rtol=1e-5, atol=1e-4),
            f"{path}: squared_error is inconsistent")
    require(set(frame["registered_rul_anchor"].astype(float).unique()) ==
            set(REGISTERED_ANCHORS), f"{path}: unexpected RUL anchors")

    duplicated = frame.duplicated(list(PAIR_KEYS) + ["method"], keep=False)
    require(not duplicated.any(), f"{path}: duplicate method/pair keys detected")
    return frame[list(PAIR_KEYS) + list(CONTEXT_COLUMNS) + ["method"] +
                 list(VALUE_COLUMNS)].copy()


def normalize_four_methods(a23: pd.DataFrame, a24: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat([a23, a24], ignore_index=True, sort=False)
    require(set(long["method"].unique()) == set(METHODS),
            "the combined prediction table does not contain all four methods")

    method_keys: dict[str, pd.MultiIndex] = {}
    for method in METHODS:
        part = long.loc[long["method"].eq(method), list(PAIR_KEYS)]
        method_keys[method] = pd.MultiIndex.from_frame(part.sort_values(list(PAIR_KEYS)))
    reference_keys = method_keys[METHODS[0]]
    for method in METHODS[1:]:
        require(reference_keys.equals(method_keys[method]),
                f"pair-key set/order differs between {METHODS[0]} and {method}")

    context = long[list(PAIR_KEYS) + list(CONTEXT_COLUMNS)].drop_duplicates()
    require(len(context) == len(reference_keys),
            "true-RUL or RUL-stage context differs across methods")

    wide: pd.DataFrame | None = None
    for method in METHODS:
        part = long.loc[long["method"].eq(method),
                        list(PAIR_KEYS) + list(VALUE_COLUMNS)].copy()
        part = part.rename(columns={name: f"{name}__{method}" for name in VALUE_COLUMNS})
        wide = part if wide is None else wide.merge(
            part, on=list(PAIR_KEYS), how="inner", validate="one_to_one"
        )
    assert wide is not None
    wide = context.merge(wide, on=list(PAIR_KEYS), how="inner", validate="one_to_one")
    require(len(wide) == len(reference_keys), "four-method inner pairing lost records")
    return wide.sort_values(list(PAIR_KEYS), kind="stable").reset_index(drop=True)


def metric_value(frame: pd.DataFrame, method: str, metric: str) -> float:
    if metric == "rmse":
        return float(math.sqrt(frame[f"squared_error__{method}"].mean()))
    if metric == "nasa_score":
        return float(frame[f"nasa_score_component__{method}"].sum())
    raise ValueError(metric)


def relative_degradation(candidate: float, reference: float) -> float:
    require(reference > 0 and math.isfinite(reference),
            "reference metric must be positive and finite")
    return candidate / reference - 1.0


def engine_win_rate(frame: pd.DataFrame, candidate: str,
                    reference: str, metric: str) -> float:
    field = "absolute_error" if metric == "rmse" else "nasa_score_component"
    c = frame[f"{field}__{candidate}"].to_numpy(dtype=float)
    r = frame[f"{field}__{reference}"].to_numpy(dtype=float)
    return float(np.mean(c < r) + 0.5 * np.mean(c == r))


def hierarchical_weight_matrix(cell_meta: pd.DataFrame, repetitions: int,
                               seed: int) -> np.ndarray:
    """Bootstrap domains, seeds, and splits; engines remain aggregated in cells.

    This is intentionally labelled a cluster-level descriptive bootstrap.  It
    avoids pretending that the already-inspected confirmation outcomes create
    a new confirmatory p-value.
    """
    lookup = {
        (str(row.target_domain), int(row.model_seed), int(row.support_split_seed)): i
        for i, row in cell_meta.reset_index(drop=True).iterrows()
    }
    domains = sorted(cell_meta["target_domain"].astype(str).unique())
    seeds_by_domain = {
        d: sorted(cell_meta.loc[cell_meta["target_domain"].astype(str).eq(d),
                                "model_seed"].astype(int).unique())
        for d in domains
    }
    splits_by_domain_seed = {
        (d, s): sorted(cell_meta.loc[
            cell_meta["target_domain"].astype(str).eq(d) &
            cell_meta["model_seed"].astype(int).eq(s),
            "support_split_seed"].astype(int).unique())
        for d in domains for s in seeds_by_domain[d]
    }
    expected = sum(len(splits_by_domain_seed[(d, s)])
                   for d in domains for s in seeds_by_domain[d])
    require(expected == len(cell_meta), "cell hierarchy has duplicate or missing cells")

    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(cell_meta)), dtype=np.uint16)
    for repetition in range(repetitions):
        sampled_domains = rng.choice(domains, size=len(domains), replace=True)
        for domain in sampled_domains:
            seeds = seeds_by_domain[str(domain)]
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for model_seed in sampled_seeds:
                splits = splits_by_domain_seed[(str(domain), int(model_seed))]
                sampled_splits = rng.choice(splits, size=len(splits), replace=True)
                for split_seed in sampled_splits:
                    weights[repetition, lookup[(str(domain), int(model_seed),
                                                int(split_seed))]] += 1
    return weights


def cell_totals(frame: pd.DataFrame, candidate: str, reference: str) -> pd.DataFrame:
    aggregations = {
        f"squared_error__{candidate}": "sum",
        f"squared_error__{reference}": "sum",
        f"nasa_score_component__{candidate}": "sum",
        f"nasa_score_component__{reference}": "sum",
        "engine_id": "size",
    }
    cells = frame.groupby(list(CELL_KEYS), observed=True, sort=True).agg(aggregations)
    cells = cells.rename(columns={"engine_id": "n_pairs"}).reset_index()
    return cells


def bootstrap_ci(cells: pd.DataFrame, candidate: str, reference: str,
                 metric: str, repetitions: int, seed: int) -> tuple[float, float]:
    meta = cells[["target_domain", "model_seed", "support_split_seed"]].copy()
    weights = hierarchical_weight_matrix(meta, repetitions, seed)
    if metric == "rmse":
        c = cells[f"squared_error__{candidate}"].to_numpy(dtype=float)
        r = cells[f"squared_error__{reference}"].to_numpy(dtype=float)
        ratios = np.sqrt((weights @ c) / (weights @ r)) - 1.0
    else:
        c = cells[f"nasa_score_component__{candidate}"].to_numpy(dtype=float)
        r = cells[f"nasa_score_component__{reference}"].to_numpy(dtype=float)
        ratios = (weights @ c) / (weights @ r) - 1.0
    require(np.isfinite(ratios).all(), "non-finite bootstrap relative degradation")
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def comparison_summary(normalized: pd.DataFrame, repetitions: int,
                       base_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combinations = normalized[["shot", "prefix_label", "registered_rul_anchor",
                               "rul_stage"]].drop_duplicates().sort_values(
                                   ["shot", "registered_rul_anchor"]
                               )
    sequence = 0
    for combo in combinations.itertuples(index=False):
        mask = (
            normalized["shot"].eq(combo.shot) &
            normalized["prefix_label"].eq(combo.prefix_label) &
            normalized["registered_rul_anchor"].eq(combo.registered_rul_anchor)
        )
        subset = normalized.loc[mask]
        for candidate, reference in COMPARISONS:
            cells = cell_totals(subset, candidate, reference)
            for metric in ("rmse", "nasa_score"):
                c = metric_value(subset, candidate, metric)
                r = metric_value(subset, reference, metric)
                low, high = bootstrap_ci(
                    cells, candidate, reference, metric, repetitions,
                    base_seed + sequence,
                )
                sequence += 1
                relative = relative_degradation(c, r)
                rows.append({
                    "experiment_id": EXPERIMENT_ID,
                    "analysis_role": (
                        "frozen_primary_shot_descriptive" if int(combo.shot) == PRIMARY_SHOT
                        else "frozen_secondary_shot_descriptive"
                    ),
                    "comparison": f"{candidate}_vs_{reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "shot": int(combo.shot),
                    "prefix_label": combo.prefix_label,
                    "registered_rul_anchor": float(combo.registered_rul_anchor),
                    "rul_stage": combo.rul_stage,
                    "metric": metric,
                    "n_paired_engines": int(len(subset)),
                    "candidate_value": c,
                    "reference_value": r,
                    "relative_degradation": relative,
                    "relative_improvement_pct": -100.0 * relative,
                    "descriptive_relative_ci95_low": low,
                    "descriptive_relative_ci95_high": high,
                    "candidate_engine_win_rate": engine_win_rate(
                        subset, candidate, reference, metric
                    ),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_design": (
                        "target_domain_then_model_seed_then_support_split_cluster"
                    ),
                    "confirmatory_p_value": np.nan,
                    "formal_efficacy_claim": False,
                })
    return pd.DataFrame(rows)


def domain_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["target_domain", "shot", "prefix_label",
                     "registered_rul_anchor", "rul_stage"]
    for keys, subset in normalized.groupby(group_columns, observed=True, sort=True):
        domain, shot, prefix, anchor, stage = keys
        for candidate, reference in COMPARISONS:
            for metric in ("rmse", "nasa_score"):
                c = metric_value(subset, candidate, metric)
                r = metric_value(subset, reference, metric)
                relative = relative_degradation(c, r)
                rows.append({
                    "experiment_id": EXPERIMENT_ID,
                    "target_domain": domain,
                    "comparison": f"{candidate}_vs_{reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "shot": int(shot),
                    "prefix_label": prefix,
                    "registered_rul_anchor": float(anchor),
                    "rul_stage": stage,
                    "metric": metric,
                    "n_paired_engines": int(len(subset)),
                    "candidate_value": c,
                    "reference_value": r,
                    "relative_degradation": relative,
                    "relative_improvement_pct": -100.0 * relative,
                    "candidate_engine_win_rate": engine_win_rate(
                        subset, candidate, reference, metric
                    ),
                    "formal_efficacy_claim": False,
                })
    return pd.DataFrame(rows)


def shot_summary(method_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = method_summary.groupby(
        ["comparison", "candidate", "reference", "shot", "metric"],
        observed=True, sort=True,
    )
    result = grouped.agg(
        anchor_checks=("registered_rul_anchor", "size"),
        anchors_with_mean_improvement=(
            "relative_degradation", lambda values: int((values < 0).sum())
        ),
        relative_degradation_mean=("relative_degradation", "mean"),
        relative_degradation_worst=("relative_degradation", "max"),
        engine_win_rate_mean=("candidate_engine_win_rate", "mean"),
    ).reset_index()
    result.insert(0, "experiment_id", EXPERIMENT_ID)
    result["formal_efficacy_claim"] = False
    return result


def make_integrity(bundle: InputBundle, normalized: pd.DataFrame,
                   args: argparse.Namespace) -> dict[str, Any]:
    method_counts = {
        method: int(len(normalized)) for method in METHODS
    }
    # Every normalized row contains all four methods.
    factorial = normalized[["target_domain", "model_seed", "support_split_seed",
                            "shot"]].drop_duplicates()
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "a23_4_output_dir": str(args.a23_4_output_dir.expanduser().resolve()),
        "a24_3_output_dir": str(args.a24_3_output_dir.expanduser().resolve()),
        "input_hashes": bundle.hashes,
        "shared_contract": bundle.shared_contract,
        "a23_4_complete": True,
        "a24_3_complete": True,
        "a23_4_passed": bundle.a23_decision.get("passed"),
        "a24_3_passed": bundle.a24_decision.get("passed"),
        "methods": list(METHODS),
        "method_prediction_records": method_counts,
        "normalized_four_method_pairs": int(len(normalized)),
        "factorial_cells": int(len(factorial)),
        "target_domains": sorted(normalized["target_domain"].astype(str).unique()),
        "model_seeds": sorted(normalized["model_seed"].astype(int).unique().tolist()),
        "support_split_seeds": sorted(
            normalized["support_split_seed"].astype(int).unique().tolist()
        ),
        "shots": sorted(normalized["shot"].astype(int).unique().tolist()),
        "rul_anchors": sorted(
            normalized["registered_rul_anchor"].astype(float).unique().tolist()
        ),
        "pair_keys_unique_for_every_method": True,
        "true_rul_and_stage_match_across_methods": True,
        "input_uses_future_cycles": False,
        "confirmation_used_for_training": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def make_decision(integrity: Mapping[str, Any], summary: pd.DataFrame,
                  repetitions: int) -> dict[str, Any]:
    focus = summary.loc[
        summary["comparison"].eq("meta_no_graph_k_vs_pretrain_finetune_k") &
        summary["shot"].isin([1, 2, 5])
    ]
    direction_count = int((focus["relative_degradation"] < 0).sum())
    return {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Across frozen A23.4 and A24.3 confirmation outputs, how do Reptile "
            "Meta-noGraph and Meta-GNN descriptively compare with matched scratch "
            "and ordinary pretrain-plus-finetune baselines across causal RUL anchors?"
        ),
        "complete": True,
        "passed": None,
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "post_confirmation_synthesis": True,
        "methods": list(METHODS),
        "comparisons": [f"{c}_vs_{r}" for c, r in COMPARISONS],
        "normalized_four_method_pairs": int(integrity["normalized_four_method_pairs"]),
        "completed_descriptive_checks": int(len(summary)),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_design": (
            "target_domain_then_model_seed_then_support_split_cluster"
        ),
        "confirmatory_p_values_computed": False,
        "meta_no_graph_vs_pft_mean_improvement_checks_at_k1_k2_k5": direction_count,
        "meta_no_graph_vs_pft_total_checks_at_k1_k2_k5": int(len(focus)),
        "a23_4_primary_result_frozen": True,
        "a24_3_primary_result_frozen": True,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A24.4 completed a hash-verified, engine-matched descriptive synthesis "
            "of frozen ordinary-transfer and meta-learning confirmation outputs"
        ),
        "interpretation_limit": (
            "The same confirmation outcomes had already been examined in A23.4 and "
            "A24.3. A24.4 cannot create a new confirmatory efficacy claim; its "
            "intervals and win rates are descriptive evidence for A25 design only."
        ),
        "next_action": (
            "preregister_A25_independent_meta_learning_confirmation_if_the_locked_"
            "descriptive_signal_is_scientifically_material_else_reassess_Reptile"
        ),
    }


def dry_run_payload(bundle: InputBundle, args: argparse.Namespace) -> dict[str, Any]:
    a23_header = pd.read_csv(
        args.a23_4_output_dir / A23_FILES["predictions"], nrows=0
    )
    a24_header = pd.read_csv(
        args.a24_3_output_dir / A24_FILES["predictions"], nrows=0
    )
    require_columns(a23_header, set(PAIR_KEYS) | set(VALUE_COLUMNS) | {"regime"},
                    "A23.4 prediction header")
    require_columns(a24_header, set(PAIR_KEYS) | set(VALUE_COLUMNS) | {"method"},
                    "A24.3 prediction header")
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "input_hashes_verified": True,
        "shared_protocol_and_config_hashes_verified": True,
        "methods": list(METHODS),
        "comparisons": [f"{c}_vs_{r}" for c, r in COMPARISONS],
        "pair_keys": list(PAIR_KEYS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ContractError(
            f"run lock already exists: {path}; verify no A24.4 process is running "
            "before removing a stale lock"
        ) from exc


def run(args: argparse.Namespace) -> int:
    args.a23_4_output_dir = args.a23_4_output_dir.expanduser().resolve()
    args.a24_3_output_dir = args.a24_3_output_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    decision_path = args.output_dir / "experimentA24_4_confirmation_decision.json"

    if decision_path.is_file() and args.resume:
        decision = load_json(decision_path)
        require(decision.get("complete") is True,
                "--resume found an incomplete decision; use a new output directory")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        print("[A24.4] existing complete synthesis reused")
        return 0
    if decision_path.exists() and not args.force:
        raise ContractError(
            f"completed output already exists: {decision_path}; use --resume to reuse "
            "or --force only if replacement is intentional"
        )

    bundle = preflight_inputs(args.a23_4_output_dir, args.a24_3_output_dir)
    if args.dry_run:
        preview = dry_run_payload(bundle, args)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("[A24.4] dry-run passed; no predictor was trained and no output was written")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "experimentA24_4_run.lock"
    lock_fd = acquire_lock(lock_path)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        print("[A24.4] loading frozen engine-level predictions ...", flush=True)
        a23 = load_prediction_file(
            args.a23_4_output_dir / A23_FILES["predictions"],
            "regime", {"scratch_k", "pretrain_finetune_k"},
        )
        a24 = load_prediction_file(
            args.a24_3_output_dir / A24_FILES["predictions"],
            "method", {"meta_no_graph_k", "meta_gnn_k"},
        )
        normalized = normalize_four_methods(a23, a24)
        print(f"[A24.4] paired four methods for {len(normalized)} engine endpoints", flush=True)

        integrity = make_integrity(bundle, normalized, args)
        summary = comparison_summary(
            normalized, args.bootstrap_repetitions, args.bootstrap_seed
        )
        domains = domain_summary(normalized)
        shots = shot_summary(summary)
        decision = make_decision(integrity, summary, args.bootstrap_repetitions)

        outputs = {
            "experimentA24_4_normalized_four_method_pairs.csv": normalized,
            "experimentA24_4_method_comparison_summary.csv": summary,
            "experimentA24_4_domain_summary.csv": domains,
            "experimentA24_4_shot_summary.csv": shots,
        }
        for name, frame in outputs.items():
            atomic_write_csv(args.output_dir / name, frame)
        atomic_write_json(args.output_dir / "experimentA24_4_input_integrity.json", integrity)
        atomic_write_json(decision_path, decision)

        artifact_names = list(outputs) + [
            "experimentA24_4_input_integrity.json",
            "experimentA24_4_confirmation_decision.json",
        ]
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "inputs": bundle.hashes,
            "artifacts": {
                name: sha256_file(args.output_dir / name) for name in artifact_names
            },
            "descriptive_only": True,
            "formal_efficacy_claim": False,
            "new_predictor_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_write_json(args.output_dir / "experimentA24_4_manifest.json", manifest)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        print("[A24.4] completed frozen four-method descriptive synthesis", flush=True)
        return 0
    finally:
        os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()


def self_test() -> int:
    rows: list[dict[str, Any]] = []
    for domain in ("FD001", "FD002"):
        for model_seed in (1, 2):
            for split in (11, 12):
                for engine in (1, 2, 3):
                    row: dict[str, Any] = {
                        "target_domain": domain,
                        "model_seed": model_seed,
                        "support_split_seed": split,
                        "shot": 5,
                        "engine_id": engine,
                        "prefix_label": "rul_anchor_015",
                        "registered_rul_anchor": 15.0,
                        "true_rul": 15.0,
                        "rul_stage": "low_rul_le30",
                    }
                    errors = {
                        "scratch_k": 10.0,
                        "pretrain_finetune_k": 8.0,
                        "meta_no_graph_k": 6.0,
                        "meta_gnn_k": 5.0,
                    }
                    for method, error in errors.items():
                        row[f"prediction__{method}"] = 15.0 + error
                        row[f"error__{method}"] = error
                        row[f"absolute_error__{method}"] = abs(error)
                        row[f"squared_error__{method}"] = error * error
                        row[f"nasa_score_component__{method}"] = math.exp(error / 10.0) - 1.0
                    rows.append(row)
    frame = pd.DataFrame(rows)
    result = comparison_summary(frame, repetitions=200, base_seed=123)
    require(len(result) == 8, "self-test summary row count mismatch")
    require((result["relative_degradation"] < 0).all(),
            "self-test expected every candidate to improve")
    domains = domain_summary(frame)
    require(len(domains) == 16, "self-test domain row count mismatch")
    print("[A24.4] self-test passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except (ContractError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[A24.4] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
