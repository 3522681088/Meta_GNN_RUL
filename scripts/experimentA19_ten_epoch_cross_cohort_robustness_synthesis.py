"""Experiment A19: training-only robustness synthesis for the locked 10-epoch A9 blend.

This CPU analysis reads three completed, independent training-only cohorts:
  * A9,
  * A15 budget=10,
  * A18 budget=10.

It does not train a predictor, select a new policy, read official C-MAPSS test
files, or alter the existing A9_1 official 10-epoch deployment policy.  Its
purpose is descriptive: document how consistently the already locked 10-epoch
blend improves on its baseline across independent cohorts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "experimentA19_ten_epoch_cross_cohort_robustness_synthesis_v1"
EXPERIMENT_ID = "experimentA19"
QUESTION = (
    "Across the completed independent A9, A15-budget10 and A18-budget10 "
    "training-only cohorts, does the locked 10-epoch selection-only "
    "baseline/cycle-age blend consistently retain full-endpoint efficacy and "
    "true-stage safety relative to its own baseline?"
)
DEFAULT_OUTPUT = "outputs/experimentA19_ten_epoch_cross_cohort_robustness_synthesis"
DEFAULT_REPETITIONS = 5000
BASELINE = "baseline_sensor_settings"
PAIR_KEYS = [
    "target_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
    "endpoint_seed",
]
STAGES = {
    "full_endpoint": "paired_blend_vs_baseline",
    "high_rul_gt60": "high_rul_paired_blend_vs_baseline",
    "low_or_mid_rul_le60": "low_rul_paired_blend_vs_baseline",
}
METRICS = ("nasa_score", "rmse")


@dataclass(frozen=True)
class CohortSpec:
    name: str
    prefix: str
    directory: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9-output-dir", required=True)
    parser.add_argument("--a15-budget10-output-dir", required=True)
    parser.add_argument("--a18-budget10-output-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    text = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16) % (2**32 - 1)


def normalise_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_json(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(normalise_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A19 input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"A19 expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required A19 CSV is missing or empty: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(f"A19 CSV is empty: {path}") from error


@contextmanager
def exclusive_lock(root: Path):
    lock_path = root / "experimentA19_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(f"another A19 process owns {root}; lock owner: {owner}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "script_version": SCRIPT_VERSION}))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def cohort_specs(args: argparse.Namespace) -> list[CohortSpec]:
    return [
        CohortSpec("A9", "experimentA9", Path(args.a9_output_dir).expanduser().resolve()),
        CohortSpec(
            "A15_budget10",
            "experimentA15_budget10",
            Path(args.a15_budget10_output_dir).expanduser().resolve(),
        ),
        CohortSpec(
            "A18_budget10",
            "experimentA18_budget10",
            Path(args.a18_budget10_output_dir).expanduser().resolve(),
        ),
    ]


def expected_paths(spec: CohortSpec) -> dict[str, Path]:
    paths = {
        "manifest": spec.directory / f"{spec.prefix}_manifest.json",
        "decision": spec.directory / f"{spec.prefix}_confirmation_decision.json",
    }
    for stage, suffix in STAGES.items():
        paths[stage] = spec.directory / f"{spec.prefix}_{suffix}.csv"
    return paths


def require_clean_decision(spec: CohortSpec, decision: dict[str, Any]) -> int:
    expected_id = spec.prefix
    required = {
        "experiment_id": expected_id,
        "complete": True,
        "quick_mode": False,
        "passed": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for key, value in required.items():
        if decision.get(key) != value:
            raise RuntimeError(
                f"A19 requires a clean completed cohort; {spec.name} has {key}={decision.get(key)!r}, expected {value!r}"
            )
    expected_pairs = decision.get("expected_primary_pairs")
    completed_pairs = decision.get("completed_primary_pairs")
    if not isinstance(expected_pairs, int) or expected_pairs <= 0 or completed_pairs != expected_pairs:
        raise RuntimeError(f"A19 cohort {spec.name} has invalid primary-pair counts")
    expected_training = decision.get("expected_training_cells")
    completed_training = decision.get("completed_training_cells")
    if not isinstance(expected_training, int) or expected_training <= 0 or completed_training != expected_training:
        raise RuntimeError(f"A19 cohort {spec.name} has invalid training-cell counts")
    expected_confirmation = decision.get("expected_confirmation_records")
    completed_confirmation = decision.get("completed_confirmation_records")
    if (
        not isinstance(expected_confirmation, int)
        or expected_confirmation <= 0
        or completed_confirmation != expected_confirmation
    ):
        raise RuntimeError(f"A19 cohort {spec.name} has invalid confirmation-record counts")
    return int(expected_pairs)


def experiment_config(spec: CohortSpec, manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("experiment_config")
    if not isinstance(config, dict):
        raise RuntimeError(f"A19 cohort {spec.name} manifest lacks experiment_config")
    for name in (
        "model_seeds",
        "target_split_seeds",
        "selection_endpoint_seeds",
        "confirmation_endpoint_seeds",
    ):
        values = config.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise RuntimeError(f"A19 cohort {spec.name} has invalid {name} in its manifest")
    if set(config["selection_endpoint_seeds"]) & set(config["confirmation_endpoint_seeds"]):
        raise RuntimeError(f"A19 cohort {spec.name} has overlapping selection/confirmation endpoint seeds")
    if int(config.get("target_epochs", -1)) != 10:
        raise RuntimeError(f"A19 requires a fixed 10-epoch cohort; {spec.name} is not configured for 10 epochs")
    return config


def metric_columns(frame: pd.DataFrame, candidate: str, path: Path) -> dict[str, tuple[str, str]]:
    columns: dict[str, tuple[str, str]] = {}
    for metric in METRICS:
        baseline_column = f"{metric}_{BASELINE}"
        candidate_column = f"{metric}_{candidate}"
        if baseline_column not in frame.columns or candidate_column not in frame.columns:
            raise RuntimeError(
                f"{path} lacks required pair columns {baseline_column!r} and/or {candidate_column!r}"
            )
        columns[metric] = (baseline_column, candidate_column)
    return columns


def normalise_pair_table(
    spec: CohortSpec,
    stage: str,
    path: Path,
    expected_pairs: int,
) -> pd.DataFrame:
    frame = read_csv(path)
    required = [*PAIR_KEYS, "candidate"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    if len(frame) != expected_pairs:
        raise RuntimeError(f"{path} has {len(frame)} rows; expected {expected_pairs}")
    if frame[PAIR_KEYS].isna().any().any() or frame.duplicated(PAIR_KEYS).any():
        raise RuntimeError(f"{path} does not contain unique, non-null A9 paired keys")
    candidates = [str(value) for value in frame["candidate"].dropna().unique()]
    if len(candidates) != 1:
        raise RuntimeError(f"{path} must contain exactly one candidate label, found {candidates}")
    candidate = candidates[0]
    columns = metric_columns(frame, candidate, path)
    output = frame[PAIR_KEYS].copy()
    output.insert(0, "cohort", spec.name)
    output.insert(1, "stage", stage)
    output.insert(2, "candidate", candidate)
    for metric, (baseline_column, candidate_column) in columns.items():
        baseline = pd.to_numeric(frame[baseline_column], errors="raise").to_numpy(dtype=float)
        candidate_values = pd.to_numeric(frame[candidate_column], errors="raise").to_numpy(dtype=float)
        if (
            (~np.isfinite(baseline)).any()
            or (~np.isfinite(candidate_values)).any()
            or (baseline <= 0).any()
            or (candidate_values <= 0).any()
        ):
            raise RuntimeError(f"{path} contains invalid {metric} values")
        output[f"{metric}_baseline"] = baseline
        output[f"{metric}_blend"] = candidate_values
        output[f"{metric}_relative_delta"] = (candidate_values - baseline) / baseline
        output[f"{metric}_blend_win"] = candidate_values < baseline
    output["target_domain"] = output["target_domain"].astype(str)
    output["model_seed"] = pd.to_numeric(output["model_seed"], errors="raise").astype(int)
    return output.sort_values(PAIR_KEYS).reset_index(drop=True)


def load_cohort(spec: CohortSpec) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    paths = expected_paths(spec)
    manifest = read_json(paths["manifest"])
    decision = read_json(paths["decision"])
    expected_pairs = require_clean_decision(spec, decision)
    config = experiment_config(spec, manifest)
    stage_frames = [
        normalise_pair_table(spec, stage, paths[stage], expected_pairs)
        for stage in STAGES
    ]
    all_pairs = pd.concat(stage_frames, ignore_index=True)
    audit = {
        "cohort": spec.name,
        "directory": str(spec.directory),
        "decision_path": str(paths["decision"]),
        "manifest_path": str(paths["manifest"]),
        "decision_sha256": sha256_file(paths["decision"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "stage_pair_sha256": {stage: sha256_file(paths[stage]) for stage in STAGES},
        "expected_primary_pairs_per_stage": expected_pairs,
        "completed_training_cells": decision["completed_training_cells"],
        "completed_confirmation_records": decision["completed_confirmation_records"],
        "model_seeds": config["model_seeds"],
        "selection_endpoint_seeds": config["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": config["confirmation_endpoint_seeds"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return manifest, decision, all_pairs, audit


def independence_audit(audits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    model_sets = {row["cohort"]: set(map(int, row["model_seeds"])) for row in audits}
    endpoint_sets = {
        row["cohort"]: set(map(int, row["selection_endpoint_seeds"]))
        | set(map(int, row["confirmation_endpoint_seeds"]))
        for row in audits
    }
    names = list(model_sets)
    all_models_disjoint = True
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            model_overlap = sorted(model_sets[left] & model_sets[right])
            endpoint_overlap = sorted(endpoint_sets[left] & endpoint_sets[right])
            rows.append(
                {
                    "left_cohort": left,
                    "right_cohort": right,
                    "model_seed_overlap": model_overlap,
                    "endpoint_seed_overlap": endpoint_overlap,
                    "model_seed_sets_disjoint": not bool(model_overlap),
                    "endpoint_seed_sets_disjoint": not bool(endpoint_overlap),
                }
            )
            all_models_disjoint &= not bool(model_overlap)
    if not all_models_disjoint:
        raise RuntimeError("A19 requires disjoint model-seed cohorts")
    return {
        "model_seed_sets_disjoint": True,
        "endpoint_seed_overlap_is_reported_not_a_primary_requirement": True,
        "pairwise_cohort_audit": rows,
    }


def summary_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, part in frame.groupby(group_columns, sort=True):
        common = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        for metric in METRICS:
            relative = part[f"{metric}_relative_delta"].to_numpy(dtype=float)
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "n_pairs": int(len(part)),
                    "relative_degradation_pct": float(100.0 * relative.mean()),
                    "relative_improvement_pct": float(-100.0 * relative.mean()),
                    "blend_win_rate": float(part[f"{metric}_blend_win"].mean()),
                    "baseline_metric_mean": float(part[f"{metric}_baseline"].mean()),
                    "blend_metric_mean": float(part[f"{metric}_blend"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(group_columns + ["metric"]).reset_index(drop=True)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Descriptive resampling: cohort -> domain -> model seed -> paired cell."""
    hierarchy: dict[str, dict[str, list[np.ndarray]]] = {}
    for cohort, cohort_frame in frame.groupby("cohort", sort=True):
        domains: dict[str, list[np.ndarray]] = {}
        for domain, domain_frame in cohort_frame.groupby("target_domain", sort=True):
            model_arrays = [
                part[column].to_numpy(dtype=float)
                for _, part in domain_frame.groupby("model_seed", sort=True)
            ]
            if not model_arrays or any(len(values) == 0 for values in model_arrays):
                raise RuntimeError(f"invalid A19 hierarchy at cohort={cohort}, domain={domain}")
            domains[str(domain)] = model_arrays
        if not domains:
            raise RuntimeError(f"empty A19 cohort in bootstrap: {cohort}")
        hierarchy[str(cohort)] = domains
    cohorts = list(hierarchy)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        selected: list[np.ndarray] = []
        for cohort_index in rng.integers(0, len(cohorts), size=len(cohorts)):
            domains = hierarchy[cohorts[int(cohort_index)]]
            domain_names = list(domains)
            for domain_index in rng.integers(0, len(domain_names), size=len(domain_names)):
                model_arrays = domains[domain_names[int(domain_index)]]
                for model_index in rng.integers(0, len(model_arrays), size=len(model_arrays)):
                    values = model_arrays[int(model_index)]
                    selected.append(values[rng.integers(0, len(values), size=len(values))])
        draws[repetition] = float(np.concatenate(selected).mean())
    return draws


def cross_cohort_summary(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, stage_frame in frame.groupby("stage", sort=True):
        for metric in METRICS:
            column = f"{metric}_relative_delta"
            values = stage_frame[column].to_numpy(dtype=float)
            draws = hierarchical_bootstrap(
                stage_frame,
                column,
                repetitions,
                stable_seed(EXPERIMENT_ID, str(stage), metric),
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            cohort_means = stage_frame.groupby("cohort", sort=True)[column].mean()
            rows.append(
                {
                    "stage": str(stage),
                    "metric": metric,
                    "n_cohorts": int(stage_frame["cohort"].nunique()),
                    "n_pairs": int(len(stage_frame)),
                    "relative_degradation_pct": float(100.0 * values.mean()),
                    "relative_improvement_pct": float(-100.0 * values.mean()),
                    "descriptive_relative_ci95_low": float(low),
                    "descriptive_relative_ci95_high": float(high),
                    "cohort_relative_delta_min_pct": float(100.0 * cohort_means.min()),
                    "cohort_relative_delta_max_pct": float(100.0 * cohort_means.max()),
                    "cohorts_with_mean_improvement": int((cohort_means < 0).sum()),
                    "bootstrap_repetitions": int(repetitions),
                    "bootstrap_design": "cohort_then_target_domain_then_model_seed_then_paired_cell",
                    "interpretation": "descriptive cross-cohort robustness evidence; three cohorts are not a new causal proof",
                }
            )
    return pd.DataFrame(rows).sort_values(["stage", "metric"]).reset_index(drop=True)


def root_manifest(root: Path, args: argparse.Namespace, audits: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "script_hash": sha256_file(Path(__file__)),
        "registered_primary_question": QUESTION,
        "cohorts": audits,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    path = root / "experimentA19_manifest.json"
    if path.is_file():
        existing = read_json(path)
        for key in ("script_hash", "cohorts", "bootstrap_repetitions"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"existing A19 output is incompatible at {key}; use a new output directory"
                )
    atomic_json(path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1000 and not args.dry_run:
        raise ValueError("A19 requires at least 1000 bootstrap repetitions outside --dry-run")
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root):
        final_path = root / "experimentA19_confirmation_decision.json"
        if final_path.is_file() and not args.resume:
            raise RuntimeError("A19 already has a final decision; use a new output directory or --resume")

        loaded = [load_cohort(spec) for spec in cohort_specs(args)]
        decisions = {spec.name: decision for spec, (_, decision, _, _) in zip(cohort_specs(args), loaded)}
        pairs = pd.concat([frame for _, _, frame, _ in loaded], ignore_index=True)
        audits = [audit for _, _, _, audit in loaded]
        independence = independence_audit(audits)
        manifest = root_manifest(root, args, audits)
        integrity = {
            "experiment_id": EXPERIMENT_ID,
            "passed": True,
            "cohort_count": len(audits),
            "all_cohort_decisions_passed": True,
            "model_seed_sets_disjoint": independence["model_seed_sets_disjoint"],
            "cohort_audits": audits,
            "independence_audit": independence,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(root / "experimentA19_input_integrity.json", integrity)
        dry = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "registered_primary_question": QUESTION,
            "output_dir": str(root),
            "cohorts": [audit["cohort"] for audit in audits],
            "expected_cohorts": 3,
            "completed_cohorts": len(audits),
            "expected_pairs_per_stage": int(sum(audit["expected_primary_pairs_per_stage"] for audit in audits)),
            "loaded_pairs_across_three_stages": int(len(pairs)),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "input_integrity_passed": True,
            "new_predictor_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(root / "experimentA19_dry_run.json", dry)
        print(json.dumps(dry, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            print("[A19] dry-run completed; no predictor was trained and no official test data was accessed", flush=True)
            return

        cohort_summary = summary_rows(pairs, ["cohort", "stage"])
        domain_summary = summary_rows(pairs, ["cohort", "target_domain", "stage"])
        cross_summary = cross_cohort_summary(pairs, int(args.bootstrap_repetitions))
        atomic_csv(root / "experimentA19_cohort_summary.csv", cohort_summary)
        atomic_csv(root / "experimentA19_domain_summary.csv", domain_summary)
        atomic_csv(root / "experimentA19_cross_cohort_summary.csv", cross_summary)
        atomic_csv(root / "experimentA19_normalized_paired_cells.csv", pairs)

        all_cohort_passed = all(bool(item.get("passed")) for item in decisions.values())
        expected_pairs_per_stage = int(sum(audit["expected_primary_pairs_per_stage"] for audit in audits))
        for stage in STAGES:
            stage_pairs = int((pairs["stage"] == stage).sum())
            if stage_pairs != expected_pairs_per_stage:
                raise RuntimeError(
                    f"A19 stage count mismatch for {stage}: {stage_pairs} vs {expected_pairs_per_stage}"
                )
        decision = {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": QUESTION,
            "complete": True,
            "quick_mode": False,
            "new_predictor_training": False,
            "cohorts": list(decisions),
            "expected_cohorts": 3,
            "completed_cohorts": len(decisions),
            "expected_pairs_per_stage": expected_pairs_per_stage,
            "completed_pairs_per_stage": expected_pairs_per_stage,
            "all_cohort_decisions_passed": all_cohort_passed,
            "model_seed_sets_disjoint": independence["model_seed_sets_disjoint"],
            "cross_cohort_summary": cross_summary.to_dict(orient="records"),
            "passed": bool(all_cohort_passed and independence["model_seed_sets_disjoint"]),
            "reason": (
                "A19 completed the descriptive cross-cohort robustness synthesis for the locked 10-epoch A9 blend"
                if all_cohort_passed
                else "A19 completed, but one or more input cohorts did not satisfy the registered completed-cohort requirement"
            ),
            "interpretation_limit": (
                "A19 aggregates three completed training-only cohorts; its descriptive bootstrap does not create a new causal or official-test claim."
            ),
            "next_action": "report_locked_10_epoch_A9_scope_and_do_not_reopen_official_test_tuning",
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(
            root / "experimentA19_manifest.json",
            {
                **manifest,
                "input_integrity": str(root / "experimentA19_input_integrity.json"),
                "cohort_summary": str(root / "experimentA19_cohort_summary.csv"),
                "domain_summary": str(root / "experimentA19_domain_summary.csv"),
                "cross_cohort_summary": str(root / "experimentA19_cross_cohort_summary.csv"),
            },
        )
        atomic_json(final_path, decision)
        print("[A19] completed descriptive cross-cohort robustness synthesis", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
