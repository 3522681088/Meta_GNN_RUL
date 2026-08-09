"""Experiment A16: A9 training-budget parsimony confirmation.

This is a CPU-only, no-retraining analysis of the completed A15 runs.  It
compares the complete, independently selection-fitted A9 deployment policy at
five target-adaptation epochs against the corresponding ten- and fifteen-epoch
policies.  The comparison is paired on the exact A15 confirmation cells.

The purpose is deliberately narrow: establish whether the 5-epoch protocol is
non-inferior within a pre-registered 3% relative margin for NASA score and RMSE
on full endpoints and the true high/low RUL safety stages.  It does not access
official C-MAPSS test files and it does not re-select any A9 alpha values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "experimentA16_budget_parsimony_confirmation_v1"
EXPERIMENT_ID = "experimentA16"
QUESTION = (
    "Does the complete selection-only A9 blend at five target-adaptation "
    "epochs remain non-inferior to independently selection-fitted ten- and "
    "fifteen-epoch A9 blends for full endpoint NASA/RMSE and true-stage safety?"
)

PAIR_KEYS = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"]
STAGE_FILES = {
    "full_endpoint": "paired_blend_vs_baseline",
    "high_rul_gt60": "high_rul_paired_blend_vs_baseline",
    "low_or_mid_rul_le60": "low_rul_paired_blend_vs_baseline",
}
METRICS = {
    "nasa_score": "nasa_score_crossfitted_safety_blend",
    "rmse": "rmse_crossfitted_safety_blend",
}
BASELINE_BUDGET = 5
REFERENCE_BUDGETS = (10, 15)
DEFAULT_MARGIN = 0.03
DEFAULT_REPETITIONS = 5000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a15-output-dir",
        type=Path,
        default=None,
        help="A15 root containing budget_05_epochs, budget_10_epochs and budget_15_epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "experimentA16_budget_parsimony_confirmation",
        help="New A16 analysis output directory.",
    )
    parser.add_argument(
        "--noninferiority-margin-pct",
        type=float,
        default=100.0 * DEFAULT_MARGIN,
        help="Relative non-inferiority margin in percent (default: 3).",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Hierarchical bootstrap repetitions (default: 5000).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write only the dry-run record.")
    return parser.parse_args()


def resolve_a15_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    root = project_root()
    candidates = [
        root / "outputs" / "experimentA15_training_budget_robustness_v3",
        root / "outputs" / "experimentA15_training_budget_robustness_v2",
        root / "outputs" / "experimentA15_training_budget_robustness_gpu_safe",
        root / "outputs" / "experimentA15_training_budget_robustness_fixed",
        root / "outputs" / "experimentA15_training_budget_robustness",
    ]
    for candidate in candidates:
        if all((candidate / f"budget_{budget:02d}_epochs").is_dir() for budget in (5, 10, 15)):
            return candidate.resolve()
    return candidates[0].resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A16 input is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def budget_dir(a15_root: Path, budget: int) -> Path:
    return a15_root / f"budget_{budget:02d}_epochs"


def budget_prefix(budget: int) -> str:
    return f"experimentA15_budget{budget:02d}"


def stage_path(a15_root: Path, budget: int, stage: str) -> Path:
    return budget_dir(a15_root, budget) / f"{budget_prefix(budget)}_{STAGE_FILES[stage]}.csv"


def input_integrity(a15_root: Path) -> dict[str, Any]:
    """Validate that A15 was complete and that A16 reads only training-only inputs."""
    paths: list[Path] = []
    root_decision = a15_root / "experimentA15_confirmation_decision.json"
    if root_decision.is_file():
        paths.append(root_decision)
        root_payload = read_json(root_decision)
        if not bool(root_payload.get("complete")) or not bool(root_payload.get("passed")):
            raise RuntimeError("A16 requires a complete, passing A15 root decision")
        if bool(root_payload.get("official_test_files_accessed")) or bool(root_payload.get("official_test_forward_run")):
            raise RuntimeError("A16 must not consume A15 outputs that accessed official test files")
    else:
        root_payload = {"available": False}

    budget_checks: dict[str, Any] = {}
    for budget in (5, 10, 15):
        decision_path = budget_dir(a15_root, budget) / f"{budget_prefix(budget)}_confirmation_decision.json"
        payload = read_json(decision_path)
        paths.append(decision_path)
        if not bool(payload.get("complete")) or not bool(payload.get("passed")):
            raise RuntimeError(f"A16 requires a complete, passing A15 budget={budget} result")
        if bool(payload.get("official_test_files_accessed")) or bool(payload.get("official_test_forward_run")):
            raise RuntimeError(f"A15 budget={budget} accessed official test data; A16 must stop")
        if int(payload.get("completed_primary_pairs", -1)) != int(payload.get("expected_primary_pairs", -2)):
            raise RuntimeError(f"A15 budget={budget} primary pairs are incomplete")
        budget_checks[str(budget)] = {
            "decision_path": str(decision_path),
            "complete": bool(payload.get("complete")),
            "passed": bool(payload.get("passed")),
            "completed_primary_pairs": int(payload["completed_primary_pairs"]),
            "official_test_files_accessed": bool(payload.get("official_test_files_accessed")),
            "official_test_forward_run": bool(payload.get("official_test_forward_run")),
        }
        for stage in STAGE_FILES:
            path = stage_path(a15_root, budget, stage)
            if not path.is_file():
                raise FileNotFoundError(f"missing A15 paired stage input: {path}")
            paths.append(path)

    return {
        "a15_root": str(a15_root),
        "root_decision": root_payload,
        "budget_checks": budget_checks,
        "input_hashes": {str(path.relative_to(a15_root)): sha256_file(path) for path in paths},
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def load_stage(a15_root: Path, budget: int, stage: str) -> pd.DataFrame:
    path = stage_path(a15_root, budget, stage)
    frame = pd.read_csv(path)
    required = [*PAIR_KEYS, *METRICS.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    if frame[PAIR_KEYS].isna().any().any():
        raise RuntimeError(f"{path} contains null paired-cell keys")
    if frame.duplicated(PAIR_KEYS).any():
        raise RuntimeError(f"{path} has duplicate paired cells")
    if len(frame) != 2500:
        raise RuntimeError(f"{path} has {len(frame)} pairs, but A15 requires 2500")
    output = frame[[*PAIR_KEYS, *METRICS.values()]].copy()
    for metric in METRICS.values():
        output[metric] = pd.to_numeric(output[metric], errors="raise")
        if (~np.isfinite(output[metric].to_numpy(dtype=float))).any() or (output[metric] <= 0).any():
            raise RuntimeError(f"{path} contains non-positive or non-finite {metric} values")
    return output.sort_values(PAIR_KEYS).reset_index(drop=True)


def build_paired_comparison(a15_root: Path, reference_budget: int, stage: str) -> pd.DataFrame:
    candidate = load_stage(a15_root, BASELINE_BUDGET, stage)
    reference = load_stage(a15_root, reference_budget, stage)
    merged = candidate.merge(reference, on=PAIR_KEYS, how="inner", validate="one_to_one", suffixes=("_budget05", f"_budget{reference_budget:02d}"))
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise RuntimeError(f"A15 paired-cell keys do not align for budget 5 vs {reference_budget}, stage={stage}")
    for metric, source_column in METRICS.items():
        candidate_column = f"{source_column}_budget05"
        reference_column = f"{source_column}_budget{reference_budget:02d}"
        merged[f"{metric}_relative_delta_5_minus_{reference_budget}"] = (
            merged[candidate_column] - merged[reference_column]
        ) / merged[reference_column]
        merged[f"{metric}_budget05_win"] = merged[candidate_column] < merged[reference_column]
    merged.insert(0, "comparison", f"budget05_vs_budget{reference_budget:02d}")
    merged.insert(1, "stage", stage)
    return merged


def stable_seed(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32 - 1)


def hierarchical_bootstrap(values: pd.DataFrame, column: str, repetitions: int, seed: int) -> np.ndarray:
    """Resample target domains, then model seeds, then paired cells.

    Each A15 paired row remains the elementary endpoint unit.  The first two
    levels preserve the domain/model-seed dependence introduced by source
    pretraining and target adaptation.
    """
    if repetitions < 1000:
        raise ValueError("A16 requires at least 1000 bootstrap repetitions")
    groups: dict[str, list[np.ndarray]] = {}
    for domain, domain_frame in values.groupby("target_domain", sort=True):
        seed_values = []
        for _, seed_frame in domain_frame.groupby("model_seed", sort=True):
            item = seed_frame[column].to_numpy(dtype=float)
            if len(item) == 0:
                continue
            seed_values.append(item)
        if not seed_values:
            raise RuntimeError(f"no model-seed groups available for domain={domain}")
        groups[str(domain)] = seed_values
    if not groups:
        raise RuntimeError("no bootstrap groups available")

    rng = np.random.default_rng(seed)
    domains = list(groups)
    draws = np.empty(repetitions, dtype=float)
    for draw_index in range(repetitions):
        sampled_values: list[np.ndarray] = []
        for domain_index in rng.integers(0, len(domains), size=len(domains)):
            seed_groups = groups[domains[int(domain_index)]]
            for seed_index in rng.integers(0, len(seed_groups), size=len(seed_groups)):
                unit_values = seed_groups[int(seed_index)]
                sampled_values.append(unit_values[rng.integers(0, len(unit_values), size=len(unit_values))])
        draws[draw_index] = float(np.concatenate(sampled_values).mean())
    return draws


def bootstrap_summary(frame: pd.DataFrame, column: str, repetitions: int, label: str, margin: float) -> dict[str, Any]:
    values = frame[column].to_numpy(dtype=float)
    draws = hierarchical_bootstrap(frame, column, repetitions, stable_seed(EXPERIMENT_ID, label, column))
    return {
        "n_pairs": int(len(frame)),
        "relative_delta_mean": float(values.mean()),
        "relative_ci95_low": float(np.quantile(draws, 0.025)),
        "relative_ci95_high": float(np.quantile(draws, 0.975)),
        "one_sided_bootstrap_p_noninferiority": float((np.count_nonzero(draws >= margin) + 1) / (len(draws) + 1)),
        "candidate_5_win_rate": float((values < 0).mean()),
        "noninferiority_margin": float(margin),
        "noninferiority_ci_passed": bool(float(np.quantile(draws, 0.975)) <= margin),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_design": "target_domain_then_model_seed_then_paired_cell",
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1]["one_sided_bootstrap_p_noninferiority"]))
    m = len(indexed)
    previous = 0.0
    for rank, (index, row) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * float(row["one_sided_bootstrap_p_noninferiority"]))
        adjusted = max(adjusted, previous)
        rows[index]["holm_adjusted_p_noninferiority"] = float(adjusted)
        rows[index]["holm_noninferiority_passed"] = bool(adjusted <= 0.05)
        previous = adjusted


def summarize_comparison(paired: pd.DataFrame, repetitions: int, margin: float) -> list[dict[str, Any]]:
    comparison = str(paired["comparison"].iloc[0])
    stage = str(paired["stage"].iloc[0])
    scopes: Iterable[tuple[str, pd.DataFrame]] = [("ALL", paired)]
    domain_scopes = [(str(domain), frame.copy()) for domain, frame in paired.groupby("target_domain", sort=True)]
    rows: list[dict[str, Any]] = []
    for scope, frame in [*scopes, *domain_scopes]:
        for metric in METRICS:
            delta_column = f"{metric}_relative_delta_5_minus_{comparison[-2:]}"
            # comparison strings are budget05_vs_budget10/budget15; retain an
            # explicit fallback for clarity if that naming ever changes.
            if delta_column not in frame.columns:
                reference_budget = comparison.rsplit("budget", 1)[1]
                delta_column = f"{metric}_relative_delta_5_minus_{reference_budget}"
            summary = bootstrap_summary(frame, delta_column, repetitions, f"{comparison}|{stage}|{scope}", margin)
            rows.append({
                "comparison": comparison,
                "stage": stage,
                "scope": scope,
                "metric": metric,
                "relative_degradation_pct": 100.0 * summary["relative_delta_mean"],
                "relative_improvement_pct": -100.0 * summary["relative_delta_mean"],
                "relative_ci95_low": summary["relative_ci95_low"],
                "relative_ci95_high": summary["relative_ci95_high"],
                "noninferiority_margin_pct": 100.0 * margin,
                "n_pairs": summary["n_pairs"],
                "candidate_5_win_rate": summary["candidate_5_win_rate"],
                "one_sided_bootstrap_p_noninferiority": summary["one_sided_bootstrap_p_noninferiority"],
                "noninferiority_ci_passed": summary["noninferiority_ci_passed"],
                "bootstrap_repetitions": summary["bootstrap_repetitions"],
                "bootstrap_design": summary["bootstrap_design"],
            })
    return rows


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1000:
        raise ValueError("--bootstrap-repetitions must be at least 1000")
    if not 0.0 < float(args.noninferiority_margin_pct) < 100.0:
        raise ValueError("--noninferiority-margin-pct must be between 0 and 100")

    a15_root = resolve_a15_root(args.a15_output_dir)
    output = args.output_dir.expanduser().resolve()
    margin = float(args.noninferiority_margin_pct) / 100.0
    output.mkdir(parents=True, exist_ok=True)

    integrity = input_integrity(a15_root)
    dry_run = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION,
        "a15_output_dir": str(a15_root),
        "output_dir": str(output),
        "baseline_budget_epochs": BASELINE_BUDGET,
        "reference_budget_epochs": list(REFERENCE_BUDGETS),
        "expected_pairs_per_stage": 2500,
        "stages": list(STAGE_FILES),
        "metrics": list(METRICS),
        "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "input_integrity_passed": True,
    }
    atomic_json(output / "experimentA16_dry_run.json", dry_run)
    atomic_json(output / "experimentA16_input_integrity.json", integrity)
    print(json.dumps(dry_run, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    print("[A16] loading paired A15 results and comparing 5 epochs with 10/15 epochs...", flush=True)
    paired_paths: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    all_primary_rows: list[dict[str, Any]] = []
    for reference_budget in REFERENCE_BUDGETS:
        for stage in STAGE_FILES:
            paired = build_paired_comparison(a15_root, reference_budget, stage)
            paired_path = output / f"experimentA16_{stage}_paired_budget05_vs_budget{reference_budget:02d}.csv"
            paired.to_csv(paired_path, index=False)
            paired_paths.append(str(paired_path))
            print(f"[A16] bootstrapping {stage}: budget 5 vs {reference_budget} ({len(paired)} paired cells)", flush=True)
            rows = summarize_comparison(paired, int(args.bootstrap_repetitions), margin)
            summary_rows.extend(rows)
            all_primary_rows.extend([row for row in rows if row["scope"] == "ALL"])

    # Holm is pre-registered for the twelve primary full/stage-by-metric tests.
    holm_adjust(all_primary_rows)
    primary_index = {(row["comparison"], row["stage"], row["scope"], row["metric"]): row for row in all_primary_rows}
    for row in summary_rows:
        key = (row["comparison"], row["stage"], row["scope"], row["metric"])
        if key in primary_index:
            row["holm_adjusted_p_noninferiority"] = primary_index[key]["holm_adjusted_p_noninferiority"]
            row["holm_noninferiority_passed"] = primary_index[key]["holm_noninferiority_passed"]
        else:
            row["holm_adjusted_p_noninferiority"] = np.nan
            row["holm_noninferiority_passed"] = np.nan

    summary = pd.DataFrame(summary_rows).sort_values(["comparison", "stage", "scope", "metric"]).reset_index(drop=True)
    summary.to_csv(output / "experimentA16_budget_parsimony_summary.csv", index=False)

    all_primary_passed = bool(all(
        bool(row["noninferiority_ci_passed"]) and bool(row["holm_noninferiority_passed"])
        for row in all_primary_rows
    ))
    per_reference: dict[str, Any] = {}
    for reference_budget in REFERENCE_BUDGETS:
        subset = [row for row in all_primary_rows if row["comparison"] == f"budget05_vs_budget{reference_budget:02d}"]
        per_reference[str(reference_budget)] = {
            "n_primary_checks": len(subset),
            "all_primary_checks_passed": bool(all(
                bool(row["noninferiority_ci_passed"]) and bool(row["holm_noninferiority_passed"])
                for row in subset
            )),
            "checks": subset,
        }

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION,
        "a15_output_dir": str(a15_root),
        "baseline_budget_epochs": BASELINE_BUDGET,
        "reference_budget_epochs": list(REFERENCE_BUDGETS),
        "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "input_integrity_file": "experimentA16_input_integrity.json",
        "paired_files": paired_paths,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(output / "experimentA16_manifest.json", manifest)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "complete": True,
        "quick_mode": False,
        "new_predictor_training": False,
        "reference": "separately_selection_fitted_A9_blend_at_10_or_15_target_epochs",
        "candidate": "separately_selection_fitted_A9_blend_at_5_target_epochs",
        "baseline_budget_epochs": BASELINE_BUDGET,
        "reference_budget_epochs": list(REFERENCE_BUDGETS),
        "expected_primary_checks": 12,
        "completed_primary_checks": len(all_primary_rows),
        "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "per_reference": per_reference,
        "passed": all_primary_passed,
        "reason": (
            "A16 confirmed that the 5-epoch A9 blend is non-inferior to both 10- and 15-epoch A9 blends across all registered full/stage NASA/RMSE checks"
            if all_primary_passed
            else "A16 completed, but the 5-epoch A9 blend did not meet every registered non-inferiority check"
        ),
        "next_action": (
            "report_A9_five_epoch_deployment_parsimony_without_additional_official_test_tuning"
            if all_primary_passed
            else "retain_A9_10_epoch_locked_deployment_policy_and_report_budget_scope_boundary"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(output / "experimentA16_confirmation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
