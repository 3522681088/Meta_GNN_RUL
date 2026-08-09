"""Experiment A17: direct 10-vs-15 epoch A9 budget-parsimony confirmation.

CPU-only, no-retraining follow-up to A15/A16.  It compares the complete,
independently selection-fitted A9 blend at 10 target-adaptation epochs
(candidate) against the analogous 15-epoch policy (reference) on exactly the
same A15 confirmation cells.  It never accesses official C-MAPSS test files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "experimentA17_ten_vs_fifteen_budget_parsimony_confirmation_v1"
EXPERIMENT_ID = "experimentA17"
QUESTION = (
    "Does the complete selection-only A9 blend at ten target-adaptation epochs "
    "remain non-inferior to the independently selection-fitted fifteen-epoch "
    "A9 blend for full endpoint NASA/RMSE and true-stage safety?"
)
CANDIDATE_BUDGET = 10
REFERENCE_BUDGET = 15
MARGIN = 0.03
DEFAULT_REPETITIONS = 5000
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a15-output-dir", type=Path, default=None,
                        help="A15 root containing budget_10_epochs and budget_15_epochs.")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs" / "experimentA17_ten_vs_fifteen_budget_parsimony_confirmation")
    parser.add_argument("--noninferiority-margin-pct", type=float, default=3.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_a15_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    root = project_root() / "outputs"
    candidates = [
        root / "experimentA15_training_budget_robustness_v3",
        root / "experimentA15_training_budget_robustness_v2",
        root / "experimentA15_training_budget_robustness_gpu_safe",
        root / "experimentA15_training_budget_robustness_fixed",
        root / "experimentA15_training_budget_robustness",
    ]
    for candidate in candidates:
        if (candidate / "budget_10_epochs").is_dir() and (candidate / "budget_15_epochs").is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A17 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def budget_dir(root: Path, budget: int) -> Path:
    return root / f"budget_{budget:02d}_epochs"


def prefix(budget: int) -> str:
    return f"experimentA15_budget{budget:02d}"


def stage_path(root: Path, budget: int, stage: str) -> Path:
    return budget_dir(root, budget) / f"{prefix(budget)}_{STAGE_FILES[stage]}.csv"


def verify_inputs(a15_root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    budget_status: dict[str, Any] = {}
    root_decision_path = a15_root / "experimentA15_confirmation_decision.json"
    root_decision = read_json(root_decision_path) if root_decision_path.is_file() else {"available": False}
    if root_decision_path.is_file():
        hashes[root_decision_path.name] = sha256_file(root_decision_path)
        if not bool(root_decision.get("complete")) or not bool(root_decision.get("passed")):
            raise RuntimeError("A17 requires a complete, passing A15 root decision")
        if bool(root_decision.get("official_test_files_accessed")) or bool(root_decision.get("official_test_forward_run")):
            raise RuntimeError("A17 cannot consume an A15 root result that accessed official tests")
    for budget in (CANDIDATE_BUDGET, REFERENCE_BUDGET):
        decision_path = budget_dir(a15_root, budget) / f"{prefix(budget)}_confirmation_decision.json"
        decision = read_json(decision_path)
        hashes[str(decision_path.relative_to(a15_root))] = sha256_file(decision_path)
        if not bool(decision.get("complete")) or not bool(decision.get("passed")):
            raise RuntimeError(f"A17 requires complete, passing A15 budget={budget} inputs")
        if bool(decision.get("official_test_files_accessed")) or bool(decision.get("official_test_forward_run")):
            raise RuntimeError(f"A17 cannot consume budget={budget} output that accessed official tests")
        if int(decision.get("completed_primary_pairs", -1)) != 2500:
            raise RuntimeError(f"A15 budget={budget} has incomplete primary pairs")
        budget_status[str(budget)] = {
            "complete": bool(decision["complete"]), "passed": bool(decision["passed"]),
            "completed_primary_pairs": int(decision["completed_primary_pairs"]),
        }
        for stage in STAGE_FILES:
            path = stage_path(a15_root, budget, stage)
            if not path.is_file():
                raise FileNotFoundError(f"missing A15 stage input: {path}")
            hashes[str(path.relative_to(a15_root))] = sha256_file(path)
    return {
        "a15_root": str(a15_root), "root_decision": root_decision,
        "budget_status": budget_status, "input_hashes": hashes,
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }


def load_stage(a15_root: Path, budget: int, stage: str) -> pd.DataFrame:
    path = stage_path(a15_root, budget, stage)
    frame = pd.read_csv(path)
    required = [*PAIR_KEYS, *METRICS.values()]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")
    if len(frame) != 2500 or frame.duplicated(PAIR_KEYS).any() or frame[PAIR_KEYS].isna().any().any():
        raise RuntimeError(f"{path} does not contain 2500 unique complete paired cells")
    output = frame[required].copy()
    for column in METRICS.values():
        output[column] = pd.to_numeric(output[column], errors="raise")
        values = output[column].to_numpy(dtype=float)
        if (~np.isfinite(values)).any() or (values <= 0).any():
            raise RuntimeError(f"{path} has non-positive/non-finite values in {column}")
    return output.sort_values(PAIR_KEYS).reset_index(drop=True)


def paired_stage(a15_root: Path, stage: str) -> pd.DataFrame:
    candidate = load_stage(a15_root, CANDIDATE_BUDGET, stage)
    reference = load_stage(a15_root, REFERENCE_BUDGET, stage)
    paired = candidate.merge(reference, on=PAIR_KEYS, how="inner", validate="one_to_one",
                             suffixes=("_budget10", "_budget15"))
    if len(paired) != 2500:
        raise RuntimeError(f"budget 10/15 cells do not align for stage={stage}")
    for metric, column in METRICS.items():
        candidate_column = f"{column}_budget10"
        reference_column = f"{column}_budget15"
        paired[f"{metric}_relative_delta_10_minus_15"] = (
            paired[candidate_column] - paired[reference_column]
        ) / paired[reference_column]
        paired[f"{metric}_budget10_win"] = paired[candidate_column] < paired[reference_column]
    paired.insert(0, "comparison", "budget10_vs_budget15")
    paired.insert(1, "stage", stage)
    return paired


def seed_for(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16) % (2**32 - 1)


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> np.ndarray:
    """Bootstrap target domains, then model seeds, then paired endpoint cells."""
    groups: dict[str, list[np.ndarray]] = {}
    for domain, domain_frame in frame.groupby("target_domain", sort=True):
        seed_groups = [item[column].to_numpy(dtype=float) for _, item in domain_frame.groupby("model_seed", sort=True)]
        if not seed_groups or any(len(values) == 0 for values in seed_groups):
            raise RuntimeError(f"invalid bootstrap groups in target domain {domain}")
        groups[str(domain)] = seed_groups
    rng = np.random.default_rng(seed)
    domains = list(groups)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled: list[np.ndarray] = []
        for domain_index in rng.integers(0, len(domains), size=len(domains)):
            model_groups = groups[domains[int(domain_index)]]
            for model_index in rng.integers(0, len(model_groups), size=len(model_groups)):
                values = model_groups[int(model_index)]
                sampled.append(values[rng.integers(0, len(values), size=len(values))])
        draws[index] = float(np.concatenate(sampled).mean())
    return draws


def summarize(frame: pd.DataFrame, stage: str, scope: str, metric: str, repetitions: int, margin: float) -> dict[str, Any]:
    column = f"{metric}_relative_delta_10_minus_15"
    values = frame[column].to_numpy(dtype=float)
    draws = hierarchical_bootstrap(frame, column, repetitions, seed_for(EXPERIMENT_ID, stage, scope, metric))
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    p_value = float((np.count_nonzero(draws >= margin) + 1) / (len(draws) + 1))
    return {
        "comparison": "budget10_vs_budget15", "stage": stage, "scope": scope, "metric": metric,
        "relative_degradation_pct": float(100.0 * values.mean()),
        "relative_improvement_pct": float(-100.0 * values.mean()),
        "relative_ci95_low": float(ci_low), "relative_ci95_high": float(ci_high),
        "noninferiority_margin_pct": float(100.0 * margin), "n_pairs": int(len(frame)),
        "candidate_10_win_rate": float((values < 0).mean()),
        "one_sided_bootstrap_p_noninferiority": p_value,
        "noninferiority_ci_passed": bool(ci_high <= margin),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_design": "target_domain_then_model_seed_then_paired_cell",
    }


def holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda pair: float(pair[1]["one_sided_bootstrap_p_noninferiority"]))
    previous = 0.0
    total = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(row["one_sided_bootstrap_p_noninferiority"]))
        adjusted = max(previous, adjusted)
        rows[index]["holm_adjusted_p_noninferiority"] = float(adjusted)
        rows[index]["holm_noninferiority_passed"] = bool(adjusted <= 0.05)
        previous = adjusted


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1000:
        raise ValueError("--bootstrap-repetitions must be at least 1000")
    if not 0.0 < args.noninferiority_margin_pct < 100.0:
        raise ValueError("--noninferiority-margin-pct must be between 0 and 100")
    a15_root = resolve_a15_root(args.a15_output_dir)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    margin = args.noninferiority_margin_pct / 100.0

    integrity = verify_inputs(a15_root)
    dry_run = {
        "experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION, "a15_output_dir": str(a15_root), "output_dir": str(output),
        "candidate_budget_epochs": CANDIDATE_BUDGET, "reference_budget_epochs": REFERENCE_BUDGET,
        "expected_pairs_per_stage": 2500, "stages": list(STAGE_FILES), "metrics": list(METRICS),
        "expected_primary_checks": 6, "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions), "new_predictor_training": False,
        "official_test_files_accessed": False, "official_test_forward_run": False,
        "input_integrity_passed": True,
    }
    atomic_json(output / "experimentA17_dry_run.json", dry_run)
    atomic_json(output / "experimentA17_input_integrity.json", integrity)
    print(json.dumps(dry_run, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    print("[A17] directly comparing A15 budget=10 against budget=15 ...", flush=True)
    summary_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    paired_files: list[str] = []
    for stage in STAGE_FILES:
        paired = paired_stage(a15_root, stage)
        paired_path = output / f"experimentA17_{stage}_paired_budget10_vs_budget15.csv"
        paired.to_csv(paired_path, index=False)
        paired_files.append(str(paired_path))
        print(f"[A17] bootstrapping {stage}: {len(paired)} paired cells", flush=True)
        scopes = [("ALL", paired)] + [(str(domain), frame.copy()) for domain, frame in paired.groupby("target_domain", sort=True)]
        for scope, frame in scopes:
            for metric in METRICS:
                row = summarize(frame, stage, scope, metric, args.bootstrap_repetitions, margin)
                summary_rows.append(row)
                if scope == "ALL":
                    primary_rows.append(row)

    holm(primary_rows)
    primary_map = {(row["stage"], row["metric"]): row for row in primary_rows}
    for row in summary_rows:
        if row["scope"] == "ALL":
            primary = primary_map[(row["stage"], row["metric"])]
            row["holm_adjusted_p_noninferiority"] = primary["holm_adjusted_p_noninferiority"]
            row["holm_noninferiority_passed"] = primary["holm_noninferiority_passed"]
        else:
            row["holm_adjusted_p_noninferiority"] = np.nan
            row["holm_noninferiority_passed"] = np.nan
    summary = pd.DataFrame(summary_rows).sort_values(["stage", "scope", "metric"]).reset_index(drop=True)
    summary.to_csv(output / "experimentA17_budget_parsimony_summary.csv", index=False)

    passed = bool(all(row["noninferiority_ci_passed"] and row["holm_noninferiority_passed"] for row in primary_rows))
    manifest = {
        "experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION, "a15_output_dir": str(a15_root),
        "candidate_budget_epochs": CANDIDATE_BUDGET, "reference_budget_epochs": REFERENCE_BUDGET,
        "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions), "paired_files": paired_files,
        "new_predictor_training": False, "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(output / "experimentA17_manifest.json", manifest)
    decision = {
        "experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION,
        "complete": True, "quick_mode": False, "new_predictor_training": False,
        "reference": "separately_selection_fitted_A9_blend_at_15_target_epochs",
        "candidate": "separately_selection_fitted_A9_blend_at_10_target_epochs",
        "candidate_budget_epochs": CANDIDATE_BUDGET, "reference_budget_epochs": REFERENCE_BUDGET,
        "expected_primary_checks": 6, "completed_primary_checks": len(primary_rows),
        "noninferiority_margin_pct": float(args.noninferiority_margin_pct),
        "bootstrap_repetitions": int(args.bootstrap_repetitions), "primary_checks": primary_rows,
        "passed": passed,
        "reason": (
            "A17 confirmed that the 10-epoch A9 blend is non-inferior to the 15-epoch policy across all registered full/stage NASA/RMSE checks"
            if passed else "A17 completed, but the 10-epoch A9 blend did not meet every registered non-inferiority check"
        ),
        "next_action": (
            "lock_10_epoch_A9_training_budget_and_report_A9_final_scope"
            if passed else "report_15_epoch_training_only_budget_advantage_without_changing_the_existing_official_10_epoch_deployment_policy"
        ),
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(output / "experimentA17_confirmation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
