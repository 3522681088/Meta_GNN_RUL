"""Experiment A18: independent-seed replication of the A17 10-vs-15 budget signal.

Two A9 blend arms are trained from scratch under fully new model, split and
endpoint seeds.  The 15-epoch arm is evaluated against the 10-epoch arm only
on training-data confirmation engines.  No official C-MAPSS test files are
read or forwarded in this experiment.
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

from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402


SCRIPT_VERSION = "experimentA18_independent_budget_10_vs_15_replication_v1"
EXPERIMENT_ID = "experimentA18"
QUESTION = (
    "On independent model/split/endpoint seeds, does a 15-epoch selection-only "
    "A9 blend reproduce the A17 training-only advantage over its 10-epoch "
    "counterpart while preserving full and true-stage RMSE/NASA safety?"
)
DOMAINS = ["FD001", "FD002", "FD003", "FD004"]
MODEL_SEEDS = [120, 121, 122, 123, 124]
TARGET_SPLIT_SEEDS = [6601, 6602, 6603, 6604, 6605]
ROLE_PARTITIONS = [1, 2, 3, 4, 5]
SELECTION_ENDPOINT_SEEDS = [9601, 9602, 9603, 9604, 9605]
CONFIRMATION_ENDPOINT_SEEDS = [9701, 9702, 9703, 9704, 9705]
BUDGETS = (10, 15)
CANDIDATE_BUDGET = 15
REFERENCE_BUDGET = 10
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="outputs/experimentA18_independent_budget_10_vs_15_replication")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,5,6")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--budget", type=int, choices=BUDGETS, help=argparse.SUPPRESS)
    return parser.parse_args()


def budget_id(budget: int) -> str:
    return f"experimentA18_budget{int(budget):02d}"


def budget_output(root: Path, budget: int) -> Path:
    return root / f"budget_{int(budget):02d}_epochs"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    def normalise(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(normalise(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A18 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_budget_modules(budget: int, output: Path) -> None:
    """Make A8 workers execute this wrapper and keep every budget isolated."""
    identifier = budget_id(budget)
    wrapper = str(Path(__file__).resolve())
    for module in (a8, a9):
        module.EXPERIMENT_ID = identifier
        module.SCRIPT_VERSION = f"{SCRIPT_VERSION}_{int(budget):02d}"
        module.DEFAULT_OUTPUT = str(output)
        module.MODEL_SEEDS = MODEL_SEEDS.copy()
        module.TARGET_SPLIT_SEEDS = TARGET_SPLIT_SEEDS.copy()
        module.ROLE_PARTITIONS = ROLE_PARTITIONS.copy()
        module.SELECTION_ENDPOINT_SEEDS = SELECTION_ENDPOINT_SEEDS.copy()
        module.CONFIRMATION_ENDPOINT_SEEDS = CONFIRMATION_ENDPOINT_SEEDS.copy()
    a8.__file__ = wrapper
    a8.DOMAINS = DOMAINS.copy()


_A8_WORKER_COMMAND = a8.worker_command


def a18_worker_command(args: argparse.Namespace, domain: str, seed: int, device: str, output: Path) -> list[str]:
    command = _A8_WORKER_COMMAND(args, domain, seed, device, output)
    if args.budget not in BUDGETS:
        raise RuntimeError("A18 worker command is missing its registered budget")
    command.extend(["--budget", str(int(args.budget))])
    return command


a8.worker_command = a18_worker_command


def build_budget_config(args: argparse.Namespace, root: Path, budget: int) -> tuple[dict[str, Any], dict[str, Any]]:
    output = budget_output(root, budget)
    configure_budget_modules(budget, output)
    local_args = deepcopy(args)
    local_args.output_dir = str(output)
    local_args.budget = int(budget)
    base, experiment = a9.load_config(local_args)
    experiment = deepcopy(experiment)
    base.update({"output_dir": str(output), "target_epochs": int(budget)})
    experiment.update(
        {
            "experiment_id": budget_id(budget),
            "experiment_name": "independent_budget_10_vs_15_replication",
            "domains": DOMAINS.copy(),
            "model_seeds": MODEL_SEEDS.copy(),
            "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
            "role_partitions": ROLE_PARTITIONS.copy(),
            "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
            "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
            "target_epochs": int(budget),
            "fixed_budget_no_epoch_selection": True,
            "fresh_source_pretraining_for_both_representations": True,
            "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "prediction_gate_threshold": 60.0,
            "selection_safety_margin_pct": 3.0,
            "selection_confirmation_endpoint_seeds_disjoint": True,
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "output_dir": str(output),
            "quick_mode": False,
            "a18_budget_epochs": int(budget),
        }
    )
    if args.quick:
        base.update({"target_epochs": 2, "source_pretrain_steps": 20})
        experiment.update(
            {
                "domains": ["FD004"], "model_seeds": [MODEL_SEEDS[0]],
                "target_split_seeds": [TARGET_SPLIT_SEEDS[0]], "role_partitions": [1],
                "selection_endpoint_seeds": [SELECTION_ENDPOINT_SEEDS[0]],
                "confirmation_endpoint_seeds": [CONFIRMATION_ENDPOINT_SEEDS[0]],
                "target_epochs": 2, "source_pretrain_steps": 20,
                "bootstrap_repetitions": 100, "quick_mode": True,
            }
        )
    if set(experiment["selection_endpoint_seeds"]) & set(experiment["confirmation_endpoint_seeds"]):
        raise RuntimeError("A18 selection and confirmation endpoint seed sets must be disjoint")
    return base, experiment


def root_manifest(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "script_hash": sha256_file(Path(__file__)),
        "registered_primary_question": QUESTION,
        "budgets": list(BUDGETS),
        "candidate_budget_epochs": CANDIDATE_BUDGET,
        "reference_budget_epochs": REFERENCE_BUDGET,
        "domains": DOMAINS,
        "model_seeds": MODEL_SEEDS,
        "target_split_seeds": TARGET_SPLIT_SEEDS,
        "role_partitions": ROLE_PARTITIONS,
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS,
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "fresh_source_pretraining_per_budget_and_representation": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    path = root / "experimentA18_manifest.json"
    if path.is_file():
        existing = read_json(path)
        for key in ("script_hash", "budgets", "model_seeds", "target_split_seeds", "selection_endpoint_seeds", "confirmation_endpoint_seeds"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A18 root output is incompatible at {key}; use a new output directory")
    atomic_json(path, manifest)
    return manifest


def run_one_budget(args: argparse.Namespace, root: Path, budget: int) -> dict[str, Any]:
    base, experiment = build_budget_config(args, root, budget)
    args.budget = int(budget)
    print(f"[A18] starting independently registered target-adaptation budget={budget} epochs", flush=True)
    a8.validate_config(base, experiment)
    a8.parent_main(args, base, experiment)
    if args.dry_run:
        return {"budget": int(budget), "dry_run": True}
    a9.augment_a9(args, base, experiment)
    decision_path = budget_output(root, budget) / f"{budget_id(budget)}_confirmation_decision.json"
    decision = read_json(decision_path)
    if not bool(decision.get("complete")):
        raise RuntimeError(f"A18 budget={budget} did not complete")
    if bool(decision.get("official_test_files_accessed")) or bool(decision.get("official_test_forward_run")):
        raise RuntimeError(f"A18 budget={budget} accessed official test data")
    print(f"[A18] completed independently registered target-adaptation budget={budget} epochs", flush=True)
    return decision


def stage_path(root: Path, budget: int, stage: str) -> Path:
    return budget_output(root, budget) / f"{budget_id(budget)}_{STAGE_FILES[stage]}.csv"


def load_stage(root: Path, budget: int, stage: str) -> pd.DataFrame:
    path = stage_path(root, budget, stage)
    if not path.is_file():
        raise FileNotFoundError(f"A18 required budget output is missing: {path}")
    frame = pd.read_csv(path)
    required = [*PAIR_KEYS, *METRICS.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    if len(frame) != 2500 or frame.duplicated(PAIR_KEYS).any() or frame[PAIR_KEYS].isna().any().any():
        raise RuntimeError(f"{path} must contain exactly 2500 unique paired cells")
    output = frame[required].copy()
    for metric in METRICS.values():
        output[metric] = pd.to_numeric(output[metric], errors="raise")
        values = output[metric].to_numpy(dtype=float)
        if (~np.isfinite(values)).any() or (values <= 0).any():
            raise RuntimeError(f"{path} has invalid {metric} values")
    return output.sort_values(PAIR_KEYS).reset_index(drop=True)


def direct_pair(root: Path, stage: str) -> pd.DataFrame:
    candidate = load_stage(root, CANDIDATE_BUDGET, stage)
    reference = load_stage(root, REFERENCE_BUDGET, stage)
    paired = candidate.merge(reference, on=PAIR_KEYS, how="inner", validate="one_to_one",
                             suffixes=("_budget15", "_budget10"))
    if len(paired) != 2500:
        raise RuntimeError(f"A18 budget pair alignment failed for stage={stage}")
    for metric, source in METRICS.items():
        candidate_column = f"{source}_budget15"
        reference_column = f"{source}_budget10"
        paired[f"{metric}_relative_delta_15_minus_10"] = (
            paired[candidate_column] - paired[reference_column]
        ) / paired[reference_column]
        paired[f"{metric}_budget15_win"] = paired[candidate_column] < paired[reference_column]
    paired.insert(0, "comparison", "budget15_vs_budget10")
    paired.insert(1, "stage", stage)
    return paired


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16) % (2**32 - 1)


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> np.ndarray:
    """Resample target domain, then model seed, then paired endpoint cell."""
    groups: dict[str, list[np.ndarray]] = {}
    for domain, domain_frame in frame.groupby("target_domain", sort=True):
        model_groups = [item[column].to_numpy(dtype=float) for _, item in domain_frame.groupby("model_seed", sort=True)]
        if not model_groups or any(len(values) == 0 for values in model_groups):
            raise RuntimeError(f"invalid bootstrap hierarchy in domain={domain}")
        groups[str(domain)] = model_groups
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


def criterion(stage: str, metric: str) -> tuple[str, float]:
    if metric == "nasa_score" and stage in {"full_endpoint", "high_rul_gt60"}:
        return "strict_improvement_ci95_upper_lt_0", 0.0
    return "noninferiority_ci95_upper_lte_3pct", MARGIN


def summarize(frame: pd.DataFrame, stage: str, scope: str, metric: str, repetitions: int) -> dict[str, Any]:
    column = f"{metric}_relative_delta_15_minus_10"
    values = frame[column].to_numpy(dtype=float)
    rule, threshold = criterion(stage, metric)
    draws = hierarchical_bootstrap(frame, column, repetitions, stable_seed(EXPERIMENT_ID, stage, scope, metric))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "comparison": "budget15_vs_budget10", "stage": stage, "scope": scope, "metric": metric,
        "relative_degradation_pct": float(100.0 * values.mean()),
        "relative_improvement_pct": float(-100.0 * values.mean()),
        "relative_ci95_low": float(low), "relative_ci95_high": float(high),
        "decision_rule": rule, "decision_threshold_pct": float(100.0 * threshold),
        "n_pairs": int(len(frame)), "candidate_15_win_rate": float((values < 0).mean()),
        "one_sided_bootstrap_p": float((np.count_nonzero(draws >= threshold) + 1) / (len(draws) + 1)),
        "criterion_ci_passed": bool(high < threshold) if threshold == 0.0 else bool(high <= threshold),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_design": "target_domain_then_model_seed_then_paired_cell",
    }


def holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda pair: float(pair[1]["one_sided_bootstrap_p"]))
    previous = 0.0
    total = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(row["one_sided_bootstrap_p"]))
        adjusted = max(previous, adjusted)
        rows[index]["holm_adjusted_p"] = float(adjusted)
        rows[index]["criterion_holm_passed"] = bool(adjusted <= 0.05)
        previous = adjusted


def compare_budgets(root: Path, repetitions: int, arm_decisions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    paired_files: list[str] = []
    for stage in STAGE_FILES:
        paired = direct_pair(root, stage)
        paired_path = root / f"experimentA18_{stage}_paired_budget15_vs_budget10.csv"
        paired.to_csv(paired_path, index=False)
        paired_files.append(str(paired_path))
        print(f"[A18] bootstrapping independent 15-vs-10 {stage}: {len(paired)} paired cells", flush=True)
        scopes = [("ALL", paired)] + [(str(domain), part.copy()) for domain, part in paired.groupby("target_domain", sort=True)]
        for scope, part in scopes:
            for metric in METRICS:
                row = summarize(part, stage, scope, metric, repetitions)
                summary_rows.append(row)
                if scope == "ALL":
                    primary_rows.append(row)
    holm(primary_rows)
    primary_map = {(row["stage"], row["metric"]): row for row in primary_rows}
    for row in summary_rows:
        if row["scope"] == "ALL":
            primary = primary_map[(row["stage"], row["metric"])]
            row["holm_adjusted_p"] = primary["holm_adjusted_p"]
            row["criterion_holm_passed"] = primary["criterion_holm_passed"]
        else:
            row["holm_adjusted_p"] = np.nan
            row["criterion_holm_passed"] = np.nan
    summary = pd.DataFrame(summary_rows).sort_values(["stage", "scope", "metric"]).reset_index(drop=True)
    summary.to_csv(root / "experimentA18_budget_replication_summary.csv", index=False)

    arms_passed = all(bool(decision.get("passed")) for decision in arm_decisions.values())
    primary_passed = all(bool(row["criterion_ci_passed"]) and bool(row["criterion_holm_passed"]) for row in primary_rows)
    passed = bool(arms_passed and primary_passed)
    return {
        "primary_rows": primary_rows, "paired_files": paired_files,
        "arms_passed": bool(arms_passed), "primary_passed": bool(primary_passed), "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1000 and not args.quick:
        raise ValueError("A18 requires at least 1000 bootstrap repetitions outside --quick mode")
    if args.worker_domain is not None:
        if args.worker_seed is None or args.budget not in BUDGETS:
            raise ValueError("A18 worker requires --worker-domain, --worker-seed and --budget")
        root = Path(args.output_dir).expanduser().resolve().parent
        base, experiment = build_budget_config(args, root, int(args.budget))
        a8.validate_config(base, experiment)
        a8.worker_main(args, base, experiment)
        return

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root_manifest(root, args)
    dry_run = {
        "experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION, "output_dir": str(root),
        "budgets": list(BUDGETS), "candidate_budget_epochs": CANDIDATE_BUDGET,
        "reference_budget_epochs": REFERENCE_BUDGET, "domains": DOMAINS,
        "model_seeds": MODEL_SEEDS, "target_split_seeds": TARGET_SPLIT_SEEDS,
        "role_partitions": ROLE_PARTITIONS,
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS,
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "expected_training_cells": 2 * len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * 2,
        "expected_confirmation_records": 2 * len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS) * len(CONFIRMATION_ENDPOINT_SEEDS) * 3,
        "expected_primary_pairs": len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS) * len(CONFIRMATION_ENDPOINT_SEEDS),
        "expected_primary_checks": 6, "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "fresh_source_pretraining_per_budget_and_representation": True,
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA18_dry_run.json", dry_run)
    print(json.dumps(dry_run, ensure_ascii=False, indent=2), flush=True)

    arm_decisions: dict[int, dict[str, Any]] = {}
    for budget in BUDGETS:
        arm_decisions[budget] = run_one_budget(args, root, budget)
    if args.dry_run:
        return

    comparison = compare_budgets(root, int(args.bootstrap_repetitions), arm_decisions)
    completed_training = sum(int(decision["completed_training_cells"]) for decision in arm_decisions.values())
    completed_confirmation = sum(int(decision["completed_confirmation_records"]) for decision in arm_decisions.values())
    decision = {
        "experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION,
        "complete": True, "quick_mode": False, "new_predictor_training": True,
        "candidate": "independently_trained_A9_blend_at_15_target_epochs",
        "reference": "independently_trained_A9_blend_at_10_target_epochs",
        "candidate_budget_epochs": CANDIDATE_BUDGET, "reference_budget_epochs": REFERENCE_BUDGET,
        "expected_training_cells": int(dry_run["expected_training_cells"]),
        "completed_training_cells": completed_training,
        "expected_confirmation_records": int(dry_run["expected_confirmation_records"]),
        "completed_confirmation_records": completed_confirmation,
        "expected_primary_pairs": int(dry_run["expected_primary_pairs"]),
        "completed_primary_pairs": int(dry_run["expected_primary_pairs"]),
        "expected_primary_checks": 6, "completed_primary_checks": len(comparison["primary_rows"]),
        "budget_arm_decisions": {str(key): value for key, value in arm_decisions.items()},
        "primary_checks": comparison["primary_rows"],
        "fresh_source_pretraining_per_budget_and_representation": True,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "passed": bool(comparison["passed"]),
        "reason": (
            "A18 independently replicated the registered 15-epoch training-only advantage over 10 epochs"
            if comparison["passed"] else "A18 completed, but the independent seed replication did not meet every registered efficacy/safety criterion"
        ),
        "next_action": (
            "report_replicated_15_epoch_training_only_candidate_without_reopening_the_existing_official_test"
            if comparison["passed"] else "retain_A9_1_official_10_epoch_policy_and_report_15_epoch_training_only_uncertainty"
        ),
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA18_manifest.json", {**manifest, "budget_outputs": {str(key): str(budget_output(root, key)) for key in BUDGETS}, "paired_files": comparison["paired_files"]})
    atomic_json(root / "experimentA18_confirmation_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
