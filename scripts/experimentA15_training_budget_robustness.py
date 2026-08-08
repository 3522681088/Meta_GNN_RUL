"""Experiment A15: training-budget robustness of the A9 locked safety blend.

Why this experiment
-------------------
Experiments A10--A14_1 established a scope boundary for source-availability
remediation and risk warning.  They do not change the successful A9/A9_1
finding: a selection-only bounded baseline/cycle-age blend improved official
C-MAPSS endpoints.  A15 changes direction and tests whether the *training-only*
A9 result is robust to a practically important training decision: the fixed
target-adaptation epoch budget.

Protocol
--------
* Three registered target-adaptation budgets: 5, 10 and 15 epochs.
* Fresh source pretraining for baseline and cycle-age representations at every
  budget, domain and model seed.  No checkpoint is shared across budgets.
* New model seeds 110--114 and new, disjoint selection/confirmation endpoint
  seeds.  The A2_1 training-only role protocol remains fixed.
* Within each budget, choose the A9 convex blend only from selection engines;
  confirmation endpoints are never used to select alpha.
* Compare each selected blend with the same-budget baseline representation.
* Success requires every budget to show one strict full-endpoint improvement
  (NASA or RMSE) and <= 3% upper-CI noninferiority for NASA and RMSE in both
  true-RUL stages.

This is a training-only robustness experiment.  It never reads official test
files, and it does not trigger another official-test confirmation.

Run from repository root:

    python -m py_compile scripts/experimentA15_training_budget_robustness.py
    nohup python -u scripts/experimentA15_training_budget_robustness.py \
      > experimentA15_training.log 2>&1 &

Resume an interrupted run:

    nohup python -u scripts/experimentA15_training_budget_robustness.py \
      --resume > experimentA15_resume.log 2>&1 &
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402


SCRIPT_VERSION = "experimentA15_training_budget_robustness_v1"
EXPERIMENT_ID = "experimentA15"
DEFAULT_OUTPUT = "outputs/experimentA15_training_budget_robustness"
BUDGETS = (5, 10, 15)
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (110, 111, 112, 113, 114)
TARGET_SPLIT_SEEDS = (6401, 6402, 6403, 6404, 6405)
ROLE_PARTITIONS = (1, 2, 3, 4, 5)
SELECTION_ENDPOINT_SEEDS = (9401, 9402, 9403, 9404, 9405)
CONFIRMATION_ENDPOINT_SEEDS = (9501, 9502, 9503, 9504, 9505)
HIGH_RUL_THRESHOLD = 60.0
STAGE_MARGIN_PCT = 3.0
QUESTION = (
    "Does the selection-only bounded baseline/cycle-age blend retain "
    "full-endpoint efficacy and true-stage safety across fixed target "
    "adaptation budgets of 5, 10 and 15 epochs?"
)

# A9 imports A8 and sets A8's wrapper globals during import.  Keep the original
# config/worker functions, then install this wrapper's budget-aware settings.
_A8_LOAD_CONFIG = a8.load_config
_A8_WORKER_COMMAND = a8.worker_command


def parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--budgets must be comma-separated integers") from exc
    if budgets != BUDGETS:
        raise argparse.ArgumentTypeError(
            "A15 budgets are preregistered and locked to 5,10,15"
        )
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A15 A9 training-budget robustness")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,2,4")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Required because the inherited A8 worker-command helper expects the
    # attribute.  A15 deliberately rejects quick mode: its registered claim
    # requires all budgets, domains, seeds and endpoint assignments.
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--budgets",
        type=parse_budgets,
        default=BUDGETS,
        help="registered budgets only: 5,10,15",
    )
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--budget", type=int, choices=BUDGETS, help=argparse.SUPPRESS)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A15 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(args: argparse.Namespace) -> Path:
    return Path(a1.resolve_path(args.output_dir or DEFAULT_OUTPUT))


def budget_id(budget: int) -> str:
    return f"{EXPERIMENT_ID}_budget{budget:02d}"


def budget_output(root: Path, budget: int) -> Path:
    return root / f"budget_{budget:02d}_epochs"


def scoped_args(args: argparse.Namespace, output: Path, budget: int) -> argparse.Namespace:
    values = dict(vars(args))
    values.update({"output_dir": str(output), "budget": int(budget)})
    return argparse.Namespace(**values)


def configure_budget_globals(budget: int, output: Path) -> None:
    """Set A8/A9 wrapper globals before either parent or worker execution."""
    identifier = budget_id(budget)
    a8.__file__ = str(Path(__file__).resolve())
    a8.SCRIPT_VERSION = f"{SCRIPT_VERSION}_{budget:02d}epochs"
    a8.EXPERIMENT_ID = identifier
    a8.DEFAULT_OUTPUT = str(output)
    a8.MODEL_SEEDS = list(MODEL_SEEDS)
    a8.TARGET_SPLIT_SEEDS = list(TARGET_SPLIT_SEEDS)
    a8.ROLE_PARTITIONS = list(ROLE_PARTITIONS)
    a8.SELECTION_ENDPOINT_SEEDS = list(SELECTION_ENDPOINT_SEEDS)
    a8.CONFIRMATION_ENDPOINT_SEEDS = list(CONFIRMATION_ENDPOINT_SEEDS)

    a9.EXPERIMENT_ID = identifier
    a9.DEFAULT_OUTPUT = str(output)
    a9.MODEL_SEEDS = list(MODEL_SEEDS)
    a9.TARGET_SPLIT_SEEDS = list(TARGET_SPLIT_SEEDS)
    a9.ROLE_PARTITIONS = list(ROLE_PARTITIONS)
    a9.SELECTION_ENDPOINT_SEEDS = list(SELECTION_ENDPOINT_SEEDS)
    a9.CONFIRMATION_ENDPOINT_SEEDS = list(CONFIRMATION_ENDPOINT_SEEDS)
    a9.QUESTION = (
        f"Does the selection-only A9 blend retain efficacy and stage safety "
        f"at a fixed target-adaptation budget of {budget} epochs?"
    )


def load_budget_config(args: argparse.Namespace, budget: int, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_budget_globals(budget, output)
    local_args = scoped_args(args, output, budget)
    base, experiment = _A8_LOAD_CONFIG(local_args)
    base = deepcopy(base)
    experiment = deepcopy(experiment)
    base.update({"output_dir": str(output), "target_epochs": int(budget)})
    experiment.update(
        {
            "experiment_id": budget_id(budget),
            "experiment_name": "a9_training_budget_robustness",
            "domains": list(DOMAINS),
            "model_seeds": list(MODEL_SEEDS),
            "target_split_seeds": list(TARGET_SPLIT_SEEDS),
            "role_partitions": list(ROLE_PARTITIONS),
            "selection_endpoint_seeds": list(SELECTION_ENDPOINT_SEEDS),
            "confirmation_endpoint_seeds": list(CONFIRMATION_ENDPOINT_SEEDS),
            "target_epochs": int(budget),
            "fixed_budget_no_epoch_selection": True,
            "fresh_source_pretraining_for_both_representations": True,
            "high_rul_threshold": float(HIGH_RUL_THRESHOLD),
            "stage_noninferiority_margin_pct": float(STAGE_MARGIN_PCT),
            "output_dir": str(output),
            "quick_mode": False,
            "a15_budget_epochs": int(budget),
        }
    )
    if tuple(experiment["selection_endpoint_seeds"]) == tuple(experiment["confirmation_endpoint_seeds"]):
        raise RuntimeError("A15 selection and confirmation endpoint seeds must differ")
    return base, experiment


def budget_worker_command(
    args: argparse.Namespace,
    domain: str,
    seed: int,
    device: str,
    output: Path,
) -> list[str]:
    command = _A8_WORKER_COMMAND(args, domain, seed, device, output)
    if args.budget is None:
        raise RuntimeError("A15 worker command requires a registered budget")
    command.extend(["--budget", str(int(args.budget))])
    return command


# a8.run_workers resolves a8.worker_command at runtime, so all child workers
# receive the same budget and reconstruct the same config as their parent.
a8.worker_command = budget_worker_command


def register_root_manifest(root: Path, args: argparse.Namespace) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"{EXPERIMENT_ID}_manifest.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "registered_primary_question": QUESTION,
        "budgets": list(BUDGETS),
        "model_seeds": list(MODEL_SEEDS),
        "target_split_seeds": list(TARGET_SPLIT_SEEDS),
        "role_partitions": list(ROLE_PARTITIONS),
        "selection_endpoint_seeds": list(SELECTION_ENDPOINT_SEEDS),
        "confirmation_endpoint_seeds": list(CONFIRMATION_ENDPOINT_SEEDS),
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "fresh_source_pretraining_per_budget_and_representation": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        for key in ("script_hash", "budgets", "model_seeds", "target_split_seeds", "selection_endpoint_seeds", "confirmation_endpoint_seeds"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A15 output is incompatible at {key}; use a new output directory")
    atomic_json(manifest_path, manifest)


def one_budget_dry_run(args: argparse.Namespace, root: Path, budget: int) -> dict[str, Any]:
    output = budget_output(root, budget)
    base, experiment = load_budget_config(args, budget, output)
    a8.validate_config(base, experiment)
    # A8's dry-run validates data, feature causality and a forward shape check.
    local_args = scoped_args(args, output, budget)
    a8.parent_main(local_args, base, experiment)
    return {
        "budget_epochs": budget,
        "output_dir": str(output),
        "expected_training_cells": 2 * len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS),
        "expected_primary_pairs": len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS) * len(CONFIRMATION_ENDPOINT_SEEDS),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def run_one_budget(args: argparse.Namespace, root: Path, budget: int) -> dict[str, Any]:
    output = budget_output(root, budget)
    base, experiment = load_budget_config(args, budget, output)
    a8.validate_config(base, experiment)
    local_args = scoped_args(args, output, budget)
    a8.parent_main(local_args, base, experiment)
    a9.augment_a9(local_args, base, experiment)
    decision = read_json(output / f"{budget_id(budget)}_confirmation_decision.json")
    if decision.get("official_test_files_accessed") or decision.get("official_test_forward_run"):
        raise RuntimeError("A15 training-only budget run accessed official test files")
    return decision


def compact_budget_row(decision: dict[str, Any], budget: int) -> dict[str, Any]:
    full = decision["full_endpoint_result"]
    high = decision["high_rul_safety_result"]
    low = decision["low_rul_safety_result"]
    return {
        "budget_epochs": int(budget),
        "complete": bool(decision.get("complete")),
        "passed": bool(decision.get("passed")),
        "completed_training_cells": int(decision.get("completed_training_cells", 0)),
        "completed_primary_pairs": int(decision.get("completed_primary_pairs", 0)),
        "full_nasa_improvement_pct": float(full["nasa_improvement_pct"]),
        "full_nasa_ci95_high": float(full["nasa_relative_ci95"][1]),
        "full_rmse_degradation_pct": float(full["rmse_degradation_pct"]),
        "full_rmse_ci95_high": float(full["rmse_relative_ci95"][1]),
        "high_nasa_ci95_high": float(high["nasa_relative_ci95"][1]),
        "high_rmse_ci95_high": float(high["rmse_relative_ci95"][1]),
        "low_nasa_ci95_high": float(low["nasa_relative_ci95"][1]),
        "low_rmse_ci95_high": float(low["rmse_relative_ci95"][1]),
        "full_endpoint_strict_improvement": bool(full["at_least_one_metric_strictly_improved"]),
        "high_rul_noninferiority": bool(high["noninferiority_passed"]),
        "low_rul_noninferiority": bool(low["noninferiority_passed"]),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def write_root_decision(root: Path, decisions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = [compact_budget_row(decisions[budget], budget) for budget in BUDGETS]
    summary = pd.DataFrame(rows).sort_values("budget_epochs")
    a1.atomic_write_text(root / f"{EXPERIMENT_ID}_budget_summary.csv", summary.to_csv(index=False))
    expected_training = 2 * len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(BUDGETS)
    expected_pairs = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS) * len(CONFIRMATION_ENDPOINT_SEEDS) * len(BUDGETS)
    complete = bool(summary["complete"].all())
    passed = bool(complete and summary["passed"].all())
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "complete": complete,
        "quick_mode": False,
        "budgets": list(BUDGETS),
        "expected_training_cells": expected_training,
        "completed_training_cells": int(summary["completed_training_cells"].sum()),
        "expected_primary_pairs": expected_pairs,
        "completed_primary_pairs": int(summary["completed_primary_pairs"].sum()),
        "per_budget_pass": {str(budget): bool(decisions[budget].get("passed")) for budget in BUDGETS},
        "fresh_source_pretraining_per_budget_and_representation": True,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": passed,
        "reason": (
            "A15 confirmed A9 blend robustness across all registered training budgets"
            if passed
            else "A15 completed, but the A9 blend did not meet every registered efficacy/safety criterion at every training budget"
        ),
        "next_action": (
            "report_A9_training_budget_robustness_without_additional_official_test_tuning"
            if passed
            else "report_A9_budget_scope_boundary_without_post_hoc_budget_selection"
        ),
    }
    atomic_json(root / f"{EXPERIMENT_ID}_confirmation_decision.json", decision)
    return decision


def main() -> None:
    args = parse_args()
    if args.quick:
        raise ValueError("A15 quick mode is disabled; run the registered full protocol")
    root = output_root(args)
    register_root_manifest(root, args)

    if args.worker_domain is not None:
        if args.worker_seed is None or args.budget is None:
            raise ValueError("A15 worker requires --worker-domain, --worker-seed and --budget")
        output = budget_output(root, int(args.budget))
        base, experiment = load_budget_config(args, int(args.budget), output)
        a8.validate_config(base, experiment)
        a8.worker_main(args, base, experiment)
        return

    if args.dry_run:
        plans = [one_budget_dry_run(args, root, budget) for budget in args.budgets]
        dry = {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": QUESTION,
            "budgets": list(args.budgets),
            "budget_plans": plans,
            "expected_training_cells": sum(item["expected_training_cells"] for item in plans),
            "expected_primary_pairs": sum(item["expected_primary_pairs"] for item in plans),
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(root / f"{EXPERIMENT_ID}_dry_run.json", dry)
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return

    decisions: dict[int, dict[str, Any]] = {}
    for budget in args.budgets:
        print(f"[A15] starting registered target-adaptation budget={budget} epochs")
        decisions[budget] = run_one_budget(args, root, budget)
        print(f"[A15] completed registered target-adaptation budget={budget} epochs")
    decision = write_root_decision(root, decisions)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
