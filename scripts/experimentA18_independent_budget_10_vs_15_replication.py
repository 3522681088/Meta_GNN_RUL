"""Experiment A18: independent-seed replication of the A17 10-vs-15 budget signal.

Two A9 blend arms are trained from scratch under new model and endpoint seeds,
while retaining the target splits registered in the locked A2_1 protocol.  The
15-epoch arm is evaluated against the 10-epoch arm only on training-data
confirmation engines.  No official C-MAPSS test files are read or forwarded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
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


SCRIPT_VERSION = "experimentA18_independent_budget_10_vs_15_replication_v3"
EXPERIMENT_ID = "experimentA18"
DEFAULT_OUTPUT = "outputs/experimentA18_independent_budget_10_vs_15_replication_v3"
QUESTION = (
    "Under the locked A2_1 target-split protocol and independent model/endpoint "
    "seeds, does a 15-epoch selection-only A9 blend reproduce the A17 "
    "training-only advantage over its 10-epoch counterpart while preserving "
    "full and true-stage RMSE/NASA safety?"
)
DOMAINS = ["FD001", "FD002", "FD003", "FD004"]
MODEL_SEEDS = [120, 121, 122, 123, 124]
# These split seeds are fixed by the registered A2_1 protocol.  Independence
# in A18 is supplied by new model seeds and new selection/confirmation endpoints.
TARGET_SPLIT_SEEDS = [6401, 6402, 6403, 6404, 6405]
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,5,6")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=20000)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(normalise(payload), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_parent_lock(root: Path):
    """Prevent two A18 parent processes from writing the same output tree."""
    lock_path = root / "experimentA18_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"another A18 parent already owns {root}; lock owner: {owner}"
            ) from error
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


def a18_choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]]]:
    """Apply memory/utilization safety thresholds even to explicitly named GPUs."""
    inventory = a8.a2.query_gpus()
    inventory_by_index = {int(row["index"]): row for row in inventory}
    if args.gpus:
        requested = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
        missing = sorted(set(requested) - set(inventory_by_index))
        if missing:
            raise RuntimeError(f"requested GPU indices are unavailable: {missing}")
        candidates = [inventory_by_index[index] for index in requested]
    else:
        visible = a8.a2.visible_gpu_filter()
        candidates = [
            row
            for row in inventory
            if visible is None or int(row["index"]) in visible
        ]
        candidates.sort(key=lambda row: (-int(row["free_mb"]), int(row["utilization"])))
    devices = [
        int(row["index"])
        for row in candidates
        if int(row["free_mb"]) >= int(args.min_free_memory_mb)
        and int(row["utilization"]) <= int(args.max_gpu_utilization)
    ]
    if args.max_workers > 0:
        devices = devices[: int(args.max_workers)]
    return devices, inventory


# A8 calls this function at scheduling time.  The replacement is process-local
# and makes the registered A18 GPU thresholds effective for explicit GPU lists.
a8.a4.choose_gpus = a18_choose_gpus


def validate_cli_and_registration(args: argparse.Namespace) -> None:
    if args.bootstrap_repetitions < (100 if args.quick else 1000):
        minimum = 100 if args.quick else 1000
        raise ValueError(f"A18 requires at least {minimum} bootstrap repetitions")
    if args.max_workers < 0:
        raise ValueError("--max-workers must be non-negative")
    if args.min_free_memory_mb < 0:
        raise ValueError("--min-free-memory-mb must be non-negative")
    if not 0 <= args.max_gpu_utilization <= 100:
        raise ValueError("--max-gpu-utilization must be between 0 and 100")
    if args.single_process and args.device == "auto":
        raise ValueError(
            "--single-process with --device auto bypasses safe physical-GPU routing; "
            "omit --single-process or provide an explicit --device"
        )
    if args.gpus and args.device != "auto":
        raise ValueError("use either --gpus with --device auto, or an explicit --device, not both")
    if args.gpus:
        try:
            gpu_indices = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
        except ValueError as error:
            raise ValueError("--gpus must be a comma-separated list of non-negative integers") from error
        if not gpu_indices or any(index < 0 for index in gpu_indices) or len(gpu_indices) != len(set(gpu_indices)):
            raise ValueError("--gpus must contain unique non-negative GPU indices")
    registered_lists = {
        "domains": DOMAINS,
        "model_seeds": MODEL_SEEDS,
        "target_split_seeds": TARGET_SPLIT_SEEDS,
        "role_partitions": ROLE_PARTITIONS,
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS,
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS,
    }
    for name, values in registered_lists.items():
        if not values or len(values) != len(set(values)):
            raise RuntimeError(f"A18 registration has empty or duplicate {name}")
    if set(SELECTION_ENDPOINT_SEEDS) & set(CONFIRMATION_ENDPOINT_SEEDS):
        raise RuntimeError("A18 selection and confirmation endpoint seeds overlap")
    if tuple(TARGET_SPLIT_SEEDS) != (6401, 6402, 6403, 6404, 6405):
        raise RuntimeError("A18 must use the five target splits locked by A2_1")
    if tuple(BUDGETS) != (10, 15) or CANDIDATE_BUDGET not in BUDGETS or REFERENCE_BUDGET not in BUDGETS:
        raise RuntimeError("A18 budget registration is invalid")


def build_budget_config(
    args: argparse.Namespace,
    root: Path,
    budget: int,
) -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    if budget not in BUDGETS:
        raise ValueError(f"unregistered A18 budget: {budget}")
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
    if Path(base["output_dir"]).resolve() != output.resolve() or Path(experiment["output_dir"]).resolve() != output.resolve():
        raise RuntimeError(f"A18 budget={budget} output routing is inconsistent")
    if int(base["target_epochs"]) != int(experiment["target_epochs"]):
        raise RuntimeError(f"A18 budget={budget} target epoch routing is inconsistent")
    return local_args, base, experiment


def expected_counts(experiment: dict[str, Any]) -> dict[str, int]:
    training_cells = (
        len(experiment["representations"])
        * len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    primary_pairs = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
    )
    return {
        "training_cells": int(training_cells),
        "primary_pairs": int(primary_pairs),
        "confirmation_records": int(primary_pairs * 3),
    }


def protocol_preflight(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Fail before training if data, A2_1 evidence, roles, or arm routing are invalid."""
    evidence_by_budget: dict[str, Any] = {}
    protocol_hashes: dict[str, str] | None = None
    for budget in BUDGETS:
        _, base, experiment = build_budget_config(args, root, budget)
        a8.validate_config(base, experiment)
        protocols, evidence = a8.a4.load_training_only_protocol(base, experiment)
        current_hashes = {
            domain: str(protocols[domain]["protocol_hash"])
            for domain in experiment["domains"]
        }
        if protocol_hashes is None:
            protocol_hashes = current_hashes
        elif current_hashes != protocol_hashes:
            raise RuntimeError("A18 budget arms resolved different A2_1 protocols")
        for domain in experiment["domains"]:
            for split_seed in experiment["target_split_seeds"]:
                split = protocols[domain]["role_splits"].get(str(int(split_seed)))
                if not isinstance(split, dict):
                    raise RuntimeError(f"A18 preflight lacks protocol split {domain}/{split_seed}")
                partitions = split.get("partitions", {})
                for partition in experiment["role_partitions"]:
                    roles = partitions.get(str(int(partition)))
                    if not isinstance(roles, dict):
                        raise RuntimeError(
                            f"A18 preflight lacks role partition {domain}/{split_seed}/{partition}"
                        )
                    selection = set(map(int, roles.get("selection_units", [])))
                    confirmation = set(map(int, roles.get("confirmation_units", [])))
                    if not selection or not confirmation or selection & confirmation:
                        raise RuntimeError(
                            f"A18 invalid selection/confirmation roles at {domain}/{split_seed}/{partition}"
                        )
        evidence_by_budget[str(budget)] = {
            "a2_1_root": evidence["a2_1_root"],
            "a2_1_input_hashes": evidence["a2_1_input_hashes"],
            "counts": expected_counts(experiment),
            "output_dir": experiment["output_dir"],
            "target_epochs": int(experiment["target_epochs"]),
        }
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "target_split_seeds_found": TARGET_SPLIT_SEEDS,
        "protocol_hashes": protocol_hashes,
        "budget_arms": evidence_by_budget,
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA18_preflight_audit.json", audit)
    return audit


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
        "quick_mode": bool(args.quick),
        "minimum_free_gpu_memory_mb": int(args.min_free_memory_mb),
        "maximum_gpu_utilization_pct": int(args.max_gpu_utilization),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    path = root / "experimentA18_manifest.json"
    if path.is_file():
        existing = read_json(path)
        for key in (
            "script_hash",
            "budgets",
            "model_seeds",
            "target_split_seeds",
            "selection_endpoint_seeds",
            "confirmation_endpoint_seeds",
            "bootstrap_repetitions",
            "quick_mode",
        ):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A18 root output is incompatible at {key}; use a new output directory")
    atomic_json(path, manifest)
    return manifest


def arm_decision_path(root: Path, budget: int) -> Path:
    return budget_output(root, budget) / f"{budget_id(budget)}_confirmation_decision.json"


def verify_completed_arm(
    root: Path,
    budget: int,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    path = arm_decision_path(root, budget)
    decision = read_json(path)
    counts = expected_counts(experiment)
    expected = {
        "experiment_id": budget_id(budget),
        "complete": True,
        "expected_training_cells": counts["training_cells"],
        "completed_training_cells": counts["training_cells"],
        "expected_confirmation_records": counts["confirmation_records"],
        "completed_confirmation_records": counts["confirmation_records"],
        "expected_primary_pairs": counts["primary_pairs"],
        "completed_primary_pairs": counts["primary_pairs"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for key, value in expected.items():
        if decision.get(key) != value:
            raise RuntimeError(
                f"A18 budget={budget} completion check failed: {key}={decision.get(key)!r}, expected {value!r}"
            )
    manifest_path = budget_output(root, budget) / f"{budget_id(budget)}_manifest.json"
    manifest = read_json(manifest_path)
    arm_config = manifest.get("experiment_config", {})
    config_expectations = {
        "experiment_id": budget_id(budget),
        "target_epochs": int(experiment["target_epochs"]),
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
    }
    for key, value in config_expectations.items():
        if arm_config.get(key) != value:
            raise RuntimeError(
                f"A18 budget={budget} manifest mismatch at {key}; use a new output directory"
            )
    for stage, stem in STAGE_FILES.items():
        stage_file = budget_output(root, budget) / f"{budget_id(budget)}_{stem}.csv"
        if not stage_file.is_file() or stage_file.stat().st_size == 0:
            raise RuntimeError(f"A18 budget={budget} is missing {stage} output: {stage_file}")
    return decision


def run_one_budget(args: argparse.Namespace, root: Path, budget: int) -> dict[str, Any]:
    local_args, base, experiment = build_budget_config(args, root, budget)
    if local_args.resume and arm_decision_path(root, budget).is_file():
        try:
            decision = verify_completed_arm(root, budget, experiment)
        except (FileNotFoundError, RuntimeError) as error:
            print(
                f"[A18] completed-arm verification failed for budget={budget}; "
                f"rebuilding derived outputs under --resume: {error}",
                flush=True,
            )
        else:
            print(f"[A18] resume verified and skipped completed budget={budget} arm", flush=True)
            return decision
    print(f"[A18] starting independently registered target-adaptation budget={budget} epochs", flush=True)
    a8.validate_config(base, experiment)
    a8.parent_main(local_args, base, experiment)
    if local_args.dry_run:
        return {"budget": int(budget), "dry_run": True}
    a9.augment_a9(local_args, base, experiment)
    decision = verify_completed_arm(root, budget, experiment)
    print(f"[A18] completed independently registered target-adaptation budget={budget} epochs", flush=True)
    return decision


def stage_path(root: Path, budget: int, stage: str) -> Path:
    return budget_output(root, budget) / f"{budget_id(budget)}_{STAGE_FILES[stage]}.csv"


def load_stage(root: Path, budget: int, stage: str, expected_pairs: int) -> pd.DataFrame:
    path = stage_path(root, budget, stage)
    if not path.is_file():
        raise FileNotFoundError(f"A18 required budget output is missing: {path}")
    frame = pd.read_csv(path)
    required = [*PAIR_KEYS, *METRICS.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    if len(frame) != expected_pairs or frame.duplicated(PAIR_KEYS).any() or frame[PAIR_KEYS].isna().any().any():
        raise RuntimeError(f"{path} must contain exactly {expected_pairs} unique paired cells")
    output = frame[required].copy()
    for metric in METRICS.values():
        output[metric] = pd.to_numeric(output[metric], errors="raise")
        values = output[metric].to_numpy(dtype=float)
        if (~np.isfinite(values)).any() or (values <= 0).any():
            raise RuntimeError(f"{path} has invalid {metric} values")
    return output.sort_values(PAIR_KEYS).reset_index(drop=True)


def direct_pair(root: Path, stage: str, expected_pairs: int) -> pd.DataFrame:
    candidate = load_stage(root, CANDIDATE_BUDGET, stage, expected_pairs)
    reference = load_stage(root, REFERENCE_BUDGET, stage, expected_pairs)
    paired = candidate.merge(reference, on=PAIR_KEYS, how="inner", validate="one_to_one",
                             suffixes=("_budget15", "_budget10"))
    if len(paired) != expected_pairs:
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
    # Replication efficacy: full-endpoint NASA must be strictly improved.
    # All remaining full/stage metrics are registered safety checks at +3%.
    if stage == "full_endpoint" and metric == "nasa_score":
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
    pair_counts = {
        int(decision["expected_primary_pairs"])
        for decision in arm_decisions.values()
    }
    if len(pair_counts) != 1:
        raise RuntimeError("A18 budget arms have different expected primary-pair counts")
    expected_pairs = pair_counts.pop()
    summary_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    paired_files: list[str] = []
    for stage in STAGE_FILES:
        paired = direct_pair(root, stage, expected_pairs)
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
        "arms_passed": bool(arms_passed), "primary_passed": bool(primary_passed),
        "completed_primary_pairs": int(expected_pairs), "passed": passed,
    }


def main() -> None:
    args = parse_args()
    validate_cli_and_registration(args)
    if args.worker_domain is not None:
        if (
            args.worker_seed is None
            or args.budget not in BUDGETS
            or args.worker_domain not in DOMAINS
            or int(args.worker_seed) not in MODEL_SEEDS
        ):
            raise ValueError("A18 worker requires --worker-domain, --worker-seed and --budget")
        root = Path(args.output_dir).expanduser().resolve().parent
        expected_output = budget_output(root, int(args.budget)).resolve()
        if Path(args.output_dir).expanduser().resolve() != expected_output:
            raise RuntimeError("A18 worker output directory does not match its budget arm")
        local_args, base, experiment = build_budget_config(args, root, int(args.budget))
        a8.validate_config(base, experiment)
        a8.worker_main(local_args, base, experiment)
        return

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_parent_lock(root):
        manifest = root_manifest(root, args)
        preflight = protocol_preflight(args, root)
        _, _, count_experiment = build_budget_config(args, root, REFERENCE_BUDGET)
        arm_counts = expected_counts(count_experiment)
        dry_run = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "registered_primary_question": QUESTION,
            "output_dir": str(root),
            "budgets": list(BUDGETS),
            "candidate_budget_epochs": CANDIDATE_BUDGET,
            "reference_budget_epochs": REFERENCE_BUDGET,
            "domains": count_experiment["domains"],
            "model_seeds": count_experiment["model_seeds"],
            "target_split_seeds": count_experiment["target_split_seeds"],
            "role_partitions": count_experiment["role_partitions"],
            "selection_endpoint_seeds": count_experiment["selection_endpoint_seeds"],
            "confirmation_endpoint_seeds": count_experiment["confirmation_endpoint_seeds"],
            "selection_confirmation_endpoint_seeds_disjoint": True,
            "expected_training_cells": int(len(BUDGETS) * arm_counts["training_cells"]),
            "expected_confirmation_records": int(len(BUDGETS) * arm_counts["confirmation_records"]),
            "expected_primary_pairs": int(arm_counts["primary_pairs"]),
            "expected_primary_checks": 6,
            "registered_primary_criteria": {
                "full_endpoint_nasa": "strict improvement: CI95 upper < 0",
                "all_other_full_and_stage_metrics": "noninferiority: CI95 upper <= +3%",
                "multiple_testing": "Holm-adjusted one-sided bootstrap p <= 0.05",
            },
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "fresh_source_pretraining_per_budget_and_representation": True,
            "protocol_preflight_passed": bool(preflight["passed"]),
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(root / "experimentA18_dry_run.json", dry_run)
        print(json.dumps(dry_run, ensure_ascii=False, indent=2), flush=True)

        arm_decisions: dict[int, dict[str, Any]] = {}
        for budget in BUDGETS:
            arm_decisions[budget] = run_one_budget(args, root, budget)
        if args.dry_run:
            print("[A18] dry-run and protocol/model preflight completed; no training was started", flush=True)
            return

        comparison = compare_budgets(root, int(args.bootstrap_repetitions), arm_decisions)
        completed_training = sum(
            int(decision["completed_training_cells"])
            for decision in arm_decisions.values()
        )
        completed_confirmation = sum(
            int(decision["completed_confirmation_records"])
            for decision in arm_decisions.values()
        )
        complete = bool(
            completed_training == int(dry_run["expected_training_cells"])
            and completed_confirmation == int(dry_run["expected_confirmation_records"])
            and int(comparison["completed_primary_pairs"]) == int(dry_run["expected_primary_pairs"])
            and len(comparison["primary_rows"]) == int(dry_run["expected_primary_checks"])
        )
        if not complete:
            raise RuntimeError("A18 final count audit failed")
        decision = {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": QUESTION,
            "complete": True,
            "quick_mode": bool(args.quick),
            "new_predictor_training": True,
            "candidate": "independently_trained_A9_blend_at_15_target_epochs",
            "reference": "independently_trained_A9_blend_at_10_target_epochs",
            "candidate_budget_epochs": CANDIDATE_BUDGET,
            "reference_budget_epochs": REFERENCE_BUDGET,
            "expected_training_cells": int(dry_run["expected_training_cells"]),
            "completed_training_cells": completed_training,
            "expected_confirmation_records": int(dry_run["expected_confirmation_records"]),
            "completed_confirmation_records": completed_confirmation,
            "expected_primary_pairs": int(dry_run["expected_primary_pairs"]),
            "completed_primary_pairs": int(comparison["completed_primary_pairs"]),
            "expected_primary_checks": 6,
            "completed_primary_checks": len(comparison["primary_rows"]),
            "budget_arm_decisions": {str(key): value for key, value in arm_decisions.items()},
            "primary_checks": comparison["primary_rows"],
            "both_budget_arms_passed_against_baseline": bool(comparison["arms_passed"]),
            "direct_15_vs_10_primary_checks_passed": bool(comparison["primary_passed"]),
            "fresh_source_pretraining_per_budget_and_representation": True,
            "selection_confirmation_endpoint_seeds_disjoint": True,
            "protocol_preflight_passed": True,
            "passed": bool(comparison["passed"]),
            "reason": (
                "A18 independently replicated the registered 15-epoch training-only advantage over 10 epochs"
                if comparison["passed"]
                else "A18 completed, but the independent seed replication did not meet every registered efficacy/safety criterion"
            ),
            "next_action": (
                "report_replicated_15_epoch_training_only_candidate_without_reopening_the_existing_official_test"
                if comparison["passed"]
                else "retain_A9_1_official_10_epoch_policy_and_report_15_epoch_training_only_uncertainty"
            ),
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(
            root / "experimentA18_manifest.json",
            {
                **manifest,
                "preflight_audit": str(root / "experimentA18_preflight_audit.json"),
                "budget_outputs": {
                    str(key): str(budget_output(root, key)) for key in BUDGETS
                },
                "paired_files": comparison["paired_files"],
            },
        )
        atomic_json(root / "experimentA18_confirmation_decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
