"""Experiment A4: training-only asymmetric endpoint-risk learning.

Motivation
----------
A3 showed that balanced-endpoint *epoch selection* did not improve official
endpoint RMSE, although it sometimes reduced the asymmetric NASA risk.  A4
changes direction: it keeps the model, source checkpoint, K-shot engines,
random seeds and 10-epoch adaptation budget fixed, and changes only the target
training objective.

Registered comparison
---------------------
* ``symmetric_mse``: ordinary target-head MSE;
* ``late_weighted_mse_x2``: squared errors with prediction > RUL weighted 2x.

The positive error is an RUL overestimate and therefore corresponds to a late
maintenance decision.  Both objectives start from the same verified A2 source
state and see identically seeded target batches.  There is no target-epoch
selection: every model is evaluated after exactly 10 epochs.

The formal experiment uses FD001--FD004, model seeds 80--84, target split
seeds 6401--6405, K=5, five A2_1 role partitions and five balanced endpoint
assignments.  Final metrics use only training engines assigned the confirmation
role.  Official C-MAPSS test trajectories and RUL files are never opened.

Run from the repository root:

    python -u scripts/experimentA4_asymmetric_endpoint_risk_learning.py --dry-run

    nohup python -u scripts/experimentA4_asymmetric_endpoint_risk_learning.py \
      > experimentA4_training.log 2>&1 &

All artifacts are written below
``outputs/experimentA4_asymmetric_endpoint_risk_learning``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402


SCRIPT_VERSION = "experimentA4_asymmetric_endpoint_risk_learning_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
OBJECTIVES = ("symmetric_mse", "late_weighted_mse_x2")
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
ENDPOINT_SEEDS = list(range(7501, 7506))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
DEFAULT_OUTPUT = "outputs/experimentA4_asymmetric_endpoint_risk_learning"
DEFAULT_A2_OUTPUT = "outputs/experimentA2_endpoint_consistency_validation"
DEFAULT_A2_1_OUTPUT = "outputs/experimentA2_1_endpoint_scheme_crossfit_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A4: training-only asymmetric endpoint-risk learning"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-output-dir")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 3,4,5")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required A4 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stable_seed(*parts: Any) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    base.update(
        {
            "data_dir": resolved(args.data_dir, base["data_dir"]),
            "output_dir": resolved(args.output_dir, DEFAULT_OUTPUT),
            "normalizer_seed": 2026,
            "condition_count": 6,
            "source_pretrain_steps": 1500,
            "source_pretrain_lr": 0.001,
            "source_pretrain_weight_decay": 0.0,
            "target_epochs": 10,
            "target_lr": 0.001,
            "pair_aux_weight": 0.0,
            "device": args.device,
        }
    )
    experiment = {
        "experiment_id": "experimentA4",
        "experiment_name": "asymmetric_endpoint_risk_learning",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "objectives": list(OBJECTIVES),
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "endpoint_seeds": ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "late_error_weight": 2.0,
        "fixed_budget_no_epoch_selection": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "minimum_nasa_improvement_pct": 3.0,
        "rmse_noninferiority_margin_pct": 3.0,
        "minimum_nasa_domain_wins": 3,
        "a2_output_dir": resolved(args.a2_output_dir, DEFAULT_A2_OUTPUT),
        "a2_1_output_dir": resolved(args.a2_1_output_dir, DEFAULT_A2_1_OUTPUT),
        "output_dir": base["output_dir"],
        "quick_mode": False,
    }
    if args.quick:
        experiment.update(
            {
                "domains": ["FD004"],
                "model_seeds": [80],
                "target_split_seeds": [6401],
                "role_partitions": [1],
                "endpoint_seeds": [7501],
                "target_epochs": 2,
                "bootstrap_repetitions": 100,
                "quick_mode": True,
            }
        )
        base["target_epochs"] = 2
        if args.output_dir is None:
            base["output_dir"] = resolved(None, DEFAULT_OUTPUT + "_quick")
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if experiment["architecture"] != ARCHITECTURE:
        raise ValueError(f"A4 requires architecture={ARCHITECTURE}")
    if tuple(experiment["objectives"]) != OBJECTIVES:
        raise ValueError(f"A4 requires objectives={OBJECTIVES}")
    if float(experiment["late_error_weight"]) != 2.0:
        raise ValueError("A4 late-error weight is pre-registered at 2.0")
    for name in (
        "domains",
        "model_seeds",
        "target_split_seeds",
        "role_partitions",
        "endpoint_seeds",
    ):
        values = experiment[name]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"A4 has empty/duplicate values in {name}")
    for domain in experiment["domains"]:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def root_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA4"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "run_json": output / f"{prefix}_run_level.json",
        "run_csv": output / f"{prefix}_run_level.csv",
        "fixed_run": output / f"{prefix}_fixed_endpoint_run_level.csv",
        "predictions": output / f"{prefix}_balanced_endpoint_predictions.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_objective_cells.csv",
        "fixed_paired": output / f"{prefix}_fixed_endpoint_paired_cells.csv",
        "comparisons": output / f"{prefix}_paired_objective_comparisons.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def shard_dir(output: Path, domain: str, seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{seed:03d}"


def shard_paths(output: Path, domain: str, seed: int) -> dict[str, Path]:
    directory = shard_dir(output, domain, seed)
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "run_json": directory / "run_level.json",
        "run_csv": directory / "run_level.csv",
        "fixed_run": directory / "fixed_endpoint_run_level.csv",
        "predictions": directory / "balanced_endpoint_predictions.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def verify_protocol_hash(protocol: dict, domain: str) -> None:
    if protocol.get("target_domain") != domain:
        raise ValueError(f"A2_1 protocol target mismatch for {domain}")
    stored = protocol.get("protocol_hash")
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    if not stored or stored != a1.canonical_hash(payload):
        raise ValueError(f"A2_1 protocol hash is invalid for {domain}")


def load_training_only_protocol(base: dict, experiment: dict) -> tuple[dict, dict]:
    """Load only A2_1 training protocol/audit; never A3 or official files."""
    root = Path(experiment["a2_1_output_dir"])
    required = {
        "manifest": root / "experimentA2_1_manifest.json",
        "protocol": root / "experimentA2_1_protocol.json",
        "decision": root / "experimentA2_1_confirmation_decision.json",
    }
    manifest = read_json(required["manifest"])
    protocols = read_json(required["protocol"])
    decision = read_json(required["decision"])
    required_decision = {
        "experiment_id": "experimentA2_1",
        "expected_training_cells": 200,
        "completed_training_cells": 200,
        "complete": True,
        "quick_mode": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "endpoint_protocol_gap_confirmed": True,
        "passed": True,
    }
    for key, expected in required_decision.items():
        if decision.get(key) != expected:
            raise ValueError(f"A4 requires clean completed A2_1 evidence: {key}={expected}")
    if set(protocols) != set(DOMAINS):
        raise ValueError("A2_1 protocol must contain all four domains")
    for domain in DOMAINS:
        protocol = protocols[domain]
        verify_protocol_hash(protocol, domain)
        for hashed_domain, expected_hash in protocol["train_file_hashes"].items():
            current = a1.file_sha256(a1.train_path(base["data_dir"], hashed_domain))
            if current != expected_hash:
                raise RuntimeError(f"training file changed since A2_1: {hashed_domain}")
        for split_seed in experiment["target_split_seeds"]:
            if str(split_seed) not in protocol["role_splits"]:
                raise ValueError(f"A2_1 protocol lacks split {domain}/{split_seed}")
    evidence = {
        "a2_1_root": str(root),
        "a2_1_input_hashes": {
            name: a1.file_sha256(path) for name, path in required.items()
        },
        "a2_1_protocol_hashes": {
            domain: protocols[domain]["protocol_hash"] for domain in DOMAINS
        },
        "a2_1_script_hash": manifest.get("script_hash"),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return protocols, evidence


def target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    late_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = prediction - target
    squared = error.square()
    mse = squared.mean()
    if objective == "symmetric_mse":
        return mse, mse
    if objective == "late_weighted_mse_x2":
        weight = torch.where(error > 0, torch.full_like(error, late_weight), torch.ones_like(error))
        return (weight * squared).mean(), mse
    raise ValueError(f"unknown A4 objective: {objective}")


def train_fixed_budget(
    model: torch.nn.Module,
    support,
    pool,
    cfg: dict,
    device: torch.device,
    objective: str,
    late_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("A4 model has no predictor.* parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["target_lr"]))
    history = []
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        learner.train()
        objective_losses: list[float] = []
        mse_losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss, unweighted_mse = target_loss(
                prediction,
                y,
                objective,
                late_weight,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A4 target loss became NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            objective_losses.append(float(loss.item()))
            mse_losses.append(float(unweighted_mse.item()))
        history.append(
            {
                "epoch": epoch,
                "objective_loss": float(np.mean(objective_losses)),
                "unweighted_mse_loss": float(np.mean(mse_losses)),
            }
        )
        print(
            f"A4 objective={objective} epoch={epoch:02d}/{cfg['target_epochs']} "
            f"objective_loss={np.mean(objective_losses):.4f} "
            f"mse={np.mean(mse_losses):.4f}"
        )
    predictions = a1.predict_with_units(learner, pool, device)
    del learner
    return predictions, pd.DataFrame(history)


def endpoint_risk_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics = regression_metrics(frame["label"], frame["prediction"])
    error = frame["error"].to_numpy(dtype=float)
    positive = np.maximum(error, 0.0)
    negative = np.maximum(-error, 0.0)
    return {
        **metrics,
        "mean_error": float(np.mean(error)),
        "late_rate": float(np.mean(error > 0)),
        "late_error_q95": float(np.quantile(positive, 0.95)),
        "under_error_q95": float(np.quantile(negative, 0.95)),
    }


def annotate_predictions(frame: pd.DataFrame, common: dict) -> pd.DataFrame:
    output = frame.copy()
    for column, value in reversed(list(common.items())):
        scalar = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
        output.insert(0, column, scalar)
    return output


def evaluate_confirmation_roles(
    *,
    experiment: dict,
    protocol: dict,
    objective: str,
    model_seed: int,
    split_seed: int,
    endpoint_rows: pd.DataFrame,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    domain = protocol["target_domain"]
    split = protocol["role_splits"][str(split_seed)]
    primary_results: list[dict] = []
    fixed_results: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for partition in experiment["role_partitions"]:
        roles = split["partitions"][str(partition)]
        confirmation_units = list(map(int, roles["confirmation_units"]))
        for endpoint_seed in experiment["endpoint_seeds"]:
            assignment = a21.balanced_assignment(
                confirmation_units,
                domain,
                split_seed,
                partition,
                endpoint_seed,
                "confirmation",
            )
            selected = a21.endpoint_subset(
                endpoint_rows,
                confirmation_units,
                assignment=assignment,
            )
            common = {
                "target_domain": domain,
                "objective": objective,
                "model_seed": int(model_seed),
                "target_split_seed": int(split_seed),
                "role_partition": int(partition),
                "endpoint_seed": int(endpoint_seed),
                "evaluation_protocol": "balanced_endpoint",
            }
            primary_results.append(
                {
                    **common,
                    **endpoint_risk_metrics(selected),
                    "confirmation_units": confirmation_units,
                    "confirmation_engine_count": len(confirmation_units),
                    "confirmation_used_for_training": False,
                    "selection_units_used": False,
                    "fixed_budget_epoch": int(experiment["target_epochs"]),
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )
            prediction_parts.append(annotate_predictions(selected, common))
        for fraction in experiment["endpoint_fractions"]:
            selected = a21.endpoint_subset(
                endpoint_rows,
                confirmation_units,
                fraction=float(fraction),
            )
            fixed_results.append(
                {
                    "target_domain": domain,
                    "objective": objective,
                    "model_seed": int(model_seed),
                    "target_split_seed": int(split_seed),
                    "role_partition": int(partition),
                    "endpoint_fraction": float(fraction),
                    "evaluation_protocol": f"fixed_endpoint_{int(round(100 * fraction)):03d}",
                    **endpoint_risk_metrics(selected),
                    "confirmation_units": confirmation_units,
                    "confirmation_engine_count": len(confirmation_units),
                    "confirmation_used_for_training": False,
                    "selection_units_used": False,
                    "fixed_budget_epoch": int(experiment["target_epochs"]),
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )
    return primary_results, fixed_results, pd.concat(prediction_parts, ignore_index=True)


def training_cell_id(domain: str, objective: str, model_seed: int, split_seed: int) -> str:
    return f"experimentA4_{domain.lower()}_mseed{model_seed:03d}_tsplit{split_seed}_{objective}"


def require_verified_source_cache(
    base: dict,
    experiment: dict,
    protocol: dict,
    model_seed: int,
    prior: torch.Tensor,
) -> tuple[dict, list, dict]:
    # A2_1 deliberately disables cache reuse for its own quick experiment.
    # A4 quick mode changes only the target-training budget, so it is both safe
    # and necessary to verify/reuse the same formal A2 source state here.
    cache_experiment = dict(experiment)
    cache_experiment["quick_mode"] = False
    reused = a21.reuse_a2_source_cache(
        base,
        cache_experiment,
        protocol,
        ARCHITECTURE,
        model_seed,
        prior,
    )
    if reused is None:
        raise RuntimeError(
            "A4 requires the verified A2 source cache used by A2_1. "
            "Restore the complete A2 output directory before running A4."
        )
    state, history, inventory = reused
    if inventory.get("source_cache_origin") != "verified_experimentA2":
        raise AssertionError("A4 source-cache provenance is invalid")
    return state, history, inventory


def run_training_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    objective: str,
    model_seed: int,
    split_seed: int,
    source_state: dict,
    source_history: list,
    inventory: dict,
    prior: torch.Tensor,
) -> tuple[list[dict], list[dict], pd.DataFrame, pd.DataFrame]:
    domain = protocol["target_domain"]
    run_seed = a2.target_run_seed(domain, model_seed, split_seed)
    cfg = deepcopy(base)
    cfg.update(
        {
            "seed": run_seed,
            "target_domain": domain,
            "source_domains": protocol["source_domains"],
        }
    )
    split = protocol["role_splits"][str(split_seed)]
    support_units = list(map(int, split["adaptation_units"]))
    pool_units = list(map(int, split["evaluation_pool_units"]))
    support, pool, feature_count = a21.prepare_support_pool(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        support_units,
        pool_units,
    )
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(ARCHITECTURE, feature_count, cfg, prior, prior)
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    predictions, history = train_fixed_budget(
        model,
        support,
        pool,
        cfg,
        device,
        objective,
        float(experiment["late_error_weight"]),
    )
    endpoint_rows = a21.endpoint_epoch_rows(predictions, int(experiment["target_epochs"]))
    primary, fixed, primary_predictions = evaluate_confirmation_roles(
        experiment=experiment,
        protocol=protocol,
        objective=objective,
        model_seed=model_seed,
        split_seed=split_seed,
        endpoint_rows=endpoint_rows,
    )
    identifier = training_cell_id(domain, objective, model_seed, split_seed)
    common = {
        "experiment_id": "experimentA4",
        "cell_id": identifier,
        "model": ARCHITECTURE,
        "target_run_seed": int(run_seed),
        "k": int(experiment["k"]),
        "adaptation_units": support_units,
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "late_error_weight": float(experiment["late_error_weight"]),
        "source_signature": inventory["source_signature"],
        "source_cache_origin": inventory["source_cache_origin"],
        "source_history_rows": int(len(source_history)),
    }
    for row in primary + fixed:
        row.update(common)
    for column, value in reversed(list(common.items())):
        scalar = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
        primary_predictions.insert(0, column, scalar)
    history.insert(0, "experiment_id", "experimentA4")
    history.insert(1, "cell_id", identifier)
    history.insert(2, "target_domain", domain)
    history.insert(3, "objective", objective)
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return primary, fixed, primary_predictions, history


def load_worker_state(paths: dict[str, Path]) -> dict:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    primary = read_json(paths["run_json"]) if paths["run_json"].is_file() else []
    primary = [row for row in primary if row.get("cell_id") in completed]
    state = {"completed": completed, "primary": primary}
    for name in ("fixed_run", "predictions", "history", "inventory"):
        frame = load_csv(paths[name])
        if name != "inventory" and not frame.empty:
            frame = frame[frame["cell_id"].isin(completed)]
        state[name] = frame
    return state


def save_worker_state(paths: dict[str, Path], state: dict, expected: int, experiment: dict) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["run_json"], state["primary"])
    a1.atomic_write_text(paths["run_csv"], pd.DataFrame(state["primary"]).to_csv(index=False))
    for name in ("fixed_run", "predictions", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    expected_primary = expected * len(experiment["role_partitions"]) * len(experiment["endpoint_seeds"])
    expected_fixed = expected * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"])
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected,
            "completed_primary_records": len(state["primary"]),
            "expected_primary_records": expected_primary,
            "completed_fixed_records": int(len(state["fixed_run"])),
            "expected_fixed_records": expected_fixed,
            "complete": bool(
                len(state["completed"]) == expected
                and len(state["primary"]) == expected_primary
                and len(state["fixed_run"]) == expected_fixed
            ),
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A4 worker")
    protocols, evidence = load_training_only_protocol(base, experiment)
    protocol = protocols[domain]
    output = Path(base["output_dir"])
    paths = shard_paths(output, domain, model_seed)
    worker_base = deepcopy(base)
    worker_base.update(
        {
            "output_dir": str(paths["directory"]),
            "target_domain": domain,
            "source_domains": protocol["source_domains"],
        }
    )
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(
        worker_base,
        experiment["preprocessing"],
        int(experiment["sensor_graph_k"]),
    )
    worker_manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "target_domain": domain,
        "model_seed": model_seed,
        "protocol_hash": protocol["protocol_hash"],
        "evidence_hashes": evidence["a2_1_input_hashes"],
        "graph_fit": graph_fit,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "target_domain", "model_seed", "protocol_hash", "evidence_hashes"):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(f"existing A4 shard is incompatible at {key}; use a new output directory")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], worker_manifest)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(
        paths["directory"] / "source_prior_adjacency.csv",
        pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv(),
    )
    a1.atomic_write_text(
        paths["directory"] / "source_prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )
    state = load_worker_state(paths)
    expected = len(OBJECTIVES) * len(experiment["target_split_seeds"])
    pending = [
        (objective, split_seed)
        for objective in OBJECTIVES
        for split_seed in experiment["target_split_seeds"]
        if training_cell_id(domain, objective, model_seed, split_seed) not in state["completed"]
    ]
    if pending:
        source_state, source_history, inventory = require_verified_source_cache(
            worker_base,
            experiment,
            protocol,
            model_seed,
            prior,
        )
        state["inventory"] = pd.DataFrame([{"target_domain": domain, **inventory}])
        for objective, split_seed in pending:
            primary, fixed, predictions, history = run_training_cell(
                base=worker_base,
                experiment=experiment,
                protocol=protocol,
                objective=objective,
                model_seed=model_seed,
                split_seed=int(split_seed),
                source_state=deepcopy(source_state),
                source_history=source_history,
                inventory=inventory,
                prior=prior,
            )
            state["primary"].extend(primary)
            state["fixed_run"] = pd.concat([state["fixed_run"], pd.DataFrame(fixed)], ignore_index=True)
            state["predictions"] = pd.concat([state["predictions"], predictions], ignore_index=True)
            state["history"] = pd.concat([state["history"], history], ignore_index=True)
            state["completed"].add(training_cell_id(domain, objective, model_seed, int(split_seed)))
            save_worker_state(paths, state, expected, experiment)
    save_worker_state(paths, state, expected, experiment)
    print(paths["status"].read_text(encoding="utf-8"))


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    inventory = a2.query_gpus()
    if args.gpus:
        devices = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        if not devices or len(devices) != len(set(devices)):
            raise ValueError("--gpus must contain unique physical GPU indices")
        known = {row["index"] for row in inventory}
        if not set(devices).issubset(known):
            raise RuntimeError("one or more requested GPUs are unavailable")
    else:
        visible = a2.visible_gpu_filter()
        candidates = [
            row
            for row in inventory
            if (visible is None or row["index"] in visible)
            and row["free_mb"] >= int(args.min_free_memory_mb)
            and row["utilization"] <= int(args.max_gpu_utilization)
        ]
        candidates.sort(key=lambda row: (-row["free_mb"], row["utilization"]))
        devices = [row["index"] for row in candidates]
    if args.max_workers > 0:
        devices = devices[: int(args.max_workers)]
    return devices, inventory


def worker_command(args: argparse.Namespace, domain: str, seed: int, device: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-domain",
        domain,
        "--worker-seed",
        str(seed),
        "--output-dir",
        str(output),
        "--device",
        device,
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
    ]
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.a2_output_dir:
        command.extend(["--a2-output-dir", args.a2_output_dir])
    if args.a2_1_output_dir:
        command.extend(["--a2-1-output-dir", args.a2_1_output_dir])
    if args.quick:
        command.append("--quick")
    if args.resume:
        command.append("--resume")
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]
        inventory: list[dict] = []
    else:
        devices, inventory = choose_gpus(args)
        if not devices:
            raise RuntimeError("no idle GPU met A4 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(
        json.dumps(
            {
                "scheduler": "experimentA4",
                "tasks": [{"domain": d, "seed": s} for d, s in tasks],
                "devices": devices,
                "gpu_inventory": inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    pending, active = list(tasks), {}
    while pending or active:
        for device in [item for item in devices if item not in active]:
            if not pending:
                break
            domain, seed = pending.pop(0)
            directory = shard_dir(output, domain, seed)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "worker_training.log"
            log_handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            if isinstance(device, int):
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                command = worker_command(args, domain, seed, "auto", output)
            else:
                command = worker_command(args, domain, seed, str(device), output)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[device] = {
                "process": process,
                "domain": domain,
                "seed": seed,
                "log": log_handle,
                "log_path": log_path,
            }
            print(f"[A4] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = "\n".join(
                    record["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                )
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"A4 worker failed domain={record['domain']} seed={record['seed']} exit={code}\n{tail}"
                )
            print(f"[A4] completed domain={record['domain']} seed={record['seed']} device={device}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]], experiment: dict) -> dict:
    merged: dict[str, Any] = {
        "primary": [],
        "fixed_run": [],
        "predictions": [],
        "history": [],
        "inventory": [],
    }
    expected = len(OBJECTIVES) * len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected:
            raise RuntimeError(f"incomplete A4 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get("official_test_forward_run"):
            raise RuntimeError(f"official-test contamination in A4 worker: {paths['status']}")
        merged["primary"].extend(read_json(paths["run_json"]))
        for name in ("fixed_run", "predictions", "history", "inventory"):
            merged[name].append(load_csv(paths[name]))
    for name in ("fixed_run", "predictions", "history", "inventory"):
        merged[name] = pd.concat(merged[name], ignore_index=True)
    return merged


PAIR_KEYS = [
    "target_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
    "endpoint_seed",
]
METRICS = [
    "rmse",
    "mae",
    "r2",
    "nasa_score",
    "mean_error",
    "late_rate",
    "late_error_q95",
    "under_error_q95",
]


def paired_objectives(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot(index=PAIR_KEYS, columns="objective", values=METRICS).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else column
        for column in pivot.columns
    ]
    output = pivot[PAIR_KEYS].copy()
    for metric in METRICS:
        output[f"{metric}_symmetric_mse"] = pivot[f"{metric}_symmetric_mse"]
        output[f"{metric}_late_weighted_mse_x2"] = pivot[f"{metric}_late_weighted_mse_x2"]
        output[f"{metric}_delta_asymmetric_minus_symmetric"] = (
            output[f"{metric}_late_weighted_mse_x2"]
            - output[f"{metric}_symmetric_mse"]
        )
    output["nasa_relative_delta"] = (
        output["nasa_score_delta_asymmetric_minus_symmetric"]
        / output["nasa_score_symmetric_mse"]
    )
    output["rmse_relative_delta"] = (
        output["rmse_delta_asymmetric_minus_symmetric"]
        / output["rmse_symmetric_mse"]
    )
    output["asymmetric_nasa_win"] = output["nasa_score_delta_asymmetric_minus_symmetric"] < 0
    output["asymmetric_rmse_win"] = output["rmse_delta_asymmetric_minus_symmetric"] < 0
    return output.sort_values(PAIR_KEYS)


def paired_fixed_endpoints(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_fraction"]
    pivot = results.pivot(index=keys, columns="objective", values=METRICS).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else column
        for column in pivot.columns
    ]
    output = pivot[keys].copy()
    for metric in METRICS:
        output[f"{metric}_symmetric_mse"] = pivot[f"{metric}_symmetric_mse"]
        output[f"{metric}_late_weighted_mse_x2"] = pivot[f"{metric}_late_weighted_mse_x2"]
        output[f"{metric}_delta_asymmetric_minus_symmetric"] = (
            output[f"{metric}_late_weighted_mse_x2"] - output[f"{metric}_symmetric_mse"]
        )
    return output.sort_values(keys)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    domains = sorted(frame["target_domain"].unique())
    model_seeds = sorted(frame["model_seed"].unique())
    split_seeds = sorted(frame["target_split_seed"].unique())
    partitions = sorted(frame["role_partition"].unique())
    endpoint_seeds = sorted(frame["endpoint_seed"].unique())
    lookup = frame.set_index(PAIR_KEYS)[column]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        values = []
        for domain in rng.choice(domains, len(domains), replace=True):
            chosen_models = rng.choice(model_seeds, len(model_seeds), replace=True)
            chosen_splits = rng.choice(split_seeds, len(split_seeds), replace=True)
            for model_seed in chosen_models:
                for split_seed in chosen_splits:
                    partition = int(rng.choice(partitions))
                    endpoint_seed = int(rng.choice(endpoint_seeds))
                    values.append(
                        float(
                            lookup.loc[
                                (
                                    domain,
                                    int(model_seed),
                                    int(split_seed),
                                    partition,
                                    endpoint_seed,
                                )
                            ]
                        )
                    )
        samples[repeat] = float(np.mean(values))
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    groups = ["target_domain", "objective"]
    rows = []
    for keys, frame in results.groupby(groups):
        row = {
            "target_domain": keys[0],
            "objective": keys[1],
            "n_records": int(len(frame)),
            "n_model_seeds": int(frame["model_seed"].nunique()),
            "n_target_splits": int(frame["target_split_seed"].nunique()),
            "n_role_partitions": int(frame["role_partition"].nunique()),
            "n_endpoint_seeds": int(frame["endpoint_seed"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups)


def comparison_summary(paired: pd.DataFrame, experiment: dict) -> pd.DataFrame:
    rows = []
    for scope, frame in [("ALL", paired)] + list(paired.groupby("target_domain")):
        nasa_ci = hierarchical_bootstrap(
            frame,
            "nasa_relative_delta",
            int(experiment["bootstrap_repetitions"]),
            stable_seed("A4_nasa", scope),
        )
        rmse_ci = hierarchical_bootstrap(
            frame,
            "rmse_relative_delta",
            int(experiment["bootstrap_repetitions"]),
            stable_seed("A4_rmse", scope),
        )
        rows.append(
            {
                "scope": scope,
                "n_records": int(len(frame)),
                "nasa_score_delta_mean": float(frame["nasa_score_delta_asymmetric_minus_symmetric"].mean()),
                "nasa_improvement_pct": float(-100.0 * frame["nasa_relative_delta"].mean()),
                "nasa_relative_boot_ci95_low": nasa_ci[0],
                "nasa_relative_boot_ci95_high": nasa_ci[1],
                "nasa_win_rate": float(frame["asymmetric_nasa_win"].mean()),
                "rmse_delta_mean": float(frame["rmse_delta_asymmetric_minus_symmetric"].mean()),
                "rmse_degradation_pct": float(100.0 * frame["rmse_relative_delta"].mean()),
                "rmse_relative_boot_ci95_low": rmse_ci[0],
                "rmse_relative_boot_ci95_high": rmse_ci[1],
                "rmse_win_rate": float(frame["asymmetric_rmse_win"].mean()),
                "late_error_q95_delta_mean": float(frame["late_error_q95_delta_asymmetric_minus_symmetric"].mean()),
                "under_error_q95_delta_mean": float(frame["under_error_q95_delta_asymmetric_minus_symmetric"].mean()),
                "mean_error_delta_mean": float(frame["mean_error_delta_asymmetric_minus_symmetric"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_decision(
    results: pd.DataFrame,
    paired: pd.DataFrame,
    comparisons: pd.DataFrame,
    experiment: dict,
) -> dict:
    expected_cells = (
        len(experiment["domains"])
        * len(OBJECTIVES)
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_records = (
        expected_cells
        * len(experiment["role_partitions"])
        * len(experiment["endpoint_seeds"])
    )
    primary = comparisons[comparisons["scope"] == "ALL"].iloc[0]
    domain_nasa = paired.groupby("target_domain")["nasa_relative_delta"].mean()
    domain_wins = int((domain_nasa < 0).sum())
    complete = bool(
        results["cell_id"].nunique() == expected_cells
        and len(results) == expected_records
    )
    uncontaminated = not results[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any()
    success = bool(
        complete
        and uncontaminated
        and primary["nasa_improvement_pct"]
        >= float(experiment["minimum_nasa_improvement_pct"])
        and primary["nasa_relative_boot_ci95_high"] < 0
        and 100.0 * primary["rmse_relative_boot_ci95_high"]
        <= float(experiment["rmse_noninferiority_margin_pct"])
        and domain_wins >= int(experiment["minimum_nasa_domain_wins"])
        and primary["late_error_q95_delta_mean"] < 0
    )
    decision = {
        "experiment_id": "experimentA4",
        "registered_primary_question": "Does a fixed 2x late-error target loss reduce balanced-endpoint NASA risk on training-only confirmation engines while remaining within a 3% RMSE noninferiority margin?",
        "expected_training_cells": expected_cells,
        "completed_training_cells": int(results["cell_id"].nunique()),
        "expected_primary_evaluation_records": expected_records,
        "completed_primary_evaluation_records": int(len(results)),
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "late_error_weight": float(experiment["late_error_weight"]),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "primary_result": {
            "nasa_score_delta_mean": float(primary["nasa_score_delta_mean"]),
            "nasa_improvement_pct": float(primary["nasa_improvement_pct"]),
            "nasa_relative_ci95": [
                float(primary["nasa_relative_boot_ci95_low"]),
                float(primary["nasa_relative_boot_ci95_high"]),
            ],
            "nasa_win_rate": float(primary["nasa_win_rate"]),
            "nasa_domain_win_count": domain_wins,
            "rmse_delta_mean": float(primary["rmse_delta_mean"]),
            "rmse_degradation_pct": float(primary["rmse_degradation_pct"]),
            "rmse_relative_ci95": [
                float(primary["rmse_relative_boot_ci95_low"]),
                float(primary["rmse_relative_boot_ci95_high"]),
            ],
            "late_error_q95_delta_mean": float(primary["late_error_q95_delta_mean"]),
            "under_error_q95_delta_mean": float(primary["under_error_q95_delta_mean"]),
            "mean_error_delta_mean": float(primary["mean_error_delta_mean"]),
        },
    }
    if experiment["quick_mode"]:
        decision.update(
            {
                "passed": complete,
                "reason": "quick smoke run only; do not interpret scientifically",
            }
        )
    else:
        decision.update(
            {
                "passed": success,
                "reason": (
                    "A4 confirmed training-only asymmetric endpoint-risk learning"
                    if success
                    else "A4 completed, but the asymmetric objective did not meet every registered risk/noninferiority criterion"
                ),
            }
        )
    return decision


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols, evidence = load_training_only_protocol(base, experiment)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"},
        "experiment_config": experiment,
        "evidence": evidence,
        "registered_primary_question": "Does a fixed 2x late-error target loss reduce balanced-endpoint NASA risk on training-only confirmation engines while remaining within a 3% RMSE noninferiority margin?",
        "A3_official_outputs_used_for_model_selection": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A4 output is incompatible at {key}; use a new output directory")
    atomic_json(paths["manifest"], manifest)
    selected_protocols = {domain: protocols[domain] for domain in experiment["domains"]}
    atomic_json(paths["protocol"], selected_protocols)
    a1.atomic_write_text(paths["engine_roles"], a21.protocol_rows(selected_protocols).to_csv(index=False))
    expected_cells = (
        len(experiment["domains"])
        * len(OBJECTIVES)
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_primary = expected_cells * len(experiment["role_partitions"]) * len(experiment["endpoint_seeds"])
    expected_fixed = expected_cells * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"])
    dry = {
        "experiment_id": "experimentA4",
        "domains": experiment["domains"],
        "objectives": experiment["objectives"],
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "role_partitions": experiment["role_partitions"],
        "endpoint_seeds": experiment["endpoint_seeds"],
        "expected_training_cells": expected_cells,
        "expected_primary_evaluation_records": expected_primary,
        "expected_fixed_endpoint_records": expected_fixed,
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "late_error_weight": float(experiment["late_error_weight"]),
        "evidence": evidence,
        "gpu_inventory": a2.query_gpus(),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry)
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    shard_root = output / "shards"
    if shard_root.exists() and any(shard_root.iterdir()) and not args.resume:
        raise RuntimeError("A4 contains an interrupted run; use --resume or a new output directory")
    tasks = [
        (domain, seed)
        for domain in experiment["domains"]
        for seed in experiment["model_seeds"]
    ]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    results = pd.DataFrame(merged["primary"]).sort_values(PAIR_KEYS + ["objective"])
    fixed = merged["fixed_run"].sort_values(
        ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_fraction", "objective"]
    )
    if results["cell_id"].nunique() != expected_cells or len(results) != expected_primary:
        raise RuntimeError("A4 merged primary output is incomplete")
    if len(fixed) != expected_fixed:
        raise RuntimeError("A4 merged fixed-endpoint output is incomplete")
    if results[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any():
        raise RuntimeError("A4 detected official-test contamination")
    paired = paired_objectives(results)
    fixed_paired = paired_fixed_endpoints(fixed)
    summary = summarize(results)
    comparisons = comparison_summary(paired, experiment)
    decision = make_decision(results, paired, comparisons, experiment)
    atomic_json(paths["run_json"], results.to_dict("records"))
    a1.atomic_write_text(paths["run_csv"], results.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_run"], fixed.to_csv(index=False))
    a1.atomic_write_text(paths["predictions"], merged["predictions"].to_csv(index=False))
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    a1.atomic_write_text(paths["inventory"], merged["inventory"].to_csv(index=False))
    a1.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    a1.atomic_write_text(paths["paired"], paired.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_paired"], fixed_paired.to_csv(index=False))
    a1.atomic_write_text(paths["comparisons"], comparisons.to_csv(index=False))
    atomic_json(paths["decision"], decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    base, experiment = load_config(args)
    validate_config(base, experiment)
    if args.worker_domain is not None:
        if args.worker_seed is None:
            raise ValueError("--worker-domain requires --worker-seed")
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
