"""Experiment A7: high-RUL underprediction-protective target learning.

Experiments A4--A6 showed a consistent failure mode: methods that lower
predictions reduce late-error tails but worsen the true high-RUL subset, whose
baseline predictions are already too low.  A7 moves the intervention into
target-head training.  The candidate uses the fixed, pre-registered loss

    MSE * [1 + I(label > 60 and prediction < label)]

with multiplier two.  Labels are used only while adapting on target support
engines.  Deployment still consumes only sensor windows; no future trajectory,
official-test file, calibration, or post-hoc prediction shift is used.

The symmetric-MSE baseline and the protective candidate are trained as paired
cells from identical verified A2 source states, target splits, support samples,
and random seeds.  Parent scheduling automatically uses idle GPUs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402


SCRIPT_VERSION = "experimentA7_high_rul_underprediction_protective_learning_v1"
EXPERIMENT_ID = "experimentA7"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
OBJECTIVES = ("symmetric_mse", "high_rul_underprediction_x2")
BASELINE_OBJECTIVE = "symmetric_mse"
CANDIDATE_OBJECTIVE = "high_rul_underprediction_x2"
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
SELECTION_ENDPOINT_SEEDS = list(range(8201, 8206))
CONFIRMATION_ENDPOINT_SEEDS = list(range(8301, 8306))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
HIGH_RUL_THRESHOLD = 60.0
PROTECTIVE_MULTIPLIER = 2.0
DEFAULT_OUTPUT = "outputs/experimentA7_high_rul_underprediction_protective_learning"
DEFAULT_A2_OUTPUT = a4.DEFAULT_A2_OUTPUT
DEFAULT_A2_1_OUTPUT = a4.DEFAULT_A2_1_OUTPUT
METRICS = a4.METRICS
PAIR_KEYS = [
    "target_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
    "endpoint_seed",
]
FIXED_KEYS = [
    "target_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
    "endpoint_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A7: high-RUL underprediction-protective learning"
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
        raise FileNotFoundError(f"required A7 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": "high_rul_underprediction_protective_learning",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "objectives": list(OBJECTIVES),
        "baseline_objective": BASELINE_OBJECTIVE,
        "candidate_objective": CANDIDATE_OBJECTIVE,
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "high_rul_threshold": HIGH_RUL_THRESHOLD,
        "protective_multiplier": PROTECTIVE_MULTIPLIER,
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "fixed_budget_no_epoch_selection": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "overall_nasa_noninferiority_margin_pct": 3.0,
        "overall_rmse_noninferiority_margin_pct": 3.0,
        "low_rul_nasa_noninferiority_margin_pct": 3.0,
        "high_rul_nasa_ci_upper_max": 0.0,
        "high_rul_rmse_ci_upper_max": 0.0,
        "minimum_high_rul_mean_error_uplift": 0.0,
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
                "selection_endpoint_seeds": [8201],
                "confirmation_endpoint_seeds": [8301],
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
        raise ValueError(f"A7 requires architecture={ARCHITECTURE}")
    if tuple(experiment["objectives"]) != OBJECTIVES:
        raise ValueError("A7 objective set is locked")
    if float(experiment["high_rul_threshold"]) != HIGH_RUL_THRESHOLD:
        raise ValueError("A7 high-RUL threshold is locked at 60")
    if float(experiment["protective_multiplier"]) != PROTECTIVE_MULTIPLIER:
        raise ValueError("A7 protective multiplier is locked at 2")
    if set(experiment["selection_endpoint_seeds"]) & set(
        experiment["confirmation_endpoint_seeds"]
    ):
        raise ValueError("selection and confirmation endpoint seeds must be disjoint")
    for name in (
        "domains",
        "model_seeds",
        "target_split_seeds",
        "role_partitions",
        "selection_endpoint_seeds",
        "confirmation_endpoint_seeds",
    ):
        values = experiment[name]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"A7 has empty/duplicate values in {name}")
    for domain in experiment["domains"]:
        if not a1.train_path(base["data_dir"], domain).is_file():
            raise FileNotFoundError(f"missing training file for {domain}")


def root_paths(output: Path) -> dict[str, Path]:
    p = EXPERIMENT_ID
    return {
        "manifest": output / f"{p}_manifest.json",
        "protocol": output / f"{p}_protocol.json",
        "engine_roles": output / f"{p}_engine_roles.csv",
        "dry_run": output / f"{p}_dry_run.json",
        "endpoint_predictions": output / f"{p}_pool_endpoint_predictions.csv",
        "training_protection": output / f"{p}_training_protection_audit.csv",
        "history": output / f"{p}_target_history.csv",
        "inventory": output / f"{p}_source_inventory.csv",
        "selection_predictions": output / f"{p}_selection_endpoint_predictions.csv",
        "confirmation_predictions": output / f"{p}_confirmation_endpoint_predictions.csv",
        "selection_run": output / f"{p}_selection_run_level.csv",
        "confirmation_run": output / f"{p}_confirmation_run_level.csv",
        "fixed_run": output / f"{p}_fixed_endpoint_run_level.csv",
        "paired": output / f"{p}_paired_protective_vs_symmetric.csv",
        "fixed_paired": output / f"{p}_fixed_endpoint_paired_protective_vs_symmetric.csv",
        "high_paired": output / f"{p}_high_rul_paired_protective_vs_symmetric.csv",
        "low_paired": output / f"{p}_low_rul_paired_protective_vs_symmetric.csv",
        "comparison": output / f"{p}_comparison_summary.csv",
        "decision": output / f"{p}_confirmation_decision.json",
    }


def shard_dir(output: Path, domain: str, model_seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{model_seed:03d}"


def shard_paths(output: Path, domain: str, model_seed: int) -> dict[str, Path]:
    directory = shard_dir(output, domain, model_seed)
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "endpoints": directory / "pool_endpoint_predictions.csv",
        "protection": directory / "training_protection_audit.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def training_cell_id(
    domain: str, objective: str, model_seed: int, split_seed: int
) -> str:
    return (
        f"{EXPERIMENT_ID}_{domain.lower()}_{objective}_"
        f"mseed{model_seed:03d}_tsplit{split_seed}"
    )


def target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    high_rul_threshold: float,
    protective_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = prediction - target
    squared = error.square()
    unweighted_mse = squared.mean()
    if objective == BASELINE_OBJECTIVE:
        protected = torch.zeros_like(error, dtype=torch.bool)
        return unweighted_mse, unweighted_mse, protected
    if objective != CANDIDATE_OBJECTIVE:
        raise ValueError(f"unknown A7 objective: {objective}")
    protected = (target > high_rul_threshold) & (error < 0)
    weights = torch.where(
        protected,
        torch.full_like(squared, protective_multiplier),
        torch.ones_like(squared),
    )
    return (weights * squared).mean(), unweighted_mse, protected


def train_target_head(
    model: torch.nn.Module,
    support: Any,
    pool: Any,
    cfg: dict,
    device: torch.device,
    objective: str,
    experiment: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("A7 model has no predictor.* parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["target_lr"]))
    history: list[dict[str, Any]] = []
    total_windows = 0
    protected_windows = 0
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        learner.train()
        objective_losses: list[float] = []
        mse_losses: list[float] = []
        epoch_windows = 0
        epoch_protected = 0
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss, mse, protected = target_loss(
                prediction,
                y,
                objective,
                float(experiment["high_rul_threshold"]),
                float(experiment["protective_multiplier"]),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A7 target loss became NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            count = int(y.numel())
            epoch_windows += count
            epoch_protected += int(protected.sum().item())
            objective_losses.append(float(loss.item()))
            mse_losses.append(float(mse.item()))
        total_windows += epoch_windows
        protected_windows += epoch_protected
        history.append(
            {
                "epoch": epoch,
                "objective_loss": float(np.mean(objective_losses)),
                "unweighted_mse_loss": float(np.mean(mse_losses)),
                "support_windows": epoch_windows,
                "protected_windows": epoch_protected,
                "protected_window_rate": float(epoch_protected / epoch_windows),
            }
        )
        print(
            f"A7 objective={objective} epoch={epoch:02d}/{cfg['target_epochs']} "
            f"objective_loss={np.mean(objective_losses):.4f} "
            f"mse={np.mean(mse_losses):.4f} protected={epoch_protected}/{epoch_windows}"
        )
    predictions = a1.predict_with_units(learner, pool, device)
    del learner
    return predictions, pd.DataFrame(history), {
        "objective": objective,
        "training_windows": total_windows,
        "protected_windows": protected_windows,
        "protected_window_rate": float(protected_windows / total_windows),
    }


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    # Recreate loaders per objective with the same seed.  This makes weighted
    # support sampling identical across paired baseline/candidate training.
    a1.seed_everything(run_seed)
    support, pool, feature_count = a21.prepare_support_pool(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        support_units,
        pool_units,
    )
    model = exp17b.build_model_17b(ARCHITECTURE, feature_count, cfg, prior, prior)
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    predictions, history, protection = train_target_head(
        model,
        support,
        pool,
        cfg,
        device,
        objective,
        experiment,
    )
    endpoints = a21.endpoint_epoch_rows(predictions, int(experiment["target_epochs"]))
    identifier = training_cell_id(domain, objective, model_seed, split_seed)
    common = {
        "experiment_id": EXPERIMENT_ID,
        "cell_id": identifier,
        "target_domain": domain,
        "model": ARCHITECTURE,
        "objective": objective,
        "model_seed": int(model_seed),
        "target_split_seed": int(split_seed),
        "target_run_seed": int(run_seed),
        "k": int(experiment["k"]),
        "adaptation_units": json.dumps(support_units, ensure_ascii=False),
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "high_rul_threshold": float(experiment["high_rul_threshold"]),
        "protective_multiplier": float(experiment["protective_multiplier"]),
        "source_signature": inventory["source_signature"],
        "source_cache_origin": inventory["source_cache_origin"],
        "source_history_rows": int(len(source_history)),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for column, value in reversed(list(common.items())):
        endpoints.insert(0, column, value)
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", identifier)
    history.insert(2, "target_domain", domain)
    history.insert(3, "objective", objective)
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    protection_frame = pd.DataFrame([{**common, **protection}])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return endpoints, protection_frame, history


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    state = {
        "completed": completed,
        "endpoints": load_csv(paths["endpoints"]),
        "protection": load_csv(paths["protection"]),
        "history": load_csv(paths["history"]),
        "inventory": load_csv(paths["inventory"]),
    }
    for name in ("endpoints", "protection", "history"):
        if not state[name].empty:
            state[name] = state[name][state[name]["cell_id"].isin(completed)]
    return state


def save_worker_state(paths: dict[str, Path], state: dict[str, Any], expected: int) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in ("endpoints", "protection", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected,
            "endpoint_rows": int(len(state["endpoints"])),
            "protection_audit_rows": int(len(state["protection"])),
            "complete": len(state["completed"]) == expected,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A7 worker")
    protocols, evidence = a4.load_training_only_protocol(base, experiment)
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
    manifest = {
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
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"existing A7 shard is incompatible at {key}; use a new output directory"
                )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], manifest)
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
        if training_cell_id(domain, objective, model_seed, split_seed)
        not in state["completed"]
    ]
    if pending:
        source_state, source_history, inventory = a4.require_verified_source_cache(
            worker_base, experiment, protocol, model_seed, prior
        )
        state["inventory"] = pd.DataFrame([{"target_domain": domain, **inventory}])
        for objective, split_seed in pending:
            endpoints, protection, history = run_training_cell(
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
            state["endpoints"] = pd.concat([state["endpoints"], endpoints], ignore_index=True)
            state["protection"] = pd.concat([state["protection"], protection], ignore_index=True)
            state["history"] = pd.concat([state["history"], history], ignore_index=True)
            state["completed"].add(
                training_cell_id(domain, objective, model_seed, int(split_seed))
            )
            save_worker_state(paths, state, expected)
    save_worker_state(paths, state, expected)
    print(paths["status"].read_text(encoding="utf-8"))


def worker_command(
    args: argparse.Namespace, domain: str, seed: int, device: str, output: Path
) -> list[str]:
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker-domain", domain, "--worker-seed", str(seed),
        "--output-dir", str(output), "--device", device,
        "--bootstrap-repetitions", str(args.bootstrap_repetitions),
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
        devices, inventory = a4.choose_gpus(args)
        if not devices:
            raise RuntimeError("no idle GPU met A7 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(json.dumps({"scheduler": EXPERIMENT_ID, "tasks": [{"domain": d, "seed": s} for d, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending = list(tasks)
    active: dict[str | int, dict[str, Any]] = {}
    while pending or active:
        for device in [item for item in devices if item not in active]:
            if not pending:
                break
            domain, seed = pending.pop(0)
            directory = shard_dir(output, domain, seed)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "worker_training.log"
            handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            if isinstance(device, int):
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                command = worker_command(args, domain, seed, "auto", output)
            else:
                command = worker_command(args, domain, seed, str(device), output)
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active[device] = {"process": process, "domain": domain, "seed": seed, "handle": handle, "log_path": log_path}
            print(f"[A7] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished: list[str | int] = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["handle"].close()
            if code != 0:
                tail = "\n".join(record["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(f"A7 worker failed domain={record['domain']} seed={record['seed']} exit={code}\n{tail}")
            print(f"[A7] completed domain={record['domain']} seed={record['seed']} device={device}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]], experiment: dict) -> dict[str, pd.DataFrame]:
    merged: dict[str, list[pd.DataFrame]] = {"endpoints": [], "protection": [], "history": [], "inventory": []}
    expected = len(OBJECTIVES) * len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected:
            raise RuntimeError(f"incomplete A7 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get("official_test_forward_run"):
            raise RuntimeError(f"official-test contamination: {paths['status']}")
        for name in merged:
            merged[name].append(load_csv(paths[name]))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in merged.items()}


def evaluate_objective(frame: pd.DataFrame) -> dict[str, float]:
    return a4.endpoint_risk_metrics(frame)


def evaluate_roles(endpoints: pd.DataFrame, protocols: dict[str, dict], experiment: dict) -> dict[str, pd.DataFrame]:
    selection_parts: list[pd.DataFrame] = []
    confirmation_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for values, frame in endpoints.groupby(["target_domain", "model_seed", "target_split_seed", "objective"]):
        domain, model_seed, split_seed, objective = values
        protocol = protocols[str(domain)]
        split = protocol["role_splits"][str(int(split_seed))]
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]
            common_role = {
                "target_domain": str(domain), "model_seed": int(model_seed),
                "target_split_seed": int(split_seed), "objective": str(objective),
                "role_partition": int(partition), "fixed_budget_epoch": int(experiment["target_epochs"]),
                "official_test_files_accessed": False, "official_test_forward_run": False,
            }
            for role, units, seeds, output_parts, output_rows in (
                ("selection", list(map(int, roles["selection_units"])), experiment["selection_endpoint_seeds"], selection_parts, selection_rows),
                ("confirmation", list(map(int, roles["confirmation_units"])), experiment["confirmation_endpoint_seeds"], confirmation_parts, confirmation_rows),
            ):
                for endpoint_seed in seeds:
                    assignment = a21.balanced_assignment(units, str(domain), int(split_seed), int(partition), int(endpoint_seed), role)
                    selected = a21.endpoint_subset(frame, units, assignment=assignment).copy()
                    selected["role_partition"] = int(partition)
                    selected["endpoint_seed"] = int(endpoint_seed)
                    selected["evaluation_role"] = role
                    output_parts.append(selected)
                    output_rows.append({
                        **common_role, "endpoint_seed": int(endpoint_seed), "evaluation_role": role,
                        "evaluation_protocol": "balanced_endpoint", **evaluate_objective(selected),
                        "evaluation_engine_count": int(selected["unit"].nunique()),
                    })
            confirmation_units = list(map(int, roles["confirmation_units"]))
            for fraction in experiment["endpoint_fractions"]:
                selected = a21.endpoint_subset(frame, confirmation_units, fraction=float(fraction))
                fixed_rows.append({
                    **common_role, "endpoint_fraction": float(fraction),
                    "evaluation_protocol": f"fixed_endpoint_{int(round(100 * float(fraction))):03d}",
                    **evaluate_objective(selected), "evaluation_engine_count": int(selected["unit"].nunique()),
                })
    return {
        "selection_predictions": pd.concat(selection_parts, ignore_index=True),
        "confirmation_predictions": pd.concat(confirmation_parts, ignore_index=True),
        "selection_run": pd.DataFrame(selection_rows),
        "confirmation_run": pd.DataFrame(confirmation_rows),
        "fixed_run": pd.DataFrame(fixed_rows),
    }


def paired_objectives(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pivot = results.pivot(index=keys, columns="objective", values=METRICS).reset_index()
    pivot.columns = ["_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else column for column in pivot.columns]
    output = pivot[keys].copy()
    for metric in METRICS:
        baseline = pivot[f"{metric}_{BASELINE_OBJECTIVE}"].astype(float)
        candidate = pivot[f"{metric}_{CANDIDATE_OBJECTIVE}"].astype(float)
        output[f"{metric}_{BASELINE_OBJECTIVE}"] = baseline
        output[f"{metric}_{CANDIDATE_OBJECTIVE}"] = candidate
        output[f"{metric}_delta_candidate_minus_baseline"] = candidate - baseline
    output["candidate"] = CANDIDATE_OBJECTIVE
    output["nasa_relative_delta"] = output["nasa_score_delta_candidate_minus_baseline"] / output[f"nasa_score_{BASELINE_OBJECTIVE}"]
    output["rmse_relative_delta"] = output["rmse_delta_candidate_minus_baseline"] / output[f"rmse_{BASELINE_OBJECTIVE}"]
    output["candidate_nasa_win"] = output["nasa_score_delta_candidate_minus_baseline"] < 0
    output["candidate_rmse_win"] = output["rmse_delta_candidate_minus_baseline"] < 0
    return output.sort_values(keys)


def stage_pairs(predictions: pd.DataFrame, high: bool, experiment: dict) -> pd.DataFrame:
    threshold = float(experiment["high_rul_threshold"])
    stage = predictions[predictions["label"] > threshold].copy() if high else predictions[predictions["label"] <= threshold].copy()
    row_keys = PAIR_KEYS + ["unit", "endpoint_fraction", "unit_window_index", "label"]
    pivot = stage.pivot(index=row_keys, columns="objective", values=["prediction", "error", "nasa_contribution"]).reset_index()
    pivot.columns = ["_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else column for column in pivot.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in pivot.groupby(PAIR_KEYS):
        row = dict(zip(PAIR_KEYS, keys))
        row["rul_stage"] = "high_rul_gt60" if high else "low_or_mid_rul_le60"
        row["rul_threshold"] = threshold
        row["stage_engine_count"] = int(group["unit"].nunique())
        for objective in OBJECTIVES:
            prediction = group[f"prediction_{objective}"].to_numpy(dtype=float)
            label = group["label"].to_numpy(dtype=float)
            error = prediction - label
            evaluated = pd.DataFrame({"label": label, "prediction": prediction, "error": error, "nasa_contribution": nasa_contribution(error)})
            metrics = a4.endpoint_risk_metrics(evaluated)
            for metric in METRICS:
                row[f"{metric}_{objective}"] = float(metrics[metric])
        for metric in METRICS:
            row[f"{metric}_delta_candidate_minus_baseline"] = row[f"{metric}_{CANDIDATE_OBJECTIVE}"] - row[f"{metric}_{BASELINE_OBJECTIVE}"]
        row["nasa_relative_delta"] = row["nasa_score_delta_candidate_minus_baseline"] / row[f"nasa_score_{BASELINE_OBJECTIVE}"]
        row["rmse_relative_delta"] = row["rmse_delta_candidate_minus_baseline"] / row[f"rmse_{BASELINE_OBJECTIVE}"]
        row["candidate_nasa_win"] = row["nasa_score_delta_candidate_minus_baseline"] < 0
        row["candidate_rmse_win"] = row["rmse_delta_candidate_minus_baseline"] < 0
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS)
    expected = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    if len(output) != expected:
        raise RuntimeError("one or more A7 confirmation cells lack requested RUL-stage observations")
    return output


def nasa_contribution(error: np.ndarray) -> np.ndarray:
    value = np.asarray(error, dtype=float)
    return np.where(value < 0, np.exp(-value / 13.0) - 1.0, np.exp(value / 10.0) - 1.0)


def comparison_summary(paired: pd.DataFrame, experiment: dict, comparison: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in [("ALL", paired)] + list(paired.groupby("target_domain")):
        nasa_ci = a4.hierarchical_bootstrap(frame, "nasa_relative_delta", int(experiment["bootstrap_repetitions"]), a4.stable_seed(EXPERIMENT_ID, comparison, "nasa", scope))
        rmse_ci = a4.hierarchical_bootstrap(frame, "rmse_relative_delta", int(experiment["bootstrap_repetitions"]), a4.stable_seed(EXPERIMENT_ID, comparison, "rmse", scope))
        rows.append({
            "comparison": comparison, "scope": scope, "n_records": int(len(frame)),
            "nasa_score_delta_mean": float(frame["nasa_score_delta_candidate_minus_baseline"].mean()),
            "nasa_improvement_pct": float(-100 * frame["nasa_relative_delta"].mean()),
            "nasa_relative_boot_ci95_low": nasa_ci[0], "nasa_relative_boot_ci95_high": nasa_ci[1],
            "nasa_win_rate": float(frame["candidate_nasa_win"].mean()),
            "rmse_delta_mean": float(frame["rmse_delta_candidate_minus_baseline"].mean()),
            "rmse_degradation_pct": float(100 * frame["rmse_relative_delta"].mean()),
            "rmse_relative_boot_ci95_low": rmse_ci[0], "rmse_relative_boot_ci95_high": rmse_ci[1],
            "rmse_win_rate": float(frame["candidate_rmse_win"].mean()),
            "late_error_q95_delta_mean": float(frame["late_error_q95_delta_candidate_minus_baseline"].mean()),
            "under_error_q95_delta_mean": float(frame["under_error_q95_delta_candidate_minus_baseline"].mean()),
            "mean_error_delta_mean": float(frame["mean_error_delta_candidate_minus_baseline"].mean()),
        })
    return pd.DataFrame(rows)


def stage_summary(paired: pd.DataFrame, experiment: dict, label: str) -> dict[str, Any]:
    nasa_ci = a4.hierarchical_bootstrap(paired, "nasa_relative_delta", int(experiment["bootstrap_repetitions"]), a4.stable_seed(EXPERIMENT_ID, label, "nasa"))
    rmse_ci = a4.hierarchical_bootstrap(paired, "rmse_relative_delta", int(experiment["bootstrap_repetitions"]), a4.stable_seed(EXPERIMENT_ID, label, "rmse"))
    return {
        "stage": label, "n_records": int(len(paired)),
        "nasa_improvement_pct": float(-100 * paired["nasa_relative_delta"].mean()),
        "nasa_relative_ci95": [nasa_ci[0], nasa_ci[1]],
        "rmse_degradation_pct": float(100 * paired["rmse_relative_delta"].mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
        "mean_error_delta_mean": float(paired["mean_error_delta_candidate_minus_baseline"].mean()),
        "nasa_win_rate": float(paired["candidate_nasa_win"].mean()),
        "rmse_win_rate": float(paired["candidate_rmse_win"].mean()),
    }


def make_decision(*, endpoints: pd.DataFrame, protection: pd.DataFrame, confirmation: pd.DataFrame, paired: pd.DataFrame, comparisons: pd.DataFrame, high: dict[str, Any], low: dict[str, Any], experiment: dict) -> dict[str, Any]:
    expected_training = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(OBJECTIVES)
    expected_confirmation = expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    expected_pairs = expected_confirmation // len(OBJECTIVES)
    primary = comparisons[(comparisons["comparison"] == "full_endpoint_protective_vs_symmetric") & (comparisons["scope"] == "ALL")].iloc[0]
    complete = bool(endpoints["cell_id"].nunique() == expected_training and len(confirmation) == expected_confirmation and len(paired) == expected_pairs and len(protection) == expected_training)
    uncontaminated = not endpoints[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any()
    high_success = bool(high["nasa_relative_ci95"][1] < float(experiment["high_rul_nasa_ci_upper_max"]) and high["rmse_relative_ci95"][1] < float(experiment["high_rul_rmse_ci_upper_max"]) and high["mean_error_delta_mean"] > float(experiment["minimum_high_rul_mean_error_uplift"]))
    overall_safe = bool(100 * primary["nasa_relative_boot_ci95_high"] <= float(experiment["overall_nasa_noninferiority_margin_pct"]) and 100 * primary["rmse_relative_boot_ci95_high"] <= float(experiment["overall_rmse_noninferiority_margin_pct"]))
    low_safe = bool(100 * low["nasa_relative_ci95"][1] <= float(experiment["low_rul_nasa_noninferiority_margin_pct"]))
    candidate_protection = protection[protection["objective"] == CANDIDATE_OBJECTIVE]
    protection_active = bool((candidate_protection["protected_windows"] > 0).all() and (candidate_protection["protected_window_rate"] > 0).all())
    passed = bool(complete and uncontaminated and protection_active and high_success and overall_safe and low_safe)
    decision: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": "Does the fixed high-RUL-underprediction protective loss improve true-high-RUL NASA/RMSE and bias without materially harming full-endpoint or low-RUL risk?",
        "expected_training_cells": expected_training, "completed_training_cells": int(endpoints["cell_id"].nunique()),
        "expected_confirmation_records": expected_confirmation, "completed_confirmation_records": int(len(confirmation)),
        "expected_primary_pairs": expected_pairs, "completed_primary_pairs": int(len(paired)),
        "complete": complete, "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "high_rul_threshold": float(experiment["high_rul_threshold"]), "protective_multiplier": float(experiment["protective_multiplier"]),
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"], "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "official_test_files_accessed": False, "official_test_forward_run": False,
        "protection_audit": {
            "candidate_training_cells": int(len(candidate_protection)),
            "protected_window_rate_mean": float(candidate_protection["protected_window_rate"].mean()),
            "protected_window_rate_min": float(candidate_protection["protected_window_rate"].min()),
            "protected_windows_total": int(candidate_protection["protected_windows"].sum()),
            "active_in_all_candidate_cells": protection_active,
        },
        "full_endpoint_result": {
            "nasa_improvement_pct": float(primary["nasa_improvement_pct"]),
            "nasa_relative_ci95": [float(primary["nasa_relative_boot_ci95_low"]), float(primary["nasa_relative_boot_ci95_high"])],
            "rmse_degradation_pct": float(primary["rmse_degradation_pct"]),
            "rmse_relative_ci95": [float(primary["rmse_relative_boot_ci95_low"]), float(primary["rmse_relative_boot_ci95_high"])],
        },
        "high_rul_result": {**high, "strict_success": high_success},
        "low_rul_safety_result": {**low, "nasa_noninferiority_passed": low_safe},
        "passed": passed if not experiment["quick_mode"] else complete and protection_active,
        "reason": (
            "quick smoke run only; do not interpret scientifically" if experiment["quick_mode"] else
            ("A7 confirmed high-RUL underprediction-protective learning" if passed else "A7 completed, but the protective loss did not meet every registered high-RUL/safety criterion")
        ),
        "next_action": None if experiment["quick_mode"] else ("run_fresh_seed_confirmation_without_official_test_access" if passed else "stop_predictor_loss_tuning_and_reassess_experimentA8_representation_direction"),
    }
    return decision


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols, evidence = a4.load_training_only_protocol(base, experiment)
    manifest = {
        "script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"}, "experiment_config": experiment, "evidence": evidence,
        "registered_primary_question": "Does the fixed high-RUL-underprediction protective loss improve true-high-RUL NASA/RMSE and bias without materially harming full-endpoint or low-RUL risk?",
        "candidate_uses_labels_only_during_target_training": True, "candidate_uses_labels_at_deployment": False,
        "candidate_uses_future_windows": False, "A6_results_used_for_hypothesis_generation": True, "A6_confirmation_outputs_used_for_runtime_fitting": False,
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        existing = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A7 output is incompatible at {key}; use a new output directory")
    atomic_json(paths["manifest"], manifest)
    selected_protocols = {domain: protocols[domain] for domain in experiment["domains"]}
    atomic_json(paths["protocol"], selected_protocols)
    a1.atomic_write_text(paths["engine_roles"], a21.protocol_rows(selected_protocols).to_csv(index=False))
    base_training = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"])
    expected_training = base_training * len(OBJECTIVES)
    dry = {
        "experiment_id": EXPERIMENT_ID, "objectives": list(OBJECTIVES), "domains": experiment["domains"], "model_seeds": experiment["model_seeds"], "target_split_seeds": experiment["target_split_seeds"], "role_partitions": experiment["role_partitions"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"], "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"], "endpoint_seed_sets_disjoint": True,
        "high_rul_threshold": float(experiment["high_rul_threshold"]), "protective_multiplier": float(experiment["protective_multiplier"]),
        "loss": "MSE * [1 + I(label > 60 and prediction < label)]", "expected_training_cells": expected_training,
        "expected_selection_records": expected_training * len(experiment["role_partitions"]) * len(experiment["selection_endpoint_seeds"]),
        "expected_confirmation_records": expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"]),
        "expected_fixed_endpoint_records": expected_training * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]), "evidence": evidence, "gpu_inventory": a2.query_gpus(),
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry)
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2)); return
    shard_root = output / "shards"
    if shard_root.exists() and any(shard_root.iterdir()) and not args.resume:
        raise RuntimeError("A7 contains an interrupted run; use --resume or a new output directory")
    tasks = [(domain, seed) for domain in experiment["domains"] for seed in experiment["model_seeds"]]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    endpoints = merged["endpoints"].sort_values(["target_domain", "objective", "model_seed", "target_split_seed", "unit", "endpoint_fraction"])
    if endpoints["cell_id"].nunique() != expected_training:
        raise RuntimeError("A7 endpoint outputs are incomplete")
    if endpoints[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any():
        raise RuntimeError("A7 detected official-test contamination")
    evaluated = evaluate_roles(endpoints, selected_protocols, experiment)
    selection = evaluated["selection_run"].sort_values(PAIR_KEYS + ["objective"])
    confirmation = evaluated["confirmation_run"].sort_values(PAIR_KEYS + ["objective"])
    fixed = evaluated["fixed_run"].sort_values(FIXED_KEYS + ["objective"])
    expected_selection = expected_training * len(experiment["role_partitions"]) * len(experiment["selection_endpoint_seeds"])
    expected_confirmation = expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    expected_fixed = expected_training * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"])
    if len(selection) != expected_selection or len(confirmation) != expected_confirmation or len(fixed) != expected_fixed:
        raise RuntimeError("A7 role/fixed evaluation output is incomplete")
    paired = paired_objectives(confirmation, PAIR_KEYS)
    fixed_paired = paired_objectives(fixed, FIXED_KEYS)
    high_paired = stage_pairs(evaluated["confirmation_predictions"], True, experiment)
    low_paired = stage_pairs(evaluated["confirmation_predictions"], False, experiment)
    comparisons = pd.concat([
        comparison_summary(paired, experiment, "full_endpoint_protective_vs_symmetric"),
        comparison_summary(high_paired, experiment, "high_rul_protective_vs_symmetric"),
        comparison_summary(low_paired, experiment, "low_rul_protective_vs_symmetric"),
    ], ignore_index=True)
    high = stage_summary(high_paired, experiment, "high_rul_gt60")
    low = stage_summary(low_paired, experiment, "low_or_mid_rul_le60")
    decision = make_decision(endpoints=endpoints, protection=merged["protection"], confirmation=confirmation, paired=paired, comparisons=comparisons, high=high, low=low, experiment=experiment)
    a1.atomic_write_text(paths["endpoint_predictions"], endpoints.to_csv(index=False))
    a1.atomic_write_text(paths["training_protection"], merged["protection"].to_csv(index=False))
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    inventory = merged["inventory"].drop_duplicates(["target_domain", "model_seed"])
    a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False))
    for name, frame in (("selection_predictions", evaluated["selection_predictions"]), ("confirmation_predictions", evaluated["confirmation_predictions"]), ("selection_run", selection), ("confirmation_run", confirmation), ("fixed_run", fixed), ("paired", paired), ("fixed_paired", fixed_paired), ("high_paired", high_paired), ("low_paired", low_paired), ("comparison", comparisons)):
        a1.atomic_write_text(paths[name], frame.to_csv(index=False))
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
