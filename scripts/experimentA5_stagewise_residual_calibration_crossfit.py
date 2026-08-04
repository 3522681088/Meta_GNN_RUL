"""Experiment A5: stagewise residual calibration with cross-fit evaluation.

A4 and A4_1 showed that a global 2x late-error objective can reduce endpoint
NASA risk, but it shifts high-RUL predictions downward.  A5 changes direction:
the target model is trained only with symmetric MSE, then a small deployable
residual correction is estimated on selection engines and locked before it is
evaluated on engine-disjoint confirmation engines.

The primary candidate is ``stagewise_guarded``.  Selection predictions are
binned by *predicted* RUL, so the same rule is available at deployment:

    [-inf, 20), [20, 40), [40, 60), [60, inf)

Only positive mean residuals in the first three bins can produce a downward
correction.  The correction is reliability-shrunk by n/(n+30), capped at three
RUL units, and fixed to zero for predicted RUL >= 60.  ``global_positive`` is
retained only as a secondary ablation.  Official C-MAPSS test files are never
opened.

Run from the repository root::

    python -u scripts/experimentA5_stagewise_residual_calibration_crossfit.py --dry-run

    nohup python -u scripts/experimentA5_stagewise_residual_calibration_crossfit.py \
      > experimentA5_training.log 2>&1 &

All artifacts are written under
``outputs/experimentA5_stagewise_residual_calibration_crossfit``.  The parent
process automatically assigns domain/model-seed workers to idle GPUs.
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
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402


SCRIPT_VERSION = "experimentA5_stagewise_residual_calibration_crossfit_v1"
EXPERIMENT_ID = "experimentA5"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
SELECTION_ENDPOINT_SEEDS = list(range(7801, 7806))
CONFIRMATION_ENDPOINT_SEEDS = list(range(7901, 7906))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
VARIANTS = ("baseline_symmetric", "stagewise_guarded", "global_positive")
PRIMARY_VARIANT = "stagewise_guarded"
PREDICTION_BIN_EDGES = (-np.inf, 20.0, 40.0, 60.0, np.inf)
PREDICTION_BIN_LABELS = ("pred_lt20", "pred_20_40", "pred_40_60", "pred_ge60")
HIGH_PREDICTION_GUARD = 60.0
HIGH_TRUE_RUL_THRESHOLD = 60.0
SHRINKAGE_PRIOR_COUNT = 30.0
MAX_CORRECTION = 3.0
DEFAULT_OUTPUT = "outputs/experimentA5_stagewise_residual_calibration_crossfit"
DEFAULT_A2_OUTPUT = a4.DEFAULT_A2_OUTPUT
DEFAULT_A2_1_OUTPUT = a4.DEFAULT_A2_1_OUTPUT
METRICS = a4.METRICS
BASE_KEYS = ["target_domain", "model_seed", "target_split_seed"]
CALIBRATION_KEYS = BASE_KEYS + ["role_partition"]
PAIR_KEYS = CALIBRATION_KEYS + ["endpoint_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A5: stagewise residual calibration cross-fit"
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
        raise FileNotFoundError(f"required A5 input is missing: {path}")
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
        "experiment_name": "stagewise_residual_calibration_crossfit",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "objective": "symmetric_mse",
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "variants": list(VARIANTS),
        "primary_variant": PRIMARY_VARIANT,
        "prediction_bin_edges": ["-inf", 20.0, 40.0, 60.0, "inf"],
        "prediction_bin_labels": list(PREDICTION_BIN_LABELS),
        "high_prediction_guard": HIGH_PREDICTION_GUARD,
        "high_true_rul_threshold": HIGH_TRUE_RUL_THRESHOLD,
        "shrinkage_prior_count": SHRINKAGE_PRIOR_COUNT,
        "max_correction": MAX_CORRECTION,
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "fixed_budget_no_epoch_selection": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "minimum_nasa_improvement_pct": 3.0,
        "rmse_noninferiority_margin_pct": 3.0,
        "high_rul_rmse_noninferiority_margin_pct": 3.0,
        "high_rul_nasa_noninferiority_margin_pct": 3.0,
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
                "selection_endpoint_seeds": [7801],
                "confirmation_endpoint_seeds": [7901],
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
        raise ValueError(f"A5 requires architecture={ARCHITECTURE}")
    if experiment["objective"] != "symmetric_mse":
        raise ValueError("A5 trains only the symmetric MSE target model")
    if float(experiment["high_prediction_guard"]) != 60.0:
        raise ValueError("A5 high-prediction guard is locked at 60 RUL")
    if float(experiment["shrinkage_prior_count"]) != 30.0:
        raise ValueError("A5 shrinkage prior count is locked at 30")
    if float(experiment["max_correction"]) != 3.0:
        raise ValueError("A5 maximum correction is locked at 3 RUL")
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
            raise ValueError(f"A5 has empty/duplicate values in {name}")
    for domain in experiment["domains"]:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def root_paths(output: Path) -> dict[str, Path]:
    prefix = EXPERIMENT_ID
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "pool_predictions": output / f"{prefix}_pool_endpoint_predictions.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "calibration": output / f"{prefix}_calibration_parameters.csv",
        "selection_predictions": output / f"{prefix}_selection_calibration_predictions.csv",
        "confirmation_predictions": output / f"{prefix}_confirmation_predictions.csv",
        "confirmation_run": output / f"{prefix}_confirmation_run_level.csv",
        "fixed_run": output / f"{prefix}_fixed_endpoint_run_level.csv",
        "paired_primary": output / f"{prefix}_paired_stagewise_vs_baseline.csv",
        "paired_global": output / f"{prefix}_paired_global_vs_baseline.csv",
        "fixed_primary": output / f"{prefix}_fixed_endpoint_paired_stagewise_vs_baseline.csv",
        "high_rul_primary": output / f"{prefix}_high_rul_paired_stagewise_vs_baseline.csv",
        "high_rul_global": output / f"{prefix}_high_rul_paired_global_vs_baseline.csv",
        "comparisons": output / f"{prefix}_comparison_summary.csv",
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
        "pool_predictions": directory / "pool_endpoint_predictions.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def training_cell_id(domain: str, model_seed: int, split_seed: int) -> str:
    return f"{EXPERIMENT_ID}_{domain.lower()}_mseed{model_seed:03d}_tsplit{split_seed}"


def run_training_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_seed: int,
    split_seed: int,
    source_state: dict,
    source_history: list,
    inventory: dict,
    prior: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    predictions, history = a4.train_fixed_budget(
        model,
        support,
        pool,
        cfg,
        device,
        "symmetric_mse",
        1.0,
    )
    endpoint_rows = a21.endpoint_epoch_rows(
        predictions,
        int(experiment["target_epochs"]),
    )
    identifier = training_cell_id(domain, model_seed, split_seed)
    common = {
        "experiment_id": EXPERIMENT_ID,
        "cell_id": identifier,
        "target_domain": domain,
        "model": ARCHITECTURE,
        "objective": "symmetric_mse",
        "model_seed": int(model_seed),
        "target_split_seed": int(split_seed),
        "target_run_seed": int(run_seed),
        "k": int(experiment["k"]),
        "adaptation_units": json.dumps(support_units, ensure_ascii=False),
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "source_signature": inventory["source_signature"],
        "source_cache_origin": inventory["source_cache_origin"],
        "source_history_rows": int(len(source_history)),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for column, value in reversed(list(common.items())):
        endpoint_rows.insert(0, column, value)
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", identifier)
    history.insert(2, "target_domain", domain)
    history.insert(3, "objective", "symmetric_mse")
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return endpoint_rows, history


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    predictions = load_csv(paths["pool_predictions"])
    history = load_csv(paths["history"])
    if not predictions.empty:
        predictions = predictions[predictions["cell_id"].isin(completed)]
    if not history.empty:
        history = history[history["cell_id"].isin(completed)]
    return {
        "completed": completed,
        "pool_predictions": predictions,
        "history": history,
        "inventory": load_csv(paths["inventory"]),
    }


def save_worker_state(
    paths: dict[str, Path],
    state: dict[str, Any],
    expected_cells: int,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in ("pool_predictions", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected_cells,
            "pool_endpoint_prediction_rows": int(len(state["pool_predictions"])),
            "complete": len(state["completed"]) == expected_cells,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain = str(args.worker_domain)
    model_seed = int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A5 worker")
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
        for key in (
            "script_hash",
            "target_domain",
            "model_seed",
            "protocol_hash",
            "evidence_hashes",
        ):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(
                    f"existing A5 shard is incompatible at {key}; use a new output directory"
                )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], worker_manifest)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(
        paths["directory"] / "source_prior_adjacency.csv",
        pd.DataFrame(
            prior.numpy().astype(int), index=sensors, columns=sensors
        ).to_csv(),
    )
    a1.atomic_write_text(
        paths["directory"] / "source_prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )
    state = load_worker_state(paths)
    expected_cells = len(experiment["target_split_seeds"])
    pending = [
        split_seed
        for split_seed in experiment["target_split_seeds"]
        if training_cell_id(domain, model_seed, split_seed) not in state["completed"]
    ]
    if pending:
        source_state, source_history, inventory = a4.require_verified_source_cache(
            worker_base,
            experiment,
            protocol,
            model_seed,
            prior,
        )
        state["inventory"] = pd.DataFrame([{"target_domain": domain, **inventory}])
        for split_seed in pending:
            endpoint_rows, history = run_training_cell(
                base=worker_base,
                experiment=experiment,
                protocol=protocol,
                model_seed=model_seed,
                split_seed=int(split_seed),
                source_state=deepcopy(source_state),
                source_history=source_history,
                inventory=inventory,
                prior=prior,
            )
            state["pool_predictions"] = pd.concat(
                [state["pool_predictions"], endpoint_rows], ignore_index=True
            )
            state["history"] = pd.concat(
                [state["history"], history], ignore_index=True
            )
            state["completed"].add(
                training_cell_id(domain, model_seed, int(split_seed))
            )
            save_worker_state(paths, state, expected_cells)
    save_worker_state(paths, state, expected_cells)
    print(paths["status"].read_text(encoding="utf-8"))


def worker_command(
    args: argparse.Namespace,
    domain: str,
    seed: int,
    device: str,
    output: Path,
) -> list[str]:
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


def run_workers(
    args: argparse.Namespace,
    tasks: list[tuple[str, int]],
    output: Path,
) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]
        inventory: list[dict] = []
    else:
        devices, inventory = a4.choose_gpus(args)
        if not devices:
            raise RuntimeError(
                "no idle GPU met A5 thresholds; inventory="
                + json.dumps(inventory, ensure_ascii=False)
            )
    print(
        json.dumps(
            {
                "scheduler": EXPERIMENT_ID,
                "tasks": [{"domain": d, "seed": s} for d, s in tasks],
                "devices": devices,
                "gpu_inventory": inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
            print(
                f"[A5] launched domain={domain} seed={seed} "
                f"device={device} pid={process.pid}"
            )
        finished: list[str | int] = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = "\n".join(
                    record["log_path"]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-80:]
                )
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"A5 worker failed domain={record['domain']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A5] completed domain={record['domain']} "
                f"seed={record['seed']} device={device}"
            )
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(
    output: Path,
    tasks: list[tuple[str, int]],
    experiment: dict,
) -> dict[str, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    inventories: list[pd.DataFrame] = []
    expected_cells = len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected_cells:
            raise RuntimeError(f"incomplete A5 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get(
            "official_test_forward_run"
        ):
            raise RuntimeError(f"official-test contamination: {paths['status']}")
        predictions.append(load_csv(paths["pool_predictions"]))
        histories.append(load_csv(paths["history"]))
        inventories.append(load_csv(paths["inventory"]))
    return {
        "pool_predictions": pd.concat(predictions, ignore_index=True),
        "history": pd.concat(histories, ignore_index=True),
        "inventory": pd.concat(inventories, ignore_index=True),
    }


def prediction_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=PREDICTION_BIN_EDGES,
        labels=PREDICTION_BIN_LABELS,
        right=False,
        include_lowest=True,
    )


def nasa_contribution(error: np.ndarray) -> np.ndarray:
    error = np.asarray(error, dtype=float)
    return np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )


def metrics_for_prediction(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    working = frame[["label", prediction_column]].copy()
    working["prediction"] = working[prediction_column].astype(float)
    working["error"] = working["prediction"] - working["label"].astype(float)
    working["nasa_contribution"] = nasa_contribution(working["error"].to_numpy())
    return a4.endpoint_risk_metrics(working)


def fit_calibration(
    selection: pd.DataFrame,
    common: dict[str, Any],
    experiment: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = selection.copy()
    working["prediction_bin"] = prediction_bin(working["prediction"])
    rows: list[dict[str, Any]] = []
    prior_count = float(experiment["shrinkage_prior_count"])
    cap = float(experiment["max_correction"])
    for label in PREDICTION_BIN_LABELS:
        group = working[working["prediction_bin"] == label]
        count = int(len(group))
        raw_bias = float(group["error"].mean()) if count else 0.0
        bias_std = float(group["error"].std(ddof=1)) if count > 1 else 0.0
        reliability = float(count / (count + prior_count)) if count else 0.0
        eligible = label != "pred_ge60"
        positive_bias = max(raw_bias, 0.0) if eligible else 0.0
        correction = min(cap, positive_bias * reliability)
        rows.append(
            {
                **common,
                "calibration_variant": PRIMARY_VARIANT,
                "prediction_bin": label,
                "selection_rows": count,
                "selection_engine_count": int(group["unit"].nunique()),
                "raw_mean_error": raw_bias,
                "error_std": bias_std,
                "reliability": reliability,
                "positive_bias_only": True,
                "high_prediction_guarded": not eligible,
                "correction": correction,
            }
        )
    global_count = int(len(working))
    global_bias = float(working["error"].mean()) if global_count else 0.0
    global_reliability = float(global_count / (global_count + prior_count))
    global_correction = min(cap, max(global_bias, 0.0) * global_reliability)
    rows.append(
        {
            **common,
            "calibration_variant": "global_positive",
            "prediction_bin": "all_predictions",
            "selection_rows": global_count,
            "selection_engine_count": int(working["unit"].nunique()),
            "raw_mean_error": global_bias,
            "error_std": float(working["error"].std(ddof=1)),
            "reliability": global_reliability,
            "positive_bias_only": True,
            "high_prediction_guarded": False,
            "correction": global_correction,
        }
    )
    calibration = pd.DataFrame(rows)
    working["selection_prediction_bin"] = working["prediction_bin"].astype(str)
    return calibration, working


def apply_calibration(
    frame: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    output["prediction_bin"] = prediction_bin(output["prediction"]).astype(str)
    stage = calibration[calibration["calibration_variant"] == PRIMARY_VARIANT]
    correction_lookup = stage.set_index("prediction_bin")["correction"].to_dict()
    global_row = calibration[
        calibration["calibration_variant"] == "global_positive"
    ].iloc[0]
    output["stagewise_correction"] = (
        output["prediction_bin"].map(correction_lookup).fillna(0.0).astype(float)
    )
    output["global_correction"] = float(global_row["correction"])
    output["prediction_baseline_symmetric"] = output["prediction"].astype(float)
    output["prediction_stagewise_guarded"] = (
        output["prediction_baseline_symmetric"] - output["stagewise_correction"]
    )
    output["prediction_global_positive"] = (
        output["prediction_baseline_symmetric"] - output["global_correction"]
    )
    return output


def evaluate_variants(
    frame: pd.DataFrame,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = {
        "baseline_symmetric": "prediction_baseline_symmetric",
        "stagewise_guarded": "prediction_stagewise_guarded",
        "global_positive": "prediction_global_positive",
    }
    for variant, column in columns.items():
        rows.append(
            {
                **common,
                "variant": variant,
                **metrics_for_prediction(frame, column),
                "evaluation_engine_count": int(frame["unit"].nunique()),
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
        )
    return rows


def crossfit_evaluation(
    pool_predictions: pd.DataFrame,
    protocols: dict[str, dict],
    experiment: dict,
) -> dict[str, pd.DataFrame]:
    calibration_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    confirmation_parts: list[pd.DataFrame] = []
    confirmation_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    for base_values, endpoint_rows in pool_predictions.groupby(BASE_KEYS):
        domain, model_seed, split_seed = base_values
        protocol = protocols[str(domain)]
        split = protocol["role_splits"][str(int(split_seed))]
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]
            selection_units = list(map(int, roles["selection_units"]))
            confirmation_units = list(map(int, roles["confirmation_units"]))
            if set(selection_units) & set(confirmation_units):
                raise AssertionError("selection and confirmation engines overlap")
            selection_seed_parts: list[pd.DataFrame] = []
            for endpoint_seed in experiment["selection_endpoint_seeds"]:
                assignment = a21.balanced_assignment(
                    selection_units,
                    str(domain),
                    int(split_seed),
                    int(partition),
                    int(endpoint_seed),
                    "selection",
                )
                selected = a21.endpoint_subset(
                    endpoint_rows,
                    selection_units,
                    assignment=assignment,
                ).copy()
                selected["role_partition"] = int(partition)
                selected["endpoint_seed"] = int(endpoint_seed)
                selected["evaluation_role"] = "selection"
                selection_seed_parts.append(selected)
            selection = pd.concat(selection_seed_parts, ignore_index=True)
            common_cal = {
                "target_domain": str(domain),
                "model_seed": int(model_seed),
                "target_split_seed": int(split_seed),
                "role_partition": int(partition),
            }
            calibration, selection_audit = fit_calibration(
                selection,
                common_cal,
                experiment,
            )
            calibration_parts.append(calibration)
            selection_parts.append(selection_audit)
            for endpoint_seed in experiment["confirmation_endpoint_seeds"]:
                assignment = a21.balanced_assignment(
                    confirmation_units,
                    str(domain),
                    int(split_seed),
                    int(partition),
                    int(endpoint_seed),
                    "confirmation",
                )
                selected = a21.endpoint_subset(
                    endpoint_rows,
                    confirmation_units,
                    assignment=assignment,
                ).copy()
                applied = apply_calibration(selected, calibration)
                applied["target_domain"] = str(domain)
                applied["model_seed"] = int(model_seed)
                applied["target_split_seed"] = int(split_seed)
                applied["role_partition"] = int(partition)
                applied["endpoint_seed"] = int(endpoint_seed)
                applied["evaluation_role"] = "confirmation"
                confirmation_parts.append(applied)
                common = {
                    **common_cal,
                    "endpoint_seed": int(endpoint_seed),
                    "evaluation_protocol": "balanced_endpoint",
                }
                confirmation_rows.extend(evaluate_variants(applied, common))
                high = applied[
                    applied["label"] > float(experiment["high_true_rul_threshold"])
                ]
                if high.empty:
                    raise RuntimeError("A5 confirmation cell lacks high-RUL engines")
                high_common = {
                    **common,
                    "high_true_rul_threshold": float(
                        experiment["high_true_rul_threshold"]
                    ),
                }
                high_rows.extend(evaluate_variants(high, high_common))
            for fraction in experiment["endpoint_fractions"]:
                selected = a21.endpoint_subset(
                    endpoint_rows,
                    confirmation_units,
                    fraction=float(fraction),
                )
                applied = apply_calibration(selected, calibration)
                common = {
                    **common_cal,
                    "endpoint_fraction": float(fraction),
                    "evaluation_protocol": f"fixed_endpoint_{int(round(100*fraction)):03d}",
                }
                fixed_rows.extend(evaluate_variants(applied, common))
    return {
        "calibration": pd.concat(calibration_parts, ignore_index=True),
        "selection_predictions": pd.concat(selection_parts, ignore_index=True),
        "confirmation_predictions": pd.concat(confirmation_parts, ignore_index=True),
        "confirmation_run": pd.DataFrame(confirmation_rows),
        "fixed_run": pd.DataFrame(fixed_rows),
        "high_run": pd.DataFrame(high_rows),
    }


def paired_variants(
    results: pd.DataFrame,
    candidate: str,
    keys: list[str],
) -> pd.DataFrame:
    pivot = results.pivot(index=keys, columns="variant", values=METRICS).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else column
        for column in pivot.columns
    ]
    output = pivot[keys].copy()
    for metric in METRICS:
        baseline = pivot[f"{metric}_baseline_symmetric"].astype(float)
        candidate_values = pivot[f"{metric}_{candidate}"].astype(float)
        output[f"{metric}_baseline_symmetric"] = baseline
        output[f"{metric}_{candidate}"] = candidate_values
        output[f"{metric}_delta_candidate_minus_baseline"] = candidate_values - baseline
    output["candidate"] = candidate
    output["nasa_relative_delta"] = (
        output["nasa_score_delta_candidate_minus_baseline"]
        / output["nasa_score_baseline_symmetric"]
    )
    output["rmse_relative_delta"] = (
        output["rmse_delta_candidate_minus_baseline"]
        / output["rmse_baseline_symmetric"]
    )
    output["candidate_nasa_win"] = (
        output["nasa_score_delta_candidate_minus_baseline"] < 0
    )
    output["candidate_rmse_win"] = (
        output["rmse_delta_candidate_minus_baseline"] < 0
    )
    return output.sort_values(keys)


def comparison_summary(
    paired: pd.DataFrame,
    experiment: dict,
    comparison: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in [("ALL", paired)] + list(paired.groupby("target_domain")):
        nasa_ci = a4.hierarchical_bootstrap(
            frame,
            "nasa_relative_delta",
            int(experiment["bootstrap_repetitions"]),
            a4.stable_seed(EXPERIMENT_ID, comparison, "nasa", scope),
        )
        rmse_ci = a4.hierarchical_bootstrap(
            frame,
            "rmse_relative_delta",
            int(experiment["bootstrap_repetitions"]),
            a4.stable_seed(EXPERIMENT_ID, comparison, "rmse", scope),
        )
        rows.append(
            {
                "comparison": comparison,
                "scope": scope,
                "n_records": int(len(frame)),
                "nasa_score_delta_mean": float(
                    frame["nasa_score_delta_candidate_minus_baseline"].mean()
                ),
                "nasa_improvement_pct": float(-100.0 * frame["nasa_relative_delta"].mean()),
                "nasa_relative_boot_ci95_low": nasa_ci[0],
                "nasa_relative_boot_ci95_high": nasa_ci[1],
                "nasa_win_rate": float(frame["candidate_nasa_win"].mean()),
                "rmse_delta_mean": float(
                    frame["rmse_delta_candidate_minus_baseline"].mean()
                ),
                "rmse_degradation_pct": float(100.0 * frame["rmse_relative_delta"].mean()),
                "rmse_relative_boot_ci95_low": rmse_ci[0],
                "rmse_relative_boot_ci95_high": rmse_ci[1],
                "rmse_win_rate": float(frame["candidate_rmse_win"].mean()),
                "late_error_q95_delta_mean": float(
                    frame["late_error_q95_delta_candidate_minus_baseline"].mean()
                ),
                "under_error_q95_delta_mean": float(
                    frame["under_error_q95_delta_candidate_minus_baseline"].mean()
                ),
                "mean_error_delta_mean": float(
                    frame["mean_error_delta_candidate_minus_baseline"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def high_rul_summary(
    paired: pd.DataFrame,
    experiment: dict,
) -> dict[str, Any]:
    nasa_ci = a4.hierarchical_bootstrap(
        paired,
        "nasa_relative_delta",
        int(experiment["bootstrap_repetitions"]),
        a4.stable_seed(EXPERIMENT_ID, "high_rul", "nasa"),
    )
    rmse_ci = a4.hierarchical_bootstrap(
        paired,
        "rmse_relative_delta",
        int(experiment["bootstrap_repetitions"]),
        a4.stable_seed(EXPERIMENT_ID, "high_rul", "rmse"),
    )
    return {
        "threshold": float(experiment["high_true_rul_threshold"]),
        "n_records": int(len(paired)),
        "nasa_improvement_pct": float(-100.0 * paired["nasa_relative_delta"].mean()),
        "nasa_relative_ci95": [nasa_ci[0], nasa_ci[1]],
        "rmse_degradation_pct": float(100.0 * paired["rmse_relative_delta"].mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
    }


def make_decision(
    *,
    pool_predictions: pd.DataFrame,
    confirmation_run: pd.DataFrame,
    calibration: pd.DataFrame,
    paired_primary: pd.DataFrame,
    comparisons: pd.DataFrame,
    high_rul: dict[str, Any],
    experiment: dict,
) -> dict[str, Any]:
    expected_training = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_confirmation = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
        * len(VARIANTS)
    )
    expected_pairs = expected_confirmation // len(VARIANTS)
    expected_calibration = (
        expected_training * len(experiment["role_partitions"]) * 5
    )
    primary = comparisons[
        (comparisons["comparison"] == "stagewise_guarded_vs_baseline")
        & (comparisons["scope"] == "ALL")
    ].iloc[0]
    domains = comparisons[
        (comparisons["comparison"] == "stagewise_guarded_vs_baseline")
        & (comparisons["scope"] != "ALL")
    ]
    domain_wins = int((domains["nasa_improvement_pct"] > 0).sum())
    complete = bool(
        pool_predictions["cell_id"].nunique() == expected_training
        and len(confirmation_run) == expected_confirmation
        and len(paired_primary) == expected_pairs
        and len(calibration) == expected_calibration
    )
    uncontaminated = not pool_predictions[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any()
    high_rmse_safe = bool(
        100.0 * float(high_rul["rmse_relative_ci95"][1])
        <= float(experiment["high_rul_rmse_noninferiority_margin_pct"])
    )
    high_nasa_safe = bool(
        100.0 * float(high_rul["nasa_relative_ci95"][1])
        <= float(experiment["high_rul_nasa_noninferiority_margin_pct"])
    )
    success = bool(
        complete
        and uncontaminated
        and float(primary["nasa_improvement_pct"])
        >= float(experiment["minimum_nasa_improvement_pct"])
        and float(primary["nasa_relative_boot_ci95_high"]) < 0
        and 100.0 * float(primary["rmse_relative_boot_ci95_high"])
        <= float(experiment["rmse_noninferiority_margin_pct"])
        and domain_wins >= int(experiment["minimum_nasa_domain_wins"])
        and float(primary["late_error_q95_delta_mean"]) < 0
        and high_rmse_safe
        and high_nasa_safe
    )
    stage = calibration[calibration["calibration_variant"] == PRIMARY_VARIANT]
    decision: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Does selection-only positive-bias residual calibration, applied "
            "only below predicted RUL 60, improve confirmation endpoint NASA "
            "risk while preserving overall and high-RUL RMSE/NASA safety?"
        ),
        "expected_training_cells": expected_training,
        "completed_training_cells": int(pool_predictions["cell_id"].nunique()),
        "expected_confirmation_records": expected_confirmation,
        "completed_confirmation_records": int(len(confirmation_run)),
        "expected_primary_pairs": expected_pairs,
        "completed_primary_pairs": int(len(paired_primary)),
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "prediction_bins": experiment["prediction_bin_labels"],
        "high_prediction_guard": float(experiment["high_prediction_guard"]),
        "shrinkage_prior_count": float(experiment["shrinkage_prior_count"]),
        "max_correction": float(experiment["max_correction"]),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "calibration_result": {
            "n_stage_parameters": int(len(stage)),
            "nonzero_correction_rate": float((stage["correction"] > 0).mean()),
            "mean_nonzero_correction": float(
                stage.loc[stage["correction"] > 0, "correction"].mean()
                if (stage["correction"] > 0).any()
                else 0.0
            ),
            "maximum_observed_correction": float(stage["correction"].max()),
            "high_prediction_nonzero_corrections": int(
                (
                    (stage["prediction_bin"] == "pred_ge60")
                    & (stage["correction"] != 0)
                ).sum()
            ),
        },
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
            "late_error_q95_delta_mean": float(
                primary["late_error_q95_delta_mean"]
            ),
            "under_error_q95_delta_mean": float(
                primary["under_error_q95_delta_mean"]
            ),
            "mean_error_delta_mean": float(primary["mean_error_delta_mean"]),
        },
        "high_rul_safety_result": {
            **high_rul,
            "rmse_noninferiority_passed": high_rmse_safe,
            "nasa_noninferiority_passed": high_nasa_safe,
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
                    "A5 confirmed stagewise residual calibration with high-RUL safety"
                    if success
                    else "A5 completed, but stagewise residual calibration did not meet every registered criterion"
                ),
                "next_action": (
                    "run_fresh_seed_external_holdout_confirmation"
                    if success
                    else "stop_stagewise_calibration_and_reassess_experimentA6_direction"
                ),
            }
        )
    return decision


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols, evidence = a4.load_training_only_protocol(base, experiment)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"},
        "experiment_config": experiment,
        "evidence": evidence,
        "registered_primary_question": (
            "Does selection-only positive-bias residual calibration, applied "
            "only below predicted RUL 60, improve confirmation endpoint NASA "
            "risk while preserving overall and high-RUL RMSE/NASA safety?"
        ),
        "calibration_uses_selection_labels_only": True,
        "confirmation_used_for_calibration": False,
        "A3_official_outputs_used_for_model_selection": False,
        "A4_1_results_used_for_hypothesis_generation": True,
        "A4_1_confirmation_outputs_used_for_runtime_calibration": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"existing A5 output is incompatible at {key}; use a new output directory"
                )
    atomic_json(paths["manifest"], manifest)
    selected_protocols = {
        domain: protocols[domain] for domain in experiment["domains"]
    }
    atomic_json(paths["protocol"], selected_protocols)
    a1.atomic_write_text(
        paths["engine_roles"],
        a21.protocol_rows(selected_protocols).to_csv(index=False),
    )
    expected_training = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_confirmation = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
        * len(VARIANTS)
    )
    expected_fixed = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["endpoint_fractions"])
        * len(VARIANTS)
    )
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "domains": experiment["domains"],
        "objective": experiment["objective"],
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "role_partitions": experiment["role_partitions"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "endpoint_seed_sets_disjoint": True,
        "prediction_bins": experiment["prediction_bin_labels"],
        "calibration_rule": (
            "max(0, mean_error) * n/(n+30), capped at 3; "
            "correction=0 for predicted RUL>=60"
        ),
        "expected_training_cells": expected_training,
        "expected_confirmation_records": expected_confirmation,
        "expected_fixed_endpoint_records": expected_fixed,
        "fixed_budget_epoch": int(experiment["target_epochs"]),
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
        raise RuntimeError(
            "A5 contains an interrupted run; use --resume or a new output directory"
        )
    tasks = [
        (domain, seed)
        for domain in experiment["domains"]
        for seed in experiment["model_seeds"]
    ]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    pool_predictions = merged["pool_predictions"].sort_values(
        BASE_KEYS + ["unit", "endpoint_fraction"]
    )
    if pool_predictions["cell_id"].nunique() != expected_training:
        raise RuntimeError("A5 merged pool predictions are incomplete")
    if pool_predictions[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any():
        raise RuntimeError("A5 detected official-test contamination")
    evaluated = crossfit_evaluation(
        pool_predictions,
        selected_protocols,
        experiment,
    )
    confirmation_run = evaluated["confirmation_run"].sort_values(
        PAIR_KEYS + ["variant"]
    )
    fixed_run = evaluated["fixed_run"].sort_values(
        CALIBRATION_KEYS + ["endpoint_fraction", "variant"]
    )
    if len(confirmation_run) != expected_confirmation:
        raise RuntimeError("A5 confirmation output is incomplete")
    if len(fixed_run) != expected_fixed:
        raise RuntimeError("A5 fixed-endpoint output is incomplete")
    paired_primary = paired_variants(
        confirmation_run,
        PRIMARY_VARIANT,
        PAIR_KEYS,
    )
    paired_global = paired_variants(
        confirmation_run,
        "global_positive",
        PAIR_KEYS,
    )
    fixed_primary = paired_variants(
        fixed_run,
        PRIMARY_VARIANT,
        CALIBRATION_KEYS + ["endpoint_fraction"],
    )
    high_primary = paired_variants(
        evaluated["high_run"],
        PRIMARY_VARIANT,
        PAIR_KEYS,
    )
    high_global = paired_variants(
        evaluated["high_run"],
        "global_positive",
        PAIR_KEYS,
    )
    primary_comparison = comparison_summary(
        paired_primary,
        experiment,
        "stagewise_guarded_vs_baseline",
    )
    global_comparison = comparison_summary(
        paired_global,
        experiment,
        "global_positive_vs_baseline",
    )
    comparisons = pd.concat(
        [primary_comparison, global_comparison], ignore_index=True
    )
    high_summary = high_rul_summary(high_primary, experiment)
    decision = make_decision(
        pool_predictions=pool_predictions,
        confirmation_run=confirmation_run,
        calibration=evaluated["calibration"],
        paired_primary=paired_primary,
        comparisons=comparisons,
        high_rul=high_summary,
        experiment=experiment,
    )
    a1.atomic_write_text(paths["pool_predictions"], pool_predictions.to_csv(index=False))
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    inventory = merged["inventory"].drop_duplicates(
        ["target_domain", "model_seed"]
    )
    a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False))
    a1.atomic_write_text(
        paths["calibration"], evaluated["calibration"].to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["selection_predictions"],
        evaluated["selection_predictions"].to_csv(index=False),
    )
    a1.atomic_write_text(
        paths["confirmation_predictions"],
        evaluated["confirmation_predictions"].to_csv(index=False),
    )
    a1.atomic_write_text(paths["confirmation_run"], confirmation_run.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_run"], fixed_run.to_csv(index=False))
    a1.atomic_write_text(paths["paired_primary"], paired_primary.to_csv(index=False))
    a1.atomic_write_text(paths["paired_global"], paired_global.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_primary"], fixed_primary.to_csv(index=False))
    a1.atomic_write_text(paths["high_rul_primary"], high_primary.to_csv(index=False))
    a1.atomic_write_text(paths["high_rul_global"], high_global.to_csv(index=False))
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
