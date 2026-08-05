"""Experiment A8: causal cycle-age representation validation.

A4--A7 show that changing the loss or post-processing can exchange high-RUL
underprediction for low-RUL overprediction, but does not make a shared target
head stage-aware.  A8 changes the *input representation* only.

The candidate appends a causal age feature to each existing sensor/settings
window:

    causal_cycle_age_z = (log1p(cycle) - source_mean) / source_std

``source_mean`` and ``source_std`` are fitted on source-domain training cycles
only.  A unit's final cycle, true RUL at deployment, future observations and
official C-MAPSS test files are never read.  The baseline and candidate both
use symmetric MSE, the same 10 target epochs and the A2_1 role protocol.

Unlike A4--A7, both representations are freshly source-pretrained inside A8.
This makes the representation comparison self-contained: no A2 source model
state is reused as a computational shortcut.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.rul_generator import add_train_rul  # noqa: E402
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402
from scripts.experiment8_transfer_baseline import train_source_supervised  # noqa: E402


SCRIPT_VERSION = "experimentA8_causal_cycle_age_representation_validation_v1"
EXPERIMENT_ID = "experimentA8"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
REPRESENTATIONS = ("baseline_sensor_settings", "causal_cycle_age")
BASELINE = "baseline_sensor_settings"
CANDIDATE = "causal_cycle_age"
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
SELECTION_ENDPOINT_SEEDS = list(range(8401, 8406))
CONFIRMATION_ENDPOINT_SEEDS = list(range(8501, 8506))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
HIGH_RUL_THRESHOLD = 60.0
AGE_FEATURE = "causal_cycle_age_z"
DEFAULT_OUTPUT = "outputs/experimentA8_causal_cycle_age_representation_validation"
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
        description="Experiment A8: causal cycle-age representation validation"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,1,3")
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
        raise FileNotFoundError(f"required A8 input is missing: {path}")
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
        "experiment_name": "causal_cycle_age_representation_validation",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "representations": list(REPRESENTATIONS),
        "baseline_representation": BASELINE,
        "candidate_representation": CANDIDATE,
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "high_rul_threshold": HIGH_RUL_THRESHOLD,
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "fixed_budget_no_epoch_selection": True,
        "fresh_source_pretraining_for_both_representations": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "overall_noninferiority_margin_pct": 3.0,
        "stage_noninferiority_margin_pct": 3.0,
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
                "selection_endpoint_seeds": [8401],
                "confirmation_endpoint_seeds": [8501],
                "target_epochs": 2,
                "source_pretrain_steps": 20,
                "bootstrap_repetitions": 100,
                "quick_mode": True,
            }
        )
        base["target_epochs"] = 2
        base["source_pretrain_steps"] = 20
        if args.output_dir is None:
            base["output_dir"] = resolved(None, DEFAULT_OUTPUT + "_quick")
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if experiment["architecture"] != ARCHITECTURE:
        raise ValueError(f"A8 requires architecture={ARCHITECTURE}")
    if tuple(experiment["representations"]) != REPRESENTATIONS:
        raise ValueError("A8 representation set is locked")
    if experiment["preprocessing"] != "condition_settings":
        raise ValueError("A8 requires condition_settings preprocessing")
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
            raise ValueError(f"A8 has empty/duplicate values in {name}")
    for domain in experiment["domains"]:
        if not a1.train_path(base["data_dir"], domain).is_file():
            raise FileNotFoundError(f"missing training file for {domain}")


def root_paths(output: Path) -> dict[str, Path]:
    prefix = EXPERIMENT_ID
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "causality": output / f"{prefix}_feature_causality_audit.json",
        "age_audit": output / f"{prefix}_age_feature_audit.csv",
        "source_inventory": output / f"{prefix}_source_pretrain_inventory.csv",
        "source_history": output / f"{prefix}_source_pretrain_history.csv",
        "endpoint_predictions": output / f"{prefix}_pool_endpoint_predictions.csv",
        "target_history": output / f"{prefix}_target_history.csv",
        "selection_predictions": output / f"{prefix}_selection_endpoint_predictions.csv",
        "confirmation_predictions": output / f"{prefix}_confirmation_endpoint_predictions.csv",
        "selection_run": output / f"{prefix}_selection_run_level.csv",
        "confirmation_run": output / f"{prefix}_confirmation_run_level.csv",
        "fixed_run": output / f"{prefix}_fixed_endpoint_run_level.csv",
        "paired": output / f"{prefix}_paired_age_vs_baseline.csv",
        "fixed_paired": output / f"{prefix}_fixed_endpoint_paired_age_vs_baseline.csv",
        "high_paired": output / f"{prefix}_high_rul_paired_age_vs_baseline.csv",
        "low_paired": output / f"{prefix}_low_rul_paired_age_vs_baseline.csv",
        "comparison": output / f"{prefix}_comparison_summary.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
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
        "target_history": directory / "target_history.csv",
        "source_history": directory / "source_pretrain_history.csv",
        "source_inventory": directory / "source_pretrain_inventory.csv",
        "age_audit": directory / "age_feature_audit.csv",
    }


def source_cache_path(output: Path, domain: str, model_seed: int, representation: str) -> Path:
    return (
        shard_dir(output, domain, model_seed)
        / "source_cache"
        / f"{EXPERIMENT_ID}_{representation}_{domain}_mseed{model_seed:03d}.pt"
    )


def training_cell_id(
    domain: str, representation: str, model_seed: int, split_seed: int
) -> str:
    return (
        f"{EXPERIMENT_ID}_{domain.lower()}_{representation}_"
        f"mseed{model_seed:03d}_tsplit{split_seed}"
    )


def fit_cycle_age(source_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    values = np.log1p(
        pd.concat(list(source_frames.values()), ignore_index=True)["cycle"].to_numpy(
            dtype=np.float64
        )
    )
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
        raise ValueError("source-only cycle-age fit is invalid")
    return {
        "feature": AGE_FEATURE,
        "transform": "log1p(cycle)",
        "fit_scope": "source_training_rows_only",
        "source_log1p_cycle_mean": mean,
        "source_log1p_cycle_std": std,
        "source_cycle_min": int(min(frame["cycle"].min() for frame in source_frames.values())),
        "source_cycle_max": int(max(frame["cycle"].max() for frame in source_frames.values())),
        "source_row_count": int(sum(len(frame) for frame in source_frames.values())),
        "uses_unit_max_cycle": False,
        "uses_true_rul_as_feature": False,
        "uses_future_windows": False,
        "uses_official_test": False,
    }


def append_age_feature(frame: pd.DataFrame, age_spec: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    transformed = np.log1p(output["cycle"].to_numpy(dtype=np.float64))
    output[AGE_FEATURE] = (
        (transformed - float(age_spec["source_log1p_cycle_mean"]))
        / float(age_spec["source_log1p_cycle_std"])
    ).astype(np.float32)
    if not np.isfinite(output[AGE_FEATURE].to_numpy(dtype=float)).all():
        raise ValueError("A8 age feature contains NaN/Inf")
    return output


def prepare_representation_data(cfg: dict, representation: str) -> dict[str, Any]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown A8 representation: {representation}")
    sensors = list(cfg["sensor_columns"])
    source_raw, normalizer = a1.fit_source_normalizer_train_only(
        cfg, "condition_settings"
    )
    age_spec = fit_cycle_age(source_raw)
    source_frames = {
        domain: normalizer.transform(frame, sensors)
        for domain, frame in source_raw.items()
    }
    target_raw = add_train_rul(
        a1.load_train_domain(cfg["data_dir"], cfg["target_domain"]), cfg["rul_cap"]
    )
    target_frame = normalizer.transform(target_raw, sensors)
    features = sensors + list(a1.SETTING_FEATURE_COLUMNS)
    if representation == CANDIDATE:
        source_frames = {
            domain: append_age_feature(frame, age_spec)
            for domain, frame in source_frames.items()
        }
        target_frame = append_age_feature(target_frame, age_spec)
        features = features + [AGE_FEATURE]
    audit = {
        "representation": representation,
        "age_feature_included": representation == CANDIDATE,
        "feature_count": int(len(features)),
        "feature_columns": json.dumps(features, ensure_ascii=False),
        **age_spec,
        "target_domain": str(cfg["target_domain"]),
        "source_domains": json.dumps(list(cfg["source_domains"]), ensure_ascii=False),
        "target_cycle_used_for_feature_transform_only": representation == CANDIDATE,
        "target_cycle_used_for_fit": False,
        "normalizer_fit_scope": "source_training_rows_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return {
        "source_frames": source_frames,
        "target_frame": target_frame,
        "features": features,
        "age_spec": age_spec,
        "audit": audit,
    }


def make_source_tasks(data: dict[str, Any], cfg: dict, experiment: dict) -> dict[str, Any]:
    return {
        domain: a1.make_loader(
            frame,
            data["features"],
            cfg,
            training=True,
            balance_mode=experiment["balance_mode"],
            loader_seed=int(cfg["seed"]) + 1000 * (index + 1),
        )
        for index, (domain, frame) in enumerate(data["source_frames"].items())
    }


def prepare_support_pool(
    data: dict[str, Any],
    cfg: dict,
    experiment: dict,
    support_units: list[int],
    pool_units: list[int],
) -> tuple[Any, Any]:
    target = data["target_frame"]
    support = target[target["unit"].isin(support_units)].copy()
    pool = target[target["unit"].isin(pool_units)].copy()
    if support["unit"].nunique() != len(support_units):
        raise ValueError("A8 support engines are incomplete")
    if pool["unit"].nunique() != len(pool_units):
        raise ValueError("A8 evaluation-pool engines are incomplete")
    return (
        a1.make_loader(
            support,
            data["features"],
            cfg,
            training=True,
            balance_mode=experiment["balance_mode"],
            loader_seed=int(cfg["seed"]) + 9000,
        ),
        a1.make_loader(
            pool,
            data["features"],
            cfg,
            training=False,
            loader_seed=int(cfg["seed"]) + 9200,
        ),
    )


def source_signature(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    representation: str,
    model_seed: int,
    feature_count: int,
    prior: torch.Tensor,
    audit: dict[str, Any],
) -> str:
    return a1.canonical_hash(
        {
            "script_version": SCRIPT_VERSION,
            "script_hash": a1.file_sha256(Path(__file__)),
            "git_commit": a1.git_commit(PROJECT_ROOT),
            "architecture": ARCHITECTURE,
            "representation": representation,
            "model_seed": int(model_seed),
            "target_domain": base["target_domain"],
            "source_domains": list(base["source_domains"]),
            "feature_count": int(feature_count),
            "source_age_audit": audit,
            "sensor_columns": list(base["sensor_columns"]),
            "window_size": int(base["window_size"]),
            "window_stride": int(base["window_stride"]),
            "rul_cap": int(base["rul_cap"]),
            "batch_size": int(base["batch_size"]),
            "hidden_dim": int(base["hidden_dim"]),
            "embedding_dim": int(base["embedding_dim"]),
            "gat_heads": int(base["gat_heads"]),
            "dropout": float(base["dropout"]),
            "preprocessing": experiment["preprocessing"],
            "balance_mode": experiment["balance_mode"],
            "source_pretrain_steps": int(base["source_pretrain_steps"]),
            "source_pretrain_lr": float(base["source_pretrain_lr"]),
            "source_pretrain_weight_decay": float(base["source_pretrain_weight_decay"]),
            "train_file_hashes": protocol["train_file_hashes"],
            "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
        }
    )


def train_or_load_source(
    *,
    output: Path,
    base: dict,
    experiment: dict,
    protocol: dict,
    representation: str,
    model_seed: int,
    data: dict[str, Any],
    prior: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict], dict[str, Any]]:
    cfg = deepcopy(base)
    cfg["seed"] = int(model_seed)
    signature = source_signature(
        base=cfg,
        experiment=experiment,
        protocol=protocol,
        representation=representation,
        model_seed=model_seed,
        feature_count=len(data["features"]),
        prior=prior,
        audit=data["audit"],
    )
    cache_path = source_cache_path(
        output, str(cfg["target_domain"]), int(model_seed), representation
    )
    if cache_path.is_file():
        cached = a1.safe_torch_load(cache_path)
        if cached.get("signature") == signature:
            inventory = dict(cached["inventory"])
            inventory["source_cache_origin"] = "experimentA8_fresh_source_cache"
            inventory["source_cache_path"] = str(cache_path)
            return cached["state"], cached.get("history", []), inventory
    source_tasks = make_source_tasks(data, cfg, experiment)
    a1.seed_everything(int(model_seed))
    model = exp17b.build_model_17b(
        ARCHITECTURE, len(data["features"]), cfg, prior, prior
    )
    total, predictor = exp17.parameter_count(model)
    device = a1.resolve_device(cfg["device"])
    model, history = train_source_supervised(model, source_tasks, cfg, device)
    inventory = {
        "representation": representation,
        "model": ARCHITECTURE,
        "model_seed": int(model_seed),
        "feature_count": int(len(data["features"])),
        "feature_columns": json.dumps(data["features"], ensure_ascii=False),
        "total_parameter_count": int(total),
        "predictor_parameter_count": int(predictor),
        "source_pretrain_steps": int(cfg["source_pretrain_steps"]),
        "source_signature": signature,
        "source_cache_origin": "experimentA8_fresh_source_cache",
        "source_cache_path": str(cache_path),
        "source_pretraining_reused_from_prior_experiment": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"signature": signature, "state": a1.state_to_cpu(model), "history": history, "inventory": inventory},
        cache_path,
    )
    state = a1.state_to_cpu(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, history, inventory


def run_training_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    representation: str,
    model_seed: int,
    split_seed: int,
    data: dict[str, Any],
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    source_inventory: dict[str, Any],
    prior: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain = str(protocol["target_domain"])
    run_seed = a2.target_run_seed(domain, int(model_seed), int(split_seed))
    cfg = deepcopy(base)
    cfg.update(
        {
            "seed": int(run_seed),
            "target_domain": domain,
            "source_domains": protocol["source_domains"],
        }
    )
    split = protocol["role_splits"][str(split_seed)]
    support_units = list(map(int, split["adaptation_units"]))
    pool_units = list(map(int, split["evaluation_pool_units"]))
    support, pool = prepare_support_pool(
        data, cfg, experiment, support_units, pool_units
    )
    a1.seed_everything(int(run_seed))
    model = exp17b.build_model_17b(
        ARCHITECTURE, len(data["features"]), cfg, prior, prior
    )
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
    endpoints = a21.endpoint_epoch_rows(predictions, int(experiment["target_epochs"]))
    cell_id = training_cell_id(domain, representation, model_seed, split_seed)
    common = {
        "experiment_id": EXPERIMENT_ID,
        "cell_id": cell_id,
        "target_domain": domain,
        "model": ARCHITECTURE,
        "representation": representation,
        "model_seed": int(model_seed),
        "target_split_seed": int(split_seed),
        "target_run_seed": int(run_seed),
        "k": int(experiment["k"]),
        "adaptation_units": json.dumps(support_units, ensure_ascii=False),
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "feature_count": int(len(data["features"])),
        "feature_columns": json.dumps(data["features"], ensure_ascii=False),
        "source_signature": source_inventory["source_signature"],
        "source_history_rows": int(len(source_history)),
        "source_cache_origin": source_inventory["source_cache_origin"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for column, value in reversed(list(common.items())):
        endpoints.insert(0, column, value)
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", cell_id)
    history.insert(2, "target_domain", domain)
    history.insert(3, "representation", representation)
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return endpoints, history


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    state = {
        "completed": completed,
        "endpoints": load_csv(paths["endpoints"]),
        "target_history": load_csv(paths["target_history"]),
        "source_history": load_csv(paths["source_history"]),
        "source_inventory": load_csv(paths["source_inventory"]),
        "age_audit": load_csv(paths["age_audit"]),
    }
    for name in ("endpoints", "target_history"):
        if not state[name].empty:
            state[name] = state[name][state[name]["cell_id"].isin(completed)]
    return state


def save_worker_state(paths: dict[str, Path], state: dict[str, Any], expected: int) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in (
        "endpoints",
        "target_history",
        "source_history",
        "source_inventory",
        "age_audit",
    ):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected,
            "endpoint_rows": int(len(state["endpoints"])),
            "source_representation_count": int(len(state["source_inventory"])),
            "complete": len(state["completed"]) == expected,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A8 worker")
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
        "model_seed": int(model_seed),
        "protocol_hash": protocol["protocol_hash"],
        "evidence_hashes": evidence["a2_1_input_hashes"],
        "graph_fit": graph_fit,
        "fresh_source_pretraining_for_both_representations": True,
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
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A8 shard incompatible at {key}; use a new output directory")
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
    expected = len(REPRESENTATIONS) * len(experiment["target_split_seeds"])
    data_by_representation: dict[str, dict[str, Any]] = {}
    source_by_representation: dict[str, tuple[dict, list, dict]] = {}
    for representation in REPRESENTATIONS:
        data = prepare_representation_data(worker_base, representation)
        data_by_representation[representation] = data
        source_state, source_history, source_inventory = train_or_load_source(
            output=output,
            base=worker_base,
            experiment=experiment,
            protocol=protocol,
            representation=representation,
            model_seed=model_seed,
            data=data,
            prior=prior,
        )
        source_by_representation[representation] = (
            source_state,
            source_history,
            source_inventory,
        )
        source_row = {
            "target_domain": domain,
            "model_seed": int(model_seed),
            **source_inventory,
        }
        state["source_inventory"] = pd.concat(
            [
                state["source_inventory"].loc[
                    state["source_inventory"].get("representation", pd.Series(dtype=str)) != representation
                ],
                pd.DataFrame([source_row]),
            ],
            ignore_index=True,
        )
        audit_row = {"target_domain": domain, "model_seed": int(model_seed), **data["audit"]}
        state["age_audit"] = pd.concat(
            [
                state["age_audit"].loc[
                    state["age_audit"].get("representation", pd.Series(dtype=str)) != representation
                ],
                pd.DataFrame([audit_row]),
            ],
            ignore_index=True,
        )
        history = pd.DataFrame(source_history)
        if not history.empty:
            history.insert(0, "experiment_id", EXPERIMENT_ID)
            history.insert(1, "target_domain", domain)
            history.insert(2, "representation", representation)
            history.insert(3, "model_seed", int(model_seed))
            state["source_history"] = pd.concat(
                [
                    state["source_history"].loc[
                        ~(
                            (state["source_history"].get("representation", pd.Series(dtype=str)) == representation)
                            & (state["source_history"].get("model_seed", pd.Series(dtype=int)) == int(model_seed))
                        )
                    ],
                    history,
                ],
                ignore_index=True,
            )
    pending = [
        (representation, int(split_seed))
        for representation in REPRESENTATIONS
        for split_seed in experiment["target_split_seeds"]
        if training_cell_id(domain, representation, model_seed, int(split_seed))
        not in state["completed"]
    ]
    for representation, split_seed in pending:
        source_state, source_history, source_inventory = source_by_representation[representation]
        endpoints, history = run_training_cell(
            base=worker_base,
            experiment=experiment,
            protocol=protocol,
            representation=representation,
            model_seed=model_seed,
            split_seed=split_seed,
            data=data_by_representation[representation],
            source_state=deepcopy(source_state),
            source_history=source_history,
            source_inventory=source_inventory,
            prior=prior,
        )
        state["endpoints"] = pd.concat([state["endpoints"], endpoints], ignore_index=True)
        state["target_history"] = pd.concat([state["target_history"], history], ignore_index=True)
        state["completed"].add(training_cell_id(domain, representation, model_seed, split_seed))
        save_worker_state(paths, state, expected)
    save_worker_state(paths, state, expected)
    print(paths["status"].read_text(encoding="utf-8"))


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
            raise RuntimeError("no idle GPU met A8 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(
        json.dumps(
            {
                "scheduler": EXPERIMENT_ID,
                "tasks": [{"domain": domain, "seed": seed} for domain, seed in tasks],
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
            handle = log_path.open("a", encoding="utf-8")
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
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[device] = {
                "process": process,
                "domain": domain,
                "seed": seed,
                "handle": handle,
                "log_path": log_path,
            }
            print(f"[A8] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished: list[str | int] = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["handle"].close()
            if code != 0:
                tail = "\n".join(
                    record["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                )
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"A8 worker failed domain={record['domain']} seed={record['seed']} exit={code}\n{tail}"
                )
            print(f"[A8] completed domain={record['domain']} seed={record['seed']} device={device}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]], experiment: dict) -> dict[str, pd.DataFrame]:
    names = ("endpoints", "target_history", "source_history", "source_inventory", "age_audit")
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    expected = len(REPRESENTATIONS) * len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or int(status.get("completed_training_cells", -1)) != expected:
            raise RuntimeError(f"incomplete A8 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get("official_test_forward_run"):
            raise RuntimeError(f"official-test contamination: {paths['status']}")
        for name in names:
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
    for values, frame in endpoints.groupby(
        ["target_domain", "model_seed", "target_split_seed", "representation"]
    ):
        domain, model_seed, split_seed, representation = values
        protocol = protocols[str(domain)]
        split = protocol["role_splits"][str(int(split_seed))]
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]
            common = {
                "target_domain": str(domain),
                "model_seed": int(model_seed),
                "target_split_seed": int(split_seed),
                "representation": str(representation),
                "role_partition": int(partition),
                "fixed_budget_epoch": int(experiment["target_epochs"]),
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
            for role, units, seeds, output_parts, output_rows in (
                ("selection", list(map(int, roles["selection_units"])), experiment["selection_endpoint_seeds"], selection_parts, selection_rows),
                ("confirmation", list(map(int, roles["confirmation_units"])), experiment["confirmation_endpoint_seeds"], confirmation_parts, confirmation_rows),
            ):
                for endpoint_seed in seeds:
                    assignment = a21.balanced_assignment(
                        units,
                        str(domain),
                        int(split_seed),
                        int(partition),
                        int(endpoint_seed),
                        role,
                    )
                    selected = a21.endpoint_subset(frame, units, assignment=assignment).copy()
                    selected["role_partition"] = int(partition)
                    selected["endpoint_seed"] = int(endpoint_seed)
                    selected["evaluation_role"] = role
                    output_parts.append(selected)
                    output_rows.append(
                        {
                            **common,
                            "endpoint_seed": int(endpoint_seed),
                            "evaluation_role": role,
                            "evaluation_protocol": "balanced_endpoint",
                            **evaluate_objective(selected),
                            "evaluation_engine_count": int(selected["unit"].nunique()),
                        }
                    )
            confirmation_units = list(map(int, roles["confirmation_units"]))
            for fraction in experiment["endpoint_fractions"]:
                selected = a21.endpoint_subset(
                    frame, confirmation_units, fraction=float(fraction)
                )
                fixed_rows.append(
                    {
                        **common,
                        "endpoint_fraction": float(fraction),
                        "evaluation_protocol": f"fixed_endpoint_{int(round(100 * float(fraction))):03d}",
                        **evaluate_objective(selected),
                        "evaluation_engine_count": int(selected["unit"].nunique()),
                    }
                )
    return {
        "selection_predictions": pd.concat(selection_parts, ignore_index=True),
        "confirmation_predictions": pd.concat(confirmation_parts, ignore_index=True),
        "selection_run": pd.DataFrame(selection_rows),
        "confirmation_run": pd.DataFrame(confirmation_rows),
        "fixed_run": pd.DataFrame(fixed_rows),
    }


def paired_representations(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pivot = results.pivot(index=keys, columns="representation", values=METRICS).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else column
        for column in pivot.columns
    ]
    output = pivot[keys].copy()
    for metric in METRICS:
        baseline = pivot[f"{metric}_{BASELINE}"].astype(float)
        candidate = pivot[f"{metric}_{CANDIDATE}"].astype(float)
        output[f"{metric}_{BASELINE}"] = baseline
        output[f"{metric}_{CANDIDATE}"] = candidate
        output[f"{metric}_delta_candidate_minus_baseline"] = candidate - baseline
    output["candidate"] = CANDIDATE
    output["nasa_relative_delta"] = output["nasa_score_delta_candidate_minus_baseline"] / output[f"nasa_score_{BASELINE}"]
    output["rmse_relative_delta"] = output["rmse_delta_candidate_minus_baseline"] / output[f"rmse_{BASELINE}"]
    output["candidate_nasa_win"] = output["nasa_score_delta_candidate_minus_baseline"] < 0
    output["candidate_rmse_win"] = output["rmse_delta_candidate_minus_baseline"] < 0
    return output.sort_values(keys)


def stage_pairs(predictions: pd.DataFrame, high: bool, experiment: dict) -> pd.DataFrame:
    threshold = float(experiment["high_rul_threshold"])
    stage = predictions[predictions["label"] > threshold].copy() if high else predictions[predictions["label"] <= threshold].copy()
    row_keys = PAIR_KEYS + ["unit", "endpoint_fraction", "unit_window_index", "label"]
    pivot = stage.pivot(
        index=row_keys,
        columns="representation",
        values=["prediction", "error", "nasa_contribution"],
    ).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else column
        for column in pivot.columns
    ]
    rows: list[dict[str, Any]] = []
    for values, group in pivot.groupby(PAIR_KEYS):
        row = dict(zip(PAIR_KEYS, values))
        row["rul_stage"] = "high_rul_gt60" if high else "low_or_mid_rul_le60"
        row["rul_threshold"] = threshold
        row["stage_engine_count"] = int(group["unit"].nunique())
        labels = group["label"].to_numpy(dtype=float)
        for representation in REPRESENTATIONS:
            prediction = group[f"prediction_{representation}"].to_numpy(dtype=float)
            evaluated = pd.DataFrame(
                {
                    "label": labels,
                    "prediction": prediction,
                    "error": prediction - labels,
                }
            )
            metrics = a4.endpoint_risk_metrics(evaluated)
            for metric in METRICS:
                row[f"{metric}_{representation}"] = float(metrics[metric])
        for metric in METRICS:
            row[f"{metric}_delta_candidate_minus_baseline"] = row[f"{metric}_{CANDIDATE}"] - row[f"{metric}_{BASELINE}"]
        row["nasa_relative_delta"] = row["nasa_score_delta_candidate_minus_baseline"] / row[f"nasa_score_{BASELINE}"]
        row["rmse_relative_delta"] = row["rmse_delta_candidate_minus_baseline"] / row[f"rmse_{BASELINE}"]
        row["candidate_nasa_win"] = row["nasa_score_delta_candidate_minus_baseline"] < 0
        row["candidate_rmse_win"] = row["rmse_delta_candidate_minus_baseline"] < 0
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS)
    expected = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
    )
    if len(output) != expected:
        raise RuntimeError("A8 confirmation cells lack requested RUL-stage observations")
    return output


def comparison_summary(paired: pd.DataFrame, experiment: dict, comparison: str) -> pd.DataFrame:
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
                "nasa_score_delta_mean": float(frame["nasa_score_delta_candidate_minus_baseline"].mean()),
                "nasa_improvement_pct": float(-100 * frame["nasa_relative_delta"].mean()),
                "nasa_relative_boot_ci95_low": nasa_ci[0],
                "nasa_relative_boot_ci95_high": nasa_ci[1],
                "nasa_win_rate": float(frame["candidate_nasa_win"].mean()),
                "rmse_delta_mean": float(frame["rmse_delta_candidate_minus_baseline"].mean()),
                "rmse_degradation_pct": float(100 * frame["rmse_relative_delta"].mean()),
                "rmse_relative_boot_ci95_low": rmse_ci[0],
                "rmse_relative_boot_ci95_high": rmse_ci[1],
                "rmse_win_rate": float(frame["candidate_rmse_win"].mean()),
                "late_error_q95_delta_mean": float(frame["late_error_q95_delta_candidate_minus_baseline"].mean()),
                "under_error_q95_delta_mean": float(frame["under_error_q95_delta_candidate_minus_baseline"].mean()),
                "mean_error_delta_mean": float(frame["mean_error_delta_candidate_minus_baseline"].mean()),
            }
        )
    return pd.DataFrame(rows)


def stage_summary(paired: pd.DataFrame, experiment: dict, label: str) -> dict[str, Any]:
    nasa_ci = a4.hierarchical_bootstrap(
        paired,
        "nasa_relative_delta",
        int(experiment["bootstrap_repetitions"]),
        a4.stable_seed(EXPERIMENT_ID, label, "nasa"),
    )
    rmse_ci = a4.hierarchical_bootstrap(
        paired,
        "rmse_relative_delta",
        int(experiment["bootstrap_repetitions"]),
        a4.stable_seed(EXPERIMENT_ID, label, "rmse"),
    )
    return {
        "stage": label,
        "n_records": int(len(paired)),
        "nasa_improvement_pct": float(-100 * paired["nasa_relative_delta"].mean()),
        "nasa_relative_ci95": [nasa_ci[0], nasa_ci[1]],
        "rmse_degradation_pct": float(100 * paired["rmse_relative_delta"].mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
        "mean_error_delta_mean": float(paired["mean_error_delta_candidate_minus_baseline"].mean()),
        "nasa_win_rate": float(paired["candidate_nasa_win"].mean()),
        "rmse_win_rate": float(paired["candidate_rmse_win"].mean()),
    }


def make_decision(
    *,
    endpoints: pd.DataFrame,
    inventory: pd.DataFrame,
    confirmation: pd.DataFrame,
    paired: pd.DataFrame,
    comparisons: pd.DataFrame,
    high: dict[str, Any],
    low: dict[str, Any],
    experiment: dict,
) -> dict[str, Any]:
    expected_training = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
        * len(REPRESENTATIONS)
    )
    expected_confirmation = expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    expected_pairs = expected_confirmation // len(REPRESENTATIONS)
    primary = comparisons[
        (comparisons["comparison"] == "full_endpoint_age_vs_baseline")
        & (comparisons["scope"] == "ALL")
    ].iloc[0]
    complete = bool(
        endpoints["cell_id"].nunique() == expected_training
        and len(confirmation) == expected_confirmation
        and len(paired) == expected_pairs
        and len(inventory) == len(experiment["domains"]) * len(experiment["model_seeds"]) * len(REPRESENTATIONS)
    )
    uncontaminated = not endpoints[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any()
    primary_improved = bool(
        float(primary["nasa_relative_boot_ci95_high"]) < 0
        or float(primary["rmse_relative_boot_ci95_high"]) < 0
    )
    margin = float(experiment["stage_noninferiority_margin_pct"]) / 100.0
    high_safe = bool(
        high["nasa_relative_ci95"][1] <= margin
        and high["rmse_relative_ci95"][1] <= margin
    )
    low_safe = bool(
        low["nasa_relative_ci95"][1] <= margin
        and low["rmse_relative_ci95"][1] <= margin
    )
    passed = bool(complete and uncontaminated and primary_improved and high_safe and low_safe)
    return {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": "Does a source-standardized causal cycle-age input channel improve balanced-endpoint RUL prediction while preserving high- and low/mid-RUL safety?",
        "architecture": ARCHITECTURE,
        "baseline_representation": BASELINE,
        "candidate_representation": CANDIDATE,
        "expected_training_cells": expected_training,
        "completed_training_cells": int(endpoints["cell_id"].nunique()),
        "expected_confirmation_records": expected_confirmation,
        "completed_confirmation_records": int(len(confirmation)),
        "expected_primary_pairs": expected_pairs,
        "completed_primary_pairs": int(len(paired)),
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "fresh_source_pretraining_for_both_representations": True,
        "age_feature": {
            "name": AGE_FEATURE,
            "transform": "log1p(cycle), source-standardized",
            "uses_unit_max_cycle": False,
            "uses_target_cycle_for_fit": False,
            "uses_true_rul_at_deployment": False,
            "uses_future_windows": False,
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "full_endpoint_result": {
            "nasa_improvement_pct": float(primary["nasa_improvement_pct"]),
            "nasa_relative_ci95": [float(primary["nasa_relative_boot_ci95_low"]), float(primary["nasa_relative_boot_ci95_high"])],
            "rmse_degradation_pct": float(primary["rmse_degradation_pct"]),
            "rmse_relative_ci95": [float(primary["rmse_relative_boot_ci95_low"]), float(primary["rmse_relative_boot_ci95_high"])],
            "at_least_one_metric_strictly_improved": primary_improved,
        },
        "high_rul_safety_result": {**high, "noninferiority_passed": high_safe},
        "low_rul_safety_result": {**low, "noninferiority_passed": low_safe},
        "passed": passed if not experiment["quick_mode"] else complete,
        "reason": (
            "quick smoke run only; do not interpret scientifically"
            if experiment["quick_mode"]
            else (
                "A8 confirmed a causal cycle-age representation benefit with stage safety"
                if passed
                else "A8 completed, but the causal cycle-age representation did not meet every registered efficacy/safety criterion"
            )
        ),
        "next_action": (
            None
            if experiment["quick_mode"]
            else (
                "run_fresh_seed_replication_before_any_official_test_confirmation"
                if passed
                else "stop_single_channel_representation_extension_and_reassess_multitask_or_domain_generalization_direction"
            )
        ),
    }


def write_initial_artifacts(
    paths: dict[str, Path],
    base: dict,
    experiment: dict,
    evidence: dict,
) -> dict[str, Any]:
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"},
        "experiment_config": experiment,
        "evidence": evidence,
        "registered_primary_question": "Does a source-standardized causal cycle-age input channel improve balanced-endpoint RUL prediction while preserving high- and low/mid-RUL safety?",
        "fresh_source_pretraining_for_both_representations": True,
        "candidate_uses_causal_observed_cycle_at_deployment": True,
        "candidate_uses_unit_max_cycle": False,
        "candidate_uses_true_rul_at_deployment": False,
        "candidate_uses_future_windows": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        existing = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A8 output is incompatible at {key}; use a new output directory")
    atomic_json(paths["manifest"], manifest)
    return manifest


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols, evidence = a4.load_training_only_protocol(base, experiment)
    manifest = write_initial_artifacts(paths, base, experiment, evidence)
    selected_protocols = {domain: protocols[domain] for domain in experiment["domains"]}
    atomic_json(paths["protocol"], selected_protocols)
    a1.atomic_write_text(paths["engine_roles"], a21.protocol_rows(selected_protocols).to_csv(index=False))
    base_training = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"])
    expected_training = base_training * len(REPRESENTATIONS)
    dry_cfg = deepcopy(base)
    dry_cfg.update({"target_domain": experiment["domains"][0], "source_domains": selected_protocols[experiment["domains"][0]]["source_domains"], "seed": int(experiment["model_seeds"][0])})
    dry_data = {representation: prepare_representation_data(dry_cfg, representation) for representation in REPRESENTATIONS}
    prior, _, _ = a1.source_correlation_adjacency_train_only(dry_cfg, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    dry_shapes: dict[str, Any] = {}
    for representation, data in dry_data.items():
        a1.seed_everything(int(experiment["model_seeds"][0]))
        model = exp17b.build_model_17b(ARCHITECTURE, len(data["features"]), dry_cfg, prior, prior).eval()
        source_tasks = make_source_tasks(data, dry_cfg, experiment)
        x, _ = next(iter(source_tasks[dry_cfg["source_domains"][0]]))
        with torch.no_grad():
            prediction = model(x[: min(4, len(x))])
        dry_shapes[representation] = {
            "feature_count": int(len(data["features"])),
            "input_shape": list(x[: min(4, len(x))].shape),
            "output_shape": list(prediction.shape),
            "finite_output": bool(torch.isfinite(prediction).all()),
        }
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "representations": list(REPRESENTATIONS),
        "domains": experiment["domains"],
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "role_partitions": experiment["role_partitions"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "endpoint_seed_sets_disjoint": True,
        "source_pretraining_runs": len(experiment["domains"]) * len(experiment["model_seeds"]) * len(REPRESENTATIONS),
        "expected_training_cells": expected_training,
        "expected_selection_records": expected_training * len(experiment["role_partitions"]) * len(experiment["selection_endpoint_seeds"]),
        "expected_confirmation_records": expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"]),
        "expected_fixed_endpoint_records": expected_training * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "source_pretrain_steps": int(base["source_pretrain_steps"]),
        "feature_shapes": dry_shapes,
        "age_feature_audit": {representation: data["audit"] for representation, data in dry_data.items()},
        "evidence": evidence,
        "gpu_inventory": a2.query_gpus(),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry)
    atomic_json(
        paths["causality"],
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_feature": AGE_FEATURE,
            "candidate_transform": "log1p(cycle), then source-only standardization",
            "allowed_at_deployment": ["current_observed_cycle", "current_and_past_sensor_windows", "current_and_past_operating_settings"],
            "forbidden": ["unit_max_cycle", "future_cycle", "true_rul", "official_test_trajectories", "official_test_rul_labels"],
            "source_pretraining_fresh_for_baseline": True,
            "source_pretraining_fresh_for_candidate": True,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    shard_root = output / "shards"
    if shard_root.exists() and any(shard_root.iterdir()) and not args.resume:
        raise RuntimeError("A8 contains an interrupted run; use --resume or a new output directory")
    tasks = [(domain, seed) for domain in experiment["domains"] for seed in experiment["model_seeds"]]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    endpoints = merged["endpoints"].sort_values(["target_domain", "representation", "model_seed", "target_split_seed", "unit", "endpoint_fraction"])
    if endpoints["cell_id"].nunique() != expected_training:
        raise RuntimeError("A8 endpoint outputs are incomplete")
    if endpoints[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any():
        raise RuntimeError("A8 detected official-test contamination")
    expected_sources = len(experiment["domains"]) * len(experiment["model_seeds"]) * len(REPRESENTATIONS)
    inventory = merged["source_inventory"].drop_duplicates(["target_domain", "model_seed", "representation"])
    audits = merged["age_audit"].drop_duplicates(["target_domain", "model_seed", "representation"])
    if len(inventory) != expected_sources or len(audits) != expected_sources:
        raise RuntimeError("A8 source-pretraining/age audit output is incomplete")
    if inventory["source_pretraining_reused_from_prior_experiment"].astype(bool).any():
        raise RuntimeError("A8 source model was improperly reused from a previous experiment")
    evaluated = evaluate_roles(endpoints, selected_protocols, experiment)
    selection = evaluated["selection_run"].sort_values(PAIR_KEYS + ["representation"])
    confirmation = evaluated["confirmation_run"].sort_values(PAIR_KEYS + ["representation"])
    fixed = evaluated["fixed_run"].sort_values(FIXED_KEYS + ["representation"])
    expected_selection = expected_training * len(experiment["role_partitions"]) * len(experiment["selection_endpoint_seeds"])
    expected_confirmation = expected_training * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    expected_fixed = expected_training * len(experiment["role_partitions"]) * len(experiment["endpoint_fractions"])
    if len(selection) != expected_selection or len(confirmation) != expected_confirmation or len(fixed) != expected_fixed:
        raise RuntimeError("A8 role/fixed evaluation output is incomplete")
    paired = paired_representations(confirmation, PAIR_KEYS)
    fixed_paired = paired_representations(fixed, FIXED_KEYS)
    high_paired = stage_pairs(evaluated["confirmation_predictions"], True, experiment)
    low_paired = stage_pairs(evaluated["confirmation_predictions"], False, experiment)
    comparisons = pd.concat(
        [
            comparison_summary(paired, experiment, "full_endpoint_age_vs_baseline"),
            comparison_summary(high_paired, experiment, "high_rul_age_vs_baseline"),
            comparison_summary(low_paired, experiment, "low_rul_age_vs_baseline"),
        ],
        ignore_index=True,
    )
    high = stage_summary(high_paired, experiment, "high_rul_gt60")
    low = stage_summary(low_paired, experiment, "low_or_mid_rul_le60")
    decision = make_decision(
        endpoints=endpoints,
        inventory=inventory,
        confirmation=confirmation,
        paired=paired,
        comparisons=comparisons,
        high=high,
        low=low,
        experiment=experiment,
    )
    for name, frame in (
        ("age_audit", audits),
        ("source_inventory", inventory),
        ("source_history", merged["source_history"]),
        ("endpoint_predictions", endpoints),
        ("target_history", merged["target_history"]),
        ("selection_predictions", evaluated["selection_predictions"]),
        ("confirmation_predictions", evaluated["confirmation_predictions"]),
        ("selection_run", selection),
        ("confirmation_run", confirmation),
        ("fixed_run", fixed),
        ("paired", paired),
        ("fixed_paired", fixed_paired),
        ("high_paired", high_paired),
        ("low_paired", low_paired),
        ("comparison", comparisons),
    ):
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
