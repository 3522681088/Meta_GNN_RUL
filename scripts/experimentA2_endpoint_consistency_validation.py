"""Experiment A2: train-only endpoint-consistency validation.

This experiment starts a new validation direction after A1--A1_3 showed that
the prior sensor graph does not have a stable advantage.  A2 asks whether
model selection on every window of a complete run-to-failure trajectory agrees
with deployment-like selection at a truncated trajectory endpoint.

Registered formal design
------------------------

* targets FD001--FD004 in leave-one-domain-out source transfer;
* window_no_graph, window_graph, and sensor_graph_prior;
* K=5 target adaptation engines;
* five model seeds (80--84) crossed with five support splits (6401--6405);
* fixed, engine-disjoint selection and confirmation engines;
* source-train-only normalization and source-train-only prior graphs;
* one target-head training trajectory stores both the best complete-window
  epoch and the best single-endpoint epoch;
* confirmation is scored on all windows, one deterministic endpoint per
  engine, and four diagnostic endpoint fractions;
* no test_FDxxx.txt or RUL_FDxxx.txt access.

The single entry point automatically discovers idle GPUs, assigns one
(target-domain, model-seed) task to each GPU, writes isolated resumable shards,
and merges all artifacts into one output directory.

Run from the repository root:

    python -u scripts/experimentA2_endpoint_consistency_validation.py

Outputs are written to
``outputs/experimentA2_endpoint_consistency_validation``.
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
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_1_prior_window_stability_confirmation as a11  # noqa: E402
from scripts import experimentA1_2_seed_ensemble_stability as a12  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402


SCRIPT_VERSION = "experimentA2_endpoint_consistency_validation_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODELS = ("window_no_graph", "window_graph", "sensor_graph_prior")
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
SELECTION_PROTOCOLS = (
    "full_trajectory_selection",
    "single_endpoint_selection",
)
EVALUATION_PROTOCOLS = (
    "full_trajectory",
    "single_endpoint",
    "stratified_endpoint",
)
COMPARISONS = (
    ("window_graph", "window_no_graph", "window_graph_vs_window_no_graph"),
    (
        "sensor_graph_prior",
        "window_graph",
        "prior_sensor_vs_window_graph",
    ),
    (
        "sensor_graph_prior",
        "window_no_graph",
        "prior_sensor_vs_window_no_graph",
    ),
)
DEFAULT_OUTPUT = "outputs/experimentA2_endpoint_consistency_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A2: train-only endpoint consistency"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 3,4,5")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="FD004, one seed, one split, five source steps",
    )
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    base["data_dir"] = resolved(args.data_dir, base["data_dir"])
    base["output_dir"] = resolved(args.output_dir, DEFAULT_OUTPUT)
    base["normalizer_seed"] = 2026
    base["condition_count"] = 6
    base["source_pretrain_steps"] = 1500
    base["source_pretrain_lr"] = 0.001
    base["source_pretrain_weight_decay"] = 0.0
    base["target_epochs"] = 10
    base["target_lr"] = 0.001
    base["pair_aux_weight"] = 0.0
    base["device"] = args.device
    experiment = {
        "experiment_id": "experimentA2",
        "experiment_name": "endpoint_consistency_validation",
        "domains": list(DOMAINS),
        "models": list(MODELS),
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "k": 5,
        "selection_count": 20,
        "confirmation_count": 30,
        "selection_seed_base": 7200,
        "confirmation_seed_base": 7300,
        "endpoint_seed_base": 7400,
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "minimum_endpoint_improvement_pct": 3.0,
        "minimum_domain_win_count": 3,
        "rank_flip_domain_rate_threshold": 0.20,
        "minimum_rank_flip_domains": 3,
        "output_dir": base["output_dir"],
        "quick_mode": False,
    }
    if args.quick:
        experiment["domains"] = ["FD004"]
        experiment["model_seeds"] = [80]
        experiment["target_split_seeds"] = [6401]
        experiment["source_pretrain_steps"] = 5
        experiment["target_epochs"] = 2
        experiment["bootstrap_repetitions"] = 100
        experiment["quick_mode"] = True
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 2
        if args.output_dir is None:
            base["output_dir"] = resolved(
                None,
                "outputs/experimentA2_endpoint_consistency_validation_quick",
            )
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if not set(experiment["domains"]).issubset(DOMAINS):
        raise ValueError("A2 contains an unknown C-MAPSS domain")
    if tuple(experiment["models"]) != MODELS:
        raise ValueError(f"A2 requires models={MODELS}")
    if len(set(experiment["model_seeds"])) != len(experiment["model_seeds"]):
        raise ValueError("duplicate model seeds")
    if len(set(experiment["target_split_seeds"])) != len(
        experiment["target_split_seeds"]
    ):
        raise ValueError("duplicate target split seeds")
    if experiment["k"] < 1:
        raise ValueError("K must be positive")
    for domain in DOMAINS:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )


def domain_index(domain: str) -> int:
    return list(DOMAINS).index(domain) + 1


def previous_fd004_exposure(base: dict) -> set[int]:
    prior_base = deepcopy(base)
    prior_base["target_domain"] = "FD004"
    prior_base["source_domains"] = ["FD001", "FD002", "FD003"]
    prior_experiment = deepcopy(a12.DEFAULT_EXPERIMENT)
    protocol = a12.build_protocol(prior_base, prior_experiment)
    exposed: set[int] = set()
    for units in protocol["historical_engine_sets"].values():
        exposed.update(map(int, units))
    exposed.update(map(int, protocol["selection_units"]))
    exposed.update(map(int, protocol["confirmation_units"]))
    for units in protocol["adaptation_units_by_target_split_seed"].values():
        exposed.update(map(int, units))
    return exposed


def build_domain_protocol(
    base: dict,
    experiment: dict,
    target_domain: str,
) -> dict:
    frame = a1.load_train_domain(base["data_dir"], target_domain)
    all_units = np.asarray(sorted(frame["unit"].unique()), dtype=int)
    excluded = previous_fd004_exposure(base) if target_domain == "FD004" else set()
    eligible = np.asarray(
        [unit for unit in all_units if int(unit) not in excluded], dtype=int
    )
    index = domain_index(target_domain)
    selection_seed = int(experiment["selection_seed_base"]) + index
    confirmation_seed = int(experiment["confirmation_seed_base"]) + index
    selection = np.random.default_rng(selection_seed).permutation(eligible)[
        : int(experiment["selection_count"])
    ]
    selection_set = set(map(int, selection))
    confirmation_pool = np.asarray(
        [unit for unit in eligible if int(unit) not in selection_set], dtype=int
    )
    confirmation = np.random.default_rng(confirmation_seed).permutation(
        confirmation_pool
    )[: int(experiment["confirmation_count"])]
    confirmation_set = set(map(int, confirmation))
    candidates = np.asarray(
        [
            unit
            for unit in eligible
            if int(unit) not in selection_set
            and int(unit) not in confirmation_set
        ],
        dtype=int,
    )
    if int(experiment["k"]) > len(candidates):
        raise ValueError(f"not enough A2 adaptation engines for {target_domain}")
    adaptation: dict[str, list[int]] = {}
    for split_seed in experiment["target_split_seeds"]:
        units = np.random.default_rng(int(split_seed)).permutation(candidates)[
            : int(experiment["k"])
        ]
        adaptation[str(split_seed)] = list(map(int, units))
    source_domains = [domain for domain in DOMAINS if domain != target_domain]
    protocol = {
        "protocol_version": SCRIPT_VERSION,
        "target_domain": target_domain,
        "source_domains": source_domains,
        "train_engine_count": int(len(all_units)),
        "historical_exclusion_applied": target_domain == "FD004",
        "historically_excluded_units": sorted(excluded),
        "eligible_engine_count": int(len(eligible)),
        "selection_seed": selection_seed,
        "selection_units": list(map(int, selection)),
        "selection_role": "dual_epoch_selection_only",
        "confirmation_seed": confirmation_seed,
        "confirmation_units": list(map(int, confirmation)),
        "confirmation_role": "final_metrics_only",
        "adaptation_candidate_count": int(len(candidates)),
        "adaptation_units_by_target_split_seed": adaptation,
        "target_split_seeds": list(map(int, experiment["target_split_seeds"])),
        "model_seeds": list(map(int, experiment["model_seeds"])),
        "models": list(experiment["models"]),
        "k": int(experiment["k"]),
        "endpoint_seed": int(experiment["endpoint_seed_base"]) + index,
        "endpoint_fractions": list(map(float, experiment["endpoint_fractions"])),
        "normalizer_fit_scope": "source_train_only",
        "prior_graph_fit_scope": "source_train_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "train_file_hashes": {
            domain: a1.file_sha256(a1.train_path(base["data_dir"], domain))
            for domain in DOMAINS
        },
    }
    protocol["protocol_hash"] = a1.canonical_hash(protocol)
    return protocol


def protocol_rows(protocols: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for domain, protocol in protocols.items():
        for role in ("selection", "confirmation"):
            for unit in protocol[f"{role}_units"]:
                rows.append(
                    {
                        "target_domain": domain,
                        "target_split_seed": "fixed",
                        "role": role,
                        "unit": int(unit),
                    }
                )
        for split_seed, units in protocol[
            "adaptation_units_by_target_split_seed"
        ].items():
            for unit in units:
                rows.append(
                    {
                        "target_domain": domain,
                        "target_split_seed": int(split_seed),
                        "role": "adaptation",
                        "unit": int(unit),
                    }
                )
    return pd.DataFrame(rows)


def endpoint_position(count: int, fraction: float) -> int:
    if count < 1:
        raise ValueError("endpoint selection received an empty engine")
    return int(np.clip(round(float(fraction) * (count - 1)), 0, count - 1))


def add_within_engine_index(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["unit_window_index"] = frame.groupby("unit").cumcount()
    return frame


def single_endpoint_subset(
    predictions: pd.DataFrame,
    target_domain: str,
    endpoint_seed: int,
    fractions: list[float],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for unit, group in predictions.groupby("unit", sort=True):
        ordered = group.sort_values("unit_window_index")
        payload = f"{target_domain}:{int(unit)}:{endpoint_seed}".encode("utf-8")
        choice = int(hashlib.sha256(payload).hexdigest()[:8], 16) % len(fractions)
        fraction = float(fractions[choice])
        selected = ordered.iloc[endpoint_position(len(ordered), fraction)].copy()
        selected["endpoint_fraction"] = fraction
        selected["endpoint_role"] = "single_primary"
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def stratified_endpoint_subset(
    predictions: pd.DataFrame,
    fractions: list[float],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in predictions.groupby("unit", sort=True):
        ordered = group.sort_values("unit_window_index")
        used: set[int] = set()
        for fraction in fractions:
            position = endpoint_position(len(ordered), float(fraction))
            if position in used:
                continue
            used.add(position)
            selected = ordered.iloc[position].copy()
            selected["endpoint_fraction"] = float(fraction)
            selected["endpoint_role"] = "stratified_diagnostic"
            rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def target_run_seed(domain: str, model_seed: int, split_seed: int) -> int:
    payload = f"experimentA2:{domain}:{model_seed}:{split_seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def train_target_head_dual_selection(
    model: torch.nn.Module,
    support,
    selection,
    cfg: dict,
    device: torch.device,
    target_domain: str,
    endpoint_seed: int,
    fractions: list[float],
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict], dict[str, int]]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("model has no predictor.* parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["target_lr"]))
    best_full = a1.state_to_cpu(learner)
    best_endpoint = a1.state_to_cpu(learner)
    best_full_rmse = float("inf")
    best_endpoint_rmse = float("inf")
    best_epochs = {
        "full_trajectory_selection": 0,
        "single_endpoint_selection": 0,
    }
    history: list[dict] = []
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        learner.train()
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss = F.mse_loss(prediction, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A2 target-head loss became NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        full_predictions = add_within_engine_index(
            a1.predict_with_units(learner, selection, device)
        )
        endpoint_predictions = single_endpoint_subset(
            full_predictions,
            target_domain,
            endpoint_seed,
            fractions,
        )
        full_metrics = regression_metrics(
            full_predictions["label"], full_predictions["prediction"]
        )
        endpoint_metrics = regression_metrics(
            endpoint_predictions["label"], endpoint_predictions["prediction"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"full_selection_{k}": v for k, v in full_metrics.items()},
                **{
                    f"endpoint_selection_{k}": v
                    for k, v in endpoint_metrics.items()
                },
            }
        )
        print(
            f"A2 target_epoch={epoch:02d}/{cfg['target_epochs']} "
            f"loss={np.mean(losses):.4f} full_rmse={full_metrics['rmse']:.4f} "
            f"endpoint_rmse={endpoint_metrics['rmse']:.4f}"
        )
        if full_metrics["rmse"] < best_full_rmse:
            best_full_rmse = float(full_metrics["rmse"])
            best_full = a1.state_to_cpu(learner)
            best_epochs["full_trajectory_selection"] = epoch
        if endpoint_metrics["rmse"] < best_endpoint_rmse:
            best_endpoint_rmse = float(endpoint_metrics["rmse"])
            best_endpoint = a1.state_to_cpu(learner)
            best_epochs["single_endpoint_selection"] = epoch
    return (
        {
            "full_trajectory_selection": best_full,
            "single_endpoint_selection": best_endpoint,
        },
        history,
        best_epochs,
    )


def cell_id(domain: str, split_seed: int, model_seed: int, model: str) -> str:
    return (
        f"experimentA2_{domain.lower()}_k05_tsplit{split_seed}_"
        f"mseed{model_seed}_{model}"
    )


def evaluate_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    split_seed: int,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    prior: torch.Tensor,
    save_checkpoint: bool,
) -> tuple[
    list[dict],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
]:
    domain = str(protocol["target_domain"])
    run_seed = target_run_seed(domain, model_seed, split_seed)
    cfg = deepcopy(base)
    cfg["seed"] = run_seed
    cfg["target_domain"] = domain
    cfg["source_domains"] = list(protocol["source_domains"])
    (
        _,
        support,
        selection,
        confirmation,
        feature_count,
        split,
    ) = a11.prepare_confirmation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=protocol["adaptation_units_by_target_split_seed"][
            str(split_seed)
        ],
    )
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(
        model_name, feature_count, cfg, prior, prior
    )
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    states, history, best_epochs = train_target_head_dual_selection(
        model,
        support,
        selection,
        cfg,
        device,
        domain,
        int(protocol["endpoint_seed"]),
        list(protocol["endpoint_fractions"]),
    )
    identifier = cell_id(domain, split_seed, model_seed, model_name)
    result_rows: list[dict] = []
    full_parts: list[pd.DataFrame] = []
    single_parts: list[pd.DataFrame] = []
    stratified_parts: list[pd.DataFrame] = []
    engine_parts: list[pd.DataFrame] = []
    for selection_protocol, state in states.items():
        selected_model = exp17b.build_model_17b(
            model_name, feature_count, cfg, prior, prior
        ).to(device)
        selected_model.load_state_dict(state)
        full = add_within_engine_index(
            a1.predict_with_units(selected_model, confirmation, device)
        )
        single = single_endpoint_subset(
            full,
            domain,
            int(protocol["endpoint_seed"]),
            list(protocol["endpoint_fractions"]),
        )
        stratified = stratified_endpoint_subset(
            full, list(protocol["endpoint_fractions"])
        )
        evaluations = {
            "full_trajectory": full,
            "single_endpoint": single,
            "stratified_endpoint": stratified,
        }
        for evaluation_protocol, predictions in evaluations.items():
            metrics = regression_metrics(
                predictions["label"], predictions["prediction"]
            )
            result_rows.append(
                {
                    **metrics,
                    "experiment_id": "experimentA2",
                    "cell_id": identifier,
                    "target_domain": domain,
                    "source_domains": list(protocol["source_domains"]),
                    "model": model_name,
                    "model_seed": int(model_seed),
                    "target_split_seed": int(split_seed),
                    "target_run_seed": int(run_seed),
                    "k": int(experiment["k"]),
                    "selection_protocol": selection_protocol,
                    "evaluation_protocol": evaluation_protocol,
                    "selected_epoch": int(best_epochs[selection_protocol]),
                    "prediction_count": int(len(predictions)),
                    "adaptation_units": protocol[
                        "adaptation_units_by_target_split_seed"
                    ][str(split_seed)],
                    "selection_units": protocol["selection_units"],
                    "confirmation_units": protocol["confirmation_units"],
                    "source_pretrain_steps": int(base["source_pretrain_steps"]),
                    "target_epochs_planned": int(base["target_epochs"]),
                    "total_parameter_count": inventory[
                        "total_parameter_count"
                    ],
                    "target_trainable_parameter_count": inventory[
                        "predictor_parameter_count"
                    ],
                    "source_signature": inventory["source_signature"],
                    "source_history_rows": int(len(source_history)),
                    "normalizer_fit_scope": "source_train_only",
                    "confirmation_used_for_selection": False,
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )
            annotated = predictions.copy()
            for position, (column, value) in enumerate(
                [
                    ("cell_id", identifier),
                    ("target_domain", domain),
                    ("model", model_name),
                    ("model_seed", int(model_seed)),
                    ("target_split_seed", int(split_seed)),
                    ("selection_protocol", selection_protocol),
                    ("evaluation_protocol", evaluation_protocol),
                ]
            ):
                annotated.insert(position, column, value)
            engines = a1.per_engine_metrics(annotated)
            for position, (column, value) in enumerate(
                [
                    ("cell_id", identifier),
                    ("target_domain", domain),
                    ("model", model_name),
                    ("model_seed", int(model_seed)),
                    ("target_split_seed", int(split_seed)),
                    ("selection_protocol", selection_protocol),
                    ("evaluation_protocol", evaluation_protocol),
                ]
            ):
                engines.insert(position, column, value)
            engine_parts.append(engines)
            if evaluation_protocol == "full_trajectory":
                full_parts.append(annotated)
            elif evaluation_protocol == "single_endpoint":
                single_parts.append(annotated)
            else:
                stratified_parts.append(annotated)
        del selected_model
    history_rows = [
        {
            "cell_id": identifier,
            "target_domain": domain,
            "model": model_name,
            "model_seed": int(model_seed),
            "target_split_seed": int(split_seed),
            **row,
        }
        for row in history
    ]
    if save_checkpoint:
        path = Path(base["output_dir"]) / "checkpoints" / f"{identifier}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "states": states,
                "best_epochs": best_epochs,
                "history": history,
                "split": split,
            },
            path,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        result_rows,
        pd.concat(full_parts, ignore_index=True),
        pd.concat(single_parts, ignore_index=True),
        pd.concat(stratified_parts, ignore_index=True),
        pd.concat(engine_parts, ignore_index=True),
        history_rows,
    )


def root_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA2"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "run_json": output / f"{prefix}_run_level.json",
        "run_csv": output / f"{prefix}_run_level.csv",
        "full_predictions": output / f"{prefix}_full_trajectory_predictions.csv",
        "single_predictions": output / f"{prefix}_single_endpoint_predictions.csv",
        "stratified_predictions": output
        / f"{prefix}_stratified_endpoint_predictions.csv",
        "per_engine": output / f"{prefix}_per_engine_metrics.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "summary": output / f"{prefix}_summary.csv",
        "rankings": output / f"{prefix}_model_rankings.csv",
        "rank_flips": output / f"{prefix}_rank_flips.csv",
        "regret": output / f"{prefix}_selection_regret.csv",
        "paired_cells": output / f"{prefix}_paired_model_cells.csv",
        "paired_comparisons": output
        / f"{prefix}_paired_model_comparisons.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
        "lock_candidate": output / f"{prefix}_lock_candidate.json",
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
        "full_predictions": directory / "full_trajectory_predictions.csv",
        "single_predictions": directory / "single_endpoint_predictions.csv",
        "stratified_predictions": directory
        / "stratified_endpoint_predictions.csv",
        "per_engine": directory / "per_engine_metrics.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    completed: set[str] = set()
    if paths["status"].is_file():
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        completed = set(map(str, status.get("completed_cell_ids", [])))
    results = []
    if paths["run_json"].is_file():
        results = json.loads(paths["run_json"].read_text(encoding="utf-8"))
        results = [row for row in results if row["cell_id"] in completed]
    frames = {
        name: load_csv(paths[name])
        for name in (
            "full_predictions",
            "single_predictions",
            "stratified_predictions",
            "per_engine",
            "history",
            "inventory",
        )
    }
    for name in (
        "full_predictions",
        "single_predictions",
        "stratified_predictions",
        "per_engine",
        "history",
    ):
        if not frames[name].empty:
            frames[name] = frames[name][frames[name]["cell_id"].isin(completed)]
    return {"completed": completed, "results": results, **frames}


def save_worker_state(
    paths: dict[str, Path],
    state: dict[str, Any],
    expected_cells: int,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["run_json"], state["results"])
    a1.atomic_write_text(
        paths["run_csv"], pd.DataFrame(state["results"]).to_csv(index=False)
    )
    for name in (
        "full_predictions",
        "single_predictions",
        "stratified_predictions",
        "per_engine",
        "history",
        "inventory",
    ):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    status = {
        "completed_cell_ids": sorted(state["completed"]),
        "completed_cells": len(state["completed"]),
        "expected_cells": int(expected_cells),
        "complete": len(state["completed"]) == int(expected_cells),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["status"], status)


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain = str(args.worker_domain)
    model_seed = int(args.worker_seed)
    if domain not in experiment["domains"]:
        raise ValueError(f"unregistered A2 worker domain: {domain}")
    if model_seed not in experiment["model_seeds"]:
        raise ValueError(f"unregistered A2 worker seed: {model_seed}")
    output = Path(base["output_dir"])
    paths = shard_paths(output, domain, model_seed)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    worker_base = deepcopy(base)
    worker_base["output_dir"] = str(paths["directory"])
    worker_base["target_domain"] = domain
    worker_base["source_domains"] = [value for value in DOMAINS if value != domain]
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"
    protocol = build_domain_protocol(worker_base, experiment, domain)
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(
        worker_base,
        experiment["preprocessing"],
        int(experiment["sensor_graph_k"]),
    )
    script_hash = a1.file_sha256(Path(__file__))
    git_commit = a1.git_commit(PROJECT_ROOT)
    atomic_json(
        paths["manifest"],
        {
            "script_version": SCRIPT_VERSION,
            "script_hash": script_hash,
            "git_commit": git_commit,
            "target_domain": domain,
            "model_seed": model_seed,
            "protocol_hash": protocol["protocol_hash"],
            "graph_fit": graph_fit,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(
        paths["directory"] / "prior_adjacency.csv",
        pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv(),
    )
    a1.atomic_write_text(
        paths["directory"] / "prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )
    state = load_worker_state(paths)
    expected_cells = len(MODELS) * len(experiment["target_split_seeds"])
    for model_name in MODELS:
        pending = [
            split_seed
            for split_seed in experiment["target_split_seeds"]
            if cell_id(domain, split_seed, model_seed, model_name)
            not in state["completed"]
        ]
        if not pending:
            continue
        source_experiment = {
            **experiment,
            "target_split_seeds": list(experiment["target_split_seeds"]),
            "output_dir": worker_base["output_dir"],
        }
        source_state, source_history, inventory = a12.load_or_train_source(
            base=worker_base,
            experiment=source_experiment,
            protocol=protocol,
            architecture=model_name,
            model_seed=model_seed,
            prior=prior,
            git_commit=git_commit,
            script_hash=script_hash,
        )
        inventory = {
            "target_domain": domain,
            **inventory,
        }
        if state["inventory"].empty:
            state["inventory"] = pd.DataFrame([inventory])
        else:
            keep = ~(
                state["inventory"]["target_domain"].eq(domain)
                & state["inventory"]["model"].eq(model_name)
                & state["inventory"]["model_seed"].eq(model_seed)
            )
            state["inventory"] = pd.concat(
                [state["inventory"][keep], pd.DataFrame([inventory])],
                ignore_index=True,
            )
        for split_seed in pending:
            (
                rows,
                full,
                single,
                stratified,
                engines,
                history,
            ) = evaluate_cell(
                base=worker_base,
                experiment=experiment,
                protocol=protocol,
                model_name=model_name,
                model_seed=model_seed,
                split_seed=int(split_seed),
                source_state=deepcopy(source_state),
                source_history=source_history,
                inventory=inventory,
                prior=prior,
                save_checkpoint=args.save_target_checkpoints,
            )
            state["results"].extend(rows)
            for name, frame in (
                ("full_predictions", full),
                ("single_predictions", single),
                ("stratified_predictions", stratified),
                ("per_engine", engines),
                ("history", pd.DataFrame(history)),
            ):
                state[name] = pd.concat(
                    [state[name], frame], ignore_index=True
                )
            state["completed"].add(
                cell_id(domain, split_seed, model_seed, model_name)
            )
            save_worker_state(paths, state, expected_cells)
    save_worker_state(paths, state, expected_cells)
    print(paths["status"].read_text(encoding="utf-8"))


def query_gpus() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        try:
            index, free, total, utilization = map(int, values)
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "free_mb": free,
                "total_mb": total,
                "utilization": utilization,
            }
        )
    return rows


def visible_gpu_filter() -> set[int] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or not value.strip():
        return None
    parts = [part.strip() for part in value.split(",")]
    return {int(part) for part in parts} if all(part.isdigit() for part in parts) else None


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    inventory = query_gpus()
    if args.gpus:
        requested = [int(value.strip()) for value in args.gpus.split(",")]
        if len(set(requested)) != len(requested):
            raise ValueError("--gpus contains duplicate GPU indices")
        known = {row["index"] for row in inventory}
        if not set(requested).issubset(known):
            raise RuntimeError("one or more requested GPUs are unavailable")
        if args.max_workers > 0:
            requested = requested[: int(args.max_workers)]
        return requested, inventory
    visible = visible_gpu_filter()
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
    if args.quick:
        command.append("--quick")
    if args.save_target_checkpoints:
        command.append("--save-target-checkpoints")
    return command


def run_workers(
    args: argparse.Namespace,
    tasks: list[tuple[str, int]],
    output: Path,
) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {
        "auto",
        "cpu",
    }:
        devices: list[str | int] = [args.device]
        inventory: list[dict] = []
    else:
        devices, inventory = choose_gpus(args)
        if not devices:
            raise RuntimeError(
                "no idle GPU met A2 thresholds; inventory="
                + json.dumps(inventory, ensure_ascii=False)
            )
    print(
        json.dumps(
            {
                "scheduler": "experimentA2",
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
        for device in [value for value in devices if value not in active]:
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
                f"[A2] launched domain={domain} seed={seed} "
                f"device={device} pid={process.pid}"
            )
        finished = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = "\n".join(
                    record["log_path"].read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-50:]
                )
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"A2 worker failed domain={record['domain']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A2] completed domain={record['domain']} "
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
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "results": [],
        "full_predictions": [],
        "single_predictions": [],
        "stratified_predictions": [],
        "per_engine": [],
        "history": [],
        "inventory": [],
    }
    expected_per_worker = len(MODELS) * len(experiment["target_split_seeds"])
    for domain, seed in tasks:
        paths = shard_paths(output, domain, seed)
        if not paths["status"].is_file():
            raise RuntimeError(f"missing A2 worker status: {paths['status']}")
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        if not status.get("complete") or status.get("completed_cells") != expected_per_worker:
            raise RuntimeError(f"incomplete A2 worker: {paths['status']}")
        merged["results"].extend(
            json.loads(paths["run_json"].read_text(encoding="utf-8"))
        )
        for name in (
            "full_predictions",
            "single_predictions",
            "stratified_predictions",
            "per_engine",
            "history",
            "inventory",
        ):
            merged[name].append(load_csv(paths[name]))
    for name in (
        "full_predictions",
        "single_predictions",
        "stratified_predictions",
        "per_engine",
        "history",
        "inventory",
    ):
        merged[name] = pd.concat(merged[name], ignore_index=True)
    return merged


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "target_domain",
        "selection_protocol",
        "evaluation_protocol",
        "model",
    ]
    for keys, group in results.groupby(group_columns):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_cells": int(len(group)),
                "n_model_seeds": int(group["model_seed"].nunique()),
                "n_target_splits": int(group["target_split_seed"].nunique()),
            }
        )
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns)


def build_rankings_and_regret(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["target_domain", "model_seed", "target_split_seed"]
    full = results[
        results["selection_protocol"].eq("full_trajectory_selection")
        & results["evaluation_protocol"].eq("full_trajectory")
    ].copy()
    endpoint = results[
        results["selection_protocol"].eq("single_endpoint_selection")
        & results["evaluation_protocol"].eq("single_endpoint")
    ].copy()
    full["ranking_protocol"] = "full_trajectory"
    endpoint["ranking_protocol"] = "single_endpoint"
    rankings = pd.concat([full, endpoint], ignore_index=True)
    rankings["rmse_rank"] = rankings.groupby(keys + ["ranking_protocol"])[
        "rmse"
    ].rank(method="min")
    rankings = rankings[
        keys + ["ranking_protocol", "model", "rmse", "nasa_score", "rmse_rank"]
    ]
    full_best = (
        full.sort_values("rmse").groupby(keys, as_index=False).first()[
            keys + ["model", "rmse"]
        ].rename(columns={"model": "full_selected_model", "rmse": "full_rmse"})
    )
    endpoint_best = (
        endpoint.sort_values("rmse").groupby(keys, as_index=False).first()[
            keys + ["model", "rmse"]
        ].rename(
            columns={
                "model": "endpoint_selected_model",
                "rmse": "endpoint_best_rmse",
            }
        )
    )
    flips = full_best.merge(endpoint_best, on=keys)
    flips["rank_flip"] = (
        flips["full_selected_model"] != flips["endpoint_selected_model"]
    )
    endpoint_common = endpoint[keys + ["model", "rmse"]].rename(
        columns={"rmse": "endpoint_rmse_endpoint_selected_state"}
    )
    full_choice_common = full_best.merge(
        endpoint_common,
        left_on=keys + ["full_selected_model"],
        right_on=keys + ["model"],
    )
    full_choice_cross = results[
        results["selection_protocol"].eq("full_trajectory_selection")
        & results["evaluation_protocol"].eq("single_endpoint")
    ][keys + ["model", "rmse"]].rename(
        columns={"rmse": "endpoint_rmse_full_selected_state"}
    )
    regret = flips.merge(
        full_choice_common[
            keys
            + [
                "full_selected_model",
                "endpoint_rmse_endpoint_selected_state",
            ]
        ],
        on=keys + ["full_selected_model"],
    ).merge(
        full_choice_cross,
        left_on=keys + ["full_selected_model"],
        right_on=keys + ["model"],
    )
    regret["model_selection_regret"] = (
        regret["endpoint_rmse_endpoint_selected_state"]
        - regret["endpoint_best_rmse"]
    )
    regret["end_to_end_protocol_delta"] = (
        regret["endpoint_rmse_full_selected_state"]
        - regret["endpoint_best_rmse"]
    )
    regret = regret.drop(columns=["model"])
    return rankings, flips, regret


def paired_model_cells(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "target_domain",
        "model_seed",
        "target_split_seed",
        "selection_protocol",
        "evaluation_protocol",
    ]
    for values, group in results.groupby(keys):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        for candidate, reference, comparison in COMPARISONS:
            if candidate not in by_model or reference not in by_model:
                continue
            candidate_row, reference_row = by_model[candidate], by_model[reference]
            row = dict(zip(keys, values))
            row.update(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "reference": reference,
                    "rmse_candidate": float(candidate_row["rmse"]),
                    "rmse_reference": float(reference_row["rmse"]),
                }
            )
            for metric in ("rmse", "mae", "r2", "nasa_score"):
                row[f"{metric}_delta_candidate_minus_reference"] = float(
                    candidate_row[metric] - reference_row[metric]
                )
            row["rmse_candidate_win"] = float(
                row["rmse_delta_candidate_minus_reference"] < 0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def crossed_bootstrap(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    domains = sorted(frame["target_domain"].unique())
    rng = np.random.default_rng(seed)
    samples = np.empty(int(repetitions), dtype=float)
    for repeat in range(int(repetitions)):
        selected_domains = rng.choice(domains, len(domains), replace=True)
        domain_means = []
        for domain in selected_domains:
            subset = frame[frame["target_domain"].eq(domain)]
            matrix = subset.pivot(
                index="model_seed",
                columns="target_split_seed",
                values=column,
            ).to_numpy(float)
            seed_index = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            split_index = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            domain_means.append(matrix[np.ix_(seed_index, split_index)].mean())
        samples[repeat] = float(np.mean(domain_means))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison_summary(
    paired: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    group_columns = ["selection_protocol", "evaluation_protocol", "comparison"]
    for keys, group in paired.groupby(group_columns):
        scoped_groups = [("ALL", group)] + [
            (domain, subset)
            for domain, subset in group.groupby("target_domain")
        ]
        for scope, scoped in scoped_groups:
            candidate = str(scoped["candidate"].iloc[0])
            reference = str(scoped["reference"].iloc[0])
            low, high = crossed_bootstrap(
                scoped,
                "rmse_delta_candidate_minus_reference",
                repetitions,
                8200 + sum(map(ord, "".join(keys) + scope)),
            )
            domain_means = scoped.groupby("target_domain")[
                "rmse_delta_candidate_minus_reference"
            ].mean()
            reference_rmse = float(scoped["rmse_reference"].mean())
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "scope": scope,
                    "candidate": candidate,
                    "reference": reference,
                    "n_cells": int(len(scoped)),
                    "n_domains": int(scoped["target_domain"].nunique()),
                    "rmse_delta_mean": float(
                        scoped["rmse_delta_candidate_minus_reference"].mean()
                    ),
                    "rmse_improvement_pct": float(
                        -100
                        * scoped["rmse_delta_candidate_minus_reference"].mean()
                        / reference_rmse
                    ),
                    "rmse_cell_win_rate": float(
                        scoped["rmse_candidate_win"].mean()
                    ),
                    "rmse_domain_win_count": int((domain_means < 0).sum()),
                    "rmse_boot_ci95_low": low,
                    "rmse_boot_ci95_high": high,
                    "mae_delta_mean": float(
                        scoped["mae_delta_candidate_minus_reference"].mean()
                    ),
                    "r2_delta_mean": float(
                        scoped["r2_delta_candidate_minus_reference"].mean()
                    ),
                    "nasa_score_delta_mean": float(
                        scoped[
                            "nasa_score_delta_candidate_minus_reference"
                        ].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_regret(
    regret: pd.DataFrame,
    column: str,
    repetitions: int,
) -> tuple[float, float]:
    renamed = regret.rename(columns={column: "value"})
    return crossed_bootstrap(renamed, "value", repetitions, 8299)


def make_decision(
    experiment: dict,
    results: pd.DataFrame,
    flips: pd.DataFrame,
    regret: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> tuple[dict, dict]:
    expected_cells = (
        len(experiment["domains"])
        * len(experiment["models"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_records = expected_cells * len(SELECTION_PROTOCOLS) * len(
        EVALUATION_PROTOCOLS
    )
    model_regret_ci = bootstrap_regret(
        regret,
        "model_selection_regret",
        int(experiment["bootstrap_repetitions"]),
    )
    end_to_end_ci = bootstrap_regret(
        regret,
        "end_to_end_protocol_delta",
        int(experiment["bootstrap_repetitions"]),
    )
    domain_flip_rates = flips.groupby("target_domain")["rank_flip"].mean()
    affected_domains = int(
        (
            domain_flip_rates
            >= float(experiment["rank_flip_domain_rate_threshold"])
        ).sum()
    )
    protocol_gap = bool(
        affected_domains >= int(experiment["minimum_rank_flip_domains"])
        or model_regret_ci[0] > 0
        or end_to_end_ci[0] > 0
    )
    endpoint_rows = comparisons[
        comparisons["scope"].eq("ALL")
        & comparisons["selection_protocol"].eq("single_endpoint_selection")
        & comparisons["evaluation_protocol"].eq("single_endpoint")
        & comparisons["comparison"].isin(
            [
                "window_graph_vs_window_no_graph",
                "prior_sensor_vs_window_no_graph",
            ]
        )
    ].copy()
    endpoint_rows["eligible"] = (
        (
            endpoint_rows["rmse_improvement_pct"]
            >= float(experiment["minimum_endpoint_improvement_pct"])
        )
        & (
            endpoint_rows["rmse_domain_win_count"]
            >= int(experiment["minimum_domain_win_count"])
        )
        & endpoint_rows["rmse_boot_ci95_high"].lt(0)
        & endpoint_rows["nasa_score_delta_mean"].le(0)
    )
    eligible = endpoint_rows[endpoint_rows["eligible"]]
    if eligible.empty:
        candidate = "window_no_graph"
        eligible_for_official = False
        candidate_reason = "no graph model met the registered endpoint criteria"
    else:
        chosen = eligible.sort_values("rmse_delta_mean").iloc[0]
        candidate = str(chosen["candidate"])
        eligible_for_official = True
        candidate_reason = "candidate met all registered endpoint criteria"
    lock_candidate = {
        "experiment_id": "experimentA2",
        "candidate_model": candidate,
        "eligible_for_locked_official_confirmation": eligible_for_official,
        "reason": candidate_reason,
        "selection_protocol": "single_endpoint_selection",
        "evaluation_protocol": "single_endpoint",
        "k": int(experiment["k"]),
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    decision = {
        "experiment_id": "experimentA2",
        "expected_training_cells": expected_cells,
        "completed_training_cells": int(results["cell_id"].nunique()),
        "expected_evaluation_records": expected_records,
        "completed_evaluation_records": int(len(results)),
        "complete": bool(
            results["cell_id"].nunique() == expected_cells
            and len(results) == expected_records
        ),
        "quick_mode": bool(experiment["quick_mode"]),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "rank_flip_rate": float(flips["rank_flip"].mean()),
        "rank_flip_rate_by_domain": {
            key: float(value) for key, value in domain_flip_rates.items()
        },
        "rank_flip_affected_domains": affected_domains,
        "model_selection_regret_mean": float(
            regret["model_selection_regret"].mean()
        ),
        "model_selection_regret_ci95": list(model_regret_ci),
        "end_to_end_protocol_delta_mean": float(
            regret["end_to_end_protocol_delta"].mean()
        ),
        "end_to_end_protocol_delta_ci95": list(end_to_end_ci),
        "endpoint_protocol_gap_confirmed": protocol_gap,
        "lock_candidate": lock_candidate,
    }
    if experiment["quick_mode"]:
        decision["passed"] = decision["complete"]
        decision["reason"] = "quick smoke run only; do not interpret"
    else:
        decision["passed"] = bool(decision["complete"] and protocol_gap)
        decision["reason"] = (
            "A2 confirmed a consequential full-trajectory/endpoint protocol gap"
            if decision["passed"]
            else "A2 completed without meeting the registered protocol-gap criteria"
        )
    return decision, lock_candidate


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols = {
        domain: build_domain_protocol(base, experiment, domain)
        for domain in experiment["domains"]
    }
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {
            key: value for key, value in base.items() if key != "device"
        },
        "experiment_config": experiment,
        "protocol_hashes": {
            domain: protocol["protocol_hash"]
            for domain, protocol in protocols.items()
        },
        "registered_primary_question": (
            "Does complete-window model selection cause endpoint rank flips "
            "or positive endpoint selection regret?"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["protocol"], protocols)
    a1.atomic_write_text(
        paths["engine_roles"], protocol_rows(protocols).to_csv(index=False)
    )
    dry_report = {
        "experiment_id": "experimentA2",
        "domains": experiment["domains"],
        "models": experiment["models"],
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "expected_training_cells": (
            len(experiment["domains"])
            * len(experiment["models"])
            * len(experiment["model_seeds"])
            * len(experiment["target_split_seeds"])
        ),
        "expected_evaluation_records": (
            len(experiment["domains"])
            * len(experiment["models"])
            * len(experiment["model_seeds"])
            * len(experiment["target_split_seeds"])
            * len(SELECTION_PROTOCOLS)
            * len(EVALUATION_PROTOCOLS)
        ),
        "protocol_hashes": manifest["protocol_hashes"],
        "gpu_inventory": query_gpus(),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry_report)
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return
    tasks = [
        (domain, seed)
        for domain in experiment["domains"]
        for seed in experiment["model_seeds"]
    ]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    results = pd.DataFrame(merged["results"]).sort_values(
        [
            "target_domain",
            "model_seed",
            "target_split_seed",
            "model",
            "selection_protocol",
            "evaluation_protocol",
        ]
    )
    expected_cells = (
        len(experiment["domains"])
        * len(experiment["models"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    if results["cell_id"].nunique() != expected_cells:
        raise RuntimeError("A2 merged training-cell count is incomplete")
    if bool(
        results[
            ["official_test_files_accessed", "official_test_forward_run"]
        ].astype(bool).any().any()
    ):
        raise RuntimeError("A2 detected an official-test contamination flag")
    summary = summarize_results(results)
    rankings, flips, regret = build_rankings_and_regret(results)
    paired = paired_model_cells(results)
    comparison_summary = paired_comparison_summary(
        paired, int(experiment["bootstrap_repetitions"])
    )
    decision, lock_candidate = make_decision(
        experiment, results, flips, regret, comparison_summary
    )
    atomic_json(paths["run_json"], results.to_dict("records"))
    a1.atomic_write_text(paths["run_csv"], results.to_csv(index=False))
    for name in (
        "full_predictions",
        "single_predictions",
        "stratified_predictions",
        "per_engine",
        "history",
        "inventory",
    ):
        a1.atomic_write_text(paths[name], merged[name].to_csv(index=False))
    a1.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    a1.atomic_write_text(paths["rankings"], rankings.to_csv(index=False))
    a1.atomic_write_text(paths["rank_flips"], flips.to_csv(index=False))
    a1.atomic_write_text(paths["regret"], regret.to_csv(index=False))
    a1.atomic_write_text(paths["paired_cells"], paired.to_csv(index=False))
    a1.atomic_write_text(
        paths["paired_comparisons"], comparison_summary.to_csv(index=False)
    )
    atomic_json(paths["decision"], decision)
    atomic_json(paths["lock_candidate"], lock_candidate)
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
