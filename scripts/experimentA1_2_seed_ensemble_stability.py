"""Experiment A1_2: model-seed ensemble stability confirmation.

A1_1 showed that model seeds explained most of the paired RMSE variation
between the source-prior sensor graph and the window graph.  A1_2 tests one
pre-registered stabilization method: average predictions from independently
pretrained and independently adapted model seeds.

The formal protocol uses:

* FD004, K=5;
* 10 new model seeds (70--79) crossed with 10 new support splits (5401--5410);
* sensor_graph_prior and window_graph;
* disjoint seed ensembles of size 2, 5 and 10;
* the former A1_1 confirmation engines only for target-epoch selection;
* 30 new confirmation engines that had no role in A1 or A1_1;
* no official test trajectories or official RUL labels.

The script is a single entry point.  Run from the repository root:

    python -u scripts/experimentA1_2_seed_ensemble_stability.py

All outputs are written below
``outputs/experimentA1_2_seed_ensemble_stability``.
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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_1_prior_window_stability_confirmation as a11  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    train_source_supervised,
)


SCRIPT_VERSION = "experimentA1_2_seed_ensemble_stability_v1"
ARCHITECTURES = ("sensor_graph_prior", "window_graph")

DEFAULT_EXPERIMENT = {
    "experiment_id": "experimentA1_2",
    "experiment_name": "seed_ensemble_stability",
    "target_domain": "FD004",
    "preprocessing": "condition_settings",
    "balance_mode": "engine_stage",
    "k": 5,
    "sensor_graph_k": 4,
    "a1_validation_count": 20,
    "a1_validation_seed": 2026,
    "a1_target_split_seeds": [3027, 3028, 3029, 3030, 3031],
    "a1_k": 5,
    "a1_1_confirmation_count": 30,
    "a1_1_confirmation_seed": 4101,
    "a1_1_target_split_seeds": list(range(4201, 4211)),
    "a1_1_k": 5,
    "new_confirmation_count": 30,
    "new_confirmation_seed": 5301,
    "target_split_seeds": list(range(5401, 5411)),
    "model_seeds": list(range(70, 80)),
    "architectures": list(ARCHITECTURES),
    "ensemble_sizes": [2, 5, 10],
    "source_pretrain_steps": 1500,
    "source_pretrain_lr": 0.001,
    "source_pretrain_weight_decay": 0.0,
    "target_epochs": 10,
    "target_lr": 0.001,
    "bootstrap_repetitions": 10000,
    "minimum_rmse_improvement_pct": 3.0,
    "minimum_target_split_win_rate": 0.8,
    "output_dir": "outputs/experimentA1_2_seed_ensemble_stability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A1_2: model-seed ensemble stability"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one-split, two-seed smoke run in a separate output directory",
    )
    return parser.parse_args()


def resolve_optional_path(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    experiment = deepcopy(DEFAULT_EXPERIMENT)
    base["target_domain"] = "FD004"
    base["source_domains"] = ["FD001", "FD002", "FD003"]
    base["data_dir"] = resolve_optional_path(args.data_dir, base["data_dir"])
    base["output_dir"] = resolve_optional_path(
        args.output_dir,
        experiment["output_dir"],
    )
    base["normalizer_seed"] = 2026
    base["condition_count"] = 6
    base["source_pretrain_steps"] = int(experiment["source_pretrain_steps"])
    base["source_pretrain_lr"] = float(experiment["source_pretrain_lr"])
    base["source_pretrain_weight_decay"] = float(
        experiment["source_pretrain_weight_decay"]
    )
    base["target_epochs"] = int(experiment["target_epochs"])
    base["target_lr"] = float(experiment["target_lr"])
    base["pair_aux_weight"] = 0.0
    if args.device is not None:
        base["device"] = args.device

    if args.quick:
        experiment["quick_mode"] = True
        experiment["target_split_seeds"] = [5401]
        experiment["model_seeds"] = [70, 71]
        experiment["ensemble_sizes"] = [2]
        experiment["source_pretrain_steps"] = 5
        experiment["target_epochs"] = 1
        experiment["bootstrap_repetitions"] = 100
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 1
        if args.output_dir is None:
            base["output_dir"] = str(
                a1.resolve_path(
                    "outputs/experimentA1_2_seed_ensemble_stability_quick"
                )
            )
    experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if base["target_domain"] != "FD004":
        raise ValueError("Experiment A1_2 is registered for FD004 only")
    if tuple(experiment["architectures"]) != ARCHITECTURES:
        raise ValueError(f"A1_2 requires architectures={ARCHITECTURES}")
    if len(set(experiment["model_seeds"])) != len(experiment["model_seeds"]):
        raise ValueError("model seeds contain duplicates")
    if len(set(experiment["target_split_seeds"])) != len(
        experiment["target_split_seeds"]
    ):
        raise ValueError("target split seeds contain duplicates")
    if max(experiment["ensemble_sizes"]) > len(experiment["model_seeds"]):
        raise ValueError("ensemble size exceeds the available model seeds")
    if experiment["preprocessing"] not in a1.PREPROCESSING_MODES:
        raise ValueError("unknown preprocessing mode")
    if experiment["balance_mode"] not in a1.BALANCE_MODES:
        raise ValueError("unknown balance mode")


def historical_engine_sets(
    all_units: np.ndarray,
    experiment: dict,
) -> dict[str, set[int]]:
    a1_selection = np.random.default_rng(
        int(experiment["a1_validation_seed"])
    ).permutation(all_units)[: int(experiment["a1_validation_count"])]
    a1_selection_set = set(map(int, a1_selection))
    a1_candidates = np.asarray(
        [unit for unit in all_units if int(unit) not in a1_selection_set],
        dtype=int,
    )
    a1_support: set[int] = set()
    for split_seed in experiment["a1_target_split_seeds"]:
        chosen = np.random.default_rng(int(split_seed)).permutation(
            a1_candidates
        )[: int(experiment["a1_k"])]
        a1_support.update(map(int, chosen))

    a1_1_confirmation_pool = np.asarray(
        [
            unit
            for unit in all_units
            if int(unit) not in a1_selection_set
            and int(unit) not in a1_support
        ],
        dtype=int,
    )
    a1_1_confirmation = np.random.default_rng(
        int(experiment["a1_1_confirmation_seed"])
    ).permutation(a1_1_confirmation_pool)[
        : int(experiment["a1_1_confirmation_count"])
    ]
    a1_1_confirmation_set = set(map(int, a1_1_confirmation))
    a1_1_adaptation_candidates = np.asarray(
        [
            unit
            for unit in all_units
            if int(unit) not in a1_selection_set
            and int(unit) not in a1_1_confirmation_set
        ],
        dtype=int,
    )
    a1_1_support: set[int] = set()
    for split_seed in experiment["a1_1_target_split_seeds"]:
        chosen = np.random.default_rng(int(split_seed)).permutation(
            a1_1_adaptation_candidates
        )[: int(experiment["a1_1_k"])]
        a1_1_support.update(map(int, chosen))

    return {
        "a1_selection": a1_selection_set,
        "a1_support": a1_support,
        "a1_1_confirmation": a1_1_confirmation_set,
        "a1_1_support": a1_1_support,
    }


def build_protocol(base: dict, experiment: dict) -> dict:
    target_frame = a1.load_train_domain(
        base["data_dir"],
        base["target_domain"],
    )
    all_units = np.asarray(
        sorted(target_frame["unit"].unique()),
        dtype=int,
    )
    historical = historical_engine_sets(all_units, experiment)
    historical_exposure = set().union(*historical.values())
    fresh_pool = np.asarray(
        [
            unit
            for unit in all_units
            if int(unit) not in historical_exposure
        ],
        dtype=int,
    )
    confirmation_count = int(experiment["new_confirmation_count"])
    if confirmation_count >= len(fresh_pool):
        raise ValueError("not enough historically unseen confirmation engines")
    confirmation = np.random.default_rng(
        int(experiment["new_confirmation_seed"])
    ).permutation(fresh_pool)[:confirmation_count]
    confirmation_set = set(map(int, confirmation))
    adaptation_candidates = np.asarray(
        [
            unit
            for unit in fresh_pool
            if int(unit) not in confirmation_set
        ],
        dtype=int,
    )
    if int(experiment["k"]) > len(adaptation_candidates):
        raise ValueError("K exceeds fresh adaptation candidates")

    adaptation_by_split: dict[str, list[int]] = {}
    for split_seed in experiment["target_split_seeds"]:
        chosen = np.random.default_rng(int(split_seed)).permutation(
            adaptation_candidates
        )[: int(experiment["k"])]
        selected = list(map(int, chosen))
        if set(selected) & historical_exposure:
            raise AssertionError("A1_2 adaptation reused historical engines")
        if set(selected) & confirmation_set:
            raise AssertionError("adaptation and confirmation engines overlap")
        adaptation_by_split[str(split_seed)] = selected

    selection_units = sorted(historical["a1_1_confirmation"])
    selection_set = set(selection_units)
    if selection_set & confirmation_set:
        raise AssertionError("selection and confirmation engines overlap")
    if selection_set & set(adaptation_candidates):
        raise AssertionError("selection engines entered fresh adaptation pool")

    train_file_hashes = {
        domain: a1.file_sha256(a1.train_path(base["data_dir"], domain))
        for domain in [*base["source_domains"], base["target_domain"]]
    }
    protocol = {
        "protocol_version": "experimentA1_2_seed_ensemble_stability_v1",
        "target_domain": base["target_domain"],
        "source_domains": list(base["source_domains"]),
        "train_engine_count": int(len(all_units)),
        "historical_exposure_count": int(len(historical_exposure)),
        "historical_engine_sets": {
            name: sorted(values) for name, values in historical.items()
        },
        "selection_units": selection_units,
        "selection_role": "target_epoch_selection_only",
        "selection_source": "former_A1_1_confirmation",
        "fresh_engine_pool_count": int(len(fresh_pool)),
        "confirmation_seed": int(experiment["new_confirmation_seed"]),
        "confirmation_units": list(map(int, confirmation)),
        "confirmation_role": "final_metrics_only",
        "confirmation_units_seen_in_A1_or_A1_1": False,
        "adaptation_candidate_count": int(len(adaptation_candidates)),
        "adaptation_units_seen_in_A1_or_A1_1": False,
        "k": int(experiment["k"]),
        "target_split_seeds": list(map(int, experiment["target_split_seeds"])),
        "model_seeds": list(map(int, experiment["model_seeds"])),
        "adaptation_units_by_target_split_seed": adaptation_by_split,
        "ensemble_groups": ensemble_groups(experiment["model_seeds"]),
        "train_file_hashes": train_file_hashes,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    protocol["protocol_hash"] = a1.canonical_hash(protocol)
    return protocol


def ensemble_groups(model_seeds: list[int]) -> dict[str, list[list[int]]]:
    seeds = list(map(int, model_seeds))
    if len(seeds) == 2:
        return {"2": [seeds]}
    if len(seeds) != 10:
        raise ValueError("formal A1_2 requires 10 seeds; quick mode requires 2")
    return {
        "2": [seeds[index : index + 2] for index in range(0, 10, 2)],
        "5": [seeds[:5], seeds[5:]],
        "10": [seeds],
    }


def protocol_frame(protocol: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for unit in protocol["selection_units"]:
        rows.append(
            {
                "target_split_seed": "fixed",
                "role": "epoch_selection",
                "unit": int(unit),
            }
        )
    for unit in protocol["confirmation_units"]:
        rows.append(
            {
                "target_split_seed": "fixed",
                "role": "new_independent_confirmation",
                "unit": int(unit),
            }
        )
    for split_seed, units in protocol[
        "adaptation_units_by_target_split_seed"
    ].items():
        for unit in units:
            rows.append(
                {
                    "target_split_seed": int(split_seed),
                    "role": "fresh_adaptation",
                    "unit": int(unit),
                }
            )
    return pd.DataFrame(rows)


def result_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA1_2"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "prior_adjacency": output / f"{prefix}_prior_adjacency.csv",
        "prior_correlation": output / f"{prefix}_prior_correlation.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "raw": output / f"{prefix}_individual_run_level.json",
        "run_csv": output / f"{prefix}_individual_run_level.csv",
        "window_predictions": output
        / f"{prefix}_individual_window_predictions.csv",
        "per_engine": output / f"{prefix}_individual_per_engine_metrics.csv",
        "summary": output / f"{prefix}_individual_summary.csv",
        "paired_cell": output / f"{prefix}_individual_paired_cells.csv",
        "paired_split": output
        / f"{prefix}_individual_paired_target_splits.csv",
        "comparisons": output
        / f"{prefix}_individual_paired_comparisons.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "ensemble_raw": output / f"{prefix}_ensemble_run_level.json",
        "ensemble_run": output / f"{prefix}_ensemble_run_level.csv",
        "ensemble_predictions": output
        / f"{prefix}_ensemble_window_predictions.csv",
        "ensemble_per_engine": output
        / f"{prefix}_ensemble_per_engine_metrics.csv",
        "ensemble_summary": output / f"{prefix}_ensemble_summary.csv",
        "ensemble_paired": output / f"{prefix}_ensemble_paired_cells.csv",
        "ensemble_comparisons": output
        / f"{prefix}_ensemble_paired_comparisons.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def target_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = (
        f"{model_seed}:{target_split_seed}:experimentA1_2"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def source_signature(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    architecture: str,
    model_seed: int,
    feature_count: int,
    prior: torch.Tensor,
    git_commit: str,
    script_hash: str,
) -> str:
    return a1.canonical_hash(
        {
            "script_version": SCRIPT_VERSION,
            "script_hash": script_hash,
            "git_commit": git_commit,
            "architecture": architecture,
            "model_seed": int(model_seed),
            "target_domain": base["target_domain"],
            "source_domains": list(base["source_domains"]),
            "feature_count": int(feature_count),
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
            "source_pretrain_weight_decay": float(
                base["source_pretrain_weight_decay"]
            ),
            "train_file_hashes": protocol["train_file_hashes"],
            "prior_hash": hashlib.sha256(
                prior.numpy().tobytes()
            ).hexdigest(),
        }
    )


def load_or_train_source(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    architecture: str,
    model_seed: int,
    prior: torch.Tensor,
    git_commit: str,
    script_hash: str,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    first_split = experiment["target_split_seeds"][0]
    cfg = dict(base)
    cfg["seed"] = int(model_seed)
    (
        source_tasks,
        _,
        _,
        _,
        feature_count,
        _,
    ) = a11.prepare_confirmation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=protocol[
            "adaptation_units_by_target_split_seed"
        ][str(first_split)],
    )
    a1.seed_everything(model_seed)
    model = exp17b.build_model_17b(
        architecture,
        feature_count,
        cfg,
        prior,
        prior,
    )
    total, predictor = exp17.parameter_count(model)
    signature = source_signature(
        base=cfg,
        experiment=experiment,
        protocol=protocol,
        architecture=architecture,
        model_seed=model_seed,
        feature_count=feature_count,
        prior=prior,
        git_commit=git_commit,
        script_hash=script_hash,
    )
    cache_path = (
        Path(base["output_dir"])
        / "source_cache"
        / (
            f"experimentA1_2_{architecture}_{base['target_domain']}_"
            f"mseed{model_seed}.pt"
        )
    )
    if cache_path.is_file():
        cached = a1.safe_torch_load(cache_path)
        if cached.get("signature") == signature:
            return (
                cached["state"],
                cached.get("history", []),
                cached["inventory"],
            )

    device = a1.resolve_device(base["device"])
    model, history = train_source_supervised(
        model,
        source_tasks,
        cfg,
        device,
    )
    inventory = {
        "model": architecture,
        "model_seed": int(model_seed),
        "feature_count": int(feature_count),
        "total_parameter_count": int(total),
        "predictor_parameter_count": int(predictor),
        "source_pretrain_steps": int(base["source_pretrain_steps"]),
        "source_signature": signature,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    state = a1.state_to_cpu(model)
    torch.save(
        {
            "signature": signature,
            "state": state,
            "history": history,
            "inventory": inventory,
        },
        cache_path,
    )
    return state, history, inventory


def run_target_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    architecture: str,
    model_seed: int,
    target_split_seed: int,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    prior: torch.Tensor,
    save_checkpoint: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    support_units = protocol["adaptation_units_by_target_split_seed"][
        str(target_split_seed)
    ]
    run_seed = target_run_seed(model_seed, target_split_seed)
    cfg = dict(base)
    cfg["seed"] = int(run_seed)
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
        adaptation_units=support_units,
    )
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(
        architecture,
        feature_count,
        cfg,
        prior,
        prior,
    )
    model.load_state_dict(source_state)
    device = a1.resolve_device(base["device"])
    model, target_history, best_epoch = exp17.train_target_head(
        model,
        support,
        selection,
        cfg,
        device,
    )
    predictions = a1.predict_with_units(model, confirmation, device)
    predictions.insert(0, "window_index", np.arange(len(predictions)))
    metrics = regression_metrics(
        predictions["label"],
        predictions["prediction"],
    )
    replicate_id = (
        f"experimentA1_2_{base['target_domain'].lower()}_"
        f"k{int(experiment['k']):02d}_tsplit{target_split_seed}_"
        f"mseed{model_seed}_{architecture}"
    )
    result = {
        **metrics,
        "experiment_id": "experimentA1_2",
        "experiment_name": "seed_ensemble_stability",
        "replicate_id": replicate_id,
        "evaluation_scope": "train_only_new_independent_confirmation",
        "model": architecture,
        "target_domain": base["target_domain"],
        "k": int(experiment["k"]),
        "target_split_seed": int(target_split_seed),
        "model_seed": int(model_seed),
        "target_run_seed": int(run_seed),
        "adaptation_units": list(map(int, support_units)),
        "selection_units": list(map(int, protocol["selection_units"])),
        "confirmation_units": list(map(int, protocol["confirmation_units"])),
        "best_target_epoch_by_selection": int(best_epoch),
        "source_pretrain_steps": int(base["source_pretrain_steps"]),
        "target_epochs_planned": int(base["target_epochs"]),
        "preprocessing_mode": experiment["preprocessing"],
        "balance_mode": experiment["balance_mode"],
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_parameter_count": inventory[
            "predictor_parameter_count"
        ],
        "source_signature": inventory["source_signature"],
        "source_history_rows": int(len(source_history)),
        "selection_used_for_final_metrics": False,
        "confirmation_used_for_epoch_selection": False,
        "confirmation_historically_unseen": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "split_protocol": split["protocol"],
    }
    for column, value in reversed(
        [
            ("replicate_id", replicate_id),
            ("model", architecture),
            ("model_seed", int(model_seed)),
            ("target_split_seed", int(target_split_seed)),
        ]
    ):
        predictions.insert(0, column, value)
    engine_metrics = a1.per_engine_metrics(predictions)
    engine_metrics.insert(0, "replicate_id", replicate_id)
    engine_metrics.insert(1, "model", architecture)
    engine_metrics.insert(2, "model_seed", int(model_seed))
    engine_metrics.insert(3, "target_split_seed", int(target_split_seed))

    if save_checkpoint:
        checkpoint_path = (
            Path(base["output_dir"])
            / "checkpoints"
            / f"{replicate_id}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": a1.state_to_cpu(model),
                "metrics": result,
                "target_history": target_history,
                "split": split,
            },
            checkpoint_path,
        )
    return result, predictions, engine_metrics


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )


def build_ensemble_outputs(
    *,
    results: list[dict],
    predictions: pd.DataFrame,
    protocol: dict,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    if not results or predictions.empty:
        return [], pd.DataFrame(), pd.DataFrame()
    result_frame = pd.DataFrame(results)
    ensemble_results: list[dict] = []
    ensemble_predictions: list[pd.DataFrame] = []
    ensemble_engines: list[pd.DataFrame] = []

    for size_text, groups in protocol["ensemble_groups"].items():
        size = int(size_text)
        for group_index, members in enumerate(groups, start=1):
            member_set = set(map(int, members))
            for split_seed in protocol["target_split_seeds"]:
                for architecture in ARCHITECTURES:
                    selected = predictions[
                        predictions["model"].eq(architecture)
                        & predictions["target_split_seed"].eq(split_seed)
                        & predictions["model_seed"].isin(member_set)
                    ].copy()
                    counts = selected.groupby("window_index")[
                        "model_seed"
                    ].nunique()
                    if counts.empty or not bool((counts == size).all()):
                        raise RuntimeError(
                            "incomplete prediction members for an ensemble"
                        )
                    labels_per_window = selected.groupby("window_index")[
                        "label"
                    ].nunique()
                    if not bool((labels_per_window == 1).all()):
                        raise AssertionError(
                            "ensemble members have inconsistent labels"
                        )
                    averaged = (
                        selected.groupby(
                            ["window_index", "unit"],
                            as_index=False,
                        )
                        .agg(
                            label=("label", "first"),
                            prediction=("prediction", "mean"),
                        )
                        .sort_values("window_index")
                        .reset_index(drop=True)
                    )
                    averaged["error"] = (
                        averaged["prediction"] - averaged["label"]
                    )
                    averaged["nasa_contribution"] = np.where(
                        averaged["error"] < 0,
                        np.exp(-averaged["error"] / 13.0) - 1.0,
                        np.exp(averaged["error"] / 10.0) - 1.0,
                    )
                    metrics = regression_metrics(
                        averaged["label"],
                        averaged["prediction"],
                    )
                    ensemble_id = (
                        f"experimentA1_2_ensemble{size:02d}_"
                        f"group{group_index:02d}_tsplit{split_seed}_"
                        f"{architecture}"
                    )
                    member_rows = result_frame[
                        result_frame["model"].eq(architecture)
                        & result_frame["target_split_seed"].eq(split_seed)
                        & result_frame["model_seed"].isin(member_set)
                    ]
                    if len(member_rows) != size:
                        raise RuntimeError("ensemble member results are missing")
                    ensemble_results.append(
                        {
                            **metrics,
                            "experiment_id": "experimentA1_2",
                            "ensemble_id": ensemble_id,
                            "model": architecture,
                            "ensemble_size": size,
                            "ensemble_group": int(group_index),
                            "model_seed_members": list(map(int, members)),
                            "target_split_seed": int(split_seed),
                            "k": int(protocol["k"]),
                            "member_rmse_mean": float(
                                member_rows["rmse"].mean()
                            ),
                            "member_rmse_std": float(
                                member_rows["rmse"].std(ddof=1)
                            ),
                            "official_test_files_accessed": False,
                            "official_test_forward_run": False,
                        }
                    )
                    averaged.insert(0, "ensemble_id", ensemble_id)
                    averaged.insert(1, "model", architecture)
                    averaged.insert(2, "ensemble_size", size)
                    averaged.insert(3, "ensemble_group", int(group_index))
                    averaged.insert(
                        4,
                        "target_split_seed",
                        int(split_seed),
                    )
                    ensemble_predictions.append(averaged)
                    engine = a1.per_engine_metrics(averaged)
                    engine.insert(0, "ensemble_id", ensemble_id)
                    engine.insert(1, "model", architecture)
                    engine.insert(2, "ensemble_size", size)
                    engine.insert(3, "ensemble_group", int(group_index))
                    engine.insert(
                        4,
                        "target_split_seed",
                        int(split_seed),
                    )
                    ensemble_engines.append(engine)

    return (
        ensemble_results,
        pd.concat(ensemble_predictions, ignore_index=True),
        pd.concat(ensemble_engines, ignore_index=True),
    )


def ensemble_summary(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame()
    for (size, model), group in frame.groupby(["ensemble_size", "model"]):
        row = {
            "ensemble_size": int(size),
            "model": model,
            "n_ensemble_cells": int(len(group)),
            "n_target_splits": int(group["target_split_seed"].nunique()),
            "n_seed_groups": int(group["ensemble_group"].nunique()),
        }
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_cell_std"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1
                else 0.0
            )
            split_mean = group.groupby("target_split_seed")[metric].mean()
            row[f"{metric}_target_split_std"] = (
                float(split_mean.std(ddof=1))
                if len(split_mean) > 1
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["ensemble_size", "rmse_mean"]
    )


def ensemble_paired_cells(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame()
    for (size, group_id, split_seed), group in frame.groupby(
        ["ensemble_size", "ensemble_group", "target_split_seed"]
    ):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        if not set(ARCHITECTURES).issubset(by_model):
            continue
        candidate = by_model["sensor_graph_prior"]
        reference = by_model["window_graph"]
        row = {
            "ensemble_size": int(size),
            "ensemble_group": int(group_id),
            "target_split_seed": int(split_seed),
            "candidate": "sensor_graph_prior",
            "reference": "window_graph",
        }
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            delta = float(candidate[metric] - reference[metric])
            row[f"{metric}_delta_candidate_minus_reference"] = delta
        row["rmse_candidate_win"] = float(
            row["rmse_delta_candidate_minus_reference"] < 0
        )
        row["nasa_score_candidate_win"] = float(
            row["nasa_score_delta_candidate_minus_reference"] < 0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["ensemble_size", "target_split_seed", "ensemble_group"]
    )


def hierarchical_bootstrap(
    group: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    matrix = group.pivot(
        index="target_split_seed",
        columns="ensemble_group",
        values=column,
    ).sort_index().sort_index(axis=1)
    if matrix.empty or bool(matrix.isna().any().any()):
        return float("nan"), float("nan")
    values = matrix.to_numpy(float)
    n_splits, n_groups = values.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        split_indices = rng.integers(0, n_splits, size=n_splits)
        group_indices = rng.integers(0, n_groups, size=n_groups)
        samples[index] = values[np.ix_(split_indices, group_indices)].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def ensemble_comparisons(
    results: list[dict],
    paired: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    if not results or paired.empty:
        return pd.DataFrame()
    raw = pd.DataFrame(results)
    rows: list[dict] = []
    for size, group in paired.groupby("ensemble_size"):
        rmse_column = "rmse_delta_candidate_minus_reference"
        split_means = group.groupby("target_split_seed")[rmse_column].mean()
        low, high = hierarchical_bootstrap(
            group,
            rmse_column,
            repetitions,
            seed=6120 + int(size),
        )
        reference_rmse = raw[
            raw["ensemble_size"].eq(size)
            & raw["model"].eq("window_graph")
        ]["rmse"].mean()
        nasa_delta = float(
            group["nasa_score_delta_candidate_minus_reference"].mean()
        )
        rows.append(
            {
                "ensemble_size": int(size),
                "n_paired_cells": int(len(group)),
                "n_target_splits": int(
                    group["target_split_seed"].nunique()
                ),
                "n_seed_groups": int(group["ensemble_group"].nunique()),
                "rmse_delta_mean": float(group[rmse_column].mean()),
                "rmse_improvement_pct": float(
                    -100.0 * group[rmse_column].mean() / reference_rmse
                ),
                "rmse_cell_win_rate": float(
                    group["rmse_candidate_win"].mean()
                ),
                "rmse_target_split_win_rate": float(
                    (split_means < 0).mean()
                ),
                "rmse_hier_boot_ci95_low": low,
                "rmse_hier_boot_ci95_high": high,
                "rmse_split_t_p": exp17b.split_level_pvalue(
                    split_means.to_numpy(float)
                ),
                "mae_delta_mean": float(
                    group[
                        "mae_delta_candidate_minus_reference"
                    ].mean()
                ),
                "r2_delta_mean": float(
                    group["r2_delta_candidate_minus_reference"].mean()
                ),
                "nasa_score_delta_mean": nasa_delta,
                "nasa_score_target_split_win_rate": float(
                    (
                        group.groupby("target_split_seed")[
                            "nasa_score_delta_candidate_minus_reference"
                        ].mean()
                        < 0
                    ).mean()
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("ensemble_size").reset_index(
        drop=True
    )
    frame["rmse_split_t_p_holm"] = exp17b.holm_adjust(
        frame["rmse_split_t_p"].tolist()
    )
    frame["strict_success"] = (
        (frame["rmse_improvement_pct"] >= 3.0)
        & (frame["rmse_target_split_win_rate"] >= 0.8)
        & (frame["rmse_hier_boot_ci95_high"] < 0.0)
        & (frame["rmse_split_t_p_holm"] < 0.05)
        & (frame["nasa_score_delta_mean"] <= 0.0)
    )
    return frame


def save_ensemble_outputs(
    *,
    paths: dict[str, Path],
    results: list[dict],
    predictions: pd.DataFrame,
    engines: pd.DataFrame,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = ensemble_summary(results)
    paired = ensemble_paired_cells(results)
    comparisons = ensemble_comparisons(results, paired, repetitions)
    atomic_json(paths["ensemble_raw"], results)
    a1.atomic_write_text(
        paths["ensemble_run"],
        pd.DataFrame(results).to_csv(index=False),
    )
    a1.atomic_write_text(
        paths["ensemble_predictions"],
        predictions.to_csv(index=False),
    )
    a1.atomic_write_text(
        paths["ensemble_per_engine"],
        engines.to_csv(index=False),
    )
    a1.atomic_write_text(paths["ensemble_summary"], summary.to_csv(index=False))
    a1.atomic_write_text(paths["ensemble_paired"], paired.to_csv(index=False))
    a1.atomic_write_text(
        paths["ensemble_comparisons"],
        comparisons.to_csv(index=False),
    )
    return summary, comparisons


def confirmation_decision(
    *,
    experiment: dict,
    individual_results: list[dict],
    ensemble_results: list[dict],
    ensemble_summary_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> dict:
    expected_individual = (
        len(experiment["architectures"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    groups = ensemble_groups(experiment["model_seeds"])
    expected_ensemble = (
        len(experiment["architectures"])
        * len(experiment["target_split_seeds"])
        * sum(len(value) for value in groups.values())
    )
    decision: dict[str, Any] = {
        "experiment_id": "experimentA1_2",
        "expected_individual_cells": int(expected_individual),
        "completed_individual_cells": int(len(individual_results)),
        "expected_ensemble_cells": int(expected_ensemble),
        "completed_ensemble_cells": int(len(ensemble_results)),
        "complete": (
            len(individual_results) == expected_individual
            and len(ensemble_results) == expected_ensemble
        ),
        "primary_ensemble_size": int(max(experiment["ensemble_sizes"])),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "quick_mode": bool(experiment.get("quick_mode", False)),
    }
    if experiment.get("quick_mode", False):
        decision["passed"] = bool(decision["complete"])
        decision["reason"] = (
            "quick smoke experiment completed"
            if decision["passed"]
            else "quick smoke experiment is incomplete"
        )
        return decision
    primary_size = int(max(experiment["ensemble_sizes"]))
    selected = comparison_frame[
        comparison_frame["ensemble_size"].eq(primary_size)
    ]
    if len(selected) != 1:
        decision["passed"] = False
        decision["reason"] = "primary ensemble comparison is missing"
        return decision
    row = selected.iloc[0]
    primary = {
        "ensemble_size": primary_size,
        "rmse_delta_mean": float(row["rmse_delta_mean"]),
        "rmse_improvement_pct": float(row["rmse_improvement_pct"]),
        "rmse_target_split_win_rate": float(
            row["rmse_target_split_win_rate"]
        ),
        "rmse_hier_boot_ci95_low": float(
            row["rmse_hier_boot_ci95_low"]
        ),
        "rmse_hier_boot_ci95_high": float(
            row["rmse_hier_boot_ci95_high"]
        ),
        "rmse_split_t_p_holm": float(row["rmse_split_t_p_holm"]),
        "nasa_score_delta_mean": float(row["nasa_score_delta_mean"]),
        "strict_success": bool(row["strict_success"]),
    }
    decision["primary_result"] = primary

    prior_primary = ensemble_summary_frame[
        ensemble_summary_frame["ensemble_size"].eq(primary_size)
        & ensemble_summary_frame["model"].eq("sensor_graph_prior")
    ]
    if len(prior_primary) == 1:
        decision["prior_ensemble_rmse_target_split_std"] = float(
            prior_primary.iloc[0]["rmse_target_split_std"]
        )
    individual_frame = pd.DataFrame(individual_results)
    decision["prior_individual_rmse_cell_std"] = float(
        individual_frame[
            individual_frame["model"].eq("sensor_graph_prior")
        ]["rmse"].std(ddof=1)
    )
    decision["passed"] = bool(decision["complete"] and primary["strict_success"])
    decision["reason"] = (
        "A1_2 confirmed that seed ensembling stabilizes the prior sensor graph"
        if decision["passed"]
        else (
            "A1_2 completed, but the 10-seed prior ensemble did not meet "
            "the registered stability criteria"
        )
    )
    return decision


def run_signature(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    prior: torch.Tensor,
    script_hash: str,
) -> str:
    return a1.canonical_hash(
        {
            "script_version": SCRIPT_VERSION,
            "script_hash": script_hash,
            "base": {
                key: value
                for key, value in base.items()
                if key not in {"output_dir", "device"}
            },
            "experiment": {
                key: value
                for key, value in experiment.items()
                if key != "output_dir"
            },
            "protocol_hash": protocol["protocol_hash"],
            "prior_hash": hashlib.sha256(
                prior.numpy().tobytes()
            ).hexdigest(),
        }
    )


def main() -> None:
    args = parse_args()
    base, experiment = load_config(args)
    validate_config(base, experiment)
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = result_paths(output)
    git_commit = a1.git_commit(PROJECT_ROOT)
    script_hash = a1.file_sha256(Path(__file__))

    protocol = build_protocol(base, experiment)
    prior, correlation, graph_fit = (
        a1.source_correlation_adjacency_train_only(
            base,
            experiment["preprocessing"],
            int(experiment["sensor_graph_k"]),
        )
    )
    signature = run_signature(
        base=base,
        experiment=experiment,
        protocol=protocol,
        prior=prior,
        script_hash=script_hash,
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": script_hash,
        "git_commit": git_commit,
        "run_signature": signature,
        "base_config": base,
        "experiment_config": experiment,
        "protocol_hash": protocol["protocol_hash"],
        "graph_fit": graph_fit,
        "registered_primary_comparison": (
            "10-seed sensor_graph_prior ensemble vs "
            "10-seed window_graph ensemble"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file() and paths["raw"].is_file():
        previous = json.loads(
            paths["manifest"].read_text(encoding="utf-8")
        )
        if previous.get("run_signature") != signature:
            raise RuntimeError(
                "existing A1_2 results use a different protocol; "
                "use a new --output-dir"
            )
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["protocol"], protocol)
    a1.atomic_write_text(
        paths["engine_roles"],
        protocol_frame(protocol).to_csv(index=False),
    )
    sensors = list(base["sensor_columns"])
    a1.atomic_write_text(
        paths["prior_adjacency"],
        pd.DataFrame(
            prior.numpy().astype(int),
            index=sensors,
            columns=sensors,
        ).to_csv(),
    )
    a1.atomic_write_text(
        paths["prior_correlation"],
        pd.DataFrame(
            correlation,
            index=sensors,
            columns=sensors,
        ).to_csv(),
    )

    first_seed = experiment["model_seeds"][0]
    first_split = experiment["target_split_seeds"][0]
    dry_cfg = dict(base)
    dry_cfg["seed"] = int(first_seed)
    (
        source_tasks,
        support,
        selection,
        confirmation,
        feature_count,
        split,
    ) = a11.prepare_confirmation_experiment(
        dry_cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=protocol[
            "adaptation_units_by_target_split_seed"
        ][str(first_split)],
    )
    a1.seed_everything(first_seed)
    dry_models = {
        architecture: exp17b.build_model_17b(
            architecture,
            feature_count,
            dry_cfg,
            prior,
            prior,
        )
        for architecture in ARCHITECTURES
    }
    x, _ = next(iter(source_tasks[base["source_domains"][0]]))
    dry_report = {
        "experiment_id": "experimentA1_2",
        "expected_individual_cells": (
            len(experiment["architectures"])
            * len(experiment["model_seeds"])
            * len(experiment["target_split_seeds"])
        ),
        "feature_count": int(feature_count),
        "source_batch_shape": list(x.shape),
        "support_windows": int(len(support.dataset)),
        "selection_windows": int(len(selection.dataset)),
        "confirmation_windows": int(len(confirmation.dataset)),
        "model_output_shapes": {
            name: list(model(x[: min(4, len(x))]).shape)
            for name, model in dry_models.items()
        },
        "ensemble_groups": protocol["ensemble_groups"],
        "split": split,
        "run_signature": signature,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry_report)
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return

    (
        individual_results,
        prediction_parts,
        engine_parts,
        inventory_rows,
    ) = a11.load_resume_state(paths)
    completed = a11.completed_keys(individual_results)

    for model_seed in experiment["model_seeds"]:
        cfg = dict(base)
        cfg["seed"] = int(model_seed)
        for architecture in experiment["architectures"]:
            pending = [
                split_seed
                for split_seed in experiment["target_split_seeds"]
                if (
                    int(split_seed),
                    int(model_seed),
                    architecture,
                )
                not in completed
            ]
            if not pending:
                continue
            source_state, source_history, inventory = (
                load_or_train_source(
                    base=cfg,
                    experiment=experiment,
                    protocol=protocol,
                    architecture=architecture,
                    model_seed=model_seed,
                    prior=prior,
                    git_commit=git_commit,
                    script_hash=script_hash,
                )
            )
            inventory_rows = [
                row
                for row in inventory_rows
                if not (
                    str(row.get("model")) == architecture
                    and int(row.get("model_seed", -1)) == model_seed
                )
            ]
            inventory_rows.append(inventory)
            for split_seed in pending:
                result, predictions, engine_metrics = run_target_cell(
                    base=cfg,
                    experiment=experiment,
                    protocol=protocol,
                    architecture=architecture,
                    model_seed=model_seed,
                    target_split_seed=split_seed,
                    source_state=deepcopy(source_state),
                    source_history=source_history,
                    inventory=inventory,
                    prior=prior,
                    save_checkpoint=args.save_target_checkpoints,
                )
                individual_results.append(result)
                prediction_parts.append(predictions)
                engine_parts.append(engine_metrics)
                completed.add(
                    (
                        int(split_seed),
                        int(model_seed),
                        architecture,
                    )
                )
                a1.save_progress(
                    paths=paths,
                    results=individual_results,
                    predictions=prediction_parts,
                    engine_metrics=engine_parts,
                    inventory=inventory_rows,
                    bootstrap_repetitions=min(
                        200,
                        int(experiment["bootstrap_repetitions"]),
                    ),
                )

    a1.save_progress(
        paths=paths,
        results=individual_results,
        predictions=prediction_parts,
        engine_metrics=engine_parts,
        inventory=inventory_rows,
        bootstrap_repetitions=int(experiment["bootstrap_repetitions"]),
    )
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    ensemble_results, ensemble_prediction_frame, ensemble_engine_frame = (
        build_ensemble_outputs(
            results=individual_results,
            predictions=all_predictions,
            protocol=protocol,
        )
    )
    ensemble_summary_frame, comparison_frame = save_ensemble_outputs(
        paths=paths,
        results=ensemble_results,
        predictions=ensemble_prediction_frame,
        engines=ensemble_engine_frame,
        repetitions=int(experiment["bootstrap_repetitions"]),
    )
    decision = confirmation_decision(
        experiment=experiment,
        individual_results=individual_results,
        ensemble_results=ensemble_results,
        ensemble_summary_frame=ensemble_summary_frame,
        comparison_frame=comparison_frame,
    )
    atomic_json(paths["decision"], decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
