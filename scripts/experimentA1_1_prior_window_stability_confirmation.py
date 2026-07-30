"""Experiment A1_1: independent stability confirmation of the prior sensor graph.

Experiment A1 reproduced Experiment 17B and established that the source-only
correlation sensor graph beats both a sensor no-graph control and a
degree-matched random graph.  Its average result also beat the original window
graph, but that primary comparison did not pass the hierarchical-bootstrap
criterion.

A1_1 keeps the same research direction and tests only the remaining question:
does the prior sensor graph reliably beat the window graph on engines that were
not used for A1 validation or A1 target adaptation?

The registered formal design is:

* FD004 target domain and K=5 labelled adaptation engines;
* 30 fixed confirmation engines unseen by A1 validation/adaptation;
* the 20 former A1 validation engines used only for target-epoch selection;
* 10 new target split seeds crossed with 10 new model seeds;
* three regimes: prior sensor graph, window graph and window no-graph;
* 300 train-only confirmation cells in total;
* no official ``test_FDxxx.txt`` or ``RUL_FDxxx.txt`` access.

Run once from the repository root:

    python -u scripts/experimentA1_1_prior_window_stability_confirmation.py

All artifacts are written below
``outputs/experimentA1_1_prior_window_stability_confirmation``.
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
from preprocess.rul_generator import add_train_rul  # noqa: E402
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    train_source_supervised,
)


SCRIPT_VERSION = "experimentA1_1_prior_window_stability_confirmation_v1"
MODELS = ("sensor_graph_prior", "window_graph", "window_no_graph")

DEFAULT_EXPERIMENT = {
    "experiment_id": "experimentA1_1",
    "experiment_name": "prior_window_stability_confirmation",
    "target_domain": "FD004",
    "preprocessing": "condition_settings",
    "balance_mode": "engine_stage",
    "k": 5,
    "sensor_graph_k": 4,
    "selection_count": 20,
    "selection_seed": 2026,
    "a1_exposure_target_split_seeds": [3027, 3028, 3029, 3030, 3031],
    "a1_exposure_k": 5,
    "confirmation_count": 30,
    "confirmation_seed": 4101,
    "target_split_seeds": list(range(4201, 4211)),
    "model_seeds": list(range(52, 62)),
    "models": list(MODELS),
    "source_pretrain_steps": 1500,
    "source_pretrain_lr": 0.001,
    "source_pretrain_weight_decay": 0.0,
    "target_epochs": 10,
    "target_lr": 0.001,
    "bootstrap_repetitions": 10000,
    "minimum_rmse_improvement_pct": 3.0,
    "minimum_target_split_win_rate": 0.8,
    "output_dir": (
        "outputs/experimentA1_1_prior_window_stability_confirmation"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment A1_1: independent prior-sensor vs window-graph "
            "stability confirmation"
        )
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a one-split, one-seed smoke experiment in a separate output",
    )
    return parser.parse_args()


def resolve_optional_path(value: str | None, fallback: str) -> str:
    selected = fallback if value is None else value
    return str(a1.resolve_path(selected))


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    experiment = deepcopy(DEFAULT_EXPERIMENT)
    base["target_domain"] = experiment["target_domain"]
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
        experiment["target_split_seeds"] = [4201]
        experiment["model_seeds"] = [52]
        experiment["source_pretrain_steps"] = 5
        experiment["target_epochs"] = 1
        experiment["bootstrap_repetitions"] = 100
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 1
        if args.output_dir is None:
            base["output_dir"] = str(
                a1.resolve_path(
                    "outputs/"
                    "experimentA1_1_prior_window_stability_confirmation_quick"
                )
            )
    experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if base["target_domain"] != "FD004":
        raise ValueError("Experiment A1_1 is registered for FD004 only")
    if experiment["k"] <= 0:
        raise ValueError("K must be positive")
    if len(set(experiment["models"])) != len(experiment["models"]):
        raise ValueError("models contains duplicates")
    if set(experiment["models"]) != set(MODELS):
        raise ValueError(f"A1_1 requires exactly these models: {MODELS}")
    if len(set(experiment["target_split_seeds"])) != len(
        experiment["target_split_seeds"]
    ):
        raise ValueError("target split seeds contain duplicates")
    if len(set(experiment["model_seeds"])) != len(experiment["model_seeds"]):
        raise ValueError("model seeds contain duplicates")
    if experiment["preprocessing"] not in a1.PREPROCESSING_MODES:
        raise ValueError("unknown preprocessing mode")
    if experiment["balance_mode"] not in a1.BALANCE_MODES:
        raise ValueError("unknown balance mode")
    if not 1 <= int(experiment["sensor_graph_k"]) < len(
        base["sensor_columns"]
    ):
        raise ValueError("sensor_graph_k is outside the valid range")


def build_confirmation_protocol(base: dict, experiment: dict) -> dict:
    target = base["target_domain"]
    frame = a1.load_train_domain(base["data_dir"], target)
    units = np.asarray(sorted(frame["unit"].unique()), dtype=int)
    if not 1 <= int(experiment["selection_count"]) < len(units):
        raise ValueError("selection engine count is invalid")

    selection_order = np.random.default_rng(
        int(experiment["selection_seed"])
    ).permutation(units)
    selection_units = selection_order[
        : int(experiment["selection_count"])
    ]
    selection_set = set(selection_units.tolist())
    a1_candidates = np.asarray(
        [unit for unit in units if int(unit) not in selection_set],
        dtype=int,
    )

    a1_exposed: set[int] = set()
    exposure_rows: list[dict] = []
    for split_seed in experiment["a1_exposure_target_split_seeds"]:
        order = np.random.default_rng(int(split_seed)).permutation(
            a1_candidates
        )
        exposed = order[: int(experiment["a1_exposure_k"])]
        a1_exposed.update(int(unit) for unit in exposed)
        exposure_rows.append(
            {
                "target_split_seed": int(split_seed),
                "units": exposed.astype(int).tolist(),
            }
        )

    confirmation_pool = np.asarray(
        [
            unit
            for unit in units
            if int(unit) not in selection_set
            and int(unit) not in a1_exposed
        ],
        dtype=int,
    )
    confirmation_count = int(experiment["confirmation_count"])
    if confirmation_count > len(confirmation_pool):
        raise ValueError("not enough A1-unseen engines for confirmation")
    confirmation_order = np.random.default_rng(
        int(experiment["confirmation_seed"])
    ).permutation(confirmation_pool)
    confirmation_units = confirmation_order[:confirmation_count]
    confirmation_set = set(confirmation_units.tolist())

    adaptation_candidates = np.asarray(
        [
            unit
            for unit in units
            if int(unit) not in selection_set
            and int(unit) not in confirmation_set
        ],
        dtype=int,
    )
    if int(experiment["k"]) > len(adaptation_candidates):
        raise ValueError("K exceeds available adaptation engines")

    adaptation_units: dict[str, list[int]] = {}
    for split_seed in experiment["target_split_seeds"]:
        order = np.random.default_rng(int(split_seed)).permutation(
            adaptation_candidates
        )
        selected = order[: int(experiment["k"])].astype(int).tolist()
        if len(set(selected)) != int(experiment["k"]):
            raise AssertionError("adaptation unit count does not match K")
        if set(selected) & selection_set:
            raise AssertionError("adaptation and selection engines overlap")
        if set(selected) & confirmation_set:
            raise AssertionError("adaptation and confirmation engines overlap")
        adaptation_units[str(split_seed)] = selected

    train_file_hashes = {
        domain: a1.file_sha256(a1.train_path(base["data_dir"], domain))
        for domain in [*base["source_domains"], target]
    }
    protocol = {
        "protocol_version": "experimentA1_1_independent_confirmation_v1",
        "target_domain": target,
        "source_domains": list(base["source_domains"]),
        "train_engine_count": int(len(units)),
        "selection_seed": int(experiment["selection_seed"]),
        "selection_units": selection_units.astype(int).tolist(),
        "selection_role": "target_epoch_selection_only",
        "a1_exposure_target_split_seeds": [
            int(value)
            for value in experiment["a1_exposure_target_split_seeds"]
        ],
        "a1_exposure_k": int(experiment["a1_exposure_k"]),
        "a1_exposed_support_by_split": exposure_rows,
        "a1_exposed_support_units": sorted(a1_exposed),
        "confirmation_seed": int(experiment["confirmation_seed"]),
        "confirmation_units": confirmation_units.astype(int).tolist(),
        "confirmation_role": "final_metrics_only",
        "confirmation_units_seen_in_a1_validation": False,
        "confirmation_units_seen_in_a1_adaptation": False,
        "adaptation_candidate_count": int(len(adaptation_candidates)),
        "k": int(experiment["k"]),
        "target_split_seeds": [
            int(value) for value in experiment["target_split_seeds"]
        ],
        "model_seeds": [
            int(value) for value in experiment["model_seeds"]
        ],
        "adaptation_units_by_target_split_seed": adaptation_units,
        "train_file_hashes": train_file_hashes,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    protocol["protocol_hash"] = a1.canonical_hash(protocol)
    return protocol


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
                "role": "independent_confirmation",
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
                    "role": "adaptation",
                    "unit": int(unit),
                }
            )
    return pd.DataFrame(rows)


def prepare_confirmation_experiment(
    cfg: dict,
    preprocessing: str,
    balance_mode: str,
    *,
    selection_units: list[int],
    confirmation_units: list[int],
    adaptation_units: list[int],
):
    sensors = list(cfg["sensor_columns"])
    source_frames, normalizer = a1.fit_source_normalizer_train_only(
        cfg,
        preprocessing,
    )
    target_train = add_train_rul(
        a1.load_train_domain(cfg["data_dir"], cfg["target_domain"]),
        cfg["rul_cap"],
    )
    include_settings = preprocessing in {
        "global_settings",
        "condition_settings",
    }
    features = (
        sensors + a1.SETTING_FEATURE_COLUMNS
        if include_settings
        else sensors
    )
    normalized_sources = {
        domain: normalizer.transform(frame, sensors)
        for domain, frame in source_frames.items()
    }
    normalized_target = normalizer.transform(target_train, sensors)

    source_tasks = {
        domain: a1.make_loader(
            normalized_sources[domain],
            features,
            cfg,
            training=True,
            balance_mode=balance_mode,
            loader_seed=cfg["seed"] + 1000 * (index + 1),
        )
        for index, domain in enumerate(cfg["source_domains"])
    }

    adaptation = np.asarray(adaptation_units, dtype=int)
    selection = np.asarray(selection_units, dtype=int)
    confirmation = np.asarray(confirmation_units, dtype=int)
    if set(adaptation.tolist()) & set(selection.tolist()):
        raise AssertionError("adaptation and selection engines overlap")
    if set(adaptation.tolist()) & set(confirmation.tolist()):
        raise AssertionError("adaptation and confirmation engines overlap")
    if set(selection.tolist()) & set(confirmation.tolist()):
        raise AssertionError("selection and confirmation engines overlap")

    support_frame = normalized_target.query("unit in @adaptation")
    selection_frame = normalized_target.query("unit in @selection")
    confirmation_frame = normalized_target.query("unit in @confirmation")
    if support_frame["unit"].nunique() != len(adaptation_units):
        raise ValueError("support engine count does not match K")
    if selection_frame["unit"].nunique() != len(selection_units):
        raise ValueError("selection engine count is inconsistent")
    if confirmation_frame["unit"].nunique() != len(confirmation_units):
        raise ValueError("confirmation engine count is inconsistent")

    support = a1.make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        loader_seed=cfg["seed"] + 9000,
    )
    selection_loader = a1.make_loader(
        selection_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9100,
    )
    confirmation_loader = a1.make_loader(
        confirmation_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9200,
    )
    split = {
        "protocol": "experimentA1_1_train_only_independent_confirmation",
        "target_domain": cfg["target_domain"],
        "adaptation_units": [int(value) for value in adaptation_units],
        "selection_units": [int(value) for value in selection_units],
        "confirmation_units": [
            int(value) for value in confirmation_units
        ],
        "feature_columns": list(features),
        "normalizer_fit_scope": "source_train_only",
        "target_epoch_selection_scope": "selection_engines_only",
        "final_metric_scope": "independent_confirmation_engines_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return (
        source_tasks,
        support,
        selection_loader,
        confirmation_loader,
        len(features),
        split,
    )


def result_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA1_1"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_splits": output / f"{prefix}_engine_roles.csv",
        "prior_adjacency": output / f"{prefix}_prior_adjacency.csv",
        "prior_correlation": output / f"{prefix}_prior_correlation.csv",
        "raw": output / f"{prefix}_run_level.json",
        "run_csv": output / f"{prefix}_run_level.csv",
        "window_predictions": output / f"{prefix}_window_predictions.csv",
        "per_engine": output / f"{prefix}_per_engine_metrics.csv",
        "summary": output / f"{prefix}_summary.csv",
        "paired_cell": output / f"{prefix}_paired_cells.csv",
        "paired_split": output / f"{prefix}_paired_target_splits.csv",
        "comparisons": output / f"{prefix}_paired_comparisons.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
        "dry_run": output / f"{prefix}_dry_run.json",
    }


def target_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = (
        f"{model_seed}:{target_split_seed}:experimentA1_1"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def source_signature(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    feature_count: int,
    prior: torch.Tensor,
    git_commit: str,
    script_hash: str,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "script_hash": script_hash,
        "git_commit": git_commit,
        "model": model_name,
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
    return a1.canonical_hash(payload)


def load_or_train_source(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    prior: torch.Tensor,
    git_commit: str,
    script_hash: str,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    first_split = experiment["target_split_seeds"][0]
    units = protocol["adaptation_units_by_target_split_seed"][
        str(first_split)
    ]
    cfg = dict(base)
    cfg["seed"] = int(model_seed)
    (
        source_tasks,
        _,
        _,
        _,
        feature_count,
        _,
    ) = prepare_confirmation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=units,
    )

    a1.seed_everything(model_seed)
    model = exp17b.build_model_17b(
        model_name,
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
        model_name=model_name,
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
            f"experimentA1_1_{model_name}_{base['target_domain']}_"
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
        "model": model_name,
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
    model_name: str,
    model_seed: int,
    target_split_seed: int,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    prior: torch.Tensor,
    save_checkpoint: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    units = protocol["adaptation_units_by_target_split_seed"][
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
    ) = prepare_confirmation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=units,
    )

    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(
        model_name,
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
    window_predictions = a1.predict_with_units(
        model,
        confirmation,
        device,
    )
    metrics = regression_metrics(
        window_predictions["label"],
        window_predictions["prediction"],
    )
    replicate_id = (
        f"experimentA1_1_{base['target_domain'].lower()}_"
        f"k{int(experiment['k']):02d}_tsplit{target_split_seed}_"
        f"mseed{model_seed}_{model_name}"
    )
    result = {
        **metrics,
        "experiment_id": "experimentA1_1",
        "experiment_name": "prior_window_stability_confirmation",
        "replicate_id": replicate_id,
        "evaluation_scope": "train_only_independent_confirmation",
        "model": model_name,
        "target_domain": base["target_domain"],
        "k": int(experiment["k"]),
        "target_split_seed": int(target_split_seed),
        "model_seed": int(model_seed),
        "target_run_seed": int(run_seed),
        "adaptation_units": [int(value) for value in units],
        "selection_units": [
            int(value) for value in protocol["selection_units"]
        ],
        "confirmation_units": [
            int(value) for value in protocol["confirmation_units"]
        ],
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
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "split_protocol": split["protocol"],
    }
    window_predictions.insert(0, "replicate_id", replicate_id)
    engine_metrics = a1.per_engine_metrics(window_predictions)
    engine_metrics.insert(0, "replicate_id", replicate_id)

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
    return result, window_predictions, engine_metrics


def load_csv_if_populated(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_resume_state(
    paths: dict[str, Path],
) -> tuple[
    list[dict],
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[dict],
]:
    results: list[dict] = []
    if paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
    prediction_frame = load_csv_if_populated(paths["window_predictions"])
    engine_frame = load_csv_if_populated(paths["per_engine"])
    inventory_frame = load_csv_if_populated(paths["inventory"])
    if not results:
        return (
            [],
            [prediction_frame] if not prediction_frame.empty else [],
            [engine_frame] if not engine_frame.empty else [],
            inventory_frame.to_dict("records"),
        )

    result_ids = {str(row["replicate_id"]) for row in results}
    prediction_ids = (
        set(prediction_frame["replicate_id"].astype(str))
        if not prediction_frame.empty
        else set()
    )
    engine_ids = (
        set(engine_frame["replicate_id"].astype(str))
        if not engine_frame.empty
        else set()
    )
    consistent_ids = result_ids & prediction_ids & engine_ids
    results = [
        row for row in results if str(row["replicate_id"]) in consistent_ids
    ]
    if not prediction_frame.empty:
        prediction_frame = prediction_frame[
            prediction_frame["replicate_id"].astype(str).isin(consistent_ids)
        ]
    if not engine_frame.empty:
        engine_frame = engine_frame[
            engine_frame["replicate_id"].astype(str).isin(consistent_ids)
        ]
    return (
        results,
        [prediction_frame] if not prediction_frame.empty else [],
        [engine_frame] if not engine_frame.empty else [],
        inventory_frame.to_dict("records"),
    )


def completed_keys(
    results: list[dict],
) -> set[tuple[int, int, str]]:
    return {
        (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            str(row["model"]),
        )
        for row in results
    }


def primary_decision(
    *,
    experiment: dict,
    results: list[dict],
) -> dict:
    expected = (
        len(experiment["models"])
        * len(experiment["target_split_seeds"])
        * len(experiment["model_seeds"])
    )
    decision: dict[str, Any] = {
        "experiment_id": "experimentA1_1",
        "expected_cells": int(expected),
        "completed_cells": int(len(results)),
        "complete": len(results) == expected,
        "primary_comparison": "prior_sensor_vs_window_graph",
        "minimum_rmse_improvement_pct": float(
            experiment["minimum_rmse_improvement_pct"]
        ),
        "minimum_target_split_win_rate": float(
            experiment["minimum_target_split_win_rate"]
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "quick_mode": bool(experiment.get("quick_mode", False)),
    }
    if not results:
        decision["passed"] = False
        decision["reason"] = "no completed confirmation cells"
        return decision
    if experiment.get("quick_mode", False):
        decision["passed"] = bool(decision["complete"])
        decision["reason"] = (
            "quick smoke experiment completed"
            if decision["passed"]
            else "quick smoke experiment is incomplete"
        )
        return decision

    paired = exp17b.paired_cells(results)
    comparisons = exp17b.comparison_summary(
        results,
        paired,
        int(experiment["bootstrap_repetitions"]),
    )
    selected = comparisons[
        comparisons["comparison"] == "prior_sensor_vs_window_graph"
    ]
    if len(selected) != 1:
        decision["passed"] = False
        decision["reason"] = "primary paired comparison is missing"
        return decision
    row = selected.iloc[0]
    primary = {
        "k": int(row["k"]),
        "candidate": str(row["candidate"]),
        "reference": str(row["reference"]),
        "rmse_delta_mean": float(row["rmse_delta_mean"]),
        "rmse_improvement_pct": float(row["rmse_improvement_pct"]),
        "rmse_cell_win_rate": float(row["rmse_cell_win_rate"]),
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
    decision["passed"] = bool(
        decision["complete"]
        and primary["strict_success"]
        and primary["rmse_improvement_pct"]
        >= float(experiment["minimum_rmse_improvement_pct"])
        and primary["rmse_target_split_win_rate"]
        >= float(experiment["minimum_target_split_win_rate"])
    )
    decision["reason"] = (
        "A1_1 independently confirmed a stable prior-sensor-graph advantage"
        if decision["passed"]
        else (
            "A1_1 completed, but the prior-sensor advantage did not meet "
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
    base_payload = {
        key: value
        for key, value in base.items()
        if key not in {"output_dir", "device"}
    }
    experiment_payload = {
        key: value
        for key, value in experiment.items()
        if key != "output_dir"
    }
    return a1.canonical_hash(
        {
            "script_version": SCRIPT_VERSION,
            "script_hash": script_hash,
            "base": base_payload,
            "experiment": experiment_payload,
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

    protocol = build_confirmation_protocol(base, experiment)
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
        "prior_graph_hash": hashlib.sha256(
            prior.numpy().tobytes()
        ).hexdigest(),
        "registered_primary_comparison": (
            "sensor_graph_prior vs window_graph"
        ),
        "official_test_policy": (
            "official test and RUL files are not loaded in Experiment A1_1"
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
                "existing A1_1 results use a different registered protocol; "
                "use a new --output-dir instead of mixing runs"
            )

    a1.atomic_write_text(
        paths["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
    )
    a1.atomic_write_text(
        paths["protocol"],
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
    )
    a1.atomic_write_text(
        paths["engine_splits"],
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
    first_units = protocol["adaptation_units_by_target_split_seed"][
        str(first_split)
    ]
    dry_cfg = dict(base)
    dry_cfg["seed"] = int(first_seed)
    (
        source_tasks,
        support,
        selection,
        confirmation,
        feature_count,
        dry_split,
    ) = prepare_confirmation_experiment(
        dry_cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        selection_units=protocol["selection_units"],
        confirmation_units=protocol["confirmation_units"],
        adaptation_units=first_units,
    )
    a1.seed_everything(first_seed)
    dry_models = {
        model_name: exp17b.build_model_17b(
            model_name,
            feature_count,
            dry_cfg,
            prior,
            prior,
        )
        for model_name in experiment["models"]
    }
    x, _ = next(iter(source_tasks[base["source_domains"][0]]))
    dry_shapes = {
        name: list(model(x[: min(4, len(x))]).shape)
        for name, model in dry_models.items()
    }
    dry_report = {
        "experiment_id": "experimentA1_1",
        "expected_formal_cells": (
            len(DEFAULT_EXPERIMENT["models"])
            * len(DEFAULT_EXPERIMENT["target_split_seeds"])
            * len(DEFAULT_EXPERIMENT["model_seeds"])
        ),
        "feature_count": int(feature_count),
        "source_batch_shape": list(x.shape),
        "support_windows": int(len(support.dataset)),
        "selection_windows": int(len(selection.dataset)),
        "confirmation_windows": int(len(confirmation.dataset)),
        "model_output_shapes": dry_shapes,
        "split": dry_split,
        "run_signature": signature,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    a1.atomic_write_text(
        paths["dry_run"],
        json.dumps(dry_report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return

    (
        results,
        predictions,
        engine_metrics,
        inventory_rows,
    ) = load_resume_state(paths)
    done = completed_keys(results)

    for model_seed in experiment["model_seeds"]:
        cfg = dict(base)
        cfg["seed"] = int(model_seed)
        for model_name in experiment["models"]:
            pending = [
                split_seed
                for split_seed in experiment["target_split_seeds"]
                if (
                    int(split_seed),
                    int(model_seed),
                    str(model_name),
                )
                not in done
            ]
            if not pending:
                continue
            source_state, source_history, inventory = (
                load_or_train_source(
                    base=cfg,
                    experiment=experiment,
                    protocol=protocol,
                    model_name=model_name,
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
                    str(row.get("model")) == model_name
                    and int(row.get("model_seed", -1)) == model_seed
                )
            ]
            inventory_rows.append(inventory)

            for split_seed in pending:
                result, window_frame, engine_frame = run_target_cell(
                    base=cfg,
                    experiment=experiment,
                    protocol=protocol,
                    model_name=model_name,
                    model_seed=model_seed,
                    target_split_seed=split_seed,
                    source_state=deepcopy(source_state),
                    source_history=source_history,
                    inventory=inventory,
                    prior=prior,
                    save_checkpoint=args.save_target_checkpoints,
                )
                results.append(result)
                predictions.append(window_frame)
                engine_metrics.append(engine_frame)
                done.add(
                    (
                        int(split_seed),
                        int(model_seed),
                        str(model_name),
                    )
                )
                a1.save_progress(
                    paths=paths,
                    results=results,
                    predictions=predictions,
                    engine_metrics=engine_metrics,
                    inventory=inventory_rows,
                    bootstrap_repetitions=min(
                        200,
                        int(experiment["bootstrap_repetitions"]),
                    ),
                )

    a1.save_progress(
        paths=paths,
        results=results,
        predictions=predictions,
        engine_metrics=engine_metrics,
        inventory=inventory_rows,
        bootstrap_repetitions=int(
            experiment["bootstrap_repetitions"]
        ),
    )
    decision = primary_decision(
        experiment=experiment,
        results=results,
    )
    a1.atomic_write_text(
        paths["decision"],
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
