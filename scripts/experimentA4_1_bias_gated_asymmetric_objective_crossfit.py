"""Experiment A4_1: bias-gated asymmetric objective cross-fit.

Scientific question
-------------------
A4 showed that a fixed 2x penalty for positive RUL errors reduced the late
error tail, but it also shifted almost every prediction downward.  The shift
helped target cells whose symmetric model overestimated RUL and harmed cells
that were already conservative, especially FD004.  A4_1 tests one locked,
mechanism-based policy instead of sweeping more loss weights:

* train ``symmetric_mse`` and ``late_weighted_mse_x2`` from the same verified
  A2 source state for the same fixed ten-epoch target budget;
* estimate the signed error of the symmetric model on selection engines only;
* select the asymmetric model iff the mean selection error is strictly
  positive (RUL overestimation); otherwise retain the symmetric model;
* evaluate the locked choice on engine-disjoint confirmation engines with
  independent endpoint-assignment seeds.

The zero threshold is pre-registered and has an operational interpretation;
it is not tuned.  Official C-MAPSS test trajectories and RUL files are never
opened.  A3 official outputs are not read or used for model selection.

Run from the repository root::

    python -u scripts/experimentA4_1_bias_gated_asymmetric_objective_crossfit.py --dry-run

    nohup python -u scripts/experimentA4_1_bias_gated_asymmetric_objective_crossfit.py \
      > experimentA4_1_training.log 2>&1 &

The script automatically assigns domain/model-seed workers to idle GPUs.  All
artifacts are written below
``outputs/experimentA4_1_bias_gated_asymmetric_objective_crossfit``.
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


SCRIPT_VERSION = "experimentA4_1_bias_gated_asymmetric_objective_crossfit_v1"
EXPERIMENT_ID = "experimentA4_1"
DOMAINS = a4.DOMAINS
ARCHITECTURE = a4.ARCHITECTURE
OBJECTIVES = a4.OBJECTIVES
MODEL_SEEDS = a4.MODEL_SEEDS
TARGET_SPLIT_SEEDS = a4.TARGET_SPLIT_SEEDS
ROLE_PARTITIONS = a4.ROLE_PARTITIONS
SELECTION_ENDPOINT_SEEDS = list(range(7601, 7606))
CONFIRMATION_ENDPOINT_SEEDS = list(range(7701, 7706))
ENDPOINT_FRACTIONS = a4.ENDPOINT_FRACTIONS
BIAS_GATE_THRESHOLD = 0.0
HIGH_RUL_THRESHOLD = 60.0
DEFAULT_OUTPUT = "outputs/experimentA4_1_bias_gated_asymmetric_objective_crossfit"
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
GATE_KEYS = [
    "target_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A4_1: bias-gated asymmetric objective cross-fit"
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
        raise FileNotFoundError(f"required A4_1 input is missing: {path}")
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
        "experiment_name": "bias_gated_asymmetric_objective_crossfit",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "objectives": list(OBJECTIVES),
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
        "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "late_error_weight": 2.0,
        "bias_gate_threshold": BIAS_GATE_THRESHOLD,
        "high_rul_threshold": HIGH_RUL_THRESHOLD,
        "fixed_budget_no_epoch_selection": True,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "minimum_nasa_improvement_pct": 3.0,
        "rmse_noninferiority_margin_pct": 3.0,
        "high_rul_rmse_noninferiority_margin_pct": 3.0,
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
                "selection_endpoint_seeds": [7601],
                "confirmation_endpoint_seeds": [7701],
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
        raise ValueError(f"A4_1 requires architecture={ARCHITECTURE}")
    if tuple(experiment["objectives"]) != OBJECTIVES:
        raise ValueError(f"A4_1 requires objectives={OBJECTIVES}")
    if float(experiment["late_error_weight"]) != 2.0:
        raise ValueError("A4_1 late-error weight is locked at 2.0")
    if float(experiment["bias_gate_threshold"]) != 0.0:
        raise ValueError("A4_1 bias-gate threshold is pre-registered at zero")
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
            raise ValueError(f"A4_1 has empty/duplicate values in {name}")
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
        "selection_run": output / f"{prefix}_selection_run_level.csv",
        "confirmation_run": output / f"{prefix}_confirmation_run_level.csv",
        "fixed_run": output / f"{prefix}_fixed_endpoint_run_level.csv",
        "selection_predictions": output / f"{prefix}_selection_endpoint_predictions.csv",
        "confirmation_predictions": output / f"{prefix}_confirmation_endpoint_predictions.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "gate_decisions": output / f"{prefix}_gate_decisions.csv",
        "gated_run_csv": output / f"{prefix}_gated_run_level.csv",
        "gated_run_json": output / f"{prefix}_gated_run_level.json",
        "paired_gated": output / f"{prefix}_paired_gated_vs_symmetric.csv",
        "paired_global": output / f"{prefix}_paired_global_x2_vs_symmetric.csv",
        "high_rul_paired": output / f"{prefix}_high_rul_paired_gated_vs_symmetric.csv",
        "fixed_gated": output / f"{prefix}_fixed_endpoint_paired_gated_vs_symmetric.csv",
        "comparisons": output / f"{prefix}_comparison_summary.csv",
        "gate_summary": output / f"{prefix}_gate_summary.csv",
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
        "selection_run": directory / "selection_run_level.csv",
        "confirmation_run": directory / "confirmation_run_level.csv",
        "fixed_run": directory / "fixed_endpoint_run_level.csv",
        "selection_predictions": directory / "selection_endpoint_predictions.csv",
        "confirmation_predictions": directory / "confirmation_endpoint_predictions.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def training_cell_id(
    domain: str,
    objective: str,
    model_seed: int,
    split_seed: int,
) -> str:
    return (
        f"{EXPERIMENT_ID}_{domain.lower()}_mseed{model_seed:03d}_"
        f"tsplit{split_seed}_{objective}"
    )


def role_endpoint_evaluations(
    *,
    experiment: dict,
    protocol: dict,
    objective: str,
    model_seed: int,
    split_seed: int,
    endpoint_rows: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, pd.DataFrame]:
    domain = protocol["target_domain"]
    split = protocol["role_splits"][str(split_seed)]
    selection_results: list[dict] = []
    confirmation_results: list[dict] = []
    fixed_results: list[dict] = []
    selection_prediction_parts: list[pd.DataFrame] = []
    confirmation_prediction_parts: list[pd.DataFrame] = []

    for partition in experiment["role_partitions"]:
        roles = split["partitions"][str(partition)]
        selection_units = list(map(int, roles["selection_units"]))
        confirmation_units = list(map(int, roles["confirmation_units"]))
        if set(selection_units) & set(confirmation_units):
            raise AssertionError("selection and confirmation engines overlap")

        for endpoint_seed in experiment["selection_endpoint_seeds"]:
            assignment = a21.balanced_assignment(
                selection_units,
                domain,
                split_seed,
                partition,
                endpoint_seed,
                "selection",
            )
            selected = a21.endpoint_subset(
                endpoint_rows,
                selection_units,
                assignment=assignment,
            )
            common = {
                "target_domain": domain,
                "objective": objective,
                "model_seed": int(model_seed),
                "target_split_seed": int(split_seed),
                "role_partition": int(partition),
                "endpoint_seed": int(endpoint_seed),
                "evaluation_role": "selection",
                "evaluation_protocol": "balanced_endpoint",
            }
            selection_results.append(
                {
                    **common,
                    **a4.endpoint_risk_metrics(selected),
                    "evaluation_units": selection_units,
                    "evaluation_engine_count": len(selection_units),
                    "evaluation_used_for_gate": True,
                    "evaluation_used_for_training": False,
                    "fixed_budget_epoch": int(experiment["target_epochs"]),
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )
            selection_prediction_parts.append(a4.annotate_predictions(selected, common))

        for endpoint_seed in experiment["confirmation_endpoint_seeds"]:
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
                "evaluation_role": "confirmation",
                "evaluation_protocol": "balanced_endpoint",
            }
            confirmation_results.append(
                {
                    **common,
                    **a4.endpoint_risk_metrics(selected),
                    "evaluation_units": confirmation_units,
                    "evaluation_engine_count": len(confirmation_units),
                    "evaluation_used_for_gate": False,
                    "evaluation_used_for_training": False,
                    "fixed_budget_epoch": int(experiment["target_epochs"]),
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )
            confirmation_prediction_parts.append(a4.annotate_predictions(selected, common))

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
                    "evaluation_role": "confirmation",
                    "evaluation_protocol": f"fixed_endpoint_{int(round(100 * fraction)):03d}",
                    **a4.endpoint_risk_metrics(selected),
                    "evaluation_units": confirmation_units,
                    "evaluation_engine_count": len(confirmation_units),
                    "evaluation_used_for_gate": False,
                    "evaluation_used_for_training": False,
                    "fixed_budget_epoch": int(experiment["target_epochs"]),
                    "official_test_files_accessed": False,
                    "official_test_forward_run": False,
                }
            )

    return (
        selection_results,
        confirmation_results,
        fixed_results,
        pd.concat(selection_prediction_parts, ignore_index=True),
        pd.concat(confirmation_prediction_parts, ignore_index=True),
    )


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
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        objective,
        float(experiment["late_error_weight"]),
    )
    endpoint_rows = a21.endpoint_epoch_rows(
        predictions,
        int(experiment["target_epochs"]),
    )
    (
        selection,
        confirmation,
        fixed,
        selection_predictions,
        confirmation_predictions,
    ) = role_endpoint_evaluations(
        experiment=experiment,
        protocol=protocol,
        objective=objective,
        model_seed=model_seed,
        split_seed=split_seed,
        endpoint_rows=endpoint_rows,
    )

    identifier = training_cell_id(domain, objective, model_seed, split_seed)
    common = {
        "experiment_id": EXPERIMENT_ID,
        "cell_id": identifier,
        "model": ARCHITECTURE,
        "target_run_seed": int(run_seed),
        "k": int(experiment["k"]),
        "adaptation_units": support_units,
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "late_error_weight": float(experiment["late_error_weight"]),
        "bias_gate_threshold": float(experiment["bias_gate_threshold"]),
        "source_signature": inventory["source_signature"],
        "source_cache_origin": inventory["source_cache_origin"],
        "source_history_rows": int(len(source_history)),
    }
    for row in selection + confirmation + fixed:
        row.update(common)
    for frame in (selection_predictions, confirmation_predictions):
        for column, value in reversed(list(common.items())):
            scalar = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, tuple, dict))
                else value
            )
            frame.insert(0, column, scalar)
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", identifier)
    history.insert(2, "target_domain", domain)
    history.insert(3, "objective", objective)
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        selection,
        confirmation,
        fixed,
        selection_predictions,
        confirmation_predictions,
        history,
    )


def empty_worker_state() -> dict[str, Any]:
    return {
        "completed": set(),
        "selection_run": pd.DataFrame(),
        "confirmation_run": pd.DataFrame(),
        "fixed_run": pd.DataFrame(),
        "selection_predictions": pd.DataFrame(),
        "confirmation_predictions": pd.DataFrame(),
        "history": pd.DataFrame(),
        "inventory": pd.DataFrame(),
    }


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    state = empty_worker_state()
    if paths["status"].is_file():
        state["completed"] = set(
            read_json(paths["status"]).get("completed_cell_ids", [])
        )
    for name in (
        "selection_run",
        "confirmation_run",
        "fixed_run",
        "selection_predictions",
        "confirmation_predictions",
        "history",
        "inventory",
    ):
        frame = load_csv(paths[name])
        if name != "inventory" and not frame.empty:
            frame = frame[frame["cell_id"].isin(state["completed"])]
        state[name] = frame
    return state


def save_worker_state(
    paths: dict[str, Path],
    state: dict[str, Any],
    expected_cells: int,
    experiment: dict,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in (
        "selection_run",
        "confirmation_run",
        "fixed_run",
        "selection_predictions",
        "confirmation_predictions",
        "history",
        "inventory",
    ):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    expected_selection = (
        expected_cells
        * len(experiment["role_partitions"])
        * len(experiment["selection_endpoint_seeds"])
    )
    expected_confirmation = (
        expected_cells
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
    )
    expected_fixed = (
        expected_cells
        * len(experiment["role_partitions"])
        * len(experiment["endpoint_fractions"])
    )
    complete = bool(
        len(state["completed"]) == expected_cells
        and len(state["selection_run"]) == expected_selection
        and len(state["confirmation_run"]) == expected_confirmation
        and len(state["fixed_run"]) == expected_fixed
    )
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected_cells,
            "completed_selection_records": int(len(state["selection_run"])),
            "expected_selection_records": expected_selection,
            "completed_confirmation_records": int(len(state["confirmation_run"])),
            "expected_confirmation_records": expected_confirmation,
            "completed_fixed_records": int(len(state["fixed_run"])),
            "expected_fixed_records": expected_fixed,
            "complete": complete,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain = str(args.worker_domain)
    model_seed = int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A4_1 worker")
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
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "bias_gate_threshold": experiment["bias_gate_threshold"],
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
            "selection_endpoint_seeds",
            "confirmation_endpoint_seeds",
            "bias_gate_threshold",
        ):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(
                    f"existing A4_1 shard is incompatible at {key}; "
                    "use a new output directory"
                )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], worker_manifest)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(
        paths["directory"] / "source_prior_adjacency.csv",
        pd.DataFrame(
            prior.numpy().astype(int),
            index=sensors,
            columns=sensors,
        ).to_csv(),
    )
    a1.atomic_write_text(
        paths["directory"] / "source_prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )

    state = load_worker_state(paths)
    expected_cells = len(OBJECTIVES) * len(experiment["target_split_seeds"])
    pending = [
        (objective, split_seed)
        for objective in OBJECTIVES
        for split_seed in experiment["target_split_seeds"]
        if training_cell_id(domain, objective, model_seed, split_seed)
        not in state["completed"]
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
        for objective, split_seed in pending:
            (
                selection,
                confirmation,
                fixed,
                selection_predictions,
                confirmation_predictions,
                history,
            ) = run_training_cell(
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
            state["selection_run"] = pd.concat(
                [state["selection_run"], pd.DataFrame(selection)],
                ignore_index=True,
            )
            state["confirmation_run"] = pd.concat(
                [state["confirmation_run"], pd.DataFrame(confirmation)],
                ignore_index=True,
            )
            state["fixed_run"] = pd.concat(
                [state["fixed_run"], pd.DataFrame(fixed)],
                ignore_index=True,
            )
            state["selection_predictions"] = pd.concat(
                [state["selection_predictions"], selection_predictions],
                ignore_index=True,
            )
            state["confirmation_predictions"] = pd.concat(
                [state["confirmation_predictions"], confirmation_predictions],
                ignore_index=True,
            )
            state["history"] = pd.concat(
                [state["history"], history],
                ignore_index=True,
            )
            state["completed"].add(
                training_cell_id(domain, objective, model_seed, int(split_seed))
            )
            save_worker_state(paths, state, expected_cells, experiment)
    save_worker_state(paths, state, expected_cells, experiment)
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
                "no idle GPU met A4_1 thresholds; inventory="
                + json.dumps(inventory, ensure_ascii=False)
            )
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
                f"[A4_1] launched domain={domain} seed={seed} "
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
                    f"A4_1 worker failed domain={record['domain']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A4_1] completed domain={record['domain']} "
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
    names = (
        "selection_run",
        "confirmation_run",
        "fixed_run",
        "selection_predictions",
        "confirmation_predictions",
        "history",
        "inventory",
    )
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    expected_cells = len(OBJECTIVES) * len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected_cells:
            raise RuntimeError(f"incomplete A4_1 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get(
            "official_test_forward_run"
        ):
            raise RuntimeError(f"official-test contamination: {paths['status']}")
        for name in names:
            merged[name].append(load_csv(paths[name]))
    return {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in merged.items()
    }


def objective_pivot(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pivot = frame.pivot(index=keys, columns="objective", values=METRICS).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else column
        for column in pivot.columns
    ]
    return pivot


def make_gate_decisions(selection: pd.DataFrame, experiment: dict) -> pd.DataFrame:
    symmetric = selection[selection["objective"] == "symmetric_mse"].copy()
    decisions = (
        symmetric.groupby(GATE_KEYS, as_index=False)
        .agg(
            selection_bias=("mean_error", "mean"),
            selection_bias_std=("mean_error", "std"),
            selection_nasa_score_mean=("nasa_score", "mean"),
            selection_rmse_mean=("rmse", "mean"),
            selection_endpoint_seed_count=("endpoint_seed", "nunique"),
            selection_engine_count=("evaluation_engine_count", "first"),
        )
        .sort_values(GATE_KEYS)
    )
    decisions["bias_gate_threshold"] = float(experiment["bias_gate_threshold"])
    decisions["selected_objective"] = np.where(
        decisions["selection_bias"] > float(experiment["bias_gate_threshold"]),
        "late_weighted_mse_x2",
        "symmetric_mse",
    )
    decisions["selected_asymmetric"] = (
        decisions["selected_objective"] == "late_weighted_mse_x2"
    )
    decisions["selection_only_gate"] = True
    decisions["confirmation_used_for_gate"] = False
    expected = (
        len(experiment["domains"])
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
        * len(experiment["role_partitions"])
    )
    if len(decisions) != expected:
        raise RuntimeError("A4_1 gate-decision table is incomplete")
    if not (
        decisions["selection_endpoint_seed_count"]
        == len(experiment["selection_endpoint_seeds"])
    ).all():
        raise RuntimeError("A4_1 gate did not use every selection endpoint seed")
    return decisions


def gated_confirmation(
    confirmation: pd.DataFrame,
    gates: pd.DataFrame,
) -> pd.DataFrame:
    pivot = objective_pivot(confirmation, PAIR_KEYS)
    output = pivot[PAIR_KEYS].merge(
        gates[GATE_KEYS + ["selection_bias", "selected_objective", "selected_asymmetric"]],
        on=GATE_KEYS,
        how="left",
        validate="many_to_one",
    )
    if output["selected_objective"].isna().any():
        raise RuntimeError("A4_1 confirmation row lacks a selection-only gate")
    for metric in METRICS:
        symmetric = pivot[f"{metric}_symmetric_mse"].to_numpy(dtype=float)
        asymmetric = pivot[f"{metric}_late_weighted_mse_x2"].to_numpy(dtype=float)
        selected = np.where(output["selected_asymmetric"], asymmetric, symmetric)
        output[f"{metric}_symmetric_mse"] = symmetric
        output[f"{metric}_late_weighted_mse_x2"] = asymmetric
        output[f"{metric}_gated_policy"] = selected
        output[f"{metric}_delta_gated_minus_symmetric"] = selected - symmetric
    output["nasa_relative_delta"] = (
        output["nasa_score_delta_gated_minus_symmetric"]
        / output["nasa_score_symmetric_mse"]
    )
    output["rmse_relative_delta"] = (
        output["rmse_delta_gated_minus_symmetric"]
        / output["rmse_symmetric_mse"]
    )
    output["gated_nasa_win"] = output["nasa_score_delta_gated_minus_symmetric"] < 0
    output["gated_rmse_win"] = output["rmse_delta_gated_minus_symmetric"] < 0
    return output.sort_values(PAIR_KEYS)


def gated_fixed_endpoints(
    fixed: pd.DataFrame,
    gates: pd.DataFrame,
) -> pd.DataFrame:
    keys = GATE_KEYS + ["endpoint_fraction"]
    pivot = objective_pivot(fixed, keys)
    output = pivot[keys].merge(
        gates[GATE_KEYS + ["selection_bias", "selected_objective", "selected_asymmetric"]],
        on=GATE_KEYS,
        how="left",
        validate="many_to_one",
    )
    for metric in METRICS:
        symmetric = pivot[f"{metric}_symmetric_mse"].to_numpy(dtype=float)
        asymmetric = pivot[f"{metric}_late_weighted_mse_x2"].to_numpy(dtype=float)
        selected = np.where(output["selected_asymmetric"], asymmetric, symmetric)
        output[f"{metric}_symmetric_mse"] = symmetric
        output[f"{metric}_late_weighted_mse_x2"] = asymmetric
        output[f"{metric}_gated_policy"] = selected
        output[f"{metric}_delta_gated_minus_symmetric"] = selected - symmetric
    return output.sort_values(keys)


def comparison_summary(
    paired: pd.DataFrame,
    experiment: dict,
    comparison: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [("ALL", paired)] + list(paired.groupby("target_domain"))
    for scope, frame in grouped:
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
                    frame["nasa_score_delta_gated_minus_symmetric"].mean()
                ),
                "nasa_improvement_pct": float(-100.0 * frame["nasa_relative_delta"].mean()),
                "nasa_relative_boot_ci95_low": nasa_ci[0],
                "nasa_relative_boot_ci95_high": nasa_ci[1],
                "nasa_win_rate": float(frame["gated_nasa_win"].mean()),
                "rmse_delta_mean": float(frame["rmse_delta_gated_minus_symmetric"].mean()),
                "rmse_degradation_pct": float(100.0 * frame["rmse_relative_delta"].mean()),
                "rmse_relative_boot_ci95_low": rmse_ci[0],
                "rmse_relative_boot_ci95_high": rmse_ci[1],
                "rmse_win_rate": float(frame["gated_rmse_win"].mean()),
                "late_error_q95_delta_mean": float(
                    frame["late_error_q95_delta_gated_minus_symmetric"].mean()
                ),
                "under_error_q95_delta_mean": float(
                    frame["under_error_q95_delta_gated_minus_symmetric"].mean()
                ),
                "mean_error_delta_mean": float(
                    frame["mean_error_delta_gated_minus_symmetric"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def global_x2_pairs(confirmation: pd.DataFrame) -> pd.DataFrame:
    paired = a4.paired_objectives(confirmation)
    rename = {
        column: column.replace("_delta_asymmetric_minus_symmetric", "_delta_gated_minus_symmetric")
        for column in paired.columns
        if "_delta_asymmetric_minus_symmetric" in column
    }
    paired = paired.rename(columns=rename)
    paired["gated_nasa_win"] = paired.pop("asymmetric_nasa_win")
    paired["gated_rmse_win"] = paired.pop("asymmetric_rmse_win")
    paired["selected_objective"] = "late_weighted_mse_x2"
    paired["selected_asymmetric"] = True
    return paired


def high_rul_pairs(
    confirmation_predictions: pd.DataFrame,
    gates: pd.DataFrame,
    experiment: dict,
) -> pd.DataFrame:
    threshold = float(experiment["high_rul_threshold"])
    frame = confirmation_predictions[
        confirmation_predictions["label"] > threshold
    ].copy()
    row_keys = PAIR_KEYS + [
        "unit",
        "unit_window_index",
        "endpoint_fraction",
        "label",
    ]
    pivot = frame.pivot(
        index=row_keys,
        columns="objective",
        values=["error", "nasa_contribution"],
    ).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else column
        for column in pivot.columns
    ]
    pivot = pivot.merge(
        gates[GATE_KEYS + ["selected_asymmetric"]],
        on=GATE_KEYS,
        how="left",
        validate="many_to_one",
    )
    pivot["selected_error"] = np.where(
        pivot["selected_asymmetric"],
        pivot["error_late_weighted_mse_x2"],
        pivot["error_symmetric_mse"],
    )
    pivot["selected_nasa"] = np.where(
        pivot["selected_asymmetric"],
        pivot["nasa_contribution_late_weighted_mse_x2"],
        pivot["nasa_contribution_symmetric_mse"],
    )
    rows: list[dict[str, Any]] = []
    for keys, group in pivot.groupby(PAIR_KEYS):
        symmetric_rmse = float(np.sqrt(np.mean(group["error_symmetric_mse"] ** 2)))
        gated_rmse = float(np.sqrt(np.mean(group["selected_error"] ** 2)))
        symmetric_nasa = float(group["nasa_contribution_symmetric_mse"].sum())
        gated_nasa = float(group["selected_nasa"].sum())
        row = dict(zip(PAIR_KEYS, keys))
        row.update(
            {
                "high_rul_threshold": threshold,
                "n_high_rul_engines": int(group["unit"].nunique()),
                "rmse_symmetric_mse": symmetric_rmse,
                "rmse_gated_policy": gated_rmse,
                "rmse_delta_gated_minus_symmetric": gated_rmse - symmetric_rmse,
                "rmse_relative_delta": (gated_rmse - symmetric_rmse) / symmetric_rmse,
                "nasa_score_symmetric_mse": symmetric_nasa,
                "nasa_score_gated_policy": gated_nasa,
                "nasa_score_delta_gated_minus_symmetric": gated_nasa - symmetric_nasa,
                "nasa_relative_delta": (gated_nasa - symmetric_nasa) / symmetric_nasa,
                "gated_nasa_win": gated_nasa < symmetric_nasa,
                "gated_rmse_win": gated_rmse < symmetric_rmse,
            }
        )
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
        raise RuntimeError(
            "one or more A4_1 confirmation cells lack high-RUL observations"
        )
    return output


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
        "threshold": float(experiment["high_rul_threshold"]),
        "n_records": int(len(paired)),
        "nasa_improvement_pct": float(-100.0 * paired["nasa_relative_delta"].mean()),
        "nasa_relative_ci95": [nasa_ci[0], nasa_ci[1]],
        "rmse_degradation_pct": float(100.0 * paired["rmse_relative_delta"].mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
    }


def gate_summary(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in [("ALL", gates)] + list(gates.groupby("target_domain")):
        rows.append(
            {
                "scope": scope,
                "n_gate_decisions": int(len(frame)),
                "selection_bias_mean": float(frame["selection_bias"].mean()),
                "selection_bias_std": float(frame["selection_bias"].std(ddof=1)),
                "asymmetric_selection_rate": float(frame["selected_asymmetric"].mean()),
                "symmetric_selection_rate": float((~frame["selected_asymmetric"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def make_decision(
    *,
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
    fixed: pd.DataFrame,
    gates: pd.DataFrame,
    gated: pd.DataFrame,
    comparisons: pd.DataFrame,
    high_rul: dict[str, Any],
    experiment: dict,
) -> dict[str, Any]:
    expected_training = (
        len(experiment["domains"])
        * len(OBJECTIVES)
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_selection = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["selection_endpoint_seeds"])
    )
    expected_confirmation = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
    )
    expected_fixed = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["endpoint_fractions"])
    )
    expected_gated = expected_confirmation // len(OBJECTIVES)
    primary = comparisons[
        (comparisons["comparison"] == "bias_gated_vs_symmetric")
        & (comparisons["scope"] == "ALL")
    ].iloc[0]
    domain_rows = comparisons[
        (comparisons["comparison"] == "bias_gated_vs_symmetric")
        & (comparisons["scope"] != "ALL")
    ]
    domain_wins = int((domain_rows["nasa_improvement_pct"] > 0).sum())
    complete = bool(
        selection["cell_id"].nunique() == expected_training
        and confirmation["cell_id"].nunique() == expected_training
        and len(selection) == expected_selection
        and len(confirmation) == expected_confirmation
        and len(fixed) == expected_fixed
        and len(gated) == expected_gated
    )
    uncontaminated = not pd.concat([selection, confirmation, fixed])[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any()
    high_rul_safe = bool(
        100.0 * float(high_rul["rmse_relative_ci95"][1])
        <= float(experiment["high_rul_rmse_noninferiority_margin_pct"])
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
        and high_rul_safe
    )
    decision: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Does a selection-only zero-bias gate retain the 2x asymmetric "
            "objective only when the symmetric model overestimates RUL, thereby "
            "improving confirmation-engine NASA risk without violating RMSE "
            "noninferiority?"
        ),
        "expected_training_cells": expected_training,
        "completed_training_cells": int(selection["cell_id"].nunique()),
        "expected_selection_records": expected_selection,
        "completed_selection_records": int(len(selection)),
        "expected_confirmation_records": expected_confirmation,
        "completed_confirmation_records": int(len(confirmation)),
        "expected_gated_records": expected_gated,
        "completed_gated_records": int(len(gated)),
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "late_error_weight": float(experiment["late_error_weight"]),
        "bias_gate_threshold": float(experiment["bias_gate_threshold"]),
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "gate_result": {
            "n_gate_decisions": int(len(gates)),
            "asymmetric_selection_rate": float(gates["selected_asymmetric"].mean()),
            "selection_bias_mean": float(gates["selection_bias"].mean()),
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
            "rmse_noninferiority_passed": high_rul_safe,
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
                    "A4_1 confirmed selection-only bias-gated asymmetric risk learning"
                    if success
                    else "A4_1 completed, but the bias-gated policy did not meet every registered criterion"
                ),
                "next_action": (
                    "run_fresh_seed_robustness_without_official_test_access"
                    if success
                    else "stop_A4_asymmetric_objective_direction_and_move_to_experimentA5"
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
            "Does a selection-only zero-bias gate retain the 2x asymmetric "
            "objective only when the symmetric model overestimates RUL, thereby "
            "improving confirmation-engine NASA risk without violating RMSE "
            "noninferiority?"
        ),
        "gate_uses_selection_labels_only": True,
        "confirmation_used_for_gate": False,
        "A3_official_outputs_used_for_model_selection": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"existing A4_1 output is incompatible at {key}; "
                    "use a new output directory"
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
        * len(OBJECTIVES)
        * len(experiment["model_seeds"])
        * len(experiment["target_split_seeds"])
    )
    expected_selection = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["selection_endpoint_seeds"])
    )
    expected_confirmation = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["confirmation_endpoint_seeds"])
    )
    expected_fixed = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["endpoint_fractions"])
    )
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "domains": experiment["domains"],
        "objectives": experiment["objectives"],
        "model_seeds": experiment["model_seeds"],
        "target_split_seeds": experiment["target_split_seeds"],
        "role_partitions": experiment["role_partitions"],
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "endpoint_seed_sets_disjoint": True,
        "bias_gate_rule": "select x2 iff mean symmetric selection error > 0",
        "expected_training_cells": expected_training,
        "expected_selection_records": expected_selection,
        "expected_confirmation_records": expected_confirmation,
        "expected_fixed_endpoint_records": expected_fixed,
        "expected_gate_decisions": expected_training // len(OBJECTIVES) * len(experiment["role_partitions"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "late_error_weight": float(experiment["late_error_weight"]),
        "bias_gate_threshold": float(experiment["bias_gate_threshold"]),
        "high_rul_threshold": float(experiment["high_rul_threshold"]),
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
            "A4_1 contains an interrupted run; use --resume or a new output directory"
        )
    tasks = [
        (domain, seed)
        for domain in experiment["domains"]
        for seed in experiment["model_seeds"]
    ]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    selection = merged["selection_run"].sort_values(PAIR_KEYS + ["objective"])
    confirmation = merged["confirmation_run"].sort_values(PAIR_KEYS + ["objective"])
    fixed = merged["fixed_run"].sort_values(
        GATE_KEYS + ["endpoint_fraction", "objective"]
    )
    if selection["cell_id"].nunique() != expected_training or len(selection) != expected_selection:
        raise RuntimeError("A4_1 merged selection output is incomplete")
    if confirmation["cell_id"].nunique() != expected_training or len(confirmation) != expected_confirmation:
        raise RuntimeError("A4_1 merged confirmation output is incomplete")
    if len(fixed) != expected_fixed:
        raise RuntimeError("A4_1 merged fixed-endpoint output is incomplete")
    if pd.concat([selection, confirmation, fixed])[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any():
        raise RuntimeError("A4_1 detected official-test contamination")

    gates = make_gate_decisions(selection, experiment)
    gated = gated_confirmation(confirmation, gates)
    fixed_gated = gated_fixed_endpoints(fixed, gates)
    global_pairs = global_x2_pairs(confirmation)
    gated_comparisons = comparison_summary(
        gated,
        experiment,
        "bias_gated_vs_symmetric",
    )
    global_comparisons = comparison_summary(
        global_pairs,
        experiment,
        "global_x2_vs_symmetric",
    )
    comparisons = pd.concat(
        [gated_comparisons, global_comparisons],
        ignore_index=True,
    )
    high_pairs = high_rul_pairs(
        merged["confirmation_predictions"],
        gates,
        experiment,
    )
    high_summary = high_rul_summary(high_pairs, experiment)
    decision = make_decision(
        selection=selection,
        confirmation=confirmation,
        fixed=fixed,
        gates=gates,
        gated=gated,
        comparisons=comparisons,
        high_rul=high_summary,
        experiment=experiment,
    )

    a1.atomic_write_text(paths["selection_run"], selection.to_csv(index=False))
    a1.atomic_write_text(paths["confirmation_run"], confirmation.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_run"], fixed.to_csv(index=False))
    a1.atomic_write_text(
        paths["selection_predictions"],
        merged["selection_predictions"].to_csv(index=False),
    )
    a1.atomic_write_text(
        paths["confirmation_predictions"],
        merged["confirmation_predictions"].to_csv(index=False),
    )
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    inventory = merged["inventory"].drop_duplicates(
        ["target_domain", "model_seed"]
    )
    a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False))
    a1.atomic_write_text(paths["gate_decisions"], gates.to_csv(index=False))
    a1.atomic_write_text(paths["gated_run_csv"], gated.to_csv(index=False))
    atomic_json(paths["gated_run_json"], gated.to_dict("records"))
    a1.atomic_write_text(paths["paired_gated"], gated.to_csv(index=False))
    a1.atomic_write_text(paths["paired_global"], global_pairs.to_csv(index=False))
    a1.atomic_write_text(paths["high_rul_paired"], high_pairs.to_csv(index=False))
    a1.atomic_write_text(paths["fixed_gated"], fixed_gated.to_csv(index=False))
    a1.atomic_write_text(paths["comparisons"], comparisons.to_csv(index=False))
    a1.atomic_write_text(paths["gate_summary"], gate_summary(gates).to_csv(index=False))
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
