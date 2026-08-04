"""Experiment A6: causal trajectory-monotonicity confirmation.

A4/A5 improved average endpoint risk by shifting predictions downward, but
also worsened the true-high-RUL subset.  A6 changes direction.  It trains only
the locked symmetric-MSE ``window_no_graph`` baseline and applies a causal
local isotonic projection to each engine's prediction history.  The primary
candidate uses the current and two preceding predictions only; it never reads
future windows, the engine's final lifetime, or labels.  A strict cumulative
minimum is retained only as a secondary mechanism diagnostic.

The script first reconstructs window-end cycles with the same ordering used by
``make_windows`` and refuses to continue unless unit, label, cycle and final
window alignment audits pass.  Official C-MAPSS test files are never opened.

Run from the repository root::

    python -u scripts/experimentA6_trajectory_monotonicity_confirmation.py --dry-run

    nohup python -u scripts/experimentA6_trajectory_monotonicity_confirmation.py \
      > experimentA6_training.log 2>&1 &

All artifacts are written below
``outputs/experimentA6_trajectory_monotonicity_confirmation``.  The parent
process automatically distributes domain/model-seed workers over idle GPUs.
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

from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402


SCRIPT_VERSION = "experimentA6_trajectory_monotonicity_confirmation_v1"
EXPERIMENT_ID = "experimentA6"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
SELECTION_ENDPOINT_SEEDS = list(range(8001, 8006))
CONFIRMATION_ENDPOINT_SEEDS = list(range(8101, 8106))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
VARIANTS = (
    "baseline_symmetric",
    "causal_rolling_pava_w3",
    "causal_cumulative_min",
)
PRIMARY_VARIANT = "causal_rolling_pava_w3"
SECONDARY_VARIANT = "causal_cumulative_min"
ROLLING_MEMORY = 3
HIGH_TRUE_RUL_THRESHOLD = 60.0
DEFAULT_OUTPUT = "outputs/experimentA6_trajectory_monotonicity_confirmation"
DEFAULT_A2_OUTPUT = a4.DEFAULT_A2_OUTPUT
DEFAULT_A2_1_OUTPUT = a4.DEFAULT_A2_1_OUTPUT
METRICS = a4.METRICS
BASE_KEYS = ["target_domain", "model_seed", "target_split_seed"]
ROLE_KEYS = BASE_KEYS + ["role_partition"]
PAIR_KEYS = ROLE_KEYS + ["endpoint_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A6: causal trajectory-monotonicity confirmation"
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
        raise FileNotFoundError(f"required A6 input is missing: {path}")
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
        "experiment_name": "trajectory_monotonicity_confirmation",
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
        "secondary_variant": SECONDARY_VARIANT,
        "rolling_memory": ROLLING_MEMORY,
        "high_true_rul_threshold": HIGH_TRUE_RUL_THRESHOLD,
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
        "minimum_nasa_domain_wins": 3,
        "minimum_upward_jump_magnitude_reduction_pct": 20.0,
        "high_rul_nasa_ci_upper_max": 0.0,
        "high_rul_rmse_ci_upper_max": 0.0,
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
                "selection_endpoint_seeds": [8001],
                "confirmation_endpoint_seeds": [8101],
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
        raise ValueError(f"A6 requires architecture={ARCHITECTURE}")
    if experiment["objective"] != "symmetric_mse":
        raise ValueError("A6 trains only the symmetric-MSE target model")
    if int(experiment["rolling_memory"]) != 3:
        raise ValueError("A6 rolling PAVA memory is locked at three predictions")
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
            raise ValueError(f"A6 has empty/duplicate values in {name}")
    for domain in experiment["domains"]:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def root_paths(output: Path) -> dict[str, Path]:
    p = EXPERIMENT_ID
    return {
        "manifest": output / f"{p}_manifest.json",
        "protocol": output / f"{p}_protocol.json",
        "engine_roles": output / f"{p}_engine_roles.csv",
        "dry_run": output / f"{p}_dry_run.json",
        "trajectory_predictions": output / f"{p}_trajectory_predictions.csv",
        "time_audit": output / f"{p}_time_alignment_audit.csv",
        "history": output / f"{p}_target_history.csv",
        "inventory": output / f"{p}_source_inventory.csv",
        "selection_predictions": output / f"{p}_selection_endpoint_predictions.csv",
        "confirmation_predictions": output / f"{p}_confirmation_endpoint_predictions.csv",
        "selection_run": output / f"{p}_selection_run_level.csv",
        "confirmation_run": output / f"{p}_confirmation_run_level.csv",
        "fixed_run": output / f"{p}_fixed_endpoint_run_level.csv",
        "paired_primary": output / f"{p}_paired_rolling_pava_vs_baseline.csv",
        "paired_secondary": output / f"{p}_paired_cummin_vs_baseline.csv",
        "fixed_primary": output / f"{p}_fixed_endpoint_paired_rolling_pava_vs_baseline.csv",
        "high_primary": output / f"{p}_high_rul_paired_rolling_pava_vs_baseline.csv",
        "high_secondary": output / f"{p}_high_rul_paired_cummin_vs_baseline.csv",
        "comparison": output / f"{p}_comparison_summary.csv",
        "decision": output / f"{p}_confirmation_decision.json",
    }


def shard_dir(output: Path, domain: str, seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{seed:03d}"


def shard_paths(output: Path, domain: str, seed: int) -> dict[str, Path]:
    directory = shard_dir(output, domain, seed)
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "predictions": directory / "trajectory_predictions.csv",
        "time_audit": directory / "time_alignment_audit.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
    }


def training_cell_id(domain: str, model_seed: int, split_seed: int) -> str:
    return f"{EXPERIMENT_ID}_{domain.lower()}_mseed{model_seed:03d}_tsplit{split_seed}"


def make_window_time_metadata(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recreate make_windows ordering and attach the real window-end cycle."""
    rows: list[dict[str, Any]] = []
    window_size = int(cfg["window_size"])
    stride = int(cfg["window_stride"])
    for unit, group in frame.groupby("unit", sort=True):
        ordered = group.sort_values("cycle").reset_index(drop=True)
        original_count = len(ordered)
        if original_count < 1:
            raise ValueError("A6 received an empty engine trajectory")
        pad = max(0, window_size - original_count)
        padded_count = original_count + pad
        ends = (
            [padded_count]
            if original_count < window_size
            else list(range(window_size, padded_count + 1, stride))
        )
        if not ends or ends[-1] != padded_count:
            ends.append(padded_count)
        for index, end in enumerate(ends):
            original_index = int(np.clip(end - pad - 1, 0, original_count - 1))
            source = ordered.iloc[original_index]
            rows.append(
                {
                    "unit": int(unit),
                    "unit_window_index": int(index),
                    "window_end_cycle": int(source["cycle"]),
                    "metadata_label": float(source["rul"]),
                    "engine_final_cycle": int(ordered["cycle"].max()),
                    "is_final_window": bool(end == padded_count),
                }
            )
    return pd.DataFrame(rows)


def prepare_support_pool_with_metadata(
    cfg: dict,
    preprocessing: str,
    balance_mode: str,
    support_units: list[int],
    pool_units: list[int],
) -> tuple[Any, Any, int, pd.DataFrame]:
    sensors = list(cfg["sensor_columns"])
    _, normalizer = a1.fit_source_normalizer_train_only(cfg, preprocessing)
    target = a1.add_train_rul(
        a1.load_train_domain(cfg["data_dir"], cfg["target_domain"]),
        cfg["rul_cap"],
    )
    features = (
        sensors + a1.SETTING_FEATURE_COLUMNS
        if preprocessing in {"global_settings", "condition_settings"}
        else sensors
    )
    normalized = normalizer.transform(target, sensors)
    support_frame = normalized[normalized["unit"].isin(support_units)].copy()
    pool_frame = normalized[normalized["unit"].isin(pool_units)].copy()
    if support_frame["unit"].nunique() != len(support_units):
        raise ValueError("A6 support engine preparation is incomplete")
    if pool_frame["unit"].nunique() != len(pool_units):
        raise ValueError("A6 pool engine preparation is incomplete")
    support = a1.make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        loader_seed=cfg["seed"] + 9000,
    )
    pool = a1.make_loader(
        pool_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9200,
    )
    metadata = make_window_time_metadata(pool_frame, cfg)
    return support, pool, len(features), metadata


def nonincreasing_pava(values: np.ndarray) -> np.ndarray:
    """Least-squares projection onto y[0] >= y[1] >= ... >= y[n-1]."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("PAVA requires a finite non-empty one-dimensional array")
    means: list[float] = []
    counts: list[int] = []
    for value in array:
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] < means[-1]:
            total = means[-2] * counts[-2] + means[-1] * counts[-1]
            count = counts[-2] + counts[-1]
            means[-2:] = [total / count]
            counts[-2:] = [count]
    fitted = np.concatenate(
        [np.full(count, mean, dtype=float) for mean, count in zip(means, counts)]
    )
    if len(fitted) != len(array) or np.any(np.diff(fitted) > 1e-10):
        raise AssertionError("PAVA projection failed the non-increasing audit")
    return fitted


def causal_rolling_pava(values: np.ndarray, memory: int) -> np.ndarray:
    """Return the current endpoint of a PAVA fit over the last ``memory`` values."""
    array = np.asarray(values, dtype=float)
    if memory < 2:
        raise ValueError("rolling PAVA memory must be at least two")
    output = np.empty(len(array), dtype=float)
    for index in range(len(array)):
        start = max(0, index - memory + 1)
        output[index] = nonincreasing_pava(array[start : index + 1])[-1]
    if np.any(output - array > 1e-9):
        raise AssertionError("causal rolling PAVA unexpectedly raised a prediction")
    return output


def causal_cumulative_min(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.minimum.accumulate(array)


def upward_jump_stats(values: np.ndarray) -> tuple[int, float]:
    delta = np.diff(np.asarray(values, dtype=float))
    positive = np.maximum(delta, 0.0)
    return int(np.sum(positive > 0)), float(np.sum(positive))


def attach_time_and_variants(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    experiment: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(predictions) != len(metadata):
        raise AssertionError("prediction/metadata row counts do not match")
    units_match = np.array_equal(
        predictions["unit"].to_numpy(dtype=int),
        metadata["unit"].to_numpy(dtype=int),
    )
    label_difference = np.abs(
        predictions["label"].to_numpy(dtype=float)
        - metadata["metadata_label"].to_numpy(dtype=float)
    )
    max_label_difference = float(label_difference.max(initial=0.0))
    if not units_match or max_label_difference > 1e-5:
        raise AssertionError(
            "A6 time metadata is not aligned with model predictions: "
            f"units_match={units_match}, max_label_difference={max_label_difference}"
        )
    output = predictions.copy().reset_index(drop=True)
    for column in (
        "unit_window_index",
        "window_end_cycle",
        "engine_final_cycle",
        "is_final_window",
    ):
        output[column] = metadata[column].to_numpy()
    output["prediction_baseline_symmetric"] = output["prediction"].astype(float)
    output["prediction_causal_rolling_pava_w3"] = np.nan
    output["prediction_causal_cumulative_min"] = np.nan
    audit_rows: list[dict[str, Any]] = []
    for unit, index in output.groupby("unit", sort=True).groups.items():
        positions = np.asarray(list(index), dtype=int)
        group = output.loc[positions].sort_values("unit_window_index")
        positions = group.index.to_numpy(dtype=int)
        raw = group["prediction_baseline_symmetric"].to_numpy(dtype=float)
        rolling = causal_rolling_pava(raw, int(experiment["rolling_memory"]))
        cumulative = causal_cumulative_min(raw)
        output.loc[positions, "prediction_causal_rolling_pava_w3"] = rolling
        output.loc[positions, "prediction_causal_cumulative_min"] = cumulative
        raw_count, raw_magnitude = upward_jump_stats(raw)
        rolling_count, rolling_magnitude = upward_jump_stats(rolling)
        cumulative_count, cumulative_magnitude = upward_jump_stats(cumulative)
        audit_rows.append(
            {
                "unit": int(unit),
                "window_count": int(len(group)),
                "first_cycle": int(group["window_end_cycle"].iloc[0]),
                "last_cycle": int(group["window_end_cycle"].iloc[-1]),
                "engine_final_cycle": int(group["engine_final_cycle"].iloc[-1]),
                "cycle_strictly_increasing": bool(
                    np.all(np.diff(group["window_end_cycle"].to_numpy()) > 0)
                ),
                "final_window_aligned": bool(
                    group["window_end_cycle"].iloc[-1]
                    == group["engine_final_cycle"].iloc[-1]
                ),
                "raw_upward_jump_count": raw_count,
                "raw_upward_jump_magnitude": raw_magnitude,
                "rolling_upward_jump_count": rolling_count,
                "rolling_upward_jump_magnitude": rolling_magnitude,
                "cumulative_upward_jump_count": cumulative_count,
                "cumulative_upward_jump_magnitude": cumulative_magnitude,
                "rolling_adjusted_rate": float(np.mean(np.abs(rolling - raw) > 1e-9)),
                "rolling_mean_adjustment": float(np.mean(rolling - raw)),
                "rolling_max_abs_adjustment": float(np.max(np.abs(rolling - raw))),
                "cumulative_adjusted_rate": float(
                    np.mean(np.abs(cumulative - raw) > 1e-9)
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["cycle_strictly_increasing"].all():
        raise AssertionError("A6 reconstructed non-increasing/duplicate cycle metadata")
    if not audit["final_window_aligned"].all():
        raise AssertionError("A6 final window cycle does not match engine final cycle")
    if (audit["cumulative_upward_jump_count"] != 0).any():
        raise AssertionError("A6 cumulative-min diagnostic is not monotone")
    if output[list(VARIANT_COLUMN_MAP.values())].isna().any().any():
        raise AssertionError("A6 failed to populate trajectory variants")
    summary = {
        "prediction_rows": int(len(output)),
        "engine_count": int(output["unit"].nunique()),
        "unit_order_aligned": units_match,
        "max_label_alignment_error": max_label_difference,
        "cycle_alignment_passed": True,
        "causal_prefix_audit_passed": causal_prefix_audit(output, experiment),
    }
    return output, {"summary": summary, "per_engine": audit}


VARIANT_COLUMN_MAP = {
    "baseline_symmetric": "prediction_baseline_symmetric",
    "causal_rolling_pava_w3": "prediction_causal_rolling_pava_w3",
    "causal_cumulative_min": "prediction_causal_cumulative_min",
}


def causal_prefix_audit(frame: pd.DataFrame, experiment: dict) -> bool:
    """Recompute several prefixes and prove that stored outputs use no future rows."""
    memory = int(experiment["rolling_memory"])
    for _, group in frame.groupby("unit", sort=True):
        ordered = group.sort_values("unit_window_index")
        raw = ordered["prediction_baseline_symmetric"].to_numpy(dtype=float)
        stored = ordered["prediction_causal_rolling_pava_w3"].to_numpy(dtype=float)
        checkpoints = sorted({0, len(raw) // 2, len(raw) - 1})
        for end in checkpoints:
            recomputed = causal_rolling_pava(raw[: end + 1], memory)[-1]
            if not np.isclose(recomputed, stored[end], atol=1e-10, rtol=0):
                return False
    return True


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
    support, pool, feature_count, metadata = prepare_support_pool_with_metadata(
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
    raw_predictions, history = a4.train_fixed_budget(
        model,
        support,
        pool,
        cfg,
        device,
        "symmetric_mse",
        1.0,
    )
    predictions, audit_payload = attach_time_and_variants(
        raw_predictions,
        metadata,
        experiment,
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
        predictions.insert(0, column, value)
    audit = audit_payload["per_engine"]
    for column, value in reversed(list(common.items())):
        audit.insert(0, column, value)
    for key, value in audit_payload["summary"].items():
        audit[key] = value
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", identifier)
    history.insert(2, "target_domain", domain)
    history.insert(3, "objective", "symmetric_mse")
    history.insert(4, "model_seed", int(model_seed))
    history.insert(5, "target_split_seed", int(split_seed))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, audit, history


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    state = {
        "completed": completed,
        "predictions": load_csv(paths["predictions"]),
        "time_audit": load_csv(paths["time_audit"]),
        "history": load_csv(paths["history"]),
        "inventory": load_csv(paths["inventory"]),
    }
    for name in ("predictions", "time_audit", "history"):
        if not state[name].empty:
            state[name] = state[name][state[name]["cell_id"].isin(completed)]
    return state


def save_worker_state(
    paths: dict[str, Path],
    state: dict[str, Any],
    expected_cells: int,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "time_audit", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected_cells,
            "trajectory_prediction_rows": int(len(state["predictions"])),
            "time_audit_rows": int(len(state["time_audit"])),
            "complete": len(state["completed"]) == expected_cells,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
    )


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain = str(args.worker_domain)
    model_seed = int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A6 worker")
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
                    f"existing A6 shard is incompatible at {key}; use a new output directory"
                )
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
            predictions, audit, history = run_training_cell(
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
            state["predictions"] = pd.concat(
                [state["predictions"], predictions], ignore_index=True
            )
            state["time_audit"] = pd.concat(
                [state["time_audit"], audit], ignore_index=True
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
                "no idle GPU met A6 thresholds; inventory="
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
                f"[A6] launched domain={domain} seed={seed} "
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
                    f"A6 worker failed domain={record['domain']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A6] completed domain={record['domain']} "
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
    merged: dict[str, list[pd.DataFrame]] = {
        "predictions": [],
        "time_audit": [],
        "history": [],
        "inventory": [],
    }
    expected_cells = len(experiment["target_split_seeds"])
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected_cells:
            raise RuntimeError(f"incomplete A6 worker: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get(
            "official_test_forward_run"
        ):
            raise RuntimeError(f"official-test contamination: {paths['status']}")
        for name in merged:
            merged[name].append(load_csv(paths[name]))
    return {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in merged.items()
    }


def nasa_contribution(error: np.ndarray) -> np.ndarray:
    array = np.asarray(error, dtype=float)
    return np.where(
        array < 0,
        np.exp(-array / 13.0) - 1.0,
        np.exp(array / 10.0) - 1.0,
    )


def metrics_for_variant(frame: pd.DataFrame, variant: str) -> dict[str, float]:
    column = VARIANT_COLUMN_MAP[variant]
    working = frame[["label", column]].copy()
    working["prediction"] = working[column].astype(float)
    working["error"] = working["prediction"] - working["label"].astype(float)
    working["nasa_contribution"] = nasa_contribution(working["error"].to_numpy())
    return a4.endpoint_risk_metrics(working)


def evaluate_variants(
    frame: pd.DataFrame,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = frame[VARIANT_COLUMN_MAP["baseline_symmetric"]].to_numpy(dtype=float)
    for variant in VARIANTS:
        prediction = frame[VARIANT_COLUMN_MAP[variant]].to_numpy(dtype=float)
        rows.append(
            {
                **common,
                "variant": variant,
                **metrics_for_variant(frame, variant),
                "evaluation_engine_count": int(frame["unit"].nunique()),
                "adjusted_rate": float(np.mean(np.abs(prediction - baseline) > 1e-9)),
                "mean_adjustment": float(np.mean(prediction - baseline)),
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
        )
    return rows


def endpoint_pool(frame: pd.DataFrame, experiment: dict) -> pd.DataFrame:
    selected = a2.stratified_endpoint_subset(
        frame,
        list(map(float, experiment["endpoint_fractions"])),
    )
    expected_minimum = frame["unit"].nunique()
    if selected["unit"].nunique() != expected_minimum:
        raise AssertionError("A6 endpoint construction lost one or more engines")
    return selected


def evaluate_roles(
    trajectories: pd.DataFrame,
    protocols: dict[str, dict],
    experiment: dict,
) -> dict[str, pd.DataFrame]:
    selection_predictions: list[pd.DataFrame] = []
    confirmation_predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    for base_values, trajectory in trajectories.groupby(BASE_KEYS):
        domain, model_seed, split_seed = base_values
        protocol = protocols[str(domain)]
        split = protocol["role_splits"][str(int(split_seed))]
        endpoints = endpoint_pool(trajectory, experiment)
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]
            selection_units = list(map(int, roles["selection_units"]))
            confirmation_units = list(map(int, roles["confirmation_units"]))
            if set(selection_units) & set(confirmation_units):
                raise AssertionError("selection and confirmation engines overlap")
            common_role = {
                "target_domain": str(domain),
                "model_seed": int(model_seed),
                "target_split_seed": int(split_seed),
                "role_partition": int(partition),
            }
            for role, units, seeds, prediction_parts, result_rows in (
                (
                    "selection",
                    selection_units,
                    experiment["selection_endpoint_seeds"],
                    selection_predictions,
                    selection_rows,
                ),
                (
                    "confirmation",
                    confirmation_units,
                    experiment["confirmation_endpoint_seeds"],
                    confirmation_predictions,
                    confirmation_rows,
                ),
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
                    chosen = a21.endpoint_subset(
                        endpoints,
                        units,
                        assignment=assignment,
                    ).copy()
                    chosen["role_partition"] = int(partition)
                    chosen["endpoint_seed"] = int(endpoint_seed)
                    chosen["evaluation_role"] = role
                    prediction_parts.append(chosen)
                    common = {
                        **common_role,
                        "endpoint_seed": int(endpoint_seed),
                        "evaluation_role": role,
                        "evaluation_protocol": "balanced_endpoint",
                    }
                    result_rows.extend(evaluate_variants(chosen, common))
                    if role == "confirmation":
                        high = chosen[
                            chosen["label"]
                            > float(experiment["high_true_rul_threshold"])
                        ]
                        if high.empty:
                            raise RuntimeError("A6 confirmation cell lacks high-RUL engines")
                        high_rows.extend(
                            evaluate_variants(
                                high,
                                {
                                    **common,
                                    "high_true_rul_threshold": float(
                                        experiment["high_true_rul_threshold"]
                                    ),
                                },
                            )
                        )
            for fraction in experiment["endpoint_fractions"]:
                chosen = a21.endpoint_subset(
                    endpoints,
                    confirmation_units,
                    fraction=float(fraction),
                )
                fixed_rows.extend(
                    evaluate_variants(
                        chosen,
                        {
                            **common_role,
                            "endpoint_fraction": float(fraction),
                            "evaluation_protocol": (
                                f"fixed_endpoint_{int(round(100 * float(fraction))):03d}"
                            ),
                        },
                    )
                )
    return {
        "selection_predictions": pd.concat(selection_predictions, ignore_index=True),
        "confirmation_predictions": pd.concat(
            confirmation_predictions, ignore_index=True
        ),
        "selection_run": pd.DataFrame(selection_rows),
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
                "nasa_improvement_pct": float(
                    -100.0 * frame["nasa_relative_delta"].mean()
                ),
                "nasa_relative_boot_ci95_low": nasa_ci[0],
                "nasa_relative_boot_ci95_high": nasa_ci[1],
                "nasa_win_rate": float(frame["candidate_nasa_win"].mean()),
                "rmse_delta_mean": float(
                    frame["rmse_delta_candidate_minus_baseline"].mean()
                ),
                "rmse_degradation_pct": float(
                    100.0 * frame["rmse_relative_delta"].mean()
                ),
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
        "nasa_win_rate": float(paired["candidate_nasa_win"].mean()),
        "rmse_degradation_pct": float(100.0 * paired["rmse_relative_delta"].mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
        "rmse_win_rate": float(paired["candidate_rmse_win"].mean()),
    }


def trajectory_summary(time_audit: pd.DataFrame) -> dict[str, Any]:
    raw = float(time_audit["raw_upward_jump_magnitude"].sum())
    rolling = float(time_audit["rolling_upward_jump_magnitude"].sum())
    reduction = 100.0 * (raw - rolling) / raw if raw > 0 else 0.0
    return {
        "audit_rows": int(len(time_audit)),
        "audited_training_cells": int(time_audit["cell_id"].nunique()),
        "audited_engines": int(len(time_audit)),
        "unit_order_aligned": bool(time_audit["unit_order_aligned"].astype(bool).all()),
        "cycle_alignment_passed": bool(
            time_audit["cycle_alignment_passed"].astype(bool).all()
        ),
        "causal_prefix_audit_passed": bool(
            time_audit["causal_prefix_audit_passed"].astype(bool).all()
        ),
        "maximum_label_alignment_error": float(
            time_audit["max_label_alignment_error"].max()
        ),
        "raw_upward_jump_count": int(time_audit["raw_upward_jump_count"].sum()),
        "rolling_upward_jump_count": int(
            time_audit["rolling_upward_jump_count"].sum()
        ),
        "raw_upward_jump_magnitude": raw,
        "rolling_upward_jump_magnitude": rolling,
        "upward_jump_magnitude_reduction_pct": reduction,
        "rolling_adjusted_rate": float(
            np.average(
                time_audit["rolling_adjusted_rate"],
                weights=time_audit["window_count"],
            )
        ),
    }


def make_decision(
    *,
    trajectories: pd.DataFrame,
    confirmation_run: pd.DataFrame,
    paired_primary: pd.DataFrame,
    comparisons: pd.DataFrame,
    high_rul: dict[str, Any],
    trajectory: dict[str, Any],
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
    primary = comparisons[
        (comparisons["comparison"] == "rolling_pava_vs_baseline")
        & (comparisons["scope"] == "ALL")
    ].iloc[0]
    domain_rows = comparisons[
        (comparisons["comparison"] == "rolling_pava_vs_baseline")
        & (comparisons["scope"] != "ALL")
    ]
    domain_wins = int((domain_rows["nasa_improvement_pct"] > 0).sum())
    complete = bool(
        trajectories["cell_id"].nunique() == expected_training
        and len(confirmation_run) == expected_confirmation
        and len(paired_primary) == expected_pairs
        and trajectory["audited_training_cells"] == expected_training
    )
    uncontaminated = not trajectories[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any()
    high_nasa_safe = bool(
        float(high_rul["nasa_relative_ci95"][1])
        <= float(experiment["high_rul_nasa_ci_upper_max"])
    )
    high_rmse_safe = bool(
        float(high_rul["rmse_relative_ci95"][1])
        <= float(experiment["high_rul_rmse_ci_upper_max"])
    )
    trajectory_passed = bool(
        trajectory["unit_order_aligned"]
        and trajectory["cycle_alignment_passed"]
        and trajectory["causal_prefix_audit_passed"]
        and trajectory["maximum_label_alignment_error"] <= 1e-5
        and trajectory["upward_jump_magnitude_reduction_pct"]
        >= float(experiment["minimum_upward_jump_magnitude_reduction_pct"])
    )
    success = bool(
        complete
        and uncontaminated
        and trajectory_passed
        and float(primary["nasa_improvement_pct"])
        >= float(experiment["minimum_nasa_improvement_pct"])
        and float(primary["nasa_relative_boot_ci95_high"]) < 0
        and 100.0 * float(primary["rmse_relative_boot_ci95_high"])
        <= float(experiment["rmse_noninferiority_margin_pct"])
        and domain_wins >= int(experiment["minimum_nasa_domain_wins"])
        and float(primary["late_error_q95_delta_mean"]) < 0
        and high_nasa_safe
        and high_rmse_safe
    )
    decision: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Does a causal three-prediction rolling PAVA reduce endpoint NASA "
            "risk while preserving strict true-high-RUL NASA/RMSE safety?"
        ),
        "expected_training_cells": expected_training,
        "completed_training_cells": int(trajectories["cell_id"].nunique()),
        "expected_confirmation_records": expected_confirmation,
        "completed_confirmation_records": int(len(confirmation_run)),
        "expected_primary_pairs": expected_pairs,
        "completed_primary_pairs": int(len(paired_primary)),
        "complete": complete,
        "quick_mode": bool(experiment["quick_mode"]),
        "fixed_budget_epoch": int(experiment["target_epochs"]),
        "rolling_memory": int(experiment["rolling_memory"]),
        "selection_endpoint_seeds": experiment["selection_endpoint_seeds"],
        "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"],
        "selection_confirmation_endpoint_seeds_disjoint": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "time_and_causality_audit": {
            **trajectory,
            "passed": trajectory_passed,
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
            "late_error_q95_delta_mean": float(primary["late_error_q95_delta_mean"]),
            "under_error_q95_delta_mean": float(
                primary["under_error_q95_delta_mean"]
            ),
            "mean_error_delta_mean": float(primary["mean_error_delta_mean"]),
        },
        "high_rul_safety_result": {
            **high_rul,
            "nasa_no_worsening_passed": high_nasa_safe,
            "rmse_no_worsening_passed": high_rmse_safe,
        },
    }
    if experiment["quick_mode"]:
        decision.update(
            {
                "passed": complete and trajectory_passed,
                "reason": "quick smoke run only; do not interpret scientifically",
            }
        )
    else:
        decision.update(
            {
                "passed": success,
                "reason": (
                    "A6 confirmed causal trajectory consistency with strict high-RUL safety"
                    if success
                    else "A6 completed, but causal rolling PAVA did not meet every registered criterion"
                ),
                "next_action": (
                    "run_fresh_seed_confirmation_without_official_test_access"
                    if success
                    else "stop_monotonic_postprocessing_and_reassess_experimentA7_direction"
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
            "Does a causal three-prediction rolling PAVA reduce endpoint NASA "
            "risk while preserving strict true-high-RUL NASA/RMSE safety?"
        ),
        "candidate_uses_current_and_past_predictions_only": True,
        "future_windows_used": False,
        "engine_final_lifetime_used_by_candidate": False,
        "labels_used_by_candidate": False,
        "A5_results_used_for_hypothesis_generation": True,
        "A5_confirmation_outputs_used_for_runtime_fitting": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "experiment_config", "evidence"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"existing A6 output is incompatible at {key}; use a new output directory"
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
    expected_selection = (
        expected_training
        * len(experiment["role_partitions"])
        * len(experiment["selection_endpoint_seeds"])
        * len(VARIANTS)
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
        "primary_variant": PRIMARY_VARIANT,
        "rolling_memory": int(experiment["rolling_memory"]),
        "candidate_information_boundary": (
            "current and previous two predictions from the same engine only"
        ),
        "expected_training_cells": expected_training,
        "expected_selection_records": expected_selection,
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
            "A6 contains an interrupted run; use --resume or a new output directory"
        )
    tasks = [
        (domain, seed)
        for domain in experiment["domains"]
        for seed in experiment["model_seeds"]
    ]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    trajectories = merged["predictions"].sort_values(
        BASE_KEYS + ["unit", "unit_window_index"]
    )
    time_audit = merged["time_audit"].sort_values(BASE_KEYS + ["unit"])
    if trajectories["cell_id"].nunique() != expected_training:
        raise RuntimeError("A6 merged trajectories are incomplete")
    if trajectories[
        ["official_test_files_accessed", "official_test_forward_run"]
    ].astype(bool).any().any():
        raise RuntimeError("A6 detected official-test contamination")
    trajectory_result = trajectory_summary(time_audit)
    if not (
        trajectory_result["unit_order_aligned"]
        and trajectory_result["cycle_alignment_passed"]
        and trajectory_result["causal_prefix_audit_passed"]
        and trajectory_result["maximum_label_alignment_error"] <= 1e-5
    ):
        raise RuntimeError("A6 time/causality audit failed; performance evaluation aborted")
    evaluated = evaluate_roles(trajectories, selected_protocols, experiment)
    selection_run = evaluated["selection_run"].sort_values(
        PAIR_KEYS + ["variant"]
    )
    confirmation_run = evaluated["confirmation_run"].sort_values(
        PAIR_KEYS + ["variant"]
    )
    fixed_run = evaluated["fixed_run"].sort_values(
        ROLE_KEYS + ["endpoint_fraction", "variant"]
    )
    if len(selection_run) != expected_selection:
        raise RuntimeError("A6 selection output is incomplete")
    if len(confirmation_run) != expected_confirmation:
        raise RuntimeError("A6 confirmation output is incomplete")
    if len(fixed_run) != expected_fixed:
        raise RuntimeError("A6 fixed-endpoint output is incomplete")
    paired_primary = paired_variants(confirmation_run, PRIMARY_VARIANT, PAIR_KEYS)
    paired_secondary = paired_variants(
        confirmation_run,
        SECONDARY_VARIANT,
        PAIR_KEYS,
    )
    fixed_primary = paired_variants(
        fixed_run,
        PRIMARY_VARIANT,
        ROLE_KEYS + ["endpoint_fraction"],
    )
    high_primary = paired_variants(
        evaluated["high_run"],
        PRIMARY_VARIANT,
        PAIR_KEYS,
    )
    high_secondary = paired_variants(
        evaluated["high_run"],
        SECONDARY_VARIANT,
        PAIR_KEYS,
    )
    comparisons = pd.concat(
        [
            comparison_summary(
                paired_primary,
                experiment,
                "rolling_pava_vs_baseline",
            ),
            comparison_summary(
                paired_secondary,
                experiment,
                "cumulative_min_vs_baseline",
            ),
        ],
        ignore_index=True,
    )
    high_summary = high_rul_summary(high_primary, experiment)
    decision = make_decision(
        trajectories=trajectories,
        confirmation_run=confirmation_run,
        paired_primary=paired_primary,
        comparisons=comparisons,
        high_rul=high_summary,
        trajectory=trajectory_result,
        experiment=experiment,
    )
    a1.atomic_write_text(paths["trajectory_predictions"], trajectories.to_csv(index=False))
    a1.atomic_write_text(paths["time_audit"], time_audit.to_csv(index=False))
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    inventory = merged["inventory"].drop_duplicates(["target_domain", "model_seed"])
    a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False))
    for path_name, frame in (
        ("selection_predictions", evaluated["selection_predictions"]),
        ("confirmation_predictions", evaluated["confirmation_predictions"]),
        ("selection_run", selection_run),
        ("confirmation_run", confirmation_run),
        ("fixed_run", fixed_run),
        ("paired_primary", paired_primary),
        ("paired_secondary", paired_secondary),
        ("fixed_primary", fixed_primary),
        ("high_primary", high_primary),
        ("high_secondary", high_secondary),
        ("comparison", comparisons),
    ):
        a1.atomic_write_text(paths[path_name], frame.to_csv(index=False))
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
