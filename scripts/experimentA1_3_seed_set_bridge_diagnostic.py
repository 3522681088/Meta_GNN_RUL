"""Experiment A1_3: terminal seed-set bridge diagnostic.

A1_1 and A1_2 changed model seeds, target-support splits, and confirmation
engines at the same time.  Their opposite conclusions therefore cannot tell
whether the reversal came from model initialization or from the confirmation
engine distribution.  A1_3 changes only the model-seed set:

* FD004, K=5, sensor_graph_prior versus window_graph;
* the A1_2 support splits (5401--5410), epoch-selection engines, and 30 new
  independent confirmation engines are held fixed;
* the A1_1 seed set (52--61) is trained on that fixed A1_2 protocol;
* the completed A1_2 seed set (70--79) is imported only after protocol and
  completeness checks;
* individual and ten-seed ensemble comparisons are reported by seed set;
* no official C-MAPSS test trajectories or official RUL labels are accessed.

The parent process automatically queries idle GPUs and assigns one model seed
to each available GPU.  Every worker writes an isolated resumable shard, then
the parent merges all shards and the registered A1_2 results into one output
directory.

Run from the repository root:

    python -u scripts/experimentA1_3_seed_set_bridge_diagnostic.py

All outputs are written below
``outputs/experimentA1_3_seed_set_bridge_diagnostic``.
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

from evaluation.metrics import regression_metrics  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_1_prior_window_stability_confirmation as a11  # noqa: E402
from scripts import experimentA1_2_seed_ensemble_stability as a12  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402


SCRIPT_VERSION = "experimentA1_3_seed_set_bridge_diagnostic_v1"
ARCHITECTURES = ("sensor_graph_prior", "window_graph")
OLD_GROUP = "a1_1_old_seeds_52_61"
NEW_GROUP = "a1_2_new_seeds_70_79"
OLD_SEEDS = list(range(52, 62))
NEW_SEEDS = list(range(70, 80))
TARGET_SPLITS = list(range(5401, 5411))
DEFAULT_OUTPUT = "outputs/experimentA1_3_seed_set_bridge_diagnostic"
DEFAULT_A1_2_OUTPUT = "outputs/experimentA1_2_seed_ensemble_stability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A1_3: terminal seed-set bridge diagnostic"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a1-2-dir")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (default), cpu, or a torch device for single-process use",
    )
    parser.add_argument(
        "--gpus",
        help="optional physical GPU indices, for example 0,2,3",
    )
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="disable the automatic multi-GPU scheduler",
    )
    parser.add_argument(
        "--retrain-new-seeds",
        action="store_true",
        help="retrain seeds 70--79 instead of importing registered A1_2",
    )
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one old seed, one new seed, one split, short smoke training",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)

    # Internal worker arguments.  Users should run the parent command only.
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-group", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def experiment_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    experiment = deepcopy(a12.DEFAULT_EXPERIMENT)
    base["target_domain"] = "FD004"
    base["source_domains"] = ["FD001", "FD002", "FD003"]
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

    experiment.update(
        {
            "experiment_id": "experimentA1_3",
            "experiment_name": "seed_set_bridge_diagnostic",
            "target_split_seeds": TARGET_SPLITS.copy(),
            # Keep the A1_2 seeds here so build_protocol reproduces its exact
            # engine roles and protocol hash.  Workers may train other seeds.
            "model_seeds": NEW_SEEDS.copy(),
            "architectures": list(ARCHITECTURES),
            "ensemble_sizes": [10],
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "output_dir": base["output_dir"],
        }
    )
    if args.quick:
        experiment["target_split_seeds"] = [5401]
        experiment["model_seeds"] = [70, 71]
        experiment["ensemble_sizes"] = [2]
        experiment["source_pretrain_steps"] = 5
        experiment["target_epochs"] = 1
        experiment["bootstrap_repetitions"] = 100
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 1
        if args.output_dir is None:
            base["output_dir"] = resolved(
                None, "outputs/experimentA1_3_seed_set_bridge_diagnostic_quick"
            )
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def result_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA1_3"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_roles": output / f"{prefix}_engine_roles.csv",
        "dry_run": output / f"{prefix}_dry_run.json",
        "raw": output / f"{prefix}_bridge_run_level.json",
        "run_csv": output / f"{prefix}_bridge_run_level.csv",
        "window_predictions": output / f"{prefix}_bridge_window_predictions.csv",
        "per_engine": output / f"{prefix}_bridge_per_engine_metrics.csv",
        "summary": output / f"{prefix}_bridge_summary.csv",
        "paired_cell": output / f"{prefix}_bridge_paired_cells.csv",
        "paired_split": output / f"{prefix}_bridge_paired_target_splits.csv",
        "comparisons": output / f"{prefix}_bridge_paired_comparisons.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "seed_summary": output / f"{prefix}_per_seed_delta_summary.csv",
        "interaction": output / f"{prefix}_seed_group_interaction.json",
        "ensemble_raw": output / f"{prefix}_group_ensemble_run_level.json",
        "ensemble_run": output / f"{prefix}_group_ensemble_run_level.csv",
        "ensemble_predictions": output
        / f"{prefix}_group_ensemble_window_predictions.csv",
        "ensemble_per_engine": output
        / f"{prefix}_group_ensemble_per_engine_metrics.csv",
        "ensemble_summary": output / f"{prefix}_group_ensemble_summary.csv",
        "ensemble_paired": output / f"{prefix}_group_ensemble_paired_cells.csv",
        "ensemble_comparisons": output
        / f"{prefix}_group_ensemble_paired_comparisons.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )


def shard_paths(output: Path, group: str, seed: int) -> dict[str, Path]:
    return result_paths(output / "shards" / f"{group}_mseed{seed:03d}")


def normalize_worker_outputs(
    result: dict,
    predictions: pd.DataFrame,
    engines: pd.DataFrame,
    group: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    old_id = str(result["replicate_id"])
    new_id = old_id.replace("experimentA1_2_", "experimentA1_3_")
    result["source_experiment_id"] = "experimentA1_3_worker"
    result["experiment_id"] = "experimentA1_3"
    result["experiment_name"] = "seed_set_bridge_diagnostic"
    result["seed_group"] = group
    result["replicate_id"] = new_id
    predictions = predictions.copy()
    engines = engines.copy()
    predictions["replicate_id"] = new_id
    engines["replicate_id"] = new_id
    predictions.insert(0, "seed_group", group)
    engines.insert(0, "seed_group", group)
    return result, predictions, engines


def worker_main(
    args: argparse.Namespace,
    base: dict,
    experiment: dict,
) -> None:
    seed = int(args.worker_seed)
    group = str(args.worker_group)
    if group not in {OLD_GROUP, NEW_GROUP}:
        raise ValueError(f"unknown worker group: {group}")
    allowed = OLD_SEEDS if group == OLD_GROUP else NEW_SEEDS
    if not args.quick and seed not in allowed:
        raise ValueError(f"seed {seed} is not registered for {group}")

    paths = shard_paths(Path(base["output_dir"]), group, seed)
    worker_output = paths["manifest"].parent
    worker_output.mkdir(parents=True, exist_ok=True)
    worker_base = deepcopy(base)
    worker_base["output_dir"] = str(worker_output)
    # CUDA_VISIBLE_DEVICES maps the assigned physical device to cuda:0.
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"

    protocol = a12.build_protocol(worker_base, experiment)
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(
        worker_base,
        experiment["preprocessing"],
        int(experiment["sensor_graph_k"]),
    )
    script_hash = a1.file_sha256(Path(__file__))
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": script_hash,
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "seed_group": group,
        "model_seed": seed,
        "target_split_seeds": experiment["target_split_seeds"],
        "protocol_hash": protocol["protocol_hash"],
        "graph_fit": graph_fit,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["protocol"], protocol)
    a1.atomic_write_text(
        paths["engine_roles"], a12.protocol_frame(protocol).to_csv(index=False)
    )
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(
        worker_output / "experimentA1_3_prior_adjacency.csv",
        pd.DataFrame(
            prior.numpy().astype(int), index=sensors, columns=sensors
        ).to_csv(),
    )
    a1.atomic_write_text(
        worker_output / "experimentA1_3_prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )

    results, prediction_parts, engine_parts, inventory_rows = (
        a11.load_resume_state(paths)
    )
    completed = a11.completed_keys(results)
    cfg = deepcopy(worker_base)
    cfg["seed"] = seed
    for architecture in ARCHITECTURES:
        pending = [
            split
            for split in experiment["target_split_seeds"]
            if (int(split), seed, architecture) not in completed
        ]
        if not pending:
            continue
        state, history, inventory = a12.load_or_train_source(
            base=cfg,
            experiment=experiment,
            protocol=protocol,
            architecture=architecture,
            model_seed=seed,
            prior=prior,
            git_commit=manifest["git_commit"],
            script_hash=script_hash,
        )
        inventory["seed_group"] = group
        inventory_rows = [
            row
            for row in inventory_rows
            if not (
                str(row.get("model")) == architecture
                and int(row.get("model_seed", -1)) == seed
            )
        ]
        inventory_rows.append(inventory)
        for split in pending:
            result, predictions, engines = a12.run_target_cell(
                base=cfg,
                experiment=experiment,
                protocol=protocol,
                architecture=architecture,
                model_seed=seed,
                target_split_seed=int(split),
                source_state=deepcopy(state),
                source_history=history,
                inventory=inventory,
                prior=prior,
                save_checkpoint=args.save_target_checkpoints,
            )
            result, predictions, engines = normalize_worker_outputs(
                result, predictions, engines, group
            )
            results.append(result)
            prediction_parts.append(predictions)
            engine_parts.append(engines)
            completed.add((int(split), seed, architecture))
            a1.save_progress(
                paths=paths,
                results=results,
                predictions=prediction_parts,
                engine_metrics=engine_parts,
                inventory=inventory_rows,
                bootstrap_repetitions=min(
                    200, int(experiment["bootstrap_repetitions"])
                ),
            )
    a1.save_progress(
        paths=paths,
        results=results,
        predictions=prediction_parts,
        engine_metrics=engine_parts,
        inventory=inventory_rows,
        bootstrap_repetitions=int(experiment["bootstrap_repetitions"]),
    )
    expected = len(ARCHITECTURES) * len(experiment["target_split_seeds"])
    status = {
        "experiment_id": "experimentA1_3",
        "seed_group": group,
        "model_seed": seed,
        "expected_cells": expected,
        "completed_cells": len(results),
        "complete": len(results) == expected,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(worker_output / "worker_complete.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


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
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, free, total, utilization = map(int, fields)
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
    values = [part.strip() for part in value.split(",")]
    if all(part.isdigit() for part in values):
        return {int(part) for part in values}
    return None


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    inventory = query_gpus()
    if args.gpus:
        requested = [int(value.strip()) for value in args.gpus.split(",")]
        known = {row["index"] for row in inventory}
        missing = [value for value in requested if value not in known]
        if missing:
            raise RuntimeError(f"requested GPUs not visible to nvidia-smi: {missing}")
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
    chosen = [row["index"] for row in candidates]
    if args.max_workers > 0:
        chosen = chosen[: int(args.max_workers)]
    return chosen, inventory


def worker_command(
    args: argparse.Namespace,
    group: str,
    seed: int,
    device: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-seed",
        str(seed),
        "--worker-group",
        group,
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
    if not tasks:
        return
    use_cpu = args.device == "cpu"
    if args.single_process or use_cpu or args.device not in {"auto", "cpu"}:
        devices = [args.device]
        gpu_inventory: list[dict] = []
    else:
        devices, gpu_inventory = choose_gpus(args)
        if not devices:
            details = json.dumps(gpu_inventory, ensure_ascii=False)
            raise RuntimeError(
                "no idle GPU met the registered thresholds; "
                f"inventory={details}. Use --gpus explicitly or wait."
            )
    print(
        json.dumps(
            {
                "scheduler": "experimentA1_3",
                "tasks": [{"group": g, "seed": s} for g, s in tasks],
                "devices": devices,
                "gpu_inventory": gpu_inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    pending = list(tasks)
    active: dict[str | int, dict[str, Any]] = {}
    while pending or active:
        free_devices = [device for device in devices if device not in active]
        for device in free_devices:
            if not pending:
                break
            group, seed = pending.pop(0)
            shard = shard_paths(output, group, seed)["manifest"].parent
            shard.mkdir(parents=True, exist_ok=True)
            log_path = shard / "worker_training.log"
            log_handle = log_path.open("a", encoding="utf-8")
            if isinstance(device, int):
                command = worker_command(args, group, seed, "auto", output)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
            else:
                command = worker_command(
                    args, group, seed, str(device), output
                )
                environment = os.environ.copy()
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
                "group": group,
                "seed": seed,
                "log": log_handle,
                "log_path": log_path,
            }
            print(
                f"[A1_3] launched group={group} seed={seed} "
                f"device={device} pid={process.pid}"
            )

        finished: list[str | int] = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = ""
                try:
                    tail = "\n".join(
                        record["log_path"].read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-40:]
                    )
                except OSError:
                    pass
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"worker failed group={record['group']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A1_3] completed group={record['group']} "
                f"seed={record['seed']} device={device}"
            )
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_registered_a1_2(
    path: Path,
    protocol: dict,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = {
        "manifest": path / "experimentA1_2_manifest.json",
        "protocol": path / "experimentA1_2_protocol.json",
        "decision": path / "experimentA1_2_confirmation_decision.json",
        "raw": path / "experimentA1_2_individual_run_level.json",
        "predictions": path
        / "experimentA1_2_individual_window_predictions.csv",
        "engines": path / "experimentA1_2_individual_per_engine_metrics.csv",
        "inventory": path / "experimentA1_2_source_inventory.csv",
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError(
            "registered A1_2 artifacts are missing; use --retrain-new-seeds "
            f"or restore: {missing}"
        )
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    old_protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    decision = json.loads(required["decision"].read_text(encoding="utf-8"))
    if old_protocol.get("protocol_hash") != protocol.get("protocol_hash"):
        raise RuntimeError("A1_2 protocol hash does not match the A1_3 bridge")
    if manifest.get("official_test_files_accessed") or manifest.get(
        "official_test_forward_run"
    ):
        raise RuntimeError("A1_2 manifest reports official-test access")
    if not decision.get("complete"):
        raise RuntimeError("A1_2 is incomplete and cannot be imported")
    results = json.loads(required["raw"].read_text(encoding="utf-8"))
    predictions = load_csv(required["predictions"])
    engines = load_csv(required["engines"])
    inventory = load_csv(required["inventory"])
    expected = len(NEW_SEEDS) * len(TARGET_SPLITS) * len(ARCHITECTURES)
    if len(results) != expected:
        raise RuntimeError(f"A1_2 has {len(results)} cells; expected {expected}")
    frame = pd.DataFrame(results)
    if set(frame["model_seed"].astype(int)) != set(NEW_SEEDS):
        raise RuntimeError("A1_2 model seeds are not exactly 70--79")
    if set(frame["target_split_seed"].astype(int)) != set(TARGET_SPLITS):
        raise RuntimeError("A1_2 target split seeds are not exactly 5401--5410")
    result_ids = set(frame["replicate_id"].astype(str))
    if result_ids != set(predictions["replicate_id"].astype(str)):
        raise RuntimeError("A1_2 run-level and prediction IDs are inconsistent")
    if result_ids != set(engines["replicate_id"].astype(str)):
        raise RuntimeError("A1_2 run-level and per-engine IDs are inconsistent")
    return results, predictions, engines, inventory, decision


def normalize_imported_group(
    results: list[dict],
    predictions: pd.DataFrame,
    engines: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping: dict[str, str] = {}
    normalized: list[dict] = []
    for original in results:
        row = dict(original)
        old_id = str(row["replicate_id"])
        new_id = old_id.replace("experimentA1_2_", "experimentA1_3_")
        mapping[old_id] = new_id
        row["source_experiment_id"] = "experimentA1_2"
        row["experiment_id"] = "experimentA1_3"
        row["experiment_name"] = "seed_set_bridge_diagnostic"
        row["seed_group"] = NEW_GROUP
        row["replicate_id"] = new_id
        normalized.append(row)
    predictions = predictions.copy()
    engines = engines.copy()
    inventory = inventory.copy()
    predictions["replicate_id"] = predictions["replicate_id"].map(mapping)
    engines["replicate_id"] = engines["replicate_id"].map(mapping)
    predictions.insert(0, "seed_group", NEW_GROUP)
    engines.insert(0, "seed_group", NEW_GROUP)
    inventory.insert(0, "seed_group", NEW_GROUP)
    return normalized, predictions, engines, inventory


def load_shards(
    output: Path,
    tasks: list[tuple[str, int]],
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    results: list[dict] = []
    predictions: list[pd.DataFrame] = []
    engines: list[pd.DataFrame] = []
    inventories: list[pd.DataFrame] = []
    for group, seed in tasks:
        paths = shard_paths(output, group, seed)
        status_path = paths["manifest"].parent / "worker_complete.json"
        if not status_path.is_file():
            raise RuntimeError(f"missing worker completion status: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("complete"):
            raise RuntimeError(f"incomplete worker shard: {status_path}")
        rows = json.loads(paths["raw"].read_text(encoding="utf-8"))
        results.extend(rows)
        predictions.append(load_csv(paths["window_predictions"]))
        engines.append(load_csv(paths["per_engine"]))
        inventories.append(load_csv(paths["inventory"]))
    return results, predictions, engines, inventories


def bridge_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (group_name, model), values in frame.groupby(["seed_group", "model"]):
        row = {
            "seed_group": group_name,
            "model": model,
            "n_cells": int(len(values)),
            "n_model_seeds": int(values["model_seed"].nunique()),
            "n_target_splits": int(values["target_split_seed"].nunique()),
        }
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_cell_std"] = float(values[metric].std(ddof=1))
            split_mean = values.groupby("target_split_seed")[metric].mean()
            row[f"{metric}_target_split_std"] = float(
                split_mean.std(ddof=1)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed_group", "rmse_mean"])


def group_comparisons(
    results: list[dict], repetitions: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired_parts: list[pd.DataFrame] = []
    split_parts: list[pd.DataFrame] = []
    comparison_parts: list[pd.DataFrame] = []
    frame = pd.DataFrame(results)
    for group_name, group_frame in frame.groupby("seed_group"):
        records = group_frame.to_dict("records")
        paired = exp17b.paired_cells(records)
        split = exp17b.paired_by_target_split(paired)
        comparisons = exp17b.comparison_summary(
            records, paired, int(repetitions)
        )
        for part in (paired, split, comparisons):
            part.insert(0, "seed_group", group_name)
        paired_parts.append(paired)
        split_parts.append(split)
        comparison_parts.append(comparisons)
    return (
        pd.concat(paired_parts, ignore_index=True),
        pd.concat(split_parts, ignore_index=True),
        pd.concat(comparison_parts, ignore_index=True),
    )


def interaction_analysis(
    paired: pd.DataFrame,
    repetitions: int,
) -> tuple[pd.DataFrame, dict]:
    column = "rmse_delta_candidate_minus_reference"
    per_seed = (
        paired.groupby(["seed_group", "model_seed"])[column]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "rmse_delta_mean"})
    )
    wins = (
        paired.assign(win=paired[column] < 0)
        .groupby(["seed_group", "model_seed"])["win"]
        .mean()
        .reset_index(name="rmse_cell_win_rate")
    )
    per_seed = per_seed.merge(wins, on=["seed_group", "model_seed"])
    matrices: dict[str, np.ndarray] = {}
    for group_name in (OLD_GROUP, NEW_GROUP):
        selected = paired[paired["seed_group"].eq(group_name)]
        matrix = selected.pivot(
            index="model_seed",
            columns="target_split_seed",
            values=column,
        ).sort_index().sort_index(axis=1)
        matrices[group_name] = matrix.to_numpy(float)
    if any(value.size == 0 for value in matrices.values()):
        raise RuntimeError("both seed groups are required for interaction analysis")
    old = matrices[OLD_GROUP]
    new = matrices[NEW_GROUP]
    rng = np.random.default_rng(6137)
    samples = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        old_seed = rng.integers(0, old.shape[0], size=old.shape[0])
        old_split = rng.integers(0, old.shape[1], size=old.shape[1])
        new_seed = rng.integers(0, new.shape[0], size=new.shape[0])
        new_split = rng.integers(0, new.shape[1], size=new.shape[1])
        samples[index] = (
            old[np.ix_(old_seed, old_split)].mean()
            - new[np.ix_(new_seed, new_split)].mean()
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    report = {
        "contrast": "old_seed_group_delta_minus_new_seed_group_delta",
        "old_group_rmse_delta_mean": float(old.mean()),
        "new_group_rmse_delta_mean": float(new.mean()),
        "interaction_delta": float(old.mean() - new.mean()),
        "interaction_bootstrap_ci95_low": float(low),
        "interaction_bootstrap_ci95_high": float(high),
        "bootstrap_repetitions": int(repetitions),
    }
    return per_seed, report


def build_group_ensembles(
    results: list[dict], predictions: pd.DataFrame, protocol: dict
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_frame = pd.DataFrame(results)
    all_results: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_engines: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    comparisons: list[pd.DataFrame] = []
    for group_name, seeds in ((OLD_GROUP, OLD_SEEDS), (NEW_GROUP, NEW_SEEDS)):
        available = sorted(
            result_frame[result_frame["seed_group"].eq(group_name)][
                "model_seed"
            ].unique()
        )
        if not available:
            continue
        group_protocol = deepcopy(protocol)
        size = len(available)
        group_protocol["ensemble_groups"] = {str(size): [available]}
        selected_results = result_frame[
            result_frame["seed_group"].eq(group_name)
        ].to_dict("records")
        selected_predictions = predictions[
            predictions["seed_group"].eq(group_name)
        ].copy()
        ensemble_results, ensemble_predictions, ensemble_engines = (
            a12.build_ensemble_outputs(
                results=selected_results,
                predictions=selected_predictions,
                protocol=group_protocol,
            )
        )
        for row in ensemble_results:
            row["experiment_id"] = "experimentA1_3"
            row["seed_group"] = group_name
            row["ensemble_id"] = row["ensemble_id"].replace(
                "experimentA1_2_", f"experimentA1_3_{group_name}_"
            )
        id_mapping = {
            old: old.replace(
                "experimentA1_2_", f"experimentA1_3_{group_name}_"
            )
            for old in ensemble_predictions["ensemble_id"].unique()
        }
        ensemble_predictions["ensemble_id"] = ensemble_predictions[
            "ensemble_id"
        ].map(id_mapping)
        ensemble_engines["ensemble_id"] = ensemble_engines["ensemble_id"].map(
            id_mapping
        )
        ensemble_predictions.insert(0, "seed_group", group_name)
        ensemble_engines.insert(0, "seed_group", group_name)
        summary = a12.ensemble_summary(ensemble_results)
        paired = a12.ensemble_paired_cells(ensemble_results)
        comparison = a12.ensemble_comparisons(
            ensemble_results, paired, int(protocol.get("bootstrap_repetitions", 10000))
        )
        for part in (summary, comparison):
            part.insert(0, "seed_group", group_name)
        all_results.extend(ensemble_results)
        all_predictions.append(ensemble_predictions)
        all_engines.append(ensemble_engines)
        summaries.append(summary)
        paired.insert(0, "seed_group", group_name)
        comparisons.append(comparison)
    return (
        all_results,
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_engines, ignore_index=True),
        pd.concat(summaries, ignore_index=True),
        pd.concat(comparisons, ignore_index=True),
    )


def strict_result(row: pd.Series) -> bool:
    return bool(
        row["rmse_improvement_pct"] >= 3.0
        and row["rmse_target_split_win_rate"] >= 0.8
        and row["rmse_hier_boot_ci95_high"] < 0.0
        and row["rmse_split_t_p_holm"] < 0.05
        and row["nasa_score_delta_mean"] <= 0.0
    )


def final_decision(
    experiment: dict,
    comparisons: pd.DataFrame,
    ensemble_comparisons: pd.DataFrame,
    interaction: dict,
    quick: bool,
) -> dict:
    expected_per_group = (
        len(ARCHITECTURES)
        * len(experiment["target_split_seeds"])
        * (2 if quick else 10)
    )
    result: dict[str, Any] = {
        "experiment_id": "experimentA1_3",
        "expected_cells_per_seed_group": expected_per_group,
        "primary_question": (
            "Does the prior sensor graph pass the same strict criteria for "
            "both old and new model-seed sets on the fixed A1_2 protocol?"
        ),
        "quick_mode": bool(quick),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "seed_group_results": {},
        "interaction": interaction,
    }
    for group_name in (OLD_GROUP, NEW_GROUP):
        selected = comparisons[
            comparisons["seed_group"].eq(group_name)
            & comparisons["comparison"].eq("prior_sensor_vs_window_graph")
        ]
        ensemble_selected = ensemble_comparisons[
            ensemble_comparisons["seed_group"].eq(group_name)
        ]
        if len(selected) != 1 or len(ensemble_selected) != 1:
            result["seed_group_results"][group_name] = {"complete": False}
            continue
        row = selected.iloc[0]
        ensemble_row = ensemble_selected.iloc[0]
        result["seed_group_results"][group_name] = {
            "complete": True,
            "individual_rmse_delta_mean": float(row["rmse_delta_mean"]),
            "individual_rmse_improvement_pct": float(
                row["rmse_improvement_pct"]
            ),
            "individual_rmse_target_split_win_rate": float(
                row["rmse_target_split_win_rate"]
            ),
            "individual_ci95": [
                float(row["rmse_hier_boot_ci95_low"]),
                float(row["rmse_hier_boot_ci95_high"]),
            ],
            "individual_nasa_score_delta_mean": float(
                row["nasa_score_delta_mean"]
            ),
            "individual_strict_success": strict_result(row),
            "ensemble_size": int(ensemble_row["ensemble_size"]),
            "ensemble_rmse_delta_mean": float(
                ensemble_row["rmse_delta_mean"]
            ),
            "ensemble_rmse_improvement_pct": float(
                ensemble_row["rmse_improvement_pct"]
            ),
            "ensemble_ci95": [
                float(ensemble_row["rmse_hier_boot_ci95_low"]),
                float(ensemble_row["rmse_hier_boot_ci95_high"]),
            ],
            "ensemble_nasa_score_delta_mean": float(
                ensemble_row["nasa_score_delta_mean"]
            ),
            "ensemble_strict_success": strict_result(ensemble_row),
        }
    complete = all(
        result["seed_group_results"].get(group, {}).get("complete", False)
        for group in (OLD_GROUP, NEW_GROUP)
    )
    result["complete"] = complete
    if quick:
        result["passed"] = complete
        result["recommendation"] = "quick smoke run only; do not interpret"
        return result
    old_success = bool(
        result["seed_group_results"][OLD_GROUP]["individual_strict_success"]
        and result["seed_group_results"][OLD_GROUP][
            "ensemble_strict_success"
        ]
    )
    new_success = bool(
        result["seed_group_results"][NEW_GROUP]["individual_strict_success"]
        and result["seed_group_results"][NEW_GROUP][
            "ensemble_strict_success"
        ]
    )
    result["passed"] = bool(complete and old_success and new_success)
    if old_success and new_success:
        recommendation = "continue_sensor_graph_direction"
        reason = "both seed sets confirmed the prior-sensor advantage"
    elif old_success and not new_success:
        recommendation = "stop_or_redesign_due_to_model_seed_instability"
        reason = "only the old seed set confirmed the prior-sensor advantage"
    elif not old_success and not new_success:
        recommendation = "stop_sensor_graph_superiority_direction"
        reason = "neither seed set confirmed the prior-sensor advantage"
    else:
        recommendation = "stop_or_redesign_due_to_model_seed_instability"
        reason = "only the new seed set confirmed the prior-sensor advantage"
    result["recommendation"] = recommendation
    result["reason"] = reason
    result["next_direction_if_stopped"] = (
        "experimentA2_endpoint_consistency_validation"
    )
    return result


def parent_main(
    args: argparse.Namespace,
    base: dict,
    experiment: dict,
) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = result_paths(output)
    protocol = a12.build_protocol(base, experiment)
    protocol["bootstrap_repetitions"] = int(
        experiment["bootstrap_repetitions"]
    )
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(
        base,
        experiment["preprocessing"],
        int(experiment["sensor_graph_k"]),
    )
    a1_2_dir = Path(resolved(args.a1_2_dir, DEFAULT_A1_2_OUTPUT))
    script_hash = a1.file_sha256(Path(__file__))
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": script_hash,
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {
            key: value for key, value in base.items() if key != "device"
        },
        "experiment_config": experiment,
        "old_seed_group": OLD_SEEDS if not args.quick else [52, 53],
        "new_seed_group": NEW_SEEDS if not args.quick else [70, 71],
        "a1_2_reuse_dir": str(a1_2_dir),
        "retrain_new_seeds": bool(args.retrain_new_seeds or args.quick),
        "protocol_hash": protocol["protocol_hash"],
        "graph_fit": graph_fit,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["protocol"], protocol)
    a1.atomic_write_text(
        paths["engine_roles"], a12.protocol_frame(protocol).to_csv(index=False)
    )
    sensors = list(base["sensor_columns"])
    a1.atomic_write_text(
        output / "experimentA1_3_prior_adjacency.csv",
        pd.DataFrame(
            prior.numpy().astype(int), index=sensors, columns=sensors
        ).to_csv(),
    )
    a1.atomic_write_text(
        output / "experimentA1_3_prior_correlation.csv",
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )
    first_split = int(experiment["target_split_seeds"][0])
    dry_cfg = deepcopy(base)
    dry_cfg["seed"] = 52
    source, support, selection, confirmation, feature_count, split = (
        a11.prepare_confirmation_experiment(
            dry_cfg,
            experiment["preprocessing"],
            experiment["balance_mode"],
            selection_units=protocol["selection_units"],
            confirmation_units=protocol["confirmation_units"],
            adaptation_units=protocol[
                "adaptation_units_by_target_split_seed"
            ][str(first_split)],
        )
    )
    dry_report = {
        "experiment_id": "experimentA1_3",
        "old_seed_group": OLD_SEEDS if not args.quick else [52, 53],
        "new_seed_group": NEW_SEEDS if not args.quick else [70, 71],
        "target_split_seeds": experiment["target_split_seeds"],
        "feature_count": int(feature_count),
        "support_windows": int(len(support.dataset)),
        "selection_windows": int(len(selection.dataset)),
        "confirmation_windows": int(len(confirmation.dataset)),
        "split": split,
        "protocol_hash": protocol["protocol_hash"],
        "gpu_inventory": query_gpus(),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry_report)
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return

    old_tasks = [
        (OLD_GROUP, seed)
        for seed in (OLD_SEEDS if not args.quick else [52, 53])
    ]
    new_tasks = [
        (NEW_GROUP, seed)
        for seed in (NEW_SEEDS if not args.quick else [70, 71])
    ]
    train_tasks = old_tasks + (
        new_tasks if (args.retrain_new_seeds or args.quick) else []
    )
    registered_a1_2 = None
    if not (args.retrain_new_seeds or args.quick):
        # Fail before expensive training if the registered bridge anchor is
        # absent, incomplete, contaminated, or protocol-incompatible.
        registered_a1_2 = load_registered_a1_2(a1_2_dir, protocol)

    run_workers(args, train_tasks, output)
    results, prediction_parts, engine_parts, inventory_parts = load_shards(
        output, train_tasks
    )

    imported_decision: dict | None = None
    if not (args.retrain_new_seeds or args.quick):
        assert registered_a1_2 is not None
        imported = registered_a1_2
        new_results, new_predictions, new_engines, new_inventory = (
            normalize_imported_group(*imported[:4])
        )
        imported_decision = imported[4]
        results.extend(new_results)
        prediction_parts.append(new_predictions)
        engine_parts.append(new_engines)
        inventory_parts.append(new_inventory)

    run_frame = pd.DataFrame(results).sort_values(
        ["seed_group", "model_seed", "model", "target_split_seed"]
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    engines = pd.concat(engine_parts, ignore_index=True)
    inventory = pd.concat(inventory_parts, ignore_index=True)
    if run_frame.duplicated(
        ["seed_group", "model_seed", "target_split_seed", "model"]
    ).any():
        raise RuntimeError("duplicate bridge cells detected during merge")
    expected = (
        2
        * len(experiment["target_split_seeds"])
        * (4 if args.quick else 20)
    )
    if len(run_frame) != expected:
        raise RuntimeError(
            f"merged bridge has {len(run_frame)} cells; expected {expected}"
        )
    old_expected = OLD_SEEDS if not args.quick else [52, 53]
    new_expected = NEW_SEEDS if not args.quick else [70, 71]
    expected_keys = {
        (group_name, int(seed), int(split_seed), architecture)
        for group_name, seeds in (
            (OLD_GROUP, old_expected),
            (NEW_GROUP, new_expected),
        )
        for seed in seeds
        for split_seed in experiment["target_split_seeds"]
        for architecture in ARCHITECTURES
    }
    observed_keys = {
        (
            str(row.seed_group),
            int(row.model_seed),
            int(row.target_split_seed),
            str(row.model),
        )
        for row in run_frame.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:10]
        extra = sorted(observed_keys - expected_keys)[:10]
        raise RuntimeError(
            f"bridge cell-key mismatch; missing={missing}, extra={extra}"
        )
    result_ids = set(run_frame["replicate_id"].astype(str))
    if result_ids != set(predictions["replicate_id"].astype(str)):
        raise RuntimeError("merged run-level and prediction IDs are inconsistent")
    if result_ids != set(engines["replicate_id"].astype(str)):
        raise RuntimeError("merged run-level and per-engine IDs are inconsistent")
    for column in (
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        if column in run_frame and bool(run_frame[column].astype(bool).any()):
            raise RuntimeError(f"official-test contamination flag set: {column}")

    summary = bridge_summary(run_frame)
    paired, paired_split, comparisons = group_comparisons(
        run_frame.to_dict("records"),
        int(experiment["bootstrap_repetitions"]),
    )
    per_seed, interaction = interaction_analysis(
        paired, int(experiment["bootstrap_repetitions"])
    )
    (
        ensemble_results,
        ensemble_predictions,
        ensemble_engines,
        ensemble_summary,
        ensemble_comparisons,
    ) = build_group_ensembles(run_frame.to_dict("records"), predictions, protocol)
    ensemble_paired_parts = []
    for group_name in (OLD_GROUP, NEW_GROUP):
        subset = [
            row for row in ensemble_results if row["seed_group"] == group_name
        ]
        part = a12.ensemble_paired_cells(subset)
        part.insert(0, "seed_group", group_name)
        ensemble_paired_parts.append(part)
    ensemble_paired = pd.concat(ensemble_paired_parts, ignore_index=True)

    decision = final_decision(
        experiment,
        comparisons,
        ensemble_comparisons,
        interaction,
        bool(args.quick),
    )
    decision["completed_bridge_cells"] = int(len(run_frame))
    decision["imported_a1_2_decision"] = imported_decision

    atomic_json(paths["raw"], run_frame.to_dict("records"))
    a1.atomic_write_text(paths["run_csv"], run_frame.to_csv(index=False))
    a1.atomic_write_text(
        paths["window_predictions"], predictions.to_csv(index=False)
    )
    a1.atomic_write_text(paths["per_engine"], engines.to_csv(index=False))
    a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False))
    a1.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    a1.atomic_write_text(paths["paired_cell"], paired.to_csv(index=False))
    a1.atomic_write_text(
        paths["paired_split"], paired_split.to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False)
    )
    a1.atomic_write_text(paths["seed_summary"], per_seed.to_csv(index=False))
    atomic_json(paths["interaction"], interaction)
    atomic_json(paths["ensemble_raw"], ensemble_results)
    a1.atomic_write_text(
        paths["ensemble_run"], pd.DataFrame(ensemble_results).to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["ensemble_predictions"], ensemble_predictions.to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["ensemble_per_engine"], ensemble_engines.to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["ensemble_summary"], ensemble_summary.to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["ensemble_paired"], ensemble_paired.to_csv(index=False)
    )
    a1.atomic_write_text(
        paths["ensemble_comparisons"],
        ensemble_comparisons.to_csv(index=False),
    )
    atomic_json(paths["decision"], decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    base, experiment = experiment_config(args)
    a12.validate_config(base, experiment)
    if args.worker_seed is not None:
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
