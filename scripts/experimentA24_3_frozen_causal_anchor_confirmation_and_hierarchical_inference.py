#!/usr/bin/env python3
"""A24.3: frozen Meta-noGraph vs Meta-GNN causal-anchor confirmation.

This evaluation-only experiment consumes the completed A24.2 Reptile grid.
It never retrains, adapts, selects a policy, or reads official test files.  At
the pre-registered target K=5 it tests Meta-GNN against the parameter-matched
Meta-noGraph at causal RUL anchors 90, 45 and 15 on A23.0 confirmation engines.

Primary family: 3 anchors x (RMSE, NASA Score) = 6 superiority checks.  The
inference unit is an engine and the inferential resampling hierarchy is target
domain -> model seed -> support split -> paired engine.  Holm correction is
applied across all six checks.  Other K values are descriptive only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402
from scripts import experimentA23_2_causal_prefix_endpoint_audit as a232  # noqa: E402
from scripts import experimentA24_1_meta_no_graph_and_meta_gnn_pilot as pilot  # noqa: E402
from scripts import experimentA24_2_formal_meta_learning_factorial_training as a242  # noqa: E402


EXPERIMENT_ID = "experimentA24_3"
SCRIPT_VERSION = "experimentA24_3_frozen_causal_anchor_confirmation_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (130, 131, 132, 133, 134)
SPLITS = (7101, 7102, 7103, 7104, 7105)
SHOTS = (1, 2, 5, 10, 20)
METHODS = ("meta_no_graph_k", "meta_gnn_k")
PRIMARY_SHOT = 5
PRIMARY_CANDIDATE = "meta_gnn_k"
PRIMARY_REFERENCE = "meta_no_graph_k"
RUL_ANCHORS = (90.0, 45.0, 15.0)
METRICS = ("rmse", "nasa_score")
ALPHA = 0.05
BOOTSTRAP_REPETITIONS_MIN = 1000
EXPECTED_WORKERS = 100
EXPECTED_RUNS = 1000

# Make the reused causal-prefix dataset use exactly this locked endpoint set.
a232.REGISTERED_RUL_ANCHORS = RUL_ANCHORS
a232.EXPERIMENT_ID = EXPERIMENT_ID


class A243Error(RuntimeError):
    """Raised for any protocol, input-integrity, or evaluation failure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A24.3 frozen causal-anchor confirmation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--protocol-dir", type=Path,
                        default=Path("outputs/experimentA23_few_shot_protocol_preflight"))
    parser.add_argument("--a24-0-output-dir", type=Path,
                        default=Path("outputs/experimentA24_0_meta_learning_contract_preflight"))
    parser.add_argument("--a24-1-output-dir", type=Path,
                        default=Path("outputs/experimentA24_1_meta_no_graph_and_meta_gnn_pilot"))
    parser.add_argument("--a24-2-output-dir", type=Path,
                        default=Path("outputs/experimentA24_2_formal_meta_learning_factorial_training"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/experimentA24_3_frozen_causal_anchor_confirmation"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=243000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.torch_threads <= 0:
        raise A243Error("--torch-threads must be positive")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        raise A243Error("--inference-batch-size must be positive")
    if args.bootstrap_repetitions < BOOTSTRAP_REPETITIONS_MIN:
        raise A243Error(f"--bootstrap-repetitions must be at least {BOOTSTRAP_REPETITIONS_MIN}")
    # Populate every field required by the frozen A24.1/A24.2 config helpers.
    args.target_epochs = 10
    args.batch_size = None
    args.window_size = None
    args.rul_cap = None
    args.inner_learning_rate = None
    args.pair_aux_weight = None
    args.outer_steps = None
    args.inner_steps = None
    args.outer_learning_rate = None
    args.source_learning_rate = None
    args.learning_rate = None
    args.model_seeds = MODEL_SEEDS
    args.support_split_seeds = SPLITS
    args.shot = PRIMARY_SHOT
    return args


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A243Error(f"required JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A243Error(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A243Error(f"JSON root must be an object: {path}")
    return value


def strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise A243Error(f"{field} is not a strict Boolean: {value!r}")


def strict_bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise A243Error(f"required Boolean column is missing: {column}")
    return pd.Series([strict_bool(v, column) for v in frame[column]], index=frame.index)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n")


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def a24_config(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    local = deepcopy(args)
    local.config = resolve(args.config)
    local.protocol_dir = resolve(args.protocol_dir)
    local.learning_rate = None
    try:
        cfg = a242.formal_config(local, protocol)
    except Exception as exc:
        raise A243Error(f"could not reconstruct frozen A24 config: {exc}") from exc
    if int(cfg["outer_steps"]) != 1500 or int(cfg["meta_inner_steps"]) != 5:
        raise A243Error("frozen A24 Reptile schedule differs from A24.1/A24.2 contract")
    return cfg


def load_a242_contract(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    root = resolve(args.a24_2_output_dir)
    decision_path = root / "experimentA24_2_confirmation_decision.json"
    manifest_path = root / "experimentA24_2_manifest.json"
    run_path = root / "experimentA24_2_training_run_level.csv"
    audit_path = root / "experimentA24_2_parameter_audit.csv"
    for path in (decision_path, manifest_path, run_path, audit_path):
        if not path.is_file():
            raise A243Error(f"required A24.2 artifact is missing: {path}")
    decision, manifest = read_json(decision_path), read_json(manifest_path)
    expected_decision = {
        "experiment_id": "experimentA24_2", "complete": True, "passed": True,
        "training_only": True, "expected_worker_cells": EXPECTED_WORKERS,
        "completed_worker_cells": EXPECTED_WORKERS,
        "expected_training_records": EXPECTED_RUNS,
        "completed_training_records": EXPECTED_RUNS,
        "selection_engines_evaluated": False, "confirmation_engines_evaluated": False,
        "checkpoint_reload_passed": True,
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    for key, wanted in expected_decision.items():
        if decision.get(key) != wanted:
            raise A243Error(f"A24.2 decision integrity failure: {key}={decision.get(key)!r}")
    if tuple(decision.get("methods", ())) != METHODS or tuple(decision.get("shots", ())) != SHOTS:
        raise A243Error("A24.2 method or shot contract differs from A24.3")
    if manifest.get("script_version") != a242.SCRIPT_VERSION:
        raise A243Error("A24.2 manifest script version is not the frozen formal version")
    if manifest.get("script_sha256") != sha256(Path(a242.__file__).resolve()):
        raise A243Error("A24.2 source changed after the frozen training run")
    artifacts = manifest.get("artifacts", {})
    for path in (run_path, audit_path):
        if artifacts.get(path.name) != sha256(path):
            raise A243Error(f"A24.2 artifact SHA-256 mismatch: {path.name}")
    runs = pd.read_csv(run_path)
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "checkpoint", "checkpoint_sha256", "checkpoint_reload_passed",
        "selection_engines_evaluated", "confirmation_engines_evaluated",
        "official_test_files_accessed", "official_test_forward_run",
    }
    if missing := required - set(runs.columns):
        raise A243Error(f"A24.2 run table lacks columns: {sorted(missing)}")
    for col in ("model_seed", "support_split_seed", "shot"):
        runs[col] = pd.to_numeric(runs[col], errors="raise").astype(int)
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method"]
    expected = {(d, s, p, k, m) for d in DOMAINS for s in MODEL_SEEDS
                for p in SPLITS for k in SHOTS for m in METHODS}
    actual = set(runs[keys].itertuples(index=False, name=None))
    if len(runs) != EXPECTED_RUNS or actual != expected or runs.duplicated(keys).any():
        raise A243Error("A24.2 factorial run grid is incomplete or duplicated")
    for column in ("selection_engines_evaluated", "confirmation_engines_evaluated",
                   "official_test_files_accessed", "official_test_forward_run"):
        if strict_bool_column(runs, column).any():
            raise A243Error(f"A24.2 training-only boundary violation: {column}=true")
    if not strict_bool_column(runs, "checkpoint_reload_passed").all():
        raise A243Error("A24.2 recorded a checkpoint reload failure")
    return runs.sort_values(keys, kind="stable").reset_index(drop=True), decision, manifest


def validate_checkpoints(
    runs: pd.DataFrame, cfg: dict[str, Any], contract_hashes: dict[str, str], roles: pd.DataFrame,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(runs.iterrows(), start=1):
        path = Path(str(row["checkpoint"])).expanduser().resolve()
        if not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
            raise A243Error(f"missing or changed A24.2 checkpoint: {path}")
        try:
            payload = a242.strict_checkpoint_roundtrip(path, str(row["method"]), cfg,
                                                        int(row["model_seed"]))
        except Exception as exc:
            raise A243Error(f"checkpoint cannot be reloaded: {path}: {exc}") from exc
        expected = {
            "script_version": a242.SCRIPT_VERSION,
            "target_domain": str(row["target_domain"]),
            "model_seed": int(row["model_seed"]),
            "support_split_seed": int(row["support_split_seed"]),
            "shot": int(row["shot"]), "method": str(row["method"]),
            "target_epochs": 10,
            "selection_engines_evaluated": False,
            "confirmation_engines_evaluated": False,
        }
        for key, wanted in expected.items():
            if payload.get(key) != wanted:
                raise A243Error(f"checkpoint metadata mismatch {key}: {path}")
        if payload.get("contract_hashes") != contract_hashes:
            raise A243Error(f"A24.0 contract hash mismatch in checkpoint: {path}")
        expected_support = a23.role_engines(roles, str(row["target_domain"]),
                                            int(row["support_split_seed"]), "support_pool",
                                            int(row["shot"]))
        if list(payload.get("support_engines", ())) != list(expected_support):
            raise A243Error(f"support-engine metadata mismatch in checkpoint: {path}")
        model = pilot.make_model(str(row["method"]), cfg, int(row["model_seed"])).cpu()
        model.load_state_dict(payload["state"], strict=True)
        uses_gat = bool(getattr(model, "use_gat", False))
        if uses_gat != (str(row["method"]) == "meta_gnn_k"):
            raise A243Error(f"method/GAT identity mismatch in checkpoint: {path}")
        inventory.append({
            "target_domain": str(row["target_domain"]), "model_seed": int(row["model_seed"]),
            "support_split_seed": int(row["support_split_seed"]), "shot": int(row["shot"]),
            "method": str(row["method"]), "checkpoint_path": str(path),
            "checkpoint_sha256": str(row["checkpoint_sha256"]), "model_uses_gat": uses_gat,
        })
        del payload, model
        if index % 100 == 0 or index == len(runs):
            print(f"[A24.3] preflight checkpoints {index:04d}/{len(runs)}", flush=True)
    return inventory


def build_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = resolve(args.output_dir)
    input_root = resolve(args.a24_2_output_dir)
    if output_root == input_root:
        raise A243Error("--output-dir must differ from --a24-2-output-dir")
    try:
        protocol, _, contract_hashes = pilot.load_contract(args)
        a242.validate_a24_0_manifest(args, contract_hashes)
        a242.validate_protocol_and_config(args, protocol)
        cfg = a24_config(args, protocol)
        compat = pilot.a23_namespace(args); compat.shots = SHOTS
        _, roles, a23_hashes = a23.load_protocol(compat)
        raw = pilot.load_raw_frames(args, protocol, cfg)
    except Exception as exc:
        raise A243Error(f"A24.0/A23.0 contract validation failed: {exc}") from exc
    if sha256(resolve(args.config)) != str(protocol.get("config_sha256", "")):
        raise A243Error("current config differs from A24.0 locked config")
    runs, a242_decision, a242_manifest = load_a242_contract(args, cfg)
    inventory = validate_checkpoints(runs, cfg, contract_hashes, roles)
    coverage_rows: list[dict[str, Any]] = []
    expected_predictions = 0
    for domain in DOMAINS:
        identity = {"mean": {c: 0.0 for c in a23.FEATURE_COLUMNS},
                    "std": {c: 1.0 for c in a23.FEATURE_COLUMNS}}
        raw_target = raw[domain]
        for split in SPLITS:
            engines = a232.confirmation_engines(roles, domain, split)
            dataset = a232.CausalPrefixDataset(a23.normalize(raw_target, identity), engines,
                                               int(cfg["window_size"]))
            coverage_rows.extend({"target_domain": domain, "support_split_seed": split, **x}
                                 for x in dataset.meta)
            expected_predictions += len(dataset) * len(MODEL_SEEDS) * len(SHOTS) * len(METHODS)
    coverage = pd.DataFrame(coverage_rows)
    stage_counts = coverage["rul_stage"].value_counts().to_dict()
    required_stages = {"high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30"}
    if set(stage_counts) != required_stages or any(int(stage_counts[s]) <= 0 for s in required_stages):
        raise A243Error(f"causal anchors do not cover all RUL stages: {stage_counts}")
    batch_size = int(args.inference_batch_size or cfg["batch_size"])
    preview = {
        "experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
        "registered_primary_question": (
            "At K=5 under the frozen A24.2 Reptile grid, does Meta-GNN strictly improve "
            "parameter-matched Meta-noGraph on confirmation engines at causal RUL anchors "
            "90, 45 and 15 for both RMSE and NASA Score?"
        ),
        "a24_2_input_dir": str(input_root), "output_dir": str(output_root),
        "primary_shot": PRIMARY_SHOT, "candidate": PRIMARY_CANDIDATE,
        "reference": PRIMARY_REFERENCE, "registered_rul_anchors": list(RUL_ANCHORS),
        "primary_metrics": list(METRICS), "primary_checks": 6,
        "primary_decision_rule": "all_six_holm_superiority_checks_pass",
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
        "expected_checkpoint_runs": EXPECTED_RUNS, "validated_checkpoint_runs": len(inventory),
        "expected_prediction_records": int(expected_predictions),
        "unique_confirmation_engine_prefixes": int(len(coverage)),
        "coverage_by_rul_stage": {k: int(v) for k, v in stage_counts.items()},
        "inference_batch_size": batch_size,
        "mixed_registered_endpoints_within_inference_batch": False,
        "model_input_uses_future_cycles": False,
        "checkpoint_schema_hash_and_model_compatibility_passed": True,
        "training_file_hashes_match_A24_0": True,
        "new_predictor_training": False, "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    return preview, {
        "output_root": output_root, "input_root": input_root, "runs": runs, "roles": roles,
        "raw": raw, "cfg": cfg, "batch_size": batch_size, "contract_hashes": contract_hashes,
        "a23_hashes": a23_hashes, "coverage": coverage, "inventory": inventory,
        "a242_decision": a242_decision, "a242_manifest": a242_manifest,
    }


def shard_stem(row: pd.Series) -> str:
    return (f"{row['target_domain']}_mseed{int(row['model_seed'])}_split"
            f"{int(row['support_split_seed'])}_shot{int(row['shot']):02d}_{row['method']}")


def shard_paths(root: Path, row: pd.Series) -> tuple[Path, Path]:
    directory = root / "prediction_shards"
    stem = shard_stem(row)
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def validate_shard(csv_path: Path, status_path: Path, row: pd.Series, expected_rows: int) -> pd.DataFrame | None:
    if not csv_path.is_file() or not status_path.is_file():
        return None
    try:
        status, frame = read_json(status_path), pd.read_csv(csv_path)
        required = {"complete": True, "passed": True, "run_key": shard_stem(row),
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "prediction_sha256": sha256(csv_path), "prediction_records": expected_rows}
        if any(status.get(k) != v for k, v in required.items()):
            return None
        keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method",
                "engine_id", "prefix_label"]
        if len(frame) != expected_rows or frame.duplicated(keys).any():
            return None
        if (set(frame["target_domain"].astype(str)) != {str(row["target_domain"])}
                or set(frame["model_seed"].astype(int)) != {int(row["model_seed"])}
                or set(frame["support_split_seed"].astype(int)) != {int(row["support_split_seed"])}
                or set(frame["shot"].astype(int)) != {int(row["shot"])}
                or set(frame["method"].astype(str)) != {str(row["method"])}
                or set(frame["checkpoint_sha256"].astype(str)) != {str(row["checkpoint_sha256"])}):
            return None
        numeric = frame[["true_rul", "prediction", "error", "absolute_error", "squared_error",
                         "nasa_score_component"]].to_numpy(float)
        if not np.isfinite(numeric).all():
            return None
        if strict_bool_column(frame, "confirmation_used_for_training").any():
            return None
        return frame
    except Exception:
        return None


def normalised_target_frames(context: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for target in DOMAINS:
        sources = {d: context["raw"][d] for d in DOMAINS if d != target}
        normalizer = a23.fit_source_normalizer(sources)
        frames[target] = a23.normalize(context["raw"][target], normalizer)
    return frames


@torch.no_grad()
def infer_one(row: pd.Series, dataset: a232.CausalPrefixDataset,
              context: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    path = Path(str(row["checkpoint"])).expanduser().resolve()
    payload = a242.strict_checkpoint_roundtrip(path, str(row["method"]), context["cfg"],
                                                int(row["model_seed"]))
    model = pilot.make_model(str(row["method"]), context["cfg"], int(row["model_seed"]))
    model.load_state_dict(payload["state"], strict=True)
    model = model.to(device).eval()
    uses_gat = bool(getattr(model, "use_gat", False))
    if uses_gat != (str(row["method"]) == PRIMARY_CANDIDATE):
        raise A243Error(f"method/GAT mismatch at inference: {path}")
    predictions = np.empty(len(dataset), dtype=np.float64)
    seen = np.zeros(len(dataset), dtype=bool)
    for anchor in RUL_ANCHORS:
        label = a232.endpoint_label(anchor)
        indices = [i for i, meta in enumerate(dataset.meta) if meta["prefix_label"] == label]
        if len(indices) * len(RUL_ANCHORS) != len(dataset):
            raise A243Error(f"anchor subset cardinality mismatch: {path}")
        loader = a232.deterministic_loader(Subset(dataset, indices), context["batch_size"], device)
        for x, location in loader:
            output = model(x.to(device, non_blocking=device.type == "cuda"))
            if isinstance(output, tuple):
                output = output[0]
            values = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
            where = location.numpy().reshape(-1).astype(int)
            if len(values) != len(where):
                raise A243Error(f"prediction cardinality mismatch: {path}")
            predictions[where] = values; seen[where] = True
    if not seen.all() or not np.isfinite(predictions).all():
        raise A243Error(f"missing/non-finite predictions: {path}")
    records: list[dict[str, Any]] = []
    for i, (prediction, meta) in enumerate(zip(predictions, dataset.meta)):
        truth = float(meta["true_rul"]); error = float(prediction - truth)
        nasa = (math.exp(error / 10.0) - 1.0 if error >= 0.0
                else math.exp(-error / 13.0) - 1.0)
        records.append({
            "experiment_id": EXPERIMENT_ID, "target_domain": str(row["target_domain"]),
            "model_seed": int(row["model_seed"]), "support_split_seed": int(row["support_split_seed"]),
            "shot": int(row["shot"]), "method": str(row["method"]),
            "checkpoint_path": str(path), "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "endpoint_index_in_dataset": i, **meta, "prediction": float(prediction),
            "error": error, "absolute_error": abs(error), "squared_error": error * error,
            "nasa_score_component": float(nasa), "model_uses_gat": uses_gat,
            "mixed_registered_endpoints_within_inference_batch": False,
            "confirmation_used_for_training": False, "new_predictor_training": False,
            "official_test_files_accessed": False, "official_test_forward_run": False,
        })
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def run_inference(args: argparse.Namespace, preview: dict[str, Any], context: dict[str, Any]) -> pd.DataFrame:
    try:
        device = a232.resolve_device(args.device)
    except Exception as exc:
        raise A243Error(str(exc)) from exc
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(int(args.bootstrap_seed)); np.random.seed(int(args.bootstrap_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.bootstrap_seed))
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    frames = normalised_target_frames(context)
    datasets: dict[tuple[str, int], a232.CausalPrefixDataset] = {}
    outputs: list[pd.DataFrame] = []; reused = 0
    for index, (_, row) in enumerate(context["runs"].iterrows(), start=1):
        key = (str(row["target_domain"]), int(row["support_split_seed"]))
        if key not in datasets:
            engines = a232.confirmation_engines(context["roles"], key[0], key[1])
            datasets[key] = a232.CausalPrefixDataset(frames[key[0]], engines,
                                                      int(context["cfg"]["window_size"]))
        dataset = datasets[key]
        csv_path, status_path = shard_paths(context["output_root"], row)
        prior = validate_shard(csv_path, status_path, row, len(dataset)) if args.resume else None
        if prior is None:
            try:
                frame = pd.DataFrame(infer_one(row, dataset, context, device))
            except Exception as exc:
                raise A243Error(f"inference failed for {shard_stem(row)}: {exc}") from exc
            atomic_frame(csv_path, frame)
            atomic_json(status_path, {
                "experiment_id": EXPERIMENT_ID, "complete": True, "passed": True,
                "run_key": shard_stem(row), "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "prediction_records": len(frame), "prediction_sha256": sha256(csv_path),
                "new_predictor_training": False, "official_test_files_accessed": False,
                "official_test_forward_run": False,
            })
        else:
            frame = prior; reused += 1
        outputs.append(frame)
        if index % 20 == 0 or index == len(context["runs"]):
            print(f"[A24.3] evaluated {index:04d}/{len(context['runs']):04d} "
                  f"reused_shards={reused} device={device}", flush=True)
    predictions = pd.concat(outputs, ignore_index=True)
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method",
            "engine_id", "prefix_label"]
    if len(predictions) != int(preview["expected_prediction_records"]) or predictions.duplicated(keys).any():
        raise A243Error("prediction count/key integrity failure")
    return predictions.sort_values(keys, kind="stable").reset_index(drop=True)


def build_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method", "prefix_label"]
    for key, frame in predictions.groupby(keys, sort=True):
        errors = frame["error"].to_numpy(np.float64)
        rows.append({"experiment_id": EXPERIMENT_ID, **dict(zip(keys, key)),
                     "registered_rul_anchor": float(frame["registered_rul_anchor"].iloc[0]),
                     "rul_stage": str(frame["rul_stage"].iloc[0]),
                     "n_engines": int(frame["engine_id"].nunique()),
                     "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                     "mae": float(np.mean(np.abs(errors))), "mean_error": float(np.mean(errors)),
                     "nasa_score": float(frame["nasa_score_component"].sum())})
    return pd.DataFrame(rows)


def build_paired(predictions: pd.DataFrame) -> pd.DataFrame:
    ids = ["target_domain", "model_seed", "support_split_seed", "shot", "engine_id", "prefix_label"]
    values = ids + ["registered_rul_anchor", "rul_stage", "true_rul", "prediction", "error",
                    "absolute_error", "squared_error", "nasa_score_component"]
    candidate = predictions.loc[predictions["method"] == PRIMARY_CANDIDATE, values].copy()
    reference = predictions.loc[predictions["method"] == PRIMARY_REFERENCE, values].copy()
    if candidate.duplicated(ids).any() or reference.duplicated(ids).any():
        raise A243Error("candidate/reference prediction keys are duplicated")
    paired = candidate.merge(reference, on=ids, suffixes=("_candidate", "_reference"),
                             validate="one_to_one")
    if len(paired) != len(candidate) or len(reference) != len(candidate):
        raise A243Error("candidate/reference pairing is incomplete")
    for column in ("true_rul", "registered_rul_anchor"):
        if not np.allclose(paired[f"{column}_candidate"], paired[f"{column}_reference"], atol=0, rtol=0):
            raise A243Error(f"paired {column} values differ")
    return paired


def relative_metric(frame: pd.DataFrame, metric: str) -> float:
    if metric == "rmse":
        candidate = math.sqrt(float(frame["squared_error_candidate"].mean()))
        reference = math.sqrt(float(frame["squared_error_reference"].mean()))
    elif metric == "nasa_score":
        candidate = float(frame["nasa_score_component_candidate"].sum())
        reference = float(frame["nasa_score_component_reference"].sum())
    else:
        raise A243Error(f"unknown metric: {metric}")
    if not (math.isfinite(candidate) and math.isfinite(reference) and reference > 0):
        raise A243Error(f"invalid aggregate {metric}")
    return candidate / reference - 1.0


def hierarchical_bootstrap(frame: pd.DataFrame, metric: str, repetitions: int, seed: int) -> np.ndarray:
    nested: dict[str, dict[int, dict[int, pd.DataFrame]]] = {}
    for domain, dframe in frame.groupby("target_domain", sort=True):
        nested[str(domain)] = {}
        for model_seed, sframe in dframe.groupby("model_seed", sort=True):
            nested[str(domain)][int(model_seed)] = {
                int(split): x.reset_index(drop=True) for split, x in sframe.groupby("support_split_seed", sort=True)
            }
    if tuple(sorted(nested)) != DOMAINS:
        raise A243Error("bootstrap target domains are incomplete")
    for d in DOMAINS:
        if tuple(sorted(nested[d])) != MODEL_SEEDS:
            raise A243Error(f"bootstrap model seeds are incomplete for {d}")
        for seed_value in MODEL_SEEDS:
            if tuple(sorted(nested[d][seed_value])) != SPLITS:
                raise A243Error(f"bootstrap support splits are incomplete for {d}/{seed_value}")
    rng, samples = np.random.default_rng(int(seed)), np.empty(int(repetitions), dtype=np.float64)
    for rep in range(int(repetitions)):
        pieces: list[pd.DataFrame] = []
        for domain in rng.choice(np.asarray(DOMAINS, dtype=object), size=len(DOMAINS), replace=True):
            for model_seed in rng.choice(np.asarray(MODEL_SEEDS), size=len(MODEL_SEEDS), replace=True):
                for split in rng.choice(np.asarray(SPLITS), size=len(SPLITS), replace=True):
                    cluster = nested[str(domain)][int(model_seed)][int(split)]
                    pieces.append(cluster.iloc[rng.integers(0, len(cluster), size=len(cluster))])
        samples[rep] = relative_metric(pd.concat(pieces, ignore_index=True), metric)
    if not np.isfinite(samples).all():
        raise A243Error("bootstrap produced non-finite values")
    return samples


def holm_adjust(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise A243Error("invalid p-value vector")
    order, result, running = np.argsort(p, kind="stable"), np.empty(len(p)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float((len(p) - rank) * p[index])))
        result[index] = running
    return result.tolist()


def primary_inference(paired: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(RUL_ANCHORS):
        label = a232.endpoint_label(anchor)
        scoped = paired.loc[(paired["shot"] == PRIMARY_SHOT) & (paired["prefix_label"] == label)].copy()
        if scoped.groupby(["target_domain", "model_seed", "support_split_seed"]).ngroups != 100:
            raise A243Error(f"primary K=5 pairs are incomplete at {label}")
        for metric_index, metric in enumerate(METRICS):
            point = relative_metric(scoped, metric)
            samples = hierarchical_bootstrap(scoped, metric, repetitions,
                                             int(seed) + anchor_index * 1000 + metric_index * 100)
            if metric == "rmse":
                candidate = math.sqrt(float(scoped["squared_error_candidate"].mean()))
                reference = math.sqrt(float(scoped["squared_error_reference"].mean()))
                wins = float(np.mean(scoped["absolute_error_candidate"] < scoped["absolute_error_reference"]))
            else:
                candidate = float(scoped["nasa_score_component_candidate"].sum())
                reference = float(scoped["nasa_score_component_reference"].sum())
                wins = float(np.mean(scoped["nasa_score_component_candidate"] < scoped["nasa_score_component_reference"]))
            rows.append({
                "experiment_id": EXPERIMENT_ID, "shot": PRIMARY_SHOT, "prefix_label": label,
                "registered_rul_anchor": anchor, "rul_stage": str(scoped["rul_stage_candidate"].iloc[0]),
                "metric": metric, "n_paired_engines": int(len(scoped)),
                "candidate": PRIMARY_CANDIDATE, "reference": PRIMARY_REFERENCE,
                "candidate_value": candidate, "reference_value": reference,
                "relative_degradation": point, "relative_improvement_pct": -100.0 * point,
                "relative_ci95_low": float(np.quantile(samples, ALPHA / 2)),
                "relative_ci95_high": float(np.quantile(samples, 1 - ALPHA / 2)),
                "candidate_engine_win_rate": wins,
                "one_sided_bootstrap_tail_probability_superiority": float(
                    (1 + np.count_nonzero(samples >= 0.0)) / (len(samples) + 1)),
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
            })
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_superiority"] = holm_adjust(
        result["one_sided_bootstrap_tail_probability_superiority"].tolist())
    result["holm_superiority_passed"] = ((result["holm_adjusted_p_superiority"] < ALPHA)
                                        & (result["relative_ci95_high"] < 0.0))
    return result


def secondary_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (shot, label), frame in paired.groupby(["shot", "prefix_label"], sort=True):
        for metric in METRICS:
            point = relative_metric(frame, metric)
            rows.append({"experiment_id": EXPERIMENT_ID,
                         "analysis_role": "primary" if int(shot) == PRIMARY_SHOT else "secondary_descriptive",
                         "shot": int(shot), "prefix_label": str(label),
                         "registered_rul_anchor": float(frame["registered_rul_anchor_candidate"].iloc[0]),
                         "rul_stage": str(frame["rul_stage_candidate"].iloc[0]), "metric": metric,
                         "n_paired_engines": len(frame), "relative_degradation": point,
                         "relative_improvement_pct": -100.0 * point})
    return pd.DataFrame(rows)


def domain_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = paired.loc[paired["shot"] == PRIMARY_SHOT]
    for (domain, label), frame in primary.groupby(["target_domain", "prefix_label"], sort=True):
        for metric in METRICS:
            point = relative_metric(frame, metric)
            rows.append({"experiment_id": EXPERIMENT_ID, "target_domain": str(domain),
                         "shot": PRIMARY_SHOT, "prefix_label": str(label),
                         "registered_rul_anchor": float(frame["registered_rul_anchor_candidate"].iloc[0]),
                         "rul_stage": str(frame["rul_stage_candidate"].iloc[0]), "metric": metric,
                         "n_paired_engines": len(frame), "relative_degradation": point,
                         "relative_improvement_pct": -100.0 * point})
    return pd.DataFrame(rows)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path, self.acquired = path, False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(read_json(self.path).get("pid"))
                os.kill(pid, 0)
            except ProcessLookupError:
                self.path.unlink(missing_ok=True)
            except (ValueError, TypeError, A243Error):
                # A malformed lock cannot identify a live parent process, but it
                # is still removed only from this experiment's dedicated path.
                self.path.unlink(missing_ok=True)
            except PermissionError as exc:
                raise A243Error(f"cannot verify existing A24.3 lock: {self.path}") from exc
            else:
                raise A243Error(f"another A24.3 parent is active; lock={self.path}")
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()}
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle); handle.flush(); os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, trace) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def finalise(args: argparse.Namespace, preview: dict[str, Any], context: dict[str, Any],
             predictions: pd.DataFrame) -> dict[str, Any]:
    root = context["output_root"]
    run_metrics, paired = build_run_metrics(predictions), build_paired(predictions)
    primary = primary_inference(paired, int(args.bootstrap_repetitions), int(args.bootstrap_seed))
    secondary, domains = secondary_summary(paired), domain_summary(paired)
    passed = bool(primary["holm_superiority_passed"].all())
    artifacts = {
        "experimentA24_3_causal_anchor_predictions.csv": predictions,
        "experimentA24_3_run_level_metrics.csv": run_metrics,
        "experimentA24_3_paired_meta_gnn_vs_meta_no_graph.csv": paired,
        "experimentA24_3_primary_hierarchical_inference.csv": primary,
        "experimentA24_3_secondary_shot_summary.csv": secondary,
        "experimentA24_3_domain_summary.csv": domains,
        "experimentA24_3_prefix_coverage.csv": context["coverage"],
        "experimentA24_3_checkpoint_inventory.csv": pd.DataFrame(context["inventory"]),
    }
    for name, frame in artifacts.items():
        atomic_frame(root / name, frame)
    decision = {
        "experiment_id": EXPERIMENT_ID, "registered_primary_question": preview["registered_primary_question"],
        "complete": True, "passed": passed, "evaluation_only": True,
        "new_predictor_training": False, "target_adaptation": False,
        "policy_selection_or_tuning": False, "candidate": PRIMARY_CANDIDATE,
        "reference": PRIMARY_REFERENCE, "primary_shot": PRIMARY_SHOT,
        "registered_rul_anchors": list(RUL_ANCHORS),
        "expected_checkpoint_runs": EXPECTED_RUNS, "completed_checkpoint_evaluations": EXPECTED_RUNS,
        "expected_prediction_records": int(preview["expected_prediction_records"]),
        "completed_prediction_records": len(predictions), "expected_primary_checks": 6,
        "completed_primary_checks": len(primary),
        "primary_decision_rule": "all_six_holm_superiority_checks_pass",
        "primary_checks_passed": int(primary["holm_superiority_passed"].sum()),
        "all_primary_holm_superiority_checks_passed": passed,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed), "bootstrap_design": preview["bootstrap_design"],
        "mixed_registered_endpoints_within_inference_batch": False,
        "model_input_uses_future_cycles": False, "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": ("A24.3 confirmed Meta-GNN superiority over Meta-noGraph at all locked K=5 causal anchors and metrics"
                   if passed else "A24.3 completed frozen causal-anchor confirmation, but Meta-GNN did not pass every Holm-corrected K=5 superiority check"),
        "interpretation_limit": ("A24.3 tests only the incremental effect of the GAT branch within the frozen Reptile implementation on training-file confirmation engines. It does not compare against ordinary transfer baselines, diagnose fault classes, or create an official-test claim."),
        "next_action": ("run_A24_4_meta_vs_ordinary_transfer_comparative_synthesis"
                        if passed else "report_A24_3_graph_increment_boundary_before_meta_vs_baseline_claims"),
    }
    atomic_json(root / "experimentA24_3_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(), "registered_analysis": {
            "primary_shot": PRIMARY_SHOT, "rul_anchors": list(RUL_ANCHORS),
            "metrics": list(METRICS), "alpha": ALPHA,
            "decision_rule": "all_six_holm_superiority_checks_pass",
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "bootstrap_seed": int(args.bootstrap_seed), "bootstrap_design": preview["bootstrap_design"],
        },
        "inputs": {
            "a24_2_decision_sha256": sha256(context["input_root"] / "experimentA24_2_confirmation_decision.json"),
            "a24_2_manifest_sha256": sha256(context["input_root"] / "experimentA24_2_manifest.json"),
            "a24_2_run_level_sha256": sha256(context["input_root"] / "experimentA24_2_training_run_level.csv"),
            "a24_0_contract_hashes": context["contract_hashes"],
            "a23_protocol_hashes": context["a23_hashes"], "config_sha256": sha256(resolve(args.config)),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "artifacts": {name: sha256(root / name) for name in sorted(list(artifacts) + [
            "experimentA24_3_preflight.json", "experimentA24_3_confirmation_decision.json"])},
        "prediction_shards_excluded_from_final_hash_manifest": True,
        "new_predictor_training": False, "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA24_3_manifest.json", manifest)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve(args.output_dir)
    decision_path = root / "experimentA24_3_confirmation_decision.json"
    if decision_path.is_file():
        prior = read_json(decision_path)
        if args.resume and prior.get("complete") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2), flush=True)
            print("[A24.3] resume: existing complete decision returned", flush=True)
            return 0
        raise A243Error(f"completed decision already exists: {decision_path}; use --resume or a new output directory")
    preview, context = build_context(args)
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("[A24.3] dry-run passed; all frozen checkpoints/contracts are compatible; no model was trained", flush=True)
        return 0
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise A243Error("non-empty output directory without decision; use --resume only for validated shards")
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(root / "experimentA24_3_run.lock"):
        atomic_json(root / "experimentA24_3_preflight.json", preview)
        predictions = run_inference(args, preview, context)
        decision = finalise(args, preview, context, predictions)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (A243Error, a23.A23Error, pilot.A24Error, a242.A24FormalError) as exc:
        print(f"[A24.3] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
