#!/usr/bin/env python3
"""A23.4 formal causal-anchor evaluation and hierarchical inference.

This evaluation-only experiment consumes the immutable A23.0 protocol and all
1,500 A23.3 baseline checkpoints.  It performs no training, adaptation,
selection or hyper-parameter tuning.  Each confirmation engine is evaluated
at the pre-registered causal prefixes whose offline true-RUL anchors are
90, 45 and 15 cycles.  No observation after an anchor is part of model input.

The registered primary comparison is pretrain_finetune_k versus scratch_k at
K=5.  The primary family contains six superiority checks (three anchors by
RMSE/NASA score) with a target-domain -> model-seed -> support-split -> paired
engine hierarchical bootstrap and Holm correction.  Other shot counts are
reported as secondary, non-selective dose-response analyses.

Operational safeguards:
* only C-MAPSS training files are resolved;
* every input artifact, checkpoint and normalizer is hash/schema validated;
* GAT inference is performed separately for each registered RUL anchor;
* one atomic prediction shard is written per checkpoint;
* --resume reuses only fully validated shards;
* final artifacts are written atomically under an exclusive run lock.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402
from scripts import experimentA23_2_causal_prefix_endpoint_audit as a232  # noqa: E402


EXPERIMENT_ID = "experimentA23_4"
SCRIPT_VERSION = "experimentA23_4_formal_causal_anchor_evaluation_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (130, 131, 132, 133, 134)
SUPPORT_SPLIT_SEEDS = (7101, 7102, 7103, 7104, 7105)
SHOTS = (1, 2, 5, 10, 20)
PRIMARY_SHOT = 5
REGIMES = ("source_only", "scratch_k", "pretrain_finetune_k")
PRIMARY_CANDIDATE = "pretrain_finetune_k"
PRIMARY_REFERENCE = "scratch_k"
RUL_ANCHORS = (90.0, 45.0, 15.0)
PRIMARY_METRICS = ("rmse", "nasa_score")
ALPHA = 0.05
SECONDARY_NONINFERIORITY_MARGIN = 0.03
SOURCE_INVARIANCE_TOLERANCE = 1e-6

# Reuse A23.2's audited causal-prefix constructor and inference implementation
# with the enlarged, immutable A23.3 factorial contract.
a232.EXPERIMENT_ID = EXPERIMENT_ID
a232.REGISTERED_SHOTS = SHOTS
a232.REGISTERED_RUL_ANCHORS = RUL_ANCHORS
a232.SOURCE_INVARIANCE_TOLERANCE = SOURCE_INVARIANCE_TOLERANCE


class A234Error(RuntimeError):
    """Raised when the registered A23.4 contract cannot be honoured."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A23.4 evaluation-only causal-anchor formal inference"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--a23-3-output-dir",
        type=Path,
        default=Path("outputs/experimentA23_3_formal_few_shot_transfer_baselines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experimentA23_4_formal_causal_anchor_evaluation_and_hierarchical_inference"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=234000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated prediction shards or return a completed decision.",
    )
    args = parser.parse_args(argv)
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        raise A234Error("--inference-batch-size must be positive")
    if args.torch_threads <= 0:
        raise A234Error("--torch-threads must be positive")
    if args.bootstrap_repetitions < 1000:
        raise A234Error("--bootstrap-repetitions must be at least 1000")
    return args


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
    )


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    rows = [{key: scalar(value) for key, value in row.items()} for row in frame.to_dict("records")]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame.columns), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A234Error(f"required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A234Error(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A234Error(f"JSON artifact must contain an object: {path}")
    return payload


def strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise A234Error(f"{field} is not a strict boolean: {value!r}")


def parse_json_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise A234Error(f"{field} is not a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise A234Error(f"{field} contains invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise A234Error(f"{field} is not a JSON object")
    return parsed


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    # A23.2 already implements the registered config parsing and validation.
    try:
        return a232.load_config(args)
    except Exception as exc:
        raise A234Error(str(exc)) from exc


def load_protocol_contract(
    root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    try:
        protocol, roles, hashes = a232.load_protocol_contract(root)
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    if tuple(int(value) for value in protocol.get("support_split_seeds", ())) != SUPPORT_SPLIT_SEEDS:
        raise A234Error("A23.0 support split seeds differ from the A23.4 contract")
    return protocol, roles, hashes


def load_a233_contract(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    decision_path = root / "experimentA23_3_confirmation_decision.json"
    manifest_path = root / "experimentA23_3_manifest.json"
    run_path = root / "experimentA23_3_training_run_level.csv"
    decision = read_json(decision_path)
    manifest = read_json(manifest_path)
    if decision.get("experiment_id") != "experimentA23_3":
        raise A234Error("input decision is not experimentA23_3")
    if not (decision.get("complete") is True and decision.get("passed") is True):
        raise A234Error("A23.3 must be complete and passed")
    if decision.get("training_only") is not True or decision.get("baseline_efficacy_claim") is not False:
        raise A234Error("A23.3 decision does not preserve the registered training-only boundary")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A234Error(f"A23.3 official-test boundary violated: {field}=true")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise A234Error("A23.3 manifest lacks an artifacts mapping")
    expected_hash = artifacts.get(run_path.name)
    if not isinstance(expected_hash, str) or sha256_file(run_path) != expected_hash:
        raise A234Error("A23.3 training run-level SHA-256 does not match its manifest")
    try:
        runs = pd.read_csv(run_path)
    except Exception as exc:
        raise A234Error(f"failed to parse A23.3 training run-level CSV: {exc}") from exc
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "regime",
        "feature_count", "feature_columns", "window_size", "rul_cap", "target_epochs",
        "source_pretrain_steps", "model_checkpoint", "model_checkpoint_sha256",
        "normalizer_path", "normalizer_sha256", "protocol_hashes",
        "causal_anchor_evaluation_contract", "confirmation_used_for_training",
        "confirmation_used_for_normalizer_fit", "selection_used_for_training",
        "selection_used_for_epoch_selection", "official_test_files_accessed",
        "official_test_forward_run",
    }
    if missing := required - set(runs.columns):
        raise A234Error(f"A23.3 run-level CSV lacks columns: {sorted(missing)}")
    for column in ("model_seed", "support_split_seed", "shot", "feature_count", "window_size"):
        runs[column] = pd.to_numeric(runs[column], errors="raise").astype(int)
    if len(runs) != 1500 or len(runs) != int(decision.get("completed_run_records", -1)):
        raise A234Error(f"A23.3 requires exactly 1500 run records; found {len(runs)}")
    if tuple(sorted(runs["target_domain"].astype(str).unique())) != DOMAINS:
        raise A234Error("A23.3 target-domain set is not FD001--FD004")
    if tuple(sorted(runs["model_seed"].unique())) != MODEL_SEEDS:
        raise A234Error("A23.3 model-seed set differs from the registered set")
    if tuple(sorted(runs["support_split_seed"].unique())) != SUPPORT_SPLIT_SEEDS:
        raise A234Error("A23.3 support-split set differs from the registered set")
    if tuple(sorted(runs["shot"].unique())) != SHOTS:
        raise A234Error("A23.3 shot set differs from the registered set")
    if tuple(sorted(runs["regime"].astype(str).unique())) != tuple(sorted(REGIMES)):
        raise A234Error("A23.3 regime set differs from the registered set")
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "regime"]
    expected = {
        (domain, seed, split, shot, regime)
        for domain in DOMAINS for seed in MODEL_SEEDS for split in SUPPORT_SPLIT_SEEDS
        for shot in SHOTS for regime in REGIMES
    }
    actual = set(runs[keys].itertuples(index=False, name=None))
    if actual != expected or runs.duplicated(keys).any():
        raise A234Error("A23.3 factorial grid is incomplete or duplicated")
    for column in (
        "confirmation_used_for_training", "confirmation_used_for_normalizer_fit",
        "selection_used_for_training", "selection_used_for_epoch_selection",
        "official_test_files_accessed", "official_test_forward_run",
    ):
        if any(strict_bool(value, field=column) for value in runs[column]):
            raise A234Error(f"A23.3 integrity violation: {column}=true")
    if set(runs["causal_anchor_evaluation_contract"].astype(str)) != {
        "A23.2_v2_rul_090_045_015"
    }:
        raise A234Error("A23.3 checkpoints do not share the locked A23.2 anchor contract")
    return runs.sort_values(keys, kind="stable").reset_index(drop=True), decision, manifest


def checkpoint_path(row: pd.Series) -> Path:
    return Path(str(row["model_checkpoint"])).expanduser().resolve()


def normalizer_path(row: pd.Series) -> Path:
    return Path(str(row["normalizer_path"])).expanduser().resolve()


def validate_checkpoint(
    row: pd.Series,
    cfg: dict[str, Any],
    protocol_hashes: dict[str, str],
) -> tuple[dict[str, Any], Path, bool]:
    path = checkpoint_path(row)
    if not path.is_file():
        raise A234Error(f"A23.3 checkpoint is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != str(row["model_checkpoint_sha256"]):
        raise A234Error(
            f"checkpoint SHA-256 mismatch: expected={row['model_checkpoint_sha256']}, "
            f"actual={actual_hash}, path={path}"
        )
    try:
        payload = a232.safe_load_checkpoint(path)
        a232.validate_checkpoint_metadata(payload, row, path, protocol_hashes)
        uses_gat = a232.validate_model_state(payload["state"], cfg, path)
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    npath = normalizer_path(row)
    if not npath.is_file() or sha256_file(npath) != str(row["normalizer_sha256"]):
        raise A234Error(f"normalizer missing or SHA-256 mismatch: {npath}")
    try:
        a232.load_normalizer(npath, str(row["target_domain"]))
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    if parse_json_mapping(row["protocol_hashes"], field="protocol_hashes") != protocol_hashes:
        raise A234Error(f"run-level protocol hashes differ from current A23.0: {path}")
    return payload, path, uses_gat


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    input_root = resolve(args.a23_3_output_dir)
    protocol_root = resolve(args.protocol_dir)
    output_root = resolve(args.output_dir)
    if input_root == output_root:
        raise A234Error("--output-dir must differ from --a23-3-output-dir")
    runs, decision, manifest = load_a233_contract(input_root)
    protocol, roles, protocol_hashes = load_protocol_contract(protocol_root)
    cfg, config_path = load_config(args)
    try:
        training_files = a232.verify_training_files(resolve(args.data_dir), protocol)
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    manifest_config_hash = manifest.get("artifacts", {}).get("config_sha256")
    if manifest_config_hash != sha256_file(config_path):
        raise A234Error("current --config SHA-256 differs from A23.3 manifest")
    if any(runs["window_size"] != int(cfg["window_size"])):
        raise A234Error("current config window_size differs from A23.3")
    if any(~np.isclose(runs["rul_cap"].astype(float), float(cfg["rul_cap"]), atol=1e-12)):
        raise A234Error("current config rul_cap differs from A23.3")
    for value in runs["feature_columns"]:
        if a232.parse_feature_columns(value) != tuple(a23.FEATURE_COLUMNS):
            raise A234Error("A23.3 feature columns differ from evaluator code")
    if any(runs["feature_count"] != len(a23.FEATURE_COLUMNS)):
        raise A234Error("A23.3 feature_count differs from evaluator code")

    checkpoint_rows: list[dict[str, Any]] = []
    gat_modes: set[bool] = set()
    for index, (_, row) in enumerate(runs.iterrows(), start=1):
        payload, path, uses_gat = validate_checkpoint(row, cfg, protocol_hashes)
        gat_modes.add(uses_gat)
        checkpoint_rows.append({
            "target_domain": str(row["target_domain"]),
            "model_seed": int(row["model_seed"]),
            "support_split_seed": int(row["support_split_seed"]),
            "shot": int(row["shot"]),
            "regime": str(row["regime"]),
            "checkpoint_path": str(path),
            "checkpoint_sha256": str(row["model_checkpoint_sha256"]),
            "normalizer_path": str(normalizer_path(row)),
            "normalizer_sha256": str(row["normalizer_sha256"]),
        })
        del payload
        if index % 100 == 0 or index == len(runs):
            print(f"[A23.4] preflight checkpoints {index:04d}/{len(runs)}", flush=True)
    if len(gat_modes) != 1:
        raise A234Error("A23.3 checkpoints disagree on GAT mode")

    raw_frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    expected_predictions = 0
    for domain in DOMAINS:
        raw_frames[domain] = a23.load_domain_frame(
            Path(training_files[domain]["path"]), rul_cap=float(cfg["rul_cap"])
        )
        identity = {
            "mean": {feature: 0.0 for feature in a23.FEATURE_COLUMNS},
            "std": {feature: 1.0 for feature in a23.FEATURE_COLUMNS},
        }
        identity_frame = a23.normalize(raw_frames[domain], identity)
        for split in SUPPORT_SPLIT_SEEDS:
            engines = a232.confirmation_engines(roles, domain, split)
            dataset = a232.CausalPrefixDataset(identity_frame, engines, int(cfg["window_size"]))
            for item in dataset.meta:
                coverage_rows.append({"target_domain": domain, "support_split_seed": split, **item})
            run_count = len(MODEL_SEEDS) * len(SHOTS) * len(REGIMES)
            expected_predictions += len(dataset) * run_count
    coverage = pd.DataFrame(coverage_rows)
    stage_counts = coverage["rul_stage"].value_counts().to_dict()
    expected_stages = {"high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30"}
    if set(stage_counts) != expected_stages or any(int(stage_counts[x]) <= 0 for x in expected_stages):
        raise A234Error(f"registered anchors do not cover all RUL stages: {stage_counts}")

    result = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": (
            "At the locked five-engine target-support budget, does ordinary source pretraining "
            "plus target fine-tuning strictly improve scratch training at all registered causal "
            "RUL anchors for RMSE and NASA Score?"
        ),
        "a23_3_input_dir": str(input_root),
        "protocol_dir": str(protocol_root),
        "output_dir": str(output_root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "registered_rul_anchors": list(RUL_ANCHORS),
        "primary_shot": PRIMARY_SHOT,
        "candidate": PRIMARY_CANDIDATE,
        "reference": PRIMARY_REFERENCE,
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_checks": len(RUL_ANCHORS) * len(PRIMARY_METRICS),
        "primary_decision_rule": "all_six_holm_superiority_checks_pass",
        "secondary_noninferiority_margin_pct": 100.0 * SECONDARY_NONINFERIORITY_MARGIN,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
        "expected_checkpoint_runs": int(len(runs)),
        "validated_checkpoint_runs": int(len(checkpoint_rows)),
        "expected_prediction_records": int(expected_predictions),
        "unique_confirmation_engine_prefixes": int(len(coverage)),
        "coverage_by_rul_stage": {key: int(value) for key, value in stage_counts.items()},
        "model_uses_gat": bool(next(iter(gat_modes))),
        "mixed_registered_endpoints_within_inference_batch": False,
        "model_input_uses_future_cycles": False,
        "checkpoint_schema_hash_and_model_compatibility_passed": True,
        "training_file_hashes_match_A23_0": True,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    context = {
        "input_root": input_root,
        "output_root": output_root,
        "runs": runs,
        "roles": roles,
        "cfg": cfg,
        "protocol_hashes": protocol_hashes,
        "training_files": training_files,
        "checkpoint_rows": checkpoint_rows,
        "coverage": coverage,
        "a233_decision": decision,
        "a233_manifest": manifest,
    }
    return result, context


def shard_stem(row: pd.Series) -> str:
    return (
        f"{row['target_domain']}_mseed{int(row['model_seed'])}_split{int(row['support_split_seed'])}_"
        f"shot{int(row['shot']):02d}_{row['regime']}"
    )


def prediction_shard_paths(root: Path, row: pd.Series) -> tuple[Path, Path]:
    stem = shard_stem(row)
    directory = root / "prediction_shards"
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def validate_prediction_shard(
    csv_path: Path,
    status_path: Path,
    row: pd.Series,
    expected_rows: int,
) -> pd.DataFrame | None:
    if not (csv_path.is_file() and status_path.is_file()):
        return None
    try:
        status = read_json(status_path)
        if status.get("complete") is not True or status.get("passed") is not True:
            return None
        if status.get("run_key") != shard_stem(row):
            return None
        if status.get("checkpoint_sha256") != str(row["model_checkpoint_sha256"]):
            return None
        if status.get("prediction_sha256") != sha256_file(csv_path):
            return None
        frame = pd.read_csv(csv_path)
    except Exception:
        return None
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "regime",
        "engine_id", "prefix_label", "registered_rul_anchor", "true_rul",
        "prediction", "error", "nasa_score_component", "checkpoint_sha256",
    }
    if len(frame) != expected_rows or not required <= set(frame.columns):
        return None
    identity = (
        set(frame["target_domain"].astype(str)) == {str(row["target_domain"])}
        and set(frame["model_seed"].astype(int)) == {int(row["model_seed"])}
        and set(frame["support_split_seed"].astype(int)) == {int(row["support_split_seed"])}
        and set(frame["shot"].astype(int)) == {int(row["shot"])}
        and set(frame["regime"].astype(str)) == {str(row["regime"])}
        and set(frame["checkpoint_sha256"].astype(str)) == {str(row["model_checkpoint_sha256"])}
    )
    if not identity or frame.duplicated(["engine_id", "prefix_label"]).any():
        return None
    numeric = frame[["true_rul", "prediction", "error", "nasa_score_component"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        return None
    return frame


def make_dataset(
    row: pd.Series,
    context: dict[str, Any],
    raw_frames: dict[str, pd.DataFrame],
) -> a232.CausalPrefixDataset:
    domain = str(row["target_domain"])
    if domain not in raw_frames:
        raw_frames[domain] = a23.load_domain_frame(
            Path(context["training_files"][domain]["path"]),
            rul_cap=float(context["cfg"]["rul_cap"]),
        )
    try:
        normalizer = a232.load_normalizer(normalizer_path(row), domain)
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    normalized = a23.normalize(raw_frames[domain], normalizer)
    engines = a232.confirmation_engines(
        context["roles"], domain, int(row["support_split_seed"])
    )
    return a232.CausalPrefixDataset(
        normalized, engines, int(context["cfg"]["window_size"])
    )


def run_inference(
    args: argparse.Namespace,
    preflight_result: dict[str, Any],
    context: dict[str, Any],
) -> pd.DataFrame:
    root: Path = context["output_root"]
    runs: pd.DataFrame = context["runs"]
    try:
        device = a232.resolve_device(args.device)
    except Exception as exc:
        raise A234Error(str(exc)) from exc
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(int(args.bootstrap_seed))
    np.random.seed(int(args.bootstrap_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.bootstrap_seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    raw_frames: dict[str, pd.DataFrame] = {}
    dataset_cache: dict[tuple[str, int, int], a232.CausalPrefixDataset] = {}
    frames: list[pd.DataFrame] = []
    skipped = 0
    for run_index, (_, row) in enumerate(runs.iterrows(), start=1):
        cache_key = (
            str(row["target_domain"]), int(row["model_seed"]),
            int(row["support_split_seed"]),
        )
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = make_dataset(row, context, raw_frames)
        dataset = dataset_cache[cache_key]
        csv_path, status_path = prediction_shard_paths(root, row)
        prior = validate_prediction_shard(csv_path, status_path, row, len(dataset)) if args.resume else None
        if prior is not None:
            frames.append(prior)
            skipped += 1
        else:
            try:
                records, uses_gat = a232.infer_checkpoint(
                    row,
                    context["input_root"],
                    context["cfg"],
                    dataset,
                    device,
                    context["protocol_hashes"],
                )
            except Exception as exc:
                raise A234Error(f"inference failed for {shard_stem(row)}: {exc}") from exc
            if bool(uses_gat) != bool(preflight_result["model_uses_gat"]):
                raise A234Error("model GAT mode changed after preflight")
            frame = pd.DataFrame(records)
            frame["checkpoint_sha256"] = str(row["model_checkpoint_sha256"])
            atomic_frame(csv_path, frame)
            atomic_json(status_path, {
                "experiment_id": EXPERIMENT_ID,
                "complete": True,
                "passed": True,
                "run_key": shard_stem(row),
                "checkpoint_sha256": str(row["model_checkpoint_sha256"]),
                "prediction_records": int(len(frame)),
                "prediction_sha256": sha256_file(csv_path),
                "new_predictor_training": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            })
            frames.append(frame)
        if run_index % 20 == 0 or run_index == len(runs):
            print(
                f"[A23.4] evaluated {run_index:04d}/{len(runs)} "
                f"reused_shards={skipped} device={device}",
                flush=True,
            )
    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != int(preflight_result["expected_prediction_records"]):
        raise A234Error(
            f"prediction count={len(predictions)}, "
            f"expected={preflight_result['expected_prediction_records']}"
        )
    keys = [
        "target_domain", "model_seed", "support_split_seed", "shot", "regime",
        "engine_id", "prefix_label",
    ]
    if predictions.duplicated(keys).any():
        raise A234Error("causal-anchor predictions contain duplicate keys")
    numeric = predictions[
        ["prediction", "true_rul", "error", "absolute_error", "squared_error", "nasa_score_component"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise A234Error("causal-anchor predictions contain non-finite values")
    return predictions.sort_values(keys, kind="stable").reset_index(drop=True)


def verify_source_invariance(predictions: pd.DataFrame) -> float:
    try:
        return float(a232.verify_source_only_invariance(predictions))
    except Exception as exc:
        raise A234Error(str(exc)) from exc


def build_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "regime", "prefix_label"]
    rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(keys, sort=True):
        errors = frame["error"].to_numpy(np.float64)
        rows.append({
            "experiment_id": EXPERIMENT_ID,
            **dict(zip(keys, key)),
            "registered_rul_anchor": float(frame["registered_rul_anchor"].iloc[0]),
            "rul_stage": str(frame["rul_stage"].iloc[0]),
            "n_engines": int(frame["engine_id"].nunique()),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "mean_error": float(np.mean(errors)),
            "nasa_score": float(frame["nasa_score_component"].sum()),
        })
    return pd.DataFrame(rows)


def build_paired_engines(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "target_domain", "model_seed", "support_split_seed", "shot",
        "engine_id", "prefix_label",
    ]
    columns = identifiers + [
        "registered_rul_anchor", "rul_stage", "true_rul", "prediction",
        "error", "absolute_error", "squared_error", "nasa_score_component",
    ]
    candidate = predictions.loc[predictions["regime"] == PRIMARY_CANDIDATE, columns].copy()
    reference = predictions.loc[predictions["regime"] == PRIMARY_REFERENCE, columns].copy()
    if candidate.duplicated(identifiers).any() or reference.duplicated(identifiers).any():
        raise A234Error("candidate/reference prediction keys are duplicated")
    paired = candidate.merge(
        reference, on=identifiers, how="inner", suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate) or len(reference) != len(candidate):
        raise A234Error("candidate/reference causal-anchor pairing is incomplete")
    if not np.allclose(paired["true_rul_candidate"], paired["true_rul_reference"], atol=0, rtol=0):
        raise A234Error("paired candidate/reference true RUL values differ")
    if not np.allclose(
        paired["registered_rul_anchor_candidate"],
        paired["registered_rul_anchor_reference"], atol=0, rtol=0,
    ):
        raise A234Error("paired candidate/reference RUL anchors differ")
    return paired


def relative_metric(frame: pd.DataFrame, metric: str) -> float:
    if metric == "rmse":
        candidate = math.sqrt(float(frame["squared_error_candidate"].mean()))
        reference = math.sqrt(float(frame["squared_error_reference"].mean()))
    elif metric == "nasa_score":
        candidate = float(frame["nasa_score_component_candidate"].sum())
        reference = float(frame["nasa_score_component_reference"].sum())
    else:
        raise A234Error(f"unknown primary metric: {metric}")
    if not (math.isfinite(candidate) and math.isfinite(reference) and reference > 0.0):
        raise A234Error(f"invalid {metric} candidate/reference aggregate")
    return candidate / reference - 1.0


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    nested: dict[str, dict[int, dict[int, pd.DataFrame]]] = {}
    for domain, domain_frame in frame.groupby("target_domain", sort=True):
        nested[str(domain)] = {}
        for model_seed, model_frame in domain_frame.groupby("model_seed", sort=True):
            nested[str(domain)][int(model_seed)] = {
                int(split): split_frame.reset_index(drop=True)
                for split, split_frame in model_frame.groupby("support_split_seed", sort=True)
            }
    if tuple(sorted(nested)) != DOMAINS:
        raise A234Error("bootstrap input does not contain the four target domains")
    for domain in DOMAINS:
        if tuple(sorted(nested[domain])) != MODEL_SEEDS:
            raise A234Error(f"bootstrap input has incomplete model seeds for {domain}")
        for model_seed in MODEL_SEEDS:
            if tuple(sorted(nested[domain][model_seed])) != SUPPORT_SPLIT_SEEDS:
                raise A234Error(
                    f"bootstrap input has incomplete support splits for {domain}/seed={model_seed}"
                )
    samples = np.empty(int(repetitions), dtype=np.float64)
    domains = np.asarray(DOMAINS, dtype=object)
    model_seeds = np.asarray(MODEL_SEEDS, dtype=np.int64)
    split_seeds = np.asarray(SUPPORT_SPLIT_SEEDS, dtype=np.int64)
    for repetition in range(int(repetitions)):
        parts: list[pd.DataFrame] = []
        for domain in rng.choice(domains, size=len(domains), replace=True):
            for model_seed in rng.choice(model_seeds, size=len(model_seeds), replace=True):
                for split in rng.choice(split_seeds, size=len(split_seeds), replace=True):
                    cluster = nested[str(domain)][int(model_seed)][int(split)]
                    indices = rng.integers(0, len(cluster), size=len(cluster))
                    parts.append(cluster.iloc[indices])
        samples[repetition] = relative_metric(pd.concat(parts, ignore_index=True), metric)
    if not np.isfinite(samples).all():
        raise A234Error("hierarchical bootstrap produced non-finite values")
    return samples


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise A234Error("invalid p-values supplied to Holm adjustment")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, float((len(values) - rank) * values[index]))
        running = max(running, current)
        adjusted[index] = running
    return adjusted.tolist()


def primary_inference(
    paired: pd.DataFrame,
    repetitions: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(RUL_ANCHORS):
        label = a232.endpoint_label(anchor)
        scoped = paired.loc[
            (paired["shot"] == PRIMARY_SHOT) & (paired["prefix_label"] == label)
        ].copy()
        expected_groups = len(DOMAINS) * len(MODEL_SEEDS) * len(SUPPORT_SPLIT_SEEDS)
        if scoped.groupby(["target_domain", "model_seed", "support_split_seed"]).ngroups != expected_groups:
            raise A234Error(f"primary K=5 pairing is incomplete for {label}")
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            point = relative_metric(scoped, metric)
            samples = hierarchical_bootstrap(
                scoped, metric, repetitions,
                int(bootstrap_seed) + anchor_index * 1000 + metric_index * 100,
            )
            tail_superiority = float((1 + np.count_nonzero(samples >= 0.0)) / (len(samples) + 1))
            tail_noninferiority = float(
                (1 + np.count_nonzero(samples >= SECONDARY_NONINFERIORITY_MARGIN))
                / (len(samples) + 1)
            )
            if metric == "rmse":
                candidate_value = math.sqrt(float(scoped["squared_error_candidate"].mean()))
                reference_value = math.sqrt(float(scoped["squared_error_reference"].mean()))
                wins = float(
                    np.mean(scoped["absolute_error_candidate"] < scoped["absolute_error_reference"])
                )
            else:
                candidate_value = float(scoped["nasa_score_component_candidate"].sum())
                reference_value = float(scoped["nasa_score_component_reference"].sum())
                wins = float(
                    np.mean(
                        scoped["nasa_score_component_candidate"]
                        < scoped["nasa_score_component_reference"]
                    )
                )
            rows.append({
                "experiment_id": EXPERIMENT_ID,
                "shot": PRIMARY_SHOT,
                "prefix_label": label,
                "registered_rul_anchor": float(anchor),
                "rul_stage": str(scoped["rul_stage_candidate"].iloc[0]),
                "metric": metric,
                "n_paired_engines": int(len(scoped)),
                "candidate_value": candidate_value,
                "reference_value": reference_value,
                "relative_degradation": point,
                "relative_improvement_pct": -100.0 * point,
                "relative_ci95_low": float(np.quantile(samples, ALPHA / 2.0)),
                "relative_ci95_high": float(np.quantile(samples, 1.0 - ALPHA / 2.0)),
                "candidate_engine_win_rate": wins,
                "one_sided_bootstrap_tail_probability_superiority": tail_superiority,
                "one_sided_bootstrap_tail_probability_noninferiority_3pct": tail_noninferiority,
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_design": (
                    "target_domain_then_model_seed_then_support_split_then_paired_engine"
                ),
            })
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_superiority"] = holm_adjust(
        result["one_sided_bootstrap_tail_probability_superiority"].tolist()
    )
    result["holm_superiority_passed"] = (
        (result["holm_adjusted_p_superiority"] < ALPHA)
        & (result["relative_ci95_high"] < 0.0)
    )
    result["holm_adjusted_p_noninferiority_3pct"] = holm_adjust(
        result["one_sided_bootstrap_tail_probability_noninferiority_3pct"].tolist()
    )
    result["holm_noninferiority_3pct_passed"] = (
        (result["holm_adjusted_p_noninferiority_3pct"] < ALPHA)
        & (result["relative_ci95_high"] <= SECONDARY_NONINFERIORITY_MARGIN)
    )
    return result


def secondary_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (shot, label), frame in paired.groupby(["shot", "prefix_label"], sort=True):
        anchor = float(frame["registered_rul_anchor_candidate"].iloc[0])
        stage = str(frame["rul_stage_candidate"].iloc[0])
        for metric in PRIMARY_METRICS:
            point = relative_metric(frame, metric)
            rows.append({
                "experiment_id": EXPERIMENT_ID,
                "analysis_role": "primary" if int(shot) == PRIMARY_SHOT else "secondary",
                "shot": int(shot),
                "prefix_label": str(label),
                "registered_rul_anchor": anchor,
                "rul_stage": stage,
                "metric": metric,
                "n_paired_engines": int(len(frame)),
                "relative_degradation": point,
                "relative_improvement_pct": -100.0 * point,
            })
    return pd.DataFrame(rows)


def domain_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = paired.loc[paired["shot"] == PRIMARY_SHOT]
    for (domain, label), frame in primary.groupby(["target_domain", "prefix_label"], sort=True):
        for metric in PRIMARY_METRICS:
            rows.append({
                "experiment_id": EXPERIMENT_ID,
                "target_domain": str(domain),
                "shot": PRIMARY_SHOT,
                "prefix_label": str(label),
                "registered_rul_anchor": float(frame["registered_rul_anchor_candidate"].iloc[0]),
                "metric": metric,
                "n_paired_engines": int(len(frame)),
                "relative_degradation": relative_metric(frame, metric),
                "relative_improvement_pct": -100.0 * relative_metric(frame, metric),
            })
    return pd.DataFrame(rows)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise A234Error(
                f"run lock exists: {self.path}; verify that no A23.4 process is active, "
                "then remove only this lock file before --resume"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired and self.path.exists():
            self.path.unlink()


def finalise(
    args: argparse.Namespace,
    preflight_result: dict[str, Any],
    context: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    root: Path = context["output_root"]
    source_range = verify_source_invariance(predictions)
    run_metrics = build_run_metrics(predictions)
    paired = build_paired_engines(predictions)
    primary = primary_inference(
        paired, int(args.bootstrap_repetitions), int(args.bootstrap_seed)
    )
    secondary = secondary_summary(paired)
    domains = domain_summary(paired)
    primary_passed = bool(primary["holm_superiority_passed"].all())
    noninferiority_passed = bool(primary["holm_noninferiority_3pct_passed"].all())
    domains_with_mean_improvement = int(
        domains.groupby("target_domain")["relative_degradation"]
        .apply(lambda values: bool((values < 0.0).all()))
        .sum()
    )

    artifact_frames = {
        "experimentA23_4_causal_anchor_predictions.csv": predictions,
        "experimentA23_4_run_level_metrics.csv": run_metrics,
        "experimentA23_4_paired_engine_metrics.csv": paired,
        "experimentA23_4_primary_hierarchical_inference.csv": primary,
        "experimentA23_4_secondary_shot_summary.csv": secondary,
        "experimentA23_4_domain_summary.csv": domains,
        "experimentA23_4_prefix_coverage.csv": context["coverage"],
        "experimentA23_4_checkpoint_inventory.csv": pd.DataFrame(context["checkpoint_rows"]),
    }
    for name, frame in artifact_frames.items():
        atomic_frame(root / name, frame)

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": preflight_result["registered_primary_question"],
        "complete": True,
        "passed": primary_passed,
        "evaluation_only": True,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "candidate": PRIMARY_CANDIDATE,
        "reference": PRIMARY_REFERENCE,
        "primary_shot": PRIMARY_SHOT,
        "registered_rul_anchors": list(RUL_ANCHORS),
        "expected_checkpoint_runs": int(preflight_result["expected_checkpoint_runs"]),
        "completed_checkpoint_evaluations": int(context["runs"].shape[0]),
        "expected_prediction_records": int(preflight_result["expected_prediction_records"]),
        "completed_prediction_records": int(len(predictions)),
        "expected_primary_checks": len(RUL_ANCHORS) * len(PRIMARY_METRICS),
        "completed_primary_checks": int(len(primary)),
        "primary_decision_rule": "all_six_holm_superiority_checks_pass",
        "primary_checks_passed": int(primary["holm_superiority_passed"].sum()),
        "all_primary_holm_superiority_checks_passed": primary_passed,
        "all_primary_holm_noninferiority_3pct_checks_passed": noninferiority_passed,
        "primary_target_domains_with_improvement_on_all_six_checks": domains_with_mean_improvement,
        "source_only_prediction_invariance_max_abs_range": source_range,
        "source_only_prediction_invariance_tolerance": SOURCE_INVARIANCE_TOLERANCE,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_design": preflight_result["bootstrap_design"],
        "model_uses_gat": bool(preflight_result["model_uses_gat"]),
        "mixed_registered_endpoints_within_inference_batch": False,
        "model_input_uses_future_cycles": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A23.4 confirmed strict K=5 PFT superiority across all registered causal anchors "
            "and metrics after hierarchical bootstrap and Holm correction"
            if primary_passed else
            "A23.4 completed the registered formal causal-anchor evaluation, but K=5 PFT did "
            "not pass every Holm-corrected superiority check"
        ),
        "interpretation_limit": (
            "A23.4 compares ordinary pretraining-plus-fine-tuning with scratch training on "
            "training-file confirmation engines. It does not evaluate meta-learning, isolate "
            "a GNN increment, diagnose fault classes, or create an official-test claim."
        ),
        "next_action": (
            "implement_A24_matched_Meta_noGraph_and_Meta_GNN_under_the_unchanged_A23_contract"
            if primary_passed else
            "report_A23_4_boundary_before_deciding_whether_to_proceed_to_A24"
        ),
    }
    atomic_json(root / "experimentA23_4_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "registered_analysis": {
            "primary_shot": PRIMARY_SHOT,
            "rul_anchors": list(RUL_ANCHORS),
            "metrics": list(PRIMARY_METRICS),
            "alpha": ALPHA,
            "decision_rule": "all_six_holm_superiority_checks_pass",
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_design": preflight_result["bootstrap_design"],
            "secondary_noninferiority_margin": SECONDARY_NONINFERIORITY_MARGIN,
        },
        "inputs": {
            "a23_3_decision_sha256": sha256_file(
                context["input_root"] / "experimentA23_3_confirmation_decision.json"
            ),
            "a23_3_manifest_sha256": sha256_file(
                context["input_root"] / "experimentA23_3_manifest.json"
            ),
            "a23_3_run_level_sha256": sha256_file(
                context["input_root"] / "experimentA23_3_training_run_level.csv"
            ),
            "protocol_hashes": context["protocol_hashes"],
            "config_sha256": preflight_result["config_sha256"],
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {
            name: sha256_file(root / name)
            for name in sorted(
                list(artifact_frames)
                + [
                    "experimentA23_4_preflight.json",
                    "experimentA23_4_confirmation_decision.json",
                ]
            )
        },
        "prediction_shards_excluded_from_final_hash_manifest": True,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA23_4_manifest.json", manifest)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolve(args.output_dir)
    decision_path = output_root / "experimentA23_4_confirmation_decision.json"
    if decision_path.is_file():
        prior = read_json(decision_path)
        if args.resume and prior.get("complete") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A23.4] resume: existing complete decision returned", flush=True)
            return 0
        raise A234Error(
            f"output already contains a completed decision: {decision_path}; "
            "use --resume to return it or select a new --output-dir"
        )

    preflight_result, context = preflight(args)
    print(json.dumps(preflight_result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print(
            "[A23.4] dry-run passed: 1500 checkpoints and all data/config/protocol contracts "
            "were validated; no predictor was trained and no official test file was accessed",
            flush=True,
        )
        return 0

    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise A234Error(
            f"non-empty output directory has no completed decision: {output_root}; "
            "use --resume only for A23.4 shards, or choose a new output directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with RunLock(output_root / "experimentA23_4_run.lock"):
        atomic_json(output_root / "experimentA23_4_preflight.json", preflight_result)
        predictions = run_inference(args, preflight_result, context)
        decision = finalise(args, preflight_result, context, predictions)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (A234Error, a232.A232Error, a23.A23Error) as exc:
        print(f"[A23.4] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
