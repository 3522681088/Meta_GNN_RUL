#!/usr/bin/env python3
"""Experiment A23.2: causal-prefix endpoint audit of the A23.1 pilot.

This is an evaluation-only experiment.  It reuses every target-adapted
checkpoint from A23.1 and evaluates the locked confirmation engines at three
retrospectively sampled, causal input endpoints anchored at true RUL 90, 45,
and 15.  The true-RUL anchor is used only to define an offline benchmark
endpoint; the model input contains no observation after the selected cycle.

The experiment intentionally does not train, adapt, select, or tune a model.
It never resolves or opens C-MAPSS test files.  Its primary decision is an
integrity/coverage decision, not an efficacy claim.
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
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402


EXPERIMENT_ID = "experimentA23_2"
SCRIPT_VERSION = "experimentA23_2_causal_prefix_endpoint_audit_v2"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
REGIMES = ("source_only", "scratch_k", "pretrain_finetune_k")
REGISTERED_SHOTS = (1, 5, 20)
REGISTERED_RUL_ANCHORS = (90.0, 45.0, 15.0)
HIGH_RUL_THRESHOLD = 60.0
LOW_RUL_THRESHOLD = 30.0
SOURCE_INVARIANCE_TOLERANCE = 1e-6


class A232Error(RuntimeError):
    """Raised when an A23.2 input, integrity, or inference check fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A23.2 evaluation-only causal-prefix endpoint audit"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--a23-1-output-dir",
        type=Path,
        default=Path("outputs/experimentA23_1_few_shot_transfer_baselines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA23_2_causal_prefix_endpoint_audit"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="CPU is the registered default and avoids shared-GPU/OOM failures.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Defaults to the unchanged batch_size in configs/default.yaml.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Return an already complete A23.2 decision without overwriting artifacts.",
    )
    args = parser.parse_args(argv)
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        raise A232Error("--inference-batch-size must be positive")
    if args.torch_threads <= 0:
        raise A232Error("--torch-threads must be positive")
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
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
        raise A232Error(f"required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A232Error(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A232Error(f"JSON artifact must contain an object: {path}")
    return payload


def strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise A232Error(f"field {field!r} is not a strict boolean: {value!r}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise A232Error("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    path = resolve(args.config)
    if not path.is_file():
        raise A232Error(f"model configuration is missing: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A232Error(f"failed to parse model configuration {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise A232Error(f"model configuration must be a mapping: {path}")
    cfg = dict(cfg)
    cfg["batch_size"] = int(cfg.get("batch_size", 64))
    cfg["window_size"] = int(cfg.get("window_size", cfg.get("seq_len", 30)))
    cfg["rul_cap"] = float(cfg.get("rul_cap", cfg.get("max_rul", 125.0)))
    if args.inference_batch_size is not None:
        cfg["batch_size"] = int(args.inference_batch_size)
    if cfg["batch_size"] <= 0 or cfg["window_size"] < 2 or cfg["rul_cap"] <= 0:
        raise A232Error("configuration contains invalid batch_size, window_size, or rul_cap")
    return cfg, path


def parse_feature_columns(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise A232Error(f"feature_columns is not valid JSON: {value!r}") from exc
    if not isinstance(value, (list, tuple)):
        raise A232Error("feature_columns must be a list")
    return tuple(str(item) for item in value)


def load_a231_contract(input_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    decision_path = input_root / "experimentA23_1_confirmation_decision.json"
    manifest_path = input_root / "experimentA23_1_manifest.json"
    run_path = input_root / "experimentA23_1_confirmation_run_level.csv"
    decision = read_json(decision_path)
    manifest = read_json(manifest_path)
    if decision.get("experiment_id") != "experimentA23_1":
        raise A232Error("input decision is not experimentA23_1")
    if not (decision.get("complete") and decision.get("passed") and decision.get("pilot_only")):
        raise A232Error("A23.1 must be complete, passed, and marked pilot_only")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A232Error(f"A23.1 violates the training-only boundary: {field}=true")
    if not run_path.is_file():
        raise A232Error(f"A23.1 run-level artifact is missing: {run_path}")
    expected_hash = manifest.get("artifacts", {}).get(run_path.name)
    actual_hash = sha256_file(run_path)
    if not expected_hash or expected_hash != actual_hash:
        raise A232Error(
            f"A23.1 run-level SHA-256 mismatch: expected={expected_hash}, actual={actual_hash}"
        )
    try:
        runs = pd.read_csv(run_path)
    except Exception as exc:
        raise A232Error(f"failed to parse A23.1 run-level CSV: {exc}") from exc
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "regime",
        "feature_count", "feature_columns", "window_size", "rul_cap",
        "target_epochs", "source_pretrain_steps", "model_checkpoint",
        "confirmation_used_for_training", "confirmation_used_for_normalizer_fit",
        "selection_used_for_training", "selection_used_for_epoch_selection",
        "official_test_files_accessed", "official_test_forward_run",
    }
    if missing := required - set(runs.columns):
        raise A232Error(f"A23.1 run-level CSV lacks columns: {sorted(missing)}")
    if len(runs) != int(decision.get("completed_run_records", -1)):
        raise A232Error("A23.1 decision/run-level record counts disagree")
    if len(runs) != 72:
        raise A232Error(f"registered A23.1 pilot requires 72 runs, found {len(runs)}")
    for column in (
        "confirmation_used_for_training",
        "confirmation_used_for_normalizer_fit",
        "selection_used_for_training",
        "selection_used_for_epoch_selection",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        parsed = [strict_bool(value, field=column) for value in runs[column].tolist()]
        if any(parsed):
            raise A232Error(f"A23.1 run-level integrity violation: {column}=true")
    runs["model_seed"] = pd.to_numeric(runs["model_seed"], errors="raise").astype(int)
    runs["support_split_seed"] = pd.to_numeric(
        runs["support_split_seed"], errors="raise"
    ).astype(int)
    runs["shot"] = pd.to_numeric(runs["shot"], errors="raise").astype(int)
    if tuple(sorted(runs["target_domain"].unique())) != DOMAINS:
        raise A232Error("A23.1 target-domain set differs from the registered four domains")
    if tuple(sorted(runs["shot"].unique())) != REGISTERED_SHOTS:
        raise A232Error("A23.1 shot set differs from registered shots 1/5/20")
    if tuple(sorted(runs["regime"].unique())) != tuple(sorted(REGIMES)):
        raise A232Error("A23.1 regime set differs from the three registered baselines")
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "regime"]
    if runs.duplicated(keys).any():
        duplicates = runs.loc[runs.duplicated(keys, keep=False), keys].to_dict("records")
        raise A232Error(f"A23.1 contains duplicate run keys: {duplicates[:10]}")
    seeds = tuple(sorted(runs["model_seed"].unique()))
    splits = tuple(sorted(runs["support_split_seed"].unique()))
    expected = {
        (domain, seed, split, shot, regime)
        for domain in DOMAINS
        for seed in seeds
        for split in splits
        for shot in REGISTERED_SHOTS
        for regime in REGIMES
    }
    actual = set(runs[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise A232Error(
            f"A23.1 factorial run grid is incomplete; missing={sorted(expected-actual)[:10]}, "
            f"unexpected={sorted(actual-expected)[:10]}"
        )
    return runs.sort_values(keys, kind="stable").reset_index(drop=True), decision, manifest


def load_protocol_contract(
    protocol_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    protocol_path = protocol_root / "experimentA23_few_shot_protocol.json"
    roles_path = protocol_root / "experimentA23_engine_roles.csv"
    decision_path = protocol_root / "experimentA23_confirmation_decision.json"
    protocol = read_json(protocol_path)
    decision = read_json(decision_path)
    if not (decision.get("complete") and decision.get("passed")):
        raise A232Error("A23.0 protocol must be complete and passed")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A232Error(f"A23.0 violates the official-test boundary: {field}=true")
    try:
        roles = pd.read_csv(roles_path)
    except Exception as exc:
        raise A232Error(f"failed to parse A23.0 engine roles: {exc}") from exc
    required = {"target_domain", "support_split_seed", "engine_id", "role", "support_rank"}
    if missing := required - set(roles.columns):
        raise A232Error(f"A23.0 engine roles lack columns: {sorted(missing)}")
    roles["support_split_seed"] = pd.to_numeric(
        roles["support_split_seed"], errors="raise"
    ).astype(int)
    roles["engine_id"] = pd.to_numeric(roles["engine_id"], errors="raise").astype(int)
    allowed_roles = {"support_pool", "selection", "confirmation"}
    if unknown := set(roles["role"]) - allowed_roles:
        raise A232Error(f"A23.0 contains unknown engine roles: {sorted(unknown)}")
    digests = {
        protocol_path.name: sha256_file(protocol_path),
        roles_path.name: sha256_file(roles_path),
        decision_path.name: sha256_file(decision_path),
    }
    return protocol, roles, digests


def verify_training_files(data_root: Path, protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    inventory = protocol.get("training_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise A232Error("A23.0 protocol lacks training_file_inventory")
    locked: dict[str, str] = {}
    for item in inventory:
        if not isinstance(item, dict) or "domain" not in item or "sha256" not in item:
            raise A232Error("malformed A23.0 training_file_inventory entry")
        locked[str(item["domain"])] = str(item["sha256"])
    if set(locked) != set(DOMAINS):
        raise A232Error(f"A23.0 training-file domains differ from {DOMAINS}: {sorted(locked)}")
    result: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        path = a23.resolve_train_file(data_root, domain)
        digest = sha256_file(path)
        if digest != locked[domain]:
            raise A232Error(
                f"training file changed after A23.0 for {domain}: "
                f"expected={locked[domain]}, actual={digest}, path={path}"
            )
        result[domain] = {"path": str(path), "sha256": digest}
    return result


def confirmation_engines(
    roles: pd.DataFrame, domain: str, split_seed: int
) -> list[int]:
    frame = roles.loc[
        (roles["target_domain"] == domain)
        & (roles["support_split_seed"] == int(split_seed))
        & (roles["role"] == "confirmation")
    ]
    engines = sorted(int(value) for value in frame["engine_id"].tolist())
    if not engines or len(engines) != len(set(engines)):
        raise A232Error(
            f"confirmation engine set is empty or duplicated for {domain}/split={split_seed}"
        )
    return engines


def stage_for_rul(value: float) -> str:
    if value > HIGH_RUL_THRESHOLD:
        return "high_rul_gt60"
    if value > LOW_RUL_THRESHOLD:
        return "mid_rul_31_to_60"
    return "low_rul_le30"


def endpoint_label(rul_anchor: float) -> str:
    return f"rul_anchor_{int(round(rul_anchor)):03d}"


class CausalPrefixDataset(Dataset):
    """One strictly past-only window per registered engine/prefix endpoint."""

    def __init__(
        self,
        frame: pd.DataFrame,
        engine_ids: Sequence[int],
        window_size: int,
    ) -> None:
        self.x: list[np.ndarray] = []
        self.meta: list[dict[str, Any]] = []
        wanted = set(int(value) for value in engine_ids)
        available = set(int(value) for value in frame["unit"].unique())
        if missing := wanted - available:
            raise A232Error(f"confirmation engines are absent from target data: {sorted(missing)}")
        for unit in sorted(wanted):
            group = frame.loc[frame["unit"] == unit].sort_values("cycle", kind="stable")
            n_cycles = len(group)
            if n_cycles < 4:
                raise A232Error(
                    f"unit={unit} has only {n_cycles} cycles; three unique registered prefixes are impossible"
                )
            raw_rul = group["raw_rul"].to_numpy(dtype=np.float64, copy=True)
            endpoint_indices: list[int] = []
            for rul_anchor in REGISTERED_RUL_ANCHORS:
                matches = np.flatnonzero(
                    np.isclose(raw_rul, float(rul_anchor), atol=0.0, rtol=0.0)
                )
                if len(matches) != 1:
                    raise A232Error(
                        f"unit={unit} does not contain exactly one offline RUL anchor "
                        f"{rul_anchor:g}; matches={matches.tolist()}, max_raw_rul={raw_rul.max():g}"
                    )
                endpoint_indices.append(int(matches[0]))
            if len(set(endpoint_indices)) != len(REGISTERED_RUL_ANCHORS):
                raise A232Error(
                    f"registered RUL anchors collapse to duplicate cycles for unit={unit}: "
                    f"{endpoint_indices}"
                )
            features = group.loc[:, a23.FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            for rul_anchor, endpoint in zip(REGISTERED_RUL_ANCHORS, endpoint_indices):
                start = max(0, endpoint - window_size + 1)
                segment = features[start : endpoint + 1]
                if len(segment) < window_size:
                    padding = np.repeat(segment[:1], window_size - len(segment), axis=0)
                    segment = np.concatenate([padding, segment], axis=0)
                row = group.iloc[endpoint]
                truth = float(row["rul"])
                self.x.append(segment)
                self.meta.append(
                    {
                        "engine_id": int(unit),
                        "cycle": int(row["cycle"]),
                        "trajectory_cycles": int(n_cycles),
                        "prefix_fraction": float((endpoint + 1) / n_cycles),
                        "prefix_label": endpoint_label(rul_anchor),
                        "registered_rul_anchor": float(rul_anchor),
                        "true_rul": truth,
                        "raw_true_rul": float(row["raw_rul"]),
                        "rul_stage": stage_for_rul(truth),
                        "input_uses_future_cycles": False,
                    }
                )
        if len(self.x) != len(wanted) * len(REGISTERED_RUL_ANCHORS):
            raise A232Error("causal-prefix dataset cardinality mismatch")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.x[index]), torch.tensor(index, dtype=torch.int64)


def deterministic_loader(dataset: Dataset, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def worker_directory(input_root: Path, domain: str, seed: int, split_seed: int) -> Path:
    return input_root / "shards" / f"{domain}_mseed{seed}_split{split_seed}"


def checkpoint_path(input_root: Path, row: pd.Series) -> Path:
    directory = worker_directory(
        input_root,
        str(row["target_domain"]),
        int(row["model_seed"]),
        int(row["support_split_seed"]),
    )
    return directory / f"{row['regime']}_shot{int(row['shot'])}_target_adapted.pt"


def normalizer_path(input_root: Path, row: pd.Series) -> Path:
    return worker_directory(
        input_root,
        str(row["target_domain"]),
        int(row["model_seed"]),
        int(row["support_split_seed"]),
    ) / "source_normalizer.json"


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A232Error(f"A23.1 checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise A232Error(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise A232Error(f"checkpoint lacks a state dictionary: {path}")
    for name, tensor in payload["state"].items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise A232Error(f"invalid state entry in {path}: {name!r}")
        if not torch.isfinite(tensor).all().item():
            raise A232Error(f"non-finite checkpoint tensor in {path}: {name}")
    return payload


def validate_checkpoint_metadata(
    payload: dict[str, Any],
    row: pd.Series,
    path: Path,
    protocol_hashes: dict[str, str],
) -> None:
    expected = {
        "target_domain": str(row["target_domain"]),
        "model_seed": int(row["model_seed"]),
        "support_split_seed": int(row["support_split_seed"]),
        "shot": int(row["shot"]),
        "regime": str(row["regime"]),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise A232Error(
                f"checkpoint metadata mismatch in {path}: {field}={payload.get(field)!r}, expected={value!r}"
            )
    if tuple(payload.get("feature_columns", ())) != tuple(a23.FEATURE_COLUMNS):
        raise A232Error(f"checkpoint feature columns differ from A23.1 code: {path}")
    if payload.get("protocol_hashes") != protocol_hashes:
        raise A232Error(f"checkpoint protocol hashes do not match current A23.0 artifacts: {path}")
    expected_epochs = 0 if str(row["regime"]) == "source_only" else int(row["target_epochs"])
    if int(payload.get("target_epochs", -1)) != expected_epochs:
        raise A232Error(f"checkpoint target_epochs mismatch: {path}")


def validate_model_state(
    state: dict[str, torch.Tensor], cfg: dict[str, Any], path: Path
) -> bool:
    try:
        model = a23.build_model("gnn", len(a23.FEATURE_COLUMNS), cfg)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise A232Error(
            f"checkpoint is incompatible with current --config/model code: {path}: {exc}"
        ) from exc
    uses_gat = bool(getattr(model, "use_gat", False))
    del model
    return uses_gat


def load_normalizer(path: Path, expected_target: str) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    if tuple(payload.get("feature_columns", ())) != tuple(a23.FEATURE_COLUMNS):
        raise A232Error(f"normalizer feature columns mismatch: {path}")
    expected_sources = sorted(domain for domain in DOMAINS if domain != expected_target)
    if sorted(payload.get("fitted_domains", ())) != expected_sources:
        raise A232Error(f"normalizer source-domain set mismatch: {path}")
    if payload.get("target_domain_used_for_fit") is not False:
        raise A232Error(f"normalizer used target domain: {path}")
    if payload.get("confirmation_engines_used_for_fit") is not False:
        raise A232Error(f"normalizer used confirmation engines: {path}")
    state = payload.get("normalizer")
    if not isinstance(state, dict) or set(state) != {"mean", "std"}:
        raise A232Error(f"malformed normalizer state: {path}")
    for section in ("mean", "std"):
        if set(state[section]) != set(a23.FEATURE_COLUMNS):
            raise A232Error(f"normalizer {section} columns mismatch: {path}")
        for feature, value in state[section].items():
            if not math.isfinite(float(value)):
                raise A232Error(f"non-finite normalizer value {section}/{feature}: {path}")
    if any(float(value) <= 0.0 for value in state["std"].values()):
        raise A232Error(f"normalizer contains non-positive standard deviation: {path}")
    return state


def preflight(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str], dict[str, Any]
]:
    input_root = resolve(args.a23_1_output_dir)
    protocol_root = resolve(args.protocol_dir)
    data_root = resolve(args.data_dir)
    output_root = resolve(args.output_dir)
    if input_root == output_root:
        raise A232Error("--output-dir must differ from --a23-1-output-dir")
    runs, a231_decision, a231_manifest = load_a231_contract(input_root)
    protocol, roles, protocol_hashes = load_protocol_contract(protocol_root)
    cfg, config_path = load_config(args)
    training_files = verify_training_files(data_root, protocol)

    if any(int(value) != int(cfg["window_size"]) for value in runs["window_size"]):
        raise A232Error(
            "current --config window_size differs from A23.1; use the unchanged A23.1 config"
        )
    if any(not math.isclose(float(value), float(cfg["rul_cap"]), abs_tol=1e-12) for value in runs["rul_cap"]):
        raise A232Error("current --config rul_cap differs from A23.1")
    for value in runs["feature_columns"]:
        if parse_feature_columns(value) != tuple(a23.FEATURE_COLUMNS):
            raise A232Error("A23.1 feature columns differ from the evaluator contract")
    if any(int(value) != len(a23.FEATURE_COLUMNS) for value in runs["feature_count"]):
        raise A232Error("A23.1 feature_count differs from the evaluator contract")

    checkpoint_rows: list[dict[str, Any]] = []
    uses_gat_values: set[bool] = set()
    for _, row in runs.iterrows():
        path = checkpoint_path(input_root, row)
        payload = safe_load_checkpoint(path)
        validate_checkpoint_metadata(payload, row, path, protocol_hashes)
        uses_gat_values.add(validate_model_state(payload["state"], cfg, path))
        npath = normalizer_path(input_root, row)
        load_normalizer(npath, str(row["target_domain"]))
        checkpoint_rows.append(
            {
                "target_domain": str(row["target_domain"]),
                "model_seed": int(row["model_seed"]),
                "support_split_seed": int(row["support_split_seed"]),
                "shot": int(row["shot"]),
                "regime": str(row["regime"]),
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "normalizer_path": str(npath),
                "normalizer_sha256": sha256_file(npath),
            }
        )
        del payload
    if len(uses_gat_values) != 1:
        raise A232Error("A23.1 checkpoints disagree on whether GAT is enabled")

    raw_frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    unique_prefix_count = 0
    stage_counts = {
        "high_rul_gt60": 0,
        "mid_rul_31_to_60": 0,
        "low_rul_le30": 0,
    }
    unique_domain_splits = sorted(
        set(
            (str(row.target_domain), int(row.support_split_seed))
            for row in runs[["target_domain", "support_split_seed"]].itertuples(index=False)
        )
    )
    for domain, split_seed in unique_domain_splits:
        if domain not in raw_frames:
            raw_frames[domain] = a23.load_domain_frame(
                Path(training_files[domain]["path"]), rul_cap=float(cfg["rul_cap"])
            )
        engines = confirmation_engines(roles, domain, split_seed)
        # Coverage does not depend on feature normalization.  Use an identity
        # feature transform so the same endpoint constructor validates cycles.
        identity = {
            "mean": {column: 0.0 for column in a23.FEATURE_COLUMNS},
            "std": {column: 1.0 for column in a23.FEATURE_COLUMNS},
        }
        dataset = CausalPrefixDataset(
            a23.normalize(raw_frames[domain], identity), engines, int(cfg["window_size"])
        )
        unique_prefix_count += len(dataset)
        for item in dataset.meta:
            stage_counts[item["rul_stage"]] += 1
            coverage_rows.append(
                {
                    "target_domain": domain,
                    "support_split_seed": split_seed,
                    **item,
                }
            )
    if any(value <= 0 for value in stage_counts.values()):
        raise A232Error(f"registered prefix endpoints do not cover all RUL stages: {stage_counts}")

    seeds = sorted(int(value) for value in runs["model_seed"].unique())
    splits = sorted(int(value) for value in runs["support_split_seed"].unique())
    expected_predictions = 0
    for domain in DOMAINS:
        for split in splits:
            n_engines = len(confirmation_engines(roles, domain, split))
            expected_predictions += (
                n_engines
                * len(REGISTERED_RUL_ANCHORS)
                * len(seeds)
                * len(REGISTERED_SHOTS)
                * len(REGIMES)
            )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": (
            "Can the completed A23.1 pilot checkpoints be evaluated without retraining at "
            "pre-registered causal RUL-anchor endpoints that cover high, mid, and low RUL?"
        ),
        "a23_1_input_dir": str(input_root),
        "protocol_dir": str(protocol_root),
        "output_dir": str(output_root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "registered_rul_anchors": list(REGISTERED_RUL_ANCHORS),
        "rul_stages": {
            "high_rul_gt60": "> 60",
            "mid_rul_31_to_60": "30 < RUL <= 60",
            "low_rul_le30": "<= 30",
        },
        "expected_checkpoint_runs": int(len(runs)),
        "validated_checkpoint_runs": int(len(checkpoint_rows)),
        "unique_confirmation_engine_prefixes": int(unique_prefix_count),
        "expected_prediction_records": int(expected_predictions),
        "coverage_by_rul_stage": stage_counts,
        "model_uses_gat": bool(next(iter(uses_gat_values))),
        "mixed_registered_endpoints_within_inference_batch": False,
        "rul_anchor_uses_complete_trajectory_label_only_for_offline_endpoint_definition": True,
        "model_input_uses_future_cycles": False,
        "inference_batch_size": int(cfg["batch_size"]),
        "checkpoint_schema_and_model_compatibility_passed": True,
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
        "training_files": training_files,
        "checkpoint_rows": checkpoint_rows,
        "coverage_rows": coverage_rows,
        "a231_decision": a231_decision,
        "a231_manifest": a231_manifest,
    }
    return result, runs, roles, cfg, protocol_hashes, context


@torch.no_grad()
def infer_checkpoint(
    row: pd.Series,
    input_root: Path,
    cfg: dict[str, Any],
    dataset: CausalPrefixDataset,
    device: torch.device,
    protocol_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    path = checkpoint_path(input_root, row)
    payload = safe_load_checkpoint(path)
    validate_checkpoint_metadata(payload, row, path, protocol_hashes)
    model = a23.build_model("gnn", len(a23.FEATURE_COLUMNS), cfg)
    try:
        model.load_state_dict(payload["state"], strict=True)
    except Exception as exc:
        raise A232Error(f"strict state load failed during inference for {path}: {exc}") from exc
    model = model.to(device)
    model.eval()
    predictions = np.empty(len(dataset), dtype=np.float64)
    seen = np.zeros(len(dataset), dtype=bool)
    # A GAT model may exchange information among samples in one batch.  Never
    # put an engine's RUL=90, 45, and 15 endpoints in the same graph/batch:
    # doing so would let an earlier endpoint depend on a later endpoint.  Each
    # registered RUL anchor is inferred separately and in stable engine order.
    for rul_anchor in REGISTERED_RUL_ANCHORS:
        label = endpoint_label(rul_anchor)
        subset_indices = [
            index
            for index, meta in enumerate(dataset.meta)
            if str(meta["prefix_label"]) == label
        ]
        if len(subset_indices) * len(REGISTERED_RUL_ANCHORS) != len(dataset):
            raise A232Error(
                f"endpoint subset cardinality mismatch for RUL anchor={rul_anchor:g}: {path}"
            )
        loader = deterministic_loader(
            Subset(dataset, subset_indices), int(cfg["batch_size"]), device
        )
        for x, indices in loader:
            try:
                output = model(x.to(device, non_blocking=device.type == "cuda"))
            except Exception as exc:
                raise A232Error(
                    f"model forward failed for {path}, RUL anchor={rul_anchor:g}: {exc}"
                ) from exc
            if isinstance(output, tuple):
                output = output[0]
            values = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
            locations = indices.numpy().reshape(-1).astype(int)
            if len(values) != len(locations):
                raise A232Error(f"prediction cardinality mismatch for {path}")
            predictions[locations] = values
            seen[locations] = True
    if not seen.all() or not np.isfinite(predictions).all():
        raise A232Error(f"missing or non-finite predictions for {path}")
    uses_gat = bool(getattr(model, "use_gat", False))
    records: list[dict[str, Any]] = []
    for index, (prediction, meta) in enumerate(zip(predictions, dataset.meta)):
        truth = float(meta["true_rul"])
        error = float(prediction - truth)
        nasa_component = float(math.exp(error / 10.0) - 1.0) if error >= 0 else float(
            math.exp(-error / 13.0) - 1.0
        )
        records.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": str(row["target_domain"]),
                "model_seed": int(row["model_seed"]),
                "support_split_seed": int(row["support_split_seed"]),
                "shot": int(row["shot"]),
                "regime": str(row["regime"]),
                "checkpoint_path": str(path),
                "endpoint_index_in_dataset": int(index),
                **meta,
                "prediction": float(prediction),
                "error": error,
                "absolute_error": abs(error),
                "squared_error": error * error,
                "nasa_score_component": nasa_component,
                "model_uses_gat": uses_gat,
                "mixed_registered_endpoints_within_inference_batch": False,
                "confirmation_used_for_training": False,
                "new_predictor_training": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
        )
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, uses_gat


def metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "n": 0,
            "n_engines": 0,
            "rmse": None,
            "mae": None,
            "mean_error": None,
            "nasa_score": None,
        }
    errors = frame["error"].to_numpy(dtype=np.float64)
    return {
        "n": int(len(frame)),
        "n_engines": int(frame["engine_id"].nunique()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
        "nasa_score": float(frame["nasa_score_component"].sum()),
    }


def evaluation_scopes() -> list[tuple[str, str, Any]]:
    scopes: list[tuple[str, str, Any]] = [("all_prefixes", "all", None)]
    scopes.extend(
        (endpoint_label(value), "endpoint", endpoint_label(value))
        for value in REGISTERED_RUL_ANCHORS
    )
    scopes.extend(
        (stage, "stage", stage)
        for stage in ("high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30")
    )
    return scopes


def select_scope(frame: pd.DataFrame, kind: str, value: Any) -> pd.DataFrame:
    if kind == "all":
        return frame
    if kind == "endpoint":
        return frame.loc[frame["prefix_label"] == value]
    if kind == "stage":
        return frame.loc[frame["rul_stage"] == value]
    raise A232Error(f"unknown evaluation scope kind: {kind}")


def build_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "regime"]
    rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(keys, sort=True):
        identity = dict(zip(keys, key))
        for scope, kind, value in evaluation_scopes():
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    **identity,
                    "evaluation_scope": scope,
                    **metric_values(select_scope(frame, kind, value)),
                }
            )
    return pd.DataFrame(rows)


def build_paired_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["target_domain", "model_seed", "support_split_seed", "shot"]
    pair_keys = ["engine_id", "prefix_label"]
    rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(base_keys, sort=True):
        identity = dict(zip(base_keys, key))
        candidate = frame.loc[frame["regime"] == "pretrain_finetune_k"].copy()
        reference = frame.loc[frame["regime"] == "scratch_k"].copy()
        if candidate.duplicated(pair_keys).any() or reference.duplicated(pair_keys).any():
            raise A232Error(f"duplicate paired prediction keys for {identity}")
        merged = candidate.merge(
            reference,
            on=pair_keys,
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        expected = len(candidate)
        if len(merged) != expected or len(reference) != expected:
            raise A232Error(f"candidate/reference pairing is incomplete for {identity}")
        if not np.allclose(
            merged["true_rul_candidate"], merged["true_rul_reference"], atol=0.0, rtol=0.0
        ):
            raise A232Error(f"paired true RUL mismatch for {identity}")
        normalized = pd.DataFrame(
            {
                "engine_id": merged["engine_id"],
                "prefix_label": merged["prefix_label"],
                "prefix_fraction": merged["prefix_fraction_candidate"],
                "registered_rul_anchor": merged["registered_rul_anchor_candidate"],
                "rul_stage": merged["rul_stage_candidate"],
                "error_candidate": merged["error_candidate"],
                "error_reference": merged["error_reference"],
                "nasa_candidate": merged["nasa_score_component_candidate"],
                "nasa_reference": merged["nasa_score_component_reference"],
            }
        )
        for scope, kind, value in evaluation_scopes():
            scoped = select_scope(normalized, kind, value)
            if scoped.empty:
                row = {
                    "n_pairs": 0,
                    "candidate_rmse": None,
                    "reference_rmse": None,
                    "rmse_delta_candidate_minus_reference": None,
                    "candidate_nasa_score": None,
                    "reference_nasa_score": None,
                    "nasa_delta_candidate_minus_reference": None,
                    "candidate_rmse_win_rate": None,
                    "available": False,
                }
            else:
                ce = scoped["error_candidate"].to_numpy(dtype=np.float64)
                re = scoped["error_reference"].to_numpy(dtype=np.float64)
                crmse = float(np.sqrt(np.mean(np.square(ce))))
                rrmse = float(np.sqrt(np.mean(np.square(re))))
                cnasa = float(scoped["nasa_candidate"].sum())
                rnasa = float(scoped["nasa_reference"].sum())
                row = {
                    "n_pairs": int(len(scoped)),
                    "candidate_rmse": crmse,
                    "reference_rmse": rrmse,
                    "rmse_delta_candidate_minus_reference": crmse - rrmse,
                    "candidate_nasa_score": cnasa,
                    "reference_nasa_score": rnasa,
                    "nasa_delta_candidate_minus_reference": cnasa - rnasa,
                    "candidate_rmse_win_rate": float(np.mean(np.abs(ce) < np.abs(re))),
                    "available": True,
                }
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    **identity,
                    "candidate": "pretrain_finetune_k",
                    "reference": "scratch_k",
                    "evaluation_scope": scope,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def build_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["target_domain", "shot", "regime", "evaluation_scope"]
    rows: list[dict[str, Any]] = []
    for key, frame in run_metrics.groupby(group_keys, sort=True):
        row: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, **dict(zip(group_keys, key))}
        row["n_runs"] = int(len(frame))
        row["n_nonempty_runs"] = int((frame["n"] > 0).sum())
        for metric in ("rmse", "mae", "mean_error", "nasa_score"):
            values = [float(value) for value in frame[metric].dropna().tolist()]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        rows.append(row)
    return pd.DataFrame(rows)


def verify_source_only_invariance(predictions: pd.DataFrame) -> float:
    source = predictions.loc[predictions["regime"] == "source_only"].copy()
    keys = [
        "target_domain", "model_seed", "support_split_seed", "engine_id", "prefix_label"
    ]
    pivot = source.pivot(index=keys, columns="shot", values="prediction")
    if tuple(sorted(int(value) for value in pivot.columns)) != REGISTERED_SHOTS:
        raise A232Error("source-only invariance table lacks a registered shot")
    if pivot.isna().any().any():
        raise A232Error("source-only predictions are missing for one or more shots")
    maximum = float((pivot.max(axis=1) - pivot.min(axis=1)).max())
    if maximum > SOURCE_INVARIANCE_TOLERANCE:
        raise A232Error(
            f"source-only predictions vary by shot; max_abs_range={maximum:.9g}, "
            f"tolerance={SOURCE_INVARIANCE_TOLERANCE}"
        )
    return maximum


def dataframe_rows(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        records.append(
            {
                key: None if pd.isna(value) else value
                for key, value in raw.items()
            }
        )
    return records, list(frame.columns)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()},
            ensure_ascii=False,
        )
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise A232Error(
                f"run lock already exists: {self.path}; verify that no A23.2 process is active "
                "before removing only this lock file"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired and self.path.exists():
            self.path.unlink()


def run_evaluation(
    args: argparse.Namespace,
    preflight_result: dict[str, Any],
    runs: pd.DataFrame,
    roles: pd.DataFrame,
    cfg: dict[str, Any],
    protocol_hashes: dict[str, str],
    context: dict[str, Any],
) -> dict[str, Any]:
    input_root: Path = context["input_root"]
    output_root: Path = context["output_root"]
    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(232000)
    np.random.seed(232000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(232000)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    raw_frames: dict[str, pd.DataFrame] = {}
    dataset_cache: dict[tuple[str, int, int], CausalPrefixDataset] = {}
    prediction_records: list[dict[str, Any]] = []
    observed_gat: set[bool] = set()
    for run_index, (_, row) in enumerate(runs.iterrows(), start=1):
        domain = str(row["target_domain"])
        seed = int(row["model_seed"])
        split = int(row["support_split_seed"])
        cache_key = (domain, seed, split)
        if cache_key not in dataset_cache:
            if domain not in raw_frames:
                raw_frames[domain] = a23.load_domain_frame(
                    Path(context["training_files"][domain]["path"]),
                    rul_cap=float(cfg["rul_cap"]),
                )
            state = load_normalizer(normalizer_path(input_root, row), domain)
            normalized = a23.normalize(raw_frames[domain], state)
            engines = confirmation_engines(roles, domain, split)
            dataset_cache[cache_key] = CausalPrefixDataset(
                normalized, engines, int(cfg["window_size"])
            )
        records, uses_gat = infer_checkpoint(
            row,
            input_root,
            cfg,
            dataset_cache[cache_key],
            device,
            protocol_hashes,
        )
        observed_gat.add(uses_gat)
        prediction_records.extend(records)
        print(
            f"[A23.2] evaluated {run_index:02d}/{len(runs)} "
            f"target={domain} seed={seed} split={split} shot={int(row['shot'])} "
            f"regime={row['regime']}",
            flush=True,
        )
    if len(observed_gat) != 1 or bool(next(iter(observed_gat))) != bool(
        preflight_result["model_uses_gat"]
    ):
        raise A232Error("model GAT mode changed between preflight and inference")
    predictions = pd.DataFrame(prediction_records)
    expected_predictions = int(preflight_result["expected_prediction_records"])
    if len(predictions) != expected_predictions:
        raise A232Error(
            f"prediction count={len(predictions)}, expected={expected_predictions}"
        )
    prediction_keys = [
        "target_domain", "model_seed", "support_split_seed", "shot", "regime",
        "engine_id", "prefix_label",
    ]
    if predictions.duplicated(prediction_keys).any():
        raise A232Error("duplicate causal-prefix prediction keys")
    source_max_range = verify_source_only_invariance(predictions)
    run_metrics = build_run_metrics(predictions)
    paired = build_paired_metrics(predictions)
    summary = build_summary(run_metrics)

    expected_scopes = len(evaluation_scopes())
    expected_run_metrics = len(runs) * expected_scopes
    if len(run_metrics) != expected_run_metrics:
        raise A232Error("run-level metric record count mismatch")
    expected_pair_rows = (
        len(DOMAINS)
        * runs["model_seed"].nunique()
        * runs["support_split_seed"].nunique()
        * len(REGISTERED_SHOTS)
        * expected_scopes
    )
    if len(paired) != expected_pair_rows:
        raise A232Error("paired metric record count mismatch")

    stage_counts = {
        stage: int(
            predictions.loc[
                predictions["rul_stage"] == stage,
                ["target_domain", "support_split_seed", "engine_id", "prefix_label"],
            ].drop_duplicates().shape[0]
        )
        for stage in ("high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30")
    }
    if stage_counts != preflight_result["coverage_by_rul_stage"]:
        raise A232Error(
            f"preflight/inference RUL-stage coverage mismatch: {stage_counts} vs "
            f"{preflight_result['coverage_by_rul_stage']}"
        )

    artifact_frames = {
        "experimentA23_2_causal_prefix_predictions.csv": predictions,
        "experimentA23_2_run_level_metrics.csv": run_metrics,
        "experimentA23_2_paired_pft_vs_scratch.csv": paired,
        "experimentA23_2_summary.csv": summary,
        "experimentA23_2_prefix_coverage.csv": pd.DataFrame(context["coverage_rows"]),
        "experimentA23_2_checkpoint_inventory.csv": pd.DataFrame(context["checkpoint_rows"]),
    }
    for name, frame in artifact_frames.items():
        rows_out, fields = dataframe_rows(frame)
        atomic_csv(output_root / name, rows_out, fields)

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": preflight_result["registered_primary_question"],
        "complete": True,
        "passed": True,
        "evaluation_only": True,
        "efficacy_claim": False,
        "a23_1_pilot_checkpoints": int(len(runs)),
        "completed_checkpoint_evaluations": int(len(runs)),
        "registered_rul_anchors": list(REGISTERED_RUL_ANCHORS),
        "expected_prediction_records": expected_predictions,
        "completed_prediction_records": int(len(predictions)),
        "expected_run_metric_records": int(expected_run_metrics),
        "completed_run_metric_records": int(len(run_metrics)),
        "expected_paired_metric_records": int(expected_pair_rows),
        "completed_paired_metric_records": int(len(paired)),
        "unique_confirmation_engine_prefixes": int(
            predictions[
                ["target_domain", "support_split_seed", "engine_id", "prefix_label"]
            ].drop_duplicates().shape[0]
        ),
        "coverage_by_rul_stage": stage_counts,
        "all_registered_rul_stages_present": all(value > 0 for value in stage_counts.values()),
        "source_only_prediction_invariance_max_abs_range": source_max_range,
        "source_only_prediction_invariance_tolerance": SOURCE_INVARIANCE_TOLERANCE,
        "model_uses_gat": bool(next(iter(observed_gat))),
        "mixed_registered_endpoints_within_inference_batch": False,
        "rul_anchor_uses_complete_trajectory_label_only_for_offline_endpoint_definition": True,
        "model_input_uses_future_cycles": False,
        "inference_device": str(device),
        "inference_batch_size": int(cfg["batch_size"]),
        "checkpoint_schema_and_model_compatibility_passed": True,
        "training_file_hashes_match_A23_0": True,
        "confirmation_used_for_training": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A23.2 reconstructed all A23.1 pilot checkpoints and completed the registered "
            "causal-prefix coverage audit without retraining or official-test access"
        ),
        "interpretation_limit": (
            "A23.2 validates retrospective causal-prefix evaluation coverage for a two-seed, "
            "one-support-split pilot. It does not establish few-shot or meta-learning efficacy."
        ),
        "next_action": (
            "run_A23_3_formal_five_seed_five_support_split_baselines_then_A24_matched_meta_learning"
        ),
    }
    atomic_json(output_root / "experimentA23_2_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "inputs": {
            "a23_1_decision_sha256": sha256_file(
                input_root / "experimentA23_1_confirmation_decision.json"
            ),
            "a23_1_run_level_sha256": sha256_file(
                input_root / "experimentA23_1_confirmation_run_level.csv"
            ),
            "protocol_hashes": protocol_hashes,
            "config_sha256": preflight_result["config_sha256"],
        },
        "artifacts": {
            name: sha256_file(output_root / name)
            for name in sorted(
                list(artifact_frames)
                + ["experimentA23_2_preflight.json", "experimentA23_2_confirmation_decision.json"]
            )
        },
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(output_root / "experimentA23_2_manifest.json", manifest)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolve(args.output_dir)
    completed_path = output_root / "experimentA23_2_confirmation_decision.json"
    if completed_path.is_file():
        completed = read_json(completed_path)
        if args.resume and completed.get("complete") and completed.get("passed"):
            print(json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False))
            print("[A23.2] resume: existing complete result verified and returned", flush=True)
            return 0
        raise A232Error(
            f"A23.2 output already contains a decision: {completed_path}; use --resume to "
            "read a complete result or choose a new --output-dir"
        )

    result, runs, roles, cfg, protocol_hashes, context = preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print(
            "[A23.2] dry-run passed: all checkpoints/config/data contracts are compatible; "
            "no model was trained and no official test file was accessed",
            flush=True,
        )
        return 0

    if output_root.exists() and any(output_root.iterdir()):
        raise A232Error(
            f"non-empty A23.2 output directory has no complete decision: {output_root}; "
            "use a new output directory to prevent artifact mixing"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with RunLock(output_root / "experimentA23_2_run.lock"):
        atomic_json(output_root / "experimentA23_2_preflight.json", result)
        decision = run_evaluation(
            args, result, runs, roles, cfg, protocol_hashes, context
        )
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (A232Error, a23.A23Error) as exc:
        print(f"[A23.2] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
