#!/usr/bin/env python3
"""A23.3 formal, training-only few-shot transfer baselines.

This is the formal expansion of the A23.1 pilot under the immutable A23.0
engine-level few-shot protocol.  It trains *only* the matched non-meta
baselines below; causal RUL-anchor confirmation evaluation and all efficacy
statistics are deliberately deferred to a separate, evaluation-only script.

    source_only             source-pretrained model, no target adaptation
    scratch_k               random initialisation + k target support engines
    pretrain_finetune_k     same source initialisation + k target support engines

The fixed factorial design is:
  4 target domains x 5 model seeds x 5 support splits x 5 shots x 3 regimes.

Safety properties implemented here:
* only C-MAPSS *training* files are resolved;
* source normalisation and source pretraining exclude the target domain;
* support, selection and confirmation engines are never mixed for fitting;
* source checkpoints are keyed by every input that can change them and are
  atomically cached across support splits;
* every worker writes its own shard, and the parent merges only complete shards;
* GPU availability is checked immediately before launch, avoiding shared-GPU OOM;
* an existing complete output is immutable unless --resume is supplied.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402


EXPERIMENT_ID = "experimentA23_3"
SCRIPT_VERSION = "experimentA23_3_formal_few_shot_transfer_baselines_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
FORMAL_MODEL_SEEDS = (130, 131, 132, 133, 134)
FORMAL_SPLIT_SEEDS = (7101, 7102, 7103, 7104, 7105)
FORMAL_SHOTS = (1, 2, 5, 10, 20)
REGIMES = ("source_only", "scratch_k", "pretrain_finetune_k")
SOURCE_CACHE_WAIT_SECONDS = 5


class A233Error(RuntimeError):
    """Raised when the registered A23.3 contract cannot be honoured."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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
    atomic_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


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


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_csv_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"{name} must be non-empty and unique")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A23.3 formal few-shot transfer baselines")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/experimentA23_3_formal_few_shot_transfer_baselines"),
    )
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--rul-cap", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--source-learning-rate", type=float, default=None)
    parser.add_argument("--pair-aux-weight", type=float, default=None)
    parser.add_argument("--gpus", default="0", help="Physical GPU IDs, e.g. 0,1,2")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--min-free-memory-mb", type=int, default=16000)
    parser.add_argument("--max-gpu-utilization", type=int, default=15)
    parser.add_argument("--gpu-poll-seconds", type=int, default=15)
    parser.add_argument("--max-gpu-wait-minutes", type=int, default=180)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    parser.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.gpus = parse_csv_ints(args.gpus, name="gpus")
    if args.source_pretrain_steps <= 0 or args.target_epochs <= 0:
        raise A233Error("source-pretrain-steps and target-epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise A233Error("batch-size must be positive")
    if args.window_size is not None and args.window_size < 2:
        raise A233Error("window-size must be at least 2")
    if args.max_workers is not None and args.max_workers <= 0:
        raise A233Error("max-workers must be positive")
    if args.min_free_memory_mb <= 0:
        raise A233Error("min-free-memory-mb must be positive")
    if not 0 <= args.max_gpu_utilization <= 100:
        raise A233Error("max-gpu-utilization must be within [0, 100]")
    if args.gpu_poll_seconds <= 0 or args.max_gpu_wait_minutes <= 0:
        raise A233Error("GPU polling and wait limits must be positive")
    if args.worker and not (
        args.target_domain and args.model_seed is not None and args.support_split_seed is not None
    ):
        raise A233Error("worker mode requires target-domain, model-seed, and support-split-seed")
    if args.worker and args.model_seed not in FORMAL_MODEL_SEEDS:
        raise A233Error("worker model seed is not registered by A23.3")
    if args.worker and args.support_split_seed not in FORMAL_SPLIT_SEEDS:
        raise A233Error("worker split seed is not registered by A23.3")
    return args


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    path = resolve(args.config)
    if not path.is_file():
        raise A233Error(f"config is missing: {path}")
    # Reuse the project's registered config parser, retaining its training choices.
    cfg = a23.load_config(args)
    return cfg, path, sha256_file(path)


def strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise A233Error(f"{field} is not a strict boolean: {value!r}")


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str], dict[str, dict[str, str]]]:
    root = resolve(args.protocol_dir)
    protocol_path = root / "experimentA23_few_shot_protocol.json"
    roles_path = root / "experimentA23_engine_roles.csv"
    decision_path = root / "experimentA23_confirmation_decision.json"
    for path in (protocol_path, roles_path, decision_path):
        if not path.is_file():
            raise A233Error(f"required A23.0 artifact is missing: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not (decision.get("complete") is True and decision.get("passed") is True):
        raise A233Error("A23.0 must be complete and passed")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A233Error(f"A23.0 official-test boundary violated: {field}=true")
    if tuple(int(x) for x in protocol.get("shot_counts", ())) != (0, 1, 2, 5, 10, 20):
        raise A233Error("A23.0 locked shot counts do not match the registered protocol")
    if tuple(int(x) for x in protocol.get("support_split_seeds", ())) != FORMAL_SPLIT_SEEDS:
        raise A233Error("A23.0 support split seeds do not match A23.3's registered design")
    roles = pd.read_csv(roles_path)
    needed = {"target_domain", "support_split_seed", "engine_id", "role", "support_rank"}
    if missing := needed - set(roles.columns):
        raise A233Error(f"A23.0 roles are missing columns: {sorted(missing)}")
    roles["target_domain"] = roles["target_domain"].astype(str)
    roles["support_split_seed"] = pd.to_numeric(roles["support_split_seed"], errors="raise").astype(int)
    roles["engine_id"] = pd.to_numeric(roles["engine_id"], errors="raise").astype(int)
    roles["support_rank"] = pd.to_numeric(roles["support_rank"], errors="raise").astype(int)
    if set(roles["target_domain"].unique()) != set(DOMAINS):
        raise A233Error("A23.0 role table does not cover exactly FD001--FD004")
    if set(roles["role"].unique()) != {"support_pool", "selection", "confirmation"}:
        raise A233Error("A23.0 role table has unexpected role labels")
    hashes = {path.name: sha256_file(path) for path in (protocol_path, roles_path, decision_path)}
    inventory = protocol.get("training_file_inventory")
    if not isinstance(inventory, list):
        raise A233Error("A23.0 protocol lacks training_file_inventory")
    data_hashes: dict[str, dict[str, str]] = {}
    for item in inventory:
        if not isinstance(item, dict) or not {"domain", "sha256"} <= set(item):
            raise A233Error("malformed A23.0 training_file_inventory")
        data_hashes[str(item["domain"])] = {"sha256": str(item["sha256"])}
    if set(data_hashes) != set(DOMAINS):
        raise A233Error("A23.0 training_file_inventory does not cover the four domains")
    return protocol, roles, hashes, data_hashes


def verify_training_files(args: argparse.Namespace, expected: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    data_root = resolve(args.data_dir)
    found: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        path = a23.resolve_train_file(data_root, domain)
        digest = sha256_file(path)
        if digest != expected[domain]["sha256"]:
            raise A233Error(
                f"training data changed after A23.0 for {domain}: expected={expected[domain]['sha256']}, actual={digest}"
            )
        found[domain] = {"path": str(path), "sha256": digest}
    return found


def role_engines(roles: pd.DataFrame, domain: str, split_seed: int, role: str, shot: int | None = None) -> list[int]:
    frame = roles.loc[
        (roles["target_domain"] == domain)
        & (roles["support_split_seed"] == int(split_seed))
        & (roles["role"] == role)
    ].copy()
    if role == "support_pool":
        if shot is None:
            raise A233Error("support pool requires a shot count")
        frame = frame.loc[frame["support_rank"] <= int(shot)]
    engines = sorted(int(x) for x in frame["engine_id"].tolist())
    if not engines or len(engines) != len(set(engines)):
        raise A233Error(f"invalid {role} engine set for {domain}/split={split_seed}")
    return engines


def validate_engine_roles(roles: pd.DataFrame) -> None:
    for domain in DOMAINS:
        for split in FORMAL_SPLIT_SEEDS:
            groups = {
                "support": set(role_engines(roles, domain, split, "support_pool", max(FORMAL_SHOTS))),
                "selection": set(role_engines(roles, domain, split, "selection")),
                "confirmation": set(role_engines(roles, domain, split, "confirmation")),
            }
            if groups["support"] & groups["selection"] or groups["support"] & groups["confirmation"] or groups["selection"] & groups["confirmation"]:
                raise A233Error(f"engine-level role overlap for {domain}/split={split}")
            for shot in FORMAL_SHOTS:
                support = role_engines(roles, domain, split, "support_pool", shot)
                if len(support) != shot:
                    raise A233Error(f"locked support count differs from K={shot} for {domain}/split={split}")


def worker_dir(root: Path, domain: str, model_seed: int, split_seed: int) -> Path:
    return root / "shards" / f"{domain}_mseed{model_seed}_split{split_seed}"


def source_cache_dir(root: Path, cache_key: str) -> Path:
    return root / "source_cache" / cache_key


def expected_workers() -> list[tuple[str, int, int]]:
    # Split is outermost so concurrently launched workers preferentially use
    # distinct target/seed source caches rather than waiting on one cache lock.
    return [
        (domain, seed, split)
        for split in FORMAL_SPLIT_SEEDS
        for domain in DOMAINS
        for seed in FORMAL_MODEL_SEEDS
    ]


def source_cache_identity(
    *, domain: str, model_seed: int, config_sha256: str, training_hashes: dict[str, dict[str, str]], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "cache_schema": 1,
        "target_domain": domain,
        "model_seed": int(model_seed),
        "source_domains": [item for item in DOMAINS if item != domain],
        "config_sha256": config_sha256,
        "training_implementation_sha256": {
            "a23_1_baseline": sha256_file(Path(a23.__file__).resolve()),
            "a23_3_formal": sha256_file(Path(__file__).resolve()),
        },
        "training_file_sha256": {key: training_hashes[key]["sha256"] for key in DOMAINS},
        "source_pretrain_steps": int(args.source_pretrain_steps),
        "source_learning_rate": float(args.source_learning_rate) if args.source_learning_rate is not None else None,
        "pair_aux_weight": float(args.pair_aux_weight) if args.pair_aux_weight is not None else None,
    }


class SourceCacheLock:
    """Exclusive cache lock.  Existing locks are waited on, never deleted automatically."""

    def __init__(self, path: Path, *, wait_seconds: int) -> None:
        self.path = path
        self.wait_seconds = int(wait_seconds)
        self.acquired = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise A233Error(
                        f"timed out waiting for source-cache lock: {self.path}; do not remove it unless its owner is known to be stopped"
                    )
                time.sleep(SOURCE_CACHE_WAIT_SECONDS)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json({"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.acquired = True
            return

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False


def safe_torch_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A233Error(f"required torch artifact is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise A233Error(f"failed to load torch artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A233Error(f"torch artifact must contain a dictionary: {path}")
    return payload


def finite_state(state: Any, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state:
        raise A233Error(f"state dictionary is missing or empty: {path}")
    copied: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item():
            raise A233Error(f"invalid/non-finite tensor {name!r} in {path}")
        copied[name] = tensor.detach().cpu().clone()
    return copied


def cache_ready_payload(cache_root: Path, expected_identity: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any], Path, str] | None:
    ready = cache_root / "ready.json"
    state_path = cache_root / "source_pretrained_state.pt"
    metadata_path = cache_root / "metadata.json"
    if not (ready.is_file() and state_path.is_file() and metadata_path.is_file()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("identity") != expected_identity:
        raise A233Error(f"source cache identity mismatch: {cache_root}")
    ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    actual_sha = sha256_file(state_path)
    if ready_payload.get("state_sha256") != actual_sha:
        raise A233Error(f"source cache SHA-256 mismatch: {state_path}")
    payload = safe_torch_load(state_path)
    if payload.get("identity") != expected_identity:
        raise A233Error(f"source checkpoint identity mismatch: {state_path}")
    state = finite_state(payload.get("state"), state_path)
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != {"mean", "std"}:
        raise A233Error(f"source checkpoint normalizer is malformed: {state_path}")
    return state, normalizer, state_path, actual_sha


def obtain_source_cache(
    *, root: Path, target: str, model_seed: int, cfg: dict[str, Any], config_sha256: str,
    training_files: dict[str, dict[str, str]], args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], Path, str, bool]:
    identity = source_cache_identity(
        domain=target, model_seed=model_seed, config_sha256=config_sha256,
        training_hashes=training_files, args=args,
    )
    cache_key = stable_digest(identity)[:32]
    cache_root = source_cache_dir(root, cache_key)
    found = cache_ready_payload(cache_root, identity)
    if found is not None:
        state, normalizer, path, digest = found
        return state, normalizer, path, digest, True

    lock = SourceCacheLock(cache_root / "source_cache.lock", wait_seconds=args.max_gpu_wait_minutes * 60)
    lock.acquire()
    try:
        found = cache_ready_payload(cache_root, identity)
        if found is not None:
            state, normalizer, path, digest = found
            return state, normalizer, path, digest, True
        source_domains = [domain for domain in DOMAINS if domain != target]
        raw_frames = {
            domain: a23.load_domain_frame(Path(training_files[domain]["path"]), rul_cap=float(cfg["rul_cap"]))
            for domain in DOMAINS
        }
        normalizer = a23.fit_source_normalizer({domain: raw_frames[domain] for domain in source_domains})
        frames = {domain: a23.normalize(frame, normalizer) for domain, frame in raw_frames.items()}
        _, base_state = a23.create_base_model(len(a23.FEATURE_COLUMNS), cfg, model_seed)
        model = a23.build_model("gnn", len(a23.FEATURE_COLUMNS), cfg)
        model.load_state_dict(base_state, strict=True)
        learner, history = a23.train_source_steps(
            model,
            a23.source_loaders(frames, target, cfg, model_seed),
            steps=int(args.source_pretrain_steps),
            learning_rate=float(cfg["source_learning_rate"]),
            pair_aux_weight=float(cfg["pair_aux_weight"]),
            seed=int(model_seed),
            device=device,
        )
        state = {name: tensor.detach().cpu().clone() for name, tensor in learner.state_dict().items()}
        del learner, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload = {
            "identity": identity,
            "state": state,
            "normalizer": normalizer,
            "source_history": history,
            "feature_columns": list(a23.FEATURE_COLUMNS),
            "source_fitted_target_domain": False,
            "selection_or_confirmation_used_for_source_training": False,
        }
        state_path = cache_root / "source_pretrained_state.pt"
        atomic_torch_save(state_path, payload)
        digest = sha256_file(state_path)
        atomic_json(cache_root / "metadata.json", {
            "identity": identity,
            "created_at_utc": utc_now(),
            "source_cache_key": cache_key,
            "source_state_sha256": digest,
        })
        # ready.json is committed last: readers never consume a partial cache.
        atomic_json(cache_root / "ready.json", {"state_sha256": digest, "complete": True})
        return state, normalizer, state_path, digest, False
    finally:
        lock.release()


def metric_rows(model: torch.nn.Module, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for name, loader in loaders.items():
        values.update(a23.flatten_metrics(name, a23.evaluate(model, loader, device)))
    return values


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_worker(args: argparse.Namespace) -> None:
    root = resolve(args.output_dir)
    target, model_seed, split_seed = args.target_domain, int(args.model_seed), int(args.support_split_seed)
    directory = worker_dir(root, target, model_seed, split_seed)
    directory.mkdir(parents=True, exist_ok=True)
    status_path, run_path = directory / "worker_status.json", directory / "run_level.csv"
    if args.resume and status_path.is_file() and run_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("complete") is True and prior.get("passed") is True and prior.get("completed_run_records") == len(FORMAL_SHOTS) * len(REGIMES):
            print(f"[A23.3] resume skip {directory.name}", flush=True)
            return
    protocol, roles, protocol_hashes, locked_hashes = load_protocol(args)
    validate_engine_roles(roles)
    training_files = verify_training_files(args, locked_hashes)
    cfg, config_path, config_sha = load_config(args)
    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise A233Error("worker was assigned CUDA but CUDA is not available")
    a23.seed_everything(model_seed)

    source_state, normalizer, source_path, source_sha, cache_reused = obtain_source_cache(
        root=root, target=target, model_seed=model_seed, cfg=cfg, config_sha256=config_sha,
        training_files=training_files, args=args, device=device,
    )
    raw_target = a23.load_domain_frame(Path(training_files[target]["path"]), rul_cap=float(cfg["rul_cap"]))
    target_frame = a23.normalize(raw_target, normalizer)
    source_domains = [domain for domain in DOMAINS if domain != target]
    normalizer_path = directory / "source_normalizer.json"
    atomic_json(normalizer_path, {
        "fitted_domains": source_domains,
        "feature_columns": list(a23.FEATURE_COLUMNS),
        "normalizer": normalizer,
        "target_domain_used_for_fit": False,
        "selection_engines_used_for_fit": False,
        "confirmation_engines_used_for_fit": False,
        "source_cache_path": str(source_path),
        "source_cache_sha256": source_sha,
    })
    normalizer_sha = sha256_file(normalizer_path)

    selection_engines = role_engines(roles, target, split_seed, "selection")
    confirmation_engines = role_engines(roles, target, split_seed, "confirmation")
    selection_set, confirmation_set = set(selection_engines), set(confirmation_engines)
    selection_loader = a23.make_loader(
        a23.WindowDataset(target_frame, selection_engines, int(cfg["window_size"])),
        batch_size=int(cfg["batch_size"]), shuffle=False, seed=model_seed + 100,
    )
    confirmation_loader = a23.make_loader(
        a23.WindowDataset(target_frame, confirmation_engines, int(cfg["window_size"])),
        batch_size=int(cfg["batch_size"]), shuffle=False, seed=model_seed + 101,
    )
    eval_loaders = {"selection": selection_loader, "confirmation": confirmation_loader}
    _, base_state = a23.create_base_model(len(a23.FEATURE_COLUMNS), cfg, model_seed)
    base_sha = state_sha256(base_state)
    source_state_sha = state_sha256(source_state)

    rows: list[dict[str, Any]] = []
    for shot in FORMAL_SHOTS:
        support_engines = role_engines(roles, target, split_seed, "support_pool", shot)
        support_set = set(support_engines)
        if len(support_engines) != shot or support_set & selection_set or support_set & confirmation_set:
            raise A233Error(f"support/selection/confirmation role leakage for {target}/split={split_seed}/K={shot}")
        support_loader = a23.make_loader(
            a23.WindowDataset(target_frame, support_engines, int(cfg["window_size"])),
            batch_size=int(cfg["batch_size"]), shuffle=True,
            seed=model_seed * 1_000_000 + split_seed * 100 + shot,
        )
        for regime in REGIMES:
            a23.seed_everything(model_seed)
            initial_state = source_state if regime in {"source_only", "pretrain_finetune_k"} else base_state
            model = a23.build_model("gnn", len(a23.FEATURE_COLUMNS), cfg)
            model.load_state_dict(initial_state, strict=True)
            history: list[dict[str, float]] = []
            if regime == "source_only":
                model = model.to(device)
                epochs = 0
            else:
                model, history = a23.train_epochs(
                    model, support_loader, epochs=int(args.target_epochs),
                    learning_rate=float(cfg["inner_lr"]), pair_aux_weight=float(cfg["pair_aux_weight"]),
                    device=device,
                    label=f"A23.3 target={target} seed={model_seed} split={split_seed} K={shot} regime={regime}",
                )
                epochs = int(args.target_epochs)
            metrics = metric_rows(model, eval_loaders, device)
            checkpoint = directory / f"{regime}_shot{shot}_target_adapted.pt"
            model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            atomic_torch_save(checkpoint, {
                "experiment_id": EXPERIMENT_ID,
                "checkpoint_schema": 2,
                "state": model_state,
                "regime": regime,
                "shot": int(shot),
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split_seed,
                "support_engines": support_engines,
                "feature_columns": list(a23.FEATURE_COLUMNS),
                "config_sha256": config_sha,
                "protocol_hashes": protocol_hashes,
                "normalizer_path": str(normalizer_path),
                "normalizer_sha256": normalizer_sha,
                "source_cache_path": str(source_path),
                "source_cache_sha256": source_sha,
                "source_state_sha256": source_state_sha,
                "base_state_sha256": base_sha,
                "source_pretrain_steps": int(args.source_pretrain_steps),
                "target_epochs": epochs,
                "causal_anchor_evaluation_contract": "A23.2_v2_rul_090_045_015",
                "selection_used_for_training": False,
                "selection_used_for_epoch_selection": False,
                "confirmation_used_for_training": False,
                "confirmation_used_for_normalizer_fit": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
                "target_history": history,
            })
            checkpoint_sha = sha256_file(checkpoint)
            row: dict[str, Any] = {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split_seed,
                "shot": int(shot),
                "regime": regime,
                "source_domains": json.dumps(source_domains),
                "support_engines": json.dumps(support_engines),
                "selection_engine_count": len(selection_engines),
                "confirmation_engine_count": len(confirmation_engines),
                "feature_count": len(a23.FEATURE_COLUMNS),
                "feature_columns": json.dumps(list(a23.FEATURE_COLUMNS)),
                "window_size": int(cfg["window_size"]),
                "rul_cap": float(cfg["rul_cap"]),
                "source_pretrain_steps": int(args.source_pretrain_steps),
                "target_epochs": epochs,
                "config_sha256": config_sha,
                "protocol_hashes": json.dumps(protocol_hashes, sort_keys=True),
                "model_checkpoint": str(checkpoint),
                "model_checkpoint_sha256": checkpoint_sha,
                "normalizer_path": str(normalizer_path),
                "normalizer_sha256": normalizer_sha,
                "source_cache_path": str(source_path),
                "source_cache_sha256": source_sha,
                "source_cache_reused": bool(cache_reused),
                "source_state_sha256": source_state_sha,
                "base_state_sha256": base_sha,
                "causal_anchor_evaluation_contract": "A23.2_v2_rul_090_045_015",
                "selection_used_for_training": False,
                "selection_used_for_epoch_selection": False,
                "confirmation_used_for_training": False,
                "confirmation_used_for_normalizer_fit": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
            row.update(metrics)
            rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[A23.3] completed target={target} seed={model_seed} split={split_seed} K={shot} regime={regime}", flush=True)

    if len(rows) != len(FORMAL_SHOTS) * len(REGIMES):
        raise A233Error("worker run cardinality mismatch")
    fields = sorted({key for row in rows for key in row})
    atomic_csv(run_path, rows, fields)
    atomic_json(status_path, {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "target_domain": target,
        "model_seed": model_seed,
        "support_split_seed": split_seed,
        "expected_run_records": len(FORMAL_SHOTS) * len(REGIMES),
        "completed_run_records": len(rows),
        "source_cache_path": str(source_path),
        "source_cache_sha256": source_sha,
        "source_cache_reused": bool(cache_reused),
        "protocol_hashes": protocol_hashes,
        "config_sha256": config_sha,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })


def worker_command(args: argparse.Namespace, domain: str, seed: int, split: int) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    values = (
        ("--data-dir", args.data_dir), ("--config", args.config), ("--protocol-dir", args.protocol_dir),
        ("--output-dir", args.output_dir), ("--source-pretrain-steps", args.source_pretrain_steps),
        ("--target-epochs", args.target_epochs), ("--target-domain", domain), ("--model-seed", seed),
        ("--support-split-seed", split), ("--device", "cuda"),
        ("--max-gpu-wait-minutes", args.max_gpu_wait_minutes),
    )
    for key, value in values:
        command.extend((key, str(value)))
    for key, value in (("--batch-size", args.batch_size), ("--window-size", args.window_size), ("--rul-cap", args.rul_cap), ("--learning-rate", args.learning_rate), ("--source-learning-rate", args.source_learning_rate), ("--pair-aux-weight", args.pair_aux_weight)):
        if value is not None:
            command.extend((key, str(value)))
    if args.resume:
        command.append("--resume")
    return command


def gpu_inventory() -> dict[int, dict[str, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    except Exception as exc:
        raise A233Error(f"GPU scheduling requires nvidia-smi: {exc}") from exc
    inventory: dict[int, dict[str, int]] = {}
    for raw in result.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 3:
            raise A233Error(f"could not parse nvidia-smi row: {raw!r}")
        try:
            index, free_mb, utilization = map(int, parts)
        except ValueError as exc:
            raise A233Error(f"could not parse nvidia-smi numeric row: {raw!r}") from exc
        inventory[index] = {"free_mb": free_mb, "utilization": utilization}
    return inventory


def eligible_gpus(args: argparse.Namespace, occupied: set[int]) -> list[int]:
    inventory = gpu_inventory()
    unknown = set(args.gpus) - set(inventory)
    if unknown:
        raise A233Error(f"requested GPU ids are unavailable: {sorted(unknown)}")
    return [
        gpu for gpu in args.gpus
        if gpu not in occupied
        and inventory[gpu]["free_mb"] >= int(args.min_free_memory_mb)
        and inventory[gpu]["utilization"] <= int(args.max_gpu_utilization)
    ]


def worker_complete(root: Path, worker: tuple[str, int, int]) -> bool:
    domain, seed, split = worker
    directory = worker_dir(root, domain, seed, split)
    status, runs = directory / "worker_status.json", directory / "run_level.csv"
    if not (status.is_file() and runs.is_file()):
        return False
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
        frame = pd.read_csv(runs)
    except Exception:
        return False
    return payload.get("complete") is True and payload.get("passed") is True and len(frame) == len(FORMAL_SHOTS) * len(REGIMES)


def dataframe_rows(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for item in frame.to_dict("records"):
        rows.append({key: None if pd.isna(value) else value for key, value in item.items()})
    return rows, list(frame.columns)


def merge_and_decide(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.output_dir)
    workers = expected_workers()
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    missing: list[str] = []
    for domain, seed, split in workers:
        directory = worker_dir(root, domain, seed, split)
        if not worker_complete(root, (domain, seed, split)):
            missing.append(directory.name)
            continue
        statuses.append(json.loads((directory / "worker_status.json").read_text(encoding="utf-8")))
        frames.append(pd.read_csv(directory / "run_level.csv"))
    if missing:
        raise A233Error(f"cannot merge incomplete worker shards: {missing[:20]} (n={len(missing)})")
    merged = pd.concat(frames, ignore_index=True)
    expected_records = len(workers) * len(FORMAL_SHOTS) * len(REGIMES)
    if len(merged) != expected_records:
        raise A233Error(f"merged record count={len(merged)}, expected={expected_records}")
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "regime"]
    expected_keys = {
        (domain, seed, split, shot, regime)
        for domain, seed, split in workers
        for shot in FORMAL_SHOTS
        for regime in REGIMES
    }
    actual_keys = set(merged[keys].itertuples(index=False, name=None))
    if actual_keys != expected_keys or merged.duplicated(keys).any():
        raise A233Error("formal factorial run grid is incomplete or contains duplicate rows")
    integrity = (
        "selection_used_for_training", "selection_used_for_epoch_selection", "confirmation_used_for_training",
        "confirmation_used_for_normalizer_fit", "official_test_files_accessed", "official_test_forward_run",
    )
    for column in integrity:
        values = [strict_bool(value, field=column) for value in merged[column].tolist()]
        if any(values):
            raise A233Error(f"integrity violation: {column}=true")
    if set(merged["causal_anchor_evaluation_contract"].unique()) != {"A23.2_v2_rul_090_045_015"}:
        raise A233Error("worker checkpoints do not share the locked A23.2 causal-anchor contract")
    cache_keys = merged[["target_domain", "model_seed", "source_cache_path", "source_cache_sha256"]].drop_duplicates()
    if len(cache_keys) != len(DOMAINS) * len(FORMAL_MODEL_SEEDS):
        raise A233Error("source cache inventory is not one cache per target-domain/model-seed")
    for _, item in cache_keys.iterrows():
        path = Path(str(item["source_cache_path"]))
        if not path.is_file() or sha256_file(path) != str(item["source_cache_sha256"]):
            raise A233Error(f"missing or modified source cache: {path}")
    merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
    run_path = root / "experimentA23_3_training_run_level.csv"
    records, fields = dataframe_rows(merged)
    atomic_csv(run_path, records, fields)
    metrics = [column for column in merged.columns if column.startswith("confirmation_") and column.endswith(("_rmse", "_mae", "_nasa_score", "_mean_error"))]
    summary = merged.groupby(["target_domain", "shot", "regime"], as_index=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(str(x) for x in column if x).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary_path = root / "experimentA23_3_training_summary.csv"
    s_rows, s_fields = dataframe_rows(summary)
    atomic_csv(summary_path, s_rows, s_fields)
    cache_path = root / "experimentA23_3_source_cache_inventory.csv"
    c_rows, c_fields = dataframe_rows(cache_keys.sort_values(["target_domain", "model_seed"]))
    atomic_csv(cache_path, c_rows, c_fields)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Under locked A23.0 engine-level k-shot roles, can the full matched source-only, scratch-k, "
            "and ordinary pretrain-plus-finetune-k baseline grid be trained without leakage before causal-anchor evaluation and Meta-GNN comparison?"
        ),
        "complete": True,
        "passed": True,
        "training_only": True,
        "baseline_efficacy_claim": False,
        "expected_worker_cells": len(workers),
        "completed_worker_cells": len(statuses),
        "expected_run_records": expected_records,
        "completed_run_records": int(len(merged)),
        "model_seeds": list(FORMAL_MODEL_SEEDS),
        "support_split_seeds": list(FORMAL_SPLIT_SEEDS),
        "shots": list(FORMAL_SHOTS),
        "regimes": list(REGIMES),
        "source_cache_entries": int(len(cache_keys)),
        "source_pretraining_reused_across_splits": True,
        "causal_anchor_evaluation_contract": "A23.2_v2_rul_090_045_015",
        "selection_and_confirmation_unused_for_training_or_selection": True,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": "A23.3 completed the full training-only few-shot transfer baseline factorial grid under the locked A23.0 protocol",
        "next_action": "run_A23_4_formal_causal_anchor_evaluation_and_hierarchical_pft_vs_scratch_inference",
    }
    atomic_json(root / "experimentA23_3_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "artifacts": {
            run_path.name: sha256_file(run_path),
            summary_path.name: sha256_file(summary_path),
            cache_path.name: sha256_file(cache_path),
            "protocol_dir": str(resolve(args.protocol_dir)),
            "config_sha256": sha256_file(resolve(args.config)),
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA23_3_manifest.json", manifest)
    return decision


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    protocol, roles, protocol_hashes, locked_hashes = load_protocol(args)
    del protocol
    validate_engine_roles(roles)
    files = verify_training_files(args, locked_hashes)
    cfg, config_path, config_sha = load_config(args)
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": "A23.3 formal, training-only few-shot transfer baselines under locked A23.0 roles",
        "output_dir": str(resolve(args.output_dir)),
        "protocol_hashes": protocol_hashes,
        "training_file_inventory": files,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "model_seeds": list(FORMAL_MODEL_SEEDS),
        "support_split_seeds": list(FORMAL_SPLIT_SEEDS),
        "shots": list(FORMAL_SHOTS),
        "regimes": list(REGIMES),
        "expected_worker_cells": len(expected_workers()),
        "expected_run_records": len(expected_workers()) * len(FORMAL_SHOTS) * len(REGIMES),
        "expected_source_cache_entries": len(DOMAINS) * len(FORMAL_MODEL_SEEDS),
        "source_pretrain_steps": int(args.source_pretrain_steps),
        "target_epochs": int(args.target_epochs),
        "window_size": int(cfg["window_size"]),
        "batch_size": int(cfg["batch_size"]),
        "causal_anchor_evaluation_contract": "A23.2_v2_rul_090_045_015",
        "new_predictor_training": not args.dry_run,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def parent_main(args: argparse.Namespace) -> None:
    root = resolve(args.output_dir)
    completed = root / "experimentA23_3_confirmation_decision.json"
    if completed.is_file():
        prior = json.loads(completed.read_text(encoding="utf-8"))
        if args.resume and prior.get("complete") is True and prior.get("passed") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A23.3] resume: existing complete result returned", flush=True)
            return
        raise A233Error(f"output already contains a decision: {completed}; use a new --output-dir or --resume")
    preview = dry_run_payload(args)
    print(json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print("[A23.3] dry-run completed; no predictor was trained and no official test file was read", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    workers = [item for item in expected_workers() if not (args.resume and worker_complete(root, item))]
    print(f"[A23.3] scheduled workers={len(workers)} skipped={len(expected_workers()) - len(workers)}", flush=True)
    if args.single_process or args.device == "cpu":
        for domain, seed, split in workers:
            local = deepcopy(args)
            local.worker, local.target_domain = True, domain
            local.model_seed, local.support_split_seed = seed, split
            local.device = "cpu"
            run_worker(local)
    else:
        maximum = min(int(args.max_workers or len(args.gpus)), len(args.gpus))
        pending = list(workers)
        active: dict[int, dict[str, Any]] = {}
        no_gpu_since: float | None = None
        while pending or active:
            free_slots = maximum - len(active)
            if pending and free_slots > 0:
                candidates = eligible_gpus(args, set(active))
                for gpu in candidates[:free_slots]:
                    domain, seed, split = pending.pop(0)
                    directory = worker_dir(root, domain, seed, split)
                    directory.mkdir(parents=True, exist_ok=True)
                    log_path = directory / "worker_training.log"
                    handle = log_path.open("a", encoding="utf-8")
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    process = subprocess.Popen(worker_command(args, domain, seed, split), cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
                    active[gpu] = {"process": process, "handle": handle, "log": log_path, "item": (domain, seed, split)}
                    print(f"[A23.3] launched target={domain} seed={seed} split={split} gpu={gpu} pid={process.pid}", flush=True)
                no_gpu_since = None
            elif pending and not active:
                if no_gpu_since is None:
                    no_gpu_since = time.monotonic()
                    print("[A23.3] waiting for an eligible GPU...", flush=True)
                elif time.monotonic() - no_gpu_since > args.max_gpu_wait_minutes * 60:
                    raise A233Error("no requested GPU met the configured free-memory/utilization thresholds before timeout")
            finished: list[int] = []
            for gpu, record in list(active.items()):
                code = record["process"].poll()
                if code is None:
                    continue
                record["handle"].close()
                if code != 0:
                    tail = "\n".join(record["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                    for other in active.values():
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    raise A233Error(f"worker failed item={record['item']} exit={code}\n{tail}")
                print(f"[A23.3] completed target={record['item'][0]} seed={record['item'][1]} split={record['item'][2]} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if (pending or active) and not finished:
                time.sleep(min(args.gpu_poll_seconds, 15))
    decision = merge_and_decide(args)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
    else:
        parent_main(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A233Error as exc:
        print(f"[A23.3] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
