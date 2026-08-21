#!/usr/bin/env python3
"""A24.2 formal training-only Meta-noGraph/Meta-GNN factorial experiment.

The script freezes the successful A24.1 implementation and expands it to the
complete A24.0 target factorial grid:

    4 target domains x 5 model seeds x 5 support splits = 100 workers
    5 nested shots x 2 methods = 10 records per worker = 1000 records

Each worker runs Reptile once per method, then independently adapts the frozen
meta-initialisation at K=1/2/5/10/20.  It never evaluates target selection or
confirmation engines.  Formal causal-anchor evaluation is reserved for A24.3.
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
from typing import Any, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA24_1_meta_no_graph_and_meta_gnn_pilot as pilot  # noqa: E402


EXPERIMENT_ID = "experimentA24_2"
SCRIPT_VERSION = "experimentA24_2_formal_meta_learning_factorial_training_v2"
DOMAINS = pilot.DOMAINS
METHODS = pilot.METHODS
MODEL_SEEDS = (130, 131, 132, 133, 134)
SPLITS = (7101, 7102, 7103, 7104, 7105)
SHOTS = (1, 2, 5, 10, 20)
PRIMARY_SHOT = 5
EXPECTED_WORKERS = 100
EXPECTED_RECORDS = 1000


class A24FormalError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A24.2 formal training-only meta-learning grid")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--protocol-dir", type=Path,
                   default=Path("outputs/experimentA23_few_shot_protocol_preflight"))
    p.add_argument("--a24-0-output-dir", type=Path,
                   default=Path("outputs/experimentA24_0_meta_learning_contract_preflight"))
    p.add_argument("--a24-1-output-dir", type=Path,
                   default=Path("outputs/experimentA24_1_meta_no_graph_and_meta_gnn_pilot"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/experimentA24_2_formal_meta_learning_factorial_training"))
    p.add_argument("--target-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--rul-cap", type=float, default=None)
    p.add_argument("--inner-learning-rate", type=float, default=None)
    p.add_argument("--pair-aux-weight", type=float, default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--min-free-memory-mb", type=int, default=20000)
    p.add_argument("--max-gpu-utilization", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    p.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.target_epochs <= 0 or args.max_workers <= 0:
        raise A24FormalError("target-epochs and max-workers must be positive")
    if args.target_epochs != 10:
        raise A24FormalError("A24.2 target-epochs is frozen at the A24.1 value of 10")
    overrides = {
        "batch-size": args.batch_size,
        "window-size": args.window_size,
        "rul-cap": args.rul_cap,
        "inner-learning-rate": args.inner_learning_rate,
        "pair-aux-weight": args.pair_aux_weight,
    }
    changed = [name for name, value in overrides.items() if value is not None]
    if changed:
        raise A24FormalError(
            f"formal A24.2 forbids post-pilot training overrides: {changed}"
        )
    if args.min_free_memory_mb < 0 or not 0 <= args.max_gpu_utilization <= 100:
        raise A24FormalError("invalid GPU threshold")
    if args.worker and (args.target_domain is None or args.model_seed is None
                        or args.support_split_seed is None):
        raise A24FormalError("worker requires target-domain/model-seed/support-split-seed")
    # Complete compatibility contract consumed by A24.1/A23 helpers.
    args.model_seeds = MODEL_SEEDS
    args.support_split_seeds = SPLITS
    args.shot = PRIMARY_SHOT
    args.outer_steps = None
    args.inner_steps = None
    args.outer_learning_rate = None
    args.source_learning_rate = None
    args.learning_rate = args.inner_learning_rate
    return args


def resolve(path: Path) -> Path:
    return pilot.resolve(path)


def load_a24_1_freeze(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.a24_1_output_dir)
    decision_path = root / "experimentA24_1_confirmation_decision.json"
    manifest_path = root / "experimentA24_1_manifest.json"
    run_path = root / "experimentA24_1_run_level.csv"
    audit_path = root / "experimentA24_1_parameter_audit.csv"
    for path in (decision_path, manifest_path, run_path, audit_path):
        if not path.is_file():
            raise A24FormalError(f"required frozen A24.1 artifact is missing: {path}")
    decision = pilot.load_json(decision_path)
    manifest = pilot.load_json(manifest_path)
    required = {
        "complete": True,
        "passed": True,
        "pilot_only": True,
        "runtime_shared_parameter_assertion_passed": True,
        "checkpoint_reload_passed": True,
        "confirmation_engines_evaluated": False,
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            raise A24FormalError(f"A24.1 freeze condition failed: {key}")
    if manifest.get("script_version") != pilot.SCRIPT_VERSION:
        raise A24FormalError("A24.1 manifest/script version mismatch")
    current_pilot_hash = pilot.sha256(Path(pilot.__file__).resolve())
    if manifest.get("script_sha256") != current_pilot_hash:
        raise A24FormalError(
            "A24.1 source changed after pilot completion; restore the frozen pilot script"
        )
    artifacts = manifest.get("artifacts", {})
    for path in (run_path, audit_path):
        expected_hash = artifacts.get(path.name)
        actual_hash = pilot.sha256(path)
        if expected_hash != actual_hash:
            raise A24FormalError(
                f"frozen A24.1 artifact hash mismatch: {path.name}"
            )
    runs = pd.read_csv(run_path)
    if len(runs) != 16 or pilot.strict_bool_series(
            runs, "confirmation_used_for_evaluation").any():
        raise A24FormalError("A24.1 run-level freeze is incomplete or contaminated")
    audit = pd.read_csv(audit_path)
    if set(audit["method"].astype(str)) != set(METHODS):
        raise A24FormalError("A24.1 parameter audit method set mismatch")
    for column in ("shared_parameter_shapes_identical",
                   "shared_parameter_initialization_identical"):
        if not pilot.strict_bool_series(audit, column).all():
            raise A24FormalError(f"A24.1 parameter audit failed: {column}")
    return {
        "decision_sha256": pilot.sha256(decision_path),
        "manifest_sha256": pilot.sha256(manifest_path),
        "run_level_sha256": pilot.sha256(run_path),
        "parameter_audit_sha256": pilot.sha256(audit_path),
        "pilot_script_sha256": current_pilot_hash,
    }


def validate_a24_0_manifest(
    args: argparse.Namespace, contract_hashes: dict[str, str]
) -> str:
    root = resolve(args.a24_0_output_dir)
    path = root / "experimentA24_0_manifest.json"
    manifest = pilot.load_json(path)
    artifacts = manifest.get("artifacts", {})
    for name, actual_hash in contract_hashes.items():
        expected_hash = artifacts.get(name)
        if expected_hash != actual_hash:
            raise A24FormalError(f"A24.0 manifest hash mismatch: {name}")
    return pilot.sha256(path)


def validate_protocol_and_config(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> dict[str, str]:
    compatibility = pilot.a23_namespace(args)
    compatibility.shots = SHOTS
    _, _, current_hashes = pilot.a23.load_protocol(compatibility)
    locked_hashes = protocol.get("protocol_hashes", {})
    if current_hashes != locked_hashes:
        raise A24FormalError(
            f"current A23 protocol differs from A24.0 lock: "
            f"current={current_hashes} locked={locked_hashes}"
        )
    config_path = resolve(args.config)
    actual_config_hash = pilot.sha256(config_path)
    if actual_config_hash != protocol.get("config_sha256"):
        raise A24FormalError("current config differs from the A24.0 locked config")
    return current_hashes


def formal_config(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    return pilot.configure(args, protocol)


def worker_dir(root: Path, domain: str, seed: int, split: int) -> Path:
    return root / "shards" / f"{domain}_mseed{seed}_split{split}"


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experimentA24_2_run.lock"
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError):
            path.unlink()
        except PermissionError as exc:
            raise A24FormalError(f"cannot verify existing A24.2 lock: {path}") from exc
        else:
            raise A24FormalError(f"another A24.2 parent is running with pid={old_pid}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid())); handle.flush(); os.fsync(handle.fileno())
    return path


def release_lock(path: Path) -> None:
    if not path.exists():
        return
    try:
        owner = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if owner == os.getpid():
        path.unlink()


def worker_keys() -> list[tuple[str, int, int]]:
    result = [(d, s, p) for d in DOMAINS for s in MODEL_SEEDS for p in SPLITS]
    if len(result) != EXPECTED_WORKERS:
        raise A24FormalError("internal worker-grid cardinality error")
    return result


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def strict_checkpoint_roundtrip(
    path: Path, method: str, cfg: dict[str, Any], seed: int
) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if loaded.get("script_version") != SCRIPT_VERSION:
        raise A24FormalError(f"checkpoint script version mismatch: {path}")
    verifier = pilot.make_model(method, cfg, seed).cpu()
    verifier.load_state_dict(loaded["state"], strict=True)
    return loaded


def validate_worker_artifacts(
    directory: Path,
    *,
    cfg: dict[str, Any],
    target: str,
    seed: int,
    split: int,
) -> pd.DataFrame:
    """Validate a complete shard before resume-skip or final merge."""
    status = pilot.load_json(directory / "worker_status.json")
    required_status = {
        "complete": True,
        "passed": True,
        "script_version": SCRIPT_VERSION,
        "target_domain": target,
        "model_seed": seed,
        "support_split_seed": split,
        "expected_records": 10,
        "completed_records": 10,
        "selection_engines_evaluated": False,
        "confirmation_engines_evaluated": False,
        "checkpoint_reload_passed": True,
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise A24FormalError(
                f"invalid shard status {directory.name}: {key}={status.get(key)!r}"
            )
    run_path = directory / "training_run_level.csv"
    if not run_path.is_file():
        raise A24FormalError(f"missing run table: {run_path}")
    frame = pd.read_csv(run_path)
    if len(frame) != 10:
        raise A24FormalError(f"{directory.name} has {len(frame)} records, expected 10")
    expected_cells = {(shot, method) for shot in SHOTS for method in METHODS}
    actual_cells = set(zip(frame["shot"].astype(int), frame["method"].astype(str)))
    if actual_cells != expected_cells:
        raise A24FormalError(f"incomplete/duplicated method-shot grid: {directory.name}")
    for column in ("selection_engines_evaluated", "confirmation_engines_evaluated",
                   "official_test_files_accessed", "official_test_forward_run"):
        if pilot.strict_bool_series(frame, column).any():
            raise A24FormalError(f"training boundary violation {column}: {directory.name}")
    if not pilot.strict_bool_series(frame, "checkpoint_reload_passed").all():
        raise A24FormalError(f"checkpoint flag failure: {directory.name}")

    for record in frame.to_dict("records"):
        method = str(record["method"])
        checkpoint = Path(str(record["checkpoint"])).expanduser().resolve()
        try:
            checkpoint.relative_to(directory.resolve())
        except ValueError as exc:
            raise A24FormalError(
                f"checkpoint escapes worker directory: {checkpoint}"
            ) from exc
        if not checkpoint.is_file():
            raise A24FormalError(f"missing checkpoint: {checkpoint}")
        actual_hash = pilot.sha256(checkpoint)
        if actual_hash != str(record["checkpoint_sha256"]):
            raise A24FormalError(f"checkpoint hash mismatch: {checkpoint}")
        payload = strict_checkpoint_roundtrip(checkpoint, method, cfg, seed)
        checks = {
            "target_domain": target,
            "model_seed": seed,
            "support_split_seed": split,
            "shot": int(record["shot"]),
            "method": method,
            "selection_engines_evaluated": False,
            "confirmation_engines_evaluated": False,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise A24FormalError(f"checkpoint metadata mismatch {key}: {checkpoint}")

    for method in METHODS:
        meta_path = directory / f"{method}_meta_initialization.pt"
        if not meta_path.is_file():
            raise A24FormalError(f"missing meta initialization: {meta_path}")
        payload = strict_checkpoint_roundtrip(meta_path, method, cfg, seed)
        if (payload.get("target_domain") != target
                or payload.get("model_seed") != seed
                or payload.get("support_split_seed") != split
                or payload.get("outer_steps") != int(cfg["outer_steps"])
                or payload.get("inner_steps") != int(cfg["meta_inner_steps"])):
            raise A24FormalError(f"meta initialization metadata mismatch: {meta_path}")

    history_path = directory / "meta_history.csv"
    if not history_path.is_file():
        raise A24FormalError(f"missing meta history: {history_path}")
    history = pd.read_csv(history_path)
    if set(history["method"].astype(str)) != set(METHODS):
        raise A24FormalError(f"meta history method mismatch: {directory.name}")
    final_steps = history.groupby("method")["outer_step"].max()
    if not (final_steps == int(cfg["outer_steps"])).all():
        raise A24FormalError(f"incomplete Reptile history: {directory.name}")
    numeric = history[["inner_support_loss", "meta_validation_query_loss"]]
    if numeric.isna().any().any():
        raise A24FormalError(f"NaN in meta history: {directory.name}")

    audit_path = directory / "parameter_audit.csv"
    if not audit_path.is_file():
        raise A24FormalError(f"missing parameter audit: {audit_path}")
    audit = pd.read_csv(audit_path)
    if set(audit["method"].astype(str)) != set(METHODS):
        raise A24FormalError(f"parameter audit method mismatch: {directory.name}")
    for column in ("shared_parameter_shapes_identical",
                   "shared_parameter_initialization_identical"):
        if not pilot.strict_bool_series(audit, column).all():
            raise A24FormalError(f"parameter audit failure {column}: {directory.name}")
    return frame


def run_worker(args: argparse.Namespace) -> None:
    root = resolve(args.output_dir)
    directory = worker_dir(root, args.target_domain, args.model_seed, args.support_split_seed)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "worker_status.json"
    run_path = directory / "training_run_level.csv"
    freeze = load_a24_1_freeze(args)
    protocol, tasks, contract_hashes = pilot.load_contract(args)
    validate_a24_0_manifest(args, contract_hashes)
    validate_protocol_and_config(args, protocol)
    cfg = formal_config(args, protocol)
    target, seed, split = args.target_domain, args.model_seed, args.support_split_seed
    if args.resume and status_path.is_file() and run_path.is_file():
        try:
            validate_worker_artifacts(
                directory, cfg=cfg, target=target, seed=seed, split=split
            )
        except (A24FormalError, OSError, ValueError, KeyError) as exc:
            print(
                f"[A24.2] resume rejected shard {directory.name}: {exc}; rerunning",
                flush=True,
            )
        else:
            print(f"[A24.2] resume verified and skipped {directory.name}", flush=True)
            return
    subset = tasks.loc[(tasks["target_domain"] == target)
                       & (tasks["model_seed"].astype(int) == seed)
                       & (tasks["target_support_split_seed"].astype(int) == split)].copy()
    expected_tasks = ((len(DOMAINS) - 1) * 2
                      * int(protocol["episodes_per_source_domain_per_phase"]))
    if len(subset) != expected_tasks:
        raise A24FormalError(f"worker task count={len(subset)}, expected={expected_tasks}")

    compatibility = pilot.a23_namespace(args)
    compatibility.shots = SHOTS
    _, roles, a23_hashes = pilot.a23.load_protocol(compatibility)
    raw = pilot.load_raw_frames(args, protocol, cfg)
    pilot.validate_worker_tasks(subset, target=target, protocol=protocol, raw_frames=raw)
    source_domains = [domain for domain in DOMAINS if domain != target]
    normalizer = pilot.a23.fit_source_normalizer({d: raw[d] for d in source_domains})
    frames = {d: pilot.a23.normalize(raw[d], normalizer) for d in DOMAINS}

    selection = pilot.a23.role_engines(roles, target, split, "selection")
    confirmation = pilot.a23.role_engines(roles, target, split, "confirmation")
    support_by_shot = {
        shot: pilot.a23.role_engines(roles, target, split, "support_pool", shot)
        for shot in SHOTS
    }
    previous: set[int] = set()
    for shot in SHOTS:
        current = set(support_by_shot[shot])
        if len(current) != shot or not previous <= current:
            raise A24FormalError("target support sets are not correctly nested")
        if current & (set(selection) | set(confirmation)):
            raise A24FormalError("target role leakage")
        previous = current

    device = pilot.a23.resolve_device(args.device)
    parameter_rows = pilot.parameter_audit(cfg, seed)
    records: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for method in METHODS:
        meta_model = pilot.make_model(method, cfg, seed)
        meta_model, meta_history = pilot.reptile_train(
            meta_model, method, subset, frames, cfg, seed, device
        )
        histories.extend(meta_history)
        meta_state = cpu_state(meta_model)
        meta_path = directory / f"{method}_meta_initialization.pt"
        torch.save({
            "state": meta_state,
            "script_version": SCRIPT_VERSION,
            "method": method,
            "target_domain": target,
            "model_seed": seed,
            "support_split_seed": split,
            "meta_algorithm": "Reptile",
            "outer_steps": int(cfg["outer_steps"]),
            "inner_steps": int(cfg["meta_inner_steps"]),
            "contract_hashes": contract_hashes,
            "a24_1_freeze": freeze,
        }, meta_path)
        strict_checkpoint_roundtrip(meta_path, method, cfg, seed)
        del meta_model

        for shot in SHOTS:
            model = pilot.make_model(method, cfg, seed)
            model.load_state_dict(meta_state, strict=True)
            loader = pilot.a23.make_loader(
                pilot.a23.WindowDataset(
                    frames[target], support_by_shot[shot], cfg["window_size"]
                ),
                batch_size=cfg["batch_size"],
                shuffle=True,
                seed=seed * 100000 + split * 10 + shot,
            )
            model, target_history = pilot.a23.train_epochs(
                model,
                loader,
                epochs=args.target_epochs,
                learning_rate=cfg["inner_lr"],
                pair_aux_weight=cfg["pair_aux_weight"],
                device=device,
                label=(f"A24.2 {method} target={target} seed={seed} "
                       f"split={split} K={shot}"),
            )
            checkpoint = directory / f"{method}_shot{shot}_target_adapted.pt"
            torch.save({
                "state": cpu_state(model),
                "script_version": SCRIPT_VERSION,
                "method": method,
                "target_domain": target,
                "model_seed": seed,
                "support_split_seed": split,
                "shot": shot,
                "support_engines": support_by_shot[shot],
                "target_epochs": args.target_epochs,
                "meta_algorithm": "Reptile",
                "meta_initialization_checkpoint": str(meta_path),
                "contract_hashes": contract_hashes,
                "a23_protocol_hashes": a23_hashes,
                "a24_1_freeze": freeze,
                "target_history": target_history,
                "selection_engines_evaluated": False,
                "confirmation_engines_evaluated": False,
            }, checkpoint)
            strict_checkpoint_roundtrip(checkpoint, method, cfg, seed)
            records.append({
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": seed,
                "support_split_seed": split,
                "shot": shot,
                "method": method,
                "support_engines": json.dumps(support_by_shot[shot]),
                "outer_steps": int(cfg["outer_steps"]),
                "inner_steps": int(cfg["meta_inner_steps"]),
                "target_epochs": args.target_epochs,
                "final_target_training_loss": float(target_history[-1]["mean_loss"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": pilot.sha256(checkpoint),
                "checkpoint_reload_passed": True,
                "selection_engines_evaluated": False,
                "confirmation_engines_evaluated": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(records) != 10:
        raise A24FormalError(f"worker produced {len(records)} records, expected 10")
    pd.DataFrame(records).to_csv(run_path, index=False)
    pd.DataFrame(histories).to_csv(directory / "meta_history.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(directory / "parameter_audit.csv", index=False)
    pilot.a23.atomic_json(status_path, {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "complete": True,
        "passed": True,
        "target_domain": target,
        "model_seed": seed,
        "support_split_seed": split,
        "expected_records": 10,
        "completed_records": 10,
        "methods": list(METHODS),
        "shots": list(SHOTS),
        "selection_engines_evaluated": False,
        "confirmation_engines_evaluated": False,
        "checkpoint_reload_passed": True,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })


def worker_command(args: argparse.Namespace, key: tuple[str, int, int]) -> list[str]:
    domain, seed, split = key
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--data-dir", str(args.data_dir),
        "--config", str(args.config),
        "--protocol-dir", str(args.protocol_dir),
        "--a24-0-output-dir", str(args.a24_0_output_dir),
        "--a24-1-output-dir", str(args.a24_1_output_dir),
        "--output-dir", str(args.output_dir),
        "--target-domain", domain,
        "--model-seed", str(seed),
        "--support-split-seed", str(split),
        "--target-epochs", str(args.target_epochs),
        "--device", "cuda",
    ]
    for flag, value in (
        ("--batch-size", args.batch_size),
        ("--window-size", args.window_size),
        ("--rul-cap", args.rul_cap),
        ("--inner-learning-rate", args.inner_learning_rate),
        ("--pair-aux-weight", args.pair_aux_weight),
    ):
        if value is not None:
            command += [flag, str(value)]
    if args.resume:
        command.append("--resume")
    return command


def merge(args: argparse.Namespace, keys: list[tuple[str, int, int]]) -> dict[str, Any]:
    root = resolve(args.output_dir)
    protocol, _, contract_hashes = pilot.load_contract(args)
    validate_a24_0_manifest(args, contract_hashes)
    validate_protocol_and_config(args, protocol)
    cfg = formal_config(args, protocol)
    frames, audits = [], []
    for index, key in enumerate(keys, 1):
        directory = worker_dir(root, *key)
        frames.append(validate_worker_artifacts(
            directory, cfg=cfg, target=key[0], seed=key[1], split=key[2]
        ))
        audits.append(pd.read_csv(directory / "parameter_audit.csv"))
        if index % 10 == 0 or index == len(keys):
            print(f"[A24.2] merge integrity {index:03d}/{len(keys):03d}", flush=True)
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != EXPECTED_RECORDS:
        raise A24FormalError(f"merged records={len(merged)}, expected={EXPECTED_RECORDS}")
    expected_cells = pd.MultiIndex.from_product(
        [DOMAINS, MODEL_SEEDS, SPLITS, SHOTS, METHODS]
    )
    actual_cells = pd.MultiIndex.from_frame(
        merged[["target_domain", "model_seed", "support_split_seed", "shot", "method"]]
    )
    if len(actual_cells.unique()) != EXPECTED_RECORDS or set(actual_cells) != set(expected_cells):
        raise A24FormalError("formal factorial grid is incomplete or duplicated")
    for column in ("selection_engines_evaluated", "confirmation_engines_evaluated",
                   "official_test_files_accessed", "official_test_forward_run"):
        if pilot.strict_bool_series(merged, column).any():
            raise A24FormalError(f"training-only boundary violation: {column}")
    if not pilot.strict_bool_series(merged, "checkpoint_reload_passed").all():
        raise A24FormalError("checkpoint round-trip failure in merged records")

    run_path = root / "experimentA24_2_training_run_level.csv"
    audit_path = root / "experimentA24_2_parameter_audit.csv"
    merged.to_csv(run_path, index=False)
    pd.concat(audits, ignore_index=True).drop_duplicates().to_csv(audit_path, index=False)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Can the frozen A24.1 Reptile implementation complete the full locked A24.0 "
            "Meta-noGraph/Meta-GNN target-support factorial grid without using target "
            "selection, confirmation, or official-test outcomes?"
        ),
        "complete": True,
        "passed": True,
        "training_only": True,
        "expected_worker_cells": EXPECTED_WORKERS,
        "completed_worker_cells": EXPECTED_WORKERS,
        "expected_training_records": EXPECTED_RECORDS,
        "completed_training_records": EXPECTED_RECORDS,
        "methods": list(METHODS),
        "shots": list(SHOTS),
        "model_seeds": list(MODEL_SEEDS),
        "support_split_seeds": list(SPLITS),
        "selection_engines_evaluated": False,
        "confirmation_engines_evaluated": False,
        "formal_efficacy_claim": False,
        "checkpoint_reload_passed": True,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": "A24.2 completed the frozen formal meta-learning training grid",
        "interpretation_limit": (
            "A24.2 trains and freezes checkpoints only; it does not inspect target "
            "selection/confirmation outcomes or establish efficacy."
        ),
        "next_action": "run_A24_3_frozen_causal_anchor_confirmation_and_hierarchical_inference",
    }
    decision_path = root / "experimentA24_2_confirmation_decision.json"
    pilot.a23.atomic_json(decision_path, decision)
    pilot.a23.atomic_json(root / "experimentA24_2_manifest.json", {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "script_sha256": pilot.sha256(Path(__file__).resolve()),
        "a24_1_freeze": load_a24_1_freeze(args),
        "artifacts": {
            run_path.name: pilot.sha256(run_path),
            audit_path.name: pilot.sha256(audit_path),
            decision_path.name: pilot.sha256(decision_path),
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return decision


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[str, int, int]]]:
    freeze = load_a24_1_freeze(args)
    protocol, tasks, contract_hashes = pilot.load_contract(args)
    a24_0_manifest_hash = validate_a24_0_manifest(args, contract_hashes)
    current_protocol_hashes = validate_protocol_and_config(args, protocol)
    cfg = formal_config(args, protocol)
    if tuple(map(int, protocol["model_seeds"])) != MODEL_SEEDS:
        raise A24FormalError("A24.0 model seeds differ from A24.2 frozen grid")
    if tuple(map(int, protocol["target_support_split_seeds"])) != SPLITS:
        raise A24FormalError("A24.0 support splits differ from A24.2 frozen grid")
    if tuple(map(int, protocol["target_support_shots"])) != SHOTS:
        raise A24FormalError("A24.0 shots differ from A24.2 frozen grid")
    keys = worker_keys()
    compatibility = pilot.a23_namespace(args)
    compatibility.shots = SHOTS
    _, roles, _ = pilot.a23.load_protocol(compatibility)
    raw = pilot.load_raw_frames(args, protocol, cfg)
    expected_tasks = ((len(DOMAINS) - 1) * 2
                      * int(protocol["episodes_per_source_domain_per_phase"]))
    for domain, seed, split in keys:
        subset = tasks.loc[(tasks["target_domain"] == domain)
                           & (tasks["model_seed"].astype(int) == seed)
                           & (tasks["target_support_split_seed"].astype(int) == split)].copy()
        if len(subset) != expected_tasks:
            raise A24FormalError(f"preflight task count mismatch: {domain}/{seed}/{split}")
        pilot.validate_worker_tasks(subset, target=domain, protocol=protocol, raw_frames=raw)
        selection = set(pilot.a23.role_engines(roles, domain, split, "selection"))
        confirmation = set(pilot.a23.role_engines(roles, domain, split, "confirmation"))
        previous: set[int] = set()
        for shot in SHOTS:
            current = set(pilot.a23.role_engines(
                roles, domain, split, "support_pool", shot
            ))
            if len(current) != shot or not previous <= current:
                raise A24FormalError("preflight nested-shot failure")
            if current & (selection | confirmation):
                raise A24FormalError("preflight target-role leakage")
            previous = current
    smoke = pilot.model_runtime_smoke(cfg, MODEL_SEEDS[0])
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "a24_1_freeze": freeze,
        "a24_0_manifest_sha256": a24_0_manifest_hash,
        "contract_hashes": contract_hashes,
        "current_a23_protocol_hashes": current_protocol_hashes,
        "target_domains": list(DOMAINS),
        "model_seeds": list(MODEL_SEEDS),
        "support_split_seeds": list(SPLITS),
        "shots": list(SHOTS),
        "methods": list(METHODS),
        "outer_steps": int(cfg["outer_steps"]),
        "inner_steps": int(cfg["meta_inner_steps"]),
        "target_epochs": args.target_epochs,
        "expected_worker_cells": len(keys),
        "expected_training_records": len(keys) * len(SHOTS) * len(METHODS),
        "all_worker_task_subsets_preflighted": True,
        "all_nested_support_sets_preflighted": True,
        "runtime_model_smoke": smoke,
        "selection_engines_evaluated": False,
        "confirmation_engines_evaluated": False,
        "new_predictor_training": not args.dry_run,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return preview, keys


def parent(args: argparse.Namespace) -> None:
    preview, keys = preflight(args)
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("[A24.2] dry-run passed; the frozen 100-worker/1000-record grid is executable")
        return
    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        for domain, seed, split in keys:
            local = deepcopy(args)
            local.worker = True; local.target_domain = domain
            local.model_seed = seed; local.support_split_seed = split
            run_worker(local)
    else:
        requested = pilot.parse_csv_ints(args.gpus, "gpus")
        inventory = pilot.gpu_inventory()
        eligible = [row["index"] for row in inventory
                    if row["index"] in requested
                    and row["free_mb"] >= args.min_free_memory_mb
                    and row["utilization"] <= args.max_gpu_utilization]
        if not eligible:
            raise A24FormalError(f"no eligible GPU; inventory={inventory}")
        eligible = eligible[:min(args.max_workers, len(eligible))]
        pending, active = keys.copy(), {}
        while pending or active:
            for gpu in eligible:
                if gpu in active or not pending:
                    continue
                key = pending.pop(0)
                directory = worker_dir(root, *key); directory.mkdir(parents=True, exist_ok=True)
                log = directory / "worker_training.log"
                handle = log.open("w", encoding="utf-8")
                environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    worker_command(args, key), cwd=PROJECT_ROOT, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
                active[gpu] = (process, handle, key, log)
                print(f"[A24.2] launched target={key[0]} seed={key[1]} split={key[2]} "
                      f"gpu={gpu} pid={process.pid}", flush=True)
            finished = []
            for gpu, (process, handle, key, log) in active.items():
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                if code != 0:
                    for other, other_handle, _, _ in active.values():
                        if other.poll() is None:
                            other.terminate()
                        other_handle.close()
                    tail = "\n".join(log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-160:])
                    raise A24FormalError(f"worker failed key={key} exit={code}\n{tail}")
                print(f"[A24.2] completed target={key[0]} seed={key[1]} "
                      f"split={key[2]} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(3)
    merge(args, keys)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
    elif args.dry_run:
        parent(args)
    else:
        lock = acquire_lock(resolve(args.output_dir))
        try:
            parent(args)
        finally:
            release_lock(lock)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (A24FormalError, pilot.A24Error) as exc:
        print(f"[A24.2] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
