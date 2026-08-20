#!/usr/bin/env python3
"""A24.1: Reptile Meta-noGraph/Meta-GNN implementation pilot.

This is a training-only implementation pilot under the immutable A24.0 task
contract.  It is deliberately not a formal efficacy experiment.  The pilot
uses four held-out target domains, model seeds 130/131, split 7101 and K=5.
Source meta-query engines are validation-only; target selection/confirmation
engines never affect optimisation, epoch selection, or policy selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402
from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402


EXPERIMENT_ID = "experimentA24_1"
SCRIPT_VERSION = "experimentA24_1_meta_no_graph_and_meta_gnn_pilot_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
METHODS = ("meta_no_graph_k", "meta_gnn_k")
PILOT_MODEL_SEEDS = (130, 131)
PILOT_SPLITS = (7101,)
PRIMARY_SHOT = 5


class A24Error(RuntimeError):
    pass


def parse_csv_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"{name} must be non-empty and unique")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A24.1 Reptile Meta-noGraph/Meta-GNN pilot")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--protocol-dir", type=Path,
                   default=Path("outputs/experimentA23_few_shot_protocol_preflight"))
    p.add_argument("--a24-0-output-dir", type=Path,
                   default=Path("outputs/experimentA24_0_meta_learning_contract_preflight"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/experimentA24_1_meta_no_graph_and_meta_gnn_pilot"))
    p.add_argument("--model-seeds", default="130,131")
    p.add_argument("--support-split-seeds", default="7101")
    p.add_argument("--shot", type=int, default=PRIMARY_SHOT)
    p.add_argument("--outer-steps", type=int, default=None)
    p.add_argument("--inner-steps", type=int, default=None)
    p.add_argument("--outer-learning-rate", type=float, default=None)
    p.add_argument("--inner-learning-rate", type=float, default=None)
    p.add_argument("--target-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--rul-cap", type=float, default=None)
    p.add_argument("--pair-aux-weight", type=float, default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--min-free-memory-mb", type=int, default=16000)
    p.add_argument("--max-gpu-utilization", type=int, default=20)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    p.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.model_seeds = parse_csv_ints(args.model_seeds, "model-seeds")
    args.support_split_seeds = parse_csv_ints(args.support_split_seeds, "support-split-seeds")
    if args.shot <= 0 or args.target_epochs <= 0 or args.max_workers <= 0:
        raise A24Error("shot, target-epochs and max-workers must be positive")
    if args.worker and (args.target_domain is None or args.model_seed is None
                        or args.support_split_seed is None):
        raise A24Error("worker mode requires target-domain/model-seed/support-split-seed")
    return args


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A24Error(f"required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise A24Error(f"JSON root must be an object: {path}")
    return value


def load_contract(args: argparse.Namespace):
    root = resolve(args.a24_0_output_dir)
    decision_path = root / "experimentA24_0_confirmation_decision.json"
    protocol_path = root / "experimentA24_0_meta_protocol.json"
    tasks_path = root / "experimentA24_0_source_meta_task_inventory.csv"
    roles_audit_path = root / "experimentA24_0_engine_role_audit.csv"
    decision = load_json(decision_path)
    protocol = load_json(protocol_path)
    if not (decision.get("complete") and decision.get("passed")):
        raise A24Error("A24.0 must be complete and passed")
    if decision.get("new_predictor_training"):
        raise A24Error("A24.0 unexpectedly reports predictor training")
    if protocol.get("meta_algorithm") != "Reptile":
        raise A24Error("A24.0 meta_algorithm is not Reptile")
    tasks = pd.read_csv(tasks_path)
    audits = pd.read_csv(roles_audit_path)
    if len(tasks) != int(protocol["expected_source_meta_tasks"]):
        raise A24Error("A24.0 task inventory row count mismatch")
    required_true = ("target_domain_excluded_from_meta_train", "support_query_disjoint",
                     "meta_train_validation_engine_disjoint")
    for col in required_true:
        if col not in tasks or not tasks[col].astype(bool).all():
            raise A24Error(f"A24.0 task contract violation: {col}")
    for col in ("engine_selection_uses_labels", "engine_selection_uses_trajectory_length",
                "official_test_files_accessed"):
        if col not in tasks or tasks[col].astype(bool).any():
            raise A24Error(f"A24.0 task contract violation: {col}")
    if not audits["all_target_role_sets_disjoint"].astype(bool).all():
        raise A24Error("target engine roles are not disjoint")
    if audits["target_engines_allowed_in_meta_training"].astype(bool).any():
        raise A24Error("target engines are allowed in meta training")
    hashes = {p.name: sha256(p) for p in (decision_path, protocol_path, tasks_path, roles_audit_path)}
    return protocol, tasks, hashes


def configure(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    local = deepcopy(args)
    local.config = resolve(args.config)
    # A23.1 owns the already validated configuration-normalisation helper.
    # Supply its complete argument contract explicitly: A24.1 intentionally
    # uses a differently named inner-learning-rate option and has no ordinary
    # source-pretraining learning rate.
    local.window_size = args.window_size
    local.rul_cap = args.rul_cap
    local.learning_rate = args.inner_learning_rate
    local.source_learning_rate = None
    local.pair_aux_weight = args.pair_aux_weight
    cfg = a23.load_config(local)
    cfg["batch_size"] = int(args.batch_size or cfg["batch_size"])
    cfg["inner_lr"] = float(args.inner_learning_rate or cfg["inner_lr"])
    cfg["outer_steps"] = int(args.outer_steps or protocol["outer_steps"])
    cfg["meta_inner_steps"] = int(args.inner_steps or protocol["inner_steps"])
    cfg["outer_lr"] = float(args.outer_learning_rate or protocol["outer_learning_rate"])
    for key in ("batch_size", "outer_steps", "meta_inner_steps"):
        if int(cfg[key]) <= 0:
            raise A24Error(f"{key} must be positive")
    if not (0.0 < cfg["outer_lr"] <= 1.0 and cfg["inner_lr"] > 0.0):
        raise A24Error("invalid inner/outer learning rate")
    return cfg


def make_model(method: str, cfg: dict[str, Any], seed: int) -> torch.nn.Module:
    a23.seed_everything(seed)
    model = build_model("gnn", len(a23.FEATURE_COLUMNS), cfg)
    if not hasattr(model, "use_gat") or not hasattr(model, "gat"):
        raise A24Error("build_model('gnn') does not expose use_gat/gat required by A24")
    if method == "meta_no_graph_k":
        model.use_gat = False
        model.gat = torch.nn.Identity()
    elif method == "meta_gnn_k":
        if not bool(model.use_gat):
            raise A24Error("Meta-GNN was constructed with use_gat=False")
    else:
        raise A24Error(f"unknown method: {method}")
    return model


def parameter_audit(cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    models = {m: make_model(m, cfg, seed) for m in METHODS}
    named = {m: dict(model.named_parameters()) for m, model in models.items()}
    shared = sorted(set(named[METHODS[0]]) & set(named[METHODS[1]]))
    if not shared:
        raise A24Error("Meta-noGraph and Meta-GNN have no shared parameter keys")
    mismatch = [k for k in shared if tuple(named[METHODS[0]][k].shape) != tuple(named[METHODS[1]][k].shape)]
    if mismatch:
        raise A24Error(f"shared parameter shape mismatch: {mismatch[:10]}")
    graph_only = sorted(set(named["meta_gnn_k"]) - set(named["meta_no_graph_k"]))
    if not graph_only or not all(name.startswith("gat.") for name in graph_only):
        raise A24Error(f"unexpected Meta-GNN-only parameters: {graph_only[:10]}")
    no_graph_only = sorted(set(named["meta_no_graph_k"]) - set(named["meta_gnn_k"]))
    if no_graph_only:
        raise A24Error(f"unexpected Meta-noGraph-only parameters: {no_graph_only[:10]}")
    rows = []
    for method, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        shared_count = sum(named[method][k].numel() for k in shared)
        graph_count = sum(named[method][k].numel() for k in graph_only if k in named[method])
        rows.append({"method": method, "total_parameters": total,
                     "trainable_parameters": trainable, "shared_parameters": shared_count,
                     "graph_increment_parameters": graph_count,
                     "shared_parameter_shapes_identical": True})
    if rows[0]["shared_parameters"] != rows[1]["shared_parameters"]:
        raise A24Error("shared parameter counts differ")
    return rows


def parse_engine_list(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise A24Error("episode engine list must be a non-empty JSON list")
    result = [int(x) for x in parsed]
    if len(result) != len(set(result)):
        raise A24Error("episode engine list contains duplicates")
    return result


def loss_on_loader(model, loader, device: torch.device, update: bool,
                   steps: int, lr: float, pair_weight: float) -> float:
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr) if update else None
    iterator = iter(loader)
    losses: list[float] = []
    model.train(update)
    context = torch.enable_grad() if update else torch.no_grad()
    with context:
        for _ in range(steps):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y, _, _ = next(iterator)
            x, y = x.to(device), y.to(device)
            if optimiser is not None:
                optimiser.zero_grad(set_to_none=True)
            loss, _ = rul_training_loss(model, x, y, pair_weight)
            if not torch.isfinite(loss):
                raise A24Error("non-finite meta loss")
            if optimiser is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def reptile_train(model, method: str, task_rows: pd.DataFrame,
                  frames: dict[str, pd.DataFrame], cfg: dict[str, Any],
                  seed: int, device: torch.device):
    train_rows = task_rows.loc[task_rows["episode_phase"] == "meta_train"].reset_index(drop=True)
    val_rows = task_rows.loc[task_rows["episode_phase"] == "meta_validation"].reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        raise A24Error("worker contract lacks meta_train or meta_validation tasks")
    rng = random.Random(seed + 240100)
    order = list(range(len(train_rows)))
    history: list[dict[str, Any]] = []
    report_every = max(1, int(cfg["outer_steps"]) // 10)
    model = model.cpu()
    for outer in range(1, int(cfg["outer_steps"]) + 1):
        if not order:
            order = list(range(len(train_rows)))
            rng.shuffle(order)
        row = train_rows.iloc[order.pop()]
        support = parse_engine_list(row["meta_support_engines"])
        query = parse_engine_list(row["meta_query_engines"])
        if set(support) & set(query) or row["source_domain"] == row["target_domain"]:
            raise A24Error("episode leakage detected at runtime")
        ds = a23.WindowDataset(frames[row["source_domain"]], support, cfg["window_size"])
        loader = a23.make_loader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                 seed=seed * 100000 + outer)
        adapted = deepcopy(model)
        inner_loss = loss_on_loader(adapted, loader, device, True,
                                    int(cfg["meta_inner_steps"]), cfg["inner_lr"],
                                    cfg["pair_aux_weight"])
        adapted = adapted.cpu()
        with torch.no_grad():
            base_params = dict(model.named_parameters())
            adapted_params = dict(adapted.named_parameters())
            for name, parameter in base_params.items():
                parameter.add_(float(cfg["outer_lr"]) * (adapted_params[name] - parameter))
        if outer % report_every == 0 or outer == int(cfg["outer_steps"]):
            vr = val_rows.iloc[(outer // report_every - 1) % len(val_rows)]
            vs = parse_engine_list(vr["meta_support_engines"])
            vq = parse_engine_list(vr["meta_query_engines"])
            probe = deepcopy(model)
            sl = a23.make_loader(a23.WindowDataset(frames[vr["source_domain"]], vs, cfg["window_size"]),
                                 batch_size=cfg["batch_size"], shuffle=True, seed=seed + outer + 1)
            ql = a23.make_loader(a23.WindowDataset(frames[vr["source_domain"]], vq, cfg["window_size"]),
                                 batch_size=cfg["batch_size"], shuffle=False, seed=seed + outer + 2)
            loss_on_loader(probe, sl, device, True, int(cfg["meta_inner_steps"]),
                           cfg["inner_lr"], cfg["pair_aux_weight"])
            query_loss = loss_on_loader(probe, ql, device, False, 1,
                                        cfg["inner_lr"], cfg["pair_aux_weight"])
            history.append({"method": method, "outer_step": outer,
                            "inner_support_loss": inner_loss,
                            "meta_validation_query_loss": query_loss})
            print(f"[A24.1] {method} outer={outer:04d}/{cfg['outer_steps']} "
                  f"support={inner_loss:.6f} val_query={query_loss:.6f}", flush=True)
            del probe
        del adapted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model.cpu(), history


def worker_directory(root: Path, domain: str, seed: int, split: int) -> Path:
    return root / "shards" / f"{domain}_mseed{seed}_split{split}"


def run_worker(args: argparse.Namespace) -> None:
    out = worker_directory(resolve(args.output_dir), args.target_domain,
                           args.model_seed, args.support_split_seed)
    out.mkdir(parents=True, exist_ok=True)
    status_path, run_path = out / "worker_status.json", out / "run_level.csv"
    if args.resume and status_path.is_file() and run_path.is_file():
        status = load_json(status_path)
        if status.get("complete") and status.get("passed"):
            print(f"[A24.1] resume skip {out.name}", flush=True)
            return
    protocol, tasks, hashes = load_contract(args)
    cfg = configure(args, protocol)
    audit_rows = parameter_audit(cfg, args.model_seed)
    target, seed, split = args.target_domain, args.model_seed, args.support_split_seed
    subset = tasks.loc[(tasks["target_domain"] == target)
                       & (tasks["model_seed"].astype(int) == seed)
                       & (tasks["target_support_split_seed"].astype(int) == split)].copy()
    expected = (len(DOMAINS) - 1) * 2 * int(protocol["episodes_per_source_domain_per_phase"])
    if len(subset) != expected:
        raise A24Error(f"worker has {len(subset)} tasks, expected {expected}")

    local = deepcopy(args)
    local.protocol_dir = resolve(args.protocol_dir)
    a23_protocol, roles, a23_hashes = a23.load_protocol(local)
    data_dir = resolve(args.data_dir)
    raw = {d: a23.load_domain_frame(a23.resolve_train_file(data_dir, d), rul_cap=cfg["rul_cap"])
           for d in DOMAINS}
    source_domains = [d for d in DOMAINS if d != target]
    normalizer = a23.fit_source_normalizer({d: raw[d] for d in source_domains})
    frames = {d: a23.normalize(raw[d], normalizer) for d in DOMAINS}
    support_engines = a23.role_engines(roles, target, split, "support_pool", args.shot)
    selection = a23.role_engines(roles, target, split, "selection")
    confirmation = a23.role_engines(roles, target, split, "confirmation")
    if len(support_engines) != args.shot or (set(support_engines) & (set(selection) | set(confirmation))):
        raise A24Error("target support role mismatch or leakage")
    device = a23.resolve_device(args.device)
    confirmation_loader = a23.make_loader(
        a23.WindowDataset(frames[target], confirmation, cfg["window_size"]),
        batch_size=cfg["batch_size"], shuffle=False, seed=seed + split)
    rows, histories = [], []
    for method in METHODS:
        model = make_model(method, cfg, seed)
        model, history = reptile_train(model, method, subset, frames, cfg, seed, device)
        histories.extend(history)
        support_loader = a23.make_loader(
            a23.WindowDataset(frames[target], support_engines, cfg["window_size"]),
            batch_size=cfg["batch_size"], shuffle=True, seed=seed * 10000 + split)
        model, target_history = a23.train_epochs(
            model, support_loader, epochs=args.target_epochs, learning_rate=cfg["inner_lr"],
            pair_aux_weight=cfg["pair_aux_weight"], device=device,
            label=f"A24.1 {method} target={target} seed={seed} split={split} K={args.shot}")
        metrics = a23.evaluate(model, confirmation_loader, device)
        checkpoint = out / f"{method}_shot{args.shot}.pt"
        payload = {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "method": method, "target_domain": target, "model_seed": seed,
                   "support_split_seed": split, "shot": args.shot,
                   "support_engines": support_engines, "target_epochs": args.target_epochs,
                   "meta_algorithm": "Reptile", "contract_hashes": hashes,
                   "a23_protocol_hashes": a23_hashes, "target_history": target_history}
        torch.save(payload, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        verifier = make_model(method, cfg, seed)
        verifier.load_state_dict(loaded["state"], strict=True)
        row = {"target_domain": target, "model_seed": seed, "support_split_seed": split,
               "shot": args.shot, "method": method, "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(checkpoint), "checkpoint_reload_passed": True,
               "confirmation_used_for_training": False, "selection_used_for_training": False,
               "official_test_files_accessed": False, "official_test_forward_run": False}
        row.update(a23.flatten_metrics("confirmation", metrics))
        rows.append(row)
        del model, verifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(run_path, index=False)
    pd.DataFrame(histories).to_csv(out / "meta_history.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "parameter_audit.csv", index=False)
    a23.atomic_json(status_path, {"experiment_id": EXPERIMENT_ID, "complete": True,
                                  "passed": len(rows) == len(METHODS), "target_domain": target,
                                  "model_seed": seed, "support_split_seed": split,
                                  "completed_methods": len(rows), "expected_methods": len(METHODS),
                                  "checkpoint_reload_passed": True,
                                  "new_predictor_training": True,
                                  "official_test_files_accessed": False,
                                  "official_test_forward_run": False})


def gpu_inventory() -> list[dict[str, int]]:
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        return [{"index": int(a), "free_mb": int(b), "utilization": int(c)}
                for a, b, c in (line.split(",") for line in text.strip().splitlines())]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def command(args, domain, seed, split):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--data-dir", str(args.data_dir), "--config", str(args.config),
           "--protocol-dir", str(args.protocol_dir), "--a24-0-output-dir", str(args.a24_0_output_dir),
           "--output-dir", str(args.output_dir), "--target-domain", domain,
           "--model-seed", str(seed), "--support-split-seed", str(split),
           "--model-seeds", ",".join(map(str, args.model_seeds)),
           "--support-split-seeds", ",".join(map(str, args.support_split_seeds)),
           "--shot", str(args.shot), "--target-epochs", str(args.target_epochs),
           "--device", "cuda"]
    for flag, value in (("--outer-steps", args.outer_steps), ("--inner-steps", args.inner_steps),
                        ("--outer-learning-rate", args.outer_learning_rate),
                        ("--inner-learning-rate", args.inner_learning_rate),
                        ("--batch-size", args.batch_size),
                        ("--window-size", args.window_size),
                        ("--rul-cap", args.rul_cap),
                        ("--pair-aux-weight", args.pair_aux_weight)):
        if value is not None:
            cmd += [flag, str(value)]
    if args.resume:
        cmd.append("--resume")
    return cmd


def merge(args, workers):
    root = resolve(args.output_dir)
    frames, audits = [], []
    for domain, seed, split in workers:
        d = worker_directory(root, domain, seed, split)
        status = load_json(d / "worker_status.json")
        if not (status.get("complete") and status.get("passed")):
            raise A24Error(f"incomplete worker: {d.name}")
        frames.append(pd.read_csv(d / "run_level.csv"))
        audits.append(pd.read_csv(d / "parameter_audit.csv"))
    merged = pd.concat(frames, ignore_index=True)
    expected = len(workers) * len(METHODS)
    if len(merged) != expected or merged["checkpoint_reload_passed"].astype(bool).sum() != expected:
        raise A24Error("merged pilot record/checkpoint count mismatch")
    if merged[["confirmation_used_for_training", "selection_used_for_training",
               "official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any():
        raise A24Error("merged integrity violation")
    run_path = root / "experimentA24_1_run_level.csv"
    audit_path = root / "experimentA24_1_parameter_audit.csv"
    merged.to_csv(run_path, index=False)
    pd.concat(audits, ignore_index=True).drop_duplicates().to_csv(audit_path, index=False)
    decision = {"experiment_id": EXPERIMENT_ID, "complete": True, "passed": True,
                "pilot_only": True, "meta_algorithm": "Reptile",
                "methods": list(METHODS), "shot": args.shot,
                "expected_worker_cells": len(workers), "completed_worker_cells": len(workers),
                "expected_run_records": expected, "completed_run_records": len(merged),
                "runtime_shared_parameter_assertion_passed": True,
                "checkpoint_reload_passed": True, "pilot_efficacy_claim": False,
                "new_predictor_training": True, "official_test_files_accessed": False,
                "official_test_forward_run": False,
                "reason": "A24.1 completed the Reptile Meta-noGraph/Meta-GNN implementation pilot",
                "interpretation_limit": "Pilot metrics are implementation diagnostics, not formal efficacy evidence.",
                "next_action": "freeze_A24_1_implementation_then_run_A24_2_formal_factorial_grid"}
    a23.atomic_json(root / "experimentA24_1_confirmation_decision.json", decision)
    a23.atomic_json(root / "experimentA24_1_manifest.json",
                    {"experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
                     "script_sha256": sha256(Path(__file__).resolve()),
                     "artifacts": {run_path.name: sha256(run_path), audit_path.name: sha256(audit_path)},
                     "official_test_files_accessed": False, "official_test_forward_run": False})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def parent(args):
    protocol, tasks, hashes = load_contract(args)
    cfg = configure(args, protocol)
    audit = parameter_audit(cfg, args.model_seeds[0])
    allowed_seeds = set(map(int, protocol["model_seeds"]))
    allowed_splits = set(map(int, protocol["target_support_split_seeds"]))
    if not set(args.model_seeds) <= allowed_seeds or not set(args.support_split_seeds) <= allowed_splits:
        raise A24Error("pilot seeds/splits are outside the locked A24.0 contract")
    if args.shot != int(protocol["primary_shot"]):
        raise A24Error("A24.1 pilot must use locked primary_shot=5")
    workers = [(d, s, p) for d in DOMAINS for s in args.model_seeds for p in args.support_split_seeds]
    preview = {"experiment_id": EXPERIMENT_ID, "script_version": SCRIPT_VERSION,
               "meta_algorithm": "Reptile", "contract_hashes": hashes,
               "methods": list(METHODS), "target_domains": list(DOMAINS),
               "model_seeds": list(args.model_seeds),
               "support_split_seeds": list(args.support_split_seeds), "shot": args.shot,
               "outer_steps": cfg["outer_steps"], "inner_steps": cfg["meta_inner_steps"],
               "outer_learning_rate": cfg["outer_lr"], "inner_learning_rate": cfg["inner_lr"],
               "expected_worker_cells": len(workers),
               "expected_run_records": len(workers) * len(METHODS),
               "runtime_parameter_audit": audit, "new_predictor_training": not args.dry_run,
               "official_test_files_accessed": False, "official_test_forward_run": False}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("[A24.1] dry-run passed; model schemas and locked contracts are compatible")
        return
    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        for d, s, p in workers:
            local = deepcopy(args); local.worker = True; local.target_domain = d
            local.model_seed = s; local.support_split_seed = p
            run_worker(local)
    else:
        requested = parse_csv_ints(args.gpus, "gpus")
        inventory = gpu_inventory()
        eligible = [x["index"] for x in inventory if x["index"] in requested
                    and x["free_mb"] >= args.min_free_memory_mb
                    and x["utilization"] <= args.max_gpu_utilization]
        if not eligible:
            raise A24Error(f"no eligible GPU; inventory={inventory}")
        eligible = eligible[:min(args.max_workers, len(eligible))]
        pending, active = workers.copy(), {}
        while pending or active:
            for gpu in eligible:
                if gpu in active or not pending:
                    continue
                item = pending.pop(0); d = worker_directory(root, *item); d.mkdir(parents=True, exist_ok=True)
                handle = (d / "worker_training.log").open("a", encoding="utf-8")
                env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                proc = subprocess.Popen(command(args, *item), cwd=PROJECT_ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT, text=True)
                active[gpu] = (proc, handle, item, d / "worker_training.log")
                print(f"[A24.1] launched target={item[0]} seed={item[1]} split={item[2]} gpu={gpu} pid={proc.pid}", flush=True)
            finished = []
            for gpu, (proc, handle, item, log) in active.items():
                code = proc.poll()
                if code is None: continue
                handle.close()
                if code:
                    for p, h, _, _ in active.values():
                        if p.poll() is None: p.terminate()
                        h.close()
                    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                    raise A24Error(f"worker failed item={item} exit={code}\n{tail}")
                print(f"[A24.1] completed target={item[0]} seed={item[1]} split={item[2]} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished: del active[gpu]
            if active and not finished: time.sleep(3)
    merge(args, workers)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker: run_worker(args)
    else: parent(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A24Error as exc:
        print(f"[A24.1] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
