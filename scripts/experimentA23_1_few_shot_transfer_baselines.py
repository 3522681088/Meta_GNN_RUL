#!/usr/bin/env python3
"""Experiment A23.1: training-only few-shot transfer-learning baselines.

This script is the first predictive experiment after A23.0.  It consumes the
immutable engine-level protocol produced by A23.0 and compares three matched
baselines on confirmation engines that have never been used for fitting,
normalisation, epoch selection, or source pretraining:

``source_only``
    Multi-source supervised pretraining followed by direct target inference.
``scratch_k``
    Random initialisation trained only on the k labelled target engines.
``pretrain_finetune_k``
    The same multi-source initialisation adapted on the same k engines.

The default is a deliberately small pilot (k=1/5/20, model seeds=130/131,
one locked support split).  It never opens C-MAPSS test files or RUL test-label
files.  It is not a meta-learning experiment; A24 is reserved for the
Meta-noGraph / Reptile Meta-GNN comparison after this baseline pilot passes.
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
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402


EXPERIMENT_ID = "experimentA23_1"
SCRIPT_VERSION = "experimentA23_1_few_shot_transfer_baselines_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
REGIMES = ("source_only", "scratch_k", "pretrain_finetune_k")
PILOT_SHOTS = (1, 5, 20)
PILOT_MODEL_SEEDS = (130, 131)
PILOT_SPLIT_SEEDS = (7101,)
FEATURE_COLUMNS = (
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14",
    "s15", "s17", "s20", "s21", "op_setting1", "op_setting2", "op_setting3",
)
RUL_COLUMNS = ("unit", "cycle", "op_setting1", "op_setting2", "op_setting3") + tuple(
    f"s{i}" for i in range(1, 22)
)
HIGH_RUL_THRESHOLD = 60.0


class A23Error(RuntimeError):
    """Raised for a protocol or training-integrity failure."""


def parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from exc
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must be non-empty and unique")
    return parsed


def parse_str_list(value: str, *, name: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must be non-empty and unique")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A23.1 training-only engine-level few-shot transfer baselines"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA23_1_few_shot_transfer_baselines"),
    )
    parser.add_argument("--shots", default=",".join(map(str, PILOT_SHOTS)))
    parser.add_argument("--model-seeds", default=",".join(map(str, PILOT_MODEL_SEEDS)))
    parser.add_argument("--support-split-seeds", default=",".join(map(str, PILOT_SPLIT_SEEDS)))
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--rul-cap", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--source-learning-rate", type=float, default=None)
    parser.add_argument("--pair-aux-weight", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpus", default="0", help="Physical GPU ids for the parent scheduler, e.g. 5,6,7")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    parser.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.shots = parse_int_list(args.shots, name="shots")
    args.model_seeds = parse_int_list(args.model_seeds, name="model-seeds")
    args.support_split_seeds = parse_int_list(
        args.support_split_seeds, name="support-split-seeds"
    )
    args.regimes = parse_str_list(args.regimes, name="regimes")
    unknown = set(args.regimes) - set(REGIMES)
    if unknown:
        raise A23Error(f"unsupported regimes: {sorted(unknown)}")
    if any(shot <= 0 for shot in args.shots):
        raise A23Error("A23.1 baselines require positive target-engine shot counts")
    if args.source_pretrain_steps <= 0 or args.target_epochs <= 0:
        raise A23Error("source-pretrain-steps and target-epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise A23Error("batch-size must be positive")
    if args.window_size is not None and args.window_size < 2:
        raise A23Error("window-size must be at least 2")
    if args.max_workers is not None and args.max_workers <= 0:
        raise A23Error("max-workers must be positive")
    if args.worker:
        if not (args.target_domain and args.model_seed is not None and args.support_split_seed is not None):
            raise A23Error("worker mode requires target-domain, model-seed and support-split-seed")
    return args


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
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def resolve_train_file(data_dir: Path, domain: str) -> Path:
    name = f"train_{domain}.txt"
    direct = data_dir / name
    if direct.is_file():
        return direct.resolve()
    matches = sorted(data_dir.rglob(name)) if data_dir.is_dir() else []
    if len(matches) != 1:
        raise A23Error(f"could not uniquely resolve {name} below {data_dir}; matches={matches}")
    return matches[0].resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise A23Error("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve(PROJECT_ROOT, args.config)
    if not config_path.is_file():
        raise A23Error(f"config is missing: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise A23Error(f"config must contain a mapping: {config_path}")
    cfg = deepcopy(cfg)
    cfg["batch_size"] = int(args.batch_size or cfg.get("batch_size", 64))
    cfg["window_size"] = int(args.window_size or cfg.get("window_size", cfg.get("seq_len", 30)))
    cfg["rul_cap"] = float(args.rul_cap if args.rul_cap is not None else cfg.get("rul_cap", cfg.get("max_rul", 125.0)))
    cfg["inner_lr"] = float(args.learning_rate if args.learning_rate is not None else cfg.get("inner_lr", cfg.get("learning_rate", 1e-3)))
    cfg["source_learning_rate"] = float(args.source_learning_rate if args.source_learning_rate is not None else cfg.get("source_pretrain_lr", cfg["inner_lr"]))
    cfg["pair_aux_weight"] = float(args.pair_aux_weight if args.pair_aux_weight is not None else cfg.get("pair_aux_weight", 0.0))
    if cfg["window_size"] < 2 or cfg["batch_size"] < 1 or cfg["rul_cap"] <= 0:
        raise A23Error("invalid window_size, batch_size or rul_cap in configuration")
    return cfg


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    protocol_dir = resolve(PROJECT_ROOT, args.protocol_dir)
    protocol_path = protocol_dir / "experimentA23_few_shot_protocol.json"
    roles_path = protocol_dir / "experimentA23_engine_roles.csv"
    decision_path = protocol_dir / "experimentA23_confirmation_decision.json"
    for path in (protocol_path, roles_path, decision_path):
        if not path.is_file():
            raise A23Error(f"required A23.0 artifact is missing: {path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not (decision.get("complete") and decision.get("passed")):
        raise A23Error("A23.0 protocol must be complete and passed before baseline training")
    if decision.get("official_test_files_accessed") or decision.get("official_test_forward_run"):
        raise A23Error("A23.0 protocol violates training-only official-test boundary")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    roles = pd.read_csv(roles_path)
    required = {"target_domain", "support_split_seed", "engine_id", "role", "support_rank"}
    if missing := required - set(roles.columns):
        raise A23Error(f"engine role table lacks columns: {sorted(missing)}")
    known_shots = {int(value) for value in protocol.get("shot_counts", [])}
    missing_shots = set(args.shots) - known_shots
    if missing_shots:
        raise A23Error(f"requested shots are not locked by A23.0: {sorted(missing_shots)}")
    known_splits = {int(value) for value in protocol.get("support_split_seeds", [])}
    missing_splits = set(args.support_split_seeds) - known_splits
    if missing_splits:
        raise A23Error(f"requested split seeds are not locked by A23.0: {sorted(missing_splits)}")
    digests = {
        protocol_path.name: sha256_file(protocol_path),
        roles_path.name: sha256_file(roles_path),
        decision_path.name: sha256_file(decision_path),
    }
    return protocol, roles, digests


def load_domain_frame(path: Path, *, rul_cap: float) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception as exc:
        raise A23Error(f"failed to parse C-MAPSS training data {path}: {exc}") from exc
    if frame.shape[1] != len(RUL_COLUMNS):
        raise A23Error(f"unexpected C-MAPSS column count in {path}: {frame.shape[1]}")
    frame.columns = RUL_COLUMNS
    frame["unit"] = pd.to_numeric(frame["unit"], errors="raise").astype(int)
    frame["cycle"] = pd.to_numeric(frame["cycle"], errors="raise").astype(int)
    frame = frame.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["unit", "cycle"]).any():
        raise A23Error(f"duplicate unit/cycle rows in {path}")
    frame["raw_rul"] = frame.groupby("unit")["cycle"].transform("max") - frame["cycle"]
    frame["rul"] = frame["raw_rul"].clip(upper=float(rul_cap)).astype(np.float32)
    return frame


def fit_source_normalizer(source_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    source = pd.concat([frame.loc[:, FEATURE_COLUMNS] for frame in source_frames.values()], ignore_index=True)
    means = source.mean(axis=0)
    stds = source.std(axis=0, ddof=0).replace(0.0, 1.0)
    return {
        "mean": {column: float(means[column]) for column in FEATURE_COLUMNS},
        "std": {column: float(stds[column]) for column in FEATURE_COLUMNS},
    }


def normalize(frame: pd.DataFrame, state: dict[str, dict[str, float]]) -> pd.DataFrame:
    copy = frame.copy()
    for column in FEATURE_COLUMNS:
        copy[column] = (copy[column].astype(np.float32) - state["mean"][column]) / state["std"][column]
    return copy


class WindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, engine_ids: Sequence[int], window_size: int) -> None:
        self.x: list[np.ndarray] = []
        self.y: list[float] = []
        self.unit: list[int] = []
        self.cycle: list[int] = []
        wanted = set(int(value) for value in engine_ids)
        if not wanted:
            raise A23Error("attempted to construct a dataset with zero engines")
        for unit, group in frame.loc[frame["unit"].isin(wanted)].groupby("unit", sort=True):
            group = group.sort_values("cycle", kind="stable")
            features = group.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            labels = group["rul"].to_numpy(dtype=np.float32, copy=True)
            cycles = group["cycle"].to_numpy(dtype=np.int64, copy=True)
            if len(features) == 0:
                raise A23Error(f"unit={unit} has no observations")
            for end in range(len(features)):
                start = max(0, end - window_size + 1)
                segment = features[start : end + 1]
                if len(segment) < window_size:
                    pad = np.repeat(segment[:1], repeats=window_size - len(segment), axis=0)
                    segment = np.concatenate([pad, segment], axis=0)
                self.x.append(segment)
                self.y.append(float(labels[end]))
                self.unit.append(int(unit))
                self.cycle.append(int(cycles[end]))
        if not self.x:
            raise A23Error("dataset contains no causal windows")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index]),
            torch.tensor(self.y[index], dtype=torch.float32),
            torch.tensor(self.unit[index], dtype=torch.int64),
            torch.tensor(self.cycle[index], dtype=torch.int64),
        )


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def role_engines(roles: pd.DataFrame, domain: str, split_seed: int, role: str, shot: int | None = None) -> list[int]:
    frame = roles.loc[
        (roles["target_domain"] == domain)
        & (roles["support_split_seed"].astype(int) == int(split_seed))
        & (roles["role"] == role)
    ].copy()
    if role == "support_pool":
        if shot is None:
            raise A23Error("support role requires a shot count")
        frame["support_rank"] = pd.to_numeric(frame["support_rank"], errors="raise").astype(int)
        frame = frame.loc[frame["support_rank"] <= int(shot)]
    return sorted(int(value) for value in frame["engine_id"].tolist())


def train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    pair_aux_weight: float,
    device: torch.device,
    label: str,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    learner = model.to(device)
    optimiser = torch.optim.Adam(learner.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    learner.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss, _ = rul_training_loss(learner, x, y, pair_aux_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
            optimiser.step()
            losses.append(float(loss.detach().item()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": float(epoch), "mean_loss": mean_loss})
        print(f"[A23.1] {label} epoch={epoch:02d}/{epochs} loss={mean_loss:.6f}", flush=True)
    return learner, history


def train_source_steps(
    model: torch.nn.Module,
    source_loaders: dict[str, DataLoader],
    *,
    steps: int,
    learning_rate: float,
    pair_aux_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    learner = model.to(device)
    optimiser = torch.optim.Adam(learner.parameters(), lr=learning_rate)
    task_names = sorted(source_loaders)
    iterators = {name: iter(source_loaders[name]) for name in task_names}
    schedule_rng = random.Random(int(seed) + 230101)
    schedule: list[str] = []
    history: list[dict[str, float]] = []
    losses: list[float] = []
    report_every = max(1, steps // 10)
    learner.train()
    for step in range(1, steps + 1):
        if not schedule:
            schedule = task_names.copy()
            schedule_rng.shuffle(schedule)
        domain = schedule.pop()
        try:
            x, y, _, _ = next(iterators[domain])
        except StopIteration:
            iterators[domain] = iter(source_loaders[domain])
            x, y, _, _ = next(iterators[domain])
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss, _ = rul_training_loss(learner, x, y, pair_aux_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
        optimiser.step()
        losses.append(float(loss.detach().item()))
        if step % report_every == 0 or step == steps:
            mean_loss = float(np.mean(losses))
            history.append({"source_step": float(step), "mean_source_loss": mean_loss})
            print(f"[A23.1] source step={step:04d}/{steps} loss={mean_loss:.6f}", flush=True)
            losses.clear()
    return learner, history


def nasa_score(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction - truth
    under = error < 0.0
    return float(np.sum(np.exp(-error[under] / 13.0) - 1.0) + np.sum(np.exp(error[~under] / 10.0) - 1.0))


def metric_block(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if len(truth) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "nasa_score": float("nan"), "mean_error": float("nan"), "n": 0}
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": nasa_score(prediction, truth),
        "mean_error": float(np.mean(error)),
        "n": int(len(truth)),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    units: list[np.ndarray] = []
    cycles: list[np.ndarray] = []
    for x, y, unit, cycle in loader:
        pred = model(x.to(device)).detach().cpu().numpy().reshape(-1)
        predictions.append(pred)
        truths.append(y.numpy().reshape(-1))
        units.append(unit.numpy().reshape(-1))
        cycles.append(cycle.numpy().reshape(-1))
    pred = np.concatenate(predictions)
    truth = np.concatenate(truths)
    unit = np.concatenate(units)
    cycle = np.concatenate(cycles)
    high = truth > HIGH_RUL_THRESHOLD
    full = metric_block(pred, truth)
    result = {
        "all_windows": full,
        "high_rul_gt60": metric_block(pred[high], truth[high]),
        "low_or_mid_rul_le60": metric_block(pred[~high], truth[~high]),
    }
    endpoint_mask = np.zeros(len(unit), dtype=bool)
    for engine in np.unique(unit):
        locations = np.flatnonzero(unit == engine)
        endpoint_mask[locations[np.argmax(cycle[locations])]] = True
    endpoint_pred, endpoint_truth = pred[endpoint_mask], truth[endpoint_mask]
    endpoint_high = endpoint_truth > HIGH_RUL_THRESHOLD
    result["engine_endpoint"] = metric_block(endpoint_pred, endpoint_truth)
    result["engine_endpoint_high_rul_gt60"] = metric_block(endpoint_pred[endpoint_high], endpoint_truth[endpoint_high])
    result["engine_endpoint_low_or_mid_rul_le60"] = metric_block(endpoint_pred[~endpoint_high], endpoint_truth[~endpoint_high])
    return result


def flatten_metrics(prefix: str, metrics: dict[str, dict[str, float]]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for stage, values in metrics.items():
        for name, value in values.items():
            output[f"{prefix}_{stage}_{name}"] = value
    return output


def worker_dir(root: Path, target_domain: str, model_seed: int, split_seed: int) -> Path:
    return root / "shards" / f"{target_domain}_mseed{model_seed}_split{split_seed}"


def expected_keys(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    return [(domain, seed, split) for domain in DOMAINS for seed in args.model_seeds for split in args.support_split_seeds]


def create_base_model(feature_count: int, cfg: dict[str, Any], seed: int) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    seed_everything(seed)
    model = build_model("gnn", feature_count, cfg)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return model, state


def source_loaders(
    frames: dict[str, pd.DataFrame],
    target_domain: str,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for index, domain in enumerate(DOMAINS):
        if domain == target_domain:
            continue
        dataset = WindowDataset(frames[domain], sorted(frames[domain]["unit"].unique()), cfg["window_size"])
        loaders[domain] = make_loader(dataset, batch_size=cfg["batch_size"], shuffle=True, seed=seed + 1000 + index)
    return loaders


def run_worker(args: argparse.Namespace) -> None:
    root = resolve(PROJECT_ROOT, args.output_dir)
    directory = worker_dir(root, args.target_domain, args.model_seed, args.support_split_seed)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "worker_status.json"
    run_path = directory / "run_level.csv"
    if args.resume and status_path.is_file() and run_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("complete") and status.get("passed"):
            print(f"[A23.1] resume skip {directory.name}")
            return

    protocol, roles, protocol_hashes = load_protocol(args)
    cfg = load_config(args)
    data_dir = resolve(PROJECT_ROOT, args.data_dir)
    device = resolve_device(args.device)
    target = args.target_domain
    source_domains = [domain for domain in DOMAINS if domain != target]
    seed_everything(args.model_seed)

    raw_frames = {
        domain: load_domain_frame(resolve_train_file(data_dir, domain), rul_cap=cfg["rul_cap"])
        for domain in DOMAINS
    }
    normalizer = fit_source_normalizer({domain: raw_frames[domain] for domain in source_domains})
    frames = {domain: normalize(frame, normalizer) for domain, frame in raw_frames.items()}
    source_state_path = directory / "source_normalizer.json"
    atomic_json(
        source_state_path,
        {
            "fitted_domains": source_domains,
            "feature_columns": list(FEATURE_COLUMNS),
            "normalizer": normalizer,
            "target_domain_used_for_fit": False,
            "confirmation_engines_used_for_fit": False,
        },
    )

    selection_engines = role_engines(roles, target, args.support_split_seed, "selection")
    confirmation_engines = role_engines(roles, target, args.support_split_seed, "confirmation")
    if set(selection_engines) & set(confirmation_engines):
        raise A23Error("selection and confirmation engine leakage")
    selection_dataset = WindowDataset(frames[target], selection_engines, cfg["window_size"])
    confirmation_dataset = WindowDataset(frames[target], confirmation_engines, cfg["window_size"])
    selection_loader = make_loader(selection_dataset, batch_size=cfg["batch_size"], shuffle=False, seed=args.model_seed + 11)
    confirmation_loader = make_loader(confirmation_dataset, batch_size=cfg["batch_size"], shuffle=False, seed=args.model_seed + 12)

    _, base_state = create_base_model(len(FEATURE_COLUMNS), cfg, args.model_seed)
    source_model = build_model("gnn", len(FEATURE_COLUMNS), cfg)
    source_model.load_state_dict(base_state)
    pretrained_model, source_history = train_source_steps(
        source_model,
        source_loaders(frames, target, cfg, args.model_seed),
        steps=args.source_pretrain_steps,
        learning_rate=cfg["source_learning_rate"],
        pair_aux_weight=cfg["pair_aux_weight"],
        seed=args.model_seed,
        device=device,
    )
    pretrained_state = {
        name: tensor.detach().cpu().clone() for name, tensor in pretrained_model.state_dict().items()
    }
    source_checkpoint = directory / "source_pretrained_state.pt"
    torch.save(
        {
            "state": pretrained_state,
            "model_seed": args.model_seed,
            "target_domain": target,
            "source_domains": source_domains,
            "feature_columns": list(FEATURE_COLUMNS),
            "source_pretrain_steps": args.source_pretrain_steps,
            "source_history": source_history,
            "protocol_hashes": protocol_hashes,
        },
        source_checkpoint,
    )
    del pretrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for shot in args.shots:
        support_engines = role_engines(roles, target, args.support_split_seed, "support_pool", shot)
        if len(support_engines) != shot:
            raise A23Error(f"locked support engines do not match requested shot={shot}")
        if set(support_engines) & set(selection_engines) or set(support_engines) & set(confirmation_engines):
            raise A23Error("support engine leakage")
        support_dataset = WindowDataset(frames[target], support_engines, cfg["window_size"])
        for regime_index, regime in enumerate(args.regimes):
            seed_everything(args.model_seed)
            if regime == "source_only":
                state = pretrained_state
                target_history: list[dict[str, float]] = []
            elif regime == "scratch_k":
                state = base_state
                target_history = []
            else:
                state = pretrained_state
                target_history = []
            model = build_model("gnn", len(FEATURE_COLUMNS), cfg)
            model.load_state_dict(state)
            if regime != "source_only":
                support_loader = make_loader(
                    support_dataset,
                    batch_size=cfg["batch_size"],
                    shuffle=True,
                    seed=args.model_seed * 100000 + args.support_split_seed * 10 + shot,
                )
                model, target_history = train_epochs(
                    model,
                    support_loader,
                    epochs=args.target_epochs,
                    learning_rate=cfg["inner_lr"],
                    pair_aux_weight=cfg["pair_aux_weight"],
                    device=device,
                    label=f"target={target} seed={args.model_seed} split={args.support_split_seed} shot={shot} regime={regime}",
                )
            selection_metrics = evaluate(model, selection_loader, device)
            confirmation_metrics = evaluate(model, confirmation_loader, device)
            checkpoint = directory / f"{regime}_shot{shot}_target_adapted.pt"
            torch.save(
                {
                    "state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                    "regime": regime,
                    "shot": shot,
                    "target_domain": target,
                    "model_seed": args.model_seed,
                    "support_split_seed": args.support_split_seed,
                    "support_engines": support_engines,
                    "feature_columns": list(FEATURE_COLUMNS),
                    "normalizer_path": str(source_state_path),
                    "source_checkpoint": str(source_checkpoint),
                    "target_epochs": 0 if regime == "source_only" else args.target_epochs,
                    "source_pretrain_steps": args.source_pretrain_steps,
                    "protocol_hashes": protocol_hashes,
                    "target_history": target_history,
                },
                checkpoint,
            )
            row: dict[str, Any] = {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": args.model_seed,
                "support_split_seed": args.support_split_seed,
                "shot": shot,
                "regime": regime,
                "source_domains": json.dumps(source_domains),
                "support_engines": json.dumps(support_engines),
                "selection_engine_count": len(selection_engines),
                "confirmation_engine_count": len(confirmation_engines),
                "feature_count": len(FEATURE_COLUMNS),
                "feature_columns": json.dumps(FEATURE_COLUMNS),
                "window_size": cfg["window_size"],
                "rul_cap": cfg["rul_cap"],
                "source_pretrain_steps": args.source_pretrain_steps,
                "target_epochs": 0 if regime == "source_only" else args.target_epochs,
                "model_checkpoint": str(checkpoint),
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
                "confirmation_used_for_training": False,
                "confirmation_used_for_normalizer_fit": False,
                "selection_used_for_training": False,
                "selection_used_for_epoch_selection": False,
            }
            row.update(flatten_metrics("selection", selection_metrics))
            row.update(flatten_metrics("confirmation", confirmation_metrics))
            rows.append(row)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "[A23.1] completed "
                f"target={target} seed={args.model_seed} split={args.support_split_seed} "
                f"shot={shot} regime={regime}",
                flush=True,
            )

    fields = sorted({key for row in rows for key in row})
    atomic_csv(run_path, rows, fields)
    status = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": len(rows) == len(args.shots) * len(args.regimes),
        "target_domain": target,
        "model_seed": args.model_seed,
        "support_split_seed": args.support_split_seed,
        "expected_runs": len(args.shots) * len(args.regimes),
        "completed_runs": len(rows),
        "protocol_hashes": protocol_hashes,
        "normalizer_fitted_source_domains": source_domains,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "confirmation_used_for_training": False,
        "new_predictor_training": True,
    }
    atomic_json(status_path, status)


def worker_command(args: argparse.Namespace, domain: str, seed: int, split_seed: int) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for key, value in (
        ("--data-dir", args.data_dir),
        ("--config", args.config),
        ("--protocol-dir", args.protocol_dir),
        ("--output-dir", args.output_dir),
        ("--shots", ",".join(map(str, args.shots))),
        ("--model-seeds", ",".join(map(str, args.model_seeds))),
        ("--support-split-seeds", ",".join(map(str, args.support_split_seeds))),
        ("--regimes", ",".join(args.regimes)),
        ("--source-pretrain-steps", args.source_pretrain_steps),
        ("--target-epochs", args.target_epochs),
        ("--target-domain", domain),
        ("--model-seed", seed),
        ("--support-split-seed", split_seed),
        ("--device", "cuda"),
    ):
        command.extend([key, str(value)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.window_size is not None:
        command.extend(["--window-size", str(args.window_size)])
    if args.rul_cap is not None:
        command.extend(["--rul-cap", str(args.rul_cap)])
    if args.learning_rate is not None:
        command.extend(["--learning-rate", str(args.learning_rate)])
    if args.source_learning_rate is not None:
        command.extend(["--source-learning-rate", str(args.source_learning_rate)])
    if args.pair_aux_weight is not None:
        command.extend(["--pair-aux-weight", str(args.pair_aux_weight)])
    if args.resume:
        command.append("--resume")
    return command


def merge_and_decide(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(PROJECT_ROOT, args.output_dir)
    records: list[pd.DataFrame] = []
    expected = expected_keys(args)
    missing: list[str] = []
    statuses: list[dict[str, Any]] = []
    for domain, seed, split_seed in expected:
        directory = worker_dir(root, domain, seed, split_seed)
        status_path, run_path = directory / "worker_status.json", directory / "run_level.csv"
        if not status_path.is_file() or not run_path.is_file():
            missing.append(directory.name)
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not (status.get("complete") and status.get("passed")):
            missing.append(directory.name)
            continue
        statuses.append(status)
        records.append(pd.read_csv(run_path))
    if missing:
        raise A23Error(f"cannot merge incomplete worker shards: {missing}")
    merged = pd.concat(records, ignore_index=True)
    expected_runs = len(expected) * len(args.shots) * len(args.regimes)
    if len(merged) != expected_runs:
        raise A23Error(f"merged run count={len(merged)} but expected={expected_runs}")
    integrity_columns = [
        "official_test_files_accessed",
        "official_test_forward_run",
        "confirmation_used_for_training",
        "confirmation_used_for_normalizer_fit",
        "selection_used_for_training",
        "selection_used_for_epoch_selection",
    ]
    for column in integrity_columns:
        if merged[column].astype(bool).any():
            raise A23Error(f"integrity violation in merged runs: {column}")
    run_level = root / "experimentA23_1_confirmation_run_level.csv"
    merged.to_csv(run_level, index=False)
    metric_columns = [
        "confirmation_engine_endpoint_rmse",
        "confirmation_engine_endpoint_mae",
        "confirmation_engine_endpoint_nasa_score",
        "confirmation_all_windows_rmse",
        "confirmation_all_windows_nasa_score",
    ]
    summary = (
        merged.groupby(["target_domain", "shot", "regime"], as_index=False)[metric_columns]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(str(item) for item in column if item).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary_path = root / "experimentA23_1_pilot_summary.csv"
    summary.to_csv(summary_path, index=False)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Under the locked A23.0 engine-level few-shot protocol, can matched source-only, "
            "scratch-k and ordinary pretrain-plus-finetune-k baselines complete without leakage "
            "before Meta-GNN efficacy is tested?"
        ),
        "complete": True,
        "passed": True,
        "pilot_only": True,
        "expected_worker_cells": len(expected),
        "completed_worker_cells": len(statuses),
        "expected_run_records": expected_runs,
        "completed_run_records": int(len(merged)),
        "shots": list(args.shots),
        "model_seeds": list(args.model_seeds),
        "support_split_seeds": list(args.support_split_seeds),
        "regimes": list(args.regimes),
        "source_pretrain_steps": args.source_pretrain_steps,
        "target_epochs": args.target_epochs,
        "confirmation_metrics_reported": True,
        "baseline_efficacy_claim": False,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": "A23.1 completed the training-only few-shot transfer baseline pilot under the locked A23.0 protocol",
        "next_action": "inspect_pilot_integrity_then_implement_A24_meta_no_graph_and_reptile_meta_gnn",
    }
    atomic_json(root / "experimentA23_1_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "artifacts": {
            run_level.name: sha256_file(run_level),
            summary_path.name: sha256_file(summary_path),
            "protocol_dir": str(resolve(PROJECT_ROOT, args.protocol_dir)),
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA23_1_manifest.json", manifest)
    return decision


def parent_main(args: argparse.Namespace) -> None:
    protocol, _, protocol_hashes = load_protocol(args)
    cfg = load_config(args)
    data_dir = resolve(PROJECT_ROOT, args.data_dir)
    inventory = {domain: str(resolve_train_file(data_dir, domain)) for domain in DOMAINS}
    workers = expected_keys(args)
    expected_runs = len(workers) * len(args.shots) * len(args.regimes)
    dry = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": "A23.1 training-only few-shot transfer baseline pilot",
        "protocol_hashes": protocol_hashes,
        "data_files": inventory,
        "data_file_sha256_locked_by_A23_0": {
            item["domain"]: item["sha256"] for item in protocol["training_file_inventory"]
        },
        "shots": list(args.shots),
        "model_seeds": list(args.model_seeds),
        "support_split_seeds": list(args.support_split_seeds),
        "regimes": list(args.regimes),
        "expected_worker_cells": len(workers),
        "expected_run_records": expected_runs,
        "model_architecture": "existing_baselines.build_model('gnn', 17, config)",
        "feature_columns": list(FEATURE_COLUMNS),
        "window_size": cfg["window_size"],
        "rul_cap": cfg["rul_cap"],
        "source_pretrain_steps": args.source_pretrain_steps,
        "target_epochs": args.target_epochs,
        "new_predictor_training": not args.dry_run,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    print(json.dumps(dry, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print("[A23.1] dry-run completed; no predictor was trained and no official test file was read")
        return

    root = resolve(PROJECT_ROOT, args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.single_process or args.device == "cpu":
        for domain, seed, split_seed in workers:
            local = deepcopy(args)
            local.worker, local.target_domain = True, domain
            local.model_seed, local.support_split_seed = seed, split_seed
            local.device = "cpu" if args.device == "cpu" else "auto"
            run_worker(local)
    else:
        gpu_ids = parse_int_list(args.gpus, name="gpus")
        max_workers = min(args.max_workers or len(gpu_ids), len(gpu_ids))
        devices = list(gpu_ids[:max_workers])
        pending = workers.copy()
        active: dict[int, dict[str, Any]] = {}
        while pending or active:
            for gpu in [item for item in devices if item not in active]:
                if not pending:
                    break
                domain, seed, split_seed = pending.pop(0)
                directory = worker_dir(root, domain, seed, split_seed)
                directory.mkdir(parents=True, exist_ok=True)
                log_path = directory / "worker_training.log"
                log_handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    worker_command(args, domain, seed, split_seed),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[gpu] = {"process": process, "handle": log_handle, "log": log_path, "item": (domain, seed, split_seed)}
                print(f"[A23.1] launched target={domain} seed={seed} split={split_seed} gpu={gpu} pid={process.pid}", flush=True)
            finished: list[int] = []
            for gpu, record in active.items():
                code = record["process"].poll()
                if code is None:
                    continue
                record["handle"].close()
                if code != 0:
                    tail = "\n".join(record["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                    for other in active.values():
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    raise A23Error(f"worker failed item={record['item']} exit={code}\n{tail}")
                domain, seed, split_seed = record["item"]
                print(f"[A23.1] completed target={domain} seed={seed} split={split_seed} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(3)
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
    except A23Error as exc:
        print(f"[A23.1] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
