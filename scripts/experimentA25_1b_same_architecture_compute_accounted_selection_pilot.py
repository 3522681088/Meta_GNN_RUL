#!/usr/bin/env python3
"""A25.1b: compute-accounted same-architecture 2x2 selection-only pilot.

This training script consumes the immutable A25.1a contract.  It compares
ordinary source pretraining with Reptile within *each* architecture, not across
different architectures:

  ordinary_no_graph_pft  vs  reptile_meta_no_graph
  ordinary_gnn_pft       vs  reptile_meta_gnn

For an algorithm pair, the initial state, source episode-support batch
schedule, number of source gradient updates, source window presentations,
target support engines, target batch order and target epochs are all matched.
The target selection engines are used only for descriptive diagnostics.  The
script neither opens nor evaluates confirmation engines or official test files.
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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402
from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402


EXPERIMENT_ID = "experimentA25_1b"
SCRIPT_VERSION = "experimentA25_1b_same_architecture_compute_accounted_selection_pilot_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
METHODS = (
    "ordinary_no_graph_pft",
    "reptile_meta_no_graph",
    "ordinary_gnn_pft",
    "reptile_meta_gnn",
)
PAIRS = {
    "no_graph": ("ordinary_no_graph_pft", "reptile_meta_no_graph"),
    "gnn": ("ordinary_gnn_pft", "reptile_meta_gnn"),
}
HASH_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


class A251bError(RuntimeError):
    """Raised when a locked A25.1 contract would be violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_schema(state: Mapping[str, torch.Tensor]) -> tuple[int, int, str]:
    if not state:
        raise A251bError("state dictionary is empty")
    schema = "\n".join(
        f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        for name, tensor in sorted(state.items())
    )
    return (
        int(sum(int(tensor.numel()) for tensor in state.values())),
        int(len(state)),
        hashlib.sha256(schema.encode("utf-8")).hexdigest(),
    )


def state_value_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
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


def atomic_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise A251bError(f"refusing to write empty CSV: {path.name}")
    fields = list(materialized[0])
    for index, row in enumerate(materialized):
        if set(row) != set(fields):
            raise A251bError(f"row schema mismatch at index={index} for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise A251bError(f"required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A251bError(f"failed to parse {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A251bError(f"{label} must be a JSON object: {path}")
    return value


def load_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise A251bError(f"required {label} is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise A251bError(f"failed to parse {label}: {path}: {exc}") from exc
    if frame.empty:
        raise A251bError(f"{label} is empty: {path}")
    return frame


def strict_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise A251bError(f"required Boolean column missing: {column}")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    accepted = {"true", "false", "1", "0"}
    invalid = sorted(set(normalized) - accepted)
    if invalid:
        raise A251bError(f"invalid Boolean values in {column}: {invalid[:8]}")
    return normalized.isin({"true", "1"})


def parse_ints(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must be non-empty, unique, non-negative integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--a25-1a-output-dir",
        type=Path,
        default=Path("outputs/experimentA25_1a_independent_matched_2x2_preflight"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA25_1b_same_architecture_compute_accounted_selection_pilot"),
    )
    parser.add_argument("--gpus", default="0", help="Physical GPU ids, e.g. 6,7")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-free-memory-mb", type=int, default=16000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    parser.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.gpu_ids = parse_ints(args.gpus, name="gpus")
    if args.max_workers < 1:
        raise A251bError("--max-workers must be positive")
    if args.min_free_memory_mb < 0 or not 0 <= args.max_gpu_utilization <= 100:
        raise A251bError("invalid GPU eligibility threshold")
    if args.torch_threads < 1:
        raise A251bError("--torch-threads must be positive")
    if args.worker and (
        args.target_domain is None or args.model_seed is None or args.support_split_seed is None
    ):
        raise A251bError("worker mode requires target-domain, model-seed and support-split-seed")
    return args


def require_complete_passed(payload: Mapping[str, Any], experiment: str) -> None:
    if payload.get("experiment_id") != experiment:
        raise A251bError(f"expected {experiment}, observed {payload.get('experiment_id')!r}")
    if payload.get("complete") is not True or payload.get("passed") is not True:
        raise A251bError(f"{experiment} must be complete=true and passed=true")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if payload.get(field) is not False:
            raise A251bError(f"{experiment} violates training-only boundary: {field}")


def validate_artifact_hashes(root: Path, decision: Mapping[str, Any]) -> dict[str, str]:
    artifacts = decision.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise A251bError("A25.1a decision lacks artifact_sha256")
    observed: dict[str, str] = {}
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str) or HASH_RE.fullmatch(expected) is None:
            raise A251bError(f"invalid A25.1a artifact hash entry: {name!r}")
        path = root / name
        if not path.is_file():
            raise A251bError(f"A25.1a artifact is missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise A251bError(f"A25.1a artifact hash mismatch: {path}")
        observed[name] = digest
    return observed


def load_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, str]]:
    root = resolve(args.a25_1a_output_dir)
    decision_path = root / "experimentA25_1a_confirmation_decision.json"
    protocol_path = root / "experimentA25_1a_protocol.json"
    decision = read_json(decision_path, label="A25.1a decision")
    protocol = read_json(protocol_path, label="A25.1a protocol")
    require_complete_passed(decision, "experimentA25_1a")
    if decision.get("preflight_only") is not True:
        raise A251bError("A25.1a is not a preflight-only contract")
    required_true = (
        "same_architecture_parameter_and_initialization_contract_locked",
        "equal_source_gradient_update_and_window_presentation_budget_locked",
        "runtime_compute_accounting_required",
        "A25_1b_selection_only",
    )
    for field in required_true:
        if decision.get(field) is not True:
            raise A251bError(f"A25.1a decision requires {field}=true")
    if decision.get("A25_1b_confirmation_engines_evaluated") is not False:
        raise A251bError("A25.1a confirmation boundary is not sealed")
    if protocol.get("experiment_id") != "experimentA25_1a":
        raise A251bError("A25.1a protocol identity mismatch")
    expected_methods = list(METHODS)
    if protocol.get("methods") != expected_methods:
        raise A251bError(f"A25.1a method order mismatch: {protocol.get('methods')}")
    if protocol.get("design") != "prospective_training_file_selection_only_2x2_pilot_preflight":
        raise A251bError("A25.1a contract is not selection-only pilot design")
    hashes = validate_artifact_hashes(root, decision)
    files = {
        "roles": root / "experimentA25_1a_target_engine_roles.csv",
        "tasks": root / "experimentA25_1a_source_episode_inventory.csv",
        "methods": root / "experimentA25_1a_method_contract.csv",
        "compute": root / "experimentA25_1a_compute_contract.csv",
        "statistics": root / "experimentA25_1a_statistical_analysis_plan.json",
        "integrity": root / "experimentA25_1a_input_integrity.json",
    }
    frames = {key: load_csv(path, label=f"A25.1a {key}") for key, path in files.items() if path.suffix == ".csv"}
    statistics = read_json(files["statistics"], label="A25.1a statistical plan")
    integrity = read_json(files["integrity"], label="A25.1a input integrity")
    if integrity.get("all_required_inputs_validated") is not True:
        raise A251bError("A25.1a input integrity is not complete")
    if statistics.get("A25_1b_evaluation_scope") != "selection_engines_only":
        raise A251bError("A25.1a statistical plan does not lock selection-only evaluation")
    if statistics.get("A25_1b_confirmatory_p_values_allowed") is not False:
        raise A251bError("A25.1a permits confirmatory p-values in the pilot")
    if statistics.get("A25_1b_policy_selection_allowed") is not False:
        raise A251bError("A25.1a permits policy selection in the pilot")
    frames["statistics"] = pd.DataFrame([statistics])
    frames["integrity"] = pd.DataFrame([integrity])
    hashes.update({
        decision_path.name: sha256(decision_path),
        protocol_path.name: sha256(protocol_path),
    })
    return protocol, frames, hashes


def load_config(protocol: Mapping[str, Any], config_arg: Path) -> tuple[dict[str, Any], Path]:
    path = resolve(config_arg)
    if not path.is_file():
        raise A251bError(f"configuration is missing: {path}")
    digest = sha256(path)
    contract = protocol.get("config_contract")
    if not isinstance(contract, dict) or contract.get("sha256") != digest:
        raise A251bError("configuration hash differs from the locked A25.1a contract")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A251bError(f"failed to parse configuration: {exc}") from exc
    if not isinstance(cfg, dict):
        raise A251bError("configuration must be a mapping")
    cfg = deepcopy(cfg)
    cfg["batch_size"] = int(contract["batch_size"])
    cfg["window_size"] = int(contract["window_size"])
    cfg["rul_cap"] = float(contract["rul_cap"])
    cfg["inner_lr"] = float(protocol["inner_learning_rate"])
    cfg["outer_lr"] = float(protocol["outer_learning_rate"])
    cfg["pair_aux_weight"] = float(cfg.get("pair_aux_weight", 0.0))
    if cfg["batch_size"] < 1 or cfg["window_size"] < 2 or cfg["rul_cap"] <= 0:
        raise A251bError("invalid locked config values")
    return cfg, path


def resolve_train_file(data_dir: Path, domain: str) -> Path:
    root = resolve(data_dir)
    if not root.is_dir():
        raise A251bError(f"data directory is missing: {root}")
    try:
        path = a23.resolve_train_file(root, domain)
    except Exception as exc:
        raise A251bError(f"failed to resolve train_{domain}.txt: {exc}") from exc
    lowered = path.name.lower()
    if lowered.startswith("test_") or "rul_" in lowered:
        raise A251bError(f"refusing non-training input: {path}")
    return path


def load_frames(args: argparse.Namespace, protocol: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    inventory = protocol.get("training_file_inventory")
    if not isinstance(inventory, list):
        raise A251bError("A25.1a training file inventory is missing")
    expected = {str(item.get("domain")): item for item in inventory if isinstance(item, dict)}
    if set(expected) != set(DOMAINS):
        raise A251bError("A25.1a training file inventory is incomplete")
    frames: dict[str, pd.DataFrame] = {}
    for domain in DOMAINS:
        path = resolve_train_file(args.data_dir, domain)
        if sha256(path) != expected[domain].get("sha256"):
            raise A251bError(f"training file hash mismatch for {domain}")
        frame = a23.load_domain_frame(path, rul_cap=float(cfg["rul_cap"]))
        if int(frame["unit"].nunique()) != int(expected[domain].get("engines", -1)):
            raise A251bError(f"training engine count mismatch for {domain}")
        frames[domain] = frame
    return frames


def parse_engine_json(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(str(value))
    except Exception as exc:
        raise A251bError(f"invalid JSON engine list in {label}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise A251bError(f"engine list must be non-empty in {label}")
    engines = tuple(int(item) for item in parsed)
    if len(engines) != len(set(engines)) or any(item < 1 for item in engines):
        raise A251bError(f"invalid engines in {label}")
    return engines


def role_engines(roles: pd.DataFrame, domain: str, split: int, role: str, shot: int | None = None) -> list[int]:
    required = {"target_domain", "support_split_seed", "engine_id", "role", "support_rank"}
    if missing := required - set(roles.columns):
        raise A251bError(f"role table lacks columns: {sorted(missing)}")
    frame = roles.loc[
        (roles["target_domain"].astype(str) == domain)
        & (pd.to_numeric(roles["support_split_seed"], errors="raise").astype(int) == int(split))
        & (roles["role"].astype(str) == role)
    ].copy()
    if role == "support_pool":
        if shot is None:
            raise A251bError("support_pool requires a shot value")
        rank = pd.to_numeric(frame["support_rank"], errors="raise")
        if rank.isna().any() or not np.allclose(rank.to_numpy(), np.floor(rank.to_numpy())):
            raise A251bError("support_rank is not integral")
        frame = frame.loc[rank.astype(int) <= int(shot)]
    result = sorted(int(value) for value in pd.to_numeric(frame["engine_id"], errors="raise"))
    if not result:
        raise A251bError(f"empty role assignment: {domain}/{split}/{role}")
    return result


def worker_tasks(tasks: pd.DataFrame, domain: str, seed: int, split: int, protocol: Mapping[str, Any]) -> pd.DataFrame:
    required = {
        "target_domain", "model_seed", "target_support_split_seed", "source_domain",
        "episode_phase", "episode_index", "meta_support_engine_ids", "meta_query_engine_ids",
        "support_query_disjoint", "target_domain_excluded",
    }
    if missing := required - set(tasks.columns):
        raise A251bError(f"source task table lacks columns: {sorted(missing)}")
    frame = tasks.loc[
        (tasks["target_domain"].astype(str) == domain)
        & (pd.to_numeric(tasks["model_seed"], errors="raise").astype(int) == int(seed))
        & (pd.to_numeric(tasks["target_support_split_seed"], errors="raise").astype(int) == int(split))
    ].copy()
    per_phase = int(protocol["episodes_per_source_domain"])
    expected = (len(DOMAINS) - 1) * 2 * per_phase
    if len(frame) != expected:
        raise A251bError(f"task count for {domain}/{seed}/{split} is {len(frame)}, expected {expected}")
    if strict_bool(frame, "support_query_disjoint").eq(False).any() or strict_bool(frame, "target_domain_excluded").eq(False).any():
        raise A251bError("source task leakage flag is false")
    if set(frame["source_domain"].astype(str)) != set(DOMAINS) - {domain}:
        raise A251bError("source task domain set is not leave-one-target-domain-out")
    counts = frame.groupby(["source_domain", "episode_phase"]).size().to_dict()
    for source in set(DOMAINS) - {domain}:
        for phase in ("meta_train", "meta_validation"):
            if int(counts.get((source, phase), 0)) != per_phase:
                raise A251bError(f"source task count mismatch {domain}/{source}/{phase}")
    for row in frame.itertuples(index=False):
        support = parse_engine_json(row.meta_support_engine_ids, label="meta_support_engine_ids")
        query = parse_engine_json(row.meta_query_engine_ids, label="meta_query_engine_ids")
        if set(support) & set(query):
            raise A251bError("source episode support/query engine overlap")
    return frame.sort_values(["source_domain", "episode_phase", "episode_index"], kind="stable").reset_index(drop=True)


def method_contract(methods: pd.DataFrame, compute: pd.DataFrame, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    method_columns = {
        "method", "architecture", "algorithm", "reference_state_tensor_numel",
        "reference_state_tensor_count", "reference_state_schema_sha256",
        "runtime_exact_state_schema_assertion_required",
        "runtime_total_parameter_equality_within_pair_required",
        "runtime_trainable_parameter_equality_within_pair_required",
        "runtime_initialization_equality_within_pair_required",
    }
    if missing := method_columns - set(methods.columns):
        raise A251bError(f"method contract lacks columns: {sorted(missing)}")
    if set(methods["method"].astype(str)) != set(METHODS) or len(methods) != len(METHODS):
        raise A251bError("method contract does not contain exactly four methods")
    lookup: dict[str, dict[str, Any]] = {}
    for row in methods.to_dict(orient="records"):
        method = str(row["method"])
        for key in (
            "runtime_exact_state_schema_assertion_required",
            "runtime_total_parameter_equality_within_pair_required",
            "runtime_trainable_parameter_equality_within_pair_required",
            "runtime_initialization_equality_within_pair_required",
        ):
            value = str(row[key]).strip().lower()
            if value not in {"true", "1"}:
                raise A251bError(f"method contract requires {key}=true for {method}")
        lookup[method] = row
    compute_required = {"method", "source_gradient_updates_budget", "source_window_presentations_budget", "target_epochs"}
    if missing := compute_required - set(compute.columns):
        raise A251bError(f"compute contract lacks columns: {sorted(missing)}")
    expected_updates = int(protocol["source_gradient_updates_per_method_cell"])
    expected_windows = int(protocol["source_window_presentations_per_method_cell"])
    for row in compute.to_dict(orient="records"):
        method = str(row["method"])
        if method not in lookup:
            raise A251bError(f"unknown compute-contract method: {method}")
        if int(row["source_gradient_updates_budget"]) != expected_updates:
            raise A251bError(f"source update budget mismatch for {method}")
        if int(row["source_window_presentations_budget"]) != expected_windows:
            raise A251bError(f"source window budget mismatch for {method}")
        if int(row["target_epochs"]) != int(protocol["target_epochs"]):
            raise A251bError(f"target epoch budget mismatch for {method}")
    if len(compute) != len(METHODS):
        raise A251bError("compute contract does not contain exactly four methods")
    return lookup


def make_model(method: str, cfg: Mapping[str, Any], seed: int) -> torch.nn.Module:
    a23.seed_everything(int(seed))
    model = build_model("gnn", len(a23.FEATURE_COLUMNS), dict(cfg))
    if not hasattr(model, "use_gat") or not hasattr(model, "gat"):
        raise A251bError("build_model('gnn') must expose use_gat and gat")
    if method in {"ordinary_no_graph_pft", "reptile_meta_no_graph"}:
        model.use_gat = False
        model.gat = torch.nn.Identity()
    elif method in {"ordinary_gnn_pft", "reptile_meta_gnn"}:
        if not bool(model.use_gat):
            raise A251bError("GNN arm constructed with use_gat=false")
    else:
        raise A251bError(f"unknown method: {method}")
    return model


def runtime_model_audit(cfg: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    models = {method: make_model(method, cfg, seed).cpu() for method in METHODS}
    rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for method, model in models.items():
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        states[method] = state
        schema_numel, schema_count, schema_hash = state_schema(state)
        contract = contracts[method]
        if schema_numel != int(contract["reference_state_tensor_numel"]):
            raise A251bError(f"state tensor numel mismatch for {method}")
        if schema_count != int(contract["reference_state_tensor_count"]):
            raise A251bError(f"state tensor count mismatch for {method}")
        if schema_hash != str(contract["reference_state_schema_sha256"]):
            raise A251bError(f"state schema hash mismatch for {method}")
        total = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        if total != trainable or total != schema_numel:
            raise A251bError(f"unexpected parameter/state count relationship for {method}")
        rows.append({
            "method": method,
            "architecture": str(contract["architecture"]),
            "state_tensor_numel": schema_numel,
            "state_tensor_count": schema_count,
            "state_schema_sha256": schema_hash,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "initial_state_sha256": state_value_hash(state),
            "runtime_schema_matches_A25_1a": True,
        })
    for architecture, (ordinary, reptile) in PAIRS.items():
        ordinary_state, reptile_state = states[ordinary], states[reptile]
        if set(ordinary_state) != set(reptile_state):
            raise A251bError(f"state keys differ within {architecture} pair")
        for name in ordinary_state:
            if not torch.equal(ordinary_state[name], reptile_state[name]):
                raise A251bError(f"initial state differs within {architecture} pair at {name}")
    return rows


def runtime_smoke(cfg: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = torch.randn(8, int(cfg["window_size"]), len(a23.FEATURE_COLUMNS))
    y = torch.linspace(5.0, 105.0, 8)
    with tempfile.TemporaryDirectory(prefix="a25_1b_smoke_") as temporary:
        for method in METHODS:
            model = make_model(method, cfg, seed).cpu()
            model.train()
            loss, _ = rul_training_loss(model, x, y, float(cfg["pair_aux_weight"]))
            if not torch.isfinite(loss):
                raise A251bError(f"non-finite smoke loss for {method}")
            loss.backward()
            checkpoint = Path(temporary) / f"{method}.pt"
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            torch.save({"state": state}, checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            verifier = make_model(method, cfg, seed).cpu()
            verifier.load_state_dict(loaded["state"], strict=True)
            rows.append({
                "method": method,
                "forward_passed": True,
                "backward_passed": True,
                "checkpoint_roundtrip_passed": True,
                "smoke_loss": float(loss.detach()),
            })
    return rows


def selection_and_support(roles: pd.DataFrame, domain: str, split: int, shots: Sequence[int]) -> tuple[dict[int, list[int]], list[int]]:
    selection = role_engines(roles, domain, split, "selection")
    support = {int(shot): role_engines(roles, domain, split, "support_pool", int(shot)) for shot in shots}
    if len(set(selection)) != len(selection):
        raise A251bError("duplicate selection engines")
    prior: set[int] = set()
    for shot in sorted(support):
        now = set(support[shot])
        if len(now) != shot or not prior.issubset(now):
            raise A251bError(f"non-nested support K sets for {domain}/{split}")
        if now & set(selection):
            raise A251bError(f"support/selection leakage for {domain}/{split}")
        prior = now
    return support, selection


def source_fit_engines(tasks: pd.DataFrame, target: str, frames: Mapping[str, pd.DataFrame]) -> dict[str, tuple[int, ...]]:
    train = tasks.loc[tasks["episode_phase"].astype(str) == "meta_train"].copy()
    if train.empty:
        raise A251bError("worker has no meta_train episodes")
    result: dict[str, tuple[int, ...]] = {}
    for source in sorted(set(DOMAINS) - {target}):
        source_rows = train.loc[train["source_domain"].astype(str) == source]
        union: set[int] = set()
        for row in source_rows.itertuples(index=False):
            union.update(parse_engine_json(row.meta_support_engine_ids, label="meta_support_engine_ids"))
            union.update(parse_engine_json(row.meta_query_engine_ids, label="meta_query_engine_ids"))
        available = set(int(value) for value in frames[source]["unit"].unique())
        if not union or not union <= available:
            raise A251bError(f"invalid source engine union for {target}/{source}")
        result[source] = tuple(sorted(union))
    return result


def source_normalize(raw: Mapping[str, pd.DataFrame], target: str, source_engines: Mapping[str, Sequence[int]]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_frames = {
        source: raw[source].loc[raw[source]["unit"].isin(set(source_engines[source]))].copy()
        for source in source_engines
    }
    if set(source_frames) != set(DOMAINS) - {target} or any(frame.empty for frame in source_frames.values()):
        raise A251bError("source normalizer would use an incomplete source engine subset")
    normalizer = a23.fit_source_normalizer(source_frames)
    normalized = {domain: a23.normalize(frame, normalizer) for domain, frame in raw.items()}
    audit = {
        "fitted_domains": sorted(source_frames),
        "source_engine_ids": {domain: list(source_engines[domain]) for domain in sorted(source_engines)},
        "target_domain_used_for_fit": False,
        "selection_engines_used_for_fit": False,
        "confirmation_engines_used_for_fit": False,
        "normalizer": normalizer,
    }
    return normalized, audit


def deterministic_schedule(tasks: pd.DataFrame, outer_steps: int, seed: int) -> list[dict[str, Any]]:
    train = tasks.loc[tasks["episode_phase"].astype(str) == "meta_train"].copy()
    train = train.sort_values(["source_domain", "episode_index"], kind="stable").reset_index(drop=True)
    entries = train.to_dict(orient="records")
    if not entries:
        raise A251bError("cannot create schedule with zero meta_train tasks")
    rng = random.Random(int(seed) + 251100)
    schedule: list[dict[str, Any]] = []
    order: list[int] = []
    for _ in range(int(outer_steps)):
        if not order:
            order = list(range(len(entries)))
            rng.shuffle(order)
        schedule.append(entries[order.pop()])
    if len(schedule) != int(outer_steps):
        raise A251bError("outer schedule length mismatch")
    return schedule


def episode_loader(
    normalized: Mapping[str, pd.DataFrame],
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    cache: dict[tuple[str, tuple[int, ...]], Any],
    seed: int,
):
    source = str(row["source_domain"])
    support = parse_engine_json(row["meta_support_engine_ids"], label="meta_support_engine_ids")
    query = parse_engine_json(row["meta_query_engine_ids"], label="meta_query_engine_ids")
    if set(support) & set(query):
        raise A251bError("source support/query overlap while building loader")
    key = (source, tuple(sorted(support)))
    if key not in cache:
        cache[key] = a23.WindowDataset(normalized[source], support, int(cfg["window_size"]))
    return a23.make_loader(cache[key], batch_size=int(cfg["batch_size"]), shuffle=True, seed=int(seed))


def initial_accounting(method: str, architecture: str, source_budget: int, source_windows_budget: int) -> dict[str, Any]:
    return {
        "method": method,
        "architecture": architecture,
        "source_gradient_updates": 0,
        "source_forward_calls": 0,
        "source_backward_calls": 0,
        "source_window_presentations": 0,
        "source_gradient_updates_budget": int(source_budget),
        "source_window_presentations_budget": int(source_windows_budget),
        "target_gradient_updates": 0,
        "target_forward_calls": 0,
        "target_backward_calls": 0,
        "target_window_presentations": 0,
        "selection_forward_calls": 0,
        "selection_window_presentations": 0,
        "peak_cuda_memory_bytes": 0,
    }


def step_update(model: torch.nn.Module, optimiser: torch.optim.Optimizer, batch, device: torch.device, pair_weight: float, accounting: dict[str, Any], phase: str) -> float:
    x, y, _, _ = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    optimiser.zero_grad(set_to_none=True)
    loss, _ = rul_training_loss(model, x, y, pair_weight)
    if not torch.isfinite(loss):
        raise A251bError(f"non-finite {phase} loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimiser.step()
    accounting[f"{phase}_gradient_updates"] += 1
    accounting[f"{phase}_forward_calls"] += 1
    accounting[f"{phase}_backward_calls"] += 1
    accounting[f"{phase}_window_presentations"] += int(y.numel())
    if device.type == "cuda":
        accounting["peak_cuda_memory_bytes"] = max(
            int(accounting["peak_cuda_memory_bytes"]), int(torch.cuda.max_memory_allocated(device))
        )
    return float(loss.detach().cpu())


def ordinary_source_train(model: torch.nn.Module, schedule: Sequence[Mapping[str, Any]], normalized: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any], seed: int, device: torch.device, accounting: dict[str, Any]) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    learner = model.to(device)
    optimiser = torch.optim.Adam(learner.parameters(), lr=float(cfg["inner_lr"]))
    cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    report_every = max(1, len(schedule) // 10)
    history: list[dict[str, Any]] = []
    running: list[float] = []
    for outer_index, row in enumerate(schedule, start=1):
        loader = episode_loader(normalized, row, cfg, cache, seed + outer_index * 31)
        iterator = iter(loader)
        for _ in range(int(cfg["inner_steps"])):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            running.append(step_update(learner, optimiser, batch, device, float(cfg["pair_aux_weight"]), accounting, "source"))
        if outer_index % report_every == 0 or outer_index == len(schedule):
            history.append({
                "method": accounting["method"], "phase": "source_train", "outer_block": outer_index,
                "source_gradient_updates": accounting["source_gradient_updates"],
                "mean_loss_since_last_report": float(np.mean(running)),
            })
            print(f"[A25.1b] {accounting['method']} source block={outer_index:04d}/{len(schedule)} loss={np.mean(running):.6f}", flush=True)
            running.clear()
    if accounting["source_gradient_updates"] != accounting["source_gradient_updates_budget"]:
        raise A251bError("ordinary source gradient update count differs from contract")
    if accounting["source_window_presentations"] != accounting["source_window_presentations_budget"]:
        raise A251bError("ordinary source window presentation count differs from contract")
    return learner.cpu(), history


def reptile_source_train(model: torch.nn.Module, schedule: Sequence[Mapping[str, Any]], normalized: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any], seed: int, device: torch.device, accounting: dict[str, Any]) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    base = model.cpu()
    cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    report_every = max(1, len(schedule) // 10)
    history: list[dict[str, Any]] = []
    running: list[float] = []
    for outer_index, row in enumerate(schedule, start=1):
        adapted = deepcopy(base).to(device)
        optimiser = torch.optim.Adam(adapted.parameters(), lr=float(cfg["inner_lr"]))
        loader = episode_loader(normalized, row, cfg, cache, seed + outer_index * 31)
        iterator = iter(loader)
        for _ in range(int(cfg["inner_steps"])):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            running.append(step_update(adapted, optimiser, batch, device, float(cfg["pair_aux_weight"]), accounting, "source"))
        adapted = adapted.cpu()
        with torch.no_grad():
            base_parameters = dict(base.named_parameters())
            adapted_parameters = dict(adapted.named_parameters())
            if set(base_parameters) != set(adapted_parameters):
                raise A251bError("Reptile base/adapted parameter keys differ")
            for name, parameter in base_parameters.items():
                parameter.add_(float(cfg["outer_lr"]) * (adapted_parameters[name] - parameter))
        if outer_index % report_every == 0 or outer_index == len(schedule):
            history.append({
                "method": accounting["method"], "phase": "source_train", "outer_block": outer_index,
                "source_gradient_updates": accounting["source_gradient_updates"],
                "mean_loss_since_last_report": float(np.mean(running)),
            })
            print(f"[A25.1b] {accounting['method']} outer={outer_index:04d}/{len(schedule)} loss={np.mean(running):.6f}", flush=True)
            running.clear()
        del adapted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if accounting["source_gradient_updates"] != accounting["source_gradient_updates_budget"]:
        raise A251bError("Reptile source gradient update count differs from contract")
    if accounting["source_window_presentations"] != accounting["source_window_presentations_budget"]:
        raise A251bError("Reptile source window presentation count differs from contract")
    return base.cpu(), history


def adapt_target(model: torch.nn.Module, dataset, cfg: Mapping[str, Any], seed: int, device: torch.device, accounting: dict[str, Any], label: str) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    loader = a23.make_loader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, seed=int(seed))
    learner = model.to(device)
    optimiser = torch.optim.Adam(learner.parameters(), lr=float(cfg["inner_lr"]))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        iterator = iter(loader)
        losses: list[float] = []
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            losses.append(step_update(learner, optimiser, batch, device, float(cfg["pair_aux_weight"]), accounting, "target"))
        if not losses:
            raise A251bError("target support loader produced zero batches")
        history.append({
            "method": accounting["method"], "phase": "target_adaptation", "epoch": epoch,
            "target_gradient_updates": accounting["target_gradient_updates"],
            "mean_loss": float(np.mean(losses)),
        })
        print(f"[A25.1b] {label} epoch={epoch:02d}/{cfg['target_epochs']} loss={np.mean(losses):.6f}", flush=True)
    return learner, history


def evaluate_selection(model: torch.nn.Module, dataset, cfg: Mapping[str, Any], device: torch.device, accounting: dict[str, Any]) -> dict[str, dict[str, float]]:
    loader = a23.make_loader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, seed=251199)
    metrics = a23.evaluate(model, loader, device)
    accounting["selection_forward_calls"] = int(math.ceil(len(dataset) / int(cfg["batch_size"])))
    accounting["selection_window_presentations"] = int(len(dataset))
    return metrics


def worker_root(root: Path, domain: str, seed: int, split: int) -> Path:
    return root / "shards" / f"{domain}_mseed{seed}_split{split}"


def worker_status_is_compatible(path: Path, expected_contract_hashes: Mapping[str, str]) -> bool:
    try:
        status = read_json(path, label="worker status")
    except A251bError:
        return False
    return bool(
        status.get("experiment_id") == EXPERIMENT_ID
        and status.get("script_version") == SCRIPT_VERSION
        and status.get("complete") is True
        and status.get("passed") is True
        and status.get("selection_only_diagnostics") is True
        and status.get("confirmation_used_for_training") is False
        and status.get("confirmation_used_for_evaluation") is False
        and status.get("contract_hashes") == dict(expected_contract_hashes)
    )


def run_worker(args: argparse.Namespace) -> None:
    if args.torch_threads:
        torch.set_num_threads(int(args.torch_threads))
    output_root = resolve(args.output_dir)
    protocol, contract_frames, contract_hashes = load_contract(args)
    directory = worker_root(output_root, args.target_domain, args.model_seed, args.support_split_seed)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "worker_status.json"
    run_path = directory / "run_level.csv"
    if args.resume and status_path.is_file() and run_path.is_file() and worker_status_is_compatible(status_path, contract_hashes):
        print(f"[A25.1b] resume skip {directory.name}", flush=True)
        return

    cfg, config_path = load_config(protocol, args.config)
    contracts = method_contract(contract_frames["methods"], contract_frames["compute"], protocol)
    audit_rows = runtime_model_audit(cfg, contracts, args.model_seed)
    raw = load_frames(args, protocol, cfg)
    target = str(args.target_domain)
    seed, split = int(args.model_seed), int(args.support_split_seed)
    allowed_seeds = set(int(value) for value in protocol["model_seeds"])
    allowed_splits = set(int(value) for value in protocol["support_split_seeds"])
    if seed not in allowed_seeds or split not in allowed_splits:
        raise A251bError("worker seed/split is not locked by A25.1a")
    roles = contract_frames["roles"]
    tasks = worker_tasks(contract_frames["tasks"], target, seed, split, protocol)
    shots = tuple(int(value) for value in protocol["shots"])
    support_by_shot, selection_engines = selection_and_support(roles, target, split, shots)
    if set(selection_engines) & set().union(*(set(values) for values in support_by_shot.values())):
        raise A251bError("target support/selection engine overlap")
    source_engines = source_fit_engines(tasks, target, raw)
    frames, normalizer_audit = source_normalize(raw, target, source_engines)
    atomic_json(directory / "source_normalizer.json", normalizer_audit)
    schedule = deterministic_schedule(tasks, int(protocol["outer_steps"]), seed + split)
    device = torch.device("cpu") if args.device == "cpu" else a23.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    selection_dataset = a23.WindowDataset(frames[target], selection_engines, int(cfg["window_size"]))
    source_budget = int(protocol["source_gradient_updates_per_method_cell"])
    source_windows_budget = int(protocol["source_window_presentations_per_method_cell"])
    method_audit = {row["method"]: row for row in audit_rows}
    run_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    method_source_states: dict[str, dict[str, torch.Tensor]] = {}
    method_accountings: dict[str, dict[str, Any]] = {}
    start_worker = time.perf_counter()
    for method in METHODS:
        architecture = str(contracts[method]["architecture"])
        a23.seed_everything(seed)
        initial_model = make_model(method, cfg, seed)
        initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
        if state_value_hash(initial_state) != str(method_audit[method]["initial_state_sha256"]):
            raise A251bError(f"initial-state audit mismatch for {method}")
        accounting = initial_accounting(method, architecture, source_budget, source_windows_budget)
        source_start = time.perf_counter()
        if method.startswith("ordinary_"):
            source_model, history = ordinary_source_train(initial_model, schedule, frames, cfg, seed + split, device, accounting)
        else:
            source_model, history = reptile_source_train(initial_model, schedule, frames, cfg, seed + split, device, accounting)
        accounting["source_wall_time_seconds"] = float(time.perf_counter() - source_start)
        histories.extend(history)
        method_source_states[method] = {name: tensor.detach().cpu().clone() for name, tensor in source_model.state_dict().items()}
        method_accountings[method] = accounting

    for architecture, (ordinary, reptile) in PAIRS.items():
        left, right = method_accountings[ordinary], method_accountings[reptile]
        fields = ("source_gradient_updates", "source_window_presentations", "source_forward_calls", "source_backward_calls")
        if any(left[field] != right[field] for field in fields):
            raise A251bError(f"source compute mismatch within {architecture} pair")
        if int(method_audit[ordinary]["initial_state_sha256"] != method_audit[reptile]["initial_state_sha256"]):
            raise A251bError(f"initialization mismatch within {architecture} pair")

    for method in METHODS:
        architecture = str(contracts[method]["architecture"])
        source_state = method_source_states[method]
        for shot in shots:
            support = support_by_shot[shot]
            target_dataset = a23.WindowDataset(frames[target], support, int(cfg["window_size"]))
            model = make_model(method, cfg, seed)
            model.load_state_dict(source_state, strict=True)
            accounting = deepcopy(method_accountings[method])
            target_start = time.perf_counter()
            model, target_history = adapt_target(
                model, target_dataset, cfg,
                seed=seed * 1_000_000 + split * 100 + shot,
                device=device, accounting=accounting,
                label=f"{method} target={target} seed={seed} split={split} K={shot}",
            )
            accounting["target_wall_time_seconds"] = float(time.perf_counter() - target_start)
            metrics = evaluate_selection(model, selection_dataset, cfg, device, accounting)
            accounting["total_wall_time_seconds"] = float(accounting["source_wall_time_seconds"] + accounting["target_wall_time_seconds"])
            checkpoint = directory / f"{method}_shot{shot}_target_adapted.pt"
            payload = {
                "state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "experiment_id": EXPERIMENT_ID,
                "script_version": SCRIPT_VERSION,
                "method": method,
                "architecture": architecture,
                "target_domain": target,
                "model_seed": seed,
                "support_split_seed": split,
                "shot": int(shot),
                "target_support_engine_ids": support,
                "target_selection_engine_ids": selection_engines,
                "confirmation_engines_used": False,
                "contract_hashes": contract_hashes,
                "config_sha256": sha256(config_path),
                "runtime_parameter_audit": method_audit[method],
                "compute_accounting": accounting,
                "target_history": target_history,
            }
            torch.save(payload, checkpoint)
            try:
                loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(checkpoint, map_location="cpu")
            verifier = make_model(method, cfg, seed).cpu()
            verifier.load_state_dict(loaded["state"], strict=True)
            final_numel, final_count, final_hash = state_schema(loaded["state"])
            if (
                final_numel != int(contracts[method]["reference_state_tensor_numel"])
                or final_count != int(contracts[method]["reference_state_tensor_count"])
                or final_hash != str(contracts[method]["reference_state_schema_sha256"])
            ):
                raise A251bError(f"final checkpoint schema mismatch for {method}/K={shot}")
            row = {
                "target_domain": target,
                "model_seed": seed,
                "support_split_seed": split,
                "shot": int(shot),
                "method": method,
                "architecture": architecture,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_reload_passed": True,
                "initial_state_sha256": method_audit[method]["initial_state_sha256"],
                "state_schema_sha256": final_hash,
                "state_tensor_numel": final_numel,
                "state_tensor_count": final_count,
                "total_parameters": method_audit[method]["total_parameters"],
                "trainable_parameters": method_audit[method]["trainable_parameters"],
                "source_gradient_updates": accounting["source_gradient_updates"],
                "source_window_presentations": accounting["source_window_presentations"],
                "source_forward_calls": accounting["source_forward_calls"],
                "source_backward_calls": accounting["source_backward_calls"],
                "target_gradient_updates": accounting["target_gradient_updates"],
                "target_window_presentations": accounting["target_window_presentations"],
                "target_forward_calls": accounting["target_forward_calls"],
                "target_backward_calls": accounting["target_backward_calls"],
                "selection_forward_calls": accounting["selection_forward_calls"],
                "selection_window_presentations": accounting["selection_window_presentations"],
                "source_wall_time_seconds": accounting["source_wall_time_seconds"],
                "target_wall_time_seconds": accounting["target_wall_time_seconds"],
                "total_wall_time_seconds": accounting["total_wall_time_seconds"],
                "peak_cuda_memory_bytes": accounting["peak_cuda_memory_bytes"],
                "selection_used_for_training": False,
                "selection_used_for_evaluation": True,
                "confirmation_used_for_training": False,
                "confirmation_used_for_evaluation": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
            row.update(a23.flatten_metrics("selection", metrics))
            run_rows.append(row)
            histories.extend(target_history)
            del model, verifier
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected_records = len(METHODS) * len(shots)
    if len(run_rows) != expected_records:
        raise A251bError(f"worker record count={len(run_rows)}, expected={expected_records}")
    if time.perf_counter() - start_worker <= 0:
        raise A251bError("non-positive worker runtime")
    pd.DataFrame(run_rows).to_csv(run_path, index=False)
    pd.DataFrame(histories).to_csv(directory / "training_history.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(directory / "runtime_parameter_audit.csv", index=False)
    atomic_json(status_path, {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "complete": True,
        "passed": True,
        "target_domain": target,
        "model_seed": seed,
        "support_split_seed": split,
        "expected_run_records": expected_records,
        "completed_run_records": len(run_rows),
        "source_gradient_update_budget_verified": True,
        "source_window_presentation_budget_verified": True,
        "same_architecture_initialization_verified": True,
        "checkpoint_reload_passed": True,
        "selection_only_diagnostics": True,
        "selection_used_for_training": False,
        "selection_used_for_evaluation": True,
        "confirmation_used_for_training": False,
        "confirmation_used_for_evaluation": False,
        "contract_hashes": contract_hashes,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })


def gpu_inventory() -> list[dict[str, int]]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], text=True,
        )
        result = []
        for line in text.strip().splitlines():
            index, free, utilization = [part.strip() for part in line.split(",")]
            result.append({"index": int(index), "free_mb": int(free), "utilization": int(utilization)})
        return result
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def worker_command(args: argparse.Namespace, domain: str, seed: int, split: int) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--data-dir", str(args.data_dir), "--config", str(args.config),
        "--a25-1a-output-dir", str(args.a25_1a_output_dir),
        "--output-dir", str(args.output_dir),
        "--target-domain", domain, "--model-seed", str(seed),
        "--support-split-seed", str(split), "--device", "cuda",
        "--torch-threads", str(args.torch_threads),
    ]
    if args.resume:
        command.append("--resume")
    return command


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experimentA25_1b_run.lock"
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError):
            path.unlink()
        except PermissionError as exc:
            raise A251bError(f"cannot inspect existing run lock: {path}") from exc
        else:
            raise A251bError(f"another A25.1b parent is active with pid={old_pid}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
        handle.flush(); os.fsync(handle.fileno())
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


def verify_worker_preflight(args: argparse.Namespace, protocol: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], contract_frames: Mapping[str, pd.DataFrame]) -> list[tuple[str, int, int]]:
    workers = [
        (domain, int(seed), int(split))
        for domain in DOMAINS
        for seed in protocol["model_seeds"]
        for split in protocol["support_split_seeds"]
    ]
    tasks = contract_frames["tasks"]
    roles = contract_frames["roles"]
    for domain, seed, split in workers:
        subset = worker_tasks(tasks, domain, seed, split, protocol)
        source_engines = source_fit_engines(subset, domain, frames)
        if set(source_engines) != set(DOMAINS) - {domain}:
            raise A251bError("worker source engine pool domain mismatch")
        support, selection = selection_and_support(roles, domain, split, tuple(int(x) for x in protocol["shots"]))
        if set(selection) & set().union(*(set(values) for values in support.values())):
            raise A251bError("worker target role leakage")
    return workers


def merge(args: argparse.Namespace, protocol: Mapping[str, Any], workers: Sequence[tuple[str, int, int]], contract_hashes: Mapping[str, str]) -> None:
    root = resolve(args.output_dir)
    frames: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    expected_per_worker = len(METHODS) * len(protocol["shots"])
    for domain, seed, split in workers:
        directory = worker_root(root, domain, seed, split)
        status = read_json(directory / "worker_status.json", label="worker status")
        if not worker_status_is_compatible(directory / "worker_status.json", contract_hashes):
            raise A251bError(f"incomplete or incompatible worker: {directory.name}")
        if int(status.get("completed_run_records", -1)) != expected_per_worker:
            raise A251bError(f"worker record count mismatch: {directory.name}")
        frame = load_csv(directory / "run_level.csv", label="worker run level")
        if len(frame) != expected_per_worker:
            raise A251bError(f"worker run level count mismatch: {directory.name}")
        frames.append(frame)
        audits.append(load_csv(directory / "runtime_parameter_audit.csv", label="worker parameter audit"))
        histories.append(load_csv(directory / "training_history.csv", label="worker history"))
    merged = pd.concat(frames, ignore_index=True)
    expected = len(workers) * expected_per_worker
    if len(merged) != expected:
        raise A251bError(f"merged run records={len(merged)}, expected={expected}")
    required_false = (
        "selection_used_for_training", "confirmation_used_for_training",
        "confirmation_used_for_evaluation", "official_test_files_accessed", "official_test_forward_run",
    )
    for column in required_false:
        if strict_bool(merged, column).any():
            raise A251bError(f"merged boundary violation: {column}")
    if not strict_bool(merged, "selection_used_for_evaluation").all():
        raise A251bError("selection diagnostics were not evaluated for every record")
    if not strict_bool(merged, "checkpoint_reload_passed").all():
        raise A251bError("one or more checkpoint reloads failed")
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method"]
    if merged.duplicated(keys).any():
        raise A251bError("duplicate merged method/shot record")
    source_budget = int(protocol["source_gradient_updates_per_method_cell"])
    source_window_budget = int(protocol["source_window_presentations_per_method_cell"])
    if not (pd.to_numeric(merged["source_gradient_updates"], errors="raise") == source_budget).all():
        raise A251bError("merged source gradient accounting differs from contract")
    if not (pd.to_numeric(merged["source_window_presentations"], errors="raise") == source_window_budget).all():
        raise A251bError("merged source window accounting differs from contract")
    pair_rows = []
    for architecture, (ordinary, reptile) in PAIRS.items():
        for shot in protocol["shots"]:
            left = merged.loc[(merged["method"] == ordinary) & (merged["shot"] == int(shot))].sort_values(keys[:3])
            right = merged.loc[(merged["method"] == reptile) & (merged["shot"] == int(shot))].sort_values(keys[:3])
            if len(left) != len(right) or len(left) != len(workers):
                raise A251bError(f"algorithm pair row mismatch: {architecture}/K={shot}")
            compare_columns = (
                "initial_state_sha256", "state_schema_sha256", "state_tensor_numel", "state_tensor_count",
                "total_parameters", "trainable_parameters", "source_gradient_updates",
                "source_window_presentations", "source_forward_calls", "source_backward_calls",
                "target_gradient_updates", "target_window_presentations", "target_forward_calls", "target_backward_calls",
            )
            for column in compare_columns:
                if left[column].tolist() != right[column].tolist():
                    raise A251bError(f"matched accounting mismatch: {architecture}/K={shot}/{column}")
            pair_rows.append({
                "architecture": architecture,
                "shot": int(shot),
                "paired_worker_cells": len(left),
                "initialization_identical": True,
                "state_schema_identical": True,
                "parameter_counts_identical": True,
                "source_gradient_updates_identical": True,
                "source_window_presentations_identical": True,
                "target_gradient_updates_identical": True,
                "target_window_presentations_identical": True,
            })
    run_path = root / "experimentA25_1b_selection_run_level.csv"
    audit_path = root / "experimentA25_1b_runtime_parameter_audit.csv"
    history_path = root / "experimentA25_1b_training_history.csv"
    matched_path = root / "experimentA25_1b_matched_compute_audit.csv"
    merged.to_csv(run_path, index=False)
    pd.concat(audits, ignore_index=True).drop_duplicates().to_csv(audit_path, index=False)
    pd.concat(histories, ignore_index=True).to_csv(history_path, index=False)
    pd.DataFrame(pair_rows).to_csv(matched_path, index=False)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "pilot_only": True,
        "selection_only_diagnostics": True,
        "formal_efficacy_claim": False,
        "methods": list(METHODS),
        "shots": [int(value) for value in protocol["shots"]],
        "expected_worker_cells": len(workers),
        "completed_worker_cells": len(workers),
        "expected_run_records": expected,
        "completed_run_records": int(len(merged)),
        "same_architecture_initialization_assertion_passed": True,
        "same_architecture_state_schema_assertion_passed": True,
        "same_architecture_compute_accounting_assertion_passed": True,
        "checkpoint_reload_passed": True,
        "confirmation_engines_evaluated": False,
        "selection_used_for_training": False,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": "A25.1b completed the same-architecture, compute-accounted 2x2 selection-only pilot",
        "interpretation_limit": "A25.1b selection metrics are implementation diagnostics only. They cannot create a confirmatory meta-learning efficacy claim or select a deployment policy.",
        "next_action": "inspect_A25_1b_integrity_then_preregister_external_or_sealed_confirmation_before_efficacy_claims",
    }
    atomic_json(root / "experimentA25_1b_confirmation_decision.json", decision)
    input_hashes = dict(contract_hashes)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "artifacts": {},
        "pilot_only": True,
        "selection_only_diagnostics": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for path in (run_path, audit_path, history_path, matched_path, root / "experimentA25_1b_confirmation_decision.json"):
        manifest["artifacts"][path.name] = sha256(path)
    atomic_json(root / "experimentA25_1b_manifest.json", manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def parent(args: argparse.Namespace) -> None:
    if args.torch_threads:
        torch.set_num_threads(int(args.torch_threads))
    protocol, frames_contract, contract_hashes = load_contract(args)
    cfg, config_path = load_config(protocol, args.config)
    contracts = method_contract(frames_contract["methods"], frames_contract["compute"], protocol)
    raw = load_frames(args, protocol, cfg)
    workers = verify_worker_preflight(args, protocol, raw, frames_contract)
    parameter_audit = runtime_model_audit(cfg, contracts, int(protocol["model_seeds"][0]))
    smoke = runtime_smoke(cfg, int(protocol["model_seeds"][0]))
    expected_records = len(workers) * len(METHODS) * len(protocol["shots"])
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": protocol["registered_primary_question"],
        "output_dir": str(resolve(args.output_dir)),
        "methods": list(METHODS),
        "model_seeds": protocol["model_seeds"],
        "support_split_seeds": protocol["support_split_seeds"],
        "shots": protocol["shots"],
        "source_gradient_updates_per_method_cell": protocol["source_gradient_updates_per_method_cell"],
        "source_window_presentations_per_method_cell": protocol["source_window_presentations_per_method_cell"],
        "target_epochs": protocol["target_epochs"],
        "expected_worker_cells": len(workers),
        "expected_run_records": expected_records,
        "runtime_parameter_audit": parameter_audit,
        "runtime_model_smoke": smoke,
        "all_worker_roles_and_source_tasks_preflighted": True,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "selection_only_diagnostics": True,
        "confirmation_engines_evaluated": False,
        "new_predictor_training": not args.dry_run,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print("[A25.1b] dry-run passed; all runtime contracts are compatible and no predictor was trained", flush=True)
        return
    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        for domain, seed, split in workers:
            local = deepcopy(args)
            local.worker, local.target_domain = True, domain
            local.model_seed, local.support_split_seed = seed, split
            run_worker(local)
    else:
        inventory = gpu_inventory()
        eligible = [
            item["index"] for item in inventory
            if item["index"] in set(args.gpu_ids)
            and item["free_mb"] >= args.min_free_memory_mb
            and item["utilization"] <= args.max_gpu_utilization
        ]
        if not eligible:
            raise A251bError(f"no eligible GPU; inventory={inventory}")
        eligible = eligible[: min(int(args.max_workers), len(eligible))]
        pending = list(workers)
        active: dict[int, tuple[subprocess.Popen[str], Any, tuple[str, int, int], Path]] = {}
        while pending or active:
            for gpu in eligible:
                if gpu in active or not pending:
                    continue
                item = pending.pop(0)
                directory = worker_root(root, *item)
                directory.mkdir(parents=True, exist_ok=True)
                log = directory / "worker_training.log"
                handle = log.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    worker_command(args, *item), cwd=PROJECT_ROOT, env=env,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
                active[gpu] = (process, handle, item, log)
                print(f"[A25.1b] launched target={item[0]} seed={item[1]} split={item[2]} gpu={gpu} pid={process.pid}", flush=True)
            finished: list[int] = []
            for gpu, (process, handle, item, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                if code != 0:
                    for running, output, _, _ in active.values():
                        if running.poll() is None:
                            running.terminate()
                        output.close()
                    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-160:])
                    raise A251bError(f"worker failed item={item} exit={code}\n{tail}")
                print(f"[A25.1b] completed target={item[0]} seed={item[1]} split={item[2]} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(3)
    merge(args, protocol, workers, contract_hashes)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
        return 0
    if args.dry_run:
        parent(args)
        return 0
    root = resolve(args.output_dir)
    decision_path = root / "experimentA25_1b_confirmation_decision.json"
    if decision_path.is_file():
        prior = read_json(decision_path, label="existing A25.1b decision")
        if args.resume and prior.get("complete") is True and prior.get("passed") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A25.1b] resume: existing complete decision returned", flush=True)
            return 0
        raise A251bError(
            f"output already contains a final decision: {decision_path}; use --resume or a new directory"
        )
    lock = acquire_lock(root)
    try:
        parent(args)
    finally:
        release_lock(lock)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A251bError as exc:
        print(f"[A25.1b] error: {exc.__class__.__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("[A25.1b] interrupted by user", file=sys.stderr, flush=True)
        raise SystemExit(130)
