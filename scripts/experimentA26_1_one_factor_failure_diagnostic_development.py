#!/usr/bin/env python3
"""A26.1: one-factor failure diagnostics on development roles only.

The experiment consumes the frozen A26.0 contract and the audited A25.1b
selection-only pilot.  It never accepts an A25.2b path and therefore cannot
read sealed-confirmation predictions or metrics.  Official C-MAPSS test files
are also prohibited.

For every target-domain/model-seed/support-split worker, A26.1:

* evaluates the four frozen A25.1b K=5 checkpoints on causal prefixes from the
  *selection* engines (development evidence only);
* trains one compute-matched Reptile outer-learning-rate-half diagnostic for
  each architecture;
* evaluates target-adaptation snapshots at 0, 1, and 10 epochs from the exact
  same source state; and
* evaluates a GNN graph-bypass inference ablation from each identical saved
  state (no additional training or checkpoint selection).

All results are descriptive and exploratory.  The program reports every
registered cell and deliberately contains no automatic winner/candidate
selection rule.
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
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_2_causal_prefix_endpoint_audit as a232  # noqa: E402
from scripts import experimentA25_1b_same_architecture_compute_accounted_selection_pilot as a251b  # noqa: E402


EXPERIMENT_ID = "experimentA26_1"
SCRIPT_VERSION = "experimentA26_1_one_factor_failure_diagnostic_development_v1"
RUN_TOKEN = "A26.1_EXPLORATORY_RUN"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (140, 141)
SUPPORT_SPLIT_SEEDS = (7501, 7502)
PRIMARY_SHOT = 5
ANCHORS = (90.0, 45.0, 15.0)
TARGET_SNAPSHOT_EPOCHS = (0, 1, 10)
OUTER_LR_MULTIPLIER = 0.5
HISTORICAL_METHODS = (
    "ordinary_no_graph_pft",
    "reptile_meta_no_graph",
    "ordinary_gnn_pft",
    "reptile_meta_gnn",
)
REPTILE_METHOD_BY_ARCHITECTURE = {
    "no_graph": "reptile_meta_no_graph",
    "gnn": "reptile_meta_gnn",
}
EXPECTED_VARIANTS = (
    "ordinary_no_graph_locked_target10",
    "reptile_no_graph_locked_target10",
    "ordinary_gnn_locked_target10",
    "reptile_gnn_locked_target10",
    "reptile_no_graph_outer_half_target0",
    "reptile_no_graph_outer_half_target1",
    "reptile_no_graph_outer_half_target10",
    "reptile_gnn_outer_half_target0",
    "reptile_gnn_outer_half_target1",
    "reptile_gnn_outer_half_target10",
    "reptile_gnn_outer_half_target0_graph_bypass",
    "reptile_gnn_outer_half_target1_graph_bypass",
    "reptile_gnn_outer_half_target10_graph_bypass",
)
EXPECTED_PAIR_TYPES = {
    "historical_algorithm_effect": 2,
    "outer_lr_half_effect": 2,
    "target_adaptation_0_vs_10": 2,
    "target_adaptation_1_vs_10": 2,
    "graph_bypass_effect": 3,
}
HASH_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
TRUE_TEXT = {"true", "1", "yes"}
FALSE_TEXT = {"false", "0", "no"}


class A261Error(RuntimeError):
    """Raised when the A26.1 diagnostic or a frozen boundary is violated."""


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_value_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_delta_l2(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if set(left) != set(right):
        raise A261Error("state keys differ while computing parameter drift")
    total = 0.0
    for name in left:
        delta = right[name].detach().cpu().double() - left[name].detach().cpu().double()
        total += float(torch.sum(delta * delta))
    return float(math.sqrt(total))


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


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise A261Error(f"refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise A261Error(f"required {label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A261Error(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A261Error(f"{label} must be a JSON object: {path}")
    return value


def read_frame(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise A261Error(f"required {label} is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise A261Error(f"cannot read {label}: {path}: {exc}") from exc
    if frame.empty:
        raise A261Error(f"{label} is empty: {path}")
    return frame


def require_fields(payload: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = sorted(set(fields) - set(payload))
    if missing:
        raise A261Error(f"{label} lacks required fields: {missing}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise A261Error(f"{label} lacks required columns: {missing}")


def strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise A261Error(f"{label} must be Boolean, observed {value!r}")


def require_true(value: Any, *, label: str) -> None:
    if not strict_bool(value, label=label):
        raise A261Error(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if strict_bool(value, label=label):
        raise A261Error(f"{label} must be false")


def require_hash(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if HASH_RE.fullmatch(text) is None:
        raise A261Error(f"{label} is not a SHA256 digest: {value!r}")
    return text


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise A261Error("--gpus must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise A261Error("--gpus must be non-empty, unique, non-negative integers")
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
        "--a25-1b-output-dir",
        type=Path,
        default=Path("outputs/experimentA25_1b_same_architecture_compute_accounted_selection_pilot_v3"),
    )
    parser.add_argument(
        "--a25-1b-script",
        type=Path,
        default=Path("scripts/experimentA25_1b_same_architecture_compute_accounted_selection_pilot.py"),
    )
    parser.add_argument(
        "--a26-0-output-dir",
        type=Path,
        default=Path("outputs/experimentA26_0_failure_diagnostic_contract_preflight"),
    )
    parser.add_argument(
        "--a26-0-script",
        type=Path,
        default=Path("scripts/experimentA26_0_failure_diagnostic_contract_preflight.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA26_1_one_factor_failure_diagnostic_development"),
    )
    parser.add_argument("--gpus", default="0", help="Physical GPU ids, e.g. 6,7")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-free-memory-mb", type=int, default=16000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-run", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    parser.add_argument("--model-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--support-split-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.gpu_ids = parse_gpu_ids(args.gpus)
    if args.max_workers < 1 or args.torch_threads < 1:
        raise A261Error("--max-workers and --torch-threads must be positive")
    if args.min_free_memory_mb < 0 or not 0 <= args.max_gpu_utilization <= 100:
        raise A261Error("invalid GPU eligibility thresholds")
    if not args.dry_run and args.confirm_run != RUN_TOKEN:
        raise A261Error(f"formal exploratory run requires --confirm-run {RUN_TOKEN}")
    if args.worker and (
        args.target_domain is None or args.model_seed is None or args.support_split_seed is None
    ):
        raise A261Error("worker mode requires target-domain, model-seed and support-split-seed")
    return args


def validate_hash_map(root: Path, mapping: Any, *, label: str) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise A261Error(f"{label} must be a non-empty hash mapping")
    verified: dict[str, str] = {}
    for name, expected in sorted(mapping.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise A261Error(f"unsafe artifact name in {label}: {name!r}")
        expected_hash = require_hash(expected, label=f"{label} {name}")
        path = root / name
        if not path.is_file():
            raise A261Error(f"artifact in {label} is missing: {path}")
        observed = sha256(path)
        if observed != expected_hash:
            raise A261Error(
                f"artifact hash mismatch in {label}: {name}: expected={expected_hash}, observed={observed}"
            )
        verified[name] = observed
    return verified


def validate_a26_0(args: argparse.Namespace) -> dict[str, str]:
    root = resolve(args.a26_0_output_dir)
    script = resolve(args.a26_0_script)
    if not root.is_dir() or not script.is_file():
        raise A261Error("A26.0 output directory or script is missing")
    manifest_path = root / "experimentA26_0_manifest.json"
    decision_path = root / "experimentA26_0_confirmation_decision.json"
    manifest = read_json(manifest_path, label="A26.0 manifest")
    decision = read_json(decision_path, label="A26.0 decision")
    if manifest.get("experiment_id") != "experimentA26_0" or decision.get("experiment_id") != "experimentA26_0":
        raise A261Error("A26.0 identity mismatch")
    if manifest.get("script_version") != "experimentA26_0_failure_diagnostic_contract_preflight_v1":
        raise A261Error("unsupported A26.0 script version")
    expected_script_hash = require_hash(manifest.get("script_sha256"), label="A26.0 script hash")
    if sha256(script) != expected_script_hash:
        raise A261Error("A26.0 script hash mismatch")
    verified = validate_hash_map(root, manifest.get("artifacts"), label="A26.0 artifacts")
    required_true = (
        "complete",
        "passed",
        "preflight_only",
        "exploratory_only",
        "A25_2b_confirmation_frozen_read_only",
        "one_factor_at_a_time_diagnostics_registered",
        "same_architecture_and_initialization_required",
        "matched_compute_required",
    )
    required_false = (
        "A25_2b_confirmation_reused_for_tuning",
        "A25_2b_confirmation_available_to_A26_1_candidate_selector",
        "source_formal_efficacy_claim_supported",
        "new_predictor_training",
        "checkpoint_tensors_opened",
        "model_forward_run",
        "official_test_files_accessed",
        "official_test_forward_run",
        "formal_efficacy_claim",
    )
    for field in required_true:
        require_true(decision.get(field), label=f"A26.0 decision {field}")
    for field in required_false:
        require_false(decision.get(field), label=f"A26.0 decision {field}")
    expected_counts = {
        "source_no_graph_checks_passed": 2,
        "source_no_graph_checks_expected": 6,
        "source_gnn_checks_passed": 0,
        "source_gnn_checks_expected": 6,
    }
    for field, expected in expected_counts.items():
        if int(decision.get(field, -1)) != expected:
            raise A261Error(f"A26.0 decision changed {field}")
    require_false(decision.get("source_low_rul_safety_gate_passed"), label="A26.0 low-RUL gate")
    if decision.get("next_action") != "implement_A26_1_one_factor_failure_diagnostic_development_experiment":
        raise A261Error("A26.0 does not authorize the registered A26.1 diagnostic")

    registry = read_frame(root / "experimentA26_0_component_diagnostic_registry.csv", label="A26.0 diagnostic registry")
    require_columns(
        registry,
        (
            "diagnostic_id",
            "phase",
            "change_policy",
            "A25_2b_confirmation_available_to_candidate_selector",
            "official_test_allowed",
            "confirmatory_claim_allowed",
        ),
        label="A26.0 diagnostic registry",
    )
    if set(registry["diagnostic_id"].astype(str)) != {f"D{i}" for i in range(1, 7)} or len(registry) != 6:
        raise A261Error("A26.0 diagnostic registry must contain exactly D1-D6")
    if set(registry["phase"].astype(str)) != {"A26.1_development_only"}:
        raise A261Error("A26.0 diagnostic phase changed")
    for column in (
        "A25_2b_confirmation_available_to_candidate_selector",
        "official_test_allowed",
        "confirmatory_claim_allowed",
    ):
        if registry[column].map(lambda value: strict_bool(value, label=column)).any():
            raise A261Error(f"A26.0 registry permits forbidden action: {column}")

    roles = read_frame(root / "experimentA26_0_data_role_contract.csv", label="A26.0 data-role contract")
    require_columns(roles, ("data_or_evidence", "role", "A26_1_access", "prohibited_actions"), label="A26.0 data roles")
    official = roles.loc[roles["data_or_evidence"].astype(str) == "official C-MAPSS test files"]
    confirmation = roles.loc[roles["data_or_evidence"].astype(str) == "A25.2b confirmation outcomes"]
    if len(official) != 1 or str(official.iloc[0]["A26_1_access"]) != "forbidden":
        raise A261Error("A26.0 does not keep official test data sealed in A26.1")
    if len(confirmation) != 1 or "not_available" not in str(confirmation.iloc[0]["A26_1_access"]):
        raise A261Error("A26.0 does not isolate A25.2b confirmation from A26.1")

    protocol = read_json(root / "experimentA26_0_diagnostic_protocol.json", label="A26.0 protocol")
    for field in (
        "A25_2b_confirmation_is_immutable",
        "A26_1_exploratory_only",
        "one_factor_at_a_time_required",
        "matched_compute_required",
        "same_architecture_and_initialization_required",
    ):
        require_true(protocol.get(field), label=f"A26.0 protocol {field}")
    for field in (
        "A25_2b_confirmation_available_to_A26_1_candidate_selector",
        "A25_2b_confirmation_reuse_for_tuning",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(protocol.get(field), label=f"A26.0 protocol {field}")
    verified.update(
        {
            manifest_path.name: sha256(manifest_path),
            "A26_0_script": expected_script_hash,
        }
    )
    return {f"A26_0::{key}": value for key, value in verified.items()}


def locate_checkpoint(root: Path, row: Mapping[str, Any]) -> Path:
    original = Path(str(row["checkpoint"])).expanduser()
    if original.is_file():
        path = original.resolve()
    else:
        directory = root / "shards" / (
            f"{row['target_domain']}_mseed{int(row['model_seed'])}_split{int(row['support_split_seed'])}"
        )
        path = (directory / original.name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise A261Error(f"A25.1b checkpoint escapes its output root: {path}") from exc
    if not path.is_file():
        raise A261Error(f"A25.1b checkpoint is missing: {path}")
    expected = require_hash(row["checkpoint_sha256"], label=f"checkpoint {path.name}")
    if sha256(path) != expected:
        raise A261Error(f"A25.1b checkpoint hash mismatch: {path}")
    return path


def validate_a25_1b(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    root = resolve(args.a25_1b_output_dir)
    script = resolve(args.a25_1b_script)
    if not root.is_dir() or not script.is_file():
        raise A261Error("A25.1b output directory or script is missing")
    imported_script = Path(a251b.__file__).resolve()
    if imported_script != script:
        raise A261Error(
            f"imported A25.1b module differs from --a25-1b-script: {imported_script} != {script}"
        )
    manifest_path = root / "experimentA25_1b_manifest.json"
    decision_path = root / "experimentA25_1b_confirmation_decision.json"
    manifest = read_json(manifest_path, label="A25.1b manifest")
    decision = read_json(decision_path, label="A25.1b decision")
    if manifest.get("experiment_id") != "experimentA25_1b" or decision.get("experiment_id") != "experimentA25_1b":
        raise A261Error("A25.1b identity mismatch")
    if manifest.get("script_version") != a251b.SCRIPT_VERSION:
        raise A261Error("A25.1b manifest version differs from imported module")
    script_hash = require_hash(manifest.get("script_sha256"), label="A25.1b script hash")
    if sha256(script) != script_hash:
        raise A261Error("A25.1b script hash mismatch")
    verified = validate_hash_map(root, manifest.get("artifacts"), label="A25.1b artifacts")
    for field in (
        "complete",
        "passed",
        "pilot_only",
        "selection_only_diagnostics",
        "same_architecture_initialization_assertion_passed",
        "same_architecture_state_schema_assertion_passed",
        "same_architecture_compute_accounting_assertion_passed",
        "checkpoint_reload_passed",
    ):
        require_true(decision.get(field), label=f"A25.1b decision {field}")
    for field in (
        "formal_efficacy_claim",
        "confirmation_engines_evaluated",
        "selection_used_for_training",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(decision.get(field), label=f"A25.1b decision {field}")
    if int(decision.get("completed_worker_cells", -1)) != 16 or int(decision.get("completed_run_records", -1)) != 192:
        raise A261Error("A25.1b completion cardinality mismatch")

    run = read_frame(root / "experimentA25_1b_selection_run_level.csv", label="A25.1b run level")
    required = {
        "target_domain",
        "model_seed",
        "support_split_seed",
        "shot",
        "method",
        "architecture",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_reload_passed",
        "state_schema_sha256",
        "source_gradient_updates",
        "source_window_presentations",
        "selection_used_for_training",
        "selection_used_for_evaluation",
        "confirmation_used_for_training",
        "confirmation_used_for_evaluation",
        "official_test_files_accessed",
        "official_test_forward_run",
    }
    require_columns(run, required, label="A25.1b run level")
    if len(run) != 192:
        raise A261Error("A25.1b run level must contain 192 rows")
    primary = run.loc[pd.to_numeric(run["shot"], errors="raise").astype(int) == PRIMARY_SHOT].copy()
    if len(primary) != 64:
        raise A261Error("A25.1b K=5 inventory must contain 64 checkpoints")
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method"]
    if primary.duplicated(keys).any():
        raise A261Error("A25.1b K=5 checkpoint keys are duplicated")
    expected_keys = {
        (domain, seed, split, PRIMARY_SHOT, method)
        for domain in DOMAINS
        for seed in MODEL_SEEDS
        for split in SUPPORT_SPLIT_SEEDS
        for method in HISTORICAL_METHODS
    }
    observed_keys = {
        (str(row.target_domain), int(row.model_seed), int(row.support_split_seed), int(row.shot), str(row.method))
        for row in primary.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise A261Error("A25.1b K=5 checkpoint factorial is incomplete")
    for index, row in primary.iterrows():
        label = f"A25.1b K=5 row {index}"
        require_true(row["checkpoint_reload_passed"], label=f"{label} checkpoint reload")
        require_false(row["selection_used_for_training"], label=f"{label} selection training")
        require_true(row["selection_used_for_evaluation"], label=f"{label} selection evaluation")
        require_false(row["confirmation_used_for_training"], label=f"{label} confirmation training")
        require_false(row["confirmation_used_for_evaluation"], label=f"{label} confirmation evaluation")
        require_false(row["official_test_files_accessed"], label=f"{label} official test access")
        require_false(row["official_test_forward_run"], label=f"{label} official test forward")
        if int(row["source_gradient_updates"]) != 7500 or int(row["source_window_presentations"]) != 480000:
            raise A261Error(f"{label} source compute differs from locked budget")
        primary.at[index, "resolved_checkpoint"] = str(locate_checkpoint(root, row))
    verified.update(
        {
            manifest_path.name: sha256(manifest_path),
            "A25_1b_script": script_hash,
        }
    )
    return primary.reset_index(drop=True), {f"A25_1b::{key}": value for key, value in verified.items()}


def load_contract_and_data(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, str], dict[str, Any], Path, dict[str, pd.DataFrame]]:
    try:
        protocol, frames, hashes = a251b.load_contract(args)
        cfg, config_path = a251b.load_config(protocol, args.config)
        raw = a251b.load_frames(args, protocol, cfg)
    except Exception as exc:
        raise A261Error(f"A25.1a contract/data validation failed: {exc}") from exc
    if tuple(int(value) for value in protocol["model_seeds"]) != MODEL_SEEDS:
        raise A261Error("A25.1a model seeds changed")
    if tuple(int(value) for value in protocol["support_split_seeds"]) != SUPPORT_SPLIT_SEEDS:
        raise A261Error("A25.1a support split seeds changed")
    if PRIMARY_SHOT not in tuple(int(value) for value in protocol["shots"]):
        raise A261Error("A25.1a does not contain the registered K=5 shot")
    if int(protocol["target_epochs"]) != 10:
        raise A261Error("A25.1a target epoch contract changed")
    if int(protocol["source_gradient_updates_per_method_cell"]) != 7500:
        raise A261Error("A25.1a source gradient budget changed")
    if int(protocol["source_window_presentations_per_method_cell"]) != 480000:
        raise A261Error("A25.1a source window budget changed")
    contracts = a251b.method_contract(frames["methods"], frames["compute"], protocol)
    return protocol, frames, hashes, cfg, config_path, raw


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise A261Error("installed PyTorch lacks safe weights_only checkpoint loading") from exc
    except Exception as exc:
        raise A261Error(f"safe checkpoint load failed: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise A261Error(f"checkpoint has no state dictionary: {path}")
    if not payload["state"] or not all(isinstance(value, torch.Tensor) for value in payload["state"].values()):
        raise A261Error(f"checkpoint state is invalid: {path}")
    for name, tensor in payload["state"].items():
        if not isinstance(name, str) or not bool(torch.isfinite(tensor).all().item()):
            raise A261Error(f"checkpoint contains invalid/non-finite state entry: {path}: {name!r}")
    return payload


def worker_root(root: Path, domain: str, seed: int, split: int) -> Path:
    return root / "shards" / f"{domain}_mseed{seed}_split{split}"


def expected_workers() -> list[tuple[str, int, int]]:
    return [
        (domain, seed, split)
        for domain in DOMAINS
        for seed in MODEL_SEEDS
        for split in SUPPORT_SPLIT_SEEDS
    ]


def variant_metadata(variant: str) -> dict[str, Any]:
    if variant.startswith("ordinary_no_graph"):
        return {
            "architecture": "no_graph",
            "base_method": "ordinary_no_graph_pft",
            "source_algorithm": "ordinary_locked_A25_1b",
            "outer_lr_multiplier": 0.0,
            "target_epochs": 10,
            "graph_bypass": False,
            "checkpoint_origin": "A25_1b_frozen",
        }
    if variant.startswith("reptile_no_graph_locked"):
        return {
            "architecture": "no_graph",
            "base_method": "reptile_meta_no_graph",
            "source_algorithm": "reptile_locked_A25_1b",
            "outer_lr_multiplier": 1.0,
            "target_epochs": 10,
            "graph_bypass": False,
            "checkpoint_origin": "A25_1b_frozen",
        }
    if variant.startswith("ordinary_gnn"):
        return {
            "architecture": "gnn",
            "base_method": "ordinary_gnn_pft",
            "source_algorithm": "ordinary_locked_A25_1b",
            "outer_lr_multiplier": 0.0,
            "target_epochs": 10,
            "graph_bypass": False,
            "checkpoint_origin": "A25_1b_frozen",
        }
    if variant.startswith("reptile_gnn_locked"):
        return {
            "architecture": "gnn",
            "base_method": "reptile_meta_gnn",
            "source_algorithm": "reptile_locked_A25_1b",
            "outer_lr_multiplier": 1.0,
            "target_epochs": 10,
            "graph_bypass": False,
            "checkpoint_origin": "A25_1b_frozen",
        }
    architecture = "gnn" if "_gnn_" in variant else "no_graph"
    target_epochs = next(
        epoch for epoch in sorted(TARGET_SNAPSHOT_EPOCHS, reverse=True)
        if f"target{epoch}" in variant
    )
    return {
        "architecture": architecture,
        "base_method": REPTILE_METHOD_BY_ARCHITECTURE[architecture],
        "source_algorithm": "reptile_outer_lr_half",
        "outer_lr_multiplier": OUTER_LR_MULTIPLIER,
        "target_epochs": target_epochs,
        "graph_bypass": variant.endswith("graph_bypass"),
        "checkpoint_origin": "A26_1_new",
    }


def historical_variant(method: str) -> str:
    mapping = {
        "ordinary_no_graph_pft": "ordinary_no_graph_locked_target10",
        "reptile_meta_no_graph": "reptile_no_graph_locked_target10",
        "ordinary_gnn_pft": "ordinary_gnn_locked_target10",
        "reptile_meta_gnn": "reptile_gnn_locked_target10",
    }
    try:
        return mapping[method]
    except KeyError as exc:
        raise A261Error(f"unknown historical method: {method}") from exc


def snapshot_variant(architecture: str, epoch: int, *, graph_bypass: bool = False) -> str:
    value = f"reptile_{architecture}_outer_half_target{epoch}"
    return value + ("_graph_bypass" if graph_bypass else "")


def source_diagnostic_train(
    model: torch.nn.Module,
    schedule: Sequence[Mapping[str, Any]],
    normalized: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    seed: int,
    device: torch.device,
    accounting: dict[str, Any],
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    base = model.cpu()
    cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    report_every = max(1, len(schedule) // 10)
    history: list[dict[str, Any]] = []
    block_losses: list[float] = []
    block_adaptation_norms: list[float] = []
    block_applied_norms: list[float] = []
    for outer_index, row in enumerate(schedule, start=1):
        adapted = deepcopy(base).to(device)
        optimiser = torch.optim.Adam(adapted.parameters(), lr=float(cfg["inner_lr"]))
        loader = a251b.episode_loader(normalized, row, cfg, cache, seed + outer_index * 31)
        iterator = iter(loader)
        for _ in range(int(cfg["inner_steps"])):
            batch, iterator = a251b.next_full_batch(iterator, loader, int(cfg["batch_size"]))
            block_losses.append(
                a251b.step_update(
                    adapted,
                    optimiser,
                    batch,
                    device,
                    float(cfg["pair_aux_weight"]),
                    accounting,
                    "source",
                )
            )
        adapted = adapted.cpu()
        base_parameters = dict(base.named_parameters())
        adapted_parameters = dict(adapted.named_parameters())
        if set(base_parameters) != set(adapted_parameters):
            raise A261Error("Reptile base/adapted parameter keys differ")
        squared = 0.0
        with torch.no_grad():
            for name, parameter in base_parameters.items():
                delta = adapted_parameters[name] - parameter
                squared += float(torch.sum(delta.detach().double() ** 2))
                parameter.add_(float(cfg["outer_lr"]) * delta)
        adaptation_norm = math.sqrt(squared)
        block_adaptation_norms.append(adaptation_norm)
        block_applied_norms.append(float(cfg["outer_lr"]) * adaptation_norm)
        if outer_index % report_every == 0 or outer_index == len(schedule):
            history.append(
                {
                    "outer_block": outer_index,
                    "source_gradient_updates": accounting["source_gradient_updates"],
                    "mean_inner_loss": float(np.mean(block_losses)),
                    "mean_episode_adaptation_delta_l2": float(np.mean(block_adaptation_norms)),
                    "mean_applied_outer_update_l2": float(np.mean(block_applied_norms)),
                    "outer_learning_rate": float(cfg["outer_lr"]),
                    "inner_learning_rate": float(cfg["inner_lr"]),
                    "inner_steps": int(cfg["inner_steps"]),
                }
            )
            print(
                f"[A26.1] {accounting['method']} outer={outer_index:04d}/{len(schedule)} "
                f"loss={np.mean(block_losses):.6f} update_l2={np.mean(block_applied_norms):.6f}",
                flush=True,
            )
            block_losses.clear()
            block_adaptation_norms.clear()
            block_applied_norms.clear()
        del adapted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if accounting["source_gradient_updates"] != accounting["source_gradient_updates_budget"]:
        raise A261Error("diagnostic source gradient update count differs from contract")
    if accounting["source_window_presentations"] != accounting["source_window_presentations_budget"]:
        raise A261Error("diagnostic source window presentation count differs from contract")
    return base.cpu(), history


def target_adaptation_snapshots(
    source_state: Mapping[str, torch.Tensor],
    method: str,
    dataset: Any,
    cfg: Mapping[str, Any],
    model_seed: int,
    loader_seed: int,
    device: torch.device,
    source_accounting: Mapping[str, Any],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    # Match A25.1b exactly: model construction resets global stochastic layers
    # with model_seed, while target batch ordering uses its separate run seed.
    model = a251b.make_model(method, cfg, model_seed).cpu()
    model.load_state_dict(source_state, strict=True)
    states = {0: {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
    accountings = {0: deepcopy(dict(source_accounting))}
    history: list[dict[str, Any]] = []
    loader = a251b.a23.make_loader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=loader_seed,
    )
    learner = model.to(device)
    optimiser = torch.optim.Adam(learner.parameters(), lr=float(cfg["inner_lr"]))
    accounting = deepcopy(dict(source_accounting))
    for epoch in range(1, max(TARGET_SNAPSHOT_EPOCHS) + 1):
        losses: list[float] = []
        for batch in loader:
            losses.append(
                a251b.step_update(
                    learner,
                    optimiser,
                    batch,
                    device,
                    float(cfg["pair_aux_weight"]),
                    accounting,
                    "target",
                )
            )
        if not losses:
            raise A261Error("target support loader produced zero batches")
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "target_gradient_updates": int(accounting["target_gradient_updates"]),
                "target_window_presentations": int(accounting["target_window_presentations"]),
            }
        )
        if epoch in TARGET_SNAPSHOT_EPOCHS:
            states[epoch] = {
                name: value.detach().cpu().clone() for name, value in learner.state_dict().items()
            }
            accountings[epoch] = deepcopy(accounting)
    if set(states) != set(TARGET_SNAPSHOT_EPOCHS):
        raise A261Error("target adaptation snapshots are incomplete")
    return states, accountings, history


@torch.inference_mode()
def predict_state(
    state: Mapping[str, torch.Tensor],
    *,
    method: str,
    variant: str,
    dataset: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    target_domain: str,
    model_seed: int,
    support_split_seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    graph_bypass: bool,
) -> pd.DataFrame:
    model = a251b.make_model(method, cfg, model_seed).cpu()
    model.load_state_dict(state, strict=True)
    original_use_gat = bool(getattr(model, "use_gat", False))
    if graph_bypass:
        if method != "reptile_meta_gnn" or not original_use_gat:
            raise A261Error("graph bypass is permitted only for a GNN checkpoint")
        model.use_gat = False
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    values = np.empty(len(dataset), dtype=np.float64)
    seen = np.zeros(len(dataset), dtype=bool)
    forward_calls = 0
    for anchor in ANCHORS:
        label = a232.endpoint_label(anchor)
        indices = [
            index for index, meta in enumerate(dataset.meta) if str(meta["prefix_label"]) == label
        ]
        if len(indices) * len(ANCHORS) != len(dataset):
            raise A261Error(f"selection causal-prefix cardinality mismatch at anchor={anchor:g}")
        loader = a232.deterministic_loader(Subset(dataset, indices), int(cfg["batch_size"]), device)
        for x, locations in loader:
            output = model(x.to(device, non_blocking=device.type == "cuda"))
            forward_calls += 1
            if isinstance(output, tuple):
                output = output[0]
            prediction = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
            target_locations = locations.numpy().reshape(-1).astype(int)
            if len(prediction) != len(target_locations):
                raise A261Error("selection prediction cardinality mismatch")
            values[target_locations] = prediction
            seen[target_locations] = True
    if not seen.all() or not np.isfinite(values).all():
        raise A261Error("selection predictions are incomplete or non-finite")
    metadata = variant_metadata(variant)
    records: list[dict[str, Any]] = []
    for index, (prediction, meta) in enumerate(zip(values, dataset.meta)):
        truth = float(meta["true_rul"])
        error = float(prediction - truth)
        nasa = float(math.exp(error / 10.0) - 1.0) if error >= 0 else float(math.exp(-error / 13.0) - 1.0)
        records.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target_domain,
                "model_seed": model_seed,
                "support_split_seed": support_split_seed,
                "shot": PRIMARY_SHOT,
                "variant": variant,
                **metadata,
                "engine_id": int(meta["engine_id"]),
                "prefix_label": str(meta["prefix_label"]),
                "registered_rul_anchor": float(meta["registered_rul_anchor"]),
                "rul_stage": str(meta["rul_stage"]),
                "true_rul": truth,
                "prediction": float(prediction),
                "error": error,
                "absolute_error": abs(error),
                "squared_error": error * error,
                "positive_error": max(error, 0.0),
                "nasa_score_component": nasa,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "model_uses_gat_before_ablation": original_use_gat,
                "model_uses_gat_during_inference": bool(getattr(model, "use_gat", False)),
                "selection_development_used_for_training": False,
                "selection_development_used_for_evaluation": True,
                "A25_2b_confirmation_used_for_training": False,
                "A25_2b_confirmation_used_for_evaluation": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
                "gradient_enabled": False,
                "backward_calls_during_evaluation": 0,
                "optimizer_steps_during_evaluation": 0,
                "forward_calls_for_variant": forward_calls,
                "endpoint_index_in_dataset": index,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["error"].to_numpy(np.float64)
    absolute = np.abs(errors)
    positive = np.maximum(errors, 0.0)
    return {
        "n_engines": int(len(frame)),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "mae": float(np.mean(absolute)),
        "mean_error": float(np.mean(errors)),
        "nasa_score": float(frame["nasa_score_component"].sum()),
        "positive_error_q90": float(np.quantile(positive, 0.90)),
        "positive_error_q95": float(np.quantile(positive, 0.95)),
        "overprediction_rate": float(np.mean(errors > 0.0)),
        "underprediction_rate": float(np.mean(errors < 0.0)),
    }


def run_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_domain",
        "model_seed",
        "support_split_seed",
        "shot",
        "variant",
        "architecture",
        "base_method",
        "source_algorithm",
        "outer_lr_multiplier",
        "target_epochs",
        "graph_bypass",
        "checkpoint_origin",
        "prefix_label",
        "registered_rul_anchor",
        "rul_stage",
    ]
    rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(keys, sort=True, dropna=False):
        row = {name: value for name, value in zip(keys, key)}
        row.update(metric_values(frame))
        rows.append({"experiment_id": EXPERIMENT_ID, **row})
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_VARIANTS) * len(ANCHORS):
        raise A261Error(
            f"worker run-level rows={len(result)}, expected={len(EXPECTED_VARIANTS) * len(ANCHORS)}"
        )
    return result


def pair_definitions() -> list[tuple[str, str, str, str]]:
    return [
        (
            "historical_algorithm_effect",
            "no_graph",
            "reptile_no_graph_locked_target10",
            "ordinary_no_graph_locked_target10",
        ),
        (
            "historical_algorithm_effect",
            "gnn",
            "reptile_gnn_locked_target10",
            "ordinary_gnn_locked_target10",
        ),
        (
            "outer_lr_half_effect",
            "no_graph",
            "reptile_no_graph_outer_half_target10",
            "reptile_no_graph_locked_target10",
        ),
        (
            "outer_lr_half_effect",
            "gnn",
            "reptile_gnn_outer_half_target10",
            "reptile_gnn_locked_target10",
        ),
        (
            "target_adaptation_0_vs_10",
            "no_graph",
            "reptile_no_graph_outer_half_target0",
            "reptile_no_graph_outer_half_target10",
        ),
        (
            "target_adaptation_0_vs_10",
            "gnn",
            "reptile_gnn_outer_half_target0",
            "reptile_gnn_outer_half_target10",
        ),
        (
            "target_adaptation_1_vs_10",
            "no_graph",
            "reptile_no_graph_outer_half_target1",
            "reptile_no_graph_outer_half_target10",
        ),
        (
            "target_adaptation_1_vs_10",
            "gnn",
            "reptile_gnn_outer_half_target1",
            "reptile_gnn_outer_half_target10",
        ),
        *[
            (
                "graph_bypass_effect",
                "gnn",
                f"reptile_gnn_outer_half_target{epoch}_graph_bypass",
                f"reptile_gnn_outer_half_target{epoch}",
            )
            for epoch in TARGET_SNAPSHOT_EPOCHS
        ],
    ]


def paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity = ["target_domain", "model_seed", "support_split_seed", "engine_id", "prefix_label"]
    for pair_type, architecture, candidate_name, reference_name in pair_definitions():
        candidate = predictions.loc[predictions["variant"] == candidate_name].copy()
        reference = predictions.loc[predictions["variant"] == reference_name].copy()
        if candidate.duplicated(identity).any() or reference.duplicated(identity).any():
            raise A261Error(f"duplicate prediction key in pair {candidate_name}/{reference_name}")
        merged = candidate.merge(reference, on=identity, suffixes=("_candidate", "_reference"), validate="one_to_one")
        if len(merged) != len(candidate) or len(merged) != len(reference):
            raise A261Error(f"unmatched predictions in pair {candidate_name}/{reference_name}")
        if not np.allclose(merged["true_rul_candidate"], merged["true_rul_reference"], atol=0, rtol=0):
            raise A261Error(f"paired truths differ in {candidate_name}/{reference_name}")
        for anchor, frame in merged.groupby("registered_rul_anchor_candidate", sort=True):
            candidate_frame = pd.DataFrame(
                {
                    "error": frame["error_candidate"],
                    "nasa_score_component": frame["nasa_score_component_candidate"],
                }
            )
            reference_frame = pd.DataFrame(
                {
                    "error": frame["error_reference"],
                    "nasa_score_component": frame["nasa_score_component_reference"],
                }
            )
            left = metric_values(candidate_frame)
            right = metric_values(reference_frame)
            row: dict[str, Any] = {
                "experiment_id": EXPERIMENT_ID,
                "pair_type": pair_type,
                "architecture": architecture,
                "candidate_variant": candidate_name,
                "reference_variant": reference_name,
                "target_domain": str(frame["target_domain"].iloc[0]),
                "model_seed": int(frame["model_seed"].iloc[0]),
                "support_split_seed": int(frame["support_split_seed"].iloc[0]),
                "shot": PRIMARY_SHOT,
                "registered_rul_anchor": float(anchor),
                "prefix_label": str(frame["prefix_label"].iloc[0]),
                "n_paired_engines": int(len(frame)),
            }
            for metric in (
                "rmse",
                "mae",
                "mean_error",
                "nasa_score",
                "positive_error_q90",
                "positive_error_q95",
                "overprediction_rate",
            ):
                candidate_value = float(left[metric])
                reference_value = float(right[metric])
                row[f"candidate_{metric}"] = candidate_value
                row[f"reference_{metric}"] = reference_value
                row[f"delta_{metric}"] = candidate_value - reference_value
                row[f"relative_degradation_{metric}"] = (
                    (candidate_value - reference_value) / reference_value
                    if metric != "mean_error" and reference_value > 0
                    else float("nan")
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = sum(EXPECTED_PAIR_TYPES.values()) * len(ANCHORS)
    if len(result) != expected:
        raise A261Error(f"worker paired rows={len(result)}, expected={expected}")
    counts = Counter(result["pair_type"].astype(str))
    for pair_type, pairs in EXPECTED_PAIR_TYPES.items():
        if counts[pair_type] != pairs * len(ANCHORS):
            raise A261Error(f"worker pair count mismatch for {pair_type}")
    return result


def source_stage_coverage(
    tasks: pd.DataFrame,
    raw: Mapping[str, pd.DataFrame],
    target_domain: str,
    model_seed: int,
    split: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sorted(set(DOMAINS) - {target_domain}):
        subset = tasks.loc[
            (tasks["source_domain"].astype(str) == source)
            & (tasks["episode_phase"].astype(str) == "meta_train")
        ]
        engines: set[int] = set()
        for item in subset.itertuples(index=False):
            engines.update(a251b.parse_engine_json(item.meta_support_engine_ids, label="meta_support_engine_ids"))
        frame = raw[source].loc[raw[source]["unit"].isin(engines)].copy()
        rul_column = "RUL" if "RUL" in frame.columns else "rul" if "rul" in frame.columns else None
        if rul_column is None:
            raise A261Error("training frame lacks an RUL/rul column for source coverage")
        rul = pd.to_numeric(frame[rul_column], errors="raise").to_numpy(np.float64)
        stages = {
            "high_rul_gt60": rul > 60.0,
            "mid_rul_31_to_60": (rul > 30.0) & (rul <= 60.0),
            "low_rul_le30": rul <= 30.0,
        }
        for stage, mask in stages.items():
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "target_domain": target_domain,
                    "model_seed": model_seed,
                    "support_split_seed": split,
                    "source_domain": source,
                    "rul_stage": stage,
                    "source_engines": len(engines),
                    "row_observations": int(np.sum(mask)),
                    "row_fraction": float(np.mean(mask)),
                    "episode_records": int(len(subset)),
                    "target_domain_excluded": True,
                    "confirmation_outcomes_used": False,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != (len(DOMAINS) - 1) * 3 or (result["row_observations"] <= 0).any():
        raise A261Error("source stage coverage is incomplete")
    return result


def checkpoint_payload(
    state: Mapping[str, torch.Tensor],
    *,
    variant: str,
    target_domain: str,
    model_seed: int,
    split: int,
    support_engines: Sequence[int],
    accounting: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
    config_hash: str,
    source_state_hash: str,
    target_drift_l2: float,
) -> dict[str, Any]:
    metadata = variant_metadata(variant)
    return {
        "state": {name: value.detach().cpu() for name, value in state.items()},
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "variant": variant,
        **metadata,
        "target_domain": target_domain,
        "model_seed": model_seed,
        "support_split_seed": split,
        "shot": PRIMARY_SHOT,
        "target_support_engine_ids": [int(value) for value in support_engines],
        "selection_development_used_for_training": False,
        "A25_2b_confirmation_used": False,
        "official_test_files_accessed": False,
        "contract_hashes": dict(contract_hashes),
        "config_sha256": config_hash,
        "source_state_sha256": source_state_hash,
        "target_parameter_drift_l2": target_drift_l2,
        "compute_accounting": dict(accounting),
    }


def run_worker(
    args: argparse.Namespace,
    primary_inventory: pd.DataFrame | None = None,
    frozen_hashes: Mapping[str, str] | None = None,
) -> None:
    torch.set_num_threads(int(args.torch_threads))
    root = resolve(args.output_dir)
    directory = worker_root(root, str(args.target_domain), int(args.model_seed), int(args.support_split_seed))
    directory.mkdir(parents=True, exist_ok=True)
    if primary_inventory is None or frozen_hashes is None:
        a26_hashes = validate_a26_0(args)
        primary_inventory, a25_hashes = validate_a25_1b(args)
        frozen_hashes = {**a26_hashes, **a25_hashes}
    protocol, contract_frames, contract_hashes, cfg, config_path, raw = load_contract_and_data(args)
    all_hashes = {**dict(frozen_hashes), **{f"A25_1a::{k}": v for k, v in contract_hashes.items()}}
    all_hashes["config"] = sha256(config_path)
    if args.resume and worker_complete(directory, all_hashes):
        print(f"[A26.1] resume skip {directory.name}; all frozen inputs revalidated", flush=True)
        return
    target = str(args.target_domain)
    model_seed = int(args.model_seed)
    split = int(args.support_split_seed)
    if target not in DOMAINS or model_seed not in MODEL_SEEDS or split not in SUPPORT_SPLIT_SEEDS:
        raise A261Error("worker identity is outside the frozen factorial")
    tasks = a251b.worker_tasks(contract_frames["tasks"], target, model_seed, split, protocol)
    support_by_shot, selection_engines = a251b.selection_and_support(
        contract_frames["roles"], target, split, tuple(int(value) for value in protocol["shots"])
    )
    confirmation_engines = a251b.role_engines(contract_frames["roles"], target, split, "confirmation")
    support_engines = support_by_shot[PRIMARY_SHOT]
    if set(confirmation_engines) & (set(selection_engines) | set(support_engines)):
        raise A261Error("target confirmation role overlaps A26.1 development roles")
    source_engines = a251b.source_fit_engines(tasks, target, raw)
    normalized, normalizer_audit = a251b.source_normalize(raw, target, source_engines)
    if normalizer_audit.get("confirmation_engines_used_for_fit") is not False:
        raise A261Error("source normalizer used confirmation engines")
    selection_dataset = a232.CausalPrefixDataset(
        normalized[target], selection_engines, int(cfg["window_size"])
    )
    support_dataset = a251b.a23.WindowDataset(
        normalized[target], support_engines, int(cfg["window_size"])
    )
    if len(selection_dataset) != len(selection_engines) * len(ANCHORS):
        raise A261Error("selection causal-prefix dataset does not contain three anchors per engine")
    if any(bool(meta["input_uses_future_cycles"]) for meta in selection_dataset.meta):
        raise A261Error("selection causal-prefix dataset uses future cycles")

    device = torch.device("cpu") if args.device == "cpu" else a251b.a23.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contracts = a251b.method_contract(contract_frames["methods"], contract_frames["compute"], protocol)
    runtime_audit = {
        row["method"]: row
        for row in a251b.runtime_model_audit(cfg, contracts, model_seed)
    }
    schedule = a251b.deterministic_schedule(tasks, int(protocol["outer_steps"]), model_seed + split)
    source_budget = int(protocol["source_gradient_updates_per_method_cell"])
    source_windows_budget = int(protocol["source_window_presentations_per_method_cell"])
    prediction_frames: list[pd.DataFrame] = []
    source_history_rows: list[dict[str, Any]] = []
    target_history_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    locked = primary_inventory.loc[
        (primary_inventory["target_domain"].astype(str) == target)
        & (pd.to_numeric(primary_inventory["model_seed"], errors="raise").astype(int) == model_seed)
        & (pd.to_numeric(primary_inventory["support_split_seed"], errors="raise").astype(int) == split)
    ].copy()
    if len(locked) != len(HISTORICAL_METHODS):
        raise A261Error("worker A25.1b locked checkpoint inventory is incomplete")
    for row in locked.to_dict(orient="records"):
        method = str(row["method"])
        variant = historical_variant(method)
        checkpoint = Path(str(row["resolved_checkpoint"])).resolve()
        payload = safe_load_checkpoint(checkpoint)
        if payload.get("experiment_id") != "experimentA25_1b" or payload.get("method") != method:
            raise A261Error(f"historical checkpoint metadata mismatch: {checkpoint}")
        if payload.get("confirmation_engines_used") is not False:
            raise A261Error(f"historical checkpoint used confirmation engines: {checkpoint}")
        if (
            str(payload.get("target_domain")) != target
            or int(payload.get("model_seed", -1)) != model_seed
            or int(payload.get("support_split_seed", -1)) != split
            or int(payload.get("shot", -1)) != PRIMARY_SHOT
        ):
            raise A261Error(f"historical checkpoint worker identity mismatch: {checkpoint}")
        prediction_frames.append(
            predict_state(
                payload["state"],
                method=method,
                variant=variant,
                dataset=selection_dataset,
                cfg=cfg,
                device=device,
                target_domain=target,
                model_seed=model_seed,
                support_split_seed=split,
                checkpoint=checkpoint,
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                graph_bypass=False,
            )
        )

    config_hash = sha256(config_path)
    for architecture, method in REPTILE_METHOD_BY_ARCHITECTURE.items():
        a251b.a23.seed_everything(model_seed)
        initial_model = a251b.make_model(method, cfg, model_seed)
        initial_state = {
            name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()
        }
        if state_value_hash(initial_state) != str(runtime_audit[method]["initial_state_sha256"]):
            raise A261Error(f"initial-state hash mismatch for {method}")
        diagnostic_cfg = deepcopy(cfg)
        diagnostic_cfg["outer_lr"] = float(cfg["outer_lr"]) * OUTER_LR_MULTIPLIER
        accounting = a251b.initial_accounting(
            f"reptile_{architecture}_outer_half",
            architecture,
            source_budget,
            source_windows_budget,
        )
        start = time.perf_counter()
        source_model, source_history = source_diagnostic_train(
            initial_model,
            schedule,
            normalized,
            diagnostic_cfg,
            model_seed + split,
            device,
            accounting,
        )
        accounting["source_wall_time_seconds"] = float(time.perf_counter() - start)
        source_state = {
            name: value.detach().cpu().clone() for name, value in source_model.state_dict().items()
        }
        source_hash = state_value_hash(source_state)
        for item in source_history:
            source_history_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "target_domain": target,
                    "model_seed": model_seed,
                    "support_split_seed": split,
                    "architecture": architecture,
                    "method": method,
                    "outer_lr_multiplier": OUTER_LR_MULTIPLIER,
                    **item,
                }
            )
        adaptation_seed = model_seed * 1_000_000 + split * 100 + PRIMARY_SHOT
        snapshots, snapshot_accountings, target_history = target_adaptation_snapshots(
            source_state,
            method,
            support_dataset,
            diagnostic_cfg,
            model_seed,
            adaptation_seed,
            device,
            accounting,
        )
        for item in target_history:
            target_history_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "target_domain": target,
                    "model_seed": model_seed,
                    "support_split_seed": split,
                    "architecture": architecture,
                    "method": method,
                    "outer_lr_multiplier": OUTER_LR_MULTIPLIER,
                    "shot": PRIMARY_SHOT,
                    **item,
                }
            )
        for epoch in TARGET_SNAPSHOT_EPOCHS:
            variant = snapshot_variant(architecture, epoch)
            state = snapshots[epoch]
            numel, count, schema_hash = a251b.state_schema(state)
            contract = contracts[method]
            if (
                numel != int(contract["reference_state_tensor_numel"])
                or count != int(contract["reference_state_tensor_count"])
                or schema_hash != str(contract["reference_state_schema_sha256"])
            ):
                raise A261Error(f"snapshot state schema mismatch: {variant}")
            target_drift = state_delta_l2(source_state, state)
            checkpoint = directory / f"{variant}.pt"
            atomic_torch_save(
                checkpoint,
                checkpoint_payload(
                    state,
                    variant=variant,
                    target_domain=target,
                    model_seed=model_seed,
                    split=split,
                    support_engines=support_engines,
                    accounting=snapshot_accountings[epoch],
                    contract_hashes=all_hashes,
                    config_hash=config_hash,
                    source_state_hash=source_hash,
                    target_drift_l2=target_drift,
                ),
            )
            checkpoint_hash = sha256(checkpoint)
            loaded = safe_load_checkpoint(checkpoint)
            verifier = a251b.make_model(method, cfg, model_seed).cpu()
            verifier.load_state_dict(loaded["state"], strict=True)
            if state_value_hash(loaded["state"]) != state_value_hash(state):
                raise A261Error(f"checkpoint state roundtrip mismatch: {variant}")
            checkpoint_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "target_domain": target,
                    "model_seed": model_seed,
                    "support_split_seed": split,
                    "shot": PRIMARY_SHOT,
                    "variant": variant,
                    "architecture": architecture,
                    "method": method,
                    "target_epochs": epoch,
                    "outer_lr_multiplier": OUTER_LR_MULTIPLIER,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_reload_passed": True,
                    "state_schema_sha256": schema_hash,
                    "state_tensor_numel": numel,
                    "state_tensor_count": count,
                    "source_state_sha256": source_hash,
                    "target_parameter_drift_l2": target_drift,
                    "source_gradient_updates": int(snapshot_accountings[epoch]["source_gradient_updates"]),
                    "source_window_presentations": int(snapshot_accountings[epoch]["source_window_presentations"]),
                    "source_forward_calls": int(snapshot_accountings[epoch]["source_forward_calls"]),
                    "source_backward_calls": int(snapshot_accountings[epoch]["source_backward_calls"]),
                    "target_gradient_updates": int(snapshot_accountings[epoch]["target_gradient_updates"]),
                    "target_window_presentations": int(snapshot_accountings[epoch]["target_window_presentations"]),
                    "target_forward_calls": int(snapshot_accountings[epoch]["target_forward_calls"]),
                    "target_backward_calls": int(snapshot_accountings[epoch]["target_backward_calls"]),
                    "source_wall_time_seconds": float(snapshot_accountings[epoch]["source_wall_time_seconds"]),
                    "peak_cuda_memory_bytes": int(snapshot_accountings[epoch]["peak_cuda_memory_bytes"]),
                    "selection_development_used_for_training": False,
                    "A25_2b_confirmation_used": False,
                    "official_test_files_accessed": False,
                }
            )
            prediction_frames.append(
                predict_state(
                    loaded["state"],
                    method=method,
                    variant=variant,
                    dataset=selection_dataset,
                    cfg=cfg,
                    device=device,
                    target_domain=target,
                    model_seed=model_seed,
                    support_split_seed=split,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_hash,
                    graph_bypass=False,
                )
            )
            if architecture == "gnn":
                bypass = snapshot_variant(architecture, epoch, graph_bypass=True)
                prediction_frames.append(
                    predict_state(
                        loaded["state"],
                        method=method,
                        variant=bypass,
                        dataset=selection_dataset,
                        cfg=cfg,
                        device=device,
                        target_domain=target,
                        model_seed=model_seed,
                        support_split_seed=split,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_hash,
                        graph_bypass=True,
                    )
                )
            del verifier, loaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del source_model

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if set(predictions["variant"].astype(str)) != set(EXPECTED_VARIANTS):
        raise A261Error("worker variant set is incomplete")
    prediction_keys = ["variant", "engine_id", "prefix_label"]
    if predictions.duplicated(prediction_keys).any():
        raise A261Error("worker predictions contain duplicate variant/engine/anchor keys")
    if len(predictions) != len(EXPECTED_VARIANTS) * len(selection_engines) * len(ANCHORS):
        raise A261Error("worker prediction cardinality mismatch")
    numeric = predictions[
        ["true_rul", "prediction", "error", "absolute_error", "squared_error", "nasa_score_component"]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise A261Error("worker predictions contain non-finite values")
    metrics = run_level_metrics(predictions)
    pairs = paired_comparisons(predictions)
    coverage = source_stage_coverage(tasks, raw, target, model_seed, split)
    checkpoint_inventory = pd.DataFrame(checkpoint_rows)
    if len(checkpoint_inventory) != 6:
        raise A261Error("worker must save six new diagnostic checkpoints")
    source_history_frame = pd.DataFrame(source_history_rows)
    target_history_frame = pd.DataFrame(target_history_rows)
    if len(source_history_frame) != 20 or len(target_history_frame) != 20:
        raise A261Error("worker source/target diagnostic history cardinality mismatch")

    outputs = {
        "prediction_records.csv": predictions,
        "run_level_diagnostics.csv": metrics,
        "paired_diagnostic_comparisons.csv": pairs,
        "source_meta_update_history.csv": source_history_frame,
        "target_adaptation_history.csv": target_history_frame,
        "source_stage_coverage.csv": coverage,
        "checkpoint_inventory.csv": checkpoint_inventory,
    }
    for name, frame in outputs.items():
        atomic_frame(directory / name, frame)
    output_hashes = {name: sha256(directory / name) for name in sorted(outputs)}
    status = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "complete": True,
        "passed": True,
        "development_only": True,
        "target_domain": target,
        "model_seed": model_seed,
        "support_split_seed": split,
        "selection_engine_count": len(selection_engines),
        "support_engine_count": len(support_engines),
        "confirmation_engine_count_metadata_only": len(confirmation_engines),
        "completed_variants": len(EXPECTED_VARIANTS),
        "completed_run_level_records": len(metrics),
        "completed_paired_records": len(pairs),
        "completed_new_checkpoints": len(checkpoint_inventory),
        "source_gradient_updates_per_new_training": source_budget,
        "source_window_presentations_per_new_training": source_windows_budget,
        "new_source_trainings": 2,
        "selection_development_used_for_training": False,
        "selection_development_used_for_evaluation": True,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "A25_2b_confirmation_used_for_training": False,
        "A25_2b_confirmation_used_for_evaluation": False,
        "candidate_selected": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "contract_hashes": all_hashes,
        "output_sha256": output_hashes,
    }
    atomic_json(directory / "worker_status.json", status)
    print(f"[A26.1] completed {directory.name}", flush=True)


def worker_complete(directory: Path, frozen_hashes: Mapping[str, str]) -> bool:
    try:
        status = read_json(directory / "worker_status.json", label="worker status")
        if (
            status.get("experiment_id") != EXPERIMENT_ID
            or status.get("script_version") != SCRIPT_VERSION
            or status.get("complete") is not True
            or status.get("passed") is not True
            or status.get("A25_2b_confirmation_used_for_training") is not False
            or status.get("A25_2b_confirmation_used_for_evaluation") is not False
            or status.get("official_test_files_accessed") is not False
        ):
            return False
        contract_hashes = status.get("contract_hashes")
        if not isinstance(contract_hashes, dict):
            return False
        for key, value in frozen_hashes.items():
            if contract_hashes.get(key) != value:
                return False
        output_hashes = status.get("output_sha256")
        if not isinstance(output_hashes, dict) or len(output_hashes) != 7:
            return False
        for name, expected in output_hashes.items():
            path = directory / str(name)
            if not path.is_file() or sha256(path) != str(expected):
                return False
        inventory = read_frame(directory / "checkpoint_inventory.csv", label="worker checkpoints")
        if len(inventory) != 6:
            return False
        for row in inventory.to_dict(orient="records"):
            path = Path(str(row["checkpoint"])).resolve()
            if path.parent != directory.resolve() or not path.is_file():
                return False
            if sha256(path) != str(row["checkpoint_sha256"]):
                return False
        return True
    except (A261Error, OSError, ValueError, TypeError):
        return False


def worker_command(args: argparse.Namespace, domain: str, seed: int, split: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--data-dir",
        str(args.data_dir),
        "--config",
        str(args.config),
        "--a25-1a-output-dir",
        str(args.a25_1a_output_dir),
        "--a25-1b-output-dir",
        str(args.a25_1b_output_dir),
        "--a25-1b-script",
        str(args.a25_1b_script),
        "--a26-0-output-dir",
        str(args.a26_0_output_dir),
        "--a26-0-script",
        str(args.a26_0_script),
        "--output-dir",
        str(args.output_dir),
        "--target-domain",
        domain,
        "--model-seed",
        str(seed),
        "--support-split-seed",
        str(split),
        "--device",
        "cuda",
        "--torch-threads",
        str(args.torch_threads),
        "--confirm-run",
        RUN_TOKEN,
    ]
    if args.resume:
        command.append("--resume")
    return command


def gpu_inventory() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        try:
            index, free_mb, utilization = [int(part.strip()) for part in line.split(",")]
        except (ValueError, TypeError):
            continue
        rows.append({"index": index, "free_mb": free_mb, "utilization": utilization})
    return rows


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experimentA26_1_run.lock"
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError):
            path.unlink()
        except PermissionError as exc:
            raise A261Error(f"cannot inspect existing run lock: {path}") from exc
        else:
            raise A261Error(f"another A26.1 parent is active with pid={old_pid}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
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


def verify_preflight(
    args: argparse.Namespace,
    primary_inventory: pd.DataFrame,
    protocol: Mapping[str, Any],
    contract_frames: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    raw: Mapping[str, pd.DataFrame],
) -> list[tuple[str, int, int]]:
    workers = expected_workers()
    if len(workers) != 16 or len(set(workers)) != 16:
        raise A261Error("A26.1 worker factorial mismatch")
    if len(EXPECTED_VARIANTS) != 13 or sum(EXPECTED_PAIR_TYPES.values()) != 11:
        raise A261Error("A26.1 registered variant/pair cardinality changed")
    for domain, seed, split in workers:
        tasks = a251b.worker_tasks(contract_frames["tasks"], domain, seed, split, protocol)
        support, selection = a251b.selection_and_support(
            contract_frames["roles"], domain, split, tuple(int(value) for value in protocol["shots"])
        )
        confirmation = a251b.role_engines(contract_frames["roles"], domain, split, "confirmation")
        if set(confirmation) & (set(selection) | set(support[PRIMARY_SHOT])):
            raise A261Error(f"role leakage in worker {domain}/{seed}/{split}")
        source_engines = a251b.source_fit_engines(tasks, domain, raw)
        normalized, audit = a251b.source_normalize(raw, domain, source_engines)
        if any(audit.get(field) is not False for field in (
            "target_domain_used_for_fit",
            "selection_engines_used_for_fit",
            "confirmation_engines_used_for_fit",
        )):
            raise A261Error("normalizer role boundary failed")
        target_dataset = a232.CausalPrefixDataset(
            normalized[domain], selection, int(cfg["window_size"])
        )
        if len(target_dataset) != len(selection) * len(ANCHORS):
            raise A261Error(f"causal selection coverage failed for {domain}/{seed}/{split}")
        locked = primary_inventory.loc[
            (primary_inventory["target_domain"].astype(str) == domain)
            & (pd.to_numeric(primary_inventory["model_seed"], errors="raise").astype(int) == seed)
            & (pd.to_numeric(primary_inventory["support_split_seed"], errors="raise").astype(int) == split)
        ]
        if len(locked) != 4:
            raise A261Error(f"locked checkpoint count failed for {domain}/{seed}/{split}")
        for row in locked.to_dict(orient="records"):
            path = Path(str(row["resolved_checkpoint"])).resolve()
            payload = safe_load_checkpoint(path)
            method = str(row["method"])
            if (
                payload.get("experiment_id") != "experimentA25_1b"
                or payload.get("method") != method
                or str(payload.get("target_domain")) != domain
                or int(payload.get("model_seed", -1)) != seed
                or int(payload.get("support_split_seed", -1)) != split
                or int(payload.get("shot", -1)) != PRIMARY_SHOT
            ):
                raise A261Error(f"locked checkpoint metadata failed preflight: {path}")
            if set(int(value) for value in payload.get("target_support_engine_ids", [])) != set(
                support[PRIMARY_SHOT]
            ):
                raise A261Error(f"locked checkpoint support engines changed: {path}")
            if payload.get("confirmation_engines_used") is not False:
                raise A261Error(f"locked checkpoint used confirmation engines: {path}")
            verifier = a251b.make_model(method, cfg, seed).cpu()
            verifier.load_state_dict(payload["state"], strict=True)
            _, _, schema_hash = a251b.state_schema(payload["state"])
            if schema_hash != str(row["state_schema_sha256"]):
                raise A261Error(f"locked checkpoint schema hash changed: {path}")
            del verifier, payload
        child = parse_args(worker_command(args, domain, seed, split)[2:])
        if not child.worker or child.target_domain != domain or child.confirm_run != RUN_TOKEN:
            raise A261Error("parent/worker command roundtrip failed")
    contracts = a251b.method_contract(contract_frames["methods"], contract_frames["compute"], protocol)
    a251b.runtime_model_audit(cfg, contracts, MODEL_SEEDS[0])
    smoke = a251b.runtime_smoke(cfg, MODEL_SEEDS[0])
    if len(smoke) != 4 or not all(row.get("checkpoint_roundtrip_passed") for row in smoke):
        raise A261Error("runtime model smoke failed")
    model = a251b.make_model("reptile_meta_gnn", cfg, MODEL_SEEDS[0]).cpu()
    if not bool(getattr(model, "use_gat", False)):
        raise A261Error("GNN smoke model does not use GAT")
    model.use_gat = False
    x = torch.randn(4, int(cfg["window_size"]), len(a251b.a23.FEATURE_COLUMNS))
    with torch.inference_mode():
        output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    if output.numel() != 4 or not torch.isfinite(output).all():
        raise A261Error("GNN graph-bypass forward smoke failed")
    return workers


def global_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, anchor), frame in predictions.groupby(["variant", "registered_rul_anchor"], sort=True):
        metadata = variant_metadata(str(variant))
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "analysis_role": "development_descriptive_only",
                "variant": str(variant),
                **metadata,
                "registered_rul_anchor": float(anchor),
                "n_prediction_records": int(len(frame)),
                **metric_values(frame),
                "candidate_selected": False,
                "formal_efficacy_claim_allowed": False,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_VARIANTS) * len(ANCHORS):
        raise A261Error("global diagnostic summary cardinality mismatch")
    return result


def merge(
    args: argparse.Namespace,
    workers: Sequence[tuple[str, int, int]],
    frozen_hashes: Mapping[str, str],
) -> dict[str, Any]:
    root = resolve(args.output_dir)
    collections: dict[str, list[pd.DataFrame]] = {
        "prediction_records.csv": [],
        "run_level_diagnostics.csv": [],
        "paired_diagnostic_comparisons.csv": [],
        "source_meta_update_history.csv": [],
        "target_adaptation_history.csv": [],
        "source_stage_coverage.csv": [],
        "checkpoint_inventory.csv": [],
    }
    for domain, seed, split in workers:
        directory = worker_root(root, domain, seed, split)
        if not worker_complete(directory, frozen_hashes):
            raise A261Error(f"incomplete or corrupt worker: {directory.name}")
        for name in collections:
            collections[name].append(read_frame(directory / name, label=f"{directory.name}/{name}"))
    merged = {name: pd.concat(frames, ignore_index=True) for name, frames in collections.items()}
    expected_counts = {
        "run_level_diagnostics.csv": 16 * 13 * 3,
        "paired_diagnostic_comparisons.csv": 16 * 11 * 3,
        "source_meta_update_history.csv": 16 * 20,
        "target_adaptation_history.csv": 16 * 20,
        "source_stage_coverage.csv": 16 * 9,
        "checkpoint_inventory.csv": 16 * 6,
    }
    for name, expected in expected_counts.items():
        if len(merged[name]) != expected:
            raise A261Error(f"merged {name} rows={len(merged[name])}, expected={expected}")
    predictions = merged["prediction_records.csv"]
    prediction_keys = [
        "target_domain",
        "model_seed",
        "support_split_seed",
        "variant",
        "engine_id",
        "prefix_label",
    ]
    if predictions.duplicated(prediction_keys).any():
        raise A261Error("merged prediction keys are duplicated")
    for column in (
        "selection_development_used_for_training",
        "A25_2b_confirmation_used_for_training",
        "A25_2b_confirmation_used_for_evaluation",
        "official_test_files_accessed",
        "official_test_forward_run",
        "gradient_enabled",
    ):
        if predictions[column].map(lambda value: strict_bool(value, label=column)).any():
            raise A261Error(f"merged prediction boundary violation: {column}")
    if (pd.to_numeric(predictions["backward_calls_during_evaluation"], errors="raise") != 0).any():
        raise A261Error("evaluation executed backward calls")
    if (pd.to_numeric(predictions["optimizer_steps_during_evaluation"], errors="raise") != 0).any():
        raise A261Error("evaluation executed optimizer steps")
    checkpoints = merged["checkpoint_inventory.csv"]
    if not (pd.to_numeric(checkpoints["source_gradient_updates"], errors="raise") == 7500).all():
        raise A261Error("new diagnostic checkpoints violate the source-gradient budget")
    if not (pd.to_numeric(checkpoints["source_window_presentations"], errors="raise") == 480000).all():
        raise A261Error("new diagnostic checkpoints violate the source-window budget")
    if not (pd.to_numeric(checkpoints["source_forward_calls"], errors="raise") == 7500).all():
        raise A261Error("new diagnostic checkpoints violate source forward-call accounting")
    if not (pd.to_numeric(checkpoints["source_backward_calls"], errors="raise") == 7500).all():
        raise A261Error("new diagnostic checkpoints violate source backward-call accounting")
    group_keys = ["target_domain", "model_seed", "support_split_seed", "architecture"]
    for key, frame in checkpoints.groupby(group_keys, sort=True):
        if set(pd.to_numeric(frame["target_epochs"], errors="raise").astype(int)) != set(TARGET_SNAPSHOT_EPOCHS):
            raise A261Error(f"target snapshot epochs are incomplete for {key}")
        if frame["source_state_sha256"].astype(str).nunique() != 1:
            raise A261Error(f"target snapshots do not share one source state for {key}")
        ordered = frame.sort_values("target_epochs")
        updates = pd.to_numeric(ordered["target_gradient_updates"], errors="raise").astype(int).tolist()
        windows = pd.to_numeric(ordered["target_window_presentations"], errors="raise").astype(int).tolist()
        if updates[0] != 0 or windows[0] != 0 or not (updates[0] < updates[1] < updates[2]) or not (windows[0] < windows[1] < windows[2]):
            raise A261Error(f"target adaptation accounting is not monotonic for {key}")
    for row in checkpoints.to_dict(orient="records"):
        path = Path(str(row["checkpoint"])).resolve()
        if not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
            raise A261Error(f"merged diagnostic checkpoint failed hash verification: {path}")
    summary = global_summary(predictions)
    top_outputs = {
        "experimentA26_1_development_predictions.csv": predictions,
        "experimentA26_1_run_level_diagnostics.csv": merged["run_level_diagnostics.csv"],
        "experimentA26_1_paired_diagnostic_comparisons.csv": merged["paired_diagnostic_comparisons.csv"],
        "experimentA26_1_global_diagnostic_summary.csv": summary,
        "experimentA26_1_source_meta_update_history.csv": merged["source_meta_update_history.csv"],
        "experimentA26_1_target_adaptation_history.csv": merged["target_adaptation_history.csv"],
        "experimentA26_1_source_stage_coverage.csv": merged["source_stage_coverage.csv"],
        "experimentA26_1_checkpoint_inventory.csv": checkpoints,
    }
    for name, frame in top_outputs.items():
        atomic_frame(root / name, frame)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "execution_integrity_passed": True,
        "development_only": True,
        "exploratory_only": True,
        "one_factor_at_a_time": True,
        "primary_shot": PRIMARY_SHOT,
        "registered_rul_anchors": list(ANCHORS),
        "outer_lr_multiplier_diagnostic": OUTER_LR_MULTIPLIER,
        "target_adaptation_snapshot_epochs": list(TARGET_SNAPSHOT_EPOCHS),
        "expected_worker_cells": 16,
        "completed_worker_cells": len(workers),
        "expected_variants_per_worker": len(EXPECTED_VARIANTS),
        "completed_run_level_records": len(merged["run_level_diagnostics.csv"]),
        "completed_paired_diagnostic_records": len(merged["paired_diagnostic_comparisons.csv"]),
        "completed_new_checkpoints": len(checkpoints),
        "matched_source_gradient_budget_passed": True,
        "matched_source_window_budget_passed": True,
        "same_architecture_initialization_passed": True,
        "checkpoint_reload_passed": True,
        "selection_development_used_for_training": False,
        "selection_development_used_for_evaluation": True,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "A25_2b_confirmation_used_for_training": False,
        "A25_2b_confirmation_used_for_evaluation": False,
        "A25_2b_confirmation_used_for_candidate_selection": False,
        "candidate_selected": False,
        "policy_selected": False,
        "formal_efficacy_claim": False,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": "A26.1 completed registered one-factor development diagnostics without selecting a candidate",
        "interpretation_limit": (
            "All A26.1 metrics are exploratory development diagnostics on A25 training-file support/selection roles. "
            "They cannot revise A25.2b, support an efficacy/deployment claim, or justify official-test access."
        ),
        "next_action": "analyze_A26_1_diagnostics_then_freeze_or_abandon_a_single_candidate_without_reopening_A25_2b",
    }
    decision_path = root / "experimentA26_1_confirmation_decision.json"
    atomic_json(decision_path, decision)
    artifacts = {name: sha256(root / name) for name in sorted(top_outputs)}
    artifacts[decision_path.name] = sha256(decision_path)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "frozen_input_sha256": dict(sorted(frozen_hashes.items())),
        "artifacts": dict(sorted(artifacts.items())),
        "worker_shards_excluded_from_manifest": True,
        "development_only": True,
        "candidate_selected": False,
        "formal_efficacy_claim": False,
        "new_predictor_training": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA26_1_manifest.json", manifest)
    return decision


def completed_output(args: argparse.Namespace, frozen_hashes: Mapping[str, str]) -> dict[str, Any] | None:
    root = resolve(args.output_dir)
    manifest_path = root / "experimentA26_1_manifest.json"
    decision_path = root / "experimentA26_1_confirmation_decision.json"
    if not manifest_path.is_file() and not decision_path.is_file():
        return None
    if not args.resume:
        raise A261Error(f"A26.1 output already exists; use --resume or a new directory: {root}")
    manifest = read_json(manifest_path, label="existing A26.1 manifest")
    decision = read_json(decision_path, label="existing A26.1 decision")
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("script_version") != SCRIPT_VERSION:
        raise A261Error("existing A26.1 identity/version mismatch")
    if manifest.get("script_sha256") != sha256(Path(__file__).resolve()):
        raise A261Error("existing A26.1 script hash mismatch")
    if manifest.get("frozen_input_sha256") != dict(sorted(frozen_hashes.items())):
        raise A261Error("existing A26.1 frozen input hashes changed")
    validate_hash_map(root, manifest.get("artifacts"), label="existing A26.1 artifacts")
    checkpoints = read_frame(
        root / "experimentA26_1_checkpoint_inventory.csv",
        label="existing A26.1 checkpoint inventory",
    )
    if len(checkpoints) != 96:
        raise A261Error("existing A26.1 checkpoint inventory must contain 96 rows")
    for row in checkpoints.to_dict(orient="records"):
        path = Path(str(row["checkpoint"])).resolve()
        if not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
            raise A261Error(f"existing A26.1 checkpoint failed revalidation: {path}")
    for field in ("complete", "passed", "execution_integrity_passed", "development_only"):
        require_true(decision.get(field), label=f"existing A26.1 {field}")
    for field in (
        "A25_2b_confirmation_used_for_training",
        "A25_2b_confirmation_used_for_evaluation",
        "candidate_selected",
        "formal_efficacy_claim",
        "official_test_files_accessed",
    ):
        require_false(decision.get(field), label=f"existing A26.1 {field}")
    return decision


def parent(args: argparse.Namespace) -> None:
    torch.set_num_threads(int(args.torch_threads))
    a26_hashes = validate_a26_0(args)
    primary_inventory, a25_hashes = validate_a25_1b(args)
    frozen_hashes = {**a26_hashes, **a25_hashes}
    protocol, contract_frames, contract_hashes, cfg, config_path, raw = load_contract_and_data(args)
    manifest_inputs = read_json(
        resolve(args.a25_1b_output_dir) / "experimentA25_1b_manifest.json",
        label="A25.1b manifest",
    ).get("input_sha256")
    if not isinstance(manifest_inputs, dict):
        raise A261Error("A25.1b manifest lacks input_sha256")
    for name, digest in manifest_inputs.items():
        if contract_hashes.get(name) != digest:
            raise A261Error(f"A25.1a contract hash differs from A25.1b manifest: {name}")
    frozen_hashes.update({f"A25_1a::{key}": value for key, value in contract_hashes.items()})
    frozen_hashes["config"] = sha256(config_path)
    existing = completed_output(args, frozen_hashes)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        print("[A26.1] existing complete result and frozen inputs revalidated; no work repeated")
        return
    workers = verify_preflight(args, primary_inventory, protocol, contract_frames, cfg, raw)
    expected_predictions_per_worker = "13_variants_x_3_anchors_x_selection_engine_count"
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": args.dry_run,
        "output_dir": str(resolve(args.output_dir)),
        "primary_shot": PRIMARY_SHOT,
        "registered_rul_anchors": list(ANCHORS),
        "outer_lr_multiplier_diagnostic": OUTER_LR_MULTIPLIER,
        "target_adaptation_snapshot_epochs": list(TARGET_SNAPSHOT_EPOCHS),
        "expected_worker_cells": len(workers),
        "new_source_trainings_per_worker": 2,
        "source_gradient_updates_per_new_training": int(protocol["source_gradient_updates_per_method_cell"]),
        "source_window_presentations_per_new_training": int(protocol["source_window_presentations_per_method_cell"]),
        "expected_new_source_trainings": len(workers) * 2,
        "expected_new_checkpoints": len(workers) * 6,
        "expected_variants_per_worker": len(EXPECTED_VARIANTS),
        "expected_run_level_records": len(workers) * len(EXPECTED_VARIANTS) * len(ANCHORS),
        "expected_paired_diagnostic_records": len(workers) * sum(EXPECTED_PAIR_TYPES.values()) * len(ANCHORS),
        "expected_predictions_per_worker": expected_predictions_per_worker,
        "A25_1b_K5_checkpoints_validated": len(primary_inventory),
        "A26_0_contract_validated": True,
        "all_worker_roles_and_causal_selection_prefixes_preflighted": True,
        "runtime_forward_backward_checkpoint_and_graph_bypass_smoke_passed": True,
        "selection_development_used_for_training": False,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "A25_2b_confirmation_used": False,
        "candidate_selected": False,
        "formal_efficacy_claim": False,
        "new_predictor_training": not args.dry_run,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": True,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if args.dry_run:
        print("[A26.1] dry-run passed; all development-only contracts are compatible and no predictor was trained")
        return

    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        for domain, seed, split in workers:
            local = deepcopy(args)
            local.worker = True
            local.target_domain = domain
            local.model_seed = seed
            local.support_split_seed = split
            run_worker(local, primary_inventory, frozen_hashes)
    else:
        inventory = gpu_inventory()
        eligible = [
            item["index"]
            for item in inventory
            if item["index"] in set(args.gpu_ids)
            and item["free_mb"] >= int(args.min_free_memory_mb)
            and item["utilization"] <= int(args.max_gpu_utilization)
        ]
        if not eligible:
            raise A261Error(f"no eligible GPU; inventory={inventory}")
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
                    worker_command(args, *item),
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[gpu] = (process, handle, item, log)
                print(
                    f"[A26.1] launched target={item[0]} seed={item[1]} split={item[2]} "
                    f"gpu={gpu} pid={process.pid}",
                    flush=True,
                )
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
                        if not output.closed:
                            output.close()
                    for running, _, _, _ in active.values():
                        if running.poll() is None:
                            try:
                                running.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                running.kill()
                                running.wait(timeout=15)
                    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-180:])
                    raise A261Error(f"worker failed item={item} exit={code}\n{tail}")
                print(f"[A26.1] completed target={item[0]} seed={item[1]} split={item[2]} gpu={gpu}", flush=True)
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(3)
    decision = merge(args, workers, frozen_hashes)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    print("[A26.1] completed registered one-factor development diagnostics; no candidate was selected")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
        return 0
    if args.dry_run:
        parent(args)
        return 0
    root = resolve(args.output_dir)
    lock = acquire_lock(root)
    try:
        parent(args)
    finally:
        release_lock(lock)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A261Error as exc:
        print(f"[A26.1] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
