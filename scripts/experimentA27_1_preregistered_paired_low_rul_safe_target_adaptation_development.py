#!/usr/bin/env python3
"""A27.1: paired low-RUL-safe target-adaptation development experiment.

The script consumes the frozen A27.0 preregistration and its sixteen
``reptile_gnn_outer_half_target0`` starting checkpoints.  Within every
target-domain/model-seed/support-split worker it copies the exact same starting
state into two graph-enabled GNN arms and performs the same K=5, ten-epoch
target adaptation with the same batch sequence:

* control: the locked RUL training loss;
* candidate: the locked loss plus
  ``mean(1[y<=30] * relu(prediction-y)^2)`` with lambda fixed at 1.0.

After every worker is complete, both arms are evaluated once on the frozen
selection-development causal prefixes.  The script then executes the A27.0
advancement gates without branching.  It never accepts an A25.2b confirmation
path or an official-test path.  A successful development gate is not a formal
efficacy or deployment claim.
"""

from __future__ import annotations

import argparse
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


EXPERIMENT_ID = "experimentA27_1"
SCRIPT_VERSION = "experimentA27_1_preregistered_paired_low_rul_safe_target_adaptation_development_v1"
RUN_TOKEN = "A27.1_EXPLORATORY_RUN"
A270_ID = "experimentA27_0"
A270_VERSION = "experimentA27_0_low_rul_safe_target_adaptation_preregistration_preflight_v1"
A261_ID = "experimentA26_1"
A261_VERSION = "experimentA26_1_one_factor_failure_diagnostic_development_v1"

DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (140, 141)
SUPPORT_SPLIT_SEEDS = (7501, 7502)
PRIMARY_SHOT = 5
ANCHORS = (90.0, 45.0, 15.0)
EXPECTED_WORKERS = 16

START_VARIANT = "reptile_gnn_outer_half_target0"
ARCHITECTURE = "gnn"
METHOD = "reptile_meta_gnn"
OUTER_LR_MULTIPLIER = 0.5
TARGET_EPOCHS = 10
LOW_RUL_THRESHOLD = 30.0
PENALTY_LAMBDA = 1.0
CONTROL_ARM = "locked_target_loss"
CANDIDATE_ARM = "locked_target_loss_plus_low_rul_overprediction_penalty"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)

GNN_STATE_SCHEMA_SHA256 = "75abbe68a756fd3ccedd28a86a99460262d9515ceefadde3389831faf288a663"
GNN_STATE_TENSOR_NUMEL = 1_591_199
GNN_STATE_TENSOR_COUNT = 48
SOURCE_GRADIENT_UPDATES = 7_500
SOURCE_WINDOW_PRESENTATIONS = 480_000

PRIMARY_METRICS = ("rmse", "nasa_score", "positive_error_q95")
CORE_METRICS = ("rmse", "nasa_score")
HASH_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
TRUE_TEXT = {"true", "1", "yes"}
FALSE_TEXT = {"false", "0", "no"}


class A271Error(RuntimeError):
    """Raised when the frozen A27.1 experiment or an execution invariant fails."""


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
        raise A271Error("state keys differ while computing parameter drift")
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
        raise A271Error(f"refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
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
        raise A271Error(f"required {label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A271Error(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A271Error(f"{label} must contain a JSON object: {path}")
    return value


def read_frame(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise A271Error(f"required {label} is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise A271Error(f"cannot read {label}: {path}: {exc}") from exc
    if frame.empty:
        raise A271Error(f"{label} is empty: {path}")
    return frame


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise A271Error("installed PyTorch lacks safe weights_only checkpoint loading") from exc
    except Exception as exc:
        raise A271Error(f"safe checkpoint load failed: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise A271Error(f"checkpoint has no state dictionary: {path}")
    state = payload["state"]
    if not state or not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()):
        raise A271Error(f"checkpoint state is invalid: {path}")
    for name, tensor in state.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise A271Error(f"checkpoint contains a non-finite tensor: {path}: {name}")
    return payload


def require_fields(payload: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = sorted(set(fields) - set(payload))
    if missing:
        raise A271Error(f"{label} lacks required fields: {missing}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise A271Error(f"{label} lacks required columns: {missing}")


def strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise A271Error(f"{label} must be Boolean, observed {value!r}")


def require_true(value: Any, *, label: str) -> None:
    if not strict_bool(value, label=label):
        raise A271Error(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if strict_bool(value, label=label):
        raise A271Error(f"{label} must be false")


def require_hash(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if HASH_RE.fullmatch(text) is None:
        raise A271Error(f"{label} is not a SHA256 digest: {value!r}")
    return text


def require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise A271Error(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def require_float(observed: Any, expected: float, *, label: str, tolerance: float = 1e-12) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise A271Error(f"{label} must be numeric, observed {observed!r}") from exc
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise A271Error(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise A271Error(f"invalid --gpus value: {raw!r}") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise A271Error("--gpus must contain unique non-negative ids")
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
        "--a25-1b-script",
        type=Path,
        default=Path("scripts/experimentA25_1b_same_architecture_compute_accounted_selection_pilot.py"),
    )
    parser.add_argument(
        "--a26-1-output-dir",
        type=Path,
        default=Path("outputs/experimentA26_1_one_factor_failure_diagnostic_development"),
    )
    parser.add_argument(
        "--a26-1-script",
        type=Path,
        default=Path("scripts/experimentA26_1_one_factor_failure_diagnostic_development.py"),
    )
    parser.add_argument(
        "--a27-0-output-dir",
        type=Path,
        default=Path("outputs/experimentA27_0_low_rul_safe_target_adaptation_preregistration_preflight"),
    )
    parser.add_argument(
        "--a27-0-script",
        type=Path,
        default=Path("scripts/experimentA27_0_low_rul_safe_target_adaptation_preregistration_preflight.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA27_1_preregistered_paired_low_rul_safe_target_adaptation_development"),
    )
    parser.add_argument("--gpus", default="0", help="Physical GPU ids, for example 6,7")
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
        raise A271Error("--max-workers and --torch-threads must be positive")
    if args.min_free_memory_mb < 0 or not 0 <= args.max_gpu_utilization <= 100:
        raise A271Error("invalid GPU eligibility thresholds")
    if not args.dry_run and args.confirm_run != RUN_TOKEN:
        raise A271Error(f"formal exploratory run requires --confirm-run {RUN_TOKEN}")
    if args.worker and (
        args.target_domain is None or args.model_seed is None or args.support_split_seed is None
    ):
        raise A271Error("worker mode requires target-domain, model-seed and support-split-seed")
    return args


def validate_hash_map(root: Path, mapping: Any, *, label: str) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise A271Error(f"{label} must be a non-empty hash mapping")
    verified: dict[str, str] = {}
    for name, expected in sorted(mapping.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise A271Error(f"unsafe artifact name in {label}: {name!r}")
        expected_hash = require_hash(expected, label=f"{label} {name}")
        path = root / name
        if not path.is_file():
            raise A271Error(f"artifact in {label} is missing: {path}")
        observed = sha256(path)
        if observed != expected_hash:
            raise A271Error(
                f"artifact hash mismatch in {label}: {name}: "
                f"expected={expected_hash}, observed={observed}"
            )
        verified[name] = observed
    return verified


def locate_start_checkpoint(root: Path, row: Mapping[str, Any]) -> Path:
    raw = Path(str(row["checkpoint"])).expanduser()
    direct = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    if direct.is_file() and is_relative_to(direct, root):
        return direct
    worker = (
        f"{row['target_domain']}_mseed{int(row['model_seed'])}_"
        f"split{int(row['support_split_seed'])}"
    )
    fallback = (root / "shards" / worker / raw.name).resolve()
    if not is_relative_to(fallback, root) or not fallback.is_file():
        raise A271Error(f"frozen starting checkpoint is missing: {fallback}")
    return fallback


def validate_a27_0(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    root = resolve(args.a27_0_output_dir)
    script = resolve(args.a27_0_script)
    a261_root = resolve(args.a26_1_output_dir)
    a261_script = resolve(args.a26_1_script)
    a251b_script = resolve(args.a25_1b_script)
    config = resolve(args.config)
    for path, label in (
        (root, "A27.0 output directory"),
        (a261_root, "A26.1 output directory"),
    ):
        if not path.is_dir():
            raise A271Error(f"{label} is missing: {path}")
    for path, label in (
        (script, "A27.0 script"),
        (a261_script, "A26.1 script"),
        (a251b_script, "A25.1b script"),
        (config, "configuration"),
    ):
        if not path.is_file():
            raise A271Error(f"{label} is missing: {path}")
    if Path(a251b.__file__).resolve() != a251b_script:
        raise A271Error("imported A25.1b module differs from --a25-1b-script")

    manifest_path = root / "experimentA27_0_manifest.json"
    decision_path = root / "experimentA27_0_confirmation_decision.json"
    manifest = read_json(manifest_path, label="A27.0 manifest")
    decision = read_json(decision_path, label="A27.0 decision")
    require_equal(manifest.get("experiment_id"), A270_ID, label="A27.0 manifest experiment")
    require_equal(manifest.get("script_version"), A270_VERSION, label="A27.0 manifest version")
    require_equal(decision.get("experiment_id"), A270_ID, label="A27.0 decision experiment")
    expected_script_hash = require_hash(manifest.get("script_sha256"), label="A27.0 script hash")
    require_equal(sha256(script), expected_script_hash, label="A27.0 script hash")
    verified = validate_hash_map(root, manifest.get("artifacts"), label="A27.0 artifacts")
    expected_artifacts = {
        "experimentA27_0_candidate_checkpoint_inventory.csv",
        "experimentA27_0_development_evidence_snapshot.csv",
        "experimentA27_0_data_role_contract.csv",
        "experimentA27_0_intervention_contract.json",
        "experimentA27_0_statistical_analysis_plan.json",
        "experimentA27_0_input_integrity.json",
        "experimentA27_0_confirmation_decision.json",
    }
    require_equal(set(verified), expected_artifacts, label="A27.0 artifact set")
    for field in (
        "complete",
        "passed",
        "preflight_only",
        "preregistered",
        "exploratory_development_only",
        "intervention_frozen",
        "statistical_plan_frozen",
        "data_role_contract_frozen",
        "starting_checkpoint_set_frozen",
        "graph_enabled",
        "candidate_proposal_frozen",
    ):
        require_true(decision.get(field), label=f"A27.0 {field}")
    for field in (
        "lambda_sweep_allowed",
        "candidate_selected",
        "policy_selected",
        "A25_2b_confirmation_path_accepted",
        "A25_2b_confirmation_used",
        "new_predictor_training",
        "checkpoint_tensors_opened",
        "model_forward_run",
        "formal_efficacy_claim",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(decision.get(field), label=f"A27.0 {field}")
    expected_decision = {
        "starting_checkpoint_variant": START_VARIANT,
        "architecture": ARCHITECTURE,
        "method": METHOD,
        "control_arm": CONTROL_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "frozen_starting_checkpoints": EXPECTED_WORKERS,
        "expected_A27_1_worker_cells": EXPECTED_WORKERS,
        "expected_A27_1_final_checkpoints": EXPECTED_WORKERS * len(ARMS),
        "target_epochs": TARGET_EPOCHS,
        "target_shot": PRIMARY_SHOT,
    }
    for field, expected in expected_decision.items():
        require_equal(decision.get(field), expected, label=f"A27.0 {field}")
    require_float(decision.get("low_rul_threshold"), LOW_RUL_THRESHOLD, label="A27.0 threshold")
    require_float(decision.get("penalty_lambda"), PENALTY_LAMBDA, label="A27.0 lambda")
    require_equal(
        decision.get("next_action"),
        "implement_A27_1_preregistered_paired_low_rul_safe_target_adaptation_development",
        label="A27.0 next action",
    )

    contract = read_json(root / "experimentA27_0_intervention_contract.json", label="A27.0 intervention")
    plan = read_json(root / "experimentA27_0_statistical_analysis_plan.json", label="A27.0 statistical plan")
    integrity = read_json(root / "experimentA27_0_input_integrity.json", label="A27.0 input integrity")
    frozen_inputs = manifest.get("frozen_input_sha256")
    if not isinstance(frozen_inputs, dict) or not frozen_inputs:
        raise A271Error("A27.0 manifest lacks frozen_input_sha256")
    require_equal(integrity.get("input_sha256"), frozen_inputs, label="A27.0 input integrity map")
    require_false(integrity.get("A25_2b_confirmation_path_accepted"), label="A27.0 A25.2b path")
    require_false(integrity.get("official_test_path_accepted"), label="A27.0 official path")

    require_equal(contract.get("experiment_id"), A270_ID, label="intervention experiment")
    require_equal(contract.get("starting_checkpoint_variant"), START_VARIANT, label="start variant")
    require_equal(contract.get("architecture"), ARCHITECTURE, label="intervention architecture")
    require_equal(contract.get("method"), METHOD, label="intervention method")
    require_true(contract.get("graph_enabled"), label="intervention graph enabled")
    require_false(contract.get("source_retraining_in_A27_1"), label="source retraining")
    require_equal(int(contract.get("target_shot", -1)), PRIMARY_SHOT, label="intervention shot")
    require_equal(int(contract.get("target_epochs", -1)), TARGET_EPOCHS, label="intervention epochs")
    require_float(contract.get("penalty_threshold_true_rul"), LOW_RUL_THRESHOLD, label="loss threshold")
    require_true(contract.get("penalty_uses_labeled_target_support_only"), label="support-only penalty")
    for field in (
        "true_rul_inference_gating",
        "lambda_sweep_allowed",
        "alternative_thresholds_allowed",
        "early_stopping_allowed",
        "intermediate_checkpoint_selection_allowed",
        "A25_2b_confirmation_path_accepted",
        "official_test_path_accepted",
        "formal_efficacy_claim_allowed",
    ):
        require_false(contract.get(field), label=f"intervention {field}")
    for field in (
        "paired_target_batch_sequence_required",
        "matched_target_gradient_updates_required",
        "matched_target_window_presentations_required",
        "matched_target_forward_backward_calls_required",
        "runtime_wall_time_and_peak_cuda_memory_required",
    ):
        require_true(contract.get(field), label=f"intervention {field}")
    require_equal(contract.get("paired_target_loader_seed_formula"), "model_seed*1000000 + support_split_seed*100 + shot", label="loader seed formula")
    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise A271Error("A27.0 intervention must contain exactly two arms")
    arm_map = {str(row.get("arm")): row for row in arms if isinstance(row, dict)}
    require_equal(set(arm_map), set(ARMS), label="intervention arms")
    require_float(arm_map[CONTROL_ARM].get("low_rul_penalty_lambda"), 0.0, label="control lambda")
    require_float(arm_map[CANDIDATE_ARM].get("low_rul_penalty_lambda"), PENALTY_LAMBDA, label="candidate lambda")
    require_equal(plan.get("intervention_contract_canonical_sha256"), canonical_sha256(contract), label="intervention canonical hash")
    require_equal(plan.get("reference_arm"), CONTROL_ARM, label="statistical reference arm")
    require_equal(plan.get("candidate_arm"), CANDIDATE_ARM, label="statistical candidate arm")
    require_true(plan.get("all_advancement_gates_conjunctive"), label="conjunctive gates")
    require_false(plan.get("lambda_retuning_after_A27_1"), label="lambda retuning")
    require_false(plan.get("A25_2b_reuse"), label="A25.2b reuse")
    require_false(plan.get("official_test_access"), label="official test access")
    require_false(plan.get("formal_efficacy_claim_from_A27_1"), label="formal claim")
    require_equal(plan.get("primary_safety_metrics"), list(PRIMARY_METRICS), label="primary safety metrics")
    require_float(plan["low_rul_pooled_gate"].get("maximum_candidate_to_control_relative_change"), -0.10, label="low pooled threshold")
    require_equal(plan["low_rul_worker_gate"].get("minimum_improving_workers_per_metric"), 12, label="low worker threshold")
    require_float(plan["mid_high_guardrail"].get("maximum_pooled_relative_deterioration"), 0.05, label="mid/high threshold")
    require_float(plan["domain_heterogeneity_guardrail"].get("maximum_domain_median_relative_deterioration"), 0.10, label="domain threshold")

    roles = read_frame(root / "experimentA27_0_data_role_contract.csv", label="A27.0 data roles")
    require_columns(roles, ("data_or_artifact", "training_access", "evaluation_access", "prohibited_actions"), label="A27.0 data roles")
    confirmation = roles.loc[roles["data_or_artifact"].astype(str) == "A25.2b sealed-confirmation outcomes"]
    official = roles.loc[roles["data_or_artifact"].astype(str) == "official C-MAPSS test files"]
    if len(confirmation) != 1 or "path_not_accepted" not in str(confirmation.iloc[0]["training_access"]):
        raise A271Error("A27.0 does not block A25.2b confirmation training access")
    if len(official) != 1 or "path_not_accepted" not in str(official.iloc[0]["evaluation_access"]):
        raise A271Error("A27.0 does not block official-test evaluation access")

    require_equal(sha256(config), frozen_inputs.get("config"), label="frozen configuration hash")
    require_equal(sha256(a261_script), frozen_inputs.get("A26_1::script"), label="frozen A26.1 script hash")
    a261_manifest_path = a261_root / "experimentA26_1_manifest.json"
    require_equal(sha256(a261_manifest_path), frozen_inputs.get("A26_1::manifest"), label="frozen A26.1 manifest hash")
    a261_manifest = read_json(a261_manifest_path, label="A26.1 manifest")
    require_equal(a261_manifest.get("experiment_id"), A261_ID, label="A26.1 manifest experiment")
    require_equal(a261_manifest.get("script_version"), A261_VERSION, label="A26.1 manifest version")
    a261_inputs = a261_manifest.get("frozen_input_sha256")
    if not isinstance(a261_inputs, dict) or not a261_inputs:
        raise A271Error("A26.1 manifest lacks frozen_input_sha256")
    require_equal(sha256(a251b_script), a261_inputs.get("A25_1b::A25_1b_script"), label="frozen A25.1b script hash")

    inventory = read_frame(root / "experimentA27_0_candidate_checkpoint_inventory.csv", label="A27.0 candidate checkpoint inventory")
    require_columns(
        inventory,
        (
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "source_variant",
            "architecture",
            "method",
            "graph_enabled",
            "source_outer_lr_multiplier",
            "source_gradient_updates",
            "source_window_presentations",
            "target_updates_before_A27_1",
            "target_parameter_drift_l2_before_A27_1",
            "checkpoint",
            "checkpoint_sha256",
            "source_state_sha256",
            "state_schema_sha256",
            "state_tensor_numel",
            "state_tensor_count",
            "checkpoint_tensor_opened_in_A27_0",
            "assigned_control_arm",
            "assigned_candidate_arm",
        ),
        label="A27.0 candidate checkpoint inventory",
    )
    require_equal(len(inventory), EXPECTED_WORKERS, label="frozen starting checkpoint rows")
    expected_cells = {
        (domain, seed, split)
        for domain in DOMAINS
        for seed in MODEL_SEEDS
        for split in SUPPORT_SPLIT_SEEDS
    }
    observed_cells: set[tuple[str, int, int]] = set()
    for index, row in inventory.iterrows():
        label = f"A27.0 checkpoint row {index}"
        cell = (str(row["target_domain"]), int(row["model_seed"]), int(row["support_split_seed"]))
        if cell in observed_cells:
            raise A271Error(f"duplicate frozen checkpoint cell: {cell}")
        observed_cells.add(cell)
        require_equal(int(row["shot"]), PRIMARY_SHOT, label=f"{label} shot")
        require_equal(str(row["source_variant"]), START_VARIANT, label=f"{label} variant")
        require_equal(str(row["architecture"]), ARCHITECTURE, label=f"{label} architecture")
        require_equal(str(row["method"]), METHOD, label=f"{label} method")
        require_true(row["graph_enabled"], label=f"{label} graph")
        require_float(row["source_outer_lr_multiplier"], OUTER_LR_MULTIPLIER, label=f"{label} outer LR")
        require_equal(int(row["source_gradient_updates"]), SOURCE_GRADIENT_UPDATES, label=f"{label} source updates")
        require_equal(int(row["source_window_presentations"]), SOURCE_WINDOW_PRESENTATIONS, label=f"{label} source windows")
        require_equal(int(row["target_updates_before_A27_1"]), 0, label=f"{label} prior target updates")
        require_float(row["target_parameter_drift_l2_before_A27_1"], 0.0, label=f"{label} prior drift")
        require_equal(str(row["state_schema_sha256"]), GNN_STATE_SCHEMA_SHA256, label=f"{label} schema")
        require_equal(int(row["state_tensor_numel"]), GNN_STATE_TENSOR_NUMEL, label=f"{label} numel")
        require_equal(int(row["state_tensor_count"]), GNN_STATE_TENSOR_COUNT, label=f"{label} tensors")
        require_false(row["checkpoint_tensor_opened_in_A27_0"], label=f"{label} opened in A27.0")
        require_equal(str(row["assigned_control_arm"]), CONTROL_ARM, label=f"{label} control")
        require_equal(str(row["assigned_candidate_arm"]), CANDIDATE_ARM, label=f"{label} candidate")
        checkpoint = locate_start_checkpoint(a261_root, row)
        expected_checkpoint_hash = require_hash(row["checkpoint_sha256"], label=f"{label} hash")
        require_equal(sha256(checkpoint), expected_checkpoint_hash, label=f"{label} checkpoint hash")
        frozen_key = f"A26_1_checkpoint::{cell[0]}::mseed{cell[1]}::split{cell[2]}"
        require_equal(frozen_inputs.get(frozen_key), expected_checkpoint_hash, label=f"{label} frozen hash")
        inventory.at[index, "resolved_checkpoint"] = str(checkpoint)
    require_equal(observed_cells, expected_cells, label="frozen starting checkpoint factorial")

    frozen_hashes = {
        "A27_0::script": expected_script_hash,
        "A27_0::manifest": sha256(manifest_path),
        **{f"A27_0::{name}": digest for name, digest in verified.items()},
        **{str(key): str(value) for key, value in frozen_inputs.items()},
    }
    return inventory.reset_index(drop=True), contract, plan, a261_manifest, dict(sorted(frozen_hashes.items()))


def load_contract_and_data(
    args: argparse.Namespace,
    a261_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, str], dict[str, Any], Path, dict[str, pd.DataFrame]]:
    try:
        protocol, frames, hashes = a251b.load_contract(args)
        cfg, config_path = a251b.load_config(protocol, args.config)
        raw = a251b.load_frames(args, protocol, cfg)
    except Exception as exc:
        raise A271Error(f"A25.1a contract/data validation failed: {exc}") from exc
    require_equal(tuple(int(value) for value in protocol["model_seeds"]), MODEL_SEEDS, label="model seeds")
    require_equal(tuple(int(value) for value in protocol["support_split_seeds"]), SUPPORT_SPLIT_SEEDS, label="split seeds")
    if PRIMARY_SHOT not in tuple(int(value) for value in protocol["shots"]):
        raise A271Error("A25.1a does not contain K=5")
    require_equal(int(protocol["target_epochs"]), TARGET_EPOCHS, label="target epochs")
    require_equal(int(protocol["source_gradient_updates_per_method_cell"]), SOURCE_GRADIENT_UPDATES, label="source updates")
    require_equal(int(protocol["source_window_presentations_per_method_cell"]), SOURCE_WINDOW_PRESENTATIONS, label="source windows")
    frozen = a261_manifest.get("frozen_input_sha256")
    if not isinstance(frozen, dict):
        raise A271Error("A26.1 manifest frozen inputs are invalid")
    for name, digest in hashes.items():
        require_equal(frozen.get(f"A25_1a::{name}"), digest, label=f"A25.1a frozen hash {name}")
    require_equal(frozen.get("config"), sha256(config_path), label="A26.1 frozen config hash")
    contracts = a251b.method_contract(frames["methods"], frames["compute"], protocol)
    contract = contracts[METHOD]
    require_equal(str(contract["architecture"]), ARCHITECTURE, label="method architecture")
    require_equal(int(contract["reference_state_tensor_numel"]), GNN_STATE_TENSOR_NUMEL, label="method state numel")
    require_equal(int(contract["reference_state_tensor_count"]), GNN_STATE_TENSOR_COUNT, label="method state tensors")
    require_equal(str(contract["reference_state_schema_sha256"]), GNN_STATE_SCHEMA_SHA256, label="method schema")
    return protocol, frames, hashes, cfg, config_path, raw


def expected_workers() -> list[tuple[str, int, int]]:
    return [
        (domain, seed, split)
        for domain in DOMAINS
        for seed in MODEL_SEEDS
        for split in SUPPORT_SPLIT_SEEDS
    ]


def worker_root(root: Path, domain: str, seed: int, split: int) -> Path:
    return root / "shards" / f"{domain}_mseed{seed}_split{split}"


def starting_row(inventory: pd.DataFrame, domain: str, seed: int, split: int) -> dict[str, Any]:
    selected = inventory.loc[
        (inventory["target_domain"].astype(str) == domain)
        & (pd.to_numeric(inventory["model_seed"], errors="raise").astype(int) == seed)
        & (pd.to_numeric(inventory["support_split_seed"], errors="raise").astype(int) == split)
    ]
    if len(selected) != 1:
        raise A271Error(f"frozen starting checkpoint count is not one: {domain}/{seed}/{split}")
    return dict(selected.iloc[0])


def validate_start_payload(
    row: Mapping[str, Any],
    *,
    target: str,
    model_seed: int,
    split: int,
    support_engines: Sequence[int],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = Path(str(row["resolved_checkpoint"])).resolve()
    require_equal(sha256(checkpoint), str(row["checkpoint_sha256"]), label="starting checkpoint hash")
    payload = safe_load_checkpoint(checkpoint)
    require_fields(
        payload,
        (
            "experiment_id",
            "script_version",
            "variant",
            "architecture",
            "base_method",
            "source_algorithm",
            "outer_lr_multiplier",
            "target_epochs",
            "graph_bypass",
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "target_support_engine_ids",
            "selection_development_used_for_training",
            "A25_2b_confirmation_used",
            "official_test_files_accessed",
            "config_sha256",
            "source_state_sha256",
            "target_parameter_drift_l2",
            "compute_accounting",
        ),
        label="A26.1 starting checkpoint",
    )
    expected = {
        "experiment_id": A261_ID,
        "script_version": A261_VERSION,
        "variant": START_VARIANT,
        "architecture": ARCHITECTURE,
        "base_method": METHOD,
        "source_algorithm": "reptile_outer_lr_half",
        "target_domain": target,
        "model_seed": model_seed,
        "support_split_seed": split,
        "shot": PRIMARY_SHOT,
        "target_epochs": 0,
        "graph_bypass": False,
    }
    for field, value in expected.items():
        require_equal(payload.get(field), value, label=f"starting checkpoint {field}")
    require_float(payload.get("outer_lr_multiplier"), OUTER_LR_MULTIPLIER, label="starting outer LR")
    require_equal(sorted(int(value) for value in payload["target_support_engine_ids"]), sorted(int(value) for value in support_engines), label="starting support engines")
    require_false(payload["selection_development_used_for_training"], label="starting selection training")
    require_false(payload["A25_2b_confirmation_used"], label="starting A25.2b use")
    require_false(payload["official_test_files_accessed"], label="starting official test")
    require_equal(payload["config_sha256"], sha256(resolve(Path(str(cfg["__config_path"])))), label="starting config hash")
    require_float(payload["target_parameter_drift_l2"], 0.0, label="starting target drift")
    state = {name: tensor.detach().cpu().clone() for name, tensor in payload["state"].items()}
    state_hash = state_value_hash(state)
    require_equal(state_hash, payload["source_state_sha256"], label="starting source state hash")
    require_equal(state_hash, str(row["source_state_sha256"]), label="inventory source state hash")
    numel, count, schema = a251b.state_schema(state)
    require_equal(numel, GNN_STATE_TENSOR_NUMEL, label="starting state numel")
    require_equal(count, GNN_STATE_TENSOR_COUNT, label="starting state tensor count")
    require_equal(schema, GNN_STATE_SCHEMA_SHA256, label="starting state schema")
    accounting = payload["compute_accounting"]
    if not isinstance(accounting, dict):
        raise A271Error("starting checkpoint compute_accounting is invalid")
    for field, value in (
        ("source_gradient_updates", SOURCE_GRADIENT_UPDATES),
        ("source_window_presentations", SOURCE_WINDOW_PRESENTATIONS),
        ("source_forward_calls", SOURCE_GRADIENT_UPDATES),
        ("source_backward_calls", SOURCE_GRADIENT_UPDATES),
        ("target_gradient_updates", 0),
        ("target_window_presentations", 0),
        ("target_forward_calls", 0),
        ("target_backward_calls", 0),
    ):
        require_equal(int(accounting.get(field, -1)), value, label=f"starting accounting {field}")
    verifier = a251b.make_model(METHOD, cfg, model_seed).cpu()
    verifier.load_state_dict(state, strict=True)
    require_true(bool(getattr(verifier, "use_gat", False)), label="starting model GAT")
    del verifier
    return payload, state


def hash_batch(digest: Any, epoch: int, batch_index: int, batch: Sequence[Any]) -> None:
    digest.update(f"epoch={epoch};batch={batch_index};items={len(batch)}".encode("utf-8"))
    for index, item in enumerate(batch):
        digest.update(f"item={index}".encode("utf-8"))
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        else:
            digest.update(repr(item).encode("utf-8"))


def target_step(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    batch: Sequence[Any],
    device: torch.device,
    pair_weight: float,
    penalty_lambda: float,
    accounting: dict[str, Any],
) -> tuple[float, float, float, int]:
    x, y, _, _ = batch
    x = x.to(device, non_blocking=device.type == "cuda")
    y = y.to(device, non_blocking=device.type == "cuda")
    optimiser.zero_grad(set_to_none=True)
    base_loss, prediction = a251b.rul_training_loss(model, x, y, pair_weight)
    if prediction.shape != y.shape:
        raise A271Error(f"prediction/target shape mismatch: {prediction.shape} != {y.shape}")
    low_mask = (y <= LOW_RUL_THRESHOLD).to(dtype=prediction.dtype)
    penalty = torch.mean(low_mask * torch.relu(prediction - y).pow(2))
    total_loss = base_loss + float(penalty_lambda) * penalty
    if not torch.isfinite(base_loss) or not torch.isfinite(penalty) or not torch.isfinite(total_loss):
        raise A271Error("non-finite target adaptation loss")
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimiser.step()
    accounting["target_gradient_updates"] += 1
    accounting["target_forward_calls"] += 1
    accounting["target_backward_calls"] += 1
    accounting["target_window_presentations"] += int(y.numel())
    accounting["low_rul_support_window_presentations"] += int(low_mask.sum().detach().cpu())
    if device.type == "cuda":
        accounting["peak_cuda_memory_bytes"] = max(
            int(accounting["peak_cuda_memory_bytes"]),
            int(torch.cuda.max_memory_allocated(device)),
        )
    return (
        float(base_loss.detach().cpu()),
        float(penalty.detach().cpu()),
        float(total_loss.detach().cpu()),
        int(low_mask.sum().detach().cpu()),
    )


def adapt_arm(
    start_state: Mapping[str, torch.Tensor],
    *,
    arm: str,
    dataset: Any,
    cfg: Mapping[str, Any],
    model_seed: int,
    loader_seed: int,
    device: torch.device,
    target: str,
    split: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]], str]:
    if arm not in ARMS:
        raise A271Error(f"unknown arm: {arm}")
    penalty_lambda = 0.0 if arm == CONTROL_ARM else PENALTY_LAMBDA
    a251b.a23.seed_everything(loader_seed)
    model = a251b.make_model(METHOD, cfg, model_seed).cpu()
    model.load_state_dict(start_state, strict=True)
    require_true(bool(getattr(model, "use_gat", False)), label=f"{arm} GAT enabled")
    initial_hash = state_value_hash(model.state_dict())
    require_equal(initial_hash, state_value_hash(start_state), label=f"{arm} initial state")
    loader = a251b.a23.make_loader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=loader_seed,
    )
    learner = model.to(device)
    optimiser = torch.optim.Adam(
        learner.parameters(),
        lr=float(cfg["inner_lr"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    accounting: dict[str, Any] = {
        "source_gradient_updates_in_A27_1": 0,
        "source_window_presentations_in_A27_1": 0,
        "target_gradient_updates": 0,
        "target_window_presentations": 0,
        "target_forward_calls": 0,
        "target_backward_calls": 0,
        "low_rul_support_window_presentations": 0,
        "selection_forward_calls": 0,
        "selection_window_presentations": 0,
        "peak_cuda_memory_bytes": 0,
    }
    batch_digest = hashlib.sha256()
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, TARGET_EPOCHS + 1):
        base_losses: list[float] = []
        penalties: list[float] = []
        total_losses: list[float] = []
        low_windows = 0
        batches = 0
        for batch_index, batch in enumerate(loader, start=1):
            hash_batch(batch_digest, epoch, batch_index, batch)
            base, penalty, total, low = target_step(
                learner,
                optimiser,
                batch,
                device,
                float(cfg["pair_aux_weight"]),
                penalty_lambda,
                accounting,
            )
            base_losses.append(base)
            penalties.append(penalty)
            total_losses.append(total)
            low_windows += low
            batches += 1
        if batches < 1:
            raise A271Error(f"{arm} target loader produced zero batches")
        history.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split,
                "shot": PRIMARY_SHOT,
                "arm": arm,
                "epoch": epoch,
                "batches": batches,
                "mean_locked_base_loss": float(np.mean(base_losses)),
                "mean_low_rul_overprediction_penalty": float(np.mean(penalties)),
                "mean_total_loss": float(np.mean(total_losses)),
                "low_rul_window_presentations": low_windows,
                "cumulative_target_gradient_updates": int(accounting["target_gradient_updates"]),
                "cumulative_target_window_presentations": int(accounting["target_window_presentations"]),
                "graph_enabled": True,
                "selection_development_used_for_training": False,
                "A25_2b_confirmation_used": False,
                "official_test_files_accessed": False,
            }
        )
        print(
            f"[A27.1] {target} seed={model_seed} split={split} arm={arm} "
            f"epoch={epoch:02d}/{TARGET_EPOCHS} loss={np.mean(total_losses):.6f}",
            flush=True,
        )
    accounting["target_wall_time_seconds"] = float(time.perf_counter() - start)
    if int(accounting["low_rul_support_window_presentations"]) < 1:
        raise A271Error(f"{arm} never received a labeled RUL<=30 support window")
    state = {name: tensor.detach().cpu().clone() for name, tensor in learner.state_dict().items()}
    del learner, model, optimiser
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, accounting, history, batch_digest.hexdigest()


@torch.inference_mode()
def predict_arm(
    state: Mapping[str, torch.Tensor],
    *,
    arm: str,
    dataset: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    target: str,
    model_seed: int,
    split: int,
    checkpoint: Path,
    checkpoint_hash: str,
) -> tuple[pd.DataFrame, int]:
    model = a251b.make_model(METHOD, cfg, model_seed).cpu()
    model.load_state_dict(state, strict=True)
    require_true(bool(getattr(model, "use_gat", False)), label=f"{arm} evaluation GAT")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    values = np.empty(len(dataset), dtype=np.float64)
    seen = np.zeros(len(dataset), dtype=bool)
    forward_calls = 0
    for anchor in ANCHORS:
        label = a232.endpoint_label(anchor)
        indices = [index for index, meta in enumerate(dataset.meta) if str(meta["prefix_label"]) == label]
        if len(indices) * len(ANCHORS) != len(dataset):
            raise A271Error(f"selection causal-prefix cardinality mismatch at anchor={anchor:g}")
        loader = a232.deterministic_loader(Subset(dataset, indices), int(cfg["batch_size"]), device)
        for x, locations in loader:
            output = model(x.to(device, non_blocking=device.type == "cuda"))
            forward_calls += 1
            if isinstance(output, tuple):
                output = output[0]
            predictions = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
            positions = locations.numpy().reshape(-1).astype(int)
            if len(predictions) != len(positions):
                raise A271Error("selection prediction cardinality mismatch")
            values[positions] = predictions
            seen[positions] = True
    if not seen.all() or not np.isfinite(values).all():
        raise A271Error("selection predictions are incomplete or non-finite")
    rows: list[dict[str, Any]] = []
    for index, (prediction, meta) in enumerate(zip(values, dataset.meta)):
        if bool(meta["input_uses_future_cycles"]):
            raise A271Error("selection input uses future cycles")
        truth = float(meta["true_rul"])
        error = float(prediction - truth)
        nasa = float(math.exp(error / 10.0) - 1.0) if error >= 0 else float(math.exp(-error / 13.0) - 1.0)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split,
                "shot": PRIMARY_SHOT,
                "arm": arm,
                "architecture": ARCHITECTURE,
                "method": METHOD,
                "graph_enabled": True,
                "target_epochs": TARGET_EPOCHS,
                "low_rul_penalty_lambda": 0.0 if arm == CONTROL_ARM else PENALTY_LAMBDA,
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
                "checkpoint_sha256": checkpoint_hash,
                "input_uses_future_cycles": False,
                "selection_development_used_for_training": False,
                "selection_development_used_for_evaluation": True,
                "A25_2b_confirmation_used_for_training": False,
                "A25_2b_confirmation_used_for_evaluation": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
                "gradient_enabled": False,
                "backward_calls_during_evaluation": 0,
                "optimizer_steps_during_evaluation": 0,
                "endpoint_index_in_dataset": index,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), forward_calls


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["error"].to_numpy(np.float64)
    positive = np.maximum(errors, 0.0)
    values = {
        "n_engines": int(len(frame)),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
        "nasa_score": float(frame["nasa_score_component"].sum()),
        "positive_error_q90": float(np.quantile(positive, 0.90)),
        "positive_error_q95": float(np.quantile(positive, 0.95)),
        "overprediction_rate": float(np.mean(errors > 0.0)),
        "underprediction_rate": float(np.mean(errors < 0.0)),
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise A271Error("metric computation produced a non-finite value")
    return values


def run_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_domain",
        "model_seed",
        "support_split_seed",
        "shot",
        "arm",
        "architecture",
        "method",
        "graph_enabled",
        "target_epochs",
        "low_rul_penalty_lambda",
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
    if len(result) != len(ARMS) * len(ANCHORS):
        raise A271Error(f"worker run-level rows={len(result)}, expected={len(ARMS) * len(ANCHORS)}")
    return result


def relative_change(candidate: float, reference: float) -> float | None:
    candidate = float(candidate)
    reference = float(reference)
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return None
    if abs(reference) <= 1e-12:
        return 0.0 if abs(candidate) <= 1e-12 else None
    return float((candidate - reference) / reference)


def paired_worker_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        subset = metrics.loc[np.isclose(pd.to_numeric(metrics["registered_rul_anchor"]), anchor)]
        control = subset.loc[subset["arm"].astype(str) == CONTROL_ARM]
        candidate = subset.loc[subset["arm"].astype(str) == CANDIDATE_ARM]
        if len(control) != 1 or len(candidate) != 1:
            raise A271Error(f"worker arm metrics are incomplete at anchor={anchor:g}")
        left = dict(control.iloc[0])
        right = dict(candidate.iloc[0])
        row: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "target_domain": str(left["target_domain"]),
            "model_seed": int(left["model_seed"]),
            "support_split_seed": int(left["support_split_seed"]),
            "shot": PRIMARY_SHOT,
            "registered_rul_anchor": anchor,
            "candidate_arm": CANDIDATE_ARM,
            "reference_arm": CONTROL_ARM,
            "n_paired_engines": int(left["n_engines"]),
        }
        if int(left["n_engines"]) != int(right["n_engines"]):
            raise A271Error("paired worker engine counts differ")
        for metric in (
            "rmse",
            "mae",
            "mean_error",
            "nasa_score",
            "positive_error_q90",
            "positive_error_q95",
            "overprediction_rate",
        ):
            reference = float(left[metric])
            value = float(right[metric])
            row[f"reference_{metric}"] = reference
            row[f"candidate_{metric}"] = value
            row[f"delta_{metric}"] = value - reference
            row[f"relative_change_{metric}"] = relative_change(value, reference)
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_payload(
    state: Mapping[str, torch.Tensor],
    *,
    arm: str,
    target: str,
    model_seed: int,
    split: int,
    support_engines: Sequence[int],
    selection_engines: Sequence[int],
    start_row: Mapping[str, Any],
    accounting: Mapping[str, Any],
    batch_sequence_hash: str,
    history: Sequence[Mapping[str, Any]],
    frozen_hashes: Mapping[str, str],
    config_hash: str,
) -> dict[str, Any]:
    return {
        "state": {name: tensor.detach().cpu() for name, tensor in state.items()},
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "arm": arm,
        "architecture": ARCHITECTURE,
        "method": METHOD,
        "graph_enabled": True,
        "target_domain": target,
        "model_seed": model_seed,
        "support_split_seed": split,
        "shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "low_rul_penalty_lambda": 0.0 if arm == CONTROL_ARM else PENALTY_LAMBDA,
        "starting_checkpoint": str(start_row["resolved_checkpoint"]),
        "starting_checkpoint_sha256": str(start_row["checkpoint_sha256"]),
        "starting_state_sha256": str(start_row["source_state_sha256"]),
        "target_support_engine_ids": [int(value) for value in support_engines],
        "target_selection_engine_ids": [int(value) for value in selection_engines],
        "batch_sequence_sha256": batch_sequence_hash,
        "compute_accounting": dict(accounting),
        "target_history": [dict(row) for row in history],
        "contract_hashes": dict(frozen_hashes),
        "config_sha256": config_hash,
        "source_retraining_in_A27_1": False,
        "selection_development_used_for_training": False,
        "selection_development_used_for_evaluation": True,
        "A25_2b_confirmation_used_for_training": False,
        "A25_2b_confirmation_used_for_evaluation": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def worker_output_names() -> tuple[str, ...]:
    return (
        "prediction_records.csv",
        "run_level_metrics.csv",
        "paired_worker_metrics.csv",
        "target_adaptation_history.csv",
        "checkpoint_inventory.csv",
        "matched_target_compute_audit.csv",
    )


def worker_complete(directory: Path, frozen_hashes: Mapping[str, str]) -> bool:
    try:
        status = read_json(directory / "worker_status.json", label="worker status")
        if (
            status.get("experiment_id") != EXPERIMENT_ID
            or status.get("script_version") != SCRIPT_VERSION
            or status.get("complete") is not True
            or status.get("passed") is not True
            or status.get("contract_hashes") != dict(sorted(frozen_hashes.items()))
            or status.get("selection_development_used_for_training") is not False
            or status.get("A25_2b_confirmation_used") is not False
            or status.get("official_test_files_accessed") is not False
        ):
            return False
        hashes = status.get("output_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(worker_output_names()):
            return False
        for name, expected in hashes.items():
            path = directory / str(name)
            if not path.is_file() or sha256(path) != str(expected):
                return False
        inventory = read_frame(directory / "checkpoint_inventory.csv", label="worker checkpoints")
        if len(inventory) != 2:
            return False
        for row in inventory.to_dict(orient="records"):
            path = Path(str(row["checkpoint"])).resolve()
            if path.parent != directory.resolve() or not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
                return False
        audit = read_frame(directory / "matched_target_compute_audit.csv", label="worker compute audit")
        if len(audit) != 1:
            return False
        for field in (
            "starting_state_identical",
            "batch_sequence_identical",
            "target_gradient_updates_identical",
            "target_window_presentations_identical",
            "target_forward_calls_identical",
            "target_backward_calls_identical",
            "graph_enabled_both_arms",
        ):
            if not strict_bool(audit.iloc[0][field], label=field):
                return False
        return True
    except (A271Error, OSError, ValueError, TypeError, KeyError):
        return False


def run_worker(
    args: argparse.Namespace,
    *,
    context: tuple[
        pd.DataFrame,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, str],
        dict[str, Any],
        dict[str, pd.DataFrame],
        dict[str, Any],
        Path,
        dict[str, pd.DataFrame],
    ] | None = None,
) -> None:
    torch.set_num_threads(int(args.torch_threads))
    if context is None:
        inventory, intervention, plan, a261_manifest, frozen_hashes = validate_a27_0(args)
        protocol, frames, contract_hashes, cfg, config_path, raw = load_contract_and_data(args, a261_manifest)
        frozen_hashes = {
            **frozen_hashes,
            **{f"A25_1a::{name}": digest for name, digest in contract_hashes.items()},
            "config": sha256(config_path),
        }
    else:
        (
            inventory,
            intervention,
            plan,
            a261_manifest,
            frozen_hashes,
            protocol,
            frames,
            cfg,
            config_path,
            raw,
        ) = context
    del intervention, plan, a261_manifest
    target = str(args.target_domain)
    model_seed = int(args.model_seed)
    split = int(args.support_split_seed)
    if (target, model_seed, split) not in set(expected_workers()):
        raise A271Error("worker identity is outside the frozen factorial")
    root = resolve(args.output_dir)
    directory = worker_root(root, target, model_seed, split)
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume and worker_complete(directory, frozen_hashes):
        print(f"[A27.1] resume skip {directory.name}; outputs and frozen inputs verified", flush=True)
        return

    tasks = a251b.worker_tasks(frames["tasks"], target, model_seed, split, protocol)
    support_by_shot, selection_engines = a251b.selection_and_support(
        frames["roles"], target, split, tuple(int(value) for value in protocol["shots"])
    )
    support_engines = support_by_shot[PRIMARY_SHOT]
    confirmation_engines = a251b.role_engines(frames["roles"], target, split, "confirmation")
    if set(confirmation_engines) & (set(support_engines) | set(selection_engines)):
        raise A271Error("confirmation engines overlap support or selection")
    source_engines = a251b.source_fit_engines(tasks, target, raw)
    normalized, normalizer_audit = a251b.source_normalize(raw, target, source_engines)
    for field in (
        "target_domain_used_for_fit",
        "selection_engines_used_for_fit",
        "confirmation_engines_used_for_fit",
    ):
        require_false(normalizer_audit.get(field), label=f"normalizer {field}")
    support_dataset = a251b.a23.WindowDataset(normalized[target], support_engines, int(cfg["window_size"]))
    selection_dataset = a232.CausalPrefixDataset(normalized[target], selection_engines, int(cfg["window_size"]))
    if len(support_dataset) < 1:
        raise A271Error("target support dataset is empty")
    if len(selection_dataset) != len(selection_engines) * len(ANCHORS):
        raise A271Error("selection causal-prefix coverage is incomplete")
    if any(bool(meta["input_uses_future_cycles"]) for meta in selection_dataset.meta):
        raise A271Error("selection causal-prefix input uses future cycles")

    row = starting_row(inventory, target, model_seed, split)
    cfg = deepcopy(cfg)
    cfg["__config_path"] = str(config_path)
    _, start_state = validate_start_payload(
        row,
        target=target,
        model_seed=model_seed,
        split=split,
        support_engines=support_engines,
        cfg=cfg,
    )
    device = torch.device("cpu") if args.device == "cpu" else a251b.a23.resolve_device(args.device)
    loader_seed = model_seed * 1_000_000 + split * 100 + PRIMARY_SHOT
    states: dict[str, dict[str, torch.Tensor]] = {}
    accountings: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    batch_hashes: dict[str, str] = {}
    for arm in ARMS:
        state, accounting, history, batch_hash = adapt_arm(
            start_state,
            arm=arm,
            dataset=support_dataset,
            cfg=cfg,
            model_seed=model_seed,
            loader_seed=loader_seed,
            device=device,
            target=target,
            split=split,
        )
        states[arm] = state
        accountings[arm] = accounting
        histories[arm] = history
        batch_hashes[arm] = batch_hash
    require_equal(batch_hashes[CONTROL_ARM], batch_hashes[CANDIDATE_ARM], label="paired batch sequence hash")
    for field in (
        "target_gradient_updates",
        "target_window_presentations",
        "target_forward_calls",
        "target_backward_calls",
        "low_rul_support_window_presentations",
    ):
        require_equal(accountings[CONTROL_ARM][field], accountings[CANDIDATE_ARM][field], label=f"paired accounting {field}")

    prediction_frames: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        state = states[arm]
        numel, count, schema = a251b.state_schema(state)
        require_equal(numel, GNN_STATE_TENSOR_NUMEL, label=f"{arm} state numel")
        require_equal(count, GNN_STATE_TENSOR_COUNT, label=f"{arm} tensor count")
        require_equal(schema, GNN_STATE_SCHEMA_SHA256, label=f"{arm} schema")
        final_state_hash = state_value_hash(state)
        checkpoint = directory / f"{arm}.pt"
        atomic_torch_save(
            checkpoint,
            checkpoint_payload(
                state,
                arm=arm,
                target=target,
                model_seed=model_seed,
                split=split,
                support_engines=support_engines,
                selection_engines=selection_engines,
                start_row=row,
                accounting=accountings[arm],
                batch_sequence_hash=batch_hashes[arm],
                history=histories[arm],
                frozen_hashes=frozen_hashes,
                config_hash=sha256(config_path),
            ),
        )
        checkpoint_hash = sha256(checkpoint)
        loaded = safe_load_checkpoint(checkpoint)
        verifier = a251b.make_model(METHOD, cfg, model_seed).cpu()
        verifier.load_state_dict(loaded["state"], strict=True)
        require_true(bool(getattr(verifier, "use_gat", False)), label=f"{arm} reload graph")
        require_equal(state_value_hash(loaded["state"]), final_state_hash, label=f"{arm} state roundtrip")
        predictions, selection_calls = predict_arm(
            loaded["state"],
            arm=arm,
            dataset=selection_dataset,
            cfg=cfg,
            device=device,
            target=target,
            model_seed=model_seed,
            split=split,
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
        )
        accountings[arm]["selection_forward_calls"] = selection_calls
        accountings[arm]["selection_window_presentations"] = len(predictions)
        prediction_frames.append(predictions)
        checkpoint_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split,
                "shot": PRIMARY_SHOT,
                "arm": arm,
                "architecture": ARCHITECTURE,
                "method": METHOD,
                "graph_enabled": True,
                "target_epochs": TARGET_EPOCHS,
                "low_rul_threshold": LOW_RUL_THRESHOLD,
                "low_rul_penalty_lambda": 0.0 if arm == CONTROL_ARM else PENALTY_LAMBDA,
                "starting_checkpoint": str(row["resolved_checkpoint"]),
                "starting_checkpoint_sha256": str(row["checkpoint_sha256"]),
                "starting_state_sha256": str(row["source_state_sha256"]),
                "final_state_sha256": final_state_hash,
                "target_parameter_drift_l2": state_delta_l2(start_state, state),
                "batch_sequence_sha256": batch_hashes[arm],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_reload_passed": True,
                "state_schema_sha256": schema,
                "state_tensor_numel": numel,
                "state_tensor_count": count,
                "inherited_source_gradient_updates": SOURCE_GRADIENT_UPDATES,
                "inherited_source_window_presentations": SOURCE_WINDOW_PRESENTATIONS,
                **accountings[arm],
                "selection_development_used_for_training": False,
                "selection_development_used_for_evaluation": True,
                "A25_2b_confirmation_used": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            }
        )
        del verifier, loaded

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_predictions = len(ARMS) * len(selection_engines) * len(ANCHORS)
    if len(predictions) != expected_predictions:
        raise A271Error(f"worker predictions={len(predictions)}, expected={expected_predictions}")
    prediction_keys = ["arm", "engine_id", "prefix_label"]
    if predictions.duplicated(prediction_keys).any():
        raise A271Error("worker prediction keys are duplicated")
    control_keys = set(
        map(tuple, predictions.loc[predictions["arm"] == CONTROL_ARM, ["engine_id", "prefix_label"]].to_numpy())
    )
    candidate_keys = set(
        map(tuple, predictions.loc[predictions["arm"] == CANDIDATE_ARM, ["engine_id", "prefix_label"]].to_numpy())
    )
    require_equal(candidate_keys, control_keys, label="paired prediction engine keys")
    metrics = run_level_metrics(predictions)
    pairs = paired_worker_metrics(metrics)
    history = pd.DataFrame(histories[CONTROL_ARM] + histories[CANDIDATE_ARM])
    inventory_frame = pd.DataFrame(checkpoint_rows)
    audit = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "target_domain": target,
                "model_seed": model_seed,
                "support_split_seed": split,
                "shot": PRIMARY_SHOT,
                "starting_checkpoint_sha256": str(row["checkpoint_sha256"]),
                "starting_state_sha256": str(row["source_state_sha256"]),
                "starting_state_identical": True,
                "control_batch_sequence_sha256": batch_hashes[CONTROL_ARM],
                "candidate_batch_sequence_sha256": batch_hashes[CANDIDATE_ARM],
                "batch_sequence_identical": True,
                "control_target_gradient_updates": accountings[CONTROL_ARM]["target_gradient_updates"],
                "candidate_target_gradient_updates": accountings[CANDIDATE_ARM]["target_gradient_updates"],
                "target_gradient_updates_identical": True,
                "control_target_window_presentations": accountings[CONTROL_ARM]["target_window_presentations"],
                "candidate_target_window_presentations": accountings[CANDIDATE_ARM]["target_window_presentations"],
                "target_window_presentations_identical": True,
                "control_target_forward_calls": accountings[CONTROL_ARM]["target_forward_calls"],
                "candidate_target_forward_calls": accountings[CANDIDATE_ARM]["target_forward_calls"],
                "target_forward_calls_identical": True,
                "control_target_backward_calls": accountings[CONTROL_ARM]["target_backward_calls"],
                "candidate_target_backward_calls": accountings[CANDIDATE_ARM]["target_backward_calls"],
                "target_backward_calls_identical": True,
                "control_low_rul_support_windows": accountings[CONTROL_ARM]["low_rul_support_window_presentations"],
                "candidate_low_rul_support_windows": accountings[CANDIDATE_ARM]["low_rul_support_window_presentations"],
                "low_rul_support_windows_identical": True,
                "graph_enabled_both_arms": True,
                "source_retraining_in_A27_1": False,
                "selection_development_used_for_training": False,
                "A25_2b_confirmation_used": False,
                "official_test_files_accessed": False,
            }
        ]
    )
    outputs = {
        "prediction_records.csv": predictions,
        "run_level_metrics.csv": metrics,
        "paired_worker_metrics.csv": pairs,
        "target_adaptation_history.csv": history,
        "checkpoint_inventory.csv": inventory_frame,
        "matched_target_compute_audit.csv": audit,
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
        "support_engine_count": len(support_engines),
        "selection_engine_count": len(selection_engines),
        "confirmation_engine_count_metadata_only": len(confirmation_engines),
        "completed_arms": len(ARMS),
        "completed_predictions": len(predictions),
        "completed_run_level_records": len(metrics),
        "completed_paired_records": len(pairs),
        "completed_checkpoints": len(inventory_frame),
        "starting_state_identical": True,
        "batch_sequence_identical": True,
        "matched_target_compute_passed": True,
        "selection_development_used_for_training": False,
        "selection_development_used_for_evaluation": True,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "A25_2b_confirmation_used": False,
        "candidate_selected_at_worker_level": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "contract_hashes": dict(sorted(frozen_hashes.items())),
        "output_sha256": output_hashes,
    }
    atomic_json(directory / "worker_status.json", status)
    print(f"[A27.1] completed {directory.name}", flush=True)


def global_arm_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, anchor), frame in predictions.groupby(["arm", "registered_rul_anchor"], sort=True):
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "analysis_role": "exploratory_selection_development",
                "arm": str(arm),
                "architecture": ARCHITECTURE,
                "method": METHOD,
                "graph_enabled": True,
                "shot": PRIMARY_SHOT,
                "registered_rul_anchor": float(anchor),
                **metric_values(frame),
                "candidate_selected": False,
                "formal_efficacy_claim_allowed": False,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(ARMS) * len(ANCHORS):
        raise A271Error("global arm summary cardinality mismatch")
    return result


def summary_value(summary: pd.DataFrame, arm: str, anchor: float, metric: str) -> float:
    row = summary.loc[
        (summary["arm"].astype(str) == arm)
        & np.isclose(pd.to_numeric(summary["registered_rul_anchor"]), anchor)
    ]
    if len(row) != 1:
        raise A271Error(f"global summary missing {arm}/RUL{anchor:g}")
    value = float(row.iloc[0][metric])
    if not math.isfinite(value):
        raise A271Error(f"global summary non-finite {arm}/RUL{anchor:g}/{metric}")
    return value


def make_gate_row(
    *,
    family: str,
    gate_id: str,
    anchor: float | str,
    metric: str,
    observed: float | int | None,
    operator: str,
    threshold: float | int,
    passed: bool,
    denominator: int | str = "",
    domain: str = "ALL",
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "gate_family": family,
        "gate_id": gate_id,
        "target_domain": domain,
        "registered_rul_anchor": anchor,
        "metric": metric,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "denominator": denominator,
        "passed": bool(passed),
        "preregistered": True,
        "formal_efficacy_claim_allowed": False,
    }


def finite_series(values: pd.Series) -> list[float]:
    output: list[float] = []
    for value in values.tolist():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def build_gate_results(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    low = paired.loc[np.isclose(pd.to_numeric(paired["registered_rul_anchor"]), 15.0)]
    if len(low) != EXPECTED_WORKERS:
        raise A271Error("low-RUL paired worker rows are incomplete")
    worker_threshold = int(plan["low_rul_worker_gate"]["minimum_improving_workers_per_metric"])
    for metric in PRIMARY_METRICS:
        improving = int((pd.to_numeric(low[f"delta_{metric}"], errors="coerce") < 0.0).sum())
        rows.append(
            make_gate_row(
                family="low_rul_worker",
                gate_id=f"LOW_WORKER_{metric.upper()}",
                anchor=15.0,
                metric=metric,
                observed=improving,
                operator=">=",
                threshold=worker_threshold,
                denominator=EXPECTED_WORKERS,
                passed=improving >= worker_threshold,
            )
        )
    pooled_threshold = float(plan["low_rul_pooled_gate"]["maximum_candidate_to_control_relative_change"])
    for metric in PRIMARY_METRICS:
        reference = summary_value(summary, CONTROL_ARM, 15.0, metric)
        candidate = summary_value(summary, CANDIDATE_ARM, 15.0, metric)
        observed = relative_change(candidate, reference)
        rows.append(
            make_gate_row(
                family="low_rul_pooled",
                gate_id=f"LOW_POOLED_{metric.upper()}",
                anchor=15.0,
                metric=metric,
                observed=observed,
                operator="<=",
                threshold=pooled_threshold,
                passed=observed is not None and observed <= pooled_threshold,
            )
        )
    reference_over = summary_value(summary, CONTROL_ARM, 15.0, "overprediction_rate")
    candidate_over = summary_value(summary, CANDIDATE_ARM, 15.0, "overprediction_rate")
    over_delta = candidate_over - reference_over
    over_threshold = float(plan["low_rul_overprediction_gate"]["pooled_candidate_minus_control_maximum"])
    rows.append(
        make_gate_row(
            family="low_rul_overprediction",
            gate_id="LOW_POOLED_OVERPREDICTION_RATE",
            anchor=15.0,
            metric="overprediction_rate",
            observed=over_delta,
            operator="<=",
            threshold=over_threshold,
            passed=over_delta <= over_threshold,
        )
    )
    guard_threshold = float(plan["mid_high_guardrail"]["maximum_pooled_relative_deterioration"])
    for anchor in (45.0, 90.0):
        for metric in CORE_METRICS:
            reference = summary_value(summary, CONTROL_ARM, anchor, metric)
            candidate = summary_value(summary, CANDIDATE_ARM, anchor, metric)
            observed = relative_change(candidate, reference)
            rows.append(
                make_gate_row(
                    family="mid_high_pooled_guardrail",
                    gate_id=f"RUL{int(anchor)}_POOLED_{metric.upper()}",
                    anchor=anchor,
                    metric=metric,
                    observed=observed,
                    operator="<=",
                    threshold=guard_threshold,
                    passed=observed is not None and observed <= guard_threshold,
                )
            )
    joint_threshold = int(
        plan["worker_joint_guardrail"][
            "minimum_workers_without_simultaneous_rmse_and_nasa_deterioration"
        ]
    )
    for anchor in ANCHORS:
        subset = paired.loc[np.isclose(pd.to_numeric(paired["registered_rul_anchor"]), anchor)]
        if len(subset) != EXPECTED_WORKERS:
            raise A271Error(f"worker joint rows incomplete at RUL{anchor:g}")
        safe = int(
            (~(
                (pd.to_numeric(subset["delta_rmse"], errors="coerce") > 0.0)
                & (pd.to_numeric(subset["delta_nasa_score"], errors="coerce") > 0.0)
            )).sum()
        )
        rows.append(
            make_gate_row(
                family="worker_joint_guardrail",
                gate_id=f"RUL{int(anchor)}_WORKER_JOINT",
                anchor=anchor,
                metric="not_simultaneous_rmse_nasa_deterioration",
                observed=safe,
                operator=">=",
                threshold=joint_threshold,
                denominator=EXPECTED_WORKERS,
                passed=safe >= joint_threshold,
            )
        )
    domain_threshold = float(
        plan["domain_heterogeneity_guardrail"]["maximum_domain_median_relative_deterioration"]
    )
    for domain in DOMAINS:
        for anchor in ANCHORS:
            subset = paired.loc[
                (paired["target_domain"].astype(str) == domain)
                & np.isclose(pd.to_numeric(paired["registered_rul_anchor"]), anchor)
            ]
            if len(subset) != len(MODEL_SEEDS) * len(SUPPORT_SPLIT_SEEDS):
                raise A271Error(f"domain gate rows incomplete: {domain}/RUL{anchor:g}")
            for metric in CORE_METRICS:
                values = finite_series(subset[f"relative_change_{metric}"])
                observed = float(np.median(values)) if len(values) == len(subset) else None
                rows.append(
                    make_gate_row(
                        family="domain_heterogeneity_guardrail",
                        gate_id=f"{domain}_RUL{int(anchor)}_{metric.upper()}",
                        domain=domain,
                        anchor=anchor,
                        metric=metric,
                        observed=observed,
                        operator="<=",
                        threshold=domain_threshold,
                        denominator=len(subset),
                        passed=observed is not None and observed <= domain_threshold,
                    )
                )
    result = pd.DataFrame(rows)
    if len(result) != 38 or result["gate_id"].duplicated().any():
        raise A271Error(f"advancement gate rows={len(result)}, expected=38 unique rows")
    return result


def verify_gate_engine(plan: Mapping[str, Any]) -> None:
    paired_rows: list[dict[str, Any]] = []
    for domain, seed, split in expected_workers():
        for anchor in ANCHORS:
            paired_rows.append(
                {
                    "target_domain": domain,
                    "model_seed": seed,
                    "support_split_seed": split,
                    "registered_rul_anchor": anchor,
                    "delta_rmse": -2.0,
                    "delta_nasa_score": -20.0,
                    "delta_positive_error_q95": -3.0,
                    "relative_change_rmse": -0.2,
                    "relative_change_nasa_score": -0.2,
                    "relative_change_positive_error_q95": -0.2,
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for arm, multiplier in ((CONTROL_ARM, 1.0), (CANDIDATE_ARM, 0.8)):
        for anchor in ANCHORS:
            summary_rows.append(
                {
                    "arm": arm,
                    "registered_rul_anchor": anchor,
                    "rmse": 10.0 * multiplier,
                    "nasa_score": 100.0 * multiplier,
                    "positive_error_q95": 20.0 * multiplier,
                    "overprediction_rate": 0.5 * multiplier,
                }
            )
    gates = build_gate_results(pd.DataFrame(paired_rows), pd.DataFrame(summary_rows), plan)
    if len(gates) != 38 or not gates["passed"].map(bool).all():
        raise A271Error("registered advancement gate engine failed its positive-control test")


def verify_preflight(
    args: argparse.Namespace,
    inventory: pd.DataFrame,
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    config_path: Path,
    raw: Mapping[str, pd.DataFrame],
) -> list[tuple[str, int, int]]:
    workers = expected_workers()
    if len(workers) != EXPECTED_WORKERS or len(set(workers)) != EXPECTED_WORKERS:
        raise A271Error("worker factorial mismatch")
    smoke_done = False
    for domain, seed, split in workers:
        tasks = a251b.worker_tasks(frames["tasks"], domain, seed, split, protocol)
        support, selection = a251b.selection_and_support(
            frames["roles"], domain, split, tuple(int(value) for value in protocol["shots"])
        )
        confirmation = a251b.role_engines(frames["roles"], domain, split, "confirmation")
        if set(confirmation) & (set(selection) | set(support[PRIMARY_SHOT])):
            raise A271Error(f"role leakage in worker {domain}/{seed}/{split}")
        source_engines = a251b.source_fit_engines(tasks, domain, raw)
        normalized, audit = a251b.source_normalize(raw, domain, source_engines)
        for field in (
            "target_domain_used_for_fit",
            "selection_engines_used_for_fit",
            "confirmation_engines_used_for_fit",
        ):
            require_false(audit.get(field), label=f"preflight normalizer {field}")
        support_dataset = a251b.a23.WindowDataset(
            normalized[domain], support[PRIMARY_SHOT], int(cfg["window_size"])
        )
        selection_dataset = a232.CausalPrefixDataset(
            normalized[domain], selection, int(cfg["window_size"])
        )
        if len(support_dataset) < 1 or len(selection_dataset) != len(selection) * len(ANCHORS):
            raise A271Error(f"dataset coverage failed: {domain}/{seed}/{split}")
        row = starting_row(inventory, domain, seed, split)
        local_cfg = deepcopy(cfg)
        local_cfg["__config_path"] = str(config_path)
        _, state = validate_start_payload(
            row,
            target=domain,
            model_seed=seed,
            split=split,
            support_engines=support[PRIMARY_SHOT],
            cfg=local_cfg,
        )
        if not smoke_done:
            a251b.a23.seed_everything(seed * 1_000_000 + split * 100 + PRIMARY_SHOT)
            model = a251b.make_model(METHOD, local_cfg, seed).cpu()
            model.load_state_dict(state, strict=True)
            loader = a251b.a23.make_loader(
                support_dataset,
                batch_size=int(local_cfg["batch_size"]),
                shuffle=True,
                seed=seed * 1_000_000 + split * 100 + PRIMARY_SHOT,
            )
            batch = next(iter(loader))
            x, y, _, _ = batch
            model.zero_grad(set_to_none=True)
            base_loss, prediction = a251b.rul_training_loss(
                model, x, y, float(local_cfg["pair_aux_weight"])
            )
            if prediction.shape != y.shape:
                raise A271Error("candidate-loss smoke prediction/target shape mismatch")
            low_mask = (y <= LOW_RUL_THRESHOLD).to(dtype=prediction.dtype)
            penalty = torch.mean(low_mask * torch.relu(prediction - y).pow(2))
            total_loss = base_loss + PENALTY_LAMBDA * penalty
            if not torch.isfinite(total_loss):
                raise A271Error("candidate-loss smoke produced a non-finite loss")
            total_loss.backward()
            if not any(parameter.grad is not None for parameter in model.parameters()):
                raise A271Error("candidate-loss smoke produced no gradients")
            del model
            smoke_done = True
        child = parse_args(worker_command(args, domain, seed, split)[2:])
        if not child.worker or child.target_domain != domain or child.confirm_run != RUN_TOKEN:
            raise A271Error("parent/worker command roundtrip failed")
    verify_gate_engine(plan)
    return workers


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
        "--a25-1b-script",
        str(args.a25_1b_script),
        "--a26-1-output-dir",
        str(args.a26_1_output_dir),
        "--a26-1-script",
        str(args.a26_1_script),
        "--a27-0-output-dir",
        str(args.a27_0_output_dir),
        "--a27-0-script",
        str(args.a27_0_script),
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


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experimentA27_1_run.lock"
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError):
            path.unlink()
        except PermissionError as exc:
            raise A271Error(f"cannot inspect existing run lock: {path}") from exc
        else:
            raise A271Error(f"another A27.1 parent is active with pid={old_pid}")
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


def top_output_names() -> tuple[str, ...]:
    return (
        "experimentA27_1_development_predictions.csv",
        "experimentA27_1_run_level_metrics.csv",
        "experimentA27_1_paired_worker_comparisons.csv",
        "experimentA27_1_target_adaptation_history.csv",
        "experimentA27_1_checkpoint_inventory.csv",
        "experimentA27_1_matched_target_compute_audit.csv",
        "experimentA27_1_global_arm_summary.csv",
        "experimentA27_1_advancement_gate_results.csv",
        "experimentA27_1_confirmation_decision.json",
    )


def completed_output(
    args: argparse.Namespace,
    frozen_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    root = resolve(args.output_dir)
    manifest_path = root / "experimentA27_1_manifest.json"
    decision_path = root / "experimentA27_1_confirmation_decision.json"
    if not manifest_path.is_file() and not decision_path.is_file():
        return None
    if not manifest_path.is_file() or not decision_path.is_file():
        raise A271Error("A27.1 final output is partial; use --resume to finish workers, not to trust it")
    if not args.resume:
        raise A271Error("complete A27.1 output exists; pass --resume to verify and return it")
    manifest = read_json(manifest_path, label="existing A27.1 manifest")
    decision = read_json(decision_path, label="existing A27.1 decision")
    require_equal(manifest.get("experiment_id"), EXPERIMENT_ID, label="existing experiment")
    require_equal(manifest.get("script_version"), SCRIPT_VERSION, label="existing version")
    require_equal(manifest.get("script_sha256"), sha256(Path(__file__).resolve()), label="existing script hash")
    require_equal(manifest.get("frozen_input_sha256"), dict(sorted(frozen_hashes.items())), label="existing input hashes")
    artifacts = validate_hash_map(root, manifest.get("artifacts"), label="existing A27.1 artifacts")
    require_equal(set(artifacts), set(top_output_names()), label="existing artifact set")
    for field in (
        "complete",
        "passed",
        "execution_integrity_passed",
        "development_only",
        "exploratory_only",
        "registered_analysis_executed_without_branching",
        "all_expected_workers_complete",
        "all_metrics_finite",
        "matched_target_compute_passed",
        "checkpoint_reload_passed",
    ):
        require_true(decision.get(field), label=f"existing A27.1 {field}")
    for field in (
        "formal_efficacy_claim",
        "policy_selected",
        "A25_2b_confirmation_path_accepted_by_script",
        "A25_2b_confirmation_used",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(decision.get(field), label=f"existing A27.1 {field}")
    inventory = read_frame(root / "experimentA27_1_checkpoint_inventory.csv", label="existing checkpoint inventory")
    if len(inventory) != EXPECTED_WORKERS * len(ARMS):
        raise A271Error("existing checkpoint inventory count mismatch")
    for row in inventory.to_dict(orient="records"):
        path = Path(str(row["checkpoint"])).resolve()
        if not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
            raise A271Error(f"existing final checkpoint failed hash validation: {path}")
    return decision


def merge(
    args: argparse.Namespace,
    workers: Sequence[tuple[str, int, int]],
    frozen_hashes: Mapping[str, str],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    root = resolve(args.output_dir)
    collections: dict[str, list[pd.DataFrame]] = {name: [] for name in worker_output_names()}
    for worker in workers:
        directory = worker_root(root, *worker)
        if not worker_complete(directory, frozen_hashes):
            raise A271Error(f"incomplete or corrupt worker: {directory.name}")
        for name in collections:
            collections[name].append(read_frame(directory / name, label=f"{directory.name}/{name}"))
    merged = {name: pd.concat(frames, ignore_index=True) for name, frames in collections.items()}
    expected_counts = {
        "run_level_metrics.csv": EXPECTED_WORKERS * len(ARMS) * len(ANCHORS),
        "paired_worker_metrics.csv": EXPECTED_WORKERS * len(ANCHORS),
        "target_adaptation_history.csv": EXPECTED_WORKERS * len(ARMS) * TARGET_EPOCHS,
        "checkpoint_inventory.csv": EXPECTED_WORKERS * len(ARMS),
        "matched_target_compute_audit.csv": EXPECTED_WORKERS,
    }
    for name, expected in expected_counts.items():
        require_equal(len(merged[name]), expected, label=f"merged {name} rows")
    predictions = merged["prediction_records.csv"]
    keys = ["target_domain", "model_seed", "support_split_seed", "arm", "engine_id", "prefix_label"]
    if predictions.duplicated(keys).any():
        raise A271Error("merged prediction keys are duplicated")
    for column in (
        "selection_development_used_for_training",
        "A25_2b_confirmation_used_for_training",
        "A25_2b_confirmation_used_for_evaluation",
        "official_test_files_accessed",
        "official_test_forward_run",
        "gradient_enabled",
        "input_uses_future_cycles",
    ):
        if predictions[column].map(lambda value: strict_bool(value, label=column)).any():
            raise A271Error(f"merged prediction boundary violation: {column}")
    if not predictions["selection_development_used_for_evaluation"].map(
        lambda value: strict_bool(value, label="selection evaluation")
    ).all():
        raise A271Error("selection was not evaluated for every prediction")
    numeric = predictions[["true_rul", "prediction", "error", "nasa_score_component"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise A271Error("merged predictions contain non-finite values")
    audits = merged["matched_target_compute_audit.csv"]
    for field in (
        "starting_state_identical",
        "batch_sequence_identical",
        "target_gradient_updates_identical",
        "target_window_presentations_identical",
        "target_forward_calls_identical",
        "target_backward_calls_identical",
        "low_rul_support_windows_identical",
        "graph_enabled_both_arms",
    ):
        if not audits[field].map(lambda value: strict_bool(value, label=field)).all():
            raise A271Error(f"merged compute audit failed: {field}")
    checkpoints = merged["checkpoint_inventory.csv"]
    if not checkpoints["checkpoint_reload_passed"].map(
        lambda value: strict_bool(value, label="checkpoint reload")
    ).all():
        raise A271Error("checkpoint reload audit failed")
    for row in checkpoints.to_dict(orient="records"):
        path = Path(str(row["checkpoint"])).resolve()
        if not path.is_file() or sha256(path) != str(row["checkpoint_sha256"]):
            raise A271Error(f"final checkpoint failed hash validation: {path}")
        payload = safe_load_checkpoint(path)
        require_equal(payload.get("experiment_id"), EXPERIMENT_ID, label="final checkpoint experiment")
        require_equal(payload.get("arm"), str(row["arm"]), label="final checkpoint arm")
        require_false(payload.get("A25_2b_confirmation_used_for_training"), label="final checkpoint A25.2b training")
        require_false(payload.get("A25_2b_confirmation_used_for_evaluation"), label="final checkpoint A25.2b evaluation")
        require_false(payload.get("official_test_files_accessed"), label="final checkpoint official test")
        require_equal(state_value_hash(payload["state"]), str(row["final_state_sha256"]), label="final checkpoint state hash")

    summary = global_arm_summary(predictions)
    gates = build_gate_results(merged["paired_worker_metrics.csv"], summary, plan)
    all_gates_passed = bool(gates["passed"].map(lambda value: strict_bool(value, label="gate pass")).all())
    summary["candidate_selected"] = all_gates_passed
    top_frames = {
        "experimentA27_1_development_predictions.csv": predictions,
        "experimentA27_1_run_level_metrics.csv": merged["run_level_metrics.csv"],
        "experimentA27_1_paired_worker_comparisons.csv": merged["paired_worker_metrics.csv"],
        "experimentA27_1_target_adaptation_history.csv": merged["target_adaptation_history.csv"],
        "experimentA27_1_checkpoint_inventory.csv": checkpoints,
        "experimentA27_1_matched_target_compute_audit.csv": audits,
        "experimentA27_1_global_arm_summary.csv": summary,
        "experimentA27_1_advancement_gate_results.csv": gates,
    }
    for name, frame in top_frames.items():
        atomic_frame(root / name, frame)
    family_counts = {
        family: {
            "passed": int(group["passed"].map(lambda value: strict_bool(value, label="gate")).sum()),
            "expected": int(len(group)),
        }
        for family, group in gates.groupby("gate_family", sort=True)
    }
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "execution_integrity_passed": True,
        "development_only": True,
        "exploratory_only": True,
        "registered_analysis_executed_without_branching": True,
        "all_expected_workers_complete": True,
        "all_metrics_finite": True,
        "expected_worker_cells": EXPECTED_WORKERS,
        "completed_worker_cells": len(workers),
        "expected_final_checkpoints": EXPECTED_WORKERS * len(ARMS),
        "completed_final_checkpoints": len(checkpoints),
        "expected_run_level_records": EXPECTED_WORKERS * len(ARMS) * len(ANCHORS),
        "completed_run_level_records": len(merged["run_level_metrics.csv"]),
        "completed_paired_worker_records": len(merged["paired_worker_metrics.csv"]),
        "completed_prediction_records": len(predictions),
        "completed_advancement_gate_records": len(gates),
        "advancement_gate_family_counts": family_counts,
        "all_advancement_gates_passed": all_gates_passed,
        "candidate_selected": all_gates_passed,
        "candidate_arm": CANDIDATE_ARM,
        "control_arm": CONTROL_ARM,
        "architecture": ARCHITECTURE,
        "method": METHOD,
        "graph_enabled": True,
        "target_shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "penalty_lambda": PENALTY_LAMBDA,
        "lambda_sweep_performed": False,
        "source_retraining_in_A27_1": False,
        "matched_target_compute_passed": True,
        "checkpoint_reload_passed": True,
        "selection_development_used_for_training": False,
        "selection_development_used_for_evaluation": True,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "A25_2b_confirmation_used": False,
        "new_predictor_training": True,
        "policy_selected": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A27.1 completed the preregistered paired development experiment and all advancement gates passed"
            if all_gates_passed
            else "A27.1 completed valid paired development training, but one or more preregistered advancement gates failed"
        ),
        "interpretation_limit": (
            "A27.1 uses historical training-file support/selection development roles. It cannot revise A25.2b, "
            "support a formal efficacy/deployment claim, or justify unregistered official-test access."
        ),
        "next_action": (
            "freeze_single_exploratory_candidate_then_preregister_one_time_external_or_official_test_evaluation"
            if all_gates_passed
            else "abandon_reptile_gnn_low_rul_repair_without_retuning_lambda_threshold_epochs_graph_or_gates"
        ),
    }
    decision_path = root / "experimentA27_1_confirmation_decision.json"
    atomic_json(decision_path, decision)
    artifacts = {name: sha256(root / name) for name in sorted(top_frames)}
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
        "all_advancement_gates_passed": all_gates_passed,
        "candidate_selected": all_gates_passed,
        "formal_efficacy_claim": False,
        "new_predictor_training": True,
        "A25_2b_confirmation_path_accepted_by_script": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA27_1_manifest.json", manifest)
    return decision


def parent(args: argparse.Namespace) -> None:
    torch.set_num_threads(int(args.torch_threads))
    inventory, intervention, plan, a261_manifest, frozen_hashes = validate_a27_0(args)
    protocol, frames, contract_hashes, cfg, config_path, raw = load_contract_and_data(args, a261_manifest)
    frozen_hashes = {
        **frozen_hashes,
        **{f"A25_1a::{name}": digest for name, digest in contract_hashes.items()},
        "config": sha256(config_path),
    }
    frozen_hashes = dict(sorted(frozen_hashes.items()))
    existing = completed_output(args, frozen_hashes)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        print("[A27.1] existing complete result and all frozen inputs were revalidated; no work repeated")
        return
    workers = verify_preflight(args, inventory, plan, protocol, frames, cfg, config_path, raw)
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": bool(args.dry_run),
        "registered_question": intervention["registered_question"],
        "starting_checkpoint_variant": START_VARIANT,
        "validated_starting_checkpoints": len(inventory),
        "architecture": ARCHITECTURE,
        "method": METHOD,
        "graph_enabled": True,
        "target_shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "arms": list(ARMS),
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "penalty_lambda": PENALTY_LAMBDA,
        "expected_worker_cells": len(workers),
        "expected_final_checkpoints": len(workers) * len(ARMS),
        "expected_run_level_records": len(workers) * len(ARMS) * len(ANCHORS),
        "expected_paired_worker_records": len(workers) * len(ANCHORS),
        "expected_advancement_gate_records": 38,
        "source_retraining_in_A27_1": False,
        "all_worker_roles_data_starting_tensors_and_candidate_loss_preflighted": True,
        "advancement_gate_engine_smoke_passed": True,
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
        print(
            "[A27.1] dry-run passed; all frozen contracts, data roles, starting tensors, "
            "candidate loss and gate logic are compatible; no predictor was trained",
            flush=True,
        )
        return

    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    context = (
        inventory,
        intervention,
        plan,
        a261_manifest,
        frozen_hashes,
        protocol,
        frames,
        cfg,
        config_path,
        raw,
    )
    if args.device == "cpu":
        for domain, seed, split in workers:
            local = deepcopy(args)
            local.worker = True
            local.target_domain = domain
            local.model_seed = seed
            local.support_split_seed = split
            run_worker(local, context=context)
    else:
        inventory_rows = gpu_inventory()
        eligible = [
            item["index"]
            for item in inventory_rows
            if item["index"] in set(args.gpu_ids)
            and item["free_mb"] >= int(args.min_free_memory_mb)
            and item["utilization"] <= int(args.max_gpu_utilization)
        ]
        if not eligible:
            raise A271Error(f"no eligible GPU; inventory={inventory_rows}")
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
                    f"[A27.1] launched target={item[0]} seed={item[1]} split={item[2]} "
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
                    raise A271Error(f"worker failed item={item} exit={code}\n{tail}")
                print(
                    f"[A27.1] completed target={item[0]} seed={item[1]} split={item[2]} gpu={gpu}",
                    flush=True,
                )
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(3)
    decision = merge(args, workers, frozen_hashes, plan)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    print(
        "[A27.1] completed preregistered paired low-RUL-safe target adaptation development",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
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
    except KeyboardInterrupt:
        print("[A27.1] interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[A27.1] error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
