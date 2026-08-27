#!/usr/bin/env python3
"""A25.2b: one-time sealed confirmation evaluation of frozen A25.1b models.

The A25.2a preregistration must be complete before this program is used.  A
dry-run validates all immutable artifacts, opens every checkpoint tensor and
checks strict model compatibility, but does not parse C-MAPSS training rows or
construct any confirmation example.  A formal run requires the explicit
unseal token.  Only then are the registered confirmation engines evaluated.

The formal evaluator performs no training, adaptation, tuning, early stopping,
checkpoint selection, optimizer construction or backward call.  It evaluates
the exact 192 frozen checkpoints at RUL anchors 90, 45 and 15, then executes
the preregistered K=5 same-architecture paired inference with hierarchical
bootstrap and Holm correction.  K=1 and K=2 remain secondary.
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
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA23_1_few_shot_transfer_baselines as a23  # noqa: E402
from scripts import experimentA23_2_causal_prefix_endpoint_audit as a232  # noqa: E402
from scripts import experimentA23_4_formal_causal_anchor_evaluation_and_hierarchical_inference as a234  # noqa: E402
from scripts import experimentA25_1b_same_architecture_compute_accounted_selection_pilot as a251b  # noqa: E402


EXPERIMENT_ID = "experimentA25_2b"
SCRIPT_VERSION = "experimentA25_2b_frozen_checkpoint_sealed_confirmation_evaluator_v1"
UNSEAL_TOKEN = "A25.2B_UNSEAL_CONFIRMATION"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (140, 141)
SUPPORT_SPLIT_SEEDS = (7501, 7502)
SHOTS = (1, 2, 5)
PRIMARY_SHOT = 5
RUL_ANCHORS = (90.0, 45.0, 15.0)
PRIMARY_METRICS = ("rmse", "nasa_score")
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
ALPHA = 0.05
NONINFERIORITY_MARGIN = 0.03
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 25200
HASH_RE = __import__("re").compile(r"^[0-9a-f]{64}$")

# Reuse the audited causal-prefix implementation under the A25.2 contract.
a232.EXPERIMENT_ID = EXPERIMENT_ID
a232.REGISTERED_RUL_ANCHORS = RUL_ANCHORS
a234.EXPERIMENT_ID = EXPERIMENT_ID
a234.DOMAINS = DOMAINS
a234.MODEL_SEEDS = MODEL_SEEDS
a234.SUPPORT_SPLIT_SEEDS = SUPPORT_SPLIT_SEEDS
a234.SHOTS = SHOTS
a234.PRIMARY_SHOT = PRIMARY_SHOT
a234.RUL_ANCHORS = RUL_ANCHORS
a234.PRIMARY_METRICS = PRIMARY_METRICS


class A252bError(RuntimeError):
    """Raised when the frozen confirmation evaluation cannot proceed safely."""


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


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise A252bError(f"refusing to write empty CSV: {path.name}")
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


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise A252bError(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A252bError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A252bError(f"{label} must contain a JSON object: {path}")
    return payload


def read_frame(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise A252bError(f"{label} is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise A252bError(f"cannot parse {label}: {path}: {exc}") from exc
    if frame.empty:
        raise A252bError(f"{label} is empty: {path}")
    return frame


def require_fields(payload: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = sorted(set(fields) - set(payload))
    if missing:
        raise A252bError(f"{label} lacks required fields: {missing}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise A252bError(f"{label} lacks required columns: {missing}")


def strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise A252bError(f"{label} is not a strict boolean: {value!r}")


def require_hash(value: Any, *, label: str) -> str:
    digest = str(value).strip()
    if HASH_RE.fullmatch(digest) is None:
        raise A252bError(f"{label} is not a SHA256 digest: {value!r}")
    return digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time frozen-checkpoint A25.2 sealed confirmation evaluation"
    )
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
        "--a25-2a-output-dir",
        type=Path,
        default=Path("outputs/experimentA25_2a_sealed_confirmation_preregistration_preflight"),
    )
    parser.add_argument(
        "--a25-2a-script",
        type=Path,
        default=Path("scripts/experimentA25_2a_sealed_confirmation_preregistration_preflight.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA25_2b_frozen_checkpoint_sealed_confirmation_evaluator"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--unseal-confirmation",
        default="",
        help=f"Required for formal execution; pass exactly {UNSEAL_TOKEN!r}.",
    )
    args = parser.parse_args(argv)
    if args.inference_batch_size is not None and args.inference_batch_size < 1:
        raise A252bError("--inference-batch-size must be positive")
    if args.torch_threads < 1:
        raise A252bError("--torch-threads must be positive")
    if not args.dry_run and args.unseal_confirmation != UNSEAL_TOKEN:
        raise A252bError(
            f"formal evaluation requires --unseal-confirmation {UNSEAL_TOKEN}; run --dry-run first"
        )
    return args


def validate_hash_map(root: Path, mapping: Any, *, label: str) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise A252bError(f"{label} is absent or empty")
    observed: dict[str, str] = {}
    for name, expected_raw in sorted(mapping.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise A252bError(f"{label} contains an unsafe artifact name: {name!r}")
        expected = require_hash(expected_raw, label=f"{label}[{name}]")
        path = root / name
        if not path.is_file():
            raise A252bError(f"frozen artifact is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise A252bError(f"frozen artifact hash mismatch: {path}")
        observed[name] = actual
    return observed


def load_a25_2a_contract(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.a25_2a_output_dir)
    decision_path = root / "experimentA25_2a_confirmation_decision.json"
    manifest_path = root / "experimentA25_2a_manifest.json"
    plan_path = root / "experimentA25_2a_registered_statistical_analysis_plan.json"
    evaluator_path = root / "experimentA25_2a_independent_evaluator_contract.json"
    checkpoint_path = root / "experimentA25_2a_frozen_checkpoint_inventory.csv"
    confirmation_path = root / "experimentA25_2a_confirmation_engine_inventory.csv"
    integrity_path = root / "experimentA25_2a_input_integrity.json"
    decision = read_json(decision_path, label="A25.2a decision")
    manifest = read_json(manifest_path, label="A25.2a manifest")
    plan = read_json(plan_path, label="A25.2a registered statistical plan")
    evaluator = read_json(evaluator_path, label="A25.2a evaluator contract")
    integrity = read_json(integrity_path, label="A25.2a input integrity")
    if decision.get("experiment_id") != "experimentA25_2a":
        raise A252bError("input decision is not experimentA25_2a")
    for field in (
        "complete", "passed", "preflight_only", "preregistered", "sealed_confirmation",
        "statistical_plan_frozen", "checkpoint_set_frozen", "confirmation_role_set_frozen",
        "same_architecture_pairing_verified", "compute_accounting_verified",
    ):
        if decision.get(field) is not True:
            raise A252bError(f"A25.2a requires {field}=true")
    for field in (
        "new_predictor_training", "checkpoint_tensors_opened",
        "confirmation_observations_opened", "confirmation_engines_evaluated",
        "formal_efficacy_claim", "official_test_files_accessed", "official_test_forward_run",
    ):
        if decision.get(field) is not False:
            raise A252bError(f"A25.2a requires {field}=false")
    if int(decision.get("frozen_checkpoints", -1)) != 192:
        raise A252bError("A25.2a did not freeze exactly 192 checkpoints")
    if int(decision.get("primary_K5_checkpoints", -1)) != 64:
        raise A252bError("A25.2a did not freeze exactly 64 primary K=5 checkpoints")
    if manifest.get("experiment_id") != "experimentA25_2a":
        raise A252bError("A25.2a manifest identity mismatch")
    for field in ("preregistered", "sealed_confirmation"):
        if manifest.get(field) is not True:
            raise A252bError(f"A25.2a manifest requires {field}=true")
    for field in (
        "confirmation_engines_evaluated", "new_predictor_training",
        "official_test_files_accessed", "official_test_forward_run",
    ):
        if manifest.get(field) is not False:
            raise A252bError(f"A25.2a manifest requires {field}=false")
    artifact_hashes = validate_hash_map(root, manifest.get("artifacts"), label="A25.2a artifacts")
    validate_hash_map(root, decision.get("artifact_sha256"), label="A25.2a decision artifacts")
    script_path = resolve(args.a25_2a_script)
    if not script_path.is_file():
        raise A252bError(f"A25.2a script is missing: {script_path}")
    if sha256(script_path) != require_hash(manifest.get("script_sha256"), label="A25.2a script hash"):
        raise A252bError("A25.2a script differs from its executed manifest")

    require_fields(
        plan,
        (
            "registered_before_confirmation_open", "primary_shot", "secondary_shots",
            "primary_hypothesis_family_no_graph", "replication_hypothesis_family_gnn",
            "low_rul_safety_gate", "bootstrap_design", "bootstrap_repetitions",
            "random_seed_for_bootstrap", "multiplicity_control", "family_success_rule",
            "new_predictor_training_allowed", "checkpoint_selection_allowed",
            "hyperparameter_tuning_allowed",
        ),
        label="A25.2a plan",
    )
    if plan.get("registered_before_confirmation_open") is not True:
        raise A252bError("A25.2a plan was not registered before confirmation")
    if int(plan.get("primary_shot", -1)) != PRIMARY_SHOT:
        raise A252bError("A25.2a primary shot changed")
    if tuple(map(int, plan.get("secondary_shots", []))) != (1, 2):
        raise A252bError("A25.2a secondary shots changed")
    if plan.get("bootstrap_design") != (
        "target_domain_then_model_seed_then_support_split_then_paired_engine"
    ):
        raise A252bError("A25.2a bootstrap hierarchy changed")
    if int(plan.get("bootstrap_repetitions", -1)) != BOOTSTRAP_REPETITIONS:
        raise A252bError("A25.2a bootstrap repetition count changed")
    if int(plan.get("random_seed_for_bootstrap", -1)) != BOOTSTRAP_SEED:
        raise A252bError("A25.2a bootstrap seed changed")
    if plan.get("multiplicity_control") != "Holm_within_each_six_check_family":
        raise A252bError("A25.2a multiplicity rule changed")
    if plan.get("family_success_rule") != "all_six_Holm_corrected_superiority_checks_pass":
        raise A252bError("A25.2a family success rule changed")
    for field in (
        "new_predictor_training_allowed", "checkpoint_selection_allowed",
        "hyperparameter_tuning_allowed",
    ):
        if plan.get(field) is not False:
            raise A252bError(f"A25.2a plan requires {field}=false")
    expected_families = {
        "primary_hypothesis_family_no_graph": (
            "reptile_meta_no_graph", "ordinary_no_graph_pft"
        ),
        "replication_hypothesis_family_gnn": (
            "reptile_meta_gnn", "ordinary_gnn_pft"
        ),
    }
    for name, (candidate, reference) in expected_families.items():
        family = plan.get(name)
        if not isinstance(family, dict):
            raise A252bError(f"A25.2a {name} is malformed")
        if family.get("candidate") != candidate or family.get("reference") != reference:
            raise A252bError(f"A25.2a {name} contrast changed")
        if int(family.get("shot", -1)) != PRIMARY_SHOT:
            raise A252bError(f"A25.2a {name} shot changed")
        if tuple(map(float, family.get("anchors", []))) != RUL_ANCHORS:
            raise A252bError(f"A25.2a {name} anchors changed")
        if tuple(family.get("metrics", [])) != PRIMARY_METRICS:
            raise A252bError(f"A25.2a {name} metrics changed")
    gate = plan.get("low_rul_safety_gate")
    if (
        not isinstance(gate, dict)
        or float(gate.get("anchor", -1)) != 15.0
        or tuple(gate.get("metrics", [])) != PRIMARY_METRICS
        or float(gate.get("noninferiority_margin_pct", -1)) != 3.0
        or gate.get("one_sided") is not True
    ):
        raise A252bError("A25.2a low-RUL safety gate changed")

    preregistration_hash = canonical_sha256(plan)
    if evaluator.get("experiment_id") != "experimentA25_2a":
        raise A252bError("A25.2a evaluator contract identity mismatch")
    if evaluator.get("next_experiment_id") != EXPERIMENT_ID:
        raise A252bError("A25.2a evaluator contract does not authorize A25.2b")
    if evaluator.get("required_preregistration_sha256") != preregistration_hash:
        raise A252bError("A25.2a evaluator/preregistration digest mismatch")
    required_evaluator_values = {
        "gradient_enabled": False,
        "optimizer_construction_allowed": False,
        "backward_calls_allowed": 0,
        "new_predictor_training_allowed": False,
        "early_stopping_allowed": False,
        "checkpoint_selection_allowed": False,
        "confirmation_passes": 1,
        "primary_shot": PRIMARY_SHOT,
    }
    for field, expected in required_evaluator_values.items():
        if evaluator.get(field) != expected:
            raise A252bError(f"A25.2a evaluator contract changed: {field}")
    if tuple(map(float, evaluator.get("registered_anchors", []))) != RUL_ANCHORS:
        raise A252bError("A25.2a evaluator anchors changed")
    if not all(integrity.get(field) is True for field in (
        "all_upstream_artifact_hashes_verified", "all_192_checkpoint_hashes_verified",
        "all_role_partitions_verified", "same_architecture_pairing_verified",
        "compute_accounting_verified",
    )):
        raise A252bError("A25.2a input integrity is incomplete")
    for field in (
        "checkpoint_tensors_opened", "confirmation_observations_opened",
        "confirmation_predictions_run", "official_test_files_accessed",
        "official_test_forward_run",
    ):
        if integrity.get(field) is not False:
            raise A252bError(f"A25.2a integrity requires {field}=false")

    checkpoints = read_frame(checkpoint_path, label="A25.2a frozen checkpoint inventory")
    confirmations = read_frame(confirmation_path, label="A25.2a confirmation inventory")
    return {
        "root": root,
        "decision": decision,
        "manifest": manifest,
        "plan": plan,
        "evaluator": evaluator,
        "integrity": integrity,
        "checkpoints": checkpoints,
        "confirmations": confirmations,
        "preregistration_hash": preregistration_hash,
        "artifact_hashes": artifact_hashes,
        "script_path": script_path,
    }


def verify_training_file_hashes_only(
    data_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    inventory = protocol.get("training_file_inventory")
    if not isinstance(inventory, list):
        raise A252bError("A25.1a training file inventory is missing")
    expected = {str(item.get("domain")): item for item in inventory if isinstance(item, dict)}
    if set(expected) != set(DOMAINS):
        raise A252bError("A25.1a training file inventory is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        try:
            path = a251b.resolve_train_file(data_dir, domain)
        except Exception as exc:
            raise A252bError(str(exc)) from exc
        if path.name.lower().startswith("test_") or "rul_" in path.name.lower():
            raise A252bError(f"refusing non-training input: {path}")
        actual = sha256(path)
        expected_hash = require_hash(expected[domain].get("sha256"), label=f"{domain} data hash")
        if actual != expected_hash:
            raise A252bError(f"training file hash mismatch for {domain}: {path}")
        result[domain] = {
            "path": str(path), "sha256": actual,
            "rows_not_parsed_in_dry_run": True,
        }
    return result


def validate_confirmation_inventory(
    frozen: pd.DataFrame, roles: pd.DataFrame
) -> dict[tuple[str, int], tuple[int, ...]]:
    required = {
        "target_domain", "support_split_seed", "engine_id", "role",
        "role_metadata_only", "outcome_opened_in_A25_2a", "prediction_run_in_A25_2a",
    }
    require_columns(frozen, required, label="A25.2a confirmation inventory")
    if len(frozen) != 1094:
        raise A252bError(f"A25.2a confirmation role count={len(frozen)}, expected=1094")
    for column in ("support_split_seed", "engine_id"):
        frozen[column] = pd.to_numeric(frozen[column], errors="raise").astype(int)
    if set(frozen["target_domain"].astype(str)) != set(DOMAINS):
        raise A252bError("A25.2a confirmation domains differ from FD001--FD004")
    if set(frozen["support_split_seed"]) != set(SUPPORT_SPLIT_SEEDS):
        raise A252bError("A25.2a confirmation split seeds changed")
    if set(frozen["role"].astype(str)) != {"confirmation"}:
        raise A252bError("A25.2a confirmation inventory contains a non-confirmation role")
    if frozen.duplicated(["target_domain", "support_split_seed", "engine_id"]).any():
        raise A252bError("A25.2a confirmation inventory contains duplicate role rows")
    for column, expected in (
        ("role_metadata_only", True),
        ("outcome_opened_in_A25_2a", False),
        ("prediction_run_in_A25_2a", False),
    ):
        values = {strict_bool(value, label=column) for value in frozen[column]}
        if values != {expected}:
            raise A252bError(f"A25.2a confirmation inventory requires {column}={expected}")
    result: dict[tuple[str, int], tuple[int, ...]] = {}
    for domain in DOMAINS:
        for split in SUPPORT_SPLIT_SEEDS:
            registered = tuple(
                sorted(
                    frozen.loc[
                        (frozen["target_domain"].astype(str) == domain)
                        & (frozen["support_split_seed"] == split),
                        "engine_id",
                    ].astype(int)
                )
            )
            try:
                upstream = tuple(a251b.role_engines(roles, domain, split, "confirmation"))
                support = set(a251b.role_engines(roles, domain, split, "support_pool", PRIMARY_SHOT))
                selection = set(a251b.role_engines(roles, domain, split, "selection"))
            except Exception as exc:
                raise A252bError(str(exc)) from exc
            if registered != upstream:
                raise A252bError(f"A25.2a confirmation engines changed for {domain}/split={split}")
            if set(registered) & (support | selection):
                raise A252bError(f"confirmation role leakage for {domain}/split={split}")
            if len(registered) < 20:
                raise A252bError(f"insufficient confirmation engines for {domain}/split={split}")
            result[(domain, split)] = registered
    return result


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A252bError(f"checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise A252bError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise A252bError(f"checkpoint lacks a state dictionary: {path}")
    for name, tensor in payload["state"].items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise A252bError(f"invalid state entry in {path}: {name!r}")
        if not torch.isfinite(tensor).all().item():
            raise A252bError(f"non-finite checkpoint tensor in {path}: {name}")
    return payload


def worker_normalizer_path(a25b_root: Path, domain: str, model_seed: int, split: int) -> Path:
    return (
        a25b_root / "shards" / f"{domain}_mseed{model_seed}_split{split}" /
        "source_normalizer.json"
    ).resolve()


def validate_stored_normalizer(path: Path, target_domain: str) -> dict[str, Any]:
    payload = read_json(path, label="A25.1b source normalizer")
    expected_sources = sorted(domain for domain in DOMAINS if domain != target_domain)
    if sorted(payload.get("fitted_domains", [])) != expected_sources:
        raise A252bError(f"normalizer source-domain set mismatch: {path}")
    for field in (
        "target_domain_used_for_fit", "selection_engines_used_for_fit",
        "confirmation_engines_used_for_fit",
    ):
        if payload.get(field) is not False:
            raise A252bError(f"normalizer leakage flag {field}=true: {path}")
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != {"mean", "std"}:
        raise A252bError(f"malformed normalizer: {path}")
    for section in ("mean", "std"):
        if set(normalizer[section]) != set(a23.FEATURE_COLUMNS):
            raise A252bError(f"normalizer {section} feature columns mismatch: {path}")
        for feature, value in normalizer[section].items():
            if not math.isfinite(float(value)):
                raise A252bError(f"non-finite normalizer value {section}/{feature}: {path}")
    if any(float(value) <= 0 for value in normalizer["std"].values()):
        raise A252bError(f"normalizer contains nonpositive standard deviation: {path}")
    return payload


def validate_checkpoint_inventory(
    frozen: pd.DataFrame,
    a25b_root: Path,
    cfg: Mapping[str, Any],
    config_hash: str,
    contract_hashes: Mapping[str, str],
    roles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], dict[str, Any]]]:
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "architecture", "checkpoint", "checkpoint_sha256", "primary_analysis_checkpoint",
        "secondary_analysis_checkpoint", "checkpoint_opened_in_A25_2a",
        "confirmation_evaluated_in_A25_2a",
    }
    require_columns(frozen, required, label="A25.2a checkpoint inventory")
    if len(frozen) != 192:
        raise A252bError(f"A25.2a frozen checkpoint count={len(frozen)}, expected=192")
    for column in ("model_seed", "support_split_seed", "shot"):
        frozen[column] = pd.to_numeric(frozen[column], errors="raise").astype(int)
    expected_keys = {
        (domain, seed, split, shot, method)
        for domain in DOMAINS for seed in MODEL_SEEDS for split in SUPPORT_SPLIT_SEEDS
        for shot in SHOTS for method in METHODS
    }
    keys = ["target_domain", "model_seed", "support_split_seed", "shot", "method"]
    observed = set(frozen[keys].itertuples(index=False, name=None))
    if observed != expected_keys or frozen.duplicated(keys).any():
        raise A252bError("A25.2a checkpoint factorial is incomplete or duplicated")
    if set(frozen["architecture"].astype(str)) != set(PAIRS):
        raise A252bError("A25.2a checkpoint architecture set changed")
    for field in ("checkpoint_opened_in_A25_2a", "confirmation_evaluated_in_A25_2a"):
        if any(strict_bool(value, label=field) for value in frozen[field]):
            raise A252bError(f"A25.2a checkpoint inventory requires {field}=false")
    primary_flags = [strict_bool(value, label="primary checkpoint") for value in frozen["primary_analysis_checkpoint"]]
    secondary_flags = [strict_bool(value, label="secondary checkpoint") for value in frozen["secondary_analysis_checkpoint"]]
    if sum(primary_flags) != 64 or sum(secondary_flags) != 128:
        raise A252bError("A25.2a primary/secondary checkpoint counts changed")
    if any(primary != (shot == PRIMARY_SHOT) for primary, shot in zip(primary_flags, frozen["shot"])):
        raise A252bError("A25.2a primary checkpoint flag is inconsistent with K=5")
    if any(secondary == primary for secondary, primary in zip(secondary_flags, primary_flags)):
        raise A252bError("A25.2a primary/secondary checkpoint flags are not complementary")

    a25b_shards = (a25b_root / "shards").resolve()
    validated_rows: list[dict[str, Any]] = []
    normalizers: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, row in frozen.sort_values(keys, kind="stable").reset_index(drop=True).iterrows():
        domain = str(row["target_domain"])
        seed = int(row["model_seed"])
        split = int(row["support_split_seed"])
        shot = int(row["shot"])
        method = str(row["method"])
        architecture = str(row["architecture"])
        expected_architecture = "no_graph" if "no_graph" in method else "gnn"
        if architecture != expected_architecture:
            raise A252bError(f"checkpoint architecture mismatch at inventory row {index + 2}")
        path = Path(str(row["checkpoint"])).expanduser().resolve()
        try:
            path.relative_to(a25b_shards)
        except ValueError as exc:
            raise A252bError(f"checkpoint escapes A25.1b shards: {path}") from exc
        expected_hash = require_hash(row["checkpoint_sha256"], label="checkpoint hash")
        if not path.is_file() or sha256(path) != expected_hash:
            raise A252bError(f"checkpoint missing or hash mismatch: {path}")
        payload = safe_load_checkpoint(path)
        expected_metadata = {
            "experiment_id": "experimentA25_1b", "method": method,
            "architecture": architecture, "target_domain": domain, "model_seed": seed,
            "support_split_seed": split, "shot": shot,
        }
        for field, expected in expected_metadata.items():
            if payload.get(field) != expected:
                raise A252bError(
                    f"checkpoint metadata mismatch {path}: {field}={payload.get(field)!r}, "
                    f"expected={expected!r}"
                )
        if payload.get("confirmation_engines_used") is not False:
            raise A252bError(f"checkpoint used confirmation engines: {path}")
        if payload.get("contract_hashes") != dict(contract_hashes):
            raise A252bError(f"checkpoint A25.1a contract hashes changed: {path}")
        if payload.get("config_sha256") != config_hash:
            raise A252bError(f"checkpoint config hash changed: {path}")
        expected_support = a251b.role_engines(roles, domain, split, "support_pool", shot)
        expected_selection = a251b.role_engines(roles, domain, split, "selection")
        if list(map(int, payload.get("target_support_engine_ids", []))) != expected_support:
            raise A252bError(f"checkpoint support engine set changed: {path}")
        if list(map(int, payload.get("target_selection_engine_ids", []))) != expected_selection:
            raise A252bError(f"checkpoint selection engine set changed: {path}")
        try:
            model = a251b.make_model(method, cfg, seed).cpu()
            model.load_state_dict(payload["state"], strict=True)
        except Exception as exc:
            raise A252bError(f"strict checkpoint/model compatibility failed: {path}: {exc}") from exc
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        tensor_numel, tensor_count, schema_hash = a251b.state_schema(payload["state"])
        audit = payload.get("runtime_parameter_audit")
        if not isinstance(audit, dict):
            raise A252bError(f"checkpoint lacks runtime parameter audit: {path}")
        if (
            int(audit.get("state_tensor_numel", -1)) != tensor_numel
            or int(audit.get("state_tensor_count", -1)) != tensor_count
            or audit.get("state_schema_sha256") != schema_hash
        ):
            raise A252bError(f"checkpoint state schema/audit mismatch: {path}")
        accounting = payload.get("compute_accounting")
        if not isinstance(accounting, dict):
            raise A252bError(f"checkpoint lacks compute accounting: {path}")
        if int(accounting.get("source_gradient_updates", -1)) != 7500:
            raise A252bError(f"checkpoint source update budget mismatch: {path}")
        if int(accounting.get("source_window_presentations", -1)) != 480000:
            raise A252bError(f"checkpoint source window budget mismatch: {path}")
        normalizer_key = (domain, seed, split)
        if normalizer_key not in normalizers:
            normalizer_path = worker_normalizer_path(a25b_root, domain, seed, split)
            normalizer_payload = validate_stored_normalizer(normalizer_path, domain)
            normalizers[normalizer_key] = {
                "path": normalizer_path,
                "sha256": sha256(normalizer_path),
                "payload": normalizer_payload,
            }
        validated_rows.append({
            **{column: scalar(row[column]) for column in frozen.columns},
            "checkpoint_tensor_validation_passed": True,
            "strict_model_load_passed": True,
            "state_tensor_numel": tensor_numel,
            "state_tensor_count": tensor_count,
            "state_schema_sha256": schema_hash,
            "normalizer_path": str(normalizers[normalizer_key]["path"]),
            "normalizer_sha256_pre_unseal": normalizers[normalizer_key]["sha256"],
        })
        del model, payload
        if (index + 1) % 16 == 0 or index + 1 == len(frozen):
            print(f"[A25.2b] validated checkpoints {index + 1:03d}/192", flush=True)
    validated = pd.DataFrame(validated_rows)
    return validated, normalizers


def synthetic_forward_smoke(
    validated: pd.DataFrame, cfg: Mapping[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    representative = validated.sort_values(
        ["method", "target_domain", "model_seed", "support_split_seed", "shot"],
        kind="stable",
    ).groupby("method", sort=True).head(1)
    x = torch.zeros(
        4, int(cfg["window_size"]), len(a23.FEATURE_COLUMNS), dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for row in representative.itertuples(index=False):
            payload = safe_load_checkpoint(Path(str(row.checkpoint)))
            model = a251b.make_model(str(row.method), cfg, int(row.model_seed)).to(device)
            model.load_state_dict(payload["state"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            values = output.detach().cpu().numpy().reshape(-1)
            if len(values) != len(x) or not np.isfinite(values).all():
                raise A252bError(f"synthetic forward smoke failed for {row.method}")
            rows.append({
                "method": str(row.method), "architecture": str(row.architecture),
                "synthetic_forward_passed": True, "output_count": int(len(values)),
                "gradient_enabled": False, "backward_calls": 0, "optimizer_steps": 0,
            })
            del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    a25b_root = resolve(args.a25_1b_output_dir)
    a25a_root = resolve(args.a25_1a_output_dir)
    a252a = load_a25_2a_contract(args)
    output_root = resolve(args.output_dir)
    input_roots = {a25a_root, a25b_root, a252a["root"]}
    if output_root in input_roots:
        raise A252bError("A25.2b output directory must differ from every input directory")
    for root in input_roots:
        try:
            output_root.relative_to(root)
            raise A252bError("A25.2b output directory must not be nested inside an input directory")
        except ValueError:
            pass

    contract_args = SimpleNamespace(a25_1a_output_dir=args.a25_1a_output_dir)
    try:
        protocol, contract_frames, contract_hashes = a251b.load_contract(contract_args)
        cfg, config_path = a251b.load_config(protocol, args.config)
    except Exception as exc:
        raise A252bError(str(exc)) from exc
    if args.inference_batch_size is not None:
        if int(args.inference_batch_size) != int(cfg["batch_size"]):
            raise A252bError(
                "--inference-batch-size may not change the frozen config batch_size; "
                f"required={cfg['batch_size']}"
            )
    data_files = verify_training_file_hashes_only(resolve(args.data_dir), protocol)
    confirmation_sets = validate_confirmation_inventory(
        a252a["confirmations"].copy(), contract_frames["roles"]
    )
    validated, normalizers = validate_checkpoint_inventory(
        a252a["checkpoints"].copy(), a25b_root, cfg, sha256(config_path),
        contract_hashes, contract_frames["roles"],
    )
    try:
        device = a23.resolve_device(args.device)
    except Exception as exc:
        raise A252bError(str(exc)) from exc
    torch.set_num_threads(int(args.torch_threads))
    smoke = synthetic_forward_smoke(validated, cfg, device)
    expected_predictions = sum(
        len(confirmation_sets[(str(row.target_domain), int(row.support_split_seed))])
        * len(RUL_ANCHORS)
        for row in validated.itertuples(index=False)
    )
    if expected_predictions != 78768:
        raise A252bError(
            f"registered prediction cardinality={expected_predictions}, expected=78768"
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": bool(args.dry_run),
        "registered_primary_question": protocol["registered_primary_question"],
        "a25_2a_preregistration_sha256": a252a["preregistration_hash"],
        "primary_shot": PRIMARY_SHOT,
        "secondary_shots": [1, 2],
        "registered_rul_anchors": list(RUL_ANCHORS),
        "primary_metrics": list(PRIMARY_METRICS),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
        "validated_checkpoint_runs": int(len(validated)),
        "validated_normalizer_files": int(len(normalizers)),
        "confirmation_engine_role_rows": int(sum(map(len, confirmation_sets.values()))),
        "expected_prediction_records": int(expected_predictions),
        "synthetic_forward_smoke": smoke,
        "checkpoint_tensors_opened": True,
        "checkpoint_hashes_verified": True,
        "checkpoint_state_schema_verified": True,
        "training_file_hashes_verified_without_row_parsing": True,
        "normalizers_schema_validated_pre_unseal": True,
        "normalizers_recomputed_from_frozen_source_roles": False,
        "confirmation_role_metadata_read": True,
        "confirmation_observations_opened": False,
        "confirmation_predictions_run": False,
        "new_predictor_training": False,
        "optimizer_construction": False,
        "backward_calls": 0,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": True,
    }
    context = {
        "output_root": output_root,
        "a25a_root": a25a_root,
        "a25b_root": a25b_root,
        "a252a": a252a,
        "protocol": protocol,
        "contract_frames": contract_frames,
        "contract_hashes": contract_hashes,
        "cfg": cfg,
        "config_path": config_path,
        "data_dir": resolve(args.data_dir),
        "data_files": data_files,
        "confirmation_sets": confirmation_sets,
        "validated_checkpoints": validated,
        "normalizers": normalizers,
        "device": device,
    }
    return result, context


def normalizer_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != {"mean", "std"} or set(right) != {"mean", "std"}:
        return False
    for section in ("mean", "std"):
        if set(left[section]) != set(right[section]):
            return False
        for feature in left[section]:
            if not math.isclose(
                float(left[section][feature]), float(right[section][feature]),
                rel_tol=0.0, abs_tol=0.0,
            ):
                return False
    return True


def prepare_confirmation_context(context: dict[str, Any]) -> dict[str, Any]:
    args = SimpleNamespace(data_dir=context["data_dir"])
    try:
        raw = a251b.load_frames(args, context["protocol"], context["cfg"])
    except Exception as exc:
        raise A252bError(f"failed to open registered training frames after unseal: {exc}") from exc
    datasets: dict[tuple[str, int, int], a232.CausalPrefixDataset] = {}
    normalizer_audit_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    roles = context["contract_frames"]["roles"]
    tasks = context["contract_frames"]["tasks"]
    for domain in DOMAINS:
        for seed in MODEL_SEEDS:
            for split in SUPPORT_SPLIT_SEEDS:
                try:
                    worker_tasks = a251b.worker_tasks(tasks, domain, seed, split, context["protocol"])
                    source_engines = a251b.source_fit_engines(worker_tasks, domain, raw)
                    _, recomputed_audit = a251b.source_normalize(raw, domain, source_engines)
                except Exception as exc:
                    raise A252bError(
                        f"normalizer recomputation failed for {domain}/model={seed}/split={split}: {exc}"
                    ) from exc
                key = (domain, seed, split)
                stored = context["normalizers"][key]
                if not normalizer_equal(
                    recomputed_audit["normalizer"], stored["payload"]["normalizer"]
                ):
                    raise A252bError(
                        f"stored normalizer differs from frozen-source recomputation: "
                        f"{domain}/model={seed}/split={split}"
                    )
                if recomputed_audit.get("target_domain_used_for_fit") is not False:
                    raise A252bError("recomputed normalizer used target domain")
                normalized_target = a23.normalize(raw[domain], recomputed_audit["normalizer"])
                engines = context["confirmation_sets"][(domain, split)]
                try:
                    dataset = a232.CausalPrefixDataset(
                        normalized_target, engines, int(context["cfg"]["window_size"])
                    )
                except Exception as exc:
                    raise A252bError(
                        f"causal confirmation dataset failed for {domain}/model={seed}/split={split}: {exc}"
                    ) from exc
                datasets[key] = dataset
                normalizer_audit_rows.append({
                    "target_domain": domain, "model_seed": seed,
                    "support_split_seed": split,
                    "stored_normalizer_path": str(stored["path"]),
                    "stored_normalizer_sha256": stored["sha256"],
                    "recomputed_normalizer_sha256": canonical_sha256(recomputed_audit["normalizer"]),
                    "stored_equals_recomputed": True,
                    "target_domain_used_for_fit": False,
                    "selection_engines_used_for_fit": False,
                    "confirmation_engines_used_for_fit": False,
                })
                if seed == MODEL_SEEDS[0]:
                    for item in dataset.meta:
                        coverage_rows.append({
                            "target_domain": domain, "support_split_seed": split, **item,
                        })
    coverage = pd.DataFrame(coverage_rows)
    if len(coverage) != 1094 * len(RUL_ANCHORS):
        raise A252bError(
            f"unique confirmation prefix coverage={len(coverage)}, expected={1094 * len(RUL_ANCHORS)}"
        )
    if coverage.duplicated(["target_domain", "support_split_seed", "engine_id", "prefix_label"]).any():
        raise A252bError("confirmation prefix coverage contains duplicate keys")
    if set(coverage["registered_rul_anchor"].astype(float)) != set(RUL_ANCHORS):
        raise A252bError("confirmation prefix coverage does not contain all registered anchors")
    if coverage["input_uses_future_cycles"].astype(bool).any():
        raise A252bError("a confirmation prefix uses future cycles")
    context["raw_frames"] = raw
    context["datasets"] = datasets
    context["normalizer_audit"] = pd.DataFrame(normalizer_audit_rows)
    context["coverage"] = coverage
    return context


def shard_stem(row: pd.Series) -> str:
    return (
        f"{row['target_domain']}_mseed{int(row['model_seed'])}_"
        f"split{int(row['support_split_seed'])}_shot{int(row['shot']):02d}_{row['method']}"
    )


def shard_paths(root: Path, row: pd.Series) -> tuple[Path, Path]:
    directory = root / "prediction_shards"
    stem = shard_stem(row)
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def validate_shard(
    csv_path: Path,
    status_path: Path,
    row: pd.Series,
    expected_rows: int,
    preregistration_hash: str,
) -> pd.DataFrame | None:
    if not (csv_path.is_file() and status_path.is_file()):
        return None
    try:
        status = read_json(status_path, label="prediction shard status")
        if status.get("complete") is not True or status.get("passed") is not True:
            return None
        if status.get("run_key") != shard_stem(row):
            return None
        if status.get("checkpoint_sha256") != str(row["checkpoint_sha256"]):
            return None
        if status.get("preregistration_sha256") != preregistration_hash:
            return None
        if status.get("prediction_sha256") != sha256(csv_path):
            return None
        frame = pd.read_csv(csv_path)
    except Exception:
        return None
    required = {
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "architecture", "engine_id", "prefix_label", "registered_rul_anchor", "true_rul",
        "prediction", "error", "absolute_error", "squared_error", "nasa_score_component",
        "checkpoint_sha256", "gradient_enabled", "backward_calls", "optimizer_steps",
    }
    if len(frame) != expected_rows or not required <= set(frame.columns):
        return None
    identity = (
        set(frame["target_domain"].astype(str)) == {str(row["target_domain"])}
        and set(frame["model_seed"].astype(int)) == {int(row["model_seed"])}
        and set(frame["support_split_seed"].astype(int)) == {int(row["support_split_seed"])}
        and set(frame["shot"].astype(int)) == {int(row["shot"])}
        and set(frame["method"].astype(str)) == {str(row["method"])}
        and set(frame["checkpoint_sha256"].astype(str)) == {str(row["checkpoint_sha256"])}
    )
    if not identity or frame.duplicated(["engine_id", "prefix_label"]).any():
        return None
    numeric = frame[
        ["true_rul", "prediction", "error", "absolute_error", "squared_error", "nasa_score_component"]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        return None
    if frame["gradient_enabled"].astype(bool).any():
        return None
    if (frame["backward_calls"].astype(int) != 0).any() or (frame["optimizer_steps"].astype(int) != 0).any():
        return None
    return frame


@torch.inference_mode()
def infer_one(
    row: pd.Series,
    dataset: a232.CausalPrefixDataset,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    path = Path(str(row["checkpoint"])).expanduser().resolve()
    payload = safe_load_checkpoint(path)
    model = a251b.make_model(str(row["method"]), cfg, int(row["model_seed"]))
    model.load_state_dict(payload["state"], strict=True)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictions = np.empty(len(dataset), dtype=np.float64)
    seen = np.zeros(len(dataset), dtype=bool)
    forward_calls = 0
    for anchor in RUL_ANCHORS:
        label = a232.endpoint_label(anchor)
        indices = [
            index for index, meta in enumerate(dataset.meta)
            if str(meta["prefix_label"]) == label
        ]
        if len(indices) * len(RUL_ANCHORS) != len(dataset):
            raise A252bError(f"anchor subset cardinality mismatch for {path}/anchor={anchor:g}")
        loader = a232.deterministic_loader(
            Subset(dataset, indices), int(cfg["batch_size"]), device
        )
        for x, locations in loader:
            output = model(x.to(device, non_blocking=device.type == "cuda"))
            forward_calls += 1
            if isinstance(output, tuple):
                output = output[0]
            values = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
            target_locations = locations.numpy().reshape(-1).astype(int)
            if len(values) != len(target_locations):
                raise A252bError(f"prediction cardinality mismatch: {path}")
            predictions[target_locations] = values
            seen[target_locations] = True
    if not seen.all() or not np.isfinite(predictions).all():
        raise A252bError(f"missing or non-finite predictions: {path}")
    records: list[dict[str, Any]] = []
    for index, (prediction, meta) in enumerate(zip(predictions, dataset.meta)):
        truth = float(meta["true_rul"])
        error = float(prediction - truth)
        nasa = (
            float(math.exp(error / 10.0) - 1.0)
            if error >= 0 else float(math.exp(-error / 13.0) - 1.0)
        )
        records.append({
            "experiment_id": EXPERIMENT_ID,
            "target_domain": str(row["target_domain"]),
            "model_seed": int(row["model_seed"]),
            "support_split_seed": int(row["support_split_seed"]),
            "shot": int(row["shot"]),
            "method": str(row["method"]),
            "architecture": str(row["architecture"]),
            "checkpoint": str(path),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "endpoint_index_in_dataset": int(index),
            **meta,
            "prediction": float(prediction),
            "error": error,
            "absolute_error": abs(error),
            "squared_error": error * error,
            "nasa_score_component": nasa,
            "model_uses_gat": bool(getattr(model, "use_gat", False)),
            "forward_calls_for_checkpoint": int(forward_calls),
            "gradient_enabled": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "new_predictor_training": False,
            "confirmation_used_for_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        })
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def run_inference(args: argparse.Namespace, result: Mapping[str, Any], context: dict[str, Any]) -> pd.DataFrame:
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    device: torch.device = context["device"]
    if device.type == "cuda":
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    frames: list[pd.DataFrame] = []
    reused = 0
    runs = context["validated_checkpoints"].sort_values(
        ["target_domain", "model_seed", "support_split_seed", "shot", "method"],
        kind="stable",
    ).reset_index(drop=True)
    for index, row in runs.iterrows():
        key = (str(row["target_domain"]), int(row["model_seed"]), int(row["support_split_seed"]))
        dataset = context["datasets"][key]
        csv_path, status_path = shard_paths(context["output_root"], row)
        prior = (
            validate_shard(
                csv_path, status_path, row, len(dataset),
                context["a252a"]["preregistration_hash"],
            )
            if args.resume else None
        )
        if prior is not None:
            frame = prior
            reused += 1
        else:
            try:
                frame = infer_one(row, dataset, context["cfg"], device)
            except Exception as exc:
                raise A252bError(f"inference failed for {shard_stem(row)}: {exc}") from exc
            atomic_frame(csv_path, frame)
            atomic_json(status_path, {
                "experiment_id": EXPERIMENT_ID,
                "complete": True, "passed": True,
                "run_key": shard_stem(row),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "preregistration_sha256": context["a252a"]["preregistration_hash"],
                "prediction_records": int(len(frame)),
                "prediction_sha256": sha256(csv_path),
                "gradient_enabled": False, "backward_calls": 0, "optimizer_steps": 0,
                "new_predictor_training": False,
                "official_test_files_accessed": False,
                "official_test_forward_run": False,
            })
        frames.append(frame)
        if (index + 1) % 8 == 0 or index + 1 == len(runs):
            print(
                f"[A25.2b] confirmation inference {index + 1:03d}/192 "
                f"reused_shards={reused} device={device}", flush=True,
            )
    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != int(result["expected_prediction_records"]):
        raise A252bError(
            f"prediction count={len(predictions)}, expected={result['expected_prediction_records']}"
        )
    keys = [
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "engine_id", "prefix_label",
    ]
    if predictions.duplicated(keys).any():
        raise A252bError("confirmation predictions contain duplicate keys")
    numeric = predictions[
        ["prediction", "true_rul", "error", "absolute_error", "squared_error", "nasa_score_component"]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise A252bError("confirmation predictions contain non-finite values")
    if predictions["gradient_enabled"].astype(bool).any():
        raise A252bError("gradient was enabled during confirmation evaluation")
    if (predictions["backward_calls"].astype(int) != 0).any():
        raise A252bError("a backward call occurred during confirmation evaluation")
    if (predictions["optimizer_steps"].astype(int) != 0).any():
        raise A252bError("an optimizer step occurred during confirmation evaluation")
    return predictions.sort_values(keys, kind="stable").reset_index(drop=True)


def build_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "architecture", "prefix_label",
    ]
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
    result = pd.DataFrame(rows)
    expected = len(DOMAINS) * len(MODEL_SEEDS) * len(SUPPORT_SPLIT_SEEDS) * len(SHOTS) * len(METHODS) * len(RUL_ANCHORS)
    if len(result) != expected:
        raise A252bError(f"run-metric row count={len(result)}, expected={expected}")
    return result


def build_paired(predictions: pd.DataFrame, architecture: str) -> pd.DataFrame:
    reference_method, candidate_method = PAIRS[architecture]
    identifiers = [
        "target_domain", "model_seed", "support_split_seed", "shot", "engine_id", "prefix_label",
    ]
    columns = identifiers + [
        "registered_rul_anchor", "rul_stage", "true_rul", "prediction", "error",
        "absolute_error", "squared_error", "nasa_score_component",
    ]
    reference = predictions.loc[predictions["method"] == reference_method, columns].copy()
    candidate = predictions.loc[predictions["method"] == candidate_method, columns].copy()
    if reference.duplicated(identifiers).any() or candidate.duplicated(identifiers).any():
        raise A252bError(f"duplicate paired prediction keys for {architecture}")
    paired = candidate.merge(
        reference, on=identifiers, how="inner", suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate) or len(reference) != len(candidate):
        raise A252bError(f"incomplete candidate/reference pairing for {architecture}")
    if not np.allclose(paired["true_rul_candidate"], paired["true_rul_reference"], atol=0, rtol=0):
        raise A252bError(f"paired true RUL differs for {architecture}")
    if not np.allclose(
        paired["registered_rul_anchor_candidate"],
        paired["registered_rul_anchor_reference"], atol=0, rtol=0,
    ):
        raise A252bError(f"paired anchors differ for {architecture}")
    paired.insert(0, "experiment_id", EXPERIMENT_ID)
    paired.insert(1, "architecture", architecture)
    paired.insert(2, "candidate_method", candidate_method)
    paired.insert(3, "reference_method", reference_method)
    return paired


def relative_metric(frame: pd.DataFrame, metric: str) -> float:
    if metric == "rmse":
        candidate = math.sqrt(float(frame["squared_error_candidate"].mean()))
        reference = math.sqrt(float(frame["squared_error_reference"].mean()))
    elif metric == "nasa_score":
        candidate = float(frame["nasa_score_component_candidate"].sum())
        reference = float(frame["nasa_score_component_reference"].sum())
    else:
        raise A252bError(f"unknown metric: {metric}")
    if not (math.isfinite(candidate) and math.isfinite(reference) and reference > 0):
        raise A252bError(f"invalid aggregate for {metric}")
    return candidate / reference - 1.0


def family_inference(
    paired: pd.DataFrame, architecture: str, seed_offset: int
) -> pd.DataFrame:
    reference_method, candidate_method = PAIRS[architecture]
    rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(RUL_ANCHORS):
        label = a232.endpoint_label(anchor)
        scoped = paired.loc[
            (paired["shot"] == PRIMARY_SHOT) & (paired["prefix_label"] == label)
        ].copy()
        expected_groups = len(DOMAINS) * len(MODEL_SEEDS) * len(SUPPORT_SPLIT_SEEDS)
        if scoped.groupby(["target_domain", "model_seed", "support_split_seed"]).ngroups != expected_groups:
            raise A252bError(f"primary K=5 pairing incomplete for {architecture}/{label}")
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            point = relative_metric(scoped, metric)
            try:
                samples = a234.hierarchical_bootstrap(
                    scoped, metric, BOOTSTRAP_REPETITIONS,
                    BOOTSTRAP_SEED + seed_offset + anchor_index * 1000 + metric_index * 100,
                )
            except Exception as exc:
                raise A252bError(f"hierarchical bootstrap failed for {architecture}/{label}/{metric}: {exc}") from exc
            p_superiority = float((1 + np.count_nonzero(samples >= 0.0)) / (len(samples) + 1))
            p_noninferiority = float(
                (1 + np.count_nonzero(samples >= NONINFERIORITY_MARGIN)) / (len(samples) + 1)
            )
            if metric == "rmse":
                candidate_value = math.sqrt(float(scoped["squared_error_candidate"].mean()))
                reference_value = math.sqrt(float(scoped["squared_error_reference"].mean()))
                wins = float(np.mean(scoped["absolute_error_candidate"] < scoped["absolute_error_reference"]))
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
                "hypothesis_role": "primary" if architecture == "no_graph" else "replication",
                "architecture": architecture,
                "candidate_method": candidate_method,
                "reference_method": reference_method,
                "shot": PRIMARY_SHOT,
                "prefix_label": label,
                "registered_rul_anchor": float(anchor),
                "rul_stage": str(scoped["rul_stage_candidate"].iloc[0]),
                "metric": metric,
                "n_paired_engine_records": int(len(scoped)),
                "candidate_value": candidate_value,
                "reference_value": reference_value,
                "relative_degradation": point,
                "relative_improvement_pct": -100.0 * point,
                "relative_ci95_low": float(np.quantile(samples, ALPHA / 2)),
                "relative_ci95_high": float(np.quantile(samples, 1 - ALPHA / 2)),
                "candidate_engine_win_rate": wins,
                "one_sided_bootstrap_tail_probability_superiority": p_superiority,
                "one_sided_bootstrap_tail_probability_noninferiority_3pct": p_noninferiority,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap_seed": BOOTSTRAP_SEED + seed_offset + anchor_index * 1000 + metric_index * 100,
                "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
            })
    result = pd.DataFrame(rows)
    if len(result) != 6:
        raise A252bError(f"{architecture} primary family does not contain six checks")
    try:
        result["holm_adjusted_p_superiority"] = a234.holm_adjust(
            result["one_sided_bootstrap_tail_probability_superiority"].tolist()
        )
    except Exception as exc:
        raise A252bError(f"Holm adjustment failed for {architecture}: {exc}") from exc
    result["holm_superiority_passed"] = (
        (result["holm_adjusted_p_superiority"] < ALPHA)
        & (result["relative_ci95_high"] < 0.0)
    )
    return result


def low_rul_safety(primary: pd.DataFrame, replication: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in (primary, replication):
        scoped = family.loc[family["registered_rul_anchor"] == 15.0].copy()
        if len(scoped) != 2 or set(scoped["metric"]) != set(PRIMARY_METRICS):
            raise A252bError("low-RUL safety family must contain RMSE and NASA score")
        try:
            scoped["holm_adjusted_p_noninferiority_3pct"] = a234.holm_adjust(
                scoped["one_sided_bootstrap_tail_probability_noninferiority_3pct"].tolist()
            )
        except Exception as exc:
            raise A252bError(f"low-RUL Holm adjustment failed: {exc}") from exc
        scoped["holm_noninferiority_3pct_passed"] = (
            (scoped["holm_adjusted_p_noninferiority_3pct"] < ALPHA)
            & (scoped["relative_ci95_high"] <= NONINFERIORITY_MARGIN)
        )
        frames.append(scoped)
    return pd.concat(frames, ignore_index=True)


def secondary_summary(paired_by_architecture: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for architecture, paired in paired_by_architecture.items():
        reference, candidate = PAIRS[architecture]
        for (shot, label), frame in paired.groupby(["shot", "prefix_label"], sort=True):
            for metric in PRIMARY_METRICS:
                point = relative_metric(frame, metric)
                rows.append({
                    "experiment_id": EXPERIMENT_ID,
                    "analysis_role": "primary" if int(shot) == PRIMARY_SHOT else "secondary",
                    "architecture": architecture,
                    "candidate_method": candidate,
                    "reference_method": reference,
                    "shot": int(shot),
                    "prefix_label": str(label),
                    "registered_rul_anchor": float(frame["registered_rul_anchor_candidate"].iloc[0]),
                    "metric": metric,
                    "n_paired_engine_records": int(len(frame)),
                    "relative_degradation": point,
                    "relative_improvement_pct": -100.0 * point,
                })
    return pd.DataFrame(rows)


def domain_summary(paired_by_architecture: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for architecture, paired in paired_by_architecture.items():
        reference, candidate = PAIRS[architecture]
        primary = paired.loc[paired["shot"] == PRIMARY_SHOT]
        for (domain, label), frame in primary.groupby(["target_domain", "prefix_label"], sort=True):
            for metric in PRIMARY_METRICS:
                point = relative_metric(frame, metric)
                rows.append({
                    "experiment_id": EXPERIMENT_ID,
                    "architecture": architecture,
                    "candidate_method": candidate,
                    "reference_method": reference,
                    "target_domain": str(domain),
                    "shot": PRIMARY_SHOT,
                    "prefix_label": str(label),
                    "registered_rul_anchor": float(frame["registered_rul_anchor_candidate"].iloc[0]),
                    "metric": metric,
                    "n_paired_engine_records": int(len(frame)),
                    "relative_degradation": point,
                    "relative_improvement_pct": -100.0 * point,
                })
    return pd.DataFrame(rows)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()}
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise A252bError(
                f"run lock exists: {self.path}; verify no A25.2b process is active, then remove "
                "only this lock before --resume"
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
    preflight_result: Mapping[str, Any], context: dict[str, Any], predictions: pd.DataFrame
) -> dict[str, Any]:
    root: Path = context["output_root"]
    run_metrics = build_run_metrics(predictions)
    paired = {architecture: build_paired(predictions, architecture) for architecture in PAIRS}
    primary = family_inference(paired["no_graph"], "no_graph", 0)
    replication = family_inference(paired["gnn"], "gnn", 10000)
    safety = low_rul_safety(primary, replication)
    secondary = secondary_summary(paired)
    domains = domain_summary(paired)
    primary_superiority = bool(primary["holm_superiority_passed"].all())
    replication_superiority = bool(replication["holm_superiority_passed"].all())
    primary_safety = bool(
        safety.loc[safety["architecture"] == "no_graph", "holm_noninferiority_3pct_passed"].all()
    )
    replication_safety = bool(
        safety.loc[safety["architecture"] == "gnn", "holm_noninferiority_3pct_passed"].all()
    )
    formal_supported = primary_superiority and primary_safety
    combined_paired = pd.concat(paired.values(), ignore_index=True)
    artifacts = {
        "experimentA25_2b_causal_anchor_predictions.csv": predictions,
        "experimentA25_2b_run_level_metrics.csv": run_metrics,
        "experimentA25_2b_paired_engine_metrics.csv": combined_paired,
        "experimentA25_2b_primary_no_graph_hierarchical_inference.csv": primary,
        "experimentA25_2b_gnn_replication_hierarchical_inference.csv": replication,
        "experimentA25_2b_low_rul_safety_gate.csv": safety,
        "experimentA25_2b_secondary_shot_summary.csv": secondary,
        "experimentA25_2b_domain_summary.csv": domains,
        "experimentA25_2b_prefix_coverage.csv": context["coverage"],
        "experimentA25_2b_normalizer_recomputation_audit.csv": context["normalizer_audit"],
        "experimentA25_2b_checkpoint_validation_inventory.csv": context["validated_checkpoints"],
    }
    for name, frame in artifacts.items():
        atomic_frame(root / name, frame)
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "execution_integrity_passed": True,
        "evaluation_only": True,
        "sealed_confirmation_opened": True,
        "confirmation_passes": 1,
        "registered_analysis_executed_without_branching": True,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "optimizer_construction": False,
        "evaluator_backward_calls": 0,
        "evaluator_optimizer_steps": 0,
        "primary_shot": PRIMARY_SHOT,
        "secondary_shots": [1, 2],
        "registered_rul_anchors": list(RUL_ANCHORS),
        "primary_metrics": list(PRIMARY_METRICS),
        "expected_checkpoint_evaluations": 192,
        "completed_checkpoint_evaluations": 192,
        "expected_prediction_records": int(preflight_result["expected_prediction_records"]),
        "completed_prediction_records": int(len(predictions)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "no_graph_primary_checks_passed": int(primary["holm_superiority_passed"].sum()),
        "no_graph_primary_checks_expected": 6,
        "no_graph_all_six_holm_superiority_checks_passed": primary_superiority,
        "no_graph_low_rul_safety_gate_passed": primary_safety,
        "gnn_replication_checks_passed": int(replication["holm_superiority_passed"].sum()),
        "gnn_replication_checks_expected": 6,
        "gnn_all_six_holm_superiority_checks_passed": replication_superiority,
        "gnn_low_rul_safety_gate_passed": replication_safety,
        "formal_efficacy_claim_supported": formal_supported,
        "formal_efficacy_claim": formal_supported,
        "normalizers_recomputed_from_frozen_source_roles": True,
        "all_stored_normalizers_match_recomputation": True,
        "checkpoint_hashes_match": True,
        "all_expected_units_complete": True,
        "all_metrics_finite": True,
        "model_input_uses_future_cycles": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A25.2b confirmed the preregistered no-graph Reptile family and low-RUL safety gate"
            if formal_supported else
            "A25.2b completed valid sealed confirmation, but the preregistered no-graph Reptile "
            "family and/or low-RUL safety gate did not fully pass"
        ),
        "interpretation_limit": (
            "The confirmation claim is limited to frozen A25.1b checkpoints, registered C-MAPSS "
            "training-file confirmation engines, K=5, and the 90/45/15 causal anchors. The GNN "
            "family is a separate replication result. This is not an official-test claim."
        ),
        "next_action": "analyze_A25_2b_confirmation_without_reusing_it_for_tuning",
    }
    decision_path = root / "experimentA25_2b_confirmation_decision.json"
    atomic_json(decision_path, decision)
    manifest_artifacts = list(artifacts) + [
        "experimentA25_2b_preflight.json",
        "experimentA25_2b_unseal_event.json",
        decision_path.name,
    ]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "a25_2a_preregistration_sha256": context["a252a"]["preregistration_hash"],
        "registered_analysis": {
            "primary_shot": PRIMARY_SHOT,
            "secondary_shots": [1, 2],
            "rul_anchors": list(RUL_ANCHORS),
            "metrics": list(PRIMARY_METRICS),
            "alpha": ALPHA,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
            "multiplicity_control": "Holm_within_each_six_check_family",
            "low_rul_noninferiority_margin": NONINFERIORITY_MARGIN,
        },
        "inputs": {
            "A25_2a_artifacts": context["a252a"]["artifact_hashes"],
            "A25_1a_contract_hashes": context["contract_hashes"],
            "config_sha256": sha256(context["config_path"]),
            "training_file_sha256": {
                domain: context["data_files"][domain]["sha256"] for domain in DOMAINS
            },
        },
        "artifacts": {name: sha256(root / name) for name in sorted(manifest_artifacts)},
        "prediction_shards_excluded_from_manifest": True,
        "new_predictor_training": False,
        "evaluator_backward_calls": 0,
        "evaluator_optimizer_steps": 0,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA25_2b_manifest.json", manifest)
    return decision


def completed_decision(root: Path) -> dict[str, Any] | None:
    path = root / "experimentA25_2b_confirmation_decision.json"
    if not path.exists():
        return None
    decision = read_json(path, label="existing A25.2b decision")
    if (
        decision.get("experiment_id") == EXPERIMENT_ID
        and decision.get("complete") is True
        and decision.get("passed") is True
    ):
        return decision
    raise A252bError("A25.2b output contains an invalid/incomplete final decision")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output_root = resolve(args.output_dir)
        existing = completed_decision(output_root)
        if existing is not None:
            if not args.resume:
                raise A252bError(
                    "A25.2b is already complete; use --resume to revalidate and return the result"
                )
            manifest = read_json(
                output_root / "experimentA25_2b_manifest.json", label="A25.2b manifest"
            )
            validate_hash_map(output_root, manifest.get("artifacts"), label="A25.2b artifacts")
            print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A25.2b] existing completed confirmation result revalidated", flush=True)
            return 0

        result, context = preflight(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        if args.dry_run:
            print(
                "[A25.2b] dry-run passed; 192 checkpoint tensors and evaluator flow were "
                "validated, confirmation observations remain unopened",
                flush=True,
            )
            return 0
        if output_root.exists() and any(output_root.iterdir()) and not args.resume:
            raise A252bError(
                f"non-empty A25.2b output has no completed decision: {output_root}; "
                "use --resume only after an interrupted A25.2b run"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        with RunLock(output_root / "experimentA25_2b_run.lock"):
            atomic_json(output_root / "experimentA25_2b_preflight.json", result)
            unseal_path = output_root / "experimentA25_2b_unseal_event.json"
            if unseal_path.exists():
                if not args.resume:
                    raise A252bError("confirmation was already unsealed; use --resume")
                unseal = read_json(unseal_path, label="A25.2b unseal event")
                if (
                    unseal.get("experiment_id") != EXPERIMENT_ID
                    or unseal.get("preregistration_sha256") != context["a252a"]["preregistration_hash"]
                    or unseal.get("confirmation_observations_opened") is not True
                ):
                    raise A252bError("existing A25.2b unseal event is incompatible")
            else:
                atomic_json(unseal_path, {
                    "experiment_id": EXPERIMENT_ID,
                    "unsealed_at_utc": utc_now(),
                    "authorization_token_sha256": hashlib.sha256(UNSEAL_TOKEN.encode()).hexdigest(),
                    "preregistration_sha256": context["a252a"]["preregistration_hash"],
                    "confirmation_observations_opened": True,
                    "confirmation_passes_authorized": 1,
                    "new_predictor_training": False,
                    "official_test_files_accessed": False,
                })
            context = prepare_confirmation_context(context)
            predictions = run_inference(args, result, context)
            decision = finalise(result, context, predictions)
        print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print("[A25.2b] completed one-time sealed confirmation evaluation", flush=True)
        return 0
    except A252bError as exc:
        print(f"[A25.2b] error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print(
            "[A25.2b] interrupted; if an unseal event exists, preserve the output and use --resume",
            file=sys.stderr, flush=True,
        )
        return 130
    except Exception as exc:
        print(f"[A25.2b] unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
