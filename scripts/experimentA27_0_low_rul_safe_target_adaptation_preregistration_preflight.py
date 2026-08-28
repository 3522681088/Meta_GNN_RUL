#!/usr/bin/env python3
"""A27.0: preregister one low-RUL-safe target-adaptation development test.

This program is deliberately a preflight.  It validates the completed A26.0
contract and A26.1 development diagnostics, freezes sixteen graph-enabled
``reptile_gnn_outer_half_target0`` checkpoints as paired starting states, and
registers exactly one intervention for A27.1:

    locked target loss
    + mean(1[true RUL <= 30] * relu(prediction - true RUL)^2)

The coefficient is fixed at 1.0.  No sweep, early stopping, architecture
change, source retraining, graph bypass, confirmation reuse, or official-test
access is permitted.  A27.0 hashes checkpoint bytes but never deserializes
checkpoint tensors, opens training observations, or runs a model forward pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "experimentA27_0"
SCRIPT_VERSION = "experimentA27_0_low_rul_safe_target_adaptation_preregistration_preflight_v1"
FREEZE_TOKEN = "A27.0_FREEZE"

A260_ID = "experimentA26_0"
A260_VERSION = "experimentA26_0_failure_diagnostic_contract_preflight_v1"
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
GNN_STATE_SCHEMA_SHA256 = "75abbe68a756fd3ccedd28a86a99460262d9515ceefadde3389831faf288a663"
GNN_STATE_TENSOR_NUMEL = 1_591_199
GNN_STATE_TENSOR_COUNT = 48
SOURCE_GRADIENT_UPDATES = 7_500
SOURCE_WINDOW_PRESENTATIONS = 480_000

CONTROL_ARM = "locked_target_loss"
CANDIDATE_ARM = "locked_target_loss_plus_low_rul_overprediction_penalty"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TRUE_TEXT = {"true", "1", "yes"}
FALSE_TEXT = {"false", "0", "no"}


class A270Error(RuntimeError):
    """Raised when the A27.0 development contract cannot be frozen safely."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


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
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise A270Error(f"refusing to write empty CSV: {path.name}")
    fields = list(materialized[0])
    for index, row in enumerate(materialized):
        if set(row) != set(fields):
            raise A270Error(f"row schema mismatch at index={index} for {path.name}")
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


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise A270Error(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A270Error(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A270Error(f"{label} must contain a JSON object: {path}")
    return value


def read_csv(path: Path, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise A270Error(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A270Error(f"cannot read {label}: {path}: {exc}") from exc
    if not fields or not rows:
        raise A270Error(f"{label} is empty: {path}")
    return fields, rows


def require_fields(container: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(container))
    if missing:
        raise A270Error(f"{label} lacks required fields: {missing}")


def require_columns(fields: Sequence[str], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise A270Error(f"{label} lacks required columns: {missing}")


def as_int(value: Any, *, label: str) -> int:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise A270Error(f"{label} must be an integer, observed {value!r}") from exc
    return number


def as_float(value: Any, *, label: str) -> float:
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise A270Error(f"{label} must be numeric, observed {value!r}") from exc
    if not math.isfinite(number):
        raise A270Error(f"{label} must be finite, observed {value!r}")
    return number


def as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise A270Error(f"{label} must be boolean, observed {value!r}")


def require_true(value: Any, *, label: str) -> None:
    if not as_bool(value, label=label):
        raise A270Error(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if as_bool(value, label=label):
        raise A270Error(f"{label} must be false")


def require_hash(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if HASH_RE.fullmatch(text) is None:
        raise A270Error(f"{label} is not a SHA256 digest: {value!r}")
    return text


def require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise A270Error(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
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
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA27_0_low_rul_safe_target_adaptation_preregistration_preflight"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--confirm-freeze",
        default="",
        help=f"Required for a formal freeze; pass exactly {FREEZE_TOKEN!r}.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and args.confirm_freeze != FREEZE_TOKEN:
        raise A270Error(f"formal preregistration requires --confirm-freeze {FREEZE_TOKEN}")
    return args


def validate_hash_map(root: Path, mapping: Any, *, label: str) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise A270Error(f"{label} must be a non-empty hash mapping")
    verified: dict[str, str] = {}
    for name, expected in sorted(mapping.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise A270Error(f"unsafe artifact name in {label}: {name!r}")
        expected_hash = require_hash(expected, label=f"{label} {name}")
        path = root / name
        if not path.is_file():
            raise A270Error(f"artifact in {label} is missing: {path}")
        observed = sha256(path)
        if observed != expected_hash:
            raise A270Error(
                f"artifact hash mismatch in {label}: {name}: "
                f"expected={expected_hash}, observed={observed}"
            )
        verified[name] = observed
    return verified


def validate_a26_0(args: argparse.Namespace) -> dict[str, str]:
    root = resolve(args.a26_0_output_dir)
    script = resolve(args.a26_0_script)
    if not root.is_dir():
        raise A270Error(f"A26.0 output directory is missing: {root}")
    if not script.is_file():
        raise A270Error(f"A26.0 script is missing: {script}")
    manifest_path = root / "experimentA26_0_manifest.json"
    decision_path = root / "experimentA26_0_confirmation_decision.json"
    manifest = read_json(manifest_path, label="A26.0 manifest")
    decision = read_json(decision_path, label="A26.0 decision")
    require_equal(manifest.get("experiment_id"), A260_ID, label="A26.0 manifest experiment")
    require_equal(manifest.get("script_version"), A260_VERSION, label="A26.0 script version")
    require_equal(decision.get("experiment_id"), A260_ID, label="A26.0 decision experiment")
    expected_script_hash = require_hash(manifest.get("script_sha256"), label="A26.0 script SHA256")
    require_equal(sha256(script), expected_script_hash, label="A26.0 script SHA256")
    verified = validate_hash_map(root, manifest.get("artifacts"), label="A26.0 manifest artifacts")
    require_fields(
        decision,
        {
            "complete",
            "passed",
            "preflight_only",
            "exploratory_only",
            "matched_compute_required",
            "same_architecture_and_initialization_required",
            "A25_2b_confirmation_frozen_read_only",
            "A25_2b_confirmation_reused_for_tuning",
            "A25_2b_confirmation_available_to_A26_1_candidate_selector",
            "formal_efficacy_claim",
            "new_predictor_training",
            "checkpoint_tensors_opened",
            "model_forward_run",
            "official_test_files_accessed",
            "official_test_forward_run",
        },
        label="A26.0 decision",
    )
    for field in (
        "complete",
        "passed",
        "preflight_only",
        "exploratory_only",
        "matched_compute_required",
        "same_architecture_and_initialization_required",
        "A25_2b_confirmation_frozen_read_only",
    ):
        require_true(decision[field], label=f"A26.0 {field}")
    for field in (
        "A25_2b_confirmation_reused_for_tuning",
        "A25_2b_confirmation_available_to_A26_1_candidate_selector",
        "formal_efficacy_claim",
        "new_predictor_training",
        "checkpoint_tensors_opened",
        "model_forward_run",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(decision[field], label=f"A26.0 {field}")
    return {
        "A26_0::script": expected_script_hash,
        "A26_0::manifest": sha256(manifest_path),
        **{f"A26_0::{name}": digest for name, digest in verified.items()},
    }


def locate_checkpoint(root: Path, row: Mapping[str, str]) -> Path:
    raw = Path(str(row["checkpoint"])).expanduser()
    if raw.is_absolute():
        direct = raw.resolve()
    else:
        direct = (project_root() / raw).resolve()
    if direct.is_file() and is_relative_to(direct, root):
        return direct
    name = raw.name
    if not name or name != Path(name).name:
        raise A270Error(f"unsafe checkpoint name: {raw}")
    worker = (
        f"{row['target_domain']}_mseed{as_int(row['model_seed'], label='model seed')}_"
        f"split{as_int(row['support_split_seed'], label='support split seed')}"
    )
    fallback = (root / "shards" / worker / name).resolve()
    if not is_relative_to(fallback, root) or not fallback.is_file():
        raise A270Error(f"checkpoint is missing inside A26.1 output: {fallback}")
    return fallback


def validate_selected_checkpoints(
    root: Path,
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_cells = {
        (domain, model_seed, split_seed)
        for domain in DOMAINS
        for model_seed in MODEL_SEEDS
        for split_seed in SUPPORT_SPLIT_SEEDS
    }
    selected = [row for row in rows if str(row["variant"]) == START_VARIANT]
    require_equal(len(selected), EXPECTED_WORKERS, label="A27.1 starting checkpoint count")
    observed_cells: set[tuple[str, int, int]] = set()
    frozen_rows: list[dict[str, Any]] = []
    frozen_hashes: dict[str, str] = {}
    for row in selected:
        domain = str(row["target_domain"])
        model_seed = as_int(row["model_seed"], label="checkpoint model seed")
        split_seed = as_int(row["support_split_seed"], label="checkpoint split seed")
        cell = (domain, model_seed, split_seed)
        if cell in observed_cells:
            raise A270Error(f"duplicate A27.1 starting checkpoint cell: {cell}")
        observed_cells.add(cell)
        label = f"start checkpoint {domain}/{model_seed}/{split_seed}"
        require_equal(as_int(row["shot"], label=label), PRIMARY_SHOT, label=f"{label} shot")
        require_equal(str(row["architecture"]), ARCHITECTURE, label=f"{label} architecture")
        require_equal(str(row["method"]), METHOD, label=f"{label} method")
        require_equal(as_int(row["target_epochs"], label=label), 0, label=f"{label} target epochs")
        require_equal(
            as_float(row["outer_lr_multiplier"], label=label),
            OUTER_LR_MULTIPLIER,
            label=f"{label} outer LR multiplier",
        )
        require_true(row["checkpoint_reload_passed"], label=f"{label} reload")
        require_equal(
            require_hash(row["state_schema_sha256"], label=f"{label} state schema"),
            GNN_STATE_SCHEMA_SHA256,
            label=f"{label} state schema",
        )
        require_equal(
            as_int(row["state_tensor_numel"], label=label),
            GNN_STATE_TENSOR_NUMEL,
            label=f"{label} state numel",
        )
        require_equal(
            as_int(row["state_tensor_count"], label=label),
            GNN_STATE_TENSOR_COUNT,
            label=f"{label} state tensor count",
        )
        require_equal(
            as_float(row["target_parameter_drift_l2"], label=label),
            0.0,
            label=f"{label} target drift",
        )
        for field, expected in (
            ("source_gradient_updates", SOURCE_GRADIENT_UPDATES),
            ("source_window_presentations", SOURCE_WINDOW_PRESENTATIONS),
            ("source_forward_calls", SOURCE_GRADIENT_UPDATES),
            ("source_backward_calls", SOURCE_GRADIENT_UPDATES),
            ("target_gradient_updates", 0),
            ("target_window_presentations", 0),
            ("target_forward_calls", 0),
            ("target_backward_calls", 0),
        ):
            require_equal(as_int(row[field], label=f"{label} {field}"), expected, label=f"{label} {field}")
        require_false(
            row["selection_development_used_for_training"],
            label=f"{label} selection used for training",
        )
        require_false(row["A25_2b_confirmation_used"], label=f"{label} A25.2b used")
        require_false(row["official_test_files_accessed"], label=f"{label} official test")
        expected_checkpoint_hash = require_hash(
            row["checkpoint_sha256"], label=f"{label} checkpoint SHA256"
        )
        checkpoint = locate_checkpoint(root, row)
        observed_checkpoint_hash = sha256(checkpoint)
        require_equal(
            observed_checkpoint_hash,
            expected_checkpoint_hash,
            label=f"{label} checkpoint SHA256",
        )
        source_state_hash = require_hash(row["source_state_sha256"], label=f"{label} source state")
        key = f"A26_1_checkpoint::{domain}::mseed{model_seed}::split{split_seed}"
        frozen_hashes[key] = observed_checkpoint_hash
        frozen_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_experiment_id": A261_ID,
                "target_domain": domain,
                "model_seed": model_seed,
                "support_split_seed": split_seed,
                "shot": PRIMARY_SHOT,
                "source_variant": START_VARIANT,
                "architecture": ARCHITECTURE,
                "method": METHOD,
                "graph_enabled": True,
                "source_outer_lr_multiplier": OUTER_LR_MULTIPLIER,
                "source_gradient_updates": SOURCE_GRADIENT_UPDATES,
                "source_window_presentations": SOURCE_WINDOW_PRESENTATIONS,
                "target_updates_before_A27_1": 0,
                "target_parameter_drift_l2_before_A27_1": 0.0,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed_checkpoint_hash,
                "source_state_sha256": source_state_hash,
                "state_schema_sha256": GNN_STATE_SCHEMA_SHA256,
                "state_tensor_numel": GNN_STATE_TENSOR_NUMEL,
                "state_tensor_count": GNN_STATE_TENSOR_COUNT,
                "checkpoint_tensor_opened_in_A27_0": False,
                "assigned_control_arm": CONTROL_ARM,
                "assigned_candidate_arm": CANDIDATE_ARM,
            }
        )
    require_equal(observed_cells, expected_cells, label="A27.1 starting checkpoint factorial")
    frozen_rows.sort(key=lambda row: (row["target_domain"], row["model_seed"], row["support_split_seed"]))
    return frozen_rows, frozen_hashes


def diagnostic_evidence_snapshot(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields, rows = read_csv(
        root / "experimentA26_1_global_diagnostic_summary.csv",
        label="A26.1 global diagnostic summary",
    )
    required = {
        "experiment_id",
        "analysis_role",
        "variant",
        "architecture",
        "registered_rul_anchor",
        "n_engines",
        "rmse",
        "nasa_score",
        "positive_error_q95",
        "overprediction_rate",
        "candidate_selected",
        "formal_efficacy_claim_allowed",
    }
    require_columns(fields, required, label="A26.1 global diagnostic summary")
    wanted = {
        (START_VARIANT, 15.0),
        (START_VARIANT, 90.0),
        ("reptile_gnn_outer_half_target10", 15.0),
        ("reptile_gnn_outer_half_target10", 45.0),
        ("reptile_gnn_outer_half_target10", 90.0),
        ("reptile_gnn_outer_half_target10_graph_bypass", 15.0),
        ("reptile_gnn_outer_half_target10_graph_bypass", 45.0),
        ("reptile_gnn_outer_half_target10_graph_bypass", 90.0),
    }
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for row in rows:
        if row["experiment_id"] != A261_ID:
            raise A270Error("A26.1 global summary contains a foreign experiment id")
        require_false(row["candidate_selected"], label="A26.1 global candidate selected")
        require_false(
            row["formal_efficacy_claim_allowed"],
            label="A26.1 global formal claim allowed",
        )
        variant = str(row["variant"])
        anchor = as_float(row["registered_rul_anchor"], label="diagnostic anchor")
        key = (variant, anchor)
        if key not in wanted:
            continue
        if key in seen:
            raise A270Error(f"duplicate global diagnostic evidence row: {key}")
        seen.add(key)
        require_equal(str(row["architecture"]), ARCHITECTURE, label=f"evidence {key} architecture")
        n_engines = as_int(row["n_engines"], label=f"evidence {key} engines")
        if n_engines < 1:
            raise A270Error(f"evidence {key} has no engines")
        selected.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_experiment_id": A261_ID,
                "evidence_role": "development_mechanism_rationale_only",
                "variant": variant,
                "architecture": ARCHITECTURE,
                "registered_rul_anchor": anchor,
                "n_engines": n_engines,
                "rmse": as_float(row["rmse"], label=f"evidence {key} RMSE"),
                "nasa_score": as_float(row["nasa_score"], label=f"evidence {key} NASA"),
                "positive_error_q95": as_float(
                    row["positive_error_q95"], label=f"evidence {key} q95"
                ),
                "overprediction_rate": as_float(
                    row["overprediction_rate"], label=f"evidence {key} overprediction"
                ),
                "candidate_selection_allowed_from_this_row": False,
                "formal_efficacy_claim_allowed": False,
            }
        )
    require_equal(seen, wanted, label="registered A26.1 mechanism evidence rows")
    selected.sort(key=lambda row: (row["variant"], -row["registered_rul_anchor"]))

    coverage_fields, coverage_rows = read_csv(
        root / "experimentA26_1_source_stage_coverage.csv",
        label="A26.1 source stage coverage",
    )
    require_columns(
        coverage_fields,
        {
            "experiment_id",
            "rul_stage",
            "row_observations",
            "target_domain_excluded",
            "confirmation_outcomes_used",
        },
        label="A26.1 source stage coverage",
    )
    totals: Counter[str] = Counter()
    for row in coverage_rows:
        require_equal(row["experiment_id"], A261_ID, label="coverage experiment id")
        require_true(row["target_domain_excluded"], label="coverage target excluded")
        require_false(row["confirmation_outcomes_used"], label="coverage confirmation used")
        count = as_int(row["row_observations"], label="coverage row observations")
        if count < 0:
            raise A270Error("source stage coverage has a negative observation count")
        totals[str(row["rul_stage"])] += count
    required_stages = {"high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30"}
    require_equal(set(totals), required_stages, label="source coverage stages")
    grand_total = sum(totals.values())
    if grand_total < 1:
        raise A270Error("source stage coverage has zero observations")
    coverage = {
        "source_row_observations": dict(sorted(totals.items())),
        "source_row_fractions": {
            stage: totals[stage] / grand_total for stage in sorted(totals)
        },
        "target_domain_excluded": True,
        "confirmation_outcomes_used": False,
    }
    return selected, coverage


def validate_a26_1(
    args: argparse.Namespace,
    a260_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    root = resolve(args.a26_1_output_dir)
    script = resolve(args.a26_1_script)
    config = resolve(args.config)
    if not root.is_dir():
        raise A270Error(f"A26.1 output directory is missing: {root}")
    if not script.is_file():
        raise A270Error(f"A26.1 script is missing: {script}")
    if not config.is_file():
        raise A270Error(f"configuration is missing: {config}")
    manifest_path = root / "experimentA26_1_manifest.json"
    decision_path = root / "experimentA26_1_confirmation_decision.json"
    manifest = read_json(manifest_path, label="A26.1 manifest")
    decision = read_json(decision_path, label="A26.1 decision")
    require_equal(manifest.get("experiment_id"), A261_ID, label="A26.1 manifest experiment")
    require_equal(manifest.get("script_version"), A261_VERSION, label="A26.1 script version")
    require_equal(decision.get("experiment_id"), A261_ID, label="A26.1 decision experiment")
    expected_script_hash = require_hash(manifest.get("script_sha256"), label="A26.1 script SHA256")
    require_equal(sha256(script), expected_script_hash, label="A26.1 script SHA256")
    verified = validate_hash_map(root, manifest.get("artifacts"), label="A26.1 manifest artifacts")
    require_fields(
        decision,
        {
            "complete",
            "passed",
            "execution_integrity_passed",
            "development_only",
            "exploratory_only",
            "one_factor_at_a_time",
            "primary_shot",
            "expected_worker_cells",
            "completed_worker_cells",
            "completed_run_level_records",
            "completed_paired_diagnostic_records",
            "completed_new_checkpoints",
            "matched_source_gradient_budget_passed",
            "matched_source_window_budget_passed",
            "same_architecture_initialization_passed",
            "checkpoint_reload_passed",
            "selection_development_used_for_training",
            "selection_development_used_for_evaluation",
            "A25_2b_confirmation_path_accepted_by_script",
            "A25_2b_confirmation_used_for_training",
            "A25_2b_confirmation_used_for_evaluation",
            "A25_2b_confirmation_used_for_candidate_selection",
            "candidate_selected",
            "policy_selected",
            "formal_efficacy_claim",
            "official_test_files_accessed",
            "official_test_forward_run",
        },
        label="A26.1 decision",
    )
    for field in (
        "complete",
        "passed",
        "execution_integrity_passed",
        "development_only",
        "exploratory_only",
        "one_factor_at_a_time",
        "matched_source_gradient_budget_passed",
        "matched_source_window_budget_passed",
        "same_architecture_initialization_passed",
        "checkpoint_reload_passed",
        "selection_development_used_for_evaluation",
    ):
        require_true(decision[field], label=f"A26.1 {field}")
    for field in (
        "selection_development_used_for_training",
        "A25_2b_confirmation_path_accepted_by_script",
        "A25_2b_confirmation_used_for_training",
        "A25_2b_confirmation_used_for_evaluation",
        "A25_2b_confirmation_used_for_candidate_selection",
        "candidate_selected",
        "policy_selected",
        "formal_efficacy_claim",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(decision[field], label=f"A26.1 {field}")
    require_equal(as_int(decision["primary_shot"], label="A26.1 primary shot"), 5, label="A26.1 primary shot")
    require_equal(
        as_int(decision["expected_worker_cells"], label="A26.1 expected workers"),
        EXPECTED_WORKERS,
        label="A26.1 expected workers",
    )
    require_equal(
        as_int(decision["completed_worker_cells"], label="A26.1 completed workers"),
        EXPECTED_WORKERS,
        label="A26.1 completed workers",
    )
    require_equal(
        as_int(decision["completed_run_level_records"], label="A26.1 run records"),
        624,
        label="A26.1 run records",
    )
    require_equal(
        as_int(decision["completed_paired_diagnostic_records"], label="A26.1 pairs"),
        528,
        label="A26.1 paired records",
    )
    require_equal(
        as_int(decision["completed_new_checkpoints"], label="A26.1 checkpoints"),
        96,
        label="A26.1 checkpoint count",
    )
    frozen_inputs = manifest.get("frozen_input_sha256")
    if not isinstance(frozen_inputs, dict) or not frozen_inputs:
        raise A270Error("A26.1 manifest lacks frozen_input_sha256")
    config_hash = require_hash(frozen_inputs.get("config"), label="A26.1 frozen config hash")
    require_equal(sha256(config), config_hash, label="configuration SHA256")
    expected_a260_script = a260_hashes.get("A26_0::script")
    require_equal(
        frozen_inputs.get("A26_0::A26_0_script"),
        expected_a260_script,
        label="A26.1 frozen A26.0 script hash",
    )
    inventory_fields, inventory_rows = read_csv(
        root / "experimentA26_1_checkpoint_inventory.csv",
        label="A26.1 checkpoint inventory",
    )
    require_columns(
        inventory_fields,
        {
            "experiment_id",
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "variant",
            "architecture",
            "method",
            "target_epochs",
            "outer_lr_multiplier",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_reload_passed",
            "state_schema_sha256",
            "state_tensor_numel",
            "state_tensor_count",
            "source_state_sha256",
            "target_parameter_drift_l2",
            "source_gradient_updates",
            "source_window_presentations",
            "source_forward_calls",
            "source_backward_calls",
            "target_gradient_updates",
            "target_window_presentations",
            "target_forward_calls",
            "target_backward_calls",
            "selection_development_used_for_training",
            "A25_2b_confirmation_used",
            "official_test_files_accessed",
        },
        label="A26.1 checkpoint inventory",
    )
    require_equal(len(inventory_rows), 96, label="A26.1 checkpoint inventory rows")
    if any(row["experiment_id"] != A261_ID for row in inventory_rows):
        raise A270Error("A26.1 checkpoint inventory contains a foreign experiment id")
    frozen_checkpoints, checkpoint_hashes = validate_selected_checkpoints(root, inventory_rows)
    evidence, source_coverage = diagnostic_evidence_snapshot(root)
    hashes = {
        "A26_1::script": expected_script_hash,
        "A26_1::manifest": sha256(manifest_path),
        "config": config_hash,
        **{f"A26_1::{name}": digest for name, digest in verified.items()},
        **checkpoint_hashes,
    }
    return frozen_checkpoints, evidence, source_coverage, hashes


def intervention_contract(config_hash: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "phase": "A27.1_single_candidate_exploratory_development",
        "registered_question": (
            "Does one fixed low-RUL overprediction penalty repair the RUL<=30 safety failure "
            "during K=5 target adaptation without materially degrading RUL45 or RUL90?"
        ),
        "starting_checkpoint_variant": START_VARIANT,
        "starting_checkpoint_count": EXPECTED_WORKERS,
        "architecture": ARCHITECTURE,
        "method": METHOD,
        "graph_enabled": True,
        "source_outer_lr_multiplier": OUTER_LR_MULTIPLIER,
        "source_retraining_in_A27_1": False,
        "target_shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "target_optimizer": {
            "name": "torch.optim.Adam",
            "learning_rate_source": "locked_config_inner_lr",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
        },
        "gradient_clip_max_norm": 5.0,
        "config_sha256": config_hash,
        "arms": [
            {
                "arm": CONTROL_ARM,
                "loss": "locked_rul_training_loss",
                "low_rul_penalty_lambda": 0.0,
            },
            {
                "arm": CANDIDATE_ARM,
                "loss": (
                    "locked_rul_training_loss + lambda * "
                    "mean(indicator[y<=30] * relu(prediction-y)^2)"
                ),
                "low_rul_penalty_lambda": PENALTY_LAMBDA,
            },
        ],
        "locked_rul_training_loss": (
            "MSE(prediction,y) plus the unchanged configured pairwise degradation-distance "
            "auxiliary term when pair_aux_weight>0"
        ),
        "penalty_threshold_true_rul": LOW_RUL_THRESHOLD,
        "penalty_direction": "positive_error_only",
        "penalty_reduction": "mean_over_full_batch_including_zero_contributions",
        "penalty_uses_labeled_target_support_only": True,
        "true_rul_inference_gating": False,
        "lambda_sweep_allowed": False,
        "alternative_thresholds_allowed": False,
        "early_stopping_allowed": False,
        "intermediate_checkpoint_selection_allowed": False,
        "final_epoch": TARGET_EPOCHS,
        "paired_target_batch_sequence_required": True,
        "paired_target_loader_seed_formula": "model_seed*1000000 + support_split_seed*100 + shot",
        "matched_target_gradient_updates_required": True,
        "matched_target_window_presentations_required": True,
        "matched_target_forward_backward_calls_required": True,
        "runtime_wall_time_and_peak_cuda_memory_required": True,
        "expected_worker_cells": EXPECTED_WORKERS,
        "expected_arms_per_worker": len(ARMS),
        "expected_final_checkpoints": EXPECTED_WORKERS * len(ARMS),
        "registered_rul_anchors": list(ANCHORS),
        "A25_2b_confirmation_path_accepted": False,
        "official_test_path_accepted": False,
        "formal_efficacy_claim_allowed": False,
    }


def data_role_contract() -> list[dict[str, Any]]:
    common = {
        "experiment_id": EXPERIMENT_ID,
        "formal_efficacy_claim_allowed": False,
    }
    return [
        {
            **common,
            "data_or_artifact": "A26.1 reptile_gnn_outer_half_target0 checkpoints",
            "A27_role": "paired_frozen_initialization",
            "training_access": "load_identical_state_into_both_arms",
            "evaluation_access": "not_an_outcome_dataset",
            "allowed_actions": "hash_verify;strict_load;target_adaptation_start",
            "prohibited_actions": "source_retraining;checkpoint_cherry_pick;graph_bypass",
        },
        {
            **common,
            "data_or_artifact": "A25.1a K5 target support_pool observations",
            "A27_role": "labeled_target_adaptation_training",
            "training_access": "allowed_for_both_paired_arms",
            "evaluation_access": "support_fit_diagnostics_only",
            "allowed_actions": "locked_10_epoch_target_adaptation;loss_computation",
            "prohibited_actions": "change_support_membership;change_shot;arm_specific_batch_order",
        },
        {
            **common,
            "data_or_artifact": "A25.1b selection observations",
            "A27_role": "exploratory_development_evaluation",
            "training_access": "forbidden",
            "evaluation_access": "one_complete_post_training_evaluation_of_both_arms",
            "allowed_actions": "apply_preregistered_A27_1_advancement_gates",
            "prohibited_actions": "gradient;early_stop;lambda_tuning;threshold_tuning;repeat_after_failure",
        },
        {
            **common,
            "data_or_artifact": "A26.1 development diagnostics",
            "A27_role": "frozen_mechanism_rationale",
            "training_access": "not_an_observation_source",
            "evaluation_access": "descriptive_reference_only",
            "allowed_actions": "justify_single_registered_intervention",
            "prohibited_actions": "post_freeze_branching;additional_candidate_selection",
        },
        {
            **common,
            "data_or_artifact": "A25.2b sealed-confirmation outcomes",
            "A27_role": "immutable_prior_confirmation",
            "training_access": "forbidden_and_path_not_accepted",
            "evaluation_access": "forbidden_and_path_not_accepted",
            "allowed_actions": "none_in_A27_0_or_A27_1",
            "prohibited_actions": "training;evaluation;tuning;selection;gate_modification",
        },
        {
            **common,
            "data_or_artifact": "official C-MAPSS test files",
            "A27_role": "sealed_external_final_evaluation",
            "training_access": "forbidden_and_path_not_accepted",
            "evaluation_access": "forbidden_and_path_not_accepted",
            "allowed_actions": "future_one_time_evaluation_only_after_separate_preregistration",
            "prohibited_actions": "open;parse;forward;explore;tune;select",
        },
    ]


def statistical_plan(contract_hash: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "intervention_contract_canonical_sha256": contract_hash,
        "analysis_status": "preregistered_exploratory_development_not_confirmatory",
        "paired_unit": "target_domain_model_seed_support_split_seed",
        "paired_worker_cells": EXPECTED_WORKERS,
        "reference_arm": CONTROL_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "lower_is_better_metrics": ["rmse", "nasa_score", "positive_error_q95"],
        "registered_rul_anchors": list(ANCHORS),
        "primary_safety_anchor": 15.0,
        "primary_safety_metrics": ["rmse", "nasa_score", "positive_error_q95"],
        "low_rul_worker_gate": {
            "rule": "candidate strictly lower than control",
            "minimum_improving_workers_per_metric": 12,
            "denominator": EXPECTED_WORKERS,
            "applies_separately_to": ["rmse", "nasa_score", "positive_error_q95"],
        },
        "low_rul_pooled_gate": {
            "maximum_candidate_to_control_relative_change": -0.10,
            "applies_separately_to": ["rmse", "nasa_score", "positive_error_q95"],
        },
        "low_rul_overprediction_gate": {
            "pooled_candidate_minus_control_maximum": 0.0,
            "metric": "overprediction_rate",
        },
        "mid_high_guardrail": {
            "anchors": [45.0, 90.0],
            "metrics": ["rmse", "nasa_score"],
            "maximum_pooled_relative_deterioration": 0.05,
        },
        "worker_joint_guardrail": {
            "anchors": [90.0, 45.0, 15.0],
            "minimum_workers_without_simultaneous_rmse_and_nasa_deterioration": 12,
            "denominator": EXPECTED_WORKERS,
        },
        "domain_heterogeneity_guardrail": {
            "domains": list(DOMAINS),
            "anchors": list(ANCHORS),
            "metrics": ["rmse", "nasa_score"],
            "maximum_domain_median_relative_deterioration": 0.10,
        },
        "all_advancement_gates_conjunctive": True,
        "missing_or_nonfinite_metric_action": "automatic_gate_failure",
        "incomplete_worker_action": "no_analysis_and_no_candidate_advancement",
        "success_action": (
            "freeze exactly the candidate arm as one exploratory candidate, then preregister a "
            "genuinely external or official one-time evaluation before opening it"
        ),
        "failure_action": (
            "abandon the Reptile-GNN low-RUL repair; do not tune lambda, threshold, epochs, "
            "graph mode, or gates using these outcomes"
        ),
        "lambda_retuning_after_A27_1": False,
        "A25_2b_reuse": False,
        "official_test_access": False,
        "formal_efficacy_claim_from_A27_1": False,
    }


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "experimentA27_0.run.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise A270Error(f"another or interrupted A27.0 run owns lock: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at_utc": utc_now()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def validate_output_paths(args: argparse.Namespace) -> None:
    output = resolve(args.output_dir)
    inputs = {resolve(args.a26_0_output_dir), resolve(args.a26_1_output_dir)}
    if output in inputs:
        raise A270Error("A27.0 output directory must differ from every input directory")
    for source in inputs:
        if is_relative_to(output, source) or is_relative_to(source, output):
            raise A270Error("A27.0 output directory must not overlap an input directory")


def expected_artifact_names() -> set[str]:
    return {
        "experimentA27_0_candidate_checkpoint_inventory.csv",
        "experimentA27_0_development_evidence_snapshot.csv",
        "experimentA27_0_data_role_contract.csv",
        "experimentA27_0_intervention_contract.json",
        "experimentA27_0_statistical_analysis_plan.json",
        "experimentA27_0_input_integrity.json",
        "experimentA27_0_confirmation_decision.json",
    }


def validate_existing_output(
    root: Path,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = root / "experimentA27_0_manifest.json"
    decision_path = root / "experimentA27_0_confirmation_decision.json"
    if not manifest_path.is_file() or not decision_path.is_file():
        raise A270Error("A27.0 output is partial; use a new output directory after inspection")
    manifest = read_json(manifest_path, label="existing A27.0 manifest")
    decision = read_json(decision_path, label="existing A27.0 decision")
    require_equal(manifest.get("experiment_id"), EXPERIMENT_ID, label="existing experiment id")
    require_equal(manifest.get("script_version"), SCRIPT_VERSION, label="existing script version")
    require_equal(
        manifest.get("script_sha256"),
        sha256(Path(__file__).resolve()),
        label="existing script SHA256",
    )
    require_equal(
        manifest.get("frozen_input_sha256"),
        dict(sorted(input_hashes.items())),
        label="existing frozen input hashes",
    )
    artifacts = validate_hash_map(root, manifest.get("artifacts"), label="existing A27.0 artifacts")
    require_equal(set(artifacts), expected_artifact_names(), label="existing artifact set")
    for field in (
        "complete",
        "passed",
        "preflight_only",
        "preregistered",
        "intervention_frozen",
        "starting_checkpoint_set_frozen",
    ):
        require_true(decision.get(field), label=f"existing A27.0 {field}")
    for field in (
        "new_predictor_training",
        "checkpoint_tensors_opened",
        "model_forward_run",
        "A25_2b_confirmation_path_accepted",
        "A25_2b_confirmation_used",
        "official_test_files_accessed",
        "official_test_forward_run",
        "formal_efficacy_claim",
    ):
        require_false(decision.get(field), label=f"existing A27.0 {field}")
    return decision


def build_preflight_summary(
    args: argparse.Namespace,
    frozen_checkpoints: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    source_coverage: Mapping[str, Any],
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": bool(args.dry_run),
        "registered_question": contract["registered_question"],
        "starting_checkpoint_variant": START_VARIANT,
        "frozen_starting_checkpoints": len(frozen_checkpoints),
        "architecture": ARCHITECTURE,
        "graph_enabled": True,
        "method": METHOD,
        "target_shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "arms": list(ARMS),
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "penalty_lambda": PENALTY_LAMBDA,
        "lambda_sweep_allowed": False,
        "expected_A27_1_worker_cells": EXPECTED_WORKERS,
        "expected_A27_1_final_checkpoints": EXPECTED_WORKERS * len(ARMS),
        "development_evidence_rows_frozen": len(evidence),
        "source_stage_row_fractions": source_coverage["source_row_fractions"],
        "statistical_plan_sha256": canonical_sha256(plan),
        "A26_1_development_diagnostics_read": True,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "A25_2b_confirmation_path_accepted": False,
        "A25_2b_confirmation_used": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": True,
    }


def write_outputs(
    root: Path,
    frozen_checkpoints: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    source_coverage: Mapping[str, Any],
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    atomic_csv(root / "experimentA27_0_candidate_checkpoint_inventory.csv", frozen_checkpoints)
    atomic_csv(root / "experimentA27_0_development_evidence_snapshot.csv", evidence)
    atomic_csv(root / "experimentA27_0_data_role_contract.csv", data_role_contract())
    atomic_json(root / "experimentA27_0_intervention_contract.json", contract)
    atomic_json(root / "experimentA27_0_statistical_analysis_plan.json", plan)
    atomic_json(
        root / "experimentA27_0_input_integrity.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "verified_at_utc": utc_now(),
            "input_sha256": dict(sorted(input_hashes.items())),
            "source_stage_coverage": source_coverage,
            "A25_2b_confirmation_path_accepted": False,
            "official_test_path_accepted": False,
            "checkpoint_tensors_opened": False,
        },
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "preflight_only": True,
        "preregistered": True,
        "exploratory_development_only": True,
        "intervention_frozen": True,
        "statistical_plan_frozen": True,
        "data_role_contract_frozen": True,
        "starting_checkpoint_set_frozen": True,
        "starting_checkpoint_variant": START_VARIANT,
        "frozen_starting_checkpoints": len(frozen_checkpoints),
        "architecture": ARCHITECTURE,
        "graph_enabled": True,
        "method": METHOD,
        "target_shot": PRIMARY_SHOT,
        "target_epochs": TARGET_EPOCHS,
        "control_arm": CONTROL_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "penalty_lambda": PENALTY_LAMBDA,
        "lambda_sweep_allowed": False,
        "expected_A27_1_worker_cells": EXPECTED_WORKERS,
        "expected_A27_1_final_checkpoints": EXPECTED_WORKERS * len(ARMS),
        "candidate_proposal_frozen": True,
        "candidate_selected": False,
        "policy_selected": False,
        "A26_1_development_diagnostics_read": True,
        "A25_2b_confirmation_path_accepted": False,
        "A25_2b_confirmation_used": False,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A27.0 froze one low-RUL overprediction-penalty intervention and its paired "
            "development advancement gates before A27.1 training"
        ),
        "interpretation_limit": (
            "A27.0 and A27.1 are exploratory development stages. A27.1 cannot revise A25.2b, "
            "support an efficacy/deployment claim, or authorize official-test access."
        ),
        "next_action": "implement_A27_1_preregistered_paired_low_rul_safe_target_adaptation_development",
    }
    decision_path = root / "experimentA27_0_confirmation_decision.json"
    atomic_json(decision_path, decision)
    artifacts = {
        name: sha256(root / name)
        for name in sorted(expected_artifact_names())
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "frozen_input_sha256": dict(sorted(input_hashes.items())),
        "artifacts": artifacts,
        "preflight_only": True,
        "preregistered": True,
        "candidate_proposal_frozen": True,
        "candidate_selected": False,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "A25_2b_confirmation_path_accepted": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(root / "experimentA27_0_manifest.json", manifest)
    return decision


def run(args: argparse.Namespace) -> None:
    validate_output_paths(args)
    a260_hashes = validate_a26_0(args)
    frozen_checkpoints, evidence, source_coverage, a261_hashes = validate_a26_1(args, a260_hashes)
    input_hashes = dict(sorted({**a260_hashes, **a261_hashes}.items()))
    contract = intervention_contract(input_hashes["config"])
    plan = statistical_plan(canonical_sha256(contract))
    summary = build_preflight_summary(
        args,
        frozen_checkpoints,
        evidence,
        source_coverage,
        contract,
        plan,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print(
            "[A27.0] dry-run passed; one intervention and all inputs are compatible, "
            "no checkpoint tensor was opened and no predictor was trained",
            flush=True,
        )
        return

    root = resolve(args.output_dir)
    manifest_path = root / "experimentA27_0_manifest.json"
    decision_path = root / "experimentA27_0_confirmation_decision.json"
    if manifest_path.exists() or decision_path.exists():
        if not args.resume:
            raise A270Error("complete A27.0 output already exists; pass --resume to verify and return it")
        decision = validate_existing_output(root, input_hashes)
        print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        print("[A27.0] existing frozen preregistration verified; no files were changed", flush=True)
        return
    if root.exists() and any(root.iterdir()):
        raise A270Error("A27.0 output directory is non-empty but incomplete; use a new directory")
    lock = acquire_lock(root)
    try:
        decision = write_outputs(
            root,
            frozen_checkpoints,
            evidence,
            source_coverage,
            contract,
            plan,
            input_hashes,
        )
    finally:
        release_lock(lock)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(
        "[A27.0] completed low-RUL-safe target-adaptation preregistration; "
        "A25.2b and official test remain unavailable",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except KeyboardInterrupt:
        print("[A27.0] interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[A27.0] error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
