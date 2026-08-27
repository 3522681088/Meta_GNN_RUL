#!/usr/bin/env python3
"""A26.0: freeze a failure-diagnostic contract after A25.2b confirmation.

This program is deliberately a preflight.  It verifies the completed A25.2b
sealed-confirmation result, records the confirmed stage-dependent failure
pattern, and registers what may and may not be investigated in A26.1.

It does not train a predictor, load a checkpoint tensor, run model inference,
read official C-MAPSS test files, or use A25.2b confirmation outcomes to choose
hyperparameters.  A25.2b is treated as immutable, read-only scientific evidence.
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


EXPERIMENT_ID = "experimentA26_0"
SCRIPT_VERSION = "experimentA26_0_failure_diagnostic_contract_preflight_v1"
FREEZE_TOKEN = "A26.0_FREEZE"
SOURCE_EXPERIMENT_ID = "experimentA25_2b"
SOURCE_SCRIPT_VERSION = "experimentA25_2b_frozen_checkpoint_sealed_confirmation_evaluator_v1"
PREREGISTRATION_SHA256 = "a3d15d35ffeb194e385b775f71c3415838304436e5d1bcea21d4172ebc9bdf53"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
METHODS = (
    "ordinary_no_graph_pft",
    "reptile_meta_no_graph",
    "ordinary_gnn_pft",
    "reptile_meta_gnn",
)
ARCHITECTURES = ("no_graph", "gnn")
MODEL_SEEDS = (140, 141)
SUPPORT_SPLIT_SEEDS = (7501, 7502)
SHOTS = (1, 2, 5)
PRIMARY_SHOT = 5
SECONDARY_SHOTS = (1, 2)
ANCHORS = (90.0, 45.0, 15.0)
METRICS = ("rmse", "nasa_score")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TRUE_TEXT = {"true", "1", "yes"}
FALSE_TEXT = {"false", "0", "no"}


class A260Error(RuntimeError):
    """Raised when the A26.0 diagnostic contract cannot be frozen safely."""


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


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise A260Error(f"refusing to write empty CSV: {path.name}")
    fields = list(materialized[0])
    for index, row in enumerate(materialized):
        if set(row) != set(fields):
            raise A260Error(f"row schema mismatch at index={index} for {path.name}")
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
        raise A260Error(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A260Error(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A260Error(f"{label} must contain a JSON object: {path}")
    return value


def read_csv(path: Path, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise A260Error(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A260Error(f"cannot read {label}: {path}: {exc}") from exc
    if not fields or not rows:
        raise A260Error(f"{label} is empty: {path}")
    return fields, rows


def require_fields(container: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(container))
    if missing:
        raise A260Error(f"{label} lacks required fields: {missing}")


def require_columns(fields: Sequence[str], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise A260Error(f"{label} lacks required columns: {missing}")


def as_int(value: Any, *, label: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise A260Error(f"{label} must be an integer, observed {value!r}") from exc


def as_float(value: Any, *, label: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise A260Error(f"{label} must be numeric, observed {value!r}") from exc
    if not math.isfinite(number):
        raise A260Error(f"{label} must be finite, observed {value!r}")
    return number


def as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise A260Error(f"{label} must be boolean, observed {value!r}")


def require_hash(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if HASH_RE.fullmatch(text) is None:
        raise A260Error(f"{label} is not a SHA256 digest: {value!r}")
    return text


def require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise A260Error(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def require_true(value: Any, *, label: str) -> None:
    if not as_bool(value, label=label):
        raise A260Error(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if as_bool(value, label=label):
        raise A260Error(f"{label} must be false")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate A25.2b and freeze an A26.1 exploratory failure-diagnostic contract; "
            "no training, checkpoint loading, inference, or official-test access."
        )
    )
    parser.add_argument(
        "--a25-2b-output-dir",
        type=Path,
        default=Path("outputs/experimentA25_2b_frozen_checkpoint_sealed_confirmation_evaluator"),
    )
    parser.add_argument(
        "--a25-2b-script",
        type=Path,
        default=Path("scripts/experimentA25_2b_frozen_checkpoint_sealed_confirmation_evaluator.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA26_0_failure_diagnostic_contract_preflight"),
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
        raise A260Error(
            f"formal freeze requires --confirm-freeze {FREEZE_TOKEN}; run --dry-run first"
        )
    return args


def validate_paths(source_dir: Path, source_script: Path, output_dir: Path) -> None:
    if not source_dir.is_dir():
        raise A260Error(f"A25.2b output directory is missing: {source_dir}")
    if not source_script.is_file():
        raise A260Error(f"A25.2b evaluator script is missing: {source_script}")
    if source_dir == output_dir or is_relative_to(output_dir, source_dir):
        raise A260Error("A26.0 output directory must be outside the immutable A25.2b directory")
    if is_relative_to(source_dir, output_dir):
        raise A260Error("A25.2b input directory must not be inside the A26.0 output directory")


def validate_manifest(source_dir: Path, source_script: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = source_dir / "experimentA25_2b_manifest.json"
    manifest = read_json(path, label="A25.2b manifest")
    require_fields(
        manifest,
        (
            "experiment_id",
            "script_version",
            "script_sha256",
            "a25_2a_preregistration_sha256",
            "artifacts",
            "registered_analysis",
            "new_predictor_training",
            "evaluator_backward_calls",
            "evaluator_optimizer_steps",
            "official_test_files_accessed",
            "official_test_forward_run",
        ),
        label="A25.2b manifest",
    )
    require_equal(manifest["experiment_id"], SOURCE_EXPERIMENT_ID, label="manifest experiment_id")
    require_equal(manifest["script_version"], SOURCE_SCRIPT_VERSION, label="manifest script_version")
    expected_script_hash = require_hash(manifest["script_sha256"], label="manifest script_sha256")
    require_equal(sha256(source_script), expected_script_hash, label="A25.2b evaluator script SHA256")
    require_equal(
        manifest["a25_2a_preregistration_sha256"],
        PREREGISTRATION_SHA256,
        label="A25.2a preregistration SHA256",
    )
    require_false(manifest["new_predictor_training"], label="manifest new_predictor_training")
    require_equal(as_int(manifest["evaluator_backward_calls"], label="backward calls"), 0, label="backward calls")
    require_equal(as_int(manifest["evaluator_optimizer_steps"], label="optimizer steps"), 0, label="optimizer steps")
    require_false(manifest["official_test_files_accessed"], label="manifest official_test_files_accessed")
    require_false(manifest["official_test_forward_run"], label="manifest official_test_forward_run")

    registered = manifest["registered_analysis"]
    if not isinstance(registered, dict):
        raise A260Error("manifest registered_analysis must be an object")
    require_equal(as_int(registered.get("primary_shot"), label="primary shot"), PRIMARY_SHOT, label="primary shot")
    require_equal(tuple(registered.get("secondary_shots", [])), SECONDARY_SHOTS, label="secondary shots")
    require_equal(tuple(float(x) for x in registered.get("rul_anchors", [])), ANCHORS, label="RUL anchors")
    require_equal(tuple(registered.get("metrics", [])), METRICS, label="registered metrics")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise A260Error("manifest artifacts must be a non-empty object")
    verified: dict[str, str] = {}
    for name, expected in sorted(artifacts.items()):
        if Path(name).name != name:
            raise A260Error(f"unsafe artifact name in A25.2b manifest: {name!r}")
        expected_hash = require_hash(expected, label=f"manifest artifact {name}")
        artifact = source_dir / name
        if not artifact.is_file():
            raise A260Error(f"A25.2b artifact is missing: {artifact}")
        actual_hash = sha256(artifact)
        require_equal(actual_hash, expected_hash, label=f"A25.2b artifact SHA256 {name}")
        verified[name] = actual_hash
    return manifest, verified


def validate_decision(source_dir: Path) -> dict[str, Any]:
    decision = read_json(
        source_dir / "experimentA25_2b_confirmation_decision.json",
        label="A25.2b confirmation decision",
    )
    required_true = (
        "complete",
        "passed",
        "execution_integrity_passed",
        "evaluation_only",
        "sealed_confirmation_opened",
        "registered_analysis_executed_without_branching",
        "normalizers_recomputed_from_frozen_source_roles",
        "all_stored_normalizers_match_recomputation",
        "checkpoint_hashes_match",
        "all_expected_units_complete",
        "all_metrics_finite",
    )
    required_false = (
        "new_predictor_training",
        "target_adaptation",
        "policy_selection_or_tuning",
        "optimizer_construction",
        "formal_efficacy_claim_supported",
        "formal_efficacy_claim",
        "no_graph_all_six_holm_superiority_checks_passed",
        "no_graph_low_rul_safety_gate_passed",
        "gnn_all_six_holm_superiority_checks_passed",
        "gnn_low_rul_safety_gate_passed",
        "model_input_uses_future_cycles",
        "official_test_files_accessed",
        "official_test_forward_run",
    )
    require_fields(
        decision,
        (
            "experiment_id",
            "confirmation_passes",
            "primary_shot",
            "secondary_shots",
            "registered_rul_anchors",
            "primary_metrics",
            "expected_checkpoint_evaluations",
            "completed_checkpoint_evaluations",
            "expected_prediction_records",
            "completed_prediction_records",
            "evaluator_backward_calls",
            "evaluator_optimizer_steps",
            "no_graph_primary_checks_passed",
            "no_graph_primary_checks_expected",
            "gnn_replication_checks_passed",
            "gnn_replication_checks_expected",
            *required_true,
            *required_false,
        ),
        label="A25.2b confirmation decision",
    )
    require_equal(decision["experiment_id"], SOURCE_EXPERIMENT_ID, label="decision experiment_id")
    for field in required_true:
        require_true(decision[field], label=f"decision {field}")
    for field in required_false:
        require_false(decision[field], label=f"decision {field}")
    expected_pairs = (
        ("confirmation_passes", 1),
        ("primary_shot", PRIMARY_SHOT),
        ("expected_checkpoint_evaluations", 192),
        ("completed_checkpoint_evaluations", 192),
        ("expected_prediction_records", 78768),
        ("completed_prediction_records", 78768),
        ("evaluator_backward_calls", 0),
        ("evaluator_optimizer_steps", 0),
        ("no_graph_primary_checks_passed", 2),
        ("no_graph_primary_checks_expected", 6),
        ("gnn_replication_checks_passed", 0),
        ("gnn_replication_checks_expected", 6),
    )
    for field, expected in expected_pairs:
        require_equal(as_int(decision[field], label=field), expected, label=f"decision {field}")
    require_equal(tuple(decision["secondary_shots"]), SECONDARY_SHOTS, label="decision secondary_shots")
    require_equal(tuple(float(x) for x in decision["registered_rul_anchors"]), ANCHORS, label="decision anchors")
    require_equal(tuple(decision["primary_metrics"]), METRICS, label="decision metrics")
    return decision


def validate_preflight_and_unseal(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = read_json(source_dir / "experimentA25_2b_preflight.json", label="A25.2b preflight")
    unseal = read_json(source_dir / "experimentA25_2b_unseal_event.json", label="A25.2b unseal event")
    require_equal(preflight.get("experiment_id"), SOURCE_EXPERIMENT_ID, label="preflight experiment_id")
    require_equal(preflight.get("script_version"), SOURCE_SCRIPT_VERSION, label="preflight script_version")
    require_equal(preflight.get("a25_2a_preregistration_sha256"), PREREGISTRATION_SHA256, label="preflight preregistration hash")
    for field in (
        "passed",
        "checkpoint_hashes_verified",
        "checkpoint_state_schema_verified",
        "normalizers_schema_validated_pre_unseal",
        "training_file_hashes_verified_without_row_parsing",
    ):
        require_true(preflight.get(field), label=f"preflight {field}")
    for field in (
        "new_predictor_training",
        "optimizer_construction",
        "official_test_files_accessed",
        "official_test_forward_run",
    ):
        require_false(preflight.get(field), label=f"preflight {field}")
    require_equal(as_int(preflight.get("backward_calls"), label="preflight backward_calls"), 0, label="preflight backward_calls")
    require_equal(as_int(preflight.get("validated_checkpoint_runs"), label="validated checkpoints"), 192, label="validated checkpoints")
    require_equal(as_int(preflight.get("validated_normalizer_files"), label="validated normalizers"), 16, label="validated normalizers")

    require_equal(unseal.get("experiment_id"), SOURCE_EXPERIMENT_ID, label="unseal experiment_id")
    require_equal(unseal.get("preregistration_sha256"), PREREGISTRATION_SHA256, label="unseal preregistration hash")
    require_true(unseal.get("confirmation_observations_opened"), label="unseal confirmation_observations_opened")
    require_equal(as_int(unseal.get("confirmation_passes_authorized"), label="authorized passes"), 1, label="authorized passes")
    require_false(unseal.get("new_predictor_training"), label="unseal new_predictor_training")
    require_false(unseal.get("official_test_files_accessed"), label="unseal official_test_files_accessed")
    require_hash(unseal.get("authorization_token_sha256"), label="unseal authorization token SHA256")
    return preflight, unseal


INFERENCE_COLUMNS = (
    "experiment_id",
    "hypothesis_role",
    "architecture",
    "candidate_method",
    "reference_method",
    "shot",
    "registered_rul_anchor",
    "rul_stage",
    "metric",
    "n_paired_engine_records",
    "candidate_value",
    "reference_value",
    "relative_degradation",
    "relative_improvement_pct",
    "relative_ci95_low",
    "relative_ci95_high",
    "candidate_engine_win_rate",
    "holm_adjusted_p_superiority",
    "holm_superiority_passed",
)


def validate_inference_family(
    path: Path,
    *,
    architecture: str,
    hypothesis_role: str,
    expected_passes: int,
) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label=f"A25.2b {architecture} inference")
    require_columns(fields, INFERENCE_COLUMNS, label=path.name)
    require_equal(len(rows), 6, label=f"{path.name} row count")
    seen: set[tuple[float, str]] = set()
    passes = 0
    for index, row in enumerate(rows):
        label = f"{path.name} row {index}"
        require_equal(row["experiment_id"], SOURCE_EXPERIMENT_ID, label=f"{label} experiment_id")
        require_equal(row["architecture"], architecture, label=f"{label} architecture")
        require_equal(row["hypothesis_role"], hypothesis_role, label=f"{label} hypothesis_role")
        require_equal(as_int(row["shot"], label=f"{label} shot"), PRIMARY_SHOT, label=f"{label} shot")
        anchor = as_float(row["registered_rul_anchor"], label=f"{label} anchor")
        metric = row["metric"]
        if anchor not in ANCHORS or metric not in METRICS:
            raise A260Error(f"{label} has unregistered anchor/metric: {(anchor, metric)!r}")
        key = (anchor, metric)
        if key in seen:
            raise A260Error(f"duplicate anchor/metric in {path.name}: {key}")
        seen.add(key)
        for numeric in (
            "candidate_value",
            "reference_value",
            "relative_degradation",
            "relative_improvement_pct",
            "relative_ci95_low",
            "relative_ci95_high",
            "candidate_engine_win_rate",
            "holm_adjusted_p_superiority",
        ):
            as_float(row[numeric], label=f"{label} {numeric}")
        require_equal(as_int(row["n_paired_engine_records"], label=f"{label} paired records"), 2188, label=f"{label} paired records")
        passed = as_bool(row["holm_superiority_passed"], label=f"{label} pass")
        passes += int(passed)
        if architecture == "no_graph":
            expected = anchor == 90.0
            require_equal(passed, expected, label=f"{label} locked pass pattern")
        else:
            require_false(passed, label=f"{label} locked pass pattern")
    require_equal(seen, {(a, m) for a in ANCHORS for m in METRICS}, label=f"{path.name} factorial")
    require_equal(passes, expected_passes, label=f"{path.name} passed checks")
    return rows


def validate_safety(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b low-RUL safety gate")
    require_columns(
        fields,
        (*INFERENCE_COLUMNS, "holm_adjusted_p_noninferiority_3pct", "holm_noninferiority_3pct_passed"),
        label=path.name,
    )
    require_equal(len(rows), 4, label="low-RUL safety row count")
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        label = f"low-RUL safety row {index}"
        require_equal(as_float(row["registered_rul_anchor"], label=f"{label} anchor"), 15.0, label=f"{label} anchor")
        require_equal(as_int(row["shot"], label=f"{label} shot"), PRIMARY_SHOT, label=f"{label} shot")
        key = (row["architecture"], row["metric"])
        if key in seen:
            raise A260Error(f"duplicate low-RUL architecture/metric: {key}")
        seen.add(key)
        require_false(row["holm_superiority_passed"], label=f"{label} superiority")
        require_false(row["holm_noninferiority_3pct_passed"], label=f"{label} noninferiority")
        as_float(row["holm_adjusted_p_noninferiority_3pct"], label=f"{label} adjusted p")
    require_equal(seen, {(a, m) for a in ARCHITECTURES for m in METRICS}, label="low-RUL factorial")
    return rows


def validate_domain_summary(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b domain summary")
    require_columns(
        fields,
        (
            "experiment_id",
            "architecture",
            "candidate_method",
            "reference_method",
            "target_domain",
            "shot",
            "registered_rul_anchor",
            "metric",
            "relative_degradation",
            "relative_improvement_pct",
        ),
        label=path.name,
    )
    require_equal(len(rows), 48, label="domain summary row count")
    seen: set[tuple[str, str, float, str]] = set()
    for index, row in enumerate(rows):
        label = f"domain summary row {index}"
        key = (
            row["architecture"],
            row["target_domain"],
            as_float(row["registered_rul_anchor"], label=f"{label} anchor"),
            row["metric"],
        )
        if key in seen:
            raise A260Error(f"duplicate domain-summary cell: {key}")
        seen.add(key)
        if key[0] not in ARCHITECTURES or key[1] not in DOMAINS or key[2] not in ANCHORS or key[3] not in METRICS:
            raise A260Error(f"invalid domain-summary cell: {key}")
        require_equal(as_int(row["shot"], label=f"{label} shot"), PRIMARY_SHOT, label=f"{label} shot")
        as_float(row["relative_degradation"], label=f"{label} relative degradation")
    expected = {(a, d, r, m) for a in ARCHITECTURES for d in DOMAINS for r in ANCHORS for m in METRICS}
    require_equal(seen, expected, label="domain summary factorial")
    return rows


def validate_secondary(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b secondary-shot summary")
    require_columns(
        fields,
        ("experiment_id", "analysis_role", "architecture", "shot", "registered_rul_anchor", "metric", "relative_degradation"),
        label=path.name,
    )
    require_equal(len(rows), 36, label="secondary-shot summary row count")
    seen: set[tuple[str, int, float, str]] = set()
    for index, row in enumerate(rows):
        label = f"secondary row {index}"
        shot = as_int(row["shot"], label=f"{label} shot")
        expected_role = "primary" if shot == PRIMARY_SHOT else "secondary"
        require_equal(row["analysis_role"], expected_role, label=f"{label} role")
        key = (
            row["architecture"],
            shot,
            as_float(row["registered_rul_anchor"], label=f"{label} anchor"),
            row["metric"],
        )
        if key in seen:
            raise A260Error(f"duplicate secondary cell: {key}")
        seen.add(key)
        if key[0] not in ARCHITECTURES or key[1] not in SHOTS or key[2] not in ANCHORS or key[3] not in METRICS:
            raise A260Error(f"invalid secondary cell: {key}")
        as_float(row["relative_degradation"], label=f"{label} relative degradation")
    expected = {(a, k, r, m) for a in ARCHITECTURES for k in SHOTS for r in ANCHORS for m in METRICS}
    require_equal(seen, expected, label="secondary-shot factorial")
    return rows


def validate_run_metrics(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b run-level metrics")
    require_columns(
        fields,
        (
            "experiment_id",
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "method",
            "architecture",
            "registered_rul_anchor",
            "n_engines",
            "rmse",
            "mae",
            "mean_error",
            "nasa_score",
        ),
        label=path.name,
    )
    require_equal(len(rows), 576, label="run-level metric row count")
    seen: set[tuple[str, int, int, int, str, float]] = set()
    for index, row in enumerate(rows):
        label = f"run-level row {index}"
        key = (
            row["target_domain"],
            as_int(row["model_seed"], label=f"{label} model seed"),
            as_int(row["support_split_seed"], label=f"{label} split seed"),
            as_int(row["shot"], label=f"{label} shot"),
            row["method"],
            as_float(row["registered_rul_anchor"], label=f"{label} anchor"),
        )
        if key in seen:
            raise A260Error(f"duplicate run-level cell: {key}")
        seen.add(key)
        if key[0] not in DOMAINS or key[1] not in MODEL_SEEDS or key[2] not in SUPPORT_SPLIT_SEEDS or key[3] not in SHOTS or key[4] not in METHODS or key[5] not in ANCHORS:
            raise A260Error(f"invalid run-level cell: {key}")
        for numeric in ("rmse", "mae", "mean_error", "nasa_score"):
            as_float(row[numeric], label=f"{label} {numeric}")
        if as_int(row["n_engines"], label=f"{label} n_engines") < 1:
            raise A260Error(f"{label} has no engines")
    expected = {
        (d, ms, ss, k, method, anchor)
        for d in DOMAINS
        for ms in MODEL_SEEDS
        for ss in SUPPORT_SPLIT_SEEDS
        for k in SHOTS
        for method in METHODS
        for anchor in ANCHORS
    }
    require_equal(seen, expected, label="run-level metric factorial")
    return rows


def validate_prefix_coverage(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b prefix coverage")
    require_columns(
        fields,
        (
            "target_domain",
            "support_split_seed",
            "engine_id",
            "cycle",
            "trajectory_cycles",
            "registered_rul_anchor",
            "true_rul",
            "raw_true_rul",
            "input_uses_future_cycles",
        ),
        label=path.name,
    )
    require_equal(len(rows), 3282, label="prefix coverage row count")
    counts: Counter[float] = Counter()
    for index, row in enumerate(rows):
        label = f"prefix row {index}"
        anchor = as_float(row["registered_rul_anchor"], label=f"{label} anchor")
        if anchor not in ANCHORS:
            raise A260Error(f"{label} has unregistered anchor {anchor}")
        counts[anchor] += 1
        require_false(row["input_uses_future_cycles"], label=f"{label} future-cycle flag")
        cycle = as_int(row["cycle"], label=f"{label} cycle")
        total = as_int(row["trajectory_cycles"], label=f"{label} trajectory cycles")
        if cycle < 1 or total < cycle:
            raise A260Error(f"{label} has invalid causal prefix cycle={cycle}, trajectory={total}")
        require_equal(as_float(row["true_rul"], label=f"{label} true_rul"), anchor, label=f"{label} true_rul")
    require_equal(dict(counts), {90.0: 1094, 45.0: 1094, 15.0: 1094}, label="prefix anchor counts")
    return rows


def validate_normalizers(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b normalizer audit")
    require_columns(
        fields,
        (
            "target_domain",
            "model_seed",
            "support_split_seed",
            "stored_equals_recomputed",
            "target_domain_used_for_fit",
            "selection_engines_used_for_fit",
            "confirmation_engines_used_for_fit",
        ),
        label=path.name,
    )
    require_equal(len(rows), 16, label="normalizer audit row count")
    seen: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows):
        label = f"normalizer row {index}"
        key = (
            row["target_domain"],
            as_int(row["model_seed"], label=f"{label} model seed"),
            as_int(row["support_split_seed"], label=f"{label} split seed"),
        )
        if key in seen:
            raise A260Error(f"duplicate normalizer cell: {key}")
        seen.add(key)
        require_true(row["stored_equals_recomputed"], label=f"{label} recomputation equality")
        for field in ("target_domain_used_for_fit", "selection_engines_used_for_fit", "confirmation_engines_used_for_fit"):
            require_false(row[field], label=f"{label} {field}")
    expected = {(d, ms, ss) for d in DOMAINS for ms in MODEL_SEEDS for ss in SUPPORT_SPLIT_SEEDS}
    require_equal(seen, expected, label="normalizer factorial")
    return rows


def validate_checkpoints(path: Path) -> list[dict[str, str]]:
    fields, rows = read_csv(path, label="A25.2b checkpoint validation inventory")
    require_columns(
        fields,
        (
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "method",
            "architecture",
            "checkpoint_sha256",
            "primary_analysis_checkpoint",
            "secondary_analysis_checkpoint",
            "checkpoint_opened_in_A25_2a",
            "confirmation_evaluated_in_A25_2a",
            "checkpoint_tensor_validation_passed",
            "strict_model_load_passed",
            "state_schema_sha256",
        ),
        label=path.name,
    )
    require_equal(len(rows), 192, label="checkpoint inventory row count")
    primary = secondary = 0
    seen: set[tuple[str, int, int, int, str]] = set()
    for index, row in enumerate(rows):
        label = f"checkpoint row {index}"
        key = (
            row["target_domain"],
            as_int(row["model_seed"], label=f"{label} model seed"),
            as_int(row["support_split_seed"], label=f"{label} split seed"),
            as_int(row["shot"], label=f"{label} shot"),
            row["method"],
        )
        if key in seen:
            raise A260Error(f"duplicate checkpoint cell: {key}")
        seen.add(key)
        require_hash(row["checkpoint_sha256"], label=f"{label} checkpoint SHA256")
        require_hash(row["state_schema_sha256"], label=f"{label} schema SHA256")
        is_primary = as_bool(row["primary_analysis_checkpoint"], label=f"{label} primary flag")
        is_secondary = as_bool(row["secondary_analysis_checkpoint"], label=f"{label} secondary flag")
        require_equal(is_primary, key[3] == PRIMARY_SHOT, label=f"{label} primary flag")
        require_equal(is_secondary, key[3] in SECONDARY_SHOTS, label=f"{label} secondary flag")
        primary += int(is_primary)
        secondary += int(is_secondary)
        require_false(row["checkpoint_opened_in_A25_2a"], label=f"{label} A25.2a opened")
        require_false(row["confirmation_evaluated_in_A25_2a"], label=f"{label} A25.2a evaluated")
        require_true(row["checkpoint_tensor_validation_passed"], label=f"{label} tensor validation")
        require_true(row["strict_model_load_passed"], label=f"{label} strict model load")
    expected = {(d, ms, ss, k, method) for d in DOMAINS for ms in MODEL_SEEDS for ss in SUPPORT_SPLIT_SEEDS for k in SHOTS for method in METHODS}
    require_equal(seen, expected, label="checkpoint factorial")
    require_equal(primary, 64, label="primary checkpoint count")
    require_equal(secondary, 128, label="secondary checkpoint count")
    return rows


def make_failure_inventory(
    no_graph_rows: Sequence[Mapping[str, str]],
    gnn_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in (*no_graph_rows, *gnn_rows):
        relative = as_float(row["relative_degradation"], label="relative degradation")
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_experiment_id": SOURCE_EXPERIMENT_ID,
                "evidence_role": "frozen_read_only_confirmation",
                "hypothesis_role": row["hypothesis_role"],
                "architecture": row["architecture"],
                "candidate_method": row["candidate_method"],
                "reference_method": row["reference_method"],
                "shot": as_int(row["shot"], label="shot"),
                "registered_rul_anchor": as_float(row["registered_rul_anchor"], label="anchor"),
                "rul_stage": row["rul_stage"],
                "metric": row["metric"],
                "candidate_value": as_float(row["candidate_value"], label="candidate value"),
                "reference_value": as_float(row["reference_value"], label="reference value"),
                "relative_degradation": relative,
                "relative_ci95_low": as_float(row["relative_ci95_low"], label="CI low"),
                "relative_ci95_high": as_float(row["relative_ci95_high"], label="CI high"),
                "candidate_engine_win_rate": as_float(row["candidate_engine_win_rate"], label="win rate"),
                "holm_adjusted_p_superiority": as_float(row["holm_adjusted_p_superiority"], label="Holm p"),
                "holm_superiority_passed": as_bool(row["holm_superiority_passed"], label="Holm pass"),
                "observed_direction": "candidate_better" if relative < 0 else "candidate_worse_or_equal",
                "permitted_use": "descriptive_failure_diagnosis_only",
                "prohibited_use": "candidate_selection_hyperparameter_tuning_or_new_efficacy_claim",
            }
        )
    return sorted(output, key=lambda x: (x["architecture"], -x["registered_rul_anchor"], x["metric"]))


def make_risk_register(safety_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in safety_rows:
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "risk_id": f"LOW_RUL_{row['architecture'].upper()}_{row['metric'].upper()}",
                "architecture": row["architecture"],
                "candidate_method": row["candidate_method"],
                "reference_method": row["reference_method"],
                "shot": as_int(row["shot"], label="safety shot"),
                "registered_rul_anchor": 15.0,
                "metric": row["metric"],
                "relative_degradation": as_float(row["relative_degradation"], label="safety degradation"),
                "relative_ci95_low": as_float(row["relative_ci95_low"], label="safety CI low"),
                "relative_ci95_high": as_float(row["relative_ci95_high"], label="safety CI high"),
                "holm_adjusted_p_noninferiority_3pct": as_float(row["holm_adjusted_p_noninferiority_3pct"], label="safety adjusted p"),
                "noninferiority_3pct_passed": False,
                "status": "open_confirmed_risk",
                "required_A26_1_diagnostic": "positive_error_tail_and_stage_conditioned_calibration",
                "deployment_allowed": False,
            }
        )
    return sorted(output, key=lambda x: (x["architecture"], x["metric"]))


def make_domain_inventory(domain_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in domain_rows:
        relative = as_float(row["relative_degradation"], label="domain relative degradation")
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_experiment_id": SOURCE_EXPERIMENT_ID,
                "evidence_role": "frozen_read_only_confirmation",
                "architecture": row["architecture"],
                "candidate_method": row["candidate_method"],
                "reference_method": row["reference_method"],
                "target_domain": row["target_domain"],
                "shot": as_int(row["shot"], label="domain shot"),
                "registered_rul_anchor": as_float(row["registered_rul_anchor"], label="domain anchor"),
                "metric": row["metric"],
                "n_paired_engine_records": as_int(row["n_paired_engine_records"], label="domain paired records"),
                "relative_degradation": relative,
                "candidate_better_descriptively": relative < 0,
                "permitted_use": "heterogeneity_description_only",
                "prohibited_use": "domain_specific_candidate_or_hyperparameter_selection",
            }
        )
    return sorted(output, key=lambda x: (x["architecture"], x["target_domain"], -x["registered_rul_anchor"], x["metric"]))


def component_registry() -> list[dict[str, Any]]:
    common = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "A26.1_development_only",
        "change_policy": "one_factor_at_a_time_against_locked_A25_reference",
        "A25_2b_confirmation_available_to_candidate_selector": False,
        "official_test_allowed": False,
        "confirmatory_claim_allowed": False,
    }
    rows = [
        {
            **common,
            "diagnostic_id": "D1",
            "component": "low_rul_overprediction_and_tail_risk",
            "registered_question": "Does candidate degradation concentrate in positive RUL error and its upper tail at low RUL?",
            "required_outputs": "mean_error;positive_error_q90;positive_error_q95;rmse;nasa_score;engine_level_pairs",
            "primary_stage": "low_rul_le30",
        },
        {
            **common,
            "diagnostic_id": "D2",
            "component": "rul_stage_tradeoff",
            "registered_question": "Is the high-RUL gain exchanged for mid/low-RUL degradation within the same paired development engines?",
            "required_outputs": "stage_conditioned_rmse;stage_conditioned_nasa_score;stage_conditioned_mean_error",
            "primary_stage": "all_registered_stages",
        },
        {
            **common,
            "diagnostic_id": "D3",
            "component": "reptile_meta_update_sensitivity",
            "registered_question": "Which single meta-update component causes stage-conditioned drift under matched compute?",
            "required_outputs": "outer_update_norm;inner_loss_trajectory;parameter_drift;matched_compute_audit",
            "primary_stage": "development_only",
        },
        {
            **common,
            "diagnostic_id": "D4",
            "component": "target_adaptation_drift",
            "registered_question": "Does target adaptation create or amplify low-RUL bias relative to the frozen pre-adaptation state?",
            "required_outputs": "pre_post_adaptation_error;parameter_drift;support_fit;development_query_metrics",
            "primary_stage": "low_rul_le30",
        },
        {
            **common,
            "diagnostic_id": "D5",
            "component": "gnn_graph_dependence_and_batch_sensitivity",
            "registered_question": "Does graph construction or batch composition explain the lack of GNN replication?",
            "required_outputs": "graph_ablation;batch_composition_audit;edge_weight_summary;paired_development_metrics",
            "primary_stage": "all_registered_stages",
        },
        {
            **common,
            "diagnostic_id": "D6",
            "component": "source_task_coverage",
            "registered_question": "Does source episode composition underrepresent degradation-stage transitions needed at low RUL?",
            "required_outputs": "source_stage_coverage;domain_task_balance;window_exposure_audit",
            "primary_stage": "source_roles_only",
        },
    ]
    return rows


def data_role_contract() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "data_or_evidence": "A25.2b confirmation outcomes",
            "role": "frozen_read_only_confirmation_evidence",
            "A26_0_access": "integrity_validation_and_descriptive_failure_inventory",
            "A26_1_access": "not_available_to_candidate_selector_or_tuning_loop",
            "allowed_actions": "quote_locked_results;verify_hashes;diagnostic_motivation",
            "prohibited_actions": "select_candidate;select_hyperparameter;early_stop;train;relabel_as_development",
            "formal_claim_allowed": False,
        },
        {
            "experiment_id": EXPERIMENT_ID,
            "data_or_evidence": "A25.1b historical selection outcomes",
            "role": "historical_development_evidence",
            "A26_0_access": "contract_reference_only",
            "A26_1_access": "exploratory_diagnostics_with_full_disclosure",
            "allowed_actions": "mechanism_diagnosis;debugging;exploratory_comparison",
            "prohibited_actions": "independent_confirmation_claim",
            "formal_claim_allowed": False,
        },
        {
            "experiment_id": EXPERIMENT_ID,
            "data_or_evidence": "A26.1 newly assigned development partitions",
            "role": "exploratory_candidate_development",
            "A26_0_access": "metadata_and_role_freeze_only",
            "A26_1_access": "training_selection_and_one_factor_diagnostics_allowed",
            "allowed_actions": "train;diagnose;select_exploratory_candidate;discard_failed_candidate",
            "prohibited_actions": "confirmatory_or_deployment_claim",
            "formal_claim_allowed": False,
        },
        {
            "experiment_id": EXPERIMENT_ID,
            "data_or_evidence": "official C-MAPSS test files",
            "role": "sealed_external_final_evaluation",
            "A26_0_access": "forbidden",
            "A26_1_access": "forbidden",
            "allowed_actions": "one_time_evaluation_only_after_full_model_and_analysis_freeze",
            "prohibited_actions": "exploration;tuning;selection;repeated_evaluation",
            "formal_claim_allowed": False,
        },
    ]


def statistical_plan() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "phase": "A26.1_exploratory_failure_diagnosis",
        "analysis_status": "exploratory_not_confirmatory",
        "primary_objective": "identify mechanisms of low-RUL degradation without reusing A25.2b confirmation for tuning",
        "locked_reference_methods": {
            "no_graph": "ordinary_no_graph_pft",
            "gnn": "ordinary_gnn_pft",
        },
        "locked_candidate_families": {
            "no_graph": "reptile_meta_no_graph",
            "gnn": "reptile_meta_gnn",
        },
        "registered_diagnostic_axes": [
            "target_domain",
            "model_seed",
            "support_split_seed",
            "shot",
            "rul_stage",
            "architecture",
        ],
        "registered_stages": ["high_rul_gt60", "mid_rul_31_to_60", "low_rul_le30"],
        "required_metrics": [
            "rmse",
            "mae",
            "mean_error",
            "nasa_score",
            "positive_error_q90",
            "positive_error_q95",
        ],
        "paired_unit": "same_target_domain_model_seed_support_split_engine_and_stage",
        "one_factor_at_a_time": True,
        "matched_compute_required": True,
        "same_architecture_and_initialization_required": True,
        "A25_2b_confirmation_reuse_for_tuning": False,
        "official_test_access_in_A26_0_or_A26_1": False,
        "multiplicity_status": "descriptive_exploratory; no confirmatory p-value interpretation",
        "candidate_advancement_rule": (
            "may advance only from newly assigned A26.1 development roles after complete compute, "
            "parameter, leakage, and low-RUL tail audits; advancement is not an efficacy claim"
        ),
        "future_confirmation_rule": (
            "a new efficacy claim requires a genuinely external or newly sealed evaluation set and a "
            "new preregistration frozen before any outcome is opened"
        ),
    }


def protocol_payload(
    source_dir: Path,
    source_script: Path,
    manifest: Mapping[str, Any],
    failure_inventory: Sequence[Mapping[str, Any]],
    risk_register: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "contract_type": "failure_diagnostic_preflight",
        "source_confirmation_experiment": SOURCE_EXPERIMENT_ID,
        "source_confirmation_output_dir": str(source_dir),
        "source_evaluator_script": str(source_script),
        "source_evaluator_script_sha256": manifest["script_sha256"],
        "source_preregistration_sha256": PREREGISTRATION_SHA256,
        "source_confirmation_status": "valid_but_formal_efficacy_not_supported",
        "locked_observation": {
            "no_graph_holm_superiority_checks_passed": 2,
            "no_graph_holm_superiority_checks_expected": 6,
            "no_graph_low_rul_safety_gate_passed": False,
            "gnn_holm_superiority_checks_passed": 0,
            "gnn_holm_superiority_checks_expected": 6,
            "gnn_low_rul_safety_gate_passed": False,
            "formal_efficacy_claim_supported": False,
        },
        "failure_inventory_rows": len(failure_inventory),
        "open_low_rul_risks": len(risk_register),
        "A25_2b_confirmation_is_immutable": True,
        "A25_2b_confirmation_available_to_A26_1_candidate_selector": False,
        "A25_2b_confirmation_reuse_for_tuning": False,
        "A26_1_exploratory_only": True,
        "one_factor_at_a_time_required": True,
        "matched_compute_required": True,
        "same_architecture_and_initialization_required": True,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "confirmation_predictions_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def input_integrity(
    source_dir: Path,
    source_script: Path,
    verified_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "A25_2b_output_dir": str(source_dir),
        "A25_2b_script": str(source_script),
        "A25_2b_script_sha256": sha256(source_script),
        "A25_2b_manifest_sha256": sha256(source_dir / "experimentA25_2b_manifest.json"),
        "A25_2b_artifacts": dict(sorted(verified_artifacts.items())),
        "all_manifest_artifact_hashes_verified": True,
        "large_prediction_artifacts_hashed_without_row_parsing": True,
        "checkpoint_files_opened": False,
        "official_test_files_accessed": False,
    }


def validate_existing_output(output_dir: Path, source_dir: Path, source_script: Path) -> dict[str, Any]:
    manifest_path = output_dir / "experimentA26_0_manifest.json"
    manifest = read_json(manifest_path, label="existing A26.0 manifest")
    require_equal(manifest.get("experiment_id"), EXPERIMENT_ID, label="existing manifest experiment_id")
    require_equal(manifest.get("script_version"), SCRIPT_VERSION, label="existing manifest script_version")
    require_equal(manifest.get("script_sha256"), sha256(Path(__file__).resolve()), label="existing A26.0 script SHA256")
    require_equal(manifest.get("A25_2b_manifest_sha256"), sha256(source_dir / "experimentA25_2b_manifest.json"), label="existing input manifest SHA256")
    require_equal(manifest.get("A25_2b_script_sha256"), sha256(source_script), label="existing input script SHA256")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise A260Error("existing A26.0 manifest has no artifacts")
    for name, expected in artifacts.items():
        if Path(name).name != name:
            raise A260Error(f"unsafe existing A26.0 artifact name: {name!r}")
        path = output_dir / name
        if not path.is_file():
            raise A260Error(f"existing A26.0 artifact is missing: {path}")
        require_equal(sha256(path), require_hash(expected, label=f"existing {name} hash"), label=f"existing {name} SHA256")
    decision = read_json(output_dir / "experimentA26_0_confirmation_decision.json", label="existing A26.0 decision")
    for field in ("complete", "passed", "preflight_only", "A25_2b_confirmation_frozen_read_only"):
        require_true(decision.get(field), label=f"existing decision {field}")
    return decision


def ensure_output_ready(output_dir: Path, *, resume: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise A260Error(f"output path exists but is not a directory: {output_dir}")
    entries = list(output_dir.iterdir())
    if entries and not resume:
        raise A260Error(
            f"output directory is not empty: {output_dir}; use a new directory or --resume"
        )


def build_artifacts(
    source_dir: Path,
    source_script: Path,
    manifest: Mapping[str, Any],
    verified_artifacts: Mapping[str, str],
    no_graph_rows: Sequence[Mapping[str, str]],
    gnn_rows: Sequence[Mapping[str, str]],
    safety_rows: Sequence[Mapping[str, str]],
    domain_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    failures = make_failure_inventory(no_graph_rows, gnn_rows)
    risks = make_risk_register(safety_rows)
    domains = make_domain_inventory(domain_rows)
    components = component_registry()
    roles = data_role_contract()
    stats = statistical_plan()
    protocol = protocol_payload(source_dir, source_script, manifest, failures, risks)
    integrity = input_integrity(source_dir, source_script, verified_artifacts)
    return {
        "experimentA26_0_diagnostic_protocol.json": protocol,
        "experimentA26_0_failure_pattern_inventory.csv": failures,
        "experimentA26_0_low_rul_risk_register.csv": risks,
        "experimentA26_0_domain_direction_inventory.csv": domains,
        "experimentA26_0_component_diagnostic_registry.csv": components,
        "experimentA26_0_data_role_contract.csv": roles,
        "experimentA26_0_statistical_analysis_plan.json": stats,
        "experimentA26_0_input_integrity.json": integrity,
    }


def write_outputs(output_dir: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = output_dir / name
        if name.endswith(".json"):
            atomic_json(path, payload)
        elif name.endswith(".csv"):
            atomic_csv(path, payload)
        else:
            raise A260Error(f"unsupported output artifact type: {name}")

    hashes = {name: sha256(output_dir / name) for name in sorted(artifacts)}
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "preflight_only": True,
        "exploratory_only": True,
        "A25_2b_confirmation_frozen_read_only": True,
        "A25_2b_confirmation_reused_for_tuning": False,
        "A25_2b_confirmation_available_to_A26_1_candidate_selector": False,
        "source_confirmation_valid": True,
        "source_formal_efficacy_claim_supported": False,
        "source_no_graph_checks_passed": 2,
        "source_no_graph_checks_expected": 6,
        "source_gnn_checks_passed": 0,
        "source_gnn_checks_expected": 6,
        "source_low_rul_safety_gate_passed": False,
        "one_factor_at_a_time_diagnostics_registered": True,
        "same_architecture_and_initialization_required": True,
        "matched_compute_required": True,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "formal_efficacy_claim": False,
        "artifact_sha256": dict(hashes),
        "reason": "A26.0 froze a leakage-resistant exploratory failure-diagnostic contract after the negative A25.2b confirmation",
        "interpretation_limit": (
            "A26.0 records A25.2b as immutable descriptive evidence. A26.1 may diagnose and develop "
            "candidates only on newly assigned development roles; it cannot convert A25.2b or official-test "
            "outcomes into tuning data or a confirmatory claim."
        ),
        "next_action": "implement_A26_1_one_factor_failure_diagnostic_development_experiment",
    }
    decision_name = "experimentA26_0_confirmation_decision.json"
    atomic_json(output_dir / decision_name, decision)
    hashes[decision_name] = sha256(output_dir / decision_name)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "A25_2b_manifest_sha256": artifacts["experimentA26_0_input_integrity.json"]["A25_2b_manifest_sha256"],
        "A25_2b_script_sha256": artifacts["experimentA26_0_input_integrity.json"]["A25_2b_script_sha256"],
        "artifacts": dict(sorted(hashes.items())),
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(output_dir / "experimentA26_0_manifest.json", manifest)
    return decision


def preview_payload(
    output_dir: Path,
    artifacts: Mapping[str, Any],
    verified_artifacts: Mapping[str, str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "A25_2b_manifest_artifacts_verified": len(verified_artifacts),
        "failure_pattern_rows": len(artifacts["experimentA26_0_failure_pattern_inventory.csv"]),
        "open_low_rul_risks": len(artifacts["experimentA26_0_low_rul_risk_register.csv"]),
        "domain_direction_rows": len(artifacts["experimentA26_0_domain_direction_inventory.csv"]),
        "registered_component_diagnostics": len(artifacts["experimentA26_0_component_diagnostic_registry.csv"]),
        "registered_data_roles": len(artifacts["experimentA26_0_data_role_contract.csv"]),
        "source_no_graph_checks_passed": 2,
        "source_no_graph_checks_expected": 6,
        "source_gnn_checks_passed": 0,
        "source_gnn_checks_expected": 6,
        "source_low_rul_safety_gate_passed": False,
        "A25_2b_confirmation_frozen_read_only": True,
        "A25_2b_confirmation_reused_for_tuning": False,
        "A25_2b_confirmation_available_to_A26_1_candidate_selector": False,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "model_forward_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = resolve(args.a25_2b_output_dir)
    source_script = resolve(args.a25_2b_script)
    output_dir = resolve(args.output_dir)
    validate_paths(source_dir, source_script, output_dir)

    if not args.dry_run:
        ensure_output_ready(output_dir, resume=args.resume)

    manifest, verified = validate_manifest(source_dir, source_script)
    validate_decision(source_dir)
    validate_preflight_and_unseal(source_dir)
    no_graph_rows = validate_inference_family(
        source_dir / "experimentA25_2b_primary_no_graph_hierarchical_inference.csv",
        architecture="no_graph",
        hypothesis_role="primary",
        expected_passes=2,
    )
    gnn_rows = validate_inference_family(
        source_dir / "experimentA25_2b_gnn_replication_hierarchical_inference.csv",
        architecture="gnn",
        hypothesis_role="replication",
        expected_passes=0,
    )
    safety_rows = validate_safety(source_dir / "experimentA25_2b_low_rul_safety_gate.csv")
    domain_rows = validate_domain_summary(source_dir / "experimentA25_2b_domain_summary.csv")
    validate_secondary(source_dir / "experimentA25_2b_secondary_shot_summary.csv")
    validate_run_metrics(source_dir / "experimentA25_2b_run_level_metrics.csv")
    validate_prefix_coverage(source_dir / "experimentA25_2b_prefix_coverage.csv")
    validate_normalizers(source_dir / "experimentA25_2b_normalizer_recomputation_audit.csv")
    validate_checkpoints(source_dir / "experimentA25_2b_checkpoint_validation_inventory.csv")

    if not args.dry_run and args.resume and (output_dir / "experimentA26_0_manifest.json").is_file():
        decision = validate_existing_output(output_dir, source_dir, source_script)
        print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        print("[A26.0] existing complete freeze and all A25.2b inputs revalidated; no files were changed")
        return 0

    artifacts = build_artifacts(
        source_dir,
        source_script,
        manifest,
        verified,
        no_graph_rows,
        gnn_rows,
        safety_rows,
        domain_rows,
    )
    preview = preview_payload(output_dir, artifacts, verified, dry_run=args.dry_run)
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if args.dry_run:
        print("[A26.0] dry-run passed; diagnostic contract is compatible and no files were written")
        return 0

    decision = write_outputs(output_dir, artifacts)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    print("[A26.0] completed failure-diagnostic contract freeze; no predictor was trained")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A260Error as exc:
        print(f"[A26.0] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
