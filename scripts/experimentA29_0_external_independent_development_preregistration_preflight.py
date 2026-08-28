#!/usr/bin/env python3
"""A29.0: freeze an external, independent development-data contract after A28.0 closure.

This preflight performs no model construction, inference, training, adaptation, or
metric calculation. It verifies A28.0 closure, hashes the external files without
parsing their observations, and freezes disjoint development/selection/confirmation
roles before any new method development begins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "experimentA29_0"
SCRIPT_VERSION = "experimentA29_0_external_independent_development_preregistration_preflight_v1"
MANIFEST_SCHEMA = "A29.0_external_dataset_manifest_v1"
FREEZE_TOKEN = "A29.0_EXTERNAL_INDEPENDENT_DEV_FREEZE"
ALLOWED_ROLES = {"development_train", "development_selection", "sealed_confirmation"}
REQUIRED_INDEPENDENCE = {"experimentA25_2b", "experimentA26_1", "experimentA27_1", "experimentA28_0"}
CLOSED_ROUTE = "reptile_meta_gnn_low_rul_overprediction_penalty_lambda1_threshold30_epoch10"


class A290Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise A290Error(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise A290Error(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A290Error(f"JSON object required: {path}")
    return value


def require_bool(mapping: dict[str, Any], key: str, expected: bool) -> None:
    if mapping.get(key) is not expected:
        raise A290Error(f"manifest field {key!r} must be {expected}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_a28(a28_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    decision_path = a28_dir / "experimentA28_0_confirmation_decision.json"
    manifest_path = a28_dir / "experimentA28_0_manifest.json"
    if not decision_path.is_file() or not manifest_path.is_file():
        raise A290Error("A28.0 decision and manifest are both required")
    decision = load_json(decision_path)
    manifest = load_json(manifest_path)
    if decision.get("experiment_id") != "experimentA28_0":
        raise A290Error("unexpected A28.0 experiment_id")
    for key in ("complete", "passed", "closure_only", "A27_1_candidate_route_closed"):
        if decision.get(key) is not True:
            raise A290Error(f"A28.0 decision requires {key}=true")
    if decision.get("candidate_selected") is not False or decision.get("formal_efficacy_claim") is not False:
        raise A290Error("A28.0 must record an unselected candidate and no efficacy claim")
    if decision.get("new_predictor_training") is not False or decision.get("official_test_files_accessed") is not False:
        raise A290Error("A28.0 training/test isolation assertion failed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise A290Error("A28.0 manifest artifact inventory is missing")
    verified: dict[str, str] = {}
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise A290Error("invalid A28.0 manifest artifact entry")
        path = a28_dir / name
        if not path.is_file():
            raise A290Error(f"A28.0 artifact is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise A290Error(f"A28.0 artifact hash mismatch: {name}")
        verified[name] = observed
    if verified.get(decision_path.name) != sha256(decision_path):
        raise A290Error("A28.0 decision is not covered by its manifest")
    return decision, verified


def validate_external_manifest(manifest_path: Path, dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise A290Error(f"schema_version must equal {MANIFEST_SCHEMA}")
    for key in ("dataset_id", "dataset_version", "dataset_origin", "registered_question", "candidate_family", "independence_statement"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise A290Error(f"nonempty string required: {key}")
    require_bool(manifest, "official_test_dataset", False)
    require_bool(manifest, "prior_outcomes_used_for_role_assignment", False)
    require_bool(manifest, "role_assignment_frozen", True)
    require_bool(manifest, "confirmation_sealed", True)
    require_bool(manifest, "closed_candidate_route_reused", False)
    independent = manifest.get("independent_of_experiments")
    if not isinstance(independent, list) or not REQUIRED_INDEPENDENCE.issubset(set(independent)):
        missing = sorted(REQUIRED_INDEPENDENCE - set(independent or []))
        raise A290Error(f"independent_of_experiments is missing: {missing}")
    forbidden = manifest.get("forbidden_tuning_evidence")
    if not isinstance(forbidden, list) or not {"A25.2b_confirmation", "A27.1_selection", "official_test"}.issubset(set(forbidden)):
        raise A290Error("forbidden_tuning_evidence must include A25.2b_confirmation, A27.1_selection, and official_test")
    if manifest["candidate_family"].strip() == CLOSED_ROUTE:
        raise A290Error("the closed A27.1 candidate route cannot be reused")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) < 3:
        raise A290Error("files must contain at least one file for each of the three roles")
    root = dataset_root.resolve()
    inventory: list[dict[str, Any]] = []
    logical_names: set[str] = set()
    resolved_paths: set[Path] = set()
    roles: Counter[str] = Counter()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise A290Error(f"files[{index}] must be an object")
        logical_name, relative, role, expected = (item.get("logical_name"), item.get("relative_path"), item.get("role"), item.get("sha256"))
        if not all(isinstance(value, str) and value.strip() for value in (logical_name, relative, role, expected)):
            raise A290Error(f"files[{index}] has a missing string field")
        if role not in ALLOWED_ROLES:
            raise A290Error(f"files[{index}] has invalid role: {role}")
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise A290Error(f"files[{index}] has invalid SHA256")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise A290Error(f"files[{index}] relative_path must stay under dataset root")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise A290Error(f"files[{index}] escapes dataset root") from exc
        if logical_name in logical_names or path in resolved_paths:
            raise A290Error(f"duplicate logical name or path at files[{index}]")
        if not path.is_file() or path.is_symlink():
            raise A290Error(f"regular non-symlink file required: {path}")
        observed = sha256(path)
        if observed != expected.lower():
            raise A290Error(f"external file hash mismatch: {relative}")
        size = path.stat().st_size
        declared_size = item.get("size_bytes")
        if declared_size is not None and (not isinstance(declared_size, int) or declared_size != size):
            raise A290Error(f"size_bytes mismatch: {relative}")
        inventory.append({
            "logical_name": logical_name, "relative_path": relative_path.as_posix(), "role": role,
            "size_bytes": size, "sha256": observed, "observations_parsed": False,
            "used_for_training_in_A29_0": False, "used_for_selection_in_A29_0": False,
        })
        logical_names.add(logical_name)
        resolved_paths.add(path)
        roles[role] += 1
    missing_roles = sorted(ALLOWED_ROLES - set(roles))
    if missing_roles:
        raise A290Error(f"external data roles are incomplete: {missing_roles}")
    return manifest, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a28-0-output-dir", required=True)
    parser.add_argument("--external-dataset-root", required=True)
    parser.add_argument("--external-dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confirm-freeze", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.confirm_freeze != FREEZE_TOKEN:
        raise A290Error(f"--confirm-freeze must equal {FREEZE_TOKEN}")
    a28_dir = Path(args.a28_0_output_dir).resolve()
    dataset_root = Path(args.external_dataset_root).resolve()
    manifest_path = Path(args.external_dataset_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_root.is_dir() or not manifest_path.is_file():
        raise A290Error("external dataset root and manifest must exist")
    _a28, a28_hashes = validate_a28(a28_dir)
    external, inventory = validate_external_manifest(manifest_path, dataset_root)
    roles = Counter(row["role"] for row in inventory)
    preview: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": args.dry_run,
        "dataset_id": external["dataset_id"],
        "dataset_version": external["dataset_version"],
        "registered_question": external["registered_question"],
        "candidate_family": external["candidate_family"],
        "external_files_hashed": len(inventory),
        "role_counts": dict(sorted(roles.items())),
        "independent_development_roles_verified": True,
        "role_assignment_frozen": True,
        "confirmation_sealed": True,
        "confirmation_observations_parsed": False,
        "new_predictor_training": False,
        "model_forward_run": False,
        "A25_2b_confirmation_used_for_tuning": False,
        "A27_1_selection_used_for_tuning": False,
        "official_test_files_accessed": False,
        "passed": True,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("[A29.0] dry-run passed; external independent roles are valid and no output was written")
        return 0
    final_path = output_dir / "experimentA29_0_confirmation_decision.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise A290Error(f"output directory must be new and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "experimentA29_0_external_data_role_inventory.csv"
    contract_path = output_dir / "experimentA29_0_frozen_development_contract.json"
    integrity_path = output_dir / "experimentA29_0_input_integrity.json"
    atomic_csv(inventory_path, list(inventory[0]), inventory)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": external["dataset_id"],
        "dataset_version": external["dataset_version"],
        "dataset_origin": external["dataset_origin"],
        "registered_question": external["registered_question"],
        "candidate_family": external["candidate_family"],
        "independence_statement": external["independence_statement"],
        "independent_of_experiments": sorted(set(external["independent_of_experiments"])),
        "forbidden_tuning_evidence": sorted(set(external["forbidden_tuning_evidence"])),
        "closed_candidate_route": CLOSED_ROUTE,
        "closed_candidate_route_reused": False,
        "roles": dict(sorted(roles.items())),
        "development_train_may_train_new_models_after_A29_0": True,
        "development_selection_may_select_one_candidate_after_A29_0": True,
        "sealed_confirmation_may_not_be_opened_until_a_later_preregistration": True,
        "official_test_access_authorized": False,
    }
    atomic_json(contract_path, contract)
    atomic_json(integrity_path, {
        "A28_0_artifact_sha256": a28_hashes,
        "external_manifest_sha256": sha256(manifest_path),
        "external_file_sha256": {row["logical_name"]: row["sha256"] for row in inventory},
    })
    final = {
        **preview,
        "dry_run": False,
        "complete": True,
        "preflight_only": True,
        "preregistered": True,
        "external_independent_development_contract_frozen": True,
        "closed_A27_candidate_route_remains_closed": True,
        "formal_efficacy_claim": False,
        "official_test_forward_run": False,
        "reason": "A29.0 froze independent external development, selection, and sealed-confirmation roles before new method development",
        "next_action": "implement_A29_1_new_candidate_development_only_after_the_frozen_external_data_contract_is_reviewed",
    }
    atomic_json(final_path, final)
    artifacts = {path.name: sha256(path) for path in (inventory_path, contract_path, integrity_path, final_path)}
    atomic_json(output_dir / "experimentA29_0_manifest.json", {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    })
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    print("[A29.0] completed external independent development preregistration; no predictor was trained")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A290Error as exc:
        print(f"[A29.0] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
