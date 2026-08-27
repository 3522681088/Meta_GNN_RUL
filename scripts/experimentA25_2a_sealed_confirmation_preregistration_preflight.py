#!/usr/bin/env python3
"""A25.2a: sealed confirmation preregistration and checkpoint freeze.

This script is deliberately a preflight.  It validates and freezes the exact
A25.1a statistical contract and the exact A25.1b target-adapted checkpoints
before any confirmation outcome is opened.  It reads confirmation *role
metadata* (domain, split seed and engine id), but it never opens training data,
confirmation observations, official test files, model tensors, or predictions.

The subsequent A25.2b evaluator must consume the artifacts emitted here.  It
must not train, tune, select checkpoints, or change the registered analysis.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "experimentA25_2a"
SCRIPT_VERSION = "experimentA25_2a_sealed_confirmation_preregistration_preflight_v1"
FREEZE_TOKEN = "A25.2A_FREEZE"
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
EXPECTED_SHOTS = (1, 2, 5)
PRIMARY_SHOT = 5
EXPECTED_ANCHORS = (90.0, 45.0, 15.0)
EXPECTED_MODEL_SEEDS = (140, 141)
EXPECTED_SPLIT_SEEDS = (7501, 7502)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TRUE_TEXT = {"true", "1", "yes"}
FALSE_TEXT = {"false", "0", "no"}


class A252aError(RuntimeError):
    """Raised when the sealed confirmation contract cannot be frozen safely."""


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
        raise A252aError(f"refusing to write empty CSV: {path.name}")
    fields = list(materialized[0])
    for index, row in enumerate(materialized):
        if set(row) != set(fields):
            raise A252aError(f"row schema mismatch at index={index} for {path.name}")
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
        raise A252aError(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A252aError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A252aError(f"{label} must contain a JSON object: {path}")
    return value


def read_csv(path: Path, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise A252aError(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A252aError(f"cannot read {label}: {path}: {exc}") from exc
    if not fields or not rows:
        raise A252aError(f"{label} is empty: {path}")
    return fields, rows


def require_fields(container: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(container))
    if missing:
        raise A252aError(f"{label} lacks required fields: {missing}")


def require_columns(fields: Sequence[str], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise A252aError(f"{label} lacks required columns: {missing}")


def as_int(value: Any, *, label: str) -> int:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise A252aError(f"{label} must be an integer, observed {value!r}") from exc
    return number


def as_float(value: Any, *, label: str) -> float:
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise A252aError(f"{label} must be numeric, observed {value!r}") from exc
    if not math.isfinite(number):
        raise A252aError(f"{label} must be finite, observed {value!r}")
    return number


def as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise A252aError(f"{label} must be boolean, observed {value!r}")


def require_hash(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if HASH_RE.fullmatch(text) is None:
        raise A252aError(f"{label} is not a SHA256 digest: {value!r}")
    return text


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the A25.2 sealed-confirmation preregistration without evaluating confirmation."
    )
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
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA25_2a_sealed_confirmation_preregistration_preflight"),
    )
    parser.add_argument("--minimum-confirmation-engines", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--confirm-freeze",
        default="",
        help=f"Required for a formal freeze; pass exactly {FREEZE_TOKEN!r}.",
    )
    args = parser.parse_args(argv)
    if args.minimum_confirmation_engines < 1:
        raise A252aError("--minimum-confirmation-engines must be positive")
    if not args.dry_run and args.confirm_freeze != FREEZE_TOKEN:
        raise A252aError(
            f"formal freeze requires --confirm-freeze {FREEZE_TOKEN}; run --dry-run first"
        )
    return args


def validate_distinct_roots(a25a: Path, a25b: Path, output: Path) -> None:
    if len({a25a, a25b, output}) != 3:
        raise A252aError("A25.1a, A25.1b and A25.2a output directories must be distinct")
    if is_relative_to(output, a25a) or is_relative_to(output, a25b):
        raise A252aError("A25.2a output directory must not be nested inside an input directory")
    if is_relative_to(a25a, output) or is_relative_to(a25b, output):
        raise A252aError("input directories must not be nested inside the A25.2a output directory")


def validate_complete_decision(
    payload: Mapping[str, Any], *, experiment_id: str, label: str
) -> None:
    if payload.get("experiment_id") != experiment_id:
        raise A252aError(f"{label} experiment identity mismatch")
    if payload.get("complete") is not True or payload.get("passed") is not True:
        raise A252aError(f"{label} must be complete=true and passed=true")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if payload.get(field) is not False:
            raise A252aError(f"{label} violates the official-test boundary: {field}")


def validate_hash_map(root: Path, values: Any, *, label: str) -> dict[str, str]:
    if not isinstance(values, dict) or not values:
        raise A252aError(f"{label} is absent or empty")
    observed: dict[str, str] = {}
    for name, expected_raw in sorted(values.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise A252aError(f"{label} contains an unsafe artifact name: {name!r}")
        expected = require_hash(expected_raw, label=f"{label}[{name}]")
        path = root / name
        if not path.is_file():
            raise A252aError(f"frozen artifact is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise A252aError(f"frozen artifact hash mismatch: {path}")
        observed[name] = actual
    return observed


def load_a25_1a(root: Path) -> dict[str, Any]:
    decision_path = root / "experimentA25_1a_confirmation_decision.json"
    protocol_path = root / "experimentA25_1a_protocol.json"
    statistics_path = root / "experimentA25_1a_statistical_analysis_plan.json"
    roles_path = root / "experimentA25_1a_target_engine_roles.csv"
    decision = read_json(decision_path, label="A25.1a decision")
    protocol = read_json(protocol_path, label="A25.1a protocol")
    statistics = read_json(statistics_path, label="A25.1a statistical plan")
    validate_complete_decision(decision, experiment_id="experimentA25_1a", label="A25.1a")
    if decision.get("preflight_only") is not True:
        raise A252aError("A25.1a is not a preflight-only contract")
    required_true = (
        "same_architecture_parameter_and_initialization_contract_locked",
        "equal_source_gradient_update_and_window_presentation_budget_locked",
        "runtime_compute_accounting_required",
        "A25_1b_selection_only",
    )
    for field in required_true:
        if decision.get(field) is not True:
            raise A252aError(f"A25.1a requires {field}=true")
    if decision.get("A25_1b_confirmation_engines_evaluated") is not False:
        raise A252aError("A25.1a confirmation boundary was not sealed")
    require_fields(
        protocol,
        (
            "registered_primary_question", "methods", "model_seeds", "support_split_seeds",
            "shots", "primary_shot", "source_gradient_updates_per_method_cell",
            "source_window_presentations_per_method_cell", "target_epochs",
        ),
        label="A25.1a protocol",
    )
    if tuple(protocol["methods"]) != METHODS:
        raise A252aError("A25.1a method order does not match the registered 2x2 design")
    if tuple(map(int, protocol["model_seeds"])) != EXPECTED_MODEL_SEEDS:
        raise A252aError("A25.1a model seeds differ from the locked prospective seeds")
    if tuple(map(int, protocol["support_split_seeds"])) != EXPECTED_SPLIT_SEEDS:
        raise A252aError("A25.1a support split seeds differ from the locked prospective seeds")
    if tuple(map(int, protocol["shots"])) != EXPECTED_SHOTS:
        raise A252aError("A25.1a shots differ from K=1,2,5")
    if int(protocol["primary_shot"]) != PRIMARY_SHOT:
        raise A252aError("A25.1a primary shot is not K=5")
    if int(protocol["source_gradient_updates_per_method_cell"]) != 7500:
        raise A252aError("A25.1a source update budget is not 7500")
    if int(protocol["source_window_presentations_per_method_cell"]) != 480000:
        raise A252aError("A25.1a source window budget is not 480000")

    require_fields(
        statistics,
        (
            "primary_hypothesis_family_no_graph", "replication_hypothesis_family_gnn",
            "low_rul_safety_gate", "bootstrap_design",
            "bootstrap_repetitions_for_later_confirmation",
            "independent_confirmation_required_after_A25_1b",
        ),
        label="A25.1a statistical plan",
    )
    expected_families = {
        "primary_hypothesis_family_no_graph": (
            "reptile_meta_no_graph", "ordinary_no_graph_pft"
        ),
        "replication_hypothesis_family_gnn": (
            "reptile_meta_gnn", "ordinary_gnn_pft"
        ),
    }
    for name, (candidate, reference) in expected_families.items():
        family = statistics[name]
        if not isinstance(family, dict):
            raise A252aError(f"A25.1a {name} must be an object")
        if family.get("candidate") != candidate or family.get("reference") != reference:
            raise A252aError(f"A25.1a {name} method contrast changed")
        if int(family.get("shot", -1)) != PRIMARY_SHOT:
            raise A252aError(f"A25.1a {name} primary shot changed")
        if tuple(map(float, family.get("anchors", []))) != EXPECTED_ANCHORS:
            raise A252aError(f"A25.1a {name} anchors changed")
        if tuple(family.get("metrics", [])) != ("rmse", "nasa_score"):
            raise A252aError(f"A25.1a {name} metrics changed")
        if family.get("decision_rule") not in {
            "all_six_Holm_corrected_superiority_checks_pass",
            "separate_all_six_Holm_corrected_superiority_checks_pass",
        }:
            raise A252aError(f"A25.1a {name} decision rule is unexpected")
    if statistics.get("bootstrap_design") != (
        "target_domain_then_model_seed_then_support_split_then_paired_engine"
    ):
        raise A252aError("A25.1a bootstrap hierarchy changed")
    if int(statistics.get("bootstrap_repetitions_for_later_confirmation", 0)) != 5000:
        raise A252aError("A25.1a bootstrap repetition count changed")
    if statistics.get("independent_confirmation_required_after_A25_1b") is not True:
        raise A252aError("A25.1a does not require independent confirmation")
    gate = statistics["low_rul_safety_gate"]
    if (
        not isinstance(gate, dict)
        or float(gate.get("anchor", -1)) != 15.0
        or tuple(gate.get("metrics", [])) != ("rmse", "nasa_score")
        or float(gate.get("noninferiority_margin_pct", -1)) != 3.0
        or gate.get("one_sided") is not True
    ):
        raise A252aError("A25.1a low-RUL safety gate changed")

    hashes = validate_hash_map(root, decision.get("artifact_sha256"), label="A25.1a artifact_sha256")
    hashes.update({
        decision_path.name: sha256(decision_path),
        protocol_path.name: sha256(protocol_path),
        statistics_path.name: sha256(statistics_path),
        roles_path.name: sha256(roles_path),
    })
    return {
        "decision": decision,
        "protocol": protocol,
        "statistics": statistics,
        "roles_path": roles_path,
        "hashes": dict(sorted(hashes.items())),
    }


def validate_roles(
    roles_path: Path, *, minimum_confirmation_engines: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fields, rows = read_csv(roles_path, label="A25.1a target engine roles")
    required = {
        "target_domain", "support_split_seed", "engine_id", "role", "support_rank",
        "included_in_1shot", "included_in_2shot", "included_in_5shot",
    }
    require_columns(fields, required, label="A25.1a target engine roles")
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows, start=2):
        domain = row["target_domain"].strip()
        split = as_int(row["support_split_seed"], label=f"roles row {index} split")
        engine = as_int(row["engine_id"], label=f"roles row {index} engine")
        role = row["role"].strip()
        if domain not in DOMAINS or split not in EXPECTED_SPLIT_SEEDS:
            raise A252aError(f"unexpected target partition at roles row {index}")
        if engine < 1 or role not in {"support_pool", "selection", "confirmation"}:
            raise A252aError(f"invalid engine/role at roles row {index}")
        key = (domain, split, engine)
        if key in seen:
            raise A252aError(f"duplicate engine role assignment: {key}")
        seen.add(key)
        groups[(domain, split)].append(row)
    expected_groups = {(domain, split) for domain in DOMAINS for split in EXPECTED_SPLIT_SEEDS}
    if set(groups) != expected_groups:
        raise A252aError("target role file does not contain exactly eight registered partitions")

    confirmation_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for (domain, split), group in sorted(groups.items()):
        support = [row for row in group if row["role"].strip() == "support_pool"]
        selection = [row for row in group if row["role"].strip() == "selection"]
        confirmation = [row for row in group if row["role"].strip() == "confirmation"]
        if len(support) != max(EXPECTED_SHOTS):
            raise A252aError(f"{domain}/split={split} support pool must contain five engines")
        if len(confirmation) < minimum_confirmation_engines:
            raise A252aError(
                f"{domain}/split={split} has {len(confirmation)} confirmation engines; "
                f"minimum={minimum_confirmation_engines}"
            )
        ranks = sorted(as_int(row["support_rank"], label="support rank") for row in support)
        if ranks != list(range(1, max(EXPECTED_SHOTS) + 1)):
            raise A252aError(f"{domain}/split={split} support ranks are not 1..5")
        for row in group:
            role = row["role"].strip()
            rank = (
                as_int(row["support_rank"], label="support rank")
                if role == "support_pool" else None
            )
            for shot in EXPECTED_SHOTS:
                observed = as_bool(row[f"included_in_{shot}shot"], label="shot membership")
                expected = role == "support_pool" and rank is not None and rank <= shot
                if observed != expected:
                    raise A252aError(
                        f"nested shot membership mismatch for {domain}/split={split}/"
                        f"engine={row['engine_id']}/K={shot}"
                    )
        counts["support_pool"] += len(support)
        counts["selection"] += len(selection)
        counts["confirmation"] += len(confirmation)
        for row in sorted(confirmation, key=lambda item: as_int(item["engine_id"], label="engine")):
            confirmation_rows.append({
                "target_domain": domain,
                "support_split_seed": split,
                "engine_id": as_int(row["engine_id"], label="confirmation engine"),
                "role": "confirmation",
                "role_metadata_only": True,
                "outcome_opened_in_A25_2a": False,
                "prediction_run_in_A25_2a": False,
            })
    return confirmation_rows, dict(counts)


def load_a25_1b(root: Path, script_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision_path = root / "experimentA25_1b_confirmation_decision.json"
    manifest_path = root / "experimentA25_1b_manifest.json"
    run_path = root / "experimentA25_1b_selection_run_level.csv"
    audit_path = root / "experimentA25_1b_matched_compute_audit.csv"
    decision = read_json(decision_path, label="A25.1b decision")
    manifest = read_json(manifest_path, label="A25.1b manifest")
    validate_complete_decision(decision, experiment_id="experimentA25_1b", label="A25.1b")
    required_true = (
        "pilot_only", "selection_only_diagnostics", "new_predictor_training",
        "same_architecture_initialization_assertion_passed",
        "same_architecture_state_schema_assertion_passed",
        "same_architecture_compute_accounting_assertion_passed", "checkpoint_reload_passed",
    )
    for field in required_true:
        if decision.get(field) is not True:
            raise A252aError(f"A25.1b requires {field}=true")
    for field in ("formal_efficacy_claim", "confirmation_engines_evaluated"):
        if decision.get(field) is not False:
            raise A252aError(f"A25.1b requires {field}=false")
    if decision.get("selection_used_for_training") is not False:
        raise A252aError("A25.1b selection outcomes were used for training")
    if int(decision.get("completed_worker_cells", -1)) != int(decision.get("expected_worker_cells", -2)):
        raise A252aError("A25.1b worker completion count mismatch")
    if int(decision.get("completed_run_records", -1)) != int(decision.get("expected_run_records", -2)):
        raise A252aError("A25.1b run completion count mismatch")
    if int(decision.get("completed_worker_cells", -1)) != 16:
        raise A252aError("A25.1b must contain exactly 16 worker cells")
    if int(decision.get("completed_run_records", -1)) != 192:
        raise A252aError("A25.1b must contain exactly 192 run records")

    if manifest.get("experiment_id") != "experimentA25_1b":
        raise A252aError("A25.1b manifest identity mismatch")
    for field in ("pilot_only", "selection_only_diagnostics", "new_predictor_training"):
        if manifest.get(field) is not True:
            raise A252aError(f"A25.1b manifest requires {field}=true")
    for field in ("formal_efficacy_claim", "official_test_files_accessed", "official_test_forward_run"):
        if manifest.get(field) is not False:
            raise A252aError(f"A25.1b manifest requires {field}=false")
    artifact_hashes = validate_hash_map(root, manifest.get("artifacts"), label="A25.1b artifacts")
    if not script_path.is_file():
        raise A252aError(f"A25.1b script is missing: {script_path}")
    expected_script_hash = require_hash(manifest.get("script_sha256"), label="A25.1b script hash")
    if sha256(script_path) != expected_script_hash:
        raise A252aError("A25.1b script hash differs from the executed manifest")

    run_fields, run_rows = read_csv(run_path, label="A25.1b selection run-level results")
    required_run_columns = {
        "target_domain", "model_seed", "support_split_seed", "shot", "method",
        "architecture", "checkpoint", "checkpoint_sha256", "checkpoint_reload_passed",
        "initial_state_sha256", "state_schema_sha256", "state_tensor_numel",
        "state_tensor_count", "total_parameters", "trainable_parameters",
        "source_gradient_updates", "source_window_presentations", "source_forward_calls",
        "source_backward_calls", "target_gradient_updates", "target_window_presentations",
        "target_forward_calls", "target_backward_calls", "selection_used_for_training",
        "confirmation_used_for_training", "confirmation_used_for_evaluation",
        "official_test_files_accessed", "official_test_forward_run",
    }
    require_columns(run_fields, required_run_columns, label="A25.1b run-level results")
    if len(run_rows) != 192:
        raise A252aError(f"A25.1b run-level row count={len(run_rows)}, expected=192")
    expected_keys = {
        (domain, model_seed, split, shot, method)
        for domain in DOMAINS
        for model_seed in EXPECTED_MODEL_SEEDS
        for split in EXPECTED_SPLIT_SEEDS
        for shot in EXPECTED_SHOTS
        for method in METHODS
    }
    observed_keys: set[tuple[str, int, int, int, str]] = set()
    parsed: dict[tuple[str, int, int, int, str], dict[str, Any]] = {}
    checkpoint_inventory: list[dict[str, Any]] = []
    shards_root = (root / "shards").resolve()
    seen_paths: set[Path] = set()
    for index, row in enumerate(run_rows, start=2):
        key = (
            row["target_domain"].strip(),
            as_int(row["model_seed"], label=f"run row {index} model seed"),
            as_int(row["support_split_seed"], label=f"run row {index} split seed"),
            as_int(row["shot"], label=f"run row {index} shot"),
            row["method"].strip(),
        )
        if key in observed_keys:
            raise A252aError(f"duplicate A25.1b run key: {key}")
        observed_keys.add(key)
        method = key[-1]
        expected_architecture = "no_graph" if "no_graph" in method else "gnn"
        if row["architecture"].strip() != expected_architecture:
            raise A252aError(f"architecture mismatch at A25.1b run row {index}")
        true_fields = ("checkpoint_reload_passed",)
        false_fields = (
            "selection_used_for_training", "confirmation_used_for_training",
            "confirmation_used_for_evaluation", "official_test_files_accessed",
            "official_test_forward_run",
        )
        for field in true_fields:
            if not as_bool(row[field], label=f"run row {index} {field}"):
                raise A252aError(f"A25.1b run row {index} requires {field}=true")
        for field in false_fields:
            if as_bool(row[field], label=f"run row {index} {field}"):
                raise A252aError(f"A25.1b run row {index} requires {field}=false")
        if as_int(row["source_gradient_updates"], label="source updates") != int(
            protocol["source_gradient_updates_per_method_cell"]
        ):
            raise A252aError(f"source update budget mismatch at A25.1b run row {index}")
        if as_int(row["source_window_presentations"], label="source windows") != int(
            protocol["source_window_presentations_per_method_cell"]
        ):
            raise A252aError(f"source window budget mismatch at A25.1b run row {index}")
        for field in (
            "source_forward_calls", "source_backward_calls", "target_gradient_updates",
            "target_window_presentations", "target_forward_calls", "target_backward_calls",
            "state_tensor_numel", "state_tensor_count", "total_parameters", "trainable_parameters",
        ):
            if as_int(row[field], label=f"run row {index} {field}") < 1:
                raise A252aError(f"A25.1b run row {index} has nonpositive {field}")
        checkpoint = Path(row["checkpoint"].strip()).expanduser()
        checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (project_root() / checkpoint).resolve()
        if not is_relative_to(checkpoint, shards_root):
            raise A252aError(f"checkpoint escapes the frozen A25.1b shards directory: {checkpoint}")
        if checkpoint in seen_paths:
            raise A252aError(f"duplicate checkpoint path: {checkpoint}")
        seen_paths.add(checkpoint)
        if not checkpoint.is_file():
            raise A252aError(f"checkpoint is missing: {checkpoint}")
        expected_hash = require_hash(row["checkpoint_sha256"], label="checkpoint hash")
        actual_hash = sha256(checkpoint)
        if actual_hash != expected_hash:
            raise A252aError(f"checkpoint hash mismatch: {checkpoint}")
        record = {
            "target_domain": key[0], "model_seed": key[1], "support_split_seed": key[2],
            "shot": key[3], "method": key[4], "architecture": expected_architecture,
            "checkpoint": str(checkpoint), "checkpoint_sha256": actual_hash,
            "primary_analysis_checkpoint": key[3] == PRIMARY_SHOT,
            "secondary_analysis_checkpoint": key[3] != PRIMARY_SHOT,
            "checkpoint_opened_in_A25_2a": False,
            "confirmation_evaluated_in_A25_2a": False,
        }
        checkpoint_inventory.append(record)
        parsed[key] = row
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:5]
        extra = sorted(observed_keys - expected_keys)[:5]
        raise A252aError(f"A25.1b run key factorial mismatch; missing={missing}, extra={extra}")

    compare_fields = (
        "initial_state_sha256", "state_schema_sha256", "state_tensor_numel",
        "state_tensor_count", "total_parameters", "trainable_parameters",
        "source_gradient_updates", "source_window_presentations", "source_forward_calls",
        "source_backward_calls", "target_gradient_updates", "target_window_presentations",
        "target_forward_calls", "target_backward_calls",
    )
    for architecture, (reference, candidate) in PAIRS.items():
        for domain in DOMAINS:
            for model_seed in EXPECTED_MODEL_SEEDS:
                for split in EXPECTED_SPLIT_SEEDS:
                    for shot in EXPECTED_SHOTS:
                        left = parsed[(domain, model_seed, split, shot, reference)]
                        right = parsed[(domain, model_seed, split, shot, candidate)]
                        for field in compare_fields:
                            if left[field].strip() != right[field].strip():
                                raise A252aError(
                                    f"matched pair mismatch: {architecture}/{domain}/model={model_seed}/"
                                    f"split={split}/K={shot}/{field}"
                                )

    audit_fields, audit_rows = read_csv(audit_path, label="A25.1b matched compute audit")
    audit_bool_fields = (
        "initialization_identical", "state_schema_identical", "parameter_counts_identical",
        "source_gradient_updates_identical", "source_window_presentations_identical",
        "target_gradient_updates_identical", "target_window_presentations_identical",
    )
    require_columns(
        audit_fields, {"architecture", "shot", "paired_worker_cells", *audit_bool_fields},
        label="A25.1b matched compute audit",
    )
    expected_audit = {(architecture, shot) for architecture in PAIRS for shot in EXPECTED_SHOTS}
    observed_audit: set[tuple[str, int]] = set()
    for index, row in enumerate(audit_rows, start=2):
        key = (row["architecture"].strip(), as_int(row["shot"], label="audit shot"))
        if key in observed_audit:
            raise A252aError(f"duplicate compute audit cell: {key}")
        observed_audit.add(key)
        if as_int(row["paired_worker_cells"], label="paired worker cells") != 16:
            raise A252aError(f"compute audit cell {key} does not contain 16 pairs")
        for field in audit_bool_fields:
            if not as_bool(row[field], label=f"audit {key} {field}"):
                raise A252aError(f"compute audit failed: {key}/{field}")
    if observed_audit != expected_audit:
        raise A252aError("A25.1b compute audit does not cover six architecture-shot cells")

    hashes = dict(artifact_hashes)
    hashes.update({
        decision_path.name: sha256(decision_path),
        manifest_path.name: sha256(manifest_path),
        script_path.name: sha256(script_path),
    })
    checkpoint_inventory.sort(
        key=lambda row: (
            row["target_domain"], row["model_seed"], row["support_split_seed"],
            row["shot"], METHODS.index(row["method"]),
        )
    )
    return {
        "decision": decision,
        "manifest": manifest,
        "hashes": dict(sorted(hashes.items())),
        "checkpoint_inventory": checkpoint_inventory,
    }


def build_registered_plan(
    protocol: Mapping[str, Any], upstream_statistics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "registered_before_confirmation_open": True,
        "registered_primary_question": protocol["registered_primary_question"],
        "confirmation_source": "sealed_A25_1a_confirmation_engine_roles",
        "checkpoint_source": "frozen_A25_1b_target_adapted_checkpoints",
        "new_predictor_training_allowed": False,
        "checkpoint_selection_allowed": False,
        "hyperparameter_tuning_allowed": False,
        "primary_shot": PRIMARY_SHOT,
        "secondary_shots": [1, 2],
        "primary_hypothesis_family_no_graph": upstream_statistics[
            "primary_hypothesis_family_no_graph"
        ],
        "replication_hypothesis_family_gnn": upstream_statistics[
            "replication_hypothesis_family_gnn"
        ],
        "secondary_graph_increment": upstream_statistics.get("secondary_graph_increment"),
        "low_rul_safety_gate": upstream_statistics["low_rul_safety_gate"],
        "analysis_unit": "paired_engine_anchor_prediction",
        "cluster_hierarchy": [
            "target_domain", "model_seed", "support_split_seed", "paired_engine"
        ],
        "bootstrap_design": upstream_statistics["bootstrap_design"],
        "bootstrap_repetitions": int(
            upstream_statistics["bootstrap_repetitions_for_later_confirmation"]
        ),
        "random_seed_for_bootstrap": 25200,
        "superiority_direction": "candidate_metric_lower_than_reference",
        "multiplicity_control": "Holm_within_each_six_check_family",
        "family_success_rule": "all_six_Holm_corrected_superiority_checks_pass",
        "effect_reporting": [
            "paired_absolute_difference_candidate_minus_reference",
            "paired_relative_difference_pct",
            "hierarchical_bootstrap_95pct_confidence_interval",
            "domain_level_paired_summary",
        ],
        "missingness_rule": (
            "No silent row deletion. A missing prediction, checkpoint, engine-anchor cell or "
            "nonfinite metric fails the confirmation run and requires a documented audit; "
            "results may not be selectively rerun."
        ),
        "retry_rule": (
            "Only infrastructure failures occurring before any confirmation metric is revealed "
            "may be retried from the same frozen checkpoint and deterministic evaluator."
        ),
        "failure_rule": (
            "Any checkpoint hash change, backward call, training-mode mutation, role leakage, "
            "official-test access, or post-unseal analysis-plan change invalidates A25.2."
        ),
        "interpretation_rule": (
            "No generic Reptile efficacy claim is allowed unless the registered no-graph primary "
            "family succeeds. The GNN family is reported separately as replication."
        ),
    }


def build_evaluator_contract(plan_hash: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "next_experiment_id": "experimentA25_2b",
        "required_preregistration_sha256": plan_hash,
        "execution_mode": "independent_frozen_checkpoint_confirmation_evaluation",
        "allowed_inputs": [
            "A25.2a frozen checkpoint inventory",
            "A25.2a confirmation engine inventory",
            "A25.2a registered statistical analysis plan",
            "A25.1a frozen preprocessing/config contract",
            "C-MAPSS training files restricted to registered confirmation engine IDs",
        ],
        "forbidden_inputs": [
            "A25.1b selection metrics for model or checkpoint choice",
            "official C-MAPSS test files",
            "official RUL test labels",
            "unregistered checkpoint variants",
        ],
        "model_mode": "eval",
        "gradient_enabled": False,
        "optimizer_construction_allowed": False,
        "backward_calls_allowed": 0,
        "new_predictor_training_allowed": False,
        "early_stopping_allowed": False,
        "checkpoint_selection_allowed": False,
        "confirmation_passes": 1,
        "registered_anchors": list(EXPECTED_ANCHORS),
        "primary_shot": PRIMARY_SHOT,
        "secondary_shots": [1, 2],
        "required_pre_metric_checks": [
            "all_input_hashes_match",
            "all_192_checkpoint_hashes_match",
            "role_partitions_match_A25_1a",
            "confirmation_engines_disjoint_from_support_and_selection",
            "model_state_schema_matches_frozen_inventory",
            "no_official_test_path_resolved",
        ],
        "required_postconditions": [
            "evaluator_backward_calls_equals_zero",
            "evaluator_optimizer_steps_equals_zero",
            "all_expected_engine_anchor_cells_complete",
            "all_metrics_finite",
            "registered_analysis_executed_without_branching",
            "official_test_files_accessed_equals_false",
        ],
    }


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    a25a_root = resolve(args.a25_1a_output_dir)
    a25b_root = resolve(args.a25_1b_output_dir)
    script_path = resolve(args.a25_1b_script)
    output_root = resolve(args.output_dir)
    validate_distinct_roots(a25a_root, a25b_root, output_root)
    a25a = load_a25_1a(a25a_root)
    confirmations, role_counts = validate_roles(
        a25a["roles_path"], minimum_confirmation_engines=args.minimum_confirmation_engines
    )
    a25b = load_a25_1b(a25b_root, script_path, a25a["protocol"])
    plan = build_registered_plan(a25a["protocol"], a25a["statistics"])
    plan_hash = canonical_sha256(plan)
    evaluator = build_evaluator_contract(plan_hash)
    return {
        "a25a_root": a25a_root,
        "a25b_root": a25b_root,
        "output_root": output_root,
        "protocol": a25a["protocol"],
        "statistics": a25a["statistics"],
        "a25a_hashes": a25a["hashes"],
        "a25b_hashes": a25b["hashes"],
        "confirmation_inventory": confirmations,
        "role_counts": role_counts,
        "checkpoint_inventory": a25b["checkpoint_inventory"],
        "plan": plan,
        "plan_hash": plan_hash,
        "evaluator": evaluator,
    }


def preview(context: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    primary_checkpoints = sum(
        bool(row["primary_analysis_checkpoint"]) for row in context["checkpoint_inventory"]
    )
    secondary_checkpoints = len(context["checkpoint_inventory"]) - primary_checkpoints
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": dry_run,
        "registered_before_confirmation_open": True,
        "registered_primary_question": context["protocol"]["registered_primary_question"],
        "primary_shot": PRIMARY_SHOT,
        "secondary_shots": [1, 2],
        "anchors": list(EXPECTED_ANCHORS),
        "metrics": ["rmse", "nasa_score"],
        "methods": list(METHODS),
        "confirmation_partitions": len(DOMAINS) * len(EXPECTED_SPLIT_SEEDS),
        "confirmation_engine_role_rows": len(context["confirmation_inventory"]),
        "confirmation_role_counts": context["role_counts"],
        "frozen_checkpoints": len(context["checkpoint_inventory"]),
        "primary_K5_checkpoints": primary_checkpoints,
        "secondary_K1_K2_checkpoints": secondary_checkpoints,
        "checkpoint_hashes_verified": True,
        "same_architecture_pairing_verified": True,
        "compute_accounting_verified": True,
        "statistical_plan_sha256": context["plan_hash"],
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "confirmation_role_metadata_read": True,
        "confirmation_observations_opened": False,
        "confirmation_predictions_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": True,
    }


def formal_freeze(context: Mapping[str, Any]) -> dict[str, Any]:
    root: Path = context["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "confirmation_inventory": root / "experimentA25_2a_confirmation_engine_inventory.csv",
        "checkpoint_inventory": root / "experimentA25_2a_frozen_checkpoint_inventory.csv",
        "statistics": root / "experimentA25_2a_registered_statistical_analysis_plan.json",
        "evaluator": root / "experimentA25_2a_independent_evaluator_contract.json",
        "integrity": root / "experimentA25_2a_input_integrity.json",
        "decision": root / "experimentA25_2a_confirmation_decision.json",
        "manifest": root / "experimentA25_2a_manifest.json",
    }
    atomic_csv(files["confirmation_inventory"], context["confirmation_inventory"])
    atomic_csv(files["checkpoint_inventory"], context["checkpoint_inventory"])
    atomic_json(files["statistics"], context["plan"])
    atomic_json(files["evaluator"], context["evaluator"])
    integrity = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": utc_now(),
        "A25_1a_root": str(context["a25a_root"]),
        "A25_1b_root": str(context["a25b_root"]),
        "A25_1a_input_sha256": context["a25a_hashes"],
        "A25_1b_input_sha256": context["a25b_hashes"],
        "all_upstream_artifact_hashes_verified": True,
        "all_192_checkpoint_hashes_verified": True,
        "all_role_partitions_verified": True,
        "same_architecture_pairing_verified": True,
        "compute_accounting_verified": True,
        "checkpoint_tensors_opened": False,
        "confirmation_observations_opened": False,
        "confirmation_predictions_run": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(files["integrity"], integrity)
    prerequisite_hashes = {
        path.name: sha256(path)
        for key, path in files.items()
        if key in {"confirmation_inventory", "checkpoint_inventory", "statistics", "evaluator", "integrity"}
    }
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "preflight_only": True,
        "preregistered": True,
        "sealed_confirmation": True,
        "statistical_plan_frozen": True,
        "checkpoint_set_frozen": True,
        "confirmation_role_set_frozen": True,
        "frozen_checkpoints": 192,
        "primary_K5_checkpoints": 64,
        "confirmation_engine_role_rows": len(context["confirmation_inventory"]),
        "same_architecture_pairing_verified": True,
        "compute_accounting_verified": True,
        "new_predictor_training": False,
        "checkpoint_tensors_opened": False,
        "confirmation_observations_opened": False,
        "confirmation_engines_evaluated": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "artifact_sha256": dict(sorted(prerequisite_hashes.items())),
        "reason": "A25.2a froze the preregistered sealed-confirmation contract before unsealing",
        "next_action": "independent_code_review_then_implement_A25_2b_frozen_checkpoint_confirmation_evaluator",
    }
    atomic_json(files["decision"], decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "artifacts": {
            path.name: sha256(path)
            for key, path in files.items()
            if key not in {"manifest"}
        },
        "preregistered": True,
        "sealed_confirmation": True,
        "confirmation_engines_evaluated": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(files["manifest"], manifest)
    return decision


def completed_result(root: Path) -> dict[str, Any] | None:
    path = root / "experimentA25_2a_confirmation_decision.json"
    if not path.exists():
        return None
    decision = read_json(path, label="existing A25.2a decision")
    if (
        decision.get("experiment_id") == EXPERIMENT_ID
        and decision.get("complete") is True
        and decision.get("passed") is True
    ):
        return decision
    raise A252aError(
        "A25.2a output contains an incomplete/invalid decision; use a fresh output directory"
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output_root = resolve(args.output_dir)
        existing = completed_result(output_root)
        if existing is not None:
            if not args.resume:
                raise A252aError(
                    "A25.2a is already complete; pass --resume to verify and return the frozen result"
                )
            context = build_context(args)
            manifest = read_json(
                output_root / "experimentA25_2a_manifest.json", label="existing A25.2a manifest"
            )
            validate_hash_map(output_root, manifest.get("artifacts"), label="A25.2a artifacts")
            print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A25.2a] existing sealed preregistration revalidated", flush=True)
            return 0

        context = build_context(args)
        summary = preview(context, dry_run=bool(args.dry_run))
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        if args.dry_run:
            print(
                "[A25.2a] dry-run passed; checkpoint hashes and preregistration are compatible, "
                "no confirmation observation was opened",
                flush=True,
            )
            return 0
        decision = formal_freeze(context)
        print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print(
            "[A25.2a] completed sealed confirmation preregistration; confirmation remains unopened",
            flush=True,
        )
        return 0
    except A252aError as exc:
        print(f"[A25.2a] error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("[A25.2a] interrupted; confirmation was not opened", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:  # defensive: preserve a clear boundary on unexpected failures
        print(f"[A25.2a] unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
