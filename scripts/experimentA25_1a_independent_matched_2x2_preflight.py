#!/usr/bin/env python3
"""A25.1a: prospective same-architecture 2x2 meta-learning preflight.

This script trains no predictor.  It freezes a prospective pilot contract that
can distinguish the Reptile algorithm effect from the graph-architecture
effect.  The four registered arms are ordinary/Reptile x noGraph/GNN.

The preflight deliberately uses only C-MAPSS training files and immutable
upstream artifacts.  It never resolves official test files or RUL test labels.
All target-engine assignments use new registered seeds and are independent of
model outcomes, labels and trajectory length.  The later training script must
enforce the executable parameter, initialization and accounting assertions
written here before its first optimizer step.
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "experimentA25_1a"
SCRIPT_VERSION = "experimentA25_1a_independent_matched_2x2_preflight_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
METHODS = (
    "ordinary_no_graph_pft",
    "reptile_meta_no_graph",
    "ordinary_gnn_pft",
    "reptile_meta_gnn",
)
ARCHITECTURES = ("no_graph", "gnn")
ALGORITHMS = ("ordinary_pretraining", "reptile")
DEFAULT_MODEL_SEEDS = (140, 141)
DEFAULT_SPLIT_SEEDS = (7501, 7502)
DEFAULT_SHOTS = (1, 2, 5)
PRIMARY_SHOT = 5
RUL_ANCHORS = (90.0, 45.0, 15.0)
FEATURE_COLUMNS = (
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14",
    "s15", "s17", "s20", "s21", "op_setting1", "op_setting2", "op_setting3",
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
FILE_SUFFIXES = (".json", ".csv", ".pt", ".pth", ".txt", ".yaml", ".yml")


class A251aError(RuntimeError):
    """Raised when the prospective A25.1a contract cannot be locked safely."""


@dataclass(frozen=True)
class TrainingInventory:
    domain: str
    path: Path
    sha256: str
    rows: int
    engines: tuple[int, ...]
    minimum_cycle: int
    maximum_cycle: int


def parse_int_tuple(raw: str, *, name: str, allow_zero: bool = False) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} contains duplicates: {values}")
    lower = 0 if allow_zero else 1
    if any(value < lower for value in values):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {lower}")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock the A25.1 prospective, same-architecture 2x2 pilot contract."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--a24-0-output-dir",
        type=Path,
        default=Path("outputs/experimentA24_0_meta_learning_contract_preflight"),
    )
    parser.add_argument(
        "--a25-0c-output-dir",
        type=Path,
        default=Path("outputs/experimentA25_0c_stratified_checkpoint_parameter_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA25_1a_independent_matched_2x2_preflight"),
    )
    parser.add_argument("--model-seeds", default=",".join(map(str, DEFAULT_MODEL_SEEDS)))
    parser.add_argument(
        "--support-split-seeds", default=",".join(map(str, DEFAULT_SPLIT_SEEDS))
    )
    parser.add_argument("--shots", default=",".join(map(str, DEFAULT_SHOTS)))
    parser.add_argument("--primary-shot", type=int, default=PRIMARY_SHOT)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    parser.add_argument("--min-selection-engines", type=int, default=10)
    parser.add_argument("--min-confirmation-engines", type=int, default=20)
    parser.add_argument("--meta-support-engines", type=int, default=5)
    parser.add_argument("--meta-query-engines", type=int, default=5)
    parser.add_argument("--meta-validation-fraction", type=float, default=0.20)
    parser.add_argument("--episodes-per-source-domain", type=int, default=10)
    parser.add_argument("--outer-steps", type=int, default=1500)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--source-batch-size", type=int, default=64)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--inner-learning-rate", type=float, default=1e-3)
    parser.add_argument("--outer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--partition-salt", type=int, default=25110)
    parser.add_argument("--episode-salt", type=int, default=25111)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    args.model_seeds = parse_int_tuple(args.model_seeds, name="model-seeds")
    args.support_split_seeds = parse_int_tuple(
        args.support_split_seeds, name="support-split-seeds"
    )
    args.shots = parse_int_tuple(args.shots, name="shots")
    if tuple(sorted(args.shots)) != args.shots:
        raise A251aError("--shots must be strictly increasing")
    if args.primary_shot not in args.shots:
        raise A251aError("--primary-shot must occur in --shots")
    if not 0.05 <= args.selection_fraction <= 0.40:
        raise A251aError("--selection-fraction must lie in [0.05, 0.40]")
    if not 0.05 <= args.meta_validation_fraction <= 0.40:
        raise A251aError("--meta-validation-fraction must lie in [0.05, 0.40]")
    positive_names = (
        "min_selection_engines", "min_confirmation_engines", "meta_support_engines",
        "meta_query_engines", "episodes_per_source_domain", "outer_steps", "inner_steps",
        "source_batch_size", "target_epochs",
    )
    for name in positive_names:
        if int(getattr(args, name)) < 1:
            raise A251aError(f"--{name.replace('_', '-')} must be positive")
    if args.inner_learning_rate <= 0 or args.outer_learning_rate <= 0:
        raise A251aError("learning rates must be positive")
    return args


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(values: Iterable[int], *parts: Any) -> tuple[int, ...]:
    prefix = "|".join(str(part) for part in parts)
    return tuple(
        sorted(
            (int(value) for value in values),
            key=lambda value: hashlib.sha256(
                f"{prefix}|{value}".encode("utf-8")
            ).hexdigest(),
        )
    )


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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise A251aError(f"refusing to write an empty CSV: {path.name}")
    fieldnames = list(rows[0])
    for index, row in enumerate(rows):
        if set(row) != set(fieldnames):
            raise A251aError(f"CSV row {index} schema differs in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
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
        raise A251aError(f"required {label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A251aError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A251aError(f"{label} must contain a JSON object: {path}")
    return payload


def strict_false(payload: Mapping[str, Any], field: str, *, label: str) -> None:
    if payload.get(field) is not False:
        raise A251aError(f"{label} requires {field}=false; observed {payload.get(field)!r}")


def require_complete_passed(payload: Mapping[str, Any], experiment: str) -> None:
    if payload.get("experiment_id") != experiment:
        raise A251aError(
            f"expected experiment_id={experiment}; observed {payload.get('experiment_id')!r}"
        )
    if payload.get("complete") is not True or payload.get("passed") is not True:
        raise A251aError(f"{experiment} must be complete=true and passed=true")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        strict_false(payload, field, label=experiment)


def resolve_training_file(data_dir: Path, domain: str) -> Path:
    root = resolve(data_dir)
    if not root.is_dir():
        raise A251aError(f"data directory does not exist: {root}")
    name = f"train_{domain}.txt"
    direct = root / name
    matches = [direct.resolve()] if direct.is_file() else [
        item.resolve() for item in sorted(root.rglob(name)) if item.is_file()
    ]
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise A251aError(f"required training file not found below {root}: {name}")
    if len(matches) != 1:
        joined = "\n  ".join(str(item) for item in matches)
        raise A251aError(f"ambiguous {name}; pass a narrower --data-dir:\n  {joined}")
    lowered = matches[0].name.lower()
    if lowered.startswith("test_") or "rul_" in lowered:
        raise A251aError(f"refusing non-training input: {matches[0]}")
    return matches[0]


def load_training_inventory(path: Path, domain: str) -> TrainingInventory:
    per_engine: dict[int, list[int]] = defaultdict(list)
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 26:
                raise A251aError(
                    f"invalid C-MAPSS row {path}:{line_number}; expected >=26 columns, got {len(fields)}"
                )
            try:
                unit_float, cycle_float = float(fields[0]), float(fields[1])
            except ValueError as exc:
                raise A251aError(f"non-numeric unit/cycle at {path}:{line_number}") from exc
            if not unit_float.is_integer() or not cycle_float.is_integer():
                raise A251aError(f"non-integer unit/cycle at {path}:{line_number}")
            unit, cycle = int(unit_float), int(cycle_float)
            if unit < 1 or cycle < 1:
                raise A251aError(f"unit/cycle must be positive at {path}:{line_number}")
            per_engine[unit].append(cycle)
            row_count += 1
    if not per_engine or row_count == 0:
        raise A251aError(f"training file is empty: {path}")
    for unit, cycles in per_engine.items():
        if len(cycles) != len(set(cycles)):
            raise A251aError(f"duplicate cycle in {domain} engine={unit}")
        if any(right <= left for left, right in zip(cycles, cycles[1:])):
            raise A251aError(f"cycles are not strictly increasing in {domain} engine={unit}")
    all_cycles = [cycle for cycles in per_engine.values() for cycle in cycles]
    return TrainingInventory(
        domain=domain,
        path=path,
        sha256=sha256_file(path),
        rows=row_count,
        engines=tuple(sorted(per_engine)),
        minimum_cycle=min(all_cycles),
        maximum_cycle=max(all_cycles),
    )


def validate_manifest(root: Path, experiment: str) -> tuple[Path, list[dict[str, str]]]:
    path = root / f"{experiment}_manifest.json"
    payload = read_json(path, label=f"{experiment} manifest")
    if payload.get("experiment_id") != experiment:
        raise A251aError(f"manifest identity mismatch: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise A251aError(f"{experiment} manifest lacks an artifacts object")
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for name, expected in sorted(artifacts.items()):
        is_file_contract = (
            isinstance(name, str)
            and name.lower().endswith(FILE_SUFFIXES)
            and isinstance(expected, str)
            and HASH_RE.fullmatch(expected) is not None
        )
        if not is_file_contract:
            rows.append({
                "experiment": experiment,
                "artifact": str(name),
                "expected_sha256": str(expected),
                "observed_sha256": "",
                "status": "metadata_not_file",
            })
            continue
        artifact = root / name
        if not artifact.is_file():
            observed, status = "", "missing"
            failures.append(name)
        else:
            observed = sha256_file(artifact)
            status = "passed" if observed == expected else "hash_mismatch"
            if status != "passed":
                failures.append(name)
        rows.append({
            "experiment": experiment,
            "artifact": name,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "status": status,
        })
    if failures:
        raise A251aError(f"{experiment} manifest validation failed: {sorted(failures)}")
    return path, rows


def load_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise A251aError(f"required {label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise A251aError(f"{label} has no header: {path}")
            return list(reader)
    except A251aError:
        raise
    except Exception as exc:
        raise A251aError(f"failed to parse {label} {path}: {exc}") from exc


def historical_schema_contract(a25_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    decision_path = a25_root / "experimentA25_0c_confirmation_decision.json"
    integrity_path = a25_root / "experimentA25_0c_input_integrity.json"
    inventory_path = a25_root / "experimentA25_0c_checkpoint_inventory.csv"
    decision = read_json(decision_path, label="A25.0c decision")
    require_complete_passed(decision, "experimentA25_0c")
    required_true = (
        "balanced_stratified_checkpoint_scan_passed",
        "pft_parameter_scale_recovered",
        "checkpoint_parameter_accounting_complete",
        "historical_comparison_architecture_confound_present",
    )
    for field in required_true:
        if decision.get(field) is not True:
            raise A251aError(f"A25.0c requires {field}=true")
    if decision.get("training_budget_equivalence_established") is not False:
        raise A251aError("A25.0c must retain training_budget_equivalence_established=false")

    integrity = read_json(integrity_path, label="A25.0c input integrity")
    if integrity.get("all_manifest_artifact_checks_passed") is not True:
        raise A251aError("A25.0c upstream manifest checks were not all passed")
    upstream = integrity.get("upstream_complete")
    if not isinstance(upstream, dict) or not upstream or not all(value is True for value in upstream.values()):
        raise A251aError("A25.0c upstream completion inventory is incomplete")

    rows = load_csv_rows(inventory_path, label="A25.0c checkpoint inventory")
    required_columns = {
        "method", "selected_for_tensor_scan", "tensor_scan_status", "state_tensor_numel",
        "state_tensor_count", "state_schema_sha256",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise A251aError("A25.0c checkpoint inventory schema is incomplete")
    schemas: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    for row in rows:
        if row.get("selected_for_tensor_scan", "").strip().lower() != "true":
            continue
        if row.get("tensor_scan_status") != "passed":
            raise A251aError("A25.0c contains a selected checkpoint that failed tensor scan")
        method = row.get("method", "")
        if method not in {"pretrain_finetune_k", "meta_no_graph_k", "meta_gnn_k"}:
            continue
        try:
            schema = (
                int(row["state_tensor_numel"]),
                int(row["state_tensor_count"]),
                str(row["state_schema_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise A251aError(f"invalid A25.0c tensor schema row for {method}") from exc
        if schema[0] < 1 or schema[1] < 1 or HASH_RE.fullmatch(schema[2]) is None:
            raise A251aError(f"invalid A25.0c schema values for {method}: {schema}")
        schemas[method].add(schema)
    expected_methods = {"pretrain_finetune_k", "meta_no_graph_k", "meta_gnn_k"}
    if set(schemas) != expected_methods or any(len(values) != 1 for values in schemas.values()):
        raise A251aError(f"A25.0c did not establish one stable schema per method: {schemas}")
    pft = next(iter(schemas["pretrain_finetune_k"]))
    meta_no_graph = next(iter(schemas["meta_no_graph_k"]))
    meta_gnn = next(iter(schemas["meta_gnn_k"]))
    if pft != meta_gnn:
        raise A251aError("A25.0c evidence no longer supports PFT/Meta-GNN schema equality")
    if pft == meta_no_graph:
        raise A251aError("A25.0c architecture-confound boundary unexpectedly disappeared")
    result = {
        "no_graph": {
            "reference_method": "meta_no_graph_k",
            "state_tensor_numel": meta_no_graph[0],
            "state_tensor_count": meta_no_graph[1],
            "state_schema_sha256": meta_no_graph[2],
        },
        "gnn": {
            "reference_method": "pretrain_finetune_k_and_meta_gnn_k",
            "state_tensor_numel": pft[0],
            "state_tensor_count": pft[1],
            "state_schema_sha256": pft[2],
        },
    }
    hashes = {
        path.name: sha256_file(path)
        for path in (decision_path, integrity_path, inventory_path)
    }
    return result, hashes


def load_upstream(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], list[dict[str, str]]]:
    protocol_root = resolve(args.protocol_dir)
    a24_root = resolve(args.a24_0_output_dir)
    a25_root = resolve(args.a25_0c_output_dir)

    a23_decision_path = protocol_root / "experimentA23_confirmation_decision.json"
    a23_protocol_path = protocol_root / "experimentA23_few_shot_protocol.json"
    a23_roles_path = protocol_root / "experimentA23_engine_roles.csv"
    a23_decision = read_json(a23_decision_path, label="A23.0 decision")
    a23_protocol = read_json(a23_protocol_path, label="A23.0 protocol")
    require_complete_passed(a23_decision, "experimentA23_0")
    if a23_protocol.get("experiment_id") != "experimentA23_0":
        raise A251aError("A23.0 protocol identity mismatch")
    artifact_hashes = a23_decision.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise A251aError("A23.0 decision lacks artifact_sha256")
    for path in (a23_protocol_path, a23_roles_path):
        expected = artifact_hashes.get(path.name)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise A251aError(f"A23.0 artifact hash mismatch: {path}")

    a24_decision_path = a24_root / "experimentA24_0_confirmation_decision.json"
    a24_protocol_path = a24_root / "experimentA24_0_meta_protocol.json"
    a24_decision = read_json(a24_decision_path, label="A24.0 decision")
    a24_protocol = read_json(a24_protocol_path, label="A24.0 protocol")
    require_complete_passed(a24_decision, "experimentA24_0")
    if a24_protocol.get("experiment_id") != "experimentA24_0":
        raise A251aError("A24.0 protocol identity mismatch")
    if a24_decision.get("source_episode_support_query_disjoint") is not True:
        raise A251aError("A24.0 source support/query disjointness is not established")
    if a24_decision.get("target_domain_excluded_from_meta_train_domains") is not True:
        raise A251aError("A24.0 leave-one-target-domain-out boundary is not established")

    historical_schemas, a25_hashes = historical_schema_contract(a25_root)
    manifest_rows: list[dict[str, str]] = []
    for root, experiment in ((a24_root, "experimentA24_0"), (a25_root, "experimentA25_0c")):
        manifest_path = root / f"{experiment}_manifest.json"
        if manifest_path.is_file():
            _, rows = validate_manifest(root, experiment)
            manifest_rows.extend(rows)

    hashes = {
        path.name: sha256_file(path)
        for path in (
            a23_decision_path, a23_protocol_path, a23_roles_path,
            a24_decision_path, a24_protocol_path,
        )
    }
    hashes.update(a25_hashes)
    return a23_protocol, a24_protocol, historical_schemas, hashes, manifest_rows


def config_contract(config_arg: Path, a24_protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve(config_arg)
    if not path.is_file():
        raise A251aError(f"config file is missing: {path}")
    digest = sha256_file(path)
    expected = a24_protocol.get("config_sha256")
    if isinstance(expected, str) and HASH_RE.fullmatch(expected) and digest != expected:
        raise A251aError(f"config changed after A24.0: expected={expected}, observed={digest}")
    text = path.read_text(encoding="utf-8")

    def scalar(keys: Sequence[str], default: float) -> float:
        for key in keys:
            match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([-+0-9.eE]+)\s*(?:#.*)?$", text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError as exc:
                    raise A251aError(f"invalid numeric config value for {key}") from exc
        return float(default)

    window_size = int(scalar(("window_size", "seq_len"), 50))
    batch_size = int(scalar(("batch_size",), 64))
    rul_cap = float(scalar(("rul_cap", "max_rul"), 125.0))
    if window_size < 2 or batch_size < 1 or not math.isfinite(rul_cap) or rul_cap <= 0:
        raise A251aError("config contains invalid window_size/batch_size/rul_cap")
    return path, {
        "sha256": digest,
        "window_size": window_size,
        "batch_size": batch_size,
        "rul_cap": rul_cap,
    }


def expected_training_hashes(a23_protocol: Mapping[str, Any]) -> dict[str, str]:
    inventory = a23_protocol.get("training_file_inventory")
    if not isinstance(inventory, list):
        raise A251aError("A23.0 protocol lacks training_file_inventory")
    hashes: dict[str, str] = {}
    for item in inventory:
        if not isinstance(item, dict):
            raise A251aError("A23.0 training inventory contains a non-object")
        domain, digest = item.get("domain"), item.get("sha256")
        if domain not in DOMAINS or not isinstance(digest, str) or HASH_RE.fullmatch(digest) is None:
            raise A251aError(f"malformed A23.0 training inventory item: {item}")
        hashes[str(domain)] = digest
    if set(hashes) != set(DOMAINS):
        raise A251aError("A23.0 training inventory does not cover exactly FD001-FD004")
    return hashes


def build_target_roles(
    args: argparse.Namespace,
    inventories: Mapping[str, TrainingInventory],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    maximum_shot = max(args.shots)
    seen_partition_signatures: set[str] = set()
    for domain in DOMAINS:
        engines = inventories[domain].engines
        for split_seed in args.support_split_seeds:
            ordered = stable_order(
                engines, EXPERIMENT_ID, "target_partition", args.partition_salt, domain, split_seed
            )
            desired_selection = max(
                int(args.min_selection_engines), int(round(len(engines) * args.selection_fraction))
            )
            available_after_support = len(engines) - maximum_shot
            if available_after_support < args.min_selection_engines + args.min_confirmation_engines:
                raise A251aError(
                    f"{domain} has insufficient engines for max K={maximum_shot}, selection and confirmation"
                )
            selection_count = min(
                desired_selection, available_after_support - args.min_confirmation_engines
            )
            support = ordered[:maximum_shot]
            selection = ordered[maximum_shot : maximum_shot + selection_count]
            confirmation = ordered[maximum_shot + selection_count :]
            role_sets = (set(support), set(selection), set(confirmation))
            if any(role_sets[left] & role_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
                raise A251aError(f"target engine leakage for {domain}/split={split_seed}")
            if set().union(*role_sets) != set(engines):
                raise A251aError(f"target partition does not cover {domain}/split={split_seed}")
            signature = hashlib.sha256(
                json.dumps(
                    {"domain": domain, "support": support, "selection": selection, "confirmation": confirmation},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if signature in seen_partition_signatures:
                raise A251aError("duplicate prospective target partition signature")
            seen_partition_signatures.add(signature)
            previous: set[int] = set()
            shot_sets: dict[int, set[int]] = {}
            for shot in args.shots:
                current = set(support[:shot])
                if len(current) != shot or not previous.issubset(current):
                    raise A251aError(f"nested K-shot invariant failed for {domain}/split={split_seed}")
                shot_sets[shot] = current
                previous = current
            support_rank = {engine: index + 1 for index, engine in enumerate(support)}
            for engine in engines:
                if engine in role_sets[0]:
                    role = "support_pool"
                    rank: int | str = support_rank[engine]
                elif engine in role_sets[1]:
                    role, rank = "selection", ""
                else:
                    role, rank = "confirmation", ""
                row: dict[str, Any] = {
                    "target_domain": domain,
                    "support_split_seed": int(split_seed),
                    "engine_id": int(engine),
                    "role": role,
                    "support_rank": rank,
                }
                for shot in args.shots:
                    row[f"included_in_{shot}shot"] = engine in shot_sets[shot]
                rows.append(row)
            summaries.append({
                "target_domain": domain,
                "support_split_seed": int(split_seed),
                "support_engines": len(support),
                "selection_engines": len(selection),
                "confirmation_engines": len(confirmation),
                "partition_sha256": signature,
                "roles_disjoint": True,
                "shot_sets_nested": True,
                "selection_uses_labels": False,
                "selection_uses_trajectory_length": False,
            })
    expected = len(DOMAINS) * len(args.support_split_seeds)
    if len(summaries) != expected:
        raise A251aError(f"target partition count={len(summaries)}, expected={expected}")
    return rows, summaries


def build_source_episodes(
    args: argparse.Namespace,
    inventories: Mapping[str, TrainingInventory],
) -> list[dict[str, Any]]:
    required = args.meta_support_engines + args.meta_query_engines
    rows: list[dict[str, Any]] = []
    for target_domain in DOMAINS:
        source_domains = tuple(domain for domain in DOMAINS if domain != target_domain)
        for model_seed in args.model_seeds:
            for split_seed in args.support_split_seeds:
                for source_domain in source_domains:
                    ordered = stable_order(
                        inventories[source_domain].engines,
                        EXPERIMENT_ID, "source_pool", args.episode_salt,
                        target_domain, source_domain, model_seed, split_seed,
                    )
                    validation_count = max(required, int(round(len(ordered) * args.meta_validation_fraction)))
                    validation_count = min(validation_count, len(ordered) - required)
                    if validation_count < required:
                        raise A251aError(f"{source_domain} cannot form source train/validation pools")
                    validation_pool = ordered[:validation_count]
                    training_pool = ordered[validation_count:]
                    if set(training_pool) & set(validation_pool):
                        raise A251aError("source meta-train/meta-validation pools overlap")
                    for phase, pool in (("meta_train", training_pool), ("meta_validation", validation_pool)):
                        if len(pool) < required:
                            raise A251aError(f"source {phase} pool too small in {source_domain}")
                        for episode_index in range(args.episodes_per_source_domain):
                            selected = stable_order(
                                pool, EXPERIMENT_ID, "episode", args.episode_salt,
                                target_domain, source_domain, model_seed, split_seed,
                                phase, episode_index,
                            )[:required]
                            support = tuple(sorted(selected[: args.meta_support_engines]))
                            query = tuple(sorted(selected[args.meta_support_engines :]))
                            if set(support) & set(query):
                                raise A251aError("source episode support/query overlap")
                            rows.append({
                                "target_domain": target_domain,
                                "model_seed": int(model_seed),
                                "target_support_split_seed": int(split_seed),
                                "source_domain": source_domain,
                                "episode_phase": phase,
                                "episode_index": int(episode_index),
                                "meta_support_engine_ids": json.dumps(support),
                                "meta_query_engine_ids": json.dumps(query),
                                "meta_support_count": len(support),
                                "meta_query_count": len(query),
                                "source_train_pool_count": len(training_pool),
                                "source_validation_pool_count": len(validation_pool),
                                "target_domain_excluded": True,
                                "support_query_disjoint": True,
                                "source_pool_shared_by_all_four_methods": True,
                                "engine_selection_uses_labels": False,
                                "engine_selection_uses_trajectory_length": False,
                            })
    key_fields = (
        "target_domain", "model_seed", "target_support_split_seed", "source_domain",
        "episode_phase", "episode_index",
    )
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise A251aError("duplicate prospective source episode key")
    expected = (
        len(DOMAINS) * len(args.model_seeds) * len(args.support_split_seeds)
        * (len(DOMAINS) - 1) * 2 * args.episodes_per_source_domain
    )
    if len(rows) != expected:
        raise A251aError(f"source episode count={len(rows)}, expected={expected}")
    return rows


def method_contract_rows(
    args: argparse.Namespace,
    schemas: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        architecture = "no_graph" if "no_graph" in method else "gnn"
        algorithm = "reptile" if method.startswith("reptile_") else "ordinary_pretraining"
        pair = "no_graph_algorithm_effect" if architecture == "no_graph" else "gnn_algorithm_effect"
        rows.append({
            "method": method,
            "architecture": architecture,
            "algorithm": algorithm,
            "matched_pair": pair,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": json.dumps(FEATURE_COLUMNS),
            "graph_message_passing": architecture == "gnn",
            "reference_state_tensor_numel": int(schemas[architecture]["state_tensor_numel"]),
            "reference_state_tensor_count": int(schemas[architecture]["state_tensor_count"]),
            "reference_state_schema_sha256": schemas[architecture]["state_schema_sha256"],
            "runtime_exact_state_schema_assertion_required": True,
            "runtime_total_parameter_equality_within_pair_required": True,
            "runtime_trainable_parameter_equality_within_pair_required": True,
            "runtime_initialization_equality_within_pair_required": True,
            "source_engine_pool_identical_across_four_methods": True,
            "target_support_engines_identical_across_four_methods": True,
            "source_gradient_updates": int(args.outer_steps * args.inner_steps),
            "target_epochs": int(args.target_epochs),
            "selection_metrics_allowed_for_pilot_diagnostics": True,
            "confirmation_metrics_allowed_during_A25_1b": False,
        })
    for architecture in ARCHITECTURES:
        pair_rows = [row for row in rows if row["architecture"] == architecture]
        if len(pair_rows) != 2 or {row["algorithm"] for row in pair_rows} != set(ALGORITHMS):
            raise A251aError(f"incomplete same-architecture pair: {architecture}")
        schema_keys = (
            "reference_state_tensor_numel", "reference_state_tensor_count",
            "reference_state_schema_sha256",
        )
        if any(pair_rows[0][key] != pair_rows[1][key] for key in schema_keys):
            raise A251aError(f"same-architecture schema contract differs: {architecture}")
    return rows


def compute_contract_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_updates = args.outer_steps * args.inner_steps
    source_window_presentations = source_updates * args.source_batch_size
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        algorithm = "reptile" if method.startswith("reptile_") else "ordinary_pretraining"
        rows.append({
            "method": method,
            "source_gradient_updates_budget": source_updates,
            "source_batch_size": args.source_batch_size,
            "source_window_presentations_budget": source_window_presentations,
            "reptile_outer_steps": args.outer_steps if algorithm == "reptile" else 0,
            "gradient_steps_per_outer_step": args.inner_steps if algorithm == "reptile" else 0,
            "ordinary_source_optimizer_steps": source_updates if algorithm == "ordinary_pretraining" else 0,
            "target_epochs": args.target_epochs,
            "target_optimizer_steps_count_at_runtime": True,
            "target_window_presentations_count_at_runtime": True,
            "source_forward_calls_count_at_runtime": True,
            "source_backward_calls_count_at_runtime": True,
            "query_forward_calls_count_separately": True,
            "wall_time_seconds_monotonic_count_at_runtime": True,
            "peak_gpu_memory_bytes_count_at_runtime": True,
            "gpu_identity_and_software_versions_recorded": True,
            "compute_equivalence_rule": "equal_source_gradient_updates_and_window_presentations_within_architecture_pair",
            "wall_time_is_descriptive_not_equivalence_gate": True,
        })
    return rows


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    a23_protocol, a24_protocol, schemas, input_hashes, manifest_rows = load_upstream(args)
    config_path, cfg = config_contract(args.config, a24_protocol)
    historical_model_seeds = set(int(value) for value in a23_protocol.get(
        "model_seeds_reserved_for_formal_training", []
    ))
    historical_split_seeds = set(int(value) for value in a23_protocol.get("support_split_seeds", []))
    if historical_model_seeds & set(args.model_seeds):
        raise A251aError("prospective model seeds overlap the historical A23/A24 model seeds")
    if historical_split_seeds & set(args.support_split_seeds):
        raise A251aError("prospective support-split seeds overlap historical A23/A24 split seeds")

    expected_hash = expected_training_hashes(a23_protocol)
    inventories: dict[str, TrainingInventory] = {}
    for domain in DOMAINS:
        path = resolve_training_file(args.data_dir, domain)
        inventory = load_training_inventory(path, domain)
        if inventory.sha256 != expected_hash[domain]:
            raise A251aError(
                f"training file changed after A23.0 for {domain}: "
                f"expected={expected_hash[domain]}, observed={inventory.sha256}"
            )
        inventories[domain] = inventory

    target_rows, target_summaries = build_target_roles(args, inventories)
    source_rows = build_source_episodes(args, inventories)
    method_rows = method_contract_rows(args, schemas)
    compute_rows = compute_contract_rows(args)
    expected_target_partitions = len(DOMAINS) * len(args.support_split_seeds)
    expected_worker_cells = len(DOMAINS) * len(args.model_seeds) * len(args.support_split_seeds)
    expected_training_records = expected_worker_cells * len(args.shots) * len(METHODS)
    source_updates = args.outer_steps * args.inner_steps

    plan = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "registered_primary_question": (
            "Under prospectively locked target roles, identical architecture/initialization, "
            "equal source gradient-update and window-presentation budgets, does Reptile improve "
            "ordinary source pretraining plus target fine-tuning in few-shot cross-domain RUL?"
        ),
        "design": "prospective_training_file_selection_only_2x2_pilot_preflight",
        "methods": list(METHODS),
        "architectures": list(ARCHITECTURES),
        "algorithms": list(ALGORITHMS),
        "target_domains": list(DOMAINS),
        "model_seeds": list(args.model_seeds),
        "support_split_seeds": list(args.support_split_seeds),
        "shots": list(args.shots),
        "primary_shot": int(args.primary_shot),
        "registered_rul_anchors_for_later_confirmation": list(RUL_ANCHORS),
        "expected_target_partitions": expected_target_partitions,
        "completed_target_partitions": len(target_summaries),
        "expected_source_episode_records": len(source_rows),
        "completed_source_episode_records": len(source_rows),
        "expected_A25_1b_worker_cells": expected_worker_cells,
        "expected_A25_1b_training_records": expected_training_records,
        "source_gradient_updates_per_method_cell": source_updates,
        "source_window_presentations_per_method_cell": source_updates * args.source_batch_size,
        "target_epochs": args.target_epochs,
        "inner_steps": args.inner_steps,
        "outer_steps": args.outer_steps,
        "inner_learning_rate": args.inner_learning_rate,
        "outer_learning_rate": args.outer_learning_rate,
        "config_path": str(config_path),
        "config_contract": cfg,
        "feature_columns": list(FEATURE_COLUMNS),
        "training_file_inventory": [
            {
                "domain": value.domain,
                "path": str(value.path),
                "sha256": value.sha256,
                "rows": value.rows,
                "engines": len(value.engines),
                "minimum_cycle": value.minimum_cycle,
                "maximum_cycle": value.maximum_cycle,
            }
            for value in inventories.values()
        ],
        "historical_schema_boundary": schemas,
        "input_sha256": input_hashes,
        "audits": {
            "prospective_model_seeds_disjoint_from_A23_A24": True,
            "prospective_split_seeds_disjoint_from_A23_A24": True,
            "target_engine_roles_disjoint": True,
            "shot_sets_nested": True,
            "source_support_query_disjoint": True,
            "target_domain_excluded_from_source_training": True,
            "same_architecture_parameter_assertions_registered": True,
            "same_architecture_initialization_assertions_registered": True,
            "compute_accounting_registered": True,
            "equal_source_update_budget_registered": True,
            "confirmation_engines_must_not_be_evaluated_in_A25_1b": True,
            "new_predictor_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
        "interpretation_limit": (
            "A25.1a locks a prospective training-file pilot. A25.1b may inspect selection-only "
            "diagnostics but cannot create an independent confirmatory efficacy claim because "
            "historical C-MAPSS training-file outcomes have already been examined."
        ),
    }
    statistical_plan = {
        "experiment_id": EXPERIMENT_ID,
        "primary_hypothesis_family_no_graph": {
            "candidate": "reptile_meta_no_graph",
            "reference": "ordinary_no_graph_pft",
            "shot": int(args.primary_shot),
            "anchors": list(RUL_ANCHORS),
            "metrics": ["rmse", "nasa_score"],
            "decision_rule": "all_six_Holm_corrected_superiority_checks_pass",
        },
        "replication_hypothesis_family_gnn": {
            "candidate": "reptile_meta_gnn",
            "reference": "ordinary_gnn_pft",
            "shot": int(args.primary_shot),
            "anchors": list(RUL_ANCHORS),
            "metrics": ["rmse", "nasa_score"],
            "decision_rule": "separate_all_six_Holm_corrected_superiority_checks_pass",
        },
        "secondary_graph_increment": {
            "candidate": "reptile_meta_gnn",
            "reference": "reptile_meta_no_graph",
            "confirmatory": False,
        },
        "low_rul_safety_gate": {
            "anchor": 15.0,
            "metrics": ["rmse", "nasa_score"],
            "noninferiority_margin_pct": 3.0,
            "one_sided": True,
        },
        "bootstrap_design": "target_domain_then_model_seed_then_support_split_then_paired_engine",
        "bootstrap_repetitions_for_later_confirmation": 5000,
        "A25_1b_evaluation_scope": "selection_engines_only",
        "A25_1b_confirmatory_p_values_allowed": False,
        "A25_1b_policy_selection_allowed": False,
        "independent_confirmation_required_after_A25_1b": True,
    }
    context = {
        "target_rows": target_rows,
        "target_summaries": target_summaries,
        "source_rows": source_rows,
        "method_rows": method_rows,
        "compute_rows": compute_rows,
        "statistical_plan": statistical_plan,
        "manifest_validation_rows": manifest_rows,
    }
    return plan, context


def decision_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    audits = plan["audits"]
    passed = bool(
        plan["completed_target_partitions"] == plan["expected_target_partitions"]
        and plan["completed_source_episode_records"] == plan["expected_source_episode_records"]
        and all(value is True for value in audits.values() if isinstance(value, bool) and value)
        and audits["new_predictor_training"] is False
        and audits["official_test_files_accessed"] is False
        and audits["official_test_forward_run"] is False
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": passed,
        "preflight_only": True,
        "registered_primary_question": plan["registered_primary_question"],
        "methods": plan["methods"],
        "shots": plan["shots"],
        "primary_shot": plan["primary_shot"],
        "expected_target_partitions": plan["expected_target_partitions"],
        "completed_target_partitions": plan["completed_target_partitions"],
        "expected_source_episode_records": plan["expected_source_episode_records"],
        "completed_source_episode_records": plan["completed_source_episode_records"],
        "expected_A25_1b_worker_cells": plan["expected_A25_1b_worker_cells"],
        "expected_A25_1b_training_records": plan["expected_A25_1b_training_records"],
        "prospective_seeds_disjoint_from_A23_A24": True,
        "same_architecture_parameter_and_initialization_contract_locked": True,
        "equal_source_gradient_update_and_window_presentation_budget_locked": True,
        "runtime_compute_accounting_required": True,
        "A25_1b_selection_only": True,
        "A25_1b_confirmation_engines_evaluated": False,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A25.1a locked the prospective same-architecture, compute-accounted 2x2 pilot contract"
            if passed else
            "A25.1a did not satisfy every registered preflight invariant"
        ),
        "interpretation_limit": plan["interpretation_limit"],
        "next_action": (
            "implement_A25_1b_same_architecture_compute_accounted_selection_only_pilot"
            if passed else "repair_A25_1a_before_any_predictor_training"
        ),
    }


def summary_payload(plan: Mapping[str, Any], decision: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": dry_run,
        "methods": plan["methods"],
        "model_seeds": plan["model_seeds"],
        "support_split_seeds": plan["support_split_seeds"],
        "shots": plan["shots"],
        "primary_shot": plan["primary_shot"],
        "expected_target_partitions": plan["expected_target_partitions"],
        "expected_source_episode_records": plan["expected_source_episode_records"],
        "expected_A25_1b_worker_cells": plan["expected_A25_1b_worker_cells"],
        "expected_A25_1b_training_records": plan["expected_A25_1b_training_records"],
        "source_gradient_updates_per_method_cell": plan["source_gradient_updates_per_method_cell"],
        "same_architecture_pairing": True,
        "confirmation_engines_evaluated": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "passed": decision["passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = resolve(args.output_dir)
    decision_path = output / "experimentA25_1a_confirmation_decision.json"
    if decision_path.is_file():
        prior = read_json(decision_path, label="existing A25.1a decision")
        if args.resume and prior.get("complete") is True and prior.get("passed") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A25.1a] resume: existing complete decision returned", flush=True)
            return 0
        raise A251aError(
            f"output already contains a final decision: {decision_path}; use --resume or a new directory"
        )

    plan, context = build_plan(args)
    decision = decision_from_plan(plan)
    print(json.dumps(summary_payload(plan, decision, args.dry_run), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print(
            "[A25.1a] dry-run passed; contracts and data roles were validated, no predictor was trained",
            flush=True,
        )
        return 0 if decision["passed"] else 2

    if output.exists() and any(output.iterdir()):
        raise A251aError(
            f"non-empty output directory without a final decision: {output}; choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol": output / "experimentA25_1a_protocol.json",
        "target_roles": output / "experimentA25_1a_target_engine_roles.csv",
        "target_audit": output / "experimentA25_1a_target_partition_audit.csv",
        "source_tasks": output / "experimentA25_1a_source_episode_inventory.csv",
        "methods": output / "experimentA25_1a_method_contract.csv",
        "compute": output / "experimentA25_1a_compute_contract.csv",
        "statistics": output / "experimentA25_1a_statistical_analysis_plan.json",
        "integrity": output / "experimentA25_1a_input_integrity.json",
        "decision": decision_path,
    }
    atomic_json(paths["protocol"], plan)
    atomic_csv(paths["target_roles"], context["target_rows"])
    atomic_csv(paths["target_audit"], context["target_summaries"])
    atomic_csv(paths["source_tasks"], context["source_rows"])
    atomic_csv(paths["methods"], context["method_rows"])
    atomic_csv(paths["compute"], context["compute_rows"])
    atomic_json(paths["statistics"], context["statistical_plan"])
    atomic_json(paths["integrity"], {
        "experiment_id": EXPERIMENT_ID,
        "input_sha256": plan["input_sha256"],
        "upstream_manifest_validation": context["manifest_validation_rows"],
        "training_file_sha256": {
            item["domain"]: item["sha256"] for item in plan["training_file_inventory"]
        },
        "config_sha256": plan["config_contract"]["sha256"],
        "all_required_inputs_validated": True,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })
    decision["artifact_sha256"] = {
        path.name: sha256_file(path)
        for key, path in paths.items() if key != "decision"
    }
    atomic_json(paths["decision"], decision)
    manifest_path = output / "experimentA25_1a_manifest.json"
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print("[A25.1a] completed prospective matched 2x2 preflight", flush=True)
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A251aError as exc:
        print(f"[A25.1a] error: {exc.__class__.__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("[A25.1a] interrupted by user", file=sys.stderr, flush=True)
        raise SystemExit(130)
