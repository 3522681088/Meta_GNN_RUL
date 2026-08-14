#!/usr/bin/env python3
"""Experiment A23.0: engine-level few-shot protocol preflight.

This script deliberately does not train a predictor.  It creates the immutable
engine-level partitions required before comparing ordinary fine-tuning with a
real meta-learning method.  Only C-MAPSS ``train_FD00x.txt`` files are opened;
official test files and RUL_FD00x.txt files are never read.

Primary invariants
------------------
* 1/2/5/10/20-shot sets are nested and are defined by complete target engines.
* support, selection and confirmation engines are mutually disjoint.
* the held-out target domain never appears among its meta-training domains.
* every partition is reproducible from explicit, registered seeds.
* no label- or trajectory-dependent selection is used to choose engines.

The output is a protocol artifact, not efficacy evidence.  Its successful
completion authorizes the later A23 fine-tuning baseline implementation; it
does not establish a meta-learning, GNN, fault-detection or RUL improvement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ID = "experimentA23_0"
SCRIPT_VERSION = "experimentA23_few_shot_protocol_preflight_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
SHOT_COUNTS = (0, 1, 2, 5, 10, 20)
PRIMARY_SHOT = 5
MODEL_SEEDS = (130, 131, 132, 133, 134)
SUPPORT_SPLIT_SEEDS = (7101, 7102, 7103, 7104, 7105)
SELECTION_FRACTION = 0.20
MIN_SELECTION_ENGINES = 10
MIN_CONFIRMATION_ENGINES = 20


class ProtocolError(RuntimeError):
    """Raised when a registered few-shot invariant cannot be satisfied."""


@dataclass(frozen=True)
class DomainInventory:
    domain: str
    path: Path
    sha256: str
    rows: int
    engines: tuple[int, ...]
    minimum_cycle: int
    maximum_cycle: int


def parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values: {parsed}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and audit the registered engine-level A23 few-shot protocol."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing C-MAPSS train_FD001.txt ... train_FD004.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--shot-counts",
        default=",".join(str(x) for x in SHOT_COUNTS),
        help="Registered target-engine budgets. Must include 0 and primary shot 5.",
    )
    parser.add_argument(
        "--model-seeds",
        default=",".join(str(x) for x in MODEL_SEEDS),
        help="Reserved model seeds for the later formal experiment.",
    )
    parser.add_argument(
        "--support-split-seeds",
        default=",".join(str(x) for x in SUPPORT_SPLIT_SEEDS),
        help="Seeds that determine nested target-engine partitions.",
    )
    parser.add_argument("--selection-fraction", type=float, default=SELECTION_FRACTION)
    parser.add_argument("--min-selection-engines", type=int, default=MIN_SELECTION_ENGINES)
    parser.add_argument("--min-confirmation-engines", type=int, default=MIN_CONFIRMATION_ENGINES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing protocol artifacts.",
    )
    args = parser.parse_args(argv)

    args.shot_counts = parse_int_list(args.shot_counts, name="shot-counts")
    args.model_seeds = parse_int_list(args.model_seeds, name="model-seeds")
    args.support_split_seeds = parse_int_list(
        args.support_split_seeds, name="support-split-seeds"
    )
    if tuple(sorted(args.shot_counts)) != args.shot_counts:
        raise ProtocolError("shot-counts must be strictly increasing")
    if args.shot_counts[0] != 0:
        raise ProtocolError("shot-counts must begin with 0-shot")
    if PRIMARY_SHOT not in args.shot_counts:
        raise ProtocolError(f"registered primary shot={PRIMARY_SHOT} is missing")
    if not 0.0 < args.selection_fraction < 0.5:
        raise ProtocolError("selection-fraction must be between 0 and 0.5")
    if args.min_selection_engines < 1 or args.min_confirmation_engines < 1:
        raise ProtocolError("minimum selection/confirmation engine counts must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_training_file(data_dir: Path, domain: str) -> Path:
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"data directory does not exist: {root}")
    name = f"train_{domain}.txt"
    direct = root / name
    candidates = [direct] if direct.is_file() else sorted(root.rglob(name))
    candidates = [path.resolve() for path in candidates if path.is_file()]
    unique = tuple(dict.fromkeys(candidates))
    if not unique:
        raise ProtocolError(f"required training file not found below {root}: {name}")
    if len(unique) != 1:
        joined = "\n  ".join(str(path) for path in unique)
        raise ProtocolError(
            f"training file is ambiguous for {domain}; pass a narrower --data-dir:\n  {joined}"
        )
    if unique[0].name.lower().startswith("test_") or "rul_" in unique[0].name.lower():
        raise ProtocolError(f"refusing non-training input: {unique[0]}")
    return unique[0]


def load_inventory(data_dir: Path, domain: str) -> DomainInventory:
    path = resolve_training_file(data_dir, domain)
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception as exc:  # pandas supplies actionable parser detail
        raise ProtocolError(f"failed to parse {path}: {exc}") from exc
    if frame.empty or frame.shape[1] < 2:
        raise ProtocolError(f"invalid C-MAPSS training table: {path}, shape={frame.shape}")
    units = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    cycles = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    if units.isna().any() or cycles.isna().any():
        raise ProtocolError(f"unit/cycle columns contain non-numeric values: {path}")
    if not np.allclose(units.to_numpy(), np.round(units.to_numpy())):
        raise ProtocolError(f"unit identifiers are not integers: {path}")
    if not np.allclose(cycles.to_numpy(), np.round(cycles.to_numpy())):
        raise ProtocolError(f"cycle identifiers are not integers: {path}")
    engine_ids = tuple(sorted(int(value) for value in units.unique()))
    if len(engine_ids) != len(set(engine_ids)) or not engine_ids:
        raise ProtocolError(f"could not obtain unique engine identifiers: {path}")
    grouped = frame.assign(_unit=units.astype(int), _cycle=cycles.astype(int)).groupby("_unit")
    for unit_id, group in grouped:
        ordered = group["_cycle"].to_numpy()
        if len(np.unique(ordered)) != len(ordered):
            raise ProtocolError(f"duplicate cycle for {domain} unit={unit_id}")
        if np.any(np.diff(ordered) <= 0):
            raise ProtocolError(f"non-causal cycle order for {domain} unit={unit_id}")
    return DomainInventory(
        domain=domain,
        path=path,
        sha256=sha256_file(path),
        rows=int(len(frame)),
        engines=engine_ids,
        minimum_cycle=int(cycles.min()),
        maximum_cycle=int(cycles.max()),
    )


def domain_code(domain: str) -> int:
    if domain not in DOMAINS:
        raise ProtocolError(f"unknown domain: {domain}")
    return int(domain[-3:])


def create_partition(
    inventory: DomainInventory,
    split_seed: int,
    shot_counts: tuple[int, ...],
    selection_fraction: float,
    min_selection: int,
    min_confirmation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    maximum_shot = max(shot_counts)
    n_engines = len(inventory.engines)
    desired_selection = max(min_selection, int(round(n_engines * selection_fraction)))
    available_after_support = n_engines - maximum_shot
    if available_after_support < min_selection + min_confirmation:
        raise ProtocolError(
            f"{inventory.domain} has {n_engines} engines, insufficient for max-shot={maximum_shot}, "
            f"selection>={min_selection}, confirmation>={min_confirmation}"
        )
    selection_count = min(desired_selection, available_after_support - min_confirmation)
    if selection_count < min_selection:
        raise ProtocolError(f"selection pool too small for {inventory.domain}")

    sequence = np.random.SeedSequence([int(split_seed), domain_code(inventory.domain), 2300])
    rng = np.random.default_rng(sequence)
    shuffled = np.asarray(inventory.engines, dtype=np.int64)
    rng.shuffle(shuffled)
    support = tuple(int(x) for x in shuffled[:maximum_shot])
    selection = tuple(
        int(x) for x in shuffled[maximum_shot : maximum_shot + selection_count]
    )
    confirmation = tuple(int(x) for x in shuffled[maximum_shot + selection_count :])

    support_set, selection_set, confirmation_set = set(support), set(selection), set(confirmation)
    if support_set & selection_set or support_set & confirmation_set or selection_set & confirmation_set:
        raise ProtocolError(f"engine leakage detected for {inventory.domain}/{split_seed}")
    if support_set | selection_set | confirmation_set != set(inventory.engines):
        raise ProtocolError(f"partition does not cover every engine for {inventory.domain}/{split_seed}")

    shot_members: dict[str, list[int]] = {}
    previous: set[int] = set()
    for shot in shot_counts:
        members = set(support[:shot]) if shot > 0 else set()
        if not previous.issubset(members):
            raise ProtocolError(f"shot sets are not nested for {inventory.domain}/{split_seed}")
        if len(members) != shot:
            raise ProtocolError(f"shot={shot} does not contain exactly {shot} engines")
        shot_members[str(shot)] = sorted(members)
        previous = members

    meta_train_domains = [domain for domain in DOMAINS if domain != inventory.domain]
    if inventory.domain in meta_train_domains or len(meta_train_domains) != len(DOMAINS) - 1:
        raise ProtocolError(f"invalid leave-one-domain-out set for {inventory.domain}")

    protocol = {
        "target_domain": inventory.domain,
        "meta_train_domains": meta_train_domains,
        "support_split_seed": int(split_seed),
        "partition_rng": "numpy.SeedSequence([split_seed, domain_code, 2300])",
        "maximum_shot": int(maximum_shot),
        "shot_engine_ids": shot_members,
        "support_pool_engine_ids_ordered": list(support),
        "selection_engine_ids": list(selection),
        "confirmation_engine_ids": list(confirmation),
        "support_engines": len(support),
        "selection_engines": len(selection),
        "confirmation_engines": len(confirmation),
        "engine_partition_disjoint": True,
        "shot_sets_nested": True,
        "selection_is_label_independent": True,
    }

    rows: list[dict[str, Any]] = []
    rank = {engine_id: index + 1 for index, engine_id in enumerate(support)}
    for engine_id in inventory.engines:
        if engine_id in support_set:
            role = "support_pool"
            support_rank: int | str = rank[engine_id]
        elif engine_id in selection_set:
            role = "selection"
            support_rank = ""
        else:
            role = "confirmation"
            support_rank = ""
        rows.append(
            {
                "target_domain": inventory.domain,
                "support_split_seed": int(split_seed),
                "engine_id": int(engine_id),
                "role": role,
                "support_rank": support_rank,
                "included_in_0shot": False,
                "included_in_1shot": bool(engine_id in set(support[:1])),
                "included_in_2shot": bool(engine_id in set(support[:2])),
                "included_in_5shot": bool(engine_id in set(support[:5])),
                "included_in_10shot": bool(engine_id in set(support[:10])),
                "included_in_20shot": bool(engine_id in set(support[:20])),
            }
        )
    return protocol, rows


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventories = [load_inventory(args.data_dir, domain) for domain in DOMAINS]
    protocol_sets: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for inventory in inventories:
        for split_seed in args.support_split_seeds:
            protocol, rows = create_partition(
                inventory=inventory,
                split_seed=split_seed,
                shot_counts=args.shot_counts,
                selection_fraction=float(args.selection_fraction),
                min_selection=int(args.min_selection_engines),
                min_confirmation=int(args.min_confirmation_engines),
            )
            protocol_sets.append(protocol)
            role_rows.extend(rows)

    expected_protocol_sets = len(DOMAINS) * len(args.support_split_seeds)
    if len(protocol_sets) != expected_protocol_sets:
        raise ProtocolError("protocol-set count mismatch")
    expected_formal_training_cells = (
        len(DOMAINS)
        * len(args.model_seeds)
        * len(args.support_split_seeds)
        * len(args.shot_counts)
        * 5
    )
    plan = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": (
            "Can a leakage-free, engine-level nested k-shot protocol be locked before comparing "
            "ordinary target fine-tuning with Meta-noGraph and Meta-GNN on unseen C-MAPSS domains?"
        ),
        "created_at_utc": utc_now(),
        "task": "few_shot_cross_domain_rul_regression_protocol",
        "primary_shot": PRIMARY_SHOT,
        "shot_counts": list(args.shot_counts),
        "model_seeds_reserved_for_formal_training": list(args.model_seeds),
        "support_split_seeds": list(args.support_split_seeds),
        "target_domains": list(DOMAINS),
        "meta_training_design": "leave_one_domain_out",
        "support_unit": "complete_target_engine",
        "methods_reserved_for_formal_comparison": [
            "source_only",
            "scratch_k",
            "pretrain_finetune_k",
            "meta_no_graph_k",
            "meta_gnn_k",
        ],
        "registered_primary_comparison": "meta_gnn_5shot_vs_pretrain_finetune_5shot",
        "expected_protocol_sets": expected_protocol_sets,
        "completed_protocol_sets": len(protocol_sets),
        "expected_formal_training_cells": expected_formal_training_cells,
        "selection_fraction_requested": float(args.selection_fraction),
        "minimum_selection_engines": int(args.min_selection_engines),
        "minimum_confirmation_engines": int(args.min_confirmation_engines),
        "training_file_inventory": [
            {
                "domain": item.domain,
                "path": str(item.path),
                "sha256": item.sha256,
                "rows": item.rows,
                "engines": len(item.engines),
                "minimum_cycle": item.minimum_cycle,
                "maximum_cycle": item.maximum_cycle,
            }
            for item in inventories
        ],
        "protocol_sets": protocol_sets,
        "audits": {
            "engine_level_partitioning": True,
            "support_selection_confirmation_disjoint": True,
            "shot_sets_nested": True,
            "target_domain_excluded_from_meta_train_domains": True,
            "selection_is_label_and_trajectory_independent": True,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
            "new_predictor_training": False,
        },
    }
    return plan, role_rows


def decision_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    audits = plan["audits"]
    passed = bool(
        plan["completed_protocol_sets"] == plan["expected_protocol_sets"]
        and audits["engine_level_partitioning"]
        and audits["support_selection_confirmation_disjoint"]
        and audits["shot_sets_nested"]
        and audits["target_domain_excluded_from_meta_train_domains"]
        and audits["selection_is_label_and_trajectory_independent"]
        and not audits["official_test_files_accessed"]
        and not audits["official_test_forward_run"]
        and not audits["new_predictor_training"]
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": passed,
        "registered_primary_question": plan["registered_primary_question"],
        "primary_shot": plan["primary_shot"],
        "shot_counts": plan["shot_counts"],
        "expected_protocol_sets": plan["expected_protocol_sets"],
        "completed_protocol_sets": plan["completed_protocol_sets"],
        "engine_level_partitioning": audits["engine_level_partitioning"],
        "support_selection_confirmation_disjoint": audits[
            "support_selection_confirmation_disjoint"
        ],
        "shot_sets_nested": audits["shot_sets_nested"],
        "target_domain_excluded_from_meta_train_domains": audits[
            "target_domain_excluded_from_meta_train_domains"
        ],
        "selection_is_label_and_trajectory_independent": audits[
            "selection_is_label_and_trajectory_independent"
        ],
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A23.0 locked a leakage-free engine-level nested few-shot protocol"
            if passed
            else "A23.0 protocol did not satisfy every registered integrity criterion"
        ),
        "interpretation_limit": (
            "This preflight creates data roles only; it does not establish meta-learning, "
            "GNN, fault-detection or RUL efficacy."
        ),
        "next_action": (
            "implement_A23_pretrain_finetune_kshot_baseline_using_the_locked_protocol"
            if passed
            else "repair_protocol_before_any_predictor_training"
        ),
    }


def print_summary(plan: dict[str, Any], decision: dict[str, Any], *, dry_run: bool) -> None:
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": dry_run,
        "data_files": [item["path"] for item in plan["training_file_inventory"]],
        "training_file_sha256": {
            item["domain"]: item["sha256"] for item in plan["training_file_inventory"]
        },
        "shot_counts": plan["shot_counts"],
        "primary_shot": plan["primary_shot"],
        "protocol_sets": plan["completed_protocol_sets"],
        "reserved_formal_training_cells": plan["expected_formal_training_cells"],
        "passed": decision["passed"],
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, role_rows = build_plan(args)
    decision = decision_from_plan(plan)
    if args.dry_run:
        print_summary(plan, decision, dry_run=True)
        print("[A23.0] dry-run completed; no predictor was trained and no official test file was read")
        return 0 if decision["passed"] else 2

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "experimentA23_few_shot_protocol.json"
    roles_path = output / "experimentA23_engine_roles.csv"
    manifest_path = output / "experimentA23_manifest.json"
    decision_path = output / "experimentA23_confirmation_decision.json"

    atomic_write_json(protocol_path, plan)
    atomic_write_csv(
        roles_path,
        role_rows,
        fieldnames=(
            "target_domain",
            "support_split_seed",
            "engine_id",
            "role",
            "support_rank",
            "included_in_0shot",
            "included_in_1shot",
            "included_in_2shot",
            "included_in_5shot",
            "included_in_10shot",
            "included_in_20shot",
        ),
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "output_dir": str(output),
        "artifacts": {
            protocol_path.name: sha256_file(protocol_path),
            roles_path.name: sha256_file(roles_path),
        },
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_write_json(manifest_path, manifest)
    decision["artifact_sha256"] = {
        protocol_path.name: sha256_file(protocol_path),
        roles_path.name: sha256_file(roles_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    atomic_write_json(decision_path, decision)
    print_summary(plan, decision, dry_run=False)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"[A23.0] protocol error: {exc}", file=sys.stderr)
        raise SystemExit(2)

