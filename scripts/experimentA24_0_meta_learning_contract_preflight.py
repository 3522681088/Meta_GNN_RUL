#!/usr/bin/env python3
"""A24.0: lock and audit the few-shot meta-learning comparison contract.

This script creates no predictor and trains nothing.  It converts the immutable
A23.0 target-domain protocol into an equally immutable source-domain episodic
meta-learning contract for the later Meta-noGraph and Meta-GNN experiments.

It deliberately validates only data roles, hashes, episode construction and
declared fairness constraints.  Exact executable parameter-count equality is
deferred to A24.1, where both concrete model implementations will exist and
must be asserted before any optimiser step.  This avoids falsely claiming a
parameter-count result at preflight time.

No C-MAPSS official test or RUL test-label file is resolved or read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "experimentA24_0"
SCRIPT_VERSION = "experimentA24_0_meta_learning_contract_preflight_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
FORMAL_MODEL_SEEDS = (130, 131, 132, 133, 134)
FORMAL_SPLIT_SEEDS = (7101, 7102, 7103, 7104, 7105)
FORMAL_SHOTS = (1, 2, 5, 10, 20)
PRIMARY_SHOT = 5
RUL_ANCHORS = (90.0, 45.0, 15.0)
FEATURE_COLUMNS = (
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14",
    "s15", "s17", "s20", "s21", "op_setting1", "op_setting2", "op_setting3",
)
METHODS = ("meta_no_graph_k", "meta_gnn_k")


class A240Error(RuntimeError):
    """Raised when A24.0 cannot honour its registered preflight contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A24.0 meta-learning protocol/episode fairness preflight"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("outputs/experimentA23_few_shot_protocol_preflight"),
    )
    parser.add_argument(
        "--a23-4-output-dir",
        type=Path,
        default=Path("outputs/experimentA23_4_formal_causal_anchor_evaluation_and_hierarchical_inference"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimentA24_0_meta_learning_contract_preflight"),
    )
    parser.add_argument("--meta-support-engines", type=int, default=5)
    parser.add_argument("--meta-query-engines", type=int, default=5)
    parser.add_argument("--episodes-per-source-domain", type=int, default=10)
    parser.add_argument("--meta-validation-fraction", type=float, default=0.20)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--outer-steps", type=int, default=1500)
    parser.add_argument("--outer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--episode-seed", type=int, default=240000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.meta_support_engines < 1 or args.meta_query_engines < 1:
        raise A240Error("meta support/query engine counts must be positive")
    if args.episodes_per_source_domain < 1:
        raise A240Error("episodes-per-source-domain must be positive")
    if not 0.05 <= args.meta_validation_fraction <= 0.40:
        raise A240Error("meta-validation-fraction must lie in [0.05, 0.40]")
    if args.inner_steps < 1 or args.outer_steps < 1 or args.outer_learning_rate <= 0:
        raise A240Error("inner/outer steps and outer learning rate must be positive")
    return args


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def atomic_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A240Error(f"required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A240Error(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A240Error(f"JSON artifact must contain an object: {path}")
    return value


def strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise A240Error(f"{field} is not a strict boolean: {value!r}")


def domain_code(domain: str) -> int:
    if domain not in DOMAINS:
        raise A240Error(f"unknown C-MAPSS domain: {domain}")
    return int(domain[-3:])


def resolve_training_file(data_root: Path, domain: str) -> Path:
    root = resolve(data_root)
    if not root.is_dir():
        raise A240Error(f"data directory is missing: {root}")
    name = f"train_{domain}.txt"
    direct = root / name
    matches = [direct] if direct.is_file() else sorted(root.rglob(name))
    matches = [path.resolve() for path in matches if path.is_file()]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise A240Error(
            f"expected exactly one {name} below {root}; found {len(matches)}: {matches}"
        )
    path = matches[0]
    lowered = path.name.lower()
    if lowered.startswith("test_") or "rul_" in lowered:
        raise A240Error(f"refusing non-training file: {path}")
    return path


def load_engine_inventory(path: Path, domain: str) -> tuple[int, ...]:
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1])
    except Exception as exc:
        raise A240Error(f"failed to parse training file {path}: {exc}") from exc
    if frame.empty:
        raise A240Error(f"training file is empty: {path}")
    units = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    cycles = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    if units.isna().any() or cycles.isna().any():
        raise A240Error(f"non-numeric unit/cycle values in {path}")
    if not np.allclose(units, np.round(units)) or not np.allclose(cycles, np.round(cycles)):
        raise A240Error(f"non-integer unit/cycle identifiers in {path}")
    table = pd.DataFrame({"unit": units.astype(int), "cycle": cycles.astype(int)})
    if table.duplicated(["unit", "cycle"]).any():
        raise A240Error(f"duplicate unit/cycle observations in {domain}")
    for unit, group in table.groupby("unit", sort=True):
        cycles_now = group.sort_values("cycle", kind="stable")["cycle"].to_numpy()
        if np.any(np.diff(cycles_now) <= 0):
            raise A240Error(f"non-increasing cycles for {domain} engine={unit}")
    engines = tuple(sorted(int(value) for value in table["unit"].unique()))
    if len(engines) < 2:
        raise A240Error(f"source domain lacks enough engines: {domain}")
    return engines


def load_protocol(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str], dict[str, str]]:
    root = resolve(args.protocol_dir)
    protocol_path = root / "experimentA23_few_shot_protocol.json"
    roles_path = root / "experimentA23_engine_roles.csv"
    decision_path = root / "experimentA23_confirmation_decision.json"
    protocol = read_json(protocol_path)
    decision = read_json(decision_path)
    if protocol.get("experiment_id") != "experimentA23_0":
        raise A240Error("protocol artifact is not A23.0")
    if decision.get("experiment_id") != "experimentA23_0":
        raise A240Error("protocol decision is not A23.0")
    if not (decision.get("complete") is True and decision.get("passed") is True):
        raise A240Error("A23.0 protocol must be complete and passed")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A240Error(f"A23.0 official-test boundary violated: {field}=true")
    expected_artifacts = decision.get("artifact_sha256")
    if not isinstance(expected_artifacts, dict):
        raise A240Error("A23.0 decision lacks artifact_sha256")
    for path in (protocol_path, roles_path):
        expected = expected_artifacts.get(path.name)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise A240Error(f"A23.0 hash mismatch: {path}")
    try:
        roles = pd.read_csv(roles_path)
    except Exception as exc:
        raise A240Error(f"failed to parse A23.0 role table: {exc}") from exc
    required = {"target_domain", "support_split_seed", "engine_id", "role", "support_rank"}
    if missing := required - set(roles.columns):
        raise A240Error(f"A23.0 roles lack columns: {sorted(missing)}")
    roles["target_domain"] = roles["target_domain"].astype(str)
    roles["support_split_seed"] = pd.to_numeric(
        roles["support_split_seed"], errors="raise"
    ).astype(int)
    roles["engine_id"] = pd.to_numeric(roles["engine_id"], errors="raise").astype(int)
    if tuple(sorted(roles["target_domain"].unique())) != DOMAINS:
        raise A240Error("A23.0 role table does not cover exactly the four domains")
    if set(roles["role"].astype(str)) != {"support_pool", "selection", "confirmation"}:
        raise A240Error("A23.0 role table contains an unexpected role")
    hashes = {
        protocol_path.name: sha256_file(protocol_path),
        roles_path.name: sha256_file(roles_path),
        decision_path.name: sha256_file(decision_path),
    }
    training_hashes: dict[str, str] = {}
    inventory = protocol.get("training_file_inventory")
    if not isinstance(inventory, list):
        raise A240Error("A23.0 protocol lacks training_file_inventory")
    for item in inventory:
        if not isinstance(item, dict) or not {"domain", "sha256"} <= set(item):
            raise A240Error("malformed A23.0 training-file inventory")
        training_hashes[str(item["domain"])] = str(item["sha256"])
    if set(training_hashes) != set(DOMAINS):
        raise A240Error("A23.0 training-file inventory is incomplete")
    if tuple(int(x) for x in protocol.get("model_seeds_reserved_for_formal_training", ())) != FORMAL_MODEL_SEEDS:
        raise A240Error("A23.0 model-seed contract differs from A24.0")
    if tuple(int(x) for x in protocol.get("support_split_seeds", ())) != FORMAL_SPLIT_SEEDS:
        raise A240Error("A23.0 support-split contract differs from A24.0")
    if tuple(int(x) for x in protocol.get("shot_counts", ())) != (0,) + FORMAL_SHOTS:
        raise A240Error("A23.0 shot contract differs from A24.0")
    return protocol, roles, hashes, training_hashes


def verify_target_roles(roles: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for split in FORMAL_SPLIT_SEEDS:
            frame = roles.loc[
                (roles["target_domain"] == domain)
                & (roles["support_split_seed"] == split)
            ].copy()
            support_frame = frame.loc[frame["role"] == "support_pool"].copy()
            ranks = pd.to_numeric(support_frame["support_rank"], errors="raise")
            if ranks.isna().any() or not np.allclose(ranks, np.floor(ranks)):
                raise A240Error(f"invalid support ranks for {domain}/split={split}")
            support_frame["support_rank"] = ranks.astype(int)
            sets = {
                "support_pool": set(support_frame["engine_id"].tolist()),
                "selection": set(frame.loc[frame["role"] == "selection", "engine_id"].tolist()),
                "confirmation": set(frame.loc[frame["role"] == "confirmation", "engine_id"].tolist()),
            }
            if (
                sets["support_pool"] & sets["selection"]
                or sets["support_pool"] & sets["confirmation"]
                or sets["selection"] & sets["confirmation"]
            ):
                raise A240Error(f"target engine role overlap: {domain}/split={split}")
            if len(sets["support_pool"]) != 20 or len(sets["selection"]) < 10 or len(sets["confirmation"]) < 20:
                raise A240Error(f"target role cardinality mismatch: {domain}/split={split}")
            for shot in FORMAL_SHOTS:
                nested = set(
                    support_frame.loc[support_frame["support_rank"] <= shot, "engine_id"].tolist()
                )
                if len(nested) != shot:
                    raise A240Error(f"A23.0 does not provide exactly K={shot}: {domain}/split={split}")
            rows.append({
                "target_domain": domain,
                "support_split_seed": split,
                "support_pool_engines": len(sets["support_pool"]),
                "selection_engines": len(sets["selection"]),
                "confirmation_engines": len(sets["confirmation"]),
                "all_target_role_sets_disjoint": True,
                "nested_shot_sets_verified": True,
                "target_engines_allowed_in_meta_training": False,
            })
    return rows


def load_a234_boundary(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    root = resolve(args.a23_4_output_dir)
    decision_path = root / "experimentA23_4_confirmation_decision.json"
    manifest_path = root / "experimentA23_4_manifest.json"
    decision = read_json(decision_path)
    manifest = read_json(manifest_path)
    if decision.get("experiment_id") != "experimentA23_4" or decision.get("complete") is not True:
        raise A240Error("A23.4 must be complete before A24.0 is locked")
    for field in ("official_test_files_accessed", "official_test_forward_run"):
        if strict_bool(decision.get(field), field=field):
            raise A240Error(f"A23.4 official-test boundary violated: {field}=true")
    if manifest.get("experiment_id") != "experimentA23_4":
        raise A240Error("A23.4 manifest identity mismatch")
    return decision, {
        decision_path.name: sha256_file(decision_path),
        manifest_path.name: sha256_file(manifest_path),
    }


def load_config(path_arg: Path) -> tuple[dict[str, Any], Path]:
    path = resolve(path_arg)
    if not path.is_file():
        raise A240Error(f"config is missing: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A240Error(f"failed to parse config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise A240Error("config must be a mapping")
    cfg = dict(cfg)
    cfg["batch_size"] = int(cfg.get("batch_size", 64))
    cfg["window_size"] = int(cfg.get("window_size", cfg.get("seq_len", 30)))
    cfg["rul_cap"] = float(cfg.get("rul_cap", cfg.get("max_rul", 125.0)))
    cfg["inner_lr"] = float(cfg.get("inner_lr", cfg.get("learning_rate", 1e-3)))
    if cfg["batch_size"] < 1 or cfg["window_size"] < 2 or cfg["rul_cap"] <= 0 or cfg["inner_lr"] <= 0:
        raise A240Error("config contains invalid batch/window/RUL/inner learning-rate values")
    return cfg, path


def build_source_episode_rows(
    args: argparse.Namespace,
    inventories: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = args.meta_support_engines + args.meta_query_engines
    for target_domain in DOMAINS:
        source_domains = tuple(domain for domain in DOMAINS if domain != target_domain)
        if target_domain in source_domains or len(source_domains) != 3:
            raise A240Error(f"invalid leave-one-domain-out source list for {target_domain}")
        for model_seed in FORMAL_MODEL_SEEDS:
            for target_split in FORMAL_SPLIT_SEEDS:
                for source_domain in source_domains:
                    engine_values = np.asarray(inventories[source_domain], dtype=np.int64)
                    partition_rng = np.random.default_rng(np.random.SeedSequence([
                        int(args.episode_seed), domain_code(target_domain), int(model_seed),
                        int(target_split), domain_code(source_domain), 2401,
                    ]))
                    shuffled = engine_values.copy()
                    partition_rng.shuffle(shuffled)
                    validation_count = max(required, int(round(len(shuffled) * args.meta_validation_fraction)))
                    validation_count = min(validation_count, len(shuffled) - required)
                    if validation_count < required:
                        raise A240Error(
                            f"source domain {source_domain} cannot form disjoint train/validation episodes"
                        )
                    validation_pool = shuffled[:validation_count]
                    training_pool = shuffled[validation_count:]
                    if set(training_pool.tolist()) & set(validation_pool.tolist()):
                        raise A240Error("meta-train/meta-validation engine overlap")
                    for phase_code, (phase, pool) in enumerate((
                        ("meta_train", training_pool), ("meta_validation", validation_pool),
                    ), start=1):
                        for episode_index in range(int(args.episodes_per_source_domain)):
                            episode_rng = np.random.default_rng(np.random.SeedSequence([
                                int(args.episode_seed), domain_code(target_domain), int(model_seed),
                                int(target_split), domain_code(source_domain), phase_code, episode_index,
                            ]))
                            selected = episode_rng.choice(pool, size=required, replace=False)
                            support = tuple(sorted(int(value) for value in selected[:args.meta_support_engines]))
                            query = tuple(sorted(int(value) for value in selected[args.meta_support_engines:]))
                            if set(support) & set(query):
                                raise A240Error("within-episode support/query engine overlap")
                            rows.append({
                                "target_domain": target_domain,
                                "model_seed": int(model_seed),
                                "target_support_split_seed": int(target_split),
                                "source_domain": source_domain,
                                "episode_phase": phase,
                                "episode_index": int(episode_index),
                                "meta_support_engines": json.dumps(support),
                                "meta_query_engines": json.dumps(query),
                                "meta_support_count": len(support),
                                "meta_query_count": len(query),
                                "source_train_pool_count": int(len(training_pool)),
                                "source_validation_pool_count": int(len(validation_pool)),
                                "target_domain_excluded_from_meta_train": True,
                                "support_query_disjoint": True,
                                "meta_train_validation_engine_disjoint": True,
                                "engine_selection_uses_labels": False,
                                "engine_selection_uses_trajectory_length": False,
                                "official_test_files_accessed": False,
                            })
    return rows


def parameter_audit(args: argparse.Namespace, cfg: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    common = {
        "base_backbone": "existing baselines.build_model('gnn', feature_count, config) contract",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": json.dumps(list(FEATURE_COLUMNS)),
        "window_size": int(cfg["window_size"]),
        "batch_size": int(cfg["batch_size"]),
        "rul_cap": float(cfg["rul_cap"]),
        "inner_learning_rate": float(cfg["inner_lr"]),
        "inner_steps": int(args.inner_steps),
        "outer_steps": int(args.outer_steps),
        "outer_learning_rate": float(args.outer_learning_rate),
        "meta_support_engines_per_episode": int(args.meta_support_engines),
        "meta_query_engines_per_episode": int(args.meta_query_engines),
        "episodes_per_source_domain_per_phase": int(args.episodes_per_source_domain),
        "config_sha256": sha256_file(config_path),
        "target_support_shots": json.dumps(list(FORMAL_SHOTS)),
        "primary_target_support_shot": PRIMARY_SHOT,
        "causal_anchor_contract": "A23.2_v2_rul_090_045_015",
        "initialisation_schedule": "same registered model_seed controls both methods",
        "parameter_count_verified_at_preflight": False,
        "parameter_count_verification_required_before_A24_1_optimizer_step": True,
    }
    return [
        {
            "method": "meta_no_graph_k",
            "meta_algorithm": "Reptile",
            "graph_message_passing": False,
            "graph_control": "graph aggregation is bypassed; no graph edges are constructed",
            "only_intended_difference": "absence of graph message passing",
            **common,
        },
        {
            "method": "meta_gnn_k",
            "meta_algorithm": "Reptile",
            "graph_message_passing": True,
            "graph_control": "GAT/message-passing branch may use only samples within the same causal endpoint batch",
            "only_intended_difference": "presence of graph message passing",
            **common,
        },
    ]


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = resolve(args.output_dir)
    protocol, roles, protocol_hashes, expected_training_hashes = load_protocol(args)
    a234_decision, a234_hashes = load_a234_boundary(args)
    cfg, config_path = load_config(args.config)
    target_audit = verify_target_roles(roles)

    training_inventory: dict[str, dict[str, Any]] = {}
    engine_inventories: dict[str, tuple[int, ...]] = {}
    for domain in DOMAINS:
        path = resolve_training_file(args.data_dir, domain)
        digest = sha256_file(path)
        if digest != expected_training_hashes[domain]:
            raise A240Error(
                f"training file changed after A23.0 for {domain}: "
                f"expected={expected_training_hashes[domain]}, actual={digest}"
            )
        engines = load_engine_inventory(path, domain)
        training_inventory[domain] = {
            "path": str(path), "sha256": digest, "n_engines": len(engines),
        }
        engine_inventories[domain] = engines

    episode_rows = build_source_episode_rows(args, engine_inventories)
    task_frame = pd.DataFrame(episode_rows)
    expected_tasks = (
        len(DOMAINS) * len(FORMAL_MODEL_SEEDS) * len(FORMAL_SPLIT_SEEDS)
        * (len(DOMAINS) - 1) * 2 * int(args.episodes_per_source_domain)
    )
    if len(task_frame) != expected_tasks:
        raise A240Error(f"meta task count={len(task_frame)}, expected={expected_tasks}")
    task_keys = [
        "target_domain", "model_seed", "target_support_split_seed", "source_domain",
        "episode_phase", "episode_index",
    ]
    if task_frame.duplicated(task_keys).any():
        raise A240Error("duplicate source meta-task key")
    if (task_frame["target_domain"] == task_frame["source_domain"]).any():
        raise A240Error("target domain leaked into meta source domain")
    if not task_frame["support_query_disjoint"].all() or not task_frame["meta_train_validation_engine_disjoint"].all():
        raise A240Error("episodic engine-disjointness invariant failed")
    parameter_rows = parameter_audit(args, cfg, config_path)

    formal_cells = (
        len(DOMAINS) * len(FORMAL_MODEL_SEEDS) * len(FORMAL_SPLIT_SEEDS)
        * len(FORMAL_SHOTS) * len(METHODS)
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": (
            "Can a leakage-free leave-one-target-domain-out Reptile task contract be locked "
            "before comparing Meta-noGraph and Meta-GNN under the unchanged A23 few-shot "
            "engine roles and causal anchor evaluation protocol?"
        ),
        "output_dir": str(output_root),
        "methods": list(METHODS),
        "meta_algorithm": "Reptile",
        "target_domains": list(DOMAINS),
        "model_seeds": list(FORMAL_MODEL_SEEDS),
        "target_support_split_seeds": list(FORMAL_SPLIT_SEEDS),
        "target_support_shots": list(FORMAL_SHOTS),
        "primary_shot": PRIMARY_SHOT,
        "registered_rul_anchors": list(RUL_ANCHORS),
        "causal_anchor_contract": "A23.2_v2_rul_090_045_015",
        "expected_target_training_cells_for_A24_2": formal_cells,
        "expected_source_meta_tasks": expected_tasks,
        "source_meta_tasks": int(len(task_frame)),
        "source_meta_train_tasks": int((task_frame["episode_phase"] == "meta_train").sum()),
        "source_meta_validation_tasks": int((task_frame["episode_phase"] == "meta_validation").sum()),
        "meta_support_engines_per_episode": int(args.meta_support_engines),
        "meta_query_engines_per_episode": int(args.meta_query_engines),
        "episodes_per_source_domain_per_phase": int(args.episodes_per_source_domain),
        "meta_validation_fraction": float(args.meta_validation_fraction),
        "inner_steps": int(args.inner_steps),
        "outer_steps": int(args.outer_steps),
        "outer_learning_rate": float(args.outer_learning_rate),
        "protocol_hashes": protocol_hashes,
        "a23_4_input_hashes": a234_hashes,
        "a23_4_completed": True,
        "a23_4_primary_passed": bool(a234_decision.get("passed")),
        "a23_4_result_is_frozen_not_reused_for_tuning": True,
        "training_file_inventory": training_inventory,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "target_engine_roles_disjoint": True,
        "target_domain_excluded_from_meta_train_domains": True,
        "source_episode_support_query_disjoint": True,
        "source_meta_train_validation_engines_disjoint": True,
        "engine_selection_is_label_and_trajectory_independent": True,
        "parameter_count_preflight_is_not_a_model_claim": True,
        "parameter_count_execution_assertion_required_in_A24_1": True,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    context = {
        "output_root": output_root,
        "target_audit": target_audit,
        "task_rows": episode_rows,
        "parameter_rows": parameter_rows,
        "protocol": protocol,
    }
    return result, context


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolve(args.output_dir)
    decision_path = output_root / "experimentA24_0_confirmation_decision.json"
    if decision_path.is_file():
        prior = read_json(decision_path)
        if args.resume and prior.get("complete") is True and prior.get("passed") is True:
            print(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            print("[A24.0] resume: existing complete decision returned", flush=True)
            return 0
        raise A240Error(
            f"output already contains a completed A24.0 decision: {decision_path}; "
            "use --resume or choose a new output directory"
        )

    result, context = preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print(
            "[A24.0] dry-run passed: meta-learning task and fairness contract validated; "
            "no predictor was trained and no official test file was accessed",
            flush=True,
        )
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise A240Error(
            f"non-empty output directory without a complete decision: {output_root}; "
            "choose a new output directory to avoid artifact mixing"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "experimentA24_0_meta_protocol.json", result)
    atomic_csv(
        output_root / "experimentA24_0_source_meta_task_inventory.csv",
        context["task_rows"],
        list(context["task_rows"][0].keys()),
    )
    atomic_csv(
        output_root / "experimentA24_0_engine_role_audit.csv",
        context["target_audit"],
        list(context["target_audit"][0].keys()),
    )
    atomic_csv(
        output_root / "experimentA24_0_parameter_budget_audit.csv",
        context["parameter_rows"],
        list(context["parameter_rows"][0].keys()),
    )
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "passed": True,
        "registered_primary_question": result["registered_primary_question"],
        "expected_source_meta_tasks": result["expected_source_meta_tasks"],
        "completed_source_meta_tasks": result["source_meta_tasks"],
        "expected_target_training_cells_for_A24_2": result["expected_target_training_cells_for_A24_2"],
        "target_domain_excluded_from_meta_train_domains": True,
        "source_episode_support_query_disjoint": True,
        "source_meta_train_validation_engines_disjoint": True,
        "target_engine_roles_disjoint": True,
        "parameter_count_execution_assertion_required_in_A24_1": True,
        "a23_4_primary_result_frozen": True,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "reason": (
            "A24.0 locked a leakage-free, leave-one-target-domain-out episodic meta-learning "
            "contract and a declared Meta-noGraph/Meta-GNN fairness contract"
        ),
        "interpretation_limit": (
            "A24.0 creates only roles, episodes and implementation constraints. It does not "
            "train a meta-learner, compare methods, verify executable parameter counts, or "
            "establish RUL/fault-detection efficacy."
        ),
        "next_action": "implement_A24_1_meta_no_graph_and_meta_gnn_pilot_with_runtime_parameter_assertion",
    }
    atomic_json(output_root / "experimentA24_0_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "inputs": {
            "protocol_hashes": result["protocol_hashes"],
            "a23_4_input_hashes": result["a23_4_input_hashes"],
            "config_sha256": result["config_sha256"],
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {},
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    artifact_names = (
        "experimentA24_0_meta_protocol.json",
        "experimentA24_0_source_meta_task_inventory.csv",
        "experimentA24_0_engine_role_audit.csv",
        "experimentA24_0_parameter_budget_audit.csv",
        "experimentA24_0_confirmation_decision.json",
    )
    manifest["artifacts"] = {
        name: sha256_file(output_root / name) for name in artifact_names
    }
    atomic_json(output_root / "experimentA24_0_manifest.json", manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A240Error as exc:
        print(f"[A24.0] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
