"""Experiment A3: locked endpoint-selection transfer confirmation.

Purpose
-------
A1--A1_3 found no stable sensor-graph advantage, while A2 and A2_1 established
that full-trajectory selection and endpoint-like selection lead to a material,
replicable protocol gap.  A3 deliberately stops comparing graph architectures.
It uses *only* the A2_1 ``window_no_graph`` control and asks the deployment
question that remains:

    Do selection epochs chosen with a balanced endpoint rule transfer better to
    the official C-MAPSS test endpoints than epochs chosen from full trajectories?

The two epoch-selection policies are locked from the completed A2_1 training
outputs before this script is allowed to read any official test trajectory or
official RUL label.  For every (domain, model seed, support split), A3 takes a
deterministic upper median of the A2_1 cross-fit epoch votes:

* full-trajectory policy: five role-partition votes;
* balanced-endpoint policy: 25 role-partition x endpoint-assignment votes.

No architecture, hyperparameter, source model, target support split, or epoch
is selected using the official test data.  ``--confirm-official-test`` is
mandatory for a formal run; the default/``--dry-run`` path never opens test or
RUL files.  Completed test cells are resumable, but a completed final decision
cannot be overwritten or rerun in the same output directory.

Run from the repository root (first, no official-test access):

    python -u scripts/experimentA3_locked_endpoint_transfer_confirmation.py --dry-run

Then, after inspecting the generated locked-policy file:

    nohup python -u scripts/experimentA3_locked_endpoint_transfer_confirmation.py \\
      --confirm-official-test > experimentA3_training.log 2>&1 &

All artifacts are written under ``outputs/experimentA3_locked_endpoint_transfer_confirmation``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from preprocess.cmapps_loader import load_domain  # noqa: E402
from preprocess.rul_generator import add_test_rul, add_train_rul  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experiment7_kshot_engines as exp7  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402


SCRIPT_VERSION = "experimentA3_locked_endpoint_transfer_confirmation_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
ARCHITECTURE = "window_no_graph"
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
ENDPOINT_SEEDS = list(range(7501, 7506))
SELECTION_PROTOCOLS = (
    "full_trajectory_selection",
    "balanced_endpoint_selection",
)
DEFAULT_OUTPUT = "outputs/experimentA3_locked_endpoint_transfer_confirmation"
DEFAULT_A2_OUTPUT = "outputs/experimentA2_endpoint_consistency_validation"
DEFAULT_A2_1_OUTPUT = "outputs/experimentA2_1_endpoint_scheme_crossfit_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A3: locked endpoint-selection official confirmation"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-output-dir")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 3,4,5")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="required acknowledgement before any official test/RUL file is opened",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted formal run; completed cells are never re-evaluated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write and validate the frozen A2_1 policy only; never access official test files",
    )
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required locked-input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stable_seed(*parts: Any) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def load_config(args: argparse.Namespace) -> tuple[dict, dict]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    base.update(
        {
            "data_dir": resolved(args.data_dir, base["data_dir"]),
            "output_dir": resolved(args.output_dir, DEFAULT_OUTPUT),
            "normalizer_seed": 2026,
            "condition_count": 6,
            "source_pretrain_steps": 1500,
            "source_pretrain_lr": 0.001,
            "source_pretrain_weight_decay": 0.0,
            "target_epochs": 10,
            "target_lr": 0.001,
            "pair_aux_weight": 0.0,
            "device": args.device,
        }
    )
    experiment = {
        "experiment_id": "experimentA3",
        "experiment_name": "locked_endpoint_transfer_confirmation",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "endpoint_seeds": ENDPOINT_SEEDS.copy(),
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "policy_aggregation": "upper_median_of_locked_A2_1_epoch_votes",
        "minimum_domain_wins": 3,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "a2_output_dir": resolved(args.a2_output_dir, DEFAULT_A2_OUTPUT),
        "a2_1_output_dir": resolved(args.a2_1_output_dir, DEFAULT_A2_1_OUTPUT),
        "output_dir": base["output_dir"],
        "quick_mode": False,
    }
    return base, experiment


def validate_config(base: dict, experiment: dict, args: argparse.Namespace) -> None:
    if experiment["architecture"] != ARCHITECTURE:
        raise ValueError(f"A3 requires the fixed architecture={ARCHITECTURE}")
    for name, expected in (
        ("domains", list(DOMAINS)),
        ("model_seeds", MODEL_SEEDS),
        ("target_split_seeds", TARGET_SPLIT_SEEDS),
        ("role_partitions", ROLE_PARTITIONS),
        ("endpoint_seeds", ENDPOINT_SEEDS),
    ):
        if experiment[name] != expected:
            raise ValueError(f"A3 requires locked {name}={expected}")
    if int(experiment["k"]) != 5:
        raise ValueError("A3 requires K=5")
    if args.dry_run and args.confirm_official_test:
        raise ValueError("choose either --dry-run or --confirm-official-test, not both")
    if not args.dry_run and not args.confirm_official_test:
        raise ValueError("A3 requires --dry-run or explicit --confirm-official-test")
    for domain in DOMAINS:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def root_paths(output: Path) -> dict[str, Path]:
    prefix = "experimentA3"
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "dry_run": output / f"{prefix}_dry_run.json",
        "locked_policy": output / f"{prefix}_locked_policy.json",
        "locked_policy_csv": output / f"{prefix}_locked_policy.csv",
        "run_json": output / f"{prefix}_run_level.json",
        "run_csv": output / f"{prefix}_run_level.csv",
        "predictions": output / f"{prefix}_official_endpoint_predictions.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "test_integrity": output / f"{prefix}_official_test_integrity.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_selection_policies.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def shard_dir(output: Path, domain: str, seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{seed:03d}"


def shard_paths(output: Path, domain: str, seed: int) -> dict[str, Path]:
    directory = shard_dir(output, domain, seed)
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "run_json": directory / "run_level.json",
        "run_csv": directory / "run_level.csv",
        "predictions": directory / "official_endpoint_predictions.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
        "test_audit": directory / "official_test_audit.json",
    }


def _require_a2_1_result_shape(frame: pd.DataFrame) -> None:
    required = {
        "target_domain", "model", "model_seed", "target_split_seed",
        "role_partition", "endpoint_seed", "selection_protocol",
        "evaluation_protocol", "selected_epoch",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"A2_1 run-level file is missing columns: {sorted(missing)}")


def _verify_protocol_hash(protocol: dict, domain: str) -> None:
    if protocol.get("target_domain") != domain:
        raise ValueError(f"A2_1 protocol target mismatch for {domain}")
    stored = protocol.get("protocol_hash")
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    if not stored or stored != a1.canonical_hash(payload):
        raise ValueError(f"A2_1 protocol hash is invalid for {domain}")


def load_a2_1_evidence(experiment: dict) -> dict:
    """Load only completed training-only A2_1 artifacts; never test data."""
    root = Path(experiment["a2_1_output_dir"])
    required = {
        "manifest": root / "experimentA2_1_manifest.json",
        "protocol": root / "experimentA2_1_protocol.json",
        "run": root / "experimentA2_1_run_level.csv",
        "decision": root / "experimentA2_1_confirmation_decision.json",
        "lock": root / "experimentA2_1_lock_candidate.json",
    }
    manifest = read_json(required["manifest"])
    protocols = read_json(required["protocol"])
    decision = read_json(required["decision"])
    lock = read_json(required["lock"])
    run = load_csv(required["run"])
    if run.empty:
        raise ValueError("A2_1 run-level output is empty")
    _require_a2_1_result_shape(run)
    expected_decision = {
        "experiment_id": "experimentA2_1",
        "expected_training_cells": 200,
        "completed_training_cells": 200,
        "expected_evaluation_records": 60000,
        "completed_evaluation_records": 60000,
        "complete": True,
        "quick_mode": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "endpoint_protocol_gap_confirmed": True,
        "passed": True,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            raise ValueError(f"A3 requires completed passing A2_1 evidence: {key}={expected}")
    if lock.get("candidate_model") != ARCHITECTURE:
        raise ValueError("A3 requires A2_1's no-graph control as the locked architecture")
    if bool(lock.get("eligible_for_locked_official_confirmation")):
        raise ValueError("A3 is a policy confirmation, not a graph-superiority confirmation")
    if set(protocols) != set(DOMAINS):
        raise ValueError("A2_1 protocol must contain all four C-MAPSS domains")
    for domain in DOMAINS:
        _verify_protocol_hash(protocols[domain], domain)
        if int(protocols[domain].get("k", -1)) != int(experiment["k"]):
            raise ValueError(f"A2_1 protocol K differs for {domain}")
        if protocols[domain].get("model_seeds") != experiment["model_seeds"]:
            raise ValueError(f"A2_1 model seeds differ for {domain}")
        if protocols[domain].get("target_split_seeds") != experiment["target_split_seeds"]:
            raise ValueError(f"A2_1 target split seeds differ for {domain}")
    input_hashes = {name: a1.file_sha256(path) for name, path in required.items()}
    return {
        "root": str(root),
        "manifest": manifest,
        "protocols": protocols,
        "run": run,
        "decision": decision,
        "lock": lock,
        "input_hashes": input_hashes,
    }


def cell_key(domain: str, model_seed: int, split_seed: int) -> str:
    return f"{domain}_mseed{int(model_seed):03d}_tsplit{int(split_seed)}"


def upper_median(values: list[int]) -> int:
    if not values:
        raise ValueError("cannot aggregate an empty set of locked epoch votes")
    ordered = sorted(map(int, values))
    return int(ordered[len(ordered) // 2])


def build_locked_policy(evidence: dict, experiment: dict) -> dict:
    """Freeze A2_1 choices without looking at official predictions or labels."""
    run = evidence["run"].copy()
    run = run[run["model"] == ARCHITECTURE].copy()
    required_count = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS)
    cells: dict[str, dict] = {}
    for domain in DOMAINS:
        for model_seed in MODEL_SEEDS:
            for split_seed in TARGET_SPLIT_SEEDS:
                key = cell_key(domain, model_seed, split_seed)
                subset = run[
                    (run["target_domain"] == domain)
                    & (run["model_seed"] == model_seed)
                    & (run["target_split_seed"] == split_seed)
                ]
                entries: dict[str, dict] = {}
                for selection_protocol in SELECTION_PROTOCOLS:
                    rows = subset[
                        (subset["selection_protocol"] == selection_protocol)
                        & (subset["evaluation_protocol"] == "balanced_endpoint")
                    ].copy()
                    expected_votes = len(ROLE_PARTITIONS) * len(ENDPOINT_SEEDS)
                    if len(rows) != expected_votes:
                        raise ValueError(
                            f"A2_1 has {len(rows)}, not {expected_votes}, locked votes for {key} {selection_protocol}"
                        )
                    group_columns = ["role_partition", "endpoint_seed"]
                    duplicates = rows.groupby(group_columns)["selected_epoch"].nunique()
                    if len(duplicates) != expected_votes or int(duplicates.max()) != 1:
                        raise ValueError(f"A2_1 has duplicate/inconsistent selected epochs for {key}")
                    rows = rows.sort_values(group_columns)
                    votes = [int(value) for value in rows["selected_epoch"].tolist()]
                    if not all(1 <= value <= int(experiment["target_epochs"]) for value in votes):
                        raise ValueError(f"A2_1 selected epoch is outside 1..{experiment['target_epochs']} for {key}")
                    if selection_protocol == "full_trajectory_selection":
                        per_partition = rows.groupby("role_partition")["selected_epoch"].agg(list)
                        if any(len(set(map(int, value))) != 1 for value in per_partition):
                            raise ValueError(f"full-trajectory epoch unexpectedly varies by endpoint assignment for {key}")
                        effective_votes = [int(value[0]) for _, value in per_partition.items()]
                    else:
                        effective_votes = votes
                    entries[selection_protocol] = {
                        "selected_epoch": upper_median(effective_votes),
                        "effective_epoch_votes": effective_votes,
                        "raw_epoch_votes": votes,
                        "vote_count": len(effective_votes),
                    }
                protocol = evidence["protocols"][domain]
                support = protocol["role_splits"][str(split_seed)]["adaptation_units"]
                cells[key] = {
                    "target_domain": domain,
                    "model_seed": int(model_seed),
                    "target_split_seed": int(split_seed),
                    "adaptation_units": list(map(int, support)),
                    "a2_1_protocol_hash": protocol["protocol_hash"],
                    "policies": entries,
                }
    if len(cells) != required_count:
        raise AssertionError("locked-policy cell count is incomplete")
    payload = {
        "experiment_id": "experimentA3",
        "policy_origin": "completed_A2_1_training_only_outputs",
        "registered_primary_question": "Does locked balanced-endpoint epoch selection improve official C-MAPSS endpoint deployment RMSE over locked full-trajectory epoch selection?",
        "architecture": ARCHITECTURE,
        "k": int(experiment["k"]),
        "policy_aggregation": experiment["policy_aggregation"],
        "a2_1_input_hashes": evidence["input_hashes"],
        "a2_1_protocol_hashes": {
            domain: evidence["protocols"][domain]["protocol_hash"] for domain in DOMAINS
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "cells": cells,
    }
    payload["policy_hash"] = a1.canonical_hash(payload)
    return payload


def locked_policy_frame(policy: dict) -> pd.DataFrame:
    rows = []
    for key, cell in policy["cells"].items():
        for selection_protocol, detail in cell["policies"].items():
            rows.append(
                {
                    "cell_key": key,
                    "target_domain": cell["target_domain"],
                    "model_seed": cell["model_seed"],
                    "target_split_seed": cell["target_split_seed"],
                    "adaptation_units": json.dumps(cell["adaptation_units"]),
                    "selection_protocol": selection_protocol,
                    "selected_epoch": detail["selected_epoch"],
                    "effective_epoch_votes": json.dumps(detail["effective_epoch_votes"]),
                    "raw_epoch_votes": json.dumps(detail["raw_epoch_votes"]),
                    "vote_count": detail["vote_count"],
                    "policy_hash": policy["policy_hash"],
                }
            )
    return pd.DataFrame(rows).sort_values(["target_domain", "model_seed", "target_split_seed", "selection_protocol"])


def validate_current_train_hashes(base: dict, evidence: dict) -> None:
    for domain in DOMAINS:
        expected = evidence["protocols"][domain].get("train_file_hashes", {})
        for hashed_domain, expected_hash in expected.items():
            current = a1.file_sha256(a1.train_path(base["data_dir"], hashed_domain))
            if current != expected_hash:
                raise RuntimeError(
                    f"training file changed since A2_1: {hashed_domain}; A3 must not continue"
                )


def prepare_support_loader(cfg: dict, support_units: list[int]):
    sensors = list(cfg["sensor_columns"])
    _, normalizer = a1.fit_source_normalizer_train_only(cfg, "condition_settings")
    target = add_train_rul(a1.load_train_domain(cfg["data_dir"], cfg["target_domain"]), cfg["rul_cap"])
    normalized = normalizer.transform(target, sensors)
    features = sensors + a1.SETTING_FEATURE_COLUMNS
    support_frame = normalized.query("unit in @support_units")
    if support_frame["unit"].nunique() != len(support_units):
        raise RuntimeError("A3 target support set is incomplete")
    support = a1.make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode="engine_stage",
        loader_seed=int(cfg["seed"]) + 9000,
    )
    return support, len(features)


def official_file_paths(data_dir: str, domain: str) -> dict[str, Path]:
    root = Path(data_dir)
    nested = root / domain
    folder = nested if (nested / f"train_{domain}.txt").is_file() else root
    return {
        "test": folder / f"test_{domain}.txt",
        "rul": folder / f"RUL_{domain}.txt",
    }


def prepare_official_endpoint_loader(cfg: dict, domain: str):
    """This is the only A3 function permitted to open official test/RUL files."""
    paths = official_file_paths(cfg["data_dir"], domain)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing official {name} file: {path}")
    _, test, final_rul = load_domain(cfg["data_dir"], domain)
    expected = int(exp7.EXPECTED_OFFICIAL_TEST_ENGINES[domain])
    test_units = np.asarray(sorted(test["unit"].unique()), dtype=int)
    if len(test_units) != expected:
        raise ValueError(f"{domain} has {len(test_units)} official test engines, expected {expected}")
    labeled = add_test_rul(test, final_rul, cfg["rul_cap"])
    sensors = list(cfg["sensor_columns"])
    _, normalizer = a1.fit_source_normalizer_train_only(cfg, "condition_settings")
    normalized = normalizer.transform(labeled, sensors)
    features = sensors + a1.SETTING_FEATURE_COLUMNS
    loader = a1.make_loader(
        normalized,
        features,
        cfg,
        training=False,
        last_only=True,
        loader_seed=int(cfg["seed"]) + 9900,
    )
    loader_units = np.asarray(loader.dataset.units, dtype=int)
    if len(loader_units) != expected or len(set(loader_units.tolist())) != expected:
        raise AssertionError("official endpoint loader must have exactly one window per test engine")
    return loader, len(features), {
        "target_domain": domain,
        "official_test_engine_count": expected,
        "official_test_units_hash": hashlib.sha256(loader_units.tobytes()).hexdigest(),
        "test_file_sha256": a1.file_sha256(paths["test"]),
        "rul_file_sha256": a1.file_sha256(paths["rul"]),
        "official_test_files_accessed": True,
        "official_test_forward_run": True,
    }


def train_target_to_locked_epochs(
    model: torch.nn.Module,
    support,
    cfg: dict,
    device: torch.device,
    required_epochs: set[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], pd.DataFrame]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("model has no predictor.* parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["target_lr"]))
    captured: dict[int, dict[str, torch.Tensor]] = {}
    history: list[dict] = []
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        learner.train()
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss = F.mse_loss(prediction, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A3 target-head loss became NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        if epoch in required_epochs:
            captured[epoch] = a1.state_to_cpu(learner)
        print(f"A3 target_epoch={epoch:02d}/{cfg['target_epochs']} loss={np.mean(losses):.4f}")
    missing = required_epochs.difference(captured)
    if missing:
        raise AssertionError(f"locked epochs were not captured: {sorted(missing)}")
    del learner
    return captured, pd.DataFrame(history)


def evaluate_target_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    locked_cell: dict,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    prior: torch.Tensor,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict]:
    domain = str(locked_cell["target_domain"])
    model_seed = int(locked_cell["model_seed"])
    split_seed = int(locked_cell["target_split_seed"])
    run_seed = a2.target_run_seed(domain, model_seed, split_seed)
    cfg = deepcopy(base)
    cfg.update({"seed": run_seed, "target_domain": domain, "source_domains": protocol["source_domains"]})
    support, feature_count = prepare_support_loader(cfg, list(map(int, locked_cell["adaptation_units"])))
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(ARCHITECTURE, feature_count, cfg, prior, prior)
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    selected = {
        policy: int(detail["selected_epoch"])
        for policy, detail in locked_cell["policies"].items()
    }
    states, history = train_target_to_locked_epochs(model, support, cfg, device, set(selected.values()))

    # Selection is fully frozen above.  Test files are first accessed only here.
    official_loader, official_feature_count, test_audit = prepare_official_endpoint_loader(cfg, domain)
    if feature_count != official_feature_count:
        raise AssertionError("training and official-test feature counts differ")
    predictions_by_epoch: dict[int, pd.DataFrame] = {}
    results: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    identifier = cell_key(domain, model_seed, split_seed)
    for selection_protocol in SELECTION_PROTOCOLS:
        epoch = selected[selection_protocol]
        if epoch not in predictions_by_epoch:
            model.load_state_dict(states[epoch])
            model.to(device)
            predictions_by_epoch[epoch] = a1.predict_with_units(model, official_loader, device)
        predictions = predictions_by_epoch[epoch].copy()
        if predictions["unit"].nunique() != int(test_audit["official_test_engine_count"]) or len(predictions) != int(test_audit["official_test_engine_count"]):
            raise AssertionError("official endpoint prediction count is invalid")
        metrics = regression_metrics(predictions["label"], predictions["prediction"])
        common = {
            "experiment_id": "experimentA3",
            "cell_id": identifier,
            "target_domain": domain,
            "model": ARCHITECTURE,
            "model_seed": model_seed,
            "target_split_seed": split_seed,
            "target_run_seed": run_seed,
            "k": int(experiment["k"]),
            "selection_protocol": selection_protocol,
            "selected_epoch": epoch,
            "adaptation_units": list(map(int, locked_cell["adaptation_units"])),
            "a2_1_protocol_hash": locked_cell["a2_1_protocol_hash"],
            "locked_policy_hash": experiment["locked_policy_hash"],
            "selection_was_locked_before_official_test": True,
            "source_cache_origin": inventory.get("source_cache_origin"),
            "source_signature": inventory["source_signature"],
            "source_history_rows": int(len(source_history)),
            "official_test_files_accessed": True,
            "official_test_forward_run": True,
        }
        results.append({**common, **metrics, "official_test_engine_count": int(test_audit["official_test_engine_count"])})
        for column, value in reversed(list(common.items())):
            predictions.insert(0, column, value)
        prediction_parts.append(predictions)
    history.insert(0, "cell_id", identifier)
    history.insert(1, "target_domain", domain)
    history.insert(2, "model_seed", model_seed)
    history.insert(3, "target_split_seed", split_seed)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, pd.concat(prediction_parts, ignore_index=True), history, test_audit


def load_worker_state(paths: dict[str, Path]) -> dict:
    completed: set[str] = set()
    if paths["status"].is_file():
        completed = set(read_json(paths["status"]).get("completed_cell_ids", []))
    results = read_json(paths["run_json"]) if paths["run_json"].is_file() else []
    results = [row for row in results if row.get("cell_id") in completed]
    state = {"completed": completed, "results": results}
    for name in ("predictions", "history", "inventory"):
        frame = load_csv(paths[name])
        if name != "inventory" and not frame.empty:
            frame = frame[frame["cell_id"].isin(completed)]
        state[name] = frame
    return state


def save_worker_state(paths: dict[str, Path], state: dict, expected: int, test_audit: dict | None) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["run_json"], state["results"])
    a1.atomic_write_text(paths["run_csv"], pd.DataFrame(state["results"]).to_csv(index=False))
    for name in ("predictions", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    if test_audit is not None:
        atomic_json(paths["test_audit"], test_audit)
    atomic_json(
        paths["status"],
        {
            "completed_cell_ids": sorted(state["completed"]),
            "completed_training_cells": len(state["completed"]),
            "expected_training_cells": expected,
            "completed_official_evaluations": len(state["results"]),
            "expected_official_evaluations": expected * len(SELECTION_PROTOCOLS),
            "complete": len(state["completed"]) == expected,
            "official_test_files_accessed": bool(state["completed"]),
            "official_test_forward_run": bool(state["completed"]),
        },
    )


def require_verified_a2_source_cache(
    base: dict,
    experiment: dict,
    protocol: dict,
    model_seed: int,
    prior: torch.Tensor,
) -> tuple[dict, list, dict]:
    reused = a21.reuse_a2_source_cache(
        base, experiment, protocol, ARCHITECTURE, model_seed, prior
    )
    if reused is None:
        raise RuntimeError(
            "A3 requires the verified A2 source cache used by A2_1. "
            "Do not retrain a source model after locking A2_1 selection; restore "
            "outputs/experimentA2_endpoint_consistency_validation first."
        )
    state, history, inventory = reused
    if inventory.get("source_cache_origin") != "verified_experimentA2":
        raise AssertionError("A3 source cache provenance is not verified A2")
    return state, history, inventory


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in DOMAINS or model_seed not in MODEL_SEEDS:
        raise ValueError("unregistered A3 worker")
    if not args.confirm_official_test:
        raise RuntimeError("A3 workers require --confirm-official-test")
    output = Path(base["output_dir"])
    paths = shard_paths(output, domain, model_seed)
    evidence = load_a2_1_evidence(experiment)
    validate_current_train_hashes(base, evidence)
    policy = read_json(root_paths(output)["locked_policy"])
    recalculated = build_locked_policy(evidence, experiment)
    if policy.get("policy_hash") != recalculated.get("policy_hash"):
        raise RuntimeError("locked A3 policy no longer matches the registered A2_1 evidence")
    if policy.get("policy_hash") != experiment["locked_policy_hash"]:
        raise RuntimeError("worker policy hash does not match the parent manifest")
    protocol = evidence["protocols"][domain]
    worker_base = deepcopy(base)
    worker_base.update({"output_dir": str(paths["directory"]), "target_domain": domain, "source_domains": protocol["source_domains"]})
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(worker_base, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    worker_manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "target_domain": domain,
        "model_seed": model_seed,
        "a2_1_protocol_hash": protocol["protocol_hash"],
        "locked_policy_hash": policy["policy_hash"],
        "graph_fit": graph_fit,
        "official_test_access_requires_explicit_flag": True,
    }
    if paths["manifest"].is_file() and paths["status"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("script_hash", "target_domain", "model_seed", "a2_1_protocol_hash", "locked_policy_hash"):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(f"existing A3 shard is incompatible at {key}; use a new output directory")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], worker_manifest)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(paths["directory"] / "source_prior_adjacency.csv", pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv())
    a1.atomic_write_text(paths["directory"] / "source_prior_correlation.csv", pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv())
    state = load_worker_state(paths)
    expected = len(TARGET_SPLIT_SEEDS)
    pending = [split for split in TARGET_SPLIT_SEEDS if cell_key(domain, model_seed, split) not in state["completed"]]
    test_audit = read_json(paths["test_audit"]) if paths["test_audit"].is_file() else None
    if pending:
        source_state, source_history, inventory = require_verified_a2_source_cache(worker_base, experiment, protocol, model_seed, prior)
        inventory_row = {"target_domain": domain, **inventory}
        if state["inventory"].empty:
            state["inventory"] = pd.DataFrame([inventory_row])
        else:
            state["inventory"] = pd.DataFrame([inventory_row])
        for split_seed in pending:
            locked_cell = policy["cells"][cell_key(domain, model_seed, split_seed)]
            results, predictions, history, audit = evaluate_target_cell(
                base=worker_base,
                experiment=experiment,
                protocol=protocol,
                locked_cell=locked_cell,
                source_state=deepcopy(source_state),
                source_history=source_history,
                inventory=inventory,
                prior=prior,
            )
            if test_audit is not None:
                compare_keys = ("official_test_engine_count", "official_test_units_hash", "test_file_sha256", "rul_file_sha256")
                if any(test_audit.get(key) != audit.get(key) for key in compare_keys):
                    raise RuntimeError("official test integrity changed within a single A3 worker")
            test_audit = audit
            state["results"].extend(results)
            state["predictions"] = pd.concat([state["predictions"], predictions], ignore_index=True)
            state["history"] = pd.concat([state["history"], history], ignore_index=True)
            state["completed"].add(cell_key(domain, model_seed, split_seed))
            save_worker_state(paths, state, expected, test_audit)
    save_worker_state(paths, state, expected, test_audit)
    print(paths["status"].read_text(encoding="utf-8"))


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    inventory = a2.query_gpus()
    if args.gpus:
        devices = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        if not devices or len(devices) != len(set(devices)):
            raise ValueError("--gpus must contain one or more unique GPU indices")
        known = {row["index"] for row in inventory}
        if not set(devices).issubset(known):
            raise RuntimeError("one or more requested GPUs are unavailable")
    else:
        visible = a2.visible_gpu_filter()
        candidates = [
            row for row in inventory
            if (visible is None or row["index"] in visible)
            and row["free_mb"] >= int(args.min_free_memory_mb)
            and row["utilization"] <= int(args.max_gpu_utilization)
        ]
        candidates.sort(key=lambda row: (-row["free_mb"], row["utilization"]))
        devices = [row["index"] for row in candidates]
    if args.max_workers > 0:
        devices = devices[: int(args.max_workers)]
    return devices, inventory


def worker_command(args: argparse.Namespace, domain: str, seed: int, device: str, output: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker-domain", domain, "--worker-seed", str(seed),
        "--output-dir", str(output), "--device", device,
        "--bootstrap-repetitions", str(args.bootstrap_repetitions),
        "--confirm-official-test",
    ]
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.a2_output_dir:
        command.extend(["--a2-output-dir", args.a2_output_dir])
    if args.a2_1_output_dir:
        command.extend(["--a2-1-output-dir", args.a2_1_output_dir])
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]
        inventory: list[dict] = []
    else:
        devices, inventory = choose_gpus(args)
        if not devices:
            raise RuntimeError("no idle GPU met A3 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(json.dumps({"scheduler": "experimentA3", "tasks": [{"domain": d, "seed": s} for d, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending, active = list(tasks), {}
    while pending or active:
        for device in [item for item in devices if item not in active]:
            if not pending:
                break
            domain, seed = pending.pop(0)
            directory = shard_dir(output, domain, seed)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "worker_training.log"
            log_handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            if isinstance(device, int):
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                command = worker_command(args, domain, seed, "auto", output)
            else:
                command = worker_command(args, domain, seed, str(device), output)
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            active[device] = {"process": process, "domain": domain, "seed": seed, "log": log_handle, "log_path": log_path}
            print(f"[A3] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = "\n".join(record["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(f"A3 worker failed domain={record['domain']} seed={record['seed']} exit={code}\n{tail}")
            print(f"[A3] completed domain={record['domain']} seed={record['seed']} device={device}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]]) -> dict:
    merged: dict[str, Any] = {"results": [], "predictions": [], "history": [], "inventory": [], "audits": []}
    expected = len(TARGET_SPLIT_SEEDS)
    for domain, model_seed in tasks:
        paths = shard_paths(output, domain, model_seed)
        status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected:
            raise RuntimeError(f"incomplete A3 worker: {paths['status']}")
        if not bool(status.get("official_test_files_accessed")) or not bool(status.get("official_test_forward_run")):
            raise RuntimeError(f"A3 worker lacks official-test audit flags: {paths['status']}")
        merged["results"].extend(read_json(paths["run_json"]))
        for name in ("predictions", "history", "inventory"):
            merged[name].append(load_csv(paths[name]))
        merged["audits"].append(read_json(paths["test_audit"]))
    for name in ("predictions", "history", "inventory"):
        merged[name] = pd.concat(merged[name], ignore_index=True)
    return merged


def paired_policy_results(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "target_split_seed", "cell_id"]
    value_columns = ["rmse", "mae", "r2", "nasa_score", "selected_epoch"]
    pivot = results.pivot(index=keys, columns="selection_protocol", values=value_columns).reset_index()
    pivot.columns = ["_".join(str(item) for item in col if str(item)) if isinstance(col, tuple) else col for col in pivot.columns]
    full = "full_trajectory_selection"
    balanced = "balanced_endpoint_selection"
    output = pivot[keys].copy()
    for metric in value_columns:
        output[f"{metric}_full_trajectory"] = pivot[f"{metric}_{full}"]
        output[f"{metric}_balanced_endpoint"] = pivot[f"{metric}_{balanced}"]
    for metric in ("rmse", "mae", "nasa_score"):
        output[f"{metric}_delta_full_minus_balanced"] = output[f"{metric}_full_trajectory"] - output[f"{metric}_balanced_endpoint"]
    output["balanced_endpoint_rmse_win"] = output["rmse_delta_full_minus_balanced"] > 0
    return output.sort_values(keys)


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, repetitions: int) -> tuple[float, float]:
    domains = sorted(frame["target_domain"].unique())
    seeds = sorted(frame["model_seed"].unique())
    splits = sorted(frame["target_split_seed"].unique())
    lookup = frame.set_index(["target_domain", "model_seed", "target_split_seed"])[column]
    rng = np.random.default_rng(stable_seed("experimentA3_bootstrap", column, repetitions))
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        values = []
        for domain in rng.choice(domains, len(domains), replace=True):
            chosen_seeds = rng.choice(seeds, len(seeds), replace=True)
            chosen_splits = rng.choice(splits, len(splits), replace=True)
            for model_seed in chosen_seeds:
                for split_seed in chosen_splits:
                    values.append(float(lookup.loc[(domain, int(model_seed), int(split_seed))]))
        samples[index] = float(np.mean(values))
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def summarize(results: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in results.groupby(["target_domain", "selection_protocol"]):
        row = {"target_domain": keys[0], "selection_protocol": keys[1], "n_cells": int(len(frame))}
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    primary = paired.groupby("target_domain", as_index=False).agg(
        n_cells=("cell_id", "size"),
        rmse_delta_full_minus_balanced_mean=("rmse_delta_full_minus_balanced", "mean"),
        balanced_endpoint_rmse_win_rate=("balanced_endpoint_rmse_win", "mean"),
        nasa_score_delta_full_minus_balanced_mean=("nasa_score_delta_full_minus_balanced", "mean"),
    )
    summary = pd.DataFrame(rows)
    return summary.merge(primary, on="target_domain", how="left").sort_values(["target_domain", "selection_protocol"])


def make_decision(results: pd.DataFrame, paired: pd.DataFrame, experiment: dict, policy: dict, audits: list[dict]) -> tuple[dict, dict]:
    expected_cells = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS)
    expected_records = expected_cells * len(SELECTION_PROTOCOLS)
    ci = hierarchical_bootstrap(paired, "rmse_delta_full_minus_balanced", int(experiment["bootstrap_repetitions"]))
    domain_means = paired.groupby("target_domain")["rmse_delta_full_minus_balanced"].mean()
    domain_wins = int((domain_means > 0).sum())
    complete = bool(results["cell_id"].nunique() == expected_cells and len(results) == expected_records)
    flags = results[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).all().all()
    score_delta = float(paired["nasa_score_delta_full_minus_balanced"].mean())
    success = bool(complete and flags and ci[0] > 0 and domain_wins >= int(experiment["minimum_domain_wins"]) and score_delta >= 0)
    integrity = {
        "official_test_files_accessed": True,
        "official_test_forward_run": True,
        "test_audits": audits,
        "unique_test_hash_sets": sorted({
            (a["target_domain"], a["test_file_sha256"], a["rul_file_sha256"], a["official_test_units_hash"])
            for a in audits
        }),
    }
    decision = {
        "experiment_id": "experimentA3",
        "registered_primary_question": policy["registered_primary_question"],
        "architecture": ARCHITECTURE,
        "expected_training_cells": expected_cells,
        "completed_training_cells": int(results["cell_id"].nunique()),
        "expected_official_evaluation_records": expected_records,
        "completed_official_evaluation_records": int(len(results)),
        "complete": complete,
        "quick_mode": False,
        "official_test_files_accessed": True,
        "official_test_forward_run": True,
        "locked_policy_hash": policy["policy_hash"],
        "rmse_delta_full_trajectory_minus_balanced_endpoint_mean": float(paired["rmse_delta_full_minus_balanced"].mean()),
        "rmse_delta_ci95": list(ci),
        "balanced_endpoint_rmse_win_rate": float(paired["balanced_endpoint_rmse_win"].mean()),
        "balanced_endpoint_domain_win_count": domain_wins,
        "nasa_score_delta_full_trajectory_minus_balanced_endpoint_mean": score_delta,
        "balanced_endpoint_deployment_advantage_confirmed": success,
        "passed": success,
        "reason": "A3 confirmed the locked balanced-endpoint policy on official test endpoints" if success else "A3 completed official confirmation, but the locked balanced-endpoint policy did not meet every registered deployment criterion",
    }
    return decision, integrity


def initial_manifest(base: dict, experiment: dict, evidence: dict, policy: dict) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"},
        "experiment_config": experiment,
        "registered_primary_question": policy["registered_primary_question"],
        "locked_policy_hash": policy["policy_hash"],
        "a2_1_input_hashes": evidence["input_hashes"],
        "a2_1_protocol_hashes": policy["a2_1_protocol_hashes"],
        "official_test_purpose": "locked selection-policy confirmation only; no graph-superiority claim",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def validate_or_write_initial_artifacts(paths: dict[str, Path], base: dict, experiment: dict, evidence: dict, policy: dict) -> dict:
    manifest = initial_manifest(base, experiment, evidence, policy)
    if paths["manifest"].is_file():
        existing = read_json(paths["manifest"])
        for key in ("script_hash", "locked_policy_hash", "a2_1_input_hashes", "registered_primary_question"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing A3 output is incompatible at {key}; use a new output directory")
        manifest = existing
    else:
        atomic_json(paths["manifest"], manifest)
    if paths["locked_policy"].is_file():
        existing_policy = read_json(paths["locked_policy"])
        if existing_policy.get("policy_hash") != policy.get("policy_hash"):
            raise RuntimeError("existing A3 locked policy differs from current A2_1 evidence")
    else:
        atomic_json(paths["locked_policy"], policy)
        a1.atomic_write_text(paths["locked_policy_csv"], locked_policy_frame(policy).to_csv(index=False))
    return manifest


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    if args.confirm_official_test and paths["decision"].is_file():
        raise RuntimeError("A3 official confirmation is already complete in this output directory; it cannot be rerun")
    evidence = load_a2_1_evidence(experiment)
    validate_current_train_hashes(base, evidence)
    policy = build_locked_policy(evidence, experiment)
    experiment["locked_policy_hash"] = policy["policy_hash"]
    manifest = validate_or_write_initial_artifacts(paths, base, experiment, evidence, policy)
    expected_cells = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS)
    dry = {
        "experiment_id": "experimentA3",
        "expected_training_cells": expected_cells,
        "expected_official_evaluation_records": expected_cells * len(SELECTION_PROTOCOLS),
        "locked_policy_hash": policy["policy_hash"],
        "a2_1_input_hashes": evidence["input_hashes"],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "formal_run_requires": "--confirm-official-test",
        "gpu_inventory": a2.query_gpus(),
    }
    atomic_json(paths["dry_run"], dry)
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    existing_shards = output / "shards"
    if existing_shards.exists() and any(existing_shards.iterdir()) and not args.resume:
        raise RuntimeError("A3 output contains an interrupted run; rerun with --resume so completed official-test cells are not repeated")
    if bool(manifest.get("official_test_forward_run")) and not args.resume:
        raise RuntimeError("A3 manifest already records official-test access; use --resume only for incomplete shards")
    tasks = [(domain, seed) for domain in DOMAINS for seed in MODEL_SEEDS]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks)
    results = pd.DataFrame(merged["results"]).sort_values(["target_domain", "model_seed", "target_split_seed", "selection_protocol"])
    expected_records = expected_cells * len(SELECTION_PROTOCOLS)
    if results["cell_id"].nunique() != expected_cells or len(results) != expected_records:
        raise RuntimeError("A3 merged output is incomplete")
    paired = paired_policy_results(results)
    summary = summarize(results, paired)
    decision, integrity = make_decision(results, paired, experiment, policy, merged["audits"])
    atomic_json(paths["run_json"], results.to_dict("records"))
    a1.atomic_write_text(paths["run_csv"], results.to_csv(index=False))
    a1.atomic_write_text(paths["predictions"], merged["predictions"].to_csv(index=False))
    a1.atomic_write_text(paths["history"], merged["history"].to_csv(index=False))
    a1.atomic_write_text(paths["inventory"], merged["inventory"].to_csv(index=False))
    a1.atomic_write_text(paths["paired"], paired.to_csv(index=False))
    a1.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    atomic_json(paths["test_integrity"], integrity)
    atomic_json(paths["decision"], decision)
    manifest.update({
        "official_test_files_accessed": True,
        "official_test_forward_run": True,
        "official_test_integrity_file": paths["test_integrity"].name,
        "confirmation_decision_file": paths["decision"].name,
    })
    atomic_json(paths["manifest"], manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    base, experiment = load_config(args)
    validate_config(base, experiment, args)
    if args.worker_domain is not None:
        if args.worker_seed is None:
            raise ValueError("--worker-domain requires --worker-seed")
        # Parent has already frozen the policy hash.  Workers recover it from disk.
        policy_path = root_paths(Path(base["output_dir"]))["locked_policy"]
        if not policy_path.is_file():
            raise FileNotFoundError("A3 worker requires the parent-created locked-policy file")
        experiment["locked_policy_hash"] = read_json(policy_path)["policy_hash"]
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
