"""Experiment A9_1: locked official confirmation of the A9 safety blend.

This is the *same experimental direction* as A9, not a new blend search.
It freezes the successful A9 policy before official C-MAPSS test trajectories
or RUL labels are opened:

* architecture, preprocessing, source caches, target budget (10 epochs);
* all 500 A9 selection-only (alpha_high, alpha_low) decisions;
* convex blend formula and predicted-RUL gate at 60.

For every domain/model-seed/target-split A9_1 replays the fixed target-head
adaptation, obtains one official endpoint prediction per test engine from the
baseline and cycle-age representations, then applies the already locked blend
for each role partition.  Official labels are used only in final metrics.

Run from repository root:

    # verifies/freeze inputs only; never opens official test files
    python -u scripts/experimentA9_1_locked_blend_official_confirmation.py --dry-run

    # formal, one-time official confirmation
    nohup python -u scripts/experimentA9_1_locked_blend_official_confirmation.py \\
      --confirm-official-test > experimentA9_1_training.log 2>&1 &
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.rul_generator import add_test_rul  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA3_locked_endpoint_transfer_confirmation as a3  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402
from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402


SCRIPT_VERSION = "experimentA9_1_locked_blend_official_confirmation_v1"
EXPERIMENT_ID = "experimentA9_1"
DEFAULT_OUTPUT = "outputs/experimentA9_1_locked_blend_official_confirmation"
DEFAULT_A9_OUTPUT = "outputs/experimentA9_crossfitted_cycle_age_safety_blend"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (100, 101, 102, 103, 104)
TARGET_SPLIT_SEEDS = (6401, 6402, 6403, 6404, 6405)
ROLE_PARTITIONS = (1, 2, 3, 4, 5)
BASE, AGE, BLEND = a9.BASE, a9.AGE, a9.BLEND
ARCHITECTURE = "window_no_graph"
HIGH_RUL_THRESHOLD = 60.0
PAIR_KEYS = ["target_domain", "model_seed", "target_split_seed", "role_partition"]
QUESTION = (
    "Does the A9 selection-only locked baseline/cycle-age blend improve official "
    "C-MAPSS endpoint NASA/RMSE while preserving true-high- and low/mid-RUL safety?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A9_1 locked A9 blend official confirmation")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a9-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,2,4")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--confirm-official-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, value: Any) -> None:
    a1.atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A9_1 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def root_paths(output: Path) -> dict[str, Path]:
    prefix = EXPERIMENT_ID
    return {
        "manifest": output / f"{prefix}_manifest.json",
        "dry_run": output / f"{prefix}_dry_run.json",
        "locked_policy": output / f"{prefix}_locked_policy.json",
        "locked_policy_csv": output / f"{prefix}_locked_policy.csv",
        "run_json": output / f"{prefix}_official_run_level.json",
        "run_csv": output / f"{prefix}_official_run_level.csv",
        "predictions": output / f"{prefix}_official_endpoint_predictions.csv",
        "history": output / f"{prefix}_target_history.csv",
        "inventory": output / f"{prefix}_source_inventory.csv",
        "integrity": output / f"{prefix}_official_test_integrity.json",
        "paired": output / f"{prefix}_paired_blend_vs_baseline.csv",
        "high": output / f"{prefix}_high_rul_paired_blend_vs_baseline.csv",
        "low": output / f"{prefix}_low_rul_paired_blend_vs_baseline.csv",
        "summary": output / f"{prefix}_comparison_summary.csv",
        "decision": output / f"{prefix}_confirmation_decision.json",
    }


def shard_dir(output: Path, domain: str, model_seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{model_seed:03d}"


def shard_paths(output: Path, domain: str, model_seed: int) -> dict[str, Path]:
    directory = shard_dir(output, domain, model_seed)
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "run_json": directory / "official_run_level.json",
        "run_csv": directory / "official_run_level.csv",
        "predictions": directory / "official_endpoint_predictions.csv",
        "history": directory / "target_history.csv",
        "inventory": directory / "source_inventory.csv",
        "test_audit": directory / "official_test_audit.json",
    }


def cell_id(domain: str, model_seed: int, split_seed: int) -> str:
    return f"{EXPERIMENT_ID}_{domain.lower()}_mseed{model_seed:03d}_tsplit{split_seed}"


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    base.update({
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
    })
    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": "locked_blend_official_confirmation",
        "domains": list(DOMAINS),
        "architecture": ARCHITECTURE,
        "representations": [BASE, AGE],
        "model_seeds": list(MODEL_SEEDS),
        "target_split_seeds": list(TARGET_SPLIT_SEEDS),
        "role_partitions": list(ROLE_PARTITIONS),
        "high_rul_threshold": HIGH_RUL_THRESHOLD,
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "target_epochs": 10,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "stage_noninferiority_margin_pct": 3.0,
        "a9_output_dir": resolved(args.a9_output_dir, DEFAULT_A9_OUTPUT),
        "output_dir": base["output_dir"],
    }
    return base, experiment


def validate_args(base: dict[str, Any], experiment: dict[str, Any], args: argparse.Namespace) -> None:
    if args.dry_run == args.confirm_official_test:
        raise ValueError("A9_1 requires exactly one of --dry-run or --confirm-official-test")
    for domain in DOMAINS:
        if not a1.train_path(base["data_dir"], domain).is_file():
            raise FileNotFoundError(f"missing training file for {domain}")
    if experiment["model_seeds"] != list(MODEL_SEEDS) or experiment["target_split_seeds"] != list(TARGET_SPLIT_SEEDS):
        raise ValueError("A9_1 seed sets are locked")


def a9_inputs(a9_root: Path) -> dict[str, Path]:
    return {
        "manifest": a9_root / "experimentA9_manifest.json",
        "decision": a9_root / "experimentA9_confirmation_decision.json",
        "protocol": a9_root / "experimentA9_protocol.json",
        "parameters": a9_root / "experimentA9_blend_parameters.csv",
        "source_inventory": a9_root / "experimentA9_source_pretrain_inventory.csv",
        "causality": a9_root / "experimentA9_blend_causality_audit.json",
        "feature_audit": a9_root / "experimentA9_feature_causality_audit.json",
    }


def lock_a9_policy(experiment: dict[str, Any]) -> dict[str, Any]:
    root = Path(experiment["a9_output_dir"])
    files = a9_inputs(root)
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"A9 locked-input artifact missing: {path}")
    manifest, decision, protocol = (read_json(files[name]) for name in ("manifest", "decision", "protocol"))
    causality, feature_audit = (read_json(files[name]) for name in ("causality", "feature_audit"))
    if not decision.get("complete") or not decision.get("passed"):
        raise RuntimeError("A9_1 requires completed/passed A9 training-only confirmation")
    if decision.get("official_test_files_accessed") or decision.get("official_test_forward_run"):
        raise RuntimeError("A9 input artifacts must be training-only")
    cfg = manifest.get("experiment_config", {})
    for key, expected in (("model_seeds", list(MODEL_SEEDS)), ("target_split_seeds", list(TARGET_SPLIT_SEEDS)), ("role_partitions", list(ROLE_PARTITIONS)), ("target_epochs", 10)):
        if cfg.get(key) != expected:
            raise RuntimeError(f"A9 locked input differs at {key}")
    if cfg.get("alpha_grid") != [0.0, 0.25, 0.5, 0.75, 1.0] or float(cfg.get("prediction_gate_threshold")) != 60.0:
        raise RuntimeError("A9 blend grid/gate is not the registered policy")
    if not causality.get("selection_labels_used_only_to_choose_alpha") or causality.get("confirmation_used_for_alpha_selection"):
        raise RuntimeError("A9 blend causality audit is incompatible")
    if feature_audit.get("official_test_files_accessed") or feature_audit.get("official_test_forward_run"):
        raise RuntimeError("A9 feature audit indicates test contamination")

    parameters = load_csv(files["parameters"])
    keys = ["target_domain", "model_seed", "target_split_seed", "role_partition"]
    expected_rows = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS)
    required = keys + ["alpha_high", "alpha_low", "confirmation_used_for_alpha_selection"]
    if any(column not in parameters for column in required) or len(parameters) != expected_rows or parameters.duplicated(keys).any():
        raise RuntimeError("A9 blend parameter table is incomplete or not uniquely keyed")
    if parameters["confirmation_used_for_alpha_selection"].astype(bool).any():
        raise RuntimeError("A9 confirmation data was used to choose blend alpha")
    for column in ("alpha_high", "alpha_low"):
        if not parameters[column].isin([0.0, 0.25, 0.5, 0.75, 1.0]).all():
            raise RuntimeError(f"A9 parameter table contains unregistered {column}")

    inventory = load_csv(files["source_inventory"])
    inv_keys = ["target_domain", "model_seed", "representation"]
    expected_sources = len(DOMAINS) * len(MODEL_SEEDS) * 2
    if len(inventory) != expected_sources or inventory.duplicated(inv_keys).any():
        raise RuntimeError("A9 source inventory is incomplete")
    for column in ("source_cache_path", "source_signature", "representation"):
        if column not in inventory:
            raise RuntimeError(f"A9 source inventory lacks {column}")
    for path in inventory["source_cache_path"].map(Path):
        if not path.is_file():
            raise FileNotFoundError(f"locked A9 source cache missing: {path}")

    parameter_records = parameters.sort_values(keys).to_dict("records")
    inventory_records = inventory.sort_values(inv_keys).to_dict("records")
    input_hashes = {name: a1.file_sha256(path) for name, path in files.items()}
    frozen = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "a9_root": str(root),
        "a9_manifest_hash": input_hashes["manifest"],
        "a9_decision_hash": input_hashes["decision"],
        "a9_input_hashes": input_hashes,
        "a9_source_script_hash": manifest.get("script_hash"),
        "a9_protocol_hashes": {d: protocol[d]["protocol_hash"] for d in DOMAINS},
        "architecture": ARCHITECTURE,
        "representations": [BASE, AGE],
        "model_seeds": list(MODEL_SEEDS),
        "target_split_seeds": list(TARGET_SPLIT_SEEDS),
        "role_partitions": list(ROLE_PARTITIONS),
        "target_epochs": 10,
        "high_rul_threshold": HIGH_RUL_THRESHOLD,
        "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "prediction_gate_threshold": 60.0,
        "formula": "baseline + alpha * (cycle_age - baseline)",
        "selection_only_parameters": parameter_records,
        "source_inventory": inventory_records,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    frozen["policy_hash"] = a1.canonical_hash(frozen)
    return frozen


def policy_table(policy: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(policy["selection_only_parameters"]).sort_values(PAIR_KEYS)


def cache_lookup(policy: dict[str, Any], domain: str, seed: int, representation: str) -> dict[str, Any]:
    for row in policy["source_inventory"]:
        if str(row["target_domain"]) == domain and int(row["model_seed"]) == seed and str(row["representation"]) == representation:
            return dict(row)
    raise KeyError(f"locked source cache not found for {domain}/{seed}/{representation}")


def load_locked_source(policy: dict[str, Any], domain: str, seed: int, representation: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    record = cache_lookup(policy, domain, seed, representation)
    cache = a1.safe_torch_load(Path(record["source_cache_path"]))
    inventory = dict(cache.get("inventory", {}))
    if cache.get("signature") != record.get("source_signature") or inventory.get("source_signature") != record.get("source_signature"):
        raise RuntimeError("locked A9 source cache signature changed")
    if inventory.get("representation") != representation or int(inventory.get("model_seed", -1)) != seed:
        raise RuntimeError("locked A9 source cache metadata mismatch")
    return cache["state"], record


def official_loader(cfg: dict[str, Any], data: dict[str, Any], representation: str, domain: str):
    """The only function in A9_1 allowed to open test/RUL files."""
    paths = a3.official_file_paths(cfg["data_dir"], domain)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing official {name} file: {path}")
    _, test, final_rul = a3.load_domain(cfg["data_dir"], domain)
    expected = int(a3.exp7.EXPECTED_OFFICIAL_TEST_ENGINES[domain])
    if test["unit"].nunique() != expected:
        raise RuntimeError(f"official {domain} test engine count changed")
    labeled = add_test_rul(test, final_rul, cfg["rul_cap"])
    _, normalizer = a1.fit_source_normalizer_train_only(cfg, "condition_settings")
    frame = normalizer.transform(labeled, list(cfg["sensor_columns"]))
    if representation == AGE:
        frame = a8.append_age_feature(frame, data["age_spec"])
    loader = a1.make_loader(frame, data["features"], cfg, training=False, last_only=True, loader_seed=int(cfg["seed"]) + 9900)
    units = np.asarray(loader.dataset.units, dtype=int)
    if len(units) != expected or len(np.unique(units)) != expected:
        raise RuntimeError("official loader must produce exactly one prediction per test engine")
    audit = {
        "target_domain": domain,
        "official_test_engine_count": expected,
        "official_test_units_hash": hashlib.sha256(units.tobytes()).hexdigest(),
        "test_file_sha256": a1.file_sha256(paths["test"]),
        "rul_file_sha256": a1.file_sha256(paths["rul"]),
        "official_test_files_accessed": True,
        "official_test_forward_run": True,
    }
    return loader, audit


def support_loader(data: dict[str, Any], cfg: dict[str, Any], experiment: dict[str, Any], support_units: list[int]):
    frame = data["target_frame"]
    support = frame[frame["unit"].isin(support_units)].copy()
    if support["unit"].nunique() != len(support_units):
        raise RuntimeError("target adaptation support engines are incomplete")
    return a1.make_loader(support, data["features"], cfg, training=True, balance_mode=experiment["balance_mode"], loader_seed=int(cfg["seed"]) + 9000)


def adapt_then_predict(
    *, base: dict[str, Any], experiment: dict[str, Any], protocol: dict[str, Any], policy: dict[str, Any],
    domain: str, model_seed: int, split_seed: int, representation: str, data: dict[str, Any],
    source_state: dict[str, torch.Tensor], prior: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_seed = a2.target_run_seed(domain, model_seed, split_seed)
    cfg = deepcopy(base)
    cfg.update({"seed": int(run_seed), "target_domain": domain, "source_domains": protocol["source_domains"]})
    support_units = list(map(int, protocol["role_splits"][str(split_seed)]["adaptation_units"]))
    support = support_loader(data, cfg, experiment, support_units)
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(ARCHITECTURE, len(data["features"]), cfg, prior, prior)
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    states, history = a3.train_target_to_locked_epochs(model, support, cfg, device, {int(experiment["target_epochs"])})
    # Test files are first opened only after fixed target adaptation is complete.
    loader, audit = official_loader(cfg, data, representation, domain)
    model.load_state_dict(states[int(experiment["target_epochs"])])
    model.to(device)
    prediction = a1.predict_with_units(model, loader, device)
    if prediction["unit"].nunique() != audit["official_test_engine_count"] or len(prediction) != audit["official_test_engine_count"]:
        raise RuntimeError("invalid official endpoint prediction count")
    if not np.isfinite(prediction[["label", "prediction"]].to_numpy(dtype=float)).all():
        raise RuntimeError("official prediction contains NaN/Inf")
    history.insert(0, "experiment_id", EXPERIMENT_ID)
    history.insert(1, "cell_id", cell_id(domain, model_seed, split_seed))
    history.insert(2, "target_domain", domain)
    history.insert(3, "representation", representation)
    history.insert(4, "model_seed", model_seed)
    history.insert(5, "target_split_seed", split_seed)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, history, audit


def annotate(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column, value in reversed(list(metadata.items())):
        scalar = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
        out.insert(0, column, scalar)
    return out


def run_metrics(frame: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for variant, column in ((BASE, "prediction_baseline"), (AGE, "prediction_cycle_age"), (BLEND, "prediction_blend")):
        risk = a9.risk(frame, column)
        result.append({**metadata, "variant": variant, **risk, "official_test_engine_count": int(frame["unit"].nunique()), "official_test_files_accessed": True, "official_test_forward_run": True})
    return result


def load_worker_state(paths: dict[str, Path]) -> dict[str, Any]:
    complete: set[str] = set()
    if paths["status"].is_file():
        complete = set(read_json(paths["status"]).get("completed_cell_ids", []))
    records = read_json(paths["run_json"]).get("records", []) if paths["run_json"].is_file() else []
    records = [r for r in records if r.get("cell_id") in complete]
    state = {"completed": complete, "records": records, "predictions": load_csv(paths["predictions"]), "history": load_csv(paths["history"]), "inventory": load_csv(paths["inventory"])}
    for key in ("predictions", "history"):
        if not state[key].empty:
            state[key] = state[key][state[key]["cell_id"].isin(complete)]
    return state


def save_worker_state(paths: dict[str, Path], state: dict[str, Any], expected: int, audit: dict[str, Any] | None) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["run_json"], {"records": state["records"]})
    a1.atomic_write_text(paths["run_csv"], pd.DataFrame(state["records"]).to_csv(index=False))
    for name in ("predictions", "history", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    if audit is not None:
        atomic_json(paths["test_audit"], audit)
    atomic_json(paths["status"], {
        "completed_cell_ids": sorted(state["completed"]),
        "completed_training_cells": len(state["completed"]),
        "expected_training_cells": expected,
        "completed_official_evaluation_records": len(state["records"]),
        "expected_official_evaluation_records": expected * len(ROLE_PARTITIONS) * 3,
        "complete": len(state["completed"]) == expected,
        "official_test_files_accessed": bool(state["completed"]),
        "official_test_forward_run": bool(state["completed"]),
    })


def worker_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in DOMAINS or model_seed not in MODEL_SEEDS or not args.confirm_official_test:
        raise RuntimeError("unregistered A9_1 worker or missing official-test acknowledgement")
    output, paths = Path(base["output_dir"]), shard_paths(Path(base["output_dir"]), domain, model_seed)
    policy = read_json(root_paths(output)["locked_policy"])
    if policy.get("policy_hash") != experiment.get("locked_policy_hash"):
        raise RuntimeError("worker locked policy hash differs from parent")
    protocols = read_json(Path(experiment["a9_output_dir"]) / "experimentA9_protocol.json")
    protocol = protocols[domain]
    worker_base = deepcopy(base)
    worker_base.update({"output_dir": str(paths["directory"]), "target_domain": domain, "source_domains": protocol["source_domains"]})
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(worker_base, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    worker_manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "target_domain": domain, "model_seed": model_seed, "locked_policy_hash": policy["policy_hash"], "a9_protocol_hash": protocol["protocol_hash"], "graph_fit": graph_fit, "official_test_access_requires_explicit_flag": True}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("target_domain", "model_seed", "locked_policy_hash", "a9_protocol_hash"):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(f"existing A9_1 shard is incompatible at {key}")
        if previous.get("script_hash") != worker_manifest["script_hash"]:
            if not args.resume:
                raise RuntimeError("A9_1 script changed; use --resume only after checking locked policy")
            worker_manifest["resumed_from_script_hash"] = previous.get("script_hash")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["manifest"], worker_manifest)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(paths["directory"] / "source_prior_adjacency.csv", pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv())
    a1.atomic_write_text(paths["directory"] / "source_prior_correlation.csv", pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv())
    state = load_worker_state(paths)
    params = policy_table(policy)
    expected = len(TARGET_SPLIT_SEEDS)
    pending = [split for split in TARGET_SPLIT_SEEDS if cell_id(domain, model_seed, split) not in state["completed"]]
    audit = read_json(paths["test_audit"]) if paths["test_audit"].is_file() else None
    if pending:
        data, source = {}, {}
        for representation in (BASE, AGE):
            data[representation] = a8.prepare_representation_data(worker_base, representation)
            source[representation] = load_locked_source(policy, domain, model_seed, representation)
        inventory = pd.DataFrame([{**record, "target_domain": domain, "official_test_files_accessed": True, "official_test_forward_run": True} for _, record in source.values()])
        state["inventory"] = inventory
        for split_seed in pending:
            by_rep, history_parts, current_audit = {}, [], None
            for representation in (BASE, AGE):
                state_dict, _ = source[representation]
                pred, history, one_audit = adapt_then_predict(base=worker_base, experiment=experiment, protocol=protocol, policy=policy, domain=domain, model_seed=model_seed, split_seed=split_seed, representation=representation, data=data[representation], source_state=deepcopy(state_dict), prior=prior)
                by_rep[representation] = pred.sort_values("unit").reset_index(drop=True)
                history_parts.append(history)
                if current_audit is None:
                    current_audit = one_audit
                elif any(current_audit[k] != one_audit[k] for k in ("official_test_engine_count", "official_test_units_hash", "test_file_sha256", "rul_file_sha256")):
                    raise RuntimeError("baseline/cycle-age official test integrity mismatch")
            if audit is not None and any(audit[k] != current_audit[k] for k in ("official_test_engine_count", "official_test_units_hash", "test_file_sha256", "rul_file_sha256")):
                raise RuntimeError("official test integrity changed during worker")
            audit = current_audit
            if not np.array_equal(by_rep[BASE]["unit"].to_numpy(), by_rep[AGE]["unit"].to_numpy()) or not np.allclose(by_rep[BASE]["label"], by_rep[AGE]["label"]):
                raise RuntimeError("official baseline/cycle-age endpoint alignment failed")
            wide = by_rep[BASE][["unit", "label", "prediction"]].rename(columns={"prediction": "prediction_baseline"})
            wide["prediction_cycle_age"] = by_rep[AGE]["prediction"].to_numpy(dtype=float)
            wide["prediction_mean_for_gate"] = (wide["prediction_baseline"] + wide["prediction_cycle_age"]) / 2.0
            wide["prediction_gate"] = np.where(wide["prediction_mean_for_gate"] >= 60.0, "high_pred_rul_ge60", "lower_pred_rul_lt60")
            selected = params[(params.target_domain == domain) & (params.model_seed == model_seed) & (params.target_split_seed == split_seed)]
            if len(selected) != len(ROLE_PARTITIONS):
                raise RuntimeError("locked A9 alpha policy lacks role-partition rows")
            common_base = {"experiment_id": EXPERIMENT_ID, "cell_id": cell_id(domain, model_seed, split_seed), "target_domain": domain, "model_seed": model_seed, "target_split_seed": split_seed, "target_run_seed": a2.target_run_seed(domain, model_seed, split_seed), "architecture": ARCHITECTURE, "fixed_budget_epoch": int(experiment["target_epochs"]), "selection_was_locked_before_official_test": True, "locked_policy_hash": policy["policy_hash"], "official_test_files_accessed": True, "official_test_forward_run": True}
            for row in selected.itertuples(index=False):
                applied = a9.blend(wide, float(row.alpha_high), float(row.alpha_low))
                metadata = {**common_base, "role_partition": int(row.role_partition), "alpha_high": float(row.alpha_high), "alpha_low": float(row.alpha_low), "selection_safety_feasible": bool(row.selection_safety_feasible), "fallback_to_baseline": bool(row.fallback_to_baseline)}
                state["records"].extend(run_metrics(applied, metadata))
                state["predictions"] = pd.concat([state["predictions"], annotate(applied, metadata)], ignore_index=True)
            state["history"] = pd.concat([state["history"], *history_parts], ignore_index=True)
            state["completed"].add(cell_id(domain, model_seed, split_seed))
            save_worker_state(paths, state, expected, audit)
    save_worker_state(paths, state, expected, audit)
    print(paths["status"].read_text(encoding="utf-8"))


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]]]:
    inventory = a2.query_gpus()
    if args.gpus:
        devices = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("--gpus needs unique indices")
        if not set(devices).issubset({row["index"] for row in inventory}):
            raise RuntimeError("requested GPU does not exist")
    else:
        visible = a2.visible_gpu_filter()
        devices = [row["index"] for row in sorted(inventory, key=lambda x: (-x["free_mb"], x["utilization"])) if (visible is None or row["index"] in visible) and row["free_mb"] >= args.min_free_memory_mb and row["utilization"] <= args.max_gpu_utilization]
    return (devices[:args.max_workers] if args.max_workers > 0 else devices), inventory


def worker_command(args: argparse.Namespace, domain: str, seed: int, device: str, output: Path) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-domain", domain, "--worker-seed", str(seed), "--output-dir", str(output), "--device", device, "--bootstrap-repetitions", str(args.bootstrap_repetitions), "--confirm-official-test"]
    for option, value in (("--data-dir", args.data_dir), ("--a9-output-dir", args.a9_output_dir)):
        if value: command.extend([option, value])
    if args.resume: command.append("--resume")
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]; inventory: list[dict[str, Any]] = []
    else:
        devices, inventory = choose_gpus(args)
        if not devices: raise RuntimeError("no idle GPU met A9_1 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(json.dumps({"scheduler": EXPERIMENT_ID, "tasks": [{"domain": d, "seed": s} for d, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending, active = list(tasks), {}
    while pending or active:
        for device in [d for d in devices if d not in active]:
            if not pending: break
            domain, seed = pending.pop(0); paths = shard_paths(output, domain, seed); paths["directory"].mkdir(parents=True, exist_ok=True)
            handle = (paths["directory"] / "worker_training.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            if isinstance(device, int): environment["CUDA_VISIBLE_DEVICES"] = str(device); command = worker_command(args, domain, seed, "auto", output)
            else: command = worker_command(args, domain, seed, str(device), output)
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active[device] = {"process": process, "domain": domain, "seed": seed, "handle": handle, "log": paths["directory"] / "worker_training.log"}
            print(f"[A9_1] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished = []
        for device, item in active.items():
            code = item["process"].poll()
            if code is None: continue
            item["handle"].close()
            if code != 0:
                tail = "\n".join(item["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                for other in active.values():
                    if other["process"].poll() is None: other["process"].terminate()
                raise RuntimeError(f"A9_1 worker failed domain={item['domain']} seed={item['seed']} exit={code}\n{tail}")
            print(f"[A9_1] completed domain={item['domain']} seed={item['seed']} device={device}")
            finished.append(device)
        for device in finished: del active[device]
        if active and not finished: time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records, predictions, history, inventory, audits = [], [], [], [], []
    for domain, seed in tasks:
        paths = shard_paths(output, domain, seed); status = read_json(paths["status"])
        if not status.get("complete") or int(status.get("completed_training_cells", -1)) != len(TARGET_SPLIT_SEEDS):
            raise RuntimeError(f"incomplete A9_1 shard: {paths['status']}")
        if not status.get("official_test_files_accessed") or not status.get("official_test_forward_run"):
            raise RuntimeError(f"A9_1 missing official-test audit flags: {paths['status']}")
        records.extend(read_json(paths["run_json"])["records"])
        predictions.append(load_csv(paths["predictions"])); history.append(load_csv(paths["history"])); inventory.append(load_csv(paths["inventory"])); audits.append(read_json(paths["test_audit"]))
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True), pd.concat(history, ignore_index=True), pd.concat(inventory, ignore_index=True), audits


def pair_results(results: pd.DataFrame, candidate: str = BLEND) -> pd.DataFrame:
    metrics = list(a8.METRICS)
    pivot = results.pivot(index=PAIR_KEYS, columns="variant", values=metrics).reset_index()
    pivot.columns = ["_".join(str(v) for v in c if str(v)) if isinstance(c, tuple) else c for c in pivot.columns]
    out = pivot[PAIR_KEYS].copy()
    for metric in metrics:
        out[f"{metric}_{BASE}"] = pivot[f"{metric}_{BASE}"].astype(float); out[f"{metric}_{candidate}"] = pivot[f"{metric}_{candidate}"].astype(float)
        out[f"{metric}_delta_candidate_minus_baseline"] = out[f"{metric}_{candidate}"] - out[f"{metric}_{BASE}"]
    out["candidate"] = candidate; out["nasa_relative_delta"] = out["nasa_score_delta_candidate_minus_baseline"] / out[f"nasa_score_{BASE}"]; out["rmse_relative_delta"] = out["rmse_delta_candidate_minus_baseline"] / out[f"rmse_{BASE}"]
    out["candidate_nasa_win"] = out["nasa_score_delta_candidate_minus_baseline"] < 0; out["candidate_rmse_win"] = out["rmse_delta_candidate_minus_baseline"] < 0
    return out.sort_values(PAIR_KEYS)


def stage_pairs(predictions: pd.DataFrame, high: bool) -> pd.DataFrame:
    stage = predictions[predictions["label"] > HIGH_RUL_THRESHOLD].copy() if high else predictions[predictions["label"] <= HIGH_RUL_THRESHOLD].copy()
    rows = []
    for values, group in stage.groupby(PAIR_KEYS):
        base, candidate = a9.risk(group, "prediction_baseline"), a9.risk(group, "prediction_blend")
        row = dict(zip(PAIR_KEYS, values)); row.update({"rul_stage": "high_rul_gt60" if high else "low_or_mid_rul_le60", "rul_threshold": HIGH_RUL_THRESHOLD, "stage_engine_count": int(group.unit.nunique()), "candidate": BLEND})
        for metric in a8.METRICS:
            row[f"{metric}_{BASE}"] = base[metric]; row[f"{metric}_{BLEND}"] = candidate[metric]; row[f"{metric}_delta_candidate_minus_baseline"] = candidate[metric] - base[metric]
        row["nasa_relative_delta"] = a9.rel(candidate, base, "nasa_score"); row["rmse_relative_delta"] = a9.rel(candidate, base, "rmse")
        row["candidate_nasa_win"] = candidate["nasa_score"] < base["nasa_score"]; row["candidate_rmse_win"] = candidate["rmse"] < base["rmse"]
        rows.append(row)
    out = pd.DataFrame(rows)
    expected = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS)
    if len(out) != expected: raise RuntimeError("some locked official pairs have no requested RUL stage")
    return out.sort_values(PAIR_KEYS)


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> tuple[float, float]:
    """Bootstrap the official test hierarchy: domain -> model -> split -> role.

    A9's training-only confirmation has an extra endpoint-assignment level.
    Official test endpoints do not: the locked role policy is applied to the
    same official unit set.  Reusing A4's five-level bootstrap here would
    silently require a nonexistent endpoint_seed column, so A9_1 has its own
    four-level version.
    """
    domains = sorted(frame.target_domain.unique())
    models = sorted(frame.model_seed.unique())
    splits = sorted(frame.target_split_seed.unique())
    roles = sorted(frame.role_partition.unique())
    lookup = frame.set_index(PAIR_KEYS)[column]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        values = []
        for domain in rng.choice(domains, len(domains), replace=True):
            chosen_models = rng.choice(models, len(models), replace=True)
            chosen_splits = rng.choice(splits, len(splits), replace=True)
            for model_seed in chosen_models:
                for split_seed in chosen_splits:
                    role = int(rng.choice(roles))
                    values.append(float(lookup.loc[(domain, int(model_seed), int(split_seed), role)]))
        samples[index] = float(np.mean(values))
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def comparison_summary(pairs: pd.DataFrame, experiment: dict[str, Any], label: str) -> pd.DataFrame:
    rows = []
    for scope, frame in [("ALL", pairs)] + list(pairs.groupby("target_domain")):
        nasa_ci = hierarchical_bootstrap(frame, "nasa_relative_delta", int(experiment["bootstrap_repetitions"]), stable_seed(EXPERIMENT_ID, label, "nasa", scope))
        rmse_ci = hierarchical_bootstrap(frame, "rmse_relative_delta", int(experiment["bootstrap_repetitions"]), stable_seed(EXPERIMENT_ID, label, "rmse", scope))
        rows.append({
            "comparison": label, "scope": scope, "n_records": int(len(frame)),
            "nasa_score_delta_mean": float(frame.nasa_score_delta_candidate_minus_baseline.mean()),
            "nasa_improvement_pct": float(-100 * frame.nasa_relative_delta.mean()),
            "nasa_relative_boot_ci95_low": nasa_ci[0], "nasa_relative_boot_ci95_high": nasa_ci[1],
            "nasa_win_rate": float(frame.candidate_nasa_win.mean()),
            "rmse_delta_mean": float(frame.rmse_delta_candidate_minus_baseline.mean()),
            "rmse_degradation_pct": float(100 * frame.rmse_relative_delta.mean()),
            "rmse_relative_boot_ci95_low": rmse_ci[0], "rmse_relative_boot_ci95_high": rmse_ci[1],
            "rmse_win_rate": float(frame.candidate_rmse_win.mean()),
            "late_error_q95_delta_mean": float(frame.late_error_q95_delta_candidate_minus_baseline.mean()),
            "under_error_q95_delta_mean": float(frame.under_error_q95_delta_candidate_minus_baseline.mean()),
            "mean_error_delta_mean": float(frame.mean_error_delta_candidate_minus_baseline.mean()),
        })
    return pd.DataFrame(rows)


def stage_summary(pairs: pd.DataFrame, experiment: dict[str, Any], label: str) -> dict[str, Any]:
    nasa_ci = hierarchical_bootstrap(pairs, "nasa_relative_delta", int(experiment["bootstrap_repetitions"]), stable_seed(EXPERIMENT_ID, label, "nasa"))
    rmse_ci = hierarchical_bootstrap(pairs, "rmse_relative_delta", int(experiment["bootstrap_repetitions"]), stable_seed(EXPERIMENT_ID, label, "rmse"))
    return {
        "stage": label, "n_records": int(len(pairs)),
        "nasa_improvement_pct": float(-100 * pairs.nasa_relative_delta.mean()),
        "nasa_relative_ci95": [nasa_ci[0], nasa_ci[1]],
        "rmse_degradation_pct": float(100 * pairs.rmse_relative_delta.mean()),
        "rmse_relative_ci95": [rmse_ci[0], rmse_ci[1]],
        "mean_error_delta_mean": float(pairs.mean_error_delta_candidate_minus_baseline.mean()),
        "nasa_win_rate": float(pairs.candidate_nasa_win.mean()),
        "rmse_win_rate": float(pairs.candidate_rmse_win.mean()),
    }


def integrity(audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"official_test_files_accessed": True, "official_test_forward_run": True, "test_audits": audits, "unique_test_hash_sets": sorted((a["target_domain"], a["test_file_sha256"], a["rul_file_sha256"], a["official_test_units_hash"]) for a in audits)}


def make_decision(results: pd.DataFrame, pairs: pd.DataFrame, high_pairs: pd.DataFrame, low_pairs: pd.DataFrame, summary: pd.DataFrame, policy: dict[str, Any], audits: list[dict[str, Any]], experiment: dict[str, Any]) -> dict[str, Any]:
    expected_cells = len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS)
    expected_pairs = expected_cells * len(ROLE_PARTITIONS)
    full = summary.query("comparison == 'official_blend_vs_baseline' and scope == 'ALL'").iloc[0]
    high = stage_summary(high_pairs, experiment, "official_high_rul_gt60")
    low = stage_summary(low_pairs, experiment, "official_low_or_mid_rul_le60")
    margin = float(experiment["stage_noninferiority_margin_pct"]) / 100.0
    full_ok = float(full.nasa_relative_boot_ci95_high) < 0 or float(full.rmse_relative_boot_ci95_high) < 0
    high_ok = high["nasa_relative_ci95"][1] <= margin and high["rmse_relative_ci95"][1] <= margin
    low_ok = low["nasa_relative_ci95"][1] <= margin and low["rmse_relative_ci95"][1] <= margin
    complete = results.cell_id.nunique() == expected_cells and len(results) == expected_pairs * 3 and len(pairs) == expected_pairs and len(high_pairs) == expected_pairs and len(low_pairs) == expected_pairs
    flags = bool(results[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).all().all())
    passed = bool(complete and flags and full_ok and high_ok and low_ok)
    return {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "architecture": ARCHITECTURE, "complete": bool(complete), "expected_training_cells": expected_cells * 2, "completed_training_cells": int(results.cell_id.nunique()) * 2, "expected_official_evaluation_records": expected_pairs * 3, "completed_official_evaluation_records": int(len(results)), "expected_primary_pairs": expected_pairs, "completed_primary_pairs": int(len(pairs)), "fixed_budget_epoch": 10, "locked_policy_hash": policy["policy_hash"], "selection_was_locked_before_official_test": True, "official_test_files_accessed": True, "official_test_forward_run": True, "full_endpoint_result": {"nasa_improvement_pct": float(full.nasa_improvement_pct), "nasa_relative_ci95": [float(full.nasa_relative_boot_ci95_low), float(full.nasa_relative_boot_ci95_high)], "rmse_degradation_pct": float(full.rmse_degradation_pct), "rmse_relative_ci95": [float(full.rmse_relative_boot_ci95_low), float(full.rmse_relative_boot_ci95_high)], "at_least_one_metric_strictly_improved": bool(full_ok)}, "high_rul_safety_result": {**high, "noninferiority_passed": bool(high_ok)}, "low_rul_safety_result": {**low, "noninferiority_passed": bool(low_ok)}, "passed": passed, "reason": "A9_1 confirmed the locked A9 blend on official C-MAPSS endpoints" if passed else "A9_1 completed official confirmation, but the locked blend did not meet every registered criterion", "next_action": "report_locked_official_A9_results" if passed else "stop_locked_A9_blend_claim_and_report_training_vs_official_gap"}


def write_initial(paths: dict[str, Path], base: dict[str, Any], experiment: dict[str, Any], policy: dict[str, Any], resume: bool) -> dict[str, Any]:
    manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "git_commit": a1.git_commit(PROJECT_ROOT), "base_config": {k: v for k, v in base.items() if k != "device"}, "experiment_config": experiment, "registered_primary_question": QUESTION, "locked_policy_hash": policy["policy_hash"], "a9_input_hashes": policy["a9_input_hashes"], "official_test_purpose": "A9 locked blend confirmation only; no official data may alter alpha, gate, budget, or architecture", "official_test_files_accessed": False, "official_test_forward_run": False}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("locked_policy_hash", "a9_input_hashes", "registered_primary_question"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"existing A9_1 output is incompatible at {key}")
        if previous.get("script_hash") != manifest["script_hash"]:
            if not resume: raise RuntimeError("A9_1 script changed after artifacts were made; use --resume only after review")
            manifest["resumed_from_script_hash"] = previous.get("script_hash")
    atomic_json(paths["manifest"], manifest)
    if paths["locked_policy"].is_file():
        if read_json(paths["locked_policy"]).get("policy_hash") != policy["policy_hash"]: raise RuntimeError("existing A9_1 locked policy differs from A9 inputs")
    else:
        atomic_json(paths["locked_policy"], policy); a1.atomic_write_text(paths["locked_policy_csv"], policy_table(policy).to_csv(index=False))
    return manifest


def parent_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    output = Path(base["output_dir"]); output.mkdir(parents=True, exist_ok=True); paths = root_paths(output)
    if args.confirm_official_test and paths["decision"].is_file(): raise RuntimeError("A9_1 official confirmation is complete and cannot be rerun in this output directory")
    policy = lock_a9_policy(experiment); experiment["locked_policy_hash"] = policy["policy_hash"]
    manifest = write_initial(paths, base, experiment, policy, bool(args.resume))
    dry = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "expected_training_cells": len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * 2, "expected_official_evaluation_records": len(DOMAINS) * len(MODEL_SEEDS) * len(TARGET_SPLIT_SEEDS) * len(ROLE_PARTITIONS) * 3, "locked_policy_hash": policy["policy_hash"], "a9_input_hashes": policy["a9_input_hashes"], "selection_parameter_sets": len(policy["selection_only_parameters"]), "official_test_files_accessed": False, "official_test_forward_run": False, "formal_run_requires": "--confirm-official-test", "gpu_inventory": a2.query_gpus()}
    atomic_json(paths["dry_run"], dry)
    if args.dry_run: print(json.dumps(dry, ensure_ascii=False, indent=2)); return
    shards = output / "shards"
    if shards.exists() and any(shards.iterdir()) and not args.resume: raise RuntimeError("A9_1 has interrupted shards; use --resume to avoid repeating official cells")
    tasks = [(domain, seed) for domain in DOMAINS for seed in MODEL_SEEDS]
    run_workers(args, tasks, output)
    results, predictions, history, inventory, audits = merge_shards(output, tasks)
    results = results.sort_values(PAIR_KEYS + ["variant"]); pairs = pair_results(results); high_pairs = stage_pairs(predictions, True); low_pairs = stage_pairs(predictions, False)
    summary = pd.concat([comparison_summary(pairs, experiment, "official_blend_vs_baseline"), comparison_summary(high_pairs, experiment, "official_high_rul_blend_vs_baseline"), comparison_summary(low_pairs, experiment, "official_low_rul_blend_vs_baseline")], ignore_index=True)
    decision = make_decision(results, pairs, high_pairs, low_pairs, summary, policy, audits, experiment)
    atomic_json(paths["run_json"], {"records": results.to_dict("records")}); a1.atomic_write_text(paths["run_csv"], results.to_csv(index=False)); a1.atomic_write_text(paths["predictions"], predictions.to_csv(index=False)); a1.atomic_write_text(paths["history"], history.to_csv(index=False)); a1.atomic_write_text(paths["inventory"], inventory.to_csv(index=False)); a1.atomic_write_text(paths["paired"], pairs.to_csv(index=False)); a1.atomic_write_text(paths["high"], high_pairs.to_csv(index=False)); a1.atomic_write_text(paths["low"], low_pairs.to_csv(index=False)); a1.atomic_write_text(paths["summary"], summary.to_csv(index=False)); atomic_json(paths["integrity"], integrity(audits)); atomic_json(paths["decision"], decision)
    manifest.update({"official_test_files_accessed": True, "official_test_forward_run": True, "official_test_integrity_file": paths["integrity"].name, "confirmation_decision_file": paths["decision"].name}); atomic_json(paths["manifest"], manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args(); base, experiment = load_config(args); validate_args(base, experiment, args)
    if args.worker_domain is not None:
        if args.worker_seed is None: raise ValueError("--worker-domain requires --worker-seed")
        policy_path = root_paths(Path(base["output_dir"]))["locked_policy"]
        if not policy_path.is_file(): raise FileNotFoundError("A9_1 worker requires a parent-created locked policy")
        experiment["locked_policy_hash"] = read_json(policy_path)["policy_hash"]
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
