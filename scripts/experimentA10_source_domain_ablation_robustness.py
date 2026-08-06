"""Experiment A10: training-only source-domain ablation robustness check.

This experiment does not revisit the official test set.  For each target FD it
removes one of the three source domains, freshly pretrains baseline and causal
cycle-age models with only the remaining two sources, then performs A9's
selection-only bounded blend on disjoint training engines.

The design answers a new question: is A9's benefit robust to a missing source
domain, rather than an artefact of one particular three-source pool?
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import product
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

from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402
from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402


SCRIPT_VERSION = "experimentA10_source_domain_ablation_robustness_v1"
EXPERIMENT_ID = "experimentA10"
DEFAULT_OUTPUT = "outputs/experimentA10_source_domain_ablation_robustness"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODEL_SEEDS = (100, 101, 102, 103, 104)
TARGET_SPLIT_SEEDS = (6401, 6402, 6403, 6404, 6405)
ROLE_PARTITIONS = (1, 2, 3, 4, 5)
SELECTION_ENDPOINT_SEEDS = (9001, 9002, 9003, 9004, 9005)
CONFIRMATION_ENDPOINT_SEEDS = (9101, 9102, 9103, 9104, 9105)
BASE, AGE, BLEND = a9.BASE, a9.AGE, a9.BLEND
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_THRESHOLD, MARGIN = 60.0, 0.03
PAIR_KEYS = ["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"]
CELL_KEYS = ["target_domain", "heldout_source_domain", "model_seed", "target_split_seed"]
QUESTION = "Does A9's selection-only causal cycle-age blend retain endpoint efficacy and high/low-RUL safety when one available source domain is omitted?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A10 source-domain ablation robustness")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-1-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,2,4")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-heldout", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


def atomic_json(path: Path, value: Any) -> None:
    a1.atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required A10 input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def source_options(target_domain: str) -> tuple[str, ...]:
    return tuple(domain for domain in DOMAINS if domain != target_domain)


def active_sources(target_domain: str, heldout: str) -> tuple[str, str]:
    sources = source_options(target_domain)
    if heldout not in sources:
        raise ValueError(f"{heldout} cannot be held out when target is {target_domain}")
    kept = tuple(domain for domain in sources if domain != heldout)
    if len(kept) != 2:
        raise AssertionError("A10 requires exactly two active source domains")
    return kept


def cell_id(domain: str, heldout: str, seed: int, split: int, representation: str) -> str:
    return f"{EXPERIMENT_ID}_{domain.lower()}_drop{heldout.lower()}_{representation}_mseed{seed:03d}_tsplit{split}"


def root_paths(output: Path) -> dict[str, Path]:
    p = EXPERIMENT_ID
    return {
        "manifest": output / f"{p}_manifest.json", "protocol": output / f"{p}_protocol.json",
        "roles": output / f"{p}_engine_roles.csv", "dry": output / f"{p}_dry_run.json",
        "causality": output / f"{p}_feature_causality_audit.json", "age_audit": output / f"{p}_age_feature_audit.csv",
        "inventory": output / f"{p}_source_pretrain_inventory.csv", "source_history": output / f"{p}_source_pretrain_history.csv",
        "endpoint": output / f"{p}_pool_endpoint_predictions.csv", "target_history": output / f"{p}_target_history.csv",
        "selection_prediction": output / f"{p}_selection_endpoint_predictions.csv", "confirmation_prediction": output / f"{p}_confirmation_endpoint_predictions.csv",
        "selection_run": output / f"{p}_selection_run_level.csv", "confirmation_run": output / f"{p}_confirmation_run_level.csv",
        "grid": output / f"{p}_blend_selection_grid.csv", "parameters": output / f"{p}_blend_parameters.csv",
        "paired": output / f"{p}_paired_blend_vs_baseline.csv", "high": output / f"{p}_high_rul_paired_blend_vs_baseline.csv", "low": output / f"{p}_low_rul_paired_blend_vs_baseline.csv",
        "summary": output / f"{p}_comparison_summary.csv", "ablation": output / f"{p}_source_ablation_summary.csv", "decision": output / f"{p}_confirmation_decision.json",
    }


def shard_paths(output: Path, domain: str, heldout: str, seed: int) -> dict[str, Path]:
    directory = output / "shards" / f"{domain}_drop{heldout}_mseed{seed:03d}"
    return {
        "directory": directory, "manifest": directory / "worker_manifest.json", "status": directory / "worker_status.json",
        "endpoint": directory / "pool_endpoint_predictions.csv", "target_history": directory / "target_history.csv",
        "source_history": directory / "source_pretrain_history.csv", "inventory": directory / "source_pretrain_inventory.csv", "audit": directory / "age_feature_audit.csv",
    }


def cache_path(output: Path, target: str, heldout: str, seed: int, representation: str) -> Path:
    return shard_paths(output, target, heldout, seed)["directory"] / "source_cache" / f"{EXPERIMENT_ID}_{representation}_{target}_drop{heldout}_mseed{seed:03d}.pt"


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    base = deepcopy(a1.DEFAULT_BASE_CONFIG)
    base.update({"data_dir": resolved(args.data_dir, base["data_dir"]), "output_dir": resolved(args.output_dir, DEFAULT_OUTPUT), "normalizer_seed": 2026, "condition_count": 6, "source_pretrain_steps": 1500, "source_pretrain_lr": 0.001, "source_pretrain_weight_decay": 0.0, "target_epochs": 10, "target_lr": 0.001, "pair_aux_weight": 0.0, "device": args.device})
    experiment = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": "source_domain_ablation_robustness", "domains": list(DOMAINS),
        "architecture": "window_no_graph", "representations": [BASE, AGE], "model_seeds": list(MODEL_SEEDS), "target_split_seeds": list(TARGET_SPLIT_SEEDS), "role_partitions": list(ROLE_PARTITIONS),
        "selection_endpoint_seeds": list(SELECTION_ENDPOINT_SEEDS), "confirmation_endpoint_seeds": list(CONFIRMATION_ENDPOINT_SEEDS),
        "high_rul_threshold": 60.0, "k": 5, "preprocessing": "condition_settings", "balance_mode": "engine_stage", "sensor_graph_k": 4,
        "source_pretrain_steps": 1500, "target_epochs": 10, "fixed_budget_no_epoch_selection": True, "fresh_source_pretraining_per_ablation": True,
        "alpha_grid": list(ALPHA_GRID), "prediction_gate_threshold": GATE_THRESHOLD, "selection_safety_margin_pct": 3.0, "stage_noninferiority_margin_pct": 3.0,
        "minimum_passing_ablation_conditions": 9, "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "a2_1_output_dir": resolved(args.a2_1_output_dir, a4.DEFAULT_A2_1_OUTPUT), "output_dir": base["output_dir"], "quick_mode": bool(args.quick),
    }
    if args.quick:
        experiment.update({"domains": ["FD004"], "model_seeds": [100], "target_split_seeds": [6401], "role_partitions": [1], "selection_endpoint_seeds": [9001], "confirmation_endpoint_seeds": [9101], "bootstrap_repetitions": 100, "minimum_passing_ablation_conditions": 1})
        base.update({"target_epochs": 2, "source_pretrain_steps": 20}); experiment.update({"target_epochs": 2, "source_pretrain_steps": 20})
        if args.output_dir is None: base["output_dir"] = resolved(None, DEFAULT_OUTPUT + "_quick"); experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate(base: dict[str, Any], experiment: dict[str, Any]) -> None:
    if set(experiment["selection_endpoint_seeds"]) & set(experiment["confirmation_endpoint_seeds"]):
        raise ValueError("selection/confirmation endpoint seeds must be disjoint")
    for domain in DOMAINS:
        if not a1.train_path(base["data_dir"], domain).is_file():
            raise FileNotFoundError(f"missing training file: {domain}")


def source_signature(base: dict[str, Any], experiment: dict[str, Any], target: str, heldout: str, representation: str, model_seed: int, data: dict[str, Any], prior: torch.Tensor) -> str:
    sources = active_sources(target, heldout)
    return a1.canonical_hash({"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "target": target, "heldout_source_domain": heldout, "active_source_domains": list(sources), "representation": representation, "model_seed": model_seed, "features": data["features"], "age_audit": data["audit"], "architecture": experiment["architecture"], "source_pretrain_steps": base["source_pretrain_steps"], "source_pretrain_lr": base["source_pretrain_lr"], "train_file_hashes": {d: a1.file_sha256(a1.train_path(base["data_dir"], d)) for d in sources}, "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest()})


def train_or_load_source(output: Path, base: dict[str, Any], experiment: dict[str, Any], target: str, heldout: str, representation: str, model_seed: int, data: dict[str, Any], prior: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    signature = source_signature(base, experiment, target, heldout, representation, model_seed, data, prior)
    path = cache_path(output, target, heldout, model_seed, representation)
    if path.is_file():
        saved = a1.safe_torch_load(path)
        if saved.get("signature") == signature:
            return saved["state"], saved.get("history", []), dict(saved["inventory"])
    cfg = deepcopy(base); cfg.update({"seed": model_seed, "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    a1.seed_everything(model_seed)
    model = exp17b.build_model_17b(experiment["architecture"], len(data["features"]), cfg, prior, prior)
    total, predictor = a8.exp17.parameter_count(model)
    model, history = a8.train_source_supervised(model, a8.make_source_tasks(data, cfg, experiment), cfg, a1.resolve_device(cfg["device"]))
    inventory = {"target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "representation": representation, "model": experiment["architecture"], "model_seed": model_seed, "feature_count": len(data["features"]), "feature_columns": json.dumps(data["features"]), "total_parameter_count": total, "predictor_parameter_count": predictor, "source_pretrain_steps": int(cfg["source_pretrain_steps"]), "source_signature": signature, "source_cache_origin": "experimentA10_fresh_two_source_cache", "source_cache_path": str(path), "source_pretraining_reused_from_prior_experiment": False, "official_test_files_accessed": False, "official_test_forward_run": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "state": a1.state_to_cpu(model), "history": history, "inventory": inventory}, path)
    state = a1.state_to_cpu(model); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return state, history, inventory


def run_cell(base: dict[str, Any], experiment: dict[str, Any], protocol: dict[str, Any], target: str, heldout: str, representation: str, model_seed: int, split_seed: int, data: dict[str, Any], source_state: dict[str, torch.Tensor], source_history: list[dict[str, Any]], inventory: dict[str, Any], prior: torch.Tensor) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_seed = a2.target_run_seed(target, model_seed, split_seed)
    cfg = deepcopy(base); cfg.update({"seed": run_seed, "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    split = protocol["role_splits"][str(split_seed)]
    support, pool = a8.prepare_support_pool(data, cfg, experiment, list(map(int, split["adaptation_units"])), list(map(int, split["evaluation_pool_units"])))
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(experiment["architecture"], len(data["features"]), cfg, prior, prior); model.load_state_dict(source_state)
    predictions, history = a4.train_fixed_budget(model, support, pool, cfg, a1.resolve_device(cfg["device"]), "symmetric_mse", 1.0)
    endpoint = a21.endpoint_epoch_rows(predictions, int(experiment["target_epochs"]))
    common = {"experiment_id": EXPERIMENT_ID, "cell_id": cell_id(target, heldout, model_seed, split_seed, representation), "target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "model": experiment["architecture"], "representation": representation, "model_seed": model_seed, "target_split_seed": split_seed, "target_run_seed": run_seed, "k": experiment["k"], "adaptation_units": json.dumps(list(map(int, split["adaptation_units"]))), "a2_1_protocol_hash": protocol["protocol_hash"], "feature_count": len(data["features"]), "source_signature": inventory["source_signature"], "source_cache_origin": inventory["source_cache_origin"], "source_history_rows": len(source_history), "official_test_files_accessed": False, "official_test_forward_run": False}
    for name, value in reversed(list(common.items())): endpoint.insert(0, name, value)
    history.insert(0, "experiment_id", EXPERIMENT_ID); history.insert(1, "cell_id", common["cell_id"]); history.insert(2, "target_domain", target); history.insert(3, "heldout_source_domain", heldout); history.insert(4, "representation", representation); history.insert(5, "model_seed", model_seed); history.insert(6, "target_split_seed", split_seed)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return endpoint, history


def load_state(paths: dict[str, Path]) -> dict[str, Any]:
    done = set(read_json(paths["status"]).get("completed_cell_ids", [])) if paths["status"].is_file() else set()
    state = {"completed": done, "endpoint": load_csv(paths["endpoint"]), "target_history": load_csv(paths["target_history"]), "source_history": load_csv(paths["source_history"]), "inventory": load_csv(paths["inventory"]), "audit": load_csv(paths["audit"])}
    for name in ("endpoint", "target_history"):
        if not state[name].empty: state[name] = state[name][state[name].cell_id.isin(done)]
    return state


def save_state(paths: dict[str, Path], state: dict[str, Any], expected: int) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in ("endpoint", "target_history", "source_history", "inventory", "audit"): a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(paths["status"], {"completed_cell_ids": sorted(state["completed"]), "completed_training_cells": len(state["completed"]), "expected_training_cells": expected, "endpoint_rows": len(state["endpoint"]), "source_representation_count": len(state["inventory"]), "complete": len(state["completed"]) == expected, "official_test_files_accessed": False, "official_test_forward_run": False})


def worker_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    target, heldout, model_seed = str(args.worker_domain), str(args.worker_heldout), int(args.worker_seed)
    if target not in experiment["domains"] or heldout not in source_options(target) or model_seed not in experiment["model_seeds"]: raise ValueError("unregistered A10 worker")
    protocols, evidence = a4.load_training_only_protocol(base, experiment); protocol = protocols[target]
    output, paths = Path(base["output_dir"]), shard_paths(Path(base["output_dir"]), target, heldout, model_seed)
    worker_base = deepcopy(base); worker_base.update({"output_dir": str(paths["directory"]), "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    if args.device == "auto" and torch.cuda.is_available(): worker_base["device"] = "cuda:0"
    prior, corr, graph_fit = a1.source_correlation_adjacency_train_only(worker_base, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "target_domain": target, "heldout_source_domain": heldout, "active_source_domains": list(active_sources(target, heldout)), "model_seed": model_seed, "a2_1_protocol_hash": protocol["protocol_hash"], "evidence_hashes": evidence["a2_1_input_hashes"], "graph_fit": graph_fit, "official_test_files_accessed": False, "official_test_forward_run": False}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("target_domain", "heldout_source_domain", "active_source_domains", "model_seed", "a2_1_protocol_hash", "evidence_hashes"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"incompatible A10 shard at {key}")
        if previous.get("script_hash") != manifest["script_hash"]:
            if not args.resume: raise RuntimeError("A10 script changed; use --resume only after reviewing the change")
            manifest["resumed_from_script_hash"] = previous.get("script_hash")
    paths["directory"].mkdir(parents=True, exist_ok=True); atomic_json(paths["manifest"], manifest)
    sensors = list(worker_base["sensor_columns"]); a1.atomic_write_text(paths["directory"] / "source_prior_adjacency.csv", pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv()); a1.atomic_write_text(paths["directory"] / "source_prior_correlation.csv", pd.DataFrame(corr, index=sensors, columns=sensors).to_csv())
    state = load_state(paths); expected = len(experiment["representations"]) * len(experiment["target_split_seeds"])
    data, source = {}, {}
    for representation in experiment["representations"]:
        data[representation] = a8.prepare_representation_data(worker_base, representation)
        source[representation] = train_or_load_source(output, worker_base, experiment, target, heldout, representation, model_seed, data[representation], prior)
        _, history, inventory = source[representation]
        state["inventory"] = pd.concat([state["inventory"].loc[state["inventory"].get("representation", pd.Series(dtype=str)) != representation], pd.DataFrame([inventory])], ignore_index=True)
        state["audit"] = pd.concat([state["audit"].loc[state["audit"].get("representation", pd.Series(dtype=str)) != representation], pd.DataFrame([{**data[representation]["audit"], "target_domain": target, "heldout_source_domain": heldout, "model_seed": model_seed, "active_source_domains": json.dumps(active_sources(target, heldout))}])], ignore_index=True)
        h = pd.DataFrame(history)
        if not h.empty:
            h.insert(0, "experiment_id", EXPERIMENT_ID); h.insert(1, "target_domain", target); h.insert(2, "heldout_source_domain", heldout); h.insert(3, "representation", representation); h.insert(4, "model_seed", model_seed)
            state["source_history"] = pd.concat([state["source_history"].loc[~((state["source_history"].get("representation", pd.Series(dtype=str)) == representation) & (state["source_history"].get("model_seed", pd.Series(dtype=int)) == model_seed))], h], ignore_index=True)
    for representation, split in product(experiment["representations"], experiment["target_split_seeds"]):
        key = cell_id(target, heldout, model_seed, int(split), representation)
        if key in state["completed"]: continue
        source_state, source_history, inventory = source[representation]
        endpoint, history = run_cell(worker_base, experiment, protocol, target, heldout, representation, model_seed, int(split), data[representation], deepcopy(source_state), source_history, inventory, prior)
        state["endpoint"] = pd.concat([state["endpoint"], endpoint], ignore_index=True); state["target_history"] = pd.concat([state["target_history"], history], ignore_index=True); state["completed"].add(key); save_state(paths, state, expected)
    save_state(paths, state, expected); print(paths["status"].read_text(encoding="utf-8"))


def select_paired(frame: pd.DataFrame, units: list[int], *, assignment: dict[int, float]) -> pd.DataFrame:
    parts = []
    for representation in (BASE, AGE):
        part = a21.endpoint_subset(frame[frame.representation == representation], units, assignment=assignment)
        parts.append(part[a9.PRED_KEYS + ["representation", "prediction"]])
    wide = pd.concat(parts, ignore_index=True).pivot(index=a9.PRED_KEYS, columns="representation", values="prediction").reset_index()
    if len(wide) != len(units) or BASE not in wide or AGE not in wide: raise RuntimeError("A10 baseline/cycle-age endpoint alignment failed")
    wide = wide.rename(columns={BASE: "prediction_baseline", AGE: "prediction_cycle_age"}); wide["prediction_mean_for_gate"] = (wide.prediction_baseline + wide.prediction_cycle_age) / 2.0; wide["prediction_gate"] = np.where(wide.prediction_mean_for_gate >= GATE_THRESHOLD, "high_pred_rul_ge60", "lower_pred_rul_lt60")
    return wide


def crossfit(endpoints: pd.DataFrame, protocols: dict[str, Any], experiment: dict[str, Any]) -> dict[str, pd.DataFrame]:
    selection_parts, confirmation_parts, grid_parts, parameter_rows, selection_rows, confirmation_rows = [], [], [], [], [], []
    groups = ["target_domain", "heldout_source_domain", "model_seed", "target_split_seed"]
    for values, source in endpoints.groupby(groups):
        domain, heldout, seed, split_seed = values; split = protocols[str(domain)]["role_splits"][str(int(split_seed))]
        for partition in experiment["role_partitions"]:
            roles = split["partitions"][str(partition)]; selection_units, confirmation_units = list(map(int, roles["selection_units"])), list(map(int, roles["confirmation_units"]))
            if set(selection_units) & set(confirmation_units): raise RuntimeError("A10 selection/confirmation engine overlap")
            common = {"target_domain": str(domain), "heldout_source_domain": str(heldout), "model_seed": int(seed), "target_split_seed": int(split_seed), "role_partition": int(partition), "active_source_domains": json.dumps(active_sources(str(domain), str(heldout)))}
            selection_set = []
            for endpoint_seed in experiment["selection_endpoint_seeds"]:
                assign = a21.balanced_assignment(selection_units, str(domain), int(split_seed), int(partition), int(endpoint_seed), "selection")
                selection_set.append(select_paired(source, selection_units, assignment=assign).assign(**common, endpoint_seed=int(endpoint_seed), evaluation_role="selection"))
            selection = pd.concat(selection_set, ignore_index=True); params, grid = a9.choose_alpha(selection, common, experiment); parameter_rows.append(params); grid_parts.append(grid)
            selected_applied = a9.blend(selection, params["alpha_high"], params["alpha_low"]); selection_parts.append(selected_applied)
            for endpoint_seed in experiment["selection_endpoint_seeds"]:
                selection_rows.extend(a9.evaluation_rows(selected_applied[selected_applied.endpoint_seed == endpoint_seed], {**common, "endpoint_seed": int(endpoint_seed), "evaluation_role": "selection", "alpha_high": params["alpha_high"], "alpha_low": params["alpha_low"]}))
            for endpoint_seed in experiment["confirmation_endpoint_seeds"]:
                assign = a21.balanced_assignment(confirmation_units, str(domain), int(split_seed), int(partition), int(endpoint_seed), "confirmation")
                applied = a9.blend(select_paired(source, confirmation_units, assignment=assign), params["alpha_high"], params["alpha_low"]).assign(**common, endpoint_seed=int(endpoint_seed), evaluation_role="confirmation")
                confirmation_parts.append(applied)
                confirmation_rows.extend(a9.evaluation_rows(applied, {**common, "endpoint_seed": int(endpoint_seed), "evaluation_role": "confirmation", "alpha_high": params["alpha_high"], "alpha_low": params["alpha_low"]}))
    return {"selection_prediction": pd.concat(selection_parts, ignore_index=True), "confirmation_prediction": pd.concat(confirmation_parts, ignore_index=True), "selection_run": pd.DataFrame(selection_rows), "confirmation_run": pd.DataFrame(confirmation_rows), "grid": pd.concat(grid_parts, ignore_index=True), "parameters": pd.DataFrame(parameter_rows)}


def pair_results(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot(index=PAIR_KEYS, columns="variant", values=a8.METRICS).reset_index(); pivot.columns = ["_".join(str(v) for v in c if str(v)) if isinstance(c, tuple) else c for c in pivot.columns]
    out = pivot[PAIR_KEYS].copy()
    for metric in a8.METRICS:
        out[f"{metric}_{BASE}"] = pivot[f"{metric}_{BASE}"].astype(float); out[f"{metric}_{BLEND}"] = pivot[f"{metric}_{BLEND}"].astype(float); out[f"{metric}_delta_candidate_minus_baseline"] = out[f"{metric}_{BLEND}"] - out[f"{metric}_{BASE}"]
    out["candidate"] = BLEND; out["nasa_relative_delta"] = out.nasa_score_delta_candidate_minus_baseline / out[f"nasa_score_{BASE}"]; out["rmse_relative_delta"] = out.rmse_delta_candidate_minus_baseline / out[f"rmse_{BASE}"]; out["candidate_nasa_win"] = out.nasa_score_delta_candidate_minus_baseline < 0; out["candidate_rmse_win"] = out.rmse_delta_candidate_minus_baseline < 0
    return out.sort_values(PAIR_KEYS)


def stage_pairs(predictions: pd.DataFrame, high: bool, experiment: dict[str, Any]) -> pd.DataFrame:
    selected = predictions[predictions.label > experiment["high_rul_threshold"]].copy() if high else predictions[predictions.label <= experiment["high_rul_threshold"]].copy(); rows = []
    for values, frame in selected.groupby(PAIR_KEYS):
        baseline, candidate = a9.risk(frame, "prediction_baseline"), a9.risk(frame, "prediction_blend"); row = dict(zip(PAIR_KEYS, values)); row.update({"rul_stage": "high_rul_gt60" if high else "low_or_mid_rul_le60", "rul_threshold": experiment["high_rul_threshold"], "stage_engine_count": int(frame.unit.nunique()), "candidate": BLEND})
        for metric in a8.METRICS: row[f"{metric}_{BASE}"], row[f"{metric}_{BLEND}"], row[f"{metric}_delta_candidate_minus_baseline"] = baseline[metric], candidate[metric], candidate[metric] - baseline[metric]
        row["nasa_relative_delta"], row["rmse_relative_delta"] = a9.rel(candidate, baseline, "nasa_score"), a9.rel(candidate, baseline, "rmse"); row["candidate_nasa_win"], row["candidate_rmse_win"] = candidate["nasa_score"] < baseline["nasa_score"], candidate["rmse"] < baseline["rmse"]; rows.append(row)
    out = pd.DataFrame(rows); expected = len(experiment["domains"]) * 3 * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    if len(out) != expected: raise RuntimeError("incomplete A10 stage pairs")
    return out.sort_values(PAIR_KEYS)


def bootstrap(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> tuple[float, float]:
    levels = [sorted(frame[c].unique()) for c in ["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"]]; lookup = frame.set_index(PAIR_KEYS)[column]; rng = np.random.default_rng(seed); samples = np.empty(repetitions)
    for index in range(repetitions):
        values = []
        for target in rng.choice(levels[0], len(levels[0]), replace=True):
            holds = [h for h in levels[1] if h != target]
            for heldout in rng.choice(holds, len(holds), replace=True):
                for model in rng.choice(levels[2], len(levels[2]), replace=True):
                    for split in rng.choice(levels[3], len(levels[3]), replace=True):
                        role, endpoint = int(rng.choice(levels[4])), int(rng.choice(levels[5])); values.append(float(lookup.loc[(target, heldout, model, split, role, endpoint)]))
        samples[index] = np.mean(values)
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def summary(pairs: pd.DataFrame, experiment: dict[str, Any], label: str, group_columns: list[str] | None = None) -> pd.DataFrame:
    groups = [("ALL", pairs)] if group_columns is None else list(pairs.groupby(group_columns))
    rows = []
    for scope, frame in groups:
        nasa_ci, rmse_ci = bootstrap(frame, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, label, scope, "nasa")), bootstrap(frame, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, label, scope, "rmse"))
        row = {"comparison": label, "scope": scope if isinstance(scope, str) else "|".join(map(str, scope if isinstance(scope, tuple) else (scope,))), "n_records": len(frame), "nasa_score_delta_mean": float(frame.nasa_score_delta_candidate_minus_baseline.mean()), "nasa_improvement_pct": float(-100 * frame.nasa_relative_delta.mean()), "nasa_relative_boot_ci95_low": nasa_ci[0], "nasa_relative_boot_ci95_high": nasa_ci[1], "nasa_win_rate": float(frame.candidate_nasa_win.mean()), "rmse_delta_mean": float(frame.rmse_delta_candidate_minus_baseline.mean()), "rmse_degradation_pct": float(100 * frame.rmse_relative_delta.mean()), "rmse_relative_boot_ci95_low": rmse_ci[0], "rmse_relative_boot_ci95_high": rmse_ci[1], "rmse_win_rate": float(frame.candidate_rmse_win.mean()), "late_error_q95_delta_mean": float(frame.late_error_q95_delta_candidate_minus_baseline.mean()), "under_error_q95_delta_mean": float(frame.under_error_q95_delta_candidate_minus_baseline.mean()), "mean_error_delta_mean": float(frame.mean_error_delta_candidate_minus_baseline.mean())}
        rows.append(row)
    return pd.DataFrame(rows)


def stage_summary(pairs: pd.DataFrame, experiment: dict[str, Any], label: str) -> dict[str, Any]:
    nasa_ci, rmse_ci = bootstrap(pairs, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, label, "nasa")), bootstrap(pairs, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, label, "rmse"))
    return {"stage": label, "n_records": len(pairs), "nasa_improvement_pct": float(-100 * pairs.nasa_relative_delta.mean()), "nasa_relative_ci95": list(nasa_ci), "rmse_degradation_pct": float(100 * pairs.rmse_relative_delta.mean()), "rmse_relative_ci95": list(rmse_ci), "mean_error_delta_mean": float(pairs.mean_error_delta_candidate_minus_baseline.mean()), "nasa_win_rate": float(pairs.candidate_nasa_win.mean()), "rmse_win_rate": float(pairs.candidate_rmse_win.mean())}


def ablation_summary(full: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for (target, heldout), frame in full.groupby(["target_domain", "heldout_source_domain"]):
        h = high[(high.target_domain == target) & (high.heldout_source_domain == heldout)]; l = low[(low.target_domain == target) & (low.heldout_source_domain == heldout)]
        fci = bootstrap(frame, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "full")); h_nasa, h_rmse = bootstrap(h, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "hn")), bootstrap(h, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "hr")); l_nasa, l_rmse = bootstrap(l, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "ln")), bootstrap(l, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "lr"))
        passed = fci[1] < 0 and max(h_nasa[1], h_rmse[1], l_nasa[1], l_rmse[1]) <= MARGIN
        rows.append({"target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "n_pairs": len(frame), "full_rmse_improvement_pct": float(-100 * frame.rmse_relative_delta.mean()), "full_rmse_ci95": json.dumps(fci), "high_nasa_ci95": json.dumps(h_nasa), "high_rmse_ci95": json.dumps(h_rmse), "low_nasa_ci95": json.dumps(l_nasa), "low_rmse_ci95": json.dumps(l_rmse), "robust_condition_passed": passed})
    return pd.DataFrame(rows).sort_values(["target_domain", "heldout_source_domain"])


def worker_command(args: argparse.Namespace, target: str, heldout: str, seed: int, device: str, output: Path) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-domain", target, "--worker-heldout", heldout, "--worker-seed", str(seed), "--output-dir", str(output), "--device", device, "--bootstrap-repetitions", str(args.bootstrap_repetitions)]
    if args.data_dir: command += ["--data-dir", args.data_dir]
    if args.a2_1_output_dir: command += ["--a2-1-output-dir", args.a2_1_output_dir]
    if args.quick: command.append("--quick")
    if args.resume: command.append("--resume")
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}: devices, inventory = [args.device], []
    else:
        devices, inventory = a4.choose_gpus(args)
        if not devices: raise RuntimeError("no idle GPU met A10 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(json.dumps({"scheduler": EXPERIMENT_ID, "tasks": [{"target_domain": t, "heldout_source_domain": h, "seed": s} for t, h, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending, active = list(tasks), {}
    while pending or active:
        for device in [d for d in devices if d not in active]:
            if not pending: break
            target, heldout, seed = pending.pop(0); paths = shard_paths(output, target, heldout, seed); paths["directory"].mkdir(parents=True, exist_ok=True); log = paths["directory"] / "worker_training.log"; handle = log.open("a", encoding="utf-8"); env = os.environ.copy()
            if isinstance(device, int): env["CUDA_VISIBLE_DEVICES"] = str(device); command = worker_command(args, target, heldout, seed, "auto", output)
            else: command = worker_command(args, target, heldout, seed, str(device), output)
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True); active[device] = {"process": process, "target": target, "heldout": heldout, "seed": seed, "handle": handle, "log": log}; print(f"[A10] launched target={target} holdout={heldout} seed={seed} device={device} pid={process.pid}")
        done = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None: continue
            record["handle"].close()
            if code != 0:
                tail = "\n".join(record["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                for other in active.values():
                    if other["process"].poll() is None: other["process"].terminate()
                raise RuntimeError(f"A10 worker failed target={record['target']} holdout={record['heldout']} seed={record['seed']} exit={code}\n{tail}")
            print(f"[A10] completed target={record['target']} holdout={record['heldout']} seed={record['seed']} device={device}"); done.append(device)
        for device in done: del active[device]
        if active and not done: time.sleep(5)


def merge(output: Path, tasks: list[tuple[str, str, int]], experiment: dict[str, Any]) -> dict[str, pd.DataFrame]:
    names = ("endpoint", "target_history", "source_history", "inventory", "audit"); parts: dict[str, list[pd.DataFrame]] = {n: [] for n in names}; expected = len(experiment["representations"]) * len(experiment["target_split_seeds"])
    for target, heldout, seed in tasks:
        paths = shard_paths(output, target, heldout, seed); status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected: raise RuntimeError(f"incomplete A10 shard: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get("official_test_forward_run"): raise RuntimeError("official-test contamination in A10")
        for name in names: parts[name].append(load_csv(paths[name]))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def parent_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    output = Path(base["output_dir"]); output.mkdir(parents=True, exist_ok=True); paths = root_paths(output); protocols, evidence = a4.load_training_only_protocol(base, experiment)
    manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "git_commit": a1.git_commit(PROJECT_ROOT), "base_config": {k: v for k, v in base.items() if k != "device"}, "experiment_config": experiment, "evidence": evidence, "registered_primary_question": QUESTION, "source_ablation": "for each target, omit exactly one of its three available source domains; remaining two sources are freshly pretrained", "official_test_files_accessed": False, "official_test_forward_run": False}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("experiment_config", "evidence", "registered_primary_question"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"incompatible existing A10 output at {key}")
        if previous.get("script_hash") != manifest["script_hash"]:
            if not args.resume: raise RuntimeError("A10 script changed; use --resume only after review")
            manifest["resumed_from_script_hash"] = previous.get("script_hash")
    atomic_json(paths["manifest"], manifest); atomic_json(paths["protocol"], {d: protocols[d] for d in experiment["domains"]}); a1.atomic_write_text(paths["roles"], a21.protocol_rows({d: protocols[d] for d in experiment["domains"]}).to_csv(index=False))
    tasks = [(target, heldout, seed) for target in experiment["domains"] for heldout in source_options(target) for seed in experiment["model_seeds"]]
    expected_cells = len(tasks) * len(experiment["target_split_seeds"]) * len(experiment["representations"]); expected_params = len(tasks) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]); expected_pairs = expected_params * len(experiment["confirmation_endpoint_seeds"])
    dry = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "domains": experiment["domains"], "source_domain_ablation_conditions_per_target": 3, "model_seeds": experiment["model_seeds"], "target_split_seeds": experiment["target_split_seeds"], "selection_endpoint_seeds": experiment["selection_endpoint_seeds"], "confirmation_endpoint_seeds": experiment["confirmation_endpoint_seeds"], "selection_confirmation_endpoint_seeds_disjoint": True, "expected_source_pretrain_runs": len(tasks) * 2, "expected_training_cells": expected_cells, "expected_blend_parameter_sets": expected_params, "expected_confirmation_pairs": expected_pairs, "official_test_files_accessed": False, "official_test_forward_run": False, "gpu_inventory": a2.query_gpus()}; atomic_json(paths["dry"], dry); atomic_json(paths["causality"], {"experiment_id": EXPERIMENT_ID, "candidate_feature": "causal_cycle_age_z", "feature_transform": "log1p(cycle), source-only standardization under each two-source ablation", "uses_unit_max_cycle": False, "uses_true_rul_as_feature": False, "uses_future_windows": False, "selection_labels_used_only_to_choose_alpha": True, "confirmation_used_for_alpha_selection": False, "official_test_files_accessed": False, "official_test_forward_run": False})
    if args.dry_run: print(json.dumps(dry, ensure_ascii=False, indent=2)); return
    shards = output / "shards"
    if shards.exists() and any(shards.iterdir()) and not args.resume: raise RuntimeError("A10 contains interrupted shards; use --resume")
    run_workers(args, tasks, output); merged = merge(output, tasks, experiment); endpoint = merged["endpoint"].sort_values(["target_domain", "heldout_source_domain", "representation", "model_seed", "target_split_seed", "unit", "endpoint_fraction"])
    if endpoint.cell_id.nunique() != expected_cells: raise RuntimeError("A10 endpoint output is incomplete")
    evaluated = crossfit(endpoint, protocols, experiment); confirmation = evaluated["confirmation_run"].sort_values(PAIR_KEYS + ["variant"]); pairs = pair_results(confirmation); high, low = stage_pairs(evaluated["confirmation_prediction"], True, experiment), stage_pairs(evaluated["confirmation_prediction"], False, experiment)
    comparison = pd.concat([summary(pairs, experiment, "full_endpoint_blend_vs_baseline"), summary(high, experiment, "high_rul_blend_vs_baseline"), summary(low, experiment, "low_rul_blend_vs_baseline")], ignore_index=True); ablation = ablation_summary(pairs, high, low, experiment)
    overall = comparison[(comparison.comparison == "full_endpoint_blend_vs_baseline") & (comparison.scope == "ALL")].iloc[0]; high_all, low_all = stage_summary(high, experiment, "high_rul_gt60"), stage_summary(low, experiment, "low_or_mid_rul_le60"); full_ok = overall.nasa_relative_boot_ci95_high < 0 or overall.rmse_relative_boot_ci95_high < 0; high_ok = high_all["nasa_relative_ci95"][1] <= MARGIN and high_all["rmse_relative_ci95"][1] <= MARGIN; low_ok = low_all["nasa_relative_ci95"][1] <= MARGIN and low_all["rmse_relative_ci95"][1] <= MARGIN; conditions = int(ablation.robust_condition_passed.sum()); minimum = int(experiment["minimum_passing_ablation_conditions"]); complete = len(pairs) == expected_pairs and len(evaluated["parameters"]) == expected_params and len(merged["inventory"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "representation"])) == len(tasks) * 2
    decision = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "complete": bool(complete), "quick_mode": bool(experiment["quick_mode"]), "expected_training_cells": expected_cells, "completed_training_cells": int(endpoint.cell_id.nunique()), "expected_blend_parameter_sets": expected_params, "completed_blend_parameter_sets": len(evaluated["parameters"]), "expected_confirmation_pairs": expected_pairs, "completed_confirmation_pairs": len(pairs), "source_ablation_conditions": 12 if not args.quick else 3, "passing_ablation_conditions": conditions, "minimum_passing_ablation_conditions": minimum, "official_test_files_accessed": False, "official_test_forward_run": False, "full_endpoint_result": {"nasa_improvement_pct": float(overall.nasa_improvement_pct), "nasa_relative_ci95": [float(overall.nasa_relative_boot_ci95_low), float(overall.nasa_relative_boot_ci95_high)], "rmse_degradation_pct": float(overall.rmse_degradation_pct), "rmse_relative_ci95": [float(overall.rmse_relative_boot_ci95_low), float(overall.rmse_relative_boot_ci95_high)], "strict_improvement": bool(full_ok)}, "high_rul_safety_result": {**high_all, "noninferiority_passed": bool(high_ok)}, "low_rul_safety_result": {**low_all, "noninferiority_passed": bool(low_ok)}, "passed": bool(complete and full_ok and high_ok and low_ok and conditions >= minimum) if not args.quick else bool(complete), "reason": "A10 confirmed source-domain ablation robustness" if complete and full_ok and high_ok and low_ok and conditions >= minimum else "A10 completed, but efficacy/safety or source-ablation robustness criteria were not all met", "next_action": "prepare_method_ablation_and_final_paper_tables" if complete and full_ok and high_ok and low_ok and conditions >= minimum else "report_source-dependence_boundary_without_tuning_official_test"}
    for name, frame in (("age_audit", merged["audit"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "representation"])), ("inventory", merged["inventory"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "representation"])), ("source_history", merged["source_history"]), ("endpoint", endpoint), ("target_history", merged["target_history"]), ("selection_prediction", evaluated["selection_prediction"]), ("confirmation_prediction", evaluated["confirmation_prediction"]), ("selection_run", evaluated["selection_run"]), ("confirmation_run", confirmation), ("grid", evaluated["grid"]), ("parameters", evaluated["parameters"]), ("paired", pairs), ("high", high), ("low", low), ("summary", comparison), ("ablation", ablation)): a1.atomic_write_text(paths[name], frame.to_csv(index=False))
    atomic_json(paths["decision"], decision); print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args(); base, experiment = load_config(args); validate(base, experiment)
    if args.worker_domain is not None:
        if args.worker_seed is None or args.worker_heldout is None: raise ValueError("worker requires --worker-domain, --worker-heldout, --worker-seed")
        worker_main(args, base, experiment)
    else: parent_main(args, base, experiment)


if __name__ == "__main__": main()
