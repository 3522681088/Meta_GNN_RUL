"""Experiment A12: coverage-aware source weighting under A10 source ablation.

For each A10 target/source-ablation/model-seed/target-split cell, A12 computes
two source weights from *only* the target adaptation engines' input features
and the two active source training domains.  It then uses those fixed weights
to sample source-domain batches during ordinary supervised pretraining.

The reference is the already completed A10 uniform-source A9 blend.  A12 does
not access official test files and uses the same A10 selection/confirmation
engine and endpoint partitions.  Alpha selection remains selection-only.
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

from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402
from scripts import experimentA2_1_endpoint_scheme_crossfit_confirmation as a21  # noqa: E402
from scripts import experimentA4_asymmetric_endpoint_risk_learning as a4  # noqa: E402
from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402
from scripts import experimentA9_crossfitted_cycle_age_safety_blend as a9  # noqa: E402
from scripts import experimentA10_source_domain_ablation_robustness as a10  # noqa: E402
from scripts.experiment8_transfer_baseline import train_source_supervised  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402


SCRIPT_VERSION = "experimentA12_coverage_aware_source_weighting_training_only_v1"
EXPERIMENT_ID = "experimentA12"
DEFAULT_OUTPUT = "outputs/experimentA12_coverage_aware_source_weighting_training_only"
DEFAULT_A10_OUTPUT = a10.DEFAULT_OUTPUT
DOMAINS = a10.DOMAINS
MODEL_SEEDS = a10.MODEL_SEEDS
TARGET_SPLIT_SEEDS = a10.TARGET_SPLIT_SEEDS
ROLE_PARTITIONS = a10.ROLE_PARTITIONS
SELECTION_ENDPOINT_SEEDS = a10.SELECTION_ENDPOINT_SEEDS
CONFIRMATION_ENDPOINT_SEEDS = a10.CONFIRMATION_ENDPOINT_SEEDS
BASE, AGE = a10.BASE, a10.AGE
REFERENCE = "uniform_source_a9_blend"
CANDIDATE = "coverage_aware_source_weighted_a9_blend"
PAIR_KEYS = a10.PAIR_KEYS
PRED_KEYS = a9.PRED_KEYS
MARGIN = 0.03
WEIGHT_TEMPERATURE = 2.0
WEIGHT_LOWER_BOUND = 0.25
WEIGHT_UPPER_BOUND = 0.75
QUESTION = (
    "Does fixed coverage-aware two-source pretraining, fitted only on target "
    "adaptation-engine inputs, improve A10 source-ablation robustness while "
    "preserving endpoint and stage safety against the locked uniform-source A9 reference?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A12 coverage-aware source weighting")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a10-output-dir")
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
        raise FileNotFoundError(f"required input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def source_options(target: str) -> tuple[str, ...]:
    return tuple(domain for domain in DOMAINS if domain != target)


def active_sources(target: str, heldout: str) -> tuple[str, str]:
    sources = tuple(domain for domain in source_options(target) if domain != heldout)
    if len(sources) != 2:
        raise ValueError(f"invalid target/heldout pair: {target}/{heldout}")
    return sources


def cell_id(target: str, heldout: str, seed: int, split: int, representation: str) -> str:
    return f"{EXPERIMENT_ID}_{target.lower()}_drop{heldout.lower()}_{representation}_mseed{seed:03d}_tsplit{split}"


def root_paths(output: Path) -> dict[str, Path]:
    p = EXPERIMENT_ID
    return {
        "manifest": output / f"{p}_manifest.json",
        "protocol": output / f"{p}_protocol.json",
        "roles": output / f"{p}_engine_roles.csv",
        "dry": output / f"{p}_dry_run.json",
        "reference": output / f"{p}_reference_input_integrity.json",
        "causality": output / f"{p}_source_weight_causality_audit.json",
        "weights": output / f"{p}_source_weight_parameters.csv",
        "weight_audit": output / f"{p}_source_weight_audit.csv",
        "inventory": output / f"{p}_source_pretrain_inventory.csv",
        "source_history": output / f"{p}_source_pretrain_history.csv",
        "endpoint": output / f"{p}_pool_endpoint_predictions.csv",
        "target_history": output / f"{p}_target_history.csv",
        "selection_prediction": output / f"{p}_selection_endpoint_predictions.csv",
        "confirmation_prediction": output / f"{p}_confirmation_endpoint_predictions.csv",
        "selection_run": output / f"{p}_selection_run_level.csv",
        "confirmation_run": output / f"{p}_confirmation_run_level.csv",
        "grid": output / f"{p}_blend_selection_grid.csv",
        "blend_parameters": output / f"{p}_blend_parameters.csv",
        "paired": output / f"{p}_paired_weighted_vs_uniform.csv",
        "high": output / f"{p}_high_rul_paired_weighted_vs_uniform.csv",
        "low": output / f"{p}_low_rul_paired_weighted_vs_uniform.csv",
        "summary": output / f"{p}_comparison_summary.csv",
        "ablation": output / f"{p}_source_ablation_summary.csv",
        "decision": output / f"{p}_confirmation_decision.json",
    }


def shard_paths(output: Path, target: str, heldout: str, seed: int) -> dict[str, Path]:
    directory = output / "shards" / f"{target}_drop{heldout}_mseed{seed:03d}"
    return {
        "directory": directory,
        "manifest": directory / "worker_manifest.json",
        "status": directory / "worker_status.json",
        "endpoint": directory / "pool_endpoint_predictions.csv",
        "target_history": directory / "target_history.csv",
        "source_history": directory / "source_pretrain_history.csv",
        "inventory": directory / "source_pretrain_inventory.csv",
        "weights": directory / "source_weight_parameters.csv",
        "audit": directory / "source_weight_audit.csv",
    }


def cache_path(output: Path, target: str, heldout: str, seed: int, split: int, representation: str) -> Path:
    return shard_paths(output, target, heldout, seed)["directory"] / "source_cache" / (
        f"{EXPERIMENT_ID}_{representation}_{target}_drop{heldout}_mseed{seed:03d}_tsplit{split}.pt"
    )


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
        "experiment_name": "coverage_aware_source_weighting_training_only",
        "domains": list(DOMAINS),
        "architecture": "window_no_graph",
        "representations": [BASE, AGE],
        "model_seeds": list(MODEL_SEEDS),
        "target_split_seeds": list(TARGET_SPLIT_SEEDS),
        "role_partitions": list(ROLE_PARTITIONS),
        "selection_endpoint_seeds": list(SELECTION_ENDPOINT_SEEDS),
        "confirmation_endpoint_seeds": list(CONFIRMATION_ENDPOINT_SEEDS),
        "high_rul_threshold": 60.0,
        "k": 5,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "fixed_budget_no_epoch_selection": True,
        "source_weighting": "fixed_adaptation_input_coverage_softmax",
        "weight_temperature": WEIGHT_TEMPERATURE,
        "weight_lower_bound": WEIGHT_LOWER_BOUND,
        "weight_upper_bound": WEIGHT_UPPER_BOUND,
        "alpha_grid": list(a10.ALPHA_GRID),
        "prediction_gate_threshold": a10.GATE_THRESHOLD,
        "selection_safety_margin_pct": 3.0,
        "stage_noninferiority_margin_pct": 3.0,
        "minimum_passing_ablation_conditions": 9,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "a2_1_output_dir": resolved(args.a2_1_output_dir, a4.DEFAULT_A2_1_OUTPUT),
        "a10_output_dir": resolved(args.a10_output_dir, DEFAULT_A10_OUTPUT),
        "output_dir": base["output_dir"],
        "quick_mode": bool(args.quick),
    }
    if args.quick:
        experiment.update({
            "domains": ["FD004"], "model_seeds": [100], "target_split_seeds": [6401],
            "role_partitions": [1], "selection_endpoint_seeds": [9001],
            "confirmation_endpoint_seeds": [9101], "bootstrap_repetitions": 100,
            "minimum_passing_ablation_conditions": 1,
        })
        base.update({"target_epochs": 2, "source_pretrain_steps": 20})
        experiment.update({"target_epochs": 2, "source_pretrain_steps": 20})
        if args.output_dir is None:
            base["output_dir"] = resolved(None, DEFAULT_OUTPUT + "_quick")
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate(base: dict[str, Any], experiment: dict[str, Any]) -> None:
    if set(experiment["selection_endpoint_seeds"]) & set(experiment["confirmation_endpoint_seeds"]):
        raise ValueError("selection/confirmation endpoint seeds must be disjoint")
    for domain in DOMAINS:
        if not a1.train_path(base["data_dir"], domain).is_file():
            raise FileNotFoundError(f"missing training file: {domain}")


def a10_input_paths(root: Path) -> dict[str, Path]:
    p = "experimentA10"
    return {
        "manifest": root / f"{p}_manifest.json",
        "decision": root / f"{p}_confirmation_decision.json",
        "protocol": root / f"{p}_protocol.json",
        "ablation": root / f"{p}_source_ablation_summary.csv",
        "confirmation_prediction": root / f"{p}_confirmation_endpoint_predictions.csv",
        "blend_parameters": root / f"{p}_blend_parameters.csv",
    }


def validate_reference(experiment: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    paths = a10_input_paths(Path(experiment["a10_output_dir"]))
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("A12 requires completed A10 input files:\n" + "\n".join(missing))
    manifest, decision, protocol = read_json(paths["manifest"]), read_json(paths["decision"]), read_json(paths["protocol"])
    if not decision.get("complete") or decision.get("quick_mode"):
        raise RuntimeError("A12 requires the complete formal A10 run")
    for payload in (manifest, decision):
        if payload.get("official_test_files_accessed") or payload.get("official_test_forward_run"):
            raise RuntimeError("A10 reference is contaminated by official-test access")
    ablation = load_csv(paths["ablation"])
    reference_prediction = load_csv(paths["confirmation_prediction"])
    parameters = load_csv(paths["blend_parameters"])
    expected_pairs = len(experiment["domains"]) * 3 * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    observed_pairs = reference_prediction[PAIR_KEYS].drop_duplicates().shape[0]
    expected_parameters = len(experiment["domains"]) * 3 * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"])
    if args_domains := set(experiment["domains"]):
        reference_prediction = reference_prediction[reference_prediction.target_domain.isin(args_domains)].copy()
        ablation = ablation[ablation.target_domain.isin(args_domains)].copy()
        parameters = parameters[parameters.target_domain.isin(args_domains)].copy()
    if not experiment["quick_mode"]:
        if len(ablation) != 12 or observed_pairs != expected_pairs or len(parameters) != expected_parameters:
            raise RuntimeError("A10 reference does not match the locked A12 protocol")
    return {"ablation": ablation, "prediction": reference_prediction, "parameters": parameters, "protocol": protocol}, {
        "a10_output_dir": str(Path(experiment["a10_output_dir"])),
        "a10_manifest_hash": a1.file_sha256(paths["manifest"]),
        "a10_decision_hash": a1.file_sha256(paths["decision"]),
        "a10_protocol_hash": a1.file_sha256(paths["protocol"]),
        "a10_ablation_hash": a1.file_sha256(paths["ablation"]),
        "a10_confirmation_prediction_hash": a1.file_sha256(paths["confirmation_prediction"]),
        "a10_blend_parameter_hash": a1.file_sha256(paths["blend_parameters"]),
        "reference_expected_pairs": expected_pairs,
        "reference_observed_pairs": observed_pairs,
        "reference_official_test_files_accessed": False,
        "reference_official_test_forward_run": False,
    }


def sample_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    rng = np.random.default_rng(seed)
    return frame.iloc[np.sort(rng.choice(len(frame), count, replace=False))].copy()


def nearest_distance_p95(target: np.ndarray, source: np.ndarray) -> float:
    target, source = sample_rows(pd.DataFrame(target), 1200, 1).to_numpy(), sample_rows(pd.DataFrame(source), 1800, 2).to_numpy()
    values: list[np.ndarray] = []
    for start in range(0, len(target), 128):
        current = target[start:start + 128]
        squared = ((current[:, None, :] - source[None, :, :]) ** 2).sum(axis=2)
        values.append(np.sqrt(np.maximum(squared.min(axis=1), 0.0)))
    return float(np.quantile(np.concatenate(values), 0.95))


def source_weights_from_adaptation(
    data: dict[str, Any],
    target_units: list[int],
    target: str,
    heldout: str,
    split_seed: int,
    experiment: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fixed source weights using no target label, confirmation unit or test row."""
    sources = active_sources(target, heldout)
    target_frame = data["target_frame"]
    adaptation = target_frame[target_frame.unit.isin(target_units)].copy()
    if adaptation.unit.nunique() != len(target_units):
        raise RuntimeError("A12 adaptation units are incomplete")
    features = list(data["features"])
    setting_columns = [column for column in features if column.startswith("op_setting")]
    sensor_columns = [column for column in features if column not in setting_columns]
    if not setting_columns or not sensor_columns:
        raise RuntimeError("A12 coverage features are incomplete")
    target_sample = sample_rows(adaptation, 1200, stable_seed(EXPERIMENT_ID, target, heldout, split_seed, "target"))
    rows = []
    for source in sources:
        source_frame = data["source_frames"][source]
        source_sample = sample_rows(source_frame, 1800, stable_seed(EXPERIMENT_ID, target, heldout, split_seed, source, "source"))
        s_setting = source_sample[setting_columns].to_numpy(float)
        t_setting = target_sample[setting_columns].to_numpy(float)
        setting_mean, setting_std = s_setting.mean(axis=0), s_setting.std(axis=0)
        setting_std = np.where(setting_std > 1e-8, setting_std, 1.0)
        setting_p95 = nearest_distance_p95((t_setting - setting_mean) / setting_std, (s_setting - setting_mean) / setting_std)
        s_sensor, t_sensor = source_sample[sensor_columns].to_numpy(float), target_sample[sensor_columns].to_numpy(float)
        sensor_mean = float(np.linalg.norm(t_sensor.mean(axis=0) - s_sensor.mean(axis=0)) / np.sqrt(len(sensor_columns)))
        sensor_cov = float(np.linalg.norm(np.cov(t_sensor, rowvar=False) - np.cov(s_sensor, rowvar=False), ord="fro") / len(sensor_columns))
        source_age, target_age = np.log1p(source_sample.cycle.to_numpy(float)), np.log1p(target_sample.cycle.to_numpy(float))
        age_scale = max(float(source_age.std()), 1e-8)
        age_shift = float(abs(target_age.mean() - source_age.mean()) / age_scale + abs(np.log((target_age.std() + 1e-8) / age_scale)))
        rows.append({"source_domain": source, "setting_nn_p95": setting_p95, "sensor_mean_shift": sensor_mean, "sensor_covariance_shift": sensor_cov, "age_shift": age_shift})
    table = pd.DataFrame(rows)
    components = ["setting_nn_p95", "sensor_mean_shift", "sensor_covariance_shift", "age_shift"]
    relative = np.zeros(len(table), dtype=float)
    for column in components:
        values = table[column].to_numpy(float)
        relative += values / max(float(values.mean()), 1e-8)
    table["coverage_distance"] = relative / len(components)
    raw = np.exp(-float(experiment["weight_temperature"]) * table.coverage_distance.to_numpy(float))
    raw = raw / raw.sum()
    clipped = np.clip(raw, float(experiment["weight_lower_bound"]), float(experiment["weight_upper_bound"]))
    clipped = clipped / clipped.sum()
    table["raw_softmax_weight"] = raw
    table["source_weight"] = clipped
    table["target_domain"], table["heldout_source_domain"], table["target_split_seed"] = target, heldout, int(split_seed)
    table["adaptation_units"] = json.dumps(list(map(int, target_units)))
    table["target_adaptation_engine_count"] = len(target_units)
    table["uses_target_labels"] = False
    table["uses_selection_units"] = False
    table["uses_confirmation_units"] = False
    table["uses_official_test"] = False
    if not np.isclose(table.source_weight.sum(), 1.0) or not bool(table.source_weight.between(float(experiment["weight_lower_bound"]) - 1e-8, float(experiment["weight_upper_bound"]) + 1e-8).all()):
        raise RuntimeError("A12 source weights violate registered bounds")
    return dict(zip(table.source_domain, table.source_weight)), {"rows": table, "components": components}


def weighted_source_train(model: torch.nn.Module, source_tasks: dict[str, Any], weights: dict[str, float], cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    learner = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=cfg["source_pretrain_lr"], weight_decay=cfg["source_pretrain_weight_decay"])
    names = sorted(source_tasks)
    probability = np.asarray([weights[name] for name in names], dtype=float)
    probability /= probability.sum()
    rng = np.random.default_rng(int(cfg["seed"]) + 17001)
    iterators = {name: iter(source_tasks[name]) for name in names}
    counts = {name: 0 for name in names}
    history: list[dict[str, Any]] = []
    report_every = max(1, int(cfg["source_pretrain_steps"]) // 10)
    losses: list[float] = []
    learner.train()
    for step in range(1, int(cfg["source_pretrain_steps"]) + 1):
        name = str(rng.choice(names, p=probability)); counts[name] += 1
        try:
            x, y = next(iterators[name])
        except StopIteration:
            iterators[name] = iter(source_tasks[name]); x, y = next(iterators[name])
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, _ = rul_training_loss(learner, x, y, cfg.get("pair_aux_weight", 0.0))
        loss.backward(); torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0); optimizer.step()
        losses.append(float(loss.item()))
        if step % report_every == 0 or step == int(cfg["source_pretrain_steps"]):
            row = {"source_step": step, "mean_source_loss": float(np.mean(losses)), **{f"sampled_steps_{name}": int(counts[name]) for name in names}}
            history.append(row)
            print(f"A12 weighted_source_step={step:04d}/{cfg['source_pretrain_steps']} mean_loss={row['mean_source_loss']:.4f}")
            losses.clear()
    return learner, history


def weighted_source_signature(base: dict[str, Any], experiment: dict[str, Any], target: str, heldout: str, seed: int, split: int, representation: str, data: dict[str, Any], weights: dict[str, float], prior: torch.Tensor) -> str:
    return a1.canonical_hash({
        "script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "target": target, "heldout": heldout, "model_seed": seed, "target_split_seed": split, "representation": representation, "weights": weights, "features": data["features"], "age_audit": data["audit"], "architecture": experiment["architecture"], "source_pretrain_steps": base["source_pretrain_steps"], "train_file_hashes": {d: a1.file_sha256(a1.train_path(base["data_dir"], d)) for d in active_sources(target, heldout)}, "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
    })


def train_or_load_weighted_source(output: Path, base: dict[str, Any], experiment: dict[str, Any], target: str, heldout: str, seed: int, split: int, representation: str, data: dict[str, Any], weights: dict[str, float], prior: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    signature = weighted_source_signature(base, experiment, target, heldout, seed, split, representation, data, weights, prior)
    path = cache_path(output, target, heldout, seed, split, representation)
    if path.is_file():
        saved = a1.safe_torch_load(path)
        if saved.get("signature") == signature:
            return saved["state"], saved.get("history", []), dict(saved["inventory"])
    cfg = deepcopy(base); cfg.update({"seed": seed, "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    a1.seed_everything(seed)
    model = exp17b.build_model_17b(experiment["architecture"], len(data["features"]), cfg, prior, prior)
    total, predictor = exp17.parameter_count(model)
    tasks = a8.make_source_tasks(data, cfg, experiment)
    model, history = weighted_source_train(model, tasks, weights, cfg, a1.resolve_device(cfg["device"]))
    inventory = {"target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "representation": representation, "model": experiment["architecture"], "model_seed": seed, "target_split_seed": split, "feature_count": len(data["features"]), "feature_columns": json.dumps(data["features"]), "total_parameter_count": total, "predictor_parameter_count": predictor, "source_pretrain_steps": int(cfg["source_pretrain_steps"]), "source_weights": json.dumps(weights, sort_keys=True), "source_signature": signature, "source_cache_origin": "experimentA12_fresh_coverage_weighted_two_source_cache", "source_cache_path": str(path), "source_pretraining_reused_from_A10": False, "uses_target_labels_for_weights": False, "official_test_files_accessed": False, "official_test_forward_run": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "state": a1.state_to_cpu(model), "history": history, "inventory": inventory}, path)
    state = a1.state_to_cpu(model); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return state, history, inventory


def run_cell(base: dict[str, Any], experiment: dict[str, Any], protocol: dict[str, Any], target: str, heldout: str, representation: str, seed: int, split_seed: int, data: dict[str, Any], source_state: dict[str, torch.Tensor], history: list[dict[str, Any]], inventory: dict[str, Any], prior: torch.Tensor) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_seed = a2.target_run_seed(target, seed, split_seed)
    cfg = deepcopy(base); cfg.update({"seed": run_seed, "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    split = protocol["role_splits"][str(split_seed)]
    support, pool = a8.prepare_support_pool(data, cfg, experiment, list(map(int, split["adaptation_units"])), list(map(int, split["evaluation_pool_units"])))
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(experiment["architecture"], len(data["features"]), cfg, prior, prior); model.load_state_dict(source_state)
    predictions, history_frame = a4.train_fixed_budget(model, support, pool, cfg, a1.resolve_device(cfg["device"]), "symmetric_mse", 1.0)
    endpoint = a21.endpoint_epoch_rows(predictions, int(experiment["target_epochs"]))
    common = {"experiment_id": EXPERIMENT_ID, "cell_id": cell_id(target, heldout, seed, split_seed, representation), "target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "model": experiment["architecture"], "representation": representation, "model_seed": seed, "target_split_seed": split_seed, "target_run_seed": run_seed, "k": experiment["k"], "adaptation_units": json.dumps(list(map(int, split["adaptation_units"]))), "a2_1_protocol_hash": protocol["protocol_hash"], "feature_count": len(data["features"]), "source_weights": inventory["source_weights"], "source_signature": inventory["source_signature"], "source_cache_origin": inventory["source_cache_origin"], "source_history_rows": len(history), "official_test_files_accessed": False, "official_test_forward_run": False}
    for column, value in reversed(list(common.items())): endpoint.insert(0, column, value)
    for column, value in reversed(list({"experiment_id": EXPERIMENT_ID, "cell_id": common["cell_id"], "target_domain": target, "heldout_source_domain": heldout, "representation": representation, "model_seed": seed, "target_split_seed": split_seed}.items())): history_frame.insert(0, column, value)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return endpoint, history_frame


def load_state(paths: dict[str, Path]) -> dict[str, Any]:
    done = set(read_json(paths["status"]).get("completed_cell_ids", [])) if paths["status"].is_file() else set()
    state = {"completed": done, "endpoint": load_csv(paths["endpoint"]), "target_history": load_csv(paths["target_history"]), "source_history": load_csv(paths["source_history"]), "inventory": load_csv(paths["inventory"]), "weights": load_csv(paths["weights"]), "audit": load_csv(paths["audit"])}
    for name in ("endpoint", "target_history"):
        if not state[name].empty: state[name] = state[name][state[name].cell_id.isin(done)]
    return state


def save_state(paths: dict[str, Path], state: dict[str, Any], expected: int) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in ("endpoint", "target_history", "source_history", "inventory", "weights", "audit"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(paths["status"], {"completed_cell_ids": sorted(state["completed"]), "completed_training_cells": len(state["completed"]), "expected_training_cells": expected, "endpoint_rows": len(state["endpoint"]), "source_weight_rows": len(state["weights"]), "complete": len(state["completed"]) == expected, "official_test_files_accessed": False, "official_test_forward_run": False})


def worker_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    target, heldout, seed = str(args.worker_domain), str(args.worker_heldout), int(args.worker_seed)
    if target not in experiment["domains"] or heldout not in source_options(target) or seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A12 worker")
    protocols, evidence = a4.load_training_only_protocol(base, experiment); protocol = protocols[target]
    output, paths = Path(base["output_dir"]), shard_paths(Path(base["output_dir"]), target, heldout, seed)
    worker_base = deepcopy(base); worker_base.update({"output_dir": str(paths["directory"]), "target_domain": target, "source_domains": list(active_sources(target, heldout))})
    if args.device == "auto" and torch.cuda.is_available(): worker_base["device"] = "cuda:0"
    prior, corr, graph_fit = a1.source_correlation_adjacency_train_only(worker_base, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "target_domain": target, "heldout_source_domain": heldout, "active_source_domains": list(active_sources(target, heldout)), "model_seed": seed, "a2_1_protocol_hash": protocol["protocol_hash"], "evidence_hashes": evidence["a2_1_input_hashes"], "graph_fit": graph_fit, "official_test_files_accessed": False, "official_test_forward_run": False}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("target_domain", "heldout_source_domain", "active_source_domains", "model_seed", "a2_1_protocol_hash", "evidence_hashes"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"incompatible A12 shard at {key}")
        if previous.get("script_hash") != manifest["script_hash"] and not args.resume: raise RuntimeError("A12 script changed; use --resume only after review")
    paths["directory"].mkdir(parents=True, exist_ok=True); atomic_json(paths["manifest"], manifest)
    sensors = list(worker_base["sensor_columns"]); a1.atomic_write_text(paths["directory"] / "source_prior_adjacency.csv", pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv()); a1.atomic_write_text(paths["directory"] / "source_prior_correlation.csv", pd.DataFrame(corr, index=sensors, columns=sensors).to_csv())
    state = load_state(paths); expected = len(experiment["representations"]) * len(experiment["target_split_seeds"])
    data = {representation: a8.prepare_representation_data(worker_base, representation) for representation in experiment["representations"]}
    for split_seed in experiment["target_split_seeds"]:
        adaptation_units = list(map(int, protocol["role_splits"][str(split_seed)]["adaptation_units"]))
        weights, detail = source_weights_from_adaptation(data[BASE], adaptation_units, target, heldout, int(split_seed), experiment)
        weight_rows = detail["rows"].copy(); weight_rows["model_seed"] = seed
        state["weights"] = pd.concat([state["weights"].loc[~((state["weights"].get("target_split_seed", pd.Series(dtype=int)) == int(split_seed)) & (state["weights"].get("model_seed", pd.Series(dtype=int)) == seed))], weight_rows], ignore_index=True)
        audit_row = {"target_domain": target, "heldout_source_domain": heldout, "model_seed": seed, "target_split_seed": int(split_seed), "active_source_domains": json.dumps(active_sources(target, heldout)), "weight_formula": "softmax(-2*relative_coverage_distance), clipped_to_[0.25,0.75]", "uses_target_adaptation_inputs": True, "uses_target_labels": False, "uses_selection_units": False, "uses_confirmation_units": False, "uses_official_test": False, "weight_sum": float(sum(weights.values()))}
        state["audit"] = pd.concat([state["audit"].loc[~((state["audit"].get("target_split_seed", pd.Series(dtype=int)) == int(split_seed)) & (state["audit"].get("model_seed", pd.Series(dtype=int)) == seed))], pd.DataFrame([audit_row])], ignore_index=True)
        for representation in experiment["representations"]:
            key = cell_id(target, heldout, seed, int(split_seed), representation)
            if key in state["completed"]: continue
            source_state, source_history, inventory = train_or_load_weighted_source(output, worker_base, experiment, target, heldout, seed, int(split_seed), representation, data[representation], weights, prior)
            state["inventory"] = pd.concat([state["inventory"].loc[~((state["inventory"].get("target_split_seed", pd.Series(dtype=int)) == int(split_seed)) & (state["inventory"].get("representation", pd.Series(dtype=str)) == representation))], pd.DataFrame([inventory])], ignore_index=True)
            history = pd.DataFrame(source_history)
            if not history.empty:
                for column, value in reversed(list({"experiment_id": EXPERIMENT_ID, "target_domain": target, "heldout_source_domain": heldout, "representation": representation, "model_seed": seed, "target_split_seed": int(split_seed), "source_weights": json.dumps(weights, sort_keys=True)}.items())): history.insert(0, column, value)
                state["source_history"] = pd.concat([state["source_history"], history], ignore_index=True)
            endpoint, target_history = run_cell(worker_base, experiment, protocol, target, heldout, representation, seed, int(split_seed), data[representation], deepcopy(source_state), source_history, inventory, prior)
            state["endpoint"] = pd.concat([state["endpoint"], endpoint], ignore_index=True); state["target_history"] = pd.concat([state["target_history"], target_history], ignore_index=True); state["completed"].add(key); save_state(paths, state, expected)
    save_state(paths, state, expected); print(paths["status"].read_text(encoding="utf-8"))


def worker_command(args: argparse.Namespace, target: str, heldout: str, seed: int, device: str, output: Path) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-domain", target, "--worker-heldout", heldout, "--worker-seed", str(seed), "--output-dir", str(output), "--device", device, "--bootstrap-repetitions", str(args.bootstrap_repetitions)]
    if args.data_dir: command += ["--data-dir", args.data_dir]
    if args.a2_1_output_dir: command += ["--a2-1-output-dir", args.a2_1_output_dir]
    if args.a10_output_dir: command += ["--a10-output-dir", args.a10_output_dir]
    if args.quick: command.append("--quick")
    if args.resume: command.append("--resume")
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}: devices, inventory = [args.device], []
    else:
        devices, inventory = a4.choose_gpus(args)
        if not devices: raise RuntimeError("no idle GPU met A12 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    if args.max_workers and args.max_workers > 0: devices = devices[:args.max_workers]
    print(json.dumps({"scheduler": EXPERIMENT_ID, "tasks": [{"target_domain": t, "heldout_source_domain": h, "seed": s} for t, h, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending, active = list(tasks), {}
    while pending or active:
        for device in [item for item in devices if item not in active]:
            if not pending: break
            target, heldout, seed = pending.pop(0); paths = shard_paths(output, target, heldout, seed); paths["directory"].mkdir(parents=True, exist_ok=True); log = paths["directory"] / "worker_training.log"; handle = log.open("a", encoding="utf-8"); env = os.environ.copy()
            if isinstance(device, int): env["CUDA_VISIBLE_DEVICES"] = str(device); command = worker_command(args, target, heldout, seed, "auto", output)
            else: command = worker_command(args, target, heldout, seed, str(device), output)
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True); active[device] = {"process": process, "target": target, "heldout": heldout, "seed": seed, "handle": handle, "log": log}; print(f"[A12] launched target={target} holdout={heldout} seed={seed} device={device} pid={process.pid}")
        completed = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None: continue
            record["handle"].close()
            if code != 0:
                tail = "\n".join(record["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                for other in active.values():
                    if other["process"].poll() is None: other["process"].terminate()
                raise RuntimeError(f"A12 worker failed target={record['target']} holdout={record['heldout']} seed={record['seed']} exit={code}\n{tail}")
            print(f"[A12] completed target={record['target']} holdout={record['heldout']} seed={record['seed']} device={device}"); completed.append(device)
        for device in completed: del active[device]
        if active and not completed: time.sleep(5)


def merge(output: Path, tasks: list[tuple[str, str, int]], experiment: dict[str, Any]) -> dict[str, pd.DataFrame]:
    names = ("endpoint", "target_history", "source_history", "inventory", "weights", "audit")
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    expected = len(experiment["representations"]) * len(experiment["target_split_seeds"])
    for target, heldout, seed in tasks:
        paths = shard_paths(output, target, heldout, seed); status = read_json(paths["status"])
        if not status.get("complete") or status.get("completed_training_cells") != expected: raise RuntimeError(f"incomplete A12 shard: {paths['status']}")
        if status.get("official_test_files_accessed") or status.get("official_test_forward_run"): raise RuntimeError("official-test contamination in A12")
        for name in names: parts[name].append(load_csv(paths[name]))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def strategy_pairs(reference: pd.DataFrame, candidate: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    keys = PAIR_KEYS + PRED_KEYS
    left = reference[keys + ["prediction_blend"]].rename(columns={"prediction_blend": "prediction_uniform_source"})
    right = candidate[keys + ["prediction_blend"]].rename(columns={"prediction_blend": "prediction_coverage_weighted"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right): raise RuntimeError("A12/A10 endpoint prediction alignment failed")
    rows = []
    for values, frame in merged.groupby(PAIR_KEYS, sort=True):
        baseline, weighted = a9.risk(frame, "prediction_uniform_source"), a9.risk(frame, "prediction_coverage_weighted")
        row = dict(zip(PAIR_KEYS, values)); row.update({"candidate": CANDIDATE, "reference": REFERENCE})
        for metric in a8.METRICS:
            row[f"{metric}_{REFERENCE}"] = baseline[metric]; row[f"{metric}_{CANDIDATE}"] = weighted[metric]; row[f"{metric}_delta_candidate_minus_baseline"] = weighted[metric] - baseline[metric]
        row["nasa_relative_delta"] = a9.rel(weighted, baseline, "nasa_score"); row["rmse_relative_delta"] = a9.rel(weighted, baseline, "rmse")
        row["candidate_nasa_win"] = weighted["nasa_score"] < baseline["nasa_score"]; row["candidate_rmse_win"] = weighted["rmse"] < baseline["rmse"]
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS)
    expected = len(experiment["domains"]) * 3 * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    if len(output) != expected: raise RuntimeError(f"A12 strategy pairs incomplete: {len(output)} != {expected}")
    return output


def stage_strategy_pairs(reference: pd.DataFrame, candidate: pd.DataFrame, high: bool, experiment: dict[str, Any]) -> pd.DataFrame:
    keys = PAIR_KEYS + PRED_KEYS
    left = reference[keys + ["prediction_blend"]].rename(columns={"prediction_blend": "prediction_uniform_source"})
    right = candidate[keys + ["prediction_blend"]].rename(columns={"prediction_blend": "prediction_coverage_weighted"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    selected = merged[merged.label > float(experiment["high_rul_threshold"])].copy() if high else merged[merged.label <= float(experiment["high_rul_threshold"])].copy()
    rows = []
    for values, frame in selected.groupby(PAIR_KEYS, sort=True):
        baseline, weighted = a9.risk(frame, "prediction_uniform_source"), a9.risk(frame, "prediction_coverage_weighted")
        row = dict(zip(PAIR_KEYS, values)); row.update({"rul_stage": "high_rul_gt60" if high else "low_or_mid_rul_le60", "rul_threshold": float(experiment["high_rul_threshold"]), "stage_engine_count": int(frame.unit.nunique()), "candidate": CANDIDATE, "reference": REFERENCE})
        for metric in a8.METRICS:
            row[f"{metric}_{REFERENCE}"] = baseline[metric]; row[f"{metric}_{CANDIDATE}"] = weighted[metric]; row[f"{metric}_delta_candidate_minus_baseline"] = weighted[metric] - baseline[metric]
        row["nasa_relative_delta"] = a9.rel(weighted, baseline, "nasa_score"); row["rmse_relative_delta"] = a9.rel(weighted, baseline, "rmse")
        row["candidate_nasa_win"] = weighted["nasa_score"] < baseline["nasa_score"]; row["candidate_rmse_win"] = weighted["rmse"] < baseline["rmse"]
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(PAIR_KEYS)
    expected = len(experiment["domains"]) * 3 * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    if len(output) != expected: raise RuntimeError("incomplete A12 true-stage confirmation pairs")
    return output


def ablation_summary(full: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, experiment: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for (target, heldout), frame in full.groupby(["target_domain", "heldout_source_domain"]):
        h = high[(high.target_domain == target) & (high.heldout_source_domain == heldout)]; l = low[(low.target_domain == target) & (low.heldout_source_domain == heldout)]
        fci = a10.bootstrap(frame, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "full"))
        hn = a10.bootstrap(h, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "hn")); hr = a10.bootstrap(h, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "hr")); ln = a10.bootstrap(l, "nasa_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "ln")); lr = a10.bootstrap(l, "rmse_relative_delta", experiment["bootstrap_repetitions"], stable_seed(EXPERIMENT_ID, target, heldout, "lr"))
        passed = fci[1] < 0 and max(hn[1], hr[1], ln[1], lr[1]) <= MARGIN
        rows.append({"target_domain": target, "heldout_source_domain": heldout, "active_source_domains": json.dumps(active_sources(target, heldout)), "n_pairs": len(frame), "full_rmse_improvement_pct": float(-100 * frame.rmse_relative_delta.mean()), "full_rmse_ci95": json.dumps(fci), "high_nasa_ci95": json.dumps(hn), "high_rmse_ci95": json.dumps(hr), "low_nasa_ci95": json.dumps(ln), "low_rmse_ci95": json.dumps(lr), "robust_condition_passed": passed})
    return pd.DataFrame(rows).sort_values(["target_domain", "heldout_source_domain"])


def parent_main(args: argparse.Namespace, base: dict[str, Any], experiment: dict[str, Any]) -> None:
    output = Path(base["output_dir"]); output.mkdir(parents=True, exist_ok=True); paths = root_paths(output)
    reference, reference_integrity = validate_reference(experiment)
    protocols, evidence = a4.load_training_only_protocol(base, experiment)
    manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "git_commit": a1.git_commit(PROJECT_ROOT), "base_config": {k: v for k, v in base.items() if k != "device"}, "experiment_config": experiment, "evidence": evidence, "a10_reference_hashes": reference_integrity, "registered_primary_question": QUESTION, "official_test_files_accessed": False, "official_test_forward_run": False}
    if paths["manifest"].is_file():
        previous = read_json(paths["manifest"])
        for key in ("experiment_config", "evidence", "a10_reference_hashes", "registered_primary_question"):
            if previous.get(key) != manifest.get(key): raise RuntimeError(f"incompatible existing A12 output at {key}")
        if previous.get("script_hash") != manifest["script_hash"] and not args.resume: raise RuntimeError("A12 script changed; use --resume only after review")
        if previous.get("script_hash") != manifest["script_hash"]: manifest["resumed_from_script_hash"] = previous.get("script_hash")
    atomic_json(paths["manifest"], manifest); atomic_json(paths["protocol"], {d: protocols[d] for d in experiment["domains"]}); a1.atomic_write_text(paths["roles"], a21.protocol_rows({d: protocols[d] for d in experiment["domains"]}).to_csv(index=False)); atomic_json(paths["reference"], reference_integrity)
    tasks = [(target, heldout, seed) for target in experiment["domains"] for heldout in source_options(target) for seed in experiment["model_seeds"]]
    expected_cells = len(tasks) * len(experiment["target_split_seeds"]) * len(experiment["representations"]); expected_weights = len(tasks) * len(experiment["target_split_seeds"]) * 2; expected_pairs = len(tasks) * len(experiment["target_split_seeds"]) * len(experiment["role_partitions"]) * len(experiment["confirmation_endpoint_seeds"])
    dry = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "tasks": len(tasks), "expected_training_cells": expected_cells, "expected_source_weight_rows": expected_weights, "expected_confirmation_pairs": expected_pairs, "reference": REFERENCE, "candidate": CANDIDATE, "source_weight_formula": "softmax(-2*relative_coverage_distance), clipped_to_[0.25,0.75]", "weights_use_target_adaptation_inputs_only": True, "selection_confirmation_endpoint_seeds_disjoint": True, "official_test_files_accessed": False, "official_test_forward_run": False, "gpu_inventory": a2.query_gpus()}; atomic_json(paths["dry"], dry); atomic_json(paths["causality"], {"experiment_id": EXPERIMENT_ID, "weights_fit_on": "active source training rows plus target adaptation-engine input features", "uses_target_labels": False, "uses_selection_units": False, "uses_confirmation_units": False, "uses_official_test": False, "uses_future_target_windows": False, "selection_labels_used_only_to_choose_alpha": True, "confirmation_used_for_alpha_selection": False, "official_test_files_accessed": False, "official_test_forward_run": False})
    if args.dry_run: print(json.dumps(dry, ensure_ascii=False, indent=2)); return
    shards = output / "shards"
    if shards.exists() and any(shards.iterdir()) and not args.resume: raise RuntimeError("A12 contains interrupted shards; use --resume")
    run_workers(args, tasks, output)
    merged = merge(output, tasks, experiment)
    endpoint = merged["endpoint"].sort_values(["target_domain", "heldout_source_domain", "representation", "model_seed", "target_split_seed", "unit", "endpoint_fraction"])
    if endpoint.cell_id.nunique() != expected_cells: raise RuntimeError("A12 endpoint output is incomplete")
    evaluated = a10.crossfit(endpoint, protocols, experiment)
    candidate_prediction = evaluated["confirmation_prediction"].sort_values(PAIR_KEYS + PRED_KEYS)
    reference_prediction = reference["prediction"].sort_values(PAIR_KEYS + PRED_KEYS)
    pairs = strategy_pairs(reference_prediction, candidate_prediction, experiment); high = stage_strategy_pairs(reference_prediction, candidate_prediction, True, experiment); low = stage_strategy_pairs(reference_prediction, candidate_prediction, False, experiment)
    comparison = pd.concat([a10.summary(pairs, experiment, "full_endpoint_weighted_vs_uniform"), a10.summary(high, experiment, "high_rul_weighted_vs_uniform"), a10.summary(low, experiment, "low_rul_weighted_vs_uniform")], ignore_index=True)
    ablation = ablation_summary(pairs, high, low, experiment)
    overall = comparison.query("comparison == 'full_endpoint_weighted_vs_uniform' and scope == 'ALL'").iloc[0]
    high_all, low_all = a10.stage_summary(high, experiment, "high_rul_gt60"), a10.stage_summary(low, experiment, "low_or_mid_rul_le60")
    full_ok = float(overall.rmse_relative_boot_ci95_high) < 0; high_ok = high_all["nasa_relative_ci95"][1] <= MARGIN and high_all["rmse_relative_ci95"][1] <= MARGIN; low_ok = low_all["nasa_relative_ci95"][1] <= MARGIN and low_all["rmse_relative_ci95"][1] <= MARGIN; conditions = int(ablation.robust_condition_passed.sum()); minimum = int(experiment["minimum_passing_ablation_conditions"]); complete = len(pairs) == expected_pairs and len(merged["weights"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "source_domain"])) == expected_weights
    decision = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "complete": bool(complete), "quick_mode": bool(experiment["quick_mode"]), "reference": REFERENCE, "candidate": CANDIDATE, "expected_training_cells": expected_cells, "completed_training_cells": int(endpoint.cell_id.nunique()), "expected_source_weight_rows": expected_weights, "completed_source_weight_rows": int(len(merged["weights"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "source_domain"]))), "expected_confirmation_pairs": expected_pairs, "completed_confirmation_pairs": len(pairs), "source_ablation_conditions": 12 if not args.quick else 3, "passing_ablation_conditions": conditions, "minimum_passing_ablation_conditions": minimum, "full_endpoint_result": {"nasa_improvement_pct": float(overall.nasa_improvement_pct), "nasa_relative_ci95": [float(overall.nasa_relative_boot_ci95_low), float(overall.nasa_relative_boot_ci95_high)], "rmse_improvement_pct": float(-overall.rmse_degradation_pct), "rmse_relative_ci95": [float(overall.rmse_relative_boot_ci95_low), float(overall.rmse_relative_boot_ci95_high)], "strict_rmse_improvement": bool(full_ok)}, "high_rul_safety_result": {**high_all, "noninferiority_passed": bool(high_ok)}, "low_rul_safety_result": {**low_all, "noninferiority_passed": bool(low_ok)}, "passed": bool(complete and full_ok and high_ok and low_ok and conditions >= minimum) if not args.quick else bool(complete), "reason": "A12 confirmed coverage-aware source weighting under the registered source-ablation protocol" if complete and full_ok and high_ok and low_ok and conditions >= minimum else "A12 completed, but coverage-aware source weighting did not meet every registered efficacy/safety/robustness criterion", "next_action": "run_experimentA12_1_independent_seed_confirmation" if complete and full_ok and high_ok and low_ok and conditions >= minimum else "report_A12_result_without_official_test_tuning", "official_test_files_accessed": False, "official_test_forward_run": False}
    for name, frame in (("weights", merged["weights"].sort_values(["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "source_domain"])), ("weight_audit", merged["audit"].sort_values(["target_domain", "heldout_source_domain", "model_seed", "target_split_seed"])), ("inventory", merged["inventory"].drop_duplicates(["target_domain", "heldout_source_domain", "model_seed", "target_split_seed", "representation"])), ("source_history", merged["source_history"]), ("endpoint", endpoint), ("target_history", merged["target_history"]), ("selection_prediction", evaluated["selection_prediction"]), ("confirmation_prediction", candidate_prediction), ("selection_run", evaluated["selection_run"]), ("confirmation_run", evaluated["confirmation_run"]), ("grid", evaluated["grid"]), ("blend_parameters", evaluated["parameters"]), ("paired", pairs), ("high", high), ("low", low), ("summary", comparison), ("ablation", ablation)): a1.atomic_write_text(paths[name], frame.to_csv(index=False))
    atomic_json(paths["decision"], decision); print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args(); base, experiment = load_config(args); validate(base, experiment)
    if args.worker_domain is not None:
        if args.worker_seed is None or args.worker_heldout is None: raise ValueError("worker requires --worker-domain, --worker-heldout, --worker-seed")
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
