"""Experiment A2_1: endpoint-scheme and engine-role robustness confirmation.

A2 showed that complete-trajectory model selection and deployment-like
endpoint selection frequently choose different models.  A2_1 tests whether
that result survives balanced endpoint assignment and repeated, engine-
disjoint selection/confirmation roles.  It is deliberately restricted to the
two credible controls from A2: ``window_no_graph`` and ``window_graph``.

Formal registered design
------------------------
* leave-one-domain-out validation on FD001--FD004;
* K=5, model seeds 80--84, target split seeds 6401--6405;
* 200 target-training cells (4 domains x 2 models x 5 x 5);
* five repeated role partitions per target split;
* five balanced endpoint assignments per role partition;
* endpoint fractions 0.55, 0.70, 0.85 and 0.95;
* source-only normalization and source-only graph construction;
* no official test trajectory or official RUL-label access.

The script automatically discovers idle GPUs, runs resumable worker shards,
and merges every artifact below one output directory.  When compatible A2
source caches are present they are verified and reused; otherwise the source
models are trained locally.
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
from preprocess.rul_generator import add_train_rul  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts import experimentA1_2_seed_ensemble_stability as a12  # noqa: E402
from scripts import experimentA1_protocol_refactor_regression as a1  # noqa: E402
from scripts import experimentA2_endpoint_consistency_validation as a2  # noqa: E402


SCRIPT_VERSION = "experimentA2_1_endpoint_scheme_crossfit_confirmation_v1"
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
MODELS = ("window_no_graph", "window_graph")
MODEL_SEEDS = list(range(80, 85))
TARGET_SPLIT_SEEDS = list(range(6401, 6406))
ROLE_PARTITIONS = list(range(1, 6))
ENDPOINT_SEEDS = list(range(7501, 7506))
ENDPOINT_FRACTIONS = (0.55, 0.70, 0.85, 0.95)
SELECTION_PROTOCOLS = ("full_trajectory_selection", "balanced_endpoint_selection")
EVALUATION_PROTOCOLS = (
    "full_trajectory",
    "balanced_endpoint",
    "fixed_endpoint_055",
    "fixed_endpoint_070",
    "fixed_endpoint_085",
    "fixed_endpoint_095",
)
DEFAULT_OUTPUT = "outputs/experimentA2_1_endpoint_scheme_crossfit_confirmation"
DEFAULT_A2_OUTPUT = "outputs/experimentA2_endpoint_consistency_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A2_1 endpoint robustness")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--a2-output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 3,4,5")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker-domain", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolved(value: str | None, fallback: str) -> str:
    return str(a1.resolve_path(fallback if value is None else value))


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
        "experiment_id": "experimentA2_1",
        "experiment_name": "endpoint_scheme_crossfit_confirmation",
        "domains": list(DOMAINS),
        "models": list(MODELS),
        "model_seeds": MODEL_SEEDS.copy(),
        "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
        "role_partitions": ROLE_PARTITIONS.copy(),
        "endpoint_seeds": ENDPOINT_SEEDS.copy(),
        "endpoint_fractions": list(ENDPOINT_FRACTIONS),
        "k": 5,
        "evaluation_pool_count": 64,
        "selection_count": 20,
        "confirmation_count": 30,
        "selection_seed_base": 7200,
        "confirmation_seed_base": 7300,
        "endpoint_seed_base": 7400,
        "pool_seed_base": 7600,
        "role_seed_base": 7700,
        "preprocessing": "condition_settings",
        "balance_mode": "engine_stage",
        "sensor_graph_k": 4,
        "source_pretrain_steps": 1500,
        "target_epochs": 10,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "minimum_rank_flip_rate": 0.20,
        "minimum_affected_domains": 3,
        "minimum_positive_endpoint_seeds": 4,
        "minimum_positive_role_partitions": 4,
        "minimum_graph_improvement_pct": 3.0,
        "minimum_graph_domain_wins": 3,
        "a2_output_dir": resolved(args.a2_output_dir, DEFAULT_A2_OUTPUT),
        "output_dir": base["output_dir"],
        "quick_mode": False,
    }
    if args.quick:
        experiment.update(
            {
                "domains": ["FD004"],
                "model_seeds": [80],
                "target_split_seeds": [6401],
                "role_partitions": [1],
                "endpoint_seeds": [7501],
                "source_pretrain_steps": 5,
                "target_epochs": 2,
                "bootstrap_repetitions": 100,
                "quick_mode": True,
            }
        )
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 2
        if args.output_dir is None:
            base["output_dir"] = resolved(None, DEFAULT_OUTPUT + "_quick")
            experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_config(base: dict, experiment: dict) -> None:
    if tuple(experiment["models"]) != MODELS:
        raise ValueError(f"A2_1 requires models={MODELS}")
    for name in ("model_seeds", "target_split_seeds", "role_partitions", "endpoint_seeds"):
        values = experiment[name]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate values in {name}")
    if experiment["selection_count"] + experiment["confirmation_count"] > experiment["evaluation_pool_count"]:
        raise ValueError("evaluation pool is too small for disjoint roles")
    for domain in DOMAINS:
        path = a1.train_path(base["data_dir"], domain)
        if not path.is_file():
            raise FileNotFoundError(f"missing training file: {path}")


def atomic_json(path: Path, payload: Any) -> None:
    a1.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def stable_seed(*parts: Any) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def build_protocol(base: dict, experiment: dict, domain: str) -> dict:
    a2_protocol = a2.build_domain_protocol(base, experiment, domain)
    eligible = np.asarray(a2_protocol["selection_units"] + a2_protocol["confirmation_units"] + [
        unit
        for unit in sorted(a1.load_train_domain(base["data_dir"], domain)["unit"].unique())
        if unit not in set(a2_protocol["historically_excluded_units"])
        and unit not in set(a2_protocol["selection_units"])
        and unit not in set(a2_protocol["confirmation_units"])
    ], dtype=int)
    eligible = np.asarray(sorted(set(map(int, eligible))), dtype=int)
    role_splits: dict[str, dict[str, dict]] = {}
    for split_seed in experiment["target_split_seeds"]:
        support = list(map(int, a2_protocol["adaptation_units_by_target_split_seed"][str(split_seed)]))
        remaining = np.asarray([u for u in eligible if int(u) not in set(support)], dtype=int)
        pool_count = int(experiment["evaluation_pool_count"])
        if len(remaining) < pool_count:
            raise ValueError(f"{domain} split {split_seed} has fewer than {pool_count} evaluation engines")
        pool_rng = np.random.default_rng(stable_seed("A2_1_pool", domain, split_seed, experiment["pool_seed_base"]))
        pool = pool_rng.permutation(remaining)[:pool_count]
        role_rng = np.random.default_rng(stable_seed("A2_1_role", domain, split_seed, experiment["role_seed_base"]))
        role_order = role_rng.permutation(pool)
        partitions: dict[str, dict] = {}
        for partition_index, partition in enumerate(experiment["role_partitions"]):
            order = np.roll(role_order, -13 * partition_index)
            n_selection = int(experiment["selection_count"])
            n_confirmation = int(experiment["confirmation_count"])
            partitions[str(partition)] = {
                "selection_units": list(map(int, order[:n_selection])),
                "confirmation_units": list(map(int, order[n_selection:n_selection + n_confirmation])),
            }
        role_splits[str(split_seed)] = {
            "adaptation_units": support,
            "evaluation_pool_units": list(map(int, pool)),
            "partitions": partitions,
        }
    protocol = {
        "protocol_version": SCRIPT_VERSION,
        "target_domain": domain,
        "source_domains": [value for value in DOMAINS if value != domain],
        "historically_excluded_units": a2_protocol["historically_excluded_units"],
        "eligible_engine_count": int(len(eligible)),
        "k": int(experiment["k"]),
        "models": list(experiment["models"]),
        "model_seeds": list(experiment["model_seeds"]),
        "target_split_seeds": list(experiment["target_split_seeds"]),
        "role_partitions": list(experiment["role_partitions"]),
        "endpoint_seeds": list(experiment["endpoint_seeds"]),
        "endpoint_fractions": list(experiment["endpoint_fractions"]),
        "role_splits": role_splits,
        "adaptation_units_by_target_split_seed": {
            seed: value["adaptation_units"] for seed, value in role_splits.items()
        },
        "train_file_hashes": a2_protocol["train_file_hashes"],
        "normalizer_fit_scope": "source_train_only",
        "prior_graph_fit_scope": "source_train_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    first_split = str(experiment["target_split_seeds"][0])
    first_partition = str(experiment["role_partitions"][0])
    protocol["selection_units"] = role_splits[first_split]["partitions"][first_partition]["selection_units"]
    protocol["confirmation_units"] = role_splits[first_split]["partitions"][first_partition]["confirmation_units"]
    protocol["protocol_hash"] = a1.canonical_hash(protocol)
    return protocol


def protocol_rows(protocols: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for domain, protocol in protocols.items():
        for split_seed, split in protocol["role_splits"].items():
            for unit in split["adaptation_units"]:
                rows.append({"target_domain": domain, "target_split_seed": int(split_seed), "role_partition": "fixed", "role": "adaptation", "unit": unit})
            for unit in split["evaluation_pool_units"]:
                rows.append({"target_domain": domain, "target_split_seed": int(split_seed), "role_partition": "pool", "role": "evaluation_pool", "unit": unit})
            for partition, roles in split["partitions"].items():
                for role in ("selection", "confirmation"):
                    for unit in roles[f"{role}_units"]:
                        rows.append({"target_domain": domain, "target_split_seed": int(split_seed), "role_partition": int(partition), "role": role, "unit": unit})
    return pd.DataFrame(rows)


def balanced_assignment(units: list[int], domain: str, split_seed: int, partition: int, endpoint_seed: int, role: str) -> dict[int, float]:
    rng = np.random.default_rng(stable_seed("A2_1_endpoint", domain, split_seed, partition, endpoint_seed, role))
    order = list(map(int, rng.permutation(np.asarray(units, dtype=int))))
    fractions = list(ENDPOINT_FRACTIONS)
    offset = stable_seed(domain, split_seed, partition, endpoint_seed, role) % len(fractions)
    return {unit: float(fractions[(index + offset) % len(fractions)]) for index, unit in enumerate(order)}


def prepare_support_pool(cfg: dict, preprocessing: str, balance_mode: str, support_units: list[int], pool_units: list[int]):
    sensors = list(cfg["sensor_columns"])
    _, normalizer = a1.fit_source_normalizer_train_only(cfg, preprocessing)
    target = add_train_rul(a1.load_train_domain(cfg["data_dir"], cfg["target_domain"]), cfg["rul_cap"])
    features = sensors + a1.SETTING_FEATURE_COLUMNS if preprocessing in {"global_settings", "condition_settings"} else sensors
    normalized = normalizer.transform(target, sensors)
    support_frame = normalized.query("unit in @support_units")
    pool_frame = normalized.query("unit in @pool_units")
    if support_frame["unit"].nunique() != len(support_units) or pool_frame["unit"].nunique() != len(pool_units):
        raise ValueError("A2_1 target engine preparation is incomplete")
    support = a1.make_loader(support_frame, features, cfg, training=True, balance_mode=balance_mode, loader_seed=cfg["seed"] + 9000)
    pool = a1.make_loader(pool_frame, features, cfg, training=False, loader_seed=cfg["seed"] + 9200)
    return support, pool, len(features)


def engine_sufficient_stats(predictions: pd.DataFrame, epoch: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["squared_error"] = frame["error"] ** 2
    frame["absolute_error"] = frame["error"].abs()
    frame["label_squared"] = frame["label"] ** 2
    result = frame.groupby("unit", as_index=False).agg(
        window_count=("label", "size"),
        squared_error_sum=("squared_error", "sum"),
        absolute_error_sum=("absolute_error", "sum"),
        nasa_score=("nasa_contribution", "sum"),
        label_sum=("label", "sum"),
        label_squared_sum=("label_squared", "sum"),
    )
    result.insert(0, "epoch", int(epoch))
    return result


def endpoint_epoch_rows(predictions: pd.DataFrame, epoch: int) -> pd.DataFrame:
    endpoints = a2.stratified_endpoint_subset(a2.add_within_engine_index(predictions), list(ENDPOINT_FRACTIONS))
    endpoints.insert(0, "epoch", int(epoch))
    return endpoints[["epoch", "unit", "endpoint_fraction", "label", "prediction", "error", "nasa_contribution", "unit_window_index"]]


def train_target_trajectory(model: torch.nn.Module, support, pool, cfg: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    history, engine_parts, endpoint_parts = [], [], []
    for epoch in range(1, int(cfg["target_epochs"]) + 1):
        learner.train()
        losses = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss = F.mse_loss(prediction, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("A2_1 target loss became NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        predictions = a1.predict_with_units(learner, pool, device)
        engine_parts.append(engine_sufficient_stats(predictions, epoch))
        endpoint_parts.append(endpoint_epoch_rows(predictions, epoch))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "pool_prediction_count": int(len(predictions))})
        print(f"A2_1 target_epoch={epoch:02d}/{cfg['target_epochs']} loss={np.mean(losses):.4f} pool_windows={len(predictions)}")
    del learner
    return pd.concat(engine_parts, ignore_index=True), pd.concat(endpoint_parts, ignore_index=True), pd.DataFrame(history)


def metrics_from_engine_stats(frame: pd.DataFrame) -> dict[str, float]:
    count = float(frame["window_count"].sum())
    sse = float(frame["squared_error_sum"].sum())
    label_sum = float(frame["label_sum"].sum())
    label_sq = float(frame["label_squared_sum"].sum())
    denominator = label_sq - label_sum * label_sum / count
    return {
        "rmse": float(np.sqrt(sse / count)),
        "mae": float(frame["absolute_error_sum"].sum() / count),
        "r2": float(1.0 - sse / denominator) if denominator > 0 else 0.0,
        "nasa_score": float(frame["nasa_score"].sum()),
    }


def endpoint_subset(frame: pd.DataFrame, units: list[int], assignment: dict[int, float] | None = None, fraction: float | None = None) -> pd.DataFrame:
    subset = frame[frame["unit"].isin(units)].copy()
    if assignment is not None:
        expected = pd.DataFrame({"unit": list(assignment), "endpoint_fraction": list(assignment.values())})
        subset = subset.merge(expected, on=["unit", "endpoint_fraction"], how="inner")
    elif fraction is not None:
        subset = subset[np.isclose(subset["endpoint_fraction"], float(fraction))]
    else:
        raise ValueError("endpoint subset requires assignment or fixed fraction")
    if subset["unit"].nunique() != len(units) or len(subset) != len(units):
        raise AssertionError("endpoint assignment did not produce one row per engine")
    return subset


def best_full_epoch(engine_stats: pd.DataFrame, units: list[int]) -> int:
    candidates = []
    for epoch, group in engine_stats[engine_stats["unit"].isin(units)].groupby("epoch"):
        candidates.append((metrics_from_engine_stats(group)["rmse"], int(epoch)))
    return min(candidates)[1]


def best_endpoint_epoch(endpoint_rows: pd.DataFrame, units: list[int], assignment: dict[int, float]) -> int:
    candidates = []
    for epoch, group in endpoint_rows.groupby("epoch"):
        selected = endpoint_subset(group, units, assignment=assignment)
        candidates.append((regression_metrics(selected["label"], selected["prediction"])["rmse"], int(epoch)))
    return min(candidates)[1]


def annotated_engine_rows(frame: pd.DataFrame, common: dict) -> pd.DataFrame:
    output = frame.copy()
    output["rmse"] = np.sqrt(output["squared_error_sum"] / output["window_count"])
    output["mae"] = output["absolute_error_sum"] / output["window_count"]
    for column, value in reversed(list(common.items())):
        output.insert(0, column, value)
    return output


def annotated_endpoint_rows(frame: pd.DataFrame, common: dict) -> pd.DataFrame:
    output = frame.copy()
    output["window_count"] = 1
    output["squared_error_sum"] = output["error"] ** 2
    output["absolute_error_sum"] = output["error"].abs()
    output["rmse"] = output["error"].abs()
    output["mae"] = output["error"].abs()
    output["nasa_score"] = output["nasa_contribution"]
    for column, value in reversed(list(common.items())):
        output.insert(0, column, value)
    return output


def simulate_cell(experiment: dict, protocol: dict, domain: str, model: str, model_seed: int, split_seed: int, engine_stats: pd.DataFrame, endpoint_rows: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    split = protocol["role_splits"][str(split_seed)]
    results, primary_engine_parts = [], []
    for partition in experiment["role_partitions"]:
        roles = split["partitions"][str(partition)]
        selection_units = list(map(int, roles["selection_units"]))
        confirmation_units = list(map(int, roles["confirmation_units"]))
        full_epoch = best_full_epoch(engine_stats, selection_units)
        for endpoint_seed in experiment["endpoint_seeds"]:
            selection_assignment = balanced_assignment(selection_units, domain, split_seed, partition, endpoint_seed, "selection")
            confirmation_assignment = balanced_assignment(confirmation_units, domain, split_seed, partition, endpoint_seed, "confirmation")
            balanced_epoch = best_endpoint_epoch(endpoint_rows, selection_units, selection_assignment)
            selected_epochs = {
                "full_trajectory_selection": full_epoch,
                "balanced_endpoint_selection": balanced_epoch,
            }
            for selection_protocol, selected_epoch in selected_epochs.items():
                epoch_engines = engine_stats[(engine_stats["epoch"] == selected_epoch) & engine_stats["unit"].isin(confirmation_units)]
                epoch_endpoints = endpoint_rows[endpoint_rows["epoch"] == selected_epoch]
                evaluations: list[tuple[str, dict, pd.DataFrame]] = []
                evaluations.append(("full_trajectory", metrics_from_engine_stats(epoch_engines), epoch_engines))
                balanced = endpoint_subset(epoch_endpoints, confirmation_units, assignment=confirmation_assignment)
                evaluations.append(("balanced_endpoint", regression_metrics(balanced["label"], balanced["prediction"]), balanced))
                for fraction in ENDPOINT_FRACTIONS:
                    fixed = endpoint_subset(epoch_endpoints, confirmation_units, fraction=fraction)
                    name = f"fixed_endpoint_{int(round(fraction * 100)):03d}"
                    evaluations.append((name, regression_metrics(fixed["label"], fixed["prediction"]), fixed))
                for evaluation_protocol, metrics, evaluation_frame in evaluations:
                    common = {
                        "target_domain": domain,
                        "model": model,
                        "model_seed": int(model_seed),
                        "target_split_seed": int(split_seed),
                        "role_partition": int(partition),
                        "endpoint_seed": int(endpoint_seed),
                        "selection_protocol": selection_protocol,
                        "evaluation_protocol": evaluation_protocol,
                    }
                    results.append({
                        **common,
                        **metrics,
                        "selected_epoch": int(selected_epoch),
                        "adaptation_units": split["adaptation_units"],
                        "selection_units": selection_units,
                        "confirmation_units": confirmation_units,
                        "normalizer_fit_scope": "source_train_only",
                        "confirmation_used_for_selection": False,
                        "official_test_files_accessed": False,
                        "official_test_forward_run": False,
                    })
                    primary = (
                        (selection_protocol == "full_trajectory_selection" and evaluation_protocol in {"full_trajectory", "balanced_endpoint"})
                        or (selection_protocol == "balanced_endpoint_selection" and evaluation_protocol == "balanced_endpoint")
                    )
                    if primary:
                        if evaluation_protocol == "full_trajectory":
                            primary_engine_parts.append(annotated_engine_rows(evaluation_frame, common))
                        else:
                            primary_engine_parts.append(annotated_endpoint_rows(evaluation_frame, common))
    return results, pd.concat(primary_engine_parts, ignore_index=True)


def cell_id(domain: str, model: str, model_seed: int, split_seed: int) -> str:
    return f"experimentA2_1_{domain.lower()}_k05_mseed{model_seed}_tsplit{split_seed}_{model}"


def reuse_a2_source_cache(base: dict, experiment: dict, protocol: dict, architecture: str, model_seed: int, prior: torch.Tensor) -> tuple[dict, list, dict] | None:
    if experiment["quick_mode"]:
        return None
    a2_output = Path(experiment["a2_output_dir"])
    manifest_path = a2_output / "experimentA2_manifest.json"
    cache_path = a2_output / "shards" / f"{base['target_domain']}_mseed{model_seed:03d}" / "source_cache" / f"experimentA1_2_{architecture}_{base['target_domain']}_mseed{model_seed}.pt"
    if not manifest_path.is_file() or not cache_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    a2_base = manifest.get("base_config", {})
    required_equal = ("sensor_columns", "window_size", "window_stride", "rul_cap", "batch_size", "hidden_dim", "embedding_dim", "gat_heads", "dropout", "source_pretrain_steps", "source_pretrain_lr", "source_pretrain_weight_decay")
    for key in required_equal:
        if key not in a2_base or a2_base[key] != base[key]:
            return None
    cached = a1.safe_torch_load(cache_path)
    inventory = cached.get("inventory", {})
    feature_count = int(inventory.get("feature_count", -1))
    if feature_count < 1:
        return None
    expected = a12.source_signature(
        base=base,
        experiment=experiment,
        protocol=protocol,
        architecture=architecture,
        model_seed=model_seed,
        feature_count=feature_count,
        prior=prior,
        git_commit=str(manifest.get("git_commit", "unknown")),
        script_hash=str(manifest.get("script_hash", "unknown")),
    )
    if cached.get("signature") != expected:
        return None
    inventory = {**inventory, "source_cache_origin": "verified_experimentA2", "source_cache_path": str(cache_path)}
    return cached["state"], cached.get("history", []), inventory


def load_source(base: dict, experiment: dict, protocol: dict, architecture: str, model_seed: int, prior: torch.Tensor, git_commit: str, script_hash: str):
    reused = reuse_a2_source_cache(base, experiment, protocol, architecture, model_seed, prior)
    if reused is not None:
        print(f"[A2_1] reused verified A2 source cache domain={base['target_domain']} model={architecture} seed={model_seed}")
        return reused
    state, history, inventory = a12.load_or_train_source(
        base=base,
        experiment=experiment,
        protocol=protocol,
        architecture=architecture,
        model_seed=model_seed,
        prior=prior,
        git_commit=git_commit,
        script_hash=script_hash,
    )
    return state, history, {**inventory, "source_cache_origin": "A2_1_local"}


def evaluate_training_cell(base: dict, experiment: dict, protocol: dict, model_name: str, model_seed: int, split_seed: int, source_state: dict, source_history: list, inventory: dict, prior: torch.Tensor):
    domain = protocol["target_domain"]
    run_seed = a2.target_run_seed(domain, model_seed, split_seed)
    cfg = deepcopy(base)
    cfg.update({"seed": run_seed, "target_domain": domain, "source_domains": protocol["source_domains"]})
    split = protocol["role_splits"][str(split_seed)]
    support, pool, feature_count = prepare_support_pool(cfg, experiment["preprocessing"], experiment["balance_mode"], split["adaptation_units"], split["evaluation_pool_units"])
    a1.seed_everything(run_seed)
    model = exp17b.build_model_17b(model_name, feature_count, cfg, prior, prior)
    model.load_state_dict(source_state)
    device = a1.resolve_device(cfg["device"])
    engine_stats, endpoint_rows, history = train_target_trajectory(model, support, pool, cfg, device)
    identifier = cell_id(domain, model_name, model_seed, split_seed)
    results, primary_engines = simulate_cell(experiment, protocol, domain, model_name, model_seed, split_seed, engine_stats, endpoint_rows)
    for frame in (engine_stats, endpoint_rows, history):
        frame.insert(0, "cell_id", identifier)
        frame.insert(1, "target_domain", domain)
        frame.insert(2, "model", model_name)
        frame.insert(3, "model_seed", int(model_seed))
        frame.insert(4, "target_split_seed", int(split_seed))
    for row in results:
        row.update({
            "experiment_id": "experimentA2_1",
            "cell_id": identifier,
            "target_run_seed": int(run_seed),
            "k": int(experiment["k"]),
            "source_pretrain_steps": int(base["source_pretrain_steps"]),
            "target_epochs_planned": int(base["target_epochs"]),
            "total_parameter_count": inventory["total_parameter_count"],
            "target_trainable_parameter_count": inventory["predictor_parameter_count"],
            "source_signature": inventory["source_signature"],
            "source_history_rows": int(len(source_history)),
        })
    primary_engines.insert(0, "cell_id", identifier)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, engine_stats, endpoint_rows, history, primary_engines


def root_paths(output: Path) -> dict[str, Path]:
    p = "experimentA2_1"
    return {
        "manifest": output / f"{p}_manifest.json",
        "protocol": output / f"{p}_protocol.json",
        "engine_roles": output / f"{p}_engine_roles.csv",
        "dry_run": output / f"{p}_dry_run.json",
        "run_json": output / f"{p}_run_level.json",
        "run_csv": output / f"{p}_run_level.csv",
        "engine_stats": output / f"{p}_epoch_engine_stats.csv",
        "endpoint_rows": output / f"{p}_epoch_endpoint_predictions.csv",
        "history": output / f"{p}_target_history.csv",
        "primary_engines": output / f"{p}_primary_per_engine.csv",
        "inventory": output / f"{p}_source_inventory.csv",
        "summary": output / f"{p}_summary.csv",
        "paired": output / f"{p}_paired_model_cells.csv",
        "comparisons": output / f"{p}_paired_model_comparisons.csv",
        "regret": output / f"{p}_selection_regret.csv",
        "scheme": output / f"{p}_scheme_stability.csv",
        "decision": output / f"{p}_confirmation_decision.json",
        "lock": output / f"{p}_lock_candidate.json",
    }


def shard_dir(output: Path, domain: str, seed: int) -> Path:
    return output / "shards" / f"{domain}_mseed{seed:03d}"


def shard_paths(output: Path, domain: str, seed: int) -> dict[str, Path]:
    d = shard_dir(output, domain, seed)
    return {"directory": d, "status": d / "worker_status.json", "run_json": d / "run_level.json", "run_csv": d / "run_level.csv", "engine_stats": d / "epoch_engine_stats.csv", "endpoint_rows": d / "epoch_endpoint_predictions.csv", "history": d / "target_history.csv", "primary_engines": d / "primary_per_engine.csv", "inventory": d / "source_inventory.csv"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_worker_state(paths: dict[str, Path]) -> dict:
    completed = set()
    if paths["status"].is_file():
        completed = set(json.loads(paths["status"].read_text(encoding="utf-8")).get("completed_cell_ids", []))
    results = json.loads(paths["run_json"].read_text(encoding="utf-8")) if paths["run_json"].is_file() else []
    results = [row for row in results if row["cell_id"] in completed]
    state = {"completed": completed, "results": results}
    for name in ("engine_stats", "endpoint_rows", "history", "primary_engines", "inventory"):
        frame = load_csv(paths[name])
        if name != "inventory" and not frame.empty:
            frame = frame[frame["cell_id"].isin(completed)]
        state[name] = frame
    return state


def save_worker_state(paths: dict[str, Path], state: dict, expected: int) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["run_json"], state["results"])
    a1.atomic_write_text(paths["run_csv"], pd.DataFrame(state["results"]).to_csv(index=False))
    for name in ("engine_stats", "endpoint_rows", "history", "primary_engines", "inventory"):
        a1.atomic_write_text(paths[name], state[name].to_csv(index=False))
    atomic_json(paths["status"], {
        "completed_cell_ids": sorted(state["completed"]),
        "completed_training_cells": len(state["completed"]),
        "expected_training_cells": expected,
        "complete": len(state["completed"]) == expected,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    })


def worker_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    domain, model_seed = str(args.worker_domain), int(args.worker_seed)
    if domain not in experiment["domains"] or model_seed not in experiment["model_seeds"]:
        raise ValueError("unregistered A2_1 worker")
    output = Path(base["output_dir"])
    paths = shard_paths(output, domain, model_seed)
    worker_base = deepcopy(base)
    worker_base.update({"output_dir": str(paths["directory"]), "target_domain": domain, "source_domains": [d for d in DOMAINS if d != domain]})
    if args.device == "auto" and torch.cuda.is_available():
        worker_base["device"] = "cuda:0"
    protocol = build_protocol(worker_base, experiment, domain)
    prior, correlation, graph_fit = a1.source_correlation_adjacency_train_only(worker_base, experiment["preprocessing"], int(experiment["sensor_graph_k"]))
    paths["directory"].mkdir(parents=True, exist_ok=True)
    sensors = list(worker_base["sensor_columns"])
    a1.atomic_write_text(paths["directory"] / "prior_adjacency.csv", pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv())
    a1.atomic_write_text(paths["directory"] / "prior_correlation.csv", pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv())
    worker_manifest_path = paths["directory"] / "worker_manifest.json"
    worker_manifest = {"script_version": SCRIPT_VERSION, "script_hash": a1.file_sha256(Path(__file__)), "git_commit": a1.git_commit(PROJECT_ROOT), "target_domain": domain, "model_seed": model_seed, "protocol_hash": protocol["protocol_hash"], "graph_fit": graph_fit, "official_test_files_accessed": False}
    if worker_manifest_path.is_file() and paths["status"].is_file():
        previous = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
        for key in ("script_hash", "target_domain", "model_seed", "protocol_hash"):
            if previous.get(key) != worker_manifest.get(key):
                raise RuntimeError(f"existing A2_1 shard is incompatible at {key}; use a new output directory")
    atomic_json(worker_manifest_path, worker_manifest)
    state = load_worker_state(paths)
    expected = len(MODELS) * len(experiment["target_split_seeds"])
    script_hash, git_commit = a1.file_sha256(Path(__file__)), a1.git_commit(PROJECT_ROOT)
    for model_name in MODELS:
        pending = [s for s in experiment["target_split_seeds"] if cell_id(domain, model_name, model_seed, s) not in state["completed"]]
        if not pending:
            continue
        source_state, source_history, inventory = load_source(worker_base, experiment, protocol, model_name, model_seed, prior, git_commit, script_hash)
        inventory_row = {"target_domain": domain, **inventory}
        if state["inventory"].empty:
            state["inventory"] = pd.DataFrame([inventory_row])
        else:
            keep = ~((state["inventory"]["target_domain"] == domain) & (state["inventory"]["model"] == model_name) & (state["inventory"]["model_seed"] == model_seed))
            state["inventory"] = pd.concat([state["inventory"][keep], pd.DataFrame([inventory_row])], ignore_index=True)
        for split_seed in pending:
            rows, engine_stats, endpoint_rows, history, primary = evaluate_training_cell(worker_base, experiment, protocol, model_name, model_seed, int(split_seed), deepcopy(source_state), source_history, inventory, prior)
            state["results"].extend(rows)
            for name, frame in (("engine_stats", engine_stats), ("endpoint_rows", endpoint_rows), ("history", history), ("primary_engines", primary)):
                state[name] = pd.concat([state[name], frame], ignore_index=True)
            state["completed"].add(cell_id(domain, model_name, model_seed, int(split_seed)))
            save_worker_state(paths, state, expected)
    save_worker_state(paths, state, expected)
    print(paths["status"].read_text(encoding="utf-8"))


def choose_gpus(args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    inventory = a2.query_gpus()
    if args.gpus:
        devices = [int(v.strip()) for v in args.gpus.split(",")]
        if len(devices) != len(set(devices)):
            raise ValueError("--gpus contains duplicates")
        if not set(devices).issubset({row["index"] for row in inventory}):
            raise RuntimeError("a requested GPU is unavailable")
    else:
        visible = a2.visible_gpu_filter()
        candidates = [row for row in inventory if (visible is None or row["index"] in visible) and row["free_mb"] >= args.min_free_memory_mb and row["utilization"] <= args.max_gpu_utilization]
        candidates.sort(key=lambda row: (-row["free_mb"], row["utilization"]))
        devices = [row["index"] for row in candidates]
    if args.max_workers > 0:
        devices = devices[: args.max_workers]
    return devices, inventory


def worker_command(args: argparse.Namespace, domain: str, seed: int, device: str, output: Path) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-domain", domain, "--worker-seed", str(seed), "--output-dir", str(output), "--device", device, "--bootstrap-repetitions", str(args.bootstrap_repetitions)]
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.a2_output_dir:
        command.extend(["--a2-output-dir", args.a2_output_dir])
    if args.quick:
        command.append("--quick")
    return command


def run_workers(args: argparse.Namespace, tasks: list[tuple[str, int]], output: Path) -> None:
    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]
        inventory = []
    else:
        devices, inventory = choose_gpus(args)
        if not devices:
            raise RuntimeError("no idle GPU met A2_1 thresholds; inventory=" + json.dumps(inventory, ensure_ascii=False))
    print(json.dumps({"scheduler": "experimentA2_1", "tasks": [{"domain": d, "seed": s} for d, s in tasks], "devices": devices, "gpu_inventory": inventory}, ensure_ascii=False, indent=2))
    pending, active = list(tasks), {}
    while pending or active:
        for device in [value for value in devices if value not in active]:
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
            print(f"[A2_1] launched domain={domain} seed={seed} device={device} pid={process.pid}")
        finished = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            if code != 0:
                tail = "\n".join(record["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(f"A2_1 worker failed domain={record['domain']} seed={record['seed']} exit={code}\n{tail}")
            print(f"[A2_1] completed domain={record['domain']} seed={record['seed']} device={device}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            time.sleep(5)


def merge_shards(output: Path, tasks: list[tuple[str, int]], experiment: dict) -> dict:
    merged: dict[str, Any] = {"results": [], "engine_stats": [], "endpoint_rows": [], "history": [], "primary_engines": [], "inventory": []}
    expected = len(MODELS) * len(experiment["target_split_seeds"])
    for domain, seed in tasks:
        paths = shard_paths(output, domain, seed)
        if not paths["status"].is_file():
            raise RuntimeError(f"missing worker status: {paths['status']}")
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        if not status.get("complete") or status.get("completed_training_cells") != expected:
            raise RuntimeError(f"incomplete worker: {paths['status']}")
        merged["results"].extend(json.loads(paths["run_json"].read_text(encoding="utf-8")))
        for name in ("engine_stats", "endpoint_rows", "history", "primary_engines", "inventory"):
            merged[name].append(load_csv(paths[name]))
    for name in ("engine_stats", "endpoint_rows", "history", "primary_engines", "inventory"):
        merged[name] = pd.concat(merged[name], ignore_index=True)
    return merged


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    groups = ["target_domain", "selection_protocol", "evaluation_protocol", "model"]
    rows = []
    for keys, frame in results.groupby(groups):
        row = dict(zip(groups, keys))
        row.update({"n_rows": int(len(frame)), "n_model_seeds": int(frame["model_seed"].nunique()), "n_target_splits": int(frame["target_split_seed"].nunique()), "n_role_partitions": int(frame["role_partition"].nunique()), "n_endpoint_seeds": int(frame["endpoint_seed"].nunique())})
        for metric in ("rmse", "mae", "r2", "nasa_score"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups)


def paired_cells(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed", "selection_protocol", "evaluation_protocol"]
    pivot = results.pivot(index=keys, columns="model", values=["rmse", "mae", "r2", "nasa_score"]).reset_index()
    pivot.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in pivot.columns]
    output = pivot[keys].copy()
    for metric in ("rmse", "mae", "r2", "nasa_score"):
        output[f"{metric}_window_graph"] = pivot[f"{metric}_window_graph"]
        output[f"{metric}_window_no_graph"] = pivot[f"{metric}_window_no_graph"]
        output[f"{metric}_delta_graph_minus_no_graph"] = pivot[f"{metric}_window_graph"] - pivot[f"{metric}_window_no_graph"]
    output["rmse_graph_win"] = output["rmse_delta_graph_minus_no_graph"] < 0
    return output


def build_regret(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"]
    def view(selection: str, evaluation: str, value_name: str) -> pd.DataFrame:
        frame = results[(results["selection_protocol"] == selection) & (results["evaluation_protocol"] == evaluation)]
        return frame[keys + ["model", "rmse"]].rename(columns={"rmse": value_name})
    full = view("full_trajectory_selection", "full_trajectory", "full_rmse")
    endpoint = view("balanced_endpoint_selection", "balanced_endpoint", "endpoint_rmse")
    cross = view("full_trajectory_selection", "balanced_endpoint", "cross_rmse")
    full_choice = full.sort_values("full_rmse").groupby(keys, as_index=False).first().rename(columns={"model": "full_selected_model"})
    endpoint_choice = endpoint.sort_values("endpoint_rmse").groupby(keys, as_index=False).first().rename(columns={"model": "endpoint_selected_model", "endpoint_rmse": "endpoint_best_rmse"})
    result = full_choice.merge(endpoint_choice, on=keys)
    full_endpoint_state = endpoint.rename(columns={"model": "full_selected_model", "endpoint_rmse": "endpoint_rmse_full_model_endpoint_state"})
    full_cross_state = cross.rename(columns={"model": "full_selected_model", "cross_rmse": "endpoint_rmse_full_model_full_state"})
    result = result.merge(full_endpoint_state, on=keys + ["full_selected_model"]).merge(full_cross_state, on=keys + ["full_selected_model"])
    result["rank_flip"] = result["full_selected_model"] != result["endpoint_selected_model"]
    result["model_selection_regret"] = result["endpoint_rmse_full_model_endpoint_state"] - result["endpoint_best_rmse"]
    result["end_to_end_protocol_delta"] = result["endpoint_rmse_full_model_full_state"] - result["endpoint_best_rmse"]
    return result


def sensitivity_bootstrap(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> tuple[float, float]:
    domains = sorted(frame["target_domain"].unique())
    model_seeds = sorted(frame["model_seed"].unique())
    split_seeds = sorted(frame["target_split_seed"].unique())
    partitions = sorted(frame["role_partition"].unique())
    endpoint_seeds = sorted(frame["endpoint_seed"].unique())
    lookup = frame.set_index(["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed"])[column]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        chosen_domains = rng.choice(domains, len(domains), replace=True)
        values = []
        for domain in chosen_domains:
            chosen_models = rng.choice(model_seeds, len(model_seeds), replace=True)
            chosen_splits = rng.choice(split_seeds, len(split_seeds), replace=True)
            for model_seed in chosen_models:
                for split_seed in chosen_splits:
                    partition = int(rng.choice(partitions))
                    endpoint_seed = int(rng.choice(endpoint_seeds))
                    values.append(float(lookup.loc[(domain, int(model_seed), int(split_seed), partition, endpoint_seed)]))
        samples[repeat] = float(np.mean(values))
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def comparison_summary(paired: pd.DataFrame, experiment: dict) -> pd.DataFrame:
    rows = []
    groups = ["selection_protocol", "evaluation_protocol"]
    for keys, frame in paired.groupby(groups):
        for scope, scoped in [("ALL", frame)] + list(frame.groupby("target_domain")):
            ci = sensitivity_bootstrap(scoped, "rmse_delta_graph_minus_no_graph", int(experiment["bootstrap_repetitions"]), stable_seed("comparison", *keys, scope))
            domain_means = scoped.groupby("target_domain")["rmse_delta_graph_minus_no_graph"].mean()
            reference = float(scoped["rmse_window_no_graph"].mean())
            rows.append({
                "selection_protocol": keys[0], "evaluation_protocol": keys[1], "scope": scope,
                "n_rows": int(len(scoped)), "n_domains": int(scoped["target_domain"].nunique()),
                "rmse_delta_mean": float(scoped["rmse_delta_graph_minus_no_graph"].mean()),
                "rmse_improvement_pct": float(-100 * scoped["rmse_delta_graph_minus_no_graph"].mean() / reference),
                "rmse_win_rate": float(scoped["rmse_graph_win"].mean()),
                "rmse_domain_win_count": int((domain_means < 0).sum()),
                "rmse_boot_ci95_low": ci[0], "rmse_boot_ci95_high": ci[1],
                "nasa_score_delta_mean": float(scoped["nasa_score_delta_graph_minus_no_graph"].mean()),
            })
    return pd.DataFrame(rows)


def make_decision(experiment: dict, results: pd.DataFrame, regret: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    expected_cells = len(experiment["domains"]) * len(MODELS) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"])
    expected_records = expected_cells * len(experiment["role_partitions"]) * len(experiment["endpoint_seeds"]) * len(SELECTION_PROTOCOLS) * len(EVALUATION_PROTOCOLS)
    regret_ci = sensitivity_bootstrap(regret, "model_selection_regret", int(experiment["bootstrap_repetitions"]), 8801)
    e2e_ci = sensitivity_bootstrap(regret, "end_to_end_protocol_delta", int(experiment["bootstrap_repetitions"]), 8802)
    domain_flip = regret.groupby("target_domain")["rank_flip"].mean()
    endpoint_means = regret.groupby("endpoint_seed")["model_selection_regret"].mean()
    partition_means = regret.groupby("role_partition")["model_selection_regret"].mean()
    affected = int((domain_flip >= experiment["minimum_rank_flip_rate"]).sum())
    positive_endpoint = int((endpoint_means > 0).sum())
    positive_partition = int((partition_means > 0).sum())
    protocol_pass = bool(regret_ci[0] > 0 and affected >= experiment["minimum_affected_domains"] and positive_endpoint >= experiment["minimum_positive_endpoint_seeds"] and positive_partition >= experiment["minimum_positive_role_partitions"])
    primary = comparisons[(comparisons["scope"] == "ALL") & (comparisons["selection_protocol"] == "balanced_endpoint_selection") & (comparisons["evaluation_protocol"] == "balanced_endpoint")].iloc[0]
    graph_eligible = bool(primary["rmse_improvement_pct"] >= experiment["minimum_graph_improvement_pct"] and primary["rmse_domain_win_count"] >= experiment["minimum_graph_domain_wins"] and primary["rmse_boot_ci95_high"] < 0 and primary["nasa_score_delta_mean"] <= 0)
    lock = {
        "experiment_id": "experimentA2_1", "candidate_model": "window_graph" if graph_eligible else "window_no_graph",
        "eligible_for_locked_official_confirmation": graph_eligible,
        "reason": "window graph met all registered balanced-endpoint criteria" if graph_eligible else "window graph did not meet all registered balanced-endpoint criteria",
        "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    scheme = pd.concat([
        endpoint_means.rename("model_selection_regret_mean").reset_index().assign(sensitivity_dimension="endpoint_seed").rename(columns={"endpoint_seed": "level"}),
        partition_means.rename("model_selection_regret_mean").reset_index().assign(sensitivity_dimension="role_partition").rename(columns={"role_partition": "level"}),
    ], ignore_index=True)
    decision = {
        "experiment_id": "experimentA2_1", "expected_training_cells": expected_cells,
        "completed_training_cells": int(results["cell_id"].nunique()), "expected_evaluation_records": expected_records,
        "completed_evaluation_records": int(len(results)), "complete": bool(results["cell_id"].nunique() == expected_cells and len(results) == expected_records),
        "quick_mode": bool(experiment["quick_mode"]), "official_test_files_accessed": False, "official_test_forward_run": False,
        "two_model_rank_flip_rate": float(regret["rank_flip"].mean()), "rank_flip_rate_by_domain": {k: float(v) for k, v in domain_flip.items()},
        "rank_flip_affected_domains": affected, "model_selection_regret_mean": float(regret["model_selection_regret"].mean()), "model_selection_regret_ci95": list(regret_ci),
        "end_to_end_protocol_delta_mean": float(regret["end_to_end_protocol_delta"].mean()), "end_to_end_protocol_delta_ci95": list(e2e_ci),
        "positive_endpoint_seed_count": positive_endpoint, "positive_role_partition_count": positive_partition,
        "endpoint_protocol_gap_confirmed": protocol_pass, "lock_candidate": lock,
    }
    if experiment["quick_mode"]:
        decision.update({"passed": decision["complete"], "reason": "quick smoke run only; do not interpret"})
    else:
        decision.update({"passed": bool(decision["complete"] and protocol_pass), "reason": "A2_1 confirmed endpoint-protocol robustness across role and endpoint schemes" if decision["complete"] and protocol_pass else "A2_1 did not meet every registered endpoint-robustness criterion"})
    return decision, lock, scheme


def parent_main(args: argparse.Namespace, base: dict, experiment: dict) -> None:
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = root_paths(output)
    protocols = {domain: build_protocol(base, experiment, domain) for domain in experiment["domains"]}
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_hash": a1.file_sha256(Path(__file__)),
        "git_commit": a1.git_commit(PROJECT_ROOT),
        "base_config": {key: value for key, value in base.items() if key != "device"},
        "experiment_config": experiment,
        "protocol_hashes": {domain: protocol["protocol_hash"] for domain, protocol in protocols.items()},
        "registered_primary_question": "Does the two-model endpoint-selection gap survive repeated engine roles and balanced endpoint assignments?",
        "sensitivity_repeats_are_not_independent_training_cells": True,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["protocol"], protocols)
    a1.atomic_write_text(paths["engine_roles"], protocol_rows(protocols).to_csv(index=False))
    expected_cells = len(experiment["domains"]) * len(MODELS) * len(experiment["model_seeds"]) * len(experiment["target_split_seeds"])
    expected_records = expected_cells * len(experiment["role_partitions"]) * len(experiment["endpoint_seeds"]) * len(SELECTION_PROTOCOLS) * len(EVALUATION_PROTOCOLS)
    dry = {
        "experiment_id": "experimentA2_1", "domains": experiment["domains"], "models": experiment["models"],
        "model_seeds": experiment["model_seeds"], "target_split_seeds": experiment["target_split_seeds"],
        "role_partitions": experiment["role_partitions"], "endpoint_seeds": experiment["endpoint_seeds"],
        "expected_training_cells": expected_cells, "expected_evaluation_records": expected_records,
        "protocol_hashes": manifest["protocol_hashes"], "a2_source_cache_root": experiment["a2_output_dir"],
        "gpu_inventory": a2.query_gpus(), "official_test_files_accessed": False, "official_test_forward_run": False,
    }
    atomic_json(paths["dry_run"], dry)
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    tasks = [(domain, seed) for domain in experiment["domains"] for seed in experiment["model_seeds"]]
    run_workers(args, tasks, output)
    merged = merge_shards(output, tasks, experiment)
    results = pd.DataFrame(merged["results"]).sort_values(["target_domain", "model_seed", "target_split_seed", "role_partition", "endpoint_seed", "model", "selection_protocol", "evaluation_protocol"])
    if results["cell_id"].nunique() != expected_cells or len(results) != expected_records:
        raise RuntimeError("A2_1 merged output is incomplete")
    if results[["official_test_files_accessed", "official_test_forward_run"]].astype(bool).any().any():
        raise RuntimeError("A2_1 detected an official-test contamination flag")
    summary = summarize(results)
    paired = paired_cells(results)
    comparisons = comparison_summary(paired, experiment)
    regret = build_regret(results)
    decision, lock, scheme = make_decision(experiment, results, regret, comparisons)
    atomic_json(paths["run_json"], results.to_dict("records"))
    a1.atomic_write_text(paths["run_csv"], results.to_csv(index=False))
    for name in ("engine_stats", "endpoint_rows", "history", "primary_engines", "inventory"):
        a1.atomic_write_text(paths[name], merged[name].to_csv(index=False))
    a1.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    a1.atomic_write_text(paths["paired"], paired.to_csv(index=False))
    a1.atomic_write_text(paths["comparisons"], comparisons.to_csv(index=False))
    a1.atomic_write_text(paths["regret"], regret.to_csv(index=False))
    a1.atomic_write_text(paths["scheme"], scheme.to_csv(index=False))
    atomic_json(paths["decision"], decision)
    atomic_json(paths["lock"], lock)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    base, experiment = load_config(args)
    validate_config(base, experiment)
    if args.worker_domain is not None:
        if args.worker_seed is None:
            raise ValueError("--worker-domain requires --worker-seed")
        worker_main(args, base, experiment)
    else:
        parent_main(args, base, experiment)


if __name__ == "__main__":
    main()
