"""Experiment 24: train-only truncated-endpoint validation.

The experiment compares the budget-matched static prior with the compressed
fixed-source TCSG under a nested target-engine protocol. It reads C-MAPSS
training files only. Official test files are neither loaded nor evaluated.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from preprocess import cmapps_loader  # noqa: E402
from preprocess.rul_generator import add_train_rul  # noqa: E402
from scripts import experiment7_kshot_engines as exp7  # noqa: E402
from scripts import experiment21_fixed_context_compression as exp21  # noqa: E402
from scripts.run_condition_aware_experiment import (  # noqa: E402
    SETTING_FEATURE_COLUMNS,
    SourceConditionNormalizer,
    SourceGlobalNormalizer,
)


exp18 = exp21.exp18
exp17 = exp18.exp17
exp20 = exp21.exp20

SCRIPT_VERSION = "experiment24_truncated_endpoint_validation_v1"
EXPERIMENT18_SOURCE_VERSION = "experiment18_task_conditioned_sensor_graph_v1"
MODELS = ("static_budget_prior", "tcsg_fixed_source_gate2")
COMPARISONS = (
    (
        "tcsg_fixed_source_gate2",
        "static_budget_prior",
        "fixed_vs_static_budget",
    ),
)
PRIMARY_COMPARISONS = {"fixed_vs_static_budget"}
DEFAULT_SPLIT_SEEDS = [4027, 4028, 4029, 4030, 4031]
DEFAULT_MODEL_SEEDS = [42, 43, 44, 45, 46]
RUL_ANCHORS = (20, 50, 80)
OUTER_FRACTION = 0.20
OFFICIAL_TEST_HASH = "NOT_ACCESSED_EXPERIMENT24"

_original_parse_args = exp21._original_parse_args
_original_load_source_bundle = exp21._original_load_source_bundle
_original_context_train = exp18.train_target_context_head
_original_static_train = exp17.train_target_head
_normalized_cache: dict[tuple, tuple[dict[str, pd.DataFrame], list[str], list[int] | None]] = {}
_protocol: dict | None = None
_active_split_seed: int | None = None
_active_endpoint_loader = None
_need_source_tasks = False


def parse_args():
    args = _original_parse_args()
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = f"outputs/experiment24_truncated_endpoint_validation/{args.target}"
    if args.target_split_seeds == [3027, 3028, 3029, 3030, 3031]:
        args.target_split_seeds = list(DEFAULT_SPLIT_SEEDS)
    if args.k_values == [2, 5]:
        args.k_values = [5]
    if args.evaluation_scope != "validation" or args.confirm_official_test:
        raise ValueError("Experiment 24 never permits official-test evaluation")
    if set(args.k_values) != {5}:
        raise ValueError("Experiment 24 is locked to K=5")
    args.skip_official_count_check = True
    args.save_target_checkpoints = False
    return args


def train_path(data_dir: str, domain: str) -> Path:
    root = Path(data_dir)
    candidates = (root / domain / f"train_{domain}.txt", root / f"train_{domain}.txt")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing train_{domain}.txt under {root}")
    return path


def load_train(data_dir: str, domain: str, rul_cap: int) -> pd.DataFrame:
    return add_train_rul(cmapps_loader._read(train_path(data_dir, domain)), rul_cap)


def normalized_train_frames(cfg: dict, preprocessing: str):
    key = (
        str(Path(cfg["data_dir"]).resolve()),
        tuple(cfg["source_domains"]),
        cfg["target_domain"],
        preprocessing,
        int(cfg.get("normalizer_seed", 2026)),
        int(cfg.get("condition_count", 6)),
        int(cfg["rul_cap"]),
        tuple(cfg["sensor_columns"]),
    )
    if key in _normalized_cache:
        return _normalized_cache[key]

    domains = list(cfg["source_domains"]) + [cfg["target_domain"]]
    raw = {
        domain: load_train(cfg["data_dir"], domain, cfg["rul_cap"])
        for domain in domains
    }
    sensors = list(cfg["sensor_columns"])
    source_fit = pd.concat(
        [raw[domain] for domain in cfg["source_domains"]], ignore_index=True
    )
    condition_aware = preprocessing in {"condition_norm", "condition_settings"}
    include_settings = preprocessing in {"global_settings", "condition_settings"}
    if condition_aware:
        normalizer = SourceConditionNormalizer(
            n_conditions=cfg.get("condition_count", 6),
            seed=cfg.get("normalizer_seed", 2026),
            include_settings=include_settings,
        ).fit(source_fit, sensors)
        condition_counts = list(normalizer.source_condition_counts)
    else:
        normalizer = SourceGlobalNormalizer(include_settings=include_settings).fit(
            source_fit, sensors
        )
        condition_counts = None

    features = sensors + SETTING_FEATURE_COLUMNS if include_settings else sensors
    normalized = {
        domain: normalizer.transform(frame, sensors) for domain, frame in raw.items()
    }
    _normalized_cache[key] = normalized, features, condition_counts
    return _normalized_cache[key]


def stable_hash(values) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def create_protocol(args, cfg: dict) -> dict:
    global _protocol
    target = load_train(cfg["data_dir"], args.target, cfg["rul_cap"])
    units = np.asarray(sorted(target["unit"].unique()), dtype=int)
    outer_count = max(1, int(round(len(units) * OUTER_FRACTION)))
    if outer_count + args.validation_units + 5 > len(units):
        raise ValueError("Not enough target engines for outer, validation, and K=5 sets")

    outer_by_split = {}
    validation_by_split = {}
    orders = {}
    nested = {}
    for split_seed in args.target_split_seeds:
        order = np.random.default_rng(split_seed).permutation(units)
        outer = order[:outer_count]
        validation = order[outer_count : outer_count + args.validation_units]
        candidates = order[outer_count + args.validation_units :]
        adaptation = candidates[:5]
        sets = [set(outer), set(validation), set(adaptation)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError("Nested target engine sets overlap")
        outer_by_split[str(split_seed)] = outer.astype(int).tolist()
        validation_by_split[str(split_seed)] = validation.astype(int).tolist()
        orders[str(split_seed)] = candidates.astype(int).tolist()
        nested[str(split_seed)] = {"5": adaptation.astype(int).tolist()}

    first = str(args.target_split_seeds[0])
    _protocol = {
        "script_version": SCRIPT_VERSION,
        "protocol": "train_only_nested_truncated_endpoint",
        "target_domain": args.target,
        "train_engine_count": int(len(units)),
        "train_units": units.astype(int).tolist(),
        "train_file": str(train_path(cfg["data_dir"], args.target)),
        "target_split_seeds": list(args.target_split_seeds),
        "outer_fraction": OUTER_FRACTION,
        "outer_engine_count_per_split": outer_count,
        "inner_validation_engine_count_per_split": int(args.validation_units),
        "rul_anchors": list(RUL_ANCHORS),
        "outer_units_by_target_split_seed": outer_by_split,
        "validation_units_by_target_split_seed": validation_by_split,
        "adaptation_order_by_target_split_seed": orders,
        "nested_adaptation_units_by_target_split_seed": nested,
        "validation_units": validation_by_split[first],
        "k_values": [5],
        "normalizer_fit_scope": "source_train_only",
        "official_test_files_accessed": False,
        "official_test_engine_count": 0,
        "official_test_units": [],
        "official_test_units_hash": OFFICIAL_TEST_HASH,
    }
    return _protocol


def protocol_rows(protocol: dict) -> pd.DataFrame:
    rows = []
    for split_seed in protocol["target_split_seeds"]:
        key = str(split_seed)
        outer = set(protocol["outer_units_by_target_split_seed"][key])
        validation = set(protocol["validation_units_by_target_split_seed"][key])
        adaptation = set(protocol["nested_adaptation_units_by_target_split_seed"][key]["5"])
        for unit in protocol["train_units"]:
            role = (
                "outer_truncated_endpoint"
                if unit in outer
                else "inner_validation"
                if unit in validation
                else "adaptation_k5"
                if unit in adaptation
                else "unused_target_train"
            )
            rows.append(
                {"target_split_seed": split_seed, "unit": unit, "role": role}
            )
    return pd.DataFrame(rows)


def split_seed_for(validation_units, adaptation_units) -> int:
    if _active_split_seed is not None:
        return _active_split_seed
    if _protocol is None:
        raise RuntimeError("Experiment 24 protocol was not prepared")
    validation = set(map(int, validation_units))
    adaptation = set(map(int, adaptation_units))
    matches = []
    for split_seed in _protocol["target_split_seeds"]:
        key = str(split_seed)
        expected_validation = set(_protocol["validation_units_by_target_split_seed"][key])
        expected_adaptation = set(
            _protocol["nested_adaptation_units_by_target_split_seed"][key]["5"]
        )
        if validation == expected_validation and adaptation == expected_adaptation:
            matches.append(int(split_seed))
    if len(matches) != 1:
        raise RuntimeError("Could not identify the nested target split")
    return matches[0]


def truncated_endpoint_frame(
    target: pd.DataFrame, outer_units: list[int], window_size: int
) -> pd.DataFrame:
    frames = []
    for unit in outer_units:
        engine = target[target["unit"] == unit].sort_values("cycle")
        max_cycle = int(engine["cycle"].max())
        for anchor in RUL_ANCHORS:
            cutoff = max_cycle - anchor
            prefix = engine[engine["cycle"] <= cutoff].copy()
            if cutoff < 1 or prefix.empty:
                raise ValueError(
                    f"Engine {unit} is too short for the locked RUL={anchor} endpoint"
                )
            if float(prefix.iloc[-1]["rul"]) != float(anchor):
                raise AssertionError("Truncated endpoint label does not equal its RUL anchor")
            # Synthetic unit ids keep the three prefixes independent in make_windows.
            prefix["unit"] = int(unit) * 1000 + int(anchor)
            frames.append(prefix)
    result = pd.concat(frames, ignore_index=True)
    if result["unit"].nunique() != len(outer_units) * len(RUL_ANCHORS):
        raise AssertionError("Truncated endpoint construction lost an engine-anchor pair")
    return result


def prepare_train_only(
    cfg: dict,
    preprocessing_mode: str,
    balance_mode: str,
    validation_units: list[int],
    adaptation_units: list[int],
):
    global _active_endpoint_loader
    if _protocol is None:
        raise RuntimeError("Experiment 24 protocol was not prepared")
    normalized, features, condition_counts = normalized_train_frames(
        cfg, preprocessing_mode
    )
    source_tasks = {}
    if _need_source_tasks:
        for index, domain in enumerate(cfg["source_domains"]):
            source_tasks[domain] = exp7.make_loader(
                normalized[domain],
                features,
                cfg,
                training=True,
                balance_mode=balance_mode,
                loader_seed=cfg["seed"] + 1000 * (index + 1),
            )

    target = normalized[cfg["target_domain"]]
    adaptation = np.asarray(adaptation_units, dtype=int)
    validation = np.asarray(validation_units, dtype=int)
    if set(adaptation) & set(validation):
        raise AssertionError("Adaptation and inner-validation engines overlap")
    support_frame = target.query("unit in @adaptation")
    validation_frame = target.query("unit in @validation")
    if support_frame["unit"].nunique() != len(adaptation_units):
        raise ValueError("K-shot support engine count is incorrect")
    if validation_frame["unit"].nunique() != len(validation_units):
        raise ValueError("Inner-validation engine count is incorrect")

    split_seed = split_seed_for(validation_units, adaptation_units)
    outer_units = _protocol["outer_units_by_target_split_seed"][str(split_seed)]
    if (set(outer_units) & set(adaptation)) or (set(outer_units) & set(validation)):
        raise AssertionError("Outer endpoint engines leaked into adaptation/validation")

    support = exp7.make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        loader_seed=cfg["seed"] + 9000,
    )
    inner_validation = exp7.make_loader(
        validation_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9100,
    )
    endpoints = truncated_endpoint_frame(target, outer_units, cfg["window_size"])
    endpoint_loader = exp7.make_loader(
        endpoints,
        features,
        cfg,
        training=False,
        last_only=True,
        loader_seed=cfg["seed"] + 9200,
    )
    expected_endpoints = len(outer_units) * len(RUL_ANCHORS)
    if len(endpoint_loader.dataset) != expected_endpoints:
        raise AssertionError("Endpoint loader does not contain one row per engine-anchor")
    _active_endpoint_loader = endpoint_loader

    split = {
        "protocol": "train_only_nested_truncated_endpoint",
        "target_domain": cfg["target_domain"],
        "preprocessing_mode": preprocessing_mode,
        "balance_mode": balance_mode,
        "normalizer_fit_scope": "source_train_only",
        "normalizer_seed": cfg.get("normalizer_seed", 2026),
        "feature_columns": features,
        "adaptation_engine_count": len(adaptation_units),
        "adaptation_units": list(map(int, adaptation_units)),
        "validation_engine_count": len(validation_units),
        "validation_units": list(map(int, validation_units)),
        "outer_engine_count": len(outer_units),
        "outer_units": list(map(int, outer_units)),
        "outer_units_hash": stable_hash(outer_units),
        "endpoint_count": expected_endpoints,
        "rul_anchors": list(RUL_ANCHORS),
        "official_test_files_accessed": False,
        "official_test_engine_count": 0,
        "official_test_units": [],
        "official_test_units_hash": OFFICIAL_TEST_HASH,
    }
    if condition_counts is not None:
        split["source_condition_counts"] = condition_counts
    return (
        source_tasks,
        support,
        inner_validation,
        endpoint_loader,
        len(features),
        split,
    )


def train_only_prior(cfg: dict, preprocessing: str, neighbors: int):
    normalized, _, _ = normalized_train_frames(cfg, preprocessing)
    sensors = list(cfg["sensor_columns"])
    sensor_count = len(sensors)
    if not 1 <= neighbors < sensor_count:
        raise ValueError(f"--sensor-graph-k must be in [1, {sensor_count - 1}]")
    values = pd.concat(
        [normalized[domain] for domain in cfg["source_domains"]], ignore_index=True
    )[sensors].to_numpy(np.float64)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(
        np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0
    )
    np.fill_diagonal(correlation, 1.0)
    adjacency = np.eye(sensor_count, dtype=bool)
    for sensor in range(sensor_count):
        scores = correlation[sensor].copy()
        scores[sensor] = -np.inf
        adjacency[sensor, np.argsort(scores)[-neighbors:]] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    return torch.as_tensor(adjacency), correlation


def source_signature(args, cfg, feature_count, prior) -> str:
    previous = exp18.SCRIPT_VERSION
    try:
        exp18.SCRIPT_VERSION = EXPERIMENT18_SOURCE_VERSION
        return exp18.source_signature(args, cfg, feature_count, prior)
    finally:
        exp18.SCRIPT_VERSION = previous


def load_or_train_source_bundle(args, cfg, protocol, prior):
    global _need_source_tasks
    feature_count = len(cfg["sensor_columns"]) + (
        len(SETTING_FEATURE_COLUMNS)
        if args.preprocessing in {"global_settings", "condition_settings"}
        else 0
    )
    expected_signature = source_signature(args, cfg, feature_count, prior)
    filename = f"experiment18_source_bundle_{args.target}_modelseed{cfg['seed']}.pt"
    legacy_root = PROJECT_ROOT / "outputs" / "experiment18_task_conditioned_sensor_graph"
    candidates = (legacy_root / "source_cache" / filename, legacy_root / filename)
    for path in candidates:
        if not path.is_file():
            continue
        cached = exp17.safe_torch_load(path)
        if cached.get("signature") == expected_signature:
            print(f"[Experiment 24 source cache] {path}")
            return cached["states"], cached.get("histories", {}), cached["inventory"]

    previous_version = exp18.SCRIPT_VERSION
    _need_source_tasks = True
    try:
        exp18.SCRIPT_VERSION = EXPERIMENT18_SOURCE_VERSION
        return _original_load_source_bundle(args, cfg, protocol, prior)
    finally:
        exp18.SCRIPT_VERSION = previous_version
        _need_source_tasks = False


def prepare_source_summary(args, cfg, protocol, target_split_seed, k):
    global _need_source_tasks
    _need_source_tasks = True
    try:
        return exp20.source_global_summary(
            args, cfg, protocol, target_split_seed, k
        )
    finally:
        _need_source_tasks = False


def experiment24_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = f"{model_seed}:{target_split_seed}:experiment24".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def nasa_contribution(error: np.ndarray) -> np.ndarray:
    return np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )


def endpoint_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    metrics = regression_metrics(labels, predictions)
    errors = predictions - labels
    abs_errors = np.abs(errors)
    contributions = nasa_contribution(errors)
    if not all(math.isfinite(value) for value in metrics.values()) or not np.all(
        np.isfinite(contributions)
    ):
        raise RuntimeError("Endpoint evaluation produced NaN/Inf")
    worst_count = max(1, math.ceil(len(labels) * 0.20))
    metrics.update(
        {
            "signed_error_mean": float(errors.mean()),
            "overprediction_rate": float((errors > 0).mean()),
            "underprediction_rate": float((errors < 0).mean()),
            "worst20_abs_error_mean": float(np.sort(abs_errors)[-worst_count:].mean()),
            "worst20_nasa_contribution_mean": float(
                np.sort(contributions)[-worst_count:].mean()
            ),
        }
    )
    return metrics


def run_target_cell(
    args,
    cfg,
    protocol,
    regime,
    state,
    inventory,
    target_split_seed,
    k,
    prior,
):
    global _active_split_seed
    local_protocol = dict(protocol)
    local_protocol["validation_units"] = protocol[
        "validation_units_by_target_split_seed"
    ][str(target_split_seed)]
    _active_split_seed = int(target_split_seed)
    prepare_source_summary(args, cfg, local_protocol, target_split_seed, k)
    captured: dict[str, np.ndarray] = {}

    def context_train(model, support, validation, target_cfg, device, summary):
        trained, history, best_epoch = _original_context_train(
            model, support, validation, target_cfg, device, summary
        )
        labels, predictions = exp18.predict_context(
            trained, _active_endpoint_loader, device, summary
        )
        captured.update(labels=labels, predictions=predictions)
        return trained, history, best_epoch

    def static_train(model, support, validation, target_cfg, device):
        trained, history, best_epoch = _original_static_train(
            model, support, validation, target_cfg, device
        )
        labels, predictions = exp17.predict(trained, _active_endpoint_loader, device)
        captured.update(labels=labels, predictions=predictions)
        return trained, history, best_epoch

    previous_context_train = exp18.train_target_context_head
    previous_static_train = exp17.train_target_head
    exp18.train_target_context_head = context_train
    exp17.train_target_head = static_train
    try:
        result, audit = exp21.run_target_cell(
            args,
            cfg,
            local_protocol,
            regime,
            state,
            inventory,
            target_split_seed,
            k,
            prior,
        )
    finally:
        exp18.train_target_context_head = previous_context_train
        exp17.train_target_head = previous_static_train
        _active_split_seed = None

    if set(captured) != {"labels", "predictions"}:
        raise RuntimeError("Endpoint predictions were not captured after target adaptation")
    labels = captured["labels"]
    predictions = captured["predictions"]
    synthetic_units = np.asarray(_active_endpoint_loader.dataset.units, dtype=int)
    if not (len(labels) == len(predictions) == len(synthetic_units)):
        raise AssertionError("Endpoint labels, predictions, and units do not align")

    metrics = endpoint_metrics(labels, predictions)
    result.update({metric: metrics[metric] for metric in exp18.METRICS})
    result.update(
        {
            "evaluation_scope": "truncated_endpoint_validation",
            "inner_validation_engine_count": len(local_protocol["validation_units"]),
            "outer_engine_count": len(
                protocol["outer_units_by_target_split_seed"][str(target_split_seed)]
            ),
            "endpoint_sample_count": len(labels),
            "rul_anchors": list(RUL_ANCHORS),
            "endpoint_signed_error_mean": metrics["signed_error_mean"],
            "endpoint_overprediction_rate": metrics["overprediction_rate"],
            "endpoint_underprediction_rate": metrics["underprediction_rate"],
            "endpoint_worst20_abs_error_mean": metrics["worst20_abs_error_mean"],
            "endpoint_worst20_nasa_contribution_mean": metrics[
                "worst20_nasa_contribution_mean"
            ],
            "official_test_files_accessed": False,
            "official_test_engine_count": 0,
            "official_test_units_hash": OFFICIAL_TEST_HASH,
            "official_test_forward_run": False,
            "official_test_metrics": None,
        }
    )
    for anchor in RUL_ANCHORS:
        selected = labels == float(anchor)
        anchor_metrics = regression_metrics(labels[selected], predictions[selected])
        for metric in ("rmse", "mae", "nasa_score"):
            result[f"endpoint_rul{anchor}_{metric}"] = anchor_metrics[metric]

    errors = predictions - labels
    contributions = nasa_contribution(errors)
    prediction_rows = []
    for synthetic, label, prediction, error, contribution in zip(
        synthetic_units, labels, predictions, errors, contributions
    ):
        prediction_rows.append(
            {
                "target": args.target,
                "k": int(k),
                "target_split_seed": int(target_split_seed),
                "model_seed": int(cfg["seed"]),
                "model": regime,
                "outer_unit": int(synthetic // 1000),
                "rul_anchor": int(synthetic % 1000),
                "true_rul": float(label),
                "predicted_rul": float(prediction),
                "signed_error_prediction_minus_true": float(error),
                "absolute_error": float(abs(error)),
                "nasa_contribution": float(contribution),
                "overprediction": int(error > 0),
            }
        )
    audit.update(
        {
            "evaluation_scope": "truncated_endpoint_validation",
            "outer_engine_count": result["outer_engine_count"],
            "endpoint_sample_count": result["endpoint_sample_count"],
            "official_test_files_accessed": False,
        }
    )
    return result, audit, prediction_rows


def result_paths(args) -> dict[str, Path]:
    output = exp18.resolve_path(args.output_dir)
    prefix = f"experiment24_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired_cell": output / f"{prefix}_paired_by_cell.csv",
        "paired_split": output / f"{prefix}_paired_by_target_split.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "dense_summary": output / f"{prefix}_dense_validation_summary.csv",
        "dense_paired_cell": output / f"{prefix}_dense_validation_paired_by_cell.csv",
        "dense_comparisons": output / f"{prefix}_dense_validation_comparisons.csv",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_splits": output / f"{prefix}_engine_splits.csv",
        "prior": output / f"{prefix}_prior_adjacency.csv",
        "correlation": output / f"{prefix}_source_sensor_correlation.csv",
        "budget": output / f"{prefix}_budget.json",
        "inventory": output / f"{prefix}_model_inventory.csv",
        "source_diagnostics": output / f"{prefix}_source_diagnostics.json",
        "context_audit": output / f"{prefix}_context_audit.csv",
        "endpoint_predictions": output / f"{prefix}_endpoint_predictions.csv",
        "endpoint_by_rul": output / f"{prefix}_endpoint_by_rul.csv",
        "tail_audit": output / f"{prefix}_tail_audit.csv",
        "dry_run": output / f"{prefix}_dry_run.csv",
    }


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    exp18.atomic_write_text(path, frame.to_csv(index=False), encoding="utf-8-sig")


def endpoint_by_rul(predictions: list[dict]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    frame = pd.DataFrame(predictions)
    rows = []
    keys = ["target", "k", "target_split_seed", "model_seed", "model", "rul_anchor"]
    for key, group in frame.groupby(keys):
        metrics = regression_metrics(group["true_rul"], group["predicted_rul"])
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_engines": int(group["outer_unit"].nunique()),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "nasa_score": metrics["nasa_score"],
                "signed_error_mean": float(
                    group["signed_error_prediction_minus_true"].mean()
                ),
                "overprediction_rate": float(group["overprediction"].mean()),
                "worst20_abs_error_mean": float(
                    group.nlargest(max(1, math.ceil(len(group) * 0.2)), "absolute_error")[
                        "absolute_error"
                    ].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def comparison_summary(results: list[dict], repetitions: int) -> pd.DataFrame:
    paired = exp18.paired_cells(results)
    output = exp18.comparison_summary(results, paired, repetitions)
    return output.rename(columns={"strict_success": "endpoint_model_strict_success"})


def save_progress(
    args,
    paths,
    results: list[dict],
    context_rows: list[dict],
    predictions: list[dict],
    *,
    full_tables: bool,
) -> None:
    exp18.atomic_write_text(
        paths["raw"], json.dumps(results, ensure_ascii=False, indent=2)
    )
    context = pd.DataFrame(context_rows)
    if not context.empty:
        context = context.drop_duplicates(
            subset=["target_split_seed", "model_seed", "k", "regime"], keep="last"
        )
        atomic_csv(paths["context_audit"], context)
    prediction_frame = pd.DataFrame(predictions)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.drop_duplicates(
            subset=[
                "target_split_seed",
                "model_seed",
                "k",
                "model",
                "outer_unit",
                "rul_anchor",
            ],
            keep="last",
        )
        atomic_csv(paths["endpoint_predictions"], prediction_frame)
    if not full_tables or not results:
        return
    summary = exp18.summarize(results)
    paired = exp18.paired_cells(results)
    split = exp18.paired_by_target_split(paired)
    comparisons = comparison_summary(results, args.bootstrap_repetitions)
    dense_results = [
        {
            **row,
            **{metric: row[f"validation_{metric}"] for metric in exp18.METRICS},
            "evaluation_scope": "dense_inner_validation",
        }
        for row in results
    ]
    dense_summary = exp18.summarize(dense_results)
    dense_paired = exp18.paired_cells(dense_results)
    dense_comparisons = comparison_summary(
        dense_results, args.bootstrap_repetitions
    ).rename(
        columns={"endpoint_model_strict_success": "dense_model_strict_success"}
    )
    tail_columns = [
        "target_domain",
        "target_split_seed",
        "model_seed",
        "k",
        "model",
        "outer_engine_count",
        "endpoint_sample_count",
        "endpoint_signed_error_mean",
        "endpoint_overprediction_rate",
        "endpoint_underprediction_rate",
        "endpoint_worst20_abs_error_mean",
        "endpoint_worst20_nasa_contribution_mean",
    ]
    atomic_csv(paths["summary"], summary)
    atomic_csv(paths["paired_cell"], paired)
    atomic_csv(paths["paired_split"], split)
    atomic_csv(paths["comparisons"], comparisons)
    atomic_csv(paths["dense_summary"], dense_summary)
    atomic_csv(paths["dense_paired_cell"], dense_paired)
    atomic_csv(paths["dense_comparisons"], dense_comparisons)
    atomic_csv(paths["endpoint_by_rul"], endpoint_by_rul(predictions))
    atomic_csv(paths["tail_audit"], pd.DataFrame(results)[tail_columns])


def key_from_result(row: dict) -> tuple[int, int, int, str]:
    return (
        int(row["target_split_seed"]),
        int(row["model_seed"]),
        int(row["k"]),
        str(row["model"]),
    )


def reconcile_resume(results: list[dict], predictions: list[dict]):
    result_by_key = {key_from_result(row): row for row in results}
    prediction_by_key = {
        (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            int(row["k"]),
            str(row["model"]),
            int(row["outer_unit"]),
            int(row["rul_anchor"]),
        ): row
        for row in predictions
    }
    results = list(result_by_key.values())
    predictions = list(prediction_by_key.values())
    counts = Counter(
        (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            int(row["k"]),
            str(row["model"]),
        )
        for row in predictions
    )
    kept = [
        row
        for row in results
        if counts[key_from_result(row)] == int(row.get("endpoint_sample_count", -1))
    ]
    valid_keys = {key_from_result(row) for row in kept}
    kept_predictions = [
        row
        for row in predictions
        if (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            int(row["k"]),
            str(row["model"]),
        )
        in valid_keys
    ]
    if len(kept) != len(results):
        print(f"[Experiment 24 resume repair] rerun {len(results) - len(kept)} incomplete cells")
    return kept, kept_predictions


def dry_run(args, cfg, protocol, prior, paths) -> None:
    global _active_split_seed
    split_seed = args.target_split_seeds[0]
    local = dict(protocol)
    local["validation_units"] = protocol["validation_units_by_target_split_seed"][
        str(split_seed)
    ]
    adaptation = protocol["nested_adaptation_units_by_target_split_seed"][
        str(split_seed)
    ]["5"]
    _active_split_seed = split_seed
    try:
        _, support, validation, endpoints, feature_count, split = prepare_train_only(
            cfg,
            args.preprocessing,
            args.balance_mode,
            local["validation_units"],
            adaptation,
        )
        source_summary = prepare_source_summary(args, cfg, local, split_seed, 5)
    finally:
        _active_split_seed = None

    x, _ = next(iter(endpoints))
    static = exp18.build_static_model(feature_count, cfg, prior)
    exp21._active_gate_scale = 2.0
    try:
        tcsg = exp21.build_tcsg_model(feature_count, cfg, prior, args)
    finally:
        exp21._active_gate_scale = 1.0
    with torch.no_grad():
        static_output = static(x)
        tcsg_output = tcsg(x, source_summary)
    rows = [
        {
            "target": args.target,
            "target_split_seed": split_seed,
            "model": "static_budget_prior",
            "feature_count": feature_count,
            "support_engines": len(set(support.dataset.units)),
            "inner_validation_engines": len(set(validation.dataset.units)),
            "outer_engines": split["outer_engine_count"],
            "endpoint_rows": len(endpoints.dataset),
            "endpoint_labels": sorted(set(map(float, endpoints.dataset.y.tolist()))),
            "forward_shape": list(static_output.shape),
            "official_test_files_accessed": False,
        },
        {
            "target": args.target,
            "target_split_seed": split_seed,
            "model": "tcsg_fixed_source_gate2",
            "feature_count": feature_count,
            "support_engines": len(set(support.dataset.units)),
            "inner_validation_engines": len(set(validation.dataset.units)),
            "outer_engines": split["outer_engine_count"],
            "endpoint_rows": len(endpoints.dataset),
            "endpoint_labels": sorted(set(map(float, endpoints.dataset.y.tolist()))),
            "forward_shape": list(tcsg_output.shape),
            "official_test_files_accessed": False,
        },
    ]
    atomic_csv(paths["dry_run"], pd.DataFrame(rows))
    print("\n[Experiment 24 dry run]")
    print(pd.DataFrame(rows).to_string(index=False))
    print("No model training and no official-test file access occurred.")


def configure_reused_runner() -> None:
    exp18.SCRIPT_VERSION = SCRIPT_VERSION
    exp18.MODEL_CHOICES = MODELS
    exp18.DEFAULT_MODELS = list(MODELS)
    exp18.COMPARISONS = COMPARISONS
    exp18.PRIMARY_COMPARISONS = PRIMARY_COMPARISONS
    exp18.prepare_kshot_experiment = prepare_train_only
    exp18.build_tcsg_model = exp21.build_tcsg_model
    exp18.target_run_seed = experiment24_run_seed


def main() -> None:
    configure_reused_runner()
    args = parse_args()
    args.models = list(dict.fromkeys(args.models))
    args.k_values = sorted(set(map(int, args.k_values)))
    args.target_split_seeds = list(dict.fromkeys(map(int, args.target_split_seeds)))
    args.model_seeds = list(dict.fromkeys(map(int, args.model_seeds)))
    if not args.dry_run:
        if set(args.models) != set(MODELS):
            raise ValueError(f"Formal Experiment 24 requires both models: {MODELS}")
        if len(args.target_split_seeds) < 5 or len(args.model_seeds) < 5:
            raise ValueError("Formal Experiment 24 requires at least 5x5 crossed seeds")

    cfg0 = exp18.load_config(args, args.model_seeds[0])
    protocol = create_protocol(args, cfg0)
    prior, correlation = train_only_prior(
        cfg0, args.preprocessing, args.sensor_graph_k
    )
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    sensors = list(cfg0["sensor_columns"])
    exp18.atomic_write_text(
        paths["protocol"], json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    atomic_csv(paths["engine_splits"], protocol_rows(protocol))
    atomic_csv(
        paths["prior"],
        pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors)
        .rename_axis("sensor")
        .reset_index(),
    )
    atomic_csv(
        paths["correlation"],
        pd.DataFrame(correlation, index=sensors, columns=sensors)
        .rename_axis("sensor")
        .reset_index(),
    )
    budget = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "models": args.models,
        "k_values": args.k_values,
        "target_split_seeds": args.target_split_seeds,
        "model_seeds": args.model_seeds,
        "planned_target_cells": len(args.models)
        * len(args.k_values)
        * len(args.target_split_seeds)
        * len(args.model_seeds),
        "source_pretrain_steps": args.source_pretrain_steps,
        "context_meta_or_static_extra_steps": args.context_meta_steps,
        "target_epochs": args.target_epochs,
        "target_learning_rate": args.target_lr,
        "target_adaptation_scope": "predictor_only",
        "outer_fraction": OUTER_FRACTION,
        "inner_validation_engines": args.validation_units,
        "rul_anchors": list(RUL_ANCHORS),
        "normalizer_fit_scope": "source_train_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "primary_question": (
            "Does train-only truncated-endpoint validation expose robustness "
            "and tail-risk differences hidden by dense-window validation?"
        ),
        "primary_comparison": "fixed_vs_static_budget",
    }
    exp18.atomic_write_text(
        paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2)
    )
    print("\n[Experiment 24 locked protocol and budget]")
    print(json.dumps(budget, ensure_ascii=False, indent=2))

    if args.dry_run:
        dry_run(args, cfg0, protocol, prior, paths)
        return

    results: list[dict] = []
    predictions: list[dict] = []
    context_rows: list[dict] = []
    source_diagnostics = {}
    inventories: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
    if args.resume and paths["endpoint_predictions"].is_file():
        predictions = pd.read_csv(paths["endpoint_predictions"]).to_dict("records")
    results, predictions = reconcile_resume(results, predictions)
    if args.resume and paths["context_audit"].is_file():
        context_rows = pd.read_csv(paths["context_audit"]).to_dict("records")
    if args.resume and paths["source_diagnostics"].is_file():
        source_diagnostics = json.loads(
            paths["source_diagnostics"].read_text(encoding="utf-8")
        )
    if args.resume and paths["inventory"].is_file():
        inventories = pd.read_csv(paths["inventory"]).to_dict("records")
    done = {key_from_result(row) for row in results}
    print(f"[Experiment 24 resume] {len(done)} complete cells")

    for model_seed in args.model_seeds:
        expected_for_seed = {
            (split_seed, model_seed, 5, model)
            for split_seed in args.target_split_seeds
            for model in args.models
        }
        if expected_for_seed <= done:
            print(f"[Experiment 24 skip model seed] {model_seed}")
            continue
        cfg = exp18.load_config(args, model_seed)
        print(f"\n[Experiment 24 source initialization] model_seed={model_seed}")
        states, histories, inventory = load_or_train_source_bundle(
            args, cfg, protocol, prior
        )
        source_diagnostics[str(model_seed)] = histories
        inventories.append(inventory)
        exp18.atomic_write_text(
            paths["source_diagnostics"],
            json.dumps(source_diagnostics, ensure_ascii=False, indent=2),
        )
        inventory_frame = pd.DataFrame(inventories).drop_duplicates(
            subset=["model_seed"], keep="last"
        )
        atomic_csv(paths["inventory"], inventory_frame)

        for split_seed in args.target_split_seeds:
            for regime in args.models:
                key = (split_seed, model_seed, 5, regime)
                if key in done:
                    print(
                        f"[Experiment 24 skip] split={split_seed} "
                        f"model_seed={model_seed} model={regime}"
                    )
                    continue
                print(
                    f"\n[Experiment 24] target={args.target} split={split_seed} "
                    f"model_seed={model_seed} K=5 model={regime}"
                )
                source_key = "static_budget_prior" if regime == "static_budget_prior" else "tcsg"
                result, audit, rows = run_target_cell(
                    args,
                    cfg,
                    protocol,
                    regime,
                    states[source_key],
                    inventory,
                    split_seed,
                    5,
                    prior,
                )
                results.append(result)
                context_rows.append(audit)
                predictions.extend(rows)
                done.add(key)
                save_progress(
                    args,
                    paths,
                    results,
                    context_rows,
                    predictions,
                    full_tables=False,
                )
        save_progress(
            args,
            paths,
            results,
            context_rows,
            predictions,
            full_tables=True,
        )
        states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_progress(
        args,
        paths,
        results,
        context_rows,
        predictions,
        full_tables=True,
    )
    expected = len(args.models) * len(args.target_split_seeds) * len(args.model_seeds)
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} completed cells, got {len(results)}")
    print("\n[Experiment 24 complete]")
    print(exp18.summarize(results).to_string(index=False))
    print("\n[Experiment 24 primary comparison]")
    print(comparison_summary(results, args.bootstrap_repetitions).to_string(index=False))
    print("Official test files were not accessed. Do not tune against Experiment 23 test results.")
    for name, path in paths.items():
        if name != "output" and path.exists():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
