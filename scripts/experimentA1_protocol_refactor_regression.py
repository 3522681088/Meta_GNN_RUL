"""Experiment A1: train-only refactor and Experiment-17B regression.

This is the first script in the new ``experimentA`` series.  It retains the
five controlled Experiment-17B regimes while enforcing a stronger boundary:
validation runs load only ``train_FDxxx.txt`` files.  Official test trajectories
and ``RUL_FDxxx.txt`` labels are neither opened nor materialized.

All artifacts are written below
``outputs/experimentA1_protocol_refactor_regression`` by default.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from preprocess.rul_generator import add_train_rul  # noqa: E402
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    make_loader,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    train_source_supervised,
)
from scripts.run_condition_aware_experiment import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
    SETTING_FEATURE_COLUMNS,
    SourceConditionNormalizer,
    SourceGlobalNormalizer,
)


SCRIPT_VERSION = "experimentA1_protocol_refactor_regression_v1"
MODEL_CHOICES = exp17b.MODEL_CHOICES
CMAPSS_COLUMNS = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    *[f"s{index}" for index in range(1, 22)],
]
DEFAULT_BASE_CONFIG = {
    "seed": 42,
    "data_dir": "data",
    "source_domains": ["FD001", "FD002", "FD003"],
    "target_domain": "FD004",
    "window_size": 50,
    "window_stride": 5,
    "rul_cap": 125,
    "sensor_columns": [
        "s2",
        "s3",
        "s4",
        "s7",
        "s8",
        "s9",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s17",
        "s20",
        "s21",
    ],
    "normalization": "zscore",
    "hidden_dim": 128,
    "embedding_dim": 256,
    "gat_heads": 4,
    "self_attention_heads": 4,
    "dropout": 0.2,
    "graph_k": 5,
    "graph_method": "cosine",
    "dtw_downsample": 5,
    "batch_size": 64,
    "device": "auto",
}
DEFAULT_EXPERIMENT = {
    "experiment_id": "experimentA1",
    "experiment_name": "protocol_refactor_regression",
    "target_domain": "FD004",
    "preprocessing": "condition_settings",
    "balance_mode": "engine_stage",
    "validation_units": 20,
    "validation_seed": 2026,
    "normalizer_seed": 2026,
    "condition_count": 6,
    "sensor_graph_k": 4,
    "k_values": [2, 5],
    "target_split_seeds": [3027, 3028, 3029, 3030, 3031],
    "model_seeds": [42, 43, 44, 45, 46],
    "random_graph_seeds": [3017, 3018, 3019, 3020, 3021],
    "models": list(MODEL_CHOICES),
    "source_pretrain_steps": 1500,
    "source_pretrain_lr": 0.001,
    "source_pretrain_weight_decay": 0.0,
    "target_epochs": 10,
    "target_lr": 0.001,
    "bootstrap_repetitions": 10000,
    "regression_rmse_tolerance": 0.25,
    "reference_validation_rmse": {
        "sensor_graph_prior": {2: 16.694, 5: 16.055},
        "window_graph": {2: 18.796, 5: 17.751},
    },
    "output_dir": "outputs/experimentA1_protocol_refactor_regression",
}


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def canonical_hash(payload: Any, length: int = 20) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def train_path(data_dir: str, domain: str) -> Path:
    root = Path(data_dir)
    nested = root / domain / f"train_{domain}.txt"
    flat = root / f"train_{domain}.txt"
    return nested if nested.is_file() else flat


def load_train_domain(data_dir: str, domain: str) -> pd.DataFrame:
    """Load one training split without touching official test files."""
    path = train_path(data_dir, domain)
    if not path.is_file():
        raise FileNotFoundError(f"missing C-MAPSS training file: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < len(CMAPSS_COLUMNS):
        raise ValueError(f"{path} has fewer than {len(CMAPSS_COLUMNS)} columns")
    frame = frame.iloc[:, : len(CMAPSS_COLUMNS)]
    frame.columns = CMAPSS_COLUMNS
    return frame


def build_crossed_train_only_protocol(
    *,
    data_dir: str,
    target_domain: str,
    source_domains: list[str],
    validation_count: int,
    validation_seed: int,
    target_split_seeds: list[int],
    model_seeds: list[int],
    k_values: list[int],
) -> dict:
    target_train = load_train_domain(data_dir, target_domain)
    train_units = np.asarray(sorted(target_train["unit"].unique()), dtype=int)
    if not 1 <= validation_count < len(train_units):
        raise ValueError("validation engine count is outside the valid range")
    validation_order = np.random.default_rng(validation_seed).permutation(train_units)
    validation_units = validation_order[:validation_count]
    validation_set = set(validation_units.tolist())
    candidates = np.asarray(
        [unit for unit in train_units if int(unit) not in validation_set],
        dtype=int,
    )
    if max(k_values) > len(candidates):
        raise ValueError("maximum K exceeds available adaptation engines")

    nested: dict[str, dict[str, list[int]]] = {}
    orders: dict[str, list[int]] = {}
    for split_seed in target_split_seeds:
        order = np.random.default_rng(split_seed).permutation(candidates)
        orders[str(split_seed)] = order.astype(int).tolist()
        nested[str(split_seed)] = {
            str(k): order[:k].astype(int).tolist() for k in k_values
        }
        previous: set[int] = set()
        for k in k_values:
            current = set(nested[str(split_seed)][str(k)])
            if len(current) != k:
                raise AssertionError("adaptation engine count does not match K")
            if not previous.issubset(current):
                raise AssertionError("K-shot engine sets are not nested")
            if current & validation_set:
                raise AssertionError("adaptation and validation engines overlap")
            previous = current

    train_file_hashes = {
        domain: file_sha256(train_path(data_dir, domain))
        for domain in [*source_domains, target_domain]
    }
    payload = {
        "protocol_version": "experiment_a_engine_kshot_v1",
        "target_domain": target_domain,
        "source_domains": list(source_domains),
        "train_engine_count": int(len(train_units)),
        "validation_seed": int(validation_seed),
        "validation_units": validation_units.astype(int).tolist(),
        "candidate_adaptation_engine_count": int(len(candidates)),
        "target_split_seeds": [int(value) for value in target_split_seeds],
        "model_seeds": [int(value) for value in model_seeds],
        "k_values": [int(value) for value in k_values],
        "adaptation_order_by_target_split_seed": orders,
        "nested_adaptation_units_by_target_split_seed": nested,
        "train_file_hashes": train_file_hashes,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    payload["protocol_hash"] = canonical_hash(payload)
    return payload


def fit_source_normalizer_train_only(cfg: dict, preprocessing: str):
    sensors = list(cfg["sensor_columns"])
    source_frames = {
        domain: add_train_rul(
            load_train_domain(cfg["data_dir"], domain),
            cfg["rul_cap"],
        )
        for domain in cfg["source_domains"]
    }
    source_fit = pd.concat(source_frames.values(), ignore_index=True)
    include_settings = preprocessing in {"global_settings", "condition_settings"}
    if preprocessing in {"condition_norm", "condition_settings"}:
        normalizer = SourceConditionNormalizer(
            n_conditions=cfg.get("condition_count", 6),
            seed=cfg.get("normalizer_seed", 2026),
            include_settings=include_settings,
        ).fit(source_fit, sensors)
    else:
        normalizer = SourceGlobalNormalizer(
            include_settings=include_settings
        ).fit(source_fit, sensors)
    return source_frames, normalizer


def source_correlation_adjacency_train_only(
    cfg: dict,
    preprocessing: str,
    neighbors: int,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    sensors = list(cfg["sensor_columns"])
    sensor_count = len(sensors)
    if not 1 <= neighbors < sensor_count:
        raise ValueError("sensor graph K is outside the valid range")
    source_frames, normalizer = fit_source_normalizer_train_only(cfg, preprocessing)
    normalized = [
        normalizer.transform(frame, sensors) for frame in source_frames.values()
    ]
    values = pd.concat(normalized, ignore_index=True)[sensors].to_numpy(np.float64)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(
        np.abs(correlation),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    adjacency = np.eye(sensor_count, dtype=bool)
    for sensor in range(sensor_count):
        scores = correlation[sensor].copy()
        scores[sensor] = -np.inf
        adjacency[sensor, np.argsort(scores)[-neighbors:]] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    return (
        torch.as_tensor(adjacency),
        correlation,
        {
            "fit_scope": "source_train_only",
            "source_domains": list(cfg["source_domains"]),
            "source_rows": {
                domain: int(len(frame)) for domain, frame in source_frames.items()
            },
            "official_test_files_accessed": False,
        },
    )


def prepare_validation_experiment(
    cfg: dict,
    preprocessing: str,
    balance_mode: str,
    validation_units: list[int],
    adaptation_units: list[int],
):
    sensors = list(cfg["sensor_columns"])
    source_frames, normalizer = fit_source_normalizer_train_only(cfg, preprocessing)
    target_train = add_train_rul(
        load_train_domain(cfg["data_dir"], cfg["target_domain"]),
        cfg["rul_cap"],
    )
    include_settings = preprocessing in {"global_settings", "condition_settings"}
    features = sensors + SETTING_FEATURE_COLUMNS if include_settings else sensors
    normalized_sources = {
        domain: normalizer.transform(frame, sensors)
        for domain, frame in source_frames.items()
    }
    normalized_target = normalizer.transform(target_train, sensors)
    source_tasks = {
        domain: make_loader(
            normalized_sources[domain],
            features,
            cfg,
            training=True,
            balance_mode=balance_mode,
            loader_seed=cfg["seed"] + 1000 * (index + 1),
        )
        for index, domain in enumerate(cfg["source_domains"])
    }
    adaptation = np.asarray(adaptation_units, dtype=int)
    validation = np.asarray(validation_units, dtype=int)
    if set(adaptation.tolist()) & set(validation.tolist()):
        raise AssertionError("adaptation and validation engines overlap")
    support_frame = normalized_target.query("unit in @adaptation")
    validation_frame = normalized_target.query("unit in @validation")
    if support_frame["unit"].nunique() != len(adaptation_units):
        raise ValueError("support engine count does not match K")
    if validation_frame["unit"].nunique() != len(validation_units):
        raise ValueError("validation engine count is inconsistent")
    support = make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        loader_seed=cfg["seed"] + 9000,
    )
    validation_loader = make_loader(
        validation_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9100,
    )
    split = {
        "protocol": "experiment_a_train_only_validation",
        "target_domain": cfg["target_domain"],
        "adaptation_units": [int(value) for value in adaptation_units],
        "validation_units": [int(value) for value in validation_units],
        "feature_columns": list(features),
        "normalizer_fit_scope": "source_train_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    return source_tasks, support, validation_loader, len(features), split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A1: train-only refactor and 17B regression"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--target", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES))
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one-seed smoke training; writes to a separate outputs directory",
    )
    return parser.parse_args()


def _override(value, fallback):
    return fallback if value is None else value


def load_experiment_config(args: argparse.Namespace) -> tuple[dict, dict]:
    experiment = deepcopy(DEFAULT_EXPERIMENT)
    base = deepcopy(DEFAULT_BASE_CONFIG)
    target = _override(args.target, experiment["target_domain"])
    base["target_domain"] = target
    base["source_domains"] = [
        domain for domain in EXPECTED_OFFICIAL_TEST_ENGINES if domain != target
    ]
    base["data_dir"] = str(
        resolve_path(_override(args.data_dir, base["data_dir"]))
    )
    base["output_dir"] = str(
        resolve_path(_override(args.output_dir, experiment["output_dir"]))
    )
    base["normalizer_seed"] = int(experiment["normalizer_seed"])
    base["condition_count"] = int(experiment["condition_count"])
    base["source_pretrain_steps"] = int(experiment["source_pretrain_steps"])
    base["source_pretrain_lr"] = float(experiment["source_pretrain_lr"])
    base["source_pretrain_weight_decay"] = float(
        experiment["source_pretrain_weight_decay"]
    )
    base["target_epochs"] = int(experiment["target_epochs"])
    base["target_lr"] = float(experiment["target_lr"])
    base["pair_aux_weight"] = 0.0
    if args.device is not None:
        base["device"] = args.device

    experiment["target_domain"] = target
    if args.quick:
        experiment["quick_mode"] = True
        experiment["models"] = ["window_graph", "sensor_graph_prior"]
        experiment["k_values"] = [2]
        experiment["target_split_seeds"] = [3027]
        experiment["model_seeds"] = [42]
        experiment["random_graph_seeds"] = [3017]
        experiment["source_pretrain_steps"] = 5
        experiment["target_epochs"] = 1
        experiment["bootstrap_repetitions"] = 100
        if args.output_dir is None:
            base["output_dir"] = str(
                resolve_path(
                    "outputs/experimentA1_protocol_refactor_regression_quick"
                )
            )
        base["source_pretrain_steps"] = 5
        base["target_epochs"] = 1
    experiment["output_dir"] = base["output_dir"]
    return base, experiment


def validate_configuration(base: dict, experiment: dict) -> None:
    if len(experiment["model_seeds"]) != len(experiment["random_graph_seeds"]):
        raise ValueError("random graph seeds must match model seeds one-to-one")
    if len(set(experiment["models"])) != len(experiment["models"]):
        raise ValueError("models contains duplicates")
    if len(set(experiment["k_values"])) != len(experiment["k_values"]):
        raise ValueError("k_values contains duplicates")
    if experiment["preprocessing"] not in PREPROCESSING_MODES:
        raise ValueError("unknown preprocessing mode")
    if experiment["balance_mode"] not in BALANCE_MODES:
        raise ValueError("unknown balance mode")
    if not 1 <= int(experiment["sensor_graph_k"]) < len(base["sensor_columns"]):
        raise ValueError("sensor_graph_k is outside the valid range")
    if base["target_domain"] != experiment["target_domain"]:
        raise AssertionError("target-domain configuration mismatch")


def result_paths(output: Path) -> dict[str, Path]:
    return {
        "manifest": output / "experimentA1_manifest.json",
        "protocol": output / "experimentA1_protocol.json",
        "engine_splits": output / "experimentA1_engine_splits.csv",
        "prior_adjacency": output / "experimentA1_prior_adjacency.csv",
        "prior_correlation": output / "experimentA1_prior_correlation.csv",
        "random_graph_audit": output / "experimentA1_random_graph_audit.csv",
        "raw": output / "experimentA1_run_level.json",
        "run_csv": output / "experimentA1_run_level.csv",
        "window_predictions": output / "experimentA1_window_predictions.csv",
        "per_engine": output / "experimentA1_per_engine_metrics.csv",
        "summary": output / "experimentA1_summary.csv",
        "paired_cell": output / "experimentA1_paired_cells.csv",
        "paired_split": output / "experimentA1_paired_target_splits.csv",
        "comparisons": output / "experimentA1_paired_comparisons.csv",
        "inventory": output / "experimentA1_source_inventory.csv",
        "decision": output / "experimentA1_regression_decision.json",
        "dry_run": output / "experimentA1_dry_run.json",
    }


def protocol_frame(protocol: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for unit in protocol["validation_units"]:
        rows.append(
            {
                "target_split_seed": "fixed",
                "k": "all",
                "role": "validation",
                "unit": int(unit),
            }
        )
    for split_seed, by_k in protocol[
        "nested_adaptation_units_by_target_split_seed"
    ].items():
        for k, units in by_k.items():
            for unit in units:
                rows.append(
                    {
                        "target_split_seed": int(split_seed),
                        "k": int(k),
                        "role": "adaptation",
                        "unit": int(unit),
                    }
                )
    return pd.DataFrame(rows)


def target_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = f"{model_seed}:{target_split_seed}:experiment17b".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def source_signature(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    feature_count: int,
    prior: torch.Tensor,
    randomized: torch.Tensor,
    commit: str,
) -> str:
    return canonical_hash(
        {
            "script_version": SCRIPT_VERSION,
            "git_commit": commit,
            "model": model_name,
            "model_seed": model_seed,
            "target": base["target_domain"],
            "source_domains": base["source_domains"],
            "feature_count": feature_count,
            "embedding_dim": base["embedding_dim"],
            "gat_heads": base["gat_heads"],
            "dropout": base["dropout"],
            "preprocessing": experiment["preprocessing"],
            "balance_mode": experiment["balance_mode"],
            "source_pretrain_steps": base["source_pretrain_steps"],
            "source_pretrain_lr": base["source_pretrain_lr"],
            "source_pretrain_weight_decay": base[
                "source_pretrain_weight_decay"
            ],
            "train_file_hashes": protocol["train_file_hashes"],
            "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
            "random_hash": hashlib.sha256(randomized.numpy().tobytes()).hexdigest(),
        }
    )


def state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def load_or_train_source(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    prior: torch.Tensor,
    randomized: torch.Tensor,
    commit: str,
    resume: bool,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    first_split = experiment["target_split_seeds"][0]
    first_k = min(experiment["k_values"])
    units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(first_split)
    ][str(first_k)]
    cfg = dict(base)
    cfg["seed"] = int(model_seed)
    source_tasks, _, _, feature_count, _ = prepare_validation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        protocol["validation_units"],
        units,
    )
    seed_everything(model_seed)
    model = exp17b.build_model_17b(
        model_name,
        feature_count,
        cfg,
        prior,
        randomized,
    )
    total, predictor = exp17.parameter_count(model)
    signature = source_signature(
        base=cfg,
        experiment=experiment,
        protocol=protocol,
        model_name=model_name,
        model_seed=model_seed,
        feature_count=feature_count,
        prior=prior,
        randomized=randomized,
        commit=commit,
    )
    cache_path = (
        Path(base["output_dir"])
        / "source_cache"
        / f"experimentA1_{model_name}_{base['target_domain']}_mseed{model_seed}.pt"
    )
    if resume and cache_path.is_file():
        cached = safe_torch_load(cache_path)
        if cached.get("signature") == signature:
            return cached["state"], cached.get("history", []), cached["inventory"]

    device = resolve_device(base["device"])
    model, history = train_source_supervised(model, source_tasks, cfg, device)
    inventory = {
        "model": model_name,
        "model_seed": int(model_seed),
        "feature_count": int(feature_count),
        "total_parameter_count": int(total),
        "predictor_parameter_count": int(predictor),
        "source_pretrain_steps": int(base["source_pretrain_steps"]),
        "source_signature": signature,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "state": state_to_cpu(model),
            "history": history,
            "inventory": inventory,
        },
        cache_path,
    )
    return state_to_cpu(model), history, inventory


def predict_with_units(model, loader, device) -> pd.DataFrame:
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            labels.extend(y.cpu().numpy().tolist())
            predictions.extend(output.detach().cpu().numpy().tolist())
    units = np.asarray(loader.dataset.units, dtype=int)
    if len(units) != len(labels):
        raise AssertionError("prediction count does not match unit metadata")
    frame = pd.DataFrame(
        {
            "unit": units,
            "label": np.asarray(labels, dtype=float),
            "prediction": np.asarray(predictions, dtype=float),
        }
    )
    frame["error"] = frame["prediction"] - frame["label"]
    frame["nasa_contribution"] = np.where(
        frame["error"] < 0,
        np.exp(-frame["error"] / 13.0) - 1.0,
        np.exp(frame["error"] / 10.0) - 1.0,
    )
    return frame


def per_engine_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for unit, group in predictions.groupby("unit"):
        metrics = regression_metrics(group["label"], group["prediction"])
        rows.append(
            {
                "unit": int(unit),
                "window_count": int(len(group)),
                **metrics,
                "late_rate": float((group["error"] > 0).mean()),
                "positive_error_q95": float(
                    np.quantile(np.maximum(group["error"], 0.0), 0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_target_cell(
    *,
    base: dict,
    experiment: dict,
    protocol: dict,
    model_name: str,
    model_seed: int,
    target_split_seed: int,
    k: int,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    prior: torch.Tensor,
    randomized: torch.Tensor,
    save_checkpoint: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(target_split_seed)
    ][str(k)]
    run_seed = target_run_seed(model_seed, target_split_seed)
    cfg = dict(base)
    cfg["seed"] = run_seed
    _, support, validation, feature_count, split = prepare_validation_experiment(
        cfg,
        experiment["preprocessing"],
        experiment["balance_mode"],
        protocol["validation_units"],
        units,
    )
    seed_everything(run_seed)
    model = exp17b.build_model_17b(
        model_name,
        feature_count,
        base,
        prior,
        randomized,
    )
    model.load_state_dict(source_state)
    device = resolve_device(base["device"])
    model, target_history, best_epoch = exp17.train_target_head(
        model,
        support,
        validation,
        cfg,
        device,
    )
    window_predictions = predict_with_units(model, validation, device)
    metrics = regression_metrics(
        window_predictions["label"],
        window_predictions["prediction"],
    )
    replicate_id = (
        f"experimentA1_{base['target_domain'].lower()}_k{k:02d}_"
        f"tsplit{target_split_seed}_mseed{model_seed}_{model_name}"
    )
    result = {
        **metrics,
        "experiment_id": "experimentA1",
        "experiment_name": "protocol_refactor_regression",
        "replicate_id": replicate_id,
        "evaluation_scope": "train_only_validation",
        "model": model_name,
        "target_domain": base["target_domain"],
        "k": int(k),
        "target_split_seed": int(target_split_seed),
        "model_seed": int(model_seed),
        "target_run_seed": int(run_seed),
        "adaptation_units": [int(value) for value in units],
        "validation_units": [
            int(value) for value in protocol["validation_units"]
        ],
        "best_target_epoch_by_validation": int(best_epoch),
        "source_pretrain_steps": int(base["source_pretrain_steps"]),
        "target_epochs_planned": int(base["target_epochs"]),
        "preprocessing_mode": experiment["preprocessing"],
        "balance_mode": experiment["balance_mode"],
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_parameter_count": inventory[
            "predictor_parameter_count"
        ],
        "source_signature": inventory["source_signature"],
        "source_history_rows": int(len(source_history)),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "split_protocol": split["protocol"],
    }
    window_predictions.insert(0, "replicate_id", replicate_id)
    engine = per_engine_metrics(window_predictions)
    engine.insert(0, "replicate_id", replicate_id)

    if save_checkpoint:
        path = (
            Path(base["output_dir"])
            / "checkpoints"
            / f"{replicate_id}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": state_to_cpu(model),
                "metrics": result,
                "target_history": target_history,
                "split": split,
            },
            path,
        )
    return result, window_predictions, engine


def save_progress(
    *,
    paths: dict[str, Path],
    results: list[dict],
    predictions: list[pd.DataFrame],
    engine_metrics: list[pd.DataFrame],
    inventory: list[dict],
    bootstrap_repetitions: int,
) -> None:
    atomic_write_text(
        paths["raw"],
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
    )
    run_frame = pd.DataFrame(results)
    atomic_write_text(paths["run_csv"], run_frame.to_csv(index=False))
    atomic_write_text(
        paths["window_predictions"],
        (
            pd.concat(predictions, ignore_index=True).to_csv(index=False)
            if predictions
            else ""
        ),
    )
    atomic_write_text(
        paths["per_engine"],
        (
            pd.concat(engine_metrics, ignore_index=True).to_csv(index=False)
            if engine_metrics
            else ""
        ),
    )
    atomic_write_text(paths["inventory"], pd.DataFrame(inventory).to_csv(index=False))
    if not results:
        return
    summary = exp17b.summarize(results)
    paired = exp17b.paired_cells(results)
    split = exp17b.paired_by_target_split(paired)
    comparisons = exp17b.comparison_summary(
        results,
        paired,
        bootstrap_repetitions,
    )
    atomic_write_text(paths["summary"], summary.to_csv(index=False))
    atomic_write_text(paths["paired_cell"], paired.to_csv(index=False))
    atomic_write_text(paths["paired_split"], split.to_csv(index=False))
    atomic_write_text(paths["comparisons"], comparisons.to_csv(index=False))


def regression_decision(
    *,
    experiment: dict,
    results: list[dict],
) -> dict:
    expected = (
        len(experiment["models"])
        * len(experiment["k_values"])
        * len(experiment["target_split_seeds"])
        * len(experiment["model_seeds"])
    )
    decision = {
        "experiment_id": "experimentA1",
        "expected_cells": int(expected),
        "completed_cells": int(len(results)),
        "complete": len(results) == expected,
        "rmse_tolerance": float(experiment["regression_rmse_tolerance"]),
        "reference_checks": [],
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "quick_mode": bool(experiment.get("quick_mode", False)),
    }
    if not results:
        decision["passed"] = False
        decision["reason"] = "no completed training cells"
        return decision

    if experiment.get("quick_mode", False):
        decision["passed"] = bool(decision["complete"])
        decision["reason"] = (
            "quick smoke training completed"
            if decision["passed"]
            else "quick smoke training is incomplete"
        )
        return decision

    frame = pd.DataFrame(results)
    for model, by_k in experiment.get("reference_validation_rmse", {}).items():
        for k_text, reference in by_k.items():
            k = int(k_text)
            selected = frame[(frame["model"] == model) & (frame["k"] == k)]
            observed = float(selected["rmse"].mean()) if not selected.empty else None
            difference = None if observed is None else observed - float(reference)
            passed = (
                observed is not None
                and abs(difference) <= float(experiment["regression_rmse_tolerance"])
            )
            decision["reference_checks"].append(
                {
                    "model": model,
                    "k": k,
                    "reference_rmse": float(reference),
                    "observed_rmse": observed,
                    "difference": difference,
                    "passed": bool(passed),
                }
            )
    decision["passed"] = bool(
        decision["complete"]
        and decision["reference_checks"]
        and all(row["passed"] for row in decision["reference_checks"])
    )
    decision["reason"] = (
        "A1 reproduced the registered Experiment-17B means"
        if decision["passed"]
        else "A1 is incomplete or exceeded the registered RMSE tolerance"
    )
    return decision


def completed_keys(results: list[dict]) -> set[tuple[int, int, int, str]]:
    return {
        (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            int(row["k"]),
            str(row["model"]),
        )
        for row in results
    }


def main() -> None:
    args = parse_args()
    base, experiment = load_experiment_config(args)
    validate_configuration(base, experiment)
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = result_paths(output)
    commit = git_commit(PROJECT_ROOT)

    protocol = build_crossed_train_only_protocol(
        data_dir=base["data_dir"],
        target_domain=base["target_domain"],
        source_domains=base["source_domains"],
        validation_count=int(experiment["validation_units"]),
        validation_seed=int(experiment["validation_seed"]),
        target_split_seeds=experiment["target_split_seeds"],
        model_seeds=experiment["model_seeds"],
        k_values=experiment["k_values"],
    )
    prior, correlation, graph_fit = source_correlation_adjacency_train_only(
        base,
        experiment["preprocessing"],
        int(experiment["sensor_graph_k"]),
    )
    sensors = list(base["sensor_columns"])
    random_by_model_seed: dict[int, torch.Tensor] = {}
    random_audits: list[dict] = []
    for model_seed, graph_seed in zip(
        experiment["model_seeds"],
        experiment["random_graph_seeds"],
        strict=True,
    ):
        randomized, audit = exp17b.degree_preserving_random_graph(
            prior,
            int(graph_seed),
            swaps_multiplier=20,
        )
        random_by_model_seed[int(model_seed)] = randomized
        random_audits.append({"model_seed": int(model_seed), **audit})

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": commit,
        "base_config": base,
        "experiment_config": experiment,
        "protocol_hash": protocol["protocol_hash"],
        "graph_fit": graph_fit,
        "prior_graph_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
        "official_test_policy": "test files are not loaded in Experiment A1",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_write_text(
        paths["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        paths["protocol"],
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        paths["engine_splits"],
        protocol_frame(protocol).to_csv(index=False),
    )
    atomic_write_text(
        paths["prior_adjacency"],
        pd.DataFrame(
            prior.numpy().astype(int),
            index=sensors,
            columns=sensors,
        ).to_csv(),
    )
    atomic_write_text(
        paths["prior_correlation"],
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
    )
    atomic_write_text(
        paths["random_graph_audit"],
        pd.DataFrame(random_audits).to_csv(index=False),
    )

    first_seed = experiment["model_seeds"][0]
    first_split = experiment["target_split_seeds"][0]
    first_k = min(experiment["k_values"])
    first_units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(first_split)
    ][str(first_k)]
    dry_cfg = dict(base)
    dry_cfg["seed"] = first_seed
    source_tasks, support, validation, feature_count, split = (
        prepare_validation_experiment(
            dry_cfg,
            experiment["preprocessing"],
            experiment["balance_mode"],
            protocol["validation_units"],
            first_units,
        )
    )
    seed_everything(first_seed)
    dry_models = {
        model_name: exp17b.build_model_17b(
            model_name,
            feature_count,
            dry_cfg,
            prior,
            random_by_model_seed[first_seed],
        )
        for model_name in experiment["models"]
    }
    x, _ = next(iter(source_tasks[base["source_domains"][0]]))
    dry_shapes = {
        name: list(model(x[: min(4, len(x))]).shape)
        for name, model in dry_models.items()
    }
    dry_report = {
        "experiment_id": "experimentA1",
        "feature_count": int(feature_count),
        "source_batch_shape": list(x.shape),
        "support_windows": int(len(support.dataset)),
        "validation_windows": int(len(validation.dataset)),
        "model_output_shapes": dry_shapes,
        "split": split,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    atomic_write_text(
        paths["dry_run"],
        json.dumps(dry_report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return

    results: list[dict] = []
    predictions: list[pd.DataFrame] = []
    engine_metrics: list[pd.DataFrame] = []
    inventory_rows: list[dict] = []
    if paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        if paths["window_predictions"].is_file():
            predictions = [pd.read_csv(paths["window_predictions"])]
        if paths["per_engine"].is_file():
            engine_metrics = [pd.read_csv(paths["per_engine"])]
        if paths["inventory"].is_file():
            inventory_rows = pd.read_csv(paths["inventory"]).to_dict("records")
    done = completed_keys(results)

    for model_seed in experiment["model_seeds"]:
        randomized = random_by_model_seed[model_seed]
        cfg = dict(base)
        cfg["seed"] = model_seed
        for model_name in experiment["models"]:
            pending = [
                (split_seed, k)
                for split_seed in experiment["target_split_seeds"]
                for k in experiment["k_values"]
                if (split_seed, model_seed, k, model_name) not in done
            ]
            if not pending:
                continue
            source_state, source_history, inventory = load_or_train_source(
                base=cfg,
                experiment=experiment,
                protocol=protocol,
                model_name=model_name,
                model_seed=model_seed,
                prior=prior,
                randomized=randomized,
                commit=commit,
                resume=True,
            )
            inventory_rows = [
                row
                for row in inventory_rows
                if not (
                    str(row.get("model")) == model_name
                    and int(row.get("model_seed", -1)) == model_seed
                )
            ]
            inventory_rows.append(inventory)

            for split_seed, k in pending:
                result, window_frame, engine_frame = run_target_cell(
                    base=cfg,
                    experiment=experiment,
                    protocol=protocol,
                    model_name=model_name,
                    model_seed=model_seed,
                    target_split_seed=split_seed,
                    k=k,
                    source_state=deepcopy(source_state),
                    source_history=source_history,
                    inventory=inventory,
                    prior=prior,
                    randomized=randomized,
                    save_checkpoint=args.save_target_checkpoints,
                )
                results.append(result)
                predictions.append(window_frame)
                engine_metrics.append(engine_frame)
                done.add((split_seed, model_seed, k, model_name))
                save_progress(
                    paths=paths,
                    results=results,
                    predictions=predictions,
                    engine_metrics=engine_metrics,
                    inventory=inventory_rows,
                    bootstrap_repetitions=experiment["bootstrap_repetitions"],
                )

    decision = regression_decision(experiment=experiment, results=results)
    atomic_write_text(
        paths["decision"],
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
