"""Experiment 25: audit condition identity and context use before retraining.

The script reuses Experiment 18 source checkpoints and reads C-MAPSS training
files only. It answers three questions:

1. Do K-shot target engines cover the operating conditions seen in validation?
2. Are per-condition support summaries separable before and after encoding?
3. Does swapping the matched condition context change predictions or RMSE?

No optimizer step and no official-test forward pass are performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.window_dataset import make_windows  # noqa: E402
from scripts import experiment18_task_conditioned_sensor_graph as exp18  # noqa: E402
from scripts import experiment24_truncated_endpoint_validation as exp24  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    resolve_device,
)


SCRIPT_VERSION = "experiment25_task_identifiability_audit_v1"
PREPROCESSING = "condition_settings"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 25: condition identity and context-use audit"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES), default="FD004"
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--target-split-seeds",
        nargs="+",
        type=int,
        default=[3027, 3028, 3029, 3030, 3031],
    )
    parser.add_argument(
        "--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
    )
    parser.add_argument("--protocol")
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/experiment18_task_conditioned_sensor_graph/source_cache",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--graph-residual-rank", type=int, default=4)
    parser.add_argument("--max-graph-gate", type=float, default=0.5)
    parser.add_argument("--gate-scale", type=float, default=2.0)
    parser.add_argument("--min-condition-windows", type=int, default=4)
    parser.add_argument(
        "--max-validation-windows-per-condition", type=int, default=256
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="outputs/experiment25_task_identifiability_audit"
    )
    parser.add_argument("--min-coverage-rate", type=float, default=0.8)
    parser.add_argument("--min-separation-ratio", type=float, default=1.2)
    parser.add_argument(
        "--min-counterfactual-improvement-pct", type=float, default=0.5
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    cfg["seed"] = int(args.model_seeds[0])
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain for domain in EXPECTED_OFFICIAL_TEST_ENGINES if domain != args.target
    ]
    cfg["normalizer_seed"] = int(args.normalizer_seed)
    cfg["condition_count"] = int(args.condition_count)
    cfg["data_dir"] = str(
        project_path(args.data_dir if args.data_dir is not None else cfg["data_dir"])
    )
    cfg["output_dir"] = str(project_path(args.output_dir))
    return cfg


def load_protocol(args: argparse.Namespace) -> tuple[dict, Path]:
    path = project_path(
        args.protocol
        or (
            "outputs/experiment18_task_conditioned_sensor_graph/"
            f"experiment18_{args.target}_protocol.json"
        )
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("target_domain") != args.target:
        raise ValueError("Protocol target does not match --target")
    nested = protocol.get("nested_adaptation_units_by_target_split_seed", {})
    for split_seed in args.target_split_seeds:
        for k in args.k_values:
            if str(k) not in nested.get(str(split_seed), {}):
                raise ValueError(f"Protocol lacks target split {split_seed}, K={k}")
    return protocol, path


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    if not args.k_values or min(args.k_values) <= 0:
        raise ValueError("--k-values must contain positive integers")
    if not args.model_seeds or not args.target_split_seeds:
        raise ValueError("Model seeds and target split seeds cannot be empty")
    if not 1 <= args.sensor_graph_k < len(cfg["sensor_columns"]):
        raise ValueError("--sensor-graph-k is outside the sensor count")
    if args.condition_count < 2 or args.min_condition_windows < 2:
        raise ValueError("At least two conditions and two windows are required")
    if args.max_validation_windows_per_condition <= 0:
        raise ValueError("Validation window cap must be positive")
    if args.gate_scale <= 0:
        raise ValueError("--gate-scale must be positive")


def condition_windows(frame: pd.DataFrame, features: list[str], cfg: dict):
    columns = features + ["condition_id"]
    x, y, units = make_windows(
        frame,
        columns,
        cfg["window_size"],
        cfg["window_stride"],
    )
    conditions = np.rint(x[:, -1, -1]).astype(np.int64)
    return (
        torch.as_tensor(x[:, :, :-1], dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        conditions,
        units,
    )


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    value = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    return float(1.0 - max(-1.0, min(1.0, value)))


def mean_pairwise_distance(values: list[torch.Tensor]) -> float:
    distances = [
        cosine_distance(values[i], values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return float(np.mean(distances)) if distances else float("nan")


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    return float(numerator / max(denominator, 1e-8))


def split_summary(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    sensor_count: int,
    rul_cap: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = np.random.default_rng(seed).permutation(indices)
    midpoint = len(order) // 2
    first, second = order[:midpoint], order[midpoint:]
    return (
        exp18.support_summary(x[first], y[first], sensor_count, rul_cap),
        exp18.support_summary(x[second], y[second], sensor_count, rul_cap),
    )


def prepare_cells(args: argparse.Namespace, cfg: dict, protocol: dict):
    normalized, features, source_condition_counts = exp24.normalized_train_frames(
        cfg, PREPROCESSING
    )
    target = normalized[args.target]
    sensor_count = len(cfg["sensor_columns"])
    rows = []
    cells = []
    nested = protocol["nested_adaptation_units_by_target_split_seed"]
    validation_units = set(map(int, protocol["validation_units"]))
    validation_frame = target[target["unit"].isin(validation_units)]
    vx, vy, vc, _ = condition_windows(validation_frame, features, cfg)

    for split_seed in args.target_split_seeds:
        for k in args.k_values:
            support_units = set(map(int, nested[str(split_seed)][str(k)]))
            if support_units & validation_units:
                raise AssertionError("Support and validation engines overlap")
            support_frame = target[target["unit"].isin(support_units)]
            sx, sy, sc, _ = condition_windows(support_frame, features, cfg)
            global_summary = exp18.support_summary(
                sx, sy, sensor_count, cfg["rul_cap"]
            )
            summaries = {}
            halves = {}
            for condition in range(args.condition_count):
                support_indices = np.flatnonzero(sc == condition)
                validation_count = int(np.sum(vc == condition))
                eligible = len(support_indices) >= args.min_condition_windows
                rows.append(
                    {
                        "target": args.target,
                        "target_split_seed": split_seed,
                        "k": k,
                        "condition": condition,
                        "support_windows": int(len(support_indices)),
                        "validation_windows": validation_count,
                        "validation_present": validation_count > 0,
                        "context_eligible": eligible,
                    }
                )
                if eligible:
                    summaries[condition] = exp18.support_summary(
                        sx[support_indices],
                        sy[support_indices],
                        sensor_count,
                        cfg["rul_cap"],
                    )
                    halves[condition] = split_summary(
                        sx,
                        sy,
                        support_indices,
                        sensor_count,
                        cfg["rul_cap"],
                        seed=25000 + 31 * split_seed + 7 * k + condition,
                    )
            cells.append(
                {
                    "target_split_seed": split_seed,
                    "k": k,
                    "validation_x": vx,
                    "validation_y": vy,
                    "validation_conditions": vc,
                    "global_summary": global_summary,
                    "summaries": summaries,
                    "halves": halves,
                }
            )
    return cells, pd.DataFrame(rows), features, source_condition_counts


def load_model(
    args: argparse.Namespace,
    cfg: dict,
    prior: torch.Tensor,
    feature_count: int,
    model_seed: int,
    device: torch.device,
):
    checkpoint = project_path(args.checkpoint_dir) / (
        f"experiment18_source_bundle_{args.target}_modelseed{model_seed}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Experiment 18 checkpoint: {checkpoint}")
    bundle = exp24.exp17.safe_torch_load(checkpoint)
    if "tcsg" not in bundle.get("states", {}):
        raise ValueError(f"Checkpoint has no TCSG state: {checkpoint}")
    model = exp18.build_tcsg_model(feature_count, cfg, prior, args)
    model.load_state_dict(bundle["states"]["tcsg"])
    for layer in model.graph_layers:
        layer.max_gate *= args.gate_scale
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def encode_summary(model, summary: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    return model.task_context_encoder(summary.to(device)).squeeze(0).cpu()


@torch.inference_mode()
def graph_signature(model, summary: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    context = model.task_context_encoder(summary.to(device))
    values = []
    for layer in model.graph_layers:
        residual, gate = layer.edge_residual(context)
        values.append((gate[:, :, None, None] * residual).flatten())
    return torch.cat(values).cpu()


@torch.inference_mode()
def predict(model, x: torch.Tensor, summary: torch.Tensor, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    summary = summary.to(device)
    outputs = []
    for start in range(0, len(x), batch_size):
        outputs.append(model(x[start : start + batch_size].to(device), summary).cpu())
    return torch.cat(outputs).numpy()


def summary_gradient_norm(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    summary: torch.Tensor,
    batch_size: int,
) -> float:
    device = next(model.parameters()).device
    context_input = summary.to(device).detach().clone().requires_grad_(True)
    count = min(batch_size, len(x))
    prediction = model(x[:count].to(device), context_input)
    loss = F.mse_loss(prediction, y[:count].to(device))
    gradient = torch.autograd.grad(loss, context_input)[0]
    return float(gradient.norm().item())


def rmse(labels: torch.Tensor, predictions: np.ndarray) -> float:
    target = labels.detach().cpu().numpy().astype(float)
    return float(np.sqrt(np.mean((predictions.astype(float) - target) ** 2)))


def audit_model(
    args: argparse.Namespace,
    cfg: dict,
    cells: list[dict],
    model,
    model_seed: int,
):
    geometry_rows = []
    counterfactual_rows = []
    sensor_count = len(cfg["sensor_columns"])
    batch_size = int(cfg["batch_size"])

    for cell in cells:
        summaries = cell["summaries"]
        halves = cell["halves"]
        validation_conditions = cell["validation_conditions"]
        validation_present = set(map(int, np.unique(validation_conditions)))
        covered = validation_present & set(summaries)

        raw_between = mean_pairwise_distance(list(summaries.values()))
        raw_within = finite_mean(
            cosine_distance(*halves[condition]) for condition in summaries
        )
        contexts = {
            condition: encode_summary(model, summary)
            for condition, summary in summaries.items()
        }
        encoded_halves = {
            condition: (
                encode_summary(model, halves[condition][0]),
                encode_summary(model, halves[condition][1]),
            )
            for condition in summaries
        }
        encoded_between = mean_pairwise_distance(list(contexts.values()))
        encoded_within = finite_mean(
            cosine_distance(*encoded_halves[condition]) for condition in summaries
        )
        geometry_rows.append(
            {
                "target": args.target,
                "target_split_seed": cell["target_split_seed"],
                "model_seed": model_seed,
                "k": cell["k"],
                "validation_condition_count": len(validation_present),
                "covered_validation_condition_count": len(covered),
                "coverage_rate": len(covered) / max(1, len(validation_present)),
                "raw_between_condition_cosine_distance": raw_between,
                "raw_within_condition_half_cosine_distance": raw_within,
                "raw_separation_ratio": safe_ratio(raw_between, raw_within),
                "encoded_between_condition_cosine_distance": encoded_between,
                "encoded_within_condition_half_cosine_distance": encoded_within,
                "encoded_separation_ratio": safe_ratio(
                    encoded_between, encoded_within
                ),
            }
        )

        global_summary = cell["global_summary"]
        for condition in sorted(covered):
            wrong_conditions = sorted(set(summaries) - {condition})
            if not wrong_conditions:
                continue
            indices = np.flatnonzero(validation_conditions == condition)
            if len(indices) > args.max_validation_windows_per_condition:
                indices = np.random.default_rng(
                    26000
                    + 31 * cell["target_split_seed"]
                    + 7 * cell["k"]
                    + condition
                ).choice(
                    indices,
                    size=args.max_validation_windows_per_condition,
                    replace=False,
                )
            x = cell["validation_x"][indices]
            y = cell["validation_y"][indices]
            matched_summary = summaries[condition]
            matched_prediction = predict(
                model, x, matched_summary, batch_size=batch_size
            )
            global_prediction = predict(
                model, x, global_summary, batch_size=batch_size
            )
            matched_rmse = rmse(y, matched_prediction)
            global_rmse = rmse(y, global_prediction)
            matched_graph = graph_signature(model, matched_summary)

            wrong_rmses = []
            prediction_deltas = []
            graph_deltas = []
            for wrong_condition in wrong_conditions:
                wrong_summary = summaries[wrong_condition]
                wrong_prediction = predict(
                    model, x, wrong_summary, batch_size=batch_size
                )
                wrong_rmses.append(rmse(y, wrong_prediction))
                prediction_deltas.append(
                    float(np.mean(np.abs(matched_prediction - wrong_prediction)))
                )
                graph_deltas.append(
                    float(
                        torch.linalg.vector_norm(
                            matched_graph - graph_signature(model, wrong_summary)
                        ).item()
                    )
                )
            wrong_rmse_mean = float(np.mean(wrong_rmses))
            counterfactual_rows.append(
                {
                    "target": args.target,
                    "target_split_seed": cell["target_split_seed"],
                    "model_seed": model_seed,
                    "k": cell["k"],
                    "query_condition": condition,
                    "query_windows": len(indices),
                    "wrong_context_count": len(wrong_conditions),
                    "matched_rmse": matched_rmse,
                    "global_rmse": global_rmse,
                    "wrong_context_rmse_mean": wrong_rmse_mean,
                    "matched_minus_global_rmse": matched_rmse - global_rmse,
                    "matched_vs_wrong_rmse_improvement_pct": (
                        100.0 * (wrong_rmse_mean - matched_rmse)
                        / max(wrong_rmse_mean, 1e-8)
                    ),
                    "matched_context_beats_wrong": matched_rmse < wrong_rmse_mean,
                    "matched_wrong_prediction_abs_delta_mean": float(
                        np.mean(prediction_deltas)
                    ),
                    "matched_wrong_graph_delta_norm_mean": float(
                        np.mean(graph_deltas)
                    ),
                    "summary_gradient_norm": summary_gradient_norm(
                        model, x, y, matched_summary, batch_size
                    ),
                    "matched_global_context_cosine_distance": cosine_distance(
                        contexts[condition], encode_summary(model, global_summary)
                    ),
                }
            )
    return geometry_rows, counterfactual_rows


def summarize(
    args: argparse.Namespace,
    coverage: pd.DataFrame,
    geometry: pd.DataFrame,
    counterfactual: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    decisions = {}
    for k in args.k_values:
        geo = geometry[geometry["k"] == k]
        cf = counterfactual[counterfactual["k"] == k]
        coverage_rate = finite_mean(geo["coverage_rate"])
        raw_ratio = finite_mean(geo["raw_separation_ratio"])
        encoded_ratio = finite_mean(geo["encoded_separation_ratio"])
        counterfactual_improvement = finite_mean(
            cf["matched_vs_wrong_rmse_improvement_pct"]
        )
        win_rate = finite_mean(cf["matched_context_beats_wrong"].astype(float))
        prediction_delta = finite_mean(
            cf["matched_wrong_prediction_abs_delta_mean"]
        )
        gradient_norm = finite_mean(cf["summary_gradient_norm"])

        if coverage_rate < args.min_coverage_rate:
            decision = "stop_condition_routing_insufficient_support_coverage"
        elif raw_ratio < args.min_separation_ratio:
            decision = "replace_global_statistics_with_trajectory_set_encoder"
        elif encoded_ratio < args.min_separation_ratio:
            decision = "context_encoder_collapses_condition_identity"
        elif (
            counterfactual_improvement
            < args.min_counterfactual_improvement_pct
            or win_rate < 0.6
        ):
            decision = "train_frozen_graph_adapter_with_counterfactual_objective"
        else:
            decision = "condition_context_reaches_predictions_proceed_to_25_1"
        decisions[str(k)] = decision
        rows.append(
            {
                "target": args.target,
                "k": k,
                "coverage_rate_mean": coverage_rate,
                "raw_separation_ratio_mean": raw_ratio,
                "encoded_separation_ratio_mean": encoded_ratio,
                "matched_vs_wrong_rmse_improvement_pct_mean": (
                    counterfactual_improvement
                ),
                "matched_context_win_rate": win_rate,
                "prediction_abs_delta_mean": prediction_delta,
                "summary_gradient_norm_mean": gradient_norm,
                "decision": decision,
            }
        )
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "analysis_only": True,
        "optimizer_steps": 0,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "preprocessing": PREPROCESSING,
        "gate_scale": args.gate_scale,
        "k_values": list(args.k_values),
        "target_split_seeds": list(args.target_split_seeds),
        "model_seeds": list(args.model_seeds),
        "coverage_rows": int(len(coverage)),
        "geometry_rows": int(len(geometry)),
        "counterfactual_rows": int(len(counterfactual)),
        "thresholds": {
            "min_coverage_rate": args.min_coverage_rate,
            "min_separation_ratio": args.min_separation_ratio,
            "min_counterfactual_improvement_pct": (
                args.min_counterfactual_improvement_pct
            ),
            "min_counterfactual_win_rate": 0.6,
        },
        "decision_by_k": decisions,
        "interpretation": {
            "insufficient_support_coverage": (
                "Condition routing is not viable with the current K-shot protocol."
            ),
            "raw_summary_failure": (
                "Global statistics erase condition identity; use a trajectory set encoder."
            ),
            "encoder_collapse": (
                "The raw signal exists but the learned context encoder removes it."
            ),
            "prediction_bypass": (
                "The model can distinguish contexts but does not use them for prediction."
            ),
        },
    }
    return pd.DataFrame(rows), report


def self_check() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    assert abs(cosine_distance(first, first)) < 1e-6
    assert abs(cosine_distance(first, second) - 1.0) < 1e-6
    assert safe_ratio(2.0, 1.0) == 2.0


def main() -> None:
    self_check()
    args = parse_args()
    if args.dry_run:
        args.k_values = args.k_values[:1]
        args.target_split_seeds = args.target_split_seeds[:1]
        args.model_seeds = args.model_seeds[:1]
        args.max_validation_windows_per_condition = min(
            args.max_validation_windows_per_condition, 32
        )

    cfg = load_config(args)
    validate_args(args, cfg)
    protocol, protocol_path = load_protocol(args)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[{SCRIPT_VERSION}] target={args.target} device={device}")
    print(f"[protocol] {protocol_path}")
    print("[policy] train files only; no optimizer and no official-test access")

    cells, coverage, features, source_condition_counts = prepare_cells(
        args, cfg, protocol
    )
    prior, _ = exp24.train_only_prior(cfg, PREPROCESSING, args.sensor_graph_k)
    geometry_rows = []
    counterfactual_rows = []
    checkpoints = []
    for model_seed in args.model_seeds:
        cfg["seed"] = int(model_seed)
        model, checkpoint = load_model(
            args, cfg, prior, len(features), model_seed, device
        )
        checkpoints.append(str(checkpoint))
        print(f"[model_seed={model_seed}] {checkpoint}")
        geometry, counterfactual = audit_model(
            args, cfg, cells, model, model_seed
        )
        geometry_rows.extend(geometry)
        counterfactual_rows.extend(counterfactual)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    geometry = pd.DataFrame(geometry_rows)
    counterfactual = pd.DataFrame(counterfactual_rows)
    if geometry.empty or counterfactual.empty:
        raise RuntimeError(
            "No auditable condition pairs were produced; inspect condition coverage"
        )
    summary, report = summarize(args, coverage, geometry, counterfactual)
    report.update(
        {
            "dry_run": bool(args.dry_run),
            "protocol_path": str(protocol_path),
            "checkpoint_paths": checkpoints,
            "source_condition_counts": source_condition_counts,
        }
    )

    prefix = f"experiment25_{args.target}"
    atomic_write_text(
        output / f"{prefix}_condition_coverage.csv",
        coverage.to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_context_geometry.csv",
        geometry.to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_counterfactual.csv",
        counterfactual.to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_summary.csv",
        summary.to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(summary.to_string(index=False))
    print(f"[{SCRIPT_VERSION}] complete: {output}")


if __name__ == "__main__":
    main()
