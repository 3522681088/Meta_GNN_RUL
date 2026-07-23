"""Experiment 27: source-task gradient-alignment audit.

Experiment 26-1 showed that support updates from condition-matched engines do
not reliably improve disjoint query engines. Before training another meta
learner, this source-only audit compares four task definitions:

* random: engine-disjoint windows without task matching;
* condition: same operating condition;
* rul_stage: same project-standard RUL stage;
* condition_rul_stage: same condition and RUL stage.

The frozen static graph backbone and predictor initialization are shared.
For each task, the script measures support/query predictor-gradient cosine
similarity and the actual query-loss change after one support update. Target
train files and official test files are never loaded.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_experiment26_1():
    path = PROJECT_ROOT / "scripts" / "experiment26-1_inner_loop_calibration.py"
    if not path.is_file():
        path = PROJECT_ROOT / "experiment26-1_inner_loop_calibration.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Experiment 26-1 runner: {path}")
    spec = importlib.util.spec_from_file_location("experiment26_1_for_exp27", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Experiment 26-1 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp261 = _load_experiment26_1()
exp26 = exp261.exp26
exp251 = exp261.exp251
atomic_write_text = exp261.atomic_write_text
resolve_device = exp261.resolve_device
seed_everything = exp261.seed_everything
EXPECTED_OFFICIAL_TEST_ENGINES = exp261.EXPECTED_OFFICIAL_TEST_ENGINES

SCRIPT_VERSION = "experiment27_source_task_gradient_alignment_v1"
MODES = ("random", "condition", "rul_stage", "condition_rul_stage")
STAGE_BINS = (30.0, 60.0, 90.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 27: source-task gradient-alignment audit"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES), default="FD004"
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/experiment18_task_conditioned_sensor_graph/source_cache",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--set-token-dim", type=int, default=32)
    parser.add_argument("--graph-residual-rank", type=int, default=4)
    parser.add_argument("--max-graph-gate", type=float, default=0.5)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--tasks-per-mode", type=int, default=100)
    parser.add_argument("--support-engines", type=int, default=5)
    parser.add_argument("--query-engines", type=int, default=5)
    parser.add_argument("--support-windows", type=int, default=128)
    parser.add_argument("--query-windows", type=int, default=128)
    parser.add_argument("--audit-inner-lr", type=float, default=0.0001)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="outputs/experiment27_source_task_gradient_alignment"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    positive = (
        args.tasks_per_mode,
        args.support_engines,
        args.query_engines,
        args.support_windows,
        args.query_windows,
        args.audit_inner_lr,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Task counts, window counts, and learning rate must be positive")
    if not 1 <= args.sensor_graph_k < len(cfg["sensor_columns"]):
        raise ValueError("--sensor-graph-k is outside the sensor count")


def rul_stages(labels: np.ndarray) -> np.ndarray:
    return np.digitize(labels, bins=STAGE_BINS, right=True)


def group_key(mode: str, condition: int, stage: int):
    if mode == "condition":
        return int(condition)
    if mode == "rul_stage":
        return int(stage)
    if mode == "condition_rul_stage":
        return int(condition), int(stage)
    return "all"


def valid_groups(
    data: dict,
    mode: str,
    required_engines: int,
) -> dict[object, np.ndarray]:
    groups: dict[object, list[int]] = {}
    stages = data["stages"]
    for index, (condition, stage) in enumerate(
        zip(data["conditions"], stages, strict=True)
    ):
        key = group_key(mode, int(condition), int(stage))
        groups.setdefault(key, []).append(index)
    valid = {}
    for key, values in groups.items():
        indices = np.asarray(values, dtype=int)
        if len(np.unique(data["units"][indices])) >= required_engines:
            valid[key] = indices
    return valid


def build_group_inventory(
    source_data: dict[str, dict],
    required_engines: int,
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    inventory = {
        mode: {
            domain: valid_groups(data, mode, required_engines)
            for domain, data in source_data.items()
        }
        for mode in MODES
    }
    common_domains = [
        domain
        for domain in sorted(source_data)
        if all(inventory[mode][domain] for mode in MODES)
    ]
    if not common_domains:
        raise RuntimeError("No source domain supports every task definition")
    return inventory, common_domains


def sample_tasks(
    args: argparse.Namespace,
    source_data: dict[str, dict],
    inventory: dict[str, dict[str, dict]],
    mode: str,
    domain_sequence: list[str],
) -> list[dict]:
    rng = np.random.default_rng(
        args.model_seed + 27001 + MODES.index(mode) * 1000
    )
    tasks = []
    for task_index, domain in enumerate(domain_sequence):
        data = source_data[domain]
        groups = inventory[mode][domain]
        keys = sorted(groups, key=str)
        key = keys[int(rng.integers(0, len(keys)))]
        indices = groups[key]
        units = data["units"][indices]
        selected = rng.choice(
            np.unique(units),
            size=args.support_engines + args.query_engines,
            replace=False,
        )
        support_units = selected[: args.support_engines]
        query_units = selected[args.support_engines :]
        labels = data["y"].numpy()[indices]
        support_local = exp26.exp18._sample_balanced_indices(
            units,
            labels,
            support_units,
            args.support_windows,
            rng,
        )
        query_local = exp26.exp18._sample_balanced_indices(
            units,
            labels,
            query_units,
            args.query_windows,
            rng,
        )
        support_indices = indices[support_local]
        query_indices = indices[query_local]
        if set(data["units"][support_indices]) & set(
            data["units"][query_indices]
        ):
            raise AssertionError("Support and query engines overlap")
        tasks.append(
            {
                "task_index": task_index,
                "mode": mode,
                "domain": domain,
                "group": str(key),
                "support_x": data["x"][support_indices],
                "support_y": data["y"][support_indices],
                "query_x": data["x"][query_indices],
                "query_y": data["y"][query_indices],
            }
        )
    return tasks


def loss_and_gradient(
    predictor: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = F.mse_loss(predictor(x).squeeze(-1), y)
    gradients = torch.autograd.grad(loss, tuple(predictor.parameters()))
    vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
    return loss, vector


def evaluate_task(
    initial_predictor: torch.nn.Module,
    task: dict,
    audit_inner_lr: float,
    device: torch.device,
    task_seed: int,
) -> dict:
    predictor = deepcopy(initial_predictor).to(device).eval()
    sx = task["support_x"].to(device)
    sy = task["support_y"].to(device)
    qx = task["query_x"].to(device)
    qy = task["query_y"].to(device)

    support_loss, support_gradient = loss_and_gradient(predictor, sx, sy)
    query_loss, query_gradient = loss_and_gradient(predictor, qx, qy)
    support_norm = support_gradient.norm()
    query_norm = query_gradient.norm()
    cosine = F.cosine_similarity(
        support_gradient,
        query_gradient,
        dim=0,
        eps=1e-12,
    )

    seed_everything(task_seed)
    adapted = deepcopy(initial_predictor).to(device)
    adapted.train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=audit_inner_lr)
    optimizer.zero_grad()
    adaptation_loss = F.mse_loss(adapted(sx).squeeze(-1), sy)
    adaptation_loss.backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
    optimizer.step()
    adapted.eval()
    with torch.no_grad():
        post_query_loss = F.mse_loss(adapted(qx).squeeze(-1), qy)

    gain = float(query_loss.item() - post_query_loss.item())
    row = {
        "model_seed": int(initial_predictor._experiment27_model_seed),
        "task_index": task["task_index"],
        "mode": task["mode"],
        "domain": task["domain"],
        "group": task["group"],
        "support_loss": float(support_loss.item()),
        "pre_query_loss": float(query_loss.item()),
        "post_query_loss": float(post_query_loss.item()),
        "adaptation_gain": gain,
        "relative_adaptation_gain_pct": float(
            100.0 * gain / max(float(query_loss.item()), 1e-8)
        ),
        "positive_adaptation_gain": bool(gain > 0.0),
        "gradient_cosine": float(cosine.item()),
        "positive_gradient_cosine": bool(cosine.item() > 0.0),
        "support_gradient_norm": float(support_norm.item()),
        "query_gradient_norm": float(query_norm.item()),
        "gradient_dot_product": float(
            torch.dot(support_gradient, query_gradient).item()
        ),
        "adaptation_gradient_norm_before_clip": float(unclipped_norm),
    }
    del predictor, adapted, optimizer
    return row


def summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        raw.groupby(["model_seed", "mode"], as_index=False)
        .agg(
            task_count=("task_index", "count"),
            domain_count=("domain", "nunique"),
            gradient_cosine_mean=("gradient_cosine", "mean"),
            gradient_cosine_median=("gradient_cosine", "median"),
            gradient_cosine_std=("gradient_cosine", "std"),
            positive_gradient_rate=("positive_gradient_cosine", "mean"),
            adaptation_gain_mean=("adaptation_gain", "mean"),
            adaptation_gain_median=("adaptation_gain", "median"),
            relative_adaptation_gain_pct_mean=(
                "relative_adaptation_gain_pct",
                "mean",
            ),
            positive_adaptation_rate=("positive_adaptation_gain", "mean"),
        )
        .sort_values(
            ["gradient_cosine_mean", "adaptation_gain_mean"],
            ascending=False,
        )
    )
    by_domain = (
        raw.groupby(["model_seed", "mode", "domain"], as_index=False)
        .agg(
            task_count=("task_index", "count"),
            gradient_cosine_mean=("gradient_cosine", "mean"),
            positive_gradient_rate=("positive_gradient_cosine", "mean"),
            adaptation_gain_mean=("adaptation_gain", "mean"),
            positive_adaptation_rate=("positive_adaptation_gain", "mean"),
        )
    )
    return summary, by_domain


def select_mode(summary: pd.DataFrame) -> tuple[str | None, dict]:
    indexed = summary.set_index("mode")
    baseline_cosine = max(
        float(indexed.loc["random", "gradient_cosine_mean"]),
        float(indexed.loc["condition", "gradient_cosine_mean"]),
    )
    candidates = []
    for mode in ("rul_stage", "condition_rul_stage"):
        row = indexed.loc[mode]
        eligible = bool(
            row["gradient_cosine_mean"] >= baseline_cosine + 0.05
            and row["positive_gradient_rate"] >= 0.65
            and row["adaptation_gain_mean"] > 0.0
            and row["positive_adaptation_rate"] >= 0.60
        )
        candidates.append(
            {
                "mode": mode,
                "eligible": eligible,
                "gradient_cosine_margin_vs_best_baseline": float(
                    row["gradient_cosine_mean"] - baseline_cosine
                ),
                "gradient_cosine_mean": float(row["gradient_cosine_mean"]),
                "positive_gradient_rate": float(row["positive_gradient_rate"]),
                "adaptation_gain_mean": float(row["adaptation_gain_mean"]),
                "positive_adaptation_rate": float(
                    row["positive_adaptation_rate"]
                ),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(eligible, key=lambda candidate: candidate["gradient_cosine_mean"])[
            "mode"
        ]
        if eligible
        else None
    )
    return selected, {"baseline_cosine": baseline_cosine, "candidates": candidates}


def self_check() -> None:
    assert rul_stages(np.asarray([20.0, 40.0, 70.0, 100.0])).tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert group_key("condition_rul_stage", 2, 3) == (2, 3)
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([1.0, 0.0])
    assert abs(float(F.cosine_similarity(first, second, dim=0)) - 1.0) < 1e-6


def main() -> None:
    self_check()
    args = parse_args()
    if args.dry_run:
        args.tasks_per_mode = min(args.tasks_per_mode, 2)
        args.support_windows = min(args.support_windows, 32)
        args.query_windows = min(args.query_windows, 32)

    cfg = exp251.load_config(args)
    validate_args(args, cfg)
    output = (
        exp251.exp25.project_path(args.output_dir) / f"seed{args.model_seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(
        f"[{SCRIPT_VERSION}] target_excluded={args.target} "
        f"model_seed={args.model_seed} device={device}"
    )
    print(
        "[policy] source train files only; target train and official test "
        "are not loaded"
    )

    normalized, feature_names, source_condition_counts = (
        exp261.source_only_normalized_frames(cfg)
    )
    sensors = list(cfg["sensor_columns"])
    prior = exp261.source_prior(
        normalized, cfg["source_domains"], sensors, args.sensor_graph_k
    )
    model, initialization_checkpoint = exp251.build_model(
        args, cfg, prior, len(feature_names)
    )
    exp26.disable_context_graph(model)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial_predictor = deepcopy(model.base.predictor).cpu()
    for parameter in initial_predictor.parameters():
        parameter.requires_grad_(True)
    initial_predictor._experiment27_model_seed = args.model_seed

    print("[features] encoding frozen source backbone")
    source_windows = {
        domain: exp251.frame_windows(normalized[domain], feature_names, cfg)
        for domain in cfg["source_domains"]
    }
    source_data = {
        domain: {
            **exp26.encode_data(model, data, device),
            "stages": rul_stages(data["y"].numpy()),
        }
        for domain, data in source_windows.items()
    }
    inventory, common_domains = build_group_inventory(
        source_data, args.support_engines + args.query_engines
    )
    domain_rng = np.random.default_rng(args.model_seed + 27000)
    domain_sequence = domain_rng.choice(
        common_domains, size=args.tasks_per_mode, replace=True
    ).tolist()

    raw_rows = []
    for mode in MODES:
        tasks = sample_tasks(
            args,
            source_data,
            inventory,
            mode,
            domain_sequence,
        )
        for task in tasks:
            raw_rows.append(
                evaluate_task(
                    initial_predictor,
                    task,
                    args.audit_inner_lr,
                    device,
                    args.model_seed * 100000
                    + task["task_index"],
                )
            )
        current = pd.DataFrame(
            [row for row in raw_rows if row["mode"] == mode]
        )
        print(
            f"[mode] {mode} "
            f"cosine={current['gradient_cosine'].mean():.4f} "
            f"positive_cosine={current['positive_gradient_cosine'].mean():.3f} "
            f"adapt_gain={current['adaptation_gain'].mean():.4f} "
            f"positive_adapt={current['positive_adaptation_gain'].mean():.3f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(raw_rows)
    summary, by_domain = summaries(raw)
    selected_mode, selection_audit = select_mode(summary)
    inventory_rows = [
        {
            "mode": mode,
            "domain": domain,
            "valid_group_count": len(groups),
            "valid_groups": json.dumps(list(map(str, sorted(groups, key=str)))),
        }
        for mode, domains in inventory.items()
        for domain, groups in domains.items()
    ]
    report = {
        "script_version": SCRIPT_VERSION,
        "excluded_target": args.target,
        "model_seed": args.model_seed,
        "dry_run": bool(args.dry_run),
        "scope": "source_only_static_initialization_gradient_alignment_audit",
        "gradient_measurement_mode": "predictor_eval_with_dropout_disabled",
        "adaptation_support_mode": "train",
        "adaptation_query_mode": "eval",
        "rul_stage_bins": list(STAGE_BINS),
        "audit_inner_lr": args.audit_inner_lr,
        "tasks_per_mode": args.tasks_per_mode,
        "common_source_domains": common_domains,
        "target_train_files_accessed": False,
        "official_test_files_accessed": False,
        "outer_meta_training_run": False,
        "initialization_checkpoint": str(initialization_checkpoint),
        "source_condition_counts": source_condition_counts,
        "selection_rule": {
            "gradient_cosine_margin_vs_best_baseline_at_least": 0.05,
            "positive_gradient_rate_at_least": 0.65,
            "mean_adaptation_gain_must_be_positive": True,
            "positive_adaptation_rate_at_least": 0.60,
        },
        "selection_audit": selection_audit,
        "selected_mode": selected_mode,
        "decision": (
            "compare_seed42_and_seed43_then_run_experiment27-1"
            if selected_mode is not None
            else "do_not_train; abandon_gradient_based_task_adaptation"
        ),
        "caveat": (
            "source gradient alignment is necessary but does not prove "
            "target-domain RUL improvement"
        ),
    }

    prefix = f"experiment27_{args.target}_seed{args.model_seed}"
    atomic_write_text(output / f"{prefix}_summary.csv", summary.to_csv(index=False))
    atomic_write_text(
        output / f"{prefix}_by_domain.csv", by_domain.to_csv(index=False)
    )
    atomic_write_text(
        output / f"{prefix}_group_inventory.csv",
        pd.DataFrame(inventory_rows).to_csv(index=False),
    )
    atomic_write_text(output / f"{prefix}_raw.csv", raw.to_csv(index=False))
    atomic_write_text(
        output / f"{prefix}_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(summary.to_string(index=False))
    print(f"[selected_mode] {selected_mode}")
    print(f"[{SCRIPT_VERSION}] complete: {output}")


if __name__ == "__main__":
    main()
