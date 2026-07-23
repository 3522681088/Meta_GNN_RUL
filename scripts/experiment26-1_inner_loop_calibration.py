"""Experiment 26-1: source-only inner-loop calibration.

Experiment 26 used inner_lr=0.01 and three support updates. Its logged
adaptation gain was almost always negative, so expanding target evaluation
would not be informative. This diagnostic reuses the frozen Experiment 26
backbone and evaluates a small learning-rate/step grid on identical
engine-disjoint source episodes.

No outer meta-training, target-domain sampling, or official-test access occurs.
Positive source adaptation gain is necessary, but not sufficient, evidence for
continuing Experiment 26.
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


def _load_experiment26():
    path = PROJECT_ROOT / "scripts" / "experiment26_engine_task_meta_initialization.py"
    if not path.is_file():
        path = PROJECT_ROOT / "experiment26_engine_task_meta_initialization.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Experiment 26 runner: {path}")
    spec = importlib.util.spec_from_file_location("experiment26_for_calibration", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Experiment 26 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp26 = _load_experiment26()
exp251 = exp26.exp251
exp24 = exp26.exp24
atomic_write_text = exp26.atomic_write_text
resolve_device = exp26.resolve_device
seed_everything = exp26.seed_everything
EXPECTED_OFFICIAL_TEST_ENGINES = exp26.EXPECTED_OFFICIAL_TEST_ENGINES

SCRIPT_VERSION = "experiment26-1_inner_loop_calibration_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 26-1: source-only inner-loop calibration"
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
    parser.add_argument(
        "--inner-lrs",
        nargs="+",
        type=float,
        default=[0.0001, 0.0003, 0.001, 0.003, 0.01],
    )
    parser.add_argument(
        "--inner-step-values", nargs="+", type=int, default=[1, 3]
    )
    parser.add_argument("--episodes-per-setting", type=int, default=100)
    parser.add_argument("--source-support-engines", type=int, default=5)
    parser.add_argument("--source-query-engines", type=int, default=5)
    parser.add_argument("--source-support-windows", type=int, default=128)
    parser.add_argument("--source-query-windows", type=int, default=128)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="outputs/experiment26-1_inner_loop_calibration"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    if any(value <= 0 for value in args.inner_lrs):
        raise ValueError("All --inner-lrs must be positive")
    if any(value <= 0 for value in args.inner_step_values):
        raise ValueError("All --inner-step-values must be positive")
    if args.episodes_per_setting <= 0:
        raise ValueError("--episodes-per-setting must be positive")
    if args.source_support_engines <= 0 or args.source_query_engines <= 0:
        raise ValueError("Source engine counts must be positive")
    if args.source_support_windows <= 0 or args.source_query_windows <= 0:
        raise ValueError("Source window counts must be positive")
    if not 1 <= args.sensor_graph_k < len(cfg["sensor_columns"]):
        raise ValueError("--sensor-graph-k is outside the sensor count")


def source_only_normalized_frames(cfg: dict):
    raw = {
        domain: exp24.load_train(cfg["data_dir"], domain, cfg["rul_cap"])
        for domain in cfg["source_domains"]
    }
    sensors = list(cfg["sensor_columns"])
    source_fit = pd.concat(raw.values(), ignore_index=True)
    normalizer = exp24.SourceConditionNormalizer(
        n_conditions=cfg.get("condition_count", 6),
        seed=cfg.get("normalizer_seed", 2026),
        include_settings=True,
    ).fit(source_fit, sensors)
    features = sensors + list(exp24.SETTING_FEATURE_COLUMNS)
    normalized = {
        domain: normalizer.transform(frame, sensors)
        for domain, frame in raw.items()
    }
    return (
        normalized,
        features,
        list(map(int, normalizer.source_condition_counts)),
    )


def source_prior(
    normalized: dict[str, pd.DataFrame],
    source_domains: list[str],
    sensors: list[str],
    neighbors: int,
) -> torch.Tensor:
    values = pd.concat(
        [normalized[domain] for domain in source_domains], ignore_index=True
    )[sensors].to_numpy(np.float64)
    correlation = np.nan_to_num(
        np.abs(np.corrcoef(values, rowvar=False)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    adjacency = np.eye(len(sensors), dtype=bool)
    for sensor in range(len(sensors)):
        scores = correlation[sensor].copy()
        scores[sensor] = -np.inf
        adjacency[sensor, np.argsort(scores)[-neighbors:]] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    return torch.as_tensor(adjacency)


def sample_episodes(
    args: argparse.Namespace,
    source_data: dict[str, dict],
) -> list[tuple]:
    bank = exp251.ConditionEpisodeBank(
        source_data,
        args.source_support_engines,
        args.source_query_engines,
        args.model_seed + 26101,
    )
    return [
        bank.sample(args.source_support_windows, args.source_query_windows)
        for _ in range(args.episodes_per_setting)
    ]


def evaluate_setting(
    initial_predictor: torch.nn.Module,
    episodes: list[tuple],
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> list[dict]:
    base = deepcopy(initial_predictor).to(device).eval()
    rows = []
    for episode_index, (domain, condition, sx, sy, qx, qy) in enumerate(episodes):
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        with torch.no_grad():
            pre_query_loss = F.mse_loss(base(qx).squeeze(-1), qy)

        adapted = deepcopy(base)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
        adapted.train()
        support_loss = None
        for _ in range(inner_steps):
            optimizer.zero_grad()
            support_loss = F.mse_loss(adapted(sx).squeeze(-1), sy)
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            optimizer.step()
        with torch.no_grad():
            post_query_loss = F.mse_loss(adapted(qx).squeeze(-1), qy)

        gain = float(pre_query_loss.item() - post_query_loss.item())
        rows.append(
            {
                "model_seed": int(initial_predictor._experiment26_model_seed),
                "inner_lr": float(inner_lr),
                "inner_steps": int(inner_steps),
                "episode_index": episode_index,
                "domain": domain,
                "condition": int(condition),
                "support_loss_after_inner": float(support_loss.item()),
                "pre_query_loss": float(pre_query_loss.item()),
                "post_query_loss": float(post_query_loss.item()),
                "adaptation_gain": gain,
                "relative_adaptation_gain_pct": float(
                    100.0 * gain / max(float(pre_query_loss.item()), 1e-8)
                ),
                "positive_gain": bool(gain > 0.0),
            }
        )
        del adapted, optimizer
    return rows


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_seed", "inner_lr", "inner_steps"]
    summary = (
        raw.groupby(keys, as_index=False)
        .agg(
            episode_count=("episode_index", "count"),
            pre_query_loss_mean=("pre_query_loss", "mean"),
            post_query_loss_mean=("post_query_loss", "mean"),
            adaptation_gain_mean=("adaptation_gain", "mean"),
            adaptation_gain_median=("adaptation_gain", "median"),
            adaptation_gain_std=("adaptation_gain", "std"),
            relative_adaptation_gain_pct_mean=(
                "relative_adaptation_gain_pct",
                "mean",
            ),
            positive_gain_rate=("positive_gain", "mean"),
        )
    )
    by_condition = (
        raw.groupby(keys + ["condition"], as_index=False)
        .agg(
            episode_count=("episode_index", "count"),
            adaptation_gain_mean=("adaptation_gain", "mean"),
            relative_adaptation_gain_pct_mean=(
                "relative_adaptation_gain_pct",
                "mean",
            ),
            positive_gain_rate=("positive_gain", "mean"),
        )
    )
    condition_rate = (
        by_condition.assign(
            condition_positive=lambda frame: frame["adaptation_gain_mean"] > 0.0
        )
        .groupby(keys, as_index=False)["condition_positive"]
        .mean()
        .rename(columns={"condition_positive": "positive_condition_rate"})
    )
    summary = summary.merge(condition_rate, on=keys, how="left")
    summary["eligible"] = (
        (summary["adaptation_gain_mean"] > 0.0)
        & (summary["positive_gain_rate"] >= 0.60)
        & (summary["positive_condition_rate"] >= 0.80)
    )
    summary = summary.sort_values(
        [
            "eligible",
            "relative_adaptation_gain_pct_mean",
            "positive_gain_rate",
        ],
        ascending=[False, False, False],
    )
    return summary, by_condition


def self_check() -> None:
    frame = pd.DataFrame(
        {
            "model_seed": [42, 42],
            "inner_lr": [0.001, 0.001],
            "inner_steps": [1, 1],
            "episode_index": [0, 1],
            "condition": [0, 1],
            "pre_query_loss": [10.0, 10.0],
            "post_query_loss": [9.0, 8.0],
            "adaptation_gain": [1.0, 2.0],
            "relative_adaptation_gain_pct": [10.0, 20.0],
            "positive_gain": [True, True],
        }
    )
    summary, _ = summarize(frame)
    assert bool(summary.iloc[0]["eligible"])


def main() -> None:
    self_check()
    args = parse_args()
    args.inner_lrs = list(dict.fromkeys(map(float, args.inner_lrs)))
    args.inner_step_values = list(
        dict.fromkeys(map(int, args.inner_step_values))
    )
    if args.dry_run:
        args.inner_lrs = args.inner_lrs[:1]
        args.inner_step_values = args.inner_step_values[:1]
        args.episodes_per_setting = min(args.episodes_per_setting, 4)
        args.source_support_windows = min(args.source_support_windows, 32)
        args.source_query_windows = min(args.source_query_windows, 32)

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
        source_only_normalized_frames(cfg)
    )
    sensors = list(cfg["sensor_columns"])
    prior = source_prior(
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
    initial_predictor._experiment26_model_seed = args.model_seed

    print("[features] encoding frozen source backbone")
    source_windows = {
        domain: exp251.frame_windows(normalized[domain], feature_names, cfg)
        for domain in cfg["source_domains"]
    }
    source_data = {
        domain: exp26.encode_data(model, data, device)
        for domain, data in source_windows.items()
    }
    episodes = sample_episodes(args, source_data)

    rows = []
    for inner_steps in args.inner_step_values:
        for inner_lr in args.inner_lrs:
            setting_rows = evaluate_setting(
                initial_predictor,
                episodes,
                inner_lr,
                inner_steps,
                device,
            )
            rows.extend(setting_rows)
            gains = np.asarray(
                [row["adaptation_gain"] for row in setting_rows],
                dtype=float,
            )
            print(
                f"[setting] inner_lr={inner_lr:g} inner_steps={inner_steps} "
                f"mean_gain={gains.mean():.4f} "
                f"positive_rate={(gains > 0).mean():.3f}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    summary, by_condition = summarize(raw)
    eligible = summary[summary["eligible"]]
    selected = None
    if not eligible.empty:
        best = eligible.iloc[0]
        selected = {
            "inner_lr": float(best["inner_lr"]),
            "inner_steps": int(best["inner_steps"]),
            "adaptation_gain_mean": float(best["adaptation_gain_mean"]),
            "relative_adaptation_gain_pct_mean": float(
                best["relative_adaptation_gain_pct_mean"]
            ),
            "positive_gain_rate": float(best["positive_gain_rate"]),
            "positive_condition_rate": float(best["positive_condition_rate"]),
        }
    report = {
        "script_version": SCRIPT_VERSION,
        "excluded_target": args.target,
        "model_seed": args.model_seed,
        "dry_run": bool(args.dry_run),
        "scope": "source_only_static_initialization_inner_loop_calibration",
        "target_train_files_accessed": False,
        "official_test_files_accessed": False,
        "outer_meta_training_run": False,
        "initialization_checkpoint": str(initialization_checkpoint),
        "source_condition_counts": source_condition_counts,
        "episodes_per_setting": args.episodes_per_setting,
        "inner_lrs": args.inner_lrs,
        "inner_step_values": args.inner_step_values,
        "eligibility_rule": {
            "mean_adaptation_gain_must_be_positive": True,
            "positive_episode_rate_at_least": 0.60,
            "positive_condition_rate_at_least": 0.80,
        },
        "selected_setting": selected,
        "decision": (
            "compare_seed42_and_seed43_then_run_experiment26-2"
            if selected is not None
            else "stop_experiment26_branch"
        ),
        "caveat": (
            "positive source adaptation gain is necessary but does not prove "
            "target-domain improvement"
        ),
    }

    prefix = f"experiment26-1_{args.target}_seed{args.model_seed}"
    atomic_write_text(output / f"{prefix}_summary.csv", summary.to_csv(index=False))
    atomic_write_text(
        output / f"{prefix}_by_condition.csv",
        by_condition.to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_raw.csv", raw.to_csv(index=False)
    )
    atomic_write_text(
        output / f"{prefix}_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(summary.to_string(index=False))
    print(f"[selected] {json.dumps(selected, ensure_ascii=False)}")
    print(f"[{SCRIPT_VERSION}] complete: {output}")


if __name__ == "__main__":
    main()
