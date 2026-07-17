"""Single-file condition-aware C-MAPSS experiment for Meta-GNN-RUL.

This file is an additional experiment entry point.  It does not replace
``main.py`` or any existing preprocessing/model module.

Modes
-----
original
    Source-global sensor Z-score, original 14 sensor inputs, random windows.
condition
    Source-only operating-condition sensor Z-score plus the three normalized
    operating settings as model inputs.
condition_balanced
    ``condition`` preprocessing plus engine/RUL-stage balanced sampling.

Examples (run from the project root)
------------------------------------
Quick compatibility check without training::

    python scripts/run_condition_aware_experiment.py --compare --dry-run

One-seed comparison::

    python scripts/run_condition_aware_experiment.py --compare --target FD004 \
        --support-ratio 0.05 --seed 42

Recommended final comparison::

    python scripts/run_condition_aware_experiment.py --compare --target FD004 \
        --support-ratio 0.05 --seeds 42 52 62
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from main import run_one  # noqa: E402
from preprocess.cmapps_loader import load_domain  # noqa: E402
from preprocess.rul_generator import add_test_rul, add_train_rul  # noqa: E402
from preprocess.window_dataset import WindowDataset, make_windows  # noqa: E402


SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
SETTING_FEATURE_COLUMNS = ["op_setting1", "op_setting2", "op_setting3"]
MODES = ("original", "condition", "condition_balanced")
PREPROCESSING_MODES = (
    "global",
    "global_settings",
    "condition_norm",
    "condition_settings",
)
BALANCE_MODES = (
    "none",
    "stage",
    "engine",
    "engine_stage",
    "sqrt_engine_stage",
)
LOWER_IS_BETTER = ("rmse", "mae", "nasa_score")


def safe_column_assignment(
    frame: pd.DataFrame,
    columns: Iterable[str],
    values: np.ndarray,
) -> pd.DataFrame:
    """Replace columns safely under both pandas 2.x and pandas 3.x."""
    out = frame.copy()
    for index, column in enumerate(columns):
        out[column] = values[:, index].astype(np.float32, copy=False)
    return out


class SourceGlobalNormalizer:
    """Reproduce the original source-global Z-score without pandas dtype issues."""

    def __init__(self, include_settings: bool = False):
        self.include_settings = include_settings

    def fit(self, source: pd.DataFrame, sensors: list[str]):
        values = source[sensors].to_numpy(dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        if self.include_settings:
            raw_settings = source[SETTING_COLUMNS].to_numpy(dtype=np.float64)
            self.setting_scaler = StandardScaler().fit(raw_settings)
        return self

    def transform(self, frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        values = frame[sensors].to_numpy(dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        out = safe_column_assignment(frame, sensors, normalized)
        if self.include_settings:
            raw_settings = frame[SETTING_COLUMNS].to_numpy(dtype=np.float64)
            scaled_settings = self.setting_scaler.transform(raw_settings)
            out = safe_column_assignment(out, SETTING_FEATURE_COLUMNS, scaled_settings)
        return out


class SourceConditionNormalizer:
    """Normalize sensors by operating regime learned only from source training data.

    The target RUL labels and target sensor statistics are never used during
    fitting.  Target operating settings are only used to select one of the
    source-fitted condition statistics at transformation time.
    """

    def __init__(
        self,
        n_conditions: int = 6,
        seed: int = 42,
        include_settings: bool = True,
    ):
        self.n_conditions = n_conditions
        self.seed = seed
        self.include_settings = include_settings

    def fit(self, source: pd.DataFrame, sensors: list[str]):
        raw_settings = source[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        self.setting_scaler = StandardScaler().fit(raw_settings)
        scaled_settings = self.setting_scaler.transform(raw_settings)
        self.clusterer = KMeans(
            n_clusters=self.n_conditions,
            random_state=self.seed,
            n_init=20,
        ).fit(scaled_settings)

        labels = self.clusterer.labels_
        sensor_values = source[sensors].to_numpy(dtype=np.float64)
        global_mean = sensor_values.mean(axis=0)
        global_scale = sensor_values.std(axis=0)
        global_scale[global_scale < 1e-8] = 1.0

        means = []
        scales = []
        counts = []
        for condition in range(self.n_conditions):
            selected = sensor_values[labels == condition]
            counts.append(int(len(selected)))
            if len(selected) < 2:
                means.append(global_mean)
                scales.append(global_scale)
                continue
            condition_mean = selected.mean(axis=0)
            condition_scale = selected.std(axis=0)
            condition_scale[condition_scale < 1e-8] = 1.0
            means.append(condition_mean)
            scales.append(condition_scale)
        self.sensor_mean = np.stack(means)
        self.sensor_scale = np.stack(scales)
        self.source_condition_counts = counts
        return self

    def transform(self, frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        raw_settings = frame[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        scaled_settings = self.setting_scaler.transform(raw_settings)
        labels = self.clusterer.predict(scaled_settings)
        sensor_values = frame[sensors].to_numpy(dtype=np.float64)
        normalized = (
            sensor_values - self.sensor_mean[labels]
        ) / self.sensor_scale[labels]

        out = safe_column_assignment(frame, sensors, normalized)
        if self.include_settings:
            out = safe_column_assignment(out, SETTING_FEATURE_COLUMNS, scaled_settings)
        out["condition_id"] = labels.astype(np.int64)
        return out


def split_units(df: pd.DataFrame, val_fraction: float, seed: int):
    """Use the exact engine-level split rule from the original project."""
    rng = np.random.default_rng(seed)
    units = np.asarray(sorted(df["unit"].unique()))
    rng.shuffle(units)
    validation_count = max(1, int(round(len(units) * val_fraction)))
    return units[validation_count:], units[:validation_count]


def rul_stage_ids(labels: np.ndarray) -> np.ndarray:
    """Map RUL to critical/middle/early/high-RUL stages 0..3."""
    return np.digitize(labels, bins=[30.0, 60.0, 90.0], right=True)


def engine_stage_weights(labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Give each engine equal total mass and each present stage equal mass.

    For engine u and RUL stage s, every window receives

        1 / (number_of_stages_for_u * windows_for_u_in_s)

    Consequently, long-life engines no longer dominate merely because they
    produce more overlapping windows, and high-RUL windows no longer dominate
    the sampling distribution within one engine.
    """
    labels = np.asarray(labels, dtype=np.float64)
    units = np.asarray(units)
    stages = rul_stage_ids(labels)
    weights = np.zeros(len(labels), dtype=np.float64)

    for unit in np.unique(units):
        unit_mask = units == unit
        present_stages = np.unique(stages[unit_mask])
        for stage in present_stages:
            mask = unit_mask & (stages == stage)
            weights[mask] = 1.0 / (len(present_stages) * mask.sum())

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Unable to construct positive finite sampling weights")
    return weights


def sampling_weights(
    labels: np.ndarray,
    units: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build stage-, engine-, joint-, or softly balanced window weights."""
    if mode not in BALANCE_MODES or mode == "none":
        raise ValueError(f"A weighted mode is required, got: {mode}")
    labels = np.asarray(labels, dtype=np.float64)
    units = np.asarray(units)
    stages = rul_stage_ids(labels)

    if mode == "stage":
        weights = np.zeros(len(labels), dtype=np.float64)
        for stage in np.unique(stages):
            mask = stages == stage
            weights[mask] = 1.0 / mask.sum()
    elif mode == "engine":
        weights = np.zeros(len(labels), dtype=np.float64)
        for unit in np.unique(units):
            mask = units == unit
            weights[mask] = 1.0 / mask.sum()
    else:
        weights = engine_stage_weights(labels, units)
        if mode == "sqrt_engine_stage":
            weights = np.sqrt(weights)

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Unable to construct positive finite sampling weights")
    return weights


def loader_from_df(
    df: pd.DataFrame,
    features: list[str],
    cfg: dict,
    *,
    training: bool,
    balanced: bool = False,
    balance_mode: str | None = None,
    last_only: bool = False,
    seed_offset: int = 0,
) -> DataLoader:
    x, y, units = make_windows(
        df,
        features,
        cfg["window_size"],
        cfg["window_stride"],
        last_only,
    )
    dataset = WindowDataset(x, y, units)

    selected_balance = balance_mode or ("engine_stage" if balanced else "none")
    if selected_balance not in BALANCE_MODES:
        raise ValueError(f"Unknown balance mode: {selected_balance}")
    if training and selected_balance != "none":
        weights = sampling_weights(y, units, selected_balance)
        generator = torch.Generator().manual_seed(cfg["seed"] + seed_offset)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=training,
        drop_last=False,
    )


def prepare_custom_experiment(
    cfg: dict,
    preprocessing_mode: str,
    balance_mode: str = "none",
    experiment_label: str | None = None,
):
    """Prepare any preprocessing/sampling combination with identical unit splits."""
    if preprocessing_mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing_mode}")
    if balance_mode not in BALANCE_MODES:
        raise ValueError(f"Unknown balance mode: {balance_mode}")

    domains = list(dict.fromkeys(cfg["source_domains"] + [cfg["target_domain"]]))
    raw: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for domain in domains:
        train, test, final_rul = load_domain(cfg["data_dir"], domain)
        raw[domain] = (
            add_train_rul(train, cfg["rul_cap"]),
            add_test_rul(test, final_rul, cfg["rul_cap"]),
        )

    sensors = list(cfg["sensor_columns"])
    source_fit = pd.concat(
        [raw[domain][0] for domain in cfg["source_domains"]],
        ignore_index=True,
    )

    condition_aware = preprocessing_mode in {"condition_norm", "condition_settings"}
    include_settings = preprocessing_mode in {"global_settings", "condition_settings"}
    if condition_aware:
        normalizer = SourceConditionNormalizer(
            n_conditions=cfg.get("condition_count", 6),
            seed=cfg["seed"],
            include_settings=include_settings,
        ).fit(source_fit, sensors)
    else:
        normalizer = SourceGlobalNormalizer(
            include_settings=include_settings,
        ).fit(source_fit, sensors)
    features = sensors + SETTING_FEATURE_COLUMNS if include_settings else sensors

    normalized = {
        domain: (
            normalizer.transform(train, sensors),
            normalizer.transform(test, sensors),
        )
        for domain, (train, test) in raw.items()
    }

    task_loaders = {}
    for domain_index, domain in enumerate(cfg["source_domains"]):
        train_units, _ = split_units(
            normalized[domain][0],
            cfg["validation_fraction"],
            cfg["seed"],
        )
        source_train = normalized[domain][0].query("unit in @train_units")
        task_loaders[domain] = loader_from_df(
            source_train,
            features,
            cfg,
            training=True,
            balance_mode=balance_mode,
            seed_offset=1000 * (domain_index + 1),
        )

    target_train, target_test = normalized[cfg["target_domain"]]
    target_units = np.asarray(sorted(target_train["unit"].unique()))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(target_units)
    support_ratio = cfg.get("target_support_ratio")
    requested = (
        max(2, int(round(len(target_units) * support_ratio)))
        if support_ratio is not None
        else cfg["target_support_units"]
    )
    labeled_count = min(requested, len(target_units))
    labeled_units = target_units[:labeled_count]

    if len(labeled_units) > 1:
        validation_count = max(
            1,
            int(round(len(labeled_units) * cfg["validation_fraction"])),
        )
        validation_units = labeled_units[:validation_count]
        adaptation_units = labeled_units[validation_count:]
    else:
        adaptation_units = validation_units = labeled_units

    support = loader_from_df(
        target_train.query("unit in @adaptation_units"),
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        seed_offset=9000,
    )
    validation = loader_from_df(
        target_train.query("unit in @validation_units"),
        features,
        cfg,
        training=False,
    )
    test = loader_from_df(
        target_test,
        features,
        cfg,
        training=False,
        last_only=True,
    )

    split_info = {
        "experiment_label": experiment_label or preprocessing_mode,
        "preprocessing_mode": preprocessing_mode,
        "feature_columns": features,
        "labeled_target_units": labeled_units.tolist(),
        "adaptation_units": adaptation_units.tolist(),
        "validation_units": validation_units.tolist(),
        "balanced_sampling": balance_mode != "none",
        "balance_mode": balance_mode,
        "normalizer_fit_scope": "source_train_only",
    }
    if condition_aware:
        split_info["source_condition_counts"] = normalizer.source_condition_counts

    return (
        task_loaders,
        support,
        validation,
        test,
        len(features),
        split_info,
    )


def prepare_experiment(cfg: dict, mode: str):
    """Backward-compatible three-mode entry point used by Experiment 1."""
    mapping = {
        "original": ("global", "none"),
        "condition": ("condition_settings", "none"),
        "condition_balanced": ("condition_settings", "engine_stage"),
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    preprocessing_mode, balance_mode = mapping[mode]
    return prepare_custom_experiment(
        cfg,
        preprocessing_mode,
        balance_mode,
        experiment_label=mode,
    )


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain for domain in ("FD001", "FD002", "FD003", "FD004")
        if domain != args.target
    ]
    if args.support_ratio is not None:
        if not 0 < args.support_ratio <= 1:
            raise ValueError("--support-ratio must be in (0, 1]")
        cfg["target_support_ratio"] = args.support_ratio
    if args.meta_epochs is not None:
        cfg["meta_epochs"] = args.meta_epochs
    if args.adapt_epochs is not None:
        cfg["adapt_epochs"] = args.adapt_epochs
    if args.inner_lr is not None:
        cfg["inner_lr"] = args.inner_lr
    if args.outer_lr is not None:
        cfg["outer_lr"] = args.outer_lr
    if args.pair_aux_weight is not None:
        cfg["pair_aux_weight"] = args.pair_aux_weight

    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir, PROJECT_ROOT))
    cfg["output_dir"] = str(resolve_path(args.output_dir, PROJECT_ROOT))
    cfg["condition_count"] = args.condition_count
    return cfg


def stage_distribution_from_loader(loader: DataLoader) -> dict[str, float]:
    """Return raw or expected sampled stage percentages for diagnostics."""
    labels = loader.dataset.y.detach().cpu().numpy()
    stages = rul_stage_ids(labels)
    if isinstance(loader.sampler, WeightedRandomSampler):
        probabilities = loader.sampler.weights.detach().cpu().numpy().astype(float)
        probabilities /= probabilities.sum()
        counts = np.asarray(
            [probabilities[stages == stage].sum() for stage in range(4)]
        )
    else:
        counts = np.asarray([(stages == stage).mean() for stage in range(4)])
    names = ("critical", "middle", "early", "high_rul")
    return {name: round(float(value * 100), 2) for name, value in zip(names, counts)}


def inspect_loaders(label: str, cfg: dict, model_name: str, loaders):
    """Validate one prepared experiment through a real model forward pass."""
    tasks, support, validation, test, feature_count, split_info = loaders
    example_loader = tasks[cfg["source_domains"][0]]
    x, y = next(iter(example_loader))
    model = build_model(model_name, feature_count, cfg).cpu().eval()
    with torch.no_grad():
        prediction = model(x[: min(8, len(x))])
    diagnostic = {
        "variant": label,
        "feature_count": feature_count,
        "example_shape": list(x.shape),
        "forward_output_shape": list(prediction.shape),
        "support_windows": len(support.dataset),
        "validation_windows": len(validation.dataset),
        "test_engines": len(test.dataset),
        "support_stage_distribution_pct": stage_distribution_from_loader(support),
        "split": split_info,
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return diagnostic


def dry_run(mode: str, cfg: dict, model_name: str):
    loaders = prepare_experiment(cfg, mode)
    return inspect_loaders(mode, cfg, model_name, loaders)


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    metrics = ["rmse", "mae", "r2", "nasa_score"]
    aggregate = (
        frame.groupby(["mode", "model", "target_domain"], as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    aggregate = aggregate.rename(
        columns={
            "mode_": "mode",
            "model_": "model",
            "target_domain_": "target_domain",
        }
    )
    for metric in metrics:
        aggregate[f"{metric}_std"] = aggregate[f"{metric}_std"].fillna(0.0)

    if "original" in set(aggregate["mode"]):
        baseline = aggregate.loc[aggregate["mode"] == "original"].iloc[0]
        for metric in LOWER_IS_BETTER:
            aggregate[f"{metric}_change_pct_vs_original"] = (
                100.0
                * (aggregate[f"{metric}_mean"] - baseline[f"{metric}_mean"])
                / baseline[f"{metric}_mean"]
            )
        aggregate["r2_delta_vs_original"] = (
            aggregate["r2_mean"] - baseline["r2_mean"]
        )
        aggregate["overall_verdict"] = np.where(
            (aggregate["rmse_change_pct_vs_original"] < 0)
            & (aggregate["nasa_score_change_pct_vs_original"] < 0)
            & (aggregate["r2_delta_vs_original"] > 0),
            "improved",
            np.where(aggregate["mode"] == "original", "reference", "mixed_or_worse"),
        )
    return aggregate


def print_evaluation_guide(seed_count: int):
    print("\n[how to judge]")
    print("1. rmse/mae/nasa_score_change_pct_vs_original < 0 表示改善。")
    print("2. r2_delta_vs_original > 0 表示改善，R²由负变正尤其重要。")
    print("3. condition优于original：说明工况混合是主要问题之一。")
    print("4. condition_balanced再优于condition：说明窗口/发动机失衡也有影响。")
    print("5. RMSE降低但NASA Score升高不算全面改善，表示仍存在严重偏晚预测。")
    if seed_count < 3:
        print("6. 当前少于3个随机种子，只能作为初步结果；最终结论建议至少3个种子。")
    else:
        print("6. 检查raw JSON中各随机种子的改善方向是否一致，并结合mean±std判断稳定性。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C-MAPSS工况感知归一化与发动机/RUL阶段均衡采样实验"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="覆盖配置文件中的data_dir，可指向全部C-MAPSS txt所在目录",
    )
    parser.add_argument("--model", default="meta_gnn")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="condition_balanced",
        help="单独运行一种预处理/采样方案",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="依次运行original、condition和condition_balanced",
    )
    parser.add_argument(
        "--target",
        choices=("FD001", "FD002", "FD003", "FD004"),
        default="FD004",
    )
    parser.add_argument("--support-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="多随机种子，例如 --seeds 42 52 62",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--adapt-epochs", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument(
        "--output-dir",
        default="outputs/condition_aware",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查数据、采样器和模型前向传播，不执行训练",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]
    modes = list(MODES) if args.compare else [args.mode]

    if args.dry_run:
        for seed in seeds:
            for mode in modes:
                cfg = load_config(args, seed)
                print(f"\n[dry-run] seed={seed} mode={mode}")
                dry_run(mode, cfg, args.model)
        return

    results = []
    for seed in seeds:
        for mode in modes:
            cfg = load_config(args, seed)
            print(f"\n[training] seed={seed} mode={mode}")
            loaders = prepare_experiment(cfg, mode)
            tag = f"{mode}_{args.model}"
            metrics = run_one(args.model, cfg, loaders=loaders, tag=tag)
            metrics["mode"] = mode
            results.append(metrics)

    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"condition_aware_raw_{args.model}_{args.target}.json"
    raw_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregate = aggregate_results(results)
    summary_path = output_dir / f"condition_aware_summary_{args.model}_{args.target}.csv"
    aggregate.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n[summary]")
    print(aggregate.to_string(index=False))
    print_evaluation_guide(len(seeds))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
