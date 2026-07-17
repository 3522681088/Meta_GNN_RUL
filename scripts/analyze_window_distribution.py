"""Experiment 1: profile C-MAPSS window, RUL-stage, engine and condition balance.

This script is intentionally read-only with respect to model training.  It
reuses the project's loader, RUL label generator and exact window-end rule, and
writes diagnostic CSV/JSON files under ``outputs/window_distribution``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess import add_train_rul, load_domain  # noqa: E402


def window_end_indices(
    sequence_length,
    window_size=50,
    stride=5,
    last_only=False,
):
    """Return window ends without requiring changes to ``preprocess`` exports."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    effective_length = max(sequence_length, window_size)
    ends = [effective_length] if last_only else list(
        range(window_size, effective_length + 1, stride)
    )
    if not ends or ends[-1] != effective_length:
        ends.append(effective_length)
    return ends


OFFICIAL_CONDITION_COUNTS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
STAGES = (
    ("critical", "临近失效", 0.0, 30.0),
    ("middle", "中期退化", 30.0, 60.0),
    ("early", "早期退化", 60.0, 90.0),
    ("healthy", "健康/高RUL", 90.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计C-MAPSS滑动窗口的RUL阶段、发动机和工况分布（实验一）"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(OFFICIAL_CONDITION_COUNTS),
        default=list(OFFICIAL_CONDITION_COUNTS),
        help="需要统计的子数据集，默认FD001—FD004",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="覆盖配置文件中的data_dir；可指向所有txt所在目录",
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--rul-cap", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/window_distribution",
        help="CSV/JSON输出目录",
    )
    parser.add_argument(
        "--no-window-records",
        action="store_true",
        help="不保存逐窗口明细，以减小输出文件",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def stage_name(rul: float) -> tuple[str, str]:
    for key, chinese, lower, upper in STAGES:
        if lower <= rul <= upper:
            return key, chinese
    raise ValueError(f"Unable to assign RUL={rul} to a stage")


def infer_conditions(df: pd.DataFrame, domain: str, seed: int):
    """Estimate operating regimes from the three operational settings.

    Condition IDs are local to one domain and have no cross-domain semantic
    order.  The clustering is used only for diagnostics, never as a label for
    training in this experiment.
    """
    settings = df[SETTING_COLUMNS].to_numpy(dtype=np.float64)
    n_conditions = min(OFFICIAL_CONDITION_COUNTS[domain], len(settings))
    if n_conditions == 1:
        labels = np.zeros(len(df), dtype=np.int64)
        centers = settings.mean(axis=0, keepdims=True)
        return labels, centers

    scaler = StandardScaler().fit(settings)
    scaled = scaler.transform(settings)
    model = KMeans(n_clusters=n_conditions, random_state=seed, n_init=20)
    labels = model.fit_predict(scaled).astype(np.int64)
    centers = scaler.inverse_transform(model.cluster_centers_)

    # Make IDs deterministic and human-readable by sorting raw setting centers.
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels], centers[order]


def build_window_records(
    df: pd.DataFrame,
    condition_labels: np.ndarray,
    domain: str,
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    work = df.copy()
    work["condition_id"] = condition_labels
    records: list[dict] = []

    for unit, group in work.groupby("unit", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        length = len(group)
        pad = max(0, window_size - length)
        rul = group["rul"].to_numpy(dtype=np.float32)
        conditions = group["condition_id"].to_numpy(dtype=np.int64)
        if pad:
            rul = np.pad(rul, (pad, 0), mode="edge")
            conditions = np.pad(conditions, (pad, 0), mode="edge")

        ends = window_end_indices(length, window_size, stride, last_only=False)
        for window_number, end in enumerate(ends, start=1):
            segment_conditions = conditions[end - window_size : end]
            counts = np.bincount(
                segment_conditions,
                minlength=OFFICIAL_CONDITION_COUNTS[domain],
            )
            majority_condition = int(np.flatnonzero(counts == counts.max())[0])
            switches = int(np.count_nonzero(np.diff(segment_conditions)))
            end_position = min(end - pad - 1, length - 1)
            start_position = max(0, end_position - window_size + 1)
            label = float(rul[end - 1])
            stage, stage_chinese = stage_name(label)
            records.append(
                {
                    "domain": domain,
                    "unit": int(unit),
                    "window_number": window_number,
                    "start_cycle": int(group.iloc[start_position]["cycle"]),
                    "end_cycle": int(group.iloc[end_position]["cycle"]),
                    "rul": label,
                    "stage": stage,
                    "stage_chinese": stage_chinese,
                    "endpoint_condition": int(segment_conditions[-1]),
                    "majority_condition": majority_condition,
                    "condition_count": int(np.count_nonzero(counts)),
                    "condition_switches": switches,
                    "contains_condition_switch": bool(switches > 0),
                    "left_padding": pad if window_number == 1 else 0,
                    "engine_cycles": length,
                }
            )
    return pd.DataFrame.from_records(records)


def stage_summary(records: pd.DataFrame, domain: str, rul_cap: float) -> pd.DataFrame:
    rows = []
    total = len(records)
    for key, chinese, lower, upper in STAGES:
        part = records[records["stage"] == key]
        display_upper = rul_cap if np.isinf(upper) else upper
        rows.append(
            {
                "domain": domain,
                "stage": key,
                "stage_chinese": chinese,
                "rul_lower": lower,
                "rul_upper": display_upper,
                "window_count": int(len(part)),
                "percentage": float(100.0 * len(part) / total) if total else 0.0,
                "engine_count": int(part["unit"].nunique()),
                "mean_rul": float(part["rul"].mean()) if len(part) else None,
                "min_rul": float(part["rul"].min()) if len(part) else None,
                "max_rul": float(part["rul"].max()) if len(part) else None,
            }
        )
    return pd.DataFrame(rows)


def engine_summary(records: pd.DataFrame) -> pd.DataFrame:
    base = records.groupby(["domain", "unit"], as_index=False).agg(
        engine_cycles=("engine_cycles", "first"),
        total_windows=("window_number", "count"),
        min_rul=("rul", "min"),
        max_rul=("rul", "max"),
        mixed_condition_windows=("contains_condition_switch", "sum"),
    )
    counts = (
        records.pivot_table(
            index=["domain", "unit"],
            columns="stage",
            values="window_number",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda value: f"{value}_windows")
        .reset_index()
    )
    result = base.merge(counts, on=["domain", "unit"], how="left")
    for key, *_ in STAGES:
        column = f"{key}_windows"
        if column not in result:
            result[column] = 0
        result[f"{key}_percentage"] = 100.0 * result[column] / result["total_windows"]
    return result


def condition_summary(
    df: pd.DataFrame,
    condition_labels: np.ndarray,
    records: pd.DataFrame,
    domain: str,
) -> pd.DataFrame:
    cycle_counts = pd.Series(condition_labels).value_counts().to_dict()
    endpoint_counts = records["endpoint_condition"].value_counts().to_dict()
    majority_counts = records["majority_condition"].value_counts().to_dict()
    rows = []
    for condition in range(OFFICIAL_CONDITION_COUNTS[domain]):
        rows.append(
            {
                "domain": domain,
                "condition_id": condition,
                "cycle_count": int(cycle_counts.get(condition, 0)),
                "cycle_percentage": float(
                    100.0 * cycle_counts.get(condition, 0) / len(df)
                ),
                "endpoint_window_count": int(endpoint_counts.get(condition, 0)),
                "majority_window_count": int(majority_counts.get(condition, 0)),
            }
        )
    return pd.DataFrame(rows)


def exact_rul_histogram(records: pd.DataFrame) -> pd.DataFrame:
    result = (
        records.assign(rul=records["rul"].round().astype(int))
        .groupby(["domain", "rul"], as_index=False)
        .agg(window_count=("window_number", "count"), engine_count=("unit", "nunique"))
    )
    totals = result.groupby("domain")["window_count"].transform("sum")
    result["percentage"] = 100.0 * result["window_count"] / totals
    return result


def print_domain_summary(summary: pd.DataFrame, domain: str) -> None:
    print(f"\n[{domain}] RUL阶段窗口分布")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.stage_chinese:<10} "
            f"windows={row.window_count:>6}  percentage={row.percentage:6.2f}%  "
            f"engines={row.engine_count:>3}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config["data_dir"]
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    window_size = args.window_size or int(config["window_size"])
    stride = args.window_stride or int(config["window_stride"])
    rul_cap = args.rul_cap if args.rul_cap is not None else float(config["rul_cap"])
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_stages = []
    all_engines = []
    all_conditions = []
    all_centers = []

    for domain in args.domains:
        train, _, _ = load_domain(str(data_path), domain)
        train = add_train_rul(train, rul_cap)
        condition_labels, centers = infer_conditions(train, domain, args.seed)
        records = build_window_records(
            train,
            condition_labels,
            domain,
            window_size,
            stride,
        )
        stages = stage_summary(records, domain, rul_cap)
        engines = engine_summary(records)
        conditions = condition_summary(train, condition_labels, records, domain)
        centers_frame = pd.DataFrame(centers, columns=SETTING_COLUMNS)
        centers_frame.insert(0, "condition_id", np.arange(len(centers_frame)))
        centers_frame.insert(0, "domain", domain)

        all_records.append(records)
        all_stages.append(stages)
        all_engines.append(engines)
        all_conditions.append(conditions)
        all_centers.append(centers_frame)
        print_domain_summary(stages, domain)

    records = pd.concat(all_records, ignore_index=True)
    stages = pd.concat(all_stages, ignore_index=True)
    engines = pd.concat(all_engines, ignore_index=True)
    conditions = pd.concat(all_conditions, ignore_index=True)
    centers = pd.concat(all_centers, ignore_index=True)
    histogram = exact_rul_histogram(records)

    stages.to_csv(output_dir / "rul_stage_summary.csv", index=False, encoding="utf-8-sig")
    engines.to_csv(output_dir / "engine_window_summary.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output_dir / "condition_window_summary.csv", index=False, encoding="utf-8-sig")
    centers.to_csv(output_dir / "condition_centroids.csv", index=False, encoding="utf-8-sig")
    histogram.to_csv(output_dir / "rul_histogram.csv", index=False, encoding="utf-8-sig")
    if not args.no_window_records:
        records.to_csv(output_dir / "window_records.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "experiment": "window_distribution",
        "domains": args.domains,
        "data_dir": str(data_path),
        "window_size": window_size,
        "window_stride": stride,
        "window_overlap_ratio": 1.0 - stride / window_size,
        "rul_cap": rul_cap,
        "stage_boundaries": {"critical": 30, "middle": 60, "early": 90},
        "condition_method": "KMeans on standardized setting1-setting3; IDs are local per domain",
        "seed": args.seed,
        "files": sorted(path.name for path in output_dir.glob("*")),
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n统计完成，结果保存在：{output_dir}")


if __name__ == "__main__":
    main()
