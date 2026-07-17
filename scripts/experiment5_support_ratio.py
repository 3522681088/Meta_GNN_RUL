"""实验5：目标域标注比例与少样本优势曲线。

默认在固定的工况归一化和采样协议下，对Meta-GNN与普通GNN比较
5%、10%、20%和80%目标域标注发动机。这样可以回答：

1. 标注发动机增加后，预测误差是否下降；
2. Meta-GNN是否在5%和10%的少样本区域保持更明显优势；
3. 数据充足时，普通监督GNN是否逐渐追上Meta-GNN。

每次训练结束后都会原子保存原始JSON和汇总CSV，中断后已完成结果仍会保留。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_condition_aware_experiment import (
    BALANCE_MODES,
    PREPROCESSING_MODES,
    PROJECT_ROOT,
    inspect_loaders,
    load_config,
    prepare_custom_experiment,
    resolve_path,
    run_one,
)
import main as main_module


MODEL_CHOICES = (
    "lstm",
    "cnn_lstm",
    "transformer",
    "gnn",
    "reptile_lstm",
    "meta_gnn",
    "no_gat",
    "no_attention",
)
META_MODELS = {"reptile_lstm", "meta_gnn", "no_gat", "no_attention"}
METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验5：多标注比例、多模型、多随机种子的少样本曲线"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["meta_gnn", "gnn"],
        help="默认比较Meta-GNN与普通GNN",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="兼容旧命令；设置后将覆盖--models，只运行一个模型",
    )
    parser.add_argument(
        "--target",
        default="FD004",
        choices=[f"FD00{i}" for i in range(1, 5)],
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.8],
        help="目标域有标签发动机占全部目标训练发动机的比例",
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default="condition_norm",
    )
    parser.add_argument(
        "--balance-modes",
        nargs="+",
        choices=BALANCE_MODES,
        default=["none"],
        help="主实验建议固定一个采样协议；可额外传入stage做敏感性比较",
    )
    parser.add_argument("--support-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--adapt-epochs", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment5_support_ratio",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summarize(results: list[dict]) -> pd.DataFrame:
    """对相同模型、比例、预处理和采样方式进行多种子汇总。"""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    group_columns = [
        "support_ratio",
        "model",
        "training_regime",
        "preprocessing_mode",
        "balance_mode",
    ]
    summary = frame.groupby(group_columns, as_index=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    summary = summary.rename(columns={"rmse_count": "n_runs"})
    duplicate_counts = [f"{metric}_count" for metric in METRICS if metric != "rmse"]
    summary = summary.drop(columns=[c for c in duplicate_counts if c in summary.columns])
    return summary.sort_values(["support_ratio", "balance_mode", "rmse_mean"]).reset_index(
        drop=True
    )


def meta_advantage(summary: pd.DataFrame) -> pd.DataFrame:
    """在同一比例和采样协议下计算Meta-GNN相对GNN的改变量。"""
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["support_ratio", "preprocessing_mode", "balance_mode"]
    for group_key, group in summary.groupby(keys):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        if "meta_gnn" not in by_model or "gnn" not in by_model:
            continue
        meta = by_model["meta_gnn"]
        gnn = by_model["gnn"]
        ratio, preprocessing, balance = group_key
        rows.append(
            {
                "support_ratio": ratio,
                "preprocessing_mode": preprocessing,
                "balance_mode": balance,
                "rmse_change_pct_meta_vs_gnn": 100.0
                * (meta["rmse_mean"] - gnn["rmse_mean"])
                / gnn["rmse_mean"],
                "mae_change_pct_meta_vs_gnn": 100.0
                * (meta["mae_mean"] - gnn["mae_mean"])
                / gnn["mae_mean"],
                "r2_delta_meta_vs_gnn": meta["r2_mean"] - gnn["r2_mean"],
                "nasa_change_pct_meta_vs_gnn": 100.0
                * (meta["nasa_score_mean"] - gnn["nasa_score_mean"])
                / gnn["nasa_score_mean"],
            }
        )
    return pd.DataFrame(rows).sort_values("support_ratio") if rows else pd.DataFrame()


def atomic_write_text(path: Path, text: str, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def save_progress(results: list[dict], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"experiment5_raw_{args.target}.json"
    summary_path = output / f"experiment5_summary_{args.target}.csv"
    advantage_path = output / f"experiment5_meta_advantage_{args.target}.csv"

    atomic_write_text(
        raw_path,
        json.dumps(results, ensure_ascii=False, indent=2),
        "utf-8",
    )
    summary = summarize(results)
    atomic_write_text(summary_path, summary.to_csv(index=False), "utf-8-sig")
    advantage = meta_advantage(summary)
    atomic_write_text(advantage_path, advantage.to_csv(index=False), "utf-8-sig")
    return raw_path, summary_path, advantage_path


def main() -> None:
    args = parse_args()
    ratios = list(dict.fromkeys(args.ratios))
    if any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("--ratios中的每个值都必须位于(0, 1]区间")
    seeds = list(dict.fromkeys(args.seeds or [args.seed]))
    models = [args.model] if args.model is not None else list(dict.fromkeys(args.models))
    balance_modes = list(dict.fromkeys(args.balance_modes))

    main_module.META_MODELS.update(META_MODELS)

    results: list[dict] = []
    paths: tuple[Path, Path, Path] | None = None
    for seed in seeds:
        for ratio in ratios:
            for balance_mode in balance_modes:
                for model_name in models:
                    args.support_ratio = ratio
                    cfg = load_config(args, seed)
                    label = f"{model_name}_r{ratio:g}_{args.preprocessing}_{balance_mode}"
                    loaders = prepare_custom_experiment(
                        cfg,
                        args.preprocessing,
                        balance_mode,
                        label,
                    )
                    regime = (
                        "reptile_meta" if model_name in META_MODELS else "target_supervised"
                    )
                    print(
                        f"\n[experiment5] seed={seed} ratio={ratio:g} "
                        f"model={model_name} regime={regime} "
                        f"preprocessing={args.preprocessing} balance={balance_mode}"
                    )
                    if args.dry_run:
                        inspect_loaders(label, cfg, model_name, loaders)
                        continue

                    ratio_tag = str(ratio).replace(".", "p")
                    metrics = run_one(
                        model_name,
                        cfg,
                        loaders,
                        tag=(
                            f"exp5_{model_name}_r{ratio_tag}_"
                            f"{args.preprocessing}_{balance_mode}"
                        ),
                    )
                    metrics.update(
                        {
                            "training_regime": regime,
                            "support_ratio": ratio,
                            "preprocessing_mode": args.preprocessing,
                            "balance_mode": balance_mode,
                        }
                    )
                    results.append(metrics)
                    paths = save_progress(results, args)

    if args.dry_run:
        print("\n[dry-run完成] 未训练模型，也不会生成正式结果文件。")
        return

    assert paths is not None
    summary = summarize(results)
    advantage = meta_advantage(summary)
    print("\n[experiment5 summary]")
    print(summary.to_string(index=False))
    if not advantage.empty:
        print("\n[Meta-GNN relative advantage vs GNN]")
        print(advantage.to_string(index=False))
    print(f"\nRaw: {paths[0]}\nSummary: {paths[1]}\nAdvantage: {paths[2]}")


if __name__ == "__main__":
    main()
