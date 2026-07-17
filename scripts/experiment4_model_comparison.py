"""实验4：相同数据协议下的完整模型比较与模块消融。

本脚本一次支持：
1. 传统基线：LSTM、CNN-LSTM、Transformer；
2. 图模型：GNN；
3. 元学习模型：Reptile-LSTM、Meta-GNN；
4. Meta-GNN消融：no_gat、no_attention。

所有模型共享相同的目标发动机划分、预处理、采样策略和随机种子。
每完成一次训练都会立即以“临时文件 + 原子替换”的方式保存JSON和CSV，
避免长时间实验被中断后只留下0字节文件。
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


BASE_MODELS = (
    "lstm",
    "cnn_lstm",
    "transformer",
    "gnn",
    "reptile_lstm",
    "meta_gnn",
)
ABLATION_MODELS = ("no_gat", "no_attention")
ALL_MODELS = BASE_MODELS + ABLATION_MODELS
META_MODELS = {"reptile_lstm", "meta_gnn", "no_gat", "no_attention"}
METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验4：相同预处理、采样与目标划分下的完整模型公平比较"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(BASE_MODELS),
        help="需要正式训练的模型；默认运行6个基础/完整模型",
    )
    parser.add_argument(
        "--include-meta-ablation",
        action="store_true",
        help="在--models基础上追加no_gat和no_attention",
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default="condition_norm",
        help="实验2候选最优为condition_norm",
    )
    parser.add_argument(
        "--balance-mode",
        choices=BALANCE_MODES,
        default="stage",
        help="实验3中stage偏向RMSE，sqrt_engine_stage偏向NASA Score",
    )
    parser.add_argument(
        "--target",
        default="FD004",
        choices=[f"FD00{i}" for i in range(1, 5)],
    )
    parser.add_argument("--support-ratio", type=float, default=0.05)
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
        default="outputs/experiment4_model_comparison",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summarize(results: list[dict]) -> pd.DataFrame:
    """按模型汇总多随机种子的均值、标准差和有效运行次数。"""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    group_columns = [
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
    return summary.sort_values("rmse_mean").reset_index(drop=True)


def pairwise_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """生成预先定义的成对比较，正负方向均以候选模型相对参考模型表示。"""
    if summary.empty:
        return pd.DataFrame()
    by_model = {row["model"]: row for _, row in summary.iterrows()}
    comparisons = (
        ("meta_gnn", "gnn", "Reptile对图模型的贡献"),
        ("reptile_lstm", "lstm", "Reptile对LSTM的贡献"),
        ("meta_gnn", "reptile_lstm", "完整图结构相对元学习LSTM"),
        ("meta_gnn", "no_gat", "GAT模块贡献"),
        ("meta_gnn", "no_attention", "多头自注意力模块贡献"),
    )
    rows: list[dict] = []
    for candidate, reference, purpose in comparisons:
        if candidate not in by_model or reference not in by_model:
            continue
        cand = by_model[candidate]
        ref = by_model[reference]
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "purpose": purpose,
                "rmse_change_pct": 100.0
                * (cand["rmse_mean"] - ref["rmse_mean"])
                / ref["rmse_mean"],
                "mae_change_pct": 100.0
                * (cand["mae_mean"] - ref["mae_mean"])
                / ref["mae_mean"],
                "r2_delta": cand["r2_mean"] - ref["r2_mean"],
                "nasa_score_change_pct": 100.0
                * (cand["nasa_score_mean"] - ref["nasa_score_mean"])
                / ref["nasa_score_mean"],
            }
        )
    return pd.DataFrame(rows)


def atomic_write_text(path: Path, text: str, encoding: str) -> None:
    """先写临时文件再替换，防止中断产生0字节正式结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def save_progress(results: list[dict], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"experiment4_raw_{args.target}.json"
    summary_path = output / f"experiment4_summary_{args.target}.csv"
    effects_path = output / f"experiment4_pairwise_{args.target}.csv"

    atomic_write_text(
        raw_path,
        json.dumps(results, ensure_ascii=False, indent=2),
        "utf-8",
    )
    summary = summarize(results)
    atomic_write_text(summary_path, summary.to_csv(index=False), "utf-8-sig")
    effects = pairwise_effects(summary)
    atomic_write_text(effects_path, effects.to_csv(index=False), "utf-8-sig")
    return raw_path, summary_path, effects_path


def main() -> None:
    args = parse_args()
    if not 0 < args.support_ratio <= 1:
        raise ValueError("--support-ratio必须位于(0, 1]区间")

    seeds = list(dict.fromkeys(args.seeds or [args.seed]))
    models = list(dict.fromkeys(args.models))
    if args.include_meta_ablation:
        models.extend(name for name in ABLATION_MODELS if name not in models)

    # no_gat/no_attention必须进入Reptile分支，避免被误当成普通监督模型。
    main_module.META_MODELS.update(META_MODELS)

    results: list[dict] = []
    paths: tuple[Path, Path, Path] | None = None
    for seed in seeds:
        for model_name in models:
            cfg = load_config(args, seed)
            loaders = prepare_custom_experiment(
                cfg,
                args.preprocessing,
                args.balance_mode,
                model_name,
            )
            regime = "reptile_meta" if model_name in META_MODELS else "target_supervised"
            print(
                f"\n[experiment4] seed={seed} model={model_name} "
                f"regime={regime} preprocessing={args.preprocessing} "
                f"balance={args.balance_mode}"
            )
            if args.dry_run:
                inspect_loaders(model_name, cfg, model_name, loaders)
                continue

            metrics = run_one(
                model_name,
                cfg,
                loaders,
                tag=f"exp4_{model_name}_{args.preprocessing}_{args.balance_mode}",
            )
            metrics.update(
                {
                    "training_regime": regime,
                    "preprocessing_mode": args.preprocessing,
                    "balance_mode": args.balance_mode,
                    "support_ratio": args.support_ratio,
                }
            )
            results.append(metrics)
            paths = save_progress(results, args)

    if args.dry_run:
        print("\n[dry-run完成] 未训练模型，也不会生成正式结果文件。")
        return

    assert paths is not None
    summary = summarize(results)
    effects = pairwise_effects(summary)
    print("\n[experiment4 summary]")
    print(summary.to_string(index=False))
    if not effects.empty:
        print("\n[experiment4 pairwise effects]")
        print(effects.to_string(index=False))
    print(f"\nRaw: {paths[0]}\nSummary: {paths[1]}\nPairwise: {paths[2]}")


if __name__ == "__main__":
    main()
