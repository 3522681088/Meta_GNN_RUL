"""实验12：发动机互斥ANIL的多源域划分种子稳健性实验。

实验11在10个模型随机种子下验证了发动机互斥ANIL，但所有运行共享同一个
``source_task_seed=2027``。因此实验12把随机性拆成两个正交因素：

1. ``source_task_seed``：控制FD001--FD003中哪些发动机进入源域支持集/查询集；
2. ``model_seed``：控制模型初始化、批次采样、Dropout和目标发动机顺序。

脚本默认交叉运行5个源域划分种子和5个模型种子，并以“源域划分”为高层重复
单位进行统计，避免把同一源域划分下的多个模型种子错误地当成完全独立样本。

本文件是独立实验入口，不替换实验11、``main.py``或模型模块。普通预训练和
Batch-ANIL与源域发动机划分无关，因此每个模型种子只训练一次并在不同源域划分
中共享；发动机互斥ANIL则针对每个源域划分重新训练。

第一步：协议检查（不训练）
--------------------------

Windows / PyCharm终端：

    D:\\Anaconda\\envs\\pytorch\\python.exe \
      scripts\\experiment12_source_split_robustness.py \
      --target FD004 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --model-seeds 42 43 44 45 46 \
      --k-values 2 5 20 \
      --dry-run

第二步：只用固定验证发动机完成稳健性分析
--------------------------------------

    D:\\Anaconda\\envs\\pytorch\\python.exe \
      scripts\\experiment12_source_split_robustness.py \
      --target FD004 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --model-seeds 42 43 44 45 46 \
      --k-values 2 5 20 \
      --evaluation-scope validation \
      --resume

第三步：全部规则锁定后进行官方测试评价
------------------------------------

    D:\\Anaconda\\envs\\pytorch\\python.exe \
      scripts\\experiment12_source_split_robustness.py \
      --target FD004 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --model-seeds 42 43 44 45 46 \
      --k-values 2 5 20 \
      --evaluation-scope official_test \
      --resume

禁止根据官方测试结果挑选源域划分种子、删除不利种子或继续修改超参数。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from scripts import experiment11_engine_disjoint_anil as exp11  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    BALANCE_MODES,
    EXPECTED_OFFICIAL_TEST_ENGINES,
    METRICS,
    PREPROCESSING_MODES,
    atomic_write_text,
    evaluate,
    prepare_kshot_experiment,
    protocol_split_frame,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.experiment10_anil_repair import (  # noqa: E402
    cpu_state,
    split_source_tasks_by_engine,
)
from scripts.experiment10b_anil_stability import all_tensors_finite  # noqa: E402
from scripts.experiment10c_target_kshot import train_target  # noqa: E402


SCRIPT_VERSION = "experiment12_source_split_robustness_v1"
SHARED_SOURCE_SENTINEL = 0
REGIMES = (
    "pretrained_head",
    "anil_batch_head",
    "anil_engine_disjoint_head",
    "pretrained_budget_head",
)
DEFAULT_REGIMES = (
    "pretrained_head",
    "anil_batch_head",
    "anil_engine_disjoint_head",
)
COMPARISON_SPECS = (
    (
        "anil_engine_disjoint_head",
        "pretrained_head",
        "engine_disjoint_anil_vs_ordinary_head",
    ),
    (
        "anil_engine_disjoint_head",
        "anil_batch_head",
        "engine_disjoint_vs_batch_anil",
    ),
    (
        "anil_engine_disjoint_head",
        "pretrained_budget_head",
        "engine_disjoint_anil_vs_budget_head",
    ),
    (
        "anil_batch_head",
        "pretrained_head",
        "batch_anil_vs_ordinary_head",
    ),
)
PRIMARY_COMPARISONS = {
    "engine_disjoint_anil_vs_ordinary_head",
    "engine_disjoint_vs_batch_anil",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验12：多源域发动机划分种子下验证ANIL稳健性"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 20])
    parser.add_argument(
        "--source-task-seeds",
        nargs="+",
        type=int,
        default=[2027, 2028, 2029, 2030, 2031],
        help="源域支持/查询发动机划分种子；正式实验建议至少5个",
    )
    parser.add_argument(
        "--model-seeds",
        "--seeds",
        dest="model_seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="模型初始化、采样和目标发动机顺序种子；--seeds为兼容别名",
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=REGIMES, default=list(DEFAULT_REGIMES)
    )
    parser.add_argument("--protocol-file")
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument(
        "--balance-mode", choices=BALANCE_MODES, default="engine_stage"
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--device", help="可选：覆盖配置中的device，例如cuda或cpu")

    # 完全沿用实验11的锁定源训练超参数。
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--meta-inner-lr", type=float, default=1e-5)
    parser.add_argument("--meta-inner-steps", type=int, default=1)
    parser.add_argument("--anil-meta-lr", type=float, default=1e-4)
    parser.add_argument("--anil-query-batches", type=int, default=1)
    parser.add_argument("--anil-order", choices=("first", "second"), default="first")
    parser.add_argument("--meta-clip-norm", type=float, default=0.0)
    parser.add_argument("--loss-ceiling", type=float, default=1e8)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--source-query-fraction", type=float, default=0.30)
    parser.add_argument("--outer-lr", type=float, default=0.05)
    parser.add_argument("--pair-aux-weight", type=float, default=0.0)

    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument("--budget-extra-steps", type=int)

    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clip-norm", type=float, default=0.0)

    parser.add_argument(
        "--evaluation-scope",
        choices=("validation", "official_test"),
        default="validation",
        help="先用validation完成方法检查；规则锁定后才运行official_test",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="outputs/experiment12_source_split_robustness"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[int], list[str]]:
    k_values = sorted(set(args.k_values))
    source_seeds = list(dict.fromkeys(args.source_task_seeds))
    model_seeds = list(dict.fromkeys(args.model_seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须为正整数")
    if not source_seeds or not model_seeds or not regimes:
        raise ValueError("源域划分种子、模型种子和方案列表均不能为空")
    positive = {
        "meta_epochs": args.meta_epochs,
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_query_batches": args.anil_query_batches,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "loss_ceiling": args.loss_ceiling,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正数：{invalid}")
    if not 0 < args.source_query_fraction < 1:
        raise ValueError("--source-query-fraction必须位于(0,1)")
    if args.meta_clip_norm < 0 or args.target_clip_norm < 0:
        raise ValueError("梯度裁剪阈值不能为负数")
    if args.budget_extra_steps is not None and args.budget_extra_steps <= 0:
        raise ValueError("--budget-extra-steps必须为正整数")
    if not args.dry_run and len(source_seeds) < 5:
        print("[警告] 少于5个源域划分种子，只能作为预实验。")
    if not args.dry_run and len(model_seeds) < 5:
        print("[警告] 少于5个模型种子，只能作为预实验。")
    return k_values, source_seeds, model_seeds, regimes


def worker_args(
    args: argparse.Namespace,
    output_dir: Path,
    source_task_seed: int,
    model_seeds: list[int],
) -> argparse.Namespace:
    """Construct the Namespace expected by the locked experiment11 functions."""
    proxy = argparse.Namespace(**vars(args))
    proxy.output_dir = str(output_dir)
    proxy.source_task_seed = int(source_task_seed)
    proxy.vary_source_split_by_seed = False
    proxy.seeds = list(model_seeds)
    proxy.pair_aux_weight = 0.0
    return proxy


def load_cfg(args: argparse.Namespace, seed: int) -> dict:
    cfg = exp11.load_config(args, seed)
    if args.device:
        cfg["device"] = args.device
    return cfg


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    scope = "official" if args.evaluation_scope == "official_test" else "validation"
    prefix = f"experiment12_{scope}_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "per_source_split": output / f"{prefix}_per_source_split.csv",
        "paired_cell": output / f"{prefix}_paired_by_cell.csv",
        "paired_split": output / f"{prefix}_paired_by_source_split.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_split_protocol.json",
        "target_splits": output / f"{prefix}_target_engine_splits.csv",
        "grid": output / f"{prefix}_grid_plan.json",
        "budget": output / f"{prefix}_budget.json",
        "source_splits": output / "experiment12_source_task_splits.json",
        "source_diagnostics": output / "experiment12_source_diagnostics.json",
        "parameters": output / "experiment12_parameter_inventory.json",
    }


def write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_protocol(
    args: argparse.Namespace,
    first_worker: argparse.Namespace,
    model_seeds: list[int],
    k_values: list[int],
) -> tuple[dict, Path | None, dict]:
    cfg = load_cfg(first_worker, model_seeds[0])
    protocol, source_path = exp11.load_or_extend_protocol(
        first_worker, cfg, model_seeds, k_values
    )
    expected = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        int(protocol["official_test_engine_count"]) != expected
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试发动机应为{expected}台，"
            f"当前为{protocol['official_test_engine_count']}台"
        )
    return protocol, source_path, cfg


def write_plan_files(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    source_protocol_path: Path | None,
    k_values: list[int],
    source_seeds: list[int],
    model_seeds: list[int],
    regimes: list[str],
) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied = dict(protocol)
    copied.update(
        {
            "script_version": SCRIPT_VERSION,
            "experiment12_source_protocol": (
                str(source_protocol_path) if source_protocol_path else "regenerated"
            ),
            "source_task_seeds": source_seeds,
            "model_seeds": model_seeds,
            "evaluation_scope": args.evaluation_scope,
            "hypothesis": (
                "Engine-disjoint ANIL remains better than ordinary transfer and "
                "batch-ANIL across multiple source-engine support/query splits."
            ),
        }
    )
    write_json(paths["protocol"], copied)
    atomic_write_text(
        paths["target_splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )

    baseline_regimes = [r for r in regimes if r != "anil_engine_disjoint_head"]
    expected_rows = (
        len(source_seeds) * len(model_seeds) * len(k_values) * len(regimes)
    )
    unique_target_trainings = (
        len(model_seeds) * len(k_values) * len(baseline_regimes)
        + len(source_seeds) * len(model_seeds) * len(k_values)
    )
    grid = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": args.evaluation_scope,
        "source_task_seeds": source_seeds,
        "model_seeds": model_seeds,
        "k_values": k_values,
        "regimes": regimes,
        "full_factorial_result_rows": expected_rows,
        "unique_target_trainings_after_baseline_reuse": unique_target_trainings,
        "baseline_source_states_per_model_seed": baseline_regimes,
        "engine_disjoint_source_states": len(source_seeds) * len(model_seeds),
        "statistics_high_level_unit": "source_task_seed",
        "no_best_split_selection": True,
    }
    write_json(paths["grid"], grid)

    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    extra = exp11.budget_extra_steps(
        worker_args(args, paths["output"], source_seeds[0], model_seeds), cfg
    )
    budget = {
        "script_version": SCRIPT_VERSION,
        "source_domains": cfg["source_domains"],
        "ordinary_pretraining_steps": args.source_pretrain_steps,
        "ordinary_budget_extra_steps": extra,
        "ordinary_budget_total_steps": args.source_pretrain_steps + extra,
        "meta_epochs": args.meta_epochs,
        "tasks_per_meta_batch": task_count,
        "meta_inner_steps": args.meta_inner_steps,
        "meta_inner_lr": args.meta_inner_lr,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_query_batches": args.anil_query_batches,
        "source_query_fraction": args.source_query_fraction,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "target_scope": "predictor.* only",
        "target_loss": "raw_mse",
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "official_test_policy": (
            "Run validation first; use official_test only after every setting and "
            "success rule is locked. Never select the best source split."
        ),
        "primary_success_rule": {
            "k_focus": [k for k in (2, 5) if k in k_values],
            "rmse_improvement_pct_at_least": 3.0,
            "source_split_win_rate_at_least": 0.80,
            "hierarchical_bootstrap_ci95_upper_below": 0.0,
            "holm_adjusted_split_level_p_below": 0.05,
            "nasa_score_mean_delta": "must be <= 0",
        },
    }
    write_json(paths["budget"], budget)
    return paths


def completed_key(row: dict) -> tuple[int, int, int, str]:
    return (
        int(row["model_seed"]),
        int(row["source_split_seed"]),
        int(row["k"]),
        str(row["regime"]),
    )


def load_results(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict]:
    if not args.resume or not paths["raw"].is_file():
        return []
    results = json.loads(paths["raw"].read_text(encoding="utf-8"))
    keys = [completed_key(row) for row in results]
    if len(keys) != len(set(keys)):
        raise RuntimeError("已有raw结果包含重复网格单元，请先检查文件")
    print(f"[resume] 已读取{len(results)}条{args.evaluation_scope}结果。")
    return results


def sample_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def per_source_split_summary(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    groups = ["source_split_seed", "k", "regime", "source_training"]
    for keys, group in frame.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, keys))
        row["n_model_seeds"] = int(group.model_seed.nunique())
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std_across_model_seeds"] = sample_std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["k", "regime", "source_split_seed"]
    ).reset_index(drop=True)


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    groups = ["k", "regime", "source_training", "source_task_mode"]
    for keys, group in frame.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_cells": len(group),
                "n_source_splits": int(group.source_split_seed.nunique()),
                "n_model_seeds": int(group.model_seed.nunique()),
            }
        )
        split_means = group.groupby("source_split_seed")[list(METRICS)].mean()
        for metric in METRICS:
            within_stds = group.groupby("source_split_seed")[metric].std(ddof=1)
            row[f"{metric}_mean"] = float(split_means[metric].mean())
            row[f"{metric}_cell_std"] = sample_std(group[metric])
            row[f"{metric}_source_split_std"] = sample_std(split_means[metric])
            row[f"{metric}_within_split_model_std_mean"] = float(
                within_stds.fillna(0.0).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def t_confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(stats.sem(values))
    if not math.isfinite(sem) or sem == 0:
        return mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - half, mean + half


def split_level_pvalue(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    if np.allclose(values, values[0]):
        return 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def split_level_wilcoxon(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def hierarchical_bootstrap_ci(
    merged: pd.DataFrame,
    delta_column: str,
    repetitions: int,
    random_seed: int,
) -> tuple[float, float]:
    """Crossed bootstrap: resample source-split and model-seed factors."""
    matrix = merged.pivot(
        index="source_split_seed", columns="model_seed", values=delta_column
    ).sort_index().sort_index(axis=1)
    if matrix.empty:
        return float("nan"), float("nan")
    if bool(matrix.isna().any().any()):
        raise ValueError("分层bootstrap要求完整的source split × model seed交叉网格")
    values = matrix.to_numpy(dtype=float)
    n_source_splits, n_model_seeds = values.shape
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_split_indices = rng.integers(
            0, n_source_splits, size=n_source_splits
        )
        sampled_model_indices = rng.integers(0, n_model_seeds, size=n_model_seeds)
        sampled = values[np.ix_(sampled_split_indices, sampled_model_indices)]
        bootstrap[index] = float(sampled.mean())
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def comparison_specs(regimes: list[str]) -> list[tuple[str, str, str]]:
    selected = set(regimes)
    return [
        item for item in COMPARISON_SPECS if item[0] in selected and item[1] in selected
    ]


def paired_comparisons(
    results: list[dict],
    regimes: list[str],
    bootstrap_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    cell_rows: list[dict] = []
    split_rows: list[dict] = []
    comparison_rows: list[dict] = []
    join_keys = ["source_split_seed", "model_seed", "k"]

    for candidate, reference, label in comparison_specs(regimes):
        for k in sorted(frame.k.unique()):
            left = frame[(frame.k == k) & (frame.regime == candidate)]
            right = frame[(frame.k == k) & (frame.regime == reference)]
            merged = left.merge(right, on=join_keys, suffixes=("_candidate", "_reference"))
            if merged.empty:
                continue
            for metric in METRICS:
                merged[f"{metric}_delta"] = (
                    merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
                )

            for _, row in merged.iterrows():
                item = {
                    "k": int(k),
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "source_split_seed": int(row.source_split_seed),
                    "model_seed": int(row.model_seed),
                }
                for metric in METRICS:
                    item[f"{metric}_candidate"] = float(row[f"{metric}_candidate"])
                    item[f"{metric}_reference"] = float(row[f"{metric}_reference"])
                    item[f"{metric}_delta"] = float(row[f"{metric}_delta"])
                cell_rows.append(item)

            split_means = merged.groupby("source_split_seed", as_index=False)[
                [f"{metric}_delta" for metric in METRICS]
            ].mean()
            for _, row in split_means.iterrows():
                item = {
                    "k": int(k),
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "source_split_seed": int(row.source_split_seed),
                }
                for metric in METRICS:
                    item[f"{metric}_delta_mean_over_model_seeds"] = float(
                        row[f"{metric}_delta"]
                    )
                split_rows.append(item)

            output = {
                "k": int(k),
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "is_primary_comparison": label in PRIMARY_COMPARISONS,
                "n_source_splits": int(merged.source_split_seed.nunique()),
                "n_model_seeds": int(merged.model_seed.nunique()),
                "n_cells": len(merged),
            }
            for metric in METRICS:
                delta_column = f"{metric}_delta"
                deltas = merged[delta_column].to_numpy(dtype=float)
                split_deltas = split_means[delta_column].to_numpy(dtype=float)
                references = merged[f"{metric}_reference"].to_numpy(dtype=float)
                lower_is_better = metric != "r2"
                cell_wins = deltas < 0 if lower_is_better else deltas > 0
                split_wins = split_deltas < 0 if lower_is_better else split_deltas > 0
                t_low, t_high = t_confidence_interval(split_deltas)
                digest = hashlib.sha256(f"{label}:{k}:{metric}".encode()).hexdigest()
                boot_seed = int(digest[:8], 16)
                boot_low, boot_high = hierarchical_bootstrap_ci(
                    merged,
                    delta_column,
                    bootstrap_repetitions,
                    boot_seed,
                )
                output[f"{metric}_delta_mean"] = float(deltas.mean())
                output[f"{metric}_split_delta_std"] = sample_std(split_deltas)
                output[f"{metric}_cell_win_rate"] = float(cell_wins.mean())
                output[f"{metric}_source_split_win_rate"] = float(split_wins.mean())
                output[f"{metric}_split_t_ci95_low"] = t_low
                output[f"{metric}_split_t_ci95_high"] = t_high
                output[f"{metric}_hier_boot_ci95_low"] = boot_low
                output[f"{metric}_hier_boot_ci95_high"] = boot_high
                output[f"{metric}_split_t_p"] = split_level_pvalue(split_deltas)
                output[f"{metric}_split_wilcoxon_p"] = split_level_wilcoxon(split_deltas)
                if lower_is_better and not np.isclose(references.mean(), 0.0):
                    output[f"{metric}_improvement_pct"] = float(
                        -100.0 * deltas.mean() / references.mean()
                    )
                else:
                    output[f"{metric}_improvement_pct"] = float(deltas.mean())
            comparison_rows.append(output)

    paired_cell = pd.DataFrame(cell_rows)
    paired_split = pd.DataFrame(split_rows)
    comparisons = pd.DataFrame(comparison_rows)
    if comparisons.empty:
        return paired_cell, paired_split, comparisons
    for metric in METRICS:
        comparisons[f"{metric}_split_t_p_holm"] = exp11.holm_adjust(
            comparisons[f"{metric}_split_t_p"]
        )
    comparisons["meets_rmse_effect_3pct"] = (
        comparisons.rmse_improvement_pct >= 3.0
    )
    comparisons["meets_source_split_win_rate_80pct"] = (
        comparisons.rmse_source_split_win_rate >= 0.80
    )
    comparisons["rmse_hierarchical_ci_excludes_zero"] = (
        comparisons.rmse_hier_boot_ci95_high < 0.0
    )
    comparisons["meets_rmse_holm_p_005"] = (
        comparisons.rmse_split_t_p_holm < 0.05
    )
    comparisons["nasa_not_worse"] = comparisons.nasa_score_delta_mean <= 0.0
    comparisons["robust_success"] = (
        comparisons.is_primary_comparison
        & comparisons.meets_rmse_effect_3pct
        & comparisons.meets_source_split_win_rate_80pct
        & comparisons.rmse_hierarchical_ci_excludes_zero
        & comparisons.nasa_not_worse
    )
    comparisons["strict_success"] = (
        comparisons.robust_success & comparisons.meets_rmse_holm_p_005
    )
    return (
        paired_cell.sort_values(
            ["k", "comparison", "source_split_seed", "model_seed"]
        ).reset_index(drop=True),
        paired_split.sort_values(["k", "comparison", "source_split_seed"]).reset_index(
            drop=True
        ),
        comparisons.sort_values(["k", "comparison"]).reset_index(drop=True),
    )


def save_progress(
    args: argparse.Namespace,
    paths: dict[str, Path],
    results: list[dict],
    regimes: list[str],
) -> None:
    write_json(paths["raw"], results)
    atomic_write_text(
        paths["summary"], summarize(results).to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["per_source_split"],
        per_source_split_summary(results).to_csv(index=False),
        encoding="utf-8-sig",
    )
    paired_cell, paired_split, comparisons = paired_comparisons(
        results, regimes, args.bootstrap_repetitions
    )
    atomic_write_text(
        paths["paired_cell"], paired_cell.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["paired_split"], paired_split.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False), encoding="utf-8-sig"
    )


def source_cache_or_train_engine_anil(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    model_seed: int,
    feature_count: int,
    ordinary_state: dict,
) -> tuple[dict, list, dict, dict | None]:
    cache_path = exp11.source_cache_path(args, "anil_engine_disjoint", model_seed)
    signature = exp11.source_signature(
        args,
        cfg,
        "anil_engine_disjoint",
        feature_count,
        int(args.source_task_seed),
    )
    cached = exp11.load_source_cache(cache_path, signature) if args.resume else None
    if cached is not None:
        return (
            cached["state"],
            cached.get("history", []),
            cached.get("diagnostic", {}),
            cached.get("source_split"),
        )
    state, history, diagnostic, manifest = exp11.train_anil_state(
        args,
        cfg,
        protocol,
        model_seed,
        feature_count,
        ordinary_state,
        "engine_disjoint",
    )
    exp11.save_source_cache(
        cache_path, signature, state, history, diagnostic, manifest
    )
    return state, history, diagnostic, manifest


def run_target_cell(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    regime: str,
    source_state: dict,
    source_history: list,
    inventory: dict,
    k: int,
    adaptation_units: list[int],
    source_split_seed: int,
    source_task_seed_used: int | None,
    baseline_reused: bool,
) -> dict:
    seed_everything(cfg["seed"])
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        adaptation_units,
    )
    _, support, validation, test, feature_count, split_info = loaders
    if split_info["official_test_units_hash"] != protocol["official_test_units_hash"]:
        raise AssertionError("不同运行使用了不同官方测试发动机")

    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    device = resolve_device(cfg["device"])
    model, target_history, best_epoch, trainable_count, drift, target_diag = train_target(
        model,
        support,
        validation,
        args,
        device,
        scope="rul_head",
        loss_mode="raw_mse",
    )
    validation_metrics = evaluate(model, validation, device)
    if args.evaluation_scope == "official_test":
        selected_metrics = evaluate(model, test, device)
        official_metrics = dict(selected_metrics)
    else:
        selected_metrics = dict(validation_metrics)
        official_metrics = None

    spec = exp11.regime_spec(regime)
    result = {
        **selected_metrics,
        "evaluation_scope": args.evaluation_scope,
        "regime": regime,
        "model": "meta_gnn_rul",
        "source_training": spec["source_training"],
        "source_task_mode": spec["source_task_mode"],
        "target_adaptation_scope": "rul_head",
        "target_loss_mode": "raw_mse",
        "experiment": f"experiment12_{regime}_k{k}",
        "target_domain": args.target,
        "seed": int(cfg["seed"]),
        "model_seed": int(cfg["seed"]),
        "source_split_seed": int(source_split_seed),
        "source_task_seed_used": (
            int(source_task_seed_used) if source_task_seed_used is not None else None
        ),
        "baseline_reused_across_source_splits": bool(baseline_reused),
        "replicate_id": (
            f"source{source_split_seed}_model{cfg['seed']}_k{k}_{regime}"
        ),
        "k": int(k),
        "adaptation_engine_count": len(adaptation_units),
        "adaptation_units": [int(unit) for unit in adaptation_units],
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": args.target_epochs,
        "target_learning_rate": args.target_lr,
        "target_trainable_parameter_count": int(trainable_count),
        "total_parameter_count": int(inventory["total_parameter_count"]),
        "target_trainable_fraction": trainable_count
        / inventory["total_parameter_count"],
        "source_query_fraction": args.source_query_fraction,
        "meta_epochs": args.meta_epochs,
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "official_test_metrics": official_metrics,
        "source_history_rows": len(source_history),
        "parameter_drift_by_group": drift,
        **target_diag,
    }

    if args.save_target_checkpoints:
        checkpoint_dir = result_paths(args)["output"] / "target_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / (
            f"experiment12_{args.evaluation_scope}_{regime}_k{k}_{args.target}_"
            f"source{source_split_seed}_model{cfg['seed']}.pt"
        )
        torch.save(
            {
                "model": cpu_state(model),
                "config": cfg,
                "split": split_info,
                "parameter_inventory": inventory,
                "source_history": source_history,
                "target_history": target_history,
                "metrics": result,
            },
            checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def clone_baseline_for_split(template: dict, source_split_seed: int) -> dict:
    row = deepcopy(template)
    row["source_split_seed"] = int(source_split_seed)
    row["replicate_id"] = (
        f"source{source_split_seed}_model{row['model_seed']}_k{row['k']}_"
        f"{row['regime']}"
    )
    row["baseline_reused_across_source_splits"] = True
    row["source_task_seed_used"] = None
    return row


def inspect_all_source_splits(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    model_seed: int,
    k_values: list[int],
    source_seeds: list[int],
) -> tuple[list[dict], dict, dict]:
    first = source_seeds[0]
    first_args = worker_args(
        args, result_paths(args)["output"] / "dry_run", first, [model_seed]
    )
    diagnostics, _, inventory = exp11.inspect_protocol(
        first_args, cfg, protocol, model_seed, k_values
    )
    source_tasks, _ = exp11.fresh_source_tasks(first_args, cfg, protocol, model_seed)
    manifests: dict[str, dict] = {}
    signatures: set[str] = set()
    for source_seed in source_seeds:
        _, manifest = split_source_tasks_by_engine(
            source_tasks,
            args.balance_mode,
            args.source_query_fraction,
            source_seed,
        )
        for domain, item in manifest.items():
            support = set(item["support_units"])
            query = set(item["query_units"])
            if support & query:
                raise AssertionError(
                    f"source_task_seed={source_seed}, {domain}支持/查询发动机重叠"
                )
        encoded = json.dumps(manifest, sort_keys=True).encode()
        signature = hashlib.sha256(encoded).hexdigest()[:16]
        if signature in signatures:
            raise ValueError(f"source_task_seed={source_seed}产生了重复源域划分")
        signatures.add(signature)
        manifests[str(source_seed)] = {
            "source_split_hash": signature,
            "domains": manifest,
        }
    return diagnostics, manifests, inventory


def main() -> None:
    args = parse_args()
    k_values, source_seeds, model_seeds, regimes = validate_args(args)
    paths = result_paths(args)
    shared_worker = worker_args(
        args,
        paths["output"] / "shared_source_states",
        SHARED_SOURCE_SENTINEL,
        model_seeds,
    )
    protocol, source_protocol_path, first_cfg = load_protocol(
        args, shared_worker, model_seeds, k_values
    )
    paths = write_plan_files(
        args,
        first_cfg,
        protocol,
        source_protocol_path,
        k_values,
        source_seeds,
        model_seeds,
        regimes,
    )
    print("\n[实验12固定网格]")
    print(paths["grid"].read_text(encoding="utf-8"))
    print("\n[实验12锁定预算]")
    print(paths["budget"].read_text(encoding="utf-8"))

    if args.dry_run:
        diagnostics, manifests, inventory = inspect_all_source_splits(
            args,
            first_cfg,
            protocol,
            model_seeds[0],
            k_values,
            source_seeds,
        )
        write_json(paths["source_splits"], manifests)
        write_json(paths["parameters"], inventory)
        print("\n[目标域与前向检查]")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        print("\n[全部源域发动机互斥划分检查]")
        print(json.dumps(manifests, ensure_ascii=False, indent=2))
        print("\n[dry-run完成] 未训练模型。")
        print(
            f"Protocol: {paths['protocol']}\nGrid: {paths['grid']}"
            f"\nBudget: {paths['budget']}\nSource splits: {paths['source_splits']}"
            f"\nParameters: {paths['parameters']}"
        )
        return

    results = load_results(args, paths)
    done = {completed_key(row) for row in results}
    source_diagnostics: dict = {}
    source_manifests: dict = {}
    if args.resume and paths["source_diagnostics"].is_file():
        source_diagnostics = json.loads(
            paths["source_diagnostics"].read_text(encoding="utf-8")
        )
    if args.resume and paths["source_splits"].is_file():
        source_manifests = json.loads(
            paths["source_splits"].read_text(encoding="utf-8")
        )

    baseline_regimes = [r for r in regimes if r != "anil_engine_disjoint_head"]
    for model_seed in model_seeds:
        expected_for_model = {
            (model_seed, source_seed, k, regime)
            for source_seed in source_seeds
            for k in k_values
            for regime in regimes
        }
        if expected_for_model.issubset(done):
            print(f"[skip model seed] model_seed={model_seed}已全部完成。")
            continue

        model_shared_args = worker_args(
            args,
            paths["output"] / "shared_source_states",
            SHARED_SOURCE_SENTINEL,
            model_seeds,
        )
        cfg = load_cfg(model_shared_args, model_seed)
        # 即使用户只选择发动机互斥ANIL，它也必须从同一个普通预训练状态开始。
        source_requirements = list(
            dict.fromkeys(["pretrained_head", *baseline_regimes])
        )
        print(
            f"\n[shared source initialization] model_seed={model_seed} "
            f"regimes={source_requirements}"
        )
        states, histories, source_diags, _, inventory = exp11.build_source_states(
            model_shared_args,
            cfg,
            protocol,
            model_seed,
            source_requirements,
        )
        source_diagnostics.setdefault(str(model_seed), {})["shared"] = source_diags
        write_json(paths["source_diagnostics"], source_diagnostics)
        write_json(paths["parameters"], inventory)

        # 与源域发动机划分无关的基线只适应一次，然后复制到每个source split用于配对。
        for k in k_values:
            units = protocol["nested_adaptation_units_by_seed"][str(model_seed)][str(k)]
            for regime in baseline_regimes:
                missing_splits = [
                    source_seed
                    for source_seed in source_seeds
                    if (model_seed, source_seed, k, regime) not in done
                ]
                if not missing_splits:
                    continue
                template = next(
                    (
                        row
                        for row in results
                        if int(row["model_seed"]) == model_seed
                        and int(row["k"]) == k
                        and row["regime"] == regime
                    ),
                    None,
                )
                if template is None:
                    print(
                        f"\n[experiment12 shared baseline] model_seed={model_seed} "
                        f"K={k} regime={regime}"
                    )
                    template = run_target_cell(
                        model_shared_args,
                        cfg,
                        protocol,
                        regime,
                        states[regime],
                        histories[regime],
                        inventory,
                        k,
                        units,
                        source_seeds[0],
                        None,
                        True,
                    )
                for source_seed in missing_splits:
                    row = clone_baseline_for_split(template, source_seed)
                    results.append(row)
                    done.add(completed_key(row))
                save_progress(args, paths, results, regimes)

        if "anil_engine_disjoint_head" in regimes:
            source_tasks, feature_count = exp11.fresh_source_tasks(
                model_shared_args, cfg, protocol, model_seed
            )
            del source_tasks
            ordinary_state = states["pretrained_head"]
            for source_seed in source_seeds:
                pending_k = [
                    k
                    for k in k_values
                    if (
                        model_seed,
                        source_seed,
                        k,
                        "anil_engine_disjoint_head",
                    )
                    not in done
                ]
                if not pending_k:
                    continue
                split_args = worker_args(
                    args,
                    paths["output"] / "source_states" / f"source_{source_seed}",
                    source_seed,
                    model_seeds,
                )
                split_cfg = load_cfg(split_args, model_seed)
                print(
                    f"\n[engine-disjoint source initialization] "
                    f"source_task_seed={source_seed} model_seed={model_seed}"
                )
                state, history, diagnostic, manifest = source_cache_or_train_engine_anil(
                    split_args,
                    split_cfg,
                    protocol,
                    model_seed,
                    feature_count,
                    ordinary_state,
                )
                if not all_tensors_finite(state.values()):
                    raise RuntimeError("发动机互斥ANIL源状态包含NaN/Inf")
                source_diagnostics.setdefault(str(model_seed), {})[
                    str(source_seed)
                ] = diagnostic
                if manifest is not None:
                    source_manifests.setdefault(str(source_seed), {})[
                        str(model_seed)
                    ] = manifest
                write_json(paths["source_diagnostics"], source_diagnostics)
                write_json(paths["source_splits"], source_manifests)

                for k in pending_k:
                    units = protocol["nested_adaptation_units_by_seed"][str(model_seed)][
                        str(k)
                    ]
                    print(
                        f"\n[experiment12] source_task_seed={source_seed} "
                        f"model_seed={model_seed} K={k} "
                        "regime=anil_engine_disjoint_head"
                    )
                    result = run_target_cell(
                        split_args,
                        split_cfg,
                        protocol,
                        "anil_engine_disjoint_head",
                        state,
                        history,
                        inventory,
                        k,
                        units,
                        source_seed,
                        source_seed,
                        False,
                    )
                    results.append(result)
                    done.add(completed_key(result))
                    save_progress(args, paths, results, regimes)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                del state, history
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del states, histories
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = {
        (model_seed, source_seed, k, regime)
        for model_seed in model_seeds
        for source_seed in source_seeds
        for k in k_values
        for regime in regimes
    }
    missing = sorted(expected - done)
    if missing:
        print(f"[警告] 仍缺少{len(missing)}个网格单元，前10个：{missing[:10]}")
    save_progress(args, paths, results, regimes)
    summary = summarize(results)
    _, _, comparisons = paired_comparisons(
        results, regimes, args.bootstrap_repetitions
    )
    print("\n[实验12分层汇总]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        columns = [
            "k",
            "comparison",
            "n_source_splits",
            "n_cells",
            "rmse_improvement_pct",
            "rmse_source_split_win_rate",
            "rmse_hier_boot_ci95_low",
            "rmse_hier_boot_ci95_high",
            "rmse_split_t_p_holm",
            "nasa_score_delta_mean",
            "robust_success",
            "strict_success",
        ]
        print("\n[实验12主要比较]")
        print(comparisons[columns].to_string(index=False))
    print("\n[结论判定]")
    print("1. 不能挑选最佳source_task_seed；必须汇总所有预注册源域划分。")
    print("2. source_split_win_rate≥80%说明改善不依赖单个幸运划分。")
    print("3. 分层bootstrap的RMSE差值上界<0，说明跨划分总体改善。")
    print("4. strict_success还要求改善≥3%、Holm p<0.05且NASA Score不恶化。")
    print("5. validation阶段只能形成候选结论；official_test阶段才用于最终报告。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPer source split: {paths['per_source_split']}"
        f"\nPaired cell: {paths['paired_cell']}"
        f"\nPaired source split: {paths['paired_split']}"
        f"\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nGrid: {paths['grid']}"
        f"\nBudget: {paths['budget']}"
        f"\nSource splits: {paths['source_splits']}"
        f"\nSource diagnostics: {paths['source_diagnostics']}"
    )


if __name__ == "__main__":
    main()
