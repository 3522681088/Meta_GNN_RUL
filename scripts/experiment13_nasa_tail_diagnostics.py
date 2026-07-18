#!/usr/bin/env python3
"""实验13：预算匹配ANIL的NASA尾部误差诊断。

实验12B出现了一个关键现象：发动机互斥ANIL降低了平均RMSE/MAE，
但NASA Score（NASA非对称评分）反而升高。本脚本不再调整模型超参数，
而是在实验12B锁定的FD004验证集上回答以下问题：

1. NASA恶化是否由少数发动机/少数窗口主导；
2. 恶化主要来自晚预测（predicted RUL > true RUL）还是早预测；
3. 风险集中在哪个RUL阶段；
4. 模型种子46为什么明显不稳定；
5. K=5是否比K=2具有更一致的“RMSE和NASA同时改善”。

安全约束
--------
* 只分析固定validation（验证集），不对官方test（测试集）执行预测；
* 不根据本脚本结果修改实验12B既有结果；
* 仅加载用户自己生成且可信的PyTorch checkpoint/cache文件。

输入模式
--------
``auto``（默认）
    优先加载实验12B的目标模型checkpoint；checkpoint缺失时，从已保存的
    source cache（源模型缓存）重新执行目标域RUL头适应。

``checkpoint``
    只允许读取目标模型checkpoint；缺失即报错，适合当实验12B运行时使用了
    ``--save-target-checkpoints`` 的情况。

``replay``
    从实验12B源模型缓存重新适应目标RUL头。不会重新训练源域模型。

服务器正式运行示例
------------------

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment13_nasa_tail_diagnostics.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --mode auto \
      --output-dir outputs/experiment13_nasa_tail_diagnostics \
      --resume

若实验12B实际保存在旧目录，请把 ``--experiment12b-dir`` 改为：

    outputs/experiment12_source_split_robustness

先进行不训练检查：

    python -u scripts/experiment13_nasa_tail_diagnostics.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from evaluation.metrics import regression_metrics  # noqa: E402
from scripts import experiment11_engine_disjoint_anil as exp11  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    PREPROCESSING_MODES,
    BALANCE_MODES,
    atomic_write_text,
    prepare_kshot_experiment,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.run_condition_aware_experiment import rul_stage_ids  # noqa: E402
from scripts.experiment10b_anil_stability import all_tensors_finite  # noqa: E402
from scripts.experiment10c_target_kshot import train_target  # noqa: E402


SCRIPT_VERSION = "experiment13_nasa_tail_diagnostics_v1"
REGIMES = ("pretrained_budget_head", "anil_engine_disjoint_head")
STAGE_NAMES = ("critical", "middle", "early", "high_rul")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验13：分解实验12B的NASA尾部误差与种子46负迁移"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--target", default="FD004")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--model-seeds", "--seeds", dest="model_seeds", nargs="+", type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--source-task-seeds", nargs="+", type=int,
        default=[2027, 2028, 2029, 2030, 2031],
    )
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        "--experiment12b-dir", default="outputs/experiment12b_budget_matched",
        help="实验12B输出目录，必须包含协议、raw结果和源模型缓存",
    )
    parser.add_argument("--protocol-file")
    parser.add_argument("--raw-results-file")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--mode", choices=("auto", "checkpoint", "replay"), default="auto")
    parser.add_argument("--device", help="例如cuda、cuda:0或cpu；默认沿用配置auto")
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument("--balance-mode", choices=BALANCE_MODES, default="engine_stage")
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)

    # 必须与实验12B锁定设置一致；这些参数只用于准确重放目标域适应。
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
    parser.add_argument("--budget-extra-steps", type=int, default=600)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clip-norm", type=float, default=0.0)

    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--metric-tolerance", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs/experiment13_nasa_tail_diagnostics")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-replayed-checkpoints", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    k_values = sorted(set(args.k_values))
    model_seeds = list(dict.fromkeys(args.model_seeds))
    source_seeds = list(dict.fromkeys(args.source_task_seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须为正整数")
    if not model_seeds or not source_seeds or not regimes:
        raise ValueError("模型种子、源域划分种子和方案不能为空")
    if args.top_n <= 0 or args.metric_tolerance < 0:
        raise ValueError("--top-n必须为正数，--metric-tolerance不能为负数")
    if "anil_engine_disjoint_head" in regimes and not source_seeds:
        raise ValueError("发动机互斥ANIL必须提供源域划分种子")
    return k_values, model_seeds, source_seeds, regimes


def trusted_torch_load(path: Path) -> dict:
    """Only call this for checkpoints/caches created by the current user."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    candidates = list(candidates)
    for path in candidates:
        if path.is_file():
            return path
    joined = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"找不到{description}，已检查：\n  {joined}")


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    experiment_dir = resolve_path(args.experiment12b_dir, PROJECT_ROOT)
    protocol = (
        resolve_path(args.protocol_file, PROJECT_ROOT)
        if args.protocol_file else first_existing(
            [
                experiment_dir / f"experiment12_validation_{args.target}_split_protocol.json",
                experiment_dir / f"experiment12b_validation_{args.target}_split_protocol.json",
                experiment_dir / f"experiment12_validation_{args.target}_protocol.json",
            ],
            "实验12B划分协议",
        )
    )
    raw = (
        resolve_path(args.raw_results_file, PROJECT_ROOT)
        if args.raw_results_file else first_existing(
            [
                experiment_dir / f"experiment12_validation_{args.target}_raw.json",
                experiment_dir / f"experiment12b_validation_{args.target}_raw.json",
            ],
            "实验12B raw结果",
        )
    )
    checkpoint_dir = (
        resolve_path(args.checkpoint_dir, PROJECT_ROOT)
        if args.checkpoint_dir else experiment_dir / "target_checkpoints"
    )
    return {
        "experiment": experiment_dir,
        "protocol": protocol,
        "raw": raw,
        "checkpoints": checkpoint_dir,
    }


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment13_{args.target}"
    return {
        "output": output,
        "cells": output / "prediction_cells",
        "replayed_checkpoints": output / "replayed_checkpoints",
        "predictions": output / f"{prefix}_window_predictions.csv",
        "cell_summary": output / f"{prefix}_cell_summary.csv",
        "summary": output / f"{prefix}_tail_summary.csv",
        "engine": output / f"{prefix}_engine_metrics.csv",
        "paired_window": output / f"{prefix}_paired_window_deltas.csv",
        "paired_engine": output / f"{prefix}_paired_engine_deltas.csv",
        "outliers": output / f"{prefix}_outlier_engines.csv",
        "stage": output / f"{prefix}_stage_summary.csv",
        "model_seed": output / f"{prefix}_model_seed_summary.csv",
        "source_split": output / f"{prefix}_source_split_summary.csv",
        "conclusion": output / f"{prefix}_conclusion.json",
        "audit": output / f"{prefix}_replay_audit.csv",
        "protocol": output / f"{prefix}_diagnostic_protocol.json",
        "scatter": output / f"{prefix}_rmse_vs_nasa_delta.png",
        "seed46": output / f"{prefix}_seed46_worst_engines.png",
        "stage_plot": output / f"{prefix}_stage_nasa_delta.png",
    }


def load_protocol_and_raw(paths: dict[str, Path], args: argparse.Namespace):
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    raw = pd.DataFrame(json.loads(paths["raw"].read_text(encoding="utf-8")))
    if protocol.get("target_domain") != args.target:
        raise ValueError("协议target_domain与--target不一致")
    if raw.empty:
        raise ValueError("实验12B raw结果为空")
    if set(raw.get("evaluation_scope", [])) != {"validation"}:
        raise ValueError("实验13只允许读取evaluation_scope=validation的实验12B结果")
    if raw.get("official_test_metrics", pd.Series([None] * len(raw))).notna().any():
        print("[警告] raw文件含官方测试字段；实验13仍不会读取或评估官方测试预测。")
    return protocol, raw


def source_cache_path(
    experiment_dir: Path, target: str, regime: str, model_seed: int,
    source_seed: int,
) -> Path:
    if regime == "pretrained_budget_head":
        return (
            experiment_dir / "shared_source_states" / "source_cache" /
            f"ordinary_budget_{target}_seed{model_seed}.pt"
        )
    if regime == "anil_engine_disjoint_head":
        return (
            experiment_dir / "source_states" / f"source_{source_seed}" /
            "source_cache" / f"anil_engine_disjoint_{target}_seed{model_seed}.pt"
        )
    raise ValueError(f"不支持的方案：{regime}")


def checkpoint_candidates(
    checkpoint_dir: Path, target: str, regime: str, k: int,
    model_seed: int, source_seed: int, first_source_seed: int,
) -> list[Path]:
    source_values = [source_seed]
    if regime == "pretrained_budget_head":
        source_values = list(dict.fromkeys([first_source_seed, source_seed]))
    paths: list[Path] = []
    for value in source_values:
        paths.extend(
            [
                checkpoint_dir /
                f"experiment12_validation_{regime}_k{k}_{target}_source{value}_model{model_seed}.pt",
                checkpoint_dir /
                f"experiment12b_validation_{regime}_k{k}_{target}_source{value}_model{model_seed}.pt",
            ]
        )
    return paths


def expected_raw_row(
    raw: pd.DataFrame, regime: str, k: int, model_seed: int,
    source_seed: int,
) -> dict | None:
    mask = (
        raw["regime"].eq(regime)
        & raw["k"].astype(int).eq(k)
        & raw["model_seed"].astype(int).eq(model_seed)
    )
    if regime == "anil_engine_disjoint_head":
        mask &= raw["source_split_seed"].astype(int).eq(source_seed)
    rows = raw.loc[mask]
    return rows.iloc[0].to_dict() if not rows.empty else None


def build_replay_audit(
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
    effective_source: int,
    execution_mode: str,
    best_epoch: int | None = None,
) -> dict:
    """Compare reconstructed validation predictions with Experiment 12B metrics."""
    calculated = regression_metrics(frame["true_rul"], frame["predicted_rul"])
    expected = expected_raw_row(raw, regime, k, model_seed, source_seed)
    audit = {
        "regime": regime,
        "k": k,
        "model_seed": model_seed,
        "source_split_seed": effective_source,
        "execution_mode": execution_mode,
        "best_target_epoch": best_epoch,
        "calculated_rmse": calculated["rmse"],
        "calculated_mae": calculated["mae"],
        "calculated_nasa_score": calculated["nasa_score"],
        "reported_rmse": expected.get("rmse") if expected else np.nan,
        "reported_mae": expected.get("mae") if expected else np.nan,
        "reported_nasa_score": expected.get("nasa_score") if expected else np.nan,
    }
    for metric in ("rmse", "mae", "nasa_score"):
        calculated_value = audit[f"calculated_{metric}"]
        reported_value = audit[f"reported_{metric}"]
        audit[f"{metric}_absolute_difference"] = (
            abs(calculated_value - reported_value)
            if np.isfinite(reported_value) else np.nan
        )
    return audit


def build_config(args: argparse.Namespace, model_seed: int, experiment_dir: Path) -> dict:
    proxy = argparse.Namespace(**vars(args))
    proxy.output_dir = str(experiment_dir)
    proxy.source_task_seed = 0
    proxy.vary_source_split_by_seed = False
    proxy.seeds = list(args.model_seeds)
    cfg = exp11.load_config(proxy, model_seed)
    if args.device:
        cfg["device"] = args.device
    return cfg


def load_source_state(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少实验12B源模型缓存：{path}\n"
            "请确认--experiment12b-dir指向正式实验目录；实验13不会重新训练源域模型。"
        )
    payload = trusted_torch_load(path)
    state = payload.get("state")
    if not isinstance(state, dict) or not all_tensors_finite(state.values()):
        raise RuntimeError(f"源模型缓存无效或包含NaN/Inf：{path}")
    return state, payload


def predict_validation(model, loader, device: torch.device) -> pd.DataFrame:
    model = model.to(device).eval()
    true_values: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            true_values.extend(y.cpu().numpy().astype(float).tolist())
            predictions.extend(pred.cpu().numpy().astype(float).tolist())
    y = np.asarray(true_values, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    units = np.asarray(loader.dataset.units, dtype=int)
    if not (len(y) == len(pred) == len(units)):
        raise AssertionError("验证集标签、预测和发动机编号长度不一致")
    error = pred - y
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    if np.any(exponent > 80):
        raise FloatingPointError("NASA指数项过大，模型存在极端非有限风险")
    nasa = np.expm1(exponent)
    frame = pd.DataFrame(
        {
            "unit": units,
            "true_rul": y,
            "predicted_rul": pred,
            "error_pred_minus_true": error,
            "absolute_error": np.abs(error),
            "squared_error": error ** 2,
            "nasa_contribution": nasa,
            "is_late_prediction": error > 0,
            "is_early_prediction": error < 0,
            "stage": [STAGE_NAMES[i] for i in rul_stage_ids(y)],
        }
    )
    frame["window_index_within_engine"] = frame.groupby("unit").cumcount()
    return frame


def cell_filename(regime: str, k: int, model_seed: int, source_seed: int) -> str:
    shared = 0 if regime == "pretrained_budget_head" else source_seed
    return f"{regime}_k{k}_source{shared}_model{model_seed}.csv"


def train_or_load_cell(
    args: argparse.Namespace,
    input_paths: dict[str, Path],
    output: dict[str, Path],
    protocol: dict,
    raw: pd.DataFrame,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
    first_source_seed: int,
) -> tuple[pd.DataFrame, dict]:
    effective_source = 0 if regime == "pretrained_budget_head" else source_seed
    cell_path = output["cells"] / cell_filename(regime, k, model_seed, source_seed)
    if args.resume and cell_path.is_file():
        frame = pd.read_csv(cell_path)
        required = {
            "regime", "k", "model_seed", "source_split_seed", "true_rul",
            "predicted_rul", "unit", "window_index_within_engine",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"断点CSV缺少字段{sorted(missing)}：{cell_path}")
        audit = build_replay_audit(
            frame, raw, regime, k, model_seed, source_seed, effective_source,
            "resume_csv",
        )
        return frame, audit

    cfg = build_config(args, model_seed, input_paths["experiment"])
    adaptation_units = protocol["nested_adaptation_units_by_seed"][str(model_seed)][str(k)]
    loaders = prepare_kshot_experiment(
        cfg, args.preprocessing, args.balance_mode,
        protocol["validation_units"], adaptation_units,
    )
    source_tasks, support, validation, official_test, feature_count, split_info = loaders
    del source_tasks, official_test  # Explicitly do not evaluate official test.
    if split_info["validation_units"] != protocol["validation_units"]:
        raise AssertionError("验证发动机与锁定协议不一致")

    checkpoint = next(
        (
            p for p in checkpoint_candidates(
                input_paths["checkpoints"], args.target, regime, k, model_seed,
                source_seed, first_source_seed,
            ) if p.is_file()
        ),
        None,
    )
    execution_mode = "checkpoint"
    best_epoch = None
    if checkpoint is not None and args.mode in {"auto", "checkpoint"}:
        payload = trusted_torch_load(checkpoint)
        state = payload.get("model")
        if not isinstance(state, dict) or not all_tensors_finite(state.values()):
            raise RuntimeError(f"目标checkpoint无效：{checkpoint}")
        model = build_model("meta_gnn", feature_count, cfg)
        model.load_state_dict(state)
        best_epoch = payload.get("metrics", {}).get("best_target_epoch_by_validation")
    else:
        if args.mode == "checkpoint":
            raise FileNotFoundError(
                f"checkpoint模式下缺少目标模型：regime={regime}, K={k}, "
                f"source={source_seed}, model={model_seed}"
            )
        execution_mode = "replay_target_head"
        cache = source_cache_path(
            input_paths["experiment"], args.target, regime, model_seed, source_seed
        )
        state, _ = load_source_state(cache)
        seed_everything(model_seed)
        base_model = build_model("meta_gnn", feature_count, cfg)
        base_model.load_state_dict(state)
        device = resolve_device(cfg["device"])
        model, _, best_epoch, _, _, _ = train_target(
            base_model, support, validation, args, device,
            scope="rul_head", loss_mode="raw_mse",
        )
        if args.save_replayed_checkpoints:
            output["replayed_checkpoints"].mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {n: t.detach().cpu() for n, t in model.state_dict().items()},
                    "config": cfg,
                    "split": split_info,
                    "diagnostic_only": True,
                    "regime": regime,
                    "k": k,
                    "model_seed": model_seed,
                    "source_split_seed": effective_source,
                    "best_target_epoch_by_validation": best_epoch,
                },
                output["replayed_checkpoints"] /
                f"experiment13_{regime}_k{k}_source{effective_source}_model{model_seed}.pt",
            )

    device = resolve_device(cfg["device"])
    frame = predict_validation(model, validation, device)
    frame.insert(0, "regime", regime)
    frame.insert(1, "k", k)
    frame.insert(2, "model_seed", model_seed)
    frame.insert(3, "source_split_seed", effective_source)
    frame.insert(4, "execution_mode", execution_mode)
    frame.to_csv(cell_path, index=False, encoding="utf-8-sig")

    audit = build_replay_audit(
        frame, raw, regime, k, model_seed, source_seed, effective_source,
        execution_mode, best_epoch,
    )
    return frame, audit


def cell_metrics(group: pd.DataFrame) -> dict:
    metrics = regression_metrics(group["true_rul"], group["predicted_rul"])
    error = group["error_pred_minus_true"].to_numpy(float)
    nasa = group["nasa_contribution"].to_numpy(float)
    late = error > 0
    by_engine = group.groupby("unit", as_index=False)["nasa_contribution"].sum()
    ordered = np.sort(by_engine["nasa_contribution"].to_numpy(float))[::-1]
    total = float(nasa.sum())
    metrics.update(
        {
            "window_count": len(group),
            "engine_count": group["unit"].nunique(),
            "bias": float(error.mean()),
            "late_prediction_rate": float(late.mean()),
            "late_nasa_score": float(nasa[late].sum()),
            "early_nasa_score": float(nasa[~late].sum()),
            "late_nasa_fraction": float(nasa[late].sum() / total) if total > 0 else 0.0,
            "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)),
            "absolute_error_p99": float(np.quantile(np.abs(error), 0.99)),
            "max_absolute_error": float(np.abs(error).max()),
            "top1_engine_nasa_share": float(ordered[:1].sum() / total) if total > 0 else 0.0,
            "top3_engine_nasa_share": float(ordered[:3].sum() / total) if total > 0 else 0.0,
            "top5_engine_nasa_share": float(ordered[:5].sum() / total) if total > 0 else 0.0,
            "worst_engine_unit": int(by_engine.loc[by_engine.nasa_contribution.idxmax(), "unit"]),
            "worst_engine_nasa": float(ordered[0]),
        }
    )
    return metrics


def build_engine_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["regime", "k", "model_seed", "source_split_seed", "execution_mode", "unit"]
    rows = []
    for key, group in predictions.groupby(keys, dropna=False):
        error = group["error_pred_minus_true"].to_numpy(float)
        nasa = group["nasa_contribution"].to_numpy(float)
        metrics = regression_metrics(group["true_rul"], group["predicted_rul"])
        row = dict(zip(keys, key))
        row.update(metrics)
        row.update(
            {
                "window_count": len(group),
                "bias": float(error.mean()),
                "late_prediction_rate": float((error > 0).mean()),
                "late_nasa_score": float(nasa[error > 0].sum()),
                "early_nasa_score": float(nasa[error <= 0].sum()),
                "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)),
                "max_absolute_error": float(np.abs(error).max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = predictions[predictions.regime == "pretrained_budget_head"].copy()
    candidate = predictions[predictions.regime == "anil_engine_disjoint_head"].copy()
    if baseline.empty or candidate.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = baseline.rename(
        columns={
            "predicted_rul": "predicted_rul_reference",
            "error_pred_minus_true": "error_reference",
            "absolute_error": "absolute_error_reference",
            "squared_error": "squared_error_reference",
            "nasa_contribution": "nasa_contribution_reference",
        }
    )
    keep = [
        "k", "model_seed", "unit", "window_index_within_engine", "true_rul",
        "predicted_rul_reference", "error_reference", "absolute_error_reference",
        "squared_error_reference", "nasa_contribution_reference",
    ]
    paired = candidate.merge(
        baseline[keep],
        on=["k", "model_seed", "unit", "window_index_within_engine", "true_rul"],
        how="inner", validate="many_to_one",
    )
    paired = paired.rename(
        columns={
            "predicted_rul": "predicted_rul_candidate",
            "error_pred_minus_true": "error_candidate",
            "absolute_error": "absolute_error_candidate",
            "squared_error": "squared_error_candidate",
            "nasa_contribution": "nasa_contribution_candidate",
        }
    )
    for metric in ("absolute_error", "squared_error", "nasa_contribution"):
        paired[f"{metric}_delta"] = paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
    paired["late_status_candidate"] = paired["error_candidate"] > 0
    paired["late_status_reference"] = paired["error_reference"] > 0

    engine = paired.groupby(
        ["k", "model_seed", "source_split_seed", "unit"], as_index=False
    ).agg(
        window_count=("true_rul", "size"),
        true_rul_mean=("true_rul", "mean"),
        absolute_error_candidate=("absolute_error_candidate", "mean"),
        absolute_error_reference=("absolute_error_reference", "mean"),
        absolute_error_delta=("absolute_error_delta", "mean"),
        squared_error_candidate=("squared_error_candidate", "mean"),
        squared_error_reference=("squared_error_reference", "mean"),
        squared_error_delta=("squared_error_delta", "mean"),
        nasa_score_candidate=("nasa_contribution_candidate", "sum"),
        nasa_score_reference=("nasa_contribution_reference", "sum"),
        nasa_score_delta=("nasa_contribution_delta", "sum"),
        late_rate_candidate=("late_status_candidate", "mean"),
        late_rate_reference=("late_status_reference", "mean"),
    )
    engine["rmse_candidate"] = np.sqrt(engine["squared_error_candidate"])
    engine["rmse_reference"] = np.sqrt(engine["squared_error_reference"])
    engine["rmse_delta"] = engine["rmse_candidate"] - engine["rmse_reference"]
    return paired, engine


def group_tail_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse", "mae", "r2", "nasa_score", "bias", "late_prediction_rate",
        "late_nasa_fraction", "absolute_error_p95", "absolute_error_p99",
        "max_absolute_error", "top1_engine_nasa_share", "top3_engine_nasa_share",
        "top5_engine_nasa_share",
    ]
    rows = []
    for (k, regime), group in cell_summary.groupby(["k", "regime"]):
        row = {"k": k, "regime": regime, "n_cells": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "rmse_mean"])


def paired_cell_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["k", "model_seed", "source_split_seed"]
    for key, group in paired.groupby(keys):
        mse_candidate = float(group.squared_error_candidate.mean())
        mse_reference = float(group.squared_error_reference.mean())
        row = dict(zip(keys, key))
        row.update(
            {
                "rmse_candidate": math.sqrt(mse_candidate),
                "rmse_reference": math.sqrt(mse_reference),
                "rmse_delta": math.sqrt(mse_candidate) - math.sqrt(mse_reference),
                "mae_candidate": float(group.absolute_error_candidate.mean()),
                "mae_reference": float(group.absolute_error_reference.mean()),
                "mae_delta": float(group.absolute_error_delta.mean()),
                "nasa_score_candidate": float(group.nasa_contribution_candidate.sum()),
                "nasa_score_reference": float(group.nasa_contribution_reference.sum()),
                "nasa_score_delta": float(group.nasa_contribution_delta.sum()),
                "late_rate_candidate": float(group.late_status_candidate.mean()),
                "late_rate_reference": float(group.late_status_reference.mean()),
            }
        )
        row["both_rmse_and_nasa_improved"] = (
            row["rmse_delta"] < 0 and row["nasa_score_delta"] < 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def stage_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    return paired.groupby(["k", "stage"], as_index=False).agg(
        n_windows=("true_rul", "size"),
        absolute_error_candidate_mean=("absolute_error_candidate", "mean"),
        absolute_error_reference_mean=("absolute_error_reference", "mean"),
        absolute_error_delta_mean=("absolute_error_delta", "mean"),
        nasa_candidate_mean=("nasa_contribution_candidate", "mean"),
        nasa_reference_mean=("nasa_contribution_reference", "mean"),
        nasa_delta_mean=("nasa_contribution_delta", "mean"),
        nasa_delta_sum=("nasa_contribution_delta", "sum"),
        late_rate_candidate=("late_status_candidate", "mean"),
        late_rate_reference=("late_status_reference", "mean"),
    )


def make_conclusion(
    paired_cells: pd.DataFrame, paired_engines: pd.DataFrame,
    cell_summary: pd.DataFrame,
) -> dict:
    result = {"script_version": SCRIPT_VERSION, "evaluation_scope": "validation", "by_k": {}}
    for k, group in paired_cells.groupby("k"):
        engine_group = paired_engines[paired_engines.k == k]
        seed46 = group[group.model_seed == 46]
        result["by_k"][str(int(k))] = {
            "paired_cells": len(group),
            "rmse_win_rate": float((group.rmse_delta < 0).mean()),
            "nasa_win_rate": float((group.nasa_score_delta < 0).mean()),
            "rmse_and_nasa_joint_win_rate": float(group.both_rmse_and_nasa_improved.mean()),
            "rmse_delta_mean": float(group.rmse_delta.mean()),
            "mae_delta_mean": float(group.mae_delta.mean()),
            "nasa_score_delta_mean": float(group.nasa_score_delta.mean()),
            "engine_nasa_worsening_rate": float((engine_group.nasa_score_delta > 0).mean()),
            "seed46_rmse_delta_mean": float(seed46.rmse_delta.mean()) if len(seed46) else None,
            "seed46_nasa_delta_mean": float(seed46.nasa_score_delta.mean()) if len(seed46) else None,
        }
    candidate_cells = cell_summary[cell_summary.regime == "anil_engine_disjoint_head"]
    result["tail_concentration"] = {
        "top1_engine_nasa_share_mean": float(candidate_cells.top1_engine_nasa_share.mean()),
        "top3_engine_nasa_share_mean": float(candidate_cells.top3_engine_nasa_share.mean()),
        "top5_engine_nasa_share_mean": float(candidate_cells.top5_engine_nasa_share.mean()),
        "late_nasa_fraction_mean": float(candidate_cells.late_nasa_fraction.mean()),
    }
    result["interpretation_rule"] = {
        "rmse_delta_below_zero": "发动机互斥ANIL的RMSE更好",
        "nasa_delta_below_zero": "发动机互斥ANIL的NASA Score更好",
        "high_top_engine_share": "NASA风险由少数发动机集中贡献",
        "high_late_nasa_fraction": "NASA风险主要来自偏晚预测",
    }
    return result


def make_plots(
    paths: dict[str, Path], paired_cells: pd.DataFrame,
    paired_engines: pd.DataFrame, stages: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[警告] 未安装matplotlib，跳过图片输出。")
        return

    colors = {42: "#4C78A8", 43: "#F58518", 44: "#54A24B", 45: "#E45756", 46: "#B279A2"}
    fig, axes = plt.subplots(1, len(sorted(paired_cells.k.unique())), figsize=(10, 4), dpi=180)
    axes = np.atleast_1d(axes)
    for ax, (k, group) in zip(axes, paired_cells.groupby("k")):
        for seed, rows in group.groupby("model_seed"):
            ax.scatter(rows.rmse_delta, rows.nasa_score_delta, label=f"seed {seed}",
                       color=colors.get(int(seed)), s=35, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"K={int(k)}")
        ax.set_xlabel("RMSE delta (ANIL - budget transfer)")
        ax.set_ylabel("NASA Score delta")
        ax.grid(alpha=0.2)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 13: RMSE improvement versus NASA tail risk")
    fig.tight_layout()
    fig.savefig(paths["scatter"], bbox_inches="tight")
    plt.close(fig)

    seed46 = paired_engines[paired_engines.model_seed == 46]
    if not seed46.empty:
        top = (
            seed46.groupby(["k", "unit"], as_index=False).nasa_score_delta.mean()
            .sort_values("nasa_score_delta", ascending=False).head(15)
        )
        labels = [f"K={int(r.k)}, U={int(r.unit)}" for r in top.itertuples()]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        ax.barh(labels[::-1], top.nasa_score_delta.to_numpy()[::-1], color="#B279A2")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean engine NASA delta across source splits")
        ax.set_title("Seed 46: engines contributing most to NASA deterioration")
        fig.tight_layout()
        fig.savefig(paths["seed46"], bbox_inches="tight")
        plt.close(fig)

    if not stages.empty:
        pivot = stages.pivot(index="stage", columns="k", values="nasa_delta_mean")
        pivot = pivot.reindex(STAGE_NAMES)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
        pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean window NASA contribution delta")
        ax.set_title("NASA deterioration by RUL stage")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(paths["stage_plot"], bbox_inches="tight")
        plt.close(fig)


def dry_run_report(
    args: argparse.Namespace, inputs: dict[str, Path], protocol: dict,
    raw: pd.DataFrame, k_values, model_seeds, source_seeds, regimes,
) -> None:
    first_source = source_seeds[0]
    rows = []
    for regime in regimes:
        seeds_for_regime = [first_source] if regime == "pretrained_budget_head" else source_seeds
        for k in k_values:
            for model_seed in model_seeds:
                for source_seed in seeds_for_regime:
                    checkpoints = checkpoint_candidates(
                        inputs["checkpoints"], args.target, regime, k, model_seed,
                        source_seed, first_source,
                    )
                    cache = source_cache_path(
                        inputs["experiment"], args.target, regime, model_seed, source_seed
                    )
                    rows.append(
                        {
                            "regime": regime,
                            "k": k,
                            "model_seed": model_seed,
                            "source_split_seed": 0 if regime == "pretrained_budget_head" else source_seed,
                            "checkpoint_available": any(p.is_file() for p in checkpoints),
                            "source_cache_available": cache.is_file(),
                            "raw_result_available": expected_raw_row(
                                raw, regime, k, model_seed, source_seed
                            ) is not None,
                        }
                    )
    frame = pd.DataFrame(rows)
    planned = {
        "script_version": SCRIPT_VERSION,
        "evaluation_scope": "validation_only",
        "target": args.target,
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "regimes": regimes,
        "validation_engine_count": len(protocol["validation_units"]),
        "official_test_prediction_will_run": False,
        "unique_diagnostic_cells": len(frame),
        "target_checkpoints_available": int(frame.checkpoint_available.sum()),
        "source_caches_available": int(frame.source_cache_available.sum()),
        "raw_results_available": int(frame.raw_result_available.sum()),
        "mode": args.mode,
    }
    print(json.dumps(planned, ensure_ascii=False, indent=2))
    print("\n[dry-run cell availability]")
    print(frame.groupby("regime")[["checkpoint_available", "source_cache_available", "raw_result_available"]].sum())
    if args.mode == "checkpoint":
        missing = frame[~frame.checkpoint_available]
    elif args.mode == "replay":
        missing = frame[~frame.source_cache_available]
    else:
        missing = frame[
            ~frame.checkpoint_available & ~frame.source_cache_available
        ]
    if not missing.empty:
        print(f"\n[警告] 当前模式有{len(missing)}个单元缺少可用模型来源。")
        print(missing.head(20).to_string(index=False))


def main() -> None:
    args = parse_args()
    k_values, model_seeds, source_seeds, regimes = validate_args(args)
    inputs = resolve_inputs(args)
    protocol, raw = load_protocol_and_raw(inputs, args)

    for seed in model_seeds:
        if str(seed) not in protocol["nested_adaptation_units_by_seed"]:
            raise KeyError(f"协议缺少model_seed={seed}的目标发动机顺序")
        for k in k_values:
            if str(k) not in protocol["nested_adaptation_units_by_seed"][str(seed)]:
                raise KeyError(f"协议缺少model_seed={seed}, K={k}的适应发动机")

    if args.dry_run:
        dry_run_report(
            args, inputs, protocol, raw, k_values, model_seeds, source_seeds, regimes
        )
        return

    output = output_paths(args)
    output["output"].mkdir(parents=True, exist_ok=True)
    output["cells"].mkdir(parents=True, exist_ok=True)
    first_source = source_seeds[0]
    all_predictions: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for regime in regimes:
        seeds_for_regime = [first_source] if regime == "pretrained_budget_head" else source_seeds
        for model_seed in model_seeds:
            for k in k_values:
                for source_seed in seeds_for_regime:
                    print(
                        f"\n[experiment13] regime={regime} K={k} "
                        f"model_seed={model_seed} source_seed={source_seed}"
                    )
                    frame, audit = train_or_load_cell(
                        args, inputs, output, protocol, raw, regime, k,
                        model_seed, source_seed, first_source,
                    )
                    all_predictions.append(frame)
                    audit_rows.append(audit)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    predictions = pd.concat(all_predictions, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    predictions.to_csv(output["predictions"], index=False, encoding="utf-8-sig")
    audit.to_csv(output["audit"], index=False, encoding="utf-8-sig")

    cell_rows = []
    cell_keys = ["regime", "k", "model_seed", "source_split_seed", "execution_mode"]
    for key, group in predictions.groupby(cell_keys, dropna=False):
        row = dict(zip(cell_keys, key))
        row.update(cell_metrics(group))
        cell_rows.append(row)
    cells = pd.DataFrame(cell_rows)
    tails = group_tail_summary(cells)
    engines = build_engine_metrics(predictions)
    paired_windows, paired_engines = build_paired(predictions)
    paired_cells = paired_cell_summary(paired_windows) if not paired_windows.empty else pd.DataFrame()
    stages = stage_summary(paired_windows)

    if not paired_engines.empty:
        outliers = (
            paired_engines.sort_values(
                ["k", "model_seed", "source_split_seed", "nasa_score_delta"],
                ascending=[True, True, True, False],
            )
            .groupby(["k", "model_seed", "source_split_seed"], as_index=False)
            .head(args.top_n)
            .copy()
        )
        outliers["rank_within_cell"] = outliers.groupby(
            ["k", "model_seed", "source_split_seed"]
        )["nasa_score_delta"].rank(method="first", ascending=False).astype(int)
    else:
        outliers = pd.DataFrame()

    if not paired_cells.empty:
        model_summary = paired_cells.groupby(["k", "model_seed"], as_index=False).agg(
            source_split_count=("source_split_seed", "nunique"),
            rmse_delta_mean=("rmse_delta", "mean"),
            rmse_win_rate=("rmse_delta", lambda x: float((x < 0).mean())),
            mae_delta_mean=("mae_delta", "mean"),
            nasa_score_delta_mean=("nasa_score_delta", "mean"),
            nasa_win_rate=("nasa_score_delta", lambda x: float((x < 0).mean())),
            joint_win_rate=("both_rmse_and_nasa_improved", "mean"),
        )
        split_summary = paired_cells.groupby(["k", "source_split_seed"], as_index=False).agg(
            model_seed_count=("model_seed", "nunique"),
            rmse_delta_mean=("rmse_delta", "mean"),
            rmse_win_rate=("rmse_delta", lambda x: float((x < 0).mean())),
            mae_delta_mean=("mae_delta", "mean"),
            nasa_score_delta_mean=("nasa_score_delta", "mean"),
            nasa_win_rate=("nasa_score_delta", lambda x: float((x < 0).mean())),
            joint_win_rate=("both_rmse_and_nasa_improved", "mean"),
        )
        conclusion = make_conclusion(paired_cells, paired_engines, cells)
    else:
        model_summary = split_summary = pd.DataFrame()
        conclusion = {
            "script_version": SCRIPT_VERSION,
            "warning": "未同时运行预算基线和发动机互斥ANIL，无法形成配对结论",
        }

    cells.to_csv(output["cell_summary"], index=False, encoding="utf-8-sig")
    tails.to_csv(output["summary"], index=False, encoding="utf-8-sig")
    engines.to_csv(output["engine"], index=False, encoding="utf-8-sig")
    paired_windows.to_csv(output["paired_window"], index=False, encoding="utf-8-sig")
    paired_engines.to_csv(output["paired_engine"], index=False, encoding="utf-8-sig")
    outliers.to_csv(output["outliers"], index=False, encoding="utf-8-sig")
    stages.to_csv(output["stage"], index=False, encoding="utf-8-sig")
    model_summary.to_csv(output["model_seed"], index=False, encoding="utf-8-sig")
    split_summary.to_csv(output["source_split"], index=False, encoding="utf-8-sig")
    atomic_write_text(
        output["conclusion"], json.dumps(conclusion, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        output["protocol"],
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "target": args.target,
                "evaluation_scope": "validation_only",
                "official_test_prediction_run": False,
                "experiment12b_dir": str(inputs["experiment"]),
                "protocol_file": str(inputs["protocol"]),
                "raw_results_file": str(inputs["raw"]),
                "k_values": k_values,
                "model_seeds": model_seeds,
                "source_task_seeds": source_seeds,
                "regimes": regimes,
                "mode": args.mode,
                "target_epochs": args.target_epochs,
                "target_lr": args.target_lr,
                "preprocessing": args.preprocessing,
                "balance_mode": args.balance_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    if not args.no_plots and not paired_cells.empty:
        make_plots(output, paired_cells, paired_engines, stages)

    difference_columns = [
        column for column in audit.columns
        if column.endswith("_absolute_difference")
    ]
    max_diff = (
        audit[difference_columns].max(numeric_only=True).max()
        if difference_columns else float("nan")
    )
    if np.isfinite(max_diff) and max_diff > args.metric_tolerance:
        print(
            f"\n[警告] 重放指标与实验12B报告值最大差异={max_diff:.6g}，"
            f"超过容差{args.metric_tolerance}。请检查代码版本、缓存和参数。"
        )

    print("\n[实验13核心结论数据]")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print("\n[输出文件]")
    for name, path in output.items():
        if path.exists() and path.is_file():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
