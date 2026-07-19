#!/usr/bin/env python3
"""实验15：少样本ANIL的风险权重、梯度裁剪与发动机级CVaR稳定性实验。

实验14得到两个关键结论：

1. K=5时风险感知损失能够改善RMSE，并缓解seed=46、发动机48和high-RUL
   （高剩余寿命）窗口的极端NASA损失；
2. K=2时原风险权重过强，结果对模型种子和源域发动机划分高度敏感。

实验15只使用固定validation（验证集），绝不对官方test执行模型前向。它复用：

* 实验12B的源域模型缓存；
* 实验14已经完成的三组参考结果和窗口预测，不重复训练参考组。

参考组（直接从实验14读取）
----------------------------
``budget_mse_rmse``
    预算匹配普通迁移学习。
``anil_mse_rmse``
    当前ANIL：MSE训练、按最低验证RMSE选择epoch。
``anil_exp14_risk_tail``
    实验14组合方案：较强风险权重、无梯度裁剪、尾部约束选轮。

新增训练组
----------
``anil_lowrisk_tail``
    将NASA权重由1.0降至0.25，将high-RUL低估权重由0.25降至0.05。
``anil_lowrisk_clip_tail``
    在低风险权重基础上加入梯度裁剪（默认范数1000）。
``anil_cvar_clip_tail``
    再加入发动机级CVaR（Conditional Value at Risk，条件风险价值），直接惩罚
    当前batch内风险最高的一部分发动机，默认关注最差50%的发动机。

三组新增方案使用完全相同的支持集batch、学习率、训练轮数和源模型，只更新
``predictor.*``（RUL预测头）。默认5个模型种子×5个源划分×K={2,5}，新增
150次目标适应训练。

服务器dry-run：

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment15_cvar_stability.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --experiment14-dir outputs/experiment14_tail_robust_adaptation \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --dry-run

正式运行：

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment15_cvar_stability.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --experiment14-dir outputs/experiment14_tail_robust_adaptation \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --output-dir outputs/experiment15_cvar_stability \
      --resume
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from scripts import experiment14_tail_robust_adaptation as exp14  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    BALANCE_MODES,
    EXPECTED_OFFICIAL_TEST_ENGINES,
    PREPROCESSING_MODES,
    atomic_write_text,
    prepare_kshot_experiment,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.experiment9_anil_ablation import parameter_drift  # noqa: E402
from scripts.experiment10b_anil_stability import all_tensors_finite  # noqa: E402
from scripts.experiment10c_target_kshot import (  # noqa: E402
    set_target_mode,
    set_target_scope,
    tensor_norm,
)


SCRIPT_VERSION = "experiment15_cvar_stability_v1"
REFERENCE_REGIMES = (
    "budget_mse_rmse",
    "anil_mse_rmse",
    "anil_exp14_risk_tail",
)
TRAIN_REGIMES = (
    "anil_lowrisk_tail",
    "anil_lowrisk_clip_tail",
    "anil_cvar_clip_tail",
)
COMPARISONS = (
    (
        "anil_lowrisk_tail",
        "anil_exp14_risk_tail",
        "lower_weights_vs_exp14_candidate",
    ),
    (
        "anil_lowrisk_clip_tail",
        "anil_lowrisk_tail",
        "gradient_clipping_effect",
    ),
    (
        "anil_cvar_clip_tail",
        "anil_lowrisk_clip_tail",
        "engine_cvar_effect",
    ),
    (
        "anil_cvar_clip_tail",
        "anil_mse_rmse",
        "cvar_vs_current_anil",
    ),
    (
        "anil_cvar_clip_tail",
        "anil_exp14_risk_tail",
        "cvar_vs_exp14_candidate",
    ),
    (
        "anil_cvar_clip_tail",
        "budget_mse_rmse",
        "cvar_vs_budget_transfer",
    ),
)
PRIMARY_COMPARISON = "cvar_vs_current_anil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验15：低风险权重、梯度裁剪和发动机级CVaR稳定性"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--model-seeds", "--seeds", dest="model_seeds", nargs="+", type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--source-task-seeds", nargs="+", type=int,
        default=[2027, 2028, 2029, 2030, 2031],
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=TRAIN_REGIMES,
        default=list(TRAIN_REGIMES),
    )
    parser.add_argument(
        "--experiment12b-dir", default="outputs/experiment12b_budget_matched_final"
    )
    parser.add_argument(
        "--experiment14-dir", default="outputs/experiment14_tail_robust_adaptation"
    )
    parser.add_argument("--protocol-file")
    parser.add_argument("--raw-results-file")
    parser.add_argument("--experiment14-raw-file")
    parser.add_argument("--experiment14-predictions-file")
    parser.add_argument("--device")
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES,
        default="condition_settings",
    )
    parser.add_argument(
        "--balance-mode", choices=BALANCE_MODES, default="engine_stage"
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)

    # 与实验12B源状态配置保持一致；实验15不重新训练源模型。
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

    # 目标域训练预算。
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)

    # 实验15预注册参数。不要根据官方测试结果修改。
    parser.add_argument("--low-nasa-loss-weight", type=float, default=0.25)
    parser.add_argument("--low-high-rul-loss-weight", type=float, default=0.05)
    parser.add_argument("--high-rul-threshold", type=float, default=90.0)
    parser.add_argument("--nasa-exp-clip", type=float, default=6.0)
    parser.add_argument("--target-clip-norm", type=float, default=1000.0)
    parser.add_argument("--engine-cvar-weight", type=float, default=0.25)
    parser.add_argument(
        "--engine-cvar-alpha", type=float, default=0.50,
        help="CVaR分位点；0.5表示平均当前batch风险最高的50%发动机",
    )
    parser.add_argument(
        "--selection-rmse-tolerance", type=float, default=0.02,
        help="在最低验证RMSE的2%容差内选择NASA Score最低epoch",
    )

    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="outputs/experiment15_cvar_stability"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
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
        raise ValueError("模型种子、源域划分种子和训练方案不能为空")
    positive = {
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "nasa_exp_clip": args.nasa_exp_clip,
        "target_clip_norm": args.target_clip_norm,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正数：{invalid}")
    nonnegative = {
        "target_weight_decay": args.target_weight_decay,
        "low_nasa_loss_weight": args.low_nasa_loss_weight,
        "low_high_rul_loss_weight": args.low_high_rul_loss_weight,
        "engine_cvar_weight": args.engine_cvar_weight,
        "selection_rmse_tolerance": args.selection_rmse_tolerance,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise ValueError(f"以下参数不能为负数：{invalid}")
    if not 0 <= args.engine_cvar_alpha < 1:
        raise ValueError("--engine-cvar-alpha必须位于[0,1)")
    if len(model_seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个模型种子，只能视为预实验。")
    if len(source_seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个源域划分种子，只能视为预实验。")
    return k_values, model_seeds, source_seeds, regimes


def regime_spec(regime: str, args: argparse.Namespace) -> dict:
    common = {
        "source": "anil_engine_disjoint",
        "selection": "tail_constrained",
        "nasa_weight": float(args.low_nasa_loss_weight),
        "high_weight": float(args.low_high_rul_loss_weight),
        "cvar_weight": 0.0,
        "clip_norm": 0.0,
    }
    specs = {
        "anil_lowrisk_tail": {
            **common,
            "description": "lower_risk_weights_without_clipping",
        },
        "anil_lowrisk_clip_tail": {
            **common,
            "clip_norm": float(args.target_clip_norm),
            "description": "lower_risk_weights_plus_gradient_clipping",
        },
        "anil_cvar_clip_tail": {
            **common,
            "clip_norm": float(args.target_clip_norm),
            "cvar_weight": float(args.engine_cvar_weight),
            "description": "lower_risk_weights_plus_clipping_plus_engine_cvar",
        },
    }
    return specs[regime]


def first_existing(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到{description}，已检查：\n  {checked}")


def experiment14_inputs(args: argparse.Namespace) -> dict[str, Path]:
    directory = resolve_path(args.experiment14_dir, PROJECT_ROOT)
    raw = (
        resolve_path(args.experiment14_raw_file, PROJECT_ROOT)
        if args.experiment14_raw_file else first_existing(
            [directory / f"experiment14_{args.target}_raw.json"],
            "实验14 raw结果",
        )
    )
    predictions = (
        resolve_path(args.experiment14_predictions_file, PROJECT_ROOT)
        if args.experiment14_predictions_file else first_existing(
            [directory / f"experiment14_{args.target}_window_predictions.csv"],
            "实验14窗口预测",
        )
    )
    return {"directory": directory, "raw": raw, "predictions": predictions}


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment15_{args.target}"
    return {
        "output": output,
        "cells": output / "cells",
        "checkpoints": output / "checkpoints",
        "raw": output / f"{prefix}_raw.json",
        "history": output / f"{prefix}_history.json",
        "predictions": output / f"{prefix}_window_predictions.csv",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_by_cell.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "engine": output / f"{prefix}_engine_deltas.csv",
        "stage": output / f"{prefix}_stage_deltas.csv",
        "tail": output / f"{prefix}_tail_diagnostics.csv",
        "model_seed": output / f"{prefix}_per_model_seed.csv",
        "source_split": output / f"{prefix}_per_source_split.csv",
        "reference_audit": output / f"{prefix}_reference_import_audit.csv",
        "grid": output / f"{prefix}_grid_plan.json",
        "protocol": output / f"{prefix}_protocol.json",
        "conclusion": output / f"{prefix}_conclusion.json",
    }


def load_reference_results(
    inputs: dict[str, Path],
    k_values: list[int],
    model_seeds: list[int],
    source_seeds: list[int],
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    raw_all = json.loads(inputs["raw"].read_text(encoding="utf-8"))
    raw = pd.DataFrame(raw_all)
    predictions = pd.read_csv(inputs["predictions"])
    if raw.empty or predictions.empty:
        raise ValueError("实验14参考文件为空")
    if set(raw.evaluation_scope.unique()) != {"validation"}:
        raise ValueError("实验15只允许导入实验14的validation结果")
    official_flag = raw.get(
        "official_test_prediction_run",
        pd.Series(False, index=raw.index),
    )
    if bool(official_flag.astype(bool).any()):
        raise ValueError("实验14参考结果包含官方测试预测，拒绝导入")

    mapping = {
        "budget_mse_rmse": "budget_mse_rmse",
        "anil_mse_rmse": "anil_mse_rmse",
        "anil_risk_tail": "anil_exp14_risk_tail",
    }
    rows: list[dict] = []
    audit_rows = []
    prediction_frames = []
    for old, new in mapping.items():
        selected = raw[
            raw.regime.eq(old)
            & raw.k.astype(int).isin(k_values)
            & raw.model_seed.astype(int).isin(model_seeds)
        ].copy()
        selected_predictions = predictions[
            predictions.regime.eq(old)
            & predictions.k.astype(int).isin(k_values)
            & predictions.model_seed.astype(int).isin(model_seeds)
        ].copy()
        if old != "budget_mse_rmse":
            selected = selected[
                selected.source_split_seed.astype(int).isin(source_seeds)
            ]
            selected_predictions = selected_predictions[
                selected_predictions.source_split_seed.astype(int).isin(source_seeds)
            ]
        selected["regime"] = new
        selected_predictions["regime"] = new
        selected["reference_imported_from_experiment14"] = True
        rows.extend(selected.to_dict(orient="records"))
        prediction_frames.append(selected_predictions)
        expected = (
            len(k_values) * len(model_seeds)
            if old == "budget_mse_rmse"
            else len(k_values) * len(model_seeds) * len(source_seeds)
        )
        audit_rows.append(
            {
                "experiment14_regime": old,
                "experiment15_regime": new,
                "expected_cells": expected,
                "imported_cells": len(selected),
                "complete": len(selected) == expected,
                "raw_file": str(inputs["raw"]),
                "prediction_file": str(inputs["predictions"]),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not bool(audit.complete.all()):
        raise ValueError("实验14参考结果不完整，请检查reference_import_audit")
    return rows, pd.concat(prediction_frames, ignore_index=True), audit


def materialize_support_schedule_with_units(support, epochs: int):
    """固定batch索引，并保留每个窗口所属发动机供CVaR聚合。"""
    dataset = support.dataset
    if getattr(dataset, "units", None) is None:
        raise ValueError("support dataset缺少发动机编号，无法计算发动机级CVaR")
    units_array = np.asarray(dataset.units, dtype=np.int64)
    schedule = []
    for _ in range(epochs):
        batches = []
        for batch_indices in support.batch_sampler:
            indices = np.asarray([int(index) for index in batch_indices], dtype=np.int64)
            index_tensor = torch.as_tensor(indices, dtype=torch.long)
            x = dataset.x[index_tensor].detach().cpu().clone()
            y = dataset.y[index_tensor].detach().cpu().clone()
            units = torch.as_tensor(units_array[indices], dtype=torch.long)
            batches.append((x, y, units))
        schedule.append(batches)
    return schedule


def engine_cvar(
    window_risk: torch.Tensor,
    units: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, int, int]:
    """Average the worst (1-alpha) fraction of per-engine mean risks."""
    unique_units = torch.unique(units)
    engine_risks = torch.stack(
        [window_risk[units == unit].mean() for unit in unique_units]
    )
    tail_count = max(1, int(math.ceil((1.0 - alpha) * len(engine_risks))))
    value = torch.topk(engine_risks, k=tail_count, largest=True).values.mean()
    return value, int(len(engine_risks)), tail_count


def target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    units: torch.Tensor,
    spec: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    error = prediction - target
    mse = torch.mean(error**2)
    exponent = torch.where(error < 0, -error / 13.0, error / 10.0)
    nasa_window = torch.expm1(
        torch.clamp(exponent, min=0.0, max=args.nasa_exp_clip)
    )
    nasa = nasa_window.mean()
    high_mask = target > args.high_rul_threshold
    if bool(high_mask.any().item()):
        high_under = torch.mean(
            torch.relu(target[high_mask] - prediction[high_mask]) ** 2
        )
    else:
        high_under = torch.zeros(
            (), device=prediction.device, dtype=prediction.dtype
        )
    cvar, engine_count, tail_count = engine_cvar(
        nasa_window, units, args.engine_cvar_alpha
    )
    total = (
        mse
        + spec["nasa_weight"] * nasa
        + spec["high_weight"] * high_under
        + spec["cvar_weight"] * cvar
    )
    return total, {
        "mse": mse,
        "nasa": nasa,
        "high_under": high_under,
        "engine_cvar": cvar,
        "engine_count": engine_count,
        "cvar_tail_engine_count": tail_count,
    }


def train_target(
    model: torch.nn.Module,
    support_schedule,
    validation,
    args: argparse.Namespace,
    device: torch.device,
    spec: dict,
) -> tuple[torch.nn.Module, list[dict], int, dict, dict]:
    learner = deepcopy(model).to(device)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in learner.named_parameters()
    }
    trainable = set_target_scope(learner, "rul_head")
    optimizer = torch.optim.Adam(
        trainable, lr=args.target_lr, weight_decay=args.target_weight_decay
    )
    history = []
    epoch_states = {}
    max_grad_norm = 0.0
    gradient_steps = 0
    clip_events = 0
    all_batch_engine_counts = []
    all_cvar_tail_counts = []

    for epoch, batches in enumerate(support_schedule, start=1):
        set_target_mode(learner, "rul_head")
        component_rows = []
        for x_cpu, y_cpu, unit_cpu in batches:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            units = unit_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = learner(x)
            loss, components = target_loss(prediction, y, units, spec, args)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"目标损失非有限：epoch={epoch}")
            loss.backward()
            gradients = [
                parameter.grad for parameter in trainable
                if parameter.grad is not None
            ]
            grad_norm = tensor_norm(gradients)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"目标梯度非有限：epoch={epoch}")
            max_grad_norm = max(max_grad_norm, grad_norm)
            gradient_steps += 1
            clip_norm = float(spec["clip_norm"])
            if clip_norm > 0:
                if grad_norm > clip_norm:
                    clip_events += 1
                torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
            optimizer.step()
            if not all_tensors_finite(learner.parameters()):
                raise FloatingPointError(f"目标参数非有限：epoch={epoch}")
            component_rows.append(
                {
                    "total": float(loss.detach().cpu().item()),
                    "mse": float(components["mse"].detach().cpu().item()),
                    "nasa": float(components["nasa"].detach().cpu().item()),
                    "high_under": float(
                        components["high_under"].detach().cpu().item()
                    ),
                    "engine_cvar": float(
                        components["engine_cvar"].detach().cpu().item()
                    ),
                }
            )
            all_batch_engine_counts.append(int(components["engine_count"]))
            all_cvar_tail_counts.append(
                int(components["cvar_tail_engine_count"])
            )

        validation_frame = exp14.predict_validation(learner, validation, device)
        metrics = exp14.prediction_metrics(
            validation_frame, args.high_rul_threshold
        )
        component_frame = pd.DataFrame(component_rows)
        row = {
            "epoch": epoch,
            "train_total_loss": float(component_frame.total.mean()),
            "train_mse_component": float(component_frame.mse.mean()),
            "train_nasa_component": float(component_frame.nasa.mean()),
            "train_high_rul_component": float(component_frame.high_under.mean()),
            "train_engine_cvar_component": float(
                component_frame.engine_cvar.mean()
            ),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        epoch_states[epoch] = exp14.cpu_state(learner)
        print(
            f"target_epoch={epoch:03d}/{args.target_epochs} "
            f"clip={spec['clip_norm']:.0f} cvar={spec['cvar_weight']:.3f} "
            f"train={row['train_total_loss']:.4f} "
            f"val_rmse={metrics['rmse']:.4f} "
            f"val_nasa={metrics['nasa_score']:.2f} "
            f"worst_engine={metrics['worst_engine_unit']}"
        )

    best_epoch, selection_diagnostic = exp14.select_epoch(
        history, spec["selection"], args.selection_rmse_tolerance
    )
    learner.load_state_dict(epoch_states[best_epoch])
    for row in history:
        row["selected"] = row["epoch"] == best_epoch
    diagnostics = {
        **selection_diagnostic,
        "target_max_grad_norm": max_grad_norm,
        "target_gradient_steps": gradient_steps,
        "target_clip_events": clip_events,
        "target_clip_rate": clip_events / max(1, gradient_steps),
        "target_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable)
        ),
        "batch_engine_count_min": int(min(all_batch_engine_counts)),
        "batch_engine_count_mean": float(np.mean(all_batch_engine_counts)),
        "batch_engine_count_max": int(max(all_batch_engine_counts)),
        "cvar_tail_engine_count_mean": float(np.mean(all_cvar_tail_counts)),
        "parameter_drift_by_group": parameter_drift(before, learner),
    }
    return learner, history, best_epoch, diagnostics, epoch_states[best_epoch]


def cell_signature(args: argparse.Namespace, regime: str) -> dict:
    spec = regime_spec(regime, args)
    return {
        "script_version": SCRIPT_VERSION,
        "regime": regime,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "target_weight_decay": args.target_weight_decay,
        "nasa_exp_clip": args.nasa_exp_clip,
        "high_rul_threshold": args.high_rul_threshold,
        "selection_rmse_tolerance": args.selection_rmse_tolerance,
        "engine_cvar_alpha": args.engine_cvar_alpha,
        **spec,
    }


def cell_paths(
    output: dict[str, Path], regime: str, k: int,
    model_seed: int, source_seed: int,
) -> tuple[Path, Path]:
    stem = f"{regime}_k{k}_source{source_seed}_model{model_seed}"
    return output["cells"] / f"{stem}.json", output["cells"] / f"{stem}.csv"


def cached_cell(
    args: argparse.Namespace,
    output: dict[str, Path],
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
):
    metadata, prediction = cell_paths(
        output, regime, k, model_seed, source_seed
    )
    if not metadata.is_file() or not prediction.is_file():
        return None
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("signature") != cell_signature(args, regime):
        return None
    return (
        payload["result"], payload.get("history", []),
        pd.read_csv(prediction),
    )


def save_cell(
    args: argparse.Namespace,
    output: dict[str, Path],
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
    result: dict,
    history: list[dict],
    predictions: pd.DataFrame,
) -> None:
    metadata, prediction = cell_paths(
        output, regime, k, model_seed, source_seed
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    exp14.atomic_to_csv(predictions, prediction)
    atomic_write_text(
        metadata,
        json.dumps(
            {
                "signature": cell_signature(args, regime),
                "result": result,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def run_cell(
    args: argparse.Namespace,
    output: dict[str, Path],
    cfg: dict,
    regime: str,
    source_state: dict,
    support_schedule,
    validation,
    split_info: dict,
    feature_count: int,
    k: int,
    model_seed: int,
    source_seed: int,
):
    if args.resume:
        cached = cached_cell(
            args, output, regime, k, model_seed, source_seed
        )
        if cached is not None:
            print(
                f"[resume] regime={regime} K={k} model={model_seed} "
                f"source={source_seed}"
            )
            return cached

    spec = regime_spec(regime, args)
    seed_everything(model_seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    device = resolve_device(cfg["device"])
    model, history, best_epoch, diagnostics, selected_state = train_target(
        model, support_schedule, validation, args, device, spec
    )
    frame = exp14.predict_validation(model, validation, device)
    metrics = exp14.prediction_metrics(frame, args.high_rul_threshold)
    frame.insert(0, "regime", regime)
    frame.insert(1, "k", k)
    frame.insert(2, "model_seed", model_seed)
    frame.insert(3, "source_split_seed", source_seed)
    result = {
        **metrics,
        "evaluation_scope": "validation",
        "regime": regime,
        "source_training": "engine_disjoint_anil",
        "source_state_key": spec["source"],
        "target_loss_mode": "low_risk_engine_cvar",
        "epoch_selection_mode": spec["selection"],
        "target_adaptation_scope": "rul_head",
        "target_domain": args.target,
        "k": int(k),
        "model_seed": int(model_seed),
        "source_split_seed": int(source_seed),
        "adaptation_engine_count": int(k),
        "adaptation_units": [int(unit) for unit in split_info["adaptation_units"]],
        "validation_engine_count": len(split_info["validation_units"]),
        "validation_units": [int(unit) for unit in split_info["validation_units"]],
        "official_test_engine_count": int(split_info["official_test_engine_count"]),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "official_test_prediction_run": False,
        "reference_imported_from_experiment14": False,
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": int(args.target_epochs),
        "target_lr": float(args.target_lr),
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "nasa_loss_weight": spec["nasa_weight"],
        "high_rul_loss_weight": spec["high_weight"],
        "engine_cvar_weight": spec["cvar_weight"],
        "engine_cvar_alpha": float(args.engine_cvar_alpha),
        "target_clip_norm": spec["clip_norm"],
        "high_rul_threshold": float(args.high_rul_threshold),
        "nasa_exp_clip": float(args.nasa_exp_clip),
        "selection_rmse_tolerance": float(args.selection_rmse_tolerance),
        **diagnostics,
    }
    save_cell(
        args, output, regime, k, model_seed, source_seed,
        result, history, frame,
    )
    if args.save_checkpoints:
        output["checkpoints"].mkdir(parents=True, exist_ok=True)
        checkpoint = output["checkpoints"] / (
            f"experiment15_{regime}_k{k}_{args.target}_"
            f"source{source_seed}_model{model_seed}.pt"
        )
        torch.save(
            {
                "model": selected_state,
                "result": result,
                "history": history,
                "split": split_info,
                "signature": cell_signature(args, regime),
            },
            checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, history, frame


def comparison_outputs(raw: pd.DataFrame, predictions: pd.DataFrame, repetitions: int):
    """Reuse the tested Experiment 14 paired-statistics implementation."""
    old_comparisons = exp14.COMPARISONS
    try:
        exp14.COMPARISONS = COMPARISONS
        paired = exp14.build_paired_cells(raw)
        comparisons = exp14.comparison_summary(paired, repetitions)
        engines, stages = exp14.detailed_deltas(predictions)
        tails = exp14.tail_diagnostics(paired, engines, stages)
    finally:
        exp14.COMPARISONS = old_comparisons
    return paired, comparisons, engines, stages, tails


def add_factor_robustness(
    comparisons: pd.DataFrame,
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows = []
    source_rows = []
    robustness_rows = []
    for (k, comparison), group in paired.groupby(["k", "comparison"]):
        by_model = group.groupby("model_seed", as_index=False).agg(
            rmse_delta_mean=("rmse_delta", "mean"),
            nasa_score_delta_mean=("nasa_score_delta", "mean"),
            worst_engine_nasa_delta_mean=("worst_engine_nasa_delta", "mean"),
        )
        by_model.insert(0, "comparison", comparison)
        by_model.insert(0, "k", int(k))
        model_rows.append(by_model)
        by_source = group.groupby("source_split_seed", as_index=False).agg(
            rmse_delta_mean=("rmse_delta", "mean"),
            nasa_score_delta_mean=("nasa_score_delta", "mean"),
            worst_engine_nasa_delta_mean=("worst_engine_nasa_delta", "mean"),
        )
        by_source.insert(0, "comparison", comparison)
        by_source.insert(0, "k", int(k))
        source_rows.append(by_source)
        robustness_rows.append(
            {
                "k": int(k),
                "comparison": comparison,
                "model_seed_rmse_win_rate": float(
                    (by_model.rmse_delta_mean < 0).mean()
                ),
                "model_seed_nasa_win_rate": float(
                    (by_model.nasa_score_delta_mean < 0).mean()
                ),
                "source_split_rmse_win_rate": float(
                    (by_source.rmse_delta_mean < 0).mean()
                ),
                "source_split_nasa_win_rate": float(
                    (by_source.nasa_score_delta_mean < 0).mean()
                ),
                "nasa_delta_worst_cell": float(group.nasa_score_delta.max()),
                "nasa_delta_best_cell": float(group.nasa_score_delta.min()),
                "nasa_delta_p95": float(
                    np.quantile(group.nasa_score_delta, 0.95)
                ),
            }
        )
    model = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    source = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
    robustness = pd.DataFrame(robustness_rows)
    comparisons = comparisons.merge(
        robustness, on=["k", "comparison"], how="left", validate="one_to_one"
    )
    return comparisons, model, source


def make_conclusion(comparisons: pd.DataFrame, tails: pd.DataFrame) -> dict:
    conclusion = {
        "script_version": SCRIPT_VERSION,
        "evaluation_scope": "validation_only",
        "official_test_prediction_run": False,
        "primary_comparison": PRIMARY_COMPARISON,
        "success_rule": {
            "rmse_degradation_at_most_pct": 1.0,
            "mean_nasa_delta_below": 0.0,
            "cell_nasa_win_rate_at_least": 0.8,
            "nasa_bootstrap_ci95_upper_below": 0.0,
            "model_seed_nasa_win_rate_at_least": 0.8,
            "source_split_nasa_win_rate_at_least": 0.8,
            "worst_engine_nasa_delta_below": 0.0,
            "seed46_unit48_high_rul_all_improve": True,
        },
        "by_k": {},
    }
    primary = comparisons[comparisons.comparison == PRIMARY_COMPARISON]
    primary_tail = tails[tails.comparison == PRIMARY_COMPARISON]
    for _, row in primary.iterrows():
        k = int(row.k)
        selected_tail = primary_tail[primary_tail.k == k]
        tail = selected_tail.iloc[0] if not selected_tail.empty else None
        checks = {
            "rmse_preserved": row.rmse_change_pct <= 1.0,
            "mean_nasa_improved": row.nasa_score_delta_mean < 0,
            "cell_nasa_win_rate": row.nasa_win_rate >= 0.8,
            "nasa_ci_below_zero": row.nasa_boot_ci95_high < 0,
            "model_seed_nasa_win_rate": row.model_seed_nasa_win_rate >= 0.8,
            "source_split_nasa_win_rate": row.source_split_nasa_win_rate >= 0.8,
            "worst_engine_nasa_improved": row.worst_engine_nasa_delta_mean < 0,
            "seed46_improved": (
                tail is not None and tail.seed46_nasa_delta_mean < 0
            ),
            "unit48_improved": (
                tail is not None and tail.unit48_nasa_delta_mean < 0
            ),
            "high_rul_improved": (
                tail is not None
                and tail.high_rul_nasa_delta_mean_per_cell < 0
            ),
        }
        conclusion["by_k"][str(k)] = {
            "rmse_delta_mean": float(row.rmse_delta_mean),
            "rmse_change_pct": float(row.rmse_change_pct),
            "nasa_score_delta_mean": float(row.nasa_score_delta_mean),
            "nasa_win_rate": float(row.nasa_win_rate),
            "nasa_boot_ci95": [
                float(row.nasa_boot_ci95_low),
                float(row.nasa_boot_ci95_high),
            ],
            "model_seed_nasa_win_rate": float(row.model_seed_nasa_win_rate),
            "source_split_nasa_win_rate": float(
                row.source_split_nasa_win_rate
            ),
            "seed46_nasa_delta_mean": (
                float(tail.seed46_nasa_delta_mean) if tail is not None else None
            ),
            "unit48_nasa_delta_mean": (
                float(tail.unit48_nasa_delta_mean) if tail is not None else None
            ),
            "high_rul_nasa_delta_mean_per_cell": (
                float(tail.high_rul_nasa_delta_mean_per_cell)
                if tail is not None else None
            ),
            "checks": {name: bool(value) for name, value in checks.items()},
            "strict_success": bool(all(checks.values())),
        }
    conclusion["all_k_strict_success"] = bool(
        conclusion["by_k"]
        and all(item["strict_success"] for item in conclusion["by_k"].values())
    )
    return conclusion


def dry_run_report(
    args: argparse.Namespace,
    exp12_inputs: dict[str, Path],
    exp14_inputs: dict[str, Path],
    protocol: dict,
    references: list[dict],
    audit: pd.DataFrame,
    k_values: list[int],
    model_seeds: list[int],
    source_seeds: list[int],
    regimes: list[str],
) -> None:
    cache_rows = []
    for model_seed in model_seeds:
        for source_seed in source_seeds:
            path = exp14.source_cache_path(
                exp12_inputs["experiment"], args.target,
                "anil_engine_disjoint", model_seed, source_seed,
            )
            cache_rows.append(
                {
                    "model_seed": model_seed,
                    "source_split_seed": source_seed,
                    "available": path.is_file(),
                    "path": str(path),
                }
            )
    caches = pd.DataFrame(cache_rows)
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "validation_only",
        "official_test_prediction_will_run": False,
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "reference_regimes_imported": list(REFERENCE_REGIMES),
        "new_regimes": regimes,
        "reference_cells_imported": len(references),
        "planned_new_target_trainings": (
            len(k_values) * len(model_seeds) * len(source_seeds) * len(regimes)
        ),
        "source_cache_count": len(caches),
        "source_cache_available": int(caches.available.sum()),
        "validation_engine_count": len(protocol["validation_units"]),
        "pre_registered_parameters": {
            "low_nasa_loss_weight": args.low_nasa_loss_weight,
            "low_high_rul_loss_weight": args.low_high_rul_loss_weight,
            "target_clip_norm": args.target_clip_norm,
            "engine_cvar_weight": args.engine_cvar_weight,
            "engine_cvar_alpha": args.engine_cvar_alpha,
            "selection_rmse_tolerance": args.selection_rmse_tolerance,
        },
        "experiment14_raw": str(exp14_inputs["raw"]),
        "experiment14_predictions": str(exp14_inputs["predictions"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n[实验14参考导入审计]")
    print(audit.to_string(index=False))
    print("\n[实验12B源缓存]")
    print(caches.available.value_counts().to_string())
    if not bool(caches.available.all()):
        print("\n[缺失缓存]")
        print(caches[~caches.available].to_string(index=False))
        return

    first_model, first_source, first_k = (
        model_seeds[0], source_seeds[0], k_values[0]
    )
    cfg = exp14.build_config(args, first_model)
    units = protocol["nested_adaptation_units_by_seed"][str(first_model)][str(first_k)]
    source_tasks, support, validation, official_test, feature_count, split_info = (
        prepare_kshot_experiment(
            cfg, args.preprocessing, args.balance_mode,
            protocol["validation_units"], units,
        )
    )
    del source_tasks, official_test
    schedule = materialize_support_schedule_with_units(support, 1)
    x, _, batch_units = schedule[0][0]
    state = exp14.load_source_state(
        exp14.source_cache_path(
            exp12_inputs["experiment"], args.target,
            "anil_engine_disjoint", first_model, first_source,
        )
    )
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(state)
    with torch.no_grad():
        output_shape = list(model(x[: min(8, len(x))]).shape)
    print(
        json.dumps(
            {
                "feature_count": feature_count,
                "support_windows": len(support.dataset),
                "validation_windows": len(validation.dataset),
                "first_batch_shape": list(x.shape),
                "first_batch_engine_count": int(torch.unique(batch_units).numel()),
                "forward_output_shape": output_shape,
                "official_test_engine_count_from_protocol": (
                    split_info["official_test_engine_count"]
                ),
                "official_test_forward_run": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    k_values, model_seeds, source_seeds, regimes = validate_args(args)
    exp12_inputs = exp14.resolve_inputs(args)
    protocol, _ = exp14.load_protocol_and_raw(exp12_inputs, args)
    exp14_inputs = experiment14_inputs(args)
    reference_results, reference_predictions, reference_audit = (
        load_reference_results(
            exp14_inputs, k_values, model_seeds, source_seeds
        )
    )
    for model_seed in model_seeds:
        if str(model_seed) not in protocol["nested_adaptation_units_by_seed"]:
            raise KeyError(f"协议缺少model_seed={model_seed}")
        for k in k_values:
            if str(k) not in protocol["nested_adaptation_units_by_seed"][str(model_seed)]:
                raise KeyError(f"协议缺少model_seed={model_seed}, K={k}")

    if args.dry_run:
        dry_run_report(
            args, exp12_inputs, exp14_inputs, protocol,
            reference_results, reference_audit,
            k_values, model_seeds, source_seeds, regimes,
        )
        return

    output = result_paths(args)
    output["output"].mkdir(parents=True, exist_ok=True)
    output["cells"].mkdir(parents=True, exist_ok=True)
    new_results = []
    histories = []
    new_prediction_frames = []

    for model_seed in model_seeds:
        cfg = exp14.build_config(args, model_seed)
        for k in k_values:
            adaptation_units = protocol["nested_adaptation_units_by_seed"][str(model_seed)][str(k)]
            source_tasks, support, validation, official_test, feature_count, split_info = (
                prepare_kshot_experiment(
                    cfg, args.preprocessing, args.balance_mode,
                    protocol["validation_units"], adaptation_units,
                )
            )
            del source_tasks, official_test
            if split_info["validation_units"] != protocol["validation_units"]:
                raise AssertionError("固定验证发动机发生变化")
            schedule = materialize_support_schedule_with_units(
                support, args.target_epochs
            )
            for source_seed in source_seeds:
                cache = exp14.source_cache_path(
                    exp12_inputs["experiment"], args.target,
                    "anil_engine_disjoint", model_seed, source_seed,
                )
                source_state = exp14.load_source_state(cache)
                for regime in regimes:
                    print(
                        f"\n[experiment15] regime={regime} K={k} "
                        f"model={model_seed} source={source_seed}"
                    )
                    result, history, predictions = run_cell(
                        args, output, cfg, regime, source_state, schedule,
                        validation, split_info, feature_count, k,
                        model_seed, source_seed,
                    )
                    new_results.append(result)
                    new_prediction_frames.append(predictions)
                    histories.append(
                        {
                            "regime": regime,
                            "k": k,
                            "model_seed": model_seed,
                            "source_split_seed": source_seed,
                            "epochs": history,
                        }
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del schedule, support, validation

    all_results = [*reference_results, *new_results]
    raw = pd.DataFrame(all_results)
    predictions = pd.concat(
        [reference_predictions, *new_prediction_frames], ignore_index=True
    )
    summary = exp14.summary_table(raw)
    paired, comparisons, engines, stages, tails = comparison_outputs(
        raw, predictions, args.bootstrap_repetitions
    )
    comparisons, per_model, per_source = add_factor_robustness(
        comparisons, paired
    )
    conclusion = make_conclusion(comparisons, tails)

    atomic_write_text(
        output["raw"], json.dumps(all_results, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        output["history"], json.dumps(histories, ensure_ascii=False, indent=2)
    )
    exp14.atomic_to_csv(predictions, output["predictions"])
    exp14.atomic_to_csv(summary, output["summary"])
    exp14.atomic_to_csv(paired, output["paired"])
    exp14.atomic_to_csv(comparisons, output["comparisons"])
    exp14.atomic_to_csv(engines, output["engine"])
    exp14.atomic_to_csv(stages, output["stage"])
    exp14.atomic_to_csv(tails, output["tail"])
    exp14.atomic_to_csv(per_model, output["model_seed"])
    exp14.atomic_to_csv(per_source, output["source_split"])
    exp14.atomic_to_csv(reference_audit, output["reference_audit"])

    expected_new = (
        len(k_values) * len(model_seeds) * len(source_seeds) * len(regimes)
    )
    grid = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "validation_only",
        "official_test_prediction_run": False,
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "reference_regimes": list(REFERENCE_REGIMES),
        "new_regimes": regimes,
        "reference_cells_imported": len(reference_results),
        "planned_new_target_trainings": expected_new,
        "completed_new_target_trainings": len(new_results),
        "full_grid_complete": len(new_results) == expected_new,
        "total_result_rows": len(all_results),
    }
    protocol_output = {
        **grid,
        "experiment12b_dir": str(exp12_inputs["experiment"]),
        "experiment14_dir": str(exp14_inputs["directory"]),
        "source_protocol": str(exp12_inputs["protocol"]),
        "validation_units": protocol["validation_units"],
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "target_scope": "predictor.* only",
        "low_risk_loss": {
            "nasa_loss_weight": args.low_nasa_loss_weight,
            "high_rul_loss_weight": args.low_high_rul_loss_weight,
            "high_rul_threshold": args.high_rul_threshold,
            "nasa_exp_clip": args.nasa_exp_clip,
        },
        "gradient_clip_norm": args.target_clip_norm,
        "engine_cvar": {
            "weight": args.engine_cvar_weight,
            "alpha": args.engine_cvar_alpha,
            "aggregation": "batch_per_engine_mean_clipped_nasa",
        },
        "tail_selection_rmse_tolerance": args.selection_rmse_tolerance,
        "success_rules_locked_before_official_test": True,
    }
    atomic_write_text(output["grid"], json.dumps(grid, ensure_ascii=False, indent=2))
    atomic_write_text(
        output["protocol"],
        json.dumps(protocol_output, ensure_ascii=False, indent=2),
    )
    atomic_write_text(
        output["conclusion"],
        json.dumps(conclusion, ensure_ascii=False, indent=2),
    )

    print("\n[实验15主要比较]")
    print(comparisons.to_string(index=False))
    print("\n[实验15结论]")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print("\n[输出文件]")
    for name, path in output.items():
        if path.is_file():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
