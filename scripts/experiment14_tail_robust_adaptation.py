#!/usr/bin/env python3
"""实验14：FD004少样本RUL的尾部风险鲁棒目标适应。

实验13表明，发动机互斥ANIL虽然降低了平均RMSE/MAE，但seed=46在
验证发动机48的high-RUL窗口上产生极端寿命低估，导致NASA Score恶化。
本实验不删除发动机48，也不读取官方测试预测，而是在实验12B锁定的验证协议上
进行2×2消融：

    目标损失：raw MSE / risk-aware loss（风险感知损失）
    epoch选择：最低RMSE / RMSE约束下最低NASA Score

默认方案
--------
``budget_mse_rmse``
    预算匹配普通迁移学习；MSE训练并按最低验证RMSE选epoch，仅作外部参照。

``anil_mse_rmse``
    实验13原始ANIL方案；MSE训练并按最低验证RMSE选epoch。

``anil_mse_tail``
    只改变epoch选择：先找到最低验证RMSE，再在其2%容差内选择NASA Score最低者。

``anil_risk_rmse``
    只改变训练损失：MSE + 裁剪NASA辅助项 + high-RUL低估惩罚；仍按RMSE选epoch。

``anil_risk_tail``
    同时使用风险感知损失和尾部约束epoch选择，是实验14的主要候选方案。

风险感知损失
------------
    L = MSE + lambda_nasa * mean(clipped_NASA)
            + lambda_high * mean_{RUL>90}(ReLU(y - y_hat)^2)

NASA指数输入默认裁剪为6，防止单个极端窗口造成梯度爆炸。所有方案只更新
``predictor.*``（RUL预测头），使用完全相同的目标发动机、批次顺序、训练轮数和
学习率。源模型只从实验12B缓存读取，不重新训练。

重要约束
--------
* evaluation_scope固定为validation_only；
* 代码会构建官方test loader以核验协议哈希，但不会对它执行模型前向；
* 不允许删除发动机48或根据官方测试集调参；
* 完整5×5交叉网格完成前，不形成正式显著性结论。

服务器dry-run示例：

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment14_tail_robust_adaptation.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --dry-run

正式运行示例：

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment14_tail_robust_adaptation.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --output-dir outputs/experiment14_tail_robust_adaptation \
      --resume
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from scripts.run_condition_aware_experiment import rul_stage_ids  # noqa: E402


SCRIPT_VERSION = "experiment14_tail_robust_adaptation_v1"
STAGE_NAMES = ("critical", "middle", "early", "high_rul")
REGIMES = (
    "budget_mse_rmse",
    "anil_mse_rmse",
    "anil_mse_tail",
    "anil_risk_rmse",
    "anil_risk_tail",
)
COMPARISONS = (
    (
        "anil_mse_tail",
        "anil_mse_rmse",
        "tail_selection_vs_current_anil",
    ),
    (
        "anil_risk_rmse",
        "anil_mse_rmse",
        "risk_loss_vs_current_anil",
    ),
    (
        "anil_risk_tail",
        "anil_mse_rmse",
        "combined_vs_current_anil",
    ),
    (
        "anil_risk_tail",
        "anil_mse_tail",
        "risk_loss_under_tail_selection",
    ),
    (
        "anil_risk_tail",
        "budget_mse_rmse",
        "combined_anil_vs_budget_transfer",
    ),
)
PRIMARY_COMPARISON = "combined_vs_current_anil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验14：ANIL的high-RUL/NASA尾部风险鲁棒目标适应"
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
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        "--experiment12b-dir", default="outputs/experiment12b_budget_matched_final"
    )
    parser.add_argument("--protocol-file")
    parser.add_argument("--raw-results-file")
    parser.add_argument("--device")
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument("--balance-mode", choices=BALANCE_MODES, default="engine_stage")
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)

    # 与实验12B源状态签名/配置保持一致。实验14不重新执行这些源训练。
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

    # 所有目标方案使用相同的更新预算。
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clip-norm", type=float, default=0.0)

    # 风险感知损失和尾部epoch选择的预注册参数。
    parser.add_argument("--nasa-loss-weight", type=float, default=1.0)
    parser.add_argument("--high-rul-loss-weight", type=float, default=0.25)
    parser.add_argument("--high-rul-threshold", type=float, default=90.0)
    parser.add_argument("--nasa-exp-clip", type=float, default=6.0)
    parser.add_argument(
        "--selection-rmse-tolerance", type=float, default=0.02,
        help="尾部选择只考虑RMSE不超过最低RMSE×(1+tolerance)的epoch",
    )

    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/experiment14_tail_robust_adaptation")
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
        raise ValueError("模型种子、源域划分种子和方案不能为空")
    positive = {
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "nasa_exp_clip": args.nasa_exp_clip,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正数：{invalid}")
    nonnegative = {
        "target_weight_decay": args.target_weight_decay,
        "target_clip_norm": args.target_clip_norm,
        "nasa_loss_weight": args.nasa_loss_weight,
        "high_rul_loss_weight": args.high_rul_loss_weight,
        "selection_rmse_tolerance": args.selection_rmse_tolerance,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise ValueError(f"以下参数不能为负数：{invalid}")
    if len(model_seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个模型种子，只能视为预实验。")
    if len(source_seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个源域划分种子，只能视为预实验。")
    return k_values, model_seeds, source_seeds, regimes


def regime_spec(regime: str) -> dict:
    specs = {
        "budget_mse_rmse": {
            "source": "ordinary_budget",
            "loss": "mse",
            "selection": "rmse",
            "source_training": "budget_matched_ordinary_pretraining",
        },
        "anil_mse_rmse": {
            "source": "anil_engine_disjoint",
            "loss": "mse",
            "selection": "rmse",
            "source_training": "engine_disjoint_anil",
        },
        "anil_mse_tail": {
            "source": "anil_engine_disjoint",
            "loss": "mse",
            "selection": "tail_constrained",
            "source_training": "engine_disjoint_anil",
        },
        "anil_risk_rmse": {
            "source": "anil_engine_disjoint",
            "loss": "risk",
            "selection": "rmse",
            "source_training": "engine_disjoint_anil",
        },
        "anil_risk_tail": {
            "source": "anil_engine_disjoint",
            "loss": "risk",
            "selection": "tail_constrained",
            "source_training": "engine_disjoint_anil",
        },
    }
    return specs[regime]


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    candidates = list(candidates)
    for path in candidates:
        if path.is_file():
            return path
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到{description}，已检查：\n  {checked}")


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    experiment = resolve_path(args.experiment12b_dir, PROJECT_ROOT)
    protocol = (
        resolve_path(args.protocol_file, PROJECT_ROOT)
        if args.protocol_file else first_existing(
            [
                experiment / f"experiment12_validation_{args.target}_split_protocol.json",
                experiment / f"experiment12b_validation_{args.target}_split_protocol.json",
            ],
            "实验12B划分协议",
        )
    )
    raw = (
        resolve_path(args.raw_results_file, PROJECT_ROOT)
        if args.raw_results_file else first_existing(
            [
                experiment / f"experiment12_validation_{args.target}_raw.json",
                experiment / f"experiment12b_validation_{args.target}_raw.json",
            ],
            "实验12B raw结果",
        )
    )
    return {"experiment": experiment, "protocol": protocol, "raw": raw}


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment14_{args.target}"
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
        "audit": output / f"{prefix}_reference_audit.csv",
        "protocol": output / f"{prefix}_protocol.json",
        "grid": output / f"{prefix}_grid_plan.json",
        "conclusion": output / f"{prefix}_conclusion.json",
    }


def source_cache_path(
    experiment: Path, target: str, source_key: str,
    model_seed: int, source_seed: int,
) -> Path:
    if source_key == "ordinary_budget":
        return (
            experiment / "shared_source_states" / "source_cache" /
            f"ordinary_budget_{target}_seed{model_seed}.pt"
        )
    if source_key == "anil_engine_disjoint":
        return (
            experiment / "source_states" / f"source_{source_seed}" /
            "source_cache" / f"anil_engine_disjoint_{target}_seed{model_seed}.pt"
        )
    raise ValueError(f"未知源状态：{source_key}")


def trusted_torch_load(path: Path) -> dict:
    """仅用于读取用户自己生成且可信的实验缓存。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_source_state(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少实验12B源模型缓存：{path}")
    payload = trusted_torch_load(path)
    state = payload.get("state")
    if not isinstance(state, dict) or not all_tensors_finite(state.values()):
        raise RuntimeError(f"源模型缓存无效或包含NaN/Inf：{path}")
    return state


def load_protocol_and_raw(inputs: dict[str, Path], args: argparse.Namespace):
    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    raw = pd.DataFrame(json.loads(inputs["raw"].read_text(encoding="utf-8")))
    if protocol.get("target_domain") != args.target:
        raise ValueError("实验12B协议target_domain与--target不一致")
    if raw.empty or set(raw.get("evaluation_scope", [])) != {"validation"}:
        raise ValueError("实验14只允许使用实验12B的validation结果")
    return protocol, raw


def build_config(args: argparse.Namespace, model_seed: int) -> dict:
    proxy = argparse.Namespace(**vars(args))
    proxy.output_dir = args.output_dir
    proxy.source_task_seed = 0
    proxy.vary_source_split_by_seed = False
    proxy.seeds = list(args.model_seeds)
    cfg = exp11.load_config(proxy, model_seed)
    cfg["pair_aux_weight"] = 0.0
    if args.device:
        cfg["device"] = args.device
    return cfg


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def materialize_support_schedule(support, epochs: int):
    """Freeze target batches so every regime sees exactly the same windows/order."""
    schedule: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for _ in range(epochs):
        epoch_batches = []
        for x, y in support:
            epoch_batches.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        schedule.append(epoch_batches)
    return schedule


def predict_validation(
    model: torch.nn.Module, validation, device: torch.device
) -> pd.DataFrame:
    model = model.to(device).eval()
    labels: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x, y in validation:
            prediction = model(x.to(device))
            labels.extend(y.detach().cpu().numpy().astype(float).tolist())
            predictions.extend(
                prediction.detach().cpu().numpy().astype(float).tolist()
            )
    y = np.asarray(labels, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    units = np.asarray(validation.dataset.units, dtype=int)
    if not (len(y) == len(pred) == len(units)):
        raise AssertionError("验证标签、预测和发动机编号长度不一致")
    error = pred - y
    nasa_exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    if np.any(nasa_exponent > 80) or not np.all(np.isfinite(error)):
        raise FloatingPointError("验证预测出现非有限或极端NASA指数")
    frame = pd.DataFrame(
        {
            "unit": units,
            "true_rul": y,
            "predicted_rul": pred,
            "error_pred_minus_true": error,
            "absolute_error": np.abs(error),
            "squared_error": error**2,
            "nasa_contribution": np.expm1(nasa_exponent),
            "is_late_prediction": error > 0,
            "stage": [STAGE_NAMES[index] for index in rul_stage_ids(y)],
        }
    )
    frame["window_index_within_engine"] = frame.groupby("unit").cumcount()
    return frame


def prediction_metrics(
    frame: pd.DataFrame, high_rul_threshold: float = 90.0
) -> dict:
    metrics = regression_metrics(frame["true_rul"], frame["predicted_rul"])
    engine_nasa = (
        frame.groupby("unit", as_index=False)["nasa_contribution"].sum()
        .sort_values("nasa_contribution", ascending=False)
    )
    total_nasa = float(frame["nasa_contribution"].sum())
    high = frame["true_rul"] > high_rul_threshold
    late = frame["error_pred_minus_true"] > 0
    metrics.update(
        {
            "bias": float(frame["error_pred_minus_true"].mean()),
            "late_prediction_rate": float(late.mean()),
            "late_nasa_fraction": (
                float(frame.loc[late, "nasa_contribution"].sum() / total_nasa)
                if total_nasa > 0 else 0.0
            ),
            "worst_engine_nasa": float(engine_nasa.iloc[0].nasa_contribution),
            "worst_engine_unit": int(engine_nasa.iloc[0].unit),
            "p95_engine_nasa": float(np.quantile(engine_nasa.nasa_contribution, 0.95)),
            "top1_engine_nasa_share": (
                float(engine_nasa.iloc[0].nasa_contribution / total_nasa)
                if total_nasa > 0 else 0.0
            ),
            "top3_engine_nasa_share": (
                float(engine_nasa.head(3).nasa_contribution.sum() / total_nasa)
                if total_nasa > 0 else 0.0
            ),
            "high_rul_nasa_score": float(
                frame.loc[high, "nasa_contribution"].sum()
            ),
            "high_rul_mae": float(
                frame.loc[high, "absolute_error"].mean()
            ) if bool(high.any()) else float("nan"),
            "high_rul_underprediction_rate": float(
                (frame.loc[high, "error_pred_minus_true"] < 0).mean()
            ) if bool(high.any()) else float("nan"),
        }
    )
    return metrics


def risk_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    error = prediction - target
    mse = torch.mean(error**2)
    exponent = torch.where(error < 0, -error / 13.0, error / 10.0)
    clipped_nasa = torch.expm1(torch.clamp(exponent, min=0.0, max=args.nasa_exp_clip))
    nasa = clipped_nasa.mean()
    high_mask = target > args.high_rul_threshold
    if bool(high_mask.any().item()):
        high_under = torch.mean(torch.relu(target[high_mask] - prediction[high_mask]) ** 2)
    else:
        high_under = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    total = (
        mse
        + args.nasa_loss_weight * nasa
        + args.high_rul_loss_weight * high_under
    )
    return total, {"mse": mse, "nasa": nasa, "high_under": high_under}


def training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mode: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_mode == "mse":
        mse = torch.nn.functional.mse_loss(prediction, target)
        zero = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        return mse, {"mse": mse, "nasa": zero, "high_under": zero}
    if loss_mode == "risk":
        return risk_aware_loss(prediction, target, args)
    raise ValueError(f"未知目标损失：{loss_mode}")


def select_epoch(history: list[dict], mode: str, tolerance: float) -> tuple[int, dict]:
    if not history:
        raise ValueError("目标训练历史为空")
    min_rmse = min(row["validation_rmse"] for row in history)
    if mode == "rmse":
        selected = min(history, key=lambda row: (row["validation_rmse"], row["epoch"]))
        eligible = 1
    elif mode == "tail_constrained":
        threshold = min_rmse * (1.0 + tolerance)
        candidates = [row for row in history if row["validation_rmse"] <= threshold]
        selected = min(
            candidates,
            key=lambda row: (
                row["validation_nasa_score"],
                row["validation_worst_engine_nasa"],
                row["validation_rmse"],
                row["epoch"],
            ),
        )
        eligible = len(candidates)
    else:
        raise ValueError(f"未知epoch选择方式：{mode}")
    diagnostic = {
        "minimum_validation_rmse_across_epochs": float(min_rmse),
        "selection_rmse_threshold": float(
            min_rmse * (1.0 + tolerance) if mode == "tail_constrained" else min_rmse
        ),
        "eligible_epoch_count": int(eligible),
        "selected_epoch_validation_rmse": float(selected["validation_rmse"]),
        "selected_epoch_validation_nasa_score": float(
            selected["validation_nasa_score"]
        ),
    }
    return int(selected["epoch"]), diagnostic


def train_target_robust(
    model: torch.nn.Module,
    support_schedule,
    validation,
    args: argparse.Namespace,
    device: torch.device,
    *,
    loss_mode: str,
    selection_mode: str,
) -> tuple[torch.nn.Module, list[dict], int, dict, dict]:
    learner = deepcopy(model).to(device)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in learner.named_parameters()
    }
    trainable = set_target_scope(learner, "rul_head")
    optimizer = torch.optim.Adam(
        trainable,
        lr=args.target_lr,
        weight_decay=args.target_weight_decay,
    )
    history: list[dict] = []
    epoch_states: dict[int, dict] = {}
    max_grad_norm = 0.0
    gradient_steps = 0
    clip_events = 0

    for epoch, batches in enumerate(support_schedule, start=1):
        set_target_mode(learner, "rul_head")
        total_losses: list[float] = []
        mse_losses: list[float] = []
        nasa_losses: list[float] = []
        high_losses: list[float] = []
        for x_cpu, y_cpu in batches:
            x, y = x_cpu.to(device), y_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = learner(x)
            loss, components = training_loss(prediction, y, loss_mode, args)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"目标损失非有限：epoch={epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            grad_norm = tensor_norm(gradients)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"目标梯度非有限：epoch={epoch}")
            max_grad_norm = max(max_grad_norm, grad_norm)
            gradient_steps += 1
            if args.target_clip_norm > 0:
                if grad_norm > args.target_clip_norm:
                    clip_events += 1
                torch.nn.utils.clip_grad_norm_(trainable, args.target_clip_norm)
            optimizer.step()
            if not all_tensors_finite(learner.parameters()):
                raise FloatingPointError(f"目标参数非有限：epoch={epoch}")
            total_losses.append(float(loss.detach().cpu().item()))
            mse_losses.append(float(components["mse"].detach().cpu().item()))
            nasa_losses.append(float(components["nasa"].detach().cpu().item()))
            high_losses.append(float(components["high_under"].detach().cpu().item()))

        validation_frame = predict_validation(learner, validation, device)
        metrics = prediction_metrics(validation_frame, args.high_rul_threshold)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_losses)),
            "train_mse_component": float(np.mean(mse_losses)),
            "train_nasa_component": float(np.mean(nasa_losses)),
            "train_high_rul_component": float(np.mean(high_losses)),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        epoch_states[epoch] = cpu_state(learner)
        print(
            f"target_epoch={epoch:03d}/{args.target_epochs} "
            f"loss={loss_mode} select={selection_mode} "
            f"train={row['train_total_loss']:.4f} "
            f"val_rmse={metrics['rmse']:.4f} "
            f"val_nasa={metrics['nasa_score']:.2f} "
            f"worst_engine={metrics['worst_engine_unit']}"
        )

    best_epoch, selection_diagnostic = select_epoch(
        history, selection_mode, args.selection_rmse_tolerance
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
        "parameter_drift_by_group": parameter_drift(before, learner),
    }
    return learner, history, best_epoch, diagnostics, epoch_states[best_epoch]


def cell_paths(
    output: dict[str, Path], regime: str, k: int,
    model_seed: int, source_seed: int,
) -> tuple[Path, Path]:
    stem = f"{regime}_k{k}_source{source_seed}_model{model_seed}"
    return output["cells"] / f"{stem}.json", output["cells"] / f"{stem}.csv"


def cached_cell(
    output: dict[str, Path], regime: str, k: int,
    model_seed: int, source_seed: int,
) -> tuple[dict, list[dict], pd.DataFrame] | None:
    metadata_path, prediction_path = cell_paths(
        output, regime, k, model_seed, source_seed
    )
    if not metadata_path.is_file() or not prediction_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("script_version") != SCRIPT_VERSION:
        return None
    return payload["result"], payload.get("history", []), pd.read_csv(prediction_path)


def save_cell(
    output: dict[str, Path], regime: str, k: int,
    model_seed: int, source_seed: int,
    result: dict, history: list[dict], predictions: pd.DataFrame,
) -> None:
    metadata_path, prediction_path = cell_paths(
        output, regime, k, model_seed, source_seed
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(predictions, prediction_path)
    atomic_write_text(
        metadata_path,
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
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
) -> tuple[dict, list[dict], pd.DataFrame]:
    if args.resume:
        cached = cached_cell(output, regime, k, model_seed, source_seed)
        if cached is not None:
            print(
                f"[resume] regime={regime} K={k} model={model_seed} "
                f"source={source_seed}"
            )
            return cached

    spec = regime_spec(regime)
    seed_everything(model_seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    device = resolve_device(cfg["device"])
    model, history, best_epoch, diagnostics, selected_state = train_target_robust(
        model,
        support_schedule,
        validation,
        args,
        device,
        loss_mode=spec["loss"],
        selection_mode=spec["selection"],
    )
    frame = predict_validation(model, validation, device)
    metrics = prediction_metrics(frame, args.high_rul_threshold)
    frame.insert(0, "regime", regime)
    frame.insert(1, "k", k)
    frame.insert(2, "model_seed", model_seed)
    frame.insert(3, "source_split_seed", source_seed)

    result = {
        **metrics,
        "evaluation_scope": "validation",
        "regime": regime,
        "source_training": spec["source_training"],
        "source_state_key": spec["source"],
        "target_loss_mode": spec["loss"],
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
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": int(args.target_epochs),
        "target_lr": float(args.target_lr),
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "nasa_loss_weight": float(args.nasa_loss_weight),
        "high_rul_loss_weight": float(args.high_rul_loss_weight),
        "high_rul_threshold": float(args.high_rul_threshold),
        "nasa_exp_clip": float(args.nasa_exp_clip),
        "selection_rmse_tolerance": float(args.selection_rmse_tolerance),
        **diagnostics,
    }
    save_cell(
        output, regime, k, model_seed, source_seed, result, history, frame
    )
    if args.save_checkpoints:
        output["checkpoints"].mkdir(parents=True, exist_ok=True)
        checkpoint = output["checkpoints"] / (
            f"experiment14_{regime}_k{k}_{args.target}_"
            f"source{source_seed}_model{model_seed}.pt"
        )
        torch.save(
            {
                "model": selected_state,
                "result": result,
                "history": history,
                "split": split_info,
                "script_version": SCRIPT_VERSION,
            },
            checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, history, frame


def expected_experiment12_row(
    raw: pd.DataFrame, regime: str, k: int,
    model_seed: int, source_seed: int,
) -> dict | None:
    reference = {
        "budget_mse_rmse": "pretrained_budget_head",
        "anil_mse_rmse": "anil_engine_disjoint_head",
    }.get(regime)
    if reference is None:
        return None
    mask = (
        raw.regime.eq(reference)
        & raw.k.astype(int).eq(k)
        & raw.model_seed.astype(int).eq(model_seed)
    )
    if reference == "anil_engine_disjoint_head":
        mask &= raw.source_split_seed.astype(int).eq(source_seed)
    rows = raw.loc[mask]
    return rows.iloc[0].to_dict() if not rows.empty else None


def reference_audit(results: list[dict], raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for result in results:
        expected = expected_experiment12_row(
            raw,
            result["regime"],
            int(result["k"]),
            int(result["model_seed"]),
            int(result["source_split_seed"]),
        )
        if expected is None:
            continue
        row = {
            "regime": result["regime"],
            "k": result["k"],
            "model_seed": result["model_seed"],
            "source_split_seed": result["source_split_seed"],
        }
        for metric in ("rmse", "mae", "nasa_score"):
            row[f"experiment14_{metric}"] = result[metric]
            row[f"experiment12b_{metric}"] = expected[metric]
            row[f"{metric}_absolute_difference"] = abs(
                result[metric] - expected[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "rmse", "mae", "r2", "nasa_score", "worst_engine_nasa",
        "p95_engine_nasa", "top1_engine_nasa_share", "high_rul_nasa_score",
        "high_rul_mae", "late_prediction_rate",
    )
    rows = []
    for (k, regime), group in raw.groupby(["k", "regime"]):
        row = {"k": int(k), "regime": regime, "n_cells": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "rmse_mean"])


def build_paired_cells(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "rmse", "mae", "r2", "nasa_score", "worst_engine_nasa",
        "p95_engine_nasa", "top1_engine_nasa_share", "high_rul_nasa_score",
        "high_rul_mae", "late_prediction_rate",
    )
    rows = []
    for candidate, reference, label in COMPARISONS:
        candidate_rows = raw[raw.regime == candidate].copy()
        reference_rows = raw[raw.regime == reference].copy()
        if candidate_rows.empty or reference_rows.empty:
            continue
        keys = ["k", "model_seed"]
        if reference != "budget_mse_rmse":
            keys.append("source_split_seed")
        else:
            reference_rows = reference_rows.drop_duplicates(keys)
        columns = keys + list(metrics)
        paired = candidate_rows[columns + (["source_split_seed"] if "source_split_seed" not in keys else [])]
        paired = paired.merge(
            reference_rows[columns],
            on=keys,
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="many_to_one" if reference == "budget_mse_rmse" else "one_to_one",
        )
        if "source_split_seed_candidate" in paired.columns:
            paired = paired.rename(
                columns={"source_split_seed_candidate": "source_split_seed"}
            )
        for metric in metrics:
            paired[f"{metric}_delta"] = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
            )
        paired.insert(0, "reference", reference)
        paired.insert(0, "candidate", candidate)
        paired.insert(0, "comparison", label)
        paired["rmse_and_nasa_improved"] = (
            (paired.rmse_delta < 0) & (paired.nasa_score_delta < 0)
        )
        rows.append(paired)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def crossed_bootstrap_ci(
    paired: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    matrix = paired.pivot(
        index="source_split_seed", columns="model_seed", values=value_column
    ).sort_index().sort_index(axis=1)
    if matrix.empty or bool(matrix.isna().any().any()):
        return float("nan"), float("nan")
    values = matrix.to_numpy(dtype=float)
    n_source, n_model = values.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        source_index = rng.integers(0, n_source, size=n_source)
        model_index = rng.integers(0, n_model, size=n_model)
        samples[index] = values[np.ix_(source_index, model_index)].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def comparison_summary(
    paired: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for index, ((k, comparison), group) in enumerate(
        paired.groupby(["k", "comparison"])
    ):
        rmse_low, rmse_high = crossed_bootstrap_ci(
            group, "rmse_delta", repetitions, 14000 + index
        )
        nasa_low, nasa_high = crossed_bootstrap_ci(
            group, "nasa_score_delta", repetitions, 24000 + index
        )
        reference_rmse = float(group.rmse_reference.mean())
        row = {
            "k": int(k),
            "comparison": comparison,
            "candidate": group.candidate.iloc[0],
            "reference": group.reference.iloc[0],
            "n_pairs": len(group),
            "n_model_seeds": group.model_seed.nunique(),
            "n_source_splits": group.source_split_seed.nunique(),
            "rmse_delta_mean": float(group.rmse_delta.mean()),
            "rmse_delta_median": float(group.rmse_delta.median()),
            "rmse_change_pct": float(
                100.0 * group.rmse_delta.mean() / max(reference_rmse, 1e-12)
            ),
            "rmse_win_rate": float((group.rmse_delta < 0).mean()),
            "rmse_boot_ci95_low": rmse_low,
            "rmse_boot_ci95_high": rmse_high,
            "mae_delta_mean": float(group.mae_delta.mean()),
            "nasa_score_delta_mean": float(group.nasa_score_delta.mean()),
            "nasa_score_delta_median": float(group.nasa_score_delta.median()),
            "nasa_win_rate": float((group.nasa_score_delta < 0).mean()),
            "nasa_boot_ci95_low": nasa_low,
            "nasa_boot_ci95_high": nasa_high,
            "joint_win_rate": float(group.rmse_and_nasa_improved.mean()),
            "worst_engine_nasa_delta_mean": float(
                group.worst_engine_nasa_delta.mean()
            ),
            "high_rul_nasa_delta_mean": float(
                group.high_rul_nasa_score_delta.mean()
            ),
            "top1_share_delta_mean": float(
                group.top1_engine_nasa_share_delta.mean()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "comparison"])


def pair_prediction_frames(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
) -> pd.DataFrame:
    candidate_rows = predictions[predictions.regime == candidate].copy()
    reference_rows = predictions[predictions.regime == reference].copy()
    if candidate_rows.empty or reference_rows.empty:
        return pd.DataFrame()
    merge_keys = ["k", "model_seed", "unit", "window_index_within_engine", "true_rul"]
    if reference != "budget_mse_rmse":
        merge_keys.insert(2, "source_split_seed")
    else:
        reference_rows = reference_rows.drop_duplicates(merge_keys)
    metric_columns = [
        "predicted_rul", "error_pred_minus_true", "absolute_error",
        "squared_error", "nasa_contribution",
    ]
    reference_keep = merge_keys + metric_columns
    paired = candidate_rows.merge(
        reference_rows[reference_keep],
        on=merge_keys,
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="many_to_one" if reference == "budget_mse_rmse" else "one_to_one",
    )
    for metric in ("absolute_error", "squared_error", "nasa_contribution"):
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
        )
    return paired


def detailed_deltas(predictions: pd.DataFrame):
    engine_rows = []
    stage_rows = []
    for candidate, reference, label in COMPARISONS:
        paired = pair_prediction_frames(predictions, candidate, reference)
        if paired.empty:
            continue
        engine = paired.groupby(
            ["k", "model_seed", "source_split_seed", "unit"], as_index=False
        ).agg(
            windows=("true_rul", "size"),
            nasa_candidate=("nasa_contribution_candidate", "sum"),
            nasa_reference=("nasa_contribution_reference", "sum"),
            nasa_delta=("nasa_contribution_delta", "sum"),
            absolute_error_delta=("absolute_error_delta", "mean"),
            squared_error_candidate=("squared_error_candidate", "mean"),
            squared_error_reference=("squared_error_reference", "mean"),
        )
        engine["rmse_candidate"] = np.sqrt(engine.squared_error_candidate)
        engine["rmse_reference"] = np.sqrt(engine.squared_error_reference)
        engine["rmse_delta"] = engine.rmse_candidate - engine.rmse_reference
        engine.insert(0, "reference", reference)
        engine.insert(0, "candidate", candidate)
        engine.insert(0, "comparison", label)
        engine_rows.append(engine)

        stage = paired.groupby(
            ["k", "model_seed", "source_split_seed", "stage"], as_index=False
        ).agg(
            windows=("true_rul", "size"),
            nasa_candidate=("nasa_contribution_candidate", "sum"),
            nasa_reference=("nasa_contribution_reference", "sum"),
            nasa_delta=("nasa_contribution_delta", "sum"),
            absolute_error_delta=("absolute_error_delta", "mean"),
        )
        stage.insert(0, "reference", reference)
        stage.insert(0, "candidate", candidate)
        stage.insert(0, "comparison", label)
        stage_rows.append(stage)
    engines = pd.concat(engine_rows, ignore_index=True) if engine_rows else pd.DataFrame()
    stages = pd.concat(stage_rows, ignore_index=True) if stage_rows else pd.DataFrame()
    return engines, stages


def tail_diagnostics(
    paired: pd.DataFrame, engines: pd.DataFrame, stages: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    if paired.empty:
        return pd.DataFrame()
    for (k, comparison), group in paired.groupby(["k", "comparison"]):
        engine_group = engines[(engines.k == k) & (engines.comparison == comparison)]
        stage_group = stages[(stages.k == k) & (stages.comparison == comparison)]
        seed46 = group[group.model_seed == 46]
        unit48 = engine_group[engine_group.unit == 48]
        high = stage_group[stage_group.stage == "high_rul"]
        rows.append(
            {
                "k": int(k),
                "comparison": comparison,
                "seed46_nasa_delta_mean": (
                    float(seed46.nasa_score_delta.mean()) if len(seed46) else np.nan
                ),
                "seed46_rmse_delta_mean": (
                    float(seed46.rmse_delta.mean()) if len(seed46) else np.nan
                ),
                "unit48_nasa_delta_mean": (
                    float(unit48.nasa_delta.mean()) if len(unit48) else np.nan
                ),
                "unit48_nasa_win_rate": (
                    float((unit48.nasa_delta < 0).mean()) if len(unit48) else np.nan
                ),
                "unit48_rmse_delta_mean": (
                    float(unit48.rmse_delta.mean()) if len(unit48) else np.nan
                ),
                "high_rul_nasa_delta_mean_per_cell": (
                    float(high.nasa_delta.mean()) if len(high) else np.nan
                ),
                "high_rul_nasa_win_rate": (
                    float((high.nasa_delta < 0).mean()) if len(high) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "comparison"])


def make_conclusion(
    comparisons: pd.DataFrame,
    tails: pd.DataFrame,
) -> dict:
    conclusion = {
        "script_version": SCRIPT_VERSION,
        "evaluation_scope": "validation_only",
        "official_test_prediction_run": False,
        "primary_comparison": PRIMARY_COMPARISON,
        "success_rule": {
            "rmse_relative_degradation_at_most_pct": 1.0,
            "nasa_score_delta_mean_below": 0.0,
            "nasa_win_rate_at_least": 0.8,
            "nasa_bootstrap_ci95_upper_below": 0.0,
            "seed46_nasa_delta_mean_below": 0.0,
            "unit48_nasa_delta_mean_below": 0.0,
            "high_rul_nasa_delta_mean_below": 0.0,
        },
        "by_k": {},
    }
    primary = comparisons[comparisons.comparison == PRIMARY_COMPARISON]
    primary_tail = tails[tails.comparison == PRIMARY_COMPARISON]
    for _, row in primary.iterrows():
        k = int(row.k)
        tail_rows = primary_tail[primary_tail.k == k]
        tail = tail_rows.iloc[0] if not tail_rows.empty else None
        checks = {
            "rmse_preserved": row.rmse_change_pct <= 1.0,
            "mean_nasa_improved": row.nasa_score_delta_mean < 0,
            "nasa_win_rate": row.nasa_win_rate >= 0.8,
            "nasa_ci_below_zero": row.nasa_boot_ci95_high < 0,
            "seed46_improved": tail is not None and tail.seed46_nasa_delta_mean < 0,
            "unit48_improved": tail is not None and tail.unit48_nasa_delta_mean < 0,
            "high_rul_improved": (
                tail is not None and tail.high_rul_nasa_delta_mean_per_cell < 0
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
    return conclusion


def dry_run_report(
    args: argparse.Namespace,
    inputs: dict[str, Path],
    protocol: dict,
    k_values: list[int],
    model_seeds: list[int],
    source_seeds: list[int],
    regimes: list[str],
) -> None:
    budget_needed = any(regime_spec(name)["source"] == "ordinary_budget" for name in regimes)
    anil_needed = any(regime_spec(name)["source"] == "anil_engine_disjoint" for name in regimes)
    rows = []
    for model_seed in model_seeds:
        if budget_needed:
            path = source_cache_path(
                inputs["experiment"], args.target, "ordinary_budget", model_seed, 0
            )
            rows.append(
                {"source": "ordinary_budget", "model_seed": model_seed,
                 "source_split_seed": 0, "available": path.is_file(), "path": str(path)}
            )
        if anil_needed:
            for source_seed in source_seeds:
                path = source_cache_path(
                    inputs["experiment"], args.target,
                    "anil_engine_disjoint", model_seed, source_seed,
                )
                rows.append(
                    {"source": "anil_engine_disjoint", "model_seed": model_seed,
                     "source_split_seed": source_seed, "available": path.is_file(),
                     "path": str(path)}
                )
    availability = pd.DataFrame(rows)
    budget_cells = (
        len(model_seeds) * len(k_values)
        if "budget_mse_rmse" in regimes else 0
    )
    anil_regime_count = sum(
        regime_spec(name)["source"] == "anil_engine_disjoint" for name in regimes
    )
    anil_cells = (
        len(model_seeds) * len(source_seeds) * len(k_values) * anil_regime_count
    )
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "validation_only",
        "official_test_prediction_will_run": False,
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "regimes": regimes,
        "planned_target_trainings": budget_cells + anil_cells,
        "validation_engine_count": len(protocol["validation_units"]),
        "source_cache_count": len(availability),
        "source_cache_available": int(availability.available.sum()),
        "risk_loss": {
            "nasa_loss_weight": args.nasa_loss_weight,
            "high_rul_loss_weight": args.high_rul_loss_weight,
            "high_rul_threshold": args.high_rul_threshold,
            "nasa_exp_clip": args.nasa_exp_clip,
        },
        "tail_selection_rmse_tolerance": args.selection_rmse_tolerance,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n[source cache availability]")
    print(availability.groupby("source").available.agg(["count", "sum"]).to_string())
    missing = availability[~availability.available]
    if not missing.empty:
        print("\n[缺失缓存]")
        print(missing.to_string(index=False))
        return

    # One forward-only shape check. The official test loader is discarded without forward.
    first_seed, first_k = model_seeds[0], k_values[0]
    cfg = build_config(args, first_seed)
    units = protocol["nested_adaptation_units_by_seed"][str(first_seed)][str(first_k)]
    source_tasks, support, validation, official_test, feature_count, split_info = (
        prepare_kshot_experiment(
            cfg, args.preprocessing, args.balance_mode,
            protocol["validation_units"], units,
        )
    )
    del source_tasks, official_test
    source_key = regime_spec(regimes[0])["source"]
    source_seed = 0 if source_key == "ordinary_budget" else source_seeds[0]
    state = load_source_state(
        source_cache_path(
            inputs["experiment"], args.target, source_key, first_seed, source_seed
        )
    )
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(state)
    x, _ = next(iter(support))
    with torch.no_grad():
        shape = list(model(x[: min(8, len(x))]).shape)
    print(
        json.dumps(
            {
                "feature_count": feature_count,
                "support_windows": len(support.dataset),
                "validation_windows": len(validation.dataset),
                "example_shape": list(x.shape),
                "forward_output_shape": shape,
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
    inputs = resolve_inputs(args)
    protocol, experiment12_raw = load_protocol_and_raw(inputs, args)
    for model_seed in model_seeds:
        if str(model_seed) not in protocol["nested_adaptation_units_by_seed"]:
            raise KeyError(f"协议缺少model_seed={model_seed}")
        for k in k_values:
            if str(k) not in protocol["nested_adaptation_units_by_seed"][str(model_seed)]:
                raise KeyError(f"协议缺少model_seed={model_seed}, K={k}")

    if args.dry_run:
        dry_run_report(
            args, inputs, protocol, k_values, model_seeds, source_seeds, regimes
        )
        return

    output = result_paths(args)
    output["output"].mkdir(parents=True, exist_ok=True)
    output["cells"].mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    histories: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    budget_requested = "budget_mse_rmse" in regimes
    anil_regimes = [
        name for name in regimes
        if regime_spec(name)["source"] == "anil_engine_disjoint"
    ]

    for model_seed in model_seeds:
        cfg = build_config(args, model_seed)
        for k in k_values:
            adaptation_units = protocol["nested_adaptation_units_by_seed"][str(model_seed)][str(k)]
            source_tasks, support, validation, official_test, feature_count, split_info = (
                prepare_kshot_experiment(
                    cfg,
                    args.preprocessing,
                    args.balance_mode,
                    protocol["validation_units"],
                    adaptation_units,
                )
            )
            del source_tasks, official_test  # Never run official test forward.
            if split_info["validation_units"] != protocol["validation_units"]:
                raise AssertionError("固定验证发动机发生变化")
            schedule = materialize_support_schedule(support, args.target_epochs)

            if budget_requested:
                regime = "budget_mse_rmse"
                source_seed = 0
                cache = source_cache_path(
                    inputs["experiment"], args.target,
                    "ordinary_budget", model_seed, source_seed,
                )
                source_state = load_source_state(cache)
                print(
                    f"\n[experiment14] regime={regime} K={k} "
                    f"model={model_seed} source=shared"
                )
                result, history, predictions = run_cell(
                    args, output, cfg, regime, source_state, schedule,
                    validation, split_info, feature_count, k, model_seed, source_seed,
                )
                results.append(result)
                prediction_frames.append(predictions)
                histories.append(
                    {"regime": regime, "k": k, "model_seed": model_seed,
                     "source_split_seed": source_seed, "epochs": history}
                )

            for source_seed in source_seeds:
                if not anil_regimes:
                    break
                cache = source_cache_path(
                    inputs["experiment"], args.target,
                    "anil_engine_disjoint", model_seed, source_seed,
                )
                source_state = load_source_state(cache)
                for regime in anil_regimes:
                    print(
                        f"\n[experiment14] regime={regime} K={k} "
                        f"model={model_seed} source={source_seed}"
                    )
                    result, history, predictions = run_cell(
                        args, output, cfg, regime, source_state, schedule,
                        validation, split_info, feature_count, k,
                        model_seed, source_seed,
                    )
                    results.append(result)
                    prediction_frames.append(predictions)
                    histories.append(
                        {"regime": regime, "k": k, "model_seed": model_seed,
                         "source_split_seed": source_seed, "epochs": history}
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del schedule, support, validation

    raw = pd.DataFrame(results)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summary_table(raw)
    paired = build_paired_cells(raw)
    comparisons = comparison_summary(paired, args.bootstrap_repetitions)
    engine_deltas, stage_deltas = detailed_deltas(predictions)
    tails = tail_diagnostics(paired, engine_deltas, stage_deltas)
    audit = reference_audit(results, experiment12_raw)
    conclusion = make_conclusion(comparisons, tails)

    atomic_write_text(output["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    atomic_write_text(output["history"], json.dumps(histories, ensure_ascii=False, indent=2))
    atomic_to_csv(predictions, output["predictions"])
    atomic_to_csv(summary, output["summary"])
    atomic_to_csv(paired, output["paired"])
    atomic_to_csv(comparisons, output["comparisons"])
    atomic_to_csv(engine_deltas, output["engine"])
    atomic_to_csv(stage_deltas, output["stage"])
    atomic_to_csv(tails, output["tail"])
    atomic_to_csv(audit, output["audit"])
    atomic_write_text(output["conclusion"], json.dumps(conclusion, ensure_ascii=False, indent=2))

    budget_cells = len(model_seeds) * len(k_values) if budget_requested else 0
    grid = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "validation_only",
        "official_test_prediction_run": False,
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "regimes": regimes,
        "planned_target_trainings": (
            budget_cells + len(anil_regimes) * len(model_seeds)
            * len(k_values) * len(source_seeds)
        ),
        "completed_target_trainings": len(results),
        "full_grid_complete": len(results) == (
            budget_cells + len(anil_regimes) * len(model_seeds)
            * len(k_values) * len(source_seeds)
        ),
    }
    protocol_output = {
        **grid,
        "experiment12b_dir": str(inputs["experiment"]),
        "source_protocol": str(inputs["protocol"]),
        "source_raw_results": str(inputs["raw"]),
        "validation_units": protocol["validation_units"],
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "target_scope": "predictor.* only",
        "risk_loss": {
            "nasa_loss_weight": args.nasa_loss_weight,
            "high_rul_loss_weight": args.high_rul_loss_weight,
            "high_rul_threshold": args.high_rul_threshold,
            "nasa_exp_clip": args.nasa_exp_clip,
        },
        "tail_selection_rmse_tolerance": args.selection_rmse_tolerance,
    }
    atomic_write_text(output["grid"], json.dumps(grid, ensure_ascii=False, indent=2))
    atomic_write_text(
        output["protocol"], json.dumps(protocol_output, ensure_ascii=False, indent=2)
    )

    if not audit.empty:
        diff_columns = [
            column for column in audit.columns
            if column.endswith("_absolute_difference")
        ]
        max_difference = audit[diff_columns].max(numeric_only=True).max()
        if np.isfinite(max_difference) and max_difference > 1e-3:
            print(
                f"\n[警告] 当前MSE+RMSE参考组未精确复现实验12B，"
                f"最大差异={max_difference:.6g}。"
            )

    print("\n[实验14主要比较]")
    print(comparisons.to_string(index=False))
    print("\n[实验14结论]")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print("\n[输出文件]")
    for name, path in output.items():
        if path.is_file():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
