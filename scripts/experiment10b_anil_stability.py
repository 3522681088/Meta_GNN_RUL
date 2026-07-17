"""实验10B：ANIL数值稳定性定位与修复实验。

本脚本是独立实验入口，不替换 ``main.py`` 或实验7--10。它复用：

* 实验7固定的目标发动机协议；
* 实验10的普通多源监督预训练与RUL预测头定义；
* FD001、FD002、FD003源任务和condition_settings/engine_stage预处理。

实验10B只在源域支持集/查询集上选择稳定配置，不读取官方FD004测试指标。
每个候选配置都会检查support/query loss、内循环梯度、外循环梯度、快速参数
和元模型参数。一旦出现NaN、Inf或超过loss ceiling，立即标记失败并停止，
不再把非有限参数传入后续训练。

推荐按以下顺序运行。

第一阶段：关闭裁剪，定位inner_lr和inner_steps的真实稳定边界
（15组，先用seed=42；clip_norm=0表示不裁剪）：

    python -u scripts/experiment10b_anil_stability.py \
      --stage lr_steps \
      --target FD004 \
      --seeds 42 \
      --meta-epochs 100 \
      --task-mode batch \
      --resume

第二阶段：用第一阶段选出的学习率/步数比较损失尺度（示例）：

    python -u scripts/experiment10b_anil_stability.py \
      --stage loss \
      --target FD004 \
      --seeds 42 \
      --inner-lrs 0.0001 \
      --inner-steps-values 3 \
      --resume

第三阶段：比较无裁剪与不同梯度裁剪阈值（示例）：

    python -u scripts/experiment10b_anil_stability.py \
      --stage clip \
      --target FD004 \
      --seeds 42 \
      --inner-lrs 0.0001 \
      --inner-steps-values 3 \
      --loss-modes scaled_huber \
      --resume

第四阶段：固定最佳配置后用5个种子确认稳定性：

    python -u scripts/experiment10b_anil_stability.py \
      --stage confirm \
      --target FD004 \
      --seeds 42 43 44 45 46 \
      --inner-lrs 0.0001 \
      --inner-steps-values 3 \
      --loss-modes scaled_huber \
      --clip-norms 5 \
      --task-mode batch \
      --resume

只有confirm达到5/5稳定，才进入实验10C的目标域K=2、5正式性能比较。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    load_or_create_protocol,
    train_source_supervised,
)
from scripts.experiment10_anil_repair import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
    cpu_state,
    fresh_source_tasks,
    next_batch,
    parameter_inventory,
    split_source_tasks_by_engine,
)


SCRIPT_VERSION = "experiment10b_anil_stability_v1"
LOSS_MODES = ("raw_mse", "scaled_mse", "raw_huber", "scaled_huber")
STAGES = ("lr_steps", "loss", "clip", "confirm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验10B：ANIL内循环、损失尺度和梯度裁剪稳定性网格"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument("--stage", choices=STAGES, default="lr_steps")
    parser.add_argument("--seeds", nargs="+", type=int)
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
    parser.add_argument(
        "--task-mode", choices=("batch", "engine_disjoint"), default="batch"
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--inner-lrs", nargs="+", type=float)
    parser.add_argument("--inner-steps-values", nargs="+", type=int)
    parser.add_argument("--loss-modes", nargs="+", choices=LOSS_MODES)
    parser.add_argument("--clip-norms", nargs="+", type=float)
    parser.add_argument("--anil-meta-lr", type=float, default=1e-4)
    parser.add_argument("--anil-query-batches", type=int, default=1)
    parser.add_argument("--anil-order", choices=("first", "second"), default="first")
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=10.0,
        help="Huber转折点，单位为RUL周期；缩放损失会自动同步缩放delta",
    )
    parser.add_argument(
        "--loss-ceiling",
        type=float,
        default=1e8,
        help="有限但超过该阈值也视为发散；可按损失尺度调整",
    )
    parser.add_argument("--source-query-fraction", type=float, default=0.30)
    parser.add_argument("--source-task-seed", type=int, default=2027)
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument("--outer-lr", type=float, default=0.05)
    parser.add_argument("--pair-aux-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", default="outputs/experiment10b_anil_stability")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-stable-checkpoints", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def stage_grid(args: argparse.Namespace) -> tuple[list[int], list[float], list[int], list[str], list[float]]:
    defaults = {
        "lr_steps": {
            "seeds": [42],
            "inner_lrs": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            "inner_steps": [1, 3, 5],
            "loss_modes": ["raw_mse"],
            "clip_norms": [0.0],
        },
        "loss": {
            "seeds": [42],
            "inner_lrs": [1e-4],
            "inner_steps": [3],
            "loss_modes": list(LOSS_MODES),
            "clip_norms": [0.0],
        },
        "clip": {
            "seeds": [42],
            "inner_lrs": [1e-4],
            "inner_steps": [3],
            "loss_modes": ["scaled_huber"],
            "clip_norms": [0.0, 1.0, 5.0, 10.0],
        },
        "confirm": {
            "seeds": [42, 43, 44, 45, 46],
            "inner_lrs": [1e-4],
            "inner_steps": [3],
            "loss_modes": ["scaled_huber"],
            "clip_norms": [5.0],
        },
    }[args.stage]
    seeds = list(dict.fromkeys(args.seeds or defaults["seeds"]))
    inner_lrs = sorted(set(args.inner_lrs or defaults["inner_lrs"]))
    inner_steps = sorted(set(args.inner_steps_values or defaults["inner_steps"]))
    loss_modes = list(dict.fromkeys(args.loss_modes or defaults["loss_modes"]))
    clip_norms = sorted(set(args.clip_norms or defaults["clip_norms"]))
    if not seeds:
        raise ValueError("--seeds不能为空")
    if any(value <= 0 for value in inner_lrs):
        raise ValueError("--inner-lrs必须为正数")
    if any(value <= 0 for value in inner_steps):
        raise ValueError("--inner-steps-values必须为正整数")
    if any(value < 0 for value in clip_norms):
        raise ValueError("--clip-norms不能为负数；0表示不裁剪")
    return seeds, inner_lrs, inner_steps, loss_modes, clip_norms


def load_config(args: argparse.Namespace, seed: int, inner_lr: float, inner_steps: int) -> dict:
    # 复用实验10的配置字段语义，同时把普通预训练学习率与ANIL inner_lr分离。
    from scripts.experiment10_anil_repair import load_config as load_experiment10_config

    proxy = argparse.Namespace(**vars(args))
    proxy.seed = seed
    proxy.inner_lr = inner_lr
    proxy.inner_steps = inner_steps
    proxy.budget_meta_epochs = None
    proxy.budget_anil_meta_lr = None
    proxy.source_pretrain_steps = args.source_pretrain_steps
    proxy.source_pretrain_lr = args.source_pretrain_lr
    cfg = load_experiment10_config(proxy, seed)
    cfg["pair_aux_weight"] = 0.0
    return cfg


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment10b_{args.stage}_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "ranking": output / f"{prefix}_ranking.csv",
        "failures": output / f"{prefix}_failures.csv",
        "history": output / f"{prefix}_history.json",
        "protocol": output / f"{prefix}_protocol.json",
        "manifest": output / f"{prefix}_source_task_splits.json",
        "plan": output / f"{prefix}_grid_plan.json",
    }


def pretrain_cache_path(args: argparse.Namespace, seed: int) -> Path:
    return result_paths(args)["output"] / "source_cache" / (
        f"ordinary_pretraining_{args.target}_seed{seed}.pt"
    )


def pretrain_signature(args: argparse.Namespace, cfg: dict, feature_count: int) -> str:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    payload = {
        "script": SCRIPT_VERSION,
        "target": args.target,
        "seed": cfg["seed"],
        "source_domains": cfg["source_domains"],
        "feature_count": feature_count,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "data_dir": cfg["data_dir"],
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "source_pretrain_lr": cfg["source_pretrain_lr"],
        "source_pretrain_weight_decay": cfg["source_pretrain_weight_decay"],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def all_tensors_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def tensor_list_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        norm = float(torch.linalg.vector_norm(tensor.detach().float()).cpu().item())
        if not math.isfinite(norm):
            return float("inf")
        total += norm * norm
    return math.sqrt(total)


def model_parameter_abs_max(model: torch.nn.Module) -> float:
    maximum = 0.0
    for parameter in model.parameters():
        value = float(parameter.detach().abs().max().cpu().item())
        if not math.isfinite(value):
            return float("inf")
        maximum = max(maximum, value)
    return maximum


def loss_scale(mode: str, rul_cap: float) -> float:
    return float(rul_cap) if mode.startswith("scaled_") else 1.0


def functional_stability_loss(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    rul_cap: float,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = functional_call(model, state, (x,))
    scale = loss_scale(mode, rul_cap)
    prediction = prediction / scale
    target = y / scale
    if mode.endswith("_mse"):
        return torch.nn.functional.mse_loss(prediction, target), prediction
    if mode.endswith("_huber"):
        return (
            torch.nn.functional.huber_loss(
                prediction, target, delta=huber_delta / scale
            ),
            prediction,
        )
    raise ValueError(f"未知loss mode：{mode}")


def raw_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return one objective metric shared by every candidate loss mode."""
    # Double precision avoids float32 square overflow during instability diagnosis.
    value = torch.sqrt(
        torch.mean((prediction.detach().double() - target.detach().double()) ** 2)
    )
    return float(value.cpu().item())


def failure_result(
    base: dict,
    epoch: int,
    phase: str,
    message: str,
    history: list[dict],
    diagnostics: dict,
) -> tuple[dict, list[dict], None]:
    result = {
        **base,
        "status": "nonfinite",
        "stable": False,
        "epochs_completed": max(0, epoch - 1),
        "first_failure_epoch": epoch,
        "failure_phase": phase,
        "failure_message": message,
        **diagnostics,
    }
    print(
        f"[NONFINITE] config={base['config_id']} epoch={epoch} "
        f"phase={phase} message={message}"
    )
    return result, history, None


def train_safe_anil(
    model: torch.nn.Module,
    source_tasks: dict[str, torch.utils.data.DataLoader],
    cfg: dict,
    args: argparse.Namespace,
    *,
    inner_lr: float,
    inner_steps: int,
    loss_mode: str,
    clip_norm: float,
    config_id: str,
) -> tuple[dict, list[dict], dict[str, torch.Tensor] | None, dict | None]:
    """Run one guarded ANIL source-training configuration."""
    meta_model = deepcopy(model).to(resolve_device(cfg["device"]))
    device = next(meta_model.parameters()).device
    meta_model.train()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.anil_meta_lr)
    task_names = sorted(source_tasks)
    task_count = min(cfg["tasks_per_meta_batch"], len(task_names))
    split_manifest = None

    if args.task_mode == "engine_disjoint":
        task_pairs, split_manifest = split_source_tasks_by_engine(
            source_tasks,
            args.balance_mode,
            args.source_query_fraction,
            args.source_task_seed,
        )
        support_iterators = {name: iter(task_pairs[name][0]) for name in task_names}
        query_iterators = {name: iter(task_pairs[name][1]) for name in task_names}
    else:
        task_pairs = {name: (source_tasks[name], source_tasks[name]) for name in task_names}
        support_iterators = {name: iter(source_tasks[name]) for name in task_names}
        query_iterators = support_iterators

    head_names = [
        name for name, _ in meta_model.named_parameters() if name.startswith("predictor.")
    ]
    if not head_names:
        raise RuntimeError("未找到predictor.* RUL预测头参数")

    base = {
        "config_id": config_id,
        "seed": cfg["seed"],
        "stage": args.stage,
        "target_domain": args.target,
        "task_mode": args.task_mode,
        "inner_lr": inner_lr,
        "inner_steps": inner_steps,
        "loss_mode": loss_mode,
        "loss_scale": loss_scale(loss_mode, cfg.get("rul_cap", 125)),
        "huber_delta_cycles": args.huber_delta,
        "effective_huber_delta": (
            args.huber_delta
            / loss_scale(loss_mode, cfg.get("rul_cap", 125))
        ),
        "clip_norm": clip_norm,
        "meta_lr": args.anil_meta_lr,
        "meta_epochs_planned": args.meta_epochs,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_query_fraction": args.source_query_fraction,
        "source_task_seed": args.source_task_seed,
    }
    history: list[dict] = []
    max_support_loss = 0.0
    max_query_loss = 0.0
    max_query_raw_rmse = 0.0
    max_inner_grad_norm = 0.0
    max_outer_grad_norm = 0.0
    clip_events = 0
    inner_gradient_steps = 0
    outer_gradient_steps = 0
    report_every = max(1, args.meta_epochs // 10)
    use_second_order = args.anil_order == "second"

    def diagnostics() -> dict:
        query_values = [row["query_loss"] for row in history]
        raw_rmse_values = [row["query_raw_rmse"] for row in history]
        tail = query_values[-min(10, len(query_values)) :] if query_values else []
        raw_tail = (
            raw_rmse_values[-min(10, len(raw_rmse_values)) :]
            if raw_rmse_values
            else []
        )
        return {
            "max_support_loss": max_support_loss,
            "max_query_loss": max_query_loss,
            "max_query_raw_rmse": max_query_raw_rmse,
            "max_inner_grad_norm": max_inner_grad_norm,
            "max_outer_grad_norm": max_outer_grad_norm,
            "clip_events": clip_events,
            "inner_gradient_steps": inner_gradient_steps,
            "outer_gradient_steps": outer_gradient_steps,
            "clip_rate": clip_events / max(1, inner_gradient_steps + outer_gradient_steps),
            "best_query_loss": min(query_values) if query_values else float("nan"),
            "final_query_loss": query_values[-1] if query_values else float("nan"),
            "tail_query_loss_mean": float(np.mean(tail)) if tail else float("nan"),
            "best_query_raw_rmse": (
                min(raw_rmse_values) if raw_rmse_values else float("nan")
            ),
            "final_query_raw_rmse": (
                raw_rmse_values[-1] if raw_rmse_values else float("nan")
            ),
            "tail_query_raw_rmse_mean": (
                float(np.mean(raw_tail)) if raw_tail else float("nan")
            ),
            "final_parameter_abs_max": model_parameter_abs_max(meta_model),
        }

    for epoch in range(1, args.meta_epochs + 1):
        selected = random.sample(task_names, task_count)
        task_query_losses: list[torch.Tensor] = []
        epoch_query_raw_rmses: list[float] = []
        epoch_support_losses: list[float] = []
        epoch_inner_norms: list[float] = []

        for task_name in selected:
            support_loader, query_loader = task_pairs[task_name]
            parameters = dict(meta_model.named_parameters())
            buffers = dict(meta_model.named_buffers())
            fast_head = {name: parameters[name] for name in head_names}

            for _ in range(inner_steps):
                (support_x, support_y), iterator = next_batch(
                    support_loader, support_iterators[task_name]
                )
                support_iterators[task_name] = iterator
                if args.task_mode == "batch":
                    query_iterators[task_name] = iterator
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                state = {**buffers, **parameters, **fast_head}
                support_loss, _ = functional_stability_loss(
                    meta_model,
                    state,
                    support_x,
                    support_y,
                    loss_mode,
                    cfg.get("rul_cap", 125),
                    args.huber_delta,
                )
                support_value = float(support_loss.detach().cpu().item())
                if not math.isfinite(support_value):
                    result, hist, state_out = failure_result(
                        base, epoch, "support_loss", str(support_value), history, diagnostics()
                    )
                    return result, hist, state_out, split_manifest
                if support_value > args.loss_ceiling:
                    result, hist, state_out = failure_result(
                        base,
                        epoch,
                        "support_loss_ceiling",
                        f"{support_value:.6g}>{args.loss_ceiling:.6g}",
                        history,
                        diagnostics(),
                    )
                    return result, hist, state_out, split_manifest
                max_support_loss = max(max_support_loss, support_value)
                epoch_support_losses.append(support_value)

                gradients = torch.autograd.grad(
                    support_loss,
                    tuple(fast_head.values()),
                    create_graph=use_second_order,
                    allow_unused=True,
                )
                normalized_gradients: list[torch.Tensor] = []
                for parameter, gradient in zip(fast_head.values(), gradients):
                    if gradient is None:
                        gradient = torch.zeros_like(parameter)
                    if not use_second_order:
                        gradient = gradient.detach()
                    normalized_gradients.append(gradient)
                if not all_tensors_finite(normalized_gradients):
                    result, hist, state_out = failure_result(
                        base, epoch, "inner_gradient", "NaN/Inf", history, diagnostics()
                    )
                    return result, hist, state_out, split_manifest
                grad_norm = tensor_list_norm(normalized_gradients)
                max_inner_grad_norm = max(max_inner_grad_norm, grad_norm)
                epoch_inner_norms.append(grad_norm)
                inner_gradient_steps += 1
                if clip_norm > 0:
                    coefficient = min(1.0, clip_norm / (grad_norm + 1e-12))
                    if coefficient < 1.0:
                        clip_events += 1
                else:
                    coefficient = 1.0
                updated = {
                    name: parameter - inner_lr * gradient * coefficient
                    for (name, parameter), gradient in zip(
                        fast_head.items(), normalized_gradients
                    )
                }
                if not all_tensors_finite(updated.values()):
                    result, hist, state_out = failure_result(
                        base, epoch, "fast_head_parameter", "NaN/Inf", history, diagnostics()
                    )
                    return result, hist, state_out, split_manifest
                fast_head = updated

            query_losses: list[torch.Tensor] = []
            for _ in range(args.anil_query_batches):
                (query_x, query_y), iterator = next_batch(
                    query_loader, query_iterators[task_name]
                )
                query_iterators[task_name] = iterator
                if args.task_mode == "batch":
                    support_iterators[task_name] = iterator
                query_x = query_x.to(device)
                query_y = query_y.to(device)
                state = {**buffers, **parameters, **fast_head}
                query_loss, query_prediction = functional_stability_loss(
                    meta_model,
                    state,
                    query_x,
                    query_y,
                    loss_mode,
                    cfg.get("rul_cap", 125),
                    args.huber_delta,
                )
                query_value = float(query_loss.detach().cpu().item())
                if not math.isfinite(query_value):
                    result, hist, state_out = failure_result(
                        base, epoch, "query_loss", str(query_value), history, diagnostics()
                    )
                    return result, hist, state_out, split_manifest
                if query_value > args.loss_ceiling:
                    result, hist, state_out = failure_result(
                        base,
                        epoch,
                        "query_loss_ceiling",
                        f"{query_value:.6g}>{args.loss_ceiling:.6g}",
                        history,
                        diagnostics(),
                    )
                    return result, hist, state_out, split_manifest
                query_rmse = raw_rmse(query_prediction, query_y)
                if not math.isfinite(query_rmse):
                    result, hist, state_out = failure_result(
                        base,
                        epoch,
                        "query_raw_rmse",
                        str(query_rmse),
                        history,
                        diagnostics(),
                    )
                    return result, hist, state_out, split_manifest
                epoch_query_raw_rmses.append(query_rmse)
                query_losses.append(query_loss)
            task_query_losses.append(torch.stack(query_losses).mean())

        meta_loss = torch.stack(task_query_losses).mean()
        meta_value = float(meta_loss.detach().cpu().item())
        if not math.isfinite(meta_value):
            result, hist, state_out = failure_result(
                base, epoch, "meta_query_loss", str(meta_value), history, diagnostics()
            )
            return result, hist, state_out, split_manifest
        max_query_loss = max(max_query_loss, meta_value)
        epoch_query_raw_rmse = float(np.mean(epoch_query_raw_rmses))
        max_query_raw_rmse = max(max_query_raw_rmse, epoch_query_raw_rmse)

        optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        gradients = [
            parameter.grad
            for parameter in meta_model.parameters()
            if parameter.grad is not None
        ]
        if not all_tensors_finite(gradients):
            result, hist, state_out = failure_result(
                base, epoch, "outer_gradient", "NaN/Inf", history, diagnostics()
            )
            return result, hist, state_out, split_manifest
        outer_norm = tensor_list_norm(gradients)
        max_outer_grad_norm = max(max_outer_grad_norm, outer_norm)
        outer_gradient_steps += 1
        if clip_norm > 0:
            if outer_norm > clip_norm:
                clip_events += 1
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), clip_norm)
        optimizer.step()
        if not all_tensors_finite(meta_model.parameters()):
            result, hist, state_out = failure_result(
                base, epoch, "meta_parameter", "NaN/Inf after optimizer.step", history, diagnostics()
            )
            return result, hist, state_out, split_manifest

        row = {
            "epoch": epoch,
            "query_loss": meta_value,
            "query_raw_rmse": epoch_query_raw_rmse,
            "support_loss_mean": float(np.mean(epoch_support_losses)),
            "inner_grad_norm_max": max(epoch_inner_norms),
            "outer_grad_norm": outer_norm,
            "parameter_abs_max": model_parameter_abs_max(meta_model),
            "tasks": selected,
        }
        history.append(row)
        if epoch % report_every == 0 or epoch == 1 or epoch == args.meta_epochs:
            print(
                f"stability_epoch={epoch:04d}/{args.meta_epochs} "
                f"config={config_id} query_loss={meta_value:.6g} "
                f"query_raw_rmse={epoch_query_raw_rmse:.4f} "
                f"inner_grad={row['inner_grad_norm_max']:.6g} "
                f"outer_grad={outer_norm:.6g}"
            )

    result = {
        **base,
        "status": "stable",
        "stable": True,
        "epochs_completed": args.meta_epochs,
        "first_failure_epoch": None,
        "failure_phase": None,
        "failure_message": None,
        **diagnostics(),
    }
    return result, history, cpu_state(meta_model), split_manifest


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    groups = [
        "stage",
        "task_mode",
        "inner_lr",
        "inner_steps",
        "loss_mode",
        "clip_norm",
        "meta_lr",
        "meta_epochs_planned",
    ]
    rows: list[dict] = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        stable = group[group.stable]
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_runs": len(group),
                "n_stable": int(group.stable.sum()),
                "stable_rate": float(group.stable.mean()),
                "mean_epochs_completed": float(group.epochs_completed.mean()),
                "final_query_loss_mean": stable.final_query_loss.mean(),
                "final_query_loss_std": stable.final_query_loss.std(ddof=1),
                "tail_query_loss_mean": stable.tail_query_loss_mean.mean(),
                "final_query_raw_rmse_mean": stable.final_query_raw_rmse.mean(),
                "final_query_raw_rmse_std": stable.final_query_raw_rmse.std(ddof=1),
                "tail_query_raw_rmse_mean": stable.tail_query_raw_rmse_mean.mean(),
                "max_inner_grad_norm": group.max_inner_grad_norm.max(),
                "max_outer_grad_norm": group.max_outer_grad_norm.max(),
                "clip_rate_mean": group.clip_rate.mean(),
                "max_parameter_abs": group.final_parameter_abs_max.max(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["stable_rate", "tail_query_raw_rmse_mean", "clip_rate_mean"],
        ascending=[False, True, True],
        na_position="last",
    )


def save_progress(
    args: argparse.Namespace,
    results: list[dict],
    histories: dict[str, list[dict]],
) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    summary = summarize(results)
    atomic_write_text(
        paths["summary"], summary.to_csv(index=False), encoding="utf-8-sig"
    )
    ranking = summary[
        (summary.n_stable == summary.n_runs) & (summary.n_runs > 0)
    ].copy()
    if not ranking.empty:
        ranking.insert(0, "stability_rank", np.arange(1, len(ranking) + 1))
    atomic_write_text(
        paths["ranking"], ranking.to_csv(index=False), encoding="utf-8-sig"
    )
    failures = pd.DataFrame(results)
    failures = failures[~failures.stable] if not failures.empty else failures
    atomic_write_text(
        paths["failures"], failures.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["history"], json.dumps(histories, ensure_ascii=False, indent=2)
    )
    return paths


def completed_keys(results: list[dict]) -> set[tuple]:
    return {
        (
            int(row["seed"]),
            float(row["inner_lr"]),
            int(row["inner_steps"]),
            str(row["loss_mode"]),
            float(row["clip_norm"]),
            str(row["task_mode"]),
        )
        for row in results
    }


def config_id(seed: int, lr: float, steps: int, mode: str, clip: float, task: str) -> str:
    return (
        f"seed{seed}_lr{lr:.0e}_steps{steps}_{mode}_clip{clip:g}_{task}"
        .replace("+", "")
        .replace("-0", "-")
    )


def ordinary_pretraining_state(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int]:
    source_tasks, feature_count = fresh_source_tasks(args, cfg, protocol, seed)
    cache = pretrain_cache_path(args, seed)
    signature = pretrain_signature(args, cfg, feature_count)
    if args.resume and cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("signature") == signature:
            print(f"[ordinary cache] {cache}")
            del source_tasks
            state = payload["state"]
            if not all_tensors_finite(state.values()):
                raise RuntimeError(f"普通预训练缓存含NaN/Inf：{cache}")
            return state, feature_count
        print(f"[cache ignored] 普通预训练签名变化：{cache.name}")

    seed_everything(seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model, history = train_source_supervised(
        model, source_tasks, cfg, resolve_device(cfg["device"])
    )
    state = cpu_state(model)
    if not all_tensors_finite(state.values()):
        raise RuntimeError("普通源域预训练已经产生NaN/Inf，无法继续实验10B")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"signature": signature, "state": state, "history": history}, cache
    )
    del model, source_tasks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, feature_count


def write_protocol_files(
    args: argparse.Namespace,
    protocol: dict,
    source_protocol_path: Path | None,
    grid_plan: dict,
) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied = dict(protocol)
    copied["experiment10b_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path is not None else "regenerated"
    )
    atomic_write_text(paths["protocol"], json.dumps(copied, ensure_ascii=False, indent=2))
    atomic_write_text(paths["plan"], json.dumps(grid_plan, ensure_ascii=False, indent=2))
    return paths


def main() -> None:
    args = parse_args()
    seeds, inner_lrs, inner_steps_values, loss_modes, clip_norms = stage_grid(args)
    if args.meta_epochs <= 0 or args.anil_query_batches <= 0:
        raise ValueError("meta-epochs和anil-query-batches必须为正整数")
    if args.source_pretrain_steps <= 0 or args.source_pretrain_lr <= 0:
        raise ValueError("普通预训练步数和学习率必须为正数")
    if args.anil_meta_lr <= 0 or args.huber_delta <= 0 or args.loss_ceiling <= 0:
        raise ValueError("ANIL学习率、Huber delta和loss ceiling必须为正数")
    if not 0 < args.source_query_fraction < 1:
        raise ValueError("source-query-fraction必须位于(0,1)")

    # 协议只需要K=2来复用源任务；实验10B本身不评估官方测试指标。
    first_cfg = load_config(args, seeds[0], inner_lrs[0], inner_steps_values[0])
    protocol, source_protocol_path = load_or_create_protocol(
        args, first_cfg, seeds, [2]
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

    combinations = list(
        itertools.product(seeds, inner_lrs, inner_steps_values, loss_modes, clip_norms)
    )
    plan = {
        "script_version": SCRIPT_VERSION,
        "stage": args.stage,
        "target": args.target,
        "note": "Only source support/query data are used for stability selection; official test metrics are not evaluated.",
        "seeds": seeds,
        "inner_lrs": inner_lrs,
        "inner_steps_values": inner_steps_values,
        "loss_modes": loss_modes,
        "clip_norms": clip_norms,
        "task_mode": args.task_mode,
        "meta_epochs": args.meta_epochs,
        "meta_lr": args.anil_meta_lr,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "loss_ceiling": args.loss_ceiling,
        "planned_run_count": len(combinations),
        "pass_rule": (
            "All planned epochs finite; no loss-ceiling violation; "
            "confirm stage requires 5/5 stable seeds."
        ),
    }
    paths = write_protocol_files(args, protocol, source_protocol_path, plan)
    print("\n[实验10B稳定性网格]")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.dry_run:
        source_tasks, feature_count = fresh_source_tasks(
            args, first_cfg, protocol, seeds[0]
        )
        first_domain = first_cfg["source_domains"][0]
        x, y = next(iter(source_tasks[first_domain]))
        seed_everything(seeds[0])
        model = build_model("meta_gnn", feature_count, first_cfg).cpu()
        inventory = parameter_inventory(model, 0.0)
        print(
            json.dumps(
                {
                    "feature_count": feature_count,
                    "source_example_shape": list(x.shape),
                    "source_label_shape": list(y.shape),
                    "rul_head_parameter_count": inventory["rul_head_parameter_count"],
                    "total_parameter_count": inventory["total_parameter_count"],
                    "official_test_engine_count_only_for_protocol_check": protocol[
                        "official_test_engine_count"
                    ],
                    "official_test_metrics_evaluated": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"\n[dry-run完成] 未训练模型。\nPlan: {paths['plan']}")
        return

    results: list[dict] = []
    histories: dict[str, list[dict]] = {}
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条配置结果。")
    if args.resume and paths["history"].is_file():
        histories = json.loads(paths["history"].read_text(encoding="utf-8"))
    done = completed_keys(results)
    all_manifests: dict[str, dict] = {}
    if args.resume and paths["manifest"].is_file():
        all_manifests = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    for seed in seeds:
        cfg_for_pretrain = load_config(args, seed, inner_lrs[0], inner_steps_values[0])
        ordinary_state, feature_count = ordinary_pretraining_state(
            args, cfg_for_pretrain, protocol, seed
        )
        for inner_lr, inner_steps, loss_mode, clip_norm in itertools.product(
            inner_lrs, inner_steps_values, loss_modes, clip_norms
        ):
            key = (seed, inner_lr, inner_steps, loss_mode, clip_norm, args.task_mode)
            if key in done:
                print(f"[skip] {key}")
                continue
            cfg = load_config(args, seed, inner_lr, inner_steps)
            run_id = config_id(
                seed, inner_lr, inner_steps, loss_mode, clip_norm, args.task_mode
            )
            print(f"\n[experiment10B] {run_id}")
            # 每个配置重新构造加载器，使采样器从相同随机状态开始，避免配置顺序影响批次。
            source_tasks, current_feature_count = fresh_source_tasks(
                args, cfg, protocol, seed
            )
            if current_feature_count != feature_count:
                raise AssertionError("同一协议下feature_count发生变化")
            seed_everything(seed)
            model = build_model("meta_gnn", feature_count, cfg)
            model.load_state_dict(ordinary_state)
            result, history, stable_state, manifest = train_safe_anil(
                model,
                source_tasks,
                cfg,
                args,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                loss_mode=loss_mode,
                clip_norm=clip_norm,
                config_id=run_id,
            )
            results.append(result)
            histories[run_id] = history
            done.add(key)
            if manifest is not None:
                all_manifests[run_id] = manifest
                atomic_write_text(
                    paths["manifest"],
                    json.dumps(all_manifests, ensure_ascii=False, indent=2),
                )
            if result["stable"] and args.save_stable_checkpoints and stable_state is not None:
                checkpoint = paths["output"] / "stable_checkpoints" / f"{run_id}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": stable_state,
                        "config": cfg,
                        "stability_result": result,
                        "source_history": history,
                    },
                    checkpoint,
                )
            paths = save_progress(args, results, histories)
            del model, source_tasks, stable_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(results)
    print("\n[实验10B汇总]")
    print(summary.to_string(index=False))
    fully_stable = summary[
        (summary.n_stable == summary.n_runs) & (summary.n_runs > 0)
    ]
    if args.stage == "confirm":
        if not fully_stable.empty and int(fully_stable.iloc[0].n_stable) >= 5:
            print("\n[PASS] 至少一个配置达到5/5稳定，可进入实验10C目标域比较。")
        else:
            print("\n[NOT PASS] 尚无配置达到5/5稳定，不应进入官方测试比较。")
    else:
        print("\n[判定] 只从stable_rate=1的配置中选择下一阶段候选；不得查看官方测试结果调参。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nRanking: {paths['ranking']}\nFailures: {paths['failures']}"
        f"\nHistory: {paths['history']}\nPlan: {paths['plan']}"
    )


if __name__ == "__main__":
    main()
