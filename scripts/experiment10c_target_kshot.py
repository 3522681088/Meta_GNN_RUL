"""实验10C：稳定ANIL的FD004目标域K-shot正式验证。

本脚本是独立实验入口，不替换 ``main.py`` 或实验7--10B。它复用：

* 实验7固定且严格嵌套的K-shot目标发动机划分；
* condition_settings工况特征与engine_stage平衡采样；
* 实验10B已经确认稳定的ANIL区间：raw尺度损失、inner_lr=1e-5、
  inner_steps=1；默认不裁剪梯度。若希望严格使用实验10B稳定性排名第一的设置，
  可显式传入 ``--meta-clip-norm 5``；
* FD004官方测试发动机只用于最终评估，目标训练轮次由固定验证发动机选择。

默认比较八组方案：

``pretrained_full_mse``
    普通多源监督预训练，目标域更新完整模型，使用MSE。

``pretrained_head_mse`` / ``pretrained_head_huber``
    普通预训练，目标域只更新RUL预测头；用于隔离损失函数作用。

``pretrained_budget_head_mse`` / ``pretrained_budget_head_huber``
    普通预训练后，再增加与ANIL近似匹配的源域损失梯度预算，然后只更新RUL头。
    它们是证明“元学习收益并非只是更多源域训练”的关键对照。

``reptile_full_mse``
    从随机初始化进行完整Reptile元训练，目标域更新完整模型。

``anil_head_mse`` / ``anil_head_huber``
    从普通预训练状态开始稳定ANIL源域训练，目标域只更新RUL预测头。

正式实验建议：

    python -u scripts/experiment10c_target_kshot.py \
      --target FD004 \
      --k-values 2 5 10 20 \
      --seeds 42 43 44 45 46 \
      --preprocessing condition_settings \
      --balance-mode engine_stage \
      --meta-epochs 100 \
      --meta-inner-lr 0.00001 \
      --meta-inner-steps 1 \
      --target-epochs 10 \
      --target-lr 0.001 \
      --source-pretrain-steps 1500 \
      --resume

先做协议与前向检查：

    python -u scripts/experiment10c_target_kshot.py --target FD004 --dry-run

重要：不得根据官方FD004测试结果继续调整超参数。本脚本运行结束后，结果应视为
锁定协议下的正式模型比较；若仍需调参，应只使用固定验证发动机重新建立新实验编号。
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
    train_source_meta,
)
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    load_or_create_protocol,
    train_source_supervised,
)
from scripts.experiment9_anil_ablation import parameter_drift  # noqa: E402
from scripts.experiment10_anil_repair import (  # noqa: E402
    cpu_state,
    fresh_source_tasks,
    parameter_inventory,
)
from scripts.experiment10b_anil_stability import (  # noqa: E402
    all_tensors_finite,
    load_config as load_stability_config,
    ordinary_pretraining_state,
    train_safe_anil,
)


SCRIPT_VERSION = "experiment10c_target_kshot_v1"
REGIMES = (
    "pretrained_full_mse",
    "pretrained_head_mse",
    "pretrained_head_huber",
    "pretrained_budget_head_mse",
    "pretrained_budget_head_huber",
    "reptile_full_mse",
    "anil_head_mse",
    "anil_head_huber",
)

COMPARISONS = (
    ("anil_head_mse", "pretrained_head_mse", "anil_mse_vs_ordinary_head"),
    (
        "anil_head_mse",
        "pretrained_budget_head_mse",
        "anil_mse_vs_budget_matched_head",
    ),
    (
        "anil_head_huber",
        "pretrained_head_huber",
        "anil_huber_vs_ordinary_head",
    ),
    (
        "anil_head_huber",
        "pretrained_budget_head_huber",
        "anil_huber_vs_budget_matched_head",
    ),
    ("anil_head_mse", "reptile_full_mse", "anil_head_vs_reptile_full"),
    ("anil_head_mse", "pretrained_full_mse", "anil_head_vs_pretrained_full"),
    ("anil_head_huber", "anil_head_mse", "anil_huber_vs_anil_mse"),
    (
        "pretrained_head_huber",
        "pretrained_head_mse",
        "target_huber_vs_target_mse_without_meta",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验10C：稳定ANIL与普通预训练/Reptile的FD004 K-shot正式比较"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
    )
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
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

    # 稳定ANIL源训练设置，与实验10B分离命名，避免和目标适应学习率混淆。
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--meta-inner-lr", type=float, default=1e-5)
    parser.add_argument("--meta-inner-steps", type=int, default=1)
    parser.add_argument("--anil-meta-lr", type=float, default=1e-4)
    parser.add_argument("--anil-query-batches", type=int, default=1)
    parser.add_argument("--anil-order", choices=("first", "second"), default="first")
    parser.add_argument("--anil-task-mode", choices=("batch", "engine_disjoint"), default="batch")
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--meta-clip-norm", type=float, default=0.0)
    parser.add_argument("--loss-ceiling", type=float, default=1e8)
    parser.add_argument("--source-query-fraction", type=float, default=0.30)
    parser.add_argument("--source-task-seed", type=int, default=2027)
    parser.add_argument("--outer-lr", type=float, default=0.05)
    parser.add_argument("--pair-aux-weight", type=float, default=0.0)

    # 普通源域预训练和预算匹配控制。
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--budget-extra-steps",
        type=int,
        help=(
            "普通预算匹配组额外监督更新步数；默认等于meta_epochs × "
            "tasks_per_meta_batch × (meta_inner_steps + anil_query_batches)"
        ),
    )

    # 目标域所有方案使用完全相同的适应预算。
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clip-norm", type=float, default=0.0)

    parser.add_argument("--output-dir", default="outputs/experiment10c_target_kshot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[int], list[str]]:
    k_values = sorted(set(args.k_values))
    seeds = list(dict.fromkeys(args.seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须是正整数")
    if not seeds:
        raise ValueError("--seeds不能为空")
    if not regimes:
        raise ValueError("--regimes不能为空")
    positive = {
        "meta_epochs": args.meta_epochs,
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_query_batches": args.anil_query_batches,
        "huber_delta": args.huber_delta,
        "loss_ceiling": args.loss_ceiling,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正数：{invalid}")
    if args.meta_clip_norm < 0 or args.target_clip_norm < 0:
        raise ValueError("裁剪阈值不能为负数；0表示不裁剪")
    if args.budget_extra_steps is not None and args.budget_extra_steps <= 0:
        raise ValueError("--budget-extra-steps必须为正整数")
    if not 0 < args.source_query_fraction < 1:
        raise ValueError("--source-query-fraction必须位于(0,1)")
    if len(seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个随机种子，只能视为预实验。")
    return k_values, seeds, regimes


def load_config(args: argparse.Namespace, seed: int) -> dict:
    # experiment10B的配置代理需要stage和task_mode字段。
    args.stage = "confirm"
    args.task_mode = args.anil_task_mode
    cfg = load_stability_config(
        args,
        seed,
        args.meta_inner_lr,
        args.meta_inner_steps,
    )
    cfg.update(
        {
            "target_epochs": args.target_epochs,
            "target_lr": args.target_lr,
            "target_weight_decay": args.target_weight_decay,
            "pair_aux_weight": 0.0,
            "output_dir": str(resolve_path(args.output_dir, PROJECT_ROOT)),
        }
    )
    return cfg


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment10c_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_by_seed.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_split_protocol.json",
        "splits": output / f"{prefix}_engine_splits.csv",
        "budget": output / f"{prefix}_budget.json",
        "source_diagnostics": output / f"{prefix}_source_diagnostics.json",
        "parameters": output / f"{prefix}_parameter_inventory.json",
    }


def regime_spec(regime: str) -> dict:
    specs = {
        "pretrained_full_mse": {
            "source": "ordinary",
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "full",
            "target_loss": "raw_mse",
        },
        "pretrained_head_mse": {
            "source": "ordinary",
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "rul_head",
            "target_loss": "raw_mse",
        },
        "pretrained_head_huber": {
            "source": "ordinary",
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "rul_head",
            "target_loss": "raw_huber",
        },
        "pretrained_budget_head_mse": {
            "source": "ordinary_budget",
            "source_training": "ordinary_pretraining_plus_budget_matched_continuation",
            "target_scope": "rul_head",
            "target_loss": "raw_mse",
        },
        "pretrained_budget_head_huber": {
            "source": "ordinary_budget",
            "source_training": "ordinary_pretraining_plus_budget_matched_continuation",
            "target_scope": "rul_head",
            "target_loss": "raw_huber",
        },
        "reptile_full_mse": {
            "source": "reptile",
            "source_training": "reptile_full_model_meta_training",
            "target_scope": "full",
            "target_loss": "raw_mse",
        },
        "anil_head_mse": {
            "source": "anil_raw_mse",
            "source_training": "ordinary_pretraining_plus_stable_anil_raw_mse",
            "target_scope": "rul_head",
            "target_loss": "raw_mse",
        },
        "anil_head_huber": {
            "source": "anil_raw_huber",
            "source_training": "ordinary_pretraining_plus_stable_anil_raw_huber",
            "target_scope": "rul_head",
            "target_loss": "raw_huber",
        },
    }
    return specs[regime]


def source_cache_path(args: argparse.Namespace, source_key: str, seed: int) -> Path:
    return result_paths(args)["output"] / "source_cache" / (
        f"{source_key}_{args.target}_seed{seed}.pt"
    )


def source_signature(
    args: argparse.Namespace,
    cfg: dict,
    source_key: str,
    feature_count: int,
    budget_extra_steps: int,
) -> str:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    script_path = Path(__file__).resolve()
    payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_key": source_key,
        "target": args.target,
        "seed": cfg["seed"],
        "source_domains": cfg["source_domains"],
        "feature_count": feature_count,
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "budget_extra_steps": budget_extra_steps,
        "meta_epochs": args.meta_epochs,
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_task_mode": args.anil_task_mode,
        "meta_clip_norm": args.meta_clip_norm,
        "huber_delta": args.huber_delta,
        "outer_lr": args.outer_lr,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_cached_source(path: Path, signature: str) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        print(f"[cache ignored] 源模型签名变化：{path.name}")
        return None
    state = payload.get("state")
    if not isinstance(state, dict) or not all_tensors_finite(state.values()):
        raise RuntimeError(f"源模型缓存无效或包含NaN/Inf：{path}")
    print(f"[source cache] {path}")
    return payload


def save_source_cache(
    path: Path,
    signature: str,
    state: dict,
    history: list,
    diagnostic: dict,
) -> None:
    if not all_tensors_finite(state.values()):
        raise RuntimeError("拒绝保存包含NaN/Inf的源模型")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "state": state,
            "history": history,
            "diagnostic": diagnostic,
        },
        path,
    )


def budget_extra_steps(args: argparse.Namespace, cfg: dict) -> int:
    if args.budget_extra_steps is not None:
        return int(args.budget_extra_steps)
    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    return int(
        args.meta_epochs
        * task_count
        * (args.meta_inner_steps + args.anil_query_batches)
    )


def build_source_states(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    required_regimes: list[str],
) -> tuple[dict[str, dict], dict[str, list], dict[str, dict], dict]:
    source_keys = list(dict.fromkeys(regime_spec(name)["source"] for name in required_regimes))
    source_tasks, feature_count = fresh_source_tasks(args, cfg, protocol, seed)
    del source_tasks
    extra_steps = budget_extra_steps(args, cfg)
    device = resolve_device(cfg["device"])

    # 普通预训练是ANIL和普通迁移对照的共同起点。
    ordinary_state, cached_feature_count = ordinary_pretraining_state(
        args, cfg, protocol, seed
    )
    if cached_feature_count != feature_count:
        raise AssertionError("普通预训练与当前协议的feature_count不一致")

    seed_everything(seed)
    inventory_model = build_model("meta_gnn", feature_count, cfg).cpu()
    inventory = parameter_inventory(inventory_model, 0.0)
    del inventory_model

    states_by_source: dict[str, dict] = {"ordinary": deepcopy(ordinary_state)}
    histories_by_source: dict[str, list] = {"ordinary": []}
    diagnostics_by_source: dict[str, dict] = {
        "ordinary": {
            "seed": seed,
            "source_key": "ordinary",
            "source_pretrain_steps": args.source_pretrain_steps,
            "stable": True,
        }
    }

    for source_key in source_keys:
        if source_key == "ordinary":
            continue
        cache = source_cache_path(args, source_key, seed)
        signature = source_signature(
            args, cfg, source_key, feature_count, extra_steps
        )
        payload = load_cached_source(cache, signature) if args.resume else None
        if payload is not None:
            states_by_source[source_key] = payload["state"]
            histories_by_source[source_key] = payload.get("history", [])
            diagnostics_by_source[source_key] = payload.get("diagnostic", {})
            continue

        source_tasks, current_feature_count = fresh_source_tasks(
            args, cfg, protocol, seed
        )
        if current_feature_count != feature_count:
            raise AssertionError("同一协议下feature_count发生变化")

        if source_key == "ordinary_budget":
            model = build_model("meta_gnn", feature_count, cfg)
            model.load_state_dict(ordinary_state)
            continuation_cfg = dict(cfg)
            continuation_cfg["source_pretrain_steps"] = extra_steps
            continuation_cfg["seed"] = seed + 50000
            seed_everything(seed + 50000)
            model, history = train_source_supervised(
                model, source_tasks, continuation_cfg, device
            )
            state = cpu_state(model)
            diagnostic = {
                "seed": seed,
                "source_key": source_key,
                "stable": True,
                "initial_pretrain_steps": args.source_pretrain_steps,
                "extra_supervised_steps": extra_steps,
                "total_supervised_steps": args.source_pretrain_steps + extra_steps,
                "matching_note": (
                    "Matched to ANIL by source loss-gradient batch count; optimizer "
                    "updates remain algorithmically different."
                ),
            }
        elif source_key == "reptile":
            seed_everything(seed)
            model = build_model("meta_gnn", feature_count, cfg).to(device)
            model = train_source_meta(model, source_tasks, cfg, device)
            state = cpu_state(model)
            history = []
            diagnostic = {
                "seed": seed,
                "source_key": source_key,
                "stable": True,
                "meta_epochs": args.meta_epochs,
                "inner_steps": args.meta_inner_steps,
                "inner_lr": args.meta_inner_lr,
                "outer_lr": args.outer_lr,
            }
        elif source_key in {"anil_raw_mse", "anil_raw_huber"}:
            loss_mode = "raw_mse" if source_key.endswith("mse") else "raw_huber"
            model = build_model("meta_gnn", feature_count, cfg)
            model.load_state_dict(ordinary_state)
            run_id = f"experiment10c_seed{seed}_{source_key}"
            seed_everything(seed)
            diagnostic, history, state, _ = train_safe_anil(
                model,
                source_tasks,
                cfg,
                args,
                inner_lr=args.meta_inner_lr,
                inner_steps=args.meta_inner_steps,
                loss_mode=loss_mode,
                clip_norm=args.meta_clip_norm,
                config_id=run_id,
            )
            if not diagnostic.get("stable") or state is None:
                raise RuntimeError(
                    f"{source_key}在seed={seed}源域训练不稳定："
                    f"{diagnostic.get('failure_message')}"
                )
        else:
            raise ValueError(f"未知source_key：{source_key}")

        if not all_tensors_finite(state.values()):
            raise RuntimeError(f"{source_key}源模型包含NaN/Inf")
        states_by_source[source_key] = state
        histories_by_source[source_key] = history
        diagnostics_by_source[source_key] = diagnostic
        save_source_cache(cache, signature, state, history, diagnostic)
        del model, source_tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    states = {
        regime: deepcopy(states_by_source[regime_spec(regime)["source"]])
        for regime in required_regimes
    }
    histories = {
        regime: deepcopy(histories_by_source[regime_spec(regime)["source"]])
        for regime in required_regimes
    }
    return states, histories, diagnostics_by_source, inventory


def set_target_scope(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = scope == "full" or name.startswith("predictor.")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError(f"目标域scope={scope}没有可训练参数")
    return trainable


def set_target_mode(model: torch.nn.Module, scope: str) -> None:
    if scope == "full":
        model.train()
    else:
        model.eval()
        model.predictor.train()


def target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    huber_delta: float,
) -> torch.Tensor:
    if mode == "raw_mse":
        return torch.nn.functional.mse_loss(prediction, target)
    if mode == "raw_huber":
        return torch.nn.functional.huber_loss(
            prediction, target, delta=huber_delta
        )
    raise ValueError(f"未知目标损失：{mode}")


def tensor_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        value = float(torch.linalg.vector_norm(tensor.detach().float()).cpu().item())
        if not math.isfinite(value):
            return float("inf")
        total += value * value
    return math.sqrt(total)


def train_target(
    model: torch.nn.Module,
    support,
    validation,
    args: argparse.Namespace,
    device: torch.device,
    *,
    scope: str,
    loss_mode: str,
) -> tuple[torch.nn.Module, list[dict], int, int, dict, dict]:
    learner = deepcopy(model).to(device)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in learner.named_parameters()
    }
    trainable = set_target_scope(learner, scope)
    optimizer = torch.optim.Adam(
        trainable,
        lr=args.target_lr,
        weight_decay=args.target_weight_decay,
    )
    best_state = deepcopy(learner.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []
    max_grad_norm = 0.0
    clip_events = 0
    gradient_steps = 0

    for epoch in range(1, args.target_epochs + 1):
        set_target_mode(learner, scope)
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = learner(x)
            loss = target_loss(prediction, y, loss_mode, args.huber_delta)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"目标域loss非有限：epoch={epoch}, loss={loss.item()}"
                )
            loss.backward()
            gradients = [p.grad for p in trainable if p.grad is not None]
            grad_norm = tensor_norm(gradients)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"目标域梯度非有限：epoch={epoch}")
            max_grad_norm = max(max_grad_norm, grad_norm)
            gradient_steps += 1
            if args.target_clip_norm > 0:
                if grad_norm > args.target_clip_norm:
                    clip_events += 1
                torch.nn.utils.clip_grad_norm_(trainable, args.target_clip_norm)
            optimizer.step()
            if not all_tensors_finite(learner.parameters()):
                raise FloatingPointError(f"目标域参数非有限：epoch={epoch}")
            losses.append(float(loss.detach().cpu().item()))

        validation_metrics = evaluate(learner, validation, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(row)
        print(
            f"target_epoch={epoch:03d}/{args.target_epochs} "
            f"scope={scope} loss={loss_mode} train_loss={row['train_loss']:.4f} "
            f"val_rmse={validation_metrics['rmse']:.4f}"
        )
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_state = deepcopy(learner.state_dict())

    learner.load_state_dict(best_state)
    drift = parameter_drift(before, learner)
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    diagnostics = {
        "target_max_grad_norm": max_grad_norm,
        "target_gradient_steps": gradient_steps,
        "target_clip_events": clip_events,
        "target_clip_rate": clip_events / max(1, gradient_steps),
    }
    return learner, history, best_epoch, trainable_count, drift, diagnostics


def run_target_regime(
    args: argparse.Namespace,
    regime: str,
    cfg: dict,
    loaders,
    source_state: dict,
    source_history: list,
    inventory: dict,
    k: int,
) -> dict:
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    _, support, validation, test, feature_count, split_info = loaders
    spec = regime_spec(regime)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    model, target_history, best_epoch, trainable_count, drift, target_diag = (
        train_target(
            model,
            support,
            validation,
            args,
            device,
            scope=spec["target_scope"],
            loss_mode=spec["target_loss"],
        )
    )
    validation_metrics = evaluate(model, validation, device)
    # 只在验证集已经选定best epoch之后评估一次官方测试集。
    test_metrics = evaluate(model, test, device)
    result = {
        **test_metrics,
        "regime": regime,
        "model": "meta_gnn_rul",
        "source_training": spec["source_training"],
        "source_state_key": spec["source"],
        "target_adaptation_scope": spec["target_scope"],
        "target_loss_mode": spec["target_loss"],
        "experiment": f"experiment10c_{regime}_k{k}",
        "target_domain": cfg["target_domain"],
        "seed": cfg["seed"],
        "k": k,
        "adaptation_engine_count": k,
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "best_target_epoch_by_validation": best_epoch,
        "target_epochs_planned": args.target_epochs,
        "target_learning_rate": args.target_lr,
        "target_clip_norm": args.target_clip_norm,
        "target_trainable_parameter_count": trainable_count,
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_fraction": trainable_count
        / inventory["total_parameter_count"],
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "meta_epochs": args.meta_epochs,
        "meta_clip_norm": args.meta_clip_norm,
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "parameter_drift_by_group": drift,
        **target_diag,
    }

    output = result_paths(args)["output"] / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / (
        f"experiment10c_{regime}_k{k}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
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


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    groups = [
        "k",
        "regime",
        "source_training",
        "target_adaptation_scope",
        "target_loss_mode",
    ]
    summary = frame.groupby(groups, as_index=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(columns={"rmse_count": "n_runs"})
    for metric in METRICS:
        column = f"{metric}_std"
        if column in summary:
            summary[column] = summary[column].fillna(0.0)
    redundant = [
        f"{metric}_count"
        for metric in METRICS
        if metric != "rmse" and f"{metric}_count" in summary
    ]
    summary = summary.drop(columns=redundant)
    return summary.sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def paired_comparisons(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    paired_rows: list[dict] = []
    summary_rows: list[dict] = []
    for candidate, reference, label in COMPARISONS:
        for k in sorted(frame.k.unique()):
            left = frame[(frame.k == k) & (frame.regime == candidate)]
            right = frame[(frame.k == k) & (frame.regime == reference)]
            merged = left.merge(right, on=["seed", "k"], suffixes=("_candidate", "_reference"))
            if merged.empty:
                continue
            for _, row in merged.iterrows():
                item = {
                    "k": int(k),
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "seed": int(row.seed),
                }
                for metric in METRICS:
                    item[f"{metric}_delta"] = float(
                        row[f"{metric}_candidate"] - row[f"{metric}_reference"]
                    )
                paired_rows.append(item)

            subset = [row for row in paired_rows if row["k"] == int(k) and row["comparison"] == label]
            result = {
                "k": int(k),
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "n_pairs": len(subset),
            }
            for metric in METRICS:
                values = np.asarray([row[f"{metric}_delta"] for row in subset], dtype=float)
                lower_is_better = metric != "r2"
                wins = values < 0 if lower_is_better else values > 0
                result[f"{metric}_delta_mean"] = float(values.mean())
                result[f"{metric}_win_rate"] = float(wins.mean())
                result[f"{metric}_paired_p"] = (
                    float(stats.ttest_1samp(values, 0.0).pvalue)
                    if len(values) >= 2 and not np.allclose(values, values[0])
                    else float("nan")
                )
            summary_rows.append(result)

    paired = pd.DataFrame(paired_rows)
    comparisons = pd.DataFrame(summary_rows)
    if not paired.empty:
        paired = paired.sort_values(["k", "comparison", "seed"])
    if not comparisons.empty:
        comparisons = comparisons.sort_values(["k", "comparison"])
    return paired, comparisons


def save_progress(results: list[dict], args: argparse.Namespace) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    summary = summarize(results)
    atomic_write_text(
        paths["summary"], summary.to_csv(index=False), encoding="utf-8-sig"
    )
    paired, comparisons = paired_comparisons(results)
    atomic_write_text(
        paths["paired"], paired.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False), encoding="utf-8-sig"
    )
    return paths


def completed_keys(results: list[dict]) -> set[tuple[int, int, str]]:
    return {
        (int(row["seed"]), int(row["k"]), str(row["regime"]))
        for row in results
    }


def write_protocol_files(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    source_protocol_path: Path | None,
    k_values: list[int],
    seeds: list[int],
    regimes: list[str],
) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied = dict(protocol)
    copied["experiment10c_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path is not None else "regenerated"
    )
    atomic_write_text(
        paths["protocol"], json.dumps(copied, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )
    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    extra_steps = budget_extra_steps(args, cfg)
    budget = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "k_values": k_values,
        "seeds": seeds,
        "regimes": regimes,
        "source_domains": cfg["source_domains"],
        "ordinary_pretraining_steps": args.source_pretrain_steps,
        "ordinary_budget_extra_steps": extra_steps,
        "ordinary_budget_total_steps": args.source_pretrain_steps + extra_steps,
        "meta_epochs": args.meta_epochs,
        "tasks_per_meta_batch": task_count,
        "meta_inner_steps": args.meta_inner_steps,
        "meta_inner_lr": args.meta_inner_lr,
        "anil_query_batches": args.anil_query_batches,
        "anil_inner_loss_gradient_batches": (
            args.meta_epochs * task_count * args.meta_inner_steps
        ),
        "anil_query_loss_gradient_batches": (
            args.meta_epochs * task_count * args.anil_query_batches
        ),
        "anil_outer_optimizer_steps": args.meta_epochs,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_task_mode": args.anil_task_mode,
        "meta_clip_norm": args.meta_clip_norm,
        "target_epochs_equal_for_all_regimes": args.target_epochs,
        "target_lr_equal_for_all_regimes": args.target_lr,
        "target_clip_norm_equal_for_all_regimes": args.target_clip_norm,
        "selection_rule": "best target epoch is selected only by fixed validation engines",
        "official_test_rule": "official test is evaluated after validation selection and never used for tuning",
        "budget_matching_note": (
            "Ordinary budget continuation matches the number of ANIL source "
            "support/query loss-gradient batches, not optimizer dynamics or wall-clock time."
        ),
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))
    return paths


def inspect_protocol(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    k: int,
) -> dict:
    units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks, support, validation, test, feature_count, split_info = loaders
    x, y = next(iter(source_tasks[cfg["source_domains"][0]]))
    seed_everything(seed)
    model = build_model("meta_gnn", feature_count, cfg).cpu()
    with torch.no_grad():
        output = model(x[: min(8, len(x))])
    diagnostic = {
        "seed": seed,
        "k": k,
        "feature_count": feature_count,
        "source_example_shape": list(x.shape),
        "source_label_shape": list(y.shape),
        "forward_output_shape": list(output.shape),
        "support_engine_count": len(units),
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "support_batches": len(support),
        "validation_batches": len(validation),
        "test_batches": len(test),
        "adaptation_units": units,
        **parameter_inventory(model, 0.0),
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return diagnostic


def main() -> None:
    args = parse_args()
    k_values, seeds, regimes = validate_args(args)
    args.stage = "confirm"
    args.task_mode = args.anil_task_mode

    first_cfg = load_config(args, seeds[0])
    protocol, source_protocol_path = load_or_create_protocol(
        args, first_cfg, seeds, k_values
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

    paths = write_protocol_files(
        args,
        first_cfg,
        protocol,
        source_protocol_path,
        k_values,
        seeds,
        regimes,
    )
    print("\n[实验10C固定协议与预算]")
    print(paths["budget"].read_text(encoding="utf-8"))

    if args.dry_run:
        diagnostics = [
            inspect_protocol(args, first_cfg, protocol, seeds[0], k)
            for k in k_values
        ]
        atomic_write_text(
            paths["parameters"],
            json.dumps(diagnostics[0], ensure_ascii=False, indent=2),
        )
        print("\n[dry-run完成] 未训练模型。")
        print(
            f"Protocol: {paths['protocol']}\nSplits: {paths['splits']}"
            f"\nBudget: {paths['budget']}\nParameters: {paths['parameters']}"
        )
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条目标域结果。")
    done = completed_keys(results)
    all_source_diagnostics: dict[str, dict] = {}
    if args.resume and paths["source_diagnostics"].is_file():
        all_source_diagnostics = json.loads(
            paths["source_diagnostics"].read_text(encoding="utf-8")
        )

    for seed in seeds:
        cfg = load_config(args, seed)
        pending = [
            (k, regime)
            for k in k_values
            for regime in regimes
            if (seed, k, regime) not in done
        ]
        if not pending:
            print(f"[skip seed] seed={seed}已全部完成。")
            continue
        required = list(dict.fromkeys(regime for _, regime in pending))
        print(f"\n[source initialization] seed={seed} regimes={required}")
        states, histories, source_diagnostics, inventory = build_source_states(
            args, cfg, protocol, seed, required
        )
        all_source_diagnostics[str(seed)] = source_diagnostics
        atomic_write_text(
            paths["source_diagnostics"],
            json.dumps(all_source_diagnostics, ensure_ascii=False, indent=2),
        )
        atomic_write_text(
            paths["parameters"], json.dumps(inventory, ensure_ascii=False, indent=2)
        )

        for k in k_values:
            units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
            for regime in regimes:
                key = (seed, k, regime)
                if key in done:
                    print(f"[skip] seed={seed} K={k} regime={regime}")
                    continue
                print(
                    f"\n[experiment10C] seed={seed} K={k} regime={regime} "
                    f"engines={units}"
                )
                # 每个方案重新创建同种子加载器，保证目标批次序列可比。
                seed_everything(seed)
                loaders = prepare_kshot_experiment(
                    cfg,
                    args.preprocessing,
                    args.balance_mode,
                    protocol["validation_units"],
                    units,
                )
                split_info = loaders[-1]
                if split_info["official_test_units_hash"] != protocol[
                    "official_test_units_hash"
                ]:
                    raise AssertionError("不同运行使用了不同官方测试发动机")
                result = run_target_regime(
                    args,
                    regime,
                    cfg,
                    loaders,
                    states[regime],
                    histories[regime],
                    inventory,
                    k,
                )
                results.append(result)
                done.add(key)
                paths = save_progress(results, args)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del states, histories
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(results)
    _, comparisons = paired_comparisons(results)
    print("\n[实验10C汇总]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        print("\n[配对比较：RMSE/MAE/NASA delta<0，R2 delta>0表示候选更好]")
        print(comparisons.to_string(index=False))
    print("\n[正式判定规则]")
    print("1. ANIL必须优于同损失的pretrained_head，才能说明存在初步元学习收益。")
    print("2. ANIL还必须优于pretrained_budget_head，才能排除仅由额外源训练预算造成的提升。")
    print("3. 重点看K=2和K=5的五种子配对胜率、RMSE和NASA Score；不能只看单个seed。")
    print("4. Huber若只降低方差但平均RMSE略高，应报告为稳定性收益，而非精度收益。")
    print("5. 本实验已评估官方测试集，运行结束后不得据此继续调参。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
        f"\nBudget: {paths['budget']}\nSource diagnostics: {paths['source_diagnostics']}"
        f"\nParameters: {paths['parameters']}"
    )


if __name__ == "__main__":
    main()
