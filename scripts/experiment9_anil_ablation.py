"""实验9：ANIL与预测头适应机制消融实验。

本文件是独立实验入口，不替换 ``main.py``、实验7、实验8或模型文件。
它复用实验7固定发动机K-shot协议，在完全相同的目标发动机、固定验证集、
官方测试集、预处理和目标训练预算下比较：

``pretrained_full``
    普通多源监督预训练，目标域更新全部参数（实验8最强基线）。

``pretrained_head``
    普通多源监督预训练，目标域只更新RUL预测头/辅助预测头。

``reptile_full``
    Reptile源域元训练，目标域更新全部参数（实验8的Meta-GNN）。

``anil_head``
    ANIL源域元训练：任务内循环只更新预测头，查询损失的外循环更新整网；
    目标域也只更新预测头。默认使用一阶ANIL近似，可用
    ``--anil-order second`` 运行二阶ANIL。

推荐在项目根目录运行：

    python scripts/experiment9_anil_ablation.py \
        --target FD004 \
        --k-values 2 5 10 20 \
        --seeds 42 43 44 45 46 \
        --regimes pretrained_full pretrained_head reptile_full anil_head \
        --preprocessing condition_settings \
        --balance-mode engine_stage \
        --meta-epochs 100 \
        --target-epochs 10 \
        --inner-steps 5 \
        --inner-lr 0.001 \
        --outer-lr 0.05 \
        --anil-meta-lr 0.0001 \
        --anil-order first \
        --pair-aux-weight 0.01 \
        --resume

只检查协议、模型形状和参数范围，不训练：

    python scripts/experiment9_anil_ablation.py --target FD004 --dry-run

说明
----
1. 归一化器只在源域训练数据上拟合；
2. K台目标发动机在同一种子内严格嵌套；
3. 所有方法共用固定验证发动机和官方目标测试发动机；
4. 测试集仅用于最终报告，不能据此调参；
5. ANIL比Reptile多一次每任务查询梯度，这是算法定义所需，预算会单独记录。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402
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


REGIMES = (
    "pretrained_full",
    "pretrained_head",
    "reptile_full",
    "anil_head",
)
HEAD_PREFIXES = ("predictor.", "pairwise_predictor.")
COMPARISONS = (
    (
        "pretrained_head",
        "pretrained_full",
        "head_only_vs_full_after_ordinary_pretraining",
    ),
    ("anil_head", "reptile_full", "anil_head_vs_reptile_full"),
    (
        "anil_head",
        "pretrained_head",
        "anil_vs_ordinary_pretraining_with_head_only_target_adaptation",
    ),
    (
        "anil_head",
        "pretrained_full",
        "anil_vs_experiment8_strong_baseline",
    ),
    (
        "reptile_full",
        "pretrained_full",
        "reptile_vs_ordinary_pretraining_reference",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验9：ANIL、预测头适应、Reptile与普通预训练消融"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target",
        default="FD004",
        choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES),
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=list(REGIMES),
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
    )
    parser.add_argument(
        "--protocol-file",
        help=(
            "实验7的split_protocol JSON；默认读取"
            "outputs/experiment7_kshot_engines/experiment7_split_protocol_<target>.json"
        ),
    )
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default="condition_settings",
    )
    parser.add_argument(
        "--balance-mode", choices=BALANCE_MODES, default="engine_stage"
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--target-epochs", type=int)
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument(
        "--outer-lr",
        type=float,
        help="Reptile外循环插值率；与ANIL的Adam学习率不是同一含义",
    )
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument("--source-pretrain-steps", type=int)
    parser.add_argument("--source-pretrain-lr", type=float)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--anil-meta-lr",
        type=float,
        default=1e-4,
        help="ANIL查询损失外循环Adam学习率，应只用固定验证集选择",
    )
    parser.add_argument(
        "--anil-order",
        choices=("first", "second"),
        default="first",
        help="first为一阶近似（推荐先运行），second保留二阶梯度但显存更高",
    )
    parser.add_argument(
        "--anil-query-batches",
        type=int,
        default=1,
        help="每个源任务每个meta epoch使用的查询批次数",
    )
    parser.add_argument(
        "--output-dir", default="outputs/experiment9_anil_ablation"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="仅用于mock数据冒烟测试；正式实验不要使用",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain for domain in EXPECTED_OFFICIAL_TEST_ENGINES if domain != args.target
    ]
    cfg["condition_count"] = args.condition_count
    cfg["normalizer_seed"] = args.normalizer_seed

    if args.meta_epochs is not None:
        cfg["meta_epochs"] = args.meta_epochs
    if args.target_epochs is not None:
        cfg["target_epochs"] = args.target_epochs
    else:
        cfg["target_epochs"] = cfg["adapt_epochs"]
    if args.inner_steps is not None:
        cfg["inner_steps"] = args.inner_steps
    if args.inner_lr is not None:
        cfg["inner_lr"] = args.inner_lr
    if args.outer_lr is not None:
        cfg["outer_lr"] = args.outer_lr
    if args.pair_aux_weight is not None:
        cfg["pair_aux_weight"] = args.pair_aux_weight

    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    cfg["source_pretrain_steps"] = (
        args.source_pretrain_steps
        if args.source_pretrain_steps is not None
        else cfg["meta_epochs"] * task_count * cfg["inner_steps"]
    )
    cfg["source_pretrain_lr"] = (
        args.source_pretrain_lr
        if args.source_pretrain_lr is not None
        else cfg["inner_lr"]
    )
    cfg["source_pretrain_weight_decay"] = args.source_pretrain_weight_decay
    cfg["anil_meta_lr"] = args.anil_meta_lr
    cfg["anil_order"] = args.anil_order
    cfg["anil_query_batches"] = args.anil_query_batches

    if cfg["meta_epochs"] <= 0 or cfg["target_epochs"] <= 0:
        raise ValueError("meta_epochs和target_epochs必须为正整数")
    if cfg["inner_steps"] <= 0 or cfg["anil_query_batches"] <= 0:
        raise ValueError("inner_steps和anil_query_batches必须为正整数")
    if cfg["source_pretrain_steps"] <= 0:
        raise ValueError("source_pretrain_steps必须为正整数")

    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir, PROJECT_ROOT))
    cfg["output_dir"] = str(resolve_path(args.output_dir, PROJECT_ROOT))
    return cfg


def is_head_parameter(name: str, pair_aux_weight: float) -> bool:
    if name.startswith("predictor."):
        return True
    return pair_aux_weight > 0 and name.startswith("pairwise_predictor.")


def parameter_group(name: str) -> str:
    for prefix, group in (
        ("se_block.", "sensor_se"),
        ("temporal.", "lstm"),
        ("gat.", "gat"),
        ("sensor_attention.", "self_attention"),
        ("predictor.", "rul_head"),
        ("pairwise_predictor.", "pairwise_head"),
    ):
        if name.startswith(prefix):
            return group
    return "other"


def parameter_inventory(model: torch.nn.Module, pair_aux_weight: float) -> dict:
    groups: dict[str, int] = defaultdict(int)
    head_count = 0
    total = 0
    head_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        groups[parameter_group(name)] += count
        if is_head_parameter(name, pair_aux_weight):
            head_count += count
            head_names.append(name)
    return {
        "total_parameter_count": total,
        "head_parameter_count": head_count,
        "full_trainable_fraction": 1.0,
        "head_trainable_fraction": head_count / total,
        "parameter_count_by_group": dict(groups),
        "head_parameter_names": head_names,
    }


def next_batch(
    task_name: str,
    loaders: dict[str, torch.utils.data.DataLoader],
    iterators: dict[str, Iterable],
):
    iterator = iterators[task_name]
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loaders[task_name])
        iterators[task_name] = iterator
        batch = next(iterator)
    return batch


def functional_rul_loss(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    pair_aux_weight: float,
) -> torch.Tensor:
    """Evaluate the normal project loss with a functional parameter state."""
    if pair_aux_weight > 0 and x.size(0) > 1:
        prediction, auxiliary = functional_call(
            model,
            state,
            (x,),
            {"return_attention": True},
        )
        features = auxiliary["features"]
        mate = torch.roll(torch.arange(x.size(0), device=x.device), 1)
        pair_input = torch.cat([features, features[mate]], dim=-1)
        prefix = "pairwise_predictor."
        pair_state = {
            name[len(prefix) :]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        pair_prediction = functional_call(
            model.pairwise_predictor,
            pair_state,
            (pair_input,),
        ).squeeze(-1)
        pair_target = torch.abs(y - y[mate])
        return torch.nn.functional.mse_loss(
            prediction, y
        ) + pair_aux_weight * torch.nn.functional.mse_loss(
            pair_prediction, pair_target
        )
    prediction = functional_call(model, state, (x,))
    return torch.nn.functional.mse_loss(prediction, y)


def train_source_anil(
    model: torch.nn.Module,
    source_tasks: dict[str, torch.utils.data.DataLoader],
    cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict]]:
    """ANIL source training with head-only inner updates and query outer loss.

    ``first`` detaches inner gradients (first-order ANIL/FOMAML approximation).
    ``second`` keeps the inner gradient graph and is closer to original ANIL,
    but requires substantially more memory.
    """
    meta_model = deepcopy(model).to(device)
    meta_model.train()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=cfg["anil_meta_lr"])
    task_names = sorted(source_tasks)
    task_count = min(cfg["tasks_per_meta_batch"], len(task_names))
    iterators = {name: iter(source_tasks[name]) for name in task_names}
    head_names = [
        name
        for name, _ in meta_model.named_parameters()
        if is_head_parameter(name, cfg.get("pair_aux_weight", 0.0))
    ]
    if not head_names:
        raise RuntimeError("未找到ANIL预测头参数，请检查模型中的predictor命名")

    history: list[dict] = []
    report_every = 5
    use_second_order = cfg["anil_order"] == "second"
    for epoch in range(1, cfg["meta_epochs"] + 1):
        selected = random.sample(task_names, task_count)
        task_query_losses: list[torch.Tensor] = []

        for task_name in selected:
            parameters = dict(meta_model.named_parameters())
            buffers = dict(meta_model.named_buffers())
            fast_head = {name: parameters[name] for name in head_names}

            for _ in range(cfg["inner_steps"]):
                support_x, support_y = next_batch(
                    task_name, source_tasks, iterators
                )
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                support_state = {**buffers, **parameters, **fast_head}
                support_loss = functional_rul_loss(
                    meta_model,
                    support_state,
                    support_x,
                    support_y,
                    cfg.get("pair_aux_weight", 0.0),
                )
                gradients = torch.autograd.grad(
                    support_loss,
                    tuple(fast_head.values()),
                    create_graph=use_second_order,
                    allow_unused=True,
                )
                updated_head: dict[str, torch.Tensor] = {}
                for (name, parameter), gradient in zip(
                    fast_head.items(), gradients
                ):
                    if gradient is None:
                        gradient = torch.zeros_like(parameter)
                    if not use_second_order:
                        gradient = gradient.detach()
                    updated_head[name] = parameter - cfg["inner_lr"] * gradient
                fast_head = updated_head

            query_losses: list[torch.Tensor] = []
            for _ in range(cfg["anil_query_batches"]):
                query_x, query_y = next_batch(task_name, source_tasks, iterators)
                query_x = query_x.to(device)
                query_y = query_y.to(device)
                query_state = {**buffers, **parameters, **fast_head}
                query_losses.append(
                    functional_rul_loss(
                        meta_model,
                        query_state,
                        query_x,
                        query_y,
                        cfg.get("pair_aux_weight", 0.0),
                    )
                )
            task_query_losses.append(torch.stack(query_losses).mean())

        meta_loss = torch.stack(task_query_losses).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 5.0)
        optimizer.step()

        if epoch % report_every == 0 or epoch == cfg["meta_epochs"]:
            row = {
                "meta_epoch": epoch,
                "query_loss": float(meta_loss.detach().cpu().item()),
                "anil_order": cfg["anil_order"],
                "tasks": selected,
            }
            history.append(row)
            print(
                f"anil_meta_epoch={epoch:03d}/{cfg['meta_epochs']} "
                f"query_loss={row['query_loss']:.4f} order={cfg['anil_order']}"
            )
    return meta_model, history


def set_target_scope(
    model: torch.nn.Module,
    scope: str,
    pair_aux_weight: float,
) -> list[torch.nn.Parameter]:
    if scope not in {"full", "head"}:
        raise ValueError(f"未知目标适应范围：{scope}")
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = scope == "full" or is_head_parameter(name, pair_aux_weight)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("目标域没有可训练参数")
    return trainable


def set_training_mode(model: torch.nn.Module, scope: str) -> None:
    if scope == "full":
        model.train()
        return
    # 冻结的特征提取器保持eval，避免其Dropout在少样本头部适应时漂移。
    model.eval()
    model.predictor.train()
    if hasattr(model, "pairwise_predictor"):
        model.pairwise_predictor.train()


def parameter_drift(
    before: dict[str, torch.Tensor],
    after_model: torch.nn.Module,
) -> dict[str, dict[str, float]]:
    diff_sq: dict[str, float] = defaultdict(float)
    base_sq: dict[str, float] = defaultdict(float)
    after_state = dict(after_model.named_parameters())
    for name, before_tensor in before.items():
        if name not in after_state:
            continue
        group = parameter_group(name)
        before_cpu = before_tensor.detach().cpu().float()
        after_cpu = after_state[name].detach().cpu().float()
        diff_sq[group] += float(torch.sum((after_cpu - before_cpu) ** 2).item())
        base_sq[group] += float(torch.sum(before_cpu**2).item())
    result: dict[str, dict[str, float]] = {}
    for group in sorted(diff_sq):
        absolute = float(np.sqrt(diff_sq[group]))
        base_norm = float(np.sqrt(base_sq[group]))
        result[group] = {
            "l2_change": absolute,
            "relative_l2_change": absolute / max(base_norm, 1e-12),
        }
    return result


def train_target_with_scope(
    model: torch.nn.Module,
    support: torch.utils.data.DataLoader,
    validation: torch.utils.data.DataLoader,
    cfg: dict,
    device: torch.device,
    scope: str,
) -> tuple[torch.nn.Module, list[dict], int, int, dict]:
    learner = deepcopy(model).to(device)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in learner.named_parameters()
    }
    trainable = set_target_scope(
        learner, scope, cfg.get("pair_aux_weight", 0.0)
    )
    optimizer = torch.optim.Adam(trainable, lr=cfg["inner_lr"])
    best_state = deepcopy(learner.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, cfg["target_epochs"] + 1):
        set_training_mode(learner, scope)
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, _ = rul_training_loss(
                learner,
                x,
                y,
                cfg.get("pair_aux_weight", 0.0),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

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
            f"target_epoch={epoch:03d}/{cfg['target_epochs']} "
            f"scope={scope} train_loss={row['train_loss']:.4f} "
            f"val_rmse={validation_metrics['rmse']:.4f}"
        )
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_state = deepcopy(learner.state_dict())

    learner.load_state_dict(best_state)
    drift = parameter_drift(before, learner)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    return learner, history, best_epoch, int(trainable_count), drift


def build_source_initializations(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    regimes: list[str],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[dict]], int, dict]:
    first_k = min(int(value) for value in protocol["k_values"])
    first_units = protocol["nested_adaptation_units_by_seed"][str(seed)][
        str(first_k)
    ]
    shape_loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        first_units,
    )
    feature_count = shape_loaders[4]
    seed_everything(seed)
    base_model = build_model("gnn", feature_count, cfg).cpu()
    base_state = deepcopy(base_model.state_dict())
    inventory = parameter_inventory(
        base_model, cfg.get("pair_aux_weight", 0.0)
    )
    del shape_loaders, base_model

    states: dict[str, dict[str, torch.Tensor]] = {}
    histories: dict[str, list[dict]] = {}
    device = resolve_device(cfg["device"])

    ordinary_regimes = {"pretrained_full", "pretrained_head"} & set(regimes)
    if ordinary_regimes:
        loaders = prepare_kshot_experiment(
            cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            first_units,
        )
        model = build_model("gnn", feature_count, cfg)
        model.load_state_dict(base_state)
        seed_everything(seed)
        model, history = train_source_supervised(
            model, loaders[0], cfg, device
        )
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        for regime in ordinary_regimes:
            states[regime] = deepcopy(state)
            histories[regime] = deepcopy(history)
        del loaders, model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "reptile_full" in regimes:
        loaders = prepare_kshot_experiment(
            cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            first_units,
        )
        model = build_model("meta_gnn", feature_count, cfg)
        model.load_state_dict(base_state)
        model = model.to(device)  # 避免Reptile元参数与任务副本设备不一致。
        seed_everything(seed)
        model = train_source_meta(model, loaders[0], cfg, device)
        states["reptile_full"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        histories["reptile_full"] = [
            {
                "meta_epochs": cfg["meta_epochs"],
                "tasks_per_meta_batch": min(
                    cfg["tasks_per_meta_batch"], len(cfg["source_domains"])
                ),
                "inner_steps": cfg["inner_steps"],
            }
        ]
        del loaders, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "anil_head" in regimes:
        loaders = prepare_kshot_experiment(
            cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            first_units,
        )
        model = build_model("meta_gnn", feature_count, cfg)
        model.load_state_dict(base_state)
        seed_everything(seed)
        model, history = train_source_anil(model, loaders[0], cfg, device)
        states["anil_head"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        histories["anil_head"] = history
        del loaders, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return states, histories, feature_count, inventory


def regime_description(regime: str) -> tuple[str, str]:
    return {
        "pretrained_full": (
            "ordinary_multisource_supervised_pretraining",
            "full",
        ),
        "pretrained_head": (
            "ordinary_multisource_supervised_pretraining",
            "head",
        ),
        "reptile_full": ("reptile_meta_training", "full"),
        "anil_head": ("anil_meta_training", "head"),
    }[regime]


def run_target_regime(
    args: argparse.Namespace,
    regime: str,
    cfg: dict,
    loaders,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    k: int,
) -> dict:
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    _, support, validation, test, feature_count, split_info = loaders
    source_training, scope = regime_description(regime)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    model, target_history, best_epoch, trainable_count, drift = (
        train_target_with_scope(
            model,
            support,
            validation,
            cfg,
            device,
            scope,
        )
    )
    validation_metrics = evaluate(model, validation, device)
    test_metrics = evaluate(model, test, device)

    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    source_budget = 0
    query_budget = 0
    if source_training == "ordinary_multisource_supervised_pretraining":
        source_budget = cfg["source_pretrain_steps"]
    elif source_training == "reptile_meta_training":
        source_budget = cfg["meta_epochs"] * task_count * cfg["inner_steps"]
    else:
        source_budget = cfg["meta_epochs"] * task_count * cfg["inner_steps"]
        query_budget = (
            cfg["meta_epochs"] * task_count * cfg["anil_query_batches"]
        )

    result = {
        **test_metrics,
        "regime": regime,
        "model": "meta_gnn_rul",
        "source_training": source_training,
        "target_adaptation_scope": scope,
        "experiment": f"experiment9_{regime}_k{k}",
        "target_domain": cfg["target_domain"],
        "seed": cfg["seed"],
        "k": k,
        "adaptation_engine_count": k,
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "target_epochs_completed": cfg["target_epochs"],
        "best_target_epoch_by_validation": best_epoch,
        "target_learning_rate": cfg["inner_lr"],
        "target_trainable_parameter_count": trainable_count,
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_fraction": trainable_count
        / inventory["total_parameter_count"],
        "source_inner_gradient_budget": source_budget,
        "source_query_gradient_budget": query_budget,
        "anil_order": cfg["anil_order"] if regime == "anil_head" else "not_applicable",
        "anil_meta_lr": cfg["anil_meta_lr"] if regime == "anil_head" else 0.0,
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "parameter_drift_by_group": drift,
    }

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / (
        f"experiment9_{regime}_k{k}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
    )
    torch.save(
        {
            "model": model.state_dict(),
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
        "preprocessing_mode",
        "balance_mode",
    ]
    summary = frame.groupby(groups, as_index=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    summary = summary.rename(columns={"rmse_count": "n_runs"})
    summary = summary.drop(
        columns=[
            column
            for column in (f"{metric}_count" for metric in METRICS[1:])
            if column in summary
        ]
    )
    return summary.sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def paired_comparisons(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    for (k, seed), group in frame.groupby(["k", "seed"]):
        by_regime = {row["regime"]: row for _, row in group.iterrows()}
        for candidate, reference, comparison in COMPARISONS:
            if candidate not in by_regime or reference not in by_regime:
                continue
            candidate_row = by_regime[candidate]
            reference_row = by_regime[reference]
            row = {
                "k": int(k),
                "seed": int(seed),
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
            }
            for metric in ("rmse", "mae", "nasa_score"):
                row[f"{metric}_delta_candidate_minus_reference"] = float(
                    candidate_row[metric] - reference_row[metric]
                )
                row[f"candidate_{metric}_win"] = float(
                    candidate_row[metric] < reference_row[metric]
                )
            row["r2_delta_candidate_minus_reference"] = float(
                candidate_row["r2"] - reference_row["r2"]
            )
            row["candidate_r2_win"] = float(
                candidate_row["r2"] > reference_row["r2"]
            )
            rows.append(row)
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired, pd.DataFrame()

    summaries: list[dict] = []
    delta_columns = [
        column for column in paired.columns if "_delta_candidate_minus_reference" in column
    ]
    win_columns = [column for column in paired.columns if column.endswith("_win")]
    for (k, comparison), group in paired.groupby(["k", "comparison"]):
        row = {
            "k": int(k),
            "comparison": comparison,
            "candidate": group.iloc[0]["candidate"],
            "reference": group.iloc[0]["reference"],
            "paired_seed_count": int(len(group)),
        }
        for column in delta_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        for column in win_columns:
            row[f"{column}_rate"] = float(group[column].mean())
        summaries.append(row)
    comparison_summary = pd.DataFrame(summaries).sort_values(
        ["k", "comparison"]
    )
    return paired.sort_values(["k", "comparison", "seed"]), comparison_summary


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    return {
        "output": output,
        "raw": output / f"experiment9_raw_{args.target}.json",
        "summary": output / f"experiment9_summary_{args.target}.csv",
        "paired": output / f"experiment9_paired_by_seed_{args.target}.csv",
        "comparisons": output / f"experiment9_comparisons_{args.target}.csv",
        "protocol": output / f"experiment9_split_protocol_{args.target}.json",
        "splits": output / f"experiment9_engine_splits_{args.target}.csv",
        "budget": output / f"experiment9_budget_{args.target}.json",
        "parameters": output / f"experiment9_parameter_inventory_{args.target}.json",
    }


def save_progress(results: list[dict], args: argparse.Namespace) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    atomic_write_text(
        paths["summary"], summarize(results).to_csv(index=False), encoding="utf-8-sig"
    )
    paired, comparisons = paired_comparisons(results)
    atomic_write_text(paths["paired"], paired.to_csv(index=False), encoding="utf-8-sig")
    atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False), encoding="utf-8-sig"
    )
    return paths


def completed_keys(results: list[dict]) -> set[tuple[int, int, str]]:
    return {
        (int(row["seed"]), int(row["k"]), str(row["regime"]))
        for row in results
    }


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
    tasks, support, validation, test, feature_count, split_info = loaders
    x, _ = next(iter(tasks[cfg["source_domains"][0]]))
    seed_everything(seed)
    model = build_model("gnn", feature_count, cfg).cpu().eval()
    with torch.no_grad():
        output = model(x[: min(8, len(x))])
    inventory = parameter_inventory(model, cfg.get("pair_aux_weight", 0.0))
    diagnostic = {
        "seed": seed,
        "k": k,
        "feature_count": feature_count,
        "source_example_shape": list(x.shape),
        "forward_output_shape": list(output.shape),
        "support_engine_count": len(set(support.dataset.units)),
        "validation_engine_count": len(set(validation.dataset.units)),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "adaptation_units": units,
        **inventory,
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return inventory


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k_values))
    seeds = list(dict.fromkeys(args.seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须是正整数")
    if not seeds:
        raise ValueError("--seeds不能为空")
    if len(seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个随机种子，只能视为预实验。")

    first_cfg = load_config(args, seeds[0])
    protocol, source_protocol_path = load_or_create_protocol(
        args, first_cfg, seeds, k_values
    )
    expected_count = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        int(protocol["official_test_engine_count"]) != expected_count
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试集应为{expected_count}台发动机，"
            f"当前协议为{protocol['official_test_engine_count']}台"
        )

    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied_protocol = dict(protocol)
    copied_protocol["experiment9_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path is not None else "regenerated"
    )
    atomic_write_text(
        paths["protocol"], json.dumps(copied_protocol, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["splits"], protocol_split_frame(protocol).to_csv(index=False), encoding="utf-8-sig"
    )

    task_count = min(
        first_cfg["tasks_per_meta_batch"], len(first_cfg["source_domains"])
    )
    budget = {
        "source_domains": first_cfg["source_domains"],
        "meta_epochs": first_cfg["meta_epochs"],
        "tasks_per_meta_batch": task_count,
        "inner_steps": first_cfg["inner_steps"],
        "ordinary_pretraining_optimizer_steps": first_cfg["source_pretrain_steps"],
        "reptile_inner_gradient_budget": (
            first_cfg["meta_epochs"] * task_count * first_cfg["inner_steps"]
        ),
        "anil_inner_head_gradient_budget": (
            first_cfg["meta_epochs"] * task_count * first_cfg["inner_steps"]
        ),
        "anil_query_outer_gradient_budget": (
            first_cfg["meta_epochs"]
            * task_count
            * first_cfg["anil_query_batches"]
        ),
        "anil_order": first_cfg["anil_order"],
        "anil_meta_lr": first_cfg["anil_meta_lr"],
        "target_epochs_equal_for_all_regimes": first_cfg["target_epochs"],
        "target_lr_equal_for_all_regimes": first_cfg["inner_lr"],
        "note": (
            "Target epochs/batches/lr are matched. ANIL requires query gradients by "
            "definition; they are reported separately from inner-gradient budget."
        ),
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))

    print("\n[实验9固定协议与训练预算]")
    print(
        json.dumps(
            {
                "target": args.target,
                "k_values": k_values,
                "seeds": seeds,
                "regimes": regimes,
                "fixed_validation_units": protocol["validation_units"],
                "official_test_engine_count": protocol["official_test_engine_count"],
                "official_test_units_hash": protocol["official_test_units_hash"],
                "preprocessing": args.preprocessing,
                "balance_mode": args.balance_mode,
                **budget,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run:
        inventory = None
        for k in k_values:
            inventory = inspect_protocol(args, first_cfg, protocol, seeds[0], k)
        if inventory is not None:
            atomic_write_text(
                paths["parameters"],
                json.dumps(inventory, ensure_ascii=False, indent=2),
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
        print(f"[resume] 已读取{len(results)}条结果。")
    done = completed_keys(results)

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
        states, histories, _, inventory = build_source_initializations(
            args, cfg, protocol, seed, required
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
                    f"\n[experiment9] seed={seed} K={k} regime={regime} "
                    f"engines={units}"
                )
                loaders = prepare_kshot_experiment(
                    cfg,
                    args.preprocessing,
                    args.balance_mode,
                    protocol["validation_units"],
                    units,
                )
                if loaders[-1]["official_test_units_hash"] != protocol[
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
    print("\n[experiment9 summary]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        print("\n[配对比较：RMSE/MAE/NASA delta<0、R2 delta>0表示候选更好]")
        print(comparisons.to_string(index=False))
    print("\n[判读顺序]")
    print("1. anil_head优于reptile_full：限制任务内更新可减少少样本过拟合。")
    print("2. anil_head优于pretrained_head：ANIL元训练产生了普通预训练之外的收益。")
    print("3. pretrained_head≈pretrained_full：源域特征可复用，目标域主要校准预测头。")
    print("4. pretrained_full仍最优：下一步应改进元任务构造，而不是继续只调Reptile学习率。")
    print("5. 优先看K=2、5的配对胜率，同时要求NASA Score不能明显恶化。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
        f"\nBudget: {paths['budget']}\nParameters: {paths['parameters']}"
    )


if __name__ == "__main__":
    main()
