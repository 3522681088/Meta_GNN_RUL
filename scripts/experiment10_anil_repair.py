"""实验10：ANIL机制修复与原因定位实验。

本脚本不替换 ``main.py`` 或实验7--9。它复用实验7固定的发动机K-shot
划分，并将实验9中ANIL效果不佳的几个可能原因逐项拆开：

1. 是否需要先做普通多源监督预训练；
2. 内循环是否应该只更新真正的RUL回归头，而不是同时更新pairwise辅助头；
3. 元任务支持集和查询集是否应该按发动机互斥；
4. ANIL外循环优化步数是否不足。

默认比较八组方案：

``pretrained_full``
    普通源域预训练，目标域更新全部参数。

``pretrained_head``
    普通源域预训练，目标域更新RUL头和pairwise辅助头（实验9定义）。

``pretrained_rul_only``
    普通源域预训练，目标域只更新RUL回归头。它是实验10新增的必要对照。

``current_anil``
    随机初始化、一阶ANIL、批次级支持/查询、双预测头，复现实验9设置。

``pretrained_anil_head``
    从普通预训练权重开始ANIL，仍更新双预测头。

``pretrained_anil_rul``
    从普通预训练权重开始ANIL，内循环和目标适应只更新RUL头。

``engine_anil_rul``
    在上一组基础上，使每个源域任务的支持/查询发动机严格互斥。

``budget_anil_rul``
    在上一组基础上增加ANIL外循环步数；默认与普通预训练Adam步数相同。

正式第一阶段建议只跑K=2、5：

    python -u scripts/experiment10_anil_repair.py \
      --target FD004 \
      --k-values 2 5 \
      --seeds 42 43 44 45 46 \
      --regimes pretrained_full pretrained_head pretrained_rul_only \
                current_anil pretrained_anil_head pretrained_anil_rul \
                engine_anil_rul budget_anil_rul \
      --preprocessing condition_settings \
      --balance-mode engine_stage \
      --meta-epochs 100 \
      --target-epochs 10 \
      --inner-steps 5 \
      --inner-lr 0.001 \
      --anil-meta-lr 0.0001 \
      --pair-aux-weight 0.01 \
      --resume

先检查协议、参数范围和源域发动机互斥划分，不训练模型：

    python -u scripts/experiment10_anil_repair.py --target FD004 --dry-run

重要：所有超参数必须依据固定验证发动机选择；官方测试集仅用于最终报告。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml


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
    stable_unit_hash,
)
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    load_or_create_protocol,
    train_source_supervised,
)
from scripts.experiment9_anil_ablation import (  # noqa: E402
    functional_rul_loss,
    parameter_drift,
    parameter_group,
)
from scripts.run_condition_aware_experiment import sampling_weights  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402


SCRIPT_VERSION = "experiment10_anil_repair_v1"
REGIMES = (
    "pretrained_full",
    "pretrained_head",
    "pretrained_rul_only",
    "current_anil",
    "pretrained_anil_head",
    "pretrained_anil_rul",
    "engine_anil_rul",
    "budget_anil_rul",
)

COMPARISONS = (
    ("pretrained_head", "pretrained_full", "ordinary_head_vs_full"),
    ("pretrained_rul_only", "pretrained_head", "rul_only_vs_dual_head"),
    ("current_anil", "pretrained_head", "current_anil_vs_best_small_k_baseline"),
    ("pretrained_anil_head", "current_anil", "pretraining_initialization_effect"),
    ("pretrained_anil_rul", "pretrained_anil_head", "rul_only_inner_loop_effect"),
    ("engine_anil_rul", "pretrained_anil_rul", "engine_disjoint_task_effect"),
    ("budget_anil_rul", "engine_anil_rul", "outer_budget_effect"),
    ("budget_anil_rul", "pretrained_rul_only", "repaired_anil_vs_matched_rul_baseline"),
    ("budget_anil_rul", "pretrained_head", "repaired_anil_vs_experiment9_small_k_baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验10：ANIL预训练初始化、轻量RUL头、发动机级任务和预算消融"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=REGIMES, default=list(REGIMES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
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
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument(
        "--budget-meta-epochs",
        type=int,
        help=(
            "budget_anil_rul外循环步数；默认等于source_pretrain_steps，"
            "即匹配普通预训练Adam更新次数"
        ),
    )
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--outer-lr", type=float, default=0.05)
    parser.add_argument("--pair-aux-weight", type=float, default=0.01)
    parser.add_argument("--source-pretrain-steps", type=int)
    parser.add_argument("--source-pretrain-lr", type=float)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument("--anil-meta-lr", type=float, default=1e-4)
    parser.add_argument("--budget-anil-meta-lr", type=float)
    parser.add_argument(
        "--anil-order", choices=("first", "second"), default="first"
    )
    parser.add_argument("--anil-query-batches", type=int, default=1)
    parser.add_argument(
        "--source-query-fraction",
        type=float,
        default=0.30,
        help="源域每个任务分给查询集的发动机比例",
    )
    parser.add_argument(
        "--source-task-seed",
        type=int,
        default=2027,
        help="独立于模型随机种子的源域支持/查询发动机划分种子",
    )
    parser.add_argument("--output-dir", default="outputs/experiment10_anil_repair")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.update(
        {
            "seed": seed,
            "target_domain": args.target,
            "source_domains": [
                domain
                for domain in EXPECTED_OFFICIAL_TEST_ENGINES
                if domain != args.target
            ],
            "condition_count": args.condition_count,
            "normalizer_seed": args.normalizer_seed,
            "meta_epochs": args.meta_epochs,
            "target_epochs": args.target_epochs,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "outer_lr": args.outer_lr,
            "pair_aux_weight": args.pair_aux_weight,
            "anil_meta_lr": args.anil_meta_lr,
            "anil_order": args.anil_order,
            "anil_query_batches": args.anil_query_batches,
        }
    )
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
    cfg["budget_meta_epochs"] = (
        args.budget_meta_epochs
        if args.budget_meta_epochs is not None
        else cfg["source_pretrain_steps"]
    )
    cfg["budget_anil_meta_lr"] = (
        args.budget_anil_meta_lr
        if args.budget_anil_meta_lr is not None
        else cfg["anil_meta_lr"]
    )
    for key in (
        "meta_epochs",
        "budget_meta_epochs",
        "target_epochs",
        "inner_steps",
        "source_pretrain_steps",
        "anil_query_batches",
    ):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key}必须为正整数")
    if not 0.0 < args.source_query_fraction < 1.0:
        raise ValueError("--source-query-fraction必须位于(0, 1)")
    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir, PROJECT_ROOT))
    cfg["output_dir"] = str(resolve_path(args.output_dir, PROJECT_ROOT))
    return cfg


def is_scope_parameter(name: str, scope: str, pair_aux_weight: float) -> bool:
    if scope == "full":
        return True
    if scope == "rul_head":
        return name.startswith("predictor.")
    if scope == "dual_head":
        return name.startswith("predictor.") or (
            pair_aux_weight > 0 and name.startswith("pairwise_predictor.")
        )
    raise ValueError(f"未知参数范围：{scope}")


def effective_aux_weight(cfg: dict, scope: str) -> float:
    return cfg.get("pair_aux_weight", 0.0) if scope != "rul_head" else 0.0


def parameter_inventory(model: torch.nn.Module, pair_aux_weight: float) -> dict:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    dual_head = 0
    rul_head = 0
    dual_names: list[str] = []
    rul_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        counts[parameter_group(name)] += count
        if is_scope_parameter(name, "dual_head", pair_aux_weight):
            dual_head += count
            dual_names.append(name)
        if is_scope_parameter(name, "rul_head", pair_aux_weight):
            rul_head += count
            rul_names.append(name)
    return {
        "total_parameter_count": total,
        "dual_head_parameter_count": dual_head,
        "rul_head_parameter_count": rul_head,
        "dual_head_trainable_fraction": dual_head / total,
        "rul_head_trainable_fraction": rul_head / total,
        "parameter_count_by_group": dict(counts),
        "dual_head_parameter_names": dual_names,
        "rul_head_parameter_names": rul_names,
    }


def next_batch(loader: DataLoader, iterator: Iterable):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def subset_loader(
    base_loader: DataLoader,
    indices: np.ndarray,
    *,
    training: bool,
    balance_mode: str,
    seed: int,
) -> DataLoader:
    dataset = base_loader.dataset
    indices = np.asarray(indices, dtype=int)
    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    if training and balance_mode != "none":
        labels = dataset.y[indices].detach().cpu().numpy()
        units = np.asarray(dataset.units)[indices]
        weights = sampling_weights(labels, units, balance_mode)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            subset,
            batch_size=base_loader.batch_size,
            sampler=sampler,
            drop_last=False,
        )
    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=training,
        generator=generator if training else None,
        drop_last=False,
    )


def split_source_tasks_by_engine(
    source_tasks: dict[str, DataLoader],
    balance_mode: str,
    query_fraction: float,
    split_seed: int,
) -> tuple[dict[str, tuple[DataLoader, DataLoader]], dict]:
    """Create deterministic, engine-disjoint support/query loaders per domain."""
    splits: dict[str, tuple[DataLoader, DataLoader]] = {}
    manifest: dict[str, dict] = {}
    for domain_index, domain in enumerate(sorted(source_tasks)):
        loader = source_tasks[domain]
        units_per_window = np.asarray(loader.dataset.units, dtype=int)
        unique_units = np.asarray(sorted(np.unique(units_per_window)), dtype=int)
        if len(unique_units) < 2:
            raise ValueError(f"{domain}至少需要2台发动机构造支持/查询任务")
        rng = np.random.default_rng(split_seed + 1009 * (domain_index + 1))
        order = rng.permutation(unique_units)
        query_count = int(round(len(order) * query_fraction))
        query_count = min(max(query_count, 1), len(order) - 1)
        query_units = np.asarray(order[:query_count], dtype=int)
        support_units = np.asarray(order[query_count:], dtype=int)
        if set(support_units) & set(query_units):
            raise AssertionError("源域支持发动机和查询发动机发生重叠")
        support_indices = np.flatnonzero(np.isin(units_per_window, support_units))
        query_indices = np.flatnonzero(np.isin(units_per_window, query_units))
        support_loader = subset_loader(
            loader,
            support_indices,
            training=True,
            balance_mode=balance_mode,
            seed=split_seed + 2000 + domain_index,
        )
        # 查询集保持自然分布，只打乱窗口，不执行阶段重采样。
        query_loader = subset_loader(
            loader,
            query_indices,
            training=True,
            balance_mode="none",
            seed=split_seed + 3000 + domain_index,
        )
        splits[domain] = (support_loader, query_loader)
        manifest[domain] = {
            "support_engine_count": int(len(support_units)),
            "query_engine_count": int(len(query_units)),
            "support_window_count": int(len(support_indices)),
            "query_window_count": int(len(query_indices)),
            "support_units": support_units.tolist(),
            "query_units": query_units.tolist(),
            "support_units_hash": stable_unit_hash(support_units),
            "query_units_hash": stable_unit_hash(query_units),
            "overlap_count": 0,
        }
    return splits, manifest


def train_source_anil(
    model: torch.nn.Module,
    source_tasks: dict[str, DataLoader],
    cfg: dict,
    device: torch.device,
    *,
    head_scope: str,
    task_mode: str,
    meta_epochs: int,
    meta_lr: float,
    balance_mode: str,
    source_query_fraction: float,
    source_task_seed: int,
) -> tuple[torch.nn.Module, list[dict], dict | None]:
    """Train ANIL with configurable initialization, inner head and task split."""
    meta_model = deepcopy(model).to(device)
    meta_model.train()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    task_names = sorted(source_tasks)
    task_count = min(cfg["tasks_per_meta_batch"], len(task_names))
    split_manifest = None

    if task_mode == "engine_disjoint":
        task_pairs, split_manifest = split_source_tasks_by_engine(
            source_tasks,
            balance_mode,
            source_query_fraction,
            source_task_seed,
        )
        support_iterators = {
            name: iter(task_pairs[name][0]) for name in task_names
        }
        query_iterators = {name: iter(task_pairs[name][1]) for name in task_names}
    elif task_mode == "batch":
        task_pairs = {name: (source_tasks[name], source_tasks[name]) for name in task_names}
        # 与实验9一致：同一迭代器连续取支持批次和查询批次。
        support_iterators = {name: iter(source_tasks[name]) for name in task_names}
        query_iterators = support_iterators
    else:
        raise ValueError(f"未知ANIL任务模式：{task_mode}")

    head_names = [
        name
        for name, _ in meta_model.named_parameters()
        if is_scope_parameter(name, head_scope, cfg.get("pair_aux_weight", 0.0))
    ]
    if not head_names:
        raise RuntimeError(f"未找到{head_scope}参数")

    use_second_order = cfg["anil_order"] == "second"
    aux_weight = effective_aux_weight(cfg, head_scope)
    history: list[dict] = []
    report_every = max(1, min(25, meta_epochs // 10))

    for epoch in range(1, meta_epochs + 1):
        selected = random.sample(task_names, task_count)
        task_query_losses: list[torch.Tensor] = []
        for task_name in selected:
            support_loader, query_loader = task_pairs[task_name]
            parameters = dict(meta_model.named_parameters())
            buffers = dict(meta_model.named_buffers())
            fast_head = {name: parameters[name] for name in head_names}

            for _ in range(cfg["inner_steps"]):
                (support_x, support_y), iterator = next_batch(
                    support_loader, support_iterators[task_name]
                )
                support_iterators[task_name] = iterator
                if task_mode == "batch":
                    query_iterators[task_name] = iterator
                support_x, support_y = support_x.to(device), support_y.to(device)
                state = {**buffers, **parameters, **fast_head}
                loss = functional_rul_loss(
                    meta_model, state, support_x, support_y, aux_weight
                )
                gradients = torch.autograd.grad(
                    loss,
                    tuple(fast_head.values()),
                    create_graph=use_second_order,
                    allow_unused=True,
                )
                updated: dict[str, torch.Tensor] = {}
                for (name, parameter), gradient in zip(fast_head.items(), gradients):
                    if gradient is None:
                        gradient = torch.zeros_like(parameter)
                    if not use_second_order:
                        gradient = gradient.detach()
                    updated[name] = parameter - cfg["inner_lr"] * gradient
                fast_head = updated

            query_losses: list[torch.Tensor] = []
            for _ in range(cfg["anil_query_batches"]):
                (query_x, query_y), iterator = next_batch(
                    query_loader, query_iterators[task_name]
                )
                query_iterators[task_name] = iterator
                if task_mode == "batch":
                    support_iterators[task_name] = iterator
                query_x, query_y = query_x.to(device), query_y.to(device)
                state = {**buffers, **parameters, **fast_head}
                query_losses.append(
                    functional_rul_loss(
                        meta_model, state, query_x, query_y, aux_weight
                    )
                )
            task_query_losses.append(torch.stack(query_losses).mean())

        meta_loss = torch.stack(task_query_losses).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 5.0)
        optimizer.step()

        if epoch % report_every == 0 or epoch == meta_epochs:
            row = {
                "meta_epoch": epoch,
                "query_loss": float(meta_loss.detach().cpu().item()),
                "head_scope": head_scope,
                "task_mode": task_mode,
                "meta_lr": meta_lr,
                "tasks": selected,
            }
            history.append(row)
            print(
                f"anil_meta_epoch={epoch:04d}/{meta_epochs} "
                f"query_loss={row['query_loss']:.4f} "
                f"head={head_scope} task={task_mode}"
            )
    return meta_model, history, split_manifest


def set_target_scope(
    model: torch.nn.Module, scope: str, pair_aux_weight: float
) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = is_scope_parameter(name, scope, pair_aux_weight)
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
    model.eval()
    model.predictor.train()
    if scope == "dual_head" and hasattr(model, "pairwise_predictor"):
        model.pairwise_predictor.train()


def train_target(
    model: torch.nn.Module,
    support: DataLoader,
    validation: DataLoader,
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
    aux_weight = effective_aux_weight(cfg, scope)

    for epoch in range(1, cfg["target_epochs"] + 1):
        set_training_mode(learner, scope)
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, _ = rul_training_loss(learner, x, y, aux_weight)
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
    count = int(sum(parameter.numel() for parameter in trainable))
    return learner, history, best_epoch, count, drift


def regime_spec(regime: str) -> dict:
    return {
        "pretrained_full": {
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "full",
        },
        "pretrained_head": {
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "dual_head",
        },
        "pretrained_rul_only": {
            "source_training": "ordinary_multisource_supervised_pretraining",
            "target_scope": "rul_head",
        },
        "current_anil": {
            "source_training": "anil_random_initialization",
            "target_scope": "dual_head",
            "head_scope": "dual_head",
            "task_mode": "batch",
            "initialization": "random",
            "budget": "base",
        },
        "pretrained_anil_head": {
            "source_training": "anil_from_ordinary_pretraining",
            "target_scope": "dual_head",
            "head_scope": "dual_head",
            "task_mode": "batch",
            "initialization": "ordinary_pretrained",
            "budget": "base",
        },
        "pretrained_anil_rul": {
            "source_training": "anil_from_pretraining_rul_only_inner_loop",
            "target_scope": "rul_head",
            "head_scope": "rul_head",
            "task_mode": "batch",
            "initialization": "ordinary_pretrained",
            "budget": "base",
        },
        "engine_anil_rul": {
            "source_training": "anil_engine_disjoint_rul_only",
            "target_scope": "rul_head",
            "head_scope": "rul_head",
            "task_mode": "engine_disjoint",
            "initialization": "ordinary_pretrained",
            "budget": "base",
        },
        "budget_anil_rul": {
            "source_training": "anil_engine_disjoint_rul_only_budget_matched",
            "target_scope": "rul_head",
            "head_scope": "rul_head",
            "task_mode": "engine_disjoint",
            "initialization": "ordinary_pretrained",
            "budget": "matched",
        },
    }[regime]


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    return {
        "output": output,
        "raw": output / f"experiment10_raw_{args.target}.json",
        "summary": output / f"experiment10_summary_{args.target}.csv",
        "paired": output / f"experiment10_paired_by_seed_{args.target}.csv",
        "comparisons": output / f"experiment10_comparisons_{args.target}.csv",
        "protocol": output / f"experiment10_split_protocol_{args.target}.json",
        "splits": output / f"experiment10_engine_splits_{args.target}.csv",
        "budget": output / f"experiment10_budget_{args.target}.json",
        "parameters": output / f"experiment10_parameter_inventory_{args.target}.json",
        "source_splits": output / f"experiment10_source_task_splits_{args.target}.json",
    }


def source_cache_path(args: argparse.Namespace, regime: str, seed: int) -> Path:
    return result_paths(args)["output"] / "source_cache" / (
        f"source_{regime}_{args.target}_seed{seed}.pt"
    )


def source_signature(
    args: argparse.Namespace, cfg: dict, regime: str, feature_count: int
) -> str:
    payload = {
        "script": SCRIPT_VERSION,
        "regime": regime,
        "seed": cfg["seed"],
        "target": cfg["target_domain"],
        "source_domains": cfg["source_domains"],
        "feature_count": feature_count,
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "meta_epochs": cfg["meta_epochs"],
        "budget_meta_epochs": cfg["budget_meta_epochs"],
        "inner_steps": cfg["inner_steps"],
        "inner_lr": cfg["inner_lr"],
        "pair_aux_weight": cfg["pair_aux_weight"],
        "anil_meta_lr": cfg["anil_meta_lr"],
        "budget_anil_meta_lr": cfg["budget_anil_meta_lr"],
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "source_pretrain_lr": cfg["source_pretrain_lr"],
        "source_query_fraction": args.source_query_fraction,
        "source_task_seed": args.source_task_seed,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def load_source_cache(path: Path, signature: str) -> tuple[dict, list, dict | None] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        print(f"[cache ignored] 配置签名变化：{path.name}")
        return None
    print(f"[source cache] {path}")
    return payload["state"], payload.get("history", []), payload.get("source_split")


def save_source_cache(
    path: Path,
    signature: str,
    state: dict,
    history: list,
    source_split: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "state": state,
            "history": history,
            "source_split": source_split,
        },
        path,
    )


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def fresh_source_tasks(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
) -> tuple[dict[str, DataLoader], int]:
    first_k = min(int(value) for value in protocol["k_values"])
    first_units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(first_k)]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        first_units,
    )
    return loaders[0], loaders[4]


def build_source_initializations(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    required: list[str],
) -> tuple[dict[str, dict], dict[str, list], int, dict, dict]:
    _, feature_count = fresh_source_tasks(args, cfg, protocol, seed)
    seed_everything(seed)
    base_model = build_model("meta_gnn", feature_count, cfg).cpu()
    base_state = deepcopy(base_model.state_dict())
    inventory = parameter_inventory(base_model, cfg.get("pair_aux_weight", 0.0))
    device = resolve_device(cfg["device"])
    states: dict[str, dict] = {}
    histories: dict[str, list] = {}
    source_manifests: dict[str, dict] = {}

    ordinary_required = any(
        regime != "current_anil" for regime in required
    )
    ordinary_state: dict | None = None
    ordinary_history: list = []
    ordinary_cache = source_cache_path(args, "ordinary_pretraining", seed)
    ordinary_signature = source_signature(
        args, cfg, "ordinary_pretraining", feature_count
    )
    if ordinary_required and args.resume:
        cached = load_source_cache(ordinary_cache, ordinary_signature)
        if cached is not None:
            ordinary_state, ordinary_history, _ = cached
    if ordinary_required and ordinary_state is None:
        source_tasks, _ = fresh_source_tasks(args, cfg, protocol, seed)
        model = build_model("meta_gnn", feature_count, cfg)
        model.load_state_dict(base_state)
        seed_everything(seed)
        model, ordinary_history = train_source_supervised(
            model, source_tasks, cfg, device
        )
        ordinary_state = cpu_state(model)
        save_source_cache(
            ordinary_cache,
            ordinary_signature,
            ordinary_state,
            ordinary_history,
            None,
        )
        del model, source_tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for regime in required:
        if regime in {"pretrained_full", "pretrained_head", "pretrained_rul_only"}:
            if ordinary_state is None:
                raise RuntimeError("普通预训练状态未生成")
            states[regime] = deepcopy(ordinary_state)
            histories[regime] = deepcopy(ordinary_history)
            continue

        cache = source_cache_path(args, regime, seed)
        signature = source_signature(args, cfg, regime, feature_count)
        if args.resume:
            cached = load_source_cache(cache, signature)
            if cached is not None:
                state, history, manifest = cached
                states[regime] = state
                histories[regime] = history
                if manifest is not None:
                    source_manifests[regime] = manifest
                continue

        spec = regime_spec(regime)
        initial_state = (
            base_state
            if spec["initialization"] == "random"
            else ordinary_state
        )
        if initial_state is None:
            raise RuntimeError(f"{regime}缺少普通预训练初始化")
        source_tasks, _ = fresh_source_tasks(args, cfg, protocol, seed)
        model = build_model("meta_gnn", feature_count, cfg)
        model.load_state_dict(initial_state)
        meta_epochs = (
            cfg["budget_meta_epochs"]
            if spec["budget"] == "matched"
            else cfg["meta_epochs"]
        )
        meta_lr = (
            cfg["budget_anil_meta_lr"]
            if spec["budget"] == "matched"
            else cfg["anil_meta_lr"]
        )
        seed_everything(seed)
        model, history, manifest = train_source_anil(
            model,
            source_tasks,
            cfg,
            device,
            head_scope=spec["head_scope"],
            task_mode=spec["task_mode"],
            meta_epochs=meta_epochs,
            meta_lr=meta_lr,
            balance_mode=args.balance_mode,
            source_query_fraction=args.source_query_fraction,
            source_task_seed=args.source_task_seed,
        )
        state = cpu_state(model)
        states[regime] = state
        histories[regime] = history
        if manifest is not None:
            source_manifests[regime] = manifest
        save_source_cache(cache, signature, state, history, manifest)
        del model, source_tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return states, histories, feature_count, inventory, source_manifests


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
    model, target_history, best_epoch, trainable_count, drift = train_target(
        model,
        support,
        validation,
        cfg,
        device,
        spec["target_scope"],
    )
    validation_metrics = evaluate(model, validation, device)
    test_metrics = evaluate(model, test, device)
    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    if regime.startswith("pretrained_") and "anil" not in regime:
        source_outer_steps = cfg["source_pretrain_steps"]
        source_inner_gradients = 0
        source_query_gradients = 0
    elif regime == "budget_anil_rul":
        source_outer_steps = cfg["budget_meta_epochs"]
        source_inner_gradients = (
            cfg["budget_meta_epochs"] * task_count * cfg["inner_steps"]
        )
        source_query_gradients = (
            cfg["budget_meta_epochs"] * task_count * cfg["anil_query_batches"]
        )
    else:
        source_outer_steps = cfg["meta_epochs"]
        source_inner_gradients = (
            cfg["meta_epochs"] * task_count * cfg["inner_steps"]
        )
        source_query_gradients = (
            cfg["meta_epochs"] * task_count * cfg["anil_query_batches"]
        )

    result = {
        **test_metrics,
        "regime": regime,
        "model": "meta_gnn_rul",
        "source_training": spec["source_training"],
        "target_adaptation_scope": spec["target_scope"],
        "experiment": f"experiment10_{regime}_k{k}",
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
        "source_outer_optimizer_steps": source_outer_steps,
        "source_inner_gradient_budget": source_inner_gradients,
        "source_query_gradient_budget": source_query_gradients,
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "parameter_drift_by_group": drift,
    }
    if "anil" in regime:
        result.update(
            {
                "anil_initialization": spec["initialization"],
                "anil_inner_scope": spec["head_scope"],
                "anil_task_mode": spec["task_mode"],
                "anil_order": cfg["anil_order"],
                "anil_meta_lr": (
                    cfg["budget_anil_meta_lr"]
                    if regime == "budget_anil_rul"
                    else cfg["anil_meta_lr"]
                ),
            }
        )
    else:
        result.update(
            {
                "anil_initialization": "not_applicable",
                "anil_inner_scope": "not_applicable",
                "anil_task_mode": "not_applicable",
                "anil_order": "not_applicable",
                "anil_meta_lr": 0.0,
            }
        )

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / (
        f"experiment10_{regime}_k{k}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
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
                delta = float(candidate_row[metric] - reference_row[metric])
                row[f"{metric}_delta_candidate_minus_reference"] = delta
                row[f"candidate_{metric}_win"] = float(delta < 0)
            r2_delta = float(candidate_row["r2"] - reference_row["r2"])
            row["r2_delta_candidate_minus_reference"] = r2_delta
            row["candidate_r2_win"] = float(r2_delta > 0)
            rows.append(row)
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired, pd.DataFrame()
    delta_columns = [
        column
        for column in paired.columns
        if "_delta_candidate_minus_reference" in column
    ]
    win_columns = [column for column in paired.columns if column.endswith("_win")]
    summaries: list[dict] = []
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
) -> tuple[dict, dict]:
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
    model = build_model("meta_gnn", feature_count, cfg).cpu().eval()
    with torch.no_grad():
        output = model(x[: min(8, len(x))])
    inventory = parameter_inventory(model, cfg.get("pair_aux_weight", 0.0))
    _, source_manifest = split_source_tasks_by_engine(
        tasks,
        args.balance_mode,
        args.source_query_fraction,
        args.source_task_seed,
    )
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
    return inventory, source_manifest


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
            f"{args.target}官方测试集应为{expected_count}台，"
            f"当前为{protocol['official_test_engine_count']}台"
        )

    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied_protocol = dict(protocol)
    copied_protocol["experiment10_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path is not None else "regenerated"
    )
    atomic_write_text(
        paths["protocol"], json.dumps(copied_protocol, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )

    task_count = min(
        first_cfg["tasks_per_meta_batch"], len(first_cfg["source_domains"])
    )
    budget = {
        "source_domains": first_cfg["source_domains"],
        "meta_epochs_base": first_cfg["meta_epochs"],
        "budget_meta_epochs": first_cfg["budget_meta_epochs"],
        "tasks_per_meta_batch": task_count,
        "inner_steps": first_cfg["inner_steps"],
        "ordinary_pretraining_optimizer_steps": first_cfg["source_pretrain_steps"],
        "base_anil_outer_optimizer_steps": first_cfg["meta_epochs"],
        "budget_anil_outer_optimizer_steps": first_cfg["budget_meta_epochs"],
        "base_anil_inner_gradient_budget": (
            first_cfg["meta_epochs"] * task_count * first_cfg["inner_steps"]
        ),
        "budget_anil_inner_gradient_budget": (
            first_cfg["budget_meta_epochs"] * task_count * first_cfg["inner_steps"]
        ),
        "base_anil_query_gradient_budget": (
            first_cfg["meta_epochs"] * task_count * first_cfg["anil_query_batches"]
        ),
        "budget_anil_query_gradient_budget": (
            first_cfg["budget_meta_epochs"]
            * task_count
            * first_cfg["anil_query_batches"]
        ),
        "target_epochs_equal_for_all_regimes": first_cfg["target_epochs"],
        "target_lr_equal_for_all_regimes": first_cfg["inner_lr"],
        "source_query_fraction": args.source_query_fraction,
        "source_task_seed": args.source_task_seed,
        "note": (
            "budget_anil_rul matches ordinary pretraining by outer optimizer-step "
            "count, not by wall-clock time or total gradient computations."
        ),
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))

    print("\n[实验10固定协议与训练预算]")
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
        source_manifest = None
        for k in k_values:
            inventory, source_manifest = inspect_protocol(
                args, first_cfg, protocol, seeds[0], k
            )
        if inventory is not None:
            atomic_write_text(
                paths["parameters"], json.dumps(inventory, ensure_ascii=False, indent=2)
            )
        if source_manifest is not None:
            atomic_write_text(
                paths["source_splits"],
                json.dumps(source_manifest, ensure_ascii=False, indent=2),
            )
        print("\n[dry-run完成] 未训练模型。")
        print(
            f"Protocol: {paths['protocol']}\nSplits: {paths['splits']}"
            f"\nBudget: {paths['budget']}\nParameters: {paths['parameters']}"
            f"\nSource task splits: {paths['source_splits']}"
        )
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条结果。")
    done = completed_keys(results)
    all_source_manifests: dict[str, dict] = {}

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
        states, histories, _, inventory, source_manifests = (
            build_source_initializations(
                args, cfg, protocol, seed, required
            )
        )
        atomic_write_text(
            paths["parameters"], json.dumps(inventory, ensure_ascii=False, indent=2)
        )
        if source_manifests:
            all_source_manifests[str(seed)] = source_manifests
            atomic_write_text(
                paths["source_splits"],
                json.dumps(all_source_manifests, ensure_ascii=False, indent=2),
            )

        for k in k_values:
            units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
            for regime in regimes:
                key = (seed, k, regime)
                if key in done:
                    print(f"[skip] seed={seed} K={k} regime={regime}")
                    continue
                print(
                    f"\n[experiment10] seed={seed} K={k} regime={regime} "
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
    print("\n[experiment10 summary]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        print("\n[配对比较：RMSE/MAE/NASA delta<0、R2 delta>0表示候选更好]")
        print(comparisons.to_string(index=False))
    print("\n[原因定位规则]")
    print("1. pretrained_anil_head优于current_anil：普通预训练初始化是关键。")
    print("2. pretrained_anil_rul优于pretrained_anil_head：辅助头过大或存在干扰。")
    print("3. engine_anil_rul优于pretrained_anil_rul：发动机级支持/查询更接近真实迁移。")
    print("4. budget_anil_rul优于engine_anil_rul：实验9的ANIL外循环训练不足。")
    print("5. repaired ANIL必须优于pretrained_rul_only，才能证明存在额外元学习收益。")
    print("6. 主要看K=2、5的五种子配对胜率，同时要求NASA Score不恶化。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
        f"\nBudget: {paths['budget']}\nParameters: {paths['parameters']}"
        f"\nSource task splits: {paths['source_splits']}"
    )


if __name__ == "__main__":
    main()
