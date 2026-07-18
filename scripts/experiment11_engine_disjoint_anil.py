"""实验11：发动机互斥源任务能否带来ANIL的独立元学习收益。

研究问题
========

实验10C表明，稳定ANIL明显优于Reptile全模型元学习，但尚未显著优于
“普通源域预训练 + 目标域只微调RUL预测头”。实验11只改变源域元任务构造，
验证当前ANIL收益不明显是否源于支持集和查询集过于相似。

默认比较四组方案：

``pretrained_head``
    普通多源监督预训练，目标域只更新 ``predictor.*``。

``pretrained_budget_head``
    普通预训练后增加与ANIL支持/查询损失批次数匹配的监督更新，排除“只是多训练
    了一些”的解释；目标域仍只更新 ``predictor.*``。

``anil_batch_head``
    实验10C的ANIL方式：源任务的支持批次和查询批次来自同一域、同一窗口池。

``anil_engine_disjoint_head``
    每个源域按发动机ID固定划分支持发动机和查询发动机，二者零重叠。内循环只在
    支持发动机上更新RUL头，外循环根据未出现在支持集中的查询发动机更新初始化。

四组方案使用相同的FD004 K-shot发动机、固定验证发动机、目标训练轮数、学习率、
损失函数与官方测试集。默认使用10个随机种子，以提高配对统计检验能力。

快速检查：

    python -u scripts/experiment11_engine_disjoint_anil.py \
      --target FD004 --k-values 2 5 10 20 --seeds 42 43 44 45 46 \
      --dry-run

正式实验（Linux/服务器）：

    mkdir -p outputs/experiment11_engine_disjoint_anil
    CUDA_VISIBLE_DEVICES=0 nohup python -u \
      scripts/experiment11_engine_disjoint_anil.py \
      --target FD004 \
      --k-values 2 5 10 20 \
      --seeds 42 43 44 45 46 47 48 49 50 51 \
      --preprocessing condition_settings \
      --balance-mode engine_stage \
      --meta-epochs 100 \
      --meta-inner-lr 0.00001 \
      --meta-inner-steps 1 \
      --anil-meta-lr 0.0001 \
      --source-query-fraction 0.30 \
      --source-pretrain-steps 1500 \
      --target-epochs 10 \
      --target-lr 0.001 \
      --resume \
      > outputs/experiment11_engine_disjoint_anil/experiment11_FD004.log 2>&1 &

注意：官方测试集只应在固定验证集选定目标训练轮次后用于最终评价，不得根据官方
测试结果继续调整上述超参数。
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
    target_unit_protocol,
)
from scripts.experiment8_transfer_baseline import train_source_supervised  # noqa: E402
from scripts.experiment10_anil_repair import (  # noqa: E402
    cpu_state,
    fresh_source_tasks,
    parameter_inventory,
    split_source_tasks_by_engine,
)
from scripts.experiment10b_anil_stability import (  # noqa: E402
    all_tensors_finite,
    train_safe_anil,
)
from scripts.experiment10c_target_kshot import train_target  # noqa: E402


SCRIPT_VERSION = "experiment11_engine_disjoint_anil_v1"
REGIMES = (
    "pretrained_head",
    "pretrained_budget_head",
    "anil_batch_head",
    "anil_engine_disjoint_head",
)
COMPARISONS = (
    (
        "anil_engine_disjoint_head",
        "anil_batch_head",
        "engine_disjoint_vs_batch_anil",
    ),
    (
        "anil_engine_disjoint_head",
        "pretrained_head",
        "engine_disjoint_anil_vs_ordinary_head",
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
    "engine_disjoint_anil_vs_budget_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验11：比较batch-ANIL与发动机互斥ANIL的FD004 K-shot表现"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", default="FD004", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
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

    # 实验10B确认的稳定ANIL设置。
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
    parser.add_argument(
        "--source-task-seed",
        type=int,
        default=2027,
        help="固定源域支持/查询发动机划分；默认不随模型seed变化",
    )
    parser.add_argument(
        "--vary-source-split-by-seed",
        action="store_true",
        help="若启用，源域发动机划分种子=source_task_seed+模型seed",
    )
    parser.add_argument("--outer-lr", type=float, default=0.05)
    parser.add_argument("--pair-aux-weight", type=float, default=0.0)

    # 普通源域预训练和预算匹配组。
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--budget-extra-steps",
        type=int,
        help=(
            "预算匹配组额外监督更新步数；默认=meta_epochs×任务数×"
            "(meta_inner_steps+anil_query_batches)"
        ),
    )

    # 目标域所有组严格一致，只更新RUL预测头。
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-clip-norm", type=float, default=0.0)

    parser.add_argument(
        "--output-dir", default="outputs/experiment11_engine_disjoint_anil"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[int], list[str]]:
    k_values = sorted(set(args.k_values))
    seeds = list(dict.fromkeys(args.seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须为正整数")
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
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "loss_ceiling": args.loss_ceiling,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正数：{invalid}")
    if not 0 < args.source_query_fraction < 1:
        raise ValueError("--source-query-fraction必须位于(0,1)")
    if args.meta_clip_norm < 0 or args.target_clip_norm < 0:
        raise ValueError("梯度裁剪阈值不能为负数，0表示不裁剪")
    if args.budget_extra_steps is not None and args.budget_extra_steps <= 0:
        raise ValueError("--budget-extra-steps必须为正整数")
    if len(seeds) < 5 and not args.dry_run:
        print("[警告] 当前少于5个随机种子，只能视为预实验。")
    return k_values, seeds, regimes


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
            "inner_steps": args.meta_inner_steps,
            "inner_lr": args.meta_inner_lr,
            "outer_lr": args.outer_lr,
            # 实验11不使用pairwise辅助头，避免引入第二个实验变量。
            "pair_aux_weight": 0.0,
            "anil_meta_lr": args.anil_meta_lr,
            "anil_order": args.anil_order,
            "anil_query_batches": args.anil_query_batches,
            "source_pretrain_steps": args.source_pretrain_steps,
            "source_pretrain_lr": args.source_pretrain_lr,
            "source_pretrain_weight_decay": args.source_pretrain_weight_decay,
            "target_lr": args.target_lr,
            "target_weight_decay": args.target_weight_decay,
        }
    )
    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir, PROJECT_ROOT))
    cfg["output_dir"] = str(resolve_path(args.output_dir, PROJECT_ROOT))
    return cfg


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment11_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_by_seed.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_split_protocol.json",
        "splits": output / f"{prefix}_engine_splits.csv",
        "budget": output / f"{prefix}_budget.json",
        "source_splits": output / f"{prefix}_source_task_splits.json",
        "source_diagnostics": output / f"{prefix}_source_diagnostics.json",
        "parameters": output / f"{prefix}_parameter_inventory.json",
    }


def default_experiment7_protocol(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "outputs" / "experiment7_kshot_engines" / (
        f"experiment7_split_protocol_{args.target}.json"
    )


def load_or_extend_protocol(
    args: argparse.Namespace,
    cfg: dict,
    seeds: list[int],
    k_values: list[int],
) -> tuple[dict, Path | None]:
    """Regenerate missing seeds deterministically without changing experiment7 files."""
    generated = target_unit_protocol(
        cfg["data_dir"],
        args.target,
        args.validation_units,
        args.validation_seed,
        seeds,
        k_values,
    )
    source_path = (
        resolve_path(args.protocol_file, PROJECT_ROOT)
        if args.protocol_file
        else default_experiment7_protocol(args)
    )
    if not source_path.is_file():
        print("[protocol] 未找到实验7协议，按相同规则重新生成实验11协议。")
        return generated, None

    existing = json.loads(source_path.read_text(encoding="utf-8"))
    checks = {
        "target_domain": (existing.get("target_domain"), generated["target_domain"]),
        "validation_units": (
            existing.get("validation_units"),
            generated["validation_units"],
        ),
        "official_test_units_hash": (
            existing.get("official_test_units_hash"),
            generated["official_test_units_hash"],
        ),
        "official_test_engine_count": (
            existing.get("official_test_engine_count"),
            generated["official_test_engine_count"],
        ),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(
            f"实验7协议与当前数据/验证规则不一致：{mismatches}。"
            "请确认--validation-seed、--validation-units和数据目录。"
        )
    print(f"[protocol] 复核实验7固定划分：{source_path}")
    return generated, source_path


def regime_spec(regime: str) -> dict:
    return {
        "pretrained_head": {
            "source_key": "ordinary",
            "source_training": "ordinary_multisource_pretraining",
            "source_task_mode": "none",
        },
        "pretrained_budget_head": {
            "source_key": "ordinary_budget",
            "source_training": "ordinary_pretraining_plus_budget_continuation",
            "source_task_mode": "none",
        },
        "anil_batch_head": {
            "source_key": "anil_batch",
            "source_training": "ordinary_pretraining_plus_batch_anil",
            "source_task_mode": "batch",
        },
        "anil_engine_disjoint_head": {
            "source_key": "anil_engine_disjoint",
            "source_training": "ordinary_pretraining_plus_engine_disjoint_anil",
            "source_task_mode": "engine_disjoint",
        },
    }[regime]


def budget_extra_steps(args: argparse.Namespace, cfg: dict) -> int:
    task_count = min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
    default = (
        args.meta_epochs
        * task_count
        * (args.meta_inner_steps + args.anil_query_batches)
    )
    return args.budget_extra_steps if args.budget_extra_steps is not None else default


def source_split_seed(args: argparse.Namespace, seed: int) -> int:
    if args.vary_source_split_by_seed:
        return int(args.source_task_seed + seed)
    return int(args.source_task_seed)


def source_cache_path(args: argparse.Namespace, source_key: str, seed: int) -> Path:
    return result_paths(args)["output"] / "source_cache" / (
        f"{source_key}_{args.target}_seed{seed}.pt"
    )


def source_signature(
    args: argparse.Namespace,
    cfg: dict,
    source_key: str,
    feature_count: int,
    split_seed: int,
) -> str:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    payload = {
        "script_version": SCRIPT_VERSION,
        "source_key": source_key,
        "target": args.target,
        "seed": cfg["seed"],
        "source_domains": cfg["source_domains"],
        "feature_count": feature_count,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "data_dir": cfg["data_dir"],
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "source_pretrain_weight_decay": args.source_pretrain_weight_decay,
        "budget_extra_steps": budget_extra_steps(args, cfg),
        "meta_epochs": args.meta_epochs,
        "meta_inner_lr": args.meta_inner_lr,
        "meta_inner_steps": args.meta_inner_steps,
        "anil_meta_lr": args.anil_meta_lr,
        "anil_query_batches": args.anil_query_batches,
        "anil_order": args.anil_order,
        "source_query_fraction": args.source_query_fraction,
        "source_split_seed": split_seed,
        "meta_clip_norm": args.meta_clip_norm,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_source_cache(path: Path, signature: str) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        print(f"[cache ignored] 签名变化：{path.name}")
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
    split_manifest: dict | None,
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
            "source_split": split_manifest,
        },
        path,
    )


def train_ordinary_state(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    feature_count: int,
    base_state: dict,
) -> tuple[dict, list, dict]:
    source_tasks, _ = fresh_source_tasks(args, cfg, protocol, seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(base_state)
    model, history = train_source_supervised(
        model, source_tasks, cfg, resolve_device(cfg["device"])
    )
    state = cpu_state(model)
    diagnostic = {
        "status": "stable",
        "source_training": "ordinary_multisource_pretraining",
        "optimizer_steps": args.source_pretrain_steps,
        "final_reported_loss": history[-1]["mean_source_loss"] if history else None,
    }
    return state, history, diagnostic


def train_budget_state(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    feature_count: int,
    ordinary_state: dict,
) -> tuple[dict, list, dict]:
    source_tasks, _ = fresh_source_tasks(args, cfg, protocol, seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(ordinary_state)
    continuation_cfg = dict(cfg)
    continuation_cfg["source_pretrain_steps"] = budget_extra_steps(args, cfg)
    # 使用不同且确定的调度种子，不重复普通预训练的第一段批次顺序。
    continuation_cfg["seed"] = seed + 50000
    model, history = train_source_supervised(
        model,
        source_tasks,
        continuation_cfg,
        resolve_device(cfg["device"]),
    )
    state = cpu_state(model)
    diagnostic = {
        "status": "stable",
        "source_training": "ordinary_budget_matched_continuation",
        "base_optimizer_steps": args.source_pretrain_steps,
        "extra_optimizer_steps": continuation_cfg["source_pretrain_steps"],
        "total_optimizer_steps": (
            args.source_pretrain_steps + continuation_cfg["source_pretrain_steps"]
        ),
        "final_reported_loss": history[-1]["mean_source_loss"] if history else None,
    }
    return state, history, diagnostic


def train_anil_state(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    feature_count: int,
    ordinary_state: dict,
    task_mode: str,
) -> tuple[dict, list, dict, dict | None]:
    source_tasks, _ = fresh_source_tasks(args, cfg, protocol, seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(ordinary_state)
    proxy = argparse.Namespace(**vars(args))
    proxy.stage = "experiment11"
    proxy.task_mode = task_mode
    proxy.source_task_seed = source_split_seed(args, seed)
    seed_everything(seed)
    diagnostic, history, state, split_manifest = train_safe_anil(
        model,
        source_tasks,
        cfg,
        proxy,
        inner_lr=args.meta_inner_lr,
        inner_steps=args.meta_inner_steps,
        loss_mode="raw_mse",
        clip_norm=args.meta_clip_norm,
        config_id=f"experiment11_seed{seed}_{task_mode}",
    )
    if not diagnostic.get("stable") or state is None:
        raise RuntimeError(
            f"ANIL源训练不稳定：seed={seed}, mode={task_mode}, "
            f"phase={diagnostic.get('failure_phase')}, "
            f"message={diagnostic.get('failure_message')}"
        )
    return state, history, diagnostic, split_manifest


def build_source_states(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    required_regimes: list[str],
) -> tuple[dict[str, dict], dict[str, list], dict, dict, dict]:
    _, feature_count = fresh_source_tasks(args, cfg, protocol, seed)
    seed_everything(seed)
    base_model = build_model("meta_gnn", feature_count, cfg).cpu()
    base_state = cpu_state(base_model)
    inventory = parameter_inventory(base_model, 0.0)
    required_keys = list(
        dict.fromkeys(regime_spec(regime)["source_key"] for regime in required_regimes)
    )
    # 所有ANIL和预算控制都以同一个普通预训练状态为起点。
    if any(key != "ordinary" for key in required_keys):
        required_keys = ["ordinary"] + [
            key for key in required_keys if key != "ordinary"
        ]

    states_by_key: dict[str, dict] = {}
    histories_by_key: dict[str, list] = {}
    diagnostics_by_key: dict[str, dict] = {}
    split_manifests: dict[str, dict] = {}
    split_seed = source_split_seed(args, seed)

    for source_key in required_keys:
        cache_path = source_cache_path(args, source_key, seed)
        signature = source_signature(
            args, cfg, source_key, feature_count, split_seed
        )
        cached = load_source_cache(cache_path, signature) if args.resume else None
        if cached is not None:
            states_by_key[source_key] = cached["state"]
            histories_by_key[source_key] = cached.get("history", [])
            diagnostics_by_key[source_key] = cached.get("diagnostic", {})
            if cached.get("source_split") is not None:
                split_manifests[source_key] = cached["source_split"]
            continue

        if source_key == "ordinary":
            state, history, diagnostic = train_ordinary_state(
                args, cfg, protocol, seed, feature_count, base_state
            )
            manifest = None
        elif source_key == "ordinary_budget":
            state, history, diagnostic = train_budget_state(
                args,
                cfg,
                protocol,
                seed,
                feature_count,
                states_by_key["ordinary"],
            )
            manifest = None
        elif source_key in {"anil_batch", "anil_engine_disjoint"}:
            task_mode = "batch" if source_key == "anil_batch" else "engine_disjoint"
            state, history, diagnostic, manifest = train_anil_state(
                args,
                cfg,
                protocol,
                seed,
                feature_count,
                states_by_key["ordinary"],
                task_mode,
            )
        else:
            raise ValueError(f"未知源状态：{source_key}")

        if not all_tensors_finite(state.values()):
            raise RuntimeError(f"{source_key}源状态包含NaN/Inf")
        states_by_key[source_key] = state
        histories_by_key[source_key] = history
        diagnostics_by_key[source_key] = diagnostic
        if manifest is not None:
            split_manifests[source_key] = manifest
        save_source_cache(
            cache_path,
            signature,
            state,
            history,
            diagnostic,
            manifest,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    states = {
        regime: deepcopy(states_by_key[regime_spec(regime)["source_key"]])
        for regime in required_regimes
    }
    histories = {
        regime: deepcopy(histories_by_key[regime_spec(regime)["source_key"]])
        for regime in required_regimes
    }
    return states, histories, diagnostics_by_key, split_manifests, inventory


def run_target_regime(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    regime: str,
    source_state: dict,
    source_history: list,
    inventory: dict,
    k: int,
    adaptation_units: list[int],
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
    # 官方测试集只在固定验证集选定best epoch后评估。
    test_metrics = evaluate(model, test, device)
    spec = regime_spec(regime)
    result = {
        **test_metrics,
        "regime": regime,
        "model": "meta_gnn_rul",
        "source_training": spec["source_training"],
        "source_task_mode": spec["source_task_mode"],
        "target_adaptation_scope": "rul_head",
        "target_loss_mode": "raw_mse",
        "experiment": f"experiment11_{regime}_k{k}",
        "target_domain": args.target,
        "seed": cfg["seed"],
        "k": k,
        "adaptation_engine_count": len(adaptation_units),
        "adaptation_units": [int(unit) for unit in adaptation_units],
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "best_target_epoch_by_validation": best_epoch,
        "target_epochs_planned": args.target_epochs,
        "target_learning_rate": args.target_lr,
        "target_trainable_parameter_count": trainable_count,
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_fraction": (
            trainable_count / inventory["total_parameter_count"]
        ),
        "source_task_seed": source_split_seed(args, cfg["seed"]),
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
        "parameter_drift_by_group": drift,
        **target_diag,
    }

    checkpoint_dir = result_paths(args)["output"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / (
        f"experiment11_{regime}_k{k}_{args.target}_seed{cfg['seed']}.pt"
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
    groups = ["k", "regime", "source_training", "source_task_mode"]
    summary = frame.groupby(groups, as_index=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(columns={"rmse_count": "n_runs"})
    for metric in METRICS:
        std_col = f"{metric}_std"
        if std_col in summary:
            summary[std_col] = summary[std_col].fillna(0.0)
    redundant = [
        f"{metric}_count"
        for metric in METRICS
        if metric != "rmse" and f"{metric}_count" in summary
    ]
    return summary.drop(columns=redundant).sort_values(
        ["k", "rmse_mean"]
    ).reset_index(drop=True)


def paired_pvalue(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    if np.allclose(values, values[0]):
        return 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def wilcoxon_pvalue(values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, 0.0):
        return 1.0 if len(values) >= 2 else float("nan")
    try:
        return float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(stats.sem(values))
    if not math.isfinite(sem) or sem == 0:
        return mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - half, mean + half


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted.tolist()
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparisons(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    paired_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for candidate, reference, label in COMPARISONS:
        for k in sorted(frame.k.unique()):
            left = frame[(frame.k == k) & (frame.regime == candidate)]
            right = frame[(frame.k == k) & (frame.regime == reference)]
            merged = left.merge(
                right, on=["seed", "k"], suffixes=("_candidate", "_reference")
            )
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
                    candidate_value = float(row[f"{metric}_candidate"])
                    reference_value = float(row[f"{metric}_reference"])
                    item[f"{metric}_candidate"] = candidate_value
                    item[f"{metric}_reference"] = reference_value
                    item[f"{metric}_delta"] = candidate_value - reference_value
                paired_rows.append(item)

            subset = [
                row
                for row in paired_rows
                if row["k"] == int(k) and row["comparison"] == label
            ]
            result = {
                "k": int(k),
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "n_pairs": len(subset),
                "is_primary_comparison": label in PRIMARY_COMPARISONS,
            }
            for metric in METRICS:
                deltas = np.asarray(
                    [row[f"{metric}_delta"] for row in subset], dtype=float
                )
                references = np.asarray(
                    [row[f"{metric}_reference"] for row in subset], dtype=float
                )
                lower_is_better = metric != "r2"
                wins = deltas < 0 if lower_is_better else deltas > 0
                ci_low, ci_high = confidence_interval(deltas)
                result[f"{metric}_delta_mean"] = float(deltas.mean())
                result[f"{metric}_delta_ci95_low"] = ci_low
                result[f"{metric}_delta_ci95_high"] = ci_high
                result[f"{metric}_win_rate"] = float(wins.mean())
                result[f"{metric}_paired_t_p"] = paired_pvalue(deltas)
                result[f"{metric}_wilcoxon_p"] = wilcoxon_pvalue(deltas)
                if lower_is_better and not np.isclose(references.mean(), 0.0):
                    result[f"{metric}_improvement_pct"] = float(
                        -100.0 * deltas.mean() / references.mean()
                    )
                else:
                    result[f"{metric}_improvement_pct"] = float(deltas.mean())
            comparison_rows.append(result)

    paired = pd.DataFrame(paired_rows)
    comparisons = pd.DataFrame(comparison_rows)
    if comparisons.empty:
        return paired, comparisons
    for metric in METRICS:
        p_col = f"{metric}_paired_t_p"
        comparisons[f"{metric}_paired_t_p_holm"] = holm_adjust(comparisons[p_col])

    comparisons["meets_rmse_effect_3pct"] = (
        comparisons["rmse_improvement_pct"] >= 3.0
    )
    comparisons["meets_rmse_win_rate_80pct"] = (
        comparisons["rmse_win_rate"] >= 0.80
    )
    comparisons["meets_rmse_holm_p_005"] = (
        comparisons["rmse_paired_t_p_holm"] < 0.05
    )
    comparisons["nasa_not_worse"] = comparisons["nasa_score_delta_mean"] <= 0
    comparisons["strict_success"] = (
        comparisons["is_primary_comparison"]
        & comparisons["meets_rmse_effect_3pct"]
        & comparisons["meets_rmse_win_rate_80pct"]
        & comparisons["meets_rmse_holm_p_005"]
        & comparisons["nasa_not_worse"]
    )
    return (
        paired.sort_values(["k", "comparison", "seed"]).reset_index(drop=True),
        comparisons.sort_values(["k", "comparison"]).reset_index(drop=True),
    )


def save_progress(results: list[dict], args: argparse.Namespace) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    atomic_write_text(
        paths["summary"], summarize(results).to_csv(index=False), encoding="utf-8-sig"
    )
    paired, comparisons = paired_comparisons(results)
    atomic_write_text(
        paths["paired"], paired.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"],
        comparisons.to_csv(index=False),
        encoding="utf-8-sig",
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
    copied["experiment11_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path else "regenerated"
    )
    copied["experiment11_hypothesis"] = (
        "Engine-disjoint source support/query tasks improve ANIL beyond ordinary "
        "pretraining and a source-budget-matched control."
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
    extra = budget_extra_steps(args, cfg)
    budget = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "k_values": k_values,
        "seeds": seeds,
        "regimes": regimes,
        "source_domains": cfg["source_domains"],
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "ordinary_pretraining_steps": args.source_pretrain_steps,
        "ordinary_budget_extra_steps": extra,
        "ordinary_budget_total_steps": args.source_pretrain_steps + extra,
        "meta_epochs": args.meta_epochs,
        "tasks_per_meta_batch": task_count,
        "meta_inner_steps": args.meta_inner_steps,
        "meta_inner_lr": args.meta_inner_lr,
        "anil_query_batches": args.anil_query_batches,
        "anil_support_loss_gradient_batches": (
            args.meta_epochs * task_count * args.meta_inner_steps
        ),
        "anil_query_loss_gradient_batches": (
            args.meta_epochs * task_count * args.anil_query_batches
        ),
        "anil_outer_optimizer_steps": args.meta_epochs,
        "source_query_fraction": args.source_query_fraction,
        "source_task_seed": args.source_task_seed,
        "vary_source_split_by_seed": args.vary_source_split_by_seed,
        "target_epochs_equal_for_all_regimes": args.target_epochs,
        "target_lr_equal_for_all_regimes": args.target_lr,
        "target_scope_equal_for_all_regimes": "predictor.* only",
        "target_loss_equal_for_all_regimes": "raw_mse",
        "selection_rule": "best target epoch selected only by fixed validation engines",
        "official_test_rule": (
            "official test evaluated after validation selection; never used for tuning"
        ),
        "primary_success_rule": {
            "k_focus": [2, 5],
            "rmse_improvement_pct_at_least": 3.0,
            "paired_seed_win_rate_at_least": 0.80,
            "holm_adjusted_p_below": 0.05,
            "nasa_score": "must not worsen",
        },
        "budget_matching_note": (
            "The ordinary budget control matches ANIL support/query loss-gradient "
            "batch count, not optimizer dynamics or wall-clock time."
        ),
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))
    return paths


def inspect_protocol(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    k_values: list[int],
) -> tuple[list[dict], dict, dict]:
    diagnostics: list[dict] = []
    for k in k_values:
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
        diagnostics.append(
            {
                "seed": seed,
                "k": k,
                "feature_count": feature_count,
                "source_example_shape": list(x.shape),
                "source_label_shape": list(y.shape),
                "forward_output_shape": list(output.shape),
                "adaptation_units": units,
                "support_engine_count": len(units),
                "validation_engine_count": len(split_info["validation_units"]),
                "official_test_engine_count": len(test.dataset),
                "official_test_units_hash": split_info["official_test_units_hash"],
                "support_batches": len(support),
                "validation_batches": len(validation),
                "test_batches": len(test),
            }
        )

    source_tasks, feature_count = fresh_source_tasks(args, cfg, protocol, seed)
    _, source_manifest = split_source_tasks_by_engine(
        source_tasks,
        args.balance_mode,
        args.source_query_fraction,
        source_split_seed(args, seed),
    )
    model = build_model("meta_gnn", feature_count, cfg).cpu()
    inventory = parameter_inventory(model, 0.0)
    return diagnostics, source_manifest, inventory


def main() -> None:
    args = parse_args()
    k_values, seeds, regimes = validate_args(args)
    first_cfg = load_config(args, seeds[0])
    protocol, source_protocol_path = load_or_extend_protocol(
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
    print("\n[实验11固定协议与训练预算]")
    print(paths["budget"].read_text(encoding="utf-8"))

    if args.dry_run:
        diagnostics, source_manifest, inventory = inspect_protocol(
            args, first_cfg, protocol, seeds[0], k_values
        )
        print("\n[目标域与前向检查]")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        print("\n[发动机互斥源任务检查]")
        print(json.dumps(source_manifest, ensure_ascii=False, indent=2))
        atomic_write_text(
            paths["source_splits"],
            json.dumps({str(seeds[0]): source_manifest}, ensure_ascii=False, indent=2),
        )
        atomic_write_text(
            paths["parameters"], json.dumps(inventory, ensure_ascii=False, indent=2)
        )
        print("\n[dry-run完成] 未训练模型。")
        print(
            f"Protocol: {paths['protocol']}\nSplits: {paths['splits']}"
            f"\nBudget: {paths['budget']}\nSource splits: {paths['source_splits']}"
            f"\nParameters: {paths['parameters']}"
        )
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条结果。")
    done = completed_keys(results)
    all_source_diagnostics: dict[str, dict] = {}
    all_source_splits: dict[str, dict] = {}
    if args.resume and paths["source_diagnostics"].is_file():
        all_source_diagnostics = json.loads(
            paths["source_diagnostics"].read_text(encoding="utf-8")
        )
    if args.resume and paths["source_splits"].is_file():
        all_source_splits = json.loads(
            paths["source_splits"].read_text(encoding="utf-8")
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
        states, histories, source_diags, source_splits, inventory = build_source_states(
            args, cfg, protocol, seed, required
        )
        all_source_diagnostics[str(seed)] = source_diags
        if source_splits:
            all_source_splits[str(seed)] = source_splits
        atomic_write_text(
            paths["source_diagnostics"],
            json.dumps(all_source_diagnostics, ensure_ascii=False, indent=2),
        )
        atomic_write_text(
            paths["source_splits"],
            json.dumps(all_source_splits, ensure_ascii=False, indent=2),
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
                    f"\n[experiment11] seed={seed} K={k} regime={regime} "
                    f"engines={units}"
                )
                result = run_target_regime(
                    args,
                    cfg,
                    protocol,
                    regime,
                    states[regime],
                    histories[regime],
                    inventory,
                    k,
                    units,
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
    print("\n[实验11汇总]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        print("\n[配对比较：RMSE/MAE/NASA delta<0，R2 delta>0表示候选更好]")
        print(comparisons.to_string(index=False))
    print("\n[结论判定]")
    print("1. engine-disjoint优于batch-ANIL：支持源任务构造确实影响元学习。")
    print("2. engine-disjoint优于ordinary-head：说明ANIL可能具有普通迁移之外的收益。")
    print("3. 还需优于budget-head，才能排除只是额外源训练量带来的提升。")
    print("4. 重点看K=2/5：RMSE至少改善3%、种子胜率≥80%、Holm校正p<0.05。")
    print("5. 若RMSE改善但NASA Score变差，不能称为全面改进。")
    print("6. 本次官方测试结果不得用于继续调参；新假设应建立新的实验编号。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
        f"\nBudget: {paths['budget']}\nSource splits: {paths['source_splits']}"
        f"\nSource diagnostics: {paths['source_diagnostics']}"
        f"\nParameters: {paths['parameters']}"
    )


if __name__ == "__main__":
    main()
