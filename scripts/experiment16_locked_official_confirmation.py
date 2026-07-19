#!/usr/bin/env python3
"""实验16：锁定参数后的FD004官方测试集确认实验。

目的
----
实验15只在固定validation（验证集）上比较了预算匹配普通迁移、当前ANIL和
CVaR尾部风险方案。实验16不再搜索任何超参数，而是读取实验15已经锁定的
协议与结论，重新播放目标域适应过程并核对验证结果；核对通过后，才在
C-MAPSS FD004官方test上执行一次最终评估。

正式比较的三个方案：

``budget_mse_rmse``
    预算匹配普通源域预训练 + 只微调RUL预测头。
``anil_mse_rmse``
    发动机互斥ANIL源状态 + MSE目标适应。
``anil_cvar_clip_tail``
    同一ANIL源状态 + 实验15锁定的低风险权重、梯度裁剪、发动机级CVaR和
    尾部约束epoch选择。

防止测试集泄漏
--------------
1. 正式运行前要求实验15完整网格与验证成功门槛通过；
2. 学习率、epoch、风险权重、CVaR参数和选择规则全部从实验15协议读取；
3. 每个cell先重放validation并与实验15结果核对；
4. 只有核对通过后才预测官方test；
5. 官方test结果不参与模型、epoch或超参数选择；
6. 正式运行必须显式提供 ``--confirm-official-test``。

Dry-run不会对官方test执行模型前向：

    CUDA_VISIBLE_DEVICES=2 python -u \
      scripts/experiment16_locked_official_confirmation.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --experiment15-dir outputs/experiment15_cvar_stability \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --dry-run

正式运行：

    CUDA_VISIBLE_DEVICES=2 python -u \
      scripts/experiment16_locked_official_confirmation.py \
      --target FD004 \
      --experiment12b-dir outputs/experiment12b_budget_matched_final \
      --experiment15-dir outputs/experiment15_cvar_stability \
      --k-values 2 5 \
      --model-seeds 42 43 44 45 46 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --validation-gate-policy any_k \
      --confirm-official-test \
      --output-dir outputs/experiment16_locked_official_confirmation \
      --resume
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
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
from scripts import experiment14_tail_robust_adaptation as exp14  # noqa: E402
from scripts import experiment15_cvar_stability as exp15  # noqa: E402
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


SCRIPT_VERSION = "experiment16_locked_official_confirmation_v1"
REGIMES = (
    "budget_mse_rmse",
    "anil_mse_rmse",
    "anil_cvar_clip_tail",
)
COMPARISONS = (
    (
        "anil_cvar_clip_tail",
        "anil_mse_rmse",
        "cvar_vs_current_anil",
    ),
    (
        "anil_cvar_clip_tail",
        "budget_mse_rmse",
        "cvar_vs_budget_transfer",
    ),
    (
        "anil_mse_rmse",
        "budget_mse_rmse",
        "current_anil_vs_budget_transfer",
    ),
)
PRIMARY_COMPARISON = "cvar_vs_current_anil"
INDEPENDENT_COMPARISON = "cvar_vs_budget_transfer"
LOCKED_FD004_TEST_ENGINES = 248
LOCKED_FD004_TEST_HASH = "ce6c5ef1d7149714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验16：锁定实验15参数后的FD004官方测试确认"
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
        "--experiment12b-dir", default="outputs/experiment12b_budget_matched_final"
    )
    parser.add_argument(
        "--experiment15-dir", default="outputs/experiment15_cvar_stability"
    )
    parser.add_argument("--experiment12b-protocol-file")
    parser.add_argument("--experiment12b-raw-file")
    parser.add_argument("--experiment15-grid-file")
    parser.add_argument("--experiment15-protocol-file")
    parser.add_argument("--experiment15-conclusion-file")
    parser.add_argument("--experiment15-raw-file")
    parser.add_argument("--device")

    # 这些配置必须与实验15协议相同；正式值会从实验15协议覆盖并锁定。
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

    # 仅用于正确加载实验12B源缓存和构建相同模型，不在实验16重新搜索。
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

    # 下列目标参数只是安全默认值；lock_parameters_from_experiment15会覆盖。
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--target-weight-decay", type=float, default=0.0)
    parser.add_argument("--low-nasa-loss-weight", type=float, default=0.25)
    parser.add_argument("--low-high-rul-loss-weight", type=float, default=0.05)
    parser.add_argument("--high-rul-threshold", type=float, default=90.0)
    parser.add_argument("--nasa-exp-clip", type=float, default=6.0)
    parser.add_argument("--target-clip-norm", type=float, default=1000.0)
    parser.add_argument("--engine-cvar-weight", type=float, default=0.25)
    parser.add_argument("--engine-cvar-alpha", type=float, default=0.50)
    parser.add_argument("--selection-rmse-tolerance", type=float, default=0.02)

    parser.add_argument(
        "--validation-gate-policy", choices=("any_k", "all_k"), default="any_k",
        help="any_k：实验15至少一个K严格通过；all_k：所有K均严格通过",
    )
    parser.add_argument(
        "--validation-replay-atol", type=float, default=1e-4,
        help="重放验证指标的绝对容差",
    )
    parser.add_argument(
        "--validation-replay-rtol", type=float, default=1e-5,
        help="重放验证指标的相对容差",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="outputs/experiment16_locked_official_confirmation"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--confirm-official-test", action="store_true",
        help="显式确认执行锁定后的官方test评估；正式运行必需",
    )
    return parser.parse_args()


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(int(value) for value in values))


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    candidates = list(candidates)
    for path in candidates:
        if path.is_file():
            return path
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到{description}，已检查：\n  {joined}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_experiment15_inputs(args: argparse.Namespace) -> dict[str, Path]:
    directory = resolve_path(args.experiment15_dir, PROJECT_ROOT)
    prefix = f"experiment15_{args.target}"

    def explicit_or_default(value: str | None, suffix: str, description: str) -> Path:
        if value:
            return resolve_path(value, PROJECT_ROOT)
        return first_existing([directory / f"{prefix}_{suffix}"], description)

    return {
        "directory": directory,
        "grid": explicit_or_default(
            args.experiment15_grid_file, "grid_plan.json", "实验15网格计划"
        ),
        "protocol": explicit_or_default(
            args.experiment15_protocol_file, "protocol.json", "实验15锁定协议"
        ),
        "conclusion": explicit_or_default(
            args.experiment15_conclusion_file, "conclusion.json", "实验15结论"
        ),
        "raw": explicit_or_default(
            args.experiment15_raw_file, "raw.json", "实验15 raw结果"
        ),
    }


def resolve_experiment12b_inputs(args: argparse.Namespace) -> dict[str, Path]:
    proxy = argparse.Namespace(**vars(args))
    proxy.protocol_file = args.experiment12b_protocol_file
    proxy.raw_results_file = args.experiment12b_raw_file
    return exp14.resolve_inputs(proxy)


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    prefix = f"experiment16_{args.target}"
    return {
        "output": output,
        "cells": output / "cells",
        "checkpoints": output / "checkpoints",
        "raw": output / f"{prefix}_official_raw.json",
        "history": output / f"{prefix}_target_history.json",
        "predictions": output / f"{prefix}_official_window_predictions.csv",
        "summary": output / f"{prefix}_official_summary.csv",
        "paired": output / f"{prefix}_official_paired_by_cell.csv",
        "comparisons": output / f"{prefix}_official_comparisons.csv",
        "engine": output / f"{prefix}_official_engine_deltas.csv",
        "stage": output / f"{prefix}_official_stage_deltas.csv",
        "tail": output / f"{prefix}_official_tail_diagnostics.csv",
        "model_seed": output / f"{prefix}_official_per_model_seed.csv",
        "source_split": output / f"{prefix}_official_per_source_split.csv",
        "replay_audit": output / f"{prefix}_validation_replay_audit.csv",
        "grid": output / f"{prefix}_grid_plan.json",
        "protocol": output / f"{prefix}_locked_protocol.json",
        "conclusion": output / f"{prefix}_conclusion.json",
    }


def validate_cli(args: argparse.Namespace):
    if args.target != "FD004":
        raise ValueError(
            "实验16当前预注册协议仅用于FD004；其他子集必须单独锁定协议后再运行"
        )
    k_values = unique_sorted(args.k_values)
    model_seeds = unique_sorted(args.model_seeds)
    source_seeds = unique_sorted(args.source_task_seeds)
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须为正整数")
    if not model_seeds or not source_seeds:
        raise ValueError("模型种子和源域划分种子不能为空")
    if args.validation_replay_atol < 0 or args.validation_replay_rtol < 0:
        raise ValueError("验证重放容差不能为负数")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions必须为正整数")
    if not args.dry_run and not args.confirm_official_test:
        raise PermissionError(
            "正式实验16会访问官方test。确认实验15已锁定后，请显式加入"
            " --confirm-official-test；若只检查配置，请使用--dry-run。"
        )
    return k_values, model_seeds, source_seeds


def require_equal(name: str, requested, locked) -> None:
    if requested != locked:
        raise ValueError(
            f"实验16禁止改变{name}：请求值={requested}，实验15锁定值={locked}"
        )


def load_and_lock_experiment15(
    args: argparse.Namespace,
    inputs: dict[str, Path],
    k_values: list[int],
    model_seeds: list[int],
    source_seeds: list[int],
) -> tuple[dict, dict, dict, pd.DataFrame, dict]:
    grid = json.loads(inputs["grid"].read_text(encoding="utf-8"))
    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    conclusion = json.loads(inputs["conclusion"].read_text(encoding="utf-8"))
    raw = pd.DataFrame(json.loads(inputs["raw"].read_text(encoding="utf-8")))

    if grid.get("target") != args.target or protocol.get("target") != args.target:
        raise ValueError("实验15 target与实验16 --target不一致")
    if not bool(grid.get("full_grid_complete")):
        raise RuntimeError("实验15完整网格尚未完成，禁止进入官方test")
    if int(grid.get("completed_new_target_trainings", -1)) != int(
        grid.get("planned_new_target_trainings", -2)
    ):
        raise RuntimeError("实验15完成数与计划数不一致")
    if raw.empty:
        raise ValueError("实验15 raw结果为空")
    if not bool(protocol.get("success_rules_locked_before_official_test")):
        raise RuntimeError("实验15未声明成功规则已在官方test前锁定")
    if "anil_cvar_clip_tail" not in set(grid.get("new_regimes", [])):
        raise RuntimeError("实验15网格未包含待确认的anil_cvar_clip_tail方案")
    if set(raw["evaluation_scope"].astype(str)) != {"validation"}:
        raise ValueError("实验15 raw必须全部来自validation")
    official_flag = raw.get(
        "official_test_prediction_run", pd.Series(False, index=raw.index)
    )
    if bool(official_flag.astype(bool).any()):
        raise ValueError("实验15中出现官方test预测，协议不再满足validation-only")

    locked_k = unique_sorted(grid.get("k_values", []))
    locked_model = unique_sorted(grid.get("model_seeds", []))
    locked_source = unique_sorted(grid.get("source_task_seeds", []))
    require_equal("K集合", k_values, locked_k)
    require_equal("模型种子集合", model_seeds, locked_model)
    require_equal("源域划分种子集合", source_seeds, locked_source)
    require_equal("预处理", args.preprocessing, protocol.get("preprocessing"))
    require_equal("平衡方式", args.balance_mode, protocol.get("balance_mode"))

    low = protocol.get("low_risk_loss", {})
    cvar = protocol.get("engine_cvar", {})
    locked_values = {
        "target_epochs": int(protocol["target_epochs"]),
        "target_lr": float(protocol["target_lr"]),
        "low_nasa_loss_weight": float(low["nasa_loss_weight"]),
        "low_high_rul_loss_weight": float(low["high_rul_loss_weight"]),
        "high_rul_threshold": float(low["high_rul_threshold"]),
        "nasa_exp_clip": float(low["nasa_exp_clip"]),
        "target_clip_norm": float(protocol["gradient_clip_norm"]),
        "engine_cvar_weight": float(cvar["weight"]),
        "engine_cvar_alpha": float(cvar["alpha"]),
        "selection_rmse_tolerance": float(
            protocol["tail_selection_rmse_tolerance"]
        ),
    }
    for name, value in locked_values.items():
        setattr(args, name, value)
    # MSE基线不会使用这两个权重，但exp14接口需要这些属性。
    args.nasa_loss_weight = args.low_nasa_loss_weight
    args.high_rul_loss_weight = args.low_high_rul_loss_weight

    gate_by_k = {}
    for k in k_values:
        item = conclusion.get("by_k", {}).get(str(k))
        if item is None:
            raise KeyError(f"实验15结论缺少K={k}")
        gate_by_k[str(k)] = bool(item.get("strict_success", False))
    if args.validation_gate_policy == "all_k":
        gate_passed = bool(gate_by_k) and all(gate_by_k.values())
    else:
        gate_passed = any(gate_by_k.values())
    gate = {
        "policy": args.validation_gate_policy,
        "by_k": gate_by_k,
        "passed": bool(gate_passed),
    }
    if not gate_passed and not args.dry_run:
        raise RuntimeError(
            f"实验15验证门槛未通过：{gate}。根据预注册规则，不应访问官方test。"
        )

    expected = {
        "budget_mse_rmse": len(k_values) * len(model_seeds),
        "anil_mse_rmse": len(k_values) * len(model_seeds) * len(source_seeds),
        "anil_cvar_clip_tail": (
            len(k_values) * len(model_seeds) * len(source_seeds)
        ),
    }
    counts = {}
    for regime, count in expected.items():
        selected = raw[raw.regime.eq(regime)]
        counts[regime] = int(len(selected))
        if len(selected) != count:
            raise ValueError(
                f"实验15方案{regime}结果不完整：{len(selected)}/{count}"
            )
    duplicates = raw[raw.regime.isin(REGIMES)].copy()
    budget = duplicates[duplicates.regime.eq("budget_mse_rmse")]
    if bool(budget.duplicated(["k", "model_seed"]).any()):
        raise ValueError("实验15 budget_mse_rmse存在重复cell")
    anil = duplicates[~duplicates.regime.eq("budget_mse_rmse")]
    if bool(anil.duplicated(
        ["regime", "k", "model_seed", "source_split_seed"]
    ).any()):
        raise ValueError("实验15 ANIL结果存在重复cell")

    hashes = {name: sha256_file(path) for name, path in inputs.items() if path.is_file()}
    audit = {
        "locked_values": locked_values,
        "validation_gate": gate,
        "result_counts": counts,
        "input_sha256": hashes,
    }
    return grid, protocol, conclusion, raw, audit


def expected_validation_row(
    raw: pd.DataFrame,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
) -> dict:
    selected = raw[
        raw.regime.eq(regime)
        & raw.k.astype(int).eq(int(k))
        & raw.model_seed.astype(int).eq(int(model_seed))
    ]
    if regime != "budget_mse_rmse":
        selected = selected[
            selected.source_split_seed.astype(int).eq(int(source_seed))
        ]
    if len(selected) != 1:
        raise KeyError(
            f"无法唯一定位实验15 cell：regime={regime}, K={k}, "
            f"model={model_seed}, source={source_seed}, rows={len(selected)}"
        )
    return selected.iloc[0].to_dict()


def validation_replay_audit(
    actual: dict,
    expected: dict,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
    best_epoch: int,
    args: argparse.Namespace,
) -> dict:
    row = {
        "regime": regime,
        "k": int(k),
        "model_seed": int(model_seed),
        "source_split_seed": int(source_seed),
        "expected_best_epoch": int(expected["best_target_epoch_by_validation"]),
        "replayed_best_epoch": int(best_epoch),
    }
    checks = [best_epoch == int(expected["best_target_epoch_by_validation"])]
    for metric in ("rmse", "mae", "r2", "nasa_score"):
        expected_value = float(expected[metric])
        actual_value = float(actual[metric])
        difference = actual_value - expected_value
        close = bool(
            np.isclose(
                actual_value,
                expected_value,
                atol=args.validation_replay_atol,
                rtol=args.validation_replay_rtol,
            )
        )
        row[f"expected_{metric}"] = expected_value
        row[f"replayed_{metric}"] = actual_value
        row[f"{metric}_difference"] = difference
        row[f"{metric}_close"] = close
        checks.append(close)
    row["replay_passed"] = bool(all(checks))
    return row


def cell_signature(
    args: argparse.Namespace,
    experiment15_audit: dict,
    regime: str,
) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "regime": regime,
        "target": args.target,
        "target_epochs": args.target_epochs,
        "target_lr": args.target_lr,
        "target_weight_decay": args.target_weight_decay,
        "low_nasa_loss_weight": args.low_nasa_loss_weight,
        "low_high_rul_loss_weight": args.low_high_rul_loss_weight,
        "high_rul_threshold": args.high_rul_threshold,
        "nasa_exp_clip": args.nasa_exp_clip,
        "target_clip_norm": args.target_clip_norm,
        "engine_cvar_weight": args.engine_cvar_weight,
        "engine_cvar_alpha": args.engine_cvar_alpha,
        "selection_rmse_tolerance": args.selection_rmse_tolerance,
        "experiment15_protocol_sha256": experiment15_audit["input_sha256"]["protocol"],
        "experiment15_raw_sha256": experiment15_audit["input_sha256"]["raw"],
    }


def cell_paths(
    paths: dict[str, Path], regime: str, k: int,
    model_seed: int, source_seed: int,
) -> tuple[Path, Path]:
    stem = f"{regime}_k{k}_source{source_seed}_model{model_seed}"
    return paths["cells"] / f"{stem}.json", paths["cells"] / f"{stem}.csv"


def cached_cell(
    args: argparse.Namespace,
    paths: dict[str, Path],
    experiment15_audit: dict,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
):
    metadata, predictions = cell_paths(
        paths, regime, k, model_seed, source_seed
    )
    if not (metadata.is_file() and predictions.is_file()):
        return None
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("signature") != cell_signature(
        args, experiment15_audit, regime
    ):
        return None
    return (
        payload["result"],
        payload.get("history", []),
        payload["replay_audit"],
        pd.read_csv(predictions),
    )


def save_cell(
    args: argparse.Namespace,
    paths: dict[str, Path],
    experiment15_audit: dict,
    regime: str,
    k: int,
    model_seed: int,
    source_seed: int,
    result: dict,
    history: list[dict],
    replay_audit: dict,
    predictions: pd.DataFrame,
) -> None:
    metadata, prediction_path = cell_paths(
        paths, regime, k, model_seed, source_seed
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    exp14.atomic_to_csv(predictions, prediction_path)
    atomic_write_text(
        metadata,
        json.dumps(
            {
                "signature": cell_signature(args, experiment15_audit, regime),
                "result": result,
                "history": history,
                "replay_audit": replay_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def train_locked_model(
    args: argparse.Namespace,
    cfg: dict,
    regime: str,
    source_state: dict,
    unit_schedule,
    validation,
    feature_count: int,
    model_seed: int,
):
    seed_everything(model_seed)
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(source_state)
    device = resolve_device(cfg["device"])
    if regime in {"budget_mse_rmse", "anil_mse_rmse"}:
        mse_args = deepcopy(args)
        mse_args.target_clip_norm = 0.0
        schedule = [
            [(x, y) for x, y, _ in epoch_batches]
            for epoch_batches in unit_schedule
        ]
        return exp14.train_target_robust(
            model,
            schedule,
            validation,
            mse_args,
            device,
            loss_mode="mse",
            selection_mode="rmse",
        )
    if regime == "anil_cvar_clip_tail":
        spec = exp15.regime_spec(regime, args)
        return exp15.train_target(
            model, unit_schedule, validation, args, device, spec
        )
    raise ValueError(f"未知实验16方案：{regime}")


def run_official_cell(
    args: argparse.Namespace,
    paths: dict[str, Path],
    experiment15_audit: dict,
    experiment15_raw: pd.DataFrame,
    cfg: dict,
    regime: str,
    source_state: dict,
    unit_schedule,
    validation,
    official_test,
    split_info: dict,
    feature_count: int,
    k: int,
    model_seed: int,
    source_seed: int,
):
    if args.resume:
        cached = cached_cell(
            args, paths, experiment15_audit, regime,
            k, model_seed, source_seed,
        )
        if cached is not None:
            print(
                f"[resume] regime={regime} K={k} model={model_seed} "
                f"source={source_seed}"
            )
            return cached

    expected = expected_validation_row(
        experiment15_raw, regime, k, model_seed, source_seed
    )
    model, history, best_epoch, diagnostics, selected_state = train_locked_model(
        args, cfg, regime, source_state, unit_schedule,
        validation, feature_count, model_seed,
    )
    device = resolve_device(cfg["device"])
    replay_frame = exp14.predict_validation(model, validation, device)
    replay_metrics = exp14.prediction_metrics(
        replay_frame, args.high_rul_threshold
    )
    replay = validation_replay_audit(
        replay_metrics, expected, regime, k, model_seed, source_seed,
        best_epoch, args,
    )
    if not replay["replay_passed"]:
        raise RuntimeError(
            "验证重放与实验15不一致，已在官方test前中止：\n"
            + json.dumps(replay, ensure_ascii=False, indent=2)
        )

    # 重要：best epoch完全由validation确定。下面才是第一次官方test前向。
    official_frame = exp14.predict_validation(model, official_test, device)
    official_metrics = exp14.prediction_metrics(
        official_frame, args.high_rul_threshold
    )
    official_frame.insert(0, "evaluation_scope", "official_test")
    official_frame.insert(0, "source_split_seed", int(source_seed))
    official_frame.insert(0, "model_seed", int(model_seed))
    official_frame.insert(0, "k", int(k))
    official_frame.insert(0, "regime", regime)

    source_key = (
        "ordinary_budget" if regime == "budget_mse_rmse"
        else "anil_engine_disjoint"
    )
    result = {
        **official_metrics,
        "evaluation_scope": "official_test",
        "official_test_prediction_run": True,
        "regime": regime,
        "source_state_key": source_key,
        "target_domain": args.target,
        "k": int(k),
        "model_seed": int(model_seed),
        "source_split_seed": int(source_seed),
        "adaptation_engine_count": int(k),
        "adaptation_units": [int(unit) for unit in split_info["adaptation_units"]],
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(official_test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": int(args.target_epochs),
        "target_lr": float(args.target_lr),
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_replay_passed": True,
        **{f"validation_{key}": value for key, value in replay_metrics.items()},
        **diagnostics,
    }
    save_cell(
        args, paths, experiment15_audit, regime, k, model_seed, source_seed,
        result, history, replay, official_frame,
    )
    if args.save_checkpoints:
        paths["checkpoints"].mkdir(parents=True, exist_ok=True)
        checkpoint = paths["checkpoints"] / (
            f"experiment16_{regime}_k{k}_{args.target}_"
            f"source{source_seed}_model{model_seed}.pt"
        )
        torch.save(
            {
                "model": selected_state,
                "result": result,
                "history": history,
                "replay_audit": replay,
                "signature": cell_signature(args, experiment15_audit, regime),
            },
            checkpoint,
        )
    print(
        f"[official cell completed] regime={regime} K={k} "
        f"model={model_seed} source={source_seed}; "
        "未使用test结果选择任何配置"
    )
    return result, history, replay, official_frame


def comparison_outputs(raw: pd.DataFrame, predictions: pd.DataFrame, repetitions: int):
    old = exp14.COMPARISONS
    try:
        exp14.COMPARISONS = COMPARISONS
        paired = exp14.build_paired_cells(raw)
        comparisons = exp14.comparison_summary(paired, repetitions)
        engines, stages = exp14.detailed_deltas(predictions)
        tails = exp14.tail_diagnostics(paired, engines, stages)
    finally:
        exp14.COMPARISONS = old
    comparisons, per_model, per_source = exp15.add_factor_robustness(
        comparisons, paired
    )
    return paired, comparisons, engines, stages, tails, per_model, per_source


def make_conclusion(
    comparisons: pd.DataFrame,
    tails: pd.DataFrame,
    validation_gate: dict,
) -> dict:
    conclusion = {
        "script_version": SCRIPT_VERSION,
        "evaluation_scope": "locked_official_test",
        "official_test_used_for_selection": False,
        "validation_gate": validation_gate,
        "primary_comparison": PRIMARY_COMPARISON,
        "independent_comparison": INDEPENDENT_COMPARISON,
        "confirmation_rule": {
            "rmse_degradation_at_most_pct": 1.0,
            "mean_nasa_delta_below": 0.0,
            "cell_nasa_win_rate_at_least": 0.8,
            "nasa_bootstrap_ci95_upper_below": 0.0,
            "model_seed_nasa_win_rate_at_least": 0.8,
            "source_split_nasa_win_rate_at_least": 0.8,
        },
        "by_k": {},
        "independent_advantage_by_k": {},
    }
    primary = comparisons[comparisons.comparison.eq(PRIMARY_COMPARISON)]
    independent = comparisons[
        comparisons.comparison.eq(INDEPENDENT_COMPARISON)
    ]
    tail_primary = tails[tails.comparison.eq(PRIMARY_COMPARISON)]

    for _, row in primary.iterrows():
        k = int(row.k)
        checks = {
            "rmse_preserved": bool(row.rmse_change_pct <= 1.0),
            "mean_nasa_improved": bool(row.nasa_score_delta_mean < 0),
            "cell_nasa_win_rate": bool(row.nasa_win_rate >= 0.8),
            "nasa_ci_below_zero": bool(row.nasa_boot_ci95_high < 0),
            "model_seed_nasa_win_rate": bool(
                row.model_seed_nasa_win_rate >= 0.8
            ),
            "source_split_nasa_win_rate": bool(
                row.source_split_nasa_win_rate >= 0.8
            ),
        }
        tail = tail_primary[tail_primary.k.eq(k)]
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
                float(tail.iloc[0].seed46_nasa_delta_mean)
                if not tail.empty else None
            ),
            "unit48_nasa_delta_mean": (
                float(tail.iloc[0].unit48_nasa_delta_mean)
                if not tail.empty else None
            ),
            "checks": checks,
            "official_confirmation_passed": bool(all(checks.values())),
        }

    for _, row in independent.iterrows():
        k = int(row.k)
        checks = {
            "rmse_mean_improved": bool(row.rmse_delta_mean < 0),
            "rmse_ci_below_zero": bool(row.rmse_boot_ci95_high < 0),
            "nasa_mean_not_worse": bool(row.nasa_score_delta_mean <= 0),
            "nasa_ci_below_zero": bool(row.nasa_boot_ci95_high < 0),
        }
        conclusion["independent_advantage_by_k"][str(k)] = {
            "rmse_delta_mean": float(row.rmse_delta_mean),
            "nasa_score_delta_mean": float(row.nasa_score_delta_mean),
            "checks": checks,
            "independent_anil_advantage": bool(all(checks.values())),
        }
    conclusion["any_k_official_confirmation"] = any(
        item["official_confirmation_passed"]
        for item in conclusion["by_k"].values()
    )
    conclusion["any_k_independent_anil_advantage"] = any(
        item["independent_anil_advantage"]
        for item in conclusion["independent_advantage_by_k"].values()
    )
    return conclusion


def dry_run_report(
    args: argparse.Namespace,
    exp12_inputs: dict[str, Path],
    exp15_inputs: dict[str, Path],
    protocol12: dict,
    experiment15_raw: pd.DataFrame,
    experiment15_audit: dict,
    k_values: list[int],
    model_seeds: list[int],
    source_seeds: list[int],
) -> None:
    cache_rows = []
    for model_seed in model_seeds:
        budget_path = exp14.source_cache_path(
            exp12_inputs["experiment"], args.target,
            "ordinary_budget", model_seed, 0,
        )
        cache_rows.append(
            {
                "source_state": "ordinary_budget",
                "model_seed": model_seed,
                "source_split_seed": 0,
                "available": budget_path.is_file(),
                "path": str(budget_path),
            }
        )
        for source_seed in source_seeds:
            path = exp14.source_cache_path(
                exp12_inputs["experiment"], args.target,
                "anil_engine_disjoint", model_seed, source_seed,
            )
            cache_rows.append(
                {
                    "source_state": "anil_engine_disjoint",
                    "model_seed": model_seed,
                    "source_split_seed": source_seed,
                    "available": path.is_file(),
                    "path": str(path),
                }
            )
    caches = pd.DataFrame(cache_rows)
    planned = (
        len(k_values) * len(model_seeds)
        + 2 * len(k_values) * len(model_seeds) * len(source_seeds)
    )
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "dry_run_no_official_forward",
        "official_test_prediction_will_run": False,
        "confirm_official_test_received": bool(args.confirm_official_test),
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "regimes": list(REGIMES),
        "planned_locked_target_replays": planned,
        "experiment15_validation_rows_used": len(
            experiment15_raw[experiment15_raw.regime.isin(REGIMES)]
        ),
        "validation_gate": experiment15_audit["validation_gate"],
        "locked_parameters": experiment15_audit["locked_values"],
        "source_cache_count": len(caches),
        "source_cache_available": int(caches.available.sum()),
        "validation_engine_count": len(protocol12["validation_units"]),
        "official_test_expected_count": protocol12["official_test_engine_count"],
        "official_test_expected_hash": protocol12["official_test_units_hash"],
        "experiment15_dir": str(exp15_inputs["directory"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n[源缓存可用性]")
    print(caches.groupby("source_state").available.agg(["sum", "count"]).to_string())
    if not bool(caches.available.all()):
        print("\n[缺失缓存]")
        print(caches[~caches.available].to_string(index=False))
        return

    first_model, first_k = model_seeds[0], k_values[0]
    cfg = exp14.build_config(args, first_model)
    units = protocol12["nested_adaptation_units_by_seed"][str(first_model)][str(first_k)]
    source_tasks, support, validation, official_test, feature_count, split_info = (
        prepare_kshot_experiment(
            cfg, args.preprocessing, args.balance_mode,
            protocol12["validation_units"], units,
        )
    )
    del source_tasks
    schedule = exp15.materialize_support_schedule_with_units(support, 1)
    x, _, batch_units = schedule[0][0]
    state = exp14.load_source_state(
        exp14.source_cache_path(
            exp12_inputs["experiment"], args.target,
            "ordinary_budget", first_model, 0,
        )
    )
    model = build_model("meta_gnn", feature_count, cfg)
    model.load_state_dict(state)
    with torch.no_grad():
        support_output_shape = list(model(x[: min(8, len(x))]).shape)
    print(
        json.dumps(
            {
                "feature_count": feature_count,
                "support_windows": len(support.dataset),
                "validation_windows": len(validation.dataset),
                "official_test_rows": len(official_test.dataset),
                "first_batch_shape": list(x.shape),
                "first_batch_engine_count": int(torch.unique(batch_units).numel()),
                "support_forward_output_shape": support_output_shape,
                "official_test_hash": split_info["official_test_units_hash"],
                "official_test_forward_run": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    k_values, model_seeds, source_seeds = validate_cli(args)
    exp15_inputs = resolve_experiment15_inputs(args)
    grid15, protocol15, conclusion15, raw15, experiment15_audit = (
        load_and_lock_experiment15(
            args, exp15_inputs, k_values, model_seeds, source_seeds
        )
    )
    exp12_inputs = resolve_experiment12b_inputs(args)
    protocol12, _ = exp14.load_protocol_and_raw(exp12_inputs, args)

    if protocol12.get("official_test_engine_count") != LOCKED_FD004_TEST_ENGINES:
        raise ValueError("实验12B协议的FD004官方test发动机数量不是248")
    if protocol12.get("official_test_units_hash") != LOCKED_FD004_TEST_HASH:
        raise ValueError("实验12B协议的FD004官方test发动机hash发生变化")
    if protocol15.get("validation_units") != protocol12.get("validation_units"):
        raise ValueError("实验15与实验12B使用了不同验证发动机")
    for model_seed in model_seeds:
        nested = protocol12["nested_adaptation_units_by_seed"].get(str(model_seed), {})
        for k in k_values:
            if str(k) not in nested:
                raise KeyError(f"实验12B协议缺少model_seed={model_seed}, K={k}")

    if args.dry_run:
        dry_run_report(
            args, exp12_inputs, exp15_inputs, protocol12, raw15,
            experiment15_audit, k_values, model_seeds, source_seeds,
        )
        if not experiment15_audit["validation_gate"]["passed"]:
            print("\n[停止建议] 实验15验证门槛未通过，不应正式访问官方test。")
        else:
            print("\n[dry-run通过] 可加入--confirm-official-test执行正式实验16。")
        return

    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    paths["cells"].mkdir(parents=True, exist_ok=True)
    results = []
    histories = []
    replay_rows = []
    prediction_frames = []

    for model_seed in model_seeds:
        cfg = exp14.build_config(args, model_seed)
        for k in k_values:
            adaptation_units = protocol12["nested_adaptation_units_by_seed"][
                str(model_seed)
            ][str(k)]
            source_tasks, support, validation, official_test, feature_count, split_info = (
                prepare_kshot_experiment(
                    cfg, args.preprocessing, args.balance_mode,
                    protocol12["validation_units"], adaptation_units,
                )
            )
            del source_tasks
            if split_info["validation_units"] != protocol12["validation_units"]:
                raise AssertionError("固定验证发动机发生变化")
            if len(official_test.dataset) != LOCKED_FD004_TEST_ENGINES:
                raise AssertionError("官方test样本数不是248")
            if split_info["official_test_units_hash"] != LOCKED_FD004_TEST_HASH:
                raise AssertionError("官方test发动机hash发生变化")
            unit_schedule = exp15.materialize_support_schedule_with_units(
                support, args.target_epochs
            )

            # 预算匹配普通迁移每个(model_seed, K)只运行一次。
            budget_state = exp14.load_source_state(
                exp14.source_cache_path(
                    exp12_inputs["experiment"], args.target,
                    "ordinary_budget", model_seed, 0,
                )
            )
            print(
                f"\n[experiment16] regime=budget_mse_rmse K={k} "
                f"model={model_seed} source=0"
            )
            result, history, replay, predictions = run_official_cell(
                args, paths, experiment15_audit, raw15, cfg,
                "budget_mse_rmse", budget_state, unit_schedule,
                validation, official_test, split_info, feature_count,
                k, model_seed, 0,
            )
            results.append(result)
            histories.append({
                "regime": "budget_mse_rmse", "k": k,
                "model_seed": model_seed, "source_split_seed": 0,
                "epochs": history,
            })
            replay_rows.append(replay)
            prediction_frames.append(predictions)

            for source_seed in source_seeds:
                anil_state = exp14.load_source_state(
                    exp14.source_cache_path(
                        exp12_inputs["experiment"], args.target,
                        "anil_engine_disjoint", model_seed, source_seed,
                    )
                )
                for regime in ("anil_mse_rmse", "anil_cvar_clip_tail"):
                    print(
                        f"\n[experiment16] regime={regime} K={k} "
                        f"model={model_seed} source={source_seed}"
                    )
                    result, history, replay, predictions = run_official_cell(
                        args, paths, experiment15_audit, raw15, cfg,
                        regime, anil_state, unit_schedule,
                        validation, official_test, split_info, feature_count,
                        k, model_seed, source_seed,
                    )
                    results.append(result)
                    histories.append({
                        "regime": regime, "k": k,
                        "model_seed": model_seed,
                        "source_split_seed": source_seed,
                        "epochs": history,
                    })
                    replay_rows.append(replay)
                    prediction_frames.append(predictions)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del unit_schedule, support, validation, official_test

    raw = pd.DataFrame(results)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    replay_audit = pd.DataFrame(replay_rows)
    if not bool(replay_audit.replay_passed.all()):
        raise RuntimeError("存在未通过的validation重放cell，拒绝汇总官方结果")

    expected_cells = (
        len(k_values) * len(model_seeds)
        + 2 * len(k_values) * len(model_seeds) * len(source_seeds)
    )
    if len(raw) != expected_cells:
        raise AssertionError(f"实验16结果网格不完整：{len(raw)}/{expected_cells}")

    summary = exp14.summary_table(raw)
    (
        paired, comparisons, engines, stages, tails,
        per_model, per_source,
    ) = comparison_outputs(raw, predictions, args.bootstrap_repetitions)
    conclusion = make_conclusion(
        comparisons, tails, experiment15_audit["validation_gate"]
    )

    atomic_write_text(
        paths["raw"], json.dumps(results, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["history"], json.dumps(histories, ensure_ascii=False, indent=2)
    )
    exp14.atomic_to_csv(predictions, paths["predictions"])
    exp14.atomic_to_csv(summary, paths["summary"])
    exp14.atomic_to_csv(paired, paths["paired"])
    exp14.atomic_to_csv(comparisons, paths["comparisons"])
    exp14.atomic_to_csv(engines, paths["engine"])
    exp14.atomic_to_csv(stages, paths["stage"])
    exp14.atomic_to_csv(tails, paths["tail"])
    exp14.atomic_to_csv(per_model, paths["model_seed"])
    exp14.atomic_to_csv(per_source, paths["source_split"])
    exp14.atomic_to_csv(replay_audit, paths["replay_audit"])

    grid = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "evaluation_scope": "locked_official_test",
        "k_values": k_values,
        "model_seeds": model_seeds,
        "source_task_seeds": source_seeds,
        "regimes": list(REGIMES),
        "planned_cells": expected_cells,
        "completed_cells": len(raw),
        "validation_replay_passed_cells": int(replay_audit.replay_passed.sum()),
        "full_grid_complete": len(raw) == expected_cells,
        "official_test_engine_count": LOCKED_FD004_TEST_ENGINES,
        "official_test_units_hash": LOCKED_FD004_TEST_HASH,
    }
    locked_protocol = {
        **grid,
        "experiment12b_dir": str(exp12_inputs["experiment"]),
        "experiment15_dir": str(exp15_inputs["directory"]),
        "experiment15_input_sha256": experiment15_audit["input_sha256"],
        "experiment15_validation_gate": experiment15_audit["validation_gate"],
        "locked_parameters": experiment15_audit["locked_values"],
        "target_scope": "predictor.* only",
        "epoch_selected_by": "fixed_validation_only",
        "official_test_used_for_selection": False,
        "official_test_evaluations_per_selected_model": 1,
    }
    atomic_write_text(paths["grid"], json.dumps(grid, ensure_ascii=False, indent=2))
    atomic_write_text(
        paths["protocol"], json.dumps(locked_protocol, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["conclusion"], json.dumps(conclusion, ensure_ascii=False, indent=2)
    )

    print("\n[实验16官方测试汇总]")
    print(summary.to_string(index=False))
    print("\n[实验16配对比较]")
    print(comparisons.to_string(index=False))
    print("\n[实验16结论]")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print("\n[输出文件]")
    for name, path in paths.items():
        if path.is_file():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
