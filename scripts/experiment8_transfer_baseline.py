"""实验8：元学习与普通迁移学习的贡献分离实验。

本脚本是一个独立实验入口，不替换 ``main.py``、实验7或现有模型模块。
它在实验7的固定发动机 K-shot 协议下比较三种训练方式：

``scratch_gnn``
    GNN随机初始化，只使用K台目标域发动机训练。

``pretrained_gnn``
    GNN先在全部源域上进行普通监督预训练，再使用相同K台目标域发动机微调。

``meta_gnn``
    GNN先在源域执行Reptile元训练，再使用相同K台目标域发动机适应。

公平性约束
----------
1. 直接读取实验7的固定验证发动机和嵌套K-shot划分；
2. 三种方法从同一个随机初始化开始；
3. 普通源域预训练与Reptile对齐源域梯度计算预算；
4. 三种方法使用相同目标发动机、批次顺序、学习率、损失和目标训练轮数；
5. 归一化器只在源域训练数据上拟合；
6. 最终测试始终使用官方目标域测试集的全部发动机；
7. 测试结果只用于最终报告，不能据此重新选择超参数。

推荐从项目根目录运行：

    python scripts/experiment8_transfer_baseline.py \
        --target FD004 \
        --k-values 2 5 10 20 \
        --seeds 42 43 44 45 46 \
        --regimes scratch_gnn pretrained_gnn meta_gnn \
        --preprocessing condition_settings \
        --balance-mode engine_stage \
        --meta-epochs 100 \
        --target-epochs 10 \
        --inner-lr 0.001 \
        --outer-lr 0.05 \
        --pair-aux-weight 0.01

先检查协议而不训练：

    python scripts/experiment8_transfer_baseline.py --target FD004 --dry-run
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
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
    stable_unit_hash,
    target_unit_protocol,
    train_source_meta,
    train_target_equal_budget,
)


REGIMES = ("scratch_gnn", "pretrained_gnn", "meta_gnn")
COMPARISONS = (
    ("pretrained_gnn", "scratch_gnn", "ordinary_source_pretraining_vs_scratch"),
    ("meta_gnn", "scratch_gnn", "meta_learning_vs_scratch"),
    ("meta_gnn", "pretrained_gnn", "meta_learning_vs_ordinary_pretraining"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验8：GNN从零训练、普通源域预训练与Reptile元训练的公平比较"
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
        help="默认运行从零训练、普通源域预训练和Reptile元训练三组",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--protocol-file",
        help=(
            "实验7的split_protocol JSON；默认自动查找"
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
        "--balance-mode",
        choices=BALANCE_MODES,
        default="engine_stage",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--target-epochs", type=int)
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument(
        "--source-pretrain-steps",
        type=int,
        help=(
            "普通源域预训练的优化步数；默认等于"
            "meta_epochs × tasks_per_meta_batch × inner_steps"
        ),
    )
    parser.add_argument(
        "--source-pretrain-lr",
        type=float,
        help="普通源域预训练学习率；默认与inner_lr相同",
    )
    parser.add_argument(
        "--source-pretrain-weight-decay",
        type=float,
        default=0.0,
        help="默认与Reptile内循环一致，不使用权重衰减",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment8_transfer_baseline",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="读取已有raw JSON并跳过已完成的seed/K/regime组合",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查协议、模型输入和源域预算，不执行训练",
    )
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="仅用于mock数据调试；正式实验不要使用",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config, PROJECT_ROOT)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain
        for domain in EXPECTED_OFFICIAL_TEST_ENGINES
        if domain != args.target
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

    cfg["source_pretrain_steps"] = (
        args.source_pretrain_steps
        if args.source_pretrain_steps is not None
        else (
            cfg["meta_epochs"]
            * min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
            * cfg["inner_steps"]
        )
    )
    cfg["source_pretrain_lr"] = (
        args.source_pretrain_lr
        if args.source_pretrain_lr is not None
        else cfg["inner_lr"]
    )
    cfg["source_pretrain_weight_decay"] = args.source_pretrain_weight_decay
    if cfg["source_pretrain_steps"] <= 0:
        raise ValueError("--source-pretrain-steps必须为正整数")

    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir, PROJECT_ROOT))
    cfg["output_dir"] = str(resolve_path(args.output_dir, PROJECT_ROOT))
    return cfg


def default_protocol_path(args: argparse.Namespace) -> Path:
    if args.protocol_file:
        return resolve_path(args.protocol_file, PROJECT_ROOT)
    return (
        PROJECT_ROOT
        / "outputs"
        / "experiment7_kshot_engines"
        / f"experiment7_split_protocol_{args.target}.json"
    )


def load_or_create_protocol(
    args: argparse.Namespace,
    cfg: dict,
    seeds: list[int],
    k_values: list[int],
) -> tuple[dict, Path | None]:
    protocol_path = default_protocol_path(args)
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        source_path: Path | None = protocol_path
        print(f"[protocol] 读取实验7固定划分：{protocol_path}")
    else:
        protocol = target_unit_protocol(
            cfg["data_dir"],
            args.target,
            args.validation_units,
            args.validation_seed,
            seeds,
            k_values,
        )
        source_path = None
        print("[protocol] 未找到实验7协议文件，使用相同规则重新生成。")

    if protocol.get("target_domain") != args.target:
        raise ValueError("协议文件的target_domain与--target不一致")
    protocol_k = {int(value) for value in protocol.get("k_values", [])}
    missing_k = [value for value in k_values if value not in protocol_k]
    if missing_k:
        raise ValueError(f"实验7协议缺少K值：{missing_k}")
    nested = protocol.get("nested_adaptation_units_by_seed", {})
    missing_seeds = [seed for seed in seeds if str(seed) not in nested]
    if missing_seeds:
        raise ValueError(f"实验7协议缺少随机种子：{missing_seeds}")

    validation = set(int(unit) for unit in protocol["validation_units"])
    for seed in seeds:
        previous: set[int] = set()
        for k in k_values:
            current = set(int(unit) for unit in nested[str(seed)][str(k)])
            if len(current) != k:
                raise ValueError(f"seed={seed}, K={k}的发动机数量不正确")
            if not previous.issubset(current):
                raise ValueError(f"seed={seed}的K集合没有严格嵌套")
            if current & validation:
                raise ValueError("适应发动机与固定验证发动机发生重叠")
            previous = current
    return protocol, source_path


def train_source_supervised(
    model: torch.nn.Module,
    source_tasks: dict[str, torch.utils.data.DataLoader],
    cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict]]:
    """Ordinary multi-source supervised pretraining with an exact step budget.

    Source domains are sampled in a shuffled round-robin schedule, so each
    source contributes approximately the same number of optimizer steps even
    when its raw number of windows differs substantially.
    """
    learner = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(
        learner.parameters(),
        lr=cfg["source_pretrain_lr"],
        weight_decay=cfg["source_pretrain_weight_decay"],
    )
    task_names = sorted(source_tasks)
    iterators = {name: iter(source_tasks[name]) for name in task_names}
    schedule_rng = random.Random(cfg["seed"] + 17001)
    schedule: list[str] = []
    history: list[dict] = []
    report_every = max(1, cfg["source_pretrain_steps"] // 10)
    running_losses: list[float] = []
    learner.train()

    for step in range(1, cfg["source_pretrain_steps"] + 1):
        if not schedule:
            schedule = task_names.copy()
            schedule_rng.shuffle(schedule)
        task_name = schedule.pop()
        iterator = iterators[task_name]
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(source_tasks[task_name])
            iterators[task_name] = iterator
            x, y = next(iterator)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, _ = rul_training_loss(
            learner,
            x,
            y,
            cfg.get("pair_aux_weight", 0.0),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
        optimizer.step()
        running_losses.append(float(loss.item()))

        if step % report_every == 0 or step == cfg["source_pretrain_steps"]:
            mean_loss = float(np.mean(running_losses))
            history.append(
                {
                    "source_step": step,
                    "mean_source_loss": mean_loss,
                }
            )
            print(
                f"source_pretrain_step={step:04d}/"
                f"{cfg['source_pretrain_steps']} mean_loss={mean_loss:.4f}"
            )
            running_losses.clear()

    return learner, history


def build_source_initializations(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    regimes: list[str],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[dict]], int]:
    """Train each source initialization once per seed, independent of K."""
    first_k = min(int(value) for value in protocol["k_values"])
    first_units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(first_k)]

    # A first loader construction determines feature count and base model shape.
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
    del shape_loaders

    states: dict[str, dict[str, torch.Tensor]] = {}
    histories: dict[str, list[dict]] = {}
    if "scratch_gnn" in regimes:
        states["scratch_gnn"] = deepcopy(base_state)
        histories["scratch_gnn"] = []

    device = resolve_device(cfg["device"])
    if "pretrained_gnn" in regimes:
        # Fresh loaders keep their own deterministic generators and are not
        # consumed by the Reptile branch.
        pretrain_loaders = prepare_kshot_experiment(
            cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            first_units,
        )
        source_tasks = pretrain_loaders[0]
        pretrained_model = build_model("gnn", feature_count, cfg)
        pretrained_model.load_state_dict(base_state)
        pretrained_model, history = train_source_supervised(
            pretrained_model,
            source_tasks,
            cfg,
            device,
        )
        states["pretrained_gnn"] = {
            key: value.detach().cpu().clone()
            for key, value in pretrained_model.state_dict().items()
        }
        histories["pretrained_gnn"] = history
        del pretrained_model, pretrain_loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "meta_gnn" in regimes:
        meta_loaders = prepare_kshot_experiment(
            cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            first_units,
        )
        source_tasks = meta_loaders[0]
        meta_model = build_model("meta_gnn", feature_count, cfg)
        meta_model.load_state_dict(base_state)
        meta_model = meta_model.to(device)  # 添加这一行
        seed_everything(seed)

        meta_model = train_source_meta(meta_model, source_tasks, cfg, device)
        states["meta_gnn"] = {
            key: value.detach().cpu().clone()
            for key, value in meta_model.state_dict().items()
        }
        histories["meta_gnn"] = [
            {
                "meta_epochs": cfg["meta_epochs"],
                "tasks_per_meta_batch": cfg["tasks_per_meta_batch"],
                "inner_steps": cfg["inner_steps"],
                "inner_gradient_budget": (
                    cfg["meta_epochs"]
                    * min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
                    * cfg["inner_steps"]
                ),
            }
        ]
        del meta_model, meta_loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return states, histories, feature_count


def run_target_regime(
    args: argparse.Namespace,
    regime: str,
    cfg: dict,
    loaders,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    k: int,
) -> dict:
    """Apply one fixed source initialization to one matched target split."""
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    _, support, validation, test, feature_count, split_info = loaders
    build_name = "meta_gnn" if regime == "meta_gnn" else "gnn"
    model = build_model(build_name, feature_count, cfg)
    model.load_state_dict(source_state)
    model, target_history, best_target_epoch = train_target_equal_budget(
        model,
        support,
        validation,
        cfg,
        device,
    )
    validation_metrics = evaluate(model, validation, device)
    test_metrics = evaluate(model, test, device)

    source_training = {
        "scratch_gnn": "none",
        "pretrained_gnn": "ordinary_multisource_supervised_pretraining",
        "meta_gnn": "reptile_meta_training",
    }[regime]
    result = {
        **test_metrics,
        "regime": regime,
        "model": "gnn",
        "source_training": source_training,
        "experiment": f"experiment8_{regime}_k{k}",
        "target_domain": cfg["target_domain"],
        "seed": cfg["seed"],
        "k": k,
        "adaptation_engine_count": k,
        "validation_engine_count": len(split_info["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split_info["official_test_units_hash"],
        "target_epochs_completed": cfg["target_epochs"],
        "best_target_epoch_by_validation": best_target_epoch,
        "target_learning_rate": cfg["inner_lr"],
        "meta_epochs": cfg["meta_epochs"] if regime == "meta_gnn" else 0,
        "source_pretrain_steps": (
            cfg["source_pretrain_steps"] if regime == "pretrained_gnn" else 0
        ),
        "source_gradient_budget": (
            cfg["source_pretrain_steps"]
            if regime == "pretrained_gnn"
            else (
                cfg["meta_epochs"]
                * min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
                * cfg["inner_steps"]
                if regime == "meta_gnn"
                else 0
            )
        ),
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
    }

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / (
        f"experiment8_{regime}_k{k}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "split": split_info,
            "source_history": source_history,
            "target_history": target_history,
            "metrics": result,
        },
        checkpoint_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    group_columns = [
        "k",
        "regime",
        "source_training",
        "preprocessing_mode",
        "balance_mode",
    ]
    summary = frame.groupby(group_columns, as_index=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    summary = summary.rename(columns={"rmse_count": "n_runs"})
    redundant = [f"{metric}_count" for metric in METRICS if metric != "rmse"]
    summary = summary.drop(columns=[column for column in redundant if column in summary])
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
            cand = by_regime[candidate]
            ref = by_regime[reference]
            rows.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "comparison": comparison,
                    "candidate": candidate,
                    "reference": reference,
                    "rmse_delta_candidate_minus_reference": cand["rmse"] - ref["rmse"],
                    "mae_delta_candidate_minus_reference": cand["mae"] - ref["mae"],
                    "r2_delta_candidate_minus_reference": cand["r2"] - ref["r2"],
                    "nasa_delta_candidate_minus_reference": (
                        cand["nasa_score"] - ref["nasa_score"]
                    ),
                    "candidate_rmse_win": float(cand["rmse"] < ref["rmse"]),
                    "candidate_mae_win": float(cand["mae"] < ref["mae"]),
                    "candidate_r2_win": float(cand["r2"] > ref["r2"]),
                    "candidate_nasa_win": float(
                        cand["nasa_score"] < ref["nasa_score"]
                    ),
                }
            )
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired, pd.DataFrame()

    summary_rows: list[dict] = []
    delta_columns = (
        "rmse_delta_candidate_minus_reference",
        "mae_delta_candidate_minus_reference",
        "r2_delta_candidate_minus_reference",
        "nasa_delta_candidate_minus_reference",
    )
    win_columns = (
        "candidate_rmse_win",
        "candidate_mae_win",
        "candidate_r2_win",
        "candidate_nasa_win",
    )
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
        summary_rows.append(row)
    comparison_summary = pd.DataFrame(summary_rows).sort_values(
        ["k", "comparison"]
    )
    return paired.sort_values(["k", "comparison", "seed"]), comparison_summary


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    return {
        "output": output,
        "raw": output / f"experiment8_raw_{args.target}.json",
        "summary": output / f"experiment8_summary_{args.target}.csv",
        "paired": output / f"experiment8_paired_by_seed_{args.target}.csv",
        "comparisons": output / f"experiment8_comparisons_{args.target}.csv",
        "protocol": output / f"experiment8_split_protocol_{args.target}.json",
        "splits": output / f"experiment8_engine_splits_{args.target}.csv",
        "budget": output / f"experiment8_source_budget_{args.target}.json",
    }


def save_progress(results: list[dict], args: argparse.Namespace) -> dict[str, Path]:
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths["raw"],
        json.dumps(results, ensure_ascii=False, indent=2),
    )
    atomic_write_text(
        paths["summary"],
        summarize(results).to_csv(index=False),
        encoding="utf-8-sig",
    )
    paired, comparisons = paired_comparisons(results)
    atomic_write_text(
        paths["paired"],
        paired.to_csv(index=False),
        encoding="utf-8-sig",
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


def inspect_protocol(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    seed: int,
    k: int,
) -> None:
    adaptation_units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        adaptation_units,
    )
    tasks, support, validation, test, feature_count, split_info = loaders
    x, _ = next(iter(tasks[cfg["source_domains"][0]]))
    seed_everything(seed)
    model = build_model("gnn", feature_count, cfg).cpu().eval()
    with torch.no_grad():
        output = model(x[: min(8, len(x))])
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
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "meta_inner_gradient_budget": (
            cfg["meta_epochs"]
            * min(cfg["tasks_per_meta_batch"], len(cfg["source_domains"]))
            * cfg["inner_steps"]
        ),
        "adaptation_units": adaptation_units,
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k_values))
    seeds = list(dict.fromkeys(args.seeds))
    regimes = list(dict.fromkeys(args.regimes))
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("--k-values必须全部为正整数")
    if not seeds:
        raise ValueError("--seeds不能为空")
    if len(seeds) < 5 and not args.dry_run:
        print("[警告] 当前少于5个随机种子，只能视为预实验。")

    first_cfg = load_config(args, seeds[0])
    protocol, source_protocol_path = load_or_create_protocol(
        args,
        first_cfg,
        seeds,
        k_values,
    )
    expected_test_count = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        int(protocol["official_test_engine_count"]) != expected_test_count
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试集应有{expected_test_count}台发动机，"
            f"协议中为{protocol['official_test_engine_count']}台。"
        )

    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    copied_protocol = dict(protocol)
    copied_protocol["experiment8_source_protocol"] = (
        str(source_protocol_path) if source_protocol_path is not None else "regenerated"
    )
    atomic_write_text(
        paths["protocol"],
        json.dumps(copied_protocol, ensure_ascii=False, indent=2),
    )
    atomic_write_text(
        paths["splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )
    budget = {
        "source_domains": first_cfg["source_domains"],
        "meta_epochs": first_cfg["meta_epochs"],
        "tasks_per_meta_batch": min(
            first_cfg["tasks_per_meta_batch"], len(first_cfg["source_domains"])
        ),
        "inner_steps": first_cfg["inner_steps"],
        "meta_inner_gradient_budget": (
            first_cfg["meta_epochs"]
            * min(first_cfg["tasks_per_meta_batch"], len(first_cfg["source_domains"]))
            * first_cfg["inner_steps"]
        ),
        "ordinary_pretraining_optimizer_steps": first_cfg["source_pretrain_steps"],
        "ordinary_pretraining_lr": first_cfg["source_pretrain_lr"],
        "ordinary_pretraining_weight_decay": first_cfg[
            "source_pretrain_weight_decay"
        ],
        "target_epochs_equal_for_all_regimes": first_cfg["target_epochs"],
        "target_lr_equal_for_all_regimes": first_cfg["inner_lr"],
        "note": (
            "The two source regimes match gradient-computation counts, but Reptile "
            "outer updates and ordinary Adam updates are algorithmically different."
        ),
    }
    atomic_write_text(
        paths["budget"],
        json.dumps(budget, ensure_ascii=False, indent=2),
    )

    print("\n[实验8固定协议与训练预算]")
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
        for k in k_values:
            inspect_protocol(args, first_cfg, protocol, seeds[0], k)
        print("\n[dry-run完成] 未训练模型。")
        print(
            f"Protocol: {paths['protocol']}\nSplits: {paths['splits']}"
            f"\nBudget: {paths['budget']}"
        )
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条完成结果。")
    done = completed_keys(results)

    for seed in seeds:
        cfg = load_config(args, seed)
        pending_for_seed = [
            (k, regime)
            for k in k_values
            for regime in regimes
            if (seed, k, regime) not in done
        ]
        if not pending_for_seed:
            print(f"[skip seed] seed={seed}的全部组合均已完成。")
            continue

        required_regimes = list(dict.fromkeys(regime for _, regime in pending_for_seed))
        print(
            f"\n[source initialization] seed={seed} regimes={required_regimes}"
        )
        source_states, source_histories, _ = build_source_initializations(
            args,
            cfg,
            protocol,
            seed,
            required_regimes,
        )

        for k in k_values:
            adaptation_units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
            for regime in regimes:
                key = (seed, k, regime)
                if key in done:
                    print(f"[skip] seed={seed} K={k} regime={regime}")
                    continue
                print(
                    f"\n[experiment8] seed={seed} K={k} regime={regime} "
                    f"target_epochs={cfg['target_epochs']} engines={adaptation_units}"
                )
                # Fresh loaders reset the dedicated target sampler, giving all
                # three regimes the same target batch sequence.
                loaders = prepare_kshot_experiment(
                    cfg,
                    args.preprocessing,
                    args.balance_mode,
                    protocol["validation_units"],
                    adaptation_units,
                )
                if loaders[-1]["official_test_units_hash"] != protocol["official_test_units_hash"]:
                    raise AssertionError("不同运行使用了不同官方测试发动机")
                result = run_target_regime(
                    args,
                    regime,
                    cfg,
                    loaders,
                    source_states[regime],
                    source_histories[regime],
                    k,
                )
                results.append(result)
                done.add(key)
                paths = save_progress(results, args)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del source_states, source_histories
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(results)
    paired, comparisons = paired_comparisons(results)
    print("\n[experiment8 summary]")
    print(summary.to_string(index=False))
    if not comparisons.empty:
        print("\n[配对比较；RMSE/MAE/NASA delta<0、R2 delta>0表示候选更好]")
        print(comparisons.to_string(index=False))
    print("\n[结论判定]")
    print("1. pretrained_gnn优于scratch_gnn：说明源域普通预训练有效。")
    print("2. meta_gnn优于pretrained_gnn：说明Reptile不仅仅复用了源域特征。")
    print("3. meta_gnn≈pretrained_gnn：实验7提升主要来自可复用源域表示。")
    print("4. pretrained_gnn优于meta_gnn：当前元任务构造或Reptile更新可能存在负迁移。")
    print("5. 重点观察K=2和K=5，并结合5个种子的胜率与标准差。")
    print("6. 不允许根据官方测试结果重新选择上述超参数。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nComparisons: {paths['comparisons']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
        f"\nBudget: {paths['budget']}"
    )


if __name__ == "__main__":
    main()
