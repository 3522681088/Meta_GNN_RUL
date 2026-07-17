"""实验7：固定目标发动机数量的 K-shot 少样本适应实验。

本脚本是一个独立实验入口，不替换 ``main.py`` 或任何原有模块。它用于
在严格一致的数据协议下比较 Meta-GNN 与普通 GNN：

1. K 表示目标域适应集中的发动机数量，而不是窗口比例；
2. 默认 K=2、5、10、20，且同一种子内严格嵌套；
3. 目标域验证发动机由独立的 validation_seed 固定，所有 K/模型/种子共用；
4. 同一种子下所有模型使用相同的目标发动机顺序和目标批次采样器；
5. Meta-GNN 与 GNN 使用相同的目标训练轮数、学习率、损失和采样方式；
6. 归一化器只在源域训练数据上拟合；
7. 最终测试始终使用官方目标域测试集中的全部发动机；
8. 默认运行 5 个随机种子，并输出逐次结果、均值/标准差和配对比较。

推荐从项目根目录运行：

    D:\\Anaconda\\envs\\pytorch\\python.exe \
        scripts\\experiment7_kshot_engines.py \
        --target FD004 \
        --k-values 2 5 10 20 \
        --seeds 42 43 44 45 46 \
        --models meta_gnn gnn \
        --preprocessing condition_settings \
        --balance-mode engine_stage \
        --validation-units 20 \
        --meta-epochs 100 \
        --target-epochs 10 \
        --inner-lr 0.001 \
        --outer-lr 0.05 \
        --pair-aux-weight 0.01

先做不训练的协议检查：

    D:\\Anaconda\\envs\\pytorch\\python.exe \
        scripts\\experiment7_kshot_engines.py --target FD004 --dry-run

注意：官方测试集只能用于最终报告，不能根据测试结果重新选择超参数。
超参数应在实验开始前固定，或仅根据固定验证发动机确定。
"""

from __future__ import annotations

import argparse
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
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from evaluation.metrics import regression_metrics  # noqa: E402
from meta_learning import TaskSampler, reptile_meta_step  # noqa: E402
from preprocess.cmapps_loader import load_domain  # noqa: E402
from preprocess.rul_generator import add_test_rul, add_train_rul  # noqa: E402
from preprocess.window_dataset import WindowDataset, make_windows  # noqa: E402
from train.losses import rul_training_loss  # noqa: E402
from scripts.run_condition_aware_experiment import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
    SETTING_FEATURE_COLUMNS,
    SourceConditionNormalizer,
    SourceGlobalNormalizer,
    sampling_weights,
    stage_distribution_from_loader,
)


MODEL_CHOICES = ("meta_gnn", "gnn")
METRICS = ("rmse", "mae", "r2", "nasa_score")
EXPECTED_OFFICIAL_TEST_ENGINES = {
    "FD001": 100,
    "FD002": 259,
    "FD003": 100,
    "FD004": 248,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验7：固定目标发动机数量的K-shot少样本适应实验"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target",
        default="FD004",
        choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["meta_gnn", "gnn"],
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20],
        help="目标域适应发动机数量；同一种子内自动构造嵌套集合",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="同时控制目标发动机顺序和模型训练随机性；正式结果建议至少5个",
    )
    parser.add_argument(
        "--validation-units",
        type=int,
        default=20,
        help="从目标训练集预留且固定的验证发动机数量",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=2026,
        help="只用于固定验证发动机，不随训练种子变化",
    )
    parser.add_argument(
        "--normalizer-seed",
        type=int,
        default=2026,
        help="固定工况聚类/归一化协议，避免它随训练种子变化",
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default="condition_settings",
        help="应在实验前根据先验或固定验证集确定，不能根据官方测试集更改",
    )
    parser.add_argument(
        "--balance-mode",
        choices=BALANCE_MODES,
        default="engine_stage",
        help="Meta-GNN和GNN共用的窗口采样策略",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument(
        "--target-epochs",
        type=int,
        help="两个模型完全相同的目标域训练轮数；默认读取adapt_epochs",
    )
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument(
        "--inner-lr",
        type=float,
        help="Meta内循环及两个模型目标域训练共用的学习率",
    )
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment7_kshot_engines",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="读取已有raw JSON并跳过已完成的seed/K/model组合",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="检查固定验证集、嵌套K集合、官方测试集和模型前向传播，不训练",
    )
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="仅供mock数据调试；正式C-MAPSS实验不要使用",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def stable_unit_hash(units: Iterable[int]) -> str:
    payload = ",".join(str(int(unit)) for unit in units).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config)
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

    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir))
    cfg["output_dir"] = str(resolve_path(args.output_dir))
    return cfg


def target_unit_protocol(
    data_dir: str,
    target: str,
    validation_count: int,
    validation_seed: int,
    seeds: list[int],
    k_values: list[int],
) -> dict:
    """Create one fixed validation set and nested adaptation sets per seed."""
    target_train, target_test, final_rul = load_domain(data_dir, target)
    if final_rul is None:
        raise ValueError(f"RUL_{target}.txt is required for official evaluation")

    train_units = np.asarray(sorted(target_train["unit"].unique()), dtype=int)
    test_units = np.asarray(sorted(target_test["unit"].unique()), dtype=int)
    if not 1 <= validation_count < len(train_units):
        raise ValueError(
            f"--validation-units必须位于[1, {len(train_units) - 1}]，"
            f"当前为{validation_count}"
        )

    validation_rng = np.random.default_rng(validation_seed)
    validation_order = validation_rng.permutation(train_units)
    validation_units = validation_order[:validation_count]
    candidate_units = np.asarray(
        [unit for unit in train_units if unit not in set(validation_units)],
        dtype=int,
    )
    if max(k_values) > len(candidate_units):
        raise ValueError(
            f"最大K={max(k_values)}超过可用适应发动机数量{len(candidate_units)}"
        )

    orders: dict[str, list[int]] = {}
    nested: dict[str, dict[str, list[int]]] = {}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(candidate_units)
        orders[str(seed)] = order.astype(int).tolist()
        nested[str(seed)] = {
            str(k): order[:k].astype(int).tolist() for k in k_values
        }

        previous: set[int] = set()
        for k in k_values:
            current = set(nested[str(seed)][str(k)])
            if not previous.issubset(current):
                raise AssertionError(f"seed={seed}的K集合没有严格嵌套")
            if current & set(validation_units):
                raise AssertionError("适应发动机与验证发动机发生重叠")
            previous = current

    return {
        "target_domain": target,
        "train_engine_count": int(len(train_units)),
        "official_test_engine_count": int(len(test_units)),
        "official_test_units": test_units.tolist(),
        "official_test_units_hash": stable_unit_hash(test_units),
        "validation_seed": validation_seed,
        "validation_units": validation_units.astype(int).tolist(),
        "candidate_adaptation_engine_count": int(len(candidate_units)),
        "k_values": k_values,
        "adaptation_order_by_seed": orders,
        "nested_adaptation_units_by_seed": nested,
    }


def make_loader(
    frame: pd.DataFrame,
    features: list[str],
    cfg: dict,
    *,
    training: bool,
    balance_mode: str = "none",
    last_only: bool = False,
    loader_seed: int,
) -> DataLoader:
    """Build an independently seeded loader so all models see the same batches."""
    x, y, units = make_windows(
        frame,
        features,
        cfg["window_size"],
        cfg["window_stride"],
        last_only,
    )
    dataset = WindowDataset(x, y, units)
    generator = torch.Generator().manual_seed(loader_seed)

    if training and balance_mode != "none":
        weights = sampling_weights(y, units, balance_mode)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=training,
        generator=generator if training else None,
        drop_last=False,
    )


def prepare_kshot_experiment(
    cfg: dict,
    preprocessing_mode: str,
    balance_mode: str,
    validation_units: list[int],
    adaptation_units: list[int],
):
    """Prepare source tasks, fixed target validation, K engines and official test."""
    domains = cfg["source_domains"] + [cfg["target_domain"]]
    raw: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for domain in domains:
        train, test, final_rul = load_domain(cfg["data_dir"], domain)
        raw[domain] = (
            add_train_rul(train, cfg["rul_cap"]),
            add_test_rul(test, final_rul, cfg["rul_cap"]),
        )

    sensors = list(cfg["sensor_columns"])
    source_fit = pd.concat(
        [raw[domain][0] for domain in cfg["source_domains"]],
        ignore_index=True,
    )
    condition_aware = preprocessing_mode in {"condition_norm", "condition_settings"}
    include_settings = preprocessing_mode in {"global_settings", "condition_settings"}
    if condition_aware:
        normalizer = SourceConditionNormalizer(
            n_conditions=cfg.get("condition_count", 6),
            seed=cfg.get("normalizer_seed", 2026),
            include_settings=include_settings,
        ).fit(source_fit, sensors)
    else:
        normalizer = SourceGlobalNormalizer(
            include_settings=include_settings,
        ).fit(source_fit, sensors)

    features = sensors + SETTING_FEATURE_COLUMNS if include_settings else sensors
    normalized = {
        domain: (
            normalizer.transform(train, sensors),
            normalizer.transform(test, sensors),
        )
        for domain, (train, test) in raw.items()
    }

    # 源域元训练使用全部官方源域训练发动机，不借用目标域或官方测试数据。
    source_tasks = {}
    for index, domain in enumerate(cfg["source_domains"]):
        source_tasks[domain] = make_loader(
            normalized[domain][0],
            features,
            cfg,
            training=True,
            balance_mode=balance_mode,
            loader_seed=cfg["seed"] + 1000 * (index + 1),
        )

    target_train, target_test = normalized[cfg["target_domain"]]
    adaptation_array = np.asarray(adaptation_units, dtype=int)
    validation_array = np.asarray(validation_units, dtype=int)
    if set(adaptation_array) & set(validation_array):
        raise AssertionError("适应发动机与验证发动机不能重叠")

    support_frame = target_train.query("unit in @adaptation_array")
    validation_frame = target_train.query("unit in @validation_array")
    if support_frame["unit"].nunique() != len(adaptation_units):
        raise ValueError("目标适应发动机数量与K不一致")
    if validation_frame["unit"].nunique() != len(validation_units):
        raise ValueError("固定验证发动机数量不一致")

    support = make_loader(
        support_frame,
        features,
        cfg,
        training=True,
        balance_mode=balance_mode,
        loader_seed=cfg["seed"] + 9000,
    )
    validation = make_loader(
        validation_frame,
        features,
        cfg,
        training=False,
        loader_seed=cfg["seed"] + 9100,
    )
    test = make_loader(
        target_test,
        features,
        cfg,
        training=False,
        last_only=True,
        loader_seed=cfg["seed"] + 9200,
    )

    split_info = {
        "protocol": "fixed_engine_kshot",
        "target_domain": cfg["target_domain"],
        "preprocessing_mode": preprocessing_mode,
        "balance_mode": balance_mode,
        "normalizer_fit_scope": "source_train_only",
        "normalizer_seed": cfg.get("normalizer_seed", 2026),
        "feature_columns": features,
        "adaptation_engine_count": len(adaptation_units),
        "adaptation_units": [int(unit) for unit in adaptation_units],
        "validation_engine_count": len(validation_units),
        "validation_units": [int(unit) for unit in validation_units],
        "official_test_engine_count": len(test.dataset),
        "official_test_units": [int(unit) for unit in test.dataset.units],
        "official_test_units_hash": stable_unit_hash(test.dataset.units),
    }
    if condition_aware:
        split_info["source_condition_counts"] = normalizer.source_condition_counts

    return source_tasks, support, validation, test, len(features), split_info


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device))
            labels.extend(y.cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
    return np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    labels, predictions = predict(model, loader, device)
    return regression_metrics(labels, predictions)


def train_source_meta(
    model: torch.nn.Module,
    source_tasks: dict[str, DataLoader],
    cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    sampler = TaskSampler(source_tasks, cfg["tasks_per_meta_batch"])
    for epoch in range(cfg["meta_epochs"]):
        model = reptile_meta_step(
            model,
            sampler.sample(),
            cfg["inner_steps"],
            cfg["inner_lr"],
            cfg["outer_lr"],
            device,
            cfg.get("pair_aux_weight", 0.0),
        )
        if (epoch + 1) % 5 == 0 or epoch + 1 == cfg["meta_epochs"]:
            print(f"meta_epoch={epoch + 1:03d}/{cfg['meta_epochs']}")
    return model


def train_target_equal_budget(
    model: torch.nn.Module,
    support: DataLoader,
    validation: DataLoader,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict], int]:
    """Train exactly target_epochs for both models and select by fixed validation."""
    learner = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=cfg["inner_lr"])
    best_state = deepcopy(learner.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(cfg["target_epochs"]):
        learner.train()
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
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        validation_metrics = evaluate(learner, validation, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"target_epoch={epoch + 1:03d}/{cfg['target_epochs']} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_rmse={validation_metrics['rmse']:.4f}"
        )
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch + 1
            best_state = deepcopy(learner.state_dict())

    learner.load_state_dict(best_state)
    return learner, history, best_epoch


def run_model(
    model_name: str,
    cfg: dict,
    loaders,
    k: int,
    preprocessing_mode: str,
    balance_mode: str,
) -> dict:
    """Run one matched seed/K/model experiment and save a checkpoint."""
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    source_tasks, support, validation, test, feature_count, split_info = loaders
    model = build_model(model_name, feature_count, cfg).to(device)

    if model_name == "meta_gnn":
        model = train_source_meta(model, source_tasks, cfg, device)
        training_regime = "source_reptile_then_target_adaptation"
    elif model_name == "gnn":
        training_regime = "target_from_scratch"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 两个模型从这里开始使用完全相同的目标轮数、学习率、损失和批次协议。
    model, target_history, best_target_epoch = train_target_equal_budget(
        model,
        support,
        validation,
        cfg,
        device,
    )
    validation_metrics = evaluate(model, validation, device)
    test_metrics = evaluate(model, test, device)

    result = {
        **test_metrics,
        "model": model_name,
        "training_regime": training_regime,
        "experiment": f"experiment7_{model_name}_k{k}",
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
        "meta_epochs": cfg["meta_epochs"] if model_name == "meta_gnn" else 0,
        "preprocessing_mode": preprocessing_mode,
        "balance_mode": balance_mode,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
    }

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / (
        f"experiment7_{model_name}_k{k}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "split": split_info,
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
        "model",
        "training_regime",
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
    summary = summary.drop(columns=[c for c in redundant if c in summary.columns])
    return summary.sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def paired_results(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Meta-GNN and GNN on the same seed and the same K engines."""
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    for (k, seed), group in frame.groupby(["k", "seed"]):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        if "meta_gnn" not in by_model or "gnn" not in by_model:
            continue
        meta = by_model["meta_gnn"]
        gnn = by_model["gnn"]
        rows.append(
            {
                "k": int(k),
                "seed": int(seed),
                "rmse_delta_meta_minus_gnn": meta["rmse"] - gnn["rmse"],
                "mae_delta_meta_minus_gnn": meta["mae"] - gnn["mae"],
                "r2_delta_meta_minus_gnn": meta["r2"] - gnn["r2"],
                "nasa_delta_meta_minus_gnn": (
                    meta["nasa_score"] - gnn["nasa_score"]
                ),
                "meta_rmse_win": float(meta["rmse"] < gnn["rmse"]),
                "meta_mae_win": float(meta["mae"] < gnn["mae"]),
                "meta_r2_win": float(meta["r2"] > gnn["r2"]),
                "meta_nasa_win": float(
                    meta["nasa_score"] < gnn["nasa_score"]
                ),
            }
        )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame()

    summary_rows: list[dict] = []
    for k, group in per_seed.groupby("k"):
        row = {"k": int(k), "paired_seed_count": int(len(group))}
        for column in (
            "rmse_delta_meta_minus_gnn",
            "mae_delta_meta_minus_gnn",
            "r2_delta_meta_minus_gnn",
            "nasa_delta_meta_minus_gnn",
        ):
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        for column in (
            "meta_rmse_win",
            "meta_mae_win",
            "meta_r2_win",
            "meta_nasa_win",
        ):
            row[f"{column}_rate"] = float(group[column].mean())
        summary_rows.append(row)
    return per_seed.sort_values(["k", "seed"]), pd.DataFrame(summary_rows)


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir)
    return {
        "output": output,
        "raw": output / f"experiment7_raw_{args.target}.json",
        "summary": output / f"experiment7_summary_{args.target}.csv",
        "paired": output / f"experiment7_paired_by_seed_{args.target}.csv",
        "advantage": output / f"experiment7_meta_advantage_{args.target}.csv",
        "protocol": output / f"experiment7_split_protocol_{args.target}.json",
        "splits": output / f"experiment7_engine_splits_{args.target}.csv",
    }


def protocol_split_frame(protocol: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for unit in protocol["validation_units"]:
        rows.append({"seed": "fixed", "k": "all", "role": "validation", "unit": unit})
    for seed, by_k in protocol["nested_adaptation_units_by_seed"].items():
        for k, units in by_k.items():
            for unit in units:
                rows.append({"seed": int(seed), "k": int(k), "role": "adaptation", "unit": unit})
    for unit in protocol["official_test_units"]:
        rows.append({"seed": "fixed", "k": "all", "role": "official_test", "unit": unit})
    return pd.DataFrame(rows)


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
    paired, advantage = paired_results(results)
    atomic_write_text(
        paths["paired"],
        paired.to_csv(index=False),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["advantage"],
        advantage.to_csv(index=False),
        encoding="utf-8-sig",
    )
    return paths


def completed_keys(results: list[dict]) -> set[tuple[int, int, str]]:
    return {
        (int(row["seed"]), int(row["k"]), str(row["model"]))
        for row in results
    }


def inspect_one_configuration(
    cfg: dict,
    args: argparse.Namespace,
    protocol: dict,
    seed: int,
    k: int,
) -> None:
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
    model = build_model(args.models[0], feature_count, cfg).cpu().eval()
    with torch.no_grad():
        prediction = model(x[: min(8, len(x))])
    diagnostic = {
        "seed": seed,
        "k": k,
        "feature_count": feature_count,
        "source_example_shape": list(x.shape),
        "forward_output_shape": list(prediction.shape),
        "support_windows": len(support.dataset),
        "support_engines": len(set(support.dataset.units)),
        "validation_windows": len(validation.dataset),
        "validation_engines": len(set(validation.dataset.units)),
        "official_test_engines": len(test.dataset),
        "support_stage_distribution_pct": stage_distribution_from_loader(support),
        "split": split_info,
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k_values))
    seeds = list(dict.fromkeys(args.seeds))
    models = list(dict.fromkeys(args.models))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k-values必须全部为正整数")
    if not seeds:
        raise ValueError("--seeds不能为空")
    if len(seeds) < 5 and not args.dry_run:
        print("[警告] 当前少于5个随机种子，只能视为预实验结果。")

    first_cfg = load_config(args, seeds[0])
    protocol = target_unit_protocol(
        first_cfg["data_dir"],
        args.target,
        args.validation_units,
        args.validation_seed,
        seeds,
        k_values,
    )
    expected_test_count = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        protocol["official_test_engine_count"] != expected_test_count
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试集应有{expected_test_count}台发动机，"
            f"当前检测到{protocol['official_test_engine_count']}台。"
            "请检查data目录；mock数据调试时可显式使用"
            "--skip-official-count-check。"
        )

    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths["protocol"],
        json.dumps(protocol, ensure_ascii=False, indent=2),
    )
    atomic_write_text(
        paths["splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )

    print("\n[实验7固定协议]")
    print(
        json.dumps(
            {
                "target": args.target,
                "k_values": k_values,
                "seeds": seeds,
                "models": models,
                "fixed_validation_units": protocol["validation_units"],
                "official_test_engine_count": protocol["official_test_engine_count"],
                "official_test_units_hash": protocol["official_test_units_hash"],
                "preprocessing": args.preprocessing,
                "balance_mode": args.balance_mode,
                "target_epochs_equal_for_all_models": first_cfg["target_epochs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run:
        # 检查每个K的最小样本种子；协议JSON已包含全部种子的嵌套集合。
        for k in k_values:
            inspect_one_configuration(first_cfg, args, protocol, seeds[0], k)
        print("\n[dry-run完成] 划分严格嵌套，未训练模型。")
        print(f"Protocol: {paths['protocol']}\nSplits: {paths['splits']}")
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条完成结果。")
    done = completed_keys(results)

    for seed in seeds:
        cfg = load_config(args, seed)
        for k in k_values:
            adaptation_units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
            for model_name in models:
                key = (seed, k, model_name)
                if key in done:
                    print(f"[skip] seed={seed} K={k} model={model_name}")
                    continue
                print(
                    f"\n[experiment7] seed={seed} K={k} model={model_name} "
                    f"target_epochs={cfg['target_epochs']} "
                    f"engines={adaptation_units}"
                )
                # 每个模型重新创建独立但同种子的loader，保证目标批次协议一致。
                loaders = prepare_kshot_experiment(
                    cfg,
                    args.preprocessing,
                    args.balance_mode,
                    protocol["validation_units"],
                    adaptation_units,
                )
                split_hash = loaders[-1]["official_test_units_hash"]
                if split_hash != protocol["official_test_units_hash"]:
                    raise AssertionError("不同运行使用了不同的官方测试发动机")
                result = run_model(
                    model_name,
                    cfg,
                    loaders,
                    k,
                    args.preprocessing,
                    args.balance_mode,
                )
                results.append(result)
                done.add(key)
                paths = save_progress(results, args)

    summary = summarize(results)
    paired, advantage = paired_results(results)
    print("\n[experiment7 summary]")
    print(summary.to_string(index=False))
    if not advantage.empty:
        print("\n[Meta-GNN相对GNN的配对优势；误差delta<0、R2 delta>0为改善]")
        print(advantage.to_string(index=False))
    print("\n[判断标准]")
    print("1. 同一K下，Meta-GNN的RMSE/MAE/NASA均值更低、R²更高。")
    print("2. meta_*_win_rate越接近1，说明优势在不同发动机组合上越稳定。")
    print("3. 标准差越小，说明模型对具体选中哪几台目标发动机越不敏感。")
    print("4. 若Meta-GNN主要在K=2/5胜出，且K增大后差距缩小，支持少样本优势。")
    print("5. 不允许根据上述官方测试结果继续选择超参数。")
    print(
        f"\nRaw: {paths['raw']}\nSummary: {paths['summary']}"
        f"\nPaired: {paths['paired']}\nAdvantage: {paths['advantage']}"
        f"\nProtocol: {paths['protocol']}\nSplits: {paths['splits']}"
    )


if __name__ == "__main__":
    main()
