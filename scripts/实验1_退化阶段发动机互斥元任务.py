"""实验1：退化阶段分层、发动机互斥元任务消融。

基于 Meta_GNN_RUL 原项目现有的数据处理、稳定传感器图骨干、
FOMAML/监督训练及目标域适应函数，只新增统一任务分组器。

默认比较：
1. static_init：普通源域预训练初始化；
2. supervised_budget：与FOMAML预算匹配的源域监督训练；
3. fomaml_random：随机发动机互斥任务；
4. fomaml_condition：相同工况、发动机互斥任务；
5. fomaml_rul_stage：相同RUL阶段、发动机互斥任务；
6. fomaml_condition_rul_stage：相同工况与RUL阶段任务。

默认仅使用C-MAPSS训练文件进行目标域发动机互斥验证，不读取官方测试集。

快速检查：
python 实验1_退化阶段发动机互斥元任务.py --dry-run

正式实验：
python 实验1_退化阶段发动机互斥元任务.py ^
  --target FD004 ^
  --model-seeds 42 43 44 45 46 ^
  --target-split-seeds 3027 3028 3029 3030 3031 ^
  --meta-steps 600
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


DEFAULT_PROJECT_ROOT = Path(
    r"D:\文件\文献阅读\态势感知\第十六周_元学习"
    r"\文献13代码复现\Meta_GNN_RUL_论文对齐完整代码\Meta_GNN_RUL"
)
TASK_MODES = ("random", "condition", "rul_stage", "condition_rul_stage")
STAGE_BINS = (30.0, 60.0, 90.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", choices=("FD001", "FD002", "FD003", "FD004"), default="FD004"
    )
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--target-split-seeds",
        nargs="+",
        type=int,
        default=[3027, 3028, 3029, 3030, 3031],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--protocol")
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/experiment18_task_conditioned_sensor_graph/source_cache",
    )
    parser.add_argument("--task-modes", nargs="+", choices=TASK_MODES, default=list(TASK_MODES))
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--set-token-dim", type=int, default=32)
    parser.add_argument("--graph-residual-rank", type=int, default=4)
    parser.add_argument("--max-graph-gate", type=float, default=0.5)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--meta-steps", type=int, default=600)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--meta-weight-decay", type=float, default=0.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--support-engines", type=int, default=5)
    parser.add_argument("--query-engines", type=int, default=5)
    parser.add_argument("--support-windows", type=int, default=128)
    parser.add_argument("--query-windows", type=int, default=128)
    parser.add_argument("--audit-tasks", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=1e-3)
    parser.add_argument("--min-condition-windows", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="outputs/experiment1_stage_disjoint_meta_tasks"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_args(args: argparse.Namespace, model_seed: int) -> argparse.Namespace:
    """生成原项目Experiment 26函数需要的参数对象。"""
    return argparse.Namespace(
        config=args.config,
        data_dir=args.data_dir,
        target=args.target,
        k_values=[args.k],
        target_split_seeds=list(args.target_split_seeds),
        model_seed=model_seed,
        protocol=args.protocol,
        checkpoint_dir=args.checkpoint_dir,
        condition_count=args.condition_count,
        normalizer_seed=args.normalizer_seed,
        sensor_graph_k=args.sensor_graph_k,
        context_hidden_dim=args.context_hidden_dim,
        context_dim=args.context_dim,
        set_token_dim=args.set_token_dim,
        graph_residual_rank=args.graph_residual_rank,
        max_graph_gate=args.max_graph_gate,
        gate_scale=args.gate_scale,
        meta_steps=args.meta_steps,
        meta_lr=args.meta_lr,
        meta_weight_decay=args.meta_weight_decay,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        source_support_engines=args.support_engines,
        source_query_engines=args.query_engines,
        source_support_windows=args.support_windows,
        source_query_windows=args.query_windows,
        target_epochs=args.target_epochs,
        target_lr=args.target_lr,
        min_condition_windows=args.min_condition_windows,
        device=args.device,
        output_dir=args.output_dir,
        resume=False,
        dry_run=args.dry_run,
    )


def group_key(mode: str, condition: int, stage: int):
    if mode == "condition":
        return condition
    if mode == "rul_stage":
        return stage
    if mode == "condition_rul_stage":
        return condition, stage
    return "all"


class EpisodeBank:
    """按指定规则采样发动机互斥的support/query任务。"""

    def __init__(
        self,
        source_data: dict[str, dict],
        mode: str,
        support_engines: int,
        query_engines: int,
        seed: int,
        balanced_sampler,
    ):
        self.data = source_data
        self.mode = mode
        self.support_engines = support_engines
        self.query_engines = query_engines
        self.rng = np.random.default_rng(seed)
        self.balanced_sampler = balanced_sampler
        required = support_engines + query_engines
        self.groups: list[tuple[str, object, np.ndarray]] = []
        for domain, data in source_data.items():
            labels = data["y"].numpy()
            stages = np.digitize(labels, bins=STAGE_BINS, right=True)
            grouped: dict[object, list[int]] = {}
            for index, (condition, stage) in enumerate(
                zip(data["conditions"], stages, strict=True)
            ):
                key = group_key(mode, int(condition), int(stage))
                grouped.setdefault(key, []).append(index)
            for key, indices in grouped.items():
                indices = np.asarray(indices, dtype=int)
                if len(np.unique(data["units"][indices])) >= required:
                    self.groups.append((domain, key, indices))
        if not self.groups:
            raise RuntimeError(f"{mode}没有可构造的发动机互斥任务")

    def sample(self, support_windows: int, query_windows: int):
        group_id = int(self.rng.integers(0, len(self.groups)))
        domain, _, indices = self.groups[group_id]
        data = self.data[domain]
        units = data["units"][indices]
        selected = self.rng.choice(
            np.unique(units),
            self.support_engines + self.query_engines,
            replace=False,
        )
        support_units = selected[: self.support_engines]
        query_units = selected[self.support_engines :]
        labels = data["y"].numpy()[indices]
        support_local = self.balanced_sampler(
            units, labels, support_units, support_windows, self.rng
        )
        query_local = self.balanced_sampler(
            units, labels, query_units, query_windows, self.rng
        )
        support_indices = indices[support_local]
        query_indices = indices[query_local]
        if set(data["units"][support_indices]) & set(data["units"][query_indices]):
            raise AssertionError("support与query发动机发生重叠")
        return (
            domain,
            group_id,
            data["x"][support_indices],
            data["y"][support_indices],
            data["x"][query_indices],
            data["y"][query_indices],
        )


def state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def audit_task_alignment(
    initial: torch.nn.Module,
    bank: EpisodeBank,
    task_count: int,
    inner_lr: float,
    device: torch.device,
) -> dict:
    """测量support/query梯度一致性及一次适应的真实query收益。"""
    rows = []
    predictor = deepcopy(initial).to(device).eval()
    for _ in range(task_count):
        domain, group, sx, sy, qx, qy = bank.sample(128, 128)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        support_loss = F.mse_loss(predictor(sx).squeeze(-1), sy)
        support_gradients = torch.autograd.grad(
            support_loss, tuple(predictor.parameters())
        )
        query_loss = F.mse_loss(predictor(qx).squeeze(-1), qy)
        query_gradients = torch.autograd.grad(
            query_loss, tuple(predictor.parameters())
        )
        support_vector = torch.cat([gradient.flatten() for gradient in support_gradients])
        query_vector = torch.cat([gradient.flatten() for gradient in query_gradients])
        cosine = F.cosine_similarity(
            support_vector.unsqueeze(0), query_vector.unsqueeze(0)
        ).item()

        adapted = deepcopy(predictor)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
        optimizer.zero_grad()
        loss = F.mse_loss(adapted(sx).squeeze(-1), sy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
        optimizer.step()
        adapted.eval()
        with torch.no_grad():
            post_query_loss = F.mse_loss(adapted(qx).squeeze(-1), qy).item()
        rows.append(
            {
                "domain": domain,
                "group": group,
                "gradient_cosine": cosine,
                "adaptation_gain": float(query_loss.item() - post_query_loss),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "task_count": len(frame),
        "gradient_cosine_mean": float(frame["gradient_cosine"].mean()),
        "positive_gradient_rate": float((frame["gradient_cosine"] > 0).mean()),
        "adaptation_gain_mean": float(frame["adaptation_gain"].mean()),
        "positive_adaptation_rate": float((frame["adaptation_gain"] > 0).mean()),
    }


def paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index = ["model_seed", "target_split_seed", "k"]
    for candidate in [name for name in raw["regime"].unique() if name.startswith("fomaml_")]:
        candidate_rows = raw[raw["regime"] == candidate].set_index(index)
        for reference in ("supervised_budget", "static_init"):
            reference_rows = raw[raw["regime"] == reference].set_index(index)
            paired = candidate_rows.join(
                reference_rows, lsuffix="_candidate", rsuffix="_reference"
            )
            rmse_delta = paired["rmse_candidate"] - paired["rmse_reference"]
            nasa_delta = (
                paired["nasa_score_candidate"] - paired["nasa_score_reference"]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "n_pairs": len(paired),
                    "rmse_delta_mean": float(rmse_delta.mean()),
                    "rmse_improvement_pct": float(
                        -100 * rmse_delta.mean() / paired["rmse_reference"].mean()
                    ),
                    "rmse_win_rate": float((rmse_delta < 0).mean()),
                    "nasa_delta_mean": float(nasa_delta.mean()),
                    "nasa_win_rate": float((nasa_delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def self_check(exp26) -> None:
    # ponytail: 一个小型自检覆盖任务分层和发动机互斥，无需额外测试框架。
    labels = torch.tensor(
        [10.0, 15.0, 20.0, 25.0, 70.0, 75.0, 80.0, 85.0]
    )
    units = np.arange(1, 9)
    toy = {
        "FDX": {
            "x": torch.randn(8, 3),
            "y": labels,
            "conditions": np.zeros(8, dtype=int),
            "units": units,
        }
    }
    bank = EpisodeBank(
        toy, "rul_stage", 2, 2, 1, exp26.exp18._sample_balanced_indices
    )
    _, _, _, _, _, _ = bank.sample(4, 4)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    experiment26_path = root / "scripts" / "experiment26_engine_task_meta_initialization.py"
    if not experiment26_path.is_file():
        raise FileNotFoundError(f"找不到原项目实验代码：{experiment26_path}")
    sys.path.insert(0, str(root))
    exp26 = load_module("experiment1_base_exp26", experiment26_path)
    self_check(exp26)

    if args.k != 5:
        raise ValueError("当前原项目协议只锁定K=5；其他K需要单独生成预注册划分")
    if args.dry_run:
        args.model_seeds = args.model_seeds[:1]
        args.target_split_seeds = args.target_split_seeds[:1]
        args.meta_steps = 2
        args.audit_tasks = 2
        args.target_epochs = 1
        args.support_windows = min(args.support_windows, 32)
        args.query_windows = min(args.query_windows, 32)

    output = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else root / args.output_dir
    )
    output.mkdir(parents=True, exist_ok=True)
    device = exp26.resolve_device(args.device)
    all_results: list[dict] = []
    all_audits: list[dict] = []

    print(f"[实验1] target={args.target} device={device}")
    print("[数据策略] 仅训练文件；不读取官方测试集")
    for model_seed in args.model_seeds:
        run_args = experiment_args(args, model_seed)
        exp26.seed_everything(model_seed)
        cfg = exp26.exp251.load_config(run_args)
        protocol, protocol_path = exp26.exp25.load_protocol(run_args)
        normalized, feature_names, _ = exp26.exp24.normalized_train_frames(
            cfg, exp26.PREPROCESSING
        )
        source_windows = {
            domain: exp26.exp251.frame_windows(
                normalized[domain], feature_names, cfg
            )
            for domain in cfg["source_domains"]
        }
        prior, _ = exp26.exp24.train_only_prior(
            cfg, exp26.PREPROCESSING, args.sensor_graph_k
        )
        model, checkpoint = exp26.exp251.build_model(
            run_args, cfg, prior, len(feature_names)
        )
        exp26.disable_context_graph(model)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial = deepcopy(model.base.predictor).cpu()
        for parameter in initial.parameters():
            parameter.requires_grad_(True)

        print(f"[seed={model_seed}] 编码冻结的稳定传感器图特征")
        encoded_sources = {
            domain: exp26.encode_data(model, data, device)
            for domain, data in source_windows.items()
        }

        states = {"static_init": state_to_cpu(initial)}
        random_bank = EpisodeBank(
            encoded_sources,
            "random",
            args.support_engines,
            args.query_engines,
            model_seed + 1000,
            exp26.exp18._sample_balanced_indices,
        )
        supervised, _ = exp26.train_supervised_budget(
            run_args, initial, random_bank, device
        )
        states["supervised_budget"] = state_to_cpu(supervised)

        for mode in args.task_modes:
            audit_bank = EpisodeBank(
                encoded_sources,
                mode,
                args.support_engines,
                args.query_engines,
                model_seed + 2000,
                exp26.exp18._sample_balanced_indices,
            )
            audit = audit_task_alignment(
                initial,
                audit_bank,
                args.audit_tasks,
                args.inner_lr,
                device,
            )
            all_audits.append({"model_seed": model_seed, "mode": mode, **audit})

            meta_bank = EpisodeBank(
                encoded_sources,
                mode,
                args.support_engines,
                args.query_engines,
                model_seed + 3000,
                exp26.exp18._sample_balanced_indices,
            )
            meta_model, _ = exp26.train_fomaml(
                run_args, initial, meta_bank, device
            )
            states[f"fomaml_{mode}"] = state_to_cpu(meta_model)

        target_frame = normalized[args.target]
        validation_units = set(map(int, protocol["validation_units"]))
        validation_windows = exp26.exp251.frame_windows(
            target_frame[target_frame["unit"].isin(validation_units)],
            feature_names,
            cfg,
        )
        validation = exp26.encode_data(model, validation_windows, device)
        nested = protocol["nested_adaptation_units_by_target_split_seed"]

        for split_seed in args.target_split_seeds:
            support_units = set(map(int, nested[str(split_seed)][str(args.k)]))
            if support_units & validation_units:
                raise AssertionError("目标support与validation发动机重叠")
            support_windows = exp26.exp251.frame_windows(
                target_frame[target_frame["unit"].isin(support_units)],
                feature_names,
                cfg,
            )
            support = exp26.encode_data(model, support_windows, device)
            run_seed = exp26.exp18.target_run_seed(model_seed, split_seed)
            for regime, state in states.items():
                _, best_epoch, metrics = exp26.train_target_predictor(
                    run_args,
                    initial,
                    state,
                    support,
                    validation,
                    run_seed,
                    device,
                )
                all_results.append(
                    {
                        "target": args.target,
                        "model_seed": model_seed,
                        "target_split_seed": split_seed,
                        "k": args.k,
                        "regime": regime,
                        "best_epoch": best_epoch,
                        "support_engine_count": len(support_units),
                        "validation_engine_count": len(validation_units),
                        **metrics,
                    }
                )
                print(
                    f"[结果] seed={model_seed} split={split_seed} "
                    f"{regime}: RMSE={metrics['rmse']:.4f}, "
                    f"NASA={metrics['nasa_score']:.2f}"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(all_results)
    audits = pd.DataFrame(all_audits)
    summary = (
        raw.groupby(["target", "k", "regime"], as_index=False)
        .agg(
            n_cells=("rmse", "size"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            nasa_mean=("nasa_score", "mean"),
        )
        .sort_values("rmse_mean")
    )
    audit_summary = (
        audits.groupby("mode", as_index=False)
        .agg(
            n_seeds=("model_seed", "nunique"),
            gradient_cosine_mean=("gradient_cosine_mean", "mean"),
            positive_gradient_rate=("positive_gradient_rate", "mean"),
            adaptation_gain_mean=("adaptation_gain_mean", "mean"),
            positive_adaptation_rate=("positive_adaptation_rate", "mean"),
        )
        .sort_values("gradient_cosine_mean", ascending=False)
    )
    comparisons = paired_comparisons(raw)
    raw.to_csv(output / "experiment1_raw.csv", index=False)
    summary.to_csv(output / "experiment1_summary.csv", index=False)
    audits.to_csv(output / "experiment1_task_audit_raw.csv", index=False)
    audit_summary.to_csv(output / "experiment1_task_audit_summary.csv", index=False)
    comparisons.to_csv(output / "experiment1_comparisons.csv", index=False)
    (output / "experiment1_protocol.json").write_text(
        json.dumps(
            {
                "project_root": str(root),
                "target": args.target,
                "model_seeds": args.model_seeds,
                "target_split_seeds": args.target_split_seeds,
                "k": args.k,
                "task_modes": args.task_modes,
                "stage_bins": STAGE_BINS,
                "support_query_engine_disjoint": True,
                "source_backbone": "frozen source-pretrained stable sensor graph",
                "target_adaptation": "predictor only",
                "official_test_accessed": False,
                "protocol_path": str(protocol_path),
                "initialization_checkpoint": str(checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n任务机制审计：")
    print(audit_summary.to_string(index=False))
    print("\n目标域性能：")
    print(summary.to_string(index=False))
    print("\n配对比较：")
    print(comparisons.to_string(index=False))
    print(f"\n完成：{output}")


if __name__ == "__main__":
    main()
