"""Experiment 18: support-conditioned sensor-graph meta-learning.

Experiment 17B established that a fixed, source-only sensor prior is useful,
but it did not test meta-learning.  This experiment keeps that validated prior
and asks one focused question:

    Does a graph correction inferred from the labelled target support set
    improve over an equally trained static prior graph?

The task-conditioned model (TCSG) receives a permutation-invariant summary of
the support set.  A context encoder converts that summary into low-rank edge
biases which *reweight only edges already allowed by the source prior*.  This
is deliberately conservative: Experiment 18 tests task conditioning before
allowing the model to invent new sensor edges.

Source training has two stages:

1. ordinary multi-source supervised pretraining;
2. engine-disjoint support/query episodes.  The support engines construct the
   graph context and query engines supply the outer loss.

The budget-matched static control receives the same number of additional
source optimizer steps.  Three negative controls reuse exactly the same TCSG
weights but alter only the target support context:

``tcsg_sensor_permuted``
    Permute sensor identities only while constructing the support context.
``tcsg_label_shuffled``
    Permute support labels, preserving the label histogram but destroying the
    sensor-RUL alignment used by the context.
``tcsg_zero``
    Remove all support information by supplying a zero summary.

No official test prediction is made by default.  Run from the project root.

Dry run::

    python scripts/experiment18_task_conditioned_sensor_graph.py \
      --target FD004 --dry-run

Formal validation run::

    python -u scripts/experiment18_task_conditioned_sensor_graph.py \
      --target FD004 \
      --k-values 2 5 \
      --target-split-seeds 3027 3028 3029 3030 3031 \
      --model-seeds 42 43 44 45 46 \
      --source-pretrain-steps 1500 \
      --context-meta-steps 600 \
      --target-epochs 10 \
      --evaluation-scope validation \
      --resume
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts import experiment17b_controlled_sensor_graph as exp17b  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    prepare_kshot_experiment,
    resolve_device,
    resolve_path,
    seed_everything,
)
from scripts.experiment8_transfer_baseline import train_source_supervised  # noqa: E402
from scripts.run_condition_aware_experiment import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
)


SCRIPT_VERSION = "experiment18_task_conditioned_sensor_graph_v1"
MODEL_CHOICES = (
    "static_prior",
    "static_budget_prior",
    "tcsg_true",
    "tcsg_sensor_permuted",
    "tcsg_label_shuffled",
    "tcsg_zero",
)
DEFAULT_MODELS = list(MODEL_CHOICES)
METRICS = ("rmse", "mae", "r2", "nasa_score")
LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}
COMPARISONS = (
    ("tcsg_true", "static_budget_prior", "tcsg_true_vs_static_budget"),
    ("tcsg_true", "static_prior", "tcsg_true_vs_static_prior"),
    ("tcsg_true", "tcsg_sensor_permuted", "true_vs_sensor_permuted"),
    ("tcsg_true", "tcsg_label_shuffled", "true_vs_label_shuffled"),
    ("tcsg_true", "tcsg_zero", "true_vs_zero_context"),
    (
        "static_budget_prior",
        "static_prior",
        "static_budget_vs_static_prior",
    ),
)
PRIMARY_COMPARISONS = {
    "tcsg_true_vs_static_budget",
    "true_vs_sensor_permuted",
    "true_vs_label_shuffled",
    "true_vs_zero_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验18：支持集条件化低秩传感器图元学习"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES), default="FD004"
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=DEFAULT_MODELS)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--target-split-seeds",
        nargs="+",
        type=int,
        default=[3027, 3028, 3029, 3030, 3031],
    )
    parser.add_argument(
        "--model-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
    )
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument(
        "--experiment17-protocol",
        help="可选：实验17固定协议；默认优先读取实验17B协议",
    )
    parser.add_argument(
        "--experiment17b-protocol",
        default=None,
        help="可选：实验17B交叉协议；默认按--target自动定位",
    )
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument("--balance-mode", choices=BALANCE_MODES, default="engine_stage")
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--context-meta-steps",
        type=int,
        default=600,
        help="TCSG外循环步数；静态预算控制获得相同步数的普通训练",
    )
    parser.add_argument("--context-meta-lr", type=float, default=0.0001)
    parser.add_argument("--context-meta-weight-decay", type=float, default=0.0)
    parser.add_argument("--source-support-engines", type=int, default=5)
    parser.add_argument("--source-query-engines", type=int, default=5)
    parser.add_argument("--source-support-windows", type=int, default=128)
    parser.add_argument("--source-query-windows", type=int, default=128)
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--graph-residual-rank", type=int, default=4)
    parser.add_argument("--max-graph-gate", type=float, default=0.5)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument(
        "--evaluation-scope",
        choices=("validation", "official_test"),
        default="validation",
    )
    parser.add_argument("--confirm-official-test", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--output-dir", default="outputs/experiment18_task_conditioned_sensor_graph"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-target-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace, model_seed: int) -> dict:
    path = resolve_path(args.config)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["seed"] = int(model_seed)
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain for domain in EXPECTED_OFFICIAL_TEST_ENGINES if domain != args.target
    ]
    cfg["normalizer_seed"] = int(args.normalizer_seed)
    cfg["condition_count"] = int(args.condition_count)
    cfg["source_pretrain_steps"] = int(args.source_pretrain_steps)
    cfg["source_pretrain_lr"] = float(args.source_pretrain_lr)
    cfg["source_pretrain_weight_decay"] = float(args.source_pretrain_weight_decay)
    cfg["target_epochs"] = int(args.target_epochs)
    cfg["target_lr"] = float(args.target_lr)
    cfg["pair_aux_weight"] = 0.0
    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir))
    cfg["output_dir"] = str(resolve_path(args.output_dir))
    return cfg


def load_or_create_protocol(args: argparse.Namespace, cfg: dict) -> dict:
    candidate = (
        resolve_path(args.experiment17b_protocol)
        if args.experiment17b_protocol
        else PROJECT_ROOT
        / "outputs"
        / "experiment17b_controlled_sensor_graph"
        / f"experiment17b_{args.target}_protocol.json"
    )
    if candidate.is_file():
        protocol = json.loads(candidate.read_text(encoding="utf-8"))
        if protocol.get("target_domain") != args.target:
            raise ValueError("实验17B协议target_domain与--target不一致")
        required_splits = {str(value) for value in args.target_split_seeds}
        nested = protocol.get("nested_adaptation_units_by_target_split_seed", {})
        if required_splits.issubset(nested):
            for split_seed in args.target_split_seeds:
                for k in args.k_values:
                    if str(k) not in nested[str(split_seed)]:
                        raise ValueError(f"实验17B协议缺少split={split_seed}, K={k}")
            print(f"[protocol] 复用实验17B交叉协议：{candidate}")
            return protocol
        print("[protocol] 实验17B协议不含全部请求划分，重新按相同规则生成。")

    base, source_path = exp17b.load_or_create_base_protocol(args, cfg)
    return exp17b.build_crossed_protocol(args, cfg, base, source_path)


def context_feature_dim(sensor_count: int, condition_dim: int) -> int:
    # mean, std, slope, sensor-RUL correlation, sensor correlation matrix,
    # condition mean/std, and four label statistics.
    return 4 * sensor_count + sensor_count * sensor_count + 2 * condition_dim + 4


def _safe_standard_deviation(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    return values.std(dim=dim, unbiased=False).clamp_min(1e-6)


def support_summary(
    x: torch.Tensor,
    y: torch.Tensor,
    sensor_count: int,
    rul_cap: float,
    *,
    sensor_permutation: torch.Tensor | None = None,
    label_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a fixed-length permutation-invariant support-set summary."""
    if x.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError("support x/y形状不一致")
    sensors = x[:, :, :sensor_count]
    if sensor_permutation is not None:
        sensors = sensors[:, :, sensor_permutation]
    labels = y[label_permutation] if label_permutation is not None else y

    flat = sensors.reshape(-1, sensor_count)
    mean = flat.mean(dim=0)
    std = _safe_standard_deviation(flat, dim=0)
    slope = (sensors[:, -1] - sensors[:, 0]).mean(dim=0)

    centered = flat - mean
    covariance = centered.transpose(0, 1) @ centered / max(1, flat.shape[0] - 1)
    correlation = covariance / (std[:, None] * std[None, :])
    correlation = torch.nan_to_num(correlation).clamp(-1.0, 1.0)

    window_sensor_mean = sensors.mean(dim=1)
    ws_centered = window_sensor_mean - window_sensor_mean.mean(dim=0)
    normalized_labels = labels / float(rul_cap)
    y_centered = normalized_labels - normalized_labels.mean()
    sensor_label_covariance = (ws_centered * y_centered[:, None]).mean(dim=0)
    sensor_label_std = _safe_standard_deviation(window_sensor_mean, dim=0)
    y_std = normalized_labels.std(unbiased=False).clamp_min(1e-6)
    sensor_label_correlation = torch.nan_to_num(
        sensor_label_covariance / (sensor_label_std * y_std)
    ).clamp(-1.0, 1.0)

    condition = x[:, :, sensor_count:]
    if condition.shape[-1]:
        condition_flat = condition.reshape(-1, condition.shape[-1])
        condition_features = torch.cat(
            [
                condition_flat.mean(dim=0),
                _safe_standard_deviation(condition_flat, dim=0),
            ]
        )
    else:
        condition_features = x.new_empty(0)

    label_features = torch.stack(
        [
            normalized_labels.mean(),
            normalized_labels.std(unbiased=False),
            normalized_labels.min(),
            normalized_labels.max(),
        ]
    )
    summary = torch.cat(
        [
            mean,
            std,
            slope,
            sensor_label_correlation,
            correlation.flatten(),
            condition_features,
            label_features,
        ]
    )
    if not bool(torch.isfinite(summary).all()):
        raise RuntimeError("support任务摘要包含NaN/Inf")
    return summary


class TaskContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim == 1:
            summary = summary.unsqueeze(0)
        return self.network(summary)


class TaskConditionedSensorGraphLayer(nn.Module):
    """Prior-masked attention plus support-conditioned low-rank edge bias."""

    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        dropout: float,
        adjacency: torch.Tensor,
        context_dim: int,
        residual_rank: int,
        max_gate: float,
    ):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim必须能被gat_heads整除")
        self.embedding_dim = int(embedding_dim)
        self.heads = int(heads)
        self.head_dim = embedding_dim // heads
        self.nodes = int(adjacency.shape[0])
        self.rank = int(residual_rank)
        self.max_gate = float(max_gate)
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("adjacency", adjacency.to(dtype=torch.bool))
        output_size = heads * self.nodes * residual_rank
        self.context_u = nn.Linear(context_dim, output_size)
        self.context_v = nn.Linear(context_dim, output_size)
        self.context_gate = nn.Linear(context_dim, heads)
        nn.init.normal_(self.context_u.weight, std=0.005)
        nn.init.normal_(self.context_v.weight, std=0.005)
        nn.init.zeros_(self.context_u.bias)
        nn.init.zeros_(self.context_v.bias)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -2.0)

    def edge_residual(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        shape = (batch, self.heads, self.nodes, self.rank)
        u = self.context_u(context).reshape(shape)
        v = self.context_v(context).reshape(shape)
        residual = torch.einsum("bhir,bhjr->bhij", u, v) / math.sqrt(self.rank)
        residual = 0.5 * (residual + residual.transpose(-1, -2))
        residual = torch.tanh(residual)
        gate = self.max_gate * torch.sigmoid(self.context_gate(context))
        return residual, gate

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        if context.shape[0] == 1 and batch > 1:
            context = context.expand(batch, -1)
        qkv = self.qkv(x).reshape(batch, nodes, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        residual, gate = self.edge_residual(context)
        logits = logits + gate[:, :, None, None] * residual
        allowed = self.adjacency.view(1, 1, nodes, nodes)
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = self.dropout(F.softmax(logits, dim=-1))
        mixed = torch.einsum("bhij,bhjd->bhid", attention, v)
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        x = self.norm1(x + self.dropout(self.output(mixed)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TaskConditionedSensorGraphRegressor(nn.Module):
    def __init__(
        self,
        sensor_count: int,
        condition_dim: int,
        embedding_dim: int,
        heads: int,
        dropout: float,
        adjacency: torch.Tensor,
        context_hidden_dim: int,
        context_dim: int,
        residual_rank: int,
        max_gate: float,
    ):
        super().__init__()
        self.sensor_count = int(sensor_count)
        self.condition_dim = int(condition_dim)
        self.summary_dim = context_feature_dim(sensor_count, condition_dim)
        self.temporal = exp17.SharedSensorTemporalEncoder(embedding_dim, dropout)
        self.task_context_encoder = TaskContextEncoder(
            self.summary_dim, context_hidden_dim, context_dim
        )
        self.graph_layers = nn.ModuleList(
            [
                TaskConditionedSensorGraphLayer(
                    embedding_dim,
                    heads,
                    dropout,
                    adjacency,
                    context_dim,
                    residual_rank,
                    max_gate,
                )
                for _ in range(2)
            ]
        )
        self.pool_score = nn.Linear(embedding_dim, 1)
        if condition_dim:
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
            )
            fusion_dim = 2 * embedding_dim
        else:
            self.condition_encoder = None
            fusion_dim = embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, task_summary: torch.Tensor | None = None):
        if task_summary is None:
            task_summary = x.new_zeros(self.summary_dim)
        context = self.task_context_encoder(task_summary.to(x.device))
        sensor_window = x[:, :, : self.sensor_count]
        nodes = self.temporal(sensor_window)
        for layer in self.graph_layers:
            nodes = layer(nodes, context)
        pooling = F.softmax(self.pool_score(nodes).squeeze(-1), dim=-1)
        graph_feature = torch.sum(nodes * pooling.unsqueeze(-1), dim=1)
        if self.condition_encoder is not None:
            conditions = x[:, :, self.sensor_count :].mean(dim=1)
            graph_feature = torch.cat(
                [graph_feature, self.condition_encoder(conditions)], dim=-1
            )
        return self.predictor(graph_feature).squeeze(-1)

    @torch.no_grad()
    def graph_gate_mean(self, task_summary: torch.Tensor) -> float:
        device = next(self.parameters()).device
        context = self.task_context_encoder(task_summary.to(device))
        gates = [layer.edge_residual(context)[1] for layer in self.graph_layers]
        return float(torch.stack(gates).mean().item())


def build_static_model(feature_count: int, cfg: dict, prior: torch.Tensor) -> nn.Module:
    return exp17.build_ablation_model("sensor_graph_prior", feature_count, cfg, prior)


def build_tcsg_model(
    feature_count: int, cfg: dict, prior: torch.Tensor, args: argparse.Namespace
) -> TaskConditionedSensorGraphRegressor:
    sensor_count = len(cfg["sensor_columns"])
    return TaskConditionedSensorGraphRegressor(
        sensor_count=sensor_count,
        condition_dim=feature_count - sensor_count,
        embedding_dim=int(cfg["embedding_dim"]),
        heads=int(cfg["gat_heads"]),
        dropout=float(cfg["dropout"]),
        adjacency=prior,
        context_hidden_dim=args.context_hidden_dim,
        context_dim=args.context_dim,
        residual_rank=args.graph_residual_rank,
        max_gate=args.max_graph_gate,
    )


def load_compatible_state(model: nn.Module, source: dict[str, torch.Tensor]) -> None:
    destination = model.state_dict()
    copied = 0
    for name, value in destination.items():
        if name in source and source[name].shape == value.shape:
            destination[name] = source[name].detach().clone()
            copied += 1
    model.load_state_dict(destination)
    if copied == 0:
        raise RuntimeError("普通先验图状态没有任何参数可复制到TCSG")


def _sample_balanced_indices(
    units: np.ndarray,
    labels: np.ndarray,
    selected_units: Iterable[int],
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_units = list(int(value) for value in selected_units)
    buckets: list[np.ndarray] = []
    stages = np.digitize(labels, bins=[30.0, 60.0, 90.0], right=True)
    for unit in selected_units:
        for stage in np.unique(stages[units == unit]):
            bucket = np.flatnonzero((units == unit) & (stages == stage))
            if len(bucket):
                buckets.append(bucket)
    if not buckets:
        raise RuntimeError("源域episode没有可抽样窗口")
    per_bucket = max(1, math.ceil(total / len(buckets)))
    sampled = [
        rng.choice(bucket, size=per_bucket, replace=len(bucket) < per_bucket)
        for bucket in buckets
    ]
    merged = np.concatenate(sampled)
    rng.shuffle(merged)
    if len(merged) < total:
        merged = rng.choice(merged, size=total, replace=True)
    return merged[:total]


class SourceEpisodeBank:
    def __init__(self, source_tasks: dict, seed: int):
        self.datasets = {name: loader.dataset for name, loader in source_tasks.items()}
        self.domains = sorted(self.datasets)
        self.rng = np.random.default_rng(seed)
        for domain, dataset in self.datasets.items():
            if dataset.units is None:
                raise RuntimeError(f"源域{domain}数据集缺少发动机unit")

    def sample(
        self,
        support_engines: int,
        query_engines: int,
        support_windows: int,
        query_windows: int,
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        domain = self.domains[int(self.rng.integers(0, len(self.domains)))]
        dataset = self.datasets[domain]
        units = np.asarray(dataset.units, dtype=int)
        unique = np.unique(units)
        needed = support_engines + query_engines
        if needed > len(unique):
            raise ValueError(
                f"源域{domain}仅{len(unique)}台发动机，不能互斥抽取{needed}台"
            )
        selected = self.rng.choice(unique, size=needed, replace=False)
        support_units = selected[:support_engines]
        query_units = selected[support_engines:]
        labels = dataset.y.detach().cpu().numpy()
        support_indices = _sample_balanced_indices(
            units, labels, support_units, support_windows, self.rng
        )
        query_indices = _sample_balanced_indices(
            units, labels, query_units, query_windows, self.rng
        )
        return (
            domain,
            dataset.x[support_indices],
            dataset.y[support_indices],
            dataset.x[query_indices],
            dataset.y[query_indices],
        )


def train_context_meta(
    model: TaskConditionedSensorGraphRegressor,
    source_tasks: dict,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TaskConditionedSensorGraphRegressor, list[dict]]:
    learner = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(
        learner.parameters(),
        lr=args.context_meta_lr,
        weight_decay=args.context_meta_weight_decay,
    )
    bank = SourceEpisodeBank(source_tasks, cfg["seed"] + 18018)
    report_every = max(1, args.context_meta_steps // 10)
    running: list[float] = []
    history: list[dict] = []
    for step in range(1, args.context_meta_steps + 1):
        domain, sx, sy, qx, qy = bank.sample(
            args.source_support_engines,
            args.source_query_engines,
            args.source_support_windows,
            args.source_query_windows,
        )
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        summary = support_summary(
            sx, sy, learner.sensor_count, cfg["rul_cap"]
        )
        learner.train()
        optimizer.zero_grad()
        prediction = learner(qx, summary)
        loss = F.mse_loss(prediction, qy)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("TCSG源域episode训练出现NaN/Inf")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0)
        optimizer.step()
        running.append(float(loss.item()))
        if step % report_every == 0 or step == args.context_meta_steps:
            row = {
                "meta_step": step,
                "mean_query_loss": float(np.mean(running)),
                "last_domain": domain,
                "last_gradient_norm": float(gradient_norm),
                "last_graph_gate_mean": learner.graph_gate_mean(summary),
            }
            history.append(row)
            print(
                f"context_meta_step={step:04d}/{args.context_meta_steps} "
                f"query_loss={row['mean_query_loss']:.4f} "
                f"gate={row['last_graph_gate_mean']:.4f}"
            )
            running.clear()
    return learner, history


def source_signature(
    args: argparse.Namespace,
    cfg: dict,
    feature_count: int,
    prior: torch.Tensor,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "model_seed": cfg["seed"],
        "feature_count": feature_count,
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "source_pretrain_steps": args.source_pretrain_steps,
        "source_pretrain_lr": args.source_pretrain_lr,
        "context_meta_steps": args.context_meta_steps,
        "context_meta_lr": args.context_meta_lr,
        "source_support_engines": args.source_support_engines,
        "source_query_engines": args.source_query_engines,
        "source_support_windows": args.source_support_windows,
        "source_query_windows": args.source_query_windows,
        "context_hidden_dim": args.context_hidden_dim,
        "context_dim": args.context_dim,
        "graph_residual_rank": args.graph_residual_rank,
        "max_graph_gate": args.max_graph_gate,
        "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest()[:16],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def load_or_train_source_bundle(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    prior: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list], dict]:
    first_split = args.target_split_seeds[0]
    first_k = min(args.k_values)
    units = protocol["nested_adaptation_units_by_target_split_seed"][str(first_split)][
        str(first_k)
    ]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks, _, _, _, feature_count, _ = loaders
    signature = source_signature(args, cfg, feature_count, prior)
    path = (
        Path(cfg["output_dir"])
        / "source_cache"
        / f"experiment18_source_bundle_{args.target}_modelseed{cfg['seed']}.pt"
    )
    if args.resume and path.is_file():
        cached = exp17.safe_torch_load(path)
        if cached.get("signature") == signature:
            print(f"[source cache] {path}")
            return cached["states"], cached.get("histories", {}), cached["inventory"]
        print(f"[source cache ignored] 签名不匹配：{path}")

    seed_everything(cfg["seed"])
    static_model = build_static_model(feature_count, cfg, prior)
    static_model, ordinary_history = train_source_supervised(
        static_model, source_tasks, cfg, resolve_device(cfg["device"])
    )
    ordinary_state = exp17.state_to_cpu(static_model)
    static_total, static_head = exp17.parameter_count(static_model)
    del static_model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Exact optimizer-step control: continue ordinary supervised source training
    # for the same number of additional steps used by the episodic TCSG stage.
    continuation_cfg = dict(cfg)
    continuation_cfg["seed"] = int(cfg["seed"]) + 50000
    continuation_cfg["source_pretrain_steps"] = int(args.context_meta_steps)
    budget_model = build_static_model(feature_count, cfg, prior)
    budget_model.load_state_dict(ordinary_state)
    fresh = prepare_kshot_experiment(
        continuation_cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    budget_model, budget_history = train_source_supervised(
        budget_model, fresh[0], continuation_cfg, resolve_device(cfg["device"])
    )
    budget_state = exp17.state_to_cpu(budget_model)
    del budget_model, fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed_everything(cfg["seed"] + 18018)
    tcsg = build_tcsg_model(feature_count, cfg, prior, args)
    load_compatible_state(tcsg, ordinary_state)
    fresh_meta = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    tcsg, meta_history = train_context_meta(
        tcsg, fresh_meta[0], cfg, args, resolve_device(cfg["device"])
    )
    tcsg_state = exp17.state_to_cpu(tcsg)

    tcsg_total, tcsg_head = exp17.parameter_count(tcsg)
    inventory = {
        "model_seed": int(cfg["seed"]),
        "feature_count": int(feature_count),
        "static_total_parameter_count": static_total,
        "static_predictor_parameter_count": static_head,
        "tcsg_total_parameter_count": tcsg_total,
        "tcsg_predictor_parameter_count": tcsg_head,
        "tcsg_context_parameter_count": int(
            sum(
                parameter.numel()
                for name, parameter in tcsg.named_parameters()
                if "context" in name
            )
        ),
    }
    states = {
        "static_prior": ordinary_state,
        "static_budget_prior": budget_state,
        "tcsg": tcsg_state,
    }
    histories = {
        "ordinary": ordinary_history,
        "static_budget": budget_history,
        "tcsg_meta": meta_history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "states": states,
            "histories": histories,
            "inventory": inventory,
        },
        path,
    )
    del tcsg, fresh_meta
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return states, histories, inventory


def context_for_mode(
    mode: str,
    x: torch.Tensor,
    y: torch.Tensor,
    sensor_count: int,
    rul_cap: float,
    seed: int,
) -> torch.Tensor:
    if mode == "tcsg_zero":
        condition_dim = x.shape[-1] - sensor_count
        return x.new_zeros(context_feature_dim(sensor_count, condition_dim))
    generator = torch.Generator().manual_seed(int(seed))
    if mode == "tcsg_sensor_permuted":
        permutation = torch.randperm(sensor_count, generator=generator)
        return support_summary(
            x, y, sensor_count, rul_cap, sensor_permutation=permutation
        )
    if mode == "tcsg_label_shuffled":
        permutation = torch.randperm(len(y), generator=generator)
        return support_summary(
            x, y, sensor_count, rul_cap, label_permutation=permutation
        )
    return support_summary(x, y, sensor_count, rul_cap)


def predict_context(
    model: TaskConditionedSensorGraphRegressor,
    loader,
    device: torch.device,
    summary: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    summary = summary.to(device)
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device), summary)
            labels.extend(y.numpy().tolist())
            predictions.extend(output.detach().cpu().numpy().tolist())
    return np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)


def evaluate_context(model, loader, device, summary) -> dict:
    labels, predictions = predict_context(model, loader, device, summary)
    from evaluation.metrics import regression_metrics

    return regression_metrics(labels, predictions)


def train_target_context_head(
    model: TaskConditionedSensorGraphRegressor,
    support,
    validation,
    cfg: dict,
    device: torch.device,
    summary: torch.Tensor,
) -> tuple[TaskConditionedSensorGraphRegressor, list[dict], int]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    optimizer = torch.optim.Adam(trainable, lr=cfg["target_lr"])
    best_state = deepcopy(learner.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []
    summary = summary.to(device)
    for epoch in range(1, cfg["target_epochs"] + 1):
        learner.train()
        losses = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(learner(x, summary), y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("目标预测头训练出现NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate_context(learner, validation, device, summary)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        print(
            f"target_epoch={epoch:03d}/{cfg['target_epochs']} "
            f"train_loss={np.mean(losses):.4f} val_rmse={metrics['rmse']:.4f}"
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            best_state = deepcopy(learner.state_dict())
    learner.load_state_dict(best_state)
    return learner, history, best_epoch


def target_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = f"{model_seed}:{target_split_seed}:experiment18".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def run_target_cell(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    regime: str,
    state: dict[str, torch.Tensor],
    inventory: dict,
    target_split_seed: int,
    k: int,
    prior: torch.Tensor,
) -> tuple[dict, dict]:
    units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(target_split_seed)
    ][str(k)]
    run_seed = target_run_seed(cfg["seed"], target_split_seed)
    target_cfg = dict(cfg)
    target_cfg["seed"] = run_seed
    loaders = prepare_kshot_experiment(
        target_cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    _, support, validation, test, feature_count, split = loaders
    if split["official_test_units_hash"] != protocol["official_test_units_hash"]:
        raise AssertionError("官方测试发动机集合发生变化")
    seed_everything(run_seed)
    device = resolve_device(cfg["device"])

    context_record: dict = {
        "target_split_seed": int(target_split_seed),
        "model_seed": int(cfg["seed"]),
        "k": int(k),
        "regime": regime,
    }
    if regime.startswith("tcsg_"):
        model = build_tcsg_model(feature_count, cfg, prior, args)
        model.load_state_dict(state)
        raw_x = support.dataset.x
        raw_y = support.dataset.y
        true_summary = support_summary(
            raw_x, raw_y, model.sensor_count, cfg["rul_cap"]
        )
        summary = context_for_mode(
            regime,
            raw_x,
            raw_y,
            model.sensor_count,
            cfg["rul_cap"],
            seed=run_seed + 1818,
        )
        cosine = float(
            F.cosine_similarity(summary[None], true_summary[None]).item()
        )
        model, history, best_epoch = train_target_context_head(
            model, support, validation, target_cfg, device, summary
        )
        validation_metrics = evaluate_context(model, validation, device, summary)
        if args.evaluation_scope == "official_test":
            selected_metrics = evaluate_context(model, test, device, summary)
            official_metrics = dict(selected_metrics)
        else:
            selected_metrics = dict(validation_metrics)
            official_metrics = None
        gate = model.graph_gate_mean(summary)
        total = inventory["tcsg_total_parameter_count"]
        head = inventory["tcsg_predictor_parameter_count"]
        context_record.update(
            {
                "summary_norm": float(summary.norm().item()),
                "true_summary_norm": float(true_summary.norm().item()),
                "cosine_to_true_context": cosine,
                "graph_gate_mean": gate,
                "support_window_count": int(len(raw_y)),
            }
        )
    else:
        model = build_static_model(feature_count, cfg, prior)
        model.load_state_dict(state)
        model, history, best_epoch = exp17.train_target_head(
            model, support, validation, target_cfg, device
        )
        validation_metrics = exp17.evaluate(model, validation, device)
        if args.evaluation_scope == "official_test":
            selected_metrics = exp17.evaluate(model, test, device)
            official_metrics = dict(selected_metrics)
        else:
            selected_metrics = dict(validation_metrics)
            official_metrics = None
        gate = 0.0
        total = inventory["static_total_parameter_count"]
        head = inventory["static_predictor_parameter_count"]
        context_record.update(
            {
                "summary_norm": 0.0,
                "true_summary_norm": 0.0,
                "cosine_to_true_context": float("nan"),
                "graph_gate_mean": 0.0,
                "support_window_count": int(len(support.dataset)),
            }
        )

    result = {
        **selected_metrics,
        "evaluation_scope": args.evaluation_scope,
        "model": regime,
        "source_training": (
            "ordinary_multisource_pretraining"
            if regime == "static_prior"
            else "ordinary_budget_matched_continuation"
            if regime == "static_budget_prior"
            else "engine_disjoint_support_query_context_meta_training"
        ),
        "context_mode": regime.replace("tcsg_", "") if regime.startswith("tcsg_") else "none",
        "target_adaptation_scope": "predictor_only",
        "target_domain": args.target,
        "model_seed": int(cfg["seed"]),
        "target_split_seed": int(target_split_seed),
        "target_run_seed": int(run_seed),
        "k": int(k),
        "adaptation_units": [int(value) for value in units],
        "adaptation_engine_count": len(units),
        "validation_engine_count": len(protocol["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split["official_test_units_hash"],
        "official_test_forward_run": args.evaluation_scope == "official_test",
        "official_test_metrics": official_metrics,
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": args.target_epochs,
        "target_learning_rate": args.target_lr,
        "source_pretrain_steps": args.source_pretrain_steps,
        "context_meta_or_extra_steps": args.context_meta_steps,
        "total_parameter_count": int(total),
        "target_trainable_parameter_count": int(head),
        "graph_gate_mean": float(gate),
        "context_cosine_to_true": context_record["cosine_to_true_context"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
    }
    if args.save_target_checkpoints:
        directory = Path(cfg["output_dir"]) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"experiment18_{regime}_k{k}_{args.target}_"
            f"split{target_split_seed}_model{cfg['seed']}.pt"
        )
        torch.save(
            {
                "model": exp17.state_to_cpu(model),
                "metrics": result,
                "history": history,
                "context_audit": context_record,
            },
            path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, context_record


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows = []
    for (k, model), group in frame.groupby(["k", "model"]):
        row = {
            "k": int(k),
            "model": model,
            "n_cells": int(len(group)),
            "n_target_splits": int(group["target_split_seed"].nunique()),
            "n_model_seeds": int(group["model_seed"].nunique()),
            "total_parameter_count": int(group["total_parameter_count"].iloc[0]),
            "target_trainable_parameter_count": int(
                group["target_trainable_parameter_count"].iloc[0]
            ),
            "graph_gate_mean": float(group["graph_gate_mean"].mean()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_cell_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
            split_means = group.groupby("target_split_seed")[metric].mean()
            row[f"{metric}_target_split_std"] = (
                float(split_means.std(ddof=1)) if len(split_means) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def paired_cells(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows = []
    for (k, split_seed, model_seed), group in frame.groupby(
        ["k", "target_split_seed", "model_seed"]
    ):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        for candidate, reference, comparison in COMPARISONS:
            if candidate not in by_model or reference not in by_model:
                continue
            row = {
                "k": int(k),
                "target_split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
            }
            for metric in METRICS:
                delta = float(by_model[candidate][metric] - by_model[reference][metric])
                row[f"{metric}_delta_candidate_minus_reference"] = delta
                row[f"{metric}_candidate_win"] = float(
                    delta < 0 if metric in LOWER_IS_BETTER else delta > 0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_by_target_split(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in paired.columns
        if column.endswith("_delta_candidate_minus_reference")
    ]
    grouped = paired.groupby(
        ["k", "target_split_seed", "comparison", "candidate", "reference"],
        as_index=False,
    )[numeric].mean()
    grouped["rmse_split_win"] = (
        grouped["rmse_delta_candidate_minus_reference"] < 0
    ).astype(float)
    return grouped.sort_values(["k", "comparison", "target_split_seed"])


def comparison_summary(
    results: list[dict], paired: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    raw = pd.DataFrame(results)
    rows = []
    for (k, comparison), group in paired.groupby(["k", "comparison"]):
        candidate = group["candidate"].iloc[0]
        reference = group["reference"].iloc[0]
        split_means = group.groupby("target_split_seed")[
            "rmse_delta_candidate_minus_reference"
        ].mean()
        low, high = exp17b.hierarchical_bootstrap_ci(
            group,
            "rmse_delta_candidate_minus_reference",
            repetitions,
            seed=18018 + int(k) + sum(map(ord, comparison)),
        )
        reference_rmse = raw[
            (raw["k"] == k) & (raw["model"] == reference)
        ]["rmse"].mean()
        rows.append(
            {
                "k": int(k),
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
                "n_cells": int(len(group)),
                "n_target_splits": int(group["target_split_seed"].nunique()),
                "n_model_seeds": int(group["model_seed"].nunique()),
                "rmse_delta_mean": float(
                    group["rmse_delta_candidate_minus_reference"].mean()
                ),
                "rmse_improvement_pct": float(
                    -100.0
                    * group["rmse_delta_candidate_minus_reference"].mean()
                    / reference_rmse
                ),
                "rmse_cell_win_rate": float(group["rmse_candidate_win"].mean()),
                "rmse_target_split_win_rate": float((split_means < 0).mean()),
                "rmse_hier_boot_ci95_low": low,
                "rmse_hier_boot_ci95_high": high,
                "rmse_split_t_p": exp17b.split_level_pvalue(
                    split_means.to_numpy(float)
                ),
                "mae_delta_mean": float(
                    group["mae_delta_candidate_minus_reference"].mean()
                ),
                "r2_delta_mean": float(
                    group["r2_delta_candidate_minus_reference"].mean()
                ),
                "nasa_score_delta_mean": float(
                    group["nasa_score_delta_candidate_minus_reference"].mean()
                ),
                "primary_comparison": comparison in PRIMARY_COMPARISONS,
            }
        )
    output = pd.DataFrame(rows).sort_values(["k", "comparison"]).reset_index(drop=True)
    output["rmse_split_t_p_holm"] = exp17b.holm_adjust(
        output["rmse_split_t_p"].tolist()
    )
    output["strict_success"] = (
        (output["rmse_improvement_pct"] >= 3.0)
        & (output["rmse_target_split_win_rate"] >= 0.8)
        & (output["rmse_hier_boot_ci95_high"] < 0.0)
        & (output["rmse_split_t_p_holm"] < 0.05)
        & (output["nasa_score_delta_mean"] <= 0.0)
    )
    return output


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir)
    prefix = f"experiment18_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired_cell": output / f"{prefix}_paired_by_cell.csv",
        "paired_split": output / f"{prefix}_paired_by_target_split.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_splits": output / f"{prefix}_engine_splits.csv",
        "prior": output / f"{prefix}_prior_adjacency.csv",
        "correlation": output / f"{prefix}_source_sensor_correlation.csv",
        "budget": output / f"{prefix}_budget.json",
        "inventory": output / f"{prefix}_model_inventory.csv",
        "source_diagnostics": output / f"{prefix}_source_diagnostics.json",
        "context_audit": output / f"{prefix}_context_audit.csv",
    }


def save_progress(args, paths, results, context_rows) -> None:
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    atomic_write_text(
        paths["summary"], summarize(results).to_csv(index=False), encoding="utf-8-sig"
    )
    paired = paired_cells(results)
    split = paired_by_target_split(paired)
    comparisons = comparison_summary(results, paired, args.bootstrap_repetitions)
    atomic_write_text(
        paths["paired_cell"], paired.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["paired_split"], split.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["context_audit"],
        pd.DataFrame(context_rows).to_csv(index=False),
        encoding="utf-8-sig",
    )


def completed_keys(results: list[dict]) -> set[tuple[int, int, int, str]]:
    return {
        (
            int(row["target_split_seed"]),
            int(row["model_seed"]),
            int(row["k"]),
            str(row["model"]),
        )
        for row in results
    }


def dry_run(args, cfg, protocol, prior) -> pd.DataFrame:
    split_seed = args.target_split_seeds[0]
    k = min(args.k_values)
    units = protocol["nested_adaptation_units_by_target_split_seed"][str(split_seed)][
        str(k)
    ]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks, support, validation, test, feature_count, split = loaders
    x, _ = next(iter(support))
    static = build_static_model(feature_count, cfg, prior)
    tcsg = build_tcsg_model(feature_count, cfg, prior, args)
    summaries = {
        mode: context_for_mode(
            mode,
            support.dataset.x,
            support.dataset.y,
            tcsg.sensor_count,
            cfg["rul_cap"],
            seed=18018,
        )
        for mode in (
            "tcsg_true",
            "tcsg_sensor_permuted",
            "tcsg_label_shuffled",
            "tcsg_zero",
        )
    }
    rows = []
    with torch.no_grad():
        static_output = static(x)
        for mode, summary in summaries.items():
            output = tcsg(x, summary)
            rows.append(
                {
                    "mode": mode,
                    "feature_count": feature_count,
                    "support_windows": len(support.dataset),
                    "validation_windows": len(validation.dataset),
                    "official_test_rows": len(test.dataset),
                    "input_shape": list(x.shape),
                    "forward_shape": list(output.shape),
                    "summary_dim": len(summary),
                    "summary_norm": float(summary.norm()),
                    "cosine_to_true": float(
                        F.cosine_similarity(
                            summary[None], summaries["tcsg_true"][None]
                        ).item()
                    ),
                    "graph_gate_mean": tcsg.graph_gate_mean(summary),
                    "official_test_hash": split["official_test_units_hash"],
                    "official_test_forward_run": False,
                }
            )
    rows.append(
        {
            "mode": "static_prior",
            "feature_count": feature_count,
            "support_windows": len(support.dataset),
            "validation_windows": len(validation.dataset),
            "official_test_rows": len(test.dataset),
            "input_shape": list(x.shape),
            "forward_shape": list(static_output.shape),
            "summary_dim": 0,
            "summary_norm": 0.0,
            "cosine_to_true": float("nan"),
            "graph_gate_mean": 0.0,
            "official_test_hash": split["official_test_units_hash"],
            "official_test_forward_run": False,
        }
    )
    # Verify that source engine-disjoint episode sampling is possible.
    bank = SourceEpisodeBank(source_tasks, cfg["seed"] + 18018)
    bank.sample(
        args.source_support_engines,
        args.source_query_engines,
        min(32, args.source_support_windows),
        min(32, args.source_query_windows),
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.models = list(dict.fromkeys(args.models))
    args.k_values = sorted(set(int(value) for value in args.k_values))
    args.target_split_seeds = list(
        dict.fromkeys(int(value) for value in args.target_split_seeds)
    )
    args.model_seeds = list(dict.fromkeys(int(value) for value in args.model_seeds))
    if not args.k_values or any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values必须为正整数")
    if args.context_meta_steps <= 0:
        raise ValueError("--context-meta-steps必须大于0")
    if not 0.0 < args.max_graph_gate <= 2.0:
        raise ValueError("--max-graph-gate必须位于(0,2]")
    if args.evaluation_scope == "official_test" and not args.confirm_official_test:
        raise ValueError("官方test被锁定；必须显式提供--confirm-official-test")
    if (
        (len(args.target_split_seeds) < 5 or len(args.model_seeds) < 5)
        and not args.dry_run
    ):
        print("[警告] 少于5×5交叉种子，只能视为预实验。")

    cfg0 = load_config(args, args.model_seeds[0])
    protocol = load_or_create_protocol(args, cfg0)
    expected = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        protocol["official_test_engine_count"] != expected
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试发动机应为{expected}台，"
            f"当前为{protocol['official_test_engine_count']}台"
        )
    prior, correlation = exp17.source_correlation_adjacency(
        cfg0, args.preprocessing, args.sensor_graph_k
    )
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    sensors = list(cfg0["sensor_columns"])
    atomic_write_text(
        paths["protocol"], json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["engine_splits"],
        exp17b.protocol_rows(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["prior"],
        pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv(),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["correlation"],
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
        encoding="utf-8-sig",
    )
    budget = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "models": args.models,
        "k_values": args.k_values,
        "target_split_seeds": args.target_split_seeds,
        "model_seeds": args.model_seeds,
        "planned_target_cells": (
            len(args.models)
            * len(args.k_values)
            * len(args.target_split_seeds)
            * len(args.model_seeds)
        ),
        "source_pretrain_steps": args.source_pretrain_steps,
        "context_meta_steps": args.context_meta_steps,
        "static_budget_extra_steps": args.context_meta_steps,
        "optimizer_step_budget_matched": True,
        "source_task_mode": "engine_disjoint_support_query",
        "source_support_engines": args.source_support_engines,
        "source_query_engines": args.source_query_engines,
        "source_support_windows": args.source_support_windows,
        "source_query_windows": args.source_query_windows,
        "target_adaptation_scope": "predictor_only",
        "official_test_policy": "validation_only unless explicitly confirmed after locking",
        "primary_success_rule": {
            "comparison": "tcsg_true_vs_static_budget",
            "rmse_improvement_pct_at_least": 3.0,
            "target_split_win_rate_at_least": 0.8,
            "hierarchical_bootstrap_ci95_upper_below": 0.0,
            "holm_adjusted_split_level_p_below": 0.05,
            "nasa_score_delta_must_be_leq": 0.0,
        },
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))
    print("\n[实验18锁定协议与预算]")
    print(json.dumps(budget, ensure_ascii=False, indent=2))

    if args.dry_run:
        audit = dry_run(args, cfg0, protocol, prior)
        atomic_write_text(
            paths["context_audit"], audit.to_csv(index=False), encoding="utf-8-sig"
        )
        print("\n[实验18 dry-run审计]")
        print(audit.to_string(index=False))
        print("\n[dry-run完成] 未训练模型，也未前向访问官方test。")
        for key in ("protocol", "engine_splits", "prior", "correlation", "budget", "context_audit"):
            print(f"{key}: {paths[key]}")
        return

    results: list[dict] = []
    context_rows: list[dict] = []
    source_diagnostics: dict = {}
    inventories: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条结果。")
    if args.resume and paths["context_audit"].is_file():
        context_rows = pd.read_csv(paths["context_audit"]).to_dict(orient="records")
    if args.resume and paths["source_diagnostics"].is_file():
        source_diagnostics = json.loads(
            paths["source_diagnostics"].read_text(encoding="utf-8")
        )
    if args.resume and paths["inventory"].is_file():
        inventories = pd.read_csv(paths["inventory"]).to_dict(orient="records")
    done = completed_keys(results)

    for model_seed in args.model_seeds:
        expected_for_seed = {
            (split_seed, model_seed, k, model)
            for split_seed in args.target_split_seeds
            for k in args.k_values
            for model in args.models
        }
        if expected_for_seed.issubset(done):
            print(f"[skip model seed] model_seed={model_seed}已全部完成。")
            continue
        cfg = load_config(args, model_seed)
        print(f"\n[source initialization] model_seed={model_seed}")
        states, histories, inventory = load_or_train_source_bundle(
            args, cfg, protocol, prior
        )
        inventories.append(inventory)
        source_diagnostics[str(model_seed)] = histories
        atomic_write_text(
            paths["source_diagnostics"],
            json.dumps(source_diagnostics, ensure_ascii=False, indent=2),
        )
        atomic_write_text(
            paths["inventory"],
            pd.DataFrame(inventories)
            .drop_duplicates(subset=["model_seed"], keep="last")
            .to_csv(index=False),
            encoding="utf-8-sig",
        )

        for split_seed in args.target_split_seeds:
            for k in args.k_values:
                units = protocol["nested_adaptation_units_by_target_split_seed"][
                    str(split_seed)
                ][str(k)]
                for regime in args.models:
                    key = (split_seed, model_seed, k, regime)
                    if key in done:
                        print(
                            f"[skip] split={split_seed} model_seed={model_seed} "
                            f"K={k} model={regime}"
                        )
                        continue
                    print(
                        f"\n[experiment18] split={split_seed} model_seed={model_seed} "
                        f"K={k} model={regime} engines={units}"
                    )
                    source_key = (
                        regime if regime in {"static_prior", "static_budget_prior"} else "tcsg"
                    )
                    result, context_record = run_target_cell(
                        args,
                        cfg,
                        protocol,
                        regime,
                        states[source_key],
                        inventory,
                        split_seed,
                        k,
                        prior,
                    )
                    results.append(result)
                    context_rows.append(context_record)
                    done.add(key)
                    save_progress(args, paths, results, context_rows)
        states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_progress(args, paths, results, context_rows)
    summary = summarize(results)
    comparisons = comparison_summary(
        results, paired_cells(results), args.bootstrap_repetitions
    )
    print("\n[实验18汇总]")
    print(summary.to_string(index=False))
    print("\n[实验18主要比较]")
    print(comparisons.to_string(index=False))
    print(
        "\n[结论判定]\n"
        "1. tcsg_true_vs_static_budget是元学习独立贡献的主比较。\n"
        "2. true_vs_sensor_permuted检验正确传感器身份是否必要。\n"
        "3. true_vs_label_shuffled检验support中的传感器-RUL对应是否必要。\n"
        "4. true_vs_zero_context检验收益是否真的依赖任务上下文。\n"
        "5. 只有主比较通过strict_success，才能称任务条件化具有独立优势。\n"
        "6. validation通过前不应访问官方test。"
    )
    for key in (
        "raw",
        "summary",
        "paired_cell",
        "paired_split",
        "comparisons",
        "protocol",
        "budget",
        "context_audit",
        "source_diagnostics",
        "inventory",
    ):
        print(f"{key}: {paths[key]}")


if __name__ == "__main__":
    main()
