"""Experiment 17: stable sensor-graph ablation for C-MAPSS RUL prediction.

This is an additional, single-file experiment entry point.  It does not replace
``main.py`` or any existing model.  It tests the first structural hypothesis of
TCSG-ML before adding task conditioning or meta-learning:

    Are stable sensor nodes more useful than a graph whose nodes are the
    windows in the current random mini-batch?

The default comparison contains five source-pretrained models:

``no_graph``
    Fourteen stable sensor nodes, but only self connections.  It is the
    parameter-matched no-message-passing control.
``window_graph``
    The original project model: one node per window and a cosine kNN graph
    built inside each mini-batch.
``sensor_graph_static``
    Fourteen stable sensor nodes and one fixed fully-connected graph.
``sensor_graph_prior``
    Fourteen stable sensor nodes and a fixed source-only correlation kNN graph.
``sensor_graph_learned``
    Fourteen stable sensor nodes and learnable dense edge biases, learned only
    during source pretraining.

All regimes use the same source-domain supervised pretraining budget and only
adapt ``predictor.*`` on the same K target engines.  Thus Experiment 17 tests
graph representation, not a meta-learning optimizer.  The official C-MAPSS
test set is not evaluated by default; model selection and all default outputs
are validation-only.

Run from the project root.

Dry run::

    python scripts/experiment17_sensor_graph_ablation.py --target FD004 --dry-run

Formal validation run::

    python -u scripts/experiment17_sensor_graph_ablation.py \
      --target FD004 \
      --k-values 2 5 10 20 \
      --seeds 42 43 44 45 46 \
      --models no_graph window_graph sensor_graph_static \
               sensor_graph_prior sensor_graph_learned \
      --preprocessing condition_settings \
      --balance-mode engine_stage \
      --source-pretrain-steps 1500 \
      --target-epochs 10 \
      --evaluation-scope validation \
      --resume

Only after the complete validation protocol is locked may an official-test
confirmation be run by adding both ``--evaluation-scope official_test`` and
``--confirm-official-test``.
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

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from evaluation.metrics import regression_metrics  # noqa: E402
from preprocess.cmapps_loader import load_domain  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    prepare_kshot_experiment,
    protocol_split_frame,
    resolve_device,
    resolve_path,
    seed_everything,
    target_unit_protocol,
)
from scripts.experiment8_transfer_baseline import (  # noqa: E402
    train_source_supervised,
)
from scripts.run_condition_aware_experiment import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
    SourceConditionNormalizer,
    SourceGlobalNormalizer,
)


SCRIPT_VERSION = "experiment17_sensor_graph_ablation_v1"
MODEL_CHOICES = (
    "no_graph",
    "window_graph",
    "sensor_graph_static",
    "sensor_graph_prior",
    "sensor_graph_learned",
)
LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}
METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验17：稳定传感器图与原随机batch窗口图的结构消融"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target",
        choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES),
        default="FD004",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES)
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]
    )
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument(
        "--protocol-file",
        help="可选：实验7固定划分JSON；未指定时优先读取默认实验7协议",
    )
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument(
        "--balance-mode", choices=BALANCE_MODES, default="engine_stage"
    )
    parser.add_argument(
        "--sensor-graph-k",
        type=int,
        default=4,
        help="source-only相关性先验图中每个传感器保留的邻居数",
    )
    parser.add_argument("--source-pretrain-steps", type=int, default=1500)
    parser.add_argument("--source-pretrain-lr", type=float, default=0.001)
    parser.add_argument("--source-pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument(
        "--evaluation-scope",
        choices=("validation", "official_test"),
        default="validation",
    )
    parser.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="正式锁定后才能显式确认官方test前向；默认禁止",
    )
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=5000
    )
    parser.add_argument(
        "--output-dir", default="outputs/experiment17_sensor_graph_ablation"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-target-checkpoints",
        action="store_true",
        help="默认只保存源状态和表格；启用后保存每个K的目标模型",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-official-count-check", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace, seed: int) -> dict:
    config_path = resolve_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
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
    # Pairwise auxiliary loss exists only in the original window model.  It is
    # deliberately disabled so Experiment 17 changes graph structure alone.
    cfg["pair_aux_weight"] = 0.0
    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir))
    cfg["output_dir"] = str(resolve_path(args.output_dir))
    return cfg


class SharedSensorTemporalEncoder(nn.Module):
    """Encode each sensor's length-T trajectory with shared 1-D convolutions."""

    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        middle = max(32, embedding_dim // 2)
        self.network = nn.Sequential(
            nn.Conv1d(1, middle, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(middle, embedding_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        batch, time, nodes = sensors.shape
        trajectories = sensors.transpose(1, 2).reshape(batch * nodes, 1, time)
        encoded = self.network(trajectories)
        return encoded.reshape(batch, nodes, -1)


class DenseSensorGraphLayer(nn.Module):
    """Multi-head attention restricted by a stable sensor adjacency matrix."""

    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        dropout: float,
        adjacency: torch.Tensor,
        learnable_edge_bias: bool,
    ):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim必须能被gat_heads整除")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
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
        if learnable_edge_bias:
            self.edge_bias = nn.Parameter(
                torch.zeros(heads, adjacency.shape[0], adjacency.shape[1])
            )
        else:
            self.register_parameter("edge_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, nodes, 3, self.heads, self.head_dim
        )
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        if self.edge_bias is not None:
            logits = logits + self.edge_bias.unsqueeze(0)
        allowed = self.adjacency.view(1, 1, nodes, nodes)
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        mixed = torch.einsum("bhij,bhjd->bhid", attention, v)
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        x = self.norm1(x + self.dropout(self.output(mixed)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class StableSensorGraphRegressor(nn.Module):
    """Stable-node sensor graph used by four matched Experiment-17 regimes."""

    def __init__(
        self,
        sensor_count: int,
        condition_dim: int,
        embedding_dim: int,
        heads: int,
        dropout: float,
        adjacency: torch.Tensor,
        learnable_edge_bias: bool = False,
    ):
        super().__init__()
        if adjacency.shape != (sensor_count, sensor_count):
            raise ValueError("传感器邻接矩阵形状与sensor_count不一致")
        if not bool(torch.diag(adjacency).all()):
            raise ValueError("传感器邻接矩阵必须包含自环")
        self.sensor_count = int(sensor_count)
        self.condition_dim = int(condition_dim)
        self.temporal = SharedSensorTemporalEncoder(embedding_dim, dropout)
        self.graph_layers = nn.ModuleList(
            [
                DenseSensorGraphLayer(
                    embedding_dim,
                    heads,
                    dropout,
                    adjacency,
                    learnable_edge_bias,
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

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if x.ndim != 3 or x.shape[-1] < self.sensor_count:
            raise ValueError("输入必须为[batch, time, features]")
        sensor_window = x[:, :, : self.sensor_count]
        nodes = self.temporal(sensor_window)
        for layer in self.graph_layers:
            nodes = layer(nodes)
        pooling = F.softmax(self.pool_score(nodes).squeeze(-1), dim=-1)
        graph_feature = torch.sum(nodes * pooling.unsqueeze(-1), dim=1)
        if self.condition_encoder is not None:
            conditions = x[:, :, self.sensor_count :].mean(dim=1)
            graph_feature = torch.cat(
                [graph_feature, self.condition_encoder(conditions)], dim=-1
            )
        prediction = self.predictor(graph_feature).squeeze(-1)
        if return_attention:
            return prediction, {
                "features": graph_feature,
                "sensor_pooling": pooling,
            }
        return prediction

    def learned_edge_strength(self) -> torch.Tensor | None:
        biases = [
            layer.edge_bias.detach().mean(dim=0)
            for layer in self.graph_layers
            if layer.edge_bias is not None
        ]
        if not biases:
            return None
        return torch.stack(biases).mean(dim=0)


def identity_adjacency(sensor_count: int) -> torch.Tensor:
    return torch.eye(sensor_count, dtype=torch.bool)


def full_adjacency(sensor_count: int) -> torch.Tensor:
    return torch.ones(sensor_count, sensor_count, dtype=torch.bool)


def fit_source_normalizer(cfg: dict, preprocessing: str):
    source_frames = [
        load_domain(cfg["data_dir"], domain)[0]
        for domain in cfg["source_domains"]
    ]
    source = pd.concat(source_frames, ignore_index=True)
    sensors = list(cfg["sensor_columns"])
    include_settings = preprocessing in {"global_settings", "condition_settings"}
    if preprocessing in {"condition_norm", "condition_settings"}:
        normalizer = SourceConditionNormalizer(
            n_conditions=cfg.get("condition_count", 6),
            seed=cfg.get("normalizer_seed", 2026),
            include_settings=include_settings,
        ).fit(source, sensors)
    else:
        normalizer = SourceGlobalNormalizer(
            include_settings=include_settings
        ).fit(source, sensors)
    return source_frames, normalizer


def source_correlation_adjacency(
    cfg: dict,
    preprocessing: str,
    neighbors: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Build a fixed prior graph from source-train sensors only."""
    sensors = list(cfg["sensor_columns"])
    sensor_count = len(sensors)
    if not 1 <= neighbors < sensor_count:
        raise ValueError(
            f"--sensor-graph-k必须位于[1,{sensor_count - 1}]，当前为{neighbors}"
        )
    source_frames, normalizer = fit_source_normalizer(cfg, preprocessing)
    normalized = [normalizer.transform(frame, sensors) for frame in source_frames]
    values = pd.concat(normalized, ignore_index=True)[sensors].to_numpy(np.float64)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    adjacency = np.eye(sensor_count, dtype=bool)
    for sensor in range(sensor_count):
        scores = correlation[sensor].copy()
        scores[sensor] = -np.inf
        selected = np.argsort(scores)[-neighbors:]
        adjacency[sensor, selected] = True
    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, True)
    return torch.as_tensor(adjacency), correlation


def build_ablation_model(
    model_name: str,
    feature_count: int,
    cfg: dict,
    prior_adjacency: torch.Tensor,
) -> nn.Module:
    if model_name == "window_graph":
        return build_model("gnn", feature_count, cfg)

    sensor_count = len(cfg["sensor_columns"])
    condition_dim = feature_count - sensor_count
    if condition_dim < 0:
        raise ValueError("feature_count小于固定传感器数量")
    if model_name == "no_graph":
        adjacency = identity_adjacency(sensor_count)
        learned = False
    elif model_name == "sensor_graph_static":
        adjacency = full_adjacency(sensor_count)
        learned = False
    elif model_name == "sensor_graph_prior":
        adjacency = prior_adjacency
        learned = False
    elif model_name == "sensor_graph_learned":
        adjacency = full_adjacency(sensor_count)
        learned = True
    else:
        raise ValueError(f"未知实验17模型：{model_name}")
    return StableSensorGraphRegressor(
        sensor_count=sensor_count,
        condition_dim=condition_dim,
        embedding_dim=int(cfg["embedding_dim"]),
        heads=int(cfg["gat_heads"]),
        dropout=float(cfg["dropout"]),
        adjacency=adjacency,
        learnable_edge_bias=learned,
    )


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    predictor = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("predictor.")
    )
    return int(total), int(predictor)


def predict(model: nn.Module, loader, device: torch.device):
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            labels.extend(y.numpy().tolist())
            predictions.extend(output.detach().cpu().numpy().tolist())
    return np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)


def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    labels, predictions = predict(model, loader, device)
    return regression_metrics(labels, predictions)


def train_target_head(
    model: nn.Module,
    support,
    validation,
    cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, list[dict], int]:
    learner = deepcopy(model).to(device)
    for parameter in learner.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in learner.named_parameters():
        if name.startswith("predictor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("模型没有predictor.*参数，无法进行公平的预测头适应")

    optimizer = torch.optim.Adam(trainable, lr=cfg["target_lr"])
    best_state = deepcopy(learner.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, cfg["target_epochs"] + 1):
        learner.train()
        losses: list[float] = []
        for x, y in support:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = learner(x)
            loss = F.mse_loss(prediction, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("目标预测头训练出现NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(learner, validation, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"target_epoch={epoch:03d}/{cfg['target_epochs']} "
            f"train_loss={row['train_loss']:.4f} val_rmse={metrics['rmse']:.4f}"
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            best_state = deepcopy(learner.state_dict())
    learner.load_state_dict(best_state)
    return learner, history, best_epoch


def state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def signature_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:20]


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def source_signature(
    args: argparse.Namespace,
    cfg: dict,
    model_name: str,
    feature_count: int,
    prior_adjacency: torch.Tensor,
) -> str:
    return signature_hash(
        {
            "script_version": SCRIPT_VERSION,
            "model": model_name,
            "seed": cfg["seed"],
            "target": cfg["target_domain"],
            "sources": cfg["source_domains"],
            "feature_count": feature_count,
            "embedding_dim": cfg["embedding_dim"],
            "gat_heads": cfg["gat_heads"],
            "dropout": cfg["dropout"],
            "preprocessing": args.preprocessing,
            "balance_mode": args.balance_mode,
            "source_steps": cfg["source_pretrain_steps"],
            "source_lr": cfg["source_pretrain_lr"],
            "source_weight_decay": cfg["source_pretrain_weight_decay"],
            "prior_hash": hashlib.sha256(
                prior_adjacency.numpy().tobytes()
            ).hexdigest()[:16],
        }
    )


def load_or_train_source_state(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    model_name: str,
    prior_adjacency: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    first_k = min(int(value) for value in args.k_values)
    first_units = protocol["nested_adaptation_units_by_seed"][str(cfg["seed"])][
        str(first_k)
    ]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        first_units,
    )
    source_tasks, _, _, _, feature_count, _ = loaders
    model = build_ablation_model(
        model_name, feature_count, cfg, prior_adjacency
    )
    total, predictor = parameter_count(model)
    signature = source_signature(
        args, cfg, model_name, feature_count, prior_adjacency
    )
    cache_dir = Path(cfg["output_dir"]) / "source_cache"
    cache_path = cache_dir / (
        f"experiment17_{model_name}_{cfg['target_domain']}_seed{cfg['seed']}.pt"
    )
    if args.resume and cache_path.is_file():
        cached = safe_torch_load(cache_path)
        if cached.get("signature") == signature:
            print(f"[source cache] {cache_path}")
            return cached["state"], cached.get("history", []), cached["inventory"]
        print(f"[source cache ignored] 签名不匹配：{cache_path}")

    seed_everything(cfg["seed"])
    model = build_ablation_model(
        model_name, feature_count, cfg, prior_adjacency
    )
    device = resolve_device(cfg["device"])
    model, history = train_source_supervised(model, source_tasks, cfg, device)
    state = state_to_cpu(model)
    inventory = {
        "model": model_name,
        "seed": cfg["seed"],
        "feature_count": feature_count,
        "total_parameter_count": total,
        "predictor_parameter_count": predictor,
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "source_pretrain_lr": cfg["source_pretrain_lr"],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "state": state,
            "history": history,
            "inventory": inventory,
        },
        cache_path,
    )
    del model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, history, inventory


def run_target_cell(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    model_name: str,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    k: int,
    prior_adjacency: torch.Tensor,
) -> dict:
    units = protocol["nested_adaptation_units_by_seed"][str(cfg["seed"])][str(k)]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    _, support, validation, test, feature_count, split = loaders
    if split["official_test_units_hash"] != protocol["official_test_units_hash"]:
        raise AssertionError("官方测试发动机集合发生变化")
    seed_everything(cfg["seed"])
    model = build_ablation_model(
        model_name, feature_count, cfg, prior_adjacency
    )
    model.load_state_dict(source_state)
    device = resolve_device(cfg["device"])
    model, history, best_epoch = train_target_head(
        model, support, validation, cfg, device
    )
    validation_metrics = evaluate(model, validation, device)
    if args.evaluation_scope == "official_test":
        selected_metrics = evaluate(model, test, device)
        official_metrics = dict(selected_metrics)
    else:
        selected_metrics = dict(validation_metrics)
        official_metrics = None
    result = {
        **selected_metrics,
        "evaluation_scope": args.evaluation_scope,
        "model": model_name,
        "graph_node_semantics": (
            "batch_windows" if model_name == "window_graph" else "fixed_sensors"
        ),
        "source_training": "ordinary_multisource_pretraining",
        "target_adaptation_scope": "predictor_only",
        "experiment": f"experiment17_{model_name}_k{k}",
        "target_domain": cfg["target_domain"],
        "seed": cfg["seed"],
        "k": int(k),
        "adaptation_units": [int(unit) for unit in units],
        "adaptation_engine_count": len(units),
        "validation_engine_count": len(protocol["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split["official_test_units_hash"],
        "official_test_metrics": official_metrics,
        "official_test_forward_run": args.evaluation_scope == "official_test",
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": cfg["target_epochs"],
        "target_learning_rate": cfg["target_lr"],
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "preprocessing_mode": args.preprocessing,
        "balance_mode": args.balance_mode,
        "total_parameter_count": inventory["total_parameter_count"],
        "target_trainable_parameter_count": inventory["predictor_parameter_count"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "source_history_rows": len(source_history),
    }
    if args.save_target_checkpoints:
        checkpoint_dir = Path(cfg["output_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / (
            f"experiment17_{model_name}_k{k}_{cfg['target_domain']}_"
            f"seed{cfg['seed']}.pt"
        )
        torch.save(
            {
                "model": state_to_cpu(model),
                "metrics": result,
                "target_history": history,
                "split": split,
            },
            checkpoint_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def bootstrap_ci(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_t_pvalue(values: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_1samp

        return float(ttest_1samp(values, popmean=0.0).pvalue)
    except Exception:
        return float("nan")


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    for (k, model), group in frame.groupby(["k", "model"]):
        row = {
            "k": int(k),
            "model": model,
            "n_runs": int(len(group)),
            "evaluation_scope": group["evaluation_scope"].iloc[0],
            "graph_node_semantics": group["graph_node_semantics"].iloc[0],
            "total_parameter_count": int(group["total_parameter_count"].iloc[0]),
            "target_trainable_parameter_count": int(
                group["target_trainable_parameter_count"].iloc[0]
            ),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "rmse_mean"]).reset_index(drop=True)


def paired_comparisons(
    results: list[dict], repetitions: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(results)
    references = [name for name in ("no_graph", "window_graph") if name in set(frame["model"])]
    paired_rows: list[dict] = []
    for (k, seed), group in frame.groupby(["k", "seed"]):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        for reference in references:
            if reference not in by_model:
                continue
            for candidate, candidate_row in by_model.items():
                if candidate == reference:
                    continue
                reference_row = by_model[reference]
                row = {
                    "k": int(k),
                    "seed": int(seed),
                    "candidate": candidate,
                    "reference": reference,
                }
                for metric in METRICS:
                    delta = float(candidate_row[metric] - reference_row[metric])
                    row[f"{metric}_delta_candidate_minus_reference"] = delta
                    if metric in LOWER_IS_BETTER:
                        row[f"{metric}_candidate_win"] = float(delta < 0)
                    else:
                        row[f"{metric}_candidate_win"] = float(delta > 0)
                paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    if paired.empty:
        return paired, pd.DataFrame()

    comparison_rows: list[dict] = []
    for (k, candidate, reference), group in paired.groupby(
        ["k", "candidate", "reference"]
    ):
        rmse_delta = group["rmse_delta_candidate_minus_reference"].to_numpy(float)
        nasa_delta = group["nasa_score_delta_candidate_minus_reference"].to_numpy(float)
        low, high = bootstrap_ci(
            rmse_delta,
            repetitions,
            seed=17000 + int(k) + sum(map(ord, candidate + reference)),
        )
        reference_rmse = frame[
            (frame["k"] == k) & (frame["model"] == reference)
        ]["rmse"].mean()
        comparison_rows.append(
            {
                "k": int(k),
                "candidate": candidate,
                "reference": reference,
                "paired_seed_count": int(len(group)),
                "rmse_delta_mean": float(rmse_delta.mean()),
                "rmse_improvement_pct": float(-100.0 * rmse_delta.mean() / reference_rmse),
                "rmse_win_rate": float((rmse_delta < 0).mean()),
                "rmse_bootstrap_ci95_low": low,
                "rmse_bootstrap_ci95_high": high,
                "rmse_paired_t_p": paired_t_pvalue(rmse_delta),
                "mae_delta_mean": float(
                    group["mae_delta_candidate_minus_reference"].mean()
                ),
                "r2_delta_mean": float(
                    group["r2_delta_candidate_minus_reference"].mean()
                ),
                "nasa_score_delta_mean": float(nasa_delta.mean()),
                "nasa_score_win_rate": float((nasa_delta < 0).mean()),
                "candidate_success_3pct": bool(
                    (-100.0 * rmse_delta.mean() / reference_rmse) >= 3.0
                    and (rmse_delta < 0).mean() >= 0.8
                    and nasa_delta.mean() <= 0.0
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["k", "reference", "rmse_delta_mean"]
    )
    return paired.sort_values(["k", "reference", "candidate", "seed"]), comparisons


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir)
    prefix = f"experiment17_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired": output / f"{prefix}_paired_by_seed.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_split_protocol.json",
        "splits": output / f"{prefix}_engine_splits.csv",
        "prior": output / f"{prefix}_source_sensor_correlation.csv",
        "adjacency": output / f"{prefix}_source_sensor_adjacency.csv",
        "inventory": output / f"{prefix}_model_inventory.csv",
        "audit": output / f"{prefix}_graph_audit.json",
    }


def save_progress(
    args: argparse.Namespace,
    paths: dict[str, Path],
    results: list[dict],
) -> None:
    atomic_write_text(paths["raw"], json.dumps(results, ensure_ascii=False, indent=2))
    atomic_write_text(
        paths["summary"], summarize(results).to_csv(index=False), encoding="utf-8-sig"
    )
    paired, comparisons = paired_comparisons(
        results, args.bootstrap_repetitions
    )
    atomic_write_text(
        paths["paired"], paired.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"],
        comparisons.to_csv(index=False),
        encoding="utf-8-sig",
    )


def completed_keys(results: list[dict]) -> set[tuple[int, int, str]]:
    return {
        (int(row["seed"]), int(row["k"]), str(row["model"]))
        for row in results
    }


def validate_protocol(
    protocol: dict, args: argparse.Namespace, seeds: list[int], k_values: list[int]
) -> None:
    if protocol.get("target_domain") != args.target:
        raise ValueError("协议target_domain与--target不一致")
    validation = set(int(unit) for unit in protocol["validation_units"])
    nested = protocol["nested_adaptation_units_by_seed"]
    for seed in seeds:
        if str(seed) not in nested:
            raise ValueError(f"协议缺少seed={seed}")
        previous: set[int] = set()
        for k in k_values:
            if str(k) not in nested[str(seed)]:
                raise ValueError(f"协议缺少seed={seed}, K={k}")
            current = set(int(unit) for unit in nested[str(seed)][str(k)])
            if len(current) != k:
                raise ValueError(f"seed={seed}, K={k}发动机数量错误")
            if not previous.issubset(current):
                raise ValueError(f"seed={seed}的K集合没有严格嵌套")
            if current & validation:
                raise ValueError("适应发动机与验证发动机重叠")
            previous = current


def load_protocol(
    args: argparse.Namespace, cfg: dict, seeds: list[int], k_values: list[int]
) -> tuple[dict, str | None]:
    candidates: list[Path] = []
    if args.protocol_file:
        candidates.append(resolve_path(args.protocol_file))
    else:
        candidates.append(
            PROJECT_ROOT
            / "outputs"
            / "experiment7_kshot_engines"
            / f"experiment7_split_protocol_{args.target}.json"
        )
    existing = next((path for path in candidates if path.is_file()), None)
    if existing is not None:
        protocol = json.loads(existing.read_text(encoding="utf-8"))
        validate_protocol(protocol, args, seeds, k_values)
        print(f"[protocol] 读取固定划分：{existing}")
        return protocol, str(existing)
    protocol = target_unit_protocol(
        cfg["data_dir"],
        args.target,
        args.validation_units,
        args.validation_seed,
        seeds,
        k_values,
    )
    validate_protocol(protocol, args, seeds, k_values)
    print("[protocol] 未找到实验7协议，按相同发动机级规则新建。")
    return protocol, None


def graph_audit(
    cfg: dict,
    prior: torch.Tensor,
    models: list[str],
    source_protocol_path: str | None,
) -> dict:
    sensor_count = len(cfg["sensor_columns"])
    undirected_prior_edges = int(
        (prior.to(torch.int64).sum().item() - sensor_count) // 2
    )
    return {
        "script_version": SCRIPT_VERSION,
        "hypothesis": "stable sensor nodes outperform random-batch window nodes",
        "is_meta_learning_experiment": False,
        "source_protocol_path": source_protocol_path,
        "sensor_nodes": list(cfg["sensor_columns"]),
        "sensor_node_count": sensor_count,
        "prior_undirected_edge_count_excluding_self_loops": undirected_prior_edges,
        "models": models,
        "controls": {
            "same_source_pretraining_budget": True,
            "same_target_engines": True,
            "same_target_batches": True,
            "target_adaptation_scope": "predictor_only",
            "pairwise_auxiliary_loss": 0.0,
            "official_test_default": "disabled",
        },
    }


def dry_run_models(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    models: list[str],
    prior: torch.Tensor,
) -> pd.DataFrame:
    seed = cfg["seed"]
    k = min(int(value) for value in args.k_values)
    units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
    loaders = prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks, support, validation, test, feature_count, _ = loaders
    x, _ = next(iter(source_tasks[cfg["source_domains"][0]]))
    rows: list[dict] = []
    for model_name in models:
        seed_everything(seed)
        model = build_ablation_model(model_name, feature_count, cfg, prior).eval()
        with torch.no_grad():
            prediction = model(x[: min(8, len(x))])
        total, predictor = parameter_count(model)
        rows.append(
            {
                "model": model_name,
                "graph_node_semantics": (
                    "batch_windows"
                    if model_name == "window_graph"
                    else "fixed_sensors"
                ),
                "feature_count": feature_count,
                "total_parameter_count": total,
                "predictor_parameter_count": predictor,
                "forward_output_shape": list(prediction.shape),
                "finite_output": bool(torch.isfinite(prediction).all()),
            }
        )
    print(
        json.dumps(
            {
                "seed": seed,
                "k": k,
                "source_batch_shape": list(x.shape),
                "support_windows": len(support.dataset),
                "support_engines": len(set(support.dataset.units)),
                "validation_windows": len(validation.dataset),
                "validation_engines": len(set(validation.dataset.units)),
                "official_test_engine_count_from_protocol": len(test.dataset),
                "official_test_forward_run": False,
                "models": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.k_values = sorted(set(int(value) for value in args.k_values))
    args.seeds = list(dict.fromkeys(int(value) for value in args.seeds))
    args.models = list(dict.fromkeys(args.models))
    if not args.k_values or any(value <= 0 for value in args.k_values):
        raise ValueError("--k-values必须为正整数")
    if not args.seeds:
        raise ValueError("--seeds不能为空")
    if len(args.seeds) < 5 and not args.dry_run:
        print("[警告] 少于5个随机种子，只能视为预实验。")
    if args.evaluation_scope == "official_test" and not args.confirm_official_test:
        raise ValueError(
            "官方test默认锁定；必须同时提供--evaluation-scope official_test "
            "和--confirm-official-test。"
        )

    first_cfg = load_config(args, args.seeds[0])
    protocol, source_protocol_path = load_protocol(
        args, first_cfg, args.seeds, args.k_values
    )
    expected_test_count = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        int(protocol["official_test_engine_count"]) != expected_test_count
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试发动机应为{expected_test_count}台，"
            f"协议中为{protocol['official_test_engine_count']}台。"
        )

    prior, correlation = source_correlation_adjacency(
        first_cfg, args.preprocessing, args.sensor_graph_k
    )
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    protocol_copy = dict(protocol)
    protocol_copy["experiment17_source_protocol_path"] = source_protocol_path
    atomic_write_text(
        paths["protocol"], json.dumps(protocol_copy, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["splits"],
        protocol_split_frame(protocol).to_csv(index=False),
        encoding="utf-8-sig",
    )
    sensor_names = list(first_cfg["sensor_columns"])
    atomic_write_text(
        paths["prior"],
        pd.DataFrame(correlation, index=sensor_names, columns=sensor_names).to_csv(),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["adjacency"],
        pd.DataFrame(
            prior.numpy().astype(int), index=sensor_names, columns=sensor_names
        ).to_csv(),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["audit"],
        json.dumps(
            graph_audit(first_cfg, prior, args.models, source_protocol_path),
            ensure_ascii=False,
            indent=2,
        ),
    )

    print("\n[实验17固定协议]")
    print(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "target": args.target,
                "source_domains": first_cfg["source_domains"],
                "models": args.models,
                "k_values": args.k_values,
                "seeds": args.seeds,
                "fixed_validation_units": protocol["validation_units"],
                "preprocessing": args.preprocessing,
                "balance_mode": args.balance_mode,
                "source_pretrain_steps_equal": args.source_pretrain_steps,
                "target_epochs_equal": args.target_epochs,
                "target_adaptation_scope": "predictor_only",
                "pair_aux_weight": 0.0,
                "evaluation_scope": args.evaluation_scope,
                "official_test_forward_will_run": (
                    args.evaluation_scope == "official_test"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run:
        inventory = dry_run_models(
            args, first_cfg, protocol, args.models, prior
        )
        atomic_write_text(
            paths["inventory"], inventory.to_csv(index=False), encoding="utf-8-sig"
        )
        print("\n[dry-run完成] 模型前向与发动机协议检查通过，未训练模型。")
        for key in ("protocol", "splits", "prior", "adjacency", "inventory", "audit"):
            print(f"{key}: {paths[key]}")
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条完成结果。")
    done = completed_keys(results)
    inventories: list[dict] = []

    for seed in args.seeds:
        cfg = load_config(args, seed)
        states: dict[str, dict[str, torch.Tensor]] = {}
        histories: dict[str, list[dict]] = {}
        seed_inventory: dict[str, dict] = {}
        for model_name in args.models:
            pending = [
                k
                for k in args.k_values
                if (seed, k, model_name) not in done
            ]
            if not pending:
                continue
            print(f"\n[source initialization] seed={seed} model={model_name}")
            state, history, inventory = load_or_train_source_state(
                args, cfg, protocol, model_name, prior
            )
            states[model_name] = state
            histories[model_name] = history
            seed_inventory[model_name] = inventory
            inventories.append(inventory)

        for k in args.k_values:
            units = protocol["nested_adaptation_units_by_seed"][str(seed)][str(k)]
            for model_name in args.models:
                key = (seed, k, model_name)
                if key in done:
                    print(f"[skip] seed={seed} K={k} model={model_name}")
                    continue
                print(
                    f"\n[experiment17] seed={seed} K={k} model={model_name} "
                    f"engines={units}"
                )
                result = run_target_cell(
                    args,
                    cfg,
                    protocol,
                    model_name,
                    states[model_name],
                    histories[model_name],
                    seed_inventory[model_name],
                    k,
                    prior,
                )
                results.append(result)
                done.add(key)
                save_progress(args, paths, results)

        states.clear()
        histories.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    inventory_frame = pd.DataFrame(inventories).drop_duplicates(
        subset=["model", "seed"]
    )
    atomic_write_text(
        paths["inventory"], inventory_frame.to_csv(index=False), encoding="utf-8-sig"
    )
    save_progress(args, paths, results)
    summary = summarize(results)
    _, comparisons = paired_comparisons(results, args.bootstrap_repetitions)
    print("\n[实验17汇总]")
    print(summary.to_string(index=False))
    print("\n[实验17配对比较]")
    print(comparisons.to_string(index=False))
    print(
        "\n[判定规则]\n"
        "1. 先比较sensor_graph_*与no_graph，验证跨传感器消息传递是否有价值。\n"
        "2. 再比较sensor_graph_*与window_graph，验证稳定传感器节点是否优于随机batch窗口节点。\n"
        "3. 主要看K=2和K=5；RMSE改善建议≥3%，种子胜率≥80%，且NASA Score不恶化。\n"
        "4. 本实验没有证明元学习，只验证TCSG-ML的传感器图结构假设。\n"
        "5. validation门槛未通过时，不运行官方test，也不继续增加任务条件图。"
    )
    for key in ("raw", "summary", "paired", "comparisons", "protocol", "inventory", "audit"):
        print(f"{key}: {paths[key]}")


if __name__ == "__main__":
    main()
