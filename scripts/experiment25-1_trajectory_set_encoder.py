"""Experiment 25-1: condition-routed trajectory Set Encoder pilot.

Experiment 25 showed that the hand-built support summary is noisier within an
operating condition than between conditions. This pilot replaces that summary
with a small DeepSets encoder over labelled support trajectories.

The ordinary-pretrained sensor graph is reused. During source episodic
training the temporal backbone and predictor are frozen; only the Set Encoder
and low-rank graph adapter are updated. Target adaptation updates the predictor
only. Correct condition routing is compared with global and wrong-condition
contexts. Training and evaluation use C-MAPSS train files only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import regression_metrics  # noqa: E402
from scripts import experiment18_task_conditioned_sensor_graph as exp18  # noqa: E402
from scripts import experiment24_truncated_endpoint_validation as exp24  # noqa: E402
from scripts import experiment25_task_identifiability_audit as exp25  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    resolve_device,
    seed_everything,
)


SCRIPT_VERSION = "experiment25-1_trajectory_set_encoder_v1"
PREPROCESSING = "condition_settings"
MODES = ("set_condition_true", "set_global", "set_condition_wrong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 25-1: condition-routed trajectory Set Encoder pilot"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--target", choices=tuple(EXPECTED_OFFICIAL_TEST_ENGINES), default="FD004"
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5])
    parser.add_argument(
        "--target-split-seeds", nargs="+", type=int, default=[3027, 3028]
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--protocol")
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/experiment18_task_conditioned_sensor_graph/source_cache",
    )
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--set-token-dim", type=int, default=32)
    parser.add_argument("--graph-residual-rank", type=int, default=4)
    parser.add_argument("--max-graph-gate", type=float, default=0.5)
    parser.add_argument("--gate-scale", type=float, default=2.0)
    parser.add_argument("--meta-steps", type=int, default=600)
    parser.add_argument("--meta-lr", type=float, default=0.0001)
    parser.add_argument("--meta-weight-decay", type=float, default=0.0)
    parser.add_argument("--source-support-engines", type=int, default=5)
    parser.add_argument("--source-query-engines", type=int, default=5)
    parser.add_argument("--source-support-windows", type=int, default=128)
    parser.add_argument("--source-query-windows", type=int, default=128)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--min-condition-windows", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="outputs/experiment25-1_trajectory_set_encoder"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(exp25.project_path(args.config).read_text(encoding="utf-8"))
    cfg["seed"] = int(args.model_seed)
    cfg["target_domain"] = args.target
    cfg["source_domains"] = [
        domain for domain in EXPECTED_OFFICIAL_TEST_ENGINES if domain != args.target
    ]
    cfg["normalizer_seed"] = int(args.normalizer_seed)
    cfg["condition_count"] = int(args.condition_count)
    cfg["target_epochs"] = int(args.target_epochs)
    cfg["target_lr"] = float(args.target_lr)
    cfg["data_dir"] = str(
        exp25.project_path(
            args.data_dir if args.data_dir is not None else cfg["data_dir"]
        )
    )
    return cfg


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    positive = (
        args.meta_steps,
        args.meta_lr,
        args.source_support_engines,
        args.source_query_engines,
        args.source_support_windows,
        args.source_query_windows,
        args.target_epochs,
        args.target_lr,
        args.min_condition_windows,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Training counts and learning rates must be positive")
    if not 1 <= args.sensor_graph_k < len(cfg["sensor_columns"]):
        raise ValueError("--sensor-graph-k is outside the sensor count")
    if args.context_dim <= 1 or args.set_token_dim <= 1 or args.gate_scale <= 0:
        raise ValueError("Context dimensions and gate scale must be positive")


class TrajectorySetEncoder(nn.Module):
    """Permutation-invariant labelled support encoder preserving sensor identity."""

    def __init__(
        self,
        sensor_count: int,
        condition_dim: int,
        embedding_dim: int,
        token_dim: int,
        context_dim: int,
        rul_cap: float,
    ):
        super().__init__()
        self.sensor_count = int(sensor_count)
        self.condition_dim = int(condition_dim)
        self.rul_cap = float(rul_cap)
        self.token = nn.Sequential(
            nn.Linear(embedding_dim + 1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        output_dim = sensor_count * token_dim + condition_dim
        self.output = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def forward(
        self,
        sensor_nodes: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        labels = (support_y / self.rul_cap).view(-1, 1, 1)
        labels = labels.expand(-1, self.sensor_count, 1)
        tokens = self.token(torch.cat([sensor_nodes, labels], dim=-1))
        sensor_set = tokens.mean(dim=0).reshape(-1)
        if self.condition_dim:
            settings = support_x[:, :, self.sensor_count :].mean(dim=(0, 1))
            sensor_set = torch.cat([sensor_set, settings], dim=0)
        return self.output(sensor_set).unsqueeze(0)


class TrajectorySetGraphRegressor(nn.Module):
    def __init__(
        self,
        base: exp18.TaskConditionedSensorGraphRegressor,
        token_dim: int,
        context_dim: int,
        rul_cap: float,
    ):
        super().__init__()
        self.base = base
        embedding_dim = base.pool_score.in_features
        self.set_encoder = TrajectorySetEncoder(
            base.sensor_count,
            base.condition_dim,
            embedding_dim,
            token_dim,
            context_dim,
            rul_cap,
        )

    def encode_support(
        self, support_x: torch.Tensor, support_y: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            nodes = self.base.temporal(
                support_x[:, :, : self.base.sensor_count]
            )
        return self.set_encoder(nodes, support_x, support_y)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        nodes = self.base.temporal(x[:, :, : self.base.sensor_count])
        for layer in self.base.graph_layers:
            nodes = layer(nodes, context)
        pooling = F.softmax(self.base.pool_score(nodes).squeeze(-1), dim=-1)
        graph_feature = torch.sum(nodes * pooling.unsqueeze(-1), dim=1)
        if self.base.condition_encoder is not None:
            settings = x[:, :, self.base.sensor_count :].mean(dim=1)
            graph_feature = torch.cat(
                [graph_feature, self.base.condition_encoder(settings)], dim=-1
            )
        return self.base.predictor(graph_feature).squeeze(-1)

    def freeze_for_meta(self) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        trainable = list(self.set_encoder.parameters())
        for layer in self.base.graph_layers:
            for module in (layer.context_u, layer.context_v, layer.context_gate):
                trainable.extend(module.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        return trainable

    def freeze_for_target(self) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        trainable = list(self.base.predictor.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        return trainable

    @torch.no_grad()
    def gate_mean(self, context: torch.Tensor) -> float:
        gates = [
            layer.edge_residual(context)[1]
            for layer in self.base.graph_layers
        ]
        return float(torch.stack(gates).mean().item())


def frame_windows(frame: pd.DataFrame, features: list[str], cfg: dict) -> dict:
    x, y, conditions, units = exp25.condition_windows(frame, features, cfg)
    return {"x": x, "y": y, "conditions": conditions, "units": units}


class ConditionEpisodeBank:
    def __init__(
        self,
        source_data: dict[str, dict],
        support_engines: int,
        query_engines: int,
        seed: int,
    ):
        self.data = source_data
        self.support_engines = int(support_engines)
        self.query_engines = int(query_engines)
        self.rng = np.random.default_rng(seed)
        self.domains_by_condition: dict[int, list[str]] = {}
        needed = support_engines + query_engines
        for domain, data in source_data.items():
            for condition in np.unique(data["conditions"]):
                mask = data["conditions"] == condition
                if len(np.unique(data["units"][mask])) >= needed:
                    self.domains_by_condition.setdefault(int(condition), []).append(domain)
        if len(self.domains_by_condition) < 2:
            raise RuntimeError("Fewer than two operating conditions form valid episodes")

    def sample(self, support_windows: int, query_windows: int):
        conditions = sorted(self.domains_by_condition)
        condition = conditions[int(self.rng.integers(0, len(conditions)))]
        domains = self.domains_by_condition[condition]
        domain = domains[int(self.rng.integers(0, len(domains)))]
        data = self.data[domain]
        condition_indices = np.flatnonzero(data["conditions"] == condition)
        condition_units = data["units"][condition_indices]
        unique_units = np.unique(condition_units)
        selected = self.rng.choice(
            unique_units,
            size=self.support_engines + self.query_engines,
            replace=False,
        )
        support_units = selected[: self.support_engines]
        query_units = selected[self.support_engines :]
        labels = data["y"].numpy()[condition_indices]
        support_local = exp18._sample_balanced_indices(
            condition_units,
            labels,
            support_units,
            support_windows,
            self.rng,
        )
        query_local = exp18._sample_balanced_indices(
            condition_units,
            labels,
            query_units,
            query_windows,
            self.rng,
        )
        support_indices = condition_indices[support_local]
        query_indices = condition_indices[query_local]
        return (
            domain,
            condition,
            data["x"][support_indices],
            data["y"][support_indices],
            data["x"][query_indices],
            data["y"][query_indices],
        )


def source_signature(args: argparse.Namespace, prior: torch.Tensor) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "model_seed": args.model_seed,
        "sensor_graph_k": args.sensor_graph_k,
        "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest()[:16],
        "context_dim": args.context_dim,
        "set_token_dim": args.set_token_dim,
        "graph_residual_rank": args.graph_residual_rank,
        "max_graph_gate": args.max_graph_gate,
        "gate_scale": args.gate_scale,
        "meta_steps": args.meta_steps,
        "meta_lr": args.meta_lr,
        "meta_weight_decay": args.meta_weight_decay,
        "source_support_engines": args.source_support_engines,
        "source_query_engines": args.source_query_engines,
        "source_support_windows": args.source_support_windows,
        "source_query_windows": args.source_query_windows,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_model(
    args: argparse.Namespace,
    cfg: dict,
    prior: torch.Tensor,
    feature_count: int,
) -> tuple[TrajectorySetGraphRegressor, Path]:
    checkpoint = exp25.project_path(args.checkpoint_dir) / (
        f"experiment18_source_bundle_{args.target}_modelseed{args.model_seed}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Experiment 18 checkpoint: {checkpoint}")
    bundle = exp24.exp17.safe_torch_load(checkpoint)
    if "static_prior" not in bundle.get("states", {}):
        raise ValueError("Experiment 18 checkpoint lacks the ordinary-pretrained state")
    seed_everything(args.model_seed + 25100)
    base = exp18.build_tcsg_model(feature_count, cfg, prior, args)
    exp18.load_compatible_state(base, bundle["states"]["static_prior"])
    for layer in base.graph_layers:
        layer.max_gate *= args.gate_scale
    model = TrajectorySetGraphRegressor(
        base,
        token_dim=args.set_token_dim,
        context_dim=args.context_dim,
        rul_cap=cfg["rul_cap"],
    )
    return model, checkpoint


def train_source_meta(
    args: argparse.Namespace,
    model: TrajectorySetGraphRegressor,
    source_data: dict[str, dict],
    device: torch.device,
) -> list[dict]:
    model = model.to(device)
    trainable = model.freeze_for_meta()
    optimizer = torch.optim.Adam(
        trainable, lr=args.meta_lr, weight_decay=args.meta_weight_decay
    )
    bank = ConditionEpisodeBank(
        source_data,
        args.source_support_engines,
        args.source_query_engines,
        args.model_seed + 25101,
    )
    report_every = max(1, args.meta_steps // 10)
    running = []
    history = []
    for step in range(1, args.meta_steps + 1):
        domain, condition, sx, sy, qx, qy = bank.sample(
            args.source_support_windows, args.source_query_windows
        )
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        model.eval()
        model.set_encoder.train()
        optimizer.zero_grad()
        context = model.encode_support(sx, sy)
        loss = F.mse_loss(model(qx, context), qy)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite source meta loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        running.append(float(loss.item()))
        if step % report_every == 0 or step == args.meta_steps:
            row = {
                "meta_step": step,
                "mean_query_loss": float(np.mean(running)),
                "last_domain": domain,
                "last_condition": int(condition),
                "last_gradient_norm": float(gradient_norm),
                "last_graph_gate_mean": model.gate_mean(context.detach()),
            }
            history.append(row)
            print(
                f"meta_step={step:04d}/{args.meta_steps} "
                f"query_loss={row['mean_query_loss']:.4f} "
                f"condition={condition} gate={row['last_graph_gate_mean']:.4f}"
            )
            running.clear()
    model.eval()
    return history


def load_or_train_source(
    args: argparse.Namespace,
    model: TrajectorySetGraphRegressor,
    source_data: dict[str, dict],
    prior: torch.Tensor,
    output: Path,
    device: torch.device,
) -> tuple[TrajectorySetGraphRegressor, list[dict]]:
    signature = source_signature(args, prior)
    cache = output / "source_cache" / (
        f"experiment25-1_source_{args.target}_modelseed{args.model_seed}.pt"
    )
    if args.resume and cache.is_file():
        saved = exp24.exp17.safe_torch_load(cache)
        if saved.get("signature") == signature:
            model.load_state_dict(saved["model"])
            print(f"[source cache] {cache}")
            return model.to(device).eval(), saved.get("history", [])
        print(f"[source cache ignored] signature mismatch: {cache}")
    history = train_source_meta(args, model, source_data, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "model": exp24.exp17.state_to_cpu(model),
            "history": history,
            "meta_trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
        },
        cache,
    )
    return model, history


@torch.no_grad()
def support_contexts(
    model: TrajectorySetGraphRegressor,
    x: torch.Tensor,
    y: torch.Tensor,
    conditions: np.ndarray,
    min_windows: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[int, int]]:
    model.eval()
    global_context = model.encode_support(x.to(device), y.to(device)).squeeze(0)
    contexts = {}
    counts = {}
    for condition in sorted(map(int, np.unique(conditions))):
        indices = np.flatnonzero(conditions == condition)
        counts[condition] = int(len(indices))
        if len(indices) >= min_windows:
            contexts[condition] = model.encode_support(
                x[indices].to(device), y[indices].to(device)
            ).squeeze(0)
    return contexts, global_context, counts


def routed_context(
    mode: str,
    conditions: np.ndarray,
    contexts: dict[int, torch.Tensor],
    global_context: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    available = sorted(contexts)
    wrong = {
        condition: available[(index + 1) % len(available)]
        for index, condition in enumerate(available)
    }
    selected = []
    fallback = 0
    for value in conditions:
        condition = int(value)
        if mode == "set_global":
            selected.append(global_context)
        elif mode == "set_condition_wrong" and len(available) > 1:
            if condition in wrong:
                selected.append(contexts[wrong[condition]])
            else:
                selected.append(global_context)
                fallback += 1
        elif condition in contexts:
            selected.append(contexts[condition])
        else:
            selected.append(global_context)
            fallback += 1
    return torch.stack(selected), fallback


@torch.no_grad()
def evaluate_routed(
    model: TrajectorySetGraphRegressor,
    data: dict,
    mode: str,
    contexts: dict[int, torch.Tensor],
    global_context: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, int]:
    model.eval()
    predictions = []
    fallback_total = 0
    for start in range(0, len(data["x"]), batch_size):
        stop = start + batch_size
        context, fallback = routed_context(
            mode,
            data["conditions"][start:stop],
            contexts,
            global_context,
        )
        prediction = model(data["x"][start:stop].to(device), context.to(device))
        predictions.extend(prediction.cpu().numpy().tolist())
        fallback_total += fallback
    return regression_metrics(data["y"].numpy(), predictions), fallback_total


def train_target_predictor(
    args: argparse.Namespace,
    source_model: TrajectorySetGraphRegressor,
    support: dict,
    validation: dict,
    mode: str,
    run_seed: int,
    device: torch.device,
) -> tuple[TrajectorySetGraphRegressor, list[dict], int, dict]:
    seed_everything(run_seed)
    learner = deepcopy(source_model).to(device)
    trainable = learner.freeze_for_target()
    contexts, global_context, condition_counts = support_contexts(
        learner,
        support["x"],
        support["y"],
        support["conditions"],
        args.min_condition_windows,
        device,
    )
    optimizer = torch.optim.Adam(trainable, lr=args.target_lr)
    weights = exp24.exp7.sampling_weights(
        support["y"].numpy(), support["units"], "engine_stage"
    )
    weights = torch.as_tensor(weights, dtype=torch.double)
    generator = torch.Generator().manual_seed(run_seed + 25102)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = deepcopy(learner.state_dict())
    history = []
    batch_size = 64
    for epoch in range(1, args.target_epochs + 1):
        order = torch.multinomial(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        learner.eval()
        learner.base.predictor.train()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].numpy()
            x = support["x"][indices].to(device)
            y = support["y"][indices].to(device)
            context, _ = routed_context(
                mode,
                support["conditions"][indices],
                contexts,
                global_context,
            )
            optimizer.zero_grad()
            loss = F.mse_loss(learner(x, context.to(device)), y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite target adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics, fallback = evaluate_routed(
            learner,
            validation,
            mode,
            contexts,
            global_context,
            device,
            batch_size=256,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_fallback_windows": fallback,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            best_state = deepcopy(learner.state_dict())
    learner.load_state_dict(best_state)
    metrics, fallback = evaluate_routed(
        learner,
        validation,
        mode,
        contexts,
        global_context,
        device,
        batch_size=256,
    )
    audit = {
        "context_mode": mode,
        "available_conditions": sorted(contexts),
        "condition_window_counts": condition_counts,
        "validation_fallback_windows": fallback,
        "context_pairwise_cosine_distance": exp25.mean_pairwise_distance(
            [value.cpu() for value in contexts.values()]
        ),
    }
    return learner, history, best_epoch, {**metrics, **audit}


def comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(results["k"].unique()):
        current = results[results["k"] == k]
        for reference, name in (
            ("set_global", "condition_true_vs_global"),
            ("set_condition_wrong", "condition_true_vs_wrong"),
        ):
            true = current[current["mode"] == "set_condition_true"].set_index(
                "target_split_seed"
            )
            other = current[current["mode"] == reference].set_index(
                "target_split_seed"
            )
            paired = true.join(other, lsuffix="_candidate", rsuffix="_reference")
            rmse_delta = paired["rmse_candidate"] - paired["rmse_reference"]
            nasa_delta = (
                paired["nasa_score_candidate"] - paired["nasa_score_reference"]
            )
            rows.append(
                {
                    "target": current.iloc[0]["target"],
                    "model_seed": int(current.iloc[0]["model_seed"]),
                    "k": int(k),
                    "comparison": name,
                    "candidate": "set_condition_true",
                    "reference": reference,
                    "n_target_splits": int(len(paired)),
                    "rmse_delta_mean": float(rmse_delta.mean()),
                    "rmse_improvement_pct": float(
                        -100.0 * rmse_delta.mean()
                        / max(float(paired["rmse_reference"].mean()), 1e-8)
                    ),
                    "rmse_win_rate": float((rmse_delta < 0).mean()),
                    "nasa_score_delta_mean": float(nasa_delta.mean()),
                    "nasa_score_win_rate": float((nasa_delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def self_check() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    assert abs(exp25.cosine_distance(first, second) - 1.0) < 1e-6
    contexts = {0: first, 1: second}
    routed, fallback = routed_context(
        "set_condition_wrong", np.asarray([0, 1]), contexts, first
    )
    assert fallback == 0 and torch.equal(routed[0], second)


def main() -> None:
    self_check()
    args = parse_args()
    if args.dry_run:
        args.meta_steps = min(args.meta_steps, 2)
        args.target_epochs = 1
        args.target_split_seeds = args.target_split_seeds[:1]
        args.k_values = args.k_values[:1]
        args.source_support_windows = min(args.source_support_windows, 32)
        args.source_query_windows = min(args.source_query_windows, 32)

    cfg = load_config(args)
    validate_args(args, cfg)
    protocol, protocol_path = exp25.load_protocol(args)
    output = exp25.project_path(args.output_dir) / f"seed{args.model_seed}"
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(
        f"[{SCRIPT_VERSION}] target={args.target} model_seed={args.model_seed} "
        f"device={device}"
    )
    print(f"[protocol] {protocol_path}")
    print("[policy] train files only; official test is not loaded or evaluated")

    normalized, features, source_condition_counts = exp24.normalized_train_frames(
        cfg, PREPROCESSING
    )
    source_data = {
        domain: frame_windows(normalized[domain], features, cfg)
        for domain in cfg["source_domains"]
    }
    prior, _ = exp24.train_only_prior(cfg, PREPROCESSING, args.sensor_graph_k)
    model, initialization_checkpoint = build_model(
        args, cfg, prior, len(features)
    )
    model, source_history = load_or_train_source(
        args, model, source_data, prior, output, device
    )

    target = normalized[args.target]
    validation_units = set(map(int, protocol["validation_units"]))
    validation = frame_windows(
        target[target["unit"].isin(validation_units)], features, cfg
    )
    nested = protocol["nested_adaptation_units_by_target_split_seed"]
    raw_rows = []
    target_histories = {}
    context_rows = []
    for split_seed in args.target_split_seeds:
        for k in args.k_values:
            support_units = set(map(int, nested[str(split_seed)][str(k)]))
            if support_units & validation_units:
                raise AssertionError("Support and validation engines overlap")
            support = frame_windows(
                target[target["unit"].isin(support_units)], features, cfg
            )
            run_seed = exp18.target_run_seed(args.model_seed, split_seed)
            for mode in MODES:
                learner, history, best_epoch, result = train_target_predictor(
                    args,
                    model,
                    support,
                    validation,
                    mode,
                    run_seed,
                    device,
                )
                row = {
                    "target": args.target,
                    "model_seed": args.model_seed,
                    "target_split_seed": split_seed,
                    "target_run_seed": run_seed,
                    "k": k,
                    "mode": mode,
                    "best_target_epoch": best_epoch,
                    "support_engine_count": len(support_units),
                    "support_window_count": len(support["y"]),
                    "validation_engine_count": len(validation_units),
                    "validation_window_count": len(validation["y"]),
                    **result,
                }
                raw_rows.append(row)
                context_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "target",
                            "model_seed",
                            "target_split_seed",
                            "k",
                            "mode",
                            "available_conditions",
                            "condition_window_counts",
                            "validation_fallback_windows",
                            "context_pairwise_cosine_distance",
                        }
                    }
                )
                target_histories[
                    f"split{split_seed}_k{k}_{mode}"
                ] = history
                print(
                    f"[target] split={split_seed} k={k} mode={mode} "
                    f"rmse={result['rmse']:.4f} nasa={result['nasa_score']:.2f}"
                )
                del learner
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["target", "model_seed", "k", "mode"], as_index=False)
        .agg(
            n_target_splits=("target_split_seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            nasa_score_mean=("nasa_score", "mean"),
            context_distance_mean=("context_pairwise_cosine_distance", "mean"),
            fallback_windows=("validation_fallback_windows", "sum"),
        )
        .sort_values(["k", "rmse_mean"])
    )
    comparison = comparisons(raw)
    true_vs_global = comparison[
        comparison["comparison"] == "condition_true_vs_global"
    ]
    true_vs_wrong = comparison[
        comparison["comparison"] == "condition_true_vs_wrong"
    ]
    pilot_success = bool(
        len(true_vs_global)
        and len(true_vs_wrong)
        and (true_vs_global["rmse_improvement_pct"] >= 1.0).all()
        and (true_vs_global["rmse_win_rate"] == 1.0).all()
        and (true_vs_global["nasa_score_delta_mean"] <= 0.0).all()
        and (true_vs_wrong["rmse_improvement_pct"] >= 1.0).all()
        and (true_vs_wrong["rmse_win_rate"] == 1.0).all()
    )
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "model_seed": args.model_seed,
        "dry_run": bool(args.dry_run),
        "training_scope": "set_encoder_and_low_rank_graph_adapter_only",
        "target_adaptation_scope": "predictor_only",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "protocol_path": str(protocol_path),
        "initialization_checkpoint": str(initialization_checkpoint),
        "source_condition_counts": source_condition_counts,
        "meta_steps": args.meta_steps,
        "target_split_seeds": args.target_split_seeds,
        "k_values": args.k_values,
        "modes": list(MODES),
        "pilot_success_rule": {
            "true_vs_global_rmse_improvement_pct_at_least": 1.0,
            "true_vs_global_split_win_rate": 1.0,
            "true_vs_global_nasa_delta_must_be_leq": 0.0,
            "true_vs_wrong_rmse_improvement_pct_at_least": 1.0,
            "true_vs_wrong_split_win_rate": 1.0,
        },
        "pilot_success": pilot_success,
        "next_step": (
            "expand_to_experiment25-2_formal_comparison"
            if pilot_success
            else "do_not_scale; add_counterfactual_objective_or_abandon"
        ),
    }

    prefix = f"experiment25-1_{args.target}_seed{args.model_seed}"
    atomic_write_text(output / f"{prefix}_summary.csv", summary.to_csv(index=False))
    atomic_write_text(
        output / f"{prefix}_comparisons.csv", comparison.to_csv(index=False)
    )
    atomic_write_text(
        output / f"{prefix}_context_audit.csv",
        pd.DataFrame(context_rows).to_csv(index=False),
    )
    atomic_write_text(
        output / f"{prefix}_raw.json",
        json.dumps(raw_rows, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        output / f"{prefix}_source_history.json",
        json.dumps(source_history, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        output / f"{prefix}_target_history.json",
        json.dumps(target_histories, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        output / f"{prefix}_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print(f"[{SCRIPT_VERSION}] complete: {output}")


if __name__ == "__main__":
    main()
