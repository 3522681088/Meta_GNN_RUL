"""Experiment 26: engine-task first-order meta-initialization pilot.

Experiments 25 through 25-2 showed that target-context routing barely changes
predictions. This experiment removes context conditioning and asks a more
direct question: does engine-disjoint episodic training learn a predictor
initialization that adapts better from K target engines?

Three predictor initializations share the same frozen static graph backbone:

* static_init: Experiment 18 ordinary-pretrained predictor.
* supervised_budget: direct source training with the same support/query
  gradient-evaluation budget as the meta learner.
* fomaml_engine: first-order ANIL/FOMAML on source domain-condition episodes.

Only C-MAPSS train files are accessed. Target adaptation updates the predictor
only, identically for all three regimes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_experiment25_1():
    path = PROJECT_ROOT / "scripts" / "experiment25-1_trajectory_set_encoder.py"
    if not path.is_file():
        path = PROJECT_ROOT / "experiment25-1_trajectory_set_encoder.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Experiment 25-1 runner: {path}")
    spec = importlib.util.spec_from_file_location("experiment25_1_for_exp26", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Experiment 25-1 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp251 = _load_experiment25_1()
exp18 = exp251.exp18
exp24 = exp251.exp24
exp25 = exp251.exp25
regression_metrics = exp251.regression_metrics
EXPECTED_OFFICIAL_TEST_ENGINES = exp251.EXPECTED_OFFICIAL_TEST_ENGINES
atomic_write_text = exp251.atomic_write_text
resolve_device = exp251.resolve_device
seed_everything = exp251.seed_everything

SCRIPT_VERSION = "experiment26_engine_task_meta_initialization_v2"
PREPROCESSING = exp251.PREPROCESSING
REGIMES = ("static_init", "supervised_budget", "fomaml_engine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 26: engine-task FOMAML initialization pilot"
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
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--meta-steps", type=int, default=600)
    parser.add_argument("--meta-lr", type=float, default=0.0001)
    parser.add_argument("--meta-weight-decay", type=float, default=0.0)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--source-support-engines", type=int, default=5)
    parser.add_argument("--source-query-engines", type=int, default=5)
    parser.add_argument("--source-support-windows", type=int, default=128)
    parser.add_argument("--source-query-windows", type=int, default=128)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--target-lr", type=float, default=0.001)
    parser.add_argument("--min-condition-windows", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment26_engine_task_meta_initialization",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    positive = (
        args.meta_steps,
        args.meta_lr,
        args.inner_steps,
        args.inner_lr,
        args.source_support_engines,
        args.source_query_engines,
        args.source_support_windows,
        args.source_query_windows,
        args.target_epochs,
        args.target_lr,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Training counts and learning rates must be positive")
    if args.k_values != [5]:
        raise ValueError("Experiment 26 pilot is locked to K=5")
    if len(args.target_split_seeds) < 1:
        raise ValueError("At least one target split seed is required")
    if not 1 <= args.sensor_graph_k < len(cfg["sensor_columns"]):
        raise ValueError("--sensor-graph-k is outside the sensor count")


def state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def disable_context_graph(model: exp251.TrajectorySetGraphRegressor) -> None:
    for layer in model.base.graph_layers:
        layer.max_gate = 0.0


@torch.no_grad()
def encode_windows(
    model: exp251.TrajectorySetGraphRegressor,
    windows: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run the frozen static graph once and return predictor input features."""
    model.eval()
    features = []
    base = model.base
    context_dim = next(base.graph_layers[0].context_u.parameters()).shape[-1]
    context = torch.zeros(1, context_dim, device=device)
    for start in range(0, len(windows), batch_size):
        x = windows[start : start + batch_size].to(device)
        nodes = base.temporal(x[:, :, : base.sensor_count])
        for layer in base.graph_layers:
            nodes = layer(nodes, context)
        pooling = F.softmax(base.pool_score(nodes).squeeze(-1), dim=-1)
        graph_feature = torch.sum(nodes * pooling.unsqueeze(-1), dim=1)
        if base.condition_encoder is not None:
            settings = x[:, :, base.sensor_count :].mean(dim=1)
            graph_feature = torch.cat(
                [graph_feature, base.condition_encoder(settings)], dim=-1
            )
        features.append(graph_feature.cpu())
    return torch.cat(features, dim=0)


def encode_data(
    model: exp251.TrajectorySetGraphRegressor,
    data: dict,
    device: torch.device,
) -> dict:
    return {
        "x": encode_windows(model, data["x"], device),
        "y": data["y"],
        "conditions": data["conditions"],
        "units": data["units"],
    }


def copy_query_gradients(source: nn.Module, destination: nn.Module) -> None:
    for source_parameter, destination_parameter in zip(
        source.parameters(), destination.parameters(), strict=True
    ):
        if source_parameter.grad is None:
            raise RuntimeError("Adapted predictor has a missing query gradient")
        destination_parameter.grad = source_parameter.grad.detach().clone()


def train_supervised_budget(
    args: argparse.Namespace,
    initial: nn.Module,
    bank: exp251.ConditionEpisodeBank,
    device: torch.device,
) -> tuple[nn.Module, list[dict]]:
    predictor = deepcopy(initial).to(device)
    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.meta_lr,
        weight_decay=args.meta_weight_decay,
    )
    report_every = max(1, args.meta_steps // 10)
    running = []
    history = []
    predictor.train()
    for step in range(1, args.meta_steps + 1):
        domain, condition, sx, sy, qx, qy = bank.sample(
            args.source_support_windows, args.source_query_windows
        )
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        optimizer.zero_grad()
        support_loss = F.mse_loss(predictor(sx).squeeze(-1), sy)
        query_loss = F.mse_loss(predictor(qx).squeeze(-1), qy)
        # Match one persistent outer update per episode. Weighting the support
        # term by inner_steps matches FOMAML's support-label exposure without
        # granting the control extra optimizer steps.
        loss = (
            args.inner_steps * support_loss + query_loss
        ) / (args.inner_steps + 1)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
        optimizer.step()
        running.append(
            {
                "mean_step_loss": float(loss.item()),
                "query_loss": float(query_loss.item()),
            }
        )
        if step % report_every == 0 or step == args.meta_steps:
            row = {
                "meta_step": step,
                "mean_training_loss": float(
                    np.mean([item["mean_step_loss"] for item in running])
                ),
                "mean_query_loss": float(
                    np.mean([item["query_loss"] for item in running])
                ),
                "last_domain": domain,
                "last_condition": int(condition),
                "last_gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(
                f"[supervised_budget] step={step:04d}/{args.meta_steps} "
                f"query_loss={row['mean_query_loss']:.4f} "
                f"condition={condition}"
            )
            running.clear()
    return predictor.eval(), history


def train_fomaml(
    args: argparse.Namespace,
    initial: nn.Module,
    bank: exp251.ConditionEpisodeBank,
    device: torch.device,
) -> tuple[nn.Module, list[dict]]:
    predictor = deepcopy(initial).to(device)
    outer_optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.meta_lr,
        weight_decay=args.meta_weight_decay,
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

        predictor.eval()
        with torch.no_grad():
            pre_adaptation_query_loss = F.mse_loss(
                predictor(qx).squeeze(-1), qy
            )
        adapted = deepcopy(predictor).to(device)
        inner_optimizer = torch.optim.SGD(
            adapted.parameters(), lr=args.inner_lr
        )
        adapted.train()
        support_losses = []
        for _ in range(args.inner_steps):
            inner_optimizer.zero_grad()
            support_loss = F.mse_loss(adapted(sx).squeeze(-1), sy)
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            inner_optimizer.step()
            support_losses.append(float(support_loss.item()))

        # Audit adaptation with Dropout disabled on both sides. The separate
        # train-mode query below remains the actual FOMAML outer objective.
        adapted.eval()
        with torch.no_grad():
            post_adaptation_query_loss = F.mse_loss(
                adapted(qx).squeeze(-1), qy
            )
        adapted.train()
        inner_optimizer.zero_grad()
        query_loss = F.mse_loss(adapted(qx).squeeze(-1), qy)
        query_loss.backward()
        outer_optimizer.zero_grad()
        copy_query_gradients(adapted, predictor)
        gradient_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
        outer_optimizer.step()
        running.append(
            {
                "support_loss": float(np.mean(support_losses)),
                "pre_adaptation_query_loss": float(
                    pre_adaptation_query_loss.item()
                ),
                "post_adaptation_query_loss": float(
                    post_adaptation_query_loss.item()
                ),
                "query_loss": float(query_loss.item()),
                "adaptation_gain": float(
                    pre_adaptation_query_loss.item()
                    - post_adaptation_query_loss.item()
                ),
            }
        )
        del adapted, inner_optimizer

        if step % report_every == 0 or step == args.meta_steps:
            row = {
                "meta_step": step,
                "mean_support_loss": float(
                    np.mean([item["support_loss"] for item in running])
                ),
                "mean_query_loss": float(
                    np.mean([item["query_loss"] for item in running])
                ),
                "mean_adaptation_gain": float(
                    np.mean([item["adaptation_gain"] for item in running])
                ),
                "last_domain": domain,
                "last_condition": int(condition),
                "last_gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(
                f"[fomaml_engine] step={step:04d}/{args.meta_steps} "
                f"support_loss={row['mean_support_loss']:.4f} "
                f"query_loss={row['mean_query_loss']:.4f} "
                f"adapt_gain={row['mean_adaptation_gain']:.4f} "
                f"condition={condition}"
            )
            running.clear()
    return predictor.eval(), history


def source_signature(
    args: argparse.Namespace,
    prior: torch.Tensor,
    checkpoint: Path,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "model_seed": args.model_seed,
        "initialization_checkpoint": str(checkpoint.resolve()),
        "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest()[:16],
        "meta_steps": args.meta_steps,
        "meta_lr": args.meta_lr,
        "meta_weight_decay": args.meta_weight_decay,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "source_support_engines": args.source_support_engines,
        "source_query_engines": args.source_query_engines,
        "source_support_windows": args.source_support_windows,
        "source_query_windows": args.source_query_windows,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def load_or_train_initializations(
    args: argparse.Namespace,
    initial: nn.Module,
    source_data: dict[str, dict],
    prior: torch.Tensor,
    checkpoint: Path,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[dict]]]:
    signature = source_signature(args, prior, checkpoint)
    cache = output / "source_cache" / (
        f"experiment26_source_{args.target}_modelseed{args.model_seed}.pt"
    )
    if args.resume and cache.is_file():
        saved = exp24.exp17.safe_torch_load(cache)
        if saved.get("signature") == signature:
            print(f"[source cache] {cache}")
            return saved["predictor_states"], saved.get("histories", {})
        print(f"[source cache ignored] signature mismatch: {cache}")

    static_state = state_to_cpu(initial)
    supervised_bank = exp251.ConditionEpisodeBank(
        source_data,
        args.source_support_engines,
        args.source_query_engines,
        args.model_seed + 26001,
    )
    meta_bank = exp251.ConditionEpisodeBank(
        source_data,
        args.source_support_engines,
        args.source_query_engines,
        args.model_seed + 26001,
    )
    supervised, supervised_history = train_supervised_budget(
        args, initial, supervised_bank, device
    )
    fomaml, fomaml_history = train_fomaml(args, initial, meta_bank, device)
    states = {
        "static_init": static_state,
        "supervised_budget": state_to_cpu(supervised),
        "fomaml_engine": state_to_cpu(fomaml),
    }
    histories = {
        "static_init": [],
        "supervised_budget": supervised_history,
        "fomaml_engine": fomaml_history,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "predictor_states": states,
            "histories": histories,
        },
        cache,
    )
    return states, histories


@torch.no_grad()
def evaluate_predictor(
    predictor: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    predictor.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        prediction = predictor(features[start : start + batch_size].to(device))
        predictions.extend(prediction.squeeze(-1).cpu().numpy().tolist())
    return regression_metrics(labels.numpy(), predictions)


def train_target_predictor(
    args: argparse.Namespace,
    initial: nn.Module,
    state: dict[str, torch.Tensor],
    support: dict,
    validation: dict,
    run_seed: int,
    device: torch.device,
) -> tuple[list[dict], int, dict]:
    seed_everything(run_seed)
    predictor = deepcopy(initial).to(device)
    predictor.load_state_dict(state)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.target_lr)
    weights = exp24.exp7.sampling_weights(
        support["y"].numpy(), support["units"], "engine_stage"
    )
    weights = torch.as_tensor(weights, dtype=torch.double)
    generator = torch.Generator().manual_seed(run_seed + 26002)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = state_to_cpu(predictor)
    history = []
    batch_size = 64
    for epoch in range(1, args.target_epochs + 1):
        order = torch.multinomial(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        predictor.train()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = support["x"][indices].to(device)
            y = support["y"][indices].to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(predictor(x).squeeze(-1), y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite target adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate_predictor(
            predictor, validation["x"], validation["y"], device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            best_state = state_to_cpu(predictor)
    predictor.load_state_dict(best_state)
    result = evaluate_predictor(
        predictor, validation["x"], validation["y"], device
    )
    del predictor
    return history, best_epoch, result


def comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate = results[results["regime"] == "fomaml_engine"].set_index(
        "target_split_seed"
    )
    for reference, name in (
        ("supervised_budget", "fomaml_vs_supervised_budget"),
        ("static_init", "fomaml_vs_static_init"),
    ):
        other = results[results["regime"] == reference].set_index(
            "target_split_seed"
        )
        paired = candidate.join(other, lsuffix="_candidate", rsuffix="_reference")
        rmse_delta = paired["rmse_candidate"] - paired["rmse_reference"]
        nasa_delta = (
            paired["nasa_score_candidate"] - paired["nasa_score_reference"]
        )
        rows.append(
            {
                "target": results.iloc[0]["target"],
                "model_seed": int(results.iloc[0]["model_seed"]),
                "k": int(results.iloc[0]["k"]),
                "comparison": name,
                "candidate": "fomaml_engine",
                "reference": reference,
                "n_target_splits": int(len(paired)),
                "rmse_delta_mean": float(rmse_delta.mean()),
                "rmse_improvement_pct": float(
                    -100.0
                    * rmse_delta.mean()
                    / max(float(paired["rmse_reference"].mean()), 1e-8)
                ),
                "rmse_win_rate": float((rmse_delta < 0).mean()),
                "nasa_score_delta_mean": float(nasa_delta.mean()),
                "nasa_score_win_rate": float((nasa_delta < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def self_check() -> None:
    source = nn.Linear(2, 1)
    destination = nn.Linear(2, 1)
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    copy_query_gradients(source, destination)
    assert all(
        parameter.grad is not None
        and torch.equal(parameter.grad, torch.ones_like(parameter))
        for parameter in destination.parameters()
    )


def main() -> None:
    self_check()
    args = parse_args()
    if args.dry_run:
        args.meta_steps = min(args.meta_steps, 2)
        args.inner_steps = 1
        args.target_epochs = 1
        args.target_split_seeds = args.target_split_seeds[:1]
        args.source_support_windows = min(args.source_support_windows, 32)
        args.source_query_windows = min(args.source_query_windows, 32)

    cfg = exp251.load_config(args)
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
    print(
        f"[budget] {args.meta_steps} episodes and persistent optimizer steps "
        f"per trained source regime; support weight={args.inner_steps}"
    )

    normalized, feature_names, source_condition_counts = (
        exp24.normalized_train_frames(cfg, PREPROCESSING)
    )
    source_windows = {
        domain: exp251.frame_windows(normalized[domain], feature_names, cfg)
        for domain in cfg["source_domains"]
    }
    prior, _ = exp24.train_only_prior(cfg, PREPROCESSING, args.sensor_graph_k)
    model, initialization_checkpoint = exp251.build_model(
        args, cfg, prior, len(feature_names)
    )
    disable_context_graph(model)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial_predictor = deepcopy(model.base.predictor).cpu()
    for parameter in initial_predictor.parameters():
        parameter.requires_grad_(True)

    print("[features] encoding frozen source backbone")
    encoded_sources = {
        domain: encode_data(model, data, device)
        for domain, data in source_windows.items()
    }
    predictor_states, source_histories = load_or_train_initializations(
        args,
        initial_predictor,
        encoded_sources,
        prior,
        initialization_checkpoint,
        output,
        device,
    )

    target = normalized[args.target]
    validation_units = set(map(int, protocol["validation_units"]))
    validation_windows = exp251.frame_windows(
        target[target["unit"].isin(validation_units)], feature_names, cfg
    )
    print("[features] encoding frozen target validation backbone")
    validation = encode_data(model, validation_windows, device)
    nested = protocol["nested_adaptation_units_by_target_split_seed"]
    raw_rows = []
    target_histories = {}
    for split_seed in args.target_split_seeds:
        for k in args.k_values:
            support_units = set(map(int, nested[str(split_seed)][str(k)]))
            if support_units & validation_units:
                raise AssertionError("Support and validation engines overlap")
            support_windows = exp251.frame_windows(
                target[target["unit"].isin(support_units)], feature_names, cfg
            )
            support = encode_data(model, support_windows, device)
            run_seed = exp18.target_run_seed(args.model_seed, split_seed)
            for regime in REGIMES:
                history, best_epoch, result = train_target_predictor(
                    args,
                    initial_predictor,
                    predictor_states[regime],
                    support,
                    validation,
                    run_seed,
                    device,
                )
                row = {
                    "target": args.target,
                    "model_seed": args.model_seed,
                    "target_split_seed": split_seed,
                    "target_run_seed": run_seed,
                    "k": k,
                    "regime": regime,
                    "best_target_epoch": best_epoch,
                    "support_engine_count": len(support_units),
                    "support_window_count": len(support["y"]),
                    "validation_engine_count": len(validation_units),
                    "validation_window_count": len(validation["y"]),
                    **result,
                }
                raw_rows.append(row)
                target_histories[f"split{split_seed}_k{k}_{regime}"] = history
                print(
                    f"[target] split={split_seed} k={k} regime={regime} "
                    f"rmse={result['rmse']:.4f} nasa={result['nasa_score']:.2f}"
                )

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["target", "model_seed", "k", "regime"], as_index=False)
        .agg(
            n_target_splits=("target_split_seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            nasa_score_mean=("nasa_score", "mean"),
        )
        .sort_values(["k", "rmse_mean"])
    )
    comparison = comparisons(raw)
    primary = comparison[
        comparison["comparison"] == "fomaml_vs_supervised_budget"
    ]
    static = comparison[
        comparison["comparison"] == "fomaml_vs_static_init"
    ]
    pilot_success = bool(
        len(primary)
        and len(static)
        and (primary["rmse_improvement_pct"] >= 1.0).all()
        and (primary["rmse_win_rate"] == 1.0).all()
        and (primary["nasa_score_delta_mean"] <= 0.0).all()
        and (static["rmse_delta_mean"] <= 0.0).all()
    )
    report = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "model_seed": args.model_seed,
        "dry_run": bool(args.dry_run),
        "question": "does episodic training learn a better fast-adaptation initialization",
        "source_task": "domain_condition_with_engine_disjoint_support_query",
        "source_training_scope": "predictor_only_frozen_static_graph_backbone",
        "target_adaptation_scope": "predictor_only",
        "context_graph_gate": 0.0,
        "budget_matching": {
            "persistent_optimizer_steps_per_trained_regime": args.meta_steps,
            "same_source_episodes": True,
            "supervised_support_loss_weight": args.inner_steps,
            "note": "FOMAML necessarily adds ephemeral inner-loop optimizer steps",
        },
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "protocol_path": str(protocol_path),
        "initialization_checkpoint": str(initialization_checkpoint),
        "source_condition_counts": source_condition_counts,
        "meta_steps": args.meta_steps,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "target_split_seeds": args.target_split_seeds,
        "k_values": args.k_values,
        "regimes": list(REGIMES),
        "pilot_success_rule": {
            "fomaml_vs_supervised_rmse_improvement_pct_at_least": 1.0,
            "fomaml_vs_supervised_split_win_rate": 1.0,
            "fomaml_vs_supervised_nasa_delta_must_be_leq": 0.0,
            "fomaml_must_not_worsen_rmse_vs_static_init": True,
        },
        "pilot_success": pilot_success,
        "next_step": (
            "expand_experiment26_to_seeds44_45_46"
            if pilot_success
            else "do_not_scale; inspect_source_adaptation_gap_before_new_idea"
        ),
    }

    prefix = f"experiment26_{args.target}_seed{args.model_seed}"
    atomic_write_text(output / f"{prefix}_summary.csv", summary.to_csv(index=False))
    atomic_write_text(
        output / f"{prefix}_comparisons.csv", comparison.to_csv(index=False)
    )
    atomic_write_text(
        output / f"{prefix}_raw.json",
        json.dumps(raw_rows, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_write_text(
        output / f"{prefix}_source_history.json",
        json.dumps(source_histories, ensure_ascii=False, indent=2, allow_nan=False),
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
