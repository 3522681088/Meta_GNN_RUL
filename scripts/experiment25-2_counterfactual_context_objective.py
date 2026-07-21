"""Experiment 25-2: counterfactual condition-context objective pilot.

Experiment 25-1 learned a trajectory Set Encoder, but predictions remained
invariant to correct, global, and wrong condition contexts. This follow-up
changes only the source meta-training objective: the correct context must
predict a query better than a context built from another operating condition.

The model, target adaptation, data split, and three evaluation controls are
reused from Experiment 25-1. Only C-MAPSS train files are accessed.
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
    spec = importlib.util.spec_from_file_location("experiment25_1_runner", path)
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
EXPECTED_OFFICIAL_TEST_ENGINES = exp251.EXPECTED_OFFICIAL_TEST_ENGINES
atomic_write_text = exp251.atomic_write_text
resolve_device = exp251.resolve_device
seed_everything = exp251.seed_everything

SCRIPT_VERSION = "experiment25-2_counterfactual_context_objective_v1"
PREPROCESSING = exp251.PREPROCESSING
MODES = exp251.MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 25-2: counterfactual context-objective pilot"
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
    parser.add_argument("--counterfactual-weight", type=float, default=10.0)
    parser.add_argument("--counterfactual-margin-rul", type=float, default=1.0)
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
        default="outputs/experiment25-2_counterfactual_context_objective",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, cfg: dict) -> None:
    exp251.validate_args(args, cfg)
    if args.counterfactual_weight <= 0:
        raise ValueError("--counterfactual-weight must be positive")
    if args.counterfactual_margin_rul < 0:
        raise ValueError("--counterfactual-margin-rul cannot be negative")


class CounterfactualEpisodeBank(exp251.ConditionEpisodeBank):
    """Samples matched and wrong-condition support from the same source domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditions_by_domain: dict[str, list[int]] = {}
        for condition, domains in self.domains_by_condition.items():
            for domain in domains:
                self.conditions_by_domain.setdefault(domain, []).append(condition)
        self.conditions_by_domain = {
            domain: sorted(conditions)
            for domain, conditions in self.conditions_by_domain.items()
            if len(conditions) >= 2
        }
        if not self.conditions_by_domain:
            raise RuntimeError(
                "No source domain has two conditions with enough disjoint engines"
            )

    def sample_pair(self, support_windows: int, query_windows: int):
        domains = sorted(self.conditions_by_domain)
        domain = domains[int(self.rng.integers(0, len(domains)))]
        conditions = self.conditions_by_domain[domain]
        chosen = self.rng.choice(conditions, size=2, replace=False)
        condition, wrong_condition = map(int, chosen)
        data = self.data[domain]

        indices = np.flatnonzero(data["conditions"] == condition)
        units = data["units"][indices]
        selected = self.rng.choice(
            np.unique(units),
            size=self.support_engines + self.query_engines,
            replace=False,
        )
        support_units = selected[: self.support_engines]
        query_units = selected[self.support_engines :]
        labels = data["y"].numpy()[indices]
        support_local = exp18._sample_balanced_indices(
            units, labels, support_units, support_windows, self.rng
        )
        query_local = exp18._sample_balanced_indices(
            units, labels, query_units, query_windows, self.rng
        )

        wrong_indices = np.flatnonzero(data["conditions"] == wrong_condition)
        wrong_units = data["units"][wrong_indices]
        available_wrong_units = np.setdiff1d(np.unique(wrong_units), query_units)
        if len(available_wrong_units) < self.support_engines:
            raise RuntimeError("Cannot make wrong support engine-disjoint from query")
        wrong_support_units = self.rng.choice(
            available_wrong_units, size=self.support_engines, replace=False
        )
        wrong_labels = data["y"].numpy()[wrong_indices]
        wrong_local = exp18._sample_balanced_indices(
            wrong_units,
            wrong_labels,
            wrong_support_units,
            support_windows,
            self.rng,
        )

        support_indices = indices[support_local]
        query_indices = indices[query_local]
        wrong_support_indices = wrong_indices[wrong_local]
        return (
            domain,
            condition,
            wrong_condition,
            data["x"][support_indices],
            data["y"][support_indices],
            data["x"][wrong_support_indices],
            data["y"][wrong_support_indices],
            data["x"][query_indices],
            data["y"][query_indices],
        )


def counterfactual_rank_loss(
    matched_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    target: torch.Tensor,
    margin_rul: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched_error = (matched_prediction - target).abs()
    wrong_error = (wrong_prediction - target).abs()
    loss = F.relu(float(margin_rul) + matched_error - wrong_error).mean()
    error_gap = (wrong_error - matched_error).mean()
    return loss, error_gap


def source_signature(args: argparse.Namespace, prior: torch.Tensor) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "base_signature": exp251.source_signature(args, prior),
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin_rul": args.counterfactual_margin_rul,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def train_source_meta(
    args: argparse.Namespace,
    model: exp251.TrajectorySetGraphRegressor,
    source_data: dict[str, dict],
    device: torch.device,
) -> list[dict]:
    model = model.to(device)
    trainable = model.freeze_for_meta()
    optimizer = torch.optim.Adam(
        trainable, lr=args.meta_lr, weight_decay=args.meta_weight_decay
    )
    bank = CounterfactualEpisodeBank(
        source_data,
        args.source_support_engines,
        args.source_query_engines,
        args.model_seed + 25201,
    )
    report_every = max(1, args.meta_steps // 10)
    running: list[dict] = []
    history = []
    for step in range(1, args.meta_steps + 1):
        (
            domain,
            condition,
            wrong_condition,
            sx,
            sy,
            wx,
            wy,
            qx,
            qy,
        ) = bank.sample_pair(
            args.source_support_windows, args.source_query_windows
        )
        sx, sy = sx.to(device), sy.to(device)
        wx, wy = wx.to(device), wy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        model.eval()
        model.set_encoder.train()
        optimizer.zero_grad()
        matched_context = model.encode_support(sx, sy)
        wrong_context = model.encode_support(wx, wy)
        matched_prediction = model(qx, matched_context)
        wrong_prediction = model(qx, wrong_context)
        query_loss = F.mse_loss(matched_prediction, qy)
        rank_loss, error_gap = counterfactual_rank_loss(
            matched_prediction,
            wrong_prediction,
            qy,
            args.counterfactual_margin_rul,
        )
        loss = query_loss + args.counterfactual_weight * rank_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite counterfactual meta loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        running.append(
            {
                "query_loss": float(query_loss.item()),
                "rank_loss": float(rank_loss.item()),
                "error_gap": float(error_gap.item()),
                "total_loss": float(loss.item()),
            }
        )

        if step % report_every == 0 or step == args.meta_steps:
            row = {
                "meta_step": step,
                "mean_query_loss": float(
                    np.mean([item["query_loss"] for item in running])
                ),
                "mean_counterfactual_rank_loss": float(
                    np.mean([item["rank_loss"] for item in running])
                ),
                "mean_wrong_minus_matched_abs_error": float(
                    np.mean([item["error_gap"] for item in running])
                ),
                "mean_total_loss": float(
                    np.mean([item["total_loss"] for item in running])
                ),
                "last_domain": domain,
                "last_condition": condition,
                "last_wrong_condition": wrong_condition,
                "last_gradient_norm": float(gradient_norm),
                "last_graph_gate_mean": model.gate_mean(
                    matched_context.detach()
                ),
            }
            history.append(row)
            print(
                f"meta_step={step:04d}/{args.meta_steps} "
                f"query_loss={row['mean_query_loss']:.4f} "
                f"cf_loss={row['mean_counterfactual_rank_loss']:.4f} "
                f"error_gap={row['mean_wrong_minus_matched_abs_error']:.4f} "
                f"condition={condition}->{wrong_condition} "
                f"gate={row['last_graph_gate_mean']:.4f}"
            )
            running.clear()
    model.eval()
    return history


def load_or_train_source(
    args: argparse.Namespace,
    model: exp251.TrajectorySetGraphRegressor,
    source_data: dict[str, dict],
    prior: torch.Tensor,
    output: Path,
    device: torch.device,
) -> tuple[exp251.TrajectorySetGraphRegressor, list[dict]]:
    signature = source_signature(args, prior)
    cache = output / "source_cache" / (
        f"experiment25-2_source_{args.target}_modelseed{args.model_seed}.pt"
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
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            ),
        },
        cache,
    )
    return model, history


def self_check() -> None:
    target = torch.tensor([0.0, 0.0])
    matched = torch.tensor([0.2, 0.3])
    clearly_wrong = torch.tensor([2.0, 3.0])
    indistinguishable = matched.clone()
    good_loss, good_gap = counterfactual_rank_loss(
        matched, clearly_wrong, target, margin_rul=1.0
    )
    tied_loss, tied_gap = counterfactual_rank_loss(
        matched, indistinguishable, target, margin_rul=1.0
    )
    assert good_loss.item() == 0.0 and good_gap.item() > 1.0
    assert abs(tied_loss.item() - 1.0) < 1e-6 and tied_gap.item() == 0.0
    exp251.self_check()


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
        f"[objective] matched_mse + {args.counterfactual_weight:g} * "
        f"margin_rank_loss(margin={args.counterfactual_margin_rul:g} RUL)"
    )

    normalized, features, source_condition_counts = exp24.normalized_train_frames(
        cfg, PREPROCESSING
    )
    source_data = {
        domain: exp251.frame_windows(normalized[domain], features, cfg)
        for domain in cfg["source_domains"]
    }
    prior, _ = exp24.train_only_prior(cfg, PREPROCESSING, args.sensor_graph_k)
    model, initialization_checkpoint = exp251.build_model(
        args, cfg, prior, len(features)
    )
    model, source_history = load_or_train_source(
        args, model, source_data, prior, output, device
    )

    target = normalized[args.target]
    validation_units = set(map(int, protocol["validation_units"]))
    validation = exp251.frame_windows(
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
            support = exp251.frame_windows(
                target[target["unit"].isin(support_units)], features, cfg
            )
            run_seed = exp18.target_run_seed(args.model_seed, split_seed)
            for mode in MODES:
                learner, history, best_epoch, result = exp251.train_target_predictor(
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
                target_histories[f"split{split_seed}_k{k}_{mode}"] = history
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
    comparison = exp251.comparisons(raw)
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
        "source_meta_objective": "matched_mse_plus_counterfactual_margin_ranking",
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin_rul": args.counterfactual_margin_rul,
        "wrong_context_policy": "different_condition_same_domain_query_engine_disjoint",
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
            "expand_experiment25-2_to_seeds44_45_46"
            if pilot_success
            else "abandon_condition_context_branch_and_open_experiment26"
        ),
    }

    prefix = f"experiment25-2_{args.target}_seed{args.model_seed}"
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
