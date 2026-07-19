"""Experiment 17B: controlled and crossed validation of stable sensor graphs.

Experiment 17 showed that a source-correlation sensor graph strongly beat a
sensor-centric no-graph model, but two questions remained:

1. Was the difference from the original window graph caused by node semantics,
   or by a different temporal backbone?
2. Was the source-correlation topology useful, or was any sparse graph enough?

This script addresses both questions without modifying Experiment 17.  It uses
five controlled regimes:

``window_no_graph``
    Original LSTM/SE/self-attention backbone with its batch-window GAT removed.
``window_graph``
    Original LSTM/SE/self-attention backbone with the batch-window cosine graph.
``sensor_no_graph``
    Stable sensor-node backbone with identity adjacency only.
``sensor_graph_random``
    Stable sensor-node backbone with a connected degree-preserving random
    rewiring of the prior graph.  It has exactly the same node degrees and edge
    count as the prior graph.  Five pre-registered rewiring seeds are paired
    with the five model seeds, so the conclusion cannot depend on one lucky or
    unlucky random topology.
``sensor_graph_prior``
    Stable sensor-node backbone with the source-only correlation kNN graph.

The design crosses target-engine split seeds with model seeds.  Target split
seeds choose nested K-engine sets; model seeds independently control source
initialization and optimization.  A formal default run therefore contains

    5 target splits x 5 model seeds x 2 K values x 5 regimes = 250 cells.

The official test set is never evaluated by default.  This is still a graph
structure experiment, not a meta-learning experiment.

Run from the project root.

Dry run::

    python scripts/experiment17b_controlled_sensor_graph.py --target FD004 --dry-run

Formal validation run::

    python -u scripts/experiment17b_controlled_sensor_graph.py \
      --target FD004 \
      --k-values 2 5 \
      --target-split-seeds 3027 3028 3029 3030 3031 \
      --model-seeds 42 43 44 45 46 \
      --source-pretrain-steps 1500 \
      --target-epochs 10 \
      --evaluation-scope validation \
      --resume

The primary comparisons are:

* ``window_graph`` vs ``window_no_graph``: isolated value of the batch graph;
* ``sensor_graph_prior`` vs ``sensor_no_graph``: isolated value of sensor
  message passing;
* ``sensor_graph_prior`` vs ``sensor_graph_random``: semantic prior vs matched
  sparse regularization;
* ``sensor_graph_prior`` vs ``window_graph``: overall candidate comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_model  # noqa: E402
from preprocess.cmapps_loader import load_domain  # noqa: E402
from scripts import experiment17_sensor_graph_ablation as exp17  # noqa: E402
from scripts.experiment7_kshot_engines import (  # noqa: E402
    EXPECTED_OFFICIAL_TEST_ENGINES,
    atomic_write_text,
    prepare_kshot_experiment,
    resolve_device,
    resolve_path,
    seed_everything,
    target_unit_protocol,
)
from scripts.experiment8_transfer_baseline import train_source_supervised  # noqa: E402
from scripts.run_condition_aware_experiment import (  # noqa: E402
    BALANCE_MODES,
    PREPROCESSING_MODES,
)


SCRIPT_VERSION = "experiment17b_controlled_sensor_graph_v1"
MODEL_CHOICES = (
    "window_no_graph",
    "window_graph",
    "sensor_no_graph",
    "sensor_graph_random",
    "sensor_graph_prior",
)
METRICS = ("rmse", "mae", "r2", "nasa_score")
LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}
COMPARISONS = (
    ("window_graph", "window_no_graph", "window_graph_vs_window_no_graph"),
    ("sensor_graph_prior", "sensor_no_graph", "prior_vs_sensor_no_graph"),
    ("sensor_graph_prior", "sensor_graph_random", "prior_vs_random_sparse"),
    ("sensor_graph_prior", "window_graph", "prior_sensor_vs_window_graph"),
    ("sensor_no_graph", "window_no_graph", "sensor_backbone_vs_window_backbone"),
)
PRIMARY_COMPARISONS = {
    "prior_vs_sensor_no_graph",
    "prior_vs_random_sparse",
    "prior_sensor_vs_window_graph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验17B：同骨干图消融、等度随机图和目标划分×模型种子交叉验证"
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
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 5])
    parser.add_argument(
        "--target-split-seeds",
        nargs="+",
        type=int,
        default=[3027, 3028, 3029, 3030, 3031],
        help="只控制目标适应发动机顺序，与模型初始化种子独立",
    )
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="控制模型初始化、源训练和目标适应随机性",
    )
    parser.add_argument("--validation-units", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--normalizer-seed", type=int, default=2026)
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument(
        "--experiment17-protocol",
        help="可选：实验17固定划分协议；仅复用固定验证集和官方test审计信息",
    )
    parser.add_argument(
        "--preprocessing", choices=PREPROCESSING_MODES, default="condition_settings"
    )
    parser.add_argument(
        "--balance-mode", choices=BALANCE_MODES, default="engine_stage"
    )
    parser.add_argument("--sensor-graph-k", type=int, default=4)
    parser.add_argument(
        "--random-graph-seeds",
        nargs="+",
        type=int,
        default=[3017, 3018, 3019, 3020, 3021],
        help="与--model-seeds按位置配对的等度随机图种子，数量必须相同",
    )
    parser.add_argument(
        "--random-edge-swaps-multiplier",
        type=int,
        default=20,
        help="随机图目标成功换边次数=先验无向边数×该值",
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
    parser.add_argument("--confirm-official-test", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--output-dir", default="outputs/experiment17b_controlled_sensor_graph"
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
    # Keep graph structure as the only treatment difference.
    cfg["pair_aux_weight"] = 0.0
    data_dir = args.data_dir if args.data_dir is not None else cfg["data_dir"]
    cfg["data_dir"] = str(resolve_path(data_dir))
    cfg["output_dir"] = str(resolve_path(args.output_dir))
    return cfg


def default_experiment17_protocol(args: argparse.Namespace) -> Path:
    if args.experiment17_protocol:
        return resolve_path(args.experiment17_protocol)
    return (
        PROJECT_ROOT
        / "outputs"
        / "experiment17_sensor_graph_ablation"
        / f"experiment17_{args.target}_split_protocol.json"
    )


def load_or_create_base_protocol(args: argparse.Namespace, cfg: dict) -> tuple[dict, str | None]:
    path = default_experiment17_protocol(args)
    if path.is_file():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        print(f"[protocol] 读取实验17固定验证集：{path}")
        source_path: str | None = str(path)
    else:
        # Only its validation and official-test audit fields are reused below.
        protocol = target_unit_protocol(
            cfg["data_dir"],
            args.target,
            args.validation_units,
            args.validation_seed,
            args.model_seeds,
            args.k_values,
        )
        print("[protocol] 未找到实验17协议，按相同规则生成固定验证集。")
        source_path = None
    if protocol.get("target_domain") != args.target:
        raise ValueError("实验17协议target_domain与--target不一致")
    if len(protocol["validation_units"]) != args.validation_units:
        raise ValueError("协议固定验证发动机数量与--validation-units不一致")
    return protocol, source_path


def build_crossed_protocol(
    args: argparse.Namespace,
    cfg: dict,
    base: dict,
    source_path: str | None,
) -> dict:
    target_train, _, _ = load_domain(cfg["data_dir"], args.target)
    train_units = np.asarray(sorted(target_train["unit"].unique()), dtype=int)
    validation = np.asarray(base["validation_units"], dtype=int)
    validation_set = set(validation.tolist())
    candidates = np.asarray(
        [unit for unit in train_units if int(unit) not in validation_set], dtype=int
    )
    if max(args.k_values) > len(candidates):
        raise ValueError("最大K超过固定验证集之外的目标训练发动机数量")

    nested: dict[str, dict[str, list[int]]] = {}
    orders: dict[str, list[int]] = {}
    for split_seed in args.target_split_seeds:
        order = np.random.default_rng(split_seed).permutation(candidates)
        orders[str(split_seed)] = order.astype(int).tolist()
        nested[str(split_seed)] = {
            str(k): order[:k].astype(int).tolist() for k in args.k_values
        }
        previous: set[int] = set()
        for k in args.k_values:
            current = set(nested[str(split_seed)][str(k)])
            if len(current) != k:
                raise AssertionError("目标K发动机数量错误")
            if not previous.issubset(current):
                raise AssertionError("目标K集合没有严格嵌套")
            if current & validation_set:
                raise AssertionError("适应发动机与固定验证发动机重叠")
            previous = current

    return {
        "script_version": SCRIPT_VERSION,
        "target_domain": args.target,
        "source_protocol_path": source_path,
        "train_engine_count": int(len(train_units)),
        "validation_seed": int(base.get("validation_seed", args.validation_seed)),
        "validation_units": validation.astype(int).tolist(),
        "candidate_adaptation_engine_count": int(len(candidates)),
        "target_split_seeds": list(args.target_split_seeds),
        "model_seeds": list(args.model_seeds),
        "k_values": list(args.k_values),
        "adaptation_order_by_target_split_seed": orders,
        "nested_adaptation_units_by_target_split_seed": nested,
        "official_test_engine_count": int(base["official_test_engine_count"]),
        "official_test_units": [int(unit) for unit in base["official_test_units"]],
        "official_test_units_hash": base["official_test_units_hash"],
    }


def protocol_rows(protocol: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for unit in protocol["validation_units"]:
        rows.append(
            {
                "target_split_seed": "fixed",
                "k": "all",
                "role": "validation",
                "unit": int(unit),
            }
        )
    for split_seed, by_k in protocol[
        "nested_adaptation_units_by_target_split_seed"
    ].items():
        for k, units in by_k.items():
            for unit in units:
                rows.append(
                    {
                        "target_split_seed": int(split_seed),
                        "k": int(k),
                        "role": "adaptation",
                        "unit": int(unit),
                    }
                )
    for unit in protocol["official_test_units"]:
        rows.append(
            {
                "target_split_seed": "fixed",
                "k": "all",
                "role": "official_test",
                "unit": int(unit),
            }
        )
    return pd.DataFrame(rows)


def edge_set(adjacency: torch.Tensor) -> set[tuple[int, int]]:
    matrix = adjacency.detach().cpu().numpy().astype(bool)
    return {
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[1])
        if matrix[i, j]
    }


def adjacency_from_edges(nodes: int, edges: set[tuple[int, int]]) -> torch.Tensor:
    matrix = np.eye(nodes, dtype=bool)
    for left, right in edges:
        matrix[left, right] = True
        matrix[right, left] = True
    return torch.as_tensor(matrix)


def graph_connected(nodes: int, edges: set[tuple[int, int]]) -> bool:
    neighbors = {node: set() for node in range(nodes)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == nodes


def degree_preserving_random_graph(
    prior: torch.Tensor,
    seed: int,
    swaps_multiplier: int,
) -> tuple[torch.Tensor, dict]:
    """Connected double-edge swaps preserve every node degree and edge count."""
    nodes = prior.shape[0]
    original = edge_set(prior)
    current = set(original)
    rng = random.Random(seed)
    target_swaps = max(1, swaps_multiplier * len(original))
    successful = 0
    attempts = 0
    max_attempts = max(1000, target_swaps * 100)
    while successful < target_swaps and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(tuple(current), 2)
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            proposed = {
                tuple(sorted((a, d))),
                tuple(sorted((c, b))),
            }
        else:
            proposed = {
                tuple(sorted((a, c))),
                tuple(sorted((b, d))),
            }
        if len(proposed) != 2 or any(x == y for x, y in proposed):
            continue
        remaining = current - {first, second}
        if proposed & remaining:
            continue
        candidate = remaining | proposed
        if not graph_connected(nodes, candidate):
            continue
        current = candidate
        successful += 1

    randomized = adjacency_from_edges(nodes, current)
    prior_degrees = prior.to(torch.int64).sum(dim=1) - 1
    random_degrees = randomized.to(torch.int64).sum(dim=1) - 1
    if not torch.equal(prior_degrees, random_degrees):
        raise AssertionError("等度随机图没有保持节点度数")
    if len(current) != len(original):
        raise AssertionError("等度随机图没有保持边数")
    intersection = len(original & current)
    union = len(original | current)
    audit = {
        "random_graph_seed": int(seed),
        "requested_successful_swaps": int(target_swaps),
        "successful_swaps": int(successful),
        "attempts": int(attempts),
        "undirected_edge_count": int(len(current)),
        "degree_sequence": prior_degrees.tolist(),
        "connected": graph_connected(nodes, current),
        "prior_random_edge_jaccard": float(intersection / union),
        "changed_edge_fraction": float(1.0 - intersection / len(original)),
    }
    if successful < target_swaps // 2:
        raise RuntimeError("等度随机图有效换边次数过少，请更换--random-graph-seed")
    return randomized, audit


def build_model_17b(
    model_name: str,
    feature_count: int,
    cfg: dict,
    prior_adjacency: torch.Tensor,
    random_adjacency: torch.Tensor,
) -> nn.Module:
    if model_name == "window_no_graph":
        # Build the full window model first and copy every shape-compatible
        # parameter into the no-GAT control.  Because callers reset the same
        # model seed before every regime, all shared LSTM/SE/attention/head
        # parameters start identically; the GAT is the only treatment change.
        reference = build_model("gnn", feature_count, cfg)
        control = build_model("no_gat", feature_count, cfg)
        reference_state = reference.state_dict()
        control_state = control.state_dict()
        for name, value in control_state.items():
            if name in reference_state and reference_state[name].shape == value.shape:
                control_state[name] = reference_state[name].detach().clone()
        control.load_state_dict(control_state)
        return control
    if model_name == "window_graph":
        return build_model("gnn", feature_count, cfg)
    if model_name == "sensor_no_graph":
        return exp17.build_ablation_model(
            "no_graph", feature_count, cfg, prior_adjacency
        )
    if model_name == "sensor_graph_prior":
        return exp17.build_ablation_model(
            "sensor_graph_prior", feature_count, cfg, prior_adjacency
        )
    if model_name == "sensor_graph_random":
        sensor_count = len(cfg["sensor_columns"])
        return exp17.StableSensorGraphRegressor(
            sensor_count=sensor_count,
            condition_dim=feature_count - sensor_count,
            embedding_dim=int(cfg["embedding_dim"]),
            heads=int(cfg["gat_heads"]),
            dropout=float(cfg["dropout"]),
            adjacency=random_adjacency,
            learnable_edge_bias=False,
        )
    raise ValueError(f"未知实验17B模型：{model_name}")


def source_signature(
    args: argparse.Namespace,
    cfg: dict,
    model_name: str,
    feature_count: int,
    prior: torch.Tensor,
    randomized: torch.Tensor,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "model": model_name,
        "model_seed": cfg["seed"],
        "target": cfg["target_domain"],
        "source_domains": cfg["source_domains"],
        "feature_count": feature_count,
        "embedding_dim": cfg["embedding_dim"],
        "gat_heads": cfg["gat_heads"],
        "dropout": cfg["dropout"],
        "preprocessing": args.preprocessing,
        "balance_mode": args.balance_mode,
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "source_pretrain_lr": cfg["source_pretrain_lr"],
        "source_pretrain_weight_decay": cfg["source_pretrain_weight_decay"],
        "prior_hash": hashlib.sha256(prior.numpy().tobytes()).hexdigest()[:16],
        "random_hash": hashlib.sha256(randomized.numpy().tobytes()).hexdigest()[:16],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def load_or_train_source(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    model_name: str,
    prior: torch.Tensor,
    randomized: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
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
    seed_everything(cfg["seed"])
    model = build_model_17b(
        model_name, feature_count, cfg, prior, randomized
    )
    total, predictor = exp17.parameter_count(model)
    signature = source_signature(
        args, cfg, model_name, feature_count, prior, randomized
    )
    cache_path = (
        Path(cfg["output_dir"])
        / "source_cache"
        / f"experiment17b_{model_name}_{cfg['target_domain']}_modelseed{cfg['seed']}.pt"
    )
    if args.resume and cache_path.is_file():
        cached = exp17.safe_torch_load(cache_path)
        if cached.get("signature") == signature:
            print(f"[source cache] {cache_path}")
            return cached["state"], cached.get("history", []), cached["inventory"]
        print(f"[source cache ignored] 签名不匹配：{cache_path}")

    device = resolve_device(cfg["device"])
    model, history = train_source_supervised(model, source_tasks, cfg, device)
    state = exp17.state_to_cpu(model)
    inventory = {
        "model": model_name,
        "model_seed": cfg["seed"],
        "feature_count": feature_count,
        "total_parameter_count": total,
        "predictor_parameter_count": predictor,
        "source_pretrain_steps": cfg["source_pretrain_steps"],
        "source_pretrain_lr": cfg["source_pretrain_lr"],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
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


def target_run_seed(model_seed: int, target_split_seed: int) -> int:
    payload = f"{model_seed}:{target_split_seed}:experiment17b".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def run_target_cell(
    args: argparse.Namespace,
    source_cfg: dict,
    protocol: dict,
    model_name: str,
    source_state: dict[str, torch.Tensor],
    source_history: list[dict],
    inventory: dict,
    target_split_seed: int,
    k: int,
    prior: torch.Tensor,
    randomized: torch.Tensor,
) -> dict:
    model_seed = int(source_cfg["seed"])
    units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(target_split_seed)
    ][str(k)]
    run_seed = target_run_seed(model_seed, target_split_seed)
    target_cfg = dict(source_cfg)
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
    model = build_model_17b(
        model_name, feature_count, source_cfg, prior, randomized
    )
    model.load_state_dict(source_state)
    device = resolve_device(source_cfg["device"])
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
    result = {
        **selected_metrics,
        "evaluation_scope": args.evaluation_scope,
        "model": model_name,
        "graph_node_semantics": (
            "batch_windows" if model_name.startswith("window_") else "fixed_sensors"
        ),
        "source_training": "ordinary_multisource_pretraining",
        "target_adaptation_scope": "predictor_only",
        "target_domain": source_cfg["target_domain"],
        "model_seed": model_seed,
        "target_split_seed": int(target_split_seed),
        "target_run_seed": int(run_seed),
        "replicate_id": (
            f"split{target_split_seed}_model{model_seed}_k{k}_{model_name}"
        ),
        "k": int(k),
        "adaptation_units": [int(unit) for unit in units],
        "adaptation_engine_count": len(units),
        "validation_engine_count": len(protocol["validation_units"]),
        "official_test_engine_count": len(test.dataset),
        "official_test_units_hash": split["official_test_units_hash"],
        "official_test_forward_run": args.evaluation_scope == "official_test",
        "official_test_metrics": official_metrics,
        "best_target_epoch_by_validation": int(best_epoch),
        "target_epochs_planned": target_cfg["target_epochs"],
        "target_learning_rate": target_cfg["target_lr"],
        "source_pretrain_steps": source_cfg["source_pretrain_steps"],
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
        directory = Path(source_cfg["output_dir"]) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"experiment17b_{model_name}_k{k}_{source_cfg['target_domain']}_"
            f"split{target_split_seed}_model{model_seed}.pt"
        )
        torch.save(
            {
                "model": exp17.state_to_cpu(model),
                "metrics": result,
                "history": history,
                "split": split,
            },
            path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows: list[dict] = []
    for (k, model), group in frame.groupby(["k", "model"]):
        row = {
            "k": int(k),
            "model": model,
            "n_cells": int(len(group)),
            "n_target_splits": int(group["target_split_seed"].nunique()),
            "n_model_seeds": int(group["model_seed"].nunique()),
            "evaluation_scope": group["evaluation_scope"].iloc[0],
            "total_parameter_count": int(group["total_parameter_count"].iloc[0]),
            "target_trainable_parameter_count": int(
                group["target_trainable_parameter_count"].iloc[0]
            ),
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
    rows: list[dict] = []
    for (k, split_seed, model_seed), group in frame.groupby(
        ["k", "target_split_seed", "model_seed"]
    ):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        for candidate, reference, comparison in COMPARISONS:
            if candidate not in by_model or reference not in by_model:
                continue
            candidate_row = by_model[candidate]
            reference_row = by_model[reference]
            row = {
                "k": int(k),
                "target_split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
            }
            for metric in METRICS:
                delta = float(candidate_row[metric] - reference_row[metric])
                row[f"{metric}_delta_candidate_minus_reference"] = delta
                row[f"{metric}_candidate_win"] = float(
                    delta < 0 if metric in LOWER_IS_BETTER else delta > 0
                )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["k", "comparison", "target_split_seed", "model_seed"]
    )


def paired_by_target_split(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in paired.columns
        if column.endswith("_delta_candidate_minus_reference")
    ]
    grouped = (
        paired.groupby(
            ["k", "target_split_seed", "comparison", "candidate", "reference"],
            as_index=False,
        )[numeric]
        .mean()
        .sort_values(["k", "comparison", "target_split_seed"])
    )
    grouped["rmse_split_win"] = (
        grouped["rmse_delta_candidate_minus_reference"] < 0
    ).astype(float)
    grouped["nasa_score_split_win"] = (
        grouped["nasa_score_delta_candidate_minus_reference"] < 0
    ).astype(float)
    return grouped


def hierarchical_bootstrap_ci(
    group: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    matrix = group.pivot(
        index="target_split_seed", columns="model_seed", values=value_column
    ).sort_index().sort_index(axis=1)
    if matrix.empty or bool(matrix.isna().any().any()):
        return float("nan"), float("nan")
    values = matrix.to_numpy(dtype=float)
    n_splits, n_models = values.shape
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        split_indices = rng.integers(0, n_splits, size=n_splits)
        model_indices = rng.integers(0, n_models, size=n_models)
        bootstrap[index] = values[np.ix_(split_indices, model_indices)].mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def split_level_pvalue(values: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_1samp

        return float(ttest_1samp(values, popmean=0.0).pvalue)
    except Exception:
        return float("nan")


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        current = min(1.0, (count - rank) * value)
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def comparison_summary(
    results: list[dict],
    paired: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    raw = pd.DataFrame(results)
    rows: list[dict] = []
    for (k, comparison), group in paired.groupby(["k", "comparison"]):
        candidate = group["candidate"].iloc[0]
        reference = group["reference"].iloc[0]
        split_means = group.groupby("target_split_seed")[
            "rmse_delta_candidate_minus_reference"
        ].mean()
        nasa_split_means = group.groupby("target_split_seed")[
            "nasa_score_delta_candidate_minus_reference"
        ].mean()
        low, high = hierarchical_bootstrap_ci(
            group,
            "rmse_delta_candidate_minus_reference",
            repetitions,
            seed=17017 + int(k) + sum(map(ord, comparison)),
        )
        reference_rmse = raw[
            (raw["k"] == k) & (raw["model"] == reference)
        ]["rmse"].mean()
        improvement = float(
            -100.0
            * group["rmse_delta_candidate_minus_reference"].mean()
            / reference_rmse
        )
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
                "rmse_improvement_pct": improvement,
                "rmse_cell_win_rate": float(group["rmse_candidate_win"].mean()),
                "rmse_target_split_win_rate": float((split_means < 0).mean()),
                "rmse_hier_boot_ci95_low": low,
                "rmse_hier_boot_ci95_high": high,
                "rmse_split_t_p": split_level_pvalue(split_means.to_numpy(float)),
                "mae_delta_mean": float(
                    group["mae_delta_candidate_minus_reference"].mean()
                ),
                "r2_delta_mean": float(
                    group["r2_delta_candidate_minus_reference"].mean()
                ),
                "nasa_score_delta_mean": float(
                    group["nasa_score_delta_candidate_minus_reference"].mean()
                ),
                "nasa_score_target_split_win_rate": float(
                    (nasa_split_means < 0).mean()
                ),
                "primary_comparison": comparison in PRIMARY_COMPARISONS,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["k", "comparison"]).reset_index(drop=True)
    frame["rmse_split_t_p_holm"] = holm_adjust(frame["rmse_split_t_p"].tolist())
    frame["strict_success"] = (
        (frame["rmse_improvement_pct"] >= 3.0)
        & (frame["rmse_target_split_win_rate"] >= 0.8)
        & (frame["rmse_hier_boot_ci95_high"] < 0.0)
        & (frame["rmse_split_t_p_holm"] < 0.05)
        & (frame["nasa_score_delta_mean"] <= 0.0)
    )
    return frame


def result_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolve_path(args.output_dir)
    prefix = f"experiment17b_{args.target}"
    return {
        "output": output,
        "raw": output / f"{prefix}_raw.json",
        "summary": output / f"{prefix}_summary.csv",
        "paired_cell": output / f"{prefix}_paired_by_cell.csv",
        "paired_split": output / f"{prefix}_paired_by_target_split.csv",
        "comparisons": output / f"{prefix}_comparisons.csv",
        "protocol": output / f"{prefix}_protocol.json",
        "engine_splits": output / f"{prefix}_engine_splits.csv",
        "prior_adjacency": output / f"{prefix}_prior_adjacency.csv",
        "random_adjacency": output / f"{prefix}_random_adjacencies.csv",
        "correlation": output / f"{prefix}_source_sensor_correlation.csv",
        "graph_audit": output / f"{prefix}_graph_control_audit.json",
        "inventory": output / f"{prefix}_model_inventory.csv",
        "budget": output / f"{prefix}_budget.json",
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
    paired = paired_cells(results)
    split = paired_by_target_split(paired)
    comparisons = comparison_summary(
        results, paired, args.bootstrap_repetitions
    )
    atomic_write_text(
        paths["paired_cell"], paired.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["paired_split"], split.to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["comparisons"], comparisons.to_csv(index=False), encoding="utf-8-sig"
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


def dry_run(
    args: argparse.Namespace,
    cfg: dict,
    protocol: dict,
    prior: torch.Tensor,
    randomized: torch.Tensor,
) -> pd.DataFrame:
    split_seed = args.target_split_seeds[0]
    k = min(args.k_values)
    units = protocol["nested_adaptation_units_by_target_split_seed"][str(split_seed)][
        str(k)
    ]
    run_cfg = dict(cfg)
    run_cfg["seed"] = target_run_seed(cfg["seed"], split_seed)
    loaders = prepare_kshot_experiment(
        run_cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks, support, validation, test, feature_count, _ = loaders
    x, _ = next(iter(source_tasks[cfg["source_domains"][0]]))
    rows: list[dict] = []
    for model_name in args.models:
        seed_everything(cfg["seed"])
        model = build_model_17b(
            model_name, feature_count, cfg, prior, randomized
        ).eval()
        with torch.no_grad():
            prediction = model(x[: min(8, len(x))])
        total, predictor = exp17.parameter_count(model)
        rows.append(
            {
                "model": model_name,
                "feature_count": feature_count,
                "total_parameter_count": total,
                "predictor_parameter_count": predictor,
                "forward_shape": list(prediction.shape),
                "finite_output": bool(torch.isfinite(prediction).all()),
            }
        )
    print(
        json.dumps(
            {
                "target_split_seed": split_seed,
                "model_seed": cfg["seed"],
                "k": k,
                "adaptation_units": units,
                "source_batch_shape": list(x.shape),
                "support_windows": len(support.dataset),
                "validation_windows": len(validation.dataset),
                "official_test_engine_count": len(test.dataset),
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
    args.target_split_seeds = list(
        dict.fromkeys(int(value) for value in args.target_split_seeds)
    )
    args.model_seeds = list(dict.fromkeys(int(value) for value in args.model_seeds))
    args.random_graph_seeds = [int(value) for value in args.random_graph_seeds]
    args.models = list(dict.fromkeys(args.models))
    if not args.k_values or any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values必须全部为正整数")
    if not args.target_split_seeds or not args.model_seeds:
        raise ValueError("目标划分种子和模型种子不能为空")
    if len(args.random_graph_seeds) != len(args.model_seeds):
        raise ValueError("--random-graph-seeds数量必须与--model-seeds完全相同")
    if args.evaluation_scope == "official_test" and not args.confirm_official_test:
        raise ValueError("官方test锁定；必须显式提供--confirm-official-test")
    if (
        (len(args.target_split_seeds) < 5 or len(args.model_seeds) < 5)
        and not args.dry_run
    ):
        print("[警告] 少于5×5交叉种子，只能视为预实验。")

    first_cfg = load_config(args, args.model_seeds[0])
    base_protocol, source_path = load_or_create_base_protocol(args, first_cfg)
    protocol = build_crossed_protocol(args, first_cfg, base_protocol, source_path)
    expected = EXPECTED_OFFICIAL_TEST_ENGINES[args.target]
    if (
        protocol["official_test_engine_count"] != expected
        and not args.skip_official_count_check
    ):
        raise ValueError(
            f"{args.target}官方测试发动机应为{expected}台，"
            f"当前协议为{protocol['official_test_engine_count']}台"
        )

    prior, correlation = exp17.source_correlation_adjacency(
        first_cfg, args.preprocessing, args.sensor_graph_k
    )
    random_seed_by_model_seed = dict(zip(args.model_seeds, args.random_graph_seeds))
    randomized_by_model_seed: dict[int, torch.Tensor] = {}
    random_audits: list[dict] = []
    for model_seed in args.model_seeds:
        randomized, audit = degree_preserving_random_graph(
            prior,
            random_seed_by_model_seed[model_seed],
            args.random_edge_swaps_multiplier,
        )
        randomized_by_model_seed[model_seed] = randomized
        random_audits.append({"model_seed": model_seed, **audit})
    paths = result_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    sensors = list(first_cfg["sensor_columns"])
    atomic_write_text(
        paths["protocol"], json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    atomic_write_text(
        paths["engine_splits"], protocol_rows(protocol).to_csv(index=False), encoding="utf-8-sig"
    )
    atomic_write_text(
        paths["prior_adjacency"],
        pd.DataFrame(prior.numpy().astype(int), index=sensors, columns=sensors).to_csv(),
        encoding="utf-8-sig",
    )
    random_adjacency_rows: list[dict] = []
    for model_seed, randomized in randomized_by_model_seed.items():
        for row_index, source_sensor in enumerate(sensors):
            for column_index, target_sensor in enumerate(sensors):
                random_adjacency_rows.append(
                    {
                        "model_seed": model_seed,
                        "random_graph_seed": random_seed_by_model_seed[model_seed],
                        "source_sensor": source_sensor,
                        "target_sensor": target_sensor,
                        "adjacent": int(randomized[row_index, column_index]),
                    }
                )
    atomic_write_text(
        paths["random_adjacency"],
        pd.DataFrame(random_adjacency_rows).to_csv(index=False),
        encoding="utf-8-sig",
    )
    atomic_write_text(
        paths["correlation"],
        pd.DataFrame(correlation, index=sensors, columns=sensors).to_csv(),
        encoding="utf-8-sig",
    )
    graph_audit = {
        "script_version": SCRIPT_VERSION,
        "sensor_nodes": sensors,
        "sensor_graph_k": args.sensor_graph_k,
        "prior_edge_count": len(edge_set(prior)),
        "prior_degree_sequence": (prior.to(torch.int64).sum(dim=1) - 1).tolist(),
        "random_graph_seed_by_model_seed": random_seed_by_model_seed,
        "random_graph_audits": random_audits,
    }
    atomic_write_text(
        paths["graph_audit"], json.dumps(graph_audit, ensure_ascii=False, indent=2)
    )
    planned_cells = (
        len(args.target_split_seeds)
        * len(args.model_seeds)
        * len(args.k_values)
        * len(args.models)
    )
    budget = {
        "script_version": SCRIPT_VERSION,
        "target": args.target,
        "models": args.models,
        "k_values": args.k_values,
        "target_split_seeds": args.target_split_seeds,
        "model_seeds": args.model_seeds,
        "random_graph_seeds": args.random_graph_seeds,
        "random_graph_seed_by_model_seed": random_seed_by_model_seed,
        "planned_target_cells": planned_cells,
        "source_states": len(args.model_seeds) * len(args.models),
        "source_pretrain_steps_per_state": args.source_pretrain_steps,
        "target_epochs_per_cell": args.target_epochs,
        "target_adaptation_scope": "predictor_only",
        "pair_aux_weight": 0.0,
        "shared_parameter_initialization_matched_within_backbone": True,
        "statistics_high_level_unit": "target_split_seed",
        "official_test_policy": "validation_only unless explicitly confirmed after locking",
    }
    atomic_write_text(paths["budget"], json.dumps(budget, ensure_ascii=False, indent=2))

    print("\n[实验17B交叉协议与预算]")
    print(json.dumps({**budget, "fixed_validation_units": protocol["validation_units"]}, ensure_ascii=False, indent=2))
    print("\n[等度随机图审计]")
    print(json.dumps(graph_audit, ensure_ascii=False, indent=2))

    if args.dry_run:
        inventory = dry_run(
            args,
            first_cfg,
            protocol,
            prior,
            randomized_by_model_seed[first_cfg["seed"]],
        )
        atomic_write_text(
            paths["inventory"], inventory.to_csv(index=False), encoding="utf-8-sig"
        )
        print("\n[dry-run完成] 交叉协议、随机图控制和模型前向检查通过，未训练。")
        for key in (
            "protocol",
            "engine_splits",
            "prior_adjacency",
            "random_adjacency",
            "graph_audit",
            "inventory",
            "budget",
        ):
            print(f"{key}: {paths[key]}")
        return

    results: list[dict] = []
    if args.resume and paths["raw"].is_file():
        results = json.loads(paths["raw"].read_text(encoding="utf-8"))
        print(f"[resume] 已读取{len(results)}条结果。")
    done = completed_keys(results)
    inventories: list[dict] = []
    if args.resume and paths["inventory"].is_file():
        inventories = pd.read_csv(paths["inventory"]).to_dict(orient="records")

    for model_seed in args.model_seeds:
        cfg = load_config(args, model_seed)
        randomized = randomized_by_model_seed[model_seed]
        expected_for_seed = {
            (split_seed, model_seed, k, model_name)
            for split_seed in args.target_split_seeds
            for k in args.k_values
            for model_name in args.models
        }
        if expected_for_seed.issubset(done):
            print(f"[skip model seed] model_seed={model_seed}已全部完成。")
            continue
        states: dict[str, dict[str, torch.Tensor]] = {}
        histories: dict[str, list[dict]] = {}
        seed_inventory: dict[str, dict] = {}
        for model_name in args.models:
            pending = any(
                (split_seed, model_seed, k, model_name) not in done
                for split_seed in args.target_split_seeds
                for k in args.k_values
            )
            if not pending:
                continue
            print(f"\n[source initialization] model_seed={model_seed} model={model_name}")
            state, history, inventory = load_or_train_source(
                args, cfg, protocol, model_name, prior, randomized
            )
            states[model_name] = state
            histories[model_name] = history
            seed_inventory[model_name] = inventory
            inventories.append(inventory)

        for split_seed in args.target_split_seeds:
            for k in args.k_values:
                units = protocol["nested_adaptation_units_by_target_split_seed"][
                    str(split_seed)
                ][str(k)]
                for model_name in args.models:
                    key = (split_seed, model_seed, k, model_name)
                    if key in done:
                        print(
                            f"[skip] split={split_seed} model_seed={model_seed} "
                            f"K={k} model={model_name}"
                        )
                        continue
                    print(
                        f"\n[experiment17B] split={split_seed} model_seed={model_seed} "
                        f"K={k} model={model_name} engines={units}"
                    )
                    result = run_target_cell(
                        args,
                        cfg,
                        protocol,
                        model_name,
                        states[model_name],
                        histories[model_name],
                        seed_inventory[model_name],
                        split_seed,
                        k,
                        prior,
                        randomized,
                    )
                    results.append(result)
                    done.add(key)
                    save_progress(args, paths, results)
        states.clear()
        histories.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    inventory_frame = pd.DataFrame(inventories).drop_duplicates(
        subset=["model", "model_seed"]
    )
    atomic_write_text(
        paths["inventory"], inventory_frame.to_csv(index=False), encoding="utf-8-sig"
    )
    save_progress(args, paths, results)
    summary = summarize(results)
    paired = paired_cells(results)
    comparisons = comparison_summary(results, paired, args.bootstrap_repetitions)
    print("\n[实验17B汇总]")
    print(summary.to_string(index=False))
    print("\n[实验17B主要比较]")
    print(comparisons.to_string(index=False))
    print(
        "\n[结论判定]\n"
        "1. window_graph_vs_window_no_graph检验原batch窗口图在同一骨干内是否有贡献。\n"
        "2. prior_vs_sensor_no_graph检验传感器消息传播在同一骨干内是否有贡献。\n"
        "3. prior_vs_random_sparse检验相关性拓扑是否优于等边数、等度数稀疏控制。\n"
        "4. prior_sensor_vs_window_graph是总体候选比较，但必须结合前三项因果消融解释。\n"
        "5. strict_success要求改善≥3%、划分胜率≥80%、分层CI上界<0、Holm p<0.05且NASA不恶化。\n"
        "6. 本实验仍不证明元学习；验证通过后才进入任务条件化传感器图。"
    )
    for key in (
        "raw",
        "summary",
        "paired_cell",
        "paired_split",
        "comparisons",
        "protocol",
        "graph_audit",
        "inventory",
        "budget",
    ):
        print(f"{key}: {paths[key]}")


if __name__ == "__main__":
    main()
