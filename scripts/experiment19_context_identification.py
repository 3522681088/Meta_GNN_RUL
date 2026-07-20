"""Experiment 19: identify whether TCSG gains come from support context.

Reuses Experiment 18 training and evaluation code.  The only interventions are:
1. context from the least-overlapping alternative support split;
2. a zero context;
3. the true context with gate multipliers 0, 0.5, 2, and 4.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment18_task_conditioned_sensor_graph as exp18


SCRIPT_VERSION = "experiment19_context_identification_v1"
MODELS = (
    "static_budget_prior",
    "tcsg_true",
    "tcsg_other_split",
    "tcsg_zero",
    "tcsg_gate0",
    "tcsg_gate05",
    "tcsg_gate2",
    "tcsg_gate4",
)
COMPARISONS = (
    ("tcsg_true", "static_budget_prior", "tcsg_true_vs_static_budget"),
    ("tcsg_true", "tcsg_other_split", "true_vs_other_split"),
    ("tcsg_true", "tcsg_zero", "true_vs_zero_context"),
    ("tcsg_true", "tcsg_gate0", "true_vs_gate0"),
    ("tcsg_true", "tcsg_gate05", "true_vs_gate05"),
    ("tcsg_true", "tcsg_gate2", "true_vs_gate2"),
    ("tcsg_true", "tcsg_gate4", "true_vs_gate4"),
)
PRIMARY_COMPARISONS = {
    "tcsg_true_vs_static_budget",
    "true_vs_other_split",
    "true_vs_zero_context",
    "true_vs_gate0",
}
GATE_SCALE = {
    "tcsg_true": 1.0,
    "tcsg_other_split": 1.0,
    "tcsg_zero": 1.0,
    "tcsg_gate0": 0.0,
    "tcsg_gate05": 0.5,
    "tcsg_gate2": 2.0,
    "tcsg_gate4": 4.0,
}

_original_parse_args = exp18.parse_args
_original_result_paths = exp18.result_paths
_original_build_tcsg_model = exp18.build_tcsg_model
_original_run_task = exp18.run_target_cell
_original_context_for_mode = exp18.context_for_mode
_original_load_bundle = exp18.load_or_train_source_bundle
_active_gate_scale = 1.0
_summary_cache: dict[tuple[str, int], torch.Tensor] = {}


def parse_args():
    args = _original_parse_args()
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = "outputs/experiment19_context_identification"
    return args


def result_paths(args):
    paths = _original_result_paths(args)
    output = paths["output"]
    for key, path in tuple(paths.items()):
        if key != "output":
            paths[key] = output / path.name.replace("experiment18_", "experiment19_", 1)
    return paths


def build_tcsg_model(*args, **kwargs):
    model = _original_build_tcsg_model(*args, **kwargs)
    for layer in model.graph_layers:
        layer.max_gate *= _active_gate_scale
    return model


def _other_split(args, protocol, target_split_seed: int, k: int) -> int:
    current = set(
        protocol["nested_adaptation_units_by_target_split_seed"][
            str(target_split_seed)
        ][str(k)]
    )
    candidates = [seed for seed in args.target_split_seeds if seed != target_split_seed]
    if not candidates:
        raise ValueError("tcsg_other_split requires at least two target split seeds")
    return min(
        candidates,
        key=lambda seed: (
            len(
                current
                & set(
                    protocol["nested_adaptation_units_by_target_split_seed"][
                        str(seed)
                    ][str(k)]
                )
            ),
            candidates.index(seed),
        ),
    )


def _other_summary(args, cfg, protocol, target_split_seed: int, k: int):
    donor_seed = _other_split(args, protocol, target_split_seed, k)
    key = (str(donor_seed), int(k))
    if key not in _summary_cache:
        donor_units = protocol["nested_adaptation_units_by_target_split_seed"][
            str(donor_seed)
        ][str(k)]
        donor_cfg = dict(cfg)
        donor_cfg["seed"] = exp18.target_run_seed(cfg["seed"], donor_seed)
        loaders = exp18.prepare_kshot_experiment(
            donor_cfg,
            args.preprocessing,
            args.balance_mode,
            protocol["validation_units"],
            donor_units,
        )
        support = loaders[1]
        _summary_cache[key] = exp18.support_summary(
            support.dataset.x,
            support.dataset.y,
            len(cfg["sensor_columns"]),
            cfg["rul_cap"],
        )
        del loaders
    current = set(
        protocol["nested_adaptation_units_by_target_split_seed"][
            str(target_split_seed)
        ][str(k)]
    )
    donor = set(
        protocol["nested_adaptation_units_by_target_split_seed"][
            str(donor_seed)
        ][str(k)]
    )
    return _summary_cache[key], donor_seed, len(current & donor)


def run_target_cell(
    args,
    cfg,
    protocol,
    regime,
    state,
    inventory,
    target_split_seed,
    k,
    prior,
):
    global _active_gate_scale
    donor_seed = None
    overlap = None
    previous_context = exp18.context_for_mode
    gate_scale = GATE_SCALE.get(regime)
    try:
        _active_gate_scale = 1.0 if gate_scale is None else gate_scale
        if regime == "tcsg_other_split":
            summary, donor_seed, overlap = _other_summary(
                args, cfg, protocol, target_split_seed, k
            )
            exp18.context_for_mode = (
                lambda *unused_args, **unused_kwargs: summary.clone()
            )
        result, audit = _original_run_task(
            args,
            cfg,
            protocol,
            regime,
            state,
            inventory,
            target_split_seed,
            k,
            prior,
        )
    finally:
        exp18.context_for_mode = previous_context
        _active_gate_scale = 1.0
    extra = {
        "gate_scale": gate_scale,
        "context_donor_split_seed": donor_seed,
        "context_donor_unit_overlap": overlap,
    }
    result.update(extra)
    audit.update(extra)
    return result, audit


def load_or_train_bundle(args, cfg, protocol, prior):
    cache_dir = (
        exp18.PROJECT_ROOT
        / "outputs"
        / "experiment18_task_conditioned_sensor_graph"
    )
    cache = (
        cache_dir
        / "checkpoints"
        / f"experiment18_source_bundle_{args.target}_modelseed{cfg['seed']}.pt"
    )
    bundle_cfg = dict(cfg)
    if cache.is_file():
        bundle_cfg["output_dir"] = str(cache_dir)
    return _original_load_bundle(args, bundle_cfg, protocol, prior)


def self_check() -> None:
    assert set(MODELS) - {"static_budget_prior"} == set(GATE_SCALE)
    assert all(candidate in MODELS for candidate, _, _ in COMPARISONS)
    assert all(reference in MODELS for _, reference, _ in COMPARISONS)
    assert PRIMARY_COMPARISONS <= {name for _, _, name in COMPARISONS}


def main() -> None:
    self_check()
    exp18.SCRIPT_VERSION = SCRIPT_VERSION
    exp18.MODEL_CHOICES = MODELS
    exp18.DEFAULT_MODELS = list(MODELS)
    exp18.COMPARISONS = COMPARISONS
    exp18.PRIMARY_COMPARISONS = PRIMARY_COMPARISONS
    exp18.parse_args = parse_args
    exp18.result_paths = result_paths
    exp18.build_tcsg_model = build_tcsg_model
    exp18.run_target_cell = run_target_cell
    exp18.load_or_train_source_bundle = load_or_train_bundle
    exp18.main()


if __name__ == "__main__":
    main()
