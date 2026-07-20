"""Experiment 20: task-context source and random-context ablations.

This runner reuses Experiment 18's protocol, optimizer, target adaptation,
and statistical summaries. It compares support context with source-global and
same-norm random context while keeping the graph gate fixed at 1 or 2.
"""

from __future__ import annotations

import builtins
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment18_task_conditioned_sensor_graph as exp18


SCRIPT_VERSION = "experiment20_context_source_global_v1"
MODELS = (
    "static_budget_prior",
    "tcsg_true_gate1",
    "tcsg_true_gate2",
    "tcsg_zero_gate2",
    "tcsg_global_source_gate2",
    "tcsg_random_gate2",
)
GATE_SCALE = {
    "tcsg_true_gate1": 1.0,
    "tcsg_true_gate2": 2.0,
    "tcsg_zero_gate2": 2.0,
    "tcsg_global_source_gate2": 2.0,
    "tcsg_random_gate2": 2.0,
}
COMPARISONS = (
    ("tcsg_true_gate2", "static_budget_prior", "tcsg_true_vs_static_budget"),
    ("tcsg_true_gate2", "tcsg_true_gate1", "gate2_vs_gate1"),
    ("tcsg_true_gate2", "tcsg_zero_gate2", "true_vs_zero_gate2"),
    (
        "tcsg_true_gate2",
        "tcsg_global_source_gate2",
        "true_vs_source_global_gate2",
    ),
    ("tcsg_true_gate2", "tcsg_random_gate2", "true_vs_random_gate2"),
)
PRIMARY_COMPARISONS = {
    "tcsg_true_vs_static_budget",
    "true_vs_zero_gate2",
    "true_vs_source_global_gate2",
    "true_vs_random_gate2",
}

_original_parse_args = exp18.parse_args
_original_result_paths = exp18.result_paths
_original_build_tcsg_model = exp18.build_tcsg_model
_original_run_target_cell = exp18.run_target_cell
_original_context_for_mode = exp18.context_for_mode
_original_load_source_bundle = exp18.load_or_train_source_bundle
_active_gate_scale = 1.0
_source_summary_cache: dict[int, torch.Tensor] = {}


def _experiment_print(*values, **kwargs):
    """Keep the reused runner's log labels experiment-specific."""
    converted = []
    for value in values:
        text = str(value)
        if "[结论判定]" in text:
            text = (
                "[实验20结论判定]\n"
                "1. tcsg_true_vs_static_budget检验gate=2的TCSG整体优势。\n"
                "2. true_vs_zero_gate2检验真实support上下文是否优于零上下文。\n"
                "3. true_vs_source_global_gate2检验目标support是否优于源域全局上下文。\n"
                "4. true_vs_random_gate2检验真实上下文是否优于同范数随机上下文。\n"
                "5. gate2_vs_gate1仅判断门控强度，不代表上下文因果作用。\n"
                "6. validation通过前不访问官方test。"
            )
        text = text.replace("实验18", "实验20")
        text = text.replace("[experiment18]", "[experiment20]")
        converted.append(text)
    builtins.print(*converted, **kwargs)


def parse_args():
    args = _original_parse_args()
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = "outputs/experiment20_context_source_global"
    return args


def result_paths(args):
    paths = _original_result_paths(args)
    output = paths["output"]
    for key, path in tuple(paths.items()):
        if key != "output":
            paths[key] = output / path.name.replace(
                "experiment18_", "experiment20_", 1
            )
    return paths


def build_tcsg_model(*args, **kwargs):
    model = _original_build_tcsg_model(*args, **kwargs)
    for layer in model.graph_layers:
        layer.max_gate *= _active_gate_scale
    return model


def source_global_summary(args, cfg, protocol, target_split_seed, k):
    """Average summaries from FD001-FD003 only; FD004 support is not read."""
    if k in _source_summary_cache:
        return _source_summary_cache[k]

    units = protocol["nested_adaptation_units_by_target_split_seed"][
        str(target_split_seed)
    ][str(k)]
    loaders = exp18.prepare_kshot_experiment(
        cfg,
        args.preprocessing,
        args.balance_mode,
        protocol["validation_units"],
        units,
    )
    source_tasks = loaders[0]
    summaries = []
    for loader in source_tasks.values():
        dataset = loader.dataset
        summaries.append(
            exp18.support_summary(
                dataset.x,
                dataset.y,
                len(cfg["sensor_columns"]),
                cfg["rul_cap"],
            )
        )
    _source_summary_cache[k] = torch.stack(summaries).mean(dim=0).detach().cpu()
    del loaders
    return _source_summary_cache[k]


def context_override(mode, x, y, sensor_count, rul_cap, seed, global_summary):
    if mode == "tcsg_zero_gate2":
        return _original_context_for_mode(
            "tcsg_zero", x, y, sensor_count, rul_cap, seed
        )
    if mode == "tcsg_global_source_gate2":
        return global_summary.clone()
    if mode == "tcsg_random_gate2":
        true_summary = _original_context_for_mode(
            "tcsg_true", x, y, sensor_count, rul_cap, seed
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        random_summary = torch.randn(
            true_summary.shape,
            generator=generator,
            dtype=true_summary.dtype,
        )
        return random_summary * (
            true_summary.norm() / random_summary.norm().clamp_min(1e-6)
        )
    return _original_context_for_mode(
        "tcsg_true", x, y, sensor_count, rul_cap, seed
    )


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
    previous_context = exp18.context_for_mode
    gate_scale = GATE_SCALE.get(regime)
    global_summary = None

    try:
        _active_gate_scale = 1.0 if gate_scale is None else gate_scale
        if regime == "tcsg_global_source_gate2":
            global_summary = source_global_summary(
                args, cfg, protocol, target_split_seed, k
            )

        if regime.startswith("tcsg_") and regime != "tcsg_true_gate1":
            exp18.context_for_mode = (
                lambda mode, x, y, sensor_count, rul_cap, seed: context_override(
                    mode,
                    x,
                    y,
                    sensor_count,
                    rul_cap,
                    seed,
                    global_summary,
                )
            )

        result, audit = _original_run_target_cell(
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

    intervention = {
        "gate_scale": gate_scale,
        "context_source": (
            "target_support"
            if regime.startswith("tcsg_true")
            else "source_domains"
            if regime == "tcsg_global_source_gate2"
            else "random_same_norm"
            if regime == "tcsg_random_gate2"
            else "zero"
            if regime == "tcsg_zero_gate2"
            else "none"
        ),
    }
    result.update(intervention)
    audit.update(intervention)
    return result, audit


def load_or_train_source_bundle(args, cfg, protocol, prior):
    # Reuse Experiment 18 source bundles when the matching cache exists.
    cache_dir = (
        exp18.PROJECT_ROOT
        / "outputs"
        / "experiment18_task_conditioned_sensor_graph"
    )
    cache = cache_dir / (
        f"experiment18_source_bundle_{args.target}_modelseed{cfg['seed']}.pt"
    )
    bundle_cfg = dict(cfg)
    if cache.is_file():
        bundle_cfg["output_dir"] = str(cache_dir)
    return _original_load_source_bundle(args, bundle_cfg, protocol, prior)


def self_check():
    assert set(MODELS) - {"static_budget_prior"} == set(GATE_SCALE)
    assert all(c in MODELS and r in MODELS for c, r, _ in COMPARISONS)
    assert PRIMARY_COMPARISONS <= {name for _, _, name in COMPARISONS}


def main():
    self_check()
    exp18.print = _experiment_print
    exp18.SCRIPT_VERSION = SCRIPT_VERSION
    exp18.MODEL_CHOICES = MODELS
    exp18.DEFAULT_MODELS = list(MODELS)
    exp18.COMPARISONS = COMPARISONS
    exp18.PRIMARY_COMPARISONS = PRIMARY_COMPARISONS
    exp18.parse_args = parse_args
    exp18.result_paths = result_paths
    exp18.build_tcsg_model = build_tcsg_model
    exp18.run_target_cell = run_target_cell
    exp18.load_or_train_source_bundle = load_or_train_source_bundle
    exp18.main()


if __name__ == "__main__":
    main()
