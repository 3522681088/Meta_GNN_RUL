"""Experiment 21: remove the target-time context encoder without retraining.

The fixed model encodes the FD001-FD003 global summary once, replaces the
context encoder with that fixed vector, and keeps the graph and predictor
weights unchanged. This tests lossless parameter and runtime simplification.
"""

from __future__ import annotations

import builtins
from pathlib import Path
import sys
import time
from types import MethodType

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment18_task_conditioned_sensor_graph as exp18
from scripts import experiment20_context_source_global as exp20


SCRIPT_VERSION = "experiment21_fixed_context_compression_v1"
MODELS = (
    "static_budget_prior",
    "tcsg_true_gate2",
    "tcsg_global_source_gate2",
    "tcsg_fixed_source_gate2",
)
COMPARISONS = (
    ("tcsg_fixed_source_gate2", "static_budget_prior", "fixed_vs_static_budget"),
    ("tcsg_fixed_source_gate2", "tcsg_true_gate2", "fixed_vs_true_context"),
    (
        "tcsg_fixed_source_gate2",
        "tcsg_global_source_gate2",
        "fixed_vs_full_global",
    ),
    ("tcsg_true_gate2", "tcsg_global_source_gate2", "true_vs_full_global"),
)
PRIMARY_COMPARISONS = {
    "fixed_vs_static_budget",
    "fixed_vs_full_global",
}

_original_parse_args = exp18.parse_args
_original_result_paths = exp18.result_paths
_original_build_tcsg_model = exp18.build_tcsg_model
_original_run_target_cell = exp18.run_target_cell
_original_context_for_mode = exp18.context_for_mode
_original_load_source_bundle = exp18.load_or_train_source_bundle
_active_gate_scale = 1.0
_collapse_encoder = False
_fixed_summary: torch.Tensor | None = None
_fixed_parameter_count = 0
_removed_parameter_count = 0


class FixedContextEncoder(nn.Module):
    def __init__(self, context: torch.Tensor):
        super().__init__()
        self.register_buffer("context", context.detach().clone())

    def forward(self, unused_summary: torch.Tensor) -> torch.Tensor:
        return self.context


def _experiment_print(*values, **kwargs):
    converted = []
    for value in values:
        text = str(value)
        if "[结论判定]" in text:
            text = (
                "[实验21结论判定]\n"
                "1. fixed_vs_full_global检验移除上下文编码器是否无损。\n"
                "2. fixed_vs_static_budget检验简化模型是否仍保留整体优势。\n"
                "3. 同时比较参数量与cell_wall_seconds。\n"
                "4. validation通过前不访问官方test。"
            )
        text = text.replace("实验18", "实验21")
        text = text.replace("[experiment18]", "[experiment21]")
        converted.append(text)
    builtins.print(*converted, **kwargs)


def parse_args():
    args = _original_parse_args()
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = "outputs/experiment21_fixed_context_compression"
    return args


def result_paths(args):
    paths = _original_result_paths(args)
    output = paths["output"]
    for key, path in tuple(paths.items()):
        if key != "output":
            paths[key] = output / path.name.replace(
                "experiment18_", "experiment21_", 1
            )
    return paths


def build_tcsg_model(*args, **kwargs):
    model = _original_build_tcsg_model(*args, **kwargs)
    for layer in model.graph_layers:
        layer.max_gate *= _active_gate_scale

    if not _collapse_encoder:
        return model

    original_load = model.load_state_dict

    def load_and_collapse(this, state_dict, *load_args, **load_kwargs):
        global _fixed_parameter_count, _removed_parameter_count
        result = original_load(state_dict, *load_args, **load_kwargs)
        if _fixed_summary is None:
            raise RuntimeError("fixed source summary was not prepared")
        with torch.no_grad():
            context = this.task_context_encoder(_fixed_summary)
        _removed_parameter_count = sum(
            parameter.numel()
            for parameter in this.task_context_encoder.parameters()
        )
        this.task_context_encoder = FixedContextEncoder(context)
        _fixed_parameter_count = sum(
            parameter.numel() for parameter in this.parameters()
        )
        return result

    model.load_state_dict = MethodType(load_and_collapse, model)
    return model


def context_override(mode, x, y, sensor_count, rul_cap, seed, source_summary):
    if mode in {"tcsg_global_source_gate2", "tcsg_fixed_source_gate2"}:
        return source_summary.clone()
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
    global _active_gate_scale, _collapse_encoder, _fixed_summary
    global _fixed_parameter_count, _removed_parameter_count

    previous_context = exp18.context_for_mode
    source_summary = None
    started = None
    try:
        _active_gate_scale = 2.0 if regime.startswith("tcsg_") else 1.0
        _collapse_encoder = regime == "tcsg_fixed_source_gate2"
        _fixed_parameter_count = 0
        _removed_parameter_count = 0

        if regime in {"tcsg_global_source_gate2", "tcsg_fixed_source_gate2"}:
            source_summary = exp20.source_global_summary(
                args, cfg, protocol, target_split_seed, k
            )
            _fixed_summary = source_summary
            exp18.context_for_mode = (
                lambda mode, x, y, sensor_count, rul_cap, seed: context_override(
                    mode,
                    x,
                    y,
                    sensor_count,
                    rul_cap,
                    seed,
                    source_summary,
                )
            )

        started = time.perf_counter()
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
        elapsed = (
            time.perf_counter() - started
            if started is not None
            else float("nan")
        )
        exp18.context_for_mode = previous_context
        _active_gate_scale = 1.0
        _collapse_encoder = False
        _fixed_summary = None

    if regime == "tcsg_fixed_source_gate2":
        result["total_parameter_count"] = _fixed_parameter_count
    extra = {
        "gate_scale": 2.0 if regime.startswith("tcsg_") else None,
        "context_source": (
            "target_support"
            if regime == "tcsg_true_gate2"
            else "source_global_fixed"
            if regime == "tcsg_fixed_source_gate2"
            else "source_global"
            if regime == "tcsg_global_source_gate2"
            else "none"
        ),
        "removed_context_encoder_parameters": (
            _removed_parameter_count
            if regime == "tcsg_fixed_source_gate2"
            else 0
        ),
        "cell_wall_seconds": elapsed,
    }
    result.update(extra)
    audit.update(extra)
    return result, audit


def load_or_train_source_bundle(args, cfg, protocol, prior):
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
