"""Experiment 22: cross-target validation of the fixed-context TCSG.

Run this entry once per target (FD001-FD004). It reuses Experiment 21's
validated encoder folding and compares the lightweight model with the full
target-context model and the optimizer-budget-matched static graph.
"""

from __future__ import annotations

import builtins
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment21_fixed_context_compression as exp21


SCRIPT_VERSION = "experiment22_cross_target_fixed_context_v1"
MODELS = (
    "static_budget_prior",
    "tcsg_true_gate2",
    "tcsg_fixed_source_gate2",
)
COMPARISONS = (
    ("tcsg_true_gate2", "static_budget_prior", "tcsg_true_vs_static_budget"),
    ("tcsg_fixed_source_gate2", "static_budget_prior", "fixed_vs_static_budget"),
    ("tcsg_fixed_source_gate2", "tcsg_true_gate2", "fixed_vs_true_context"),
)
PRIMARY_COMPARISONS = {
    "tcsg_true_vs_static_budget",
    "fixed_vs_static_budget",
    "fixed_vs_true_context",
}


def parse_args():
    args = exp21._original_parse_args()
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = (
            f"outputs/experiment22_cross_target_fixed_context/{args.target}"
        )
    return args


def result_paths(args):
    paths = exp21._original_result_paths(args)
    output = paths["output"]
    for key, path in tuple(paths.items()):
        if key != "output":
            paths[key] = output / path.name.replace(
                "experiment18_", "experiment22_", 1
            )
    return paths


def experiment_print(*values, **kwargs):
    converted = []
    for value in values:
        text = str(value)
        if "[结论判定]" in text:
            text = (
                "[实验22结论判定]\n"
                "1. fixed_vs_true_context检验轻量模型在当前目标域是否无损。\n"
                "2. fixed_vs_static_budget检验轻量模型是否保留整体优势。\n"
                "3. FD001-FD004均完成后再判断跨目标稳健性。\n"
                "4. validation通过前不访问官方test。"
            )
        text = text.replace("实验18", "实验22")
        text = text.replace("[experiment18]", "[experiment22]")
        converted.append(text)
    builtins.print(*converted, **kwargs)


def self_check():
    assert all(c in MODELS and r in MODELS for c, r, _ in COMPARISONS)
    assert PRIMARY_COMPARISONS <= {name for _, _, name in COMPARISONS}


def main():
    self_check()
    exp21.SCRIPT_VERSION = SCRIPT_VERSION
    exp21.MODELS = MODELS
    exp21.COMPARISONS = COMPARISONS
    exp21.PRIMARY_COMPARISONS = PRIMARY_COMPARISONS
    exp21.parse_args = parse_args
    exp21.result_paths = result_paths
    exp21._experiment_print = experiment_print
    exp21.main()


if __name__ == "__main__":
    main()
