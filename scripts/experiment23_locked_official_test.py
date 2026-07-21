"""Experiment 23: locked official-test evaluation of the final model.

Model selection is finished by Experiment 22. This entry accepts K=5 only
and requires explicit official-test confirmation. Do not tune after running.
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

from scripts import experiment22_cross_target_fixed_context as exp22


SCRIPT_VERSION = "experiment23_locked_official_test_v1"


def parse_args():
    args = exp22.exp21._original_parse_args()
    if set(args.k_values) != {5}:
        raise ValueError("Experiment 23 is locked to --k-values 5")
    if args.evaluation_scope != "official_test":
        raise ValueError(
            "Experiment 23 requires --evaluation-scope official_test"
        )
    if not args.confirm_official_test:
        raise ValueError("Experiment 23 requires --confirm-official-test")
    if args.output_dir == "outputs/experiment18_task_conditioned_sensor_graph":
        args.output_dir = f"outputs/experiment23_locked_official_test/{args.target}"
    return args


def result_paths(args):
    paths = exp22.exp21._original_result_paths(args)
    output = paths["output"]
    for key, path in tuple(paths.items()):
        if key != "output":
            paths[key] = output / path.name.replace(
                "experiment18_", "experiment23_", 1
            )
    return paths


def experiment_print(*values, **kwargs):
    converted = []
    for value in values:
        text = str(value)
        if "[结论判定]" in text:
            text = (
                "[实验23最终判定]\n"
                "1. K=5、模型、种子和训练预算已由实验22锁定。\n"
                "2. fixed_vs_true_context检验轻量化是否保持官方测试性能。\n"
                "3. fixed_vs_static_budget报告最终模型相对公平基线的优势。\n"
                "4. 本次运行后不得根据官方测试结果继续调参。"
            )
        text = text.replace("实验18", "实验23")
        text = text.replace("[experiment18]", "[experiment23]")
        converted.append(text)
    builtins.print(*converted, **kwargs)


def main():
    exp22.SCRIPT_VERSION = SCRIPT_VERSION
    exp22.parse_args = parse_args
    exp22.result_paths = result_paths
    exp22.experiment_print = experiment_print
    exp22.main()


if __name__ == "__main__":
    main()
