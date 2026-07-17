"""Experiment 2: isolate condition normalization and operating-setting inputs.

One command runs four variants with identical target-engine splits:

1. global: original source-global sensor Z-score;
2. settings_only: global sensor Z-score + three normalized settings;
3. condition_norm_only: condition-wise sensor Z-score, no setting inputs;
4. full_condition: condition-wise sensor Z-score + setting inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_condition_aware_experiment import (
    PROJECT_ROOT,
    inspect_loaders,
    load_config,
    prepare_custom_experiment,
    resolve_path,
    run_one,
)


VARIANTS = (
    ("global", "global"),
    ("settings_only", "global_settings"),
    ("condition_norm_only", "condition_norm"),
    ("full_condition", "condition_settings"),
)
METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args():
    parser = argparse.ArgumentParser(description="实验2：工况归一化与settings输入消融")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--model", default="meta_gnn")
    parser.add_argument("--target", default="FD004", choices=[f"FD00{i}" for i in range(1, 5)])
    parser.add_argument("--support-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--adapt-epochs", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument("--output-dir", default="outputs/experiment2_condition_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summarize(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    summary = frame.groupby("variant", as_index=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    reference = summary.loc[summary["variant"] == "global"].iloc[0]
    for metric in ("rmse", "mae", "nasa_score"):
        summary[f"{metric}_change_pct_vs_global"] = 100 * (
            summary[f"{metric}_mean"] - reference[f"{metric}_mean"]
        ) / reference[f"{metric}_mean"]
    summary["r2_delta_vs_global"] = summary["r2_mean"] - reference["r2_mean"]
    return summary.sort_values("rmse_mean")


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]
    results = []
    for seed in seeds:
        for label, preprocessing in VARIANTS:
            cfg = load_config(args, seed)
            loaders = prepare_custom_experiment(cfg, preprocessing, "none", label)
            print(f"\n[experiment2] seed={seed} variant={label}")
            if args.dry_run:
                inspect_loaders(label, cfg, args.model, loaders)
                continue
            metrics = run_one(args.model, cfg, loaders, tag=f"exp2_{label}_{args.model}")
            metrics.update({"variant": label, "preprocessing_mode": preprocessing})
            results.append(metrics)

    if args.dry_run:
        return
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / f"experiment2_raw_{args.model}_{args.target}.json"
    raw.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(results)
    summary_path = output / f"experiment2_summary_{args.model}_{args.target}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n[experiment2 summary]")
    print(summary.to_string(index=False))
    print("\nInterpretation: settings_only isolates setting inputs; condition_norm_only isolates normalization;")
    print("full_condition tests whether both components are complementary.")
    print(f"Raw: {raw}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
