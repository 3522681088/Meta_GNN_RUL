"""Experiment 6: rotate FD001-FD004 as unseen target domains.

Each target is evaluated with original, condition-aware, and
condition-aware-balanced pipelines.  Source domains are automatically the
remaining three subsets.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from run_condition_aware_experiment import (
    MODES,
    PROJECT_ROOT,
    inspect_loaders,
    load_config,
    prepare_experiment,
    resolve_path,
    run_one,
)


METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args():
    parser = argparse.ArgumentParser(description="实验6：FD001-FD004轮换目标域实验")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--model", default="meta_gnn")
    parser.add_argument("--targets", nargs="+", default=[f"FD00{i}" for i in range(1, 5)], choices=[f"FD00{i}" for i in range(1, 5)])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--target", default="FD004", help=argparse.SUPPRESS)
    parser.add_argument("--support-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--condition-count", type=int, default=6)
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--adapt-epochs", type=int)
    parser.add_argument("--inner-lr", type=float)
    parser.add_argument("--outer-lr", type=float)
    parser.add_argument("--pair-aux-weight", type=float)
    parser.add_argument("--output-dir", default="outputs/experiment6_cross_domain")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summarize(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    summary = frame.groupby(["target_domain", "mode"], as_index=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)

    rows = []
    for target, group in summary.groupby("target_domain"):
        reference = group.loc[group["mode"] == "original"]
        reference = reference.iloc[0] if len(reference) else None
        for _, row in group.iterrows():
            item = row.to_dict()
            if reference is not None:
                item["rmse_change_pct_vs_original"] = 100 * (
                    row["rmse_mean"] - reference["rmse_mean"]
                ) / reference["rmse_mean"]
                item["nasa_change_pct_vs_original"] = 100 * (
                    row["nasa_score_mean"] - reference["nasa_score_mean"]
                ) / reference["nasa_score_mean"]
                item["r2_delta_vs_original"] = row["r2_mean"] - reference["r2_mean"]
            rows.append(item)
    return pd.DataFrame(rows).sort_values(["target_domain", "rmse_mean"])


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]
    results = []
    for seed in seeds:
        for target in args.targets:
            args.target = target
            for mode in args.modes:
                cfg = load_config(args, seed)
                loaders = prepare_experiment(cfg, mode)
                print(f"\n[experiment6] seed={seed} target={target} mode={mode}")
                if args.dry_run:
                    inspect_loaders(f"{target}_{mode}", cfg, args.model, loaders)
                    continue
                metrics = run_one(
                    args.model,
                    cfg,
                    loaders,
                    tag=f"exp6_{target}_{mode}_{args.model}",
                )
                metrics.update({"mode": mode})
                results.append(metrics)

    if args.dry_run:
        return
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / f"experiment6_raw_{args.model}.json"
    raw.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(results)
    summary_path = output / f"experiment6_summary_{args.model}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    macro = summary.groupby("mode", as_index=False)[
        [f"{metric}_mean" for metric in METRICS]
    ].mean()
    macro_path = output / f"experiment6_macro_average_{args.model}.csv"
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")
    print("\n[experiment6 summary]")
    print(summary.to_string(index=False))
    print("\nReport each target separately and the macro-average across four targets.")
    print("\n[macro average]")
    print(macro.to_string(index=False))
    print(f"Raw: {raw}\nSummary: {summary_path}\nMacro: {macro_path}")


if __name__ == "__main__":
    main()
