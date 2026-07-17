"""Experiment 3: compare five window-sampling balance strategies.

All variants use the verified full condition-aware preprocessing.  Only the
training sampler changes; validation and official test distributions remain
untouched.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from run_condition_aware_experiment import (
    PROJECT_ROOT,
    inspect_loaders,
    load_config,
    prepare_custom_experiment,
    resolve_path,
    run_one,
)


BALANCE_VARIANTS = (
    "none",
    "stage",
    "engine",
    "sqrt_engine_stage",
    "engine_stage",
)
METRICS = ("rmse", "mae", "r2", "nasa_score")


def parse_args():
    parser = argparse.ArgumentParser(description="实验3：发动机与RUL阶段均衡策略消融")
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
    parser.add_argument("--output-dir", default="outputs/experiment3_balancing_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summarize(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    summary = frame.groupby("balance_mode", as_index=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    reference = summary.loc[summary["balance_mode"] == "none"].iloc[0]
    for metric in ("rmse", "mae", "nasa_score"):
        summary[f"{metric}_change_pct_vs_none"] = 100 * (
            summary[f"{metric}_mean"] - reference[f"{metric}_mean"]
        ) / reference[f"{metric}_mean"]
    summary["r2_delta_vs_none"] = summary["r2_mean"] - reference["r2_mean"]
    return summary.sort_values(["nasa_score_mean", "rmse_mean"])


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]
    results = []
    for seed in seeds:
        for balance_mode in BALANCE_VARIANTS:
            cfg = load_config(args, seed)
            loaders = prepare_custom_experiment(
                cfg,
                "condition_settings",
                balance_mode,
                balance_mode,
            )
            print(f"\n[experiment3] seed={seed} balance={balance_mode}")
            if args.dry_run:
                inspect_loaders(balance_mode, cfg, args.model, loaders)
                continue
            metrics = run_one(
                args.model,
                cfg,
                loaders,
                tag=f"exp3_{balance_mode}_{args.model}",
            )
            metrics.update({"balance_mode": balance_mode})
            results.append(metrics)

    if args.dry_run:
        return
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / f"experiment3_raw_{args.model}_{args.target}.json"
    raw.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(results)
    summary_path = output / f"experiment3_summary_{args.model}_{args.target}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n[experiment3 summary]")
    print(summary.to_string(index=False))
    print("\nSelect a strategy only if multi-seed RMSE and NASA Score provide an acceptable trade-off.")
    print(f"Raw: {raw}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
