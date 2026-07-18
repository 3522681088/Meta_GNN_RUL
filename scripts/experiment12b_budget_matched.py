#!/usr/bin/env python3
"""Experiment 12B: budget-matched ANIL source-split robustness.

This is a self-contained *entry point* for the existing Meta_GNN_RUL project.
It intentionally reuses the tested training/data code in:

* scripts/experiment11_engine_disjoint_anil.py
* scripts/experiment12_source_split_robustness.py

No existing file needs to be edited.  Put this file in ``scripts/`` and run it
from the project root.  Compared with Experiment 12 it makes two changes:

1. The ordinary-transfer reference is ``pretrained_budget_head``.  Experiment
   11 already implements this state as ordinary pretraining followed by the
   automatically computed extra gradient budget.  Both ANIL branches still
   start from ``pretrained_head`` (the common 1500-step state).
2. After training, duplicated shared-baseline rows are collapsed before the
   Batch-ANIL comparison.  Engine-disjoint comparisons retain the crossed
   source-split x model-seed design and use a crossed bootstrap.

Example (validation):

    CUDA_VISIBLE_DEVICES=0 python -u scripts/experiment12b_budget_matched.py \
      --target FD004 \
      --source-task-seeds 2027 2028 2029 2030 2031 \
      --model-seeds 42 43 44 45 46 \
      --k-values 2 5 \
      --regimes pretrained_budget_head anil_batch_head \
                anil_engine_disjoint_head \
      --evaluation-scope validation \
      --preprocessing condition_settings \
      --balance-mode engine_stage \
      --meta-epochs 100 --meta-inner-lr 0.00001 --meta-inner-steps 1 \
      --anil-meta-lr 0.0001 --source-query-fraction 0.30 \
      --source-pretrain-steps 1500 --source-pretrain-lr 0.001 \
      --target-epochs 10 --target-lr 0.001 \
      --output-dir outputs/experiment12b_budget_matched --resume
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experiment12_source_split_robustness as exp12  # noqa: E402


SCRIPT_VERSION = "experiment12b_budget_matched_v1"
REGIMES = (
    "pretrained_budget_head",
    "anil_batch_head",
    "anil_engine_disjoint_head",
)
COMPARISONS = (
    (
        "anil_engine_disjoint_head",
        "pretrained_budget_head",
        "engine_disjoint_anil_vs_budget_head",
    ),
    (
        "anil_engine_disjoint_head",
        "anil_batch_head",
        "engine_disjoint_vs_batch_anil",
    ),
    (
        "anil_batch_head",
        "pretrained_budget_head",
        "batch_anil_vs_budget_head",
    ),
)
PRIMARY_COMPARISONS = {"engine_disjoint_anil_vs_budget_head"}
LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}
METRICS = ("rmse", "mae", "r2", "nasa_score")


def patch_experiment12() -> None:
    """Switch Experiment 12 to its already-implemented budget baseline."""
    exp12.SCRIPT_VERSION = SCRIPT_VERSION
    exp12.REGIMES = REGIMES
    exp12.COMPARISONS = COMPARISONS
    exp12.PRIMARY_COMPARISONS = PRIMARY_COMPARISONS


def ensure_new_output_directory() -> None:
    """Prevent accidental reuse of Experiment 12's old source-state caches."""
    if "--output-dir" not in sys.argv:
        sys.argv.extend(
            ["--output-dir", "outputs/experiment12b_budget_matched"]
        )


def cli_value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan"), float("nan")
    if np.allclose(values, values[0]):
        if np.isclose(values[0], 0.0):
            return 0.0, 0.0, 1.0
        return float(values[0]), float(values[0]), 0.0
    sem = stats.sem(values)
    low, high = stats.t.interval(
        0.95, values.size - 1, loc=values.mean(), scale=sem
    )
    p_value = stats.ttest_1samp(values, 0.0).pvalue
    return float(low), float(high), float(p_value)


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(values).pvalue)
    except ValueError:
        return float("nan")


def bootstrap_model_seed(
    values: np.ndarray, repetitions: int, random_seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def crossed_bootstrap(
    paired: pd.DataFrame,
    delta_column: str,
    repetitions: int,
    random_seed: int,
) -> tuple[float, float]:
    matrix = paired.pivot(
        index="source_split_seed",
        columns="model_seed",
        values=delta_column,
    ).sort_index().sort_index(axis=1)
    if matrix.empty or bool(matrix.isna().any().any()):
        return float("nan"), float("nan")
    values = matrix.to_numpy(dtype=float)
    n_splits, n_seeds = values.shape
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        split_indices = rng.integers(0, n_splits, size=n_splits)
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        bootstrap[index] = values[np.ix_(split_indices, seed_indices)].mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm family-wise-error correction without a statsmodels dependency."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    running = 0.0
    for rank, (index, value) in enumerate(ordered.items()):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        result.loc[index] = running
    return result


def unique_rows_for_summary(raw: pd.DataFrame, regime: str) -> pd.DataFrame:
    rows = raw[raw["regime"] == regime].copy()
    if regime in {"pretrained_budget_head", "anil_batch_head"}:
        rows = rows.drop_duplicates(["k", "model_seed", "regime"])
    else:
        rows = rows.drop_duplicates(
            ["k", "source_split_seed", "model_seed", "regime"]
        )
    return rows


def corrected_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for k in sorted(int(value) for value in raw["k"].unique()):
        for regime in REGIMES:
            group = unique_rows_for_summary(raw[raw["k"] == k], regime)
            if group.empty:
                continue
            row = {
                "k": k,
                "regime": regime,
                "statistics_unit": (
                    "model_seed"
                    if regime in {"pretrained_budget_head", "anil_batch_head"}
                    else "source_split_seed_x_model_seed"
                ),
                "n_independent_results": len(group),
                "n_source_splits": group["source_split_seed"].nunique(),
                "n_model_seeds": group["model_seed"].nunique(),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1))
                row[f"{metric}_median"] = float(group[metric].median())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(
    raw: pd.DataFrame, k: int, candidate: str, reference: str
) -> tuple[pd.DataFrame, str]:
    candidate_rows = raw[
        (raw["k"] == k) & (raw["regime"] == candidate)
    ].copy()
    reference_rows = raw[
        (raw["k"] == k) & (raw["regime"] == reference)
    ].copy()

    shared_only = candidate in {
        "pretrained_budget_head",
        "anil_batch_head",
    } and reference in {"pretrained_budget_head", "anil_batch_head"}
    if shared_only:
        keys = ["model_seed"]
        candidate_rows = candidate_rows.drop_duplicates(keys)
        reference_rows = reference_rows.drop_duplicates(keys)
        unit = "model_seed"
    else:
        keys = ["source_split_seed", "model_seed"]
        candidate_rows = candidate_rows.drop_duplicates(keys)
        reference_rows = reference_rows.drop_duplicates(keys)
        unit = "source_split_seed_x_model_seed"

    columns = keys + list(METRICS)
    merged = candidate_rows[columns].merge(
        reference_rows[columns],
        on=keys,
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    for metric in METRICS:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
        )
    return merged, unit


def corrected_comparisons(
    raw: pd.DataFrame, repetitions: int = 5000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict] = []
    paired_rows: list[pd.DataFrame] = []
    for k in sorted(int(value) for value in raw["k"].unique()):
        for comparison_index, (candidate, reference, label) in enumerate(COMPARISONS):
            paired, unit = paired_table(raw, k, candidate, reference)
            if paired.empty:
                continue
            paired.insert(0, "comparison", label)
            paired.insert(0, "k", k)
            paired_rows.append(paired)
            row: dict = {
                "k": k,
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "is_primary_comparison": label in PRIMARY_COMPARISONS,
                "statistics_unit": unit,
                "n_independent_pairs": len(paired),
                "n_source_splits": (
                    paired["source_split_seed"].nunique()
                    if "source_split_seed" in paired
                    else 0
                ),
                "n_model_seeds": paired["model_seed"].nunique(),
            }
            for metric_index, metric in enumerate(METRICS):
                delta = paired[f"{metric}_delta"].to_numpy(dtype=float)
                if unit == "model_seed":
                    inference_delta = delta
                    boot_low, boot_high = bootstrap_model_seed(
                        delta,
                        repetitions,
                        12000 + 100 * k + 10 * comparison_index + metric_index,
                    )
                    split_win_rate = float("nan")
                    model_win_rate = float(
                        np.mean(delta < 0 if metric in LOWER_IS_BETTER else delta > 0)
                    )
                else:
                    boot_low, boot_high = crossed_bootstrap(
                        paired,
                        f"{metric}_delta",
                        repetitions,
                        12000 + 100 * k + 10 * comparison_index + metric_index,
                    )
                    split_delta = paired.groupby("source_split_seed")[
                        f"{metric}_delta"
                    ].mean()
                    # Source-task split is the pre-registered high-level unit.
                    # Do not run a naive t-test on all crossed cells.
                    inference_delta = split_delta.to_numpy(dtype=float)
                    split_win_rate = float(
                        np.mean(
                            split_delta < 0
                            if metric in LOWER_IS_BETTER
                            else split_delta > 0
                        )
                    )
                    by_seed = paired.groupby("model_seed")[f"{metric}_delta"].mean()
                    model_win_rate = float(
                        np.mean(
                            by_seed < 0
                            if metric in LOWER_IS_BETTER
                            else by_seed > 0
                        )
                    )
                t_low, t_high, t_p = t_interval(inference_delta)
                candidate_mean = float(paired[f"{metric}_candidate"].mean())
                reference_mean = float(paired[f"{metric}_reference"].mean())
                if metric in LOWER_IS_BETTER and not np.isclose(reference_mean, 0.0):
                    improvement = 100.0 * (reference_mean - candidate_mean) / reference_mean
                else:
                    improvement = float("nan")
                row.update(
                    {
                        f"{metric}_candidate_mean": candidate_mean,
                        f"{metric}_reference_mean": reference_mean,
                        f"{metric}_delta_mean": float(delta.mean()),
                        f"{metric}_improvement_pct": improvement,
                        f"{metric}_cell_win_rate": float(
                            np.mean(
                                delta < 0
                                if metric in LOWER_IS_BETTER
                                else delta > 0
                            )
                        ),
                        f"{metric}_source_split_win_rate": split_win_rate,
                        f"{metric}_model_seed_win_rate": model_win_rate,
                        f"{metric}_t_ci95_low": t_low,
                        f"{metric}_t_ci95_high": t_high,
                        f"{metric}_bootstrap_ci95_low": boot_low,
                        f"{metric}_bootstrap_ci95_high": boot_high,
                        f"{metric}_t_p": t_p,
                        f"{metric}_wilcoxon_p": wilcoxon_p(inference_delta),
                    }
                )
            comparison_rows.append(row)

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        for metric in METRICS:
            comparisons[f"{metric}_t_p_holm"] = holm_adjust(
                comparisons[f"{metric}_t_p"]
            )
        comparisons["meets_rmse_effect_3pct"] = (
            comparisons["rmse_improvement_pct"] >= 3.0
        )
        comparisons["meets_source_split_win_rate_80pct"] = (
            comparisons["rmse_source_split_win_rate"] >= 0.8
        )
        comparisons["rmse_bootstrap_ci_excludes_zero"] = (
            comparisons["rmse_bootstrap_ci95_high"] < 0.0
        )
        comparisons["meets_rmse_holm_p_005"] = (
            comparisons["rmse_t_p_holm"] < 0.05
        )
        comparisons["nasa_not_worse"] = (
            comparisons["nasa_score_delta_mean"] <= 0.0
        )
        comparisons["robust_success"] = (
            comparisons["is_primary_comparison"]
            & comparisons["meets_rmse_effect_3pct"]
            & comparisons["meets_source_split_win_rate_80pct"]
            & comparisons["rmse_bootstrap_ci_excludes_zero"]
            & comparisons["nasa_not_worse"]
        )
        comparisons["strict_success"] = (
            comparisons["robust_success"]
            & comparisons["meets_rmse_holm_p_005"]
        )
    paired_all = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    return comparisons, paired_all


def find_raw_file(output_dir: Path, scope: str, target: str) -> Path | None:
    exact = output_dir / f"experiment12_{scope}_{target}_raw.json"
    if exact.is_file():
        return exact
    candidates = sorted(output_dir.glob(f"*_{scope}_{target}_raw.json"))
    return candidates[-1] if candidates else None


def budget_audit(output_dir: Path) -> dict:
    candidates = sorted(output_dir.glob("*source_diagnostics*.json"))
    if not candidates:
        return {"status": "missing_source_diagnostics"}
    diagnostics = json.loads(candidates[-1].read_text(encoding="utf-8"))
    rows: list[dict] = []
    for model_seed, payload in diagnostics.items():
        shared = payload.get("shared", {}) if isinstance(payload, dict) else {}
        ordinary = shared.get("ordinary", {})
        budget = shared.get("ordinary_budget", {})
        if not budget:
            continue
        base = budget.get("base_optimizer_steps")
        extra = budget.get("extra_optimizer_steps")
        total = budget.get("total_optimizer_steps")
        rows.append(
            {
                "model_seed": int(model_seed),
                "ordinary_optimizer_steps": ordinary.get("optimizer_steps"),
                "budget_base_optimizer_steps": base,
                "budget_extra_optimizer_steps": extra,
                "budget_total_optimizer_steps": total,
                "arithmetic_valid": (
                    base is not None
                    and extra is not None
                    and total == base + extra
                ),
            }
        )
    return {
        "status": "passed" if rows and all(r["arithmetic_valid"] for r in rows) else "failed",
        "model_seed_count": len(rows),
        "rows": rows,
    }


def postprocess() -> None:
    if "--dry-run" in sys.argv:
        print("[experiment12b] dry-run完成，不生成修正统计表。")
        return
    target = cli_value("--target", "FD004")
    scope = cli_value("--evaluation-scope", "validation")
    output_dir = Path(cli_value("--output-dir", "outputs/experiment12b_budget_matched"))
    raw_path = find_raw_file(output_dir, scope, target)
    if raw_path is None:
        print(f"[experiment12b] 未找到原始结果，跳过后处理：{output_dir}")
        return
    raw = pd.DataFrame(json.loads(raw_path.read_text(encoding="utf-8")))
    required = set(REGIMES)
    present = set(raw.get("regime", pd.Series(dtype=str)).dropna().unique())
    missing = sorted(required - present)
    if missing:
        print(f"[experiment12b] 结果尚未完整，缺少方法：{missing}")
        return

    summary = corrected_summary(raw)
    comparisons, paired = corrected_comparisons(raw)
    prefix = output_dir / f"experiment12b_{scope}_{target}"
    summary_path = Path(f"{prefix}_summary_corrected.csv")
    comparisons_path = Path(f"{prefix}_comparisons_corrected.csv")
    paired_path = Path(f"{prefix}_paired_corrected.csv")
    audit_path = Path(f"{prefix}_budget_audit.json")
    summary.to_csv(summary_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)
    paired.to_csv(paired_path, index=False)
    audit_path.write_text(
        json.dumps(budget_audit(output_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[experiment12b预算匹配修正汇总]")
    columns = [
        "k",
        "regime",
        "statistics_unit",
        "n_independent_results",
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "nasa_score_mean",
    ]
    print(summary[columns].to_string(index=False))
    print("\n[experiment12b主要比较]")
    comparison_columns = [
        "k",
        "comparison",
        "statistics_unit",
        "n_independent_pairs",
        "rmse_improvement_pct",
        "rmse_source_split_win_rate",
        "rmse_model_seed_win_rate",
        "rmse_bootstrap_ci95_low",
        "rmse_bootstrap_ci95_high",
        "rmse_t_p_holm",
        "nasa_score_delta_mean",
        "robust_success",
        "strict_success",
    ]
    print(comparisons[comparison_columns].to_string(index=False))
    print(f"\nCorrected summary: {summary_path.resolve()}")
    print(f"Corrected comparisons: {comparisons_path.resolve()}")
    print(f"Corrected paired rows: {paired_path.resolve()}")
    print(f"Budget audit: {audit_path.resolve()}")


def main() -> None:
    ensure_new_output_directory()
    patch_experiment12()
    exp12.main()
    postprocess()


if __name__ == "__main__":
    main()
