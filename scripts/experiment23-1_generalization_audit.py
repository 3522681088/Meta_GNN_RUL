"""Experiment 23-1: audit validation-to-official-test generalization.

This analysis reads only saved Experiment 22/23 JSON results. It does not
train a model, load C-MAPSS data, or run another official-test forward pass.
Positive improvement values always mean that the candidate is better.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_VERSION = "experiment23-1_generalization_audit_v1"
TARGETS = ("FD001", "FD002", "FD003", "FD004")
METRICS = ("rmse", "mae", "r2", "nasa_score")
LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}
MODELS = (
    "static_budget_prior",
    "tcsg_true_gate2",
    "tcsg_fixed_source_gate2",
)
COMPARISONS = (
    (
        "fixed_vs_static_budget",
        "tcsg_fixed_source_gate2",
        "static_budget_prior",
    ),
    ("true_vs_static_budget", "tcsg_true_gate2", "static_budget_prior"),
    (
        "fixed_vs_true_context",
        "tcsg_fixed_source_gate2",
        "tcsg_true_gate2",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("outputs/experiment22_cross_target_fixed_context"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("outputs/experiment23_locked_official_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiment23-1_generalization_audit"),
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    return parser.parse_args()


def resolve(root: Path, project_root: Path) -> Path:
    return root if root.is_absolute() else project_root / root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, scope: str, target: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")

    selected = []
    for row in rows:
        if row.get("target_domain") != target or row.get("k") != 5:
            continue
        if row.get("model") not in MODELS:
            continue
        if row.get("evaluation_scope") != scope:
            raise ValueError(f"Unexpected evaluation scope in {path}")
        if scope == "official_test" and row.get("official_test_forward_run") is not True:
            raise ValueError(f"Saved official result is not marked complete: {path}")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {metric} in {path}")
        selected.append(row)

    expected = len(MODELS) * 25
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} completed K=5 rows for {target}, got "
            f"{len(selected)} in {path}"
        )
    return selected


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = (row["model"], int(row["target_split_seed"]), int(row["model_seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate result cell: {key}")
        indexed[key] = row
    return indexed


def improvement(metric: str, candidate: float, reference: float) -> float:
    return reference - candidate if metric in LOWER_IS_BETTER else candidate - reference


def relative_improvement(metric: str, candidate: float, reference: float) -> float | None:
    if metric == "r2" or reference == 0:
        return None
    return 100.0 * improvement(metric, candidate, reference) / abs(reference)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def worst_fraction_mean(values: list[float], fraction: float = 0.2) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return mean(sorted(values)[:count])


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for index in ordered[start:end]:
            result[index] = average_rank
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_cell_rows(
    target: str, validation: dict[tuple, dict], official: dict[tuple, dict]
) -> tuple[list[dict], list[dict]]:
    cell_rows = []
    reproduction_rows = []
    seeds = sorted({(key[1], key[2]) for key in validation})

    for model, split_seed, model_seed in sorted(validation):
        validation_row = validation[(model, split_seed, model_seed)]
        official_row = official[(model, split_seed, model_seed)]
        for metric in METRICS:
            first = float(validation_row[metric])
            repeated = float(official_row[f"validation_{metric}"])
            reproduction_rows.append(
                {
                    "target": target,
                    "model": model,
                    "target_split_seed": split_seed,
                    "model_seed": model_seed,
                    "metric": metric,
                    "experiment22_validation": first,
                    "experiment23_embedded_validation": repeated,
                    "absolute_difference": abs(first - repeated),
                }
            )

    for comparison, candidate, reference in COMPARISONS:
        for split_seed, model_seed in seeds:
            validation_candidate = validation[(candidate, split_seed, model_seed)]
            validation_reference = validation[(reference, split_seed, model_seed)]
            official_candidate = official[(candidate, split_seed, model_seed)]
            official_reference = official[(reference, split_seed, model_seed)]
            for metric in METRICS:
                val_candidate = float(validation_candidate[metric])
                val_reference = float(validation_reference[metric])
                test_candidate = float(official_candidate[metric])
                test_reference = float(official_reference[metric])
                val_improvement = improvement(metric, val_candidate, val_reference)
                test_improvement = improvement(metric, test_candidate, test_reference)
                cell_rows.append(
                    {
                        "target": target,
                        "target_split_seed": split_seed,
                        "model_seed": model_seed,
                        "comparison": comparison,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "validation_candidate": val_candidate,
                        "validation_reference": val_reference,
                        "validation_improvement": val_improvement,
                        "validation_win": int(val_improvement > 0),
                        "official_candidate": test_candidate,
                        "official_reference": test_reference,
                        "official_improvement": test_improvement,
                        "official_win": int(test_improvement > 0),
                        "improvement_generalization_gap": test_improvement
                        - val_improvement,
                    }
                )
    return cell_rows, reproduction_rows


def grouped(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def build_stability(cell_rows: list[dict]) -> list[dict]:
    output = []
    for axis in ("target_split_seed", "model_seed"):
        fields = ("target", "comparison", "metric", axis)
        for key, rows in sorted(grouped(cell_rows, fields).items()):
            val = [row["validation_improvement"] for row in rows]
            test = [row["official_improvement"] for row in rows]
            output.append(
                {
                    "target": key[0],
                    "comparison": key[1],
                    "metric": key[2],
                    "axis": axis,
                    "seed": key[3],
                    "n_cells": len(rows),
                    "validation_improvement_mean": mean(val),
                    "official_improvement_mean": mean(test),
                    "official_improvement_std": sample_std(test),
                    "official_win_rate": mean([row["official_win"] for row in rows]),
                    "improvement_generalization_gap": mean(test) - mean(val),
                }
            )
    return output


def build_summary(cell_rows: list[dict], stability: list[dict]) -> list[dict]:
    output = []
    fields = ("target", "comparison", "metric")
    stability_groups = grouped(stability, fields + ("axis",))
    for key, rows in sorted(grouped(cell_rows, fields).items()):
        val_candidate = [row["validation_candidate"] for row in rows]
        val_reference = [row["validation_reference"] for row in rows]
        test_candidate = [row["official_candidate"] for row in rows]
        test_reference = [row["official_reference"] for row in rows]
        val = [row["validation_improvement"] for row in rows]
        test = [row["official_improvement"] for row in rows]
        split_means = [
            row["official_improvement_mean"]
            for row in stability_groups[key + ("target_split_seed",)]
        ]
        model_means = [
            row["official_improvement_mean"]
            for row in stability_groups[key + ("model_seed",)]
        ]
        split_std, model_std = sample_std(split_means), sample_std(model_means)
        output.append(
            {
                "target": key[0],
                "comparison": key[1],
                "metric": key[2],
                "n_cells": len(rows),
                "validation_candidate_mean": mean(val_candidate),
                "validation_reference_mean": mean(val_reference),
                "validation_improvement_mean": mean(val),
                "validation_relative_improvement_pct": relative_improvement(
                    key[2], mean(val_candidate), mean(val_reference)
                ),
                "validation_win_rate": mean([row["validation_win"] for row in rows]),
                "official_candidate_mean": mean(test_candidate),
                "official_reference_mean": mean(test_reference),
                "official_improvement_mean": mean(test),
                "official_relative_improvement_pct": relative_improvement(
                    key[2], mean(test_candidate), mean(test_reference)
                ),
                "official_win_rate": mean([row["official_win"] for row in rows]),
                "improvement_generalization_gap": mean(test) - mean(val),
                "official_improvement_median": statistics.median(test),
                "official_improvement_p10": percentile(test, 0.1),
                "official_worst20_cvar_improvement": worst_fraction_mean(test),
                "pearson_validation_official_improvement": pearson(val, test),
                "spearman_validation_official_improvement": spearman(val, test),
                "target_split_mean_effect_std": split_std,
                "model_seed_mean_effect_std": model_std,
                "dominant_variability_axis": (
                    "target_split_seed" if split_std >= model_std else "model_seed"
                ),
            }
        )
    return output


def build_conflicts(cell_rows: list[dict]) -> list[dict]:
    primary = [
        row for row in cell_rows if row["comparison"] == "fixed_vs_static_budget"
    ]
    cells = grouped(primary, ("target", "target_split_seed", "model_seed"))
    output = []
    for key, rows in sorted(cells.items()):
        by_metric = {row["metric"]: row for row in rows}
        rmse = by_metric["rmse"]["official_improvement"]
        mae = by_metric["mae"]["official_improvement"]
        nasa = by_metric["nasa_score"]["official_improvement"]
        output.append(
            {
                "target": key[0],
                "target_split_seed": key[1],
                "model_seed": key[2],
                "official_rmse_improvement": rmse,
                "official_mae_improvement": mae,
                "official_nasa_improvement": nasa,
                "rmse_improves": int(rmse > 0),
                "mae_improves": int(mae > 0),
                "nasa_improves": int(nasa > 0),
                "all_three_improve": int(rmse > 0 and mae > 0 and nasa > 0),
                "rmse_improves_but_nasa_worsens": int(rmse > 0 and nasa <= 0),
                "rmse_worsens_but_mae_improves": int(rmse <= 0 and mae > 0),
                "rmse_and_nasa_worsen": int(rmse <= 0 and nasa <= 0),
            }
        )
    return output


def aggregate_reproduction(rows: list[dict]) -> list[dict]:
    output = []
    for key, group in sorted(grouped(rows, ("target", "model", "metric")).items()):
        differences = [row["absolute_difference"] for row in group]
        output.append(
            {
                "target": key[0],
                "model": key[1],
                "metric": key[2],
                "n_cells": len(group),
                "mean_absolute_difference": mean(differences),
                "max_absolute_difference": max(differences),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    validation_root = resolve(args.validation_root, project_root)
    official_root = resolve(args.official_root, project_root)
    output_dir = resolve(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cells, all_reproduction, sources = [], [], []
    for target in args.targets:
        validation_path = validation_root / target / f"experiment22_{target}_raw.json"
        official_path = official_root / target / f"experiment23_{target}_raw.json"
        validation = index_rows(load_rows(validation_path, "validation", target))
        official = index_rows(load_rows(official_path, "official_test", target))
        if validation.keys() != official.keys():
            raise ValueError(f"Experiment 22/23 cells do not match for {target}")
        cells, reproduction = build_cell_rows(target, validation, official)
        all_cells.extend(cells)
        all_reproduction.extend(reproduction)
        for label, path in (("validation", validation_path), ("official_test", official_path)):
            sources.append(
                {
                    "target": target,
                    "scope": label,
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )

    stability = build_stability(all_cells)
    summary = build_summary(all_cells, stability)
    conflicts = build_conflicts(all_cells)
    reproduction = aggregate_reproduction(all_reproduction)

    prefix = "experiment23-1"
    write_csv(output_dir / f"{prefix}_cell_audit.csv", all_cells)
    write_csv(output_dir / f"{prefix}_summary.csv", summary)
    write_csv(output_dir / f"{prefix}_stability.csv", stability)
    write_csv(output_dir / f"{prefix}_metric_conflicts.csv", conflicts)
    write_csv(output_dir / f"{prefix}_validation_reproduction.csv", reproduction)

    report = {
        "script_version": SCRIPT_VERSION,
        "analysis_only": True,
        "retraining_performed": False,
        "official_test_forward_pass_performed": False,
        "post_official_results_must_not_be_used_for_model_tuning": True,
        "targets": list(args.targets),
        "k": 5,
        "models": list(MODELS),
        "comparisons": [item[0] for item in COMPARISONS],
        "improvement_definition": (
            "positive is better: reference-candidate for RMSE/MAE/NASA; "
            "candidate-reference for R2"
        ),
        "source_files": sources,
        "row_counts": {
            "cell_audit": len(all_cells),
            "summary": len(summary),
            "stability": len(stability),
            "metric_conflicts": len(conflicts),
            "validation_reproduction": len(reproduction),
        },
        "primary_fixed_vs_static_summary": [
            row for row in summary if row["comparison"] == "fixed_vs_static_budget"
        ],
    }
    report_path = output_dir / f"{prefix}_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    expected_cells = len(args.targets) * len(COMPARISONS) * len(METRICS) * 25
    assert len(all_cells) == expected_cells
    assert len(conflicts) == len(args.targets) * 25
    print(f"[{SCRIPT_VERSION}] complete")
    print(f"Read {len(sources)} locked result files; audited {len(all_cells)} paired rows.")
    print(f"Outputs: {output_dir}")
    print("Analysis only: no training and no official-test forward pass were run.")


if __name__ == "__main__":
    main()
