"""Experiment 7: stage-wise and asymmetric-error analysis for saved checkpoints.

Only load checkpoints you created or trust.  The script rebuilds the exact
source-fitted preprocessing pipeline, evaluates the official target test set,
and writes per-engine predictions plus stage-wise metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_condition_aware_experiment import (
    PROJECT_ROOT,
    build_model,
    prepare_custom_experiment,
    resolve_path,
    rul_stage_ids,
)
from evaluation.metrics import regression_metrics


STAGE_NAMES = ("critical", "middle", "early", "high_rul")
LEGACY_PREPROCESSING = {
    "original": "global",
    "condition": "condition_settings",
    "condition_balanced": "condition_settings",
}


def parse_args():
    parser = argparse.ArgumentParser(description="实验7：模型分RUL阶段误差与偏晚预测分析")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--checkpoint-dir", default="outputs/condition_aware")
    parser.add_argument("--data-dir", help="覆盖checkpoint中保存的数据目录")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/experiment7_error_analysis")
    return parser.parse_args()


def trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_paths(args) -> list[Path]:
    if args.checkpoints:
        paths = [resolve_path(value, PROJECT_ROOT) for value in args.checkpoints]
    else:
        folder = resolve_path(args.checkpoint_dir, PROJECT_ROOT)
        paths = sorted(folder.glob("*.pt"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    if not paths:
        raise FileNotFoundError("No checkpoint files found")
    return paths


def checkpoint_pipeline(checkpoint: dict):
    split = checkpoint.get("split", {})
    preprocessing = split.get("preprocessing_mode", "global")
    preprocessing = LEGACY_PREPROCESSING.get(preprocessing, preprocessing)
    balance_mode = split.get("balance_mode")
    if balance_mode is None:
        balance_mode = "engine_stage" if split.get("balanced_sampling", False) else "none"
    return preprocessing, balance_mode


def predict(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device))
            labels.extend(y.cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
    return np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)


def extended_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    metrics = regression_metrics(y, prediction)
    error = prediction - y
    metrics.update(
        {
            "bias": float(error.mean()),
            "late_prediction_rate": float((error > 0).mean()),
            "early_prediction_rate": float((error < 0).mean()),
            "mean_late_error": float(error[error > 0].mean()) if np.any(error > 0) else 0.0,
            "mean_early_error": float(error[error < 0].mean()) if np.any(error < 0) else 0.0,
            "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)),
            "max_absolute_error": float(np.abs(error).max()),
        }
    )
    return metrics


def main():
    args = parse_args()
    paths = resolve_checkpoint_paths(args)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    output = resolve_path(args.output_dir, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    overall_rows = []
    stage_rows = []

    for path in paths:
        checkpoint = trusted_torch_load(path)
        cfg = dict(checkpoint["config"])
        if args.data_dir:
            cfg["data_dir"] = str(resolve_path(args.data_dir, PROJECT_ROOT))
        preprocessing, balance_mode = checkpoint_pipeline(checkpoint)
        loaders = prepare_custom_experiment(
            cfg,
            preprocessing,
            balance_mode,
            experiment_label=path.stem,
        )
        _, _, _, test, feature_count, _ = loaders
        model_name = checkpoint["metrics"]["model"]
        model = build_model(model_name, feature_count, cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        y, prediction = predict(model, test, device)
        units = np.asarray(test.dataset.units)
        errors = prediction - y
        stages = rul_stage_ids(y)
        experiment = checkpoint["metrics"].get("experiment", path.stem)

        overall = extended_metrics(y, prediction)
        overall.update(
            {
                "checkpoint": path.name,
                "experiment": experiment,
                "model": model_name,
                "target_domain": cfg["target_domain"],
                "seed": cfg["seed"],
                "preprocessing_mode": preprocessing,
                "balance_mode": balance_mode,
                "test_engines": len(y),
            }
        )
        overall_rows.append(overall)

        prediction_frame = pd.DataFrame(
            {
                "unit": units,
                "true_rul": y,
                "predicted_rul": prediction,
                "error_pred_minus_true": errors,
                "absolute_error": np.abs(errors),
                "stage": [STAGE_NAMES[index] for index in stages],
                "is_late_prediction": errors > 0,
            }
        )
        prediction_frame.to_csv(
            output / f"predictions_{path.stem}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for stage_id, stage_name in enumerate(STAGE_NAMES):
            mask = stages == stage_id
            if not np.any(mask):
                continue
            row = extended_metrics(y[mask], prediction[mask])
            row.update(
                {
                    "checkpoint": path.name,
                    "experiment": experiment,
                    "stage": stage_name,
                    "engine_count": int(mask.sum()),
                    "true_rul_mean": float(y[mask].mean()),
                    "predicted_rul_mean": float(prediction[mask].mean()),
                }
            )
            stage_rows.append(row)
        print(json.dumps(overall, ensure_ascii=False, indent=2))

    overall_frame = pd.DataFrame(overall_rows).sort_values("rmse")
    stage_frame = pd.DataFrame(stage_rows).sort_values(["experiment", "stage"])
    overall_path = output / "experiment7_overall_metrics.csv"
    stage_path = output / "experiment7_stage_metrics.csv"
    overall_frame.to_csv(overall_path, index=False, encoding="utf-8-sig")
    stage_frame.to_csv(stage_path, index=False, encoding="utf-8-sig")
    print("\n[overall]")
    print(overall_frame.to_string(index=False))
    print(f"\nOverall: {overall_path}\nStage metrics: {stage_path}")


if __name__ == "__main__":
    main()
