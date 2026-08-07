"""Experiment A11: explain A10's source-ablation robustness boundary.

A11 is a training-only, post-hoc mechanism diagnostic.  It does not train a
new predictor, choose new blend parameters, or access official C-MAPSS test
files.  It combines the twelve registered A10 source-ablation outcomes with:

1. target-to-active-source operating-setting coverage;
2. sensor and causal-cycle-age distribution shift;
3. A10 blend-selection behaviour; and
4. baseline/cycle-age prediction disagreement on confirmation engines.

The registered primary diagnostic asks whether A10 failure severity increases
with either (a) a source-coverage risk index or (b) confirmation prediction
disagreement.  With only twelve ablation conditions, all inference is reported
as exploratory mechanism evidence and is not used to tune the official test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_VERSION = "experimentA11_source_coverage_failure_mechanism_v1"
EXPERIMENT_ID = "experimentA11"
DEFAULT_A10_OUTPUT = "outputs/experimentA10_source_domain_ablation_robustness"
DEFAULT_OUTPUT = "outputs/experimentA11_source_coverage_failure_mechanism"

DOMAINS = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_COLUMNS = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    *[f"s{i}" for i in range(1, 22)],
]
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
DEFAULT_SENSOR_COLUMNS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]
PAIR_KEYS = [
    "target_domain",
    "heldout_source_domain",
    "model_seed",
    "target_split_seed",
    "role_partition",
    "endpoint_seed",
]
CONDITION_KEYS = ["target_domain", "heldout_source_domain"]
QUESTION = (
    "Are A10 source-ablation failures concentrated in conditions with poorer "
    "target-to-source coverage or larger baseline/cycle-age prediction disagreement?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A11 source-coverage failure mechanism diagnostic"
    )
    parser.add_argument(
        "--a10-output-dir",
        default=DEFAULT_A10_OUTPUT,
        help="completed experimentA10 output directory",
    )
    parser.add_argument(
        "--data-dir",
        help="C-MAPSS directory containing train_FD001.txt ... train_FD004.txt",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rows-per-domain", type=int, default=2500)
    parser.add_argument("--permutation-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    text = ":".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16) % (2**31 - 1)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty input is missing: {path}")
    return pd.read_csv(path)


def a10_paths(root: Path) -> dict[str, Path]:
    prefix = "experimentA10"
    return {
        "manifest": root / f"{prefix}_manifest.json",
        "protocol": root / f"{prefix}_protocol.json",
        "decision": root / f"{prefix}_confirmation_decision.json",
        "causality": root / f"{prefix}_feature_causality_audit.json",
        "ablation": root / f"{prefix}_source_ablation_summary.csv",
        "parameters": root / f"{prefix}_blend_parameters.csv",
        "grid": root / f"{prefix}_blend_selection_grid.csv",
        "prediction": root / f"{prefix}_confirmation_endpoint_predictions.csv",
        "run": root / f"{prefix}_confirmation_run_level.csv",
        "inventory": root / f"{prefix}_source_pretrain_inventory.csv",
        "age_audit": root / f"{prefix}_age_feature_audit.csv",
        "high_pairs": root / f"{prefix}_high_rul_paired_blend_vs_baseline.csv",
        "low_pairs": root / f"{prefix}_low_rul_paired_blend_vs_baseline.csv",
    }


def output_paths(root: Path) -> dict[str, Path]:
    prefix = EXPERIMENT_ID
    return {
        "manifest": root / f"{prefix}_manifest.json",
        "protocol": root / f"{prefix}_protocol.json",
        "dry_run": root / f"{prefix}_dry_run.json",
        "input_integrity": root / f"{prefix}_input_integrity.json",
        "stage_audit": root / f"{prefix}_stage_pair_audit.csv",
        "coverage": root / f"{prefix}_source_coverage_metrics.csv",
        "behaviour": root / f"{prefix}_blend_behaviour_metrics.csv",
        "conditions": root / f"{prefix}_condition_diagnostics.csv",
        "associations": root / f"{prefix}_association_tests.csv",
        "targets": root / f"{prefix}_target_domain_summary.csv",
        "heldouts": root / f"{prefix}_heldout_source_summary.csv",
        "decision": root / f"{prefix}_confirmation_decision.json",
        "report": root / f"{prefix}_analysis_report_CN.txt",
    }


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, received {value!r}")
    return [str(item) for item in parsed]


def parse_ci_upper(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        parsed = json.loads(str(value))
    if len(parsed) != 2:
        raise ValueError(f"invalid confidence interval: {value!r}")
    return float(parsed[1])


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(series[converted.isna()].astype(str).unique())
        raise ValueError(f"cannot parse boolean values: {bad}")
    return converted.astype(bool)


def load_train_domain(data_dir: Path, domain: str) -> pd.DataFrame:
    candidates = [
        data_dir / domain / f"train_{domain}.txt",
        data_dir / f"train_{domain}.txt",
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"missing training file train_{domain}.txt below {data_dir}; "
            "A11 never searches for test_ or RUL_ files"
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < len(CMAPSS_COLUMNS):
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected at least 26")
    frame = frame.iloc[:, : len(CMAPSS_COLUMNS)].copy()
    frame.columns = CMAPSS_COLUMNS
    return frame


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("sample size must be positive")
    if len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=n, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def safe_scale(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(source, axis=0)
    std = np.nanstd(source, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean, std


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Return Euclidean nearest-reference distances without a large full matrix."""
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("nearest-distance inputs must be non-empty")
    answers: list[np.ndarray] = []
    for start in range(0, len(query), chunk):
        current = query[start : start + chunk]
        squared = np.sum((current[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        answers.append(np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)))
    return np.concatenate(answers)


def quantile_rmse(left: np.ndarray, right: np.ndarray) -> float:
    probabilities = np.linspace(0.01, 0.99, 99)
    left_q = np.quantile(left, probabilities)
    right_q = np.quantile(right, probabilities)
    return float(np.sqrt(np.mean((left_q - right_q) ** 2)))


def active_sources(target: str, heldout: str) -> list[str]:
    available = [domain for domain in DOMAINS if domain != target]
    if heldout not in available:
        raise ValueError(f"{heldout} is not an available source for target {target}")
    active = [domain for domain in available if domain != heldout]
    if len(active) != 2:
        raise AssertionError("A11 expects exactly two active A10 source domains")
    return active


def coverage_metrics(
    training: dict[str, pd.DataFrame],
    sensor_columns: list[str],
    sample_rows: int,
    seed: int,
) -> pd.DataFrame:
    sampled = {
        domain: deterministic_sample(
            training[domain], sample_rows, stable_seed(seed, domain, "domain_sample")
        )
        for domain in DOMAINS
    }
    rows: list[dict[str, Any]] = []
    for target in DOMAINS:
        for heldout in [domain for domain in DOMAINS if domain != target]:
            sources = active_sources(target, heldout)
            source_full = pd.concat([training[d] for d in sources], ignore_index=True)
            source_sample = pd.concat([sampled[d] for d in sources], ignore_index=True)
            target_sample = sampled[target]

            setting_source_full = source_full[SETTING_COLUMNS].to_numpy(float)
            setting_mean, setting_std = safe_scale(setting_source_full)
            source_setting = (
                source_sample[SETTING_COLUMNS].to_numpy(float) - setting_mean
            ) / setting_std
            target_setting = (
                target_sample[SETTING_COLUMNS].to_numpy(float) - setting_mean
            ) / setting_std

            rng = np.random.default_rng(stable_seed(seed, target, heldout, "setting_split"))
            order = rng.permutation(len(source_setting))
            midpoint = max(1, len(order) // 2)
            calibration = source_setting[order[:midpoint]]
            reference = source_setting[order[midpoint:]]
            if len(reference) == 0:
                reference = calibration
            internal_distance = nearest_distances(calibration, reference)
            coverage_radius = float(np.quantile(internal_distance, 0.95))
            target_distance = nearest_distances(target_setting, source_setting)

            sensor_source_full = source_full[sensor_columns].to_numpy(float)
            sensor_mean, sensor_std = safe_scale(sensor_source_full)
            source_sensor = (
                source_sample[sensor_columns].to_numpy(float) - sensor_mean
            ) / sensor_std
            target_sensor = (
                target_sample[sensor_columns].to_numpy(float) - sensor_mean
            ) / sensor_std
            mean_shift = float(
                np.linalg.norm(np.nanmean(target_sensor, axis=0) - np.nanmean(source_sensor, axis=0))
                / np.sqrt(len(sensor_columns))
            )
            source_cov = np.cov(source_sensor, rowvar=False)
            target_cov = np.cov(target_sensor, rowvar=False)
            covariance_shift = float(
                np.linalg.norm(target_cov - source_cov, ord="fro") / len(sensor_columns)
            )
            sensor_shift = float(np.sqrt(mean_shift**2 + covariance_shift**2))

            source_age = np.log1p(source_sample["cycle"].to_numpy(float))
            target_age = np.log1p(target_sample["cycle"].to_numpy(float))
            age_mean, age_std = safe_scale(source_age[:, None])
            source_age_z = (source_age - age_mean[0]) / age_std[0]
            target_age_z = (target_age - age_mean[0]) / age_std[0]

            rows.append(
                {
                    "target_domain": target,
                    "heldout_source_domain": heldout,
                    "active_source_domains": json.dumps(sources),
                    "target_training_rows": int(len(training[target])),
                    "active_source_training_rows": int(len(source_full)),
                    "sampled_target_rows": int(len(target_sample)),
                    "sampled_source_rows": int(len(source_sample)),
                    "setting_internal_p95_radius": coverage_radius,
                    "setting_target_nn_mean": float(np.mean(target_distance)),
                    "setting_target_nn_p95": float(np.quantile(target_distance, 0.95)),
                    "setting_coverage_rate": float(np.mean(target_distance <= coverage_radius)),
                    "setting_coverage_gap": float(1.0 - np.mean(target_distance <= coverage_radius)),
                    "sensor_mean_shift": mean_shift,
                    "sensor_covariance_shift": covariance_shift,
                    "sensor_shift_score": sensor_shift,
                    "age_mean_shift_abs": float(abs(np.mean(target_age_z) - np.mean(source_age_z))),
                    "age_std_ratio_abs_log": float(
                        abs(np.log((np.std(target_age_z) + 1e-8) / (np.std(source_age_z) + 1e-8)))
                    ),
                    "age_quantile_shift": quantile_rmse(target_age_z, source_age_z),
                }
            )
    output = pd.DataFrame(rows).sort_values(CONDITION_KEYS).reset_index(drop=True)
    risk_columns = [
        "setting_target_nn_p95",
        "setting_coverage_gap",
        "sensor_shift_score",
        "age_quantile_shift",
    ]
    standardized: list[np.ndarray] = []
    for column in risk_columns:
        values = output[column].to_numpy(float)
        std = float(np.std(values))
        standardized.append((values - float(np.mean(values))) / (std if std > 1e-12 else 1.0))
    output["source_coverage_risk_index"] = np.mean(np.column_stack(standardized), axis=1)
    return output


def behaviour_metrics(parameters: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    required_parameter_columns = {
        *CONDITION_KEYS,
        "alpha_high",
        "alpha_low",
        "selection_safety_feasible",
        "fallback_to_baseline",
        "selection_gate_high_rate",
        "confirmation_used_for_alpha_selection",
    }
    missing = required_parameter_columns - set(parameters.columns)
    if missing:
        raise ValueError(f"A10 blend-parameter columns missing: {sorted(missing)}")
    parameters = parameters.copy()
    parameters["selection_safety_feasible"] = bool_series(parameters["selection_safety_feasible"])
    parameters["fallback_to_baseline"] = bool_series(parameters["fallback_to_baseline"])
    parameters["confirmation_used_for_alpha_selection"] = bool_series(
        parameters["confirmation_used_for_alpha_selection"]
    )
    if parameters["confirmation_used_for_alpha_selection"].any():
        raise RuntimeError("A10 confirmation labels were used for alpha selection")

    parameter_rows: list[dict[str, Any]] = []
    for keys, frame in parameters.groupby(CONDITION_KEYS, sort=True):
        parameter_rows.append(
            {
                "target_domain": keys[0],
                "heldout_source_domain": keys[1],
                "blend_parameter_sets": int(len(frame)),
                "alpha_high_mean": float(frame["alpha_high"].mean()),
                "alpha_high_std": float(frame["alpha_high"].std(ddof=0)),
                "alpha_low_mean": float(frame["alpha_low"].mean()),
                "alpha_low_std": float(frame["alpha_low"].std(ddof=0)),
                "alpha_instability": float(
                    np.sqrt(
                        frame["alpha_high"].var(ddof=0) + frame["alpha_low"].var(ddof=0)
                    )
                ),
                "fallback_to_baseline_rate": float(frame["fallback_to_baseline"].mean()),
                "selection_safety_infeasible_rate": float(
                    1.0 - frame["selection_safety_feasible"].mean()
                ),
                "selection_gate_high_rate_mean": float(
                    frame["selection_gate_high_rate"].mean()
                ),
            }
        )
    parameter_summary = pd.DataFrame(parameter_rows)

    prediction_required = {
        *PAIR_KEYS,
        "unit",
        "label",
        "prediction_baseline",
        "prediction_cycle_age",
        "prediction_blend",
        "prediction_gate",
    }
    missing = prediction_required - set(predictions.columns)
    if missing:
        raise ValueError(f"A10 confirmation-prediction columns missing: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["absolute_model_disagreement"] = (
        predictions["prediction_cycle_age"] - predictions["prediction_baseline"]
    ).abs()
    predictions["absolute_blend_correction"] = (
        predictions["prediction_blend"] - predictions["prediction_baseline"]
    ).abs()
    predictions["baseline_error"] = predictions["prediction_baseline"] - predictions["label"]
    predictions["blend_error"] = predictions["prediction_blend"] - predictions["label"]
    predictions["true_high_rul"] = predictions["label"] > 60.0
    predictions["predicted_high_gate"] = predictions["prediction_gate"].astype(str).str.contains(
        "high", case=False, regex=False
    )

    prediction_rows: list[dict[str, Any]] = []
    for keys, frame in predictions.groupby(CONDITION_KEYS, sort=True):
        high = frame[frame["true_high_rul"]]
        low = frame[~frame["true_high_rul"]]
        prediction_rows.append(
            {
                "target_domain": keys[0],
                "heldout_source_domain": keys[1],
                "confirmation_prediction_rows": int(len(frame)),
                "confirmation_pair_count": int(frame[PAIR_KEYS].drop_duplicates().shape[0]),
                "prediction_disagreement_mean": float(frame["absolute_model_disagreement"].mean()),
                "prediction_disagreement_p95": float(
                    frame["absolute_model_disagreement"].quantile(0.95)
                ),
                "blend_correction_mean": float(frame["absolute_blend_correction"].mean()),
                "blend_correction_p95": float(frame["absolute_blend_correction"].quantile(0.95)),
                "predicted_high_gate_rate": float(frame["predicted_high_gate"].mean()),
                "true_high_rul_rate": float(frame["true_high_rul"].mean()),
                "high_rul_baseline_bias": float(high["baseline_error"].mean()),
                "high_rul_blend_bias": float(high["blend_error"].mean()),
                "low_mid_rul_baseline_bias": float(low["baseline_error"].mean()),
                "low_mid_rul_blend_bias": float(low["blend_error"].mean()),
            }
        )
    prediction_summary = pd.DataFrame(prediction_rows)
    return parameter_summary.merge(prediction_summary, on=CONDITION_KEYS, validate="one_to_one")


def stage_pair_audit(
    predictions: pd.DataFrame,
    high_pairs: pd.DataFrame,
    low_pairs: pd.DataFrame,
    expected_pairs: int,
) -> pd.DataFrame:
    pair_frame = predictions[PAIR_KEYS + ["label"]].copy()
    grouped = pair_frame.groupby(PAIR_KEYS, sort=True)["label"]
    stage_presence = grouped.agg(
        has_high=lambda values: bool((values > 60.0).any()),
        has_low_mid=lambda values: bool((values <= 60.0).any()),
        endpoint_engine_rows="size",
    ).reset_index()
    observed_pairs = len(stage_presence)
    saved_high_unique = (
        high_pairs[PAIR_KEYS].drop_duplicates().shape[0]
        if set(PAIR_KEYS).issubset(high_pairs.columns)
        else 0
    )
    saved_low_unique = (
        low_pairs[PAIR_KEYS].drop_duplicates().shape[0]
        if set(PAIR_KEYS).issubset(low_pairs.columns)
        else 0
    )
    rows = [
        {
            "audit_item": "confirmation_prediction_unique_pairs",
            "expected": expected_pairs,
            "observed": observed_pairs,
            "passed": observed_pairs == expected_pairs,
        },
        {
            "audit_item": "pairs_with_high_rul_engines",
            "expected": expected_pairs,
            "observed": int(stage_presence["has_high"].sum()),
            "passed": bool(stage_presence["has_high"].all()),
        },
        {
            "audit_item": "pairs_with_low_mid_rul_engines",
            "expected": expected_pairs,
            "observed": int(stage_presence["has_low_mid"].sum()),
            "passed": bool(stage_presence["has_low_mid"].all()),
        },
        {
            "audit_item": "saved_high_rul_pair_rows",
            "expected": expected_pairs,
            "observed": saved_high_unique,
            "passed": len(high_pairs) == expected_pairs and saved_high_unique == expected_pairs,
        },
        {
            "audit_item": "saved_low_mid_rul_pair_rows",
            "expected": expected_pairs,
            "observed": saved_low_unique,
            "passed": len(low_pairs) == expected_pairs and saved_low_unique == expected_pairs,
        },
    ]
    return pd.DataFrame(rows)


def outcome_metrics(ablation: pd.DataFrame) -> pd.DataFrame:
    required = {
        *CONDITION_KEYS,
        "active_source_domains",
        "n_pairs",
        "full_rmse_improvement_pct",
        "full_rmse_ci95",
        "high_nasa_ci95",
        "high_rmse_ci95",
        "low_nasa_ci95",
        "low_rmse_ci95",
        "robust_condition_passed",
    }
    missing = required - set(ablation.columns)
    if missing:
        raise ValueError(f"A10 ablation columns missing: {sorted(missing)}")
    output = ablation.copy()
    if len(output) != 12 or output[CONDITION_KEYS].drop_duplicates().shape[0] != 12:
        raise RuntimeError("A11 formal analysis requires all twelve A10 source-ablation conditions")
    output["robust_condition_passed"] = bool_series(output["robust_condition_passed"])
    for source, destination in [
        ("full_rmse_ci95", "full_rmse_ci95_upper"),
        ("high_nasa_ci95", "high_nasa_ci95_upper"),
        ("high_rmse_ci95", "high_rmse_ci95_upper"),
        ("low_nasa_ci95", "low_nasa_ci95_upper"),
        ("low_rmse_ci95", "low_rmse_ci95_upper"),
    ]:
        output[destination] = output[source].map(parse_ci_upper)
    constraint_excess = np.column_stack(
        [
            output["full_rmse_ci95_upper"].to_numpy(float),
            output["high_nasa_ci95_upper"].to_numpy(float) - 0.03,
            output["high_rmse_ci95_upper"].to_numpy(float) - 0.03,
            output["low_nasa_ci95_upper"].to_numpy(float) - 0.03,
            output["low_rmse_ci95_upper"].to_numpy(float) - 0.03,
        ]
    )
    output["failure_severity"] = np.max(constraint_excess, axis=1)
    output["robustness_slack"] = -output["failure_severity"]
    output["condition_failed"] = ~output["robust_condition_passed"]
    reconstructed_pass = output["failure_severity"] <= 1e-12
    if not np.array_equal(reconstructed_pass.to_numpy(bool), output["robust_condition_passed"].to_numpy(bool)):
        raise RuntimeError("A10 pass flags do not match the registered CI constraints")
    return output


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = rank_values(left), rank_values(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def association_tests(
    conditions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if repetitions < 100:
        raise ValueError("permutation repetitions must be at least 100")
    primary = {"source_coverage_risk_index", "prediction_disagreement_p95"}
    features = [
        "source_coverage_risk_index",
        "prediction_disagreement_p95",
        "setting_target_nn_p95",
        "setting_coverage_gap",
        "sensor_shift_score",
        "age_quantile_shift",
        "alpha_instability",
        "fallback_to_baseline_rate",
        "selection_safety_infeasible_rate",
        "blend_correction_p95",
    ]
    severity = conditions["failure_severity"].to_numpy(float)
    failed = conditions["condition_failed"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = conditions[feature].to_numpy(float)
        observed_rho = spearman_rho(values, severity)
        scale = float(np.std(values))
        standardized = (values - float(np.mean(values))) / (scale if scale > 1e-12 else 1.0)
        failed_minus_passed = float(standardized[failed].mean() - standardized[~failed].mean())
        rng = np.random.default_rng(stable_seed(seed, feature, "permutation"))
        rho_exceed = 0
        effect_exceed = 0
        for _ in range(repetitions):
            permuted_severity = rng.permutation(severity)
            if spearman_rho(values, permuted_severity) >= observed_rho - 1e-15:
                rho_exceed += 1
            permuted_failed = rng.permutation(failed)
            permuted_effect = float(
                standardized[permuted_failed].mean() - standardized[~permuted_failed].mean()
            )
            if permuted_effect >= failed_minus_passed - 1e-15:
                effect_exceed += 1
        rows.append(
            {
                "feature": feature,
                "registered_primary_feature": feature in primary,
                "hypothesized_direction": "higher_feature_higher_failure_severity",
                "spearman_rho_with_failure_severity": observed_rho,
                "spearman_one_sided_permutation_p": (rho_exceed + 1) / (repetitions + 1),
                "failed_minus_passed_standardized_mean": failed_minus_passed,
                "binary_effect_one_sided_permutation_p": (effect_exceed + 1)
                / (repetitions + 1),
                "n_conditions": len(conditions),
            }
        )
    output = pd.DataFrame(rows)
    primary_mask = output["registered_primary_feature"]
    output["primary_holm_p"] = np.nan
    output.loc[primary_mask, "primary_holm_p"] = holm_adjust(
        output.loc[primary_mask, "spearman_one_sided_permutation_p"]
    )
    output["registered_mechanism_supported"] = (
        output["registered_primary_feature"]
        & (output["spearman_rho_with_failure_severity"] >= 0.50)
        & (output["primary_holm_p"] <= 0.05)
    )
    return output.sort_values(
        ["registered_primary_feature", "spearman_one_sided_permutation_p"],
        ascending=[False, True],
    ).reset_index(drop=True)


def group_summary(conditions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, frame in conditions.groupby(column, sort=True):
        rows.append(
            {
                column: value,
                "condition_count": int(len(frame)),
                "passing_condition_count": int(frame["robust_condition_passed"].sum()),
                "passing_condition_rate": float(frame["robust_condition_passed"].mean()),
                "failure_severity_mean": float(frame["failure_severity"].mean()),
                "full_rmse_improvement_pct_mean": float(
                    frame["full_rmse_improvement_pct"].mean()
                ),
                "source_coverage_risk_index_mean": float(
                    frame["source_coverage_risk_index"].mean()
                ),
                "prediction_disagreement_p95_mean": float(
                    frame["prediction_disagreement_p95"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_a10_inputs(
    paths: dict[str, Path],
    decision: dict[str, Any],
    manifest: dict[str, Any],
    causality: dict[str, Any],
    ablation: pd.DataFrame,
    parameters: pd.DataFrame,
    grid: pd.DataFrame,
    predictions: pd.DataFrame,
    run_level: pd.DataFrame,
    inventory: pd.DataFrame,
    age_audit: pd.DataFrame,
    stage_audit: pd.DataFrame,
) -> dict[str, Any]:
    expected_pairs = int(decision.get("expected_confirmation_pairs", -1))
    expected_parameters = int(decision.get("expected_blend_parameter_sets", -1))
    expected_training = int(decision.get("expected_training_cells", -1))
    unique_prediction_pairs = predictions[PAIR_KEYS].drop_duplicates().shape[0]
    unique_run_pairs = run_level[PAIR_KEYS].drop_duplicates().shape[0]
    unique_inventory = inventory[
        ["target_domain", "heldout_source_domain", "model_seed", "representation"]
    ].drop_duplicates().shape[0]
    checks = {
        "a10_complete": bool(decision.get("complete")),
        "a10_formal_not_quick": not bool(decision.get("quick_mode")),
        "official_test_not_accessed_decision": not bool(
            decision.get("official_test_files_accessed")
        ),
        "official_test_not_forward_run_decision": not bool(
            decision.get("official_test_forward_run")
        ),
        "official_test_not_accessed_manifest": not bool(
            manifest.get("official_test_files_accessed")
        ),
        "official_test_not_forward_run_manifest": not bool(
            manifest.get("official_test_forward_run")
        ),
        "causal_feature_no_unit_max_cycle": not bool(causality.get("uses_unit_max_cycle")),
        "causal_feature_no_true_rul": not bool(causality.get("uses_true_rul_as_feature")),
        "causal_feature_no_future_windows": not bool(causality.get("uses_future_windows")),
        "confirmation_not_used_for_alpha": not bool(
            causality.get("confirmation_used_for_alpha_selection")
        ),
        "twelve_ablation_conditions": len(ablation) == 12,
        "expected_parameter_sets": len(parameters) == expected_parameters,
        "expected_grid_rows": len(grid) == expected_parameters * 25,
        "expected_prediction_pairs": unique_prediction_pairs == expected_pairs,
        "expected_run_level_pairs": unique_run_pairs == expected_pairs,
        "expected_source_inventory_rows": unique_inventory == 120,
        "expected_age_audit_rows": len(age_audit) == 120,
        "training_cells_registered": expected_training == 600,
    }
    warnings = {
        "saved_stage_pair_files_complete": bool(stage_audit["passed"].all()),
    }
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "a10_experiment_id": decision.get("experiment_id"),
        "checks": checks,
        "warnings": warnings,
        "expected_confirmation_pairs": expected_pairs,
        "observed_confirmation_prediction_pairs": unique_prediction_pairs,
        "observed_confirmation_run_pairs": unique_run_pairs,
        "expected_blend_parameter_sets": expected_parameters,
        "observed_blend_parameter_sets": len(parameters),
        "observed_blend_grid_rows": len(grid),
        "observed_source_inventory_rows": unique_inventory,
        "observed_age_audit_rows": len(age_audit),
        "all_checks_passed": bool(all(checks.values())),
        "all_warnings_clear": bool(all(warnings.values())),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
        "a10_input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }
    if not audit["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("A11 input-integrity checks failed: " + ", ".join(failed))
    return audit


def make_report(
    decision: dict[str, Any],
    conditions: pd.DataFrame,
    associations: pd.DataFrame,
    targets: pd.DataFrame,
) -> str:
    primary = associations[associations["registered_primary_feature"]]
    lines = [
        "Experiment A11：源域覆盖失败机制诊断报告",
        "=" * 56,
        "",
        f"预注册问题：{QUESTION}",
        "",
        "一、运行与数据完整性",
        f"- A10消融条件：{len(conditions)}",
        f"- A10通过条件：{int(conditions['robust_condition_passed'].sum())}/{len(conditions)}",
        "- 官方测试文件访问：否",
        "- 新模型训练或参数调优：否",
        "",
        "二、目标域边界",
    ]
    for row in targets.itertuples(index=False):
        lines.append(
            f"- {row.target_domain}: {row.passing_condition_count}/{row.condition_count} 条件通过，"
            f"平均失败严重度={row.failure_severity_mean:.4f}"
        )
    lines.extend(["", "三、预注册机制检验"])
    for row in primary.itertuples(index=False):
        lines.append(
            f"- {row.feature}: Spearman rho={row.spearman_rho_with_failure_severity:.4f}, "
            f"单侧置换p={row.spearman_one_sided_permutation_p:.5f}, "
            f"Holm p={row.primary_holm_p:.5f}, 支持={bool(row.registered_mechanism_supported)}"
        )
    lines.extend(
        [
            "",
            "四、结论",
            f"- A11机制诊断是否通过：{decision['passed']}",
            f"- 原因：{decision['reason']}",
            f"- 下一步：{decision['next_action']}",
            "",
            "说明：A11是基于12个既有消融条件的探索性机制诊断，不能单独证明因果关系。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    a10_root = resolve_project_path(args.a10_output_dir)
    output_root = resolve_project_path(args.output_dir)
    paths = a10_paths(a10_root)
    outputs = output_paths(output_root)

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("A11 requires completed A10 files:\n" + "\n".join(missing))

    a10_decision = read_json(paths["decision"])
    a10_manifest = read_json(paths["manifest"])
    a10_causality = read_json(paths["causality"])
    manifest_data_dir = a10_manifest.get("base_config", {}).get("data_dir")
    data_candidates = []
    if args.data_dir:
        data_candidates.append(resolve_project_path(args.data_dir))
    if manifest_data_dir:
        data_candidates.append(Path(manifest_data_dir).expanduser())
    data_candidates.append(PROJECT_ROOT / "data")
    data_dir = next(
        (
            candidate.resolve()
            for candidate in data_candidates
            if (candidate / "train_FD001.txt").is_file()
            or (candidate / "FD001" / "train_FD001.txt").is_file()
        ),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(
            "could not resolve a training-only C-MAPSS data directory; use --data-dir"
        )

    dry = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "a10_output_dir": str(a10_root),
        "data_dir": str(data_dir),
        "output_dir": str(output_root),
        "expected_ablation_conditions": 12,
        "expected_confirmation_pairs": a10_decision.get("expected_confirmation_pairs"),
        "sample_rows_per_domain": int(args.sample_rows_per_domain),
        "permutation_repetitions": int(args.permutation_repetitions),
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(outputs["dry_run"], dry)
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return

    ablation = read_csv(paths["ablation"])
    parameters = read_csv(paths["parameters"])
    grid = read_csv(paths["grid"])
    predictions = read_csv(paths["prediction"])
    run_level = read_csv(paths["run"])
    inventory = read_csv(paths["inventory"])
    age_audit = read_csv(paths["age_audit"])
    high_pairs = read_csv(paths["high_pairs"])
    low_pairs = read_csv(paths["low_pairs"])

    expected_pairs = int(a10_decision["expected_confirmation_pairs"])
    stage_audit = stage_pair_audit(predictions, high_pairs, low_pairs, expected_pairs)
    integrity = validate_a10_inputs(
        paths,
        a10_decision,
        a10_manifest,
        a10_causality,
        ablation,
        parameters,
        grid,
        predictions,
        run_level,
        inventory,
        age_audit,
        stage_audit,
    )

    sensor_columns = list(
        a10_manifest.get("base_config", {}).get("sensor_columns", DEFAULT_SENSOR_COLUMNS)
    )
    invalid_sensors = [column for column in sensor_columns if column not in CMAPSS_COLUMNS]
    if invalid_sensors:
        raise ValueError(f"invalid A10 sensor columns: {invalid_sensors}")
    training = {domain: load_train_domain(data_dir, domain) for domain in DOMAINS}
    coverage = coverage_metrics(training, sensor_columns, args.sample_rows_per_domain, args.seed)
    behaviour = behaviour_metrics(parameters, predictions)
    outcomes = outcome_metrics(ablation)
    conditions = outcomes.merge(coverage, on=CONDITION_KEYS, validate="one_to_one").merge(
        behaviour, on=CONDITION_KEYS, validate="one_to_one"
    )
    conditions = conditions.sort_values(CONDITION_KEYS).reset_index(drop=True)

    associations = association_tests(
        conditions,
        repetitions=int(args.permutation_repetitions),
        seed=int(args.seed),
    )
    targets = group_summary(conditions, "target_domain")
    heldouts = group_summary(conditions, "heldout_source_domain")
    supported = associations[associations["registered_mechanism_supported"]]
    passed = bool(len(supported) > 0)

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "registered_primary_question": QUESTION,
        "analysis_scope": "training_only_posthoc_mechanism_diagnostic",
        "primary_features": [
            "source_coverage_risk_index",
            "prediction_disagreement_p95",
        ],
        "primary_success_criteria": {
            "direction": "positive association with A10 failure severity",
            "minimum_spearman_rho": 0.50,
            "maximum_holm_adjusted_one_sided_permutation_p": 0.05,
            "rule": "at least one of two registered primary features must pass",
        },
        "failure_severity": (
            "max(full_rmse_CI_upper, high_nasa_CI_upper-0.03, "
            "high_rmse_CI_upper-0.03, low_nasa_CI_upper-0.03, "
            "low_rmse_CI_upper-0.03)"
        ),
        "source_coverage_risk_index_components": [
            "setting_target_nn_p95",
            "setting_coverage_gap",
            "sensor_shift_score",
            "age_quantile_shift",
        ],
        "sample_rows_per_domain": int(args.sample_rows_per_domain),
        "permutation_repetitions": int(args.permutation_repetitions),
        "seed": int(args.seed),
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "script_hash": file_sha256(Path(__file__).resolve()),
        "a10_output_dir": str(a10_root),
        "data_dir": str(data_dir),
        "output_dir": str(output_root),
        "sensor_columns": sensor_columns,
        "a10_input_hashes": integrity["a10_input_hashes"],
        "registered_primary_question": QUESTION,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": QUESTION,
        "complete": True,
        "a10_conditions": int(len(conditions)),
        "a10_passing_conditions": int(conditions["robust_condition_passed"].sum()),
        "a10_failing_conditions": int(conditions["condition_failed"].sum()),
        "input_integrity_passed": bool(integrity["all_checks_passed"]),
        "registered_primary_results": associations[
            associations["registered_primary_feature"]
        ][
            [
                "feature",
                "spearman_rho_with_failure_severity",
                "spearman_one_sided_permutation_p",
                "primary_holm_p",
                "registered_mechanism_supported",
            ]
        ].to_dict(orient="records"),
        "supported_primary_mechanisms": supported["feature"].tolist(),
        "passed": passed,
        "reason": (
            "A11 found registered evidence linking A10 failure severity to source coverage or model disagreement"
            if passed
            else "A11 completed, but neither registered mechanism met the effect-size and Holm-adjusted permutation criteria"
        ),
        "next_action": (
            "design_experimentA12_coverage_aware_source_weighting_training_only"
            if passed
            else "report_A10_source_dependence_boundary_and_stop_source_coverage_modeling"
        ),
        "interpretation_limit": (
            "post-hoc diagnostic over twelve ablation conditions; association is not causal proof"
        ),
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }

    atomic_json(outputs["manifest"], manifest)
    atomic_json(outputs["protocol"], protocol)
    atomic_json(outputs["input_integrity"], integrity)
    atomic_text(outputs["stage_audit"], stage_audit.to_csv(index=False))
    atomic_text(outputs["coverage"], coverage.to_csv(index=False))
    atomic_text(outputs["behaviour"], behaviour.to_csv(index=False))
    atomic_text(outputs["conditions"], conditions.to_csv(index=False))
    atomic_text(outputs["associations"], associations.to_csv(index=False))
    atomic_text(outputs["targets"], targets.to_csv(index=False))
    atomic_text(outputs["heldouts"], heldouts.to_csv(index=False))
    atomic_json(outputs["decision"], decision)
    atomic_text(outputs["report"], make_report(decision, conditions, associations, targets))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
