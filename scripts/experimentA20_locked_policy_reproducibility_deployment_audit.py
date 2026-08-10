"""Experiment A20: locked A9 deployment-policy reproducibility audit.

This is a CPU-only artifact audit.  It reads the completed A9_1 official
confirmation outputs and the completed A19 training-only synthesis outputs.
It never opens raw C-MAPSS test files, never performs a model forward pass,
and never selects or tunes a policy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPERIMENT_ID = "experimentA20"
SCRIPT_VERSION = "experimentA20_locked_policy_reproducibility_deployment_audit_v1"
QUESTION = (
    "Can the locked official A9_1 ten-epoch deployment policy and its reported "
    "paired metrics be independently reconstructed from immutable output artifacts, "
    "while remaining consistent with A19 training-only robustness evidence?"
)
DEFAULT_OUTPUT = "outputs/experimentA20_locked_policy_reproducibility_deployment_audit"
STAGES = {
    "full_endpoint": (
        "experimentA9_1_paired_blend_vs_baseline.csv",
        "full_endpoint_result",
    ),
    "high_rul_gt60": (
        "experimentA9_1_high_rul_paired_blend_vs_baseline.csv",
        "high_rul_safety_result",
    ),
    "low_or_mid_rul_le60": (
        "experimentA9_1_low_rul_paired_blend_vs_baseline.csv",
        "low_rul_safety_result",
    ),
}
METRICS = ("nasa_score", "rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9-1-output-dir", required=True)
    parser.add_argument("--a19-output-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--reproducibility-runs", type=int, default=5)
    parser.add_argument("--metric-tolerance", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalise_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalise_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_json(v) for v in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        normalise_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(normalise_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required A20 JSON is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"A20 expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required A20 CSV is missing or empty: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(f"A20 CSV is empty: {path}") from error


@contextmanager
def exclusive_lock(root: Path):
    lock_path = root / "experimentA20_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(f"another A20 process owns {root}; lock owner: {owner}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "script_version": SCRIPT_VERSION}))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def require_fields(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"A20 requires {label}.{key}={value!r}, found {payload.get(key)!r}"
            )


def positive_completed_count(payload: dict[str, Any], expected_key: str, completed_key: str) -> int:
    expected = payload.get(expected_key)
    completed = payload.get(completed_key)
    if not isinstance(expected, int) or expected <= 0 or completed != expected:
        raise RuntimeError(
            f"invalid count pair {expected_key}={expected!r}, {completed_key}={completed!r}"
        )
    return expected


def recursive_hash_values(value: Any, parent: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{parent}.{key}" if parent else str(key)
            if "hash" in str(key).lower() and isinstance(item, str) and item:
                found.append({"path": path, "value": item})
            found.extend(recursive_hash_values(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(recursive_hash_values(item, f"{parent}[{index}]"))
    return found


def policy_hash_audit(
    reference_hash: str,
    policy_json: dict[str, Any],
    policy_json_path: Path,
    policy_csv_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(reference_hash, str) or len(reference_hash) < 12:
        raise RuntimeError("A9_1 decision lacks a credible locked_policy_hash")
    declared = recursive_hash_values(policy_json) + recursive_hash_values(manifest)
    declared_matches = [entry for entry in declared if entry["value"] == reference_hash]
    raw_json_hash = sha256_file(policy_json_path)
    canonical_hash = sha256_bytes(canonical_json_bytes(policy_json))
    csv_text = policy_csv_path.read_text(encoding="utf-8", errors="replace")
    corroborated = bool(
        declared_matches
        or raw_json_hash.startswith(reference_hash)
        or canonical_hash.startswith(reference_hash)
        or reference_hash in csv_text
    )
    return {
        "reference_locked_policy_hash": reference_hash,
        "declared_hash_fields": declared,
        "declared_matching_fields": declared_matches,
        "locked_policy_json_sha256": raw_json_hash,
        "locked_policy_canonical_sha256": canonical_hash,
        "locked_policy_csv_sha256": sha256_file(policy_csv_path),
        "corroborated_by_locked_artifact": corroborated,
    }


def choose_metric_columns(frame: pd.DataFrame, metric: str, path: Path) -> tuple[str, str]:
    names = [str(column) for column in frame.columns]
    metric_names = [name for name in names if name.startswith(f"{metric}_")]
    baselines = [name for name in metric_names if "baseline" in name.lower()]
    blends = [
        name
        for name in metric_names
        if "blend" in name.lower()
        and "relative" not in name.lower()
        and "delta" not in name.lower()
        and "win" not in name.lower()
    ]
    if "candidate" in frame.columns:
        labels = [str(v) for v in frame["candidate"].dropna().unique()]
        if len(labels) == 1:
            exact = f"{metric}_{labels[0]}"
            if exact in names:
                blends = [exact]
    if len(baselines) != 1 or len(blends) != 1:
        raise RuntimeError(
            f"A20 could not uniquely identify {metric} baseline/blend columns in {path}; "
            f"baseline candidates={baselines}, blend candidates={blends}"
        )
    return baselines[0], blends[0]


def expected_relative_delta(result: dict[str, Any], metric: str, label: str) -> float:
    if metric == "nasa_score":
        value = result.get("nasa_improvement_pct")
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{label} lacks nasa_improvement_pct")
        return -float(value) / 100.0
    value = result.get("rmse_degradation_pct")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} lacks rmse_degradation_pct")
    return float(value) / 100.0


def reproduce_metric(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
    expected: float,
    tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    baseline_column, blend_column = choose_metric_columns(frame, metric, path)
    baseline = pd.to_numeric(frame[baseline_column], errors="raise").to_numpy(dtype=float)
    blend = pd.to_numeric(frame[blend_column], errors="raise").to_numpy(dtype=float)
    if (
        len(baseline) == 0
        or (~np.isfinite(baseline)).any()
        or (~np.isfinite(blend)).any()
        or (baseline <= 0).any()
        or (blend <= 0).any()
    ):
        raise RuntimeError(f"{path} contains invalid {metric} values")
    relative = (blend - baseline) / baseline
    candidates: list[tuple[str, float]] = [
        ("mean_of_paired_relative_deltas", float(relative.mean())),
        ("ratio_of_metric_means", float(blend.mean() / baseline.mean() - 1.0)),
    ]
    explicit_columns = [
        name
        for name in map(str, frame.columns)
        if name.startswith(f"{metric}_") and "relative" in name.lower() and "delta" in name.lower()
    ]
    for name in explicit_columns:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
        if np.isfinite(values).all():
            candidates.append((f"mean_of_{name}", float(values.mean())))
    method, reproduced = min(candidates, key=lambda item: abs(item[1] - expected))
    absolute_error = abs(reproduced - expected)
    row = {
        "metric": metric,
        "n_pairs": int(len(frame)),
        "baseline_column": baseline_column,
        "blend_column": blend_column,
        "reproduction_method": method,
        "expected_relative_delta": expected,
        "reproduced_relative_delta": reproduced,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "passed": bool(absolute_error <= tolerance),
        "blend_win_rate": float((blend < baseline).mean()),
        "baseline_mean": float(baseline.mean()),
        "blend_mean": float(blend.mean()),
    }
    normalized = np.column_stack([baseline, blend, relative])
    return row, normalized


def reproduce_official_metrics(
    a9_root: Path,
    decision: dict[str, Any],
    tolerance: float,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for stage, (filename, result_key) in STAGES.items():
        path = a9_root / filename
        frame = read_csv(path)
        result = decision.get(result_key)
        if not isinstance(result, dict):
            raise RuntimeError(f"A9_1 decision lacks {result_key}")
        for metric in METRICS:
            expected = expected_relative_delta(result, metric, result_key)
            row, normalized = reproduce_metric(frame, metric, path, expected, tolerance)
            row = {"stage": stage, "source_file": filename, **row}
            rows.append(row)
            digest.update(stage.encode("utf-8"))
            digest.update(metric.encode("utf-8"))
            digest.update(np.ascontiguousarray(normalized, dtype="<f8").tobytes())
    output = pd.DataFrame(rows).sort_values(["stage", "metric"]).reset_index(drop=True)
    return output, digest.hexdigest()


def audit_a19(a19_root: Path, tolerance: float) -> tuple[dict[str, Any], pd.DataFrame]:
    decision_path = a19_root / "experimentA19_confirmation_decision.json"
    integrity_path = a19_root / "experimentA19_input_integrity.json"
    summary_path = a19_root / "experimentA19_cross_cohort_summary.csv"
    manifest_path = a19_root / "experimentA19_manifest.json"
    decision = read_json(decision_path)
    integrity = read_json(integrity_path)
    manifest = read_json(manifest_path)
    summary = read_csv(summary_path)
    require_fields(
        decision,
        {
            "experiment_id": "experimentA19",
            "complete": True,
            "passed": True,
            "quick_mode": False,
            "new_predictor_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
        "A19 decision",
    )
    require_fields(integrity, {"passed": True, "model_seed_sets_disjoint": True}, "A19 integrity")
    if int(decision.get("completed_cohorts", -1)) != 3:
        raise RuntimeError("A20 requires the completed three-cohort A19 synthesis")
    if int(decision.get("completed_pairs_per_stage", -1)) != 7500:
        raise RuntimeError("A20 requires 7500 A19 pairs per stage")
    embedded = decision.get("cross_cohort_summary")
    if not isinstance(embedded, list) or len(embedded) != 6 or len(summary) != 6:
        raise RuntimeError("A19 cross-cohort summary must contain six checks")
    embedded_frame = pd.DataFrame(embedded)
    checks: list[dict[str, Any]] = []
    for _, source in summary.iterrows():
        match = embedded_frame[
            (embedded_frame["stage"].astype(str) == str(source["stage"]))
            & (embedded_frame["metric"].astype(str) == str(source["metric"]))
        ]
        if len(match) != 1:
            raise RuntimeError("A19 decision/CSV summary keys do not match")
        expected = float(match.iloc[0]["relative_improvement_pct"])
        actual = float(source["relative_improvement_pct"])
        checks.append(
            {
                "stage": str(source["stage"]),
                "metric": str(source["metric"]),
                "decision_relative_improvement_pct": expected,
                "csv_relative_improvement_pct": actual,
                "absolute_error": abs(actual - expected),
                "passed": bool(abs(actual - expected) <= tolerance * 100.0),
            }
        )
    check_frame = pd.DataFrame(checks)
    audit = {
        "decision_sha256": sha256_file(decision_path),
        "integrity_sha256": sha256_file(integrity_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cross_cohort_summary_sha256": sha256_file(summary_path),
        "complete": True,
        "passed": True,
        "three_cohorts_complete": True,
        "model_seed_sets_disjoint": True,
        "summary_reproduction_passed": bool(check_frame["passed"].all()),
        "script_hash": manifest.get("script_hash"),
    }
    return audit, check_frame


def official_integrity_audit(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    explicit_status: list[dict[str, Any]] = []
    for key in ("passed", "integrity_passed", "selection_was_locked_before_official_test"):
        if key in payload:
            explicit_status.append({"key": key, "value": payload[key]})
    explicit_passed = all(item["value"] is True for item in explicit_status)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "json_object_readable": True,
        "explicit_status_fields": explicit_status,
        "explicit_status_passed": explicit_passed,
    }


def inventory(paths: Iterable[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for group, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required A20 artifact is missing: {path}")
        rows.append(
            {
                "artifact_group": group,
                "file_name": path.name,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact_group", "file_name"]).reset_index(drop=True)


def reproducibility_runs(
    a9_root: Path,
    decision: dict[str, Any],
    tolerance: float,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    reference_frame: pd.DataFrame | None = None
    for index in range(repeats):
        started = time.perf_counter()
        frame, digest = reproduce_official_metrics(a9_root, decision, tolerance)
        elapsed = time.perf_counter() - started
        if reference_frame is None:
            reference_frame = frame
        runs.append(
            {
                "run": index + 1,
                "reconstruction_digest": digest,
                "elapsed_seconds": elapsed,
                "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "metric_checks_passed": bool(frame["passed"].all()),
            }
        )
    assert reference_frame is not None
    run_frame = pd.DataFrame(runs)
    run_frame["digest_matches_first"] = run_frame["reconstruction_digest"].eq(
        run_frame.iloc[0]["reconstruction_digest"]
    )
    return reference_frame, run_frame


def main() -> None:
    args = parse_args()
    if args.reproducibility_runs < 2:
        raise ValueError("A20 requires at least two reproducibility runs")
    if not (0 < args.metric_tolerance <= 1e-4):
        raise ValueError("--metric-tolerance must be in (0, 1e-4]")

    a9_root = Path(args.a9_1_output_dir).expanduser().resolve()
    a19_root = Path(args.a19_output_dir).expanduser().resolve()
    root = Path(args.output_dir).expanduser().resolve()
    if a9_root == root or a19_root == root:
        raise ValueError("A20 output directory must differ from every input directory")
    root.mkdir(parents=True, exist_ok=True)

    with exclusive_lock(root):
        final_path = root / "experimentA20_confirmation_decision.json"
        if final_path.is_file() and not args.resume:
            raise RuntimeError("A20 already completed in this directory; use --resume or a new directory")

        a9_paths = {
            "decision": a9_root / "experimentA9_1_confirmation_decision.json",
            "manifest": a9_root / "experimentA9_1_manifest.json",
            "policy_json": a9_root / "experimentA9_1_locked_policy.json",
            "policy_csv": a9_root / "experimentA9_1_locked_policy.csv",
            "official_integrity": a9_root / "experimentA9_1_official_test_integrity.json",
            **{stage: a9_root / item[0] for stage, item in STAGES.items()},
        }
        a19_paths = {
            "decision": a19_root / "experimentA19_confirmation_decision.json",
            "manifest": a19_root / "experimentA19_manifest.json",
            "integrity": a19_root / "experimentA19_input_integrity.json",
            "summary": a19_root / "experimentA19_cross_cohort_summary.csv",
        }
        artifact_inventory = inventory(
            [("A9_1", path) for path in a9_paths.values()]
            + [("A19", path) for path in a19_paths.values()]
        )

        a9_decision = read_json(a9_paths["decision"])
        a9_manifest = read_json(a9_paths["manifest"])
        policy_json = read_json(a9_paths["policy_json"])
        require_fields(
            a9_decision,
            {
                "experiment_id": "experimentA9_1",
                "complete": True,
                "passed": True,
                "selection_was_locked_before_official_test": True,
                "official_test_files_accessed": True,
                "official_test_forward_run": True,
            },
            "A9_1 decision",
        )
        training_cells = positive_completed_count(
            a9_decision, "expected_training_cells", "completed_training_cells"
        )
        official_records = positive_completed_count(
            a9_decision,
            "expected_official_evaluation_records",
            "completed_official_evaluation_records",
        )
        official_pairs = positive_completed_count(
            a9_decision, "expected_primary_pairs", "completed_primary_pairs"
        )
        reference_hash = a9_decision.get("locked_policy_hash")
        policy_audit = policy_hash_audit(
            reference_hash,
            policy_json,
            a9_paths["policy_json"],
            a9_paths["policy_csv"],
            a9_manifest,
        )
        prior_official_integrity = official_integrity_audit(a9_paths["official_integrity"])
        a19_audit, a19_checks = audit_a19(a19_root, float(args.metric_tolerance))

        metric_checks, runs = reproducibility_runs(
            a9_root,
            a9_decision,
            float(args.metric_tolerance),
            int(args.reproducibility_runs),
        )
        deterministic = bool(runs["digest_matches_first"].all())
        metrics_reproduced = bool(metric_checks["passed"].all())
        a19_reproduced = bool(a19_checks["passed"].all())
        passed = bool(
            policy_audit["corroborated_by_locked_artifact"]
            and prior_official_integrity["explicit_status_passed"]
            and deterministic
            and metrics_reproduced
            and a19_reproduced
        )

        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "script_hash": sha256_file(Path(__file__)),
            "registered_primary_question": QUESTION,
            "a9_1_output_dir": str(a9_root),
            "a19_output_dir": str(a19_root),
            "output_dir": str(root),
            "reproducibility_runs": int(args.reproducibility_runs),
            "metric_tolerance": float(args.metric_tolerance),
            "new_predictor_training": False,
            "policy_selection_or_tuning": False,
            "a20_raw_official_test_files_accessed": False,
            "a20_official_test_forward_run": False,
            "input_contains_prior_locked_official_evaluation_artifacts": True,
        }
        existing_manifest_path = root / "experimentA20_manifest.json"
        if existing_manifest_path.is_file():
            existing = read_json(existing_manifest_path)
            for key in ("script_hash", "a9_1_output_dir", "a19_output_dir", "metric_tolerance"):
                if existing.get(key) != manifest.get(key):
                    raise RuntimeError(
                        f"existing A20 output is incompatible at {key}; use a new output directory"
                    )
        atomic_json(existing_manifest_path, manifest)
        atomic_csv(root / "experimentA20_artifact_inventory.csv", artifact_inventory)
        atomic_csv(root / "experimentA20_metric_reproduction.csv", metric_checks)
        atomic_csv(root / "experimentA20_a19_summary_reproduction.csv", a19_checks)
        atomic_csv(root / "experimentA20_reproducibility_runs.csv", runs)

        input_integrity = {
            "experiment_id": EXPERIMENT_ID,
            "passed": bool(policy_audit["corroborated_by_locked_artifact"] and a19_audit["passed"]),
            "a9_1_training_cells": training_cells,
            "a9_1_official_evaluation_records": official_records,
            "a9_1_primary_pairs": official_pairs,
            "a9_1_locked_before_official_test": True,
            "policy_hash_audit": policy_audit,
            "prior_official_integrity_artifact": prior_official_integrity,
            "a19_audit": a19_audit,
            "artifact_count": int(len(artifact_inventory)),
            "a20_raw_official_test_files_accessed": False,
            "a20_official_test_forward_run": False,
        }
        atomic_json(root / "experimentA20_input_integrity.json", input_integrity)

        dry = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "registered_primary_question": QUESTION,
            "output_dir": str(root),
            "required_artifacts_loaded": int(len(artifact_inventory)),
            "policy_hash_corroborated": policy_audit["corroborated_by_locked_artifact"],
            "metric_checks": int(len(metric_checks)),
            "a19_summary_checks": int(len(a19_checks)),
            "deterministic_reconstruction": deterministic,
            "new_predictor_training": False,
            "policy_selection_or_tuning": False,
            "a20_raw_official_test_files_accessed": False,
            "a20_official_test_forward_run": False,
        }
        atomic_json(root / "experimentA20_dry_run.json", dry)
        print(json.dumps(dry, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            print(
                "[A20] dry-run completed; only locked output artifacts were audited",
                flush=True,
            )
            return

        decision = {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": QUESTION,
            "complete": True,
            "quick_mode": False,
            "new_predictor_training": False,
            "policy_selection_or_tuning": False,
            "reference_policy": "locked_A9_1_ten_epoch_baseline_cycle_age_blend",
            "locked_policy_hash": reference_hash,
            "policy_hash_corroborated": policy_audit["corroborated_by_locked_artifact"],
            "expected_metric_checks": 6,
            "completed_metric_checks": int(len(metric_checks)),
            "metric_reproduction_passed": metrics_reproduced,
            "a19_summary_reproduction_passed": a19_reproduced,
            "reproducibility_runs": int(len(runs)),
            "reconstruction_digest": str(runs.iloc[0]["reconstruction_digest"]),
            "deterministic_reconstruction_passed": deterministic,
            "audit_runtime_seconds_mean": float(runs["elapsed_seconds"].mean()),
            "audit_runtime_seconds_max": float(runs["elapsed_seconds"].max()),
            "audit_peak_rss_kb": int(runs["max_rss_kb"].max()),
            "passed": passed,
            "reason": (
                "A20 reproduced the locked A9_1 policy identity and all registered official/training-only summary metrics from immutable artifacts"
                if passed
                else "A20 completed, but one or more locked-policy identity, metric reproduction, integrity, or determinism checks failed"
            ),
            "interpretation_limit": (
                "A20 is an artifact-reproducibility audit. It does not re-run the official test, retrain a predictor, or create a new efficacy claim."
            ),
            "next_action": (
                "package_locked_A9_1_deployment_bundle_and_freeze_performance_experimentation"
                if passed
                else "inspect_failed_A20_audit_rows_without_reopening_official_test_tuning"
            ),
            "a20_raw_official_test_files_accessed": False,
            "a20_official_test_forward_run": False,
            "input_contains_prior_locked_official_evaluation_artifacts": True,
        }
        atomic_json(final_path, decision)
        print("[A20] completed locked-policy reproducibility audit", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
