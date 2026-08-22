#!/usr/bin/env python3
"""A25.0: frozen compute-fairness and low-RUL risk audit.

This program is intentionally an analysis-only audit.  It consumes frozen
A23.3/A24.x/A24.4 artifacts and does *not* train, adapt, tune, or score an
official test set.  Its purpose is to establish what training-budget evidence
is available and to localise the low-RUL risk of Meta-noGraph versus ordinary
pretrain+finetune before a new independent cohort is designed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ID = "experimentA25_0"
SCRIPT_VERSION = "experimentA25_0_meta_fairness_and_low_rul_risk_audit_v1"
LOW_RUL_ANCHOR = 15.0
PRIMARY_SHOTS = (1, 2, 5)
TAIL_FRACTIONS = (0.01, 0.05)

PFT_METHOD = "pretrain_finetune_k"
META_METHOD = "meta_no_graph_k"

PREDICTION_REQUIRED = (
    "target_domain", "model_seed", "support_split_seed", "shot", "engine_id",
    "prefix_label", "registered_rul_anchor", "true_rul", "rul_stage",
    f"error__{PFT_METHOD}", f"absolute_error__{PFT_METHOD}",
    f"squared_error__{PFT_METHOD}", f"nasa_score_component__{PFT_METHOD}",
    f"error__{META_METHOD}", f"absolute_error__{META_METHOD}",
    f"squared_error__{META_METHOD}", f"nasa_score_component__{META_METHOD}",
)
PAIR_KEYS = (
    "target_domain", "model_seed", "support_split_seed", "shot", "engine_id",
    "prefix_label", "registered_rul_anchor",
)


class ContractError(RuntimeError):
    """Raised when frozen inputs do not satisfy the A25.0 audit contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen compute-fairness and low-RUL tail-risk audit."
    )
    parser.add_argument("--a23-0-protocol-dir", type=Path, required=True)
    parser.add_argument("--a23-3-output-dir", type=Path, required=True)
    parser.add_argument("--a24-0-output-dir", type=Path, required=True)
    parser.add_argument("--a24-1-output-dir", type=Path, required=True)
    parser.add_argument("--a24-2-output-dir", type=Path, required=True)
    parser.add_argument("--a24-4-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tail-fractions",
        default="0.01,0.05",
        help="Comma-separated per-method NASA-tail fractions, default: 0.01,0.05.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    try:
        fractions = tuple(float(value.strip()) for value in args.tail_fractions.split(","))
    except ValueError as exc:
        parser.error(f"invalid --tail-fractions: {exc}")
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        parser.error("all --tail-fractions values must be strictly between 0 and 1")
    if tuple(sorted(set(fractions))) != fractions:
        parser.error("--tail-fractions must be strictly increasing with no duplicates")
    args.tail_fractions = fractions
    return args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON input {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{label} lacks required columns: {missing}")


def find_exact(root: Path, filename: str, required: bool = True) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename)) if root.is_dir() else []
    if not matches:
        if required:
            raise ContractError(f"cannot find {filename} under {root}")
        return None
    require(len(matches) == 1,
            f"ambiguous {filename} under {root}: {[str(path) for path in matches]}")
    return matches[0]


def find_first(root: Path, patterns: Sequence[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    unique = sorted(set(matches))
    if not unique:
        return None
    require(len(unique) == 1,
            f"ambiguous file discovery under {root}: {[str(path) for path in unique]}")
    return unique[0]


def read_manifest_and_decision(root: Path, prefix: str, expected_id: str) -> tuple[
        Path, dict[str, Any], Path, dict[str, Any]]:
    manifest_path = find_exact(root, f"{prefix}_manifest.json")
    decision_path = find_exact(root, f"{prefix}_confirmation_decision.json")
    assert manifest_path is not None and decision_path is not None
    manifest = load_json(manifest_path)
    decision = load_json(decision_path)
    require(manifest.get("experiment_id") == expected_id,
            f"{manifest_path}: expected experiment_id={expected_id}")
    require(decision.get("experiment_id") == expected_id,
            f"{decision_path}: expected experiment_id={expected_id}")
    require(decision.get("complete") is True, f"incomplete frozen input: {decision_path}")
    require(decision.get("official_test_files_accessed") is False,
            f"{decision_path}: official test access is not allowed")
    require(decision.get("official_test_forward_run") is False,
            f"{decision_path}: official test forward run is not allowed")
    return manifest_path, manifest, decision_path, decision


def verify_declared_artifact(root: Path, manifest: Mapping[str, Any],
                             filename: str) -> str:
    artifacts = manifest.get("artifacts", {})
    require(isinstance(artifacts, dict), f"manifest artifact section invalid: {root}")
    expected = artifacts.get(filename)
    require(isinstance(expected, str) and len(expected) == 64,
            f"manifest lacks SHA-256 for {filename}: {root}")
    path = find_exact(root, filename)
    assert path is not None
    observed = sha256_file(path)
    require(observed == expected,
            f"SHA-256 mismatch for {path}: expected={expected}, observed={observed}")
    return observed


def parse_list_count(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                if text == "":
                    return 0
                return None
        return parse_list_count(parsed)
    return None


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        require(np.isfinite(frame[column].to_numpy(dtype=float)).all(),
                f"{label}:{column} contains NaN or infinity")


def collect_scalar_values(value: Any, keys: set[str], output: list[tuple[str, float]]) -> None:
    """Collect numeric values from arbitrary JSON records without guessing semantics."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, (int, float)) and np.isfinite(float(item)):
                output.append((key, float(item)))
            collect_scalar_values(item, keys, output)
    elif isinstance(value, list):
        for item in value:
            collect_scalar_values(item, keys, output)


def discover_json_scalars(root: Path, keys: set[str]) -> dict[str, list[float]]:
    found: list[tuple[str, float]] = []
    # Only JSON evidence is read; checkpoints and tensors are intentionally untouched.
    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except ContractError:
            continue
        collect_scalar_values(payload, keys, found)
    result: dict[str, list[float]] = {key: [] for key in keys}
    for key, value in found:
        result[key].append(value)
    return result


def unique_or_none(values: Iterable[float]) -> float | None:
    unique = sorted({float(value) for value in values})
    if len(unique) == 1:
        return unique[0]
    return None


@dataclass(frozen=True)
class Inputs:
    a23_0_manifest: dict[str, Any]
    a23_0_decision: dict[str, Any]
    a23_3_manifest: dict[str, Any]
    a23_3_decision: dict[str, Any]
    a24_0_manifest: dict[str, Any]
    a24_0_decision: dict[str, Any]
    a24_1_manifest: dict[str, Any]
    a24_1_decision: dict[str, Any]
    a24_2_manifest: dict[str, Any]
    a24_2_decision: dict[str, Any]
    a24_4_manifest: dict[str, Any]
    a24_4_decision: dict[str, Any]
    hashes: dict[str, str]


def preflight(args: argparse.Namespace) -> Inputs:
    for name in (
        "a23_0_protocol_dir", "a23_3_output_dir", "a24_0_output_dir",
        "a24_1_output_dir", "a24_2_output_dir", "a24_4_output_dir",
    ):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        require(path.is_dir(), f"required directory is missing: {path}")
    args.output_dir = args.output_dir.expanduser().resolve()

    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    specifications = (
        ("a23_0", args.a23_0_protocol_dir, "experimentA23", "experimentA23_0"),
        ("a23_3", args.a23_3_output_dir, "experimentA23_3", "experimentA23_3"),
        ("a24_0", args.a24_0_output_dir, "experimentA24_0", "experimentA24_0"),
        ("a24_1", args.a24_1_output_dir, "experimentA24_1", "experimentA24_1"),
        ("a24_2", args.a24_2_output_dir, "experimentA24_2", "experimentA24_2"),
        ("a24_4", args.a24_4_output_dir, "experimentA24_4", "experimentA24_4"),
    )
    for short, root, prefix, experiment_id in specifications:
        manifest_path, manifest, decision_path, decision = read_manifest_and_decision(
            root, prefix, experiment_id
        )
        values[f"{short}_manifest"] = manifest
        values[f"{short}_decision"] = decision
        hashes[f"{short}_manifest_sha256"] = sha256_file(manifest_path)
        hashes[f"{short}_decision_sha256"] = sha256_file(decision_path)

    hashes["a23_3_training_run_level_sha256"] = verify_declared_artifact(
        args.a23_3_output_dir, values["a23_3_manifest"],
        "experimentA23_3_training_run_level.csv",
    )
    hashes["a24_4_normalized_pairs_sha256"] = verify_declared_artifact(
        args.a24_4_output_dir, values["a24_4_manifest"],
        "experimentA24_4_normalized_four_method_pairs.csv",
    )
    return Inputs(hashes=hashes, **values)


def load_pft_training(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_exact(root, "experimentA23_3_training_run_level.csv")
    assert path is not None
    frame = pd.read_csv(path)
    required = (
        "regime", "target_domain", "model_seed", "support_split_seed", "shot",
        "source_cache_path", "source_pretrain_steps", "target_epochs", "support_engines",
        "feature_count", "window_size",
    )
    require_columns(frame, required, str(path))
    pft = frame.loc[frame["regime"].eq(PFT_METHOD)].copy()
    require(not pft.empty, f"{path} has no {PFT_METHOD} records")
    finite_numeric(pft, ("model_seed", "support_split_seed", "shot", "source_pretrain_steps",
                         "target_epochs", "feature_count", "window_size"), str(path))
    pft["support_engine_count"] = pft["support_engines"].map(parse_list_count)
    require(pft["support_engine_count"].notna().all(),
            f"{path}: could not parse one or more support_engines values")
    pft["support_engine_count"] = pft["support_engine_count"].astype(int)
    require((pft["support_engine_count"] > 0).all(),
            f"{path}: a PFT record has no support engine")
    pft["target_engine_epochs"] = pft["support_engine_count"] * pft["target_epochs"]

    cache = pft[["target_domain", "model_seed", "source_cache_path", "source_pretrain_steps",
                 "feature_count", "window_size"]].drop_duplicates()
    require(cache.duplicated(["target_domain", "model_seed"], keep=False).sum() == 0,
            "PFT source cache is not unique per target-domain/model-seed")
    return pft, cache


def load_normalized_pairs(root: Path) -> pd.DataFrame:
    path = find_exact(root, "experimentA24_4_normalized_four_method_pairs.csv")
    assert path is not None
    frame = pd.read_csv(path)
    require_columns(frame, PREDICTION_REQUIRED, str(path))
    finite_numeric(frame, [
        "model_seed", "support_split_seed", "shot", "engine_id",
        "registered_rul_anchor", "true_rul",
        f"error__{PFT_METHOD}", f"absolute_error__{PFT_METHOD}",
        f"squared_error__{PFT_METHOD}", f"nasa_score_component__{PFT_METHOD}",
        f"error__{META_METHOD}", f"absolute_error__{META_METHOD}",
        f"squared_error__{META_METHOD}", f"nasa_score_component__{META_METHOD}",
    ], str(path))
    require((frame[f"squared_error__{PFT_METHOD}"] >= 0).all(),
            "PFT squared errors must be non-negative")
    require((frame[f"squared_error__{META_METHOD}"] >= 0).all(),
            "Meta-noGraph squared errors must be non-negative")
    require((frame[f"nasa_score_component__{PFT_METHOD}"] >= 0).all(),
            "PFT NASA components must be non-negative")
    require((frame[f"nasa_score_component__{META_METHOD}"] >= 0).all(),
            "Meta-noGraph NASA components must be non-negative")
    require(not frame.duplicated(list(PAIR_KEYS), keep=False).any(),
            "A24.4 normalized pairs contain duplicate engine endpoints")
    require(set(frame["registered_rul_anchor"].unique()) == {15.0, 45.0, 90.0},
            "unexpected RUL anchors in A24.4 normalized pairs")
    return frame


def json_record_values(root: Path) -> dict[str, float | None]:
    keys = {
        "outer_steps", "inner_steps", "final_outer_step", "meta_outer_steps",
        "meta_inner_steps", "total_parameters", "trainable_parameters",
        "graph_increment_parameters", "wall_time_seconds", "runtime_seconds",
    }
    values = discover_json_scalars(root, keys)
    aliases = {
        "outer_steps": ("outer_steps", "meta_outer_steps", "final_outer_step"),
        "inner_steps": ("inner_steps", "meta_inner_steps"),
        "total_parameters": ("total_parameters",),
        "trainable_parameters": ("trainable_parameters",),
        "graph_increment_parameters": ("graph_increment_parameters",),
        "wall_time_seconds": ("wall_time_seconds", "runtime_seconds"),
    }
    result: dict[str, float | None] = {}
    for standard, names in aliases.items():
        collected: list[float] = []
        for name in names:
            collected.extend(values.get(name, []))
        result[standard] = unique_or_none(collected)
    return result


def compute_fairness_audit(pft: pd.DataFrame, cache: pd.DataFrame,
                           a24_1_root: Path, a24_2_root: Path,
                           a24_1_decision: Mapping[str, Any],
                           a24_2_decision: Mapping[str, Any]) -> pd.DataFrame:
    """Report auditable quantities without equating algorithmically different steps."""
    pft_source_step = unique_or_none(cache["source_pretrain_steps"].tolist())
    pft_target_epoch = unique_or_none(pft["target_epochs"].tolist())
    pft_features = unique_or_none(pft["feature_count"].tolist())
    pft_window = unique_or_none(pft["window_size"].tolist())
    meta_pilot = json_record_values(a24_1_root)
    meta_formal = json_record_values(a24_2_root)
    meta_records = a24_2_decision.get("completed_training_records")
    meta_cells = a24_2_decision.get("completed_worker_cells")
    pilot_records = a24_1_decision.get("completed_run_records")

    records: list[dict[str, Any]] = []
    def add(family: str, method: str, quantity: str, value: Any, unit: str,
            origin: str, status: str, note: str) -> None:
        records.append({
            "experiment_id": EXPERIMENT_ID,
            "method_family": family,
            "method": method,
            "quantity": quantity,
            "value": value,
            "unit": unit,
            "evidence_origin": origin,
            "availability": status,
            "interpretation": note,
            "formal_efficacy_claim": False,
        })

    add("ordinary_transfer", PFT_METHOD, "source_cache_entries", len(cache), "unique caches",
        "A23.3 source-cache/train-level artifacts", "available",
        "One source cache is reused across target support splits.")
    add("ordinary_transfer", PFT_METHOD, "declared_source_pretrain_steps_per_cache",
        pft_source_step, "optimizer steps", "A23.3 training_run_level", "available",
        "Declared supervised source-pretraining steps; not an episode count.")
    add("ordinary_transfer", PFT_METHOD, "declared_total_source_pretrain_steps",
        None if pft_source_step is None else int(len(cache) * pft_source_step), "optimizer steps",
        "A23.3 training_run_level", "available",
        "Cache count multiplied by declared steps; this is not directly comparable to Reptile inner updates.")
    add("ordinary_transfer", PFT_METHOD, "target_adapted_model_records", len(pft), "checkpoints",
        "A23.3 training_run_level", "available", "One PFT checkpoint record per factorial condition.")
    add("ordinary_transfer", PFT_METHOD, "declared_target_epochs_per_record", pft_target_epoch,
        "epochs", "A23.3 training_run_level", "available", "Epochs are not optimizer steps without batch counts.")
    add("ordinary_transfer", PFT_METHOD, "total_support_engine_epochs",
        int(pft["target_engine_epochs"].sum()), "engine-epochs", "A23.3 training_run_level",
        "available", "Support-engine count multiplied by target epochs across PFT records.")
    add("ordinary_transfer", PFT_METHOD, "feature_count", pft_features, "features",
        "A23.3 training_run_level", "available", "Input feature count for ordinary PFT.")
    add("ordinary_transfer", PFT_METHOD, "window_size", pft_window, "cycles",
        "A23.3 training_run_level", "available", "Input temporal window length.")

    add("reptile", META_METHOD, "formal_training_records", meta_records, "method-records",
        "A24.2 confirmation decision", "available" if meta_records is not None else "missing",
        "Formal grid records; this is not directly an update count.")
    add("reptile", META_METHOD, "formal_worker_cells", meta_cells, "worker-cells",
        "A24.2 confirmation decision", "available" if meta_cells is not None else "missing",
        "Frozen formal training worker cells.")
    add("reptile", META_METHOD, "pilot_method_records", pilot_records, "method-records",
        "A24.1 confirmation decision", "available" if pilot_records is not None else "missing",
        "Implementation-pilot method records.")
    for quantity, unit, note in (
        ("outer_steps", "meta steps", "A Reptile outer step may contain multiple source tasks."),
        ("inner_steps", "inner optimizer steps", "Inner steps require task/batch evidence before compute comparison."),
        ("total_parameters", "parameters", "Runtime parameter audit evidence, if unambiguous."),
        ("trainable_parameters", "parameters", "Runtime trainable-parameter evidence, if unambiguous."),
        ("wall_time_seconds", "seconds", "Wall-clock is valid only when independently captured for both families."),
    ):
        value = meta_formal.get(quantity)
        origin = "A24.2 JSON history/manifest scan"
        if value is None:
            value = meta_pilot.get(quantity)
            origin = "A24.1 JSON history/manifest scan"
        add("reptile", META_METHOD, quantity, value, unit, origin,
            "available" if value is not None else "missing", note)

    add("cross_method", "PFT_vs_Meta-noGraph", "training_budget_equivalence_established",
        False, "boolean", "A25.0 audit rule", "not_established",
        "Declared epochs, supervised optimizer steps, Reptile outer steps, and inner steps are algorithmically different units; equal numeric values cannot prove equal training opportunity.")
    return pd.DataFrame(records)


def method_tail_rows(frame: pd.DataFrame, method: str, scope: Mapping[str, Any],
                     fractions: Sequence[float]) -> list[dict[str, Any]]:
    error = frame[f"error__{method}"].to_numpy(dtype=float)
    absolute = frame[f"absolute_error__{method}"].to_numpy(dtype=float)
    squared = frame[f"squared_error__{method}"].to_numpy(dtype=float)
    nasa = frame[f"nasa_score_component__{method}"].to_numpy(dtype=float)
    total_nasa = float(nasa.sum())
    rows: list[dict[str, Any]] = []
    base = dict(scope)
    base.update({
        "method": method,
        "n_pairs": len(frame),
        "rmse": float(math.sqrt(squared.mean())),
        "mae": float(absolute.mean()),
        "mean_error": float(error.mean()),
        "median_error": float(np.median(error)),
        "positive_error_rate": float(np.mean(error > 0)),
        "negative_error_rate": float(np.mean(error < 0)),
        "nasa_score": total_nasa,
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "worst_absolute_error": float(absolute.max()),
        "formal_efficacy_claim": False,
    })
    rows.append({**base, "tail_fraction": 0.0, "tail_threshold_nasa_component": np.nan,
                 "tail_nasa_component_share": np.nan, "tail_record_count": np.nan})
    for fraction in fractions:
        threshold = float(np.quantile(nasa, 1.0 - fraction))
        # >= deliberately includes ties; the actual record count is reported.
        mask = nasa >= threshold
        rows.append({
            **base,
            "tail_fraction": float(fraction),
            "tail_threshold_nasa_component": threshold,
            "tail_nasa_component_share": float(nasa[mask].sum() / total_nasa)
            if total_nasa > 0 else 0.0,
            "tail_record_count": int(mask.sum()),
        })
    return rows


def low_rul_tail_summary(pairs: pd.DataFrame, fractions: Sequence[float]) -> pd.DataFrame:
    low = pairs.loc[pairs["registered_rul_anchor"].eq(LOW_RUL_ANCHOR)].copy()
    require(not low.empty, "no RUL=15 endpoints available for tail-risk audit")
    rows: list[dict[str, Any]] = []
    for shot, subset in low.groupby("shot", observed=True, sort=True):
        scope = {
            "experiment_id": EXPERIMENT_ID,
            "analysis_scope": "all_domains",
            "target_domain": "ALL",
            "shot": int(shot),
            "registered_rul_anchor": LOW_RUL_ANCHOR,
            "rul_stage": "low_rul_le30",
        }
        rows.extend(method_tail_rows(subset, PFT_METHOD, scope, fractions))
        rows.extend(method_tail_rows(subset, META_METHOD, scope, fractions))

        for fraction in fractions:
            pft_nasa = subset[f"nasa_score_component__{PFT_METHOD}"].to_numpy(dtype=float)
            meta_nasa = subset[f"nasa_score_component__{META_METHOD}"].to_numpy(dtype=float)
            # Candidate excess is the directly actionable safety quantity.
            excess = meta_nasa - pft_nasa
            threshold = float(np.quantile(excess, 1.0 - fraction))
            mask = excess >= threshold
            rows.append({
                **scope,
                "method": "meta_no_graph_minus_pft",
                "n_pairs": len(subset),
                "rmse": float(math.sqrt(subset[f"squared_error__{META_METHOD}"].mean())) -
                        float(math.sqrt(subset[f"squared_error__{PFT_METHOD}"].mean())),
                "mae": float(subset[f"absolute_error__{META_METHOD}"].mean()) -
                       float(subset[f"absolute_error__{PFT_METHOD}"].mean()),
                "mean_error": float(subset[f"error__{META_METHOD}"].mean()) -
                              float(subset[f"error__{PFT_METHOD}"].mean()),
                "median_error": float(np.median(subset[f"error__{META_METHOD}"])) -
                                float(np.median(subset[f"error__{PFT_METHOD}"])),
                "positive_error_rate": float(np.mean(subset[f"error__{META_METHOD}"] > 0)) -
                                       float(np.mean(subset[f"error__{PFT_METHOD}"] > 0)),
                "negative_error_rate": float(np.mean(subset[f"error__{META_METHOD}"] < 0)) -
                                       float(np.mean(subset[f"error__{PFT_METHOD}"] < 0)),
                "nasa_score": float(meta_nasa.sum() - pft_nasa.sum()),
                "p95_absolute_error": np.nan,
                "p99_absolute_error": np.nan,
                "worst_absolute_error": np.nan,
                "tail_fraction": float(fraction),
                "tail_threshold_nasa_component": threshold,
                "tail_nasa_component_share": float(excess[mask].sum() / excess.sum())
                if excess.sum() > 0 else np.nan,
                "tail_record_count": int(mask.sum()),
                "formal_efficacy_claim": False,
            })
    return pd.DataFrame(rows)


def domain_tail_summary(pairs: pd.DataFrame, fractions: Sequence[float]) -> pd.DataFrame:
    low = pairs.loc[pairs["registered_rul_anchor"].eq(LOW_RUL_ANCHOR)].copy()
    rows: list[dict[str, Any]] = []
    for (domain, shot), subset in low.groupby(["target_domain", "shot"], observed=True, sort=True):
        scope = {
            "experiment_id": EXPERIMENT_ID,
            "analysis_scope": "target_domain",
            "target_domain": str(domain),
            "shot": int(shot),
            "registered_rul_anchor": LOW_RUL_ANCHOR,
            "rul_stage": "low_rul_le30",
        }
        rows.extend(method_tail_rows(subset, PFT_METHOD, scope, fractions))
        rows.extend(method_tail_rows(subset, META_METHOD, scope, fractions))
    return pd.DataFrame(rows)


def engine_risk_inventory(pairs: pd.DataFrame, fractions: Sequence[float]) -> pd.DataFrame:
    low = pairs.loc[pairs["registered_rul_anchor"].eq(LOW_RUL_ANCHOR),
                    list(PAIR_KEYS) + ["true_rul", "rul_stage",
                                       f"error__{PFT_METHOD}",
                                       f"absolute_error__{PFT_METHOD}",
                                       f"squared_error__{PFT_METHOD}",
                                       f"nasa_score_component__{PFT_METHOD}",
                                       f"error__{META_METHOD}",
                                       f"absolute_error__{META_METHOD}",
                                       f"squared_error__{META_METHOD}",
                                       f"nasa_score_component__{META_METHOD}"]].copy()
    low["nasa_component_delta_meta_minus_pft"] = (
        low[f"nasa_score_component__{META_METHOD}"] -
        low[f"nasa_score_component__{PFT_METHOD}"]
    )
    low["squared_error_delta_meta_minus_pft"] = (
        low[f"squared_error__{META_METHOD}"] -
        low[f"squared_error__{PFT_METHOD}"]
    )
    low["error_delta_meta_minus_pft"] = (
        low[f"error__{META_METHOD}"] - low[f"error__{PFT_METHOD}"]
    )
    low["meta_positive_error"] = low[f"error__{META_METHOD}"] > 0
    low["pft_positive_error"] = low[f"error__{PFT_METHOD}"] > 0
    group = ["target_domain", "shot"]
    low["rank_by_meta_nasa_excess"] = low.groupby(group, observed=True)[
        "nasa_component_delta_meta_minus_pft"
    ].rank(method="first", ascending=False).astype(int)
    low["rank_by_meta_squared_error_excess"] = low.groupby(group, observed=True)[
        "squared_error_delta_meta_minus_pft"
    ].rank(method="first", ascending=False).astype(int)
    for fraction in fractions:
        label = f"top_{int(round(fraction * 100))}_pct_meta_nasa_excess"
        counts = low.groupby(group, observed=True)["engine_id"].transform("size")
        cutoff = np.maximum(1, np.ceil(counts * fraction)).astype(int)
        low[label] = low["rank_by_meta_nasa_excess"] <= cutoff
    low.insert(0, "experiment_id", EXPERIMENT_ID)
    low["formal_efficacy_claim"] = False
    return low.sort_values(group + ["rank_by_meta_nasa_excess"], kind="stable")


def make_integrity(inputs: Inputs, pft: pd.DataFrame, cache: pd.DataFrame,
                   pairs: pd.DataFrame, fairness: pd.DataFrame) -> dict[str, Any]:
    pft_protocol_values = pft["protocol_hashes"].astype(str).unique().tolist() \
        if "protocol_hashes" in pft.columns else []
    fairness_complete = bool((fairness["availability"].eq("missing")).sum() == 0)
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "input_hashes": inputs.hashes,
        "a23_0_complete": True,
        "a23_3_complete": True,
        "a24_0_complete": True,
        "a24_1_complete": True,
        "a24_2_complete": True,
        "a24_4_complete": True,
        "a24_4_descriptive_only": True,
        "pft_records": int(len(pft)),
        "pft_source_caches": int(len(cache)),
        "normalized_four_method_pairs": int(len(pairs)),
        "low_rul_pairs": int((pairs["registered_rul_anchor"] == LOW_RUL_ANCHOR).sum()),
        "target_domains": sorted(pairs["target_domain"].astype(str).unique().tolist()),
        "shots": sorted(pairs["shot"].astype(int).unique().tolist()),
        "pft_protocol_hash_record_variants": int(len(pft_protocol_values)),
        "compute_metadata_all_fields_available": fairness_complete,
        "training_budget_equivalence_established": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def make_decision(integrity: Mapping[str, Any], tail: pd.DataFrame,
                  fairness: pd.DataFrame, fractions: Sequence[float]) -> dict[str, Any]:
    low_rows = tail.loc[
        tail["method"].eq("meta_no_graph_minus_pft") &
        tail["shot"].isin(PRIMARY_SHOTS)
    ]
    # One comparison row is emitted per requested tail fraction; metrics repeat.
    unique_low = low_rows.drop_duplicates(["shot"])
    rmse_worse_all_primary = bool((unique_low["rmse"] > 0).all())
    missing_compute = fairness.loc[fairness["availability"].eq("missing"), "quantity"].tolist()
    return {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": (
            "Are the frozen ordinary PFT and Reptile Meta-noGraph training-budget "
            "artifacts sufficiently auditable for a fair independent-cohort design, "
            "and what low-RUL=15 error-tail risk must be addressed before that design?"
        ),
        "complete": True,
        "passed": True,
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "pft_method": PFT_METHOD,
        "meta_method": META_METHOD,
        "low_rul_anchor": LOW_RUL_ANCHOR,
        "primary_shots_reviewed": list(PRIMARY_SHOTS),
        "tail_fractions": list(fractions),
        "compute_fairness_metadata_complete": len(missing_compute) == 0,
        "missing_compute_metadata_fields": missing_compute,
        "training_budget_equivalence_established": False,
        "low_rul_meta_minus_pft_rmse_worse_at_every_primary_shot": rmse_worse_all_primary,
        "low_rul_risk_gate_passed": False,
        "reason": (
            "A25.0 completed a frozen audit. It reports budget evidence and low-RUL "
            "tail risk, but does not claim resource equivalence or a repaired meta-learning policy."
        ),
        "interpretation_limit": (
            "A25.0 reuses previously viewed confirmation outcomes and cannot create "
            "a new efficacy claim or select a deployment policy."
        ),
        "next_action": (
            "define_a_mechanism_specific_low_RUL_safety_fix_before_preregistering_"
            "an_independent_compute_matched_A25_1_cohort"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def dry_run_payload(inputs: Inputs, args: argparse.Namespace) -> dict[str, Any]:
    pft_path = find_exact(args.a23_3_output_dir, "experimentA23_3_training_run_level.csv")
    pair_path = find_exact(args.a24_4_output_dir,
                           "experimentA24_4_normalized_four_method_pairs.csv")
    assert pft_path is not None and pair_path is not None
    pft_header = pd.read_csv(pft_path, nrows=0)
    pairs_header = pd.read_csv(pair_path, nrows=0)
    require_columns(pft_header, ("regime", "source_pretrain_steps", "target_epochs",
                                 "support_engines"), "A23.3 training header")
    require_columns(pairs_header, PREDICTION_REQUIRED, "A24.4 pairs header")
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "output_dir": str(args.output_dir),
        "hash_verified_inputs": sorted(inputs.hashes),
        "audit_methods": [PFT_METHOD, META_METHOD],
        "low_rul_anchor": LOW_RUL_ANCHOR,
        "tail_fractions": list(args.tail_fractions),
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ContractError(
            f"run lock exists: {path}; confirm no A25.0 process is active before "
            "removing a stale lock"
        ) from exc


def run(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.expanduser().resolve()
    decision_path = args.output_dir / "experimentA25_0_confirmation_decision.json"
    if decision_path.is_file() and args.resume:
        decision = load_json(decision_path)
        require(decision.get("complete") is True, "--resume found incomplete output")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        print("[A25.0] existing completed audit reused")
        return 0
    if decision_path.exists() and not args.force:
        raise ContractError(
            f"output already completed: {decision_path}; use --resume or --force"
        )

    inputs = preflight(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(inputs, args), ensure_ascii=False, indent=2))
        print("[A25.0] dry-run passed; no output written and no predictor trained")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "experimentA25_0_run.lock"
    lock_fd = acquire_lock(lock_path)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        print("[A25.0] loading frozen PFT training evidence ...", flush=True)
        pft, cache = load_pft_training(args.a23_3_output_dir)
        print("[A25.0] loading frozen A24.4 four-method pairs ...", flush=True)
        pairs = load_normalized_pairs(args.a24_4_output_dir)

        fairness = compute_fairness_audit(
            pft, cache, args.a24_1_output_dir, args.a24_2_output_dir,
            inputs.a24_1_decision, inputs.a24_2_decision,
        )
        print("[A25.0] decomposing RUL=15 NASA/error tails ...", flush=True)
        tail = low_rul_tail_summary(pairs, args.tail_fractions)
        domain = domain_tail_summary(pairs, args.tail_fractions)
        engines = engine_risk_inventory(pairs, args.tail_fractions)
        integrity = make_integrity(inputs, pft, cache, pairs, fairness)
        decision = make_decision(integrity, tail, fairness, args.tail_fractions)

        outputs: dict[str, pd.DataFrame] = {
            "experimentA25_0_compute_fairness_audit.csv": fairness,
            "experimentA25_0_low_rul_error_tail_summary.csv": tail,
            "experimentA25_0_domain_tail_summary.csv": domain,
            "experimentA25_0_engine_risk_inventory.csv": engines,
        }
        for name, frame in outputs.items():
            atomic_write_csv(args.output_dir / name, frame)
        atomic_write_json(args.output_dir / "experimentA25_0_input_integrity.json", integrity)
        atomic_write_json(decision_path, decision)

        artifact_names = list(outputs) + [
            "experimentA25_0_input_integrity.json",
            "experimentA25_0_confirmation_decision.json",
        ]
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "inputs": inputs.hashes,
            "artifacts": {name: sha256_file(args.output_dir / name)
                          for name in artifact_names},
            "descriptive_only": True,
            "formal_efficacy_claim": False,
            "new_predictor_training": False,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_write_json(args.output_dir / "experimentA25_0_manifest.json", manifest)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        print("[A25.0] completed frozen fairness and low-RUL risk audit", flush=True)
        return 0
    finally:
        os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()


def self_test() -> int:
    records: list[dict[str, Any]] = []
    for domain in ("FD001", "FD002"):
        for shot in (1, 2, 5):
            for engine in range(1, 11):
                pft_error = 4.0 + engine / 10.0
                meta_error = 6.0 + engine / 5.0
                records.append({
                    "target_domain": domain,
                    "model_seed": 1,
                    "support_split_seed": 11,
                    "shot": shot,
                    "engine_id": engine,
                    "prefix_label": "rul_anchor_015",
                    "registered_rul_anchor": 15.0,
                    "true_rul": 15.0,
                    "rul_stage": "low_rul_le30",
                    f"error__{PFT_METHOD}": pft_error,
                    f"absolute_error__{PFT_METHOD}": abs(pft_error),
                    f"squared_error__{PFT_METHOD}": pft_error ** 2,
                    f"nasa_score_component__{PFT_METHOD}": math.exp(pft_error / 10.0) - 1.0,
                    f"error__{META_METHOD}": meta_error,
                    f"absolute_error__{META_METHOD}": abs(meta_error),
                    f"squared_error__{META_METHOD}": meta_error ** 2,
                    f"nasa_score_component__{META_METHOD}": math.exp(meta_error / 10.0) - 1.0,
                })
    pairs = pd.DataFrame(records)
    tail = low_rul_tail_summary(pairs, TAIL_FRACTIONS)
    domain = domain_tail_summary(pairs, TAIL_FRACTIONS)
    engine = engine_risk_inventory(pairs, TAIL_FRACTIONS)
    require(len(tail) == 24, "self-test tail row count mismatch")
    require(len(domain) == 36, "self-test domain row count mismatch")
    require(len(engine) == 60, "self-test engine row count mismatch")
    comparisons = tail.loc[tail["method"].eq("meta_no_graph_minus_pft")]
    require((comparisons["rmse"] > 0).all(), "self-test expected Meta RMSE excess")
    print("[A25.0] self-test passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except (ContractError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[A25.0] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
