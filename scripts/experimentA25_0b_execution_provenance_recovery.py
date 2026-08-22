#!/usr/bin/env python3
"""A25.0b — recover auditable execution provenance from frozen A23/A24 artifacts.

This program deliberately performs *no* predictor training, target adaptation,
hyperparameter selection, confirmation evaluation, or official-test access.  It
only inventories existing files.  Its purpose is to determine whether the
ordinary PFT and Reptile training budgets can be documented faithfully before
designing a new independent cohort.

The scanner is schema-tolerant because A23.3/A24.1/A24.2 may store metadata in
worker_status.json, meta_history.json, manifests, dry-run payloads, or nested
checkpoint dictionaries.  Unknown fields are never invented: they remain
"missing" in the output decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


EXPERIMENT_ID = "experimentA25_0b"
SCRIPT_VERSION = "experimentA25_0b_execution_provenance_recovery_v2"
METHODS = ("meta_no_graph_k", "meta_gnn_k")
REQUIRED_META_FIELDS = (
    "outer_steps",
    "inner_steps",
    "total_parameters",
    "trainable_parameters",
    "wall_time_seconds",
)

FIELD_ALIASES = {
    "outer_steps": {
        "outer_steps", "meta_outer_steps", "final_outer_step", "final_meta_step",
        "outer_step", "meta_step",
    },
    "inner_steps": {"inner_steps", "meta_inner_steps", "inner_step_count"},
    "total_parameters": {"total_parameters", "parameter_count", "num_parameters"},
    "trainable_parameters": {"trainable_parameters", "trainable_parameter_count"},
    "wall_time_seconds": {
        "wall_time_seconds", "runtime_seconds", "elapsed_seconds", "duration_seconds",
        "worker_runtime_seconds", "training_runtime_seconds",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_directory(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def one_named_file(root: Path, filename: str, label: str) -> Path:
    direct = root / filename
    candidates = [direct] if direct.is_file() else sorted(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{label} is missing below {root}: expected {filename}")
    if len(candidates) > 1:
        joined = "\n  - ".join(str(item) for item in candidates)
        raise RuntimeError(f"{label} is ambiguous; retain one canonical file:\n  - {joined}")
    return candidates[0]


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {label}: {path}: {error}") from error


def require_completed_decision(root: Path, experiment: str) -> tuple[Path, dict[str, Any]]:
    path = one_named_file(root, f"{experiment}_confirmation_decision.json", f"{experiment} decision")
    payload = load_json(path, f"{experiment} decision")
    if not isinstance(payload, dict) or payload.get("experiment_id") != experiment:
        raise RuntimeError(f"Unexpected decision schema in {path}")
    if payload.get("complete") is not True:
        raise RuntimeError(f"{experiment} is not complete according to {path}")
    return path, payload


def normalize_method(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower().replace("-", "_")
    if "meta_no_graph" in lowered or "metanograph" in lowered:
        return "meta_no_graph_k"
    if "meta_gnn" in lowered or "metagnn" in lowered:
        return "meta_gnn_k"
    return None


def method_from_path(path: Path) -> str | None:
    return normalize_method(str(path))


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, str):
        try:
            converted = float(value.strip())
        except ValueError:
            return None
        return converted if math.isfinite(converted) else None
    return None


def json_evidence(payload: Any, path: Path) -> dict[tuple[str, str], list[float]]:
    """Collect numeric field evidence, preserving the method context of nested dicts."""
    found: dict[tuple[str, str], list[float]] = defaultdict(list)
    inferred = method_from_path(path)

    def visit(value: Any, method: str | None) -> None:
        if isinstance(value, dict):
            local_method = method
            for key in ("method", "method_name", "model_method", "model_variant"):
                candidate = normalize_method(value.get(key))
                if candidate is not None:
                    local_method = candidate
                    break
            for key, child in value.items():
                lowered = str(key).strip().lower()
                numeric = number(child)
                if numeric is not None and local_method in METHODS:
                    for field, aliases in FIELD_ALIASES.items():
                        if lowered in aliases:
                            found[(local_method, field)].append(numeric)
                            break
                visit(child, local_method)
        elif isinstance(value, list):
            for child in value:
                visit(child, method)

    visit(payload, inferred)
    return found


def scan_json_root(root: Path, progress_label: str) -> tuple[pd.DataFrame, dict[tuple[str, str], list[float]], list[Path]]:
    files = sorted(root.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index, path in enumerate(files, start=1):
        try:
            payload = load_json(path, "JSON provenance artifact")
        except RuntimeError as error:
            rows.append({
                "artifact_root": progress_label,
                "relative_path": str(path.relative_to(root)),
                "method": "unassigned",
                "field": "json_parse",
                "observations": 0,
                "minimum": None,
                "maximum": None,
                "unique_values": 0,
                "availability": "unreadable",
                "notes": str(error),
            })
            continue
        evidence = json_evidence(payload, path)
        for (method, field), values in sorted(evidence.items()):
            aggregate[(method, field)].extend(values)
            rows.append({
                "artifact_root": progress_label,
                "relative_path": str(path.relative_to(root)),
                "method": method,
                "field": field,
                "observations": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "unique_values": len(set(values)),
                "availability": "available",
                "notes": "Numeric metadata recovered directly from frozen JSON.",
            })
        if index % 250 == 0 or index == len(files):
            print(f"[A25.0b] scanned {progress_label} JSON {index:04d}/{len(files):04d}", flush=True)
    columns = [
        "artifact_root", "relative_path", "method", "field", "observations", "minimum",
        "maximum", "unique_values", "availability", "notes",
    ]
    return pd.DataFrame(rows, columns=columns), aggregate, files


def scan_csv_root(root: Path, progress_label: str) -> tuple[pd.DataFrame, dict[tuple[str, str], list[float]], list[Path]]:
    """Recover method-scoped execution fields from CSV histories without loading them all at once."""
    files = sorted(root.rglob("*.csv"))
    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    method_columns = ("method", "method_name", "model_method", "model_variant")
    alias_to_field = {alias: field for field, aliases in FIELD_ALIASES.items() for alias in aliases}

    for index, path in enumerate(files, start=1):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception as error:
            rows.append({
                "artifact_root": progress_label,
                "relative_path": str(path.relative_to(root)),
                "method": "unassigned",
                "field": "csv_parse",
                "observations": 0,
                "minimum": None,
                "maximum": None,
                "unique_values": 0,
                "availability": "unreadable",
                "notes": f"{type(error).__name__}: {error}",
            })
            continue
        normalized = {str(column).strip().lower(): str(column) for column in header.columns}
        field_columns = {alias_to_field[key]: column for key, column in normalized.items() if key in alias_to_field}
        method_column = next((normalized[key] for key in method_columns if key in normalized), None)
        path_method = method_from_path(path)
        if not field_columns or (path_method is None and method_column is None):
            continue

        per_file: dict[tuple[str, str], list[float]] = defaultdict(list)
        usecols = list(dict.fromkeys(list(field_columns.values()) + ([method_column] if method_column else [])))
        try:
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
                if method_column is None:
                    method_series = pd.Series([path_method] * len(chunk), index=chunk.index)
                else:
                    method_series = chunk[method_column].map(normalize_method)
                    if path_method is not None:
                        method_series = method_series.fillna(path_method)
                for field, column in field_columns.items():
                    numeric = pd.to_numeric(chunk[column], errors="coerce")
                    for method in METHODS:
                        values = numeric[method_series == method].dropna().astype(float).tolist()
                        if values:
                            per_file[(method, field)].extend(values)
        except Exception as error:
            rows.append({
                "artifact_root": progress_label,
                "relative_path": str(path.relative_to(root)),
                "method": path_method or "unassigned",
                "field": "csv_parse",
                "observations": 0,
                "minimum": None,
                "maximum": None,
                "unique_values": 0,
                "availability": "unreadable",
                "notes": f"{type(error).__name__}: {error}",
            })
            continue

        for (method, field), values in sorted(per_file.items()):
            aggregate[(method, field)].extend(values)
            rows.append({
                "artifact_root": progress_label,
                "relative_path": str(path.relative_to(root)),
                "method": method,
                "field": field,
                "observations": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "unique_values": len(set(values)),
                "availability": "available",
                "notes": "Numeric metadata recovered directly from frozen CSV.",
            })
        if index % 250 == 0 or index == len(files):
            print(f"[A25.0b] scanned {progress_label} CSV {index:04d}/{len(files):04d}", flush=True)

    columns = [
        "artifact_root", "relative_path", "method", "field", "observations", "minimum",
        "maximum", "unique_values", "availability", "notes",
    ]
    return pd.DataFrame(rows, columns=columns), aggregate, files


def candidate_checkpoint_files(roots: Iterable[Path]) -> list[Path]:
    suffixes = {".pt", ".pth", ".ckpt"}
    output: list[Path] = []
    for root in roots:
        output.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    return sorted(set(output))


def choose_checkpoint_samples(files: list[Path], mode: str, per_method: int) -> list[Path]:
    if mode == "none":
        return []
    if mode == "all":
        return files
    selected: list[Path] = []
    for method in METHODS:
        group = [path for path in files if method_from_path(path) == method]
        selected.extend(group[:per_method])
    return selected


def find_state_dict(payload: Any, torch_module: Any, depth: int = 0) -> Mapping[str, Any] | None:
    if depth > 4 or not isinstance(payload, Mapping):
        return None
    if payload and all(isinstance(value, torch_module.Tensor) for value in payload.values()):
        return payload
    for key in ("state_dict", "model_state_dict", "meta_state", "state", "model", "checkpoint"):
        child = payload.get(key)
        answer = find_state_dict(child, torch_module, depth + 1)
        if answer is not None:
            return answer
    for child in payload.values():
        answer = find_state_dict(child, torch_module, depth + 1)
        if answer is not None:
            return answer
    return None


def checkpoint_inventory(files: list[Path], samples: list[Path], mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chosen = set(samples)
    for path in files:
        rows.append({
            "method": method_from_path(path) or "unassigned",
            "checkpoint_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "selected_for_tensor_scan": path in chosen,
            "tensor_scan_status": "not_requested",
            "state_tensor_numel": None,
            "state_tensor_count": None,
            "state_schema_sha256": None,
            "notes": "File-stat inventory only.",
        })
    if not samples:
        return pd.DataFrame(rows)
    try:
        import torch  # imported only for an explicit checkpoint validation request
    except ImportError as error:
        raise RuntimeError("PyTorch is required for checkpoint validation; use --checkpoint-validation none or the project environment") from error

    index_by_path = {Path(row["checkpoint_path"]): index for index, row in enumerate(rows)}
    for count, path in enumerate(samples, start=1):
        row = rows[index_by_path[path]]
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # compatibility with older PyTorch
                payload = torch.load(path, map_location="cpu")
            state = find_state_dict(payload, torch)
            if state is None:
                row["tensor_scan_status"] = "no_state_dict_found"
                row["notes"] = "Loaded safely on CPU, but no tensor state dictionary was found."
            else:
                schema = "\n".join(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}" for name, tensor in sorted(state.items()))
                row["tensor_scan_status"] = "passed"
                row["state_tensor_numel"] = int(sum(int(tensor.numel()) for tensor in state.values()))
                row["state_tensor_count"] = int(len(state))
                row["state_schema_sha256"] = hashlib.sha256(schema.encode("utf-8")).hexdigest()
                row["notes"] = "CPU-only state-dictionary inventory; tensor numel may include buffers and is not asserted to equal trainable parameters."
        except Exception as error:  # audit should report bad files rather than hide them
            row["tensor_scan_status"] = "failed"
            row["notes"] = f"{type(error).__name__}: {error}"
        if count % 25 == 0 or count == len(samples):
            print(f"[A25.0b] checkpoint tensor scan {count:04d}/{len(samples):04d} ({mode})", flush=True)
    return pd.DataFrame(rows)


def available_numeric_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normal = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in normal:
            numeric = pd.to_numeric(frame[normal[alias]], errors="coerce")
            if numeric.notna().any():
                return normal[alias]
    return None


def pft_summary(training_csv: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(training_csv)
    rows: list[dict[str, Any]] = []

    def add(quantity: str, aliases: tuple[str, ...], unit: str, interpretation: str) -> None:
        column = available_numeric_column(frame, aliases)
        if column is None:
            rows.append({"method": "pretrain_finetune_k", "quantity": quantity, "value": None, "unit": unit,
                         "availability": "missing", "evidence": str(training_csv), "interpretation": interpretation})
            return
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        unique = sorted(set(float(value) for value in values))
        value = unique[0] if len(unique) == 1 else f"min={min(unique):g}; max={max(unique):g}; n_unique={len(unique)}"
        rows.append({"method": "pretrain_finetune_k", "quantity": quantity, "value": value, "unit": unit,
                     "availability": "available", "evidence": f"{training_csv}::{column}", "interpretation": interpretation})

    rows.append({"method": "pretrain_finetune_k", "quantity": "target_adapted_model_records", "value": int(len(frame)),
                 "unit": "records", "availability": "available", "evidence": str(training_csv),
                 "interpretation": "Training-run records; records are not optimizer updates."})
    add("source_pretrain_steps", ("source_pretrain_steps", "declared_source_pretrain_steps"), "optimizer steps",
        "Declared ordinary supervised source-pretraining steps per source cache.")
    add("target_epochs", ("target_epochs", "target_adaptation_epochs"), "epochs",
        "Target epochs are not optimizer updates without batch accounting.")
    add("total_parameters", ("total_parameters", "parameter_count", "num_parameters"), "parameters",
        "Available only if explicitly recorded in PFT run-level artifacts.")
    add("trainable_parameters", ("trainable_parameters", "trainable_parameter_count"), "parameters",
        "Available only if explicitly recorded in PFT run-level artifacts.")
    add("wall_time_seconds", ("wall_time_seconds", "runtime_seconds", "elapsed_seconds", "duration_seconds"), "seconds",
        "Valid for fairness only if measured on comparable hardware and execution conditions.")
    return rows


def meta_summary(aggregate: Mapping[tuple[str, str], list[float]], evidence_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for field in REQUIRED_META_FIELDS:
            values = aggregate.get((method, field), [])
            if not values:
                rows.append({"method": method, "quantity": field, "value": None, "unit": "see_quantity",
                             "availability": "missing", "evidence": "No method-context numeric value found in frozen A24.1/A24.2 JSON.",
                             "interpretation": "Missing is preserved; the audit does not infer this field."})
                continue
            # Histories may contain every outer step.  The per-run final value is the maximum;
            # spread is retained in the value string rather than silently treated as equality.
            minimum, maximum = min(values), max(values)
            unit = {
                "outer_steps": "meta steps", "inner_steps": "inner optimizer steps",
                "total_parameters": "parameters", "trainable_parameters": "parameters",
                "wall_time_seconds": "seconds",
            }[field]
            if field == "wall_time_seconds":
                value: Any = f"n={len(values)}; min={minimum:g}; max={maximum:g}; sum={sum(values):g}"
            else:
                value = maximum if minimum != maximum else minimum
            rows.append({"method": method, "quantity": field, "value": value, "unit": unit,
                         "availability": "available", "evidence": f"Recovered from {len(evidence_files)} JSON/CSV artifacts; raw range min={minimum:g}, max={maximum:g}.",
                         "interpretation": "Recovered directly from frozen metadata; range is retained for audit."})
    return rows


def input_integrity(
    files: Mapping[str, Path], decisions: Mapping[str, Mapping[str, Any]], checkpoint_files: list[Path]
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "input_sha256": {name: sha256(path) for name, path in files.items()},
        "upstream_complete": {name: payload.get("complete") is True for name, payload in decisions.items()},
        "checkpoint_files_discovered": len(checkpoint_files),
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a23-3-output-dir", required=True, help="Completed A23.3 output directory.")
    parser.add_argument("--a24-1-output-dir", required=True, help="Completed A24.1 output directory.")
    parser.add_argument("--a24-2-output-dir", required=True, help="Completed A24.2 output directory.")
    parser.add_argument("--output-dir", required=True, help="New A25.0b audit output directory.")
    parser.add_argument("--checkpoint-validation", choices=("none", "sample", "all"), default="sample",
                        help="CPU-only checkpoint state scan. 'sample' reads deterministic samples per method.")
    parser.add_argument("--checkpoint-samples-per-method", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and list discovery counts without reading checkpoints or writing final artifacts.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.checkpoint_samples_per_method < 1:
        raise ValueError("--checkpoint-samples-per-method must be at least one")
    a23_root = require_directory(args.a23_3_output_dir, "A23.3")
    a241_root = require_directory(args.a24_1_output_dir, "A24.1")
    a242_root = require_directory(args.a24_2_output_dir, "A24.2")
    output = Path(args.output_dir).expanduser().resolve()

    a23_decision_path, a23_decision = require_completed_decision(a23_root, "experimentA23_3")
    a241_decision_path, a241_decision = require_completed_decision(a241_root, "experimentA24_1")
    a242_decision_path, a242_decision = require_completed_decision(a242_root, "experimentA24_2")
    a23_training = one_named_file(a23_root, "experimentA23_3_training_run_level.csv", "A23.3 run-level training CSV")
    a241_manifest = one_named_file(a241_root, "experimentA24_1_manifest.json", "A24.1 manifest")
    a242_manifest = one_named_file(a242_root, "experimentA24_2_manifest.json", "A24.2 manifest")
    files = {
        "a23_3_confirmation_decision": a23_decision_path,
        "a23_3_training_run_level": a23_training,
        "a24_1_confirmation_decision": a241_decision_path,
        "a24_1_manifest": a241_manifest,
        "a24_2_confirmation_decision": a242_decision_path,
        "a24_2_manifest": a242_manifest,
    }
    decisions = {"experimentA23_3": a23_decision, "experimentA24_1": a241_decision, "experimentA24_2": a242_decision}
    checkpoint_files = candidate_checkpoint_files([a241_root, a242_root])
    selected = choose_checkpoint_samples(checkpoint_files, args.checkpoint_validation, args.checkpoint_samples_per_method)

    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "a23_3_output_dir": str(a23_root),
        "a24_1_output_dir": str(a241_root),
        "a24_2_output_dir": str(a242_root),
        "output_dir": str(output),
        "checkpoint_files_discovered": len(checkpoint_files),
        "checkpoint_validation": args.checkpoint_validation,
        "checkpoint_files_selected_for_tensor_scan": len(selected),
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("[A25.0b] dry-run passed; no checkpoint tensor was loaded and no predictor was trained")
        return 0

    a241_inventory, a241_evidence, a241_json = scan_json_root(a241_root, "A24.1")
    a242_inventory, a242_evidence, a242_json = scan_json_root(a242_root, "A24.2")
    a241_csv_inventory, a241_csv_evidence, a241_csv = scan_csv_root(a241_root, "A24.1")
    a242_csv_inventory, a242_csv_evidence, a242_csv = scan_csv_root(a242_root, "A24.2")
    raw_evidence: dict[tuple[str, str], list[float]] = defaultdict(list)
    for source in (a241_evidence, a242_evidence, a241_csv_evidence, a242_csv_evidence):
        for key, values in source.items():
            raw_evidence[key].extend(values)
    provenance = pd.concat(
        [a241_inventory, a242_inventory, a241_csv_inventory, a242_csv_inventory],
        ignore_index=True,
    )
    checkpoints = checkpoint_inventory(checkpoint_files, selected, args.checkpoint_validation)

    evidence_files = a241_json + a242_json + a241_csv + a242_csv
    summary_rows = pft_summary(a23_training) + meta_summary(raw_evidence, evidence_files)
    summary = pd.DataFrame(summary_rows)
    meta_required = summary[(summary["method"] == "meta_no_graph_k") & summary["quantity"].isin(REQUIRED_META_FIELDS)]
    missing_meta_fields = sorted(meta_required.loc[meta_required["availability"] != "available", "quantity"].tolist())
    checkpoint_failures = int((checkpoints.get("tensor_scan_status", pd.Series(dtype=str)) == "failed").sum())
    # Completion means the registered scan executed over all available frozen artifacts.
    # Missing provenance fields affect `passed`, not `complete`.
    complete = bool(all(payload.get("complete") is True for payload in decisions.values())) and checkpoint_failures == 0
    recovered = complete and not missing_meta_fields
    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": "Can frozen A23.3/A24.1/A24.2 artifacts recover an auditable Reptile execution ledger before an independent compute-matched low-RUL safety cohort is preregistered?",
        "complete": complete,
        "passed": recovered,
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "a23_3_complete": a23_decision.get("complete") is True,
        "a24_1_complete": a241_decision.get("complete") is True,
        "a24_2_complete": a242_decision.get("complete") is True,
        "json_artifacts_scanned": len(a241_json) + len(a242_json),
        "csv_artifacts_scanned": len(a241_csv) + len(a242_csv),
        "checkpoint_files_discovered": len(checkpoint_files),
        "checkpoint_validation": args.checkpoint_validation,
        "checkpoint_files_tensor_scanned": len(selected),
        "checkpoint_tensor_scan_failures": checkpoint_failures,
        "required_meta_fields": list(REQUIRED_META_FIELDS),
        "missing_meta_no_graph_fields": missing_meta_fields,
        "compute_accounting_recovered": recovered,
        "training_budget_equivalence_established": False,
        "reason": (
            "A25.0b recovered all required Meta-noGraph execution fields from frozen artifacts; budget equivalence still requires a separately declared matching rule."
            if recovered else
            "A25.0b completed the frozen provenance scan, but one or more required Meta-noGraph execution fields remain unavailable."
        ),
        "interpretation_limit": "This is artifact provenance recovery only. It does not retrain, retune, evaluate confirmation engines, or create a performance claim.",
        "next_action": (
            "preregister_A25_1_independent_compute_matched_low_RUL_safety_pilot"
            if recovered else
            "locate_or_preserve_missing_runtime_metadata_before_any_compute_matched_meta_learning_claim"
        ),
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    integrity = input_integrity(files, decisions, checkpoint_files)
    integrity.update({
        "json_artifacts_scanned": len(a241_json) + len(a242_json),
        "csv_artifacts_scanned": len(a241_csv) + len(a242_csv),
        "compute_accounting_recovered": recovered,
        "missing_meta_no_graph_fields": missing_meta_fields,
    })

    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "experimentA25_0b_execution_provenance_inventory.csv", provenance)
    atomic_csv(output / "experimentA25_0b_parameter_inventory.csv", checkpoints)
    atomic_csv(output / "experimentA25_0b_compute_accounting_summary.csv", summary)
    atomic_json(output / "experimentA25_0b_input_integrity.json", integrity)
    atomic_json(output / "experimentA25_0b_confirmation_decision.json", decision)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {name: sha256(path) for name, path in files.items()},
        "artifacts": {},
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for name in (
        "experimentA25_0b_execution_provenance_inventory.csv",
        "experimentA25_0b_parameter_inventory.csv",
        "experimentA25_0b_compute_accounting_summary.csv",
        "experimentA25_0b_input_integrity.json",
        "experimentA25_0b_confirmation_decision.json",
    ):
        manifest["artifacts"][name] = sha256(output / name)
    atomic_json(output / "experimentA25_0b_manifest.json", manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("[A25.0b] completed frozen execution-provenance recovery")
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[A25.0b] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[A25.0b] error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
