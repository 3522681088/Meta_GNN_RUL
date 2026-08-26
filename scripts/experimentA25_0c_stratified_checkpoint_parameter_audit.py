#!/usr/bin/env python3
"""A25.0c — stratified checkpoint/schema and PFT parameter audit.

This is an evaluation-free provenance program.  It does not train or adapt a
predictor, inspect selection/confirmation outcomes, or access official test
files.  It corrects the A25.0b path-classification boundary by identifying a
method from the checkpoint *basename* before inspecting parent path parts.

Registered outputs separate four concepts that must not be conflated:
  1. artifact_audit_complete;
  2. checkpoint_parameter_accounting_complete;
  3. historical_wall_time_available;
  4. training_budget_equivalence_established.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


EXPERIMENT_ID = "experimentA25_0c"
SCRIPT_VERSION = "experimentA25_0c_stratified_checkpoint_parameter_audit_v1"
METHODS = ("pretrain_finetune_k", "meta_no_graph_k", "meta_gnn_k")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def require_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def one_file(root: Path, name: str, label: str) -> Path:
    direct = root / name
    matches = [direct] if direct.is_file() else sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{label} is missing below {root}: {name}")
    if len(matches) > 1:
        values = "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(f"{label} is ambiguous; retain one canonical file:\n  - {values}")
    return matches[0]


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def completed_decision(root: Path, experiment: str) -> tuple[Path, dict[str, Any]]:
    path = one_file(root, f"{experiment}_confirmation_decision.json", f"{experiment} decision")
    payload = load_json(path, f"{experiment} decision")
    if payload.get("experiment_id") != experiment:
        raise RuntimeError(f"Unexpected experiment_id in {path}: {payload.get('experiment_id')!r}")
    if payload.get("complete") is not True:
        raise RuntimeError(f"Upstream experiment is not complete: {path}")
    return path, payload


def validate_manifest(root: Path, experiment: str) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    path = one_file(root, f"{experiment}_manifest.json", f"{experiment} manifest")
    payload = load_json(path, f"{experiment} manifest")
    if payload.get("experiment_id") != experiment:
        raise RuntimeError(f"Unexpected manifest experiment_id in {path}")
    checks: list[dict[str, str]] = []
    for name, expected in payload.get("artifacts", {}).items():
        artifact = root / str(name)
        if not artifact.is_file():
            status, observed = "missing", ""
        else:
            observed = sha256(artifact)
            status = "passed" if observed == expected else "mismatch"
        checks.append({
            "experiment": experiment,
            "artifact": str(name),
            "expected_sha256": str(expected),
            "observed_sha256": observed,
            "status": status,
        })
    failures = [row for row in checks if row["status"] != "passed"]
    if failures:
        raise RuntimeError(f"{experiment} manifest validation failed for {len(failures)} artifact(s)")
    return path, payload, checks


def classify_basename(path: Path) -> tuple[str, str]:
    """Classify from basename first; never let a compound experiment dirname win."""
    name = path.name.lower().replace("-", "_")
    rules = (
        ("meta_no_graph_k", ("meta_no_graph", "metanograph")),
        ("meta_gnn_k", ("meta_gnn", "metagnn")),
        ("pretrain_finetune_k", ("pretrain_finetune", "pretrain_plus_finetune", "pft_", "_pft")),
    )
    for method, tokens in rules:
        if any(token in name for token in tokens):
            return method, f"basename:{path.name}"

    # Conservative fallback: accept only an entire path component, not a
    # compound directory containing multiple method names.
    component_patterns = {
        "meta_no_graph_k": re.compile(r"^(meta_no_graph_k|meta_no_graph)$"),
        "meta_gnn_k": re.compile(r"^(meta_gnn_k|meta_gnn)$"),
        "pretrain_finetune_k": re.compile(r"^(pretrain_finetune_k|pft)$"),
    }
    for component in reversed(path.parts[:-1]):
        normalized = component.lower().replace("-", "_")
        hits = [method for method, pattern in component_patterns.items() if pattern.fullmatch(normalized)]
        if len(hits) == 1:
            return hits[0], f"path_component:{component}"
    return "unclassified", "no_unambiguous_method_token"


def root_label(path: Path, roots: Mapping[str, Path]) -> str:
    for label, root in roots.items():
        try:
            path.relative_to(root)
            return label
        except ValueError:
            continue
    return "unknown"


def discover_checkpoints(roots: Mapping[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, root in roots.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            method, evidence = classify_basename(path)
            rows.append({
                "artifact_root": label,
                "method": method,
                "classification_evidence": evidence,
                "checkpoint_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "file_size_bytes": int(path.stat().st_size),
                "selected_for_tensor_scan": False,
                "tensor_scan_status": "not_requested",
                "state_tensor_numel": None,
                "state_tensor_count": None,
                "state_schema_sha256": None,
                "explicit_total_parameters": None,
                "explicit_trainable_parameters": None,
                "notes": "File-stat inventory only.",
            })
    return pd.DataFrame(rows)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count > length:
        raise ValueError(f"Cannot select {count} samples from {length} files")
    if count == 1:
        return [length // 2]
    positions = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError("Deterministic checkpoint sampler generated duplicate indices")
    return positions


def select_stratified(frame: pd.DataFrame, samples_per_method: int) -> list[int]:
    selected: list[int] = []
    for method in METHODS:
        group = frame.index[frame["method"] == method].tolist()
        if len(group) < samples_per_method:
            raise RuntimeError(
                f"A25.0c requires at least {samples_per_method} {method} checkpoints; discovered {len(group)}"
            )
        selected.extend(group[position] for position in evenly_spaced_indices(len(group), samples_per_method))
    return selected


def find_state_dict(payload: Any, torch_module: Any, depth: int = 0) -> Mapping[str, Any] | None:
    if depth > 5 or not isinstance(payload, Mapping):
        return None
    tensor_values = [value for value in payload.values() if isinstance(value, torch_module.Tensor)]
    if tensor_values and len(tensor_values) == len(payload):
        return payload
    for key in ("state_dict", "model_state_dict", "meta_state", "state", "model", "checkpoint"):
        if key in payload:
            answer = find_state_dict(payload[key], torch_module, depth + 1)
            if answer is not None:
                return answer
    for child in payload.values():
        answer = find_state_dict(child, torch_module, depth + 1)
        if answer is not None:
            return answer
    return None


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def find_numeric_key(payload: Any, aliases: set[str], depth: int = 0) -> float | None:
    if depth > 5:
        return None
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in aliases:
                result = finite_number(value)
                if result is not None:
                    return result
        for value in payload.values():
            result = find_numeric_key(value, aliases, depth + 1)
            if result is not None:
                return result
    elif isinstance(payload, list):
        for value in payload:
            result = find_numeric_key(value, aliases, depth + 1)
            if result is not None:
                return result
    return None


def scan_selected_checkpoints(frame: pd.DataFrame, selected: list[int]) -> pd.DataFrame:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; run A25.0c in the project conda environment") from error

    for ordinal, row_index in enumerate(selected, start=1):
        path = Path(str(frame.at[row_index, "checkpoint_path"]))
        frame.at[row_index, "selected_for_tensor_scan"] = True
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            state = find_state_dict(payload, torch)
            if state is None:
                frame.at[row_index, "tensor_scan_status"] = "no_state_dict_found"
                frame.at[row_index, "notes"] = "Loaded on CPU, but no tensor state dictionary was found."
            else:
                schema = "\n".join(
                    f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
                    for name, tensor in sorted(state.items())
                )
                frame.at[row_index, "tensor_scan_status"] = "passed"
                frame.at[row_index, "state_tensor_numel"] = int(
                    sum(int(tensor.numel()) for tensor in state.values())
                )
                frame.at[row_index, "state_tensor_count"] = int(len(state))
                frame.at[row_index, "state_schema_sha256"] = hashlib.sha256(schema.encode("utf-8")).hexdigest()
                frame.at[row_index, "explicit_total_parameters"] = find_numeric_key(
                    payload, {"total_parameters", "parameter_count", "num_parameters"}
                )
                frame.at[row_index, "explicit_trainable_parameters"] = find_numeric_key(
                    payload, {"trainable_parameters", "trainable_parameter_count"}
                )
                frame.at[row_index, "notes"] = (
                    "CPU state-dictionary audit. state_tensor_numel can include buffers; "
                    "explicit parameter fields are reported separately."
                )
        except Exception as error:
            frame.at[row_index, "tensor_scan_status"] = "failed"
            frame.at[row_index, "notes"] = f"{type(error).__name__}: {error}"
        print(
            f"[A25.0c] tensor scan {ordinal:02d}/{len(selected):02d} "
            f"method={frame.at[row_index, 'method']} status={frame.at[row_index, 'tensor_scan_status']}",
            flush=True,
        )
    return frame


def scalar_or_none(series: pd.Series) -> float | None:
    values = sorted(set(float(value) for value in pd.to_numeric(series, errors="coerce").dropna()))
    return values[0] if len(values) == 1 else None


def method_summary(inventory: pd.DataFrame, samples_per_method: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        discovered = inventory[inventory["method"] == method]
        scanned = discovered[discovered["selected_for_tensor_scan"] == True]
        passed = scanned[scanned["tensor_scan_status"] == "passed"]
        numel_values = sorted(set(pd.to_numeric(passed["state_tensor_numel"], errors="coerce").dropna().astype(int)))
        tensor_count_values = sorted(set(pd.to_numeric(passed["state_tensor_count"], errors="coerce").dropna().astype(int)))
        schema_values = sorted(set(passed["state_schema_sha256"].dropna().astype(str)))
        explicit_total_values = sorted(set(pd.to_numeric(passed["explicit_total_parameters"], errors="coerce").dropna()))
        explicit_trainable_values = sorted(set(pd.to_numeric(passed["explicit_trainable_parameters"], errors="coerce").dropna()))
        rows.append({
            "method": method,
            "checkpoint_files_discovered": int(len(discovered)),
            "checkpoint_files_requested": int(samples_per_method),
            "checkpoint_files_scanned": int(len(scanned)),
            "checkpoint_scans_passed": int(len(passed)),
            "checkpoint_scans_failed": int((scanned["tensor_scan_status"] != "passed").sum()),
            "state_tensor_numel_min": min(numel_values) if numel_values else None,
            "state_tensor_numel_max": max(numel_values) if numel_values else None,
            "state_tensor_numel_unique": len(numel_values),
            "state_tensor_count_values": ";".join(str(value) for value in tensor_count_values),
            "state_schema_unique": len(schema_values),
            "state_schema_sha256": schema_values[0] if len(schema_values) == 1 else None,
            "explicit_total_parameters": explicit_total_values[0] if len(explicit_total_values) == 1 else None,
            "explicit_trainable_parameters": explicit_trainable_values[0] if len(explicit_trainable_values) == 1 else None,
            "within_method_schema_consistent": bool(len(schema_values) == 1 and len(passed) == samples_per_method),
        })
    return pd.DataFrame(rows)


def wall_time_availability(a25b_root: Path) -> dict[str, bool]:
    summary_path = one_file(
        a25b_root,
        "experimentA25_0b_compute_accounting_summary.csv",
        "A25.0b compute-accounting summary",
    )
    frame = pd.read_csv(summary_path)
    output: dict[str, bool] = {}
    for method in METHODS:
        rows = frame[(frame["method"] == method) & (frame["quantity"] == "wall_time_seconds")]
        output[method] = bool(not rows.empty and (rows["availability"] == "available").all())
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a23-3-output-dir", required=True)
    parser.add_argument("--a24-1-output-dir", required=True)
    parser.add_argument("--a24-2-output-dir", required=True)
    parser.add_argument("--a25-0b-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-method", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_method < 3:
        raise ValueError("--samples-per-method must be at least 3 for the registered stratified audit")

    roots = {
        "A23.3": require_dir(args.a23_3_output_dir, "A23.3"),
        "A24.1": require_dir(args.a24_1_output_dir, "A24.1"),
        "A24.2": require_dir(args.a24_2_output_dir, "A24.2"),
    }
    a25b_root = require_dir(args.a25_0b_output_dir, "A25.0b")
    output = Path(args.output_dir).expanduser().resolve()

    input_files: dict[str, Path] = {}
    upstream_decisions: dict[str, dict[str, Any]] = {}
    integrity_rows: list[dict[str, str]] = []
    for label, experiment in (("A23.3", "experimentA23_3"), ("A24.1", "experimentA24_1"), ("A24.2", "experimentA24_2")):
        decision_path, decision = completed_decision(roots[label], experiment)
        manifest_path, _, checks = validate_manifest(roots[label], experiment)
        input_files[f"{experiment}_decision"] = decision_path
        input_files[f"{experiment}_manifest"] = manifest_path
        upstream_decisions[experiment] = decision
        integrity_rows.extend(checks)
    a25b_decision_path, a25b_decision = completed_decision(a25b_root, "experimentA25_0b")
    a25b_manifest_path, _, checks = validate_manifest(a25b_root, "experimentA25_0b")
    input_files["experimentA25_0b_decision"] = a25b_decision_path
    input_files["experimentA25_0b_manifest"] = a25b_manifest_path
    upstream_decisions["experimentA25_0b"] = a25b_decision
    integrity_rows.extend(checks)

    inventory = discover_checkpoints(roots)
    selected = select_stratified(inventory, args.samples_per_method)
    selected_preview = inventory.loc[selected, ["artifact_root", "method", "relative_path", "file_size_bytes"]]
    counts = inventory["method"].value_counts().to_dict()
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "output_dir": str(output),
        "checkpoint_files_discovered_total": int(len(inventory)),
        "checkpoint_files_by_method": {str(key): int(value) for key, value in counts.items()},
        "samples_per_method": int(args.samples_per_method),
        "selected_checkpoint_files": selected_preview.to_dict(orient="records"),
        "upstream_complete": {key: value.get("complete") is True for key, value in upstream_decisions.items()},
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "confirmation_engines_evaluated": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("[A25.0c] dry-run passed; no checkpoint tensor was loaded and no predictor was trained")
        return 0

    inventory = scan_selected_checkpoints(inventory, selected)
    summary = method_summary(inventory, args.samples_per_method)
    summary_by_method = summary.set_index("method")
    pft = summary_by_method.loc["pretrain_finetune_k"]
    no_graph = summary_by_method.loc["meta_no_graph_k"]
    pft_numel = scalar_or_none(pd.Series([pft["state_tensor_numel_min"], pft["state_tensor_numel_max"]]))
    no_graph_numel = scalar_or_none(pd.Series([no_graph["state_tensor_numel_min"], no_graph["state_tensor_numel_max"]]))
    state_numel_matches = bool(pft_numel is not None and no_graph_numel is not None and pft_numel == no_graph_numel)
    schema_matches = bool(
        pd.notna(pft["state_schema_sha256"])
        and pd.notna(no_graph["state_schema_sha256"])
        and pft["state_schema_sha256"] == no_graph["state_schema_sha256"]
    )
    exact_parameter_schema_match = bool(state_numel_matches and schema_matches)

    wall_time = wall_time_availability(a25b_root)
    historical_wall_time_available = bool(all(wall_time.values()))
    balanced_sample_passed = bool(
        (summary["checkpoint_files_scanned"] == args.samples_per_method).all()
        and (summary["checkpoint_scans_passed"] == args.samples_per_method).all()
        and summary["within_method_schema_consistent"].all()
    )
    artifact_audit_complete = bool(
        all(value.get("complete") is True for value in upstream_decisions.values())
        and all(row["status"] == "passed" for row in integrity_rows)
        and balanced_sample_passed
    )
    pft_parameter_scale_recovered = bool(
        pft["checkpoint_scans_passed"] == args.samples_per_method
        and pft["state_tensor_numel_unique"] == 1
    )

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "registered_primary_question": "Can checkpoint basenames and stratified CPU state-dictionary scans correct the A25.0b method-label boundary, recover the PFT parameter schema, and determine whether historical PFT and Meta-noGraph were architecture/parameter matched?",
        "complete": artifact_audit_complete,
        "passed": artifact_audit_complete,
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "confirmation_engines_evaluated": False,
        "samples_per_method": int(args.samples_per_method),
        "expected_tensor_scans": int(args.samples_per_method * len(METHODS)),
        "completed_tensor_scans": int(summary["checkpoint_scans_passed"].sum()),
        "balanced_stratified_checkpoint_scan_passed": balanced_sample_passed,
        "pft_parameter_scale_recovered": pft_parameter_scale_recovered,
        "pft_vs_meta_no_graph_state_tensor_numel_matches": state_numel_matches,
        "pft_vs_meta_no_graph_state_schema_matches": schema_matches,
        "pft_vs_meta_no_graph_exact_parameter_schema_match": exact_parameter_schema_match,
        "historical_wall_time_available": historical_wall_time_available,
        "wall_time_availability_by_method": wall_time,
        "checkpoint_parameter_accounting_complete": balanced_sample_passed and pft_parameter_scale_recovered,
        "training_budget_equivalence_established": False,
        "historical_comparison_architecture_confound_present": not exact_parameter_schema_match,
        "reason": "A25.0c completed the basename-stratified checkpoint audit and separated parameter-schema evidence from unavailable historical timing evidence.",
        "interpretation_limit": "Checkpoint state numel can include registered buffers. Historical wall time cannot be reconstructed from file timestamps, and this audit does not create an efficacy claim.",
        "next_action": "preregister_A25_1_same_architecture_prospective_runtime_accounted_low_RUL_safety_pilot",
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    integrity = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "input_sha256": {name: sha256(path) for name, path in input_files.items()},
        "manifest_artifact_checks": integrity_rows,
        "all_manifest_artifact_checks_passed": all(row["status"] == "passed" for row in integrity_rows),
        "upstream_complete": {key: value.get("complete") is True for key, value in upstream_decisions.items()},
        "classified_checkpoint_files": int((inventory["method"] != "unclassified").sum()),
        "unclassified_checkpoint_files": int((inventory["method"] == "unclassified").sum()),
        "new_predictor_training": False,
        "target_adaptation": False,
        "policy_selection_or_tuning": False,
        "confirmation_engines_evaluated": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "experimentA25_0c_checkpoint_inventory.csv", inventory)
    atomic_csv(output / "experimentA25_0c_parameter_schema_summary.csv", summary)
    atomic_json(output / "experimentA25_0c_input_integrity.json", integrity)
    atomic_json(output / "experimentA25_0c_confirmation_decision.json", decision)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {name: sha256(path) for name, path in input_files.items()},
        "artifacts": {},
        "descriptive_only": True,
        "formal_efficacy_claim": False,
        "new_predictor_training": False,
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }
    for name in (
        "experimentA25_0c_checkpoint_inventory.csv",
        "experimentA25_0c_parameter_schema_summary.csv",
        "experimentA25_0c_input_integrity.json",
        "experimentA25_0c_confirmation_decision.json",
    ):
        manifest["artifacts"][name] = sha256(output / name)
    atomic_json(output / "experimentA25_0c_manifest.json", manifest)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("[A25.0c] completed stratified checkpoint and PFT parameter audit")
    return 0 if artifact_audit_complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[A25.0c] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[A25.0c] error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
