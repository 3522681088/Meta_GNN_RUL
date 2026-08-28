#!/usr/bin/env python3
"""A28.0: read-only closure of the failed A27.1 low-RUL repair candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


VERSION = "experimentA28_0_negative_result_closure_preflight_v1"
EXPERIMENT = "experimentA28_0"
FREEZE = "A28.0_CLOSURE_FREEZE"
REQUIRED = (
    "experimentA27_1_confirmation_decision.json",
    "experimentA27_1_manifest.json",
    "experimentA27_1_global_arm_summary.csv",
    "experimentA27_1_advancement_gate_results.csv",
    "experimentA27_1_paired_worker_comparisons.csv",
    "experimentA27_1_matched_target_compute_audit.csv",
)
HASH_COVERED_REQUIRED = tuple(name for name in REQUIRED if name != "experimentA27_1_manifest.json")


class A280Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise A280Error(f"cannot load JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A280Error(f"JSON object required: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise A280Error(f"cannot read CSV: {path}: {exc}") from exc
    if not rows:
        raise A280Error(f"CSV is empty: {path}")
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate(input_dir: Path) -> tuple[dict, dict, list[dict[str, str]], dict[str, str]]:
    paths = {name: input_dir / name for name in REQUIRED}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise A280Error("required A27.1 artifacts are missing: " + "; ".join(missing))
    decision = load_json(paths["experimentA27_1_confirmation_decision.json"])
    manifest = load_json(paths["experimentA27_1_manifest.json"])
    if decision.get("experiment_id") != "experimentA27_1" or not decision.get("complete") or not decision.get("passed"):
        raise A280Error("A27.1 must be a completed, passed execution")
    if not decision.get("execution_integrity_passed") or not decision.get("matched_target_compute_passed"):
        raise A280Error("A27.1 integrity or matched-compute assertion failed")
    if decision.get("all_advancement_gates_passed") or decision.get("candidate_selected"):
        raise A280Error("A28.0 is only valid for an unselected A27.1 candidate")
    expected = "abandon_reptile_gnn_low_rul_repair_without_retuning_lambda_threshold_epochs_graph_or_gates"
    if decision.get("next_action") != expected:
        raise A280Error("A27.1 negative-result closure action does not match the preregistered rule")
    if decision.get("A25_2b_confirmation_used") or decision.get("official_test_files_accessed"):
        raise A280Error("A27.1 must not have used confirmation or official-test data")
    artifact_hashes = manifest.get("artifacts")
    if not isinstance(artifact_hashes, dict):
        raise A280Error("A27.1 manifest has no artifact hash inventory")
    observed_hashes: dict[str, str] = {}
    for name in HASH_COVERED_REQUIRED:
        if name not in artifact_hashes:
            raise A280Error(f"A27.1 manifest does not cover required artifact: {name}")
        observed_hashes[name] = sha256(paths[name])
        if observed_hashes[name] != artifact_hashes[name]:
            raise A280Error(f"A27.1 artifact hash mismatch: {name}")
    gates = load_csv(paths["experimentA27_1_advancement_gate_results.csv"])
    fields = {"gate_family", "gate_id", "observed", "operator", "threshold", "passed"}
    if not fields.issubset(gates[0]):
        raise A280Error("A27.1 gate table has an unexpected schema")
    failed = [row for row in gates if row.get("passed") == "False"]
    if not failed:
        raise A280Error("A27.1 decision says gates failed but gate table has no failed rows")
    if len(gates) != int(decision.get("completed_advancement_gate_records", -1)):
        raise A280Error("gate record count disagrees with A27.1 decision")
    return decision, manifest, failed, observed_hashes


def report(decision: dict, failed: list[dict[str, str]]) -> str:
    family_counts: dict[str, list[int]] = {}
    for row in failed:
        family_counts.setdefault(row["gate_family"], []).append(1)
    lines = [
        "A28.0 阴性结果闭环冻结报告",
        "=" * 28,
        "",
        "A27.1 执行完整且计算匹配，但候选臂未通过全部预注册推进门槛。",
        "本报告冻结该结论：候选不入选；禁止在 A27.1 结果上重调 lambda、RUL 阈值、epoch、图结构或门槛。",
        "",
        f"完成 worker：{decision['completed_worker_cells']}/{decision['expected_worker_cells']}",
        f"完成 run records：{decision['completed_run_level_records']}/{decision['expected_run_level_records']}",
        f"失败门槛：{len(failed)}/{decision['completed_advancement_gate_records']}",
        "",
        "失败门槛清单：",
    ]
    for row in failed:
        lines.append(f"- {row['gate_id']}: observed={row['observed']} {row['operator']} threshold={row['threshold']} ({row['gate_family']})")
    lines.extend([
        "",
        "后续边界：不得访问官方测试，不得将 A25.2b confirmation 或 A27.1 selection 结果用于该候选路线的重新选择。",
        "若未来提出全新的方法问题，必须使用新的独立开发角色和新的预注册合同。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a27-1-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confirm-freeze", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.confirm_freeze != FREEZE:
        raise A280Error(f"--confirm-freeze must equal {FREEZE}")
    input_dir, output_dir = Path(args.a27_1_output_dir).resolve(), Path(args.output_dir).resolve()
    decision, _manifest, failed, observed_hashes = validate(input_dir)
    preview = {
        "experiment_id": EXPERIMENT, "script_version": VERSION, "dry_run": args.dry_run,
        "A27_1_execution_integrity_verified": True,
        "A27_1_all_advancement_gates_passed": False,
        "failed_advancement_gates": len(failed), "candidate_selected": False,
        "new_predictor_training": False, "model_forward_run": False,
        "A25_2b_confirmation_used": False, "official_test_files_accessed": False,
        "passed": True,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("[A28.0] dry-run passed; negative-result closure is ready and no file was written")
        return 0
    final_path = output_dir / "experimentA28_0_confirmation_decision.json"
    if final_path.exists() and not args.resume:
        raise A280Error(f"output already complete: {final_path}; use a new directory or --resume")
    failed_fields = list(failed[0])
    failed_path = output_dir / "experimentA28_0_failed_gate_inventory.csv"
    report_path = output_dir / "experimentA28_0_negative_result_closure_report.txt"
    atomic_csv(failed_path, failed_fields, failed)
    atomic_write(report_path, report(decision, failed))
    final = {
        **preview, "complete": True, "preflight_only": True, "closure_only": True,
        "A27_1_candidate_route_closed": True,
        "candidate_selected": False,
        "formal_efficacy_claim": False,
        "official_test_forward_run": False,
        "reason": "A28.0 froze the preregistered A27.1 negative result and prohibited retuning this candidate route",
        "next_action": "archive_A27_A28_negative_result; any new method requires independent development roles and a new preregistration",
        "input_sha256": observed_hashes,
    }
    atomic_write(final_path, json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    manifest = {"experiment_id": EXPERIMENT, "script_version": VERSION,
                "artifacts": {p.name: sha256(p) for p in (failed_path, report_path, final_path)}}
    atomic_write(output_dir / "experimentA28_0_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print("[A28.0] completed negative-result closure; no predictor was trained")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A280Error as exc:
        print(f"[A28.0] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
