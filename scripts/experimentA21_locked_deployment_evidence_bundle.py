"""Experiment A21: package and freeze the completed locked A9_1 policy evidence.

CPU-only.  It reads only completed result artifacts, copies a strict allow-list
into a release bundle, and writes hashes.  It never trains, forwards a model,
opens raw official test data, or changes a policy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any


EXPERIMENT_ID = "experimentA21"
VERSION = "experimentA21_locked_deployment_evidence_bundle_v1"
QUESTION = (
    "Can the completed locked A9_1 ten-epoch policy, its official confirmation, "
    "A19 training-only robustness synthesis and A20 reproducibility audit be frozen "
    "as a hash-verified evidence-and-policy bundle without retraining or retuning?"
)
DEFAULT_OUTPUT = "outputs/experimentA21_locked_deployment_evidence_bundle"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a9-1-output-dir", required=True)
    p.add_argument("--a19-output-dir", required=True)
    p.add_argument("--a20-output-dir", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    return value


def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_atomic(path, json.dumps(norm(value), ensure_ascii=False, indent=2, allow_nan=False))


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required A21 artifact is missing or empty: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"A21 expected a JSON object: {path}")
    return value


def require(value: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, target in expected.items():
        if value.get(key) != target:
            raise RuntimeError(f"A21 requires {label}.{key}={target!r}, found {value.get(key)!r}")


@contextmanager
def lock(root: Path):
    p = root / "experimentA21_run.lock"
    h = p.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            h.seek(0)
            raise RuntimeError(f"another A21 process owns {root}: {h.read().strip()}") from e
        h.seek(0); h.truncate(); h.write(json.dumps({"pid": os.getpid(), "version": VERSION})); h.flush()
        yield
    finally:
        fcntl.flock(h.fileno(), fcntl.LOCK_UN)
        h.close()


def bundle_files(a9: Path, a19: Path, a20: Path) -> list[tuple[str, Path]]:
    # Deliberately exclude all raw predictions, paired metrics, raw official test
    # files, checkpoints and source data.  This is a policy/evidence bundle only.
    return [
        ("policy/experimentA9_1_locked_policy.json", a9 / "experimentA9_1_locked_policy.json"),
        ("policy/experimentA9_1_locked_policy.csv", a9 / "experimentA9_1_locked_policy.csv"),
        ("evidence/experimentA9_1_confirmation_decision.json", a9 / "experimentA9_1_confirmation_decision.json"),
        ("evidence/experimentA9_1_official_test_integrity.json", a9 / "experimentA9_1_official_test_integrity.json"),
        ("evidence/experimentA19_confirmation_decision.json", a19 / "experimentA19_confirmation_decision.json"),
        ("evidence/experimentA19_input_integrity.json", a19 / "experimentA19_input_integrity.json"),
        ("evidence/experimentA20_confirmation_decision.json", a20 / "experimentA20_confirmation_decision.json"),
        ("evidence/experimentA20_input_integrity.json", a20 / "experimentA20_input_integrity.json"),
    ]


def release_note(policy_hash: str) -> str:
    return f"""A21 LOCKED POLICY AND EVIDENCE BUNDLE
=====================================

Policy: A9_1 ten-epoch baseline/cycle-age blend
Locked policy hash: {policy_hash}

This bundle contains only the policy and decision/integrity evidence needed to
audit the completed result. It is NOT an inference-ready model package: model
weights, runtime code, source data and raw official-test predictions are absent.

Permitted use: reproduce policy identity and inspect completed evidence.
Prohibited use: retune alpha/gates/epochs, use official-test metrics for model
selection, or infer that this evidence establishes new performance claims.
"""


def main() -> None:
    ns = args()
    a9 = Path(ns.a9_1_output_dir).expanduser().resolve()
    a19 = Path(ns.a19_output_dir).expanduser().resolve()
    a20 = Path(ns.a20_output_dir).expanduser().resolve()
    root = Path(ns.output_dir).expanduser().resolve()
    if root in (a9, a19, a20):
        raise ValueError("A21 output directory must differ from every input directory")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        final = root / "experimentA21_confirmation_decision.json"
        if final.exists() and not ns.resume:
            raise RuntimeError("A21 already has a decision; use --resume or a new output directory")

        a9_decision = load(a9 / "experimentA9_1_confirmation_decision.json")
        a19_decision = load(a19 / "experimentA19_confirmation_decision.json")
        a20_decision = load(a20 / "experimentA20_confirmation_decision.json")
        require(a9_decision, {"experiment_id": "experimentA9_1", "complete": True, "passed": True,
                              "selection_was_locked_before_official_test": True}, "A9_1 decision")
        require(a19_decision, {"experiment_id": "experimentA19", "complete": True, "passed": True,
                               "new_predictor_training": False}, "A19 decision")
        require(a20_decision, {"experiment_id": "experimentA20", "complete": True, "passed": True,
                               "metric_reproduction_passed": True, "deterministic_reconstruction_passed": True}, "A20 decision")
        policy_hash = a9_decision.get("locked_policy_hash")
        if not isinstance(policy_hash, str) or policy_hash != a20_decision.get("locked_policy_hash"):
            raise RuntimeError("A9_1 and A20 locked policy hashes do not agree")

        selected = bundle_files(a9, a19, a20)
        rows = []
        for rel, source in selected:
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"required A21 source file is missing or empty: {source}")
            rows.append({"bundle_path": rel, "source_path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256(source)})
        dry = {"experiment_id": EXPERIMENT_ID, "script_version": VERSION, "registered_primary_question": QUESTION,
               "output_dir": str(root), "selected_artifacts": len(rows), "locked_policy_hash": policy_hash,
               "a9_1_passed": True, "a19_passed": True, "a20_passed": True,
               "new_predictor_training": False, "policy_selection_or_tuning": False,
               "raw_official_test_files_accessed": False, "official_test_forward_run": False}
        write_json(root / "experimentA21_dry_run.json", dry)
        print(json.dumps(dry, ensure_ascii=False, indent=2), flush=True)
        if ns.dry_run:
            print("[A21] dry-run completed; no bundle files were copied", flush=True)
            return

        bundle = root / "locked_A9_1_evidence_bundle"
        if bundle.exists() and any(bundle.iterdir()) and not ns.resume:
            raise RuntimeError("A21 bundle directory is non-empty; use a new output directory or --resume")
        for rel, source in selected:
            destination = bundle / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != sha256(source):
                raise RuntimeError(f"A21 hash mismatch after copying {rel}")
        write_atomic(bundle / "README_deployment_bundle.txt", release_note(policy_hash))
        manifest = {"experiment_id": EXPERIMENT_ID, "script_version": VERSION, "script_sha256": sha256(Path(__file__)),
                    "registered_primary_question": QUESTION, "locked_policy_hash": policy_hash,
                    "bundle_type": "policy_and_evidence_only_not_inference_ready", "artifacts": rows,
                    "new_predictor_training": False, "policy_selection_or_tuning": False,
                    "raw_official_test_files_accessed": False, "official_test_forward_run": False}
        write_json(bundle / "bundle_manifest.json", manifest)
        archive = root / "locked_A9_1_evidence_bundle.tar.gz"
        temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(bundle, arcname=bundle.name, recursive=True)
        os.replace(temporary, archive)
        manifest["archive_file"] = str(archive)
        manifest["archive_sha256"] = sha256(archive)
        write_json(root / "experimentA21_manifest.json", manifest)
        decision = {"experiment_id": EXPERIMENT_ID, "registered_primary_question": QUESTION, "complete": True,
                    "quick_mode": False, "new_predictor_training": False, "policy_selection_or_tuning": False,
                    "locked_policy_hash": policy_hash, "selected_artifacts": len(rows),
                    "bundle_hash_verified": True, "archive_sha256": manifest["archive_sha256"],
                    "passed": True,
                    "reason": "A21 froze the completed A9_1 policy and supporting A19/A20 evidence as a hash-verified policy-and-evidence bundle",
                    "interpretation_limit": "The bundle is not an inference-ready model package and does not create a new efficacy or official-test claim.",
                    "next_action": "archive_bundle_and_freeze_performance_experimentation",
                    "raw_official_test_files_accessed": False, "official_test_forward_run": False}
        write_json(final, decision)
        print("[A21] completed locked deployment evidence bundle", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
