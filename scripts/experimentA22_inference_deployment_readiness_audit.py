"""Experiment A22: locked A9_1 inference deployment-readiness audit.

This CPU/GPU-neutral audit consumes the completed A21 evidence bundle plus an
explicit deployment specification.  It never trains, tunes, selects a
checkpoint, or reads official C-MAPSS test data.  The same synthetic smoke test
is executed in two independent processes; a release bundle is created only
when artifact hashes, policy identity, contract checks and determinism pass.

Smoke-runner contract
---------------------
The declared ``smoke_runner`` Python file is called twice as:

    python RUNNER --deployment-spec SPEC --output-json OUTPUT

It must atomically write a JSON object containing:
  passed=true, fixture_type="synthetic", n_predictions>0,
  prediction_sha256=<64 lowercase hex>, prediction_shape=<non-empty list>,
  official_test_files_accessed=false, official_test_forward_run=false.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any


EXPERIMENT_ID = "experimentA22"
SCRIPT_VERSION = "experimentA22_inference_deployment_readiness_audit_v1"
SPEC_VERSION = "a22_deployment_spec_v1"
QUESTION = (
    "Can explicitly identified artifacts for the locked A9_1 ten-epoch policy "
    "form a hash-bound, deterministic, inference-ready bundle without retraining, "
    "retuning, checkpoint selection or official-test reuse?"
)
DEFAULT_OUTPUT = "outputs/experimentA22_inference_deployment_readiness_audit"
REQUIRED_ROLES = {
    "checkpoint",
    "preprocessor_state",
    "inference_entry",
    "runtime_config",
    "environment_lock",
    "smoke_runner",
}
OPTIONAL_ROLES = {"feature_schema", "model_definition", "label_mapping"}
HEX = set("0123456789abcdef")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a21-output-dir")
    p.add_argument("--project-root")
    p.add_argument("--deployment-spec")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--smoke-timeout-seconds", type=int, default=600)
    p.add_argument("--init-spec", metavar="PATH")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


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
        json.dumps(normalize(payload), ensure_ascii=False, indent=2, allow_nan=False),
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required A22 JSON is missing or empty: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"A22 expected a JSON object: {path}")
    return value


def require(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"A22 requires {label}.{key}={value!r}, found {payload.get(key)!r}"
            )


@contextmanager
def exclusive_lock(root: Path):
    path = root / "experimentA22_run.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise RuntimeError(
                f"another A22 process owns {root}: {handle.read().strip() or 'unknown'}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "version": SCRIPT_VERSION}))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def spec_template() -> dict[str, Any]:
    return {
        "schema_version": SPEC_VERSION,
        "locked_policy_hash": "0ab6977439ef7ea95043",
        "device": "cpu",
        "input_contract": {
            "window_length": 30,
            "rul_unit": "cycles",
            "feature_order": ["REPLACE_WITH_EXACT_ORDER"],
            "age_feature_transform": "log1p(cycle), source-standardized",
        },
        "artifacts": [
            {"role": "checkpoint", "path": "REPLACE/checkpoint.pt", "sha256": "REPLACE_WITH_64_HEX"},
            {"role": "preprocessor_state", "path": "REPLACE/preprocessor.json", "sha256": "REPLACE_WITH_64_HEX"},
            {"role": "inference_entry", "path": "REPLACE/inference.py", "sha256": "REPLACE_WITH_64_HEX"},
            {"role": "runtime_config", "path": "REPLACE/runtime_config.json", "sha256": "REPLACE_WITH_64_HEX"},
            {"role": "environment_lock", "path": "REPLACE/requirements-lock.txt", "sha256": "REPLACE_WITH_64_HEX"},
            {"role": "smoke_runner", "path": "REPLACE/a22_smoke_runner.py", "sha256": "REPLACE_WITH_64_HEX"},
        ],
        "synthetic_smoke_test": {"seed": 22001, "minimum_predictions": 2},
        "official_test_files_accessed": False,
        "official_test_forward_run": False,
    }


def initialize_spec(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing deployment spec: {path}")
    atomic_json(path, spec_template())
    print(f"[A22] deployment-spec template created: {path}", flush=True)
    print("[A22] fill exact artifact paths/hashes; do not let A22 choose a checkpoint", flush=True)


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_spec(
    spec_path: Path, project_root: Path, policy_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    spec = load_json(spec_path)
    require(
        spec,
        {
            "schema_version": SPEC_VERSION,
            "locked_policy_hash": policy_hash,
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
        "deployment spec",
    )
    if spec.get("device") not in ("cpu", "cuda"):
        raise RuntimeError("deployment spec device must be 'cpu' or 'cuda'")
    contract = spec.get("input_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("deployment spec lacks input_contract")
    if not isinstance(contract.get("window_length"), int) or contract["window_length"] <= 0:
        raise RuntimeError("input_contract.window_length must be a positive integer")
    if contract.get("rul_unit") != "cycles":
        raise RuntimeError("input_contract.rul_unit must be 'cycles'")
    features = contract.get("feature_order")
    if (
        not isinstance(features, list) or not features
        or any(not isinstance(x, str) or not x.strip() for x in features)
        or len(features) != len(set(features))
        or any("REPLACE" in x for x in features)
    ):
        raise RuntimeError("input_contract.feature_order must be explicit, unique and complete")
    smoke = spec.get("synthetic_smoke_test")
    if not isinstance(smoke, dict):
        raise RuntimeError("deployment spec lacks synthetic_smoke_test")
    if not isinstance(smoke.get("seed"), int):
        raise RuntimeError("synthetic_smoke_test.seed must be an integer")
    if not isinstance(smoke.get("minimum_predictions"), int) or smoke["minimum_predictions"] <= 0:
        raise RuntimeError("synthetic_smoke_test.minimum_predictions must be positive")

    entries = spec.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("deployment spec artifacts must be a non-empty list")
    rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    allowed_roles = REQUIRED_ROLES | OPTIONAL_ROLES
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("each artifact declaration must be an object")
        role = entry.get("role")
        raw_path = entry.get("path")
        declared_hash = entry.get("sha256")
        if role not in allowed_roles:
            raise RuntimeError(f"unsupported deployment artifact role: {role!r}")
        if role in paths:
            raise RuntimeError(f"artifact role must be unique: {role}")
        if not isinstance(raw_path, str) or not raw_path or "REPLACE" in raw_path:
            raise RuntimeError(f"artifact {role} has no explicit path")
        if not valid_sha256(declared_hash):
            raise RuntimeError(f"artifact {role} requires an exact lowercase SHA-256")
        unresolved = Path(raw_path).expanduser()
        candidate_path = project_root / unresolved if not unresolved.is_absolute() else unresolved
        if candidate_path.is_symlink():
            raise RuntimeError(f"artifact {role} must not be a symbolic link: {candidate_path}")
        path = candidate_path.resolve()
        if not within(path, project_root):
            raise RuntimeError(f"artifact {role} is outside project root: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"artifact {role} must be a non-empty regular non-symlink file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != declared_hash:
            raise RuntimeError(
                f"artifact {role} SHA-256 mismatch: declared={declared_hash}, actual={actual_hash}"
            )
        paths[role] = path
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": actual_hash,
            }
        )
    missing = sorted(REQUIRED_ROLES - set(paths))
    if missing:
        raise RuntimeError(f"deployment spec lacks required artifact roles: {missing}")
    for role in ("inference_entry", "smoke_runner"):
        if paths[role].suffix != ".py":
            raise RuntimeError(f"{role} must be a Python file")
        py_compile.compile(str(paths[role]), doraise=True)
    return spec, sorted(rows, key=lambda x: x["role"]), paths


def verify_a21(a21_root: Path) -> tuple[dict[str, Any], Path]:
    decision_path = a21_root / "experimentA21_confirmation_decision.json"
    archive = a21_root / "locked_A9_1_evidence_bundle.tar.gz"
    decision = load_json(decision_path)
    require(
        decision,
        {
            "experiment_id": "experimentA21",
            "complete": True,
            "passed": True,
            "bundle_hash_verified": True,
            "new_predictor_training": False,
            "policy_selection_or_tuning": False,
            "raw_official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
        "A21 decision",
    )
    if not archive.is_file() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"A21 archive is missing: {archive}")
    actual = sha256_file(archive)
    if actual != decision.get("archive_sha256"):
        raise RuntimeError(
            f"A21 archive SHA-256 mismatch: registered={decision.get('archive_sha256')}, actual={actual}"
        )
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError("A21 evidence archive is empty")
        for member in members:
            if member.issym() or member.islnk() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise RuntimeError(f"unsafe A21 archive member: {member.name}")
    return decision, archive


def validate_smoke_result(
    payload: dict[str, Any], minimum_predictions: int, label: str
) -> None:
    require(
        payload,
        {
            "passed": True,
            "fixture_type": "synthetic",
            "official_test_files_accessed": False,
            "official_test_forward_run": False,
        },
        label,
    )
    if not isinstance(payload.get("n_predictions"), int) or payload["n_predictions"] < minimum_predictions:
        raise RuntimeError(f"{label}.n_predictions is below the registered minimum")
    if not valid_sha256(payload.get("prediction_sha256")):
        raise RuntimeError(f"{label}.prediction_sha256 is invalid")
    shape = payload.get("prediction_shape")
    if not isinstance(shape, list) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise RuntimeError(f"{label}.prediction_shape must be a positive integer list")


def run_smoke(
    runner: Path,
    spec_path: Path,
    project_root: Path,
    output_path: Path,
    timeout: int,
    device: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "A22_DEVICE": device,
            "A22_SYNTHETIC_ONLY": "1",
        }
    )
    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--deployment-spec",
            str(spec_path),
            "--output-json",
            str(output_path),
        ],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"A22 smoke runner failed with exit={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-4000:]}\n"
            f"stderr tail:\n{result.stderr[-4000:]}"
        )
    payload = load_json(output_path)
    payload["elapsed_seconds"] = elapsed
    payload["stdout_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    payload["stderr_sha256"] = hashlib.sha256(result.stderr.encode()).hexdigest()
    return payload


def copy_verified(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_hash:
        raise RuntimeError(f"hash mismatch after copying deployment artifact: {destination}")


def main() -> None:
    ns = parse_args()
    if ns.init_spec:
        initialize_spec(Path(ns.init_spec).expanduser().resolve())
        return
    for option in ("a21_output_dir", "project_root", "deployment_spec"):
        if not getattr(ns, option):
            raise ValueError(f"--{option.replace('_', '-')} is required unless --init-spec is used")
    if ns.smoke_timeout_seconds < 10 or ns.smoke_timeout_seconds > 3600:
        raise ValueError("--smoke-timeout-seconds must be between 10 and 3600")

    a21_root = Path(ns.a21_output_dir).expanduser().resolve()
    project_root = Path(ns.project_root).expanduser().resolve()
    spec_path = Path(ns.deployment_spec).expanduser().resolve()
    root = Path(ns.output_dir).expanduser().resolve()
    if not project_root.is_dir():
        raise NotADirectoryError(f"project root does not exist: {project_root}")
    if root in (a21_root, project_root) or within(root, a21_root):
        raise ValueError("A22 output must be separate from the project root and A21 input")
    root.mkdir(parents=True, exist_ok=True)

    with exclusive_lock(root):
        final_path = root / "experimentA22_deployment_readiness_decision.json"
        if final_path.exists() and not ns.resume:
            raise RuntimeError("A22 already completed here; use --resume or a new output directory")
        a21_decision, a21_archive = verify_a21(a21_root)
        policy_hash = a21_decision.get("locked_policy_hash")
        if not isinstance(policy_hash, str) or len(policy_hash) < 12:
            raise RuntimeError("A21 decision lacks a credible locked policy hash")
        spec, inventory, paths = validate_spec(spec_path, project_root, policy_hash)
        spec_hash = hashlib.sha256(canonical_bytes(spec)).hexdigest()
        dry = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "registered_primary_question": QUESTION,
            "output_dir": str(root),
            "locked_policy_hash": policy_hash,
            "deployment_spec_sha256": spec_hash,
            "required_roles": sorted(REQUIRED_ROLES),
            "validated_artifacts": len(inventory),
            "artifact_hashes_passed": True,
            "python_compile_checks_passed": True,
            "a21_archive_hash_passed": True,
            "new_predictor_training": False,
            "checkpoint_selection": False,
            "policy_selection_or_tuning": False,
            "raw_official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(root / "experimentA22_dry_run.json", dry)
        print(json.dumps(dry, ensure_ascii=False, indent=2), flush=True)
        if ns.dry_run:
            print("[A22] dry-run completed; artifacts validated, inference was not executed", flush=True)
            return

        smoke_dir = root / "smoke_runs"
        smoke_dir.mkdir(exist_ok=True)
        smoke1 = run_smoke(
            paths["smoke_runner"], spec_path, project_root,
            smoke_dir / "run_1.json", ns.smoke_timeout_seconds, spec["device"],
        )
        smoke2 = run_smoke(
            paths["smoke_runner"], spec_path, project_root,
            smoke_dir / "run_2.json", ns.smoke_timeout_seconds, spec["device"],
        )
        minimum = spec["synthetic_smoke_test"]["minimum_predictions"]
        validate_smoke_result(smoke1, minimum, "smoke run 1")
        validate_smoke_result(smoke2, minimum, "smoke run 2")
        deterministic = bool(
            smoke1["prediction_sha256"] == smoke2["prediction_sha256"]
            and smoke1["prediction_shape"] == smoke2["prediction_shape"]
            and smoke1["n_predictions"] == smoke2["n_predictions"]
        )
        if not deterministic:
            raise RuntimeError("A22 independent-process smoke predictions are not deterministic")

        bundle = root / "locked_A9_1_inference_bundle"
        if bundle.exists() and any(bundle.iterdir()) and not ns.resume:
            raise RuntimeError("A22 inference bundle directory is non-empty")
        bundle.mkdir(parents=True, exist_ok=True)
        for row in inventory:
            source = Path(row["path"])
            destination = bundle / "artifacts" / row["role"] / source.name
            copy_verified(source, destination, row["sha256"])
            row["bundle_path"] = str(destination.relative_to(bundle))
        copy_verified(a21_archive, bundle / "evidence" / a21_archive.name, sha256_file(a21_archive))
        atomic_json(bundle / "deployment_spec.json", spec)
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "script_sha256": sha256_file(Path(__file__)),
            "registered_primary_question": QUESTION,
            "locked_policy_hash": policy_hash,
            "deployment_spec_sha256": spec_hash,
            "a21_archive_sha256": sha256_file(a21_archive),
            "artifacts": inventory,
            "smoke_prediction_sha256": smoke1["prediction_sha256"],
            "smoke_prediction_shape": smoke1["prediction_shape"],
            "new_predictor_training": False,
            "checkpoint_selection": False,
            "policy_selection_or_tuning": False,
            "raw_official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(bundle / "deployment_manifest.json", manifest)
        archive = root / "locked_A9_1_inference_bundle.tar.gz"
        temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(bundle, arcname=bundle.name, recursive=True)
        os.replace(temporary, archive)
        archive_hash = sha256_file(archive)
        audit = {
            "experiment_id": EXPERIMENT_ID,
            "artifact_inventory": inventory,
            "smoke_run_1": smoke1,
            "smoke_run_2": smoke2,
            "deterministic_across_independent_processes": deterministic,
            "archive_path": str(archive),
            "archive_sha256": archive_hash,
        }
        atomic_json(root / "experimentA22_deployment_audit.json", audit)
        decision = {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": QUESTION,
            "complete": True,
            "quick_mode": False,
            "locked_policy_hash": policy_hash,
            "required_artifact_roles": len(REQUIRED_ROLES),
            "validated_artifacts": len(inventory),
            "artifact_hashes_passed": True,
            "input_contract_passed": True,
            "synthetic_smoke_runs_completed": 2,
            "deterministic_inference_passed": deterministic,
            "inference_bundle_sha256": archive_hash,
            "deployment_ready": True,
            "passed": True,
            "reason": "A22 created a hash-bound inference bundle and reproduced identical synthetic predictions in two independent processes",
            "interpretation_limit": "A22 validates deployment artifacts and deterministic synthetic inference; it does not create a new efficacy or official-test claim.",
            "next_action": "stage_locked_bundle_in_a_nonproduction_environment_then_monitor_input_contract_and_runtime_health",
            "new_predictor_training": False,
            "checkpoint_selection": False,
            "policy_selection_or_tuning": False,
            "raw_official_test_files_accessed": False,
            "official_test_forward_run": False,
        }
        atomic_json(final_path, decision)
        print("[A22] completed inference deployment-readiness audit", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
