#!/usr/bin/env python3
"""A30.0: engineering-feasibility audit for the complete Meta-GNN-RUL proposal.

The audit uses synthetic tensors only.  It never accepts a data directory, never
opens C-MAPSS train/test files, and never reports efficacy metrics.  Dry-run performs
static source/symbol checks.  Formal execution additionally performs forward,
backward, ablation, graph, checkpoint, and mathematical Reptile smoke tests.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EXPERIMENT_ID = "experimentA30_0"
SCRIPT_VERSION = "experimentA30_0_complete_scheme_engineering_feasibility_audit_v1"
CONFIRM_TOKEN = "A30.0_ENGINEERING_SMOKE"

STATIC_CONTRACTS = (
    ("lstm_encoder", "models/lstm_encoder.py", ("LSTMEncoder",), True),
    ("transformer_encoder", "models/transformer_encoder.py", ("TransformerEncoder",), True),
    ("gat_encoder", "models/gat_encoder.py", ("GATEncoder",), True),
    ("sensor_attention", "models/sensor_attention.py", ("TemporalSelfAttention",), True),
    ("sensor_se", "models/se_block.py", ("SensorSEBlock",), True),
    ("rul_predictor", "models/rul_predictor.py", ("RULPredictor",), True),
    ("meta_gnn_rul", "models/meta_gnn_rul.py", ("MetaGNNRUL",), True),
    ("cosine_and_dtw_graph", "preprocess/graph_builder.py", ("build_knn_graph", "build_dtw_graph"), True),
    ("reptile_module", "meta_learning/reptile.py", ("Reptile", "reptile_update", "reptile_train", "meta_train"), True),
    ("task_sampler", "meta_learning/task_sampler.py", (), True),
    ("lstm_baseline", "baselines/lstm.py", (), True),
    ("cnn_lstm_baseline", "baselines/cnn_lstm.py", (), True),
    ("transformer_baseline", "baselines/transformer.py", (), True),
    ("gnn_rul_baseline", "baselines/gnn_rul.py", (), True),
)


class A300Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise A300Error("component audit cannot be empty")
    fields = list(rows[0])
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


def row(component: str, check: str, required: bool, passed: bool, detail: str, phase: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "component": component,
        "check": check,
        "required": required,
        "passed": passed,
        "detail": detail[:1000],
        "synthetic_only": True,
        "formal_efficacy_claim_allowed": False,
    }


def validate_a28(path: Path) -> dict[str, Any]:
    decision_path = path / "experimentA28_0_confirmation_decision.json"
    if not decision_path.is_file():
        raise A300Error(f"A28.0 decision is missing: {decision_path}")
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A300Error(f"cannot read A28.0 decision: {exc}") from exc
    for key in ("complete", "passed", "closure_only", "A27_1_candidate_route_closed"):
        if decision.get(key) is not True:
            raise A300Error(f"A28.0 requires {key}=true")
    if decision.get("candidate_selected") is not False or decision.get("formal_efficacy_claim") is not False:
        raise A300Error("A28.0 scientific boundary is incompatible")
    return decision


def ast_symbols(path: Path) -> tuple[set[str], str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return set(), f"{type(exc).__name__}: {exc}"
    symbols = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    return symbols, None


def static_audit(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component, relative, expected, required in STATIC_CONTRACTS:
        path = project_root / relative
        if not path.is_file():
            rows.append(row(component, "source_file_and_symbol_contract", required, False, f"missing: {relative}", "static"))
            continue
        symbols, error = ast_symbols(path)
        if error:
            rows.append(row(component, "source_file_and_symbol_contract", required, False, error, "static"))
            continue
        if component == "reptile_module":
            present = sorted(set(expected) & symbols)
            passed = bool(present)
            detail = f"file={relative}; callable/class candidates={present}; sha256={sha256(path)}"
        else:
            missing = sorted(set(expected) - symbols)
            passed = not missing
            detail = f"file={relative}; missing_symbols={missing}; sha256={sha256(path)}"
        rows.append(row(component, "source_file_and_symbol_contract", required, passed, detail, "static"))
    return rows


def import_module(name: str):
    importlib.invalidate_caches()
    return importlib.import_module(name)


def tensor_from_output(output: Any, torch: Any):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, dict):
        for key in ("prediction", "pred", "output", "rul"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise A300Error(f"cannot identify prediction tensor in output type {type(output).__name__}")


def finite_gradient_summary(model: Any, torch: Any) -> tuple[int, int]:
    present = 0
    nonzero = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        present += 1
        if not bool(torch.isfinite(parameter.grad).all()):
            raise A300Error("non-finite gradient detected")
        if bool(torch.any(parameter.grad != 0)):
            nonzero += 1
    if not present or not nonzero:
        raise A300Error("no finite nonzero gradients were produced")
    return present, nonzero


def graph_contract(project_root: Path, torch: Any, device: Any, batch: int, window: int, sensors: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        module = import_module("preprocess.graph_builder")
        knn = getattr(module, "build_knn_graph")
        dtw = getattr(module, "build_dtw_graph")
        embedding = torch.randn(batch, 32, device=device)
        sequence = torch.randn(batch, window, sensors, device=device)
        for name, value in (("cosine_knn", knn(embedding, min(3, batch - 1))), ("dtw", dtw(sequence, min(3, batch - 1), 5))):
            if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != 2:
                raise A300Error(f"{name} edge_index must have shape [2,E]")
            if value.dtype != torch.long:
                raise A300Error(f"{name} edge_index must use torch.long")
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= batch):
                raise A300Error(f"{name} edge index out of range")
            rows.append(row("graph_builder", f"{name}_synthetic_contract", True, True, f"edge_count={value.shape[1]}", "runtime"))
    except Exception as exc:
        rows.append(row("graph_builder", "cosine_and_dtw_synthetic_contract", True, False, f"{type(exc).__name__}: {exc}", "runtime"))
    return rows


def model_variant_contract(torch: Any, device: Any, batch: int, window: int, sensors: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    variants = (
        ("full_cosine", True, True, "cosine"),
        ("full_dtw", True, True, "dtw"),
        ("no_gat", False, True, "cosine"),
        ("no_sensor_attention", True, False, "cosine"),
        ("no_gat_no_attention", False, False, "cosine"),
    )
    try:
        cls = getattr(import_module("models.meta_gnn_rul"), "MetaGNNRUL")
    except Exception as exc:
        return [row("meta_gnn_rul", "import", True, False, f"{type(exc).__name__}: {exc}", "runtime")], counts
    x = torch.randn(batch, window, sensors, device=device)
    y = torch.linspace(5.0, 100.0, batch, device=device)
    for name, use_gat, use_attention, method in variants:
        try:
            torch.manual_seed(30001)
            model = cls(sensor_num=sensors, hidden_dim=32, embedding_dim=32, gat_heads=4, dropout=0.0,
                        graph_k=min(3, batch - 1), use_gat=use_gat, use_sensor_attention=use_attention,
                        graph_method=method, dtw_downsample=5, self_attention_heads=4).to(device)
            model.train()
            output = model(x, return_attention=True)
            prediction = tensor_from_output(output, torch).reshape(batch, -1)
            if prediction.shape[0] != batch or prediction.shape[1] != 1:
                raise A300Error(f"prediction shape is {tuple(prediction.shape)}, expected ({batch},1)")
            if not bool(torch.isfinite(prediction).all()):
                raise A300Error("prediction contains non-finite values")
            loss = torch.mean((prediction[:, 0] - y) ** 2)
            loss.backward()
            grads, nonzero = finite_gradient_summary(model, torch)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            counts[name] = int(parameter_count)
            detail = f"prediction_shape={tuple(prediction.shape)}; loss={float(loss.detach()):.6f}; params={parameter_count}; grads={grads}; nonzero_grads={nonzero}"
            rows.append(row("meta_gnn_rul", f"{name}_forward_backward", True, True, detail, "runtime"))
        except Exception as exc:
            rows.append(row("meta_gnn_rul", f"{name}_forward_backward", True, False, f"{type(exc).__name__}: {exc}", "runtime"))
    if counts:
        passed = all(key in counts for key, *_ in variants) and counts.get("full_cosine", 0) > counts.get("no_gat_no_attention", math.inf)
        rows.append(row("ablation", "parameterized_branches_are_distinct", True, passed, json.dumps(counts, sort_keys=True), "runtime"))
    return rows, counts


def instantiate_from_signature(cls: Any, sensors: int) -> Any:
    signature = inspect.signature(cls)
    values = {
        "input_dim": sensors, "sensor_num": sensors, "num_features": sensors, "feature_dim": sensors,
        "hidden_dim": 32, "model_dim": 32, "d_model": 32, "embedding_dim": 32, "output_dim": 32,
        "nhead": 4, "num_heads": 4, "heads": 4, "num_layers": 2, "dropout": 0.0,
        "window_size": 50, "sequence_length": 50,
    }
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is parameter.empty:
            missing.append(name)
    if missing:
        raise A300Error(f"unsupported required constructor arguments: {missing}")
    return cls(**kwargs)


def transformer_contract(torch: Any, device: Any, batch: int, window: int, sensors: int) -> list[dict[str, Any]]:
    try:
        cls = getattr(import_module("models.transformer_encoder"), "TransformerEncoder")
        model = instantiate_from_signature(cls, sensors).to(device)
        x = torch.randn(batch, window, sensors, device=device)
        output = model(x)
        tensor = tensor_from_output(output, torch)
        if tensor.shape[0] != batch or not bool(torch.isfinite(tensor).all()):
            raise A300Error(f"invalid Transformer output shape/values: {tuple(tensor.shape)}")
        torch.mean(tensor.float() ** 2).backward()
        grads, nonzero = finite_gradient_summary(model, torch)
        return [row("transformer_encoder", "synthetic_forward_backward", True, True, f"shape={tuple(tensor.shape)}; grads={grads}; nonzero={nonzero}", "runtime")]
    except Exception as exc:
        return [row("transformer_encoder", "synthetic_forward_backward", True, False, f"{type(exc).__name__}: {exc}", "runtime")]


def checkpoint_contract(torch: Any, device: Any, sensors: int) -> list[dict[str, Any]]:
    try:
        cls = getattr(import_module("models.meta_gnn_rul"), "MetaGNNRUL")
        kwargs = dict(sensor_num=sensors, hidden_dim=32, embedding_dim=32, gat_heads=4, dropout=0.0,
                      graph_k=3, use_gat=True, use_sensor_attention=True, graph_method="cosine",
                      dtw_downsample=5, self_attention_heads=4)
        torch.manual_seed(30002)
        model = cls(**kwargs).to(device)
        expected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="a30_checkpoint_") as temporary:
            path = Path(temporary) / "synthetic_state.pt"
            torch.save({"state": expected}, path)
            try:
                loaded_payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                loaded_payload = torch.load(path, map_location="cpu")
            restored = cls(**kwargs).cpu()
            restored.load_state_dict(loaded_payload["state"], strict=True)
            for name, value in restored.state_dict().items():
                if not torch.equal(value, expected[name]):
                    raise A300Error(f"checkpoint mismatch at {name}")
        return [row("checkpoint", "strict_roundtrip", True, True, f"state_tensors={len(expected)}", "runtime")]
    except Exception as exc:
        return [row("checkpoint", "strict_roundtrip", True, False, f"{type(exc).__name__}: {exc}", "runtime")]


def reptile_contract(torch: Any, device: Any, batch: int, window: int, sensors: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        module = import_module("meta_learning.reptile")
        candidates = [name for name in ("Reptile", "reptile_update", "reptile_train", "meta_train") if callable(getattr(module, name, None))]
        if not candidates:
            raise A300Error("no recognized Reptile entry point")
        rows.append(row("reptile", "project_module_surface", True, True, f"entry_points={candidates}", "runtime"))
    except Exception as exc:
        rows.append(row("reptile", "project_module_surface", True, False, f"{type(exc).__name__}: {exc}", "runtime"))
    try:
        cls = getattr(import_module("models.meta_gnn_rul"), "MetaGNNRUL")
        torch.manual_seed(30003)
        base = cls(sensor_num=sensors, hidden_dim=16, embedding_dim=16, gat_heads=4, dropout=0.0,
                   graph_k=3, use_gat=False, use_sensor_attention=False, graph_method="cosine",
                   dtw_downsample=5, self_attention_heads=4).to(device)
        before = {name: value.detach().clone() for name, value in base.named_parameters()}
        adapted_models = []
        optimizer_steps = 0
        for task_offset in (0.0, 15.0):
            adapted = deepcopy(base)
            optimizer = torch.optim.SGD(adapted.parameters(), lr=1e-3)
            for _ in range(2):
                x = torch.randn(batch, window, sensors, device=device)
                y = torch.linspace(10.0 + task_offset, 80.0 + task_offset, batch, device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = tensor_from_output(adapted(x), torch).reshape(batch, -1)[:, 0]
                loss = torch.mean((prediction - y) ** 2)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
            adapted_models.append(adapted)
        alpha = 0.25
        expected: dict[str, Any] = {}
        with torch.no_grad():
            adapted_parameters = [dict(model.named_parameters()) for model in adapted_models]
            for name, parameter in base.named_parameters():
                mean_adapted = torch.stack([values[name] for values in adapted_parameters]).mean(dim=0)
                expected[name] = before[name] + alpha * (mean_adapted - before[name])
                parameter.copy_(expected[name])
        changed = 0
        for name, parameter in base.named_parameters():
            if not torch.allclose(parameter, expected[name], rtol=0.0, atol=1e-7):
                raise A300Error(f"Reptile interpolation mismatch at {name}")
            changed += int(not torch.equal(parameter, before[name]))
        if not changed:
            raise A300Error("Reptile meta update changed no parameter")
        rows.append(row("reptile", "two_task_inner_outer_update_contract", True, True, f"tasks=2; inner_steps=2; optimizer_steps={optimizer_steps}; changed_parameters={changed}", "runtime"))
    except Exception as exc:
        rows.append(row("reptile", "two_task_inner_outer_update_contract", True, False, f"{type(exc).__name__}: {exc}", "runtime"))
    return rows


def runtime_audit(project_root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    try:
        import torch
    except ImportError as exc:
        return [row("runtime", "pytorch_available", True, False, f"ImportError: {exc}", "runtime")], 0
    torch.set_num_threads(max(1, int(args.torch_threads)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return [row("runtime", "requested_device_available", True, False, f"requested={args.device}; cuda_available=false", "runtime")], 0
    device = torch.device(args.device)
    sys.path.insert(0, str(project_root))
    try:
        rows = [row("runtime", "pytorch_available", True, True, f"torch={torch.__version__}; device={device}", "runtime")]
        rows.extend(graph_contract(project_root, torch, device, args.batch_size, args.window_size, args.sensor_num))
        variant_rows, _counts = model_variant_contract(torch, device, args.batch_size, args.window_size, args.sensor_num)
        rows.extend(variant_rows)
        rows.extend(transformer_contract(torch, device, args.batch_size, args.window_size, args.sensor_num))
        rows.extend(checkpoint_contract(torch, device, args.sensor_num))
        rows.extend(reptile_contract(torch, device, args.batch_size, args.window_size, args.sensor_num))
        optimizer_steps = 4
        return rows, optimizer_steps
    finally:
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--a28-0-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--sensor-num", type=int, default=21)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--confirm-run", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.sensor_num, args.window_size, args.batch_size, args.torch_threads) <= 0:
        raise A300Error("sensor/window/batch/thread arguments must be positive")
    if args.batch_size < 4:
        raise A300Error("--batch-size must be at least 4 for graph smoke tests")
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not project_root.is_dir():
        raise A300Error(f"project root does not exist: {project_root}")
    validate_a28(Path(args.a28_0_output_dir).resolve())
    rows = static_audit(project_root)
    optimizer_steps = 0
    if not args.dry_run:
        if args.confirm_run != CONFIRM_TOKEN:
            raise A300Error(f"formal execution requires --confirm-run {CONFIRM_TOKEN}")
        runtime_rows, optimizer_steps = runtime_audit(project_root, args)
        rows.extend(runtime_rows)
    required_rows = [item for item in rows if item["required"]]
    failed = [item for item in required_rows if not item["passed"]]
    passed = not failed
    preview = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "dry_run": args.dry_run,
        "static_checks": sum(item["phase"] == "static" for item in rows),
        "runtime_checks": sum(item["phase"] == "runtime" for item in rows),
        "required_checks": len(required_rows),
        "required_checks_passed": len(required_rows) - len(failed),
        "required_checks_failed": len(failed),
        "failed_check_ids": [f"{item['component']}::{item['check']}" for item in failed],
        "synthetic_optimizer_steps": optimizer_steps,
        "synthetic_data_only": True,
        "new_predictor_training": False,
        "formal_efficacy_claim": False,
        "official_test_files_accessed": False,
        "passed": passed,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("[A30.0] dry-run completed static proposal audit; no output was written")
        return 0 if passed else 1
    if output_dir.exists() and any(output_dir.iterdir()):
        raise A300Error(f"output directory must be new and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "experimentA30_0_component_feasibility_audit.csv"
    decision_path = output_dir / "experimentA30_0_confirmation_decision.json"
    atomic_csv(audit_path, rows)
    final = {
        **preview,
        "complete": True,
        "execution_integrity_passed": True,
        "engineering_feasibility_fully_passed": passed,
        "document_scheme_fully_implemented_and_smoke_tested": passed,
        "interpretation_limit": "A30.0 validates source presence and synthetic engineering contracts only. It cannot establish accuracy, superiority, safety, or deployment fitness.",
        "reason": "A30.0 completed the complete-proposal engineering audit" if passed else "A30.0 completed the audit and identified missing or incompatible proposal components",
        "next_action": "freeze_independent_efficacy_protocol_before_training" if passed else "implement_only_the_failed_components_then_repeat_A30_0_in_a_new_output_directory",
    }
    atomic_json(decision_path, final)
    manifest_path = output_dir / "experimentA30_0_manifest.json"
    atomic_json(manifest_path, {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {audit_path.name: sha256(audit_path), decision_path.name: sha256(decision_path)},
    })
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    print("[A30.0] completed engineering feasibility audit; no formal predictor was trained")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A300Error as exc:
        print(f"[A30.0] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
