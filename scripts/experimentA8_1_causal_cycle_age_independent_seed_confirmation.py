"""Experiment A8_1: independent-seed confirmation of the causal cycle-age input.

This is deliberately a *replication*, not a new modelling direction.  It keeps
all of A8's causal feature, architecture, loss, source-pretraining budget and
the registered A2_1 engine-role protocol fixed.  Only the model seeds and the
selection/confirmation endpoint seeds are new:

* model seeds: 90--94 (A8 used 80--84);
* selection endpoint seeds: 8601--8605 (A8 used 8401--8405);
* confirmation endpoint seeds: 8701--8705 (A8 used 8501--8505).

The five A2_1 target-split seeds (6401--6405) are intentionally retained:
they identify the already registered disjoint adaptation/evaluation role
partitions.  Thus A8_1 tests independence with respect to model and endpoint
randomness without silently changing the experimental protocol.

The script delegates the well-tested data, training, GPU scheduling and audit
implementation to A8.  Before A8 starts, its immutable experiment constants
are replaced in-process.  Worker subprocesses execute this wrapper too, so
they receive exactly the same A8_1 configuration and write only to the A8_1
output directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import experimentA8_causal_cycle_age_representation_validation as a8  # noqa: E402


SCRIPT_VERSION = "experimentA8_1_causal_cycle_age_independent_seed_confirmation_v1"
EXPERIMENT_ID = "experimentA8_1"
DEFAULT_OUTPUT = "outputs/experimentA8_1_causal_cycle_age_independent_seed_confirmation"
MODEL_SEEDS = [90, 91, 92, 93, 94]
TARGET_SPLIT_SEEDS = [6401, 6402, 6403, 6404, 6405]
ROLE_PARTITIONS = [1, 2, 3, 4, 5]
SELECTION_ENDPOINT_SEEDS = [8601, 8602, 8603, 8604, 8605]
CONFIRMATION_ENDPOINT_SEEDS = [8701, 8702, 8703, 8704, 8705]
REGISTERED_QUESTION = (
    "Does the source-standardized causal cycle-age input reproduce A8's "
    "balanced-endpoint RMSE benefit with low/mid-RUL NASA benefit and "
    "high-RUL NASA/RMSE safety on an independent model/endpoint seed set?"
)


def install_a8_1_configuration() -> None:
    """Make every A8 helper operate under the A8_1 identity.

    A8 looks up these names from its module globals at call time.  Replacing
    them also ensures child workers launched by A8 point back to this wrapper
    (``a8.__file__``), rather than accidentally launching the original A8
    script with A8's old seeds.
    """

    a8.__file__ = str(Path(__file__).resolve())
    a8.SCRIPT_VERSION = SCRIPT_VERSION
    a8.EXPERIMENT_ID = EXPERIMENT_ID
    a8.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    a8.MODEL_SEEDS = MODEL_SEEDS.copy()
    a8.TARGET_SPLIT_SEEDS = TARGET_SPLIT_SEEDS.copy()
    a8.ROLE_PARTITIONS = ROLE_PARTITIONS.copy()
    a8.SELECTION_ENDPOINT_SEEDS = SELECTION_ENDPOINT_SEEDS.copy()
    a8.CONFIRMATION_ENDPOINT_SEEDS = CONFIRMATION_ENDPOINT_SEEDS.copy()


install_a8_1_configuration()
_A8_LOAD_CONFIG = a8.load_config
_A8_MAKE_DECISION = a8.make_decision
_A8_WRITE_INITIAL_ARTIFACTS = a8.write_initial_artifacts


def load_config(args: Any) -> tuple[dict, dict]:
    """Load A8's locked configuration, replacing only replication randomness."""

    base, experiment = _A8_LOAD_CONFIG(args)
    experiment = deepcopy(experiment)
    experiment.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": "causal_cycle_age_independent_seed_confirmation",
            "model_seeds": MODEL_SEEDS.copy(),
            "target_split_seeds": TARGET_SPLIT_SEEDS.copy(),
            "role_partitions": ROLE_PARTITIONS.copy(),
            "selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS.copy(),
            "confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS.copy(),
            "replication_of": "experimentA8",
            "replication_changes_only": {
                "model_seeds": {"A8": [80, 81, 82, 83, 84], "A8_1": MODEL_SEEDS},
                "selection_endpoint_seeds": {
                    "A8": [8401, 8402, 8403, 8404, 8405],
                    "A8_1": SELECTION_ENDPOINT_SEEDS,
                },
                "confirmation_endpoint_seeds": {
                    "A8": [8501, 8502, 8503, 8504, 8505],
                    "A8_1": CONFIRMATION_ENDPOINT_SEEDS,
                },
            },
            "registered_success_criteria": {
                "full_endpoint_rmse_ci95_upper": "< 0 (strict improvement)",
                "low_mid_rul_nasa_ci95_upper": "< 0 (strict improvement)",
                "high_rul_nasa_ci95_upper": "<= +3% (noninferiority)",
                "high_rul_rmse_ci95_upper": "<= +3% (noninferiority)",
            },
        }
    )
    if args.quick:
        # Keep the smoke test cheap while preserving the A8_1 identity and
        # disjoint endpoint sets.  It has no scientific interpretation.
        experiment.update(
            {
                "domains": ["FD004"],
                "model_seeds": [MODEL_SEEDS[0]],
                "target_split_seeds": [TARGET_SPLIT_SEEDS[0]],
                "role_partitions": [ROLE_PARTITIONS[0]],
                "selection_endpoint_seeds": [SELECTION_ENDPOINT_SEEDS[0]],
                "confirmation_endpoint_seeds": [CONFIRMATION_ENDPOINT_SEEDS[0]],
            }
        )
    return base, experiment


def write_initial_artifacts(
    paths: dict[str, Path], base: dict, experiment: dict, evidence: dict
) -> dict[str, Any]:
    """Add the independent-replication registry to A8's normal manifest."""

    manifest = _A8_WRITE_INITIAL_ARTIFACTS(paths, base, experiment, evidence)
    manifest.update(
        {
            "registered_primary_question": REGISTERED_QUESTION,
            "replication_of": "experimentA8",
            "independence": {
                "new_model_seeds": MODEL_SEEDS,
                "new_selection_endpoint_seeds": SELECTION_ENDPOINT_SEEDS,
                "new_confirmation_endpoint_seeds": CONFIRMATION_ENDPOINT_SEEDS,
                "selection_confirmation_endpoint_seeds_disjoint": True,
                "target_split_seeds_retained_from_registered_A2_1_protocol": TARGET_SPLIT_SEEDS,
            },
            "registered_success_criteria": experiment["registered_success_criteria"],
        }
    )
    a8.atomic_json(paths["manifest"], manifest)
    return manifest


def make_decision(**kwargs: Any) -> dict[str, Any]:
    """Apply the stricter, pre-registered A8_1 replication success rule."""

    decision = _A8_MAKE_DECISION(**kwargs)
    comparisons = kwargs["comparisons"]
    high = kwargs["high"]
    low = kwargs["low"]
    experiment = kwargs["experiment"]
    primary = comparisons[
        (comparisons["comparison"] == "full_endpoint_age_vs_baseline")
        & (comparisons["scope"] == "ALL")
    ].iloc[0]
    margin = float(experiment["stage_noninferiority_margin_pct"]) / 100.0
    full_rmse_improved = bool(float(primary["rmse_relative_boot_ci95_high"]) < 0.0)
    low_nasa_improved = bool(float(low["nasa_relative_ci95"][1]) < 0.0)
    high_nasa_safe = bool(float(high["nasa_relative_ci95"][1]) <= margin)
    high_rmse_safe = bool(float(high["rmse_relative_ci95"][1]) <= margin)
    strict_success = bool(
        decision["complete"]
        and not decision["official_test_files_accessed"]
        and not decision["official_test_forward_run"]
        and full_rmse_improved
        and low_nasa_improved
        and high_nasa_safe
        and high_rmse_safe
    )
    decision.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "registered_primary_question": REGISTERED_QUESTION,
            "replication_of": "experimentA8",
            "registered_success_criteria": experiment["registered_success_criteria"],
            "strict_replication_result": {
                "full_endpoint_rmse_ci95_upper": float(
                    primary["rmse_relative_boot_ci95_high"]
                ),
                "full_endpoint_rmse_strict_improvement": full_rmse_improved,
                "low_mid_rul_nasa_ci95_upper": float(low["nasa_relative_ci95"][1]),
                "low_mid_rul_nasa_strict_improvement": low_nasa_improved,
                "high_rul_nasa_ci95_upper": float(high["nasa_relative_ci95"][1]),
                "high_rul_nasa_noninferiority": high_nasa_safe,
                "high_rul_rmse_ci95_upper": float(high["rmse_relative_ci95"][1]),
                "high_rul_rmse_noninferiority": high_rmse_safe,
            },
            "passed": strict_success if not experiment["quick_mode"] else decision["complete"],
            "reason": (
                "quick smoke run only; do not interpret scientifically"
                if experiment["quick_mode"]
                else (
                    "A8_1 independently confirmed the causal cycle-age representation; "
                    "the next permitted step is pooled lock then official confirmation"
                    if strict_success
                    else "A8_1 completed, but the independent cycle-age replication did not meet every registered criterion"
                )
            ),
            "next_action": (
                None
                if experiment["quick_mode"]
                else (
                    "run_experimentA8_2_pooled_seed_lock_then_official_confirmation"
                    if strict_success
                    else "stop_single_channel_cycle_age_direction_and_reassess_experimentA9"
                )
            ),
        }
    )
    return decision


def run_workers(
    args: Any, tasks: list[tuple[str, int]], output: Path
) -> None:
    """Use A8's idle-GPU scheduler with unambiguous A8_1 log labels."""

    if args.single_process or args.device == "cpu" or args.device not in {"auto", "cpu"}:
        devices: list[str | int] = [args.device]
        inventory: list[dict] = []
    else:
        devices, inventory = a8.a4.choose_gpus(args)
        if not devices:
            raise RuntimeError(
                "no idle GPU met A8_1 thresholds; inventory="
                + a8.json.dumps(inventory, ensure_ascii=False)
            )
    print(
        a8.json.dumps(
            {
                "scheduler": EXPERIMENT_ID,
                "tasks": [{"domain": domain, "seed": seed} for domain, seed in tasks],
                "devices": devices,
                "gpu_inventory": inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    pending = list(tasks)
    active: dict[str | int, dict[str, Any]] = {}
    while pending or active:
        for device in [item for item in devices if item not in active]:
            if not pending:
                break
            domain, seed = pending.pop(0)
            directory = a8.shard_dir(output, domain, seed)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "worker_training.log"
            handle = log_path.open("a", encoding="utf-8")
            environment = a8.os.environ.copy()
            if isinstance(device, int):
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                command = a8.worker_command(args, domain, seed, "auto", output)
            else:
                command = a8.worker_command(args, domain, seed, str(device), output)
            process = a8.subprocess.Popen(
                command,
                cwd=a8.PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=a8.subprocess.STDOUT,
                text=True,
            )
            active[device] = {
                "process": process,
                "domain": domain,
                "seed": seed,
                "handle": handle,
                "log_path": log_path,
            }
            print(
                f"[A8_1] launched domain={domain} seed={seed} "
                f"device={device} pid={process.pid}"
            )
        finished: list[str | int] = []
        for device, record in active.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["handle"].close()
            if code != 0:
                tail = "\n".join(
                    record["log_path"]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-80:]
                )
                for other in active.values():
                    if other["process"].poll() is None:
                        other["process"].terminate()
                raise RuntimeError(
                    f"A8_1 worker failed domain={record['domain']} "
                    f"seed={record['seed']} exit={code}\n{tail}"
                )
            print(
                f"[A8_1] completed domain={record['domain']} "
                f"seed={record['seed']} device={device}"
            )
            finished.append(device)
        for device in finished:
            del active[device]
        if active and not finished:
            a8.time.sleep(5)


# Patch the functions looked up by a8.parent_main and a8.main.
a8.load_config = load_config
a8.write_initial_artifacts = write_initial_artifacts
a8.make_decision = make_decision
a8.run_workers = run_workers


def main() -> None:
    a8.main()


if __name__ == "__main__":
    main()
