#!/usr/bin/env python3
"""Read local W&B run files and summarize historical PPO/SAC evidence.

This intentionally uses the installed W&B datastore reader against local
``.wandb`` files only. It never initializes a W&B run or makes a network call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

RUN_SPECS: list[dict[str, Any]] = [
    {
        "label": "PPO baseline, 2026-11-06",
        "algorithm": "PPO",
        "wandb_path": (
            "wandb/run-20261106_135852-baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0/"
            "run-baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0.wandb"
        ),
        "world_size_evidence": "Two-rank launch script and the summed epoch/agent-step deltas.",
        "supporting_paths": [
            "project/baseline_ppo_4b_carla/train/launch_2gpu.sh",
            "pufferlib/pufferl.py",
        ],
        "ppo_epoch_is_sum_reduced": True,
    },
    {
        "label": "PPO baseline, 2026-08-26",
        "algorithm": "PPO",
        "wandb_path": (
            "wandb/run-20260826_162337-baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/"
            "run-baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0.wandb"
        ),
        "world_size_evidence": "Two-rank launch script and the summed epoch/agent-step deltas.",
        "supporting_paths": [
            "project/baseline_run_sync_2026-08-24/train/launch_2gpu.sh",
            "pufferlib/pufferl.py",
        ],
        "ppo_epoch_is_sum_reduced": True,
        "related_wandb_paths": [
            "wandb/run-20260911_194756-baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/"
            "run-baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0.wandb"
        ],
    },
    {
        "label": "SAC round 2 TES, 2026-10-29 trial 98",
        "algorithm": "SAC",
        "wandb_path": (
            "wandb/run-20261029_004934-round2_tes_sac_20261029T044847.014369240_539543_trial_98/"
            "run-round2_tes_sac_20261029T044847.014369240_539543_trial_98.wandb"
        ),
        "world_size_evidence": "Two rank runtime records and the saved launch command.",
        "supporting_paths": [
            "project/soft_actor_critic_target_entropy_scheduling/runs/"
            "round2_tes_sac_20261029T044847.014369240_539543/"
            "trial_98/console.log",
            "project/soft_actor_critic_target_entropy_scheduling/runs/"
            "round2_tes_sac_20261029T044847.014369240_539543/"
            "trial_98/rank_0_runtime.json",
            "project/soft_actor_critic_target_entropy_scheduling/runs/"
            "round2_tes_sac_20261029T044847.014369240_539543/"
            "trial_98/rank_1_runtime.json",
            "project/soft_actor_critic_target_entropy_scheduling/runs/"
            "round2_tes_sac_20261029T044847.014369240_539543/"
            "trial_98/learner/config.yaml",
        ],
        "ppo_epoch_is_sum_reduced": False,
    },
]


def safe_json_value(value: Any) -> Any:
    """Convert protobuf-decoded JSON values to strict JSON-compatible values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    return value


def decode_value(value_json: str) -> Any:
    try:
        return safe_json_value(json.loads(value_json))
    except (json.JSONDecodeError, TypeError):
        return value_json


def item_key(item: Any) -> str:
    nested = list(item.nested_key)
    if nested:
        return "/".join(nested)
    return item.key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


class ReadOnlyDataStore:
    """Thin read-only wrapper around W&B's local record scanner.

    W&B 0.28.1's ``DataStore.open_for_scan`` opens files as ``r+b`` even though
    scanning only reads. This subclass keeps the SDK's record parser and CRC
    checks, while forcing a read-only file descriptor.
    """

    def __init__(self) -> None:
        from wandb.sdk.internal.datastore import DataStore

        class _Reader(DataStore):
            def open_for_scan(self, filename: str) -> None:
                self._fname = filename
                self._fp = open(filename, "rb")
                self._index = 0
                self._size_bytes = os.stat(filename).st_size
                self._opened_for_scan = True
                self._read_header()

        self._reader = _Reader()

    def scan(self, path: Path) -> dict[str, Any]:
        from wandb.proto import wandb_internal_pb2

        reader = self._reader
        reader.open_for_scan(str(path))
        history: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}
        config: dict[str, Any] = {}
        environment_records: list[dict[str, Any]] = []
        environment_args: list[str] = []
        run_record: dict[str, Any] | None = None
        exit_codes: list[int] = []
        record_counts: dict[str, int] = {}
        try:
            while data := reader.scan_data():
                record = wandb_internal_pb2.Record.FromString(data)
                record_type = record.WhichOneof("record_type")
                if record_type is None:
                    continue
                record_counts[record_type] = record_counts.get(record_type, 0) + 1

                if record_type == "history":
                    row: dict[str, Any] = {}
                    for item in record.history.item:
                        key = item_key(item)
                        if key:
                            row[key] = decode_value(item.value_json)
                    if record.history.HasField("step"):
                        row["_step"] = record.history.step.num
                    history.append(row)
                elif record_type == "summary":
                    for item in record.summary.remove:
                        key = item_key(item)
                        if key:
                            summary.pop(key, None)
                    for item in record.summary.update:
                        key = item_key(item)
                        if key:
                            summary[key] = decode_value(item.value_json)
                elif record_type == "run":
                    run = record.run
                    run_record = {
                        "run_id": run.run_id,
                        "entity": run.entity,
                        "project": run.project,
                        "group": run.run_group,
                        "job_type": run.job_type,
                        "display_name": run.display_name,
                        "host": run.host,
                        "start_time_utc": (
                            run.start_time.ToDatetime()
                            .replace(tzinfo=dt.timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                            if run.HasField("start_time")
                            else None
                        ),
                    }
                    for item in run.config.update:
                        if item.key and item.key != "_wandb" and item.value_json:
                            config[item.key] = decode_value(item.value_json)
                elif record_type == "config":
                    for item in record.config.update:
                        if item.key and item.key != "_wandb" and item.value_json:
                            config[item.key] = decode_value(item.value_json)
                elif record_type == "environment":
                    env = record.environment
                    environment_records.append(
                        {
                            "host": env.host or None,
                            "os": env.os or None,
                            "python": env.python or None,
                            "cuda_version": env.cuda_version or None,
                            "gpu_type": env.gpu_type or None,
                            "gpu_count": env.gpu_count or None,
                            "cpu_count": env.cpu_count or None,
                            "cpu_count_logical": env.cpu_count_logical or None,
                            "program": env.program or None,
                            "started_at_utc": (
                                env.started_at.ToDatetime()
                                .replace(tzinfo=dt.timezone.utc)
                                .isoformat()
                                .replace("+00:00", "Z")
                                if env.HasField("started_at")
                                else None
                            ),
                        }
                    )
                    environment_args.extend(str(arg) for arg in env.args)
                elif record_type == "exit":
                    exit_codes.append(record.exit.exit_code)
        finally:
            reader.close()

        stat = path.stat()
        return {
            "path": relative_path(path),
            "size_bytes": stat.st_size,
            "sha256": sha256(path),
            "record_counts": record_counts,
            "history": history,
            "summary": summary,
            "config": config,
            "environment_records": environment_records,
            "environment_args": environment_args,
            "run": run_record,
            "exit_codes": exit_codes,
        }


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def first_config_value(environment_args: list[str], prefix: str) -> str | None:
    for argument in environment_args:
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def summarize_history(parsed: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    history = parsed["history"]
    config = parsed["config"]
    train = config.get("train", {})
    env = config.get("env", {})
    vec = config.get("vec", {})
    algorithm = spec["algorithm"]

    steps = [row.get("agent_steps") for row in history if isinstance(row.get("agent_steps"), (int, float))]
    epochs = [row.get("epoch") for row in history if isinstance(row.get("epoch"), (int, float))]
    eval_benchmark = train.get("evaluation_benchmarks")
    eval_metric_prefix = f"eval_{eval_benchmark}/" if isinstance(eval_benchmark, str) else "eval_"
    eval_rows = [
        row
        for row in history
        if any(key.startswith(eval_metric_prefix) for key in row)
    ]
    eval_score_key = f"{eval_metric_prefix}score"
    eval_score_rows = [row for row in history if eval_score_key in row]

    configured_interval = train.get("evaluation_interval_epochs")
    world_size: int | None = None
    local_cycles: int | None = None
    if algorithm == "PPO" and spec["ppo_epoch_is_sum_reduced"] and epochs:
        epoch_deltas = [int(right - left) for left, right in zip(epochs, epochs[1:])]
        if epoch_deltas and len(set(epoch_deltas)) == 1 and epoch_deltas[0] > 0:
            world_size = epoch_deltas[0]
        if world_size:
            local_cycles = int(round(epochs[-1] / world_size))
    elif algorithm == "SAC" and epochs:
        runtime_ranks = {
            int(match.group(1))
            for path in spec.get("supporting_paths", [])
            if (match := re.fullmatch(r"rank_(\d+)_runtime", Path(path).stem))
        }
        if runtime_ranks:
            world_size = len(runtime_ranks)
        local_cycles = int(round(epochs[-1]))

    expected_eval_count = None
    if isinstance(configured_interval, int) and configured_interval > 0 and local_cycles is not None:
        # Training evaluates on interval boundaries, then runs one completion
        # evaluation if the final cycle was not already evaluated.
        expected_eval_count = math.ceil(local_cycles / configured_interval)

    local_transitions_per_cycle = None
    transitions_per_cycle_evidence: dict[str, Any] | None = None
    if algorithm == "PPO":
        horizon = train.get("bptt_horizon")
        num_envs = vec.get("num_envs")
        agents_per_env = env.get("num_agents")
        if all(isinstance(value, int) and value > 0 for value in (horizon, num_envs, agents_per_env)):
            local_transitions_per_cycle = horizon * num_envs * agents_per_env
            transitions_per_cycle_evidence = {
                "formula": "train.bptt_horizon * vec.num_envs * env.num_agents",
                "bptt_horizon": horizon,
                "vec_num_envs": num_envs,
                "env_num_agents": agents_per_env,
            }

    step_deltas = [int(right - left) for left, right in zip(steps, steps[1:])]
    expected_global_step_per_cycle = (
        local_transitions_per_cycle * world_size
        if local_transitions_per_cycle is not None and world_size is not None
        else None
    )
    step_deltas_match_cycles = bool(
        steps
        and expected_global_step_per_cycle
        and all(delta == expected_global_step_per_cycle for delta in step_deltas)
        and steps[0] == expected_global_step_per_cycle
    )

    filter_points: list[dict[str, Any]] = []
    kept_values: list[float] = []
    valid_counts_rank0: list[int] = []
    kept_counts_per_rank: list[int] = []
    if algorithm == "PPO" and local_transitions_per_cycle is not None:
        for row in history:
            kept_fraction = row.get("losses/kept_fraction")
            masked_fraction = row.get("losses/masked_fraction")
            if not isinstance(kept_fraction, (int, float)) or not isinstance(masked_fraction, (int, float)):
                continue
            valid_count = round(local_transitions_per_cycle * (1.0 - masked_fraction))
            kept_count = round(valid_count * kept_fraction)
            valid_counts_rank0.append(valid_count)
            kept_counts_per_rank.append(kept_count)
            kept_values.append(float(kept_fraction))
            filter_points.append(
                {
                    "agent_steps_global": row.get("agent_steps"),
                    "wandb_epoch": row.get("epoch"),
                    "local_cycle": (
                        int(round(row["epoch"] / world_size))
                        if world_size and isinstance(row.get("epoch"), (int, float))
                        else None
                    ),
                    "kept_fraction_rank0": kept_fraction,
                    "filtered_fraction_rank0": row.get("losses/filtered_fraction"),
                    "masked_fraction_rank0": masked_fraction,
                    "filter_threshold_rank0": row.get("losses/filter_threshold"),
                    "valid_samples_rank0": valid_count,
                    "kept_samples_per_rank_after_ddp_trim": kept_count,
                }
            )

    weighted_rank0_fraction = (
        sum(kept_counts_per_rank) / sum(valid_counts_rank0)
        if valid_counts_rank0 and sum(valid_counts_rank0)
        else None
    )
    retained_global_samples = (
        sum(kept_counts_per_rank) * world_size
        if kept_counts_per_rank and world_size is not None
        else None
    )
    raw_cli_total = first_config_value(parsed["environment_args"], "train.total_timesteps=")
    final_agent_steps = steps[-1] if steps else None
    performance_train_seconds = parsed["summary"].get("performance/train")
    performance_learn_seconds = parsed["summary"].get("performance/learn")
    update_epochs = train.get("update_epochs")
    learner_exposures: int | None = None
    if algorithm == "PPO" and retained_global_samples is not None and isinstance(update_epochs, int):
        learner_exposures = retained_global_samples * update_epochs
        exposure_source = (
            "Inferred global PPO exposures = all-rank kept samples * train.update_epochs; "
            "the filter history records a full rollout each cycle."
        )
    elif algorithm == "SAC":
        sampled = parsed["summary"].get("replay/sampled_transitions")
        if isinstance(sampled, (int, float)):
            learner_exposures = int(sampled)
            exposure_source = "Direct global replay/sampled_transitions counter from the run summary."
        else:
            exposure_source = "The saved summary has no replay/sampled_transitions counter."
    else:
        exposure_source = "No learner exposure count could be reconstructed from the saved metrics."

    train_per_received_seconds = (
        performance_train_seconds / final_agent_steps
        if isinstance(performance_train_seconds, (int, float)) and final_agent_steps
        else None
    )
    train_per_exposure_seconds = (
        performance_train_seconds / learner_exposures
        if isinstance(performance_train_seconds, (int, float)) and learner_exposures
        else None
    )
    learn_per_received_seconds = (
        performance_learn_seconds / final_agent_steps
        if isinstance(performance_learn_seconds, (int, float)) and final_agent_steps
        else None
    )
    learn_per_exposure_seconds = (
        performance_learn_seconds / learner_exposures
        if isinstance(performance_learn_seconds, (int, float)) and learner_exposures
        else None
    )

    return {
        "label": spec["label"],
        "algorithm": algorithm,
        "run": parsed["run"],
        "exit_codes": parsed["exit_codes"],
        "runtime_seconds_from_wandb_summary": parsed["summary"].get("_runtime"),
        "wandb_file": {
            "path": parsed["path"],
            "size_bytes": parsed["size_bytes"],
            "sha256": parsed["sha256"],
            "record_counts": parsed["record_counts"],
        },
        "world_size": world_size,
        "world_size_evidence": spec["world_size_evidence"],
        "raw_environment_train_total_timesteps": raw_cli_total,
        "configuration": config,
        "final_summary": parsed["summary"],
        "hardware": {
            "wandb_environment_records": parsed["environment_records"],
            "note": (
                "W&B gpu_count is host inventory; selected device IDs are unavailable "
                "unless independently recorded in the run artifacts."
            ),
        },
        "history": {
            "record_count": len(history),
            "first_agent_steps": steps[0] if steps else None,
            "final_agent_steps": steps[-1] if steps else None,
            "first_wandb_epoch": epochs[0] if epochs else None,
            "final_wandb_epoch": epochs[-1] if epochs else None,
            "training_cycles_inferred": local_cycles,
            "agent_step_delta_min": min(step_deltas) if step_deltas else None,
            "agent_step_delta_median": median_or_none([float(value) for value in step_deltas]),
            "agent_step_delta_max": max(step_deltas) if step_deltas else None,
            "agent_steps_match_one_logged_row_per_cycle": step_deltas_match_cycles,
            "eval_metric_prefix": eval_metric_prefix,
            "observed_eval_rows": len(eval_rows),
            "observed_eval_score_rows": len(eval_score_rows),
            "evaluation_interval_epochs": configured_interval,
            "expected_eval_count_from_interval_plus_completion": expected_eval_count,
            "observed_eval_steps": [row.get("agent_steps") for row in eval_score_rows],
            "epoch_interpretation": (
                "W&B epoch is dist_sum(local_epoch); divide by world_size to recover "
                "training cycles. The evaluator condition uses rank-local epoch."
                if algorithm == "PPO" and spec["ppo_epoch_is_sum_reduced"]
                else "Logged epoch advances once per observed training cycle in this run."
            ),
        },
        "filtering": {
            "enabled": train.get("adv_filter_enabled"),
            "history_points": len(filter_points),
            "metric_scope": (
                "Rank-0 metrics. The implementation all-reduces the kept count with MIN, "
                "so the logged post-trim kept count is common across ranks. Rank-1 valid "
                "counts are not logged, so a global valid-weighted fraction cannot be reconstructed."
                if filter_points
                else "No PPO advantage-filter history was logged for this run."
            ),
            "local_transitions_per_rank_per_cycle": local_transitions_per_cycle,
            "transitions_per_cycle_evidence": transitions_per_cycle_evidence,
            "all_history_rows_are_filter_cycles": len(filter_points) == len(history) if algorithm == "PPO" else False,
            "arithmetic_mean_kept_fraction_rank0": statistics.mean(kept_values) if kept_values else None,
            "median_kept_fraction_rank0": median_or_none(kept_values),
            "min_kept_fraction_rank0": min(kept_values) if kept_values else None,
            "max_kept_fraction_rank0": max(kept_values) if kept_values else None,
            "first_kept_fraction_rank0": kept_values[0] if kept_values else None,
            "final_kept_fraction_rank0": kept_values[-1] if kept_values else None,
            "valid_weighted_kept_fraction_rank0": weighted_rank0_fraction,
            "rank0_valid_samples_across_logged_cycles": sum(valid_counts_rank0) if valid_counts_rank0 else None,
            "rank0_kept_samples_across_logged_cycles": sum(kept_counts_per_rank) if kept_counts_per_rank else None,
            "global_kept_samples_across_logged_cycles_from_ddp_min": retained_global_samples,
            "global_valid_samples": None,
            "full_history": filter_points,
        },
        "learner_accounting": {
            "global_received_agent_transitions": final_agent_steps,
            "global_learner_transition_exposures": learner_exposures,
            "learner_exposure_source": exposure_source,
            "exposures_per_received_transition": (
                learner_exposures / final_agent_steps
                if learner_exposures is not None and final_agent_steps
                else None
            ),
            "sac_replay_sampled_transitions_global": parsed["summary"].get("replay/sampled_transitions"),
            "sac_replay_valid_inserts_global": parsed["summary"].get("replay/valid_inserts"),
            "sac_replay_size_final": parsed["summary"].get("replay/size"),
            "sac_learner_updates_rank0": parsed["summary"].get("sac/learner_updates"),
            "performance_train_estimated_seconds_rank0": performance_train_seconds,
            "performance_train_seconds_per_global_received_transition": train_per_received_seconds,
            "performance_train_microseconds_per_global_received_transition": (
                train_per_received_seconds * 1_000_000
                if train_per_received_seconds is not None
                else None
            ),
            "performance_train_seconds_per_global_learner_exposure": train_per_exposure_seconds,
            "performance_train_microseconds_per_global_learner_exposure": (
                train_per_exposure_seconds * 1_000_000
                if train_per_exposure_seconds is not None
                else None
            ),
            "performance_learn_estimated_seconds_rank0": performance_learn_seconds,
            "performance_learn_seconds_per_global_received_transition": learn_per_received_seconds,
            "performance_learn_microseconds_per_global_received_transition": (
                learn_per_received_seconds * 1_000_000
                if learn_per_received_seconds is not None
                else None
            ),
            "performance_learn_seconds_per_global_learner_exposure": learn_per_exposure_seconds,
            "performance_learn_microseconds_per_global_learner_exposure": (
                learn_per_exposure_seconds * 1_000_000
                if learn_per_exposure_seconds is not None
                else None
            ),
            "profiler_estimate_method": (
                "Rank-0 performance/train is the outer train() region and includes its nested "
                "trainer bookkeeping and performance/learn gradient/update subregion. "
                "performance/learn is reported separately as the narrower inner region. The "
                "shared Profile samples every 5th epoch and scales sampled elapsed time by 5; "
                "boundaries synchronize CUDA. These are frequency-scaled profiler estimates, "
                "not independently measured wall-clock learner durations."
            ),
            "profile_source_paths": ["pufferlib/pufferl.py", "pufferlib/pufferl_sac.py"],
        },
        "supporting_paths": spec.get("supporting_paths", []),
    }


def summarize_related_run(reader: ReadOnlyDataStore, relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        return {"path": relative, "available": False}
    parsed = reader.scan(path)
    filter_rows = sum("losses/kept_fraction" in row for row in parsed["history"])
    return {
        "path": parsed["path"],
        "available": True,
        "sha256": parsed["sha256"],
        "history_record_count": len(parsed["history"]),
        "history_steps": [row.get("_step") for row in parsed["history"]],
        "filter_history_points": filter_rows,
        "summary_keys": sorted(parsed["summary"]),
        "interpretation": (
            "Supplemental duplicate W&B run file. It contains no training filter history and "
            "is kept separate from the complete 2026-08-26 history."
        ),
    }


def read_supporting_json(relative: str) -> Any:
    path = ROOT / relative
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as source:
        return safe_json_value(json.load(source))


def enrich_supporting_metadata(run: dict[str, Any], spec: dict[str, Any]) -> None:
    if run["algorithm"] == "SAC":
        runtime_paths = [path for path in spec["supporting_paths"] if Path(path).name.startswith("rank_")]
        rank_data = []
        for relative in runtime_paths:
            runtime = read_supporting_json(relative)
            if runtime is not None:
                rank_data.append({"path": relative, **runtime})
        run["rank_runtime_records"] = rank_data
        launch_path = next((path for path in spec["supporting_paths"] if path.endswith("console.log")), None)
        launch_command = None
        if launch_path and (ROOT / launch_path).is_file():
            for line in (ROOT / launch_path).read_text(encoding="utf-8", errors="replace").splitlines():
                if "torch.distributed.run" in line and "--nproc-per-node" in line:
                    launch_command = line
                    break
        run["recorded_launch_command"] = launch_command
        visible_ids = sorted({entry.get("cuda_visible_id") for entry in rank_data if entry.get("cuda_visible_id")})
        run["hardware"]["allocated_visible_gpu_ids"] = visible_ids or None
    else:
        run["hardware"]["allocated_visible_gpu_ids"] = None
        run["hardware"]["allocated_device_ids_note"] = (
            "The corresponding launcher supports an environment override; the local W&B "
            "record does not preserve the actual CUDA_VISIBLE_DEVICES value."
        )


def markdown_report(document: dict[str, Any]) -> str:
    runs = document["runs"]
    lines = [
        "# Historical PPO/SAC run evidence",
        "",
        "Generated from local W&B binary histories and saved run metadata. The analyzer used the installed W&B SDK datastore parser in read-only mode; no W&B API, network request, or training job was started.",
        "",
        "## Run summary",
        "",
        "| Run | Global received agent steps | Logged rows / cycles | Kept fraction: cycle mean / final | Filtering samples | Eval rows: observed / expected | Hardware recorded |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        filtering = run["filtering"]
        history = run["history"]
        env_records = run["hardware"]["wandb_environment_records"]
        gpu_types = sorted({entry["gpu_type"] for entry in env_records if entry.get("gpu_type")})
        hw = ", ".join(gpu_types) if gpu_types else "not recorded"
        if filtering["history_points"]:
            mean = filtering["arithmetic_mean_kept_fraction_rank0"]
            final = filtering["final_kept_fraction_rank0"]
            kept = filtering["global_kept_samples_across_logged_cycles_from_ddp_min"]
            kept_text = f"{mean:.2%} / {final:.2%}"
            filter_samples = f"{kept:,} all-rank kept"
        else:
            kept_text = "not logged"
            filter_samples = "not applicable"
        lines.append(
            "| {label} | {steps:,} | {rows:,} / {cycles:,} | {kept} | {samples} | {observed} / {expected} | {hw} |".format(
                label=run["label"],
                steps=history["final_agent_steps"] or 0,
                rows=history["record_count"],
                cycles=history["training_cycles_inferred"] or 0,
                kept=kept_text,
                samples=filter_samples,
                observed=history["observed_eval_score_rows"],
                expected=history["expected_eval_count_from_interval_plus_completion"],
                hw=hw,
            )
        )

    lines.extend(["", "## Filtering history and sample accounting", ""])
    for run in runs:
        filtering = run["filtering"]
        if not filtering["history_points"]:
            accounting = run["learner_accounting"]
            if run["algorithm"] == "SAC" and accounting["sac_replay_sampled_transitions_global"] is not None:
                lines.append(
                    f"- **{run['label']}:** PPO advantage filtering does not apply, so no kept fraction is logged. SAC directly records {accounting['sac_replay_sampled_transitions_global']:,} global replay-sampled transition exposures, {accounting['sac_replay_valid_inserts_global']:,} global valid inserts, a final replay size of {accounting['sac_replay_size_final']:,}, and {accounting['sac_learner_updates_rank0']:,} learner updates per rank."
                )
            else:
                lines.append(f"- **{run['label']}:** the saved history has no advantage-filter metrics and no direct replay exposure counter.")
            continue
        lines.append(
            f"- **{run['label']}:** {filtering['history_points']} of {run['history']['record_count']} training rows include `losses/kept_fraction`. The arithmetic mean across cycles is {filtering['arithmetic_mean_kept_fraction_rank0']:.4%}, the rank-0 valid-sample-weighted mean is {filtering['valid_weighted_kept_fraction_rank0']:.4%}, and the final-cycle value is {filtering['final_kept_fraction_rank0']:.4%}. The full curve and per-cycle estimates are in `historical_summary.json`."
        )
        lines.append(
            f"  Reconstructing sample counts from the logged `masked_fraction` and `kept_fraction` gives {filtering['rank0_valid_samples_across_logged_cycles']:,} valid and {filtering['rank0_kept_samples_across_logged_cycles']:,} kept transitions on rank 0; the DDP minimum-count trim means the all-rank kept count is {filtering['global_kept_samples_across_logged_cycles_from_ddp_min']:,}. The global valid denominator is unavailable because rank-1 masked counts were not logged."
        )

    lines.extend(
        [
            "",
            "## Epoch counters and evaluation counts",
            "",
            "PPO logs `epoch` as `dist_sum(self.epoch)` in `pufferlib/pufferl.py:866`, while one call to `PuffeRL.train()` increments the local epoch once. Both PPO runs have two-rank epoch deltas, so divide the displayed epoch by two to recover cycles. Evaluation scheduling checks each rank-local epoch. The completion evaluator runs once more if the final cycle is off an interval boundary.",
            "",
        ]
    )
    for run in runs:
        history = run["history"]
        lines.append(
            f"- **{run['label']}:** {history['training_cycles_inferred']:,} cycles, interval {history['evaluation_interval_epochs']}, expected {history['expected_eval_count_from_interval_plus_completion']} evaluations; W&B records {history['observed_eval_score_rows']}."
        )

    lines.extend(
        [
            "",
            "## Learner exposure and profiled time",
            "",
            "`performance/train` is the primary total training-phase estimate. It measures the outer `train()` region; `performance/learn` is a narrower nested gradient/update region and is shown separately below.",
            "",
            "| Run | Global received transitions | Global learner transition exposures | Exposures / received | `performance/train` estimate (rank 0) | Estimated µs / received | Estimated µs / exposure |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        accounting = run["learner_accounting"]
        exposure_count = accounting["global_learner_transition_exposures"]
        exposure_text = f"{exposure_count:,}" if exposure_count is not None else "not available"
        per_received = accounting["performance_train_microseconds_per_global_received_transition"]
        per_exposure = accounting["performance_train_microseconds_per_global_learner_exposure"]
        lines.append(
            "| {label} | {received:,} | {exposures} | {ratio:.3f} | {seconds:,.1f} s | {per_received:.3f} | {per_exposure:.3f} |".format(
                label=run["label"],
                received=accounting["global_received_agent_transitions"] or 0,
                exposures=exposure_text,
                ratio=accounting["exposures_per_received_transition"] or 0,
                seconds=accounting["performance_train_estimated_seconds_rank0"] or 0,
                per_received=per_received or 0,
                per_exposure=per_exposure or 0,
            )
        )

    lines.extend(
        [
            "",
            "| Run | `performance/learn` estimate (rank 0) | Estimated µs / received | Estimated µs / exposure |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in runs:
        accounting = run["learner_accounting"]
        lines.append(
            "| {label} | {seconds:,.1f} s | {per_received:.3f} | {per_exposure:.3f} |".format(
                label=run["label"],
                seconds=accounting["performance_learn_estimated_seconds_rank0"] or 0,
                per_received=(
                    accounting["performance_learn_microseconds_per_global_received_transition"] or 0
                ),
                per_exposure=(
                    accounting["performance_learn_microseconds_per_global_learner_exposure"] or 0
                ),
            )
        )

    sac_run = next((run for run in runs if run["algorithm"] == "SAC"), None)
    if sac_run:
        sac_exposures = sac_run["learner_accounting"]["global_learner_transition_exposures"]
        sac_train_seconds = sac_run["learner_accounting"]["performance_train_estimated_seconds_rank0"]
        sac_cost_per_exposure = sac_run["learner_accounting"][
            "performance_train_microseconds_per_global_learner_exposure"
        ]
        for ppo_run in (run for run in runs if run["algorithm"] == "PPO"):
            ppo_exposures = ppo_run["learner_accounting"]["global_learner_transition_exposures"]
            if sac_exposures and ppo_exposures:
                multiple = sac_exposures / ppo_exposures
                lines.append(
                    f"- SAC replay exposure is {multiple:.2f}x the inferred exposure of {ppo_run['label']} (SAC {sac_exposures:,}; PPO {ppo_exposures:,})."
                )
            ppo_train_seconds = ppo_run["learner_accounting"][
                "performance_train_estimated_seconds_rank0"
            ]
            ppo_cost_per_exposure = ppo_run["learner_accounting"][
                "performance_train_microseconds_per_global_learner_exposure"
            ]
            if sac_train_seconds and ppo_train_seconds and ppo_cost_per_exposure:
                train_ratio = sac_train_seconds / ppo_train_seconds
                per_exposure_ratio = sac_cost_per_exposure / ppo_cost_per_exposure
                lines.append(
                    f"- SAC has {train_ratio:.2f}x the cumulative `performance/train` estimate of {ppo_run['label']}; divided by exposure, that is {per_exposure_ratio:.2f}x. Hardware and workloads differ, so this is descriptive only."
                )
    lines.extend(
        [
            "",
            "Both profile values are cumulative rank-0 estimates from each saved run summary. `performance/train` encloses the full trainer `train()` call, including the nested train-misc bookkeeping; `performance/learn` is the inner gradient/update region. PPO exposure is inferred as the globally retained sample count times `train.update_epochs`; SAC exposure comes from the direct global replay counter. The per-row ratios normalize the outer `performance/train` estimate by global received rows and global learner exposures.",
            "",
            "These normalized times remain descriptive estimates. The profiler samples every fifth cycle, multiplies each sampled interval by five, and synchronizes CUDA at profiling boundaries. PPO used NVIDIA RTX A6000 hardware while SAC used RTX A5000 hardware, and their trainer paths differ. The ratios are not same-hardware speed measurements or controlled estimates of an algorithm-only effect.",
            "",
            "## Provenance and limits",
            "",
            "- Each primary `.wandb` file is identified by repository-relative path, byte size, and SHA-256 in the JSON. Config and final summary values are copied from its local W&B protobuf records.",
            "- The 2026-08-26 PPO run has a second local W&B file dated 2026-09-11 with the same run ID. It has one supplemental history record and no filter metric; it is recorded separately and not merged into the complete 611-cycle history.",
            "- PPO filter history is rank-0 telemetry. The logged kept count is trimmed to the DDP global minimum and therefore is common across ranks, but total valid samples across ranks cannot be recovered from rank-0-only masked metrics. Fractions are stored as floating-point values; counts are reconstructed and rounded to the nearest transition.",
            "- W&B hardware metadata reports host GPU inventory (eight A6000s for PPO, eight A5000s for SAC), not allocated GPU count. The SAC trial has separate rank records showing visible IDs 2 and 3; physical IDs for the PPO runs were not recorded in the local W&B files.",
            "- `performance/*` values describe sampled trainer profiler regions. They do not isolate benchmark evaluation work in `run_training_evaluation`; evaluation counts are reported separately and no evaluation duration is inferred here.",
            "- PPO and SAC used different GPU models and different training configurations. These artifacts establish workload and evaluation counts, not a same-hardware speed comparison or a direct wall-time estimate.",
            "- W&B `summary` values are end-of-run values. The cycle mean and per-cycle counts above use the full stored history, so they are distinct from the final summary fraction.",
            "",
            "## Reproduction",
            "",
            "From the repository root, with the project virtual environment active:",
            "",
            "```bash",
            "source .venv/bin/activate",
            "python -B project/PPO_SAC_Profiling/analyze_history.py",
            "```",
            "",
            "The script writes `historical_summary.json` and `historical_summary.md` next to itself. `-B` suppresses bytecode artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output folder (default: this script's directory).",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as error:
        raise SystemExit("The active .venv must provide wandb; activate it before running this script.") from error

    reader = ReadOnlyDataStore()
    runs = []
    for spec in RUN_SPECS:
        source = ROOT / spec["wandb_path"]
        if not source.is_file():
            raise SystemExit(f"Missing local W&B history file: {relative_path(source)}")
        parsed = reader.scan(source)
        run = summarize_history(parsed, spec)
        enrich_supporting_metadata(run, spec)
        related = [summarize_related_run(reader, path) for path in spec.get("related_wandb_paths", [])]
        if related:
            run["related_wandb_files"] = related
        runs.append(run)

    sac_exposures = next(
        (
            run["learner_accounting"]["global_learner_transition_exposures"]
            for run in runs
            if run["algorithm"] == "SAC"
        ),
        None,
    )
    for run in runs:
        exposures = run["learner_accounting"]["global_learner_transition_exposures"]
        run["learner_accounting"]["sac_exposure_multiple"] = (
            sac_exposures / exposures
            if sac_exposures is not None and exposures and run["algorithm"] == "PPO"
            else (1.0 if run["algorithm"] == "SAC" and exposures is not None else None)
        )

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    document = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "repository_root": str(ROOT),
        "analyzer": relative_path(Path(__file__)),
        "python_version": sys.version.split()[0],
        "wandb_version": wandb.__version__,
        "commands": [
            "source .venv/bin/activate",
            "python -B project/PPO_SAC_Profiling/analyze_history.py",
        ],
        "read_mode": "W&B SDK datastore scanner with a read-only rb file handle; no W&B API or network calls.",
        "source_code_references": {
            "ppo_epoch_sum_reduction": "pufferlib/pufferl.py:866",
            "ppo_epoch_increment": "pufferlib/pufferl.py:527",
            "ppo_filter_and_rank_min": "pufferlib/pufferl.py:749-778",
            "ppo_evaluation_schedule": "pufferlib/pufferl.py:1628-1637, 1668-1677",
            "frequency_scaled_profiler": "pufferlib/pufferl.py:1153-1194",
            "sac_replay_exposure_counter": "pufferlib/pufferl_sac.py:1975-1976, 2050-2057",
        },
        "runs": runs,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "historical_summary.json"
    markdown_path = output_dir / "historical_summary.md"
    json_path.write_text(json.dumps(safe_json_value(document), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(document), encoding="utf-8")
    print(f"Wrote {relative_path(json_path)}")
    print(f"Wrote {relative_path(markdown_path)}")
    for run in runs:
        history = run["history"]
        filtering = run["filtering"]
        print(
            f"{run['label']}: steps={history['final_agent_steps']}, "
            f"cycles={history['training_cycles_inferred']}, "
            f"evals={history['observed_eval_score_rows']}/{history['expected_eval_count_from_interval_plus_completion']}, "
            f"kept_mean={filtering['arithmetic_mean_kept_fraction_rank0']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
