#!/usr/bin/env python3
"""Launch the configured Drive SAC grid as isolated GPU jobs."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import psutil
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_DIR / "configs" / "round1"
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUN_CONFIG_PATH = CONFIG_DIR / "SAC_run_config.yaml"
FIXED_CONFIG_PATH = CONFIG_DIR / "SAC_fixed_config.yaml"
TUNABLE_CONFIG_PATH = CONFIG_DIR / "SAC_tunable_config.yaml"
SPECIFIC_CONFIG_PATH = CONFIG_DIR / "SAC_specific_config.yaml"
CHILD_MODULE = "project.soft_actor_critic_tuning.scripts.run_trial"
METRICS_FILENAME = "metrics.jsonl"
SUMMARY_FILENAME = "summary.json"
RUNTIME_FILENAME_TEMPLATE = "rank_{rank}_runtime.json"
REPLAY_MEMORY_HEADROOM = 1.10
TERMINATION_WAIT_SECONDS = 15
KILL_WAIT_SECONDS = 10
POLL_INTERVAL_SECONDS = 0.25


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path):
    with path.open(encoding="utf-8") as source:
        data = yaml.safe_load(source)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _load_configs(config_dir=CONFIG_DIR):
    config_dir = Path(config_dir).expanduser().resolve()
    fixed_config_path = config_dir / FIXED_CONFIG_PATH.name
    tunable_config_path = config_dir / TUNABLE_CONFIG_PATH.name
    specific_config_path = config_dir / SPECIFIC_CONFIG_PATH.name
    run_config_path = config_dir / RUN_CONFIG_PATH.name
    fixed = _load_yaml(fixed_config_path)
    tunable = _load_yaml(tunable_config_path)
    specific = _load_yaml(specific_config_path)
    run_profile = _load_yaml(run_config_path).get("launcher")
    if not isinstance(run_profile, dict):
        raise ValueError(f"{run_config_path} must define a launcher mapping")

    resolved = _deep_merge(_deep_merge(tunable, specific), fixed)
    return fixed, resolved, run_profile


def _grid_axes(config):
    axes = []
    for section, section_config in config.get("controlled_exp", {}).items():
        if not isinstance(section_config, dict):
            continue
        for name, parameter_config in section_config.items():
            if isinstance(parameter_config, dict) and "values" in parameter_config:
                values = parameter_config["values"]
                if not isinstance(values, list) or not values:
                    raise ValueError(f"controlled_exp.{section}.{name}.values must be a non-empty list")
                axes.append((f"{section}.{name}", values))
    if not axes:
        raise ValueError("No value lists found under controlled_exp")
    return axes


def _set_dotted(config, path, value):
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Grid path {path!r} does not point into a config mapping")
        target = target[part]
    target[parts[-1]] = deepcopy(value)


def _parse_trial_ids(raw, trial_count, smoke):
    configured_ids = isinstance(raw, list)
    if raw is None:
        ids = [1] if smoke else list(range(1, trial_count + 1))
    elif configured_ids:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
            raise ValueError("launcher.trial_ids must contain only positive integers")
        ids = list(raw)
    else:
        if not isinstance(raw, str):
            raise ValueError("--trial-ids must be comma-separated 1-based integers")
        try:
            ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError("--trial-ids must be comma-separated 1-based integers") from error
        if not ids:
            raise ValueError("--trial-ids must include at least one 1-based integer")
    if len(ids) != len(set(ids)):
        raise ValueError("--trial-ids contains a duplicate trial ID")
    if not ids:
        if configured_ids:
            raise ValueError("launcher.trial_ids must include at least one positive integer")
        raise ValueError("--trial-ids must include at least one 1-based integer")
    invalid = [trial_id for trial_id in ids if trial_id < 1 or trial_id > trial_count]
    if invalid:
        raise ValueError(f"Trial IDs must be in [1, {trial_count}], got {invalid}")
    if configured_ids and smoke:
        return ids[:1]
    return ids


def _parse_gpu_pairs(raw):
    pairs = []
    try:
        for item in raw.split(";"):
            values = item.split(",")
            if len(values) not in (1, 2) or any(not value.strip() for value in values):
                raise ValueError
            pair = tuple(int(value.strip()) for value in values)
            if min(pair) < 0 or len(set(pair)) != len(pair):
                raise ValueError
            pairs.append(pair)
    except ValueError as error:
        raise ValueError(
            "--gpu-groups must look like '0;1;2,3' with one or two distinct nonnegative IDs per group"
        ) from error
    if not pairs:
        raise ValueError("At least one GPU group is required")
    flattened = [gpu for pair in pairs for gpu in pair]
    if len(flattened) != len(set(flattened)):
        raise ValueError("GPU IDs cannot appear in more than one configured group")
    return pairs


def _validate_launcher(profile, gpu_pairs, max_concurrent, trial_hours):
    if trial_hours is not None and (
        isinstance(trial_hours, bool)
        or not isinstance(trial_hours, numbers.Real)
        or not math.isfinite(float(trial_hours))
        or trial_hours <= 0
        or not math.isfinite(float(trial_hours) * 3600)
    ):
        raise ValueError("launcher.trial_hours / --trial-hours must be null or finite and positive")
    grace_seconds = profile.get("grace_seconds")
    if not isinstance(grace_seconds, int) or isinstance(grace_seconds, bool) or grace_seconds < 0:
        raise ValueError("launcher.grace_seconds must be a nonnegative integer")
    if not isinstance(max_concurrent, int) or isinstance(max_concurrent, bool) or max_concurrent < 1:
        raise ValueError("--max-concurrent must be a positive integer")
    if max_concurrent > len(gpu_pairs):
        raise ValueError("--max-concurrent cannot exceed the number of configured GPU groups")
    return grace_seconds


def _smoke_config(config):
    smoke = deepcopy(config)
    smoke["env"].update(
        num_agents=3200,
        scenario_length=16,
        resample_frequency=64,
    )
    smoke["vec"].update(
        num_envs=2,
        num_workers=2,
        batch_size=1,
    )
    smoke["train"].update(
        batch_size=128000,
        minibatch_size=128000,
    )
    return smoke


def _disable_loggers(config):
    config["wandb"] = False
    config["neptune"] = False
    config["tb"] = False
    config["no_model_upload"] = True


def _write_json(path, data):
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    temporary_path.replace(path)


def _write_yaml(path, data):
    with path.open("x", encoding="utf-8") as output:
        yaml.safe_dump(data, output, sort_keys=False, allow_unicode=True)


def _make_run_dir(output_dir, trial_count=16):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"sac{trial_count}_grid_{timestamp}"


def _estimate_observation_size(environment_config):
    # Match Drive.__init__ and compute_observation_size() without constructing an
    # environment, vector worker, logger or CUDA context.
    from pufferlib.ocean.drive import binding

    lane_dropout = min(max(float(environment_config["obs_dropout_lane"]), 0.0), 1.0)
    boundary_dropout = min(max(float(environment_config["obs_dropout_boundary"]), 0.0), 1.0)
    lane_count = int(environment_config["obs_slots_lane_n"] * (1.0 - lane_dropout))
    boundary_count = int(environment_config["obs_slots_boundary_n"] * (1.0 - boundary_dropout))
    return (
        binding.EGO_FEATURES
        + binding.PARTNER_FEATURES * environment_config["obs_slots_partners_n"]
        + binding.LANE_FEATURES * lane_count
        + binding.BOUNDARY_FEATURES * boundary_count
        + binding.TRAFFIC_CONTROL_FEATURES * environment_config["obs_slots_traffic_controls_n"]
        + binding.OBS_VALID_COUNT_FEATURES
        + int(bool(environment_config["reward_conditioning"])) * binding.NUM_REWARD_COEFS
        + environment_config["num_goals"] * binding.GOAL_FEATURES
    )


def _available_memory_bytes():
    return psutil.virtual_memory().available


def _memory_preflight(config, gpu_groups, max_concurrent_trials):
    replay_capacity = config["train"]["batch_size"]
    if not isinstance(replay_capacity, int) or isinstance(replay_capacity, bool) or replay_capacity <= 0:
        raise ValueError("train.batch_size must be a positive integer for replay memory preflight")
    active_groups = gpu_groups[:max_concurrent_trials]
    if len(active_groups) != max_concurrent_trials:
        raise ValueError("Replay RAM preflight requires one configured GPU group per concurrent trial")
    ranks_per_trial = [len(group) for group in active_groups]
    total_ranks = sum(ranks_per_trial)
    observation_size = _estimate_observation_size(config["env"])
    # TensorReplay allocates two float32 observations plus int64 action, float32
    # reward and float32 discount for each transition on every rank.
    bytes_per_transition = observation_size * 8 + 16
    bytes_per_rank = replay_capacity * bytes_per_transition
    required_bytes = math.ceil(
        bytes_per_rank * total_ranks * REPLAY_MEMORY_HEADROOM
    )
    available_bytes = _available_memory_bytes()
    result = {
        "observation_size": observation_size,
        "replay_capacity_per_rank": replay_capacity,
        "replay_bytes_per_rank": bytes_per_rank,
        "concurrent_trials": len(active_groups),
        "total_ranks": total_ranks,
        "rank_counts_by_trial": ranks_per_trial,
        "ranks_per_trial": ranks_per_trial[0] if len(set(ranks_per_trial)) == 1 else None,
        "headroom_factor": REPLAY_MEMORY_HEADROOM,
        "required_host_memory_bytes": required_bytes,
        "available_host_memory_bytes": available_bytes,
        "fits": required_bytes <= available_bytes,
    }
    if not result["fits"]:
        raise MemoryError(
            "Replay RAM preflight failed: "
            f"{max_concurrent_trials} concurrent trial(s) require about "
            f"{required_bytes / 1024**3:.1f} GiB including headroom, but only "
            f"{available_bytes / 1024**3:.1f} GiB is available. "
            f"Replay remains {replay_capacity:,} transitions per rank; lower "
            "--max-concurrent or free host RAM."
        )
    return result


def _build_command(config_path, trial_dir, gpu_pair, trial_hours):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={len(gpu_pair)}",
        "-m",
        CHILD_MODULE,
        "--config",
        str(config_path),
        "--artifact-dir",
        str(trial_dir),
    ]
    if trial_hours is not None:
        command.extend(("--trial-hours", str(trial_hours)))
    return command


def _trial_command_for_log(command, gpu_pair, disable_loggers):
    environment = [
        "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpu_pair)),
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ]
    if disable_loggers:
        environment.append("WANDB_MODE=disabled")
    return shlex.join([*environment, *command])


def _atomic_update_manifest(manifest_path, manifest):
    public_manifest = deepcopy(manifest)
    public_manifest["trials"] = [
        {key: value for key, value in trial.items() if not key.startswith("_")}
        for trial in manifest["trials"]
    ]
    _write_json(manifest_path, public_manifest)


def _track_owned_descendants(process, owned_descendants):
    if process.poll() is not None:
        return
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for descendant in descendants:
        try:
            identity = (descendant.pid, descendant.create_time())
        except psutil.Error:
            continue
        owned_descendants[identity] = descendant


def _terminate_owned_group(process, owned_descendants=None):
    if owned_descendants is None:
        owned_descendants = {}
    _track_owned_descendants(process, owned_descendants)

    def collect_descendants():
        return list(owned_descendants.values())

    def signal_owned(signum):
        for descendant in reversed(collect_descendants()):
            try:
                if descendant.is_running():
                    descendant.send_signal(signum)
            except psutil.Error:
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    signal_owned(signal.SIGTERM)
    if process.poll() is None:
        try:
            exit_code = process.wait(timeout=TERMINATION_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            exit_code = None
    else:
        exit_code = process.poll()
    _, alive_descendants = psutil.wait_procs(collect_descendants(), timeout=0)
    if exit_code is None or alive_descendants:
        for descendant in alive_descendants:
            try:
                if descendant.is_running():
                    descendant.kill()
            except psutil.Error:
                pass
        signal_owned(signal.SIGKILL)
        psutil.wait_procs(alive_descendants, timeout=KILL_WAIT_SECONDS)
    if exit_code is None:
        try:
            return process.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            return process.poll()
    return exit_code


def _load_json_if_present(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _last_metric_record(path):
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = json.loads(line)
    return last


def _finish_trial(trial, exit_code, hard_timeout, elapsed_seconds):
    trial_dir = Path(trial["trial_dir"])
    config = trial["_resolved_config"]
    artifact_dir = trial_dir / "learner"
    metrics = _last_metric_record(trial_dir / METRICS_FILENAME)
    stop_record = _load_json_if_present(trial_dir / "stop_reason.json") or {}
    runtime_records = []
    missing = []
    for rank in range(len(trial["gpu_pair"])):
        runtime_path = trial_dir / RUNTIME_FILENAME_TEMPLATE.format(rank=rank)
        runtime_record = _load_json_if_present(runtime_path)
        if runtime_record is None:
            missing.append(runtime_path.name)
        else:
            runtime_records.append(runtime_record)

    final_model = artifact_dir / config["train"]["final_model_name"]
    trainer_state = artifact_dir / "trainer_state.pt"
    if metrics is None:
        missing.append("metrics.jsonl (no logged metrics)")
    elif not metrics.get("sac/learner_updates"):
        missing.append("learner updates (no SAC updates logged)")
    for artifact_path in (final_model, trainer_state):
        if not artifact_path.is_file():
            missing.append(str(artifact_path.relative_to(trial_dir)))

    training_evaluation = None
    if config["train"].get("evaluation_interval_epochs") is not None:
        selected_benchmarks = config["train"].get("evaluation_benchmarks")
        if isinstance(selected_benchmarks, str):
            selected_benchmarks = list(dict.fromkeys(
                name.strip() for name in selected_benchmarks.split(",") if name.strip()
            ))
        else:
            selected_benchmarks = []

        final_epoch = None if metrics is None else metrics.get("epoch")
        final_agent_steps = None if metrics is None else metrics.get("agent_steps")
        training_evaluation = {
            "epoch": final_epoch,
            "agent_steps": final_agent_steps,
            "benchmarks": {},
        }
        if not selected_benchmarks:
            missing.append("train.evaluation_benchmarks (no selected benchmarks)")
        elif (
            isinstance(final_epoch, bool)
            or not isinstance(final_epoch, int)
            or final_epoch < 0
            or isinstance(final_agent_steps, bool)
            or not isinstance(final_agent_steps, int)
            or final_agent_steps < 0
        ):
            missing.append("training evaluation final epoch (missing from metrics.jsonl)")
        else:
            for benchmark_name in selected_benchmarks:
                summary_path = (
                    artifact_dir
                    / "eval"
                    / "training"
                    / benchmark_name
                    / f"epoch_{final_epoch:06d}_step_{final_agent_steps}"
                    / "evaluation_summary.json"
                )
                benchmark_record = {"summary_path": str(summary_path.resolve())}
                training_evaluation["benchmarks"][benchmark_name] = benchmark_record
                try:
                    benchmark_summary = _load_json_if_present(summary_path)
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    missing.append(
                        f"{summary_path.relative_to(trial_dir)} (invalid summary JSON: {error})"
                    )
                    continue
                if benchmark_summary is None:
                    missing.append(f"{summary_path.relative_to(trial_dir)} (missing summary)")
                    continue
                if not isinstance(benchmark_summary, dict):
                    missing.append(
                        f"{summary_path.relative_to(trial_dir)} (summary must be a JSON object)"
                    )
                    continue

                num_scenarios = benchmark_summary.get("num_scenarios")
                num_episodes = benchmark_summary.get("num_episodes")
                metrics_mean = benchmark_summary.get("metrics_mean")
                finite_metrics = isinstance(metrics_mean, dict) and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (isinstance(value, int) or math.isfinite(value))
                    for value in metrics_mean.values()
                )
                if (
                    isinstance(num_scenarios, bool)
                    or not isinstance(num_scenarios, int)
                    or num_scenarios <= 0
                    or isinstance(num_episodes, bool)
                    or not isinstance(num_episodes, int)
                    or num_episodes != num_scenarios
                    or not metrics_mean
                    or not finite_metrics
                ):
                    missing.append(f"{summary_path.relative_to(trial_dir)} (incomplete benchmark summary)")
                    continue

                benchmark_record.update({
                    "num_scenarios": num_scenarios,
                    "num_episodes": num_episodes,
                    "metrics_mean": metrics_mean,
                })

    stop_reason = stop_record.get("stop_reason")
    if hard_timeout:
        status = "failed_timeout"
        stop_reason = stop_reason or "parent_hard_timeout"
    elif stop_reason == "nonfinite_learner_loss":
        status = "failed_nonfinite_loss"
    elif exit_code != 0:
        status = "failed_nonzero_exit"
        stop_reason = stop_reason or "subprocess_nonzero_exit"
    elif missing:
        status = "failed_missing_artifacts"
        stop_reason = stop_reason or "required_artifact_missing"
    else:
        status = "completed"
        stop_reason = stop_reason or "training_returned"

    summary = {
        "trial_id": trial["trial_id"],
        "parameters": trial["parameters"],
        "status": status,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "hard_timeout": hard_timeout,
        "full_time_seconds": elapsed_seconds,
        "samples": None if metrics is None else metrics.get("agent_steps"),
        "agent_steps": None if metrics is None else metrics.get("agent_steps"),
        "learner_updates": None if metrics is None else metrics.get("sac/learner_updates"),
        "sampled_transitions": None if metrics is None else metrics.get("replay/sampled_transitions"),
        "epoch": None if metrics is None else metrics.get("epoch"),
        "gpu_ranks": runtime_records,
        "missing_artifacts": missing,
        "artifacts": {
            "metrics_jsonl": str((trial_dir / METRICS_FILENAME).resolve()),
            "final_model": str(final_model.resolve()),
            "trainer_state": str(trainer_state.resolve()),
        },
    }
    if training_evaluation is not None:
        summary["training_evaluation"] = training_evaluation
    _write_json(trial_dir / SUMMARY_FILENAME, summary)
    trial["status"] = status
    trial["exit_code"] = exit_code
    trial["stop_reason"] = stop_reason
    trial["summary_path"] = str((trial_dir / SUMMARY_FILENAME).resolve())
    trial["missing_artifacts"] = missing
    return summary


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write resolved configs and commands without starting torchrun")
    parser.add_argument("--output-dir", help="Unique output directory; existing directories are refused")
    parser.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Directory containing the four SAC launch YAML files",
    )
    parser.add_argument(
        "--gpu-groups", "--gpu-pairs", dest="gpu_pairs",
        help="GPU groups such as '0;1;2,3'; --gpu-pairs remains an alias",
    )
    parser.add_argument("--max-concurrent", type=int, help="Maximum concurrent trials")
    parser.add_argument("--trial-hours", type=float, help="Soft deadline from each trial start")
    parser.add_argument("--trial-ids", help="Comma-separated 1-based trial IDs from the deterministic grid")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B, Neptune, TensorBoard and model upload")
    parser.add_argument("--smoke", action="store_true", help="Run trial 1 by default with reduced replay/vector budgets and no logging")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config_dir = Path(args.config_dir).expanduser()
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    config_dir = config_dir.resolve()
    _, base_config, launcher = _load_configs(config_dir)
    axes = _grid_axes(base_config)
    combinations = list(itertools.product(*(values for _, values in axes)))
    trial_count = len(combinations)
    if args.trial_ids is not None:
        trial_ids = args.trial_ids
    elif "trial_ids" in launcher:
        trial_ids = launcher["trial_ids"]
        if not isinstance(trial_ids, list):
            raise ValueError("launcher.trial_ids must be a list of positive integers")
    else:
        trial_ids = None
    selected_ids = _parse_trial_ids(trial_ids, trial_count, args.smoke)
    trial_evaluation_intervals = launcher.get("trial_evaluation_interval_epochs", {})
    if not isinstance(trial_evaluation_intervals, dict):
        raise ValueError("launcher.trial_evaluation_interval_epochs must be a trial-ID mapping")
    for trial_id, interval in trial_evaluation_intervals.items():
        if (
            isinstance(trial_id, bool) or not isinstance(trial_id, int)
            or not 1 <= trial_id <= trial_count
            or isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
        ):
            raise ValueError(
                "launcher.trial_evaluation_interval_epochs requires valid integer trial IDs "
                "and positive integer intervals"
            )

    configured_gpu_pairs = ";".join(",".join(map(str, pair)) for pair in launcher["gpu_pairs"])
    gpu_pairs = _parse_gpu_pairs(configured_gpu_pairs if args.gpu_pairs is None else args.gpu_pairs)
    max_concurrent = args.max_concurrent if args.max_concurrent is not None else min(launcher["max_concurrent"], len(gpu_pairs))
    if args.trial_hours is not None:
        trial_hours = args.trial_hours
    elif args.smoke:
        smoke_trial_seconds = launcher["smoke_trial_seconds"]
        if (
            isinstance(smoke_trial_seconds, bool)
            or not isinstance(smoke_trial_seconds, numbers.Real)
            or not math.isfinite(float(smoke_trial_seconds))
            or smoke_trial_seconds <= 0
        ):
            raise ValueError("launcher.smoke_trial_seconds must be finite and positive")
        trial_hours = smoke_trial_seconds / 3600
    else:
        trial_hours = launcher["trial_hours"]
    grace_seconds = _validate_launcher(launcher, gpu_pairs, max_concurrent, trial_hours)
    if args.smoke:
        grace_seconds = launcher["smoke_grace_seconds"]

    default_output_root = PROJECT_DIR / "runs"
    if config_dir != CONFIG_DIR.resolve():
        default_output_root /= config_dir.name
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else _make_run_dir(default_output_root, trial_count)
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir()

    selected = set(selected_ids)
    trials = []
    for trial_id, combination in enumerate(combinations, 1):
        parameters = {path: value for (path, _), value in zip(axes, combination)}
        trial_record = {
            "trial_id": trial_id,
            "parameters": parameters,
            "status": "planned" if trial_id in selected else "skipped",
            "gpu_pair": None,
            "config_path": None,
            "trial_dir": None,
            "command": None,
        }
        if trial_id in selected:
            trial_dir = output_dir / f"trial_{trial_id:02d}"
            trial_dir.mkdir()
            resolved = deepcopy(base_config)
            for path, value in parameters.items():
                _set_dotted(resolved, path, value)
            if trial_id in trial_evaluation_intervals:
                resolved["train"]["evaluation_interval_epochs"] = trial_evaluation_intervals[trial_id]
            if args.smoke:
                resolved = _smoke_config(resolved)
                _disable_loggers(resolved)
            elif args.no_wandb:
                _disable_loggers(resolved)
            run_name = f"{output_dir.name}_trial_{trial_id:02d}"
            resolved["run_name"] = run_name
            resolved["train"]["fresh_run"] = True
            resolved["train"]["data_dir"] = str(trial_dir / "learner")
            config_path = configs_dir / f"trial_{trial_id:02d}.yaml"
            _write_yaml(config_path, resolved)
            gpu_pair = gpu_pairs[(len([record for record in trials if record["trial_id"] in selected]) % min(max_concurrent, len(gpu_pairs)))]
            command = _build_command(config_path, trial_dir, gpu_pair, trial_hours)
            trial_record.update(
                gpu_pair=list(gpu_pair),
                config_path=str(config_path.resolve()),
                trial_dir=str(trial_dir.resolve()),
                _resolved_config=resolved,
                evaluation_interval_epochs=resolved["train"].get("evaluation_interval_epochs"),
                command=command,
                log_command=_trial_command_for_log(command, gpu_pair, args.no_wandb or args.smoke),
            )
        trials.append(trial_record)

    manifest = {
        "experiment": f"SAC{trial_count}-grid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config_dir": str(config_dir),
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "selected_trial_ids": selected_ids,
        "grid_axes": [{"path": path, "values": values} for path, values in axes],
        "launcher": {
            "trial_hours": trial_hours,
            "grace_seconds": grace_seconds,
            "gpu_pairs": [list(pair) for pair in gpu_pairs],
            "max_concurrent": max_concurrent,
            "no_wandb": args.no_wandb or args.smoke,
        },
        "memory_preflight": {"status": "not_run_dry_run"} if args.dry_run else None,
        "trials": trials,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_update_manifest(manifest_path, manifest)

    for trial in trials:
        if trial["trial_id"] not in selected:
            continue
        log_path = Path(trial["trial_dir"]) / "console.log"
        with log_path.open("x", encoding="utf-8") as log:
            log.write(("DRY RUN - not launched\n" if args.dry_run else "Prepared command\n") + trial["log_command"] + "\n")
        print(f"trial {trial['trial_id']:02d}: {trial['log_command']}")

    if args.dry_run:
        print(f"Dry run wrote {len(selected_ids)} resolved config(s) and {manifest_path}")
        return 0

    active_trial_count = min(max_concurrent, len(selected_ids))
    try:
        memory_report = _memory_preflight(
            trials[selected_ids[0] - 1]["_resolved_config"], gpu_pairs, active_trial_count,
        )
    except (MemoryError, RuntimeError, ValueError) as error:
        manifest["memory_preflight"] = {"status": "failed", "error": str(error)}
        for trial_id in selected_ids:
            trial = trials[trial_id - 1]
            trial["status"] = "failed_preflight"
            trial_dir = Path(trial["trial_dir"])
            with (trial_dir / "console.log").open("a", encoding="utf-8") as log:
                log.write(f"\nPRE-FLIGHT FAILED: {error}\n")
            _write_json(trial_dir / SUMMARY_FILENAME, {
                "trial_id": trial_id,
                "parameters": trial["parameters"],
                "status": "failed_preflight",
                "stop_reason": "replay_ram_preflight_failed",
                "error": str(error),
            })
        manifest["overall_status"] = "failed_preflight"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_update_manifest(manifest_path, manifest)
        print(f"Replay RAM preflight failed: {error}", file=sys.stderr)
        return 1
    manifest["memory_preflight"] = {"status": "passed", **memory_report}
    _atomic_update_manifest(manifest_path, manifest)

    pending = [trials[trial_id - 1] for trial_id in selected_ids]
    available_pairs = list(gpu_pairs[:max_concurrent])
    running = {}
    cancelled_signal = None
    old_handlers = {}

    def request_cancel(signum, _frame):
        nonlocal cancelled_signal
        cancelled_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_cancel)

    try:
        while pending or running:
            if cancelled_signal is not None:
                for trial in pending:
                    trial["status"] = "cancelled_before_start"
                pending.clear()
                for process_info in list(running.values()):
                    process = process_info["process"]
                    process_info["hard_timeout"] = False
                    _terminate_owned_group(process, process_info["owned_descendants"])
                    process_info["log_file"].close()
                    summary = _finish_trial(
                        process_info["trial"], process.poll(), False,
                        time.monotonic() - process_info["started_monotonic"],
                    )
                    process_info["trial"]["status"] = "cancelled"
                    summary["stop_reason"] = "parent_cancelled"
                    summary["status"] = "cancelled"
                    _write_json(Path(process_info["trial"]["trial_dir"]) / SUMMARY_FILENAME, summary)
                    process_info["trial"]["status"] = "cancelled"
                    available_pairs.append(process_info["gpu_pair"])
                    del running[process.pid]
                manifest["overall_status"] = "cancelled"
                manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_update_manifest(manifest_path, manifest)
                return 128 + cancelled_signal

            while pending and len(running) < max_concurrent:
                trial = pending.pop(0)
                gpu_pair = tuple(available_pairs.pop(0))
                trial["gpu_pair"] = list(gpu_pair)
                command = _build_command(
                    Path(trial["config_path"]), Path(trial["trial_dir"]), gpu_pair, trial_hours,
                )
                trial["command"] = command
                trial["log_command"] = _trial_command_for_log(command, gpu_pair, args.no_wandb or args.smoke)
                environment = os.environ.copy()
                environment.update(
                    CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_pair)),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                )
                if trial_hours is None:
                    environment.pop("SAC16_TRIAL_DEADLINE_MONOTONIC", None)
                else:
                    environment["SAC16_TRIAL_DEADLINE_MONOTONIC"] = str(
                        time.monotonic() + trial_hours * 3600
                    )
                if args.smoke or args.no_wandb:
                    environment["WANDB_MODE"] = "disabled"
                Path(trial["_resolved_config"]["train"]["data_dir"]).mkdir(parents=True, exist_ok=False)
                log_file = (Path(trial["trial_dir"]) / "console.log").open("a", encoding="utf-8")
                log_file.write(f"\nLAUNCH: {trial['log_command']}\n")
                log_file.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log_file.close()
                    raise
                trial["status"] = "running"
                trial["gpu_pair"] = list(gpu_pair)
                trial["started_at_utc"] = datetime.now(timezone.utc).isoformat()
                trial["log_command"] = _trial_command_for_log(command, gpu_pair, args.no_wandb or args.smoke)
                running[process.pid] = {
                    "process": process,
                    "trial": trial,
                    "gpu_pair": gpu_pair,
                    "log_file": log_file,
                    "started_monotonic": time.monotonic(),
                    "hard_deadline": (
                        None
                        if trial_hours is None
                        else time.monotonic() + trial_hours * 3600 + grace_seconds
                    ),
                    "hard_timeout": False,
                    "owned_descendants": {},
                }
                _atomic_update_manifest(manifest_path, manifest)

            now = time.monotonic()
            for process_id, process_info in list(running.items()):
                process = process_info["process"]
                exit_code = process.poll()
                hard_timeout = False
                hard_deadline = process_info["hard_deadline"]
                if exit_code is None and hard_deadline is not None and now >= hard_deadline:
                    hard_timeout = True
                    process_info["hard_timeout"] = True
                    exit_code = _terminate_owned_group(process, process_info["owned_descendants"])
                if exit_code is None:
                    _track_owned_descendants(process, process_info["owned_descendants"])
                    continue
                if not hard_timeout:
                    _terminate_owned_group(process, process_info["owned_descendants"])
                process_info["log_file"].close()
                trial = process_info["trial"]
                elapsed_seconds = time.monotonic() - process_info["started_monotonic"]
                summary = _finish_trial(trial, exit_code, hard_timeout, elapsed_seconds)
                trial["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                trial["full_time_seconds"] = elapsed_seconds
                trial["status"] = summary["status"]
                available_pairs.append(process_info["gpu_pair"])
                del running[process_id]
                _atomic_update_manifest(manifest_path, manifest)
            if pending or running:
                time.sleep(POLL_INTERVAL_SECONDS)
    except BaseException as error:
        for process_info in list(running.values()):
            process = process_info["process"]
            exit_code = _terminate_owned_group(process, process_info["owned_descendants"])
            process_info["log_file"].close()
            trial = process_info["trial"]
            trial["status"] = "failed_launcher_exception"
            trial["exit_code"] = exit_code
            trial["stop_reason"] = f"launcher_exception: {type(error).__name__}"
            try:
                summary = _finish_trial(
                    trial,
                    exit_code,
                    False,
                    time.monotonic() - process_info["started_monotonic"],
                )
                summary["status"] = "failed_launcher_exception"
                summary["stop_reason"] = f"launcher_exception: {type(error).__name__}"
                _write_json(Path(trial["trial_dir"]) / SUMMARY_FILENAME, summary)
                trial["status"] = summary["status"]
                trial["stop_reason"] = summary["stop_reason"]
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        manifest["overall_status"] = "failed_launcher_exception"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            _atomic_update_manifest(manifest_path, manifest)
        except OSError:
            pass
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    failed = [trial for trial in trials if trial["trial_id"] in selected and trial["status"] != "completed"]
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["overall_status"] = "failed" if failed else "completed"
    _atomic_update_manifest(manifest_path, manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
