"""Sweep, profile, and vector-autotune entry points for Drive SAC."""

from copy import deepcopy
from functools import partial
import importlib
import itertools
import os
from pathlib import Path
import signal
import tempfile
import warnings

from pufferlib import pufferl_sac
_previous_sigint_handler = signal.getsignal(signal.SIGINT)
try:
    from pufferlib.pufferl import capture_rng_state, downsample, restore_rng_state
finally:
    signal.signal(signal.SIGINT, _previous_sigint_handler)
import pufferlib.sweep
import pufferlib.vector
import torch


def _require_single_process(workflow_name):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            f"SAC {workflow_name} currently requires one process; WORLD_SIZE={world_size}"
        )


def _new_invocation_dir(args, workflow_name):
    output_dir = Path(args["train"]["data_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{workflow_name}_", dir=output_dir))


def _reject_resume(args, workflow_name):
    if args["train"].get("resume_state_path") is not None:
        raise ValueError(
            f"SAC {workflow_name} does not support train.resume_state_path; "
            "use an actor warmstart with load_model_path or load_id"
        )


def _fresh_run_args(args, invocation_dir, trial_index, total_timesteps=None):
    trial_args = deepcopy(args)
    trial_dir = invocation_dir / f"trial_{trial_index:04d}"
    trial_dir.mkdir()
    run_name = f"{invocation_dir.name}_{trial_index:04d}"

    trial_args["run_name"] = run_name
    trial_args["train"]["run_name"] = run_name
    trial_args["train"]["data_dir"] = str(trial_dir)
    trial_args["train"]["fresh_run"] = True
    if total_timesteps is not None:
        trial_args["train"]["total_timesteps"] = total_timesteps
    return trial_args


def sweep(args, env_name="puffer_drive"):
    """Run configured SAC trials and feed their downsampled metrics to a searcher."""
    _require_single_process("sweeps")
    _reject_resume(args, "sweeps")
    sweep_config = deepcopy(args["sweep"])
    method = sweep_config.pop("method")
    max_runs = args["max_runs"]
    metric = sweep_config["metric"]
    points_per_run = sweep_config["downsample"]
    try:
        sweep_class = getattr(pufferlib.sweep, method)
    except AttributeError as error:
        raise ValueError(f"Invalid SAC sweep method {method!r}; see pufferlib.sweep") from error

    search = sweep_class(sweep_config)
    target_key = f"environment/{metric}"
    invocation_dir = _new_invocation_dir(args, "sweep")

    for trial_index in range(1, max_runs + 1):
        trial_args = deepcopy(args)
        search.suggest(trial_args)
        total_timesteps = trial_args["train"]["total_timesteps"]
        trial_args = _fresh_run_args(trial_args, invocation_dir, trial_index)
        rng_state = capture_rng_state()
        try:
            logs = pufferl_sac.train(env_name=env_name, args=trial_args)
        finally:
            restore_rng_state({"rng_state": rng_state})
        metric_logs = [entry for entry in logs if target_key in entry]
        if not metric_logs:
            raise ValueError(f"SAC sweep trial returned no {target_key!r} metrics")

        scores = downsample([entry[target_key] for entry in metric_logs], points_per_run)
        costs = downsample([entry["uptime"] for entry in metric_logs], points_per_run)
        timesteps = downsample([entry["agent_steps"] for entry in metric_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            trial_args["train"]["total_timesteps"] = timestep
            search.observe(trial_args, score, cost)
        trial_args["train"]["total_timesteps"] = total_timesteps

    return search


def controlled_exp(args, env_name="puffer_drive"):
    """Run every combination in the configured controlled SAC experiment grid."""
    _require_single_process("controlled experiments")
    _reject_resume(args, "controlled experiments")
    params = {}
    for section, section_config in args["controlled_exp"].items():
        if isinstance(section_config, dict):
            for parameter, parameter_config in section_config.items():
                if isinstance(parameter_config, dict) and "values" in parameter_config:
                    params[f"{section}.{parameter}"] = parameter_config["values"]

    if not params:
        raise ValueError("No parameters with 'values' lists found in controlled_exp")

    keys = list(params)
    combinations = list(itertools.product(*(params[key] for key in keys)))
    invocation_dir = _new_invocation_dir(args, "controlled")
    results = []
    for trial_index, combination in enumerate(combinations, 1):
        trial_args = deepcopy(args)
        for key, value in zip(keys, combination):
            section, parameter = key.split(".", 1)
            trial_args[section][parameter] = value
        trial_args = _fresh_run_args(trial_args, invocation_dir, trial_index)
        logs = pufferl_sac.train(env_name=env_name, args=trial_args)
        results.append((dict(zip(keys, combination)), logs))
    return results


def profile(args, env_name="puffer_drive"):
    """Profile one bounded SAC training run and export its trace into that run."""
    _require_single_process("profiling")
    _reject_resume(args, "profiling")
    profile_config = args["profile"]
    total_timesteps = profile_config["total_timesteps"]
    invocation_dir = _new_invocation_dir(args, "profile")
    run_args = _fresh_run_args(args, invocation_dir, 1, total_timesteps=total_timesteps)

    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_active = str(args["train"].get("device", "cpu")).startswith("cuda") and torch.cuda.is_available()
    if cuda_active:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        with torch.profiler.record_function("sac_train"):
            logs = pufferl_sac.train(env_name=env_name, args=run_args)

    trace_path = Path(run_args["train"]["data_dir"]) / "trace.json"
    profiler.export_chrome_trace(str(trace_path))
    sort_by = "cuda_time_total" if cuda_active else "cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_by, row_limit=10))
    final_agent_steps = logs[-1].get("agent_steps") if logs else None
    print(
        f"SAC profile budget={total_timesteps} received rows; "
        f"final agent_steps={final_agent_steps}; trace={trace_path}"
    )
    return str(trace_path)


def autotune(args):
    """Run PufferLib vector autotune with SAC's final-observation payload enabled."""
    _require_single_process("vector autotune")
    vector_batch_size = args["vec"]["batch_size"]
    num_envs = args["vec"]["num_envs"]
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("SAC vector autotune requires vec.num_envs to be a positive integer")
    if vector_batch_size == "auto":
        vector_batch_size = 1 if num_envs == 1 else num_envs // 2
    elif vector_batch_size is None:
        vector_batch_size = num_envs
    if not isinstance(vector_batch_size, int) or isinstance(vector_batch_size, bool) or vector_batch_size <= 0:
        raise ValueError(
            "SAC vector autotune requires vec.batch_size to resolve to a positive integer "
            "(environment instances, not SAC replay capacity)"
        )

    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    env_constructor = env_module.env_creator(args["env_name"])
    env_kwargs = dict(args["env"], capture_final_observations=True)
    make_env = partial(env_constructor, **env_kwargs)

    autotune_config = args["autotune"]
    supported_options = (
        "max_envs",
        "model_forward_s",
        "max_env_ram_gb",
        "max_batch_vram_gb",
        "time_per_test",
    )
    options = {key: autotune_config[key] for key in supported_options if key in autotune_config}
    # Drive resets inside step(); vector.autotune computes an unused mean of
    # its empty explicit-reset timings. Leave all other numeric warnings intact.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Mean of empty slice\.?$",
            category=RuntimeWarning,
            module=r"numpy\.(?:core|_core)\.fromnumeric",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^invalid value encountered in scalar divide$",
            category=RuntimeWarning,
            module=r"numpy\.(?:core|_core)\._methods",
        )
        return pufferlib.vector.autotune(make_env, batch_size=vector_batch_size, **options)
