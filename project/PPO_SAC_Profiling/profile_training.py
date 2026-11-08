#!/usr/bin/env python3
"""Bounded same-hardware PPO/SAC profiling on the saved Drive configurations.

The harness constructs the real environment, policy, replay and learner directly.
It never enters the training CLI lifecycle, so benchmark evaluation, logger setup,
resume discovery and checkpoint writes stay disabled. Run from any directory; all
environment paths are resolved from the repository root.

Examples (activate ``.venv`` first)::

    python project/PPO_SAC_Profiling/profile_training.py --algorithm ppo \
      --output-dir /tmp/puffer-profile-ppo --device cuda --compile off
    torchrun --standalone --nproc-per-node=2 \
      project/PPO_SAC_Profiling/profile_training.py --algorithm sac \
      --output-dir /tmp/puffer-profile-sac-2gpu --device cuda

The default thread count is one to match torchrun's production default. Set
``--threads 4`` for a separate CPU-threading ablation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any

import psutil
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PPO_CONFIG = REPO_ROOT / (
    "experiments/baseline_ppo_4b_carla/"
    "baseline_ppo_4b_carla_2026-11-06_13-58-32_seed0/config.yaml"
)
SAC_CONFIG = REPO_ROOT / (
    "project/soft_actor_critic_target_entropy_scheduling/runs/"
    "round2_tes_sac_20261029T044847.014369240_539543/trial_98/learner/config.yaml"
)


class ProfileSetupError(RuntimeError):
    """Raised when a bounded profile cannot safely initialize."""


class NullProfile:
    """Disable PufferLib's section profiler and its internal CUDA sync points."""

    def __call__(self, *args, **kwargs):
        return None

    def __iter__(self):
        return iter(())

    def end(self):
        return None

    def clear(self):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile-config", type=Path, default=None,
                        help="Optional YAML defaults; CLI options override them")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Must be a new directory")
    parser.add_argument("--cycles", type=positive_int, default=3, help="Measured cycles after warmup")
    parser.add_argument("--warmup", type=nonnegative_int, default=1, help="Initial cycles excluded from summaries")
    parser.add_argument("--num-envs", type=positive_int, default=4)
    parser.add_argument("--agents", type=positive_int, default=800, help="Drive agents per environment")
    parser.add_argument("--horizon", type=positive_int, default=128, help="PPO rollout horizon")
    parser.add_argument("--replay-capacity", type=positive_int, default=128_000, help="SAC replay rows per rank")
    parser.add_argument("--logical-batch", type=positive_int, default=32_768)
    parser.add_argument("--microbatch", type=positive_int, default=8_192)
    parser.add_argument("--compile", choices=("on", "off"), default="off")
    parser.add_argument("--ppo-filter", choices=("on", "off"), default="on")
    parser.add_argument("--replay-location", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--stock-profiler", choices=("on", "off"), default="off",
                        help="PufferLib's section profiler synchronizes CUDA inside phases")
    parser.add_argument("--update-epochs", type=positive_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="Torch intra-op threads (1 matches torchrun production default)")
    parser.add_argument("--variant", choices=("stock", "obs_only", "cached", "fixed_targets", "fixed_targets_cached", "fixed_targets_actor_q"), default="stock",
                        help="Non-stock SAC variants load project-local sac_variants.py")
    parser.add_argument("--trace-cycle", type=nonnegative_int, default=None,
                        help="0-based total cycle index to capture with torch.profiler")
    parser.add_argument("--weights-from", type=Path, default=None,
                        help="Initialize model weights from a local model/checkpoint; optimizer and run state stay fresh")
    parser.add_argument("--target-audit-rows", type=nonnegative_int, default=0,
                        help="After all timed cycles, compare cached targets with current SAC targets on these rows")

    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--profile-config", type=Path, default=None)
    preset_args, _ = preset_parser.parse_known_args(argv)
    if preset_args.profile_config is not None:
        preset_path = preset_args.profile_config.expanduser().resolve()
        if not preset_path.is_file():
            parser.error(f"--profile-config file does not exist: {preset_path}")
        try:
            preset_values = yaml.safe_load(preset_path.read_text())
        except yaml.YAMLError as error:
            parser.error(f"invalid profile YAML {preset_path}: {error}")
        if not isinstance(preset_values, dict):
            parser.error(f"profile YAML must contain a mapping: {preset_path}")
        allowed_destinations = {
            action.dest for action in parser._actions
            if action.dest not in ("help", "profile_config")
        }
        unknown = sorted(set(preset_values) - allowed_destinations)
        if unknown:
            parser.error(f"unknown profile YAML keys in {preset_path}: {', '.join(unknown)}")
        actions_by_destination = {
            action.dest: action for action in parser._actions
            if action.dest not in ("help", "profile_config")
        }
        for name, value in preset_values.items():
            action = actions_by_destination[name]
            if action.choices is not None and value not in action.choices:
                parser.error(
                    f"invalid {name} value {value!r} in {preset_path}; "
                    f"choose from {', '.join(map(str, action.choices))}"
                )
            if action.type is not None and value is not None:
                try:
                    preset_values[name] = action.type(str(value))
                except (argparse.ArgumentTypeError, TypeError, ValueError) as error:
                    parser.error(f"invalid {name} value in {preset_path}: {error}")
        parser.set_defaults(**preset_values)

    args = parser.parse_args(argv)

    if args.algorithm is None:
        parser.error("--algorithm is required directly or through --profile-config")
    if args.output_dir is None:
        parser.error("--output-dir is required and must name a new directory")

    if args.logical_batch % args.microbatch:
        parser.error("--logical-batch must be divisible by --microbatch")
    if args.algorithm == "ppo" and args.microbatch % args.horizon:
        parser.error("PPO --microbatch must be divisible by --horizon")
    if (args.algorithm == "ppo" and args.ppo_filter == "off"
            and int(os.environ.get("WORLD_SIZE", "1")) > 1):
        parser.error("PPO --ppo-filter off is supported only for single-process profiles")
    if args.algorithm == "ppo" and args.variant != "stock":
        parser.error("non-stock --variant choices apply to SAC only")
    if args.target_audit_rows and not args.variant.startswith("fixed_targets"):
        parser.error("--target-audit-rows requires a fixed_targets variant")
    if args.algorithm == "sac" and args.variant != "stock" and not (
        Path(__file__).resolve().parent / "sac_variants.py"
    ).is_file():
        parser.error("non-stock SAC variants require project/PPO_SAC_Profiling/sac_variants.py")
    if args.algorithm == "sac" and args.logical_batch > args.replay_capacity:
        parser.error("SAC --logical-batch cannot exceed --replay-capacity")
    if args.trace_cycle is not None and args.trace_cycle >= args.warmup + args.cycles:
        parser.error("--trace-cycle must be less than --warmup + --cycles")

    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.profile_config is not None:
        args.profile_config = args.profile_config.expanduser().resolve()
    if args.weights_from is not None:
        args.weights_from = Path(args.weights_from).expanduser().resolve()
        if not args.weights_from.is_file():
            parser.error(f"--weights-from file does not exist: {args.weights_from}")
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def initialize_output_dir(output_dir: Path, rank: int, world_size: int) -> None:
    """Create exactly one new output directory and coordinate torchrun ranks."""
    ready_path = output_dir / ".profile_ready"
    startup_token = ":".join((
        os.environ.get("MASTER_PORT", "single"),
        os.environ.get("TORCHELASTIC_RUN_ID", "local"),
    ))
    if rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            ready_path.write_text(startup_token + "\n")
        except FileExistsError as error:
            raise ProfileSetupError(f"output directory already exists: {output_dir}") from error
    elif world_size > 1:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if ready_path.is_file() and ready_path.read_text().strip() == startup_token:
                break
            time.sleep(0.05)
        if not ready_path.is_file() or ready_path.read_text().strip() != startup_token:
            raise ProfileSetupError(
                f"rank 0 did not create a new output directory within 10 seconds: {output_dir}"
            )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary_path, path)


def trace_output_path(output_dir: Path, cycle_index: int, rank: int) -> Path:
    return output_dir / f"trace_cycle_{cycle_index:03d}_rank{rank:02d}.json"


def trace_operator_output_path(trace_path: Path) -> Path:
    return trace_path.with_name(trace_path.stem + "_operators.json")


def profile_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def git_metadata() -> dict[str, Any]:
    def git_value(*command: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *command],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git_value("status", "--porcelain")
    return {"commit": git_value("rev-parse", "HEAD"), "working_tree_dirty": bool(status)}


def host_metadata(torch, device: str, rank: int, local_rank: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "host_memory_bytes": psutil.virtual_memory().total,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device == "cuda" and torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_device)
        record["gpu"] = {
            "torch_device_index": current_device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    else:
        record["gpu"] = None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        record["nvidia_smi"] = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        record["nvidia_smi"] = {"error": str(error)}
    return record


def derive_vector_batch_size(vector_config: dict[str, Any], num_envs: int) -> int:
    """Keep saved receive grouping near its saved fraction of environments."""
    original_envs = vector_config.get("num_envs", num_envs)
    saved_batch_size = vector_config.get("batch_size", num_envs)
    if not isinstance(saved_batch_size, int) or isinstance(saved_batch_size, bool):
        return num_envs
    desired = max(1, round(num_envs * saved_batch_size / original_envs))
    divisors = [candidate for candidate in range(1, num_envs + 1) if num_envs % candidate == 0]
    return min(divisors, key=lambda candidate: (abs(candidate - desired), candidate))


def load_source_config(algorithm: str) -> tuple[dict[str, Any], Path]:
    source = PPO_CONFIG if algorithm == "ppo" else SAC_CONFIG
    if not source.is_file():
        raise ProfileSetupError(f"saved {algorithm.upper()} config is missing: {source}")
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("train"), dict):
        raise ProfileSetupError(f"expected a nested training config at {source}")
    return config, source


def prepare_config(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Path]:
    config, source_path = load_source_config(args.algorithm)
    config = copy.deepcopy(config)
    source_vec_config = dict(config["vec"])
    config["env"]["num_agents"] = args.agents
    config["vec"]["num_envs"] = args.num_envs
    config["vec"]["batch_size"] = derive_vector_batch_size(source_vec_config, args.num_envs)
    config["vec"]["seed"] = args.seed
    config["vec"].setdefault("num_workers", "auto")
    if config["vec"]["num_workers"] == "auto":
        pass
    elif isinstance(config["vec"]["num_workers"], int):
        requested_workers = min(config["vec"]["num_workers"], args.num_envs)
        config["vec"]["num_workers"] = max(
            workers for workers in range(1, requested_workers + 1) if args.num_envs % workers == 0
        )

    train = config["train"]
    train.update({
        "seed": args.seed,
        "update_epochs": args.update_epochs,
        "learning_rate": train.get("learning_rate", 5e-4),
        "compile": args.compile == "on",
        "compile_fullgraph": False if args.compile == "off" else train.get("compile_fullgraph", False),
        "evaluation_interval_epochs": None,
        "evaluation_benchmarks": None,
        "resume_state_path": None,
        "checkpoint_interval": None if args.algorithm == "sac" else 1_000_000_000,
        "total_timesteps": 1_000_000_000_000_000,
        "data_dir": str(output_dir / "disabled_artifacts"),
        "cpu_offload": args.replay_location == "cpu" if args.algorithm == "sac" else train.get("cpu_offload", False),
        "run_name": f"profile_{args.algorithm}_{args.variant}",
    })
    if args.algorithm == "ppo":
        train.update({
            "batch_size": args.num_envs * args.agents * args.horizon,
            "bptt_horizon": args.horizon,
            "minibatch_size": args.logical_batch,
            "max_minibatch_size": args.microbatch,
            "adv_filter_enabled": args.ppo_filter == "on",
            "render": False,
        })
        config["render"] = 0
        config["wandb"] = config["neptune"] = config["tb"] = False
        config["load_id"] = config["load_model_path"] = None
    else:
        train.update({
            "batch_size": args.replay_capacity,
            "minibatch_size": args.logical_batch,
            "max_minibatch_size": args.microbatch,
            "checkpoint_interval": None,
            "cpu_offload": args.replay_location == "cpu",
            "fresh_run": True,
        })
        config["env"]["capture_final_observations"] = True
        config["wandb"] = config["neptune"] = config["tb"] = False
        config["no_model_upload"] = True
        config["load_id"] = config["load_model_path"] = None

    return config, source_path


def initialize_distributed(torch, device_kind: str) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise ProfileSetupError("--device cuda was requested but torch.cuda.is_available() is false")
        visible_devices = torch.cuda.device_count()
        if local_rank >= visible_devices:
            raise ProfileSetupError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count {visible_devices}"
            )
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        # PufferDrive's multiprocessing vector workers are forked, so initialize
        # the process group only after the environment has started in run_benchmark.
        return rank, local_rank, world_size
    return rank, local_rank, world_size


def synchronize_boundary(torch, device: str, distributed: bool) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    if distributed:
        torch.distributed.barrier()


def phase_start(torch, device: str, distributed: bool) -> float:
    synchronize_boundary(torch, device, distributed)
    return time.perf_counter()


def phase_end(torch, device: str, distributed: bool, started: float) -> float:
    synchronize_boundary(torch, device, distributed)
    return time.perf_counter() - started


def tensor_bytes(value: Any) -> int:
    if not hasattr(value, "numel") or not hasattr(value, "element_size"):
        return 0
    return int(value.numel() * value.element_size())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile_weights(trainer, algorithm: str, weights_path: Path, torch, ppo) -> dict[str, Any]:
    """Load only model tensors; learner optimizers, counters, and replay stay fresh."""
    if algorithm == "ppo":
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ProfileSetupError(f"PPO model artifact must be a state dict: {weights_path}")
        trainer.uncompiled_policy.load_state_dict(ppo.clean_policy_state_dict(state), strict=True)
        loaded_keys = ["policy_state_dict"]
        artifact_format = "ppo_policy_state_dict"
    else:
        # Learner-only SAC checkpoints include NumPy RNG state, which PyTorch's
        # weights-only loader rejects. This path is explicit and local; only the
        # three model state dicts below are read into the fresh trainer.
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if (not isinstance(state, dict) or state.get("algorithm") != "sac"
                or state.get("scope") != "learner_only"):
            raise ProfileSetupError(
                f"SAC weights must come from a learner-only trainer_state.pt: {weights_path}"
            )
        source_to_module = (
            ("policy_state_dict", trainer.uncompiled_policy),
            ("critic_state_dict", trainer.uncompiled_critic),
            ("target_critic_state_dict", trainer.uncompiled_target_critic),
        )
        for state_key, module in source_to_module:
            model_state = state.get(state_key)
            if not isinstance(model_state, dict):
                raise ProfileSetupError(f"SAC checkpoint is missing {state_key}: {weights_path}")
            module.load_state_dict(model_state, strict=True)
        loaded_keys = [state_key for state_key, _module in source_to_module]
        artifact_format = "sac_learner_only_trainer_state"
    del state

    return {
        "source_path": str(weights_path),
        "source_sha256": sha256_file(weights_path),
        "source_format": artifact_format,
        "model_state_keys_loaded": loaded_keys,
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "training_counters_restored": False,
        "rng_state_restored": False,
        "replay_restored": False,
        "log_alpha_restored": False,
        "initialization": "model weights only; optimizer, scheduler, counters, RNG, and replay remain fresh",
    }


def final_model_digest(trainer, algorithm: str, torch) -> dict[str, Any]:
    """Hash final model state and check parameter finiteness outside timed cycles."""
    models = [("policy", trainer.uncompiled_policy)]
    if algorithm == "sac":
        models.extend((
            ("critic", trainer.uncompiled_critic),
            ("target_critic", trainer.uncompiled_target_critic),
        ))
    digest = hashlib.sha256()
    all_parameters_finite = True
    for model_name, module in models:
        for state_name, value in sorted(module.state_dict().items()):
            digest.update(f"{model_name}.{state_name}".encode("utf-8"))
            cpu_value = value.detach().contiguous().to(device="cpu")
            digest.update(cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes())
        for parameter in module.parameters():
            cpu_parameter = parameter.detach().to(device="cpu")
            if cpu_parameter.is_floating_point() or cpu_parameter.is_complex():
                all_parameters_finite = all_parameters_finite and bool(torch.isfinite(cpu_parameter).all())
    if algorithm == "sac":
        alpha = trainer.log_alpha.detach().contiguous().to(device="cpu")
        digest.update(b"log_alpha")
        digest.update(alpha.reshape(-1).view(torch.uint8).numpy().tobytes())
        if alpha.is_floating_point() or alpha.is_complex():
            all_parameters_finite = all_parameters_finite and bool(torch.isfinite(alpha).all())
    return {
        "sha256": digest.hexdigest(),
        "model_state_keys": [name for name, _module in models],
        "includes_log_alpha": algorithm == "sac",
        "all_parameters_finite": all_parameters_finite,
        "computed_outside_timed_cycles": True,
    }


def instrument_replay(replay) -> dict[str, int]:
    counters = {"get_batch_calls": 0, "get_batch_bytes": 0, "inserted_rows": 0}
    if hasattr(replay, "get_batch"):
        original = replay.get_batch

        def get_batch(indices, _original=original):
            tensors = _original(indices)
            counters["get_batch_calls"] += 1
            counters["get_batch_bytes"] += sum(tensor_bytes(tensor) for tensor in tensors)
            return tensors

        replay.get_batch = get_batch
    if hasattr(replay, "add"):
        original = replay.add

        def add(observations, *values, _original=original):
            result = _original(observations, *values)
            counters["inserted_rows"] += int(observations.shape[0])
            return result

        replay.add = add
    if hasattr(replay, "replace"):
        original = replay.replace

        def replace(indices, *values, _original=original):
            result = _original(indices, *values)
            counters["inserted_rows"] += int(indices.numel())
            return result

        replay.replace = replace
    if hasattr(replay, "add_sequence"):
        original = replay.add_sequence

        def add_sequence(agent_ids, *values, _original=original):
            result = _original(agent_ids, *values)
            counters["inserted_rows"] += int(agent_ids.numel())
            return result

        replay.add_sequence = add_sequence
    return counters


def attach_optimizer_counters(optimizers: dict[str, Any]) -> dict[str, int]:
    counters = {name: 0 for name in optimizers}
    for name, optimizer in optimizers.items():
        optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs, counter_name=name: counters.__setitem__(
                counter_name, counters[counter_name] + 1
            )
        )
    return counters


def profiler_context(torch, device: str):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)


def profiler_operator_summary(profiler, device: str) -> dict[str, Any]:
    """Return the 30-row text table and machine-readable per-operator timings."""
    events = profiler.key_averages()
    sample = events[0] if events else None
    if device == "cuda":
        sort_by = "self_device_time_total"
        if sample is not None and not hasattr(sample, "self_device_time_total"):
            sort_by = "self_cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"

    def get_device_time(event, *field_names):
        for field_name in field_names:
            if hasattr(event, field_name):
                value = getattr(event, field_name)
                if value is not None:
                    return float(value)
        return 0.0

    key_averages = []
    for event in events:
        key_averages.append({
            "key": event.key,
            "count": int(event.count),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "cpu_time_total_us": float(event.cpu_time_total),
            "self_device_time_total_us": get_device_time(
                event, "self_device_time_total", "self_cuda_time_total",
            ),
            "device_time_total_us": get_device_time(
                event, "device_time_total", "cuda_time_total",
            ),
        })

    return {
        "sort_by": sort_by,
        "table_row_limit": 30,
        "table": events.table(sort_by=sort_by, row_limit=30),
        "key_averages": key_averages,
    }


def measurement_included(cycle_index: int, warmup: int, trace_cycle: int | None) -> bool:
    return cycle_index >= warmup and cycle_index != trace_cycle


def import_sac_variant(variant: str):
    if variant == "stock":
        return None
    module_path = Path(__file__).resolve().parent / "sac_variants.py"
    spec = importlib.util.spec_from_file_location("profile_sac_variants", module_path)
    if spec is None or spec.loader is None:
        raise ProfileSetupError(f"cannot import project-local SAC variant module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    installer = getattr(module, "install_variant", None)
    if not callable(installer):
        raise ProfileSetupError(f"{module_path} must expose install_variant(learner, variant)")
    return installer


def construct_trainer(args, config, output_dir, torch, ppo, rank, local_rank, world_size):
    """Build Drive components and return a trainer with benchmark-only counters."""
    is_distributed = world_size > 1
    device = "cuda" if args.device == "cuda" else "cpu"
    env_args = copy.deepcopy(config)
    env_args["train"]["device"] = device
    env_args["train"]["seed"] = args.seed
    torch_seed, env_seed = ppo.derive_rank_seeds(
        env_args["vec"].get("seed"), args.seed, world_size, rank,
    )
    import random
    import numpy as np

    random.seed(torch_seed)
    np.random.seed(torch_seed % 2**32)
    torch.manual_seed(torch_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(torch_seed)

    vecenv = None
    trainer = None
    try:
        vecenv = ppo.load_env("puffer_drive", env_args, seed=env_seed)
        env_module_name = "pufferlib.ocean"
        env_module = __import__(env_module_name, fromlist=["torch"])
        if args.algorithm == "ppo":
            policy_cls = getattr(env_module.torch, env_args["policy_name"])
            policy = policy_cls(vecenv.driver_env, **env_args["policy"]).to(device)
        else:
            from pufferlib import pufferl_sac
            from pufferlib.sac_replay import TensorReplay
            from pufferlib.ocean.torch_sac import DriveSACCritic

            policy = pufferl_sac.load_policy(env_args, vecenv, "puffer_drive")
            critic = DriveSACCritic(vecenv.driver_env, **env_args["policy"]).to(
                device=device, dtype=torch.float32,
            )

        if is_distributed:
            backend = "nccl" if args.device == "cuda" else "gloo"
            torch.distributed.init_process_group(backend=backend, init_method="env://")

        if args.algorithm == "ppo":
            if is_distributed:
                policy = torch.nn.parallel.DistributedDataParallel(
                    policy, device_ids=[local_rank] if args.device == "cuda" else None,
                    output_device=local_rank if args.device == "cuda" else None,
                )
                policy.forward_eval = policy.module.forward_eval

            train_config = {
                **env_args["train"],
                "env": "puffer_drive",
                "eval": {},
                "run_name": env_args["run_name"],
            }

            class MeasuredPPO(ppo.PuffeRL):
                def __init__(self, *constructor_args, **constructor_kwargs):
                    self.loss_rows = 0
                    super().__init__(*constructor_args, **constructor_kwargs)
                    if args.stock_profiler == "off":
                        self.profile = NullProfile()

                def _ppo_loss(self, observations, *loss_args, **loss_kwargs):
                    self.loss_rows += int(observations.shape[0])
                    return super()._ppo_loss(observations, *loss_args, **loss_kwargs)

                def print_dashboard(self, *dashboard_args, **dashboard_kwargs):
                    return None

                def save_checkpoint(self):
                    raise RuntimeError("profiling harness must never write a checkpoint")

            trainer = MeasuredPPO(train_config, vecenv, policy, logger=ppo.NoLogger(train_config))
            optimizer_counters = attach_optimizer_counters({"policy": trainer.optimizer})
            return trainer, vecenv, optimizer_counters, None, None

        from pufferlib import pufferl_sac
        from pufferlib.sac_sequence_replay import SequenceReplay

        train = env_args["train"]
        sac_config = {key: value for key, value in train.items() if key in pufferl_sac.SAC_TRAIN_KEYS}
        sac_config.update({
            "env": "puffer_drive",
            "run_name": env_args["run_name"],
            "action_count": policy.action_count,
            "data_dir": str(output_dir / "disabled_artifacts"),
            "device": device,
            "total_timesteps": 1_000_000_000_000_000,
            "batch_size": args.replay_capacity,
            "minibatch_size": args.logical_batch,
            "max_minibatch_size": args.microbatch,
            "update_epochs": args.update_epochs,
            "compile": args.compile == "on",
            "cpu_offload": args.replay_location == "cpu",
            "checkpoint_interval": None,
            "resume_state_path": None,
            "fresh_run": True,
            "seed": args.seed,
        })
        replay_device = "cpu" if args.replay_location == "cpu" else torch.device(device)
        replay_kwargs = {
            "device": replay_device,
            "pin_memory": args.replay_location == "cpu" and args.device == "cuda",
        }
        if sac_config.get("return_estimator") == "pql":
            replay = SequenceReplay(
                args.replay_capacity, vecenv.single_observation_space.shape,
                num_agents=vecenv.num_agents, **replay_kwargs,
            )
        else:
            replay = TensorReplay(
                args.replay_capacity, vecenv.single_observation_space.shape, **replay_kwargs,
            )
        replay_counters = instrument_replay(replay)

        class MeasuredSAC(pufferl_sac.PuffeSAC):
            def __init__(self, *constructor_args, **constructor_kwargs):
                self.critic_loss_rows = 0
                self.policy_loss_rows = 0
                self.target_forward_rows = 0
                self.target_forward_calls = 0
                super().__init__(*constructor_args, **constructor_kwargs)
                if args.stock_profiler == "off":
                    self.profile = NullProfile()

            def _critic_loss(self, observations, *loss_args, **loss_kwargs):
                self.critic_loss_rows += int(observations.shape[0])
                return super()._critic_loss(observations, *loss_args, **loss_kwargs)

            def _compute_next_values(self, observations, *target_args, **target_kwargs):
                self.target_forward_rows += int(observations.shape[0])
                self.target_forward_calls += 1
                return super()._compute_next_values(observations, *target_args, **target_kwargs)

            def _policy_loss(self, observations, *loss_args, **loss_kwargs):
                self.policy_loss_rows += int(observations.shape[0])
                return super()._policy_loss(observations, *loss_args, **loss_kwargs)

            def print_dashboard(self, *dashboard_args, **dashboard_kwargs):
                return None

            def save_checkpoint(self):
                raise RuntimeError("profiling harness must never write a checkpoint")

        trainer = MeasuredSAC(
            sac_config, vecenv=vecenv, policy=policy, critic=critic, replay=replay,
            logger=ppo.NoLogger(sac_config),
        )
        optimizer_counters = attach_optimizer_counters({
            "actor": trainer.optimizer,
            "critic": trainer.critic_optimizer,
            "alpha": trainer.alpha_optimizer,
        })
        installer = import_sac_variant(args.variant)
        if installer is not None:
            installer(trainer, args.variant)
        return trainer, vecenv, optimizer_counters, replay_counters, sac_config
    except BaseException:
        if trainer is not None:
            trainer.utilization.stop()
            trainer.utilization.join(timeout=5)
        if vecenv is not None:
            vecenv.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise


def audit_fixed_targets(trainer, requested_rows, torch):
    """Quantify target staleness using the original precompute batch context."""
    replay = trainer.replay
    row_count = min(requested_rows, len(replay))
    indices = torch.linspace(0, len(replay) - 1, row_count, dtype=torch.long, device="cpu")
    precompute_batch_size = int(trainer.max_minibatch_size)
    current = torch.empty(row_count, dtype=torch.float32, device=trainer.device)
    selected_chunk_starts = torch.div(
        indices, precompute_batch_size, rounding_mode="floor",
    ) * precompute_batch_size
    forward_rows_for_audit = 0
    with torch.no_grad():
        # Re-evaluate every original contiguous microbatch that contains an
        # audit row. Tensor/matmul kernels may choose different BF16 results
        # when the batch dimension differs from target precomputation.
        for chunk_start in torch.unique_consecutive(selected_chunk_starts).tolist():
            chunk_end = min(chunk_start + precompute_batch_size, len(replay))
            rewards = replay.rewards[chunk_start:chunk_end].to(
                trainer.device, non_blocking=True,
            )
            next_observations = replay.next_observations[chunk_start:chunk_end].to(
                trainer.device, non_blocking=True,
            )
            discounts = replay.discounts[chunk_start:chunk_end].to(
                trainer.device, non_blocking=True,
            )
            chunk_targets = trainer._compute_targets(rewards, next_observations, discounts).float()
            row_positions = torch.nonzero(
                selected_chunk_starts == chunk_start, as_tuple=True,
            )[0]
            row_offsets = indices[row_positions] - chunk_start
            current.index_copy_(
                0,
                row_positions.to(trainer.device),
                chunk_targets.index_select(0, row_offsets.to(trainer.device)),
            )
            forward_rows_for_audit += chunk_end - chunk_start

        cached = trainer.fixed_cycle_targets[indices.to(trainer.device)].float()
        difference = current - cached
        return {
            "computed_after_all_timed_cycles": True,
            "row_selection": "evenly spaced replay storage indices",
            "rows": row_count,
            "matching_precompute_batch_context": True,
            "precompute_batch_size": precompute_batch_size,
            "forward_rows_for_audit": forward_rows_for_audit,
            "cached_mean": cached.mean().item(),
            "current_mean": current.mean().item(),
            "mean_absolute_difference": difference.abs().mean().item(),
            "max_absolute_difference": difference.abs().max().item(),
            "root_mean_square_difference": difference.square().mean().sqrt().item(),
            "current_mean_absolute_target": current.abs().mean().item(),
            "all_finite": bool(torch.isfinite(current).all() and torch.isfinite(cached).all()),
            "meaning": "Cached cycle-start targets versus targets from final updated actor, target critics and alpha",
        }


def run_benchmark(args: argparse.Namespace) -> None:
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))

    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProfileSetupError("--device cuda requires torch.cuda.is_available()")
    rank, local_rank, world_size = initialize_distributed(torch, args.device)
    distributed = world_size > 1
    if args.device == "cuda" and args.compile == "off":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve and record the saved experiment before constructing Drive workers.
    config, source_config_path = prepare_config(args, args.output_dir)
    initialize_output_dir(args.output_dir, rank, world_size)
    os.chdir(REPO_ROOT)

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        from pufferlib import pufferl
    finally:
        # pufferl normally exits immediately on SIGINT; a bounded profile must
        # retain the caller's handler so the finally block can close workers.
        signal.signal(signal.SIGINT, previous_sigint_handler)

    trainer, vecenv, optimizer_counters, replay_counters, trainer_config = construct_trainer(
        args, config, args.output_dir, torch, pufferl, rank, local_rank, world_size,
    )
    if replay_counters is not None:
        trainer_config = trainer_config or trainer.config
    weight_initialization = {
        "source_path": None,
        "source_sha256": None,
        "source_format": "random_initialization",
        "model_state_keys_loaded": [],
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "training_counters_restored": False,
        "rng_state_restored": False,
        "replay_restored": False,
        "log_alpha_restored": False,
        "initialization": "random model weights; optimizer, scheduler, counters, RNG, and replay remain fresh",
    }

    git_info = git_metadata()
    host_info = host_metadata(torch, args.device, rank, local_rank)
    resolved_config = config
    if args.algorithm == "sac":
        resolved_config = copy.deepcopy(config)
        resolved_config["train"].update(trainer.config)
    config_hash = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    total_cycles = args.warmup + args.cycles
    cycle_ids = []
    measured_ids = []

    try:
        if args.weights_from is not None:
            weight_initialization = load_profile_weights(
                trainer, args.algorithm, args.weights_from, torch, pufferl,
            )
        for cycle_index in range(total_cycles):
            is_warmup = cycle_index < args.warmup
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if args.algorithm == "ppo":
                received_before = trainer.global_step
                trainer.loss_rows = 0
                valid_before = int(trainer.masks.sum().item()) if cycle_index else 0
            else:
                received_before = trainer.global_step
                agent_steps_before = trainer.agent_steps
                inserted_before = replay_counters["inserted_rows"]
                batch_calls_before = replay_counters["get_batch_calls"]
                batch_bytes_before = replay_counters["get_batch_bytes"]
                trainer.critic_loss_rows = 0
                trainer.policy_loss_rows = 0
                trainer.target_forward_rows = 0
                trainer.target_forward_calls = 0
            optimizer_before = dict(optimizer_counters)
            loss_start = trainer.loss_rows if args.algorithm == "ppo" else 0
            initial_allocated = torch.cuda.memory_allocated() if args.device == "cuda" else 0

            use_trace = args.trace_cycle == cycle_index
            collection_started = phase_start(torch, args.device, distributed)
            trace_profiler = profiler_context(torch, args.device) if use_trace else None
            if trace_profiler is not None:
                trace_profiler.start()
                with torch.profiler.record_function("profile_collection"):
                    trainer.evaluate()
            else:
                trainer.evaluate()
            collection_seconds = phase_end(torch, args.device, distributed, collection_started)
            if trace_profiler is not None:
                trace_profiler.step()

            optimization_started = phase_start(torch, args.device, distributed)
            if trace_profiler is not None:
                with torch.profiler.record_function("profile_training"):
                    trainer.train()
            else:
                trainer.train()
            optimization_seconds = phase_end(torch, args.device, distributed, optimization_started)
            if trace_profiler is not None:
                trace_profiler.step()
                trace_profiler.stop()
                trace_path = trace_output_path(args.output_dir, cycle_index, rank)
                trace_profiler.export_chrome_trace(str(trace_path))
                trace_operator_path = trace_operator_output_path(trace_path)
                atomic_json(
                    trace_operator_path,
                    {
                        "schema_version": 1,
                        "algorithm": args.algorithm,
                        "variant": args.variant,
                        "cycle_index": cycle_index,
                        "rank": rank,
                        "device": args.device,
                        **profiler_operator_summary(trace_profiler, args.device),
                    },
                )
            else:
                trace_path = None
                trace_operator_path = None

            received_local = int(trainer.global_step - received_before)
            if args.algorithm == "ppo":
                valid_rollout_local = int(trainer.masks.sum().item())
                # PPO's transition trainer revisits the retained rollout once per epoch.
                loss_rows = int(trainer.loss_rows - loss_start)
                train_rows_per_epoch = loss_rows // args.update_epochs
                inserted_local = 0
                replay_calls = 0
                replay_bytes = 0
                received_global = None
                inserted_global = 0
            else:
                valid_rollout_local = None
                loss_rows = int(trainer.critic_loss_rows + trainer.policy_loss_rows
                                + getattr(trainer, "fixed_actor_loss_rows", 0))
                train_rows_per_epoch = None
                inserted_local = int(replay_counters["inserted_rows"] - inserted_before)
                replay_calls = int(replay_counters["get_batch_calls"] - batch_calls_before)
                replay_bytes = int(replay_counters["get_batch_bytes"] - batch_bytes_before)
                received_global = int(trainer.agent_steps - agent_steps_before)
                inserted_global = None

            local_row = {
                "rank": rank,
                "local_rank": local_rank,
                "collection_seconds": collection_seconds,
                "optimization_seconds": optimization_seconds,
                "cycle_seconds": collection_seconds + optimization_seconds,
                "received_rows_local": received_local,
                "received_rows_global_counter_delta": received_global,
                "valid_rollout_rows_local": valid_rollout_local,
                "train_rows_per_epoch_local": train_rows_per_epoch,
                "loss_row_exposures_local": loss_rows,
                "critic_loss_rows_local": int(getattr(trainer, "critic_loss_rows", 0)),
                "policy_loss_rows_local": int(getattr(trainer, "policy_loss_rows", 0)
                                              + getattr(trainer, "fixed_actor_loss_rows", 0)),
                "target_forward_rows_local": int(getattr(trainer, "target_forward_rows", 0)),
                "target_forward_calls_local": int(getattr(trainer, "target_forward_calls", 0)),
                "inserted_rows_local": inserted_local,
                "inserted_rows_global_counter_delta": inserted_global,
                "replay_get_batch_calls_local": replay_calls,
                "replay_get_batch_bytes_local": replay_bytes,
                "fixed_target_precompute_rows_local": int(getattr(trainer, "fixed_target_precompute_rows", 0)),
                "fixed_target_precompute_calls_local": int(getattr(trainer, "fixed_target_precompute_calls", 0)),
                "fixed_actor_q_precompute_rows_local": int(getattr(trainer, "fixed_actor_q_precompute_rows", 0)),
                "fixed_actor_q_precompute_calls_local": int(getattr(trainer, "fixed_actor_q_precompute_calls", 0)),
                "fixed_actor_q_cache_bytes_local": tensor_bytes(trainer.fixed_cycle_actor_q)
                    if getattr(trainer, "fixed_cycle_actor_q", None) is not None else 0,
                "optimizer_steps_local": {
                    name: optimizer_counters[name] - optimizer_before[name]
                    for name in optimizer_counters
                },
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None,
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else None,
                "initial_gpu_allocated_bytes": int(initial_allocated) if args.device == "cuda" else None,
                "trace_path": str(trace_path) if trace_path else None,
                "trace_operator_summary_path": str(trace_operator_path) if trace_operator_path else None,
                "stock_profiler_section_sampled": (
                    args.stock_profiler == "on" and (cycle_index + 1) % 5 == 0
                ),
                "host": host_info,
            }
            gathered_rows = [None] * world_size
            if distributed:
                torch.distributed.all_gather_object(gathered_rows, local_row)
            else:
                gathered_rows[0] = local_row

            if rank == 0:
                global_received = sum(row["received_rows_local"] for row in gathered_rows)
                global_valid_rollout = (
                    sum(row["valid_rollout_rows_local"] for row in gathered_rows)
                    if args.algorithm == "ppo" else None
                )
                global_inserted = (
                    sum(row["inserted_rows_local"] for row in gathered_rows)
                    if args.algorithm == "sac" else 0
                )
                global_loss_rows = sum(row["loss_row_exposures_local"] for row in gathered_rows)
                global_sampled_rows = (
                    sum(row["critic_loss_rows_local"] for row in gathered_rows)
                    if args.algorithm == "sac" else global_loss_rows
                )
                global_cycle_seconds = max(row["cycle_seconds"] for row in gathered_rows)
                include_measurement = measurement_included(cycle_index, args.warmup, args.trace_cycle)
                cycle_record = {
                    "schema_version": 1,
                    "algorithm": args.algorithm,
                    "variant": args.variant,
                    "cycle_index": cycle_index,
                    "warmup_excluded": is_warmup,
                    "measurement_included": include_measurement,
                    "trace_excluded": use_trace,
                    "torch_profiler_intrusive": use_trace,
                    "trace_paths": [row["trace_path"] for row in gathered_rows if row["trace_path"]],
                    "trace_operator_summary_paths": [
                        row["trace_operator_summary_path"]
                        for row in gathered_rows if row["trace_operator_summary_path"]
                    ],
                    "config_source": str(source_config_path),
                    "config_sha256": config_hash,
                    "weight_initialization": weight_initialization,
                    "git": git_info,
                    "command_line": sys.argv,
                    "profile_settings": profile_settings(args),
                    "config": resolved_config,
                    "world_size": world_size,
                    "ranks": gathered_rows,
                    "global": {
                        "received_rows": global_received,
                        "valid_rollout_rows": global_valid_rollout,
                        "inserted_rows": global_inserted,
                        "sampled_transitions": global_sampled_rows,
                        "loss_row_exposures": global_loss_rows,
                        "replay_get_batch_calls": sum(row["replay_get_batch_calls_local"] for row in gathered_rows),
                        "replay_get_batch_bytes": sum(row["replay_get_batch_bytes_local"] for row in gathered_rows),
                        "replay_get_batch_bytes_semantics": (
                            "bytes returned by replay.get_batch; variants that bypass this method are not included"
                        ),
                        "optimizer_steps": {
                            name: sum(row["optimizer_steps_local"][name] for row in gathered_rows)
                            for name in optimizer_counters
                        },
                        "collection_seconds": max(row["collection_seconds"] for row in gathered_rows),
                        "optimization_seconds": max(row["optimization_seconds"] for row in gathered_rows),
                        "cycle_seconds": global_cycle_seconds,
                        "received_rows_per_second": global_received / max(global_cycle_seconds, 1e-12),
                        "loss_row_exposures_per_second": global_loss_rows / max(global_cycle_seconds, 1e-12),
                        "peak_gpu_allocated_bytes": max(
                            (row["peak_gpu_allocated_bytes"] or 0 for row in gathered_rows), default=0,
                        ) or None,
                        "peak_gpu_reserved_bytes": max(
                            (row["peak_gpu_reserved_bytes"] or 0 for row in gathered_rows), default=0,
                        ) or None,
                    },
                }
                atomic_json(args.output_dir / f"cycle_{cycle_index:03d}.json", cycle_record)
                cycle_ids.append(cycle_index)
                if include_measurement:
                    measured_ids.append(cycle_index)
                status = "warmup excluded" if is_warmup else (
                    "trace excluded" if use_trace else "measured"
                )
                print(
                    f"cycle {cycle_index + 1}/{total_cycles}: {status}; "
                    f"received={global_received:,}; loss_rows={global_loss_rows:,}; "
                    f"seconds={global_cycle_seconds:.2f}",
                    flush=True,
                )
            if distributed:
                torch.distributed.barrier()

        local_final_state = {
            "rank": rank,
            **final_model_digest(trainer, args.algorithm, torch),
            "fixed_target_audit": audit_fixed_targets(trainer, args.target_audit_rows, torch)
            if args.target_audit_rows else None,
        }
        final_states = [None] * world_size
        if distributed:
            torch.distributed.all_gather_object(final_states, local_final_state)
        else:
            final_states[0] = local_final_state

        if rank == 0:
            summary = {
                "schema_version": 1,
                "algorithm": args.algorithm,
                "variant": args.variant,
                "config_source": str(source_config_path),
                "config_sha256": config_hash,
                "weight_initialization": weight_initialization,
                "final_model_states": final_states,
                "all_parameters_finite": all(
                    state["all_parameters_finite"] for state in final_states
                ),
                "git": git_info,
                "command_line": sys.argv,
                "world_size": world_size,
                "warmup_cycles_excluded": list(range(args.warmup)),
                "measured_cycles": measured_ids,
                "trace_cycles_excluded": [args.trace_cycle] if args.trace_cycle is not None else [],
                "cycle_files": [f"cycle_{cycle_index:03d}.json" for cycle_index in cycle_ids],
                "profile_settings": profile_settings(args),
            }
            atomic_json(args.output_dir / "summary.json", summary)
        if distributed:
            torch.distributed.barrier()
    finally:
        # PuffeRL/PuffeSAC close() writes checkpoints; stop their monitor and
        # vector workers directly so this bounded process leaves only reports.
        try:
            if trainer is not None:
                trainer.utilization.stop()
                trainer.utilization.join(timeout=5)
        finally:
            if vecenv is not None:
                vecenv.close()
            if rank == 0:
                (args.output_dir / ".profile_ready").unlink(missing_ok=True)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileSetupError as error:
        print(f"profile setup error: {error}", file=sys.stderr)
        raise SystemExit(2)
