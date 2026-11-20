#!/usr/bin/env python3
"""Profile TES filtering cases with the existing bounded SAC harness.

Harness options are forwarded unchanged. Each invocation writes to a fresh
``--output-dir``; run baseline, filter, and all_keep separately with identical
arguments apart from ``--case`` and output directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HARNESS_PATH = REPO_ROOT / "project/PPO_SAC_Profiling/profile_training.py"


def import_harness():
    spec = importlib.util.spec_from_file_location("tes_filter_profile_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import profiling harness: {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("All profile_training.py options are forwarded. Run that script with "
                "--help to see its full option list."),
    )
    parser.add_argument("--case", choices=("baseline", "filter", "all_keep"), required=True)
    parser.add_argument("--detailed", action="store_true",
                        help="Add synchronized method-level wall timers (extra overhead)")
    parser.add_argument("--restore-filter-state", action="store_true",
                        help="Restore checkpoint EWMA/counters; valid only for --case filter")
    parser.add_argument("--restore-update-state", action="store_true",
                        help="Restore optimizers, LR schedules and entropy controller for late-run probes")
    wrapper_args, harness_argv = parser.parse_known_args(argv)
    harness = import_harness()
    args = harness.parse_args(harness_argv)
    if args.algorithm != "sac":
        parser.error("--algorithm sac is required")
    if args.source_config is None or args.weights_from is None:
        parser.error("--source-config and --weights-from are required for matched TES profiles")
    if args.variant != "stock":
        parser.error("TES experience filtering uses the stock learner; omit non-stock --variant")
    if args.update_epochs != 1:
        parser.error("This matched TES profiling wrapper requires --update-epochs 1")
    if (args.sac_filter_score != "none" or args.sac_keep_fraction != 1.0
            or args.sac_filter_mode != "fraction" or args.sac_filter_ewma_beta != 0.25
            or args.sac_filter_threshold_scale != 0.01):
        parser.error("use train.experience_filter from --source-config, not profiling-only SAC filters")
    if wrapper_args.restore_filter_state and wrapper_args.case != "filter":
        parser.error("--restore-filter-state is compatible only with --case filter")
    args.tes_filter_case = wrapper_args.case
    args.tes_filter_detailed = wrapper_args.detailed
    args.tes_restore_filter_state = wrapper_args.restore_filter_state
    args.tes_restore_update_state = wrapper_args.restore_update_state
    return harness, args


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SynchronizedWallTimers:
    """Nested inclusive/exclusive spans; CUDA sync wait is reported separately."""

    def __init__(self, torch, cuda: bool):
        self.torch = torch
        self.cuda = cuda
        self.samples: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.stack: list[dict[str, float]] = []
        self.synchronize_wall_seconds = 0.0
        self.call_count = 0

    def _sync(self):
        if not self.cuda:
            return
        started = time.perf_counter()
        self.torch.cuda.synchronize()
        self.synchronize_wall_seconds += time.perf_counter() - started

    def measure(self, name: str, function: Callable):
        self._sync()
        started = time.perf_counter()
        frame = {"children_seconds": 0.0}
        self.stack.append(frame)
        self.call_count += 1
        try:
            return function()
        finally:
            self._sync()
            inclusive = time.perf_counter() - started
            self.stack.pop()
            exclusive = max(0.0, inclusive - frame["children_seconds"])
            self.samples[name].append({"inclusive_seconds": inclusive,
                                       "exclusive_seconds": exclusive})
            if self.stack:
                self.stack[-1]["children_seconds"] += inclusive

    def snapshot(self):
        return {
            "semantics": (
                "inclusive spans include nested instrumented calls; exclusive spans subtract "
                "nested instrumented spans; do not sum inclusive values"
            ),
            "cuda_synchronize_wait_seconds": self.synchronize_wall_seconds,
            "synchronized_method_calls": self.call_count,
            "methods": {
                name: {
                    "calls": len(samples),
                    "inclusive_total_seconds": sum(row["inclusive_seconds"] for row in samples),
                    "exclusive_total_seconds": sum(row["exclusive_seconds"] for row in samples),
                    "samples": samples,
                }
                for name, samples in sorted(self.samples.items())
            },
        }


def instrument_method(trainer, name: str, timers: SynchronizedWallTimers,
                      counter: str | None = None, count_argument: int = 0):
    original = getattr(trainer, name, None)
    if not callable(original):
        return False

    def wrapped(*args, **kwargs):
        if counter is not None:
            observations = args[count_argument] if len(args) > count_argument else None
            if observations is not None and hasattr(observations, "shape"):
                trainer._tes_probe_counts[counter] += int(observations.shape[0])
        current_timers = trainer._tes_probe_timers
        return current_timers.measure(name, lambda: original(*args, **kwargs))

    setattr(trainer, name, wrapped)
    return True


def atomic_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def instrument_trainer(harness, args, trainer, torch, rank: int, world_size: int):
    trainer._tes_probe = {
        "rank": rank,
        "world_size": world_size,
        "output_dir": args.output_dir,
        "cycle_index": -1,
        "detailed": args.tes_filter_detailed,
        "evaluation_seconds": None,
        "evaluation_metrics": None,
        "weight_initialization": None,
    }
    trainer._tes_probe_counts = {}
    trainer._tes_probe_timers = SynchronizedWallTimers(torch, args.device == "cuda")

    def reset_cycle(cycle_index):
        trainer._tes_probe["cycle_index"] = cycle_index
        trainer._tes_probe["evaluation_seconds"] = None
        trainer._tes_probe["evaluation_metrics"] = None
        trainer._tes_probe_counts = {
            "unfiltered_critic_loss_rows": 0,
            "filtered_critic_loss_sum_rows": 0,
            "policy_loss_rows": 0,
        }
        trainer._tes_probe_timers = SynchronizedWallTimers(torch, args.device == "cuda")

    original_evaluate = trainer.evaluate

    def evaluate(*call_args, **kwargs):
        cycle_index = trainer._tes_probe["cycle_index"] + 1
        reset_cycle(cycle_index)
        started = time.perf_counter()
        result = original_evaluate(*call_args, **kwargs)
        trainer._tes_probe["evaluation_seconds"] = time.perf_counter() - started
        trainer._tes_probe["evaluation_metrics"] = json_safe(result)
        return result

    trainer.evaluate = evaluate

    for method_name, label, counter in (
        ("_critic_loss", "unfiltered_critic_loss_rows", "unfiltered_critic_loss_rows"),
        ("_critic_loss_sum", "filtered_critic_loss_sum_rows", "filtered_critic_loss_sum_rows"),
        ("_policy_loss_rows", "policy_loss_rows", "policy_loss_rows"),
    ):
        original = getattr(trainer, method_name, None)
        if callable(original):
            def count_rows(*call_args, _original=original, _counter=counter, **kwargs):
                observations = call_args[0] if call_args else None
                if (observations is not None and hasattr(observations, "shape")
                        and kwargs.get("record_diagnostics", True)):
                    row_count = int(observations.shape[0])
                    trainer._tes_probe_counts[_counter] += row_count
                    if _counter == "filtered_critic_loss_sum_rows":
                        trainer.critic_loss_rows += row_count
                    elif (_counter == "policy_loss_rows"
                          and getattr(trainer, "experience_filter", None) is not None):
                        trainer.policy_loss_rows += row_count
                return _original(*call_args, **kwargs)
            setattr(trainer, method_name, count_rows)

    if args.tes_filter_detailed:
        timers = trainer._tes_probe_timers
        for method_name in (
            "_compute_targets", "_prepare_n_step_targets",
            "_score_experience_filter_candidates", "_train_filtered_critic",
            "_train_filtered_actor", "_critic_loss", "_policy_loss",
        ):
            instrument_method(trainer, method_name, timers)
        for optimizer_name, optimizer in (
            ("actor", trainer.optimizer), ("critic", trainer.critic_optimizer),
            ("alpha", trainer.alpha_optimizer),
        ):
            original_step = optimizer.step

            def timed_step(*call_args, _original=original_step, _name=optimizer_name, **kwargs):
                return trainer._tes_probe_timers.measure(
                    f"optimizer.{_name}.step",
                    lambda: _original(*call_args, **kwargs),
                )

            optimizer.step = timed_step

        experience_filter = getattr(trainer, "experience_filter", None)
        if experience_filter is not None:
            original_select = experience_filter.select

            def timed_select(*call_args, **kwargs):
                return trainer._tes_probe_timers.measure(
                    "experience_filter.select",
                    lambda: original_select(*call_args, **kwargs),
                )

            experience_filter.select = timed_select

    original_train = trainer.train

    def train(*call_args, **kwargs):
        started = time.perf_counter()
        train_logs = original_train(*call_args, **kwargs)
        training_seconds = time.perf_counter() - started
        experience_filter = getattr(trainer, "experience_filter", None)
        filter_metrics = experience_filter.metrics() if experience_filter is not None else None
        record = {
            "schema_version": 1,
            "case": args.tes_filter_case,
            "detailed": args.tes_filter_detailed,
            "cycle_index": trainer._tes_probe["cycle_index"],
            "rank": rank,
            "world_size": world_size,
            "device": args.device,
            "evaluation_seconds_wall_unsynchronized": trainer._tes_probe["evaluation_seconds"],
            "evaluation_metrics": trainer._tes_probe["evaluation_metrics"],
            "training_seconds_wall_unsynchronized": training_seconds,
            "train_logs": train_logs,
            "filter_metrics": filter_metrics,
            "row_exposures_local": dict(trainer._tes_probe_counts),
            "weight_initialization": trainer._tes_probe["weight_initialization"],
            "detailed_timing": (
                trainer._tes_probe_timers.snapshot() if args.tes_filter_detailed else None
            ),
            "timing_overhead_note": (
                "Detailed runs synchronize CUDA around each named method; compare them only "
                "with a separate plain run. Synchronization wait is reported separately, "
                "and nested inclusive spans overlap. Backward is not individually instrumented."
                if args.tes_filter_detailed else
                "Plain run adds no method-level CUDA synchronization."
            ),
        }
        cycle_index = trainer._tes_probe["cycle_index"]
        harness.tes_pending_records.append((
            args.output_dir / f"tes_filter_cycle_{cycle_index:03d}_rank_{rank:03d}.json", record,
        ))
        return train_logs

    trainer.train = train


def install_wrappers(harness, args):
    harness.tes_pending_records = []
    original_filter_metrics = harness.filter_cycle_metrics

    def filter_cycle_metrics(trainer):
        result = original_filter_metrics(trainer)
        experience_filter = getattr(trainer, "experience_filter", None)
        if experience_filter is not None:
            result.update(
                candidate_rows=len(trainer.replay) * trainer.config["update_epochs"],
                retained_rows=trainer._tes_probe_counts["filtered_critic_loss_sum_rows"],
                filter_current_max=experience_filter.last_current_max,
                filter_ema_max=experience_filter.ema_max,
                filter_threshold=experience_filter.last_threshold,
            )
        return result

    harness.filter_cycle_metrics = filter_cycle_metrics
    original_prepare_config = harness.prepare_config

    def prepare_config(profile_args, output_dir):
        config, source_path = original_prepare_config(profile_args, output_dir)
        train = config["train"]
        filter_config = train.get("experience_filter")
        if profile_args.tes_filter_case == "baseline":
            train["experience_filter"] = None
        else:
            if not isinstance(filter_config, dict) or filter_config.get("enabled", True) is not True:
                raise harness.ProfileSetupError(
                    "filter and all_keep cases require an enabled train.experience_filter "
                    "in --source-config"
                )
            if filter_config.get("apply_to", "actor_critic") != "actor_critic":
                raise harness.ProfileSetupError("This probe requires shared actor_critic filtering")
            if profile_args.tes_filter_case == "all_keep":
                filter_config["threshold_scale"] = 0.0
        return config, source_path

    harness.prepare_config = prepare_config

    original_profile_settings = harness.profile_settings

    def profile_settings(profile_args):
        settings = original_profile_settings(profile_args)
        settings.update({
            "tes_filter_case": profile_args.tes_filter_case,
            "detailed_synchronized_timers": profile_args.tes_filter_detailed,
            "restore_filter_state": profile_args.tes_restore_filter_state,
            "alpha_restored_from_weights": True,
            "restore_update_state": profile_args.tes_restore_update_state,
        })
        return settings

    harness.profile_settings = profile_settings

    original_construct_trainer = harness.construct_trainer

    def construct_trainer(profile_args, config, output_dir, torch, ppo, rank, local_rank, world_size):
        result = original_construct_trainer(
            profile_args, config, output_dir, torch, ppo, rank, local_rank, world_size,
        )
        trainer = result[0]
        instrument_trainer(harness, profile_args, trainer, torch, rank, world_size)
        return result

    harness.construct_trainer = construct_trainer

    def load_profile_weights(trainer, algorithm, weights_path, torch, ppo):
        if algorithm != "sac":
            raise harness.ProfileSetupError("TES filter probe accepts SAC weights only")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if (not isinstance(state, dict) or state.get("algorithm") != "sac"
                or state.get("scope") != "learner_only"):
            raise harness.ProfileSetupError(
                f"SAC weights must be learner-only trainer_state.pt: {weights_path}"
            )
        source_to_module = (
            ("policy_state_dict", trainer.uncompiled_policy),
            ("critic_state_dict", trainer.uncompiled_critic),
            ("target_critic_state_dict", trainer.uncompiled_target_critic),
        )
        for state_key, module in source_to_module:
            model_state = state.get(state_key)
            if not isinstance(model_state, dict):
                raise harness.ProfileSetupError(f"checkpoint is missing {state_key}: {weights_path}")
            module.load_state_dict(model_state, strict=True)
        log_alpha = state.get("log_alpha")
        if not isinstance(log_alpha, torch.Tensor):
            raise harness.ProfileSetupError(f"checkpoint is missing log_alpha: {weights_path}")
        with torch.no_grad():
            trainer.log_alpha.copy_(log_alpha.to(device=trainer.device, dtype=trainer.log_alpha.dtype))
        if args.tes_restore_update_state:
            for name in ("optimizer", "critic_optimizer", "alpha_optimizer",
                         "scheduler", "critic_scheduler", "alpha_scheduler"):
                getattr(trainer, name).load_state_dict(state[f"{name}_state_dict"])
            controller = getattr(trainer, "target_entropy_controller", None)
            if controller is not None:
                controller.load_state_dict(state["target_entropy_controller_state_dict"])
                # Probe counters start at zero; preserve elapsed force-decay time
                # by forwarding the checkpoint's step offset to the controller.
                original_observe = controller.observe
                offset = int(state["agent_steps"])

                def observe(entropy, agent_steps=None):
                    return original_observe(entropy, agent_steps=agent_steps + offset)

                controller.observe = observe
        filter_state_restored = False
        if args.tes_restore_filter_state:
            experience_filter = getattr(trainer, "experience_filter", None)
            filter_state = state.get("experience_filter_state_dict")
            if experience_filter is None or filter_state is None:
                raise harness.ProfileSetupError(
                    "--restore-filter-state requires an enabled filter and checkpoint filter state"
                )
            validated_state = experience_filter.validate_state_dict(filter_state)
            experience_filter.load_state_dict(validated_state)
            filter_state_restored = True
        initialization = {
            "source_path": str(weights_path),
            "source_sha256": harness.sha256_file(weights_path),
            "source_format": "sac_learner_only_trainer_state",
            "model_state_keys_loaded": [key for key, _module in source_to_module],
            "optimizer_state_restored": args.tes_restore_update_state,
            "scheduler_state_restored": args.tes_restore_update_state,
            "entropy_controller_state_restored": args.tes_restore_update_state,
            "training_counters_restored": False,
            "rng_state_restored": False,
            "replay_restored": False,
            "log_alpha_restored": True,
            "experience_filter_state_restored": filter_state_restored,
            "initialization": (
                "policy, online/target critics, and log_alpha loaded; counters, RNG and replay "
                "are fresh; optimizer/scheduler/controller restoration is specified by flags"
            ),
        }
        trainer._tes_probe["weight_initialization"] = initialization
        del state
        return initialization

    harness.load_profile_weights = load_profile_weights


def main(argv: list[str] | None = None) -> int:
    harness, args = parse_cli(argv)
    install_wrappers(harness, args)
    try:
        harness.run_benchmark(args)
    finally:
        for path, record in harness.tes_pending_records:
            atomic_json(path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
