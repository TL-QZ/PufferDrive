"""Coordinate native PPO critic pretraining and an in-process SAC handoff."""

from contextlib import contextmanager
from copy import deepcopy
import argparse
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

import numpy as np
import torch
import yaml


REPLAY_FIELDS = 5
PPO_STORAGE_FIELDS = (
    "observations", "actions", "values", "logprobs", "rewards", "terminals",
    "truncations", "ratio", "importance", "masks", "ep_lengths", "ep_indices",
)
SAC_ISOLATION_SEED_SALT = 0x5AC0_2026
REPLAY_SEED_SALT = 0x5EED_2026


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config, world_size):
    """Reject unsupported modes before allocating a vector environment or replay."""
    if not isinstance(config, dict):
        raise ValueError("Hybrid config must be a mapping")
    for key in ("ppo", "sac", "warm_start", "provenance"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Hybrid config requires a {key!r} mapping")

    expected_world_size = config["provenance"].get("expected_world_size")
    _positive_integer(expected_world_size, "provenance.expected_world_size")
    if world_size != expected_world_size:
        raise ValueError(
            f"Configured world size is {expected_world_size}, runtime world size is {world_size}"
        )

    ppo_args = config["ppo"]
    sac_args = config["sac"]
    warm_args = config["warm_start"]
    for phase, args in (("ppo", ppo_args), ("sac", sac_args)):
        if not isinstance(args.get("train"), dict):
            raise ValueError(f"{phase}.train must be a mapping")
        train_args = args["train"]
        if train_args.get("resume_state_path") is not None:
            raise ValueError(f"{phase} resume is unsupported; start from a new run directory")
        if args.get("load_model_path") is not None or args.get("load_id") is not None:
            raise ValueError(f"{phase} actor warm starts are unsupported")
        _positive_integer(train_args.get("total_timesteps"), f"{phase}.train.total_timesteps")

    handoff_steps = warm_args.get("handoff_steps")
    sac_steps = warm_args.get("sac_steps")
    _positive_integer(handoff_steps, "warm_start.handoff_steps")
    _positive_integer(sac_steps, "warm_start.sac_steps")
    if ppo_args["train"]["total_timesteps"] < handoff_steps:
        raise ValueError("PPO schedule horizon must reach the configured handoff threshold")
    if ppo_args.get("rnn_name") is not None or ppo_args["train"].get("use_rnn", False):
        raise NotImplementedError("The hybrid coordinator requires feed-forward PPO")
    if sac_args.get("rnn_name") is not None:
        raise NotImplementedError("The hybrid coordinator requires feed-forward SAC")
    if sac_args.get("policy_name") != "DriveSAC":
        raise ValueError("SAC policy_name must be 'DriveSAC'")
    if ppo_args.get("policy", {}).get("action_type") != "discrete":
        raise ValueError("The hybrid coordinator requires a discrete PPO actor")
    if sac_args.get("policy", {}).get("action_type") != "discrete":
        raise ValueError("The hybrid coordinator requires a discrete SAC actor")
    if ppo_args.get("env") != sac_args.get("env"):
        raise ValueError("PPO and SAC environment configurations must match")
    if ppo_args.get("policy") != sac_args.get("policy"):
        raise ValueError("PPO and SAC policy configurations must match for actor transfer")
    if not ppo_args["env"].get("capture_final_observations", False):
        raise ValueError("PPO collection requires env.capture_final_observations=True")
    if not sac_args["env"].get("capture_final_observations", False):
        raise ValueError("SAC collection requires env.capture_final_observations=True")
    if sac_args["train"].get("learn_alpha", True):
        raise ValueError("Hybrid SAC requires fixed temperature: train.learn_alpha=false")
    if not math.isclose(float(sac_args["train"].get("initial_alpha", -1)), 0.1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Hybrid SAC requires fixed alpha 0.1")
    if sac_args["train"].get("anneal_lr", False):
        raise ValueError("Hybrid SAC requires non-annealed actor and critic learning rates")

    run_dir = warm_args.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("warm_start.run_dir must be a non-empty path")
    return Path(run_dir).expanduser().resolve()


def _rank_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _dist_barrier():
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.barrier()


def _sum_across_ranks(value, device):
    total = torch.tensor(int(value), dtype=torch.int64, device=device)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    return int(total.item())


def _all_ranks_true(value, device):
    ready = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
    return bool(ready.item())


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IsolatedRNG:
    """Swap complete process RNG state around SAC-only work."""

    def __init__(self, pufferl_module, seed):
        self.pufferl = pufferl_module
        self.seed = int(seed)
        self.state = None

    @contextmanager
    def activate(self):
        caller_state = self.pufferl.capture_rng_state()
        try:
            if self.state is None:
                _seed_all(self.seed)
            else:
                self.pufferl.restore_rng_state({"rng_state": self.state})
            yield
        finally:
            self.state = self.pufferl.capture_rng_state()
            self.pufferl.restore_rng_state({"rng_state": caller_state})


class ReplayRefresh:
    """Fill replay, then reservoir-sample each full-cycle refresh in bounded memory."""

    def __init__(self, replay, capacity, quota, generator):
        _positive_integer(capacity, "replay capacity")
        _positive_integer(quota, "replay refresh quota")
        if quota > capacity:
            raise ValueError("Replay refresh quota cannot exceed replay capacity")
        self.replay = replay
        self.capacity = capacity
        self.quota = quota
        self.generator = generator
        self.refreshing = False
        self.destination_slots = None
        self.candidates_seen = 0
        self.cycle_start_size = len(replay)

    def begin_cycle(self, all_ranks_full):
        self.refreshing = bool(all_ranks_full)
        self.destination_slots = None
        self.candidates_seen = 0
        self.cycle_start_size = len(self.replay)
        if self.refreshing:
            self.destination_slots = torch.randperm(
                self.capacity, generator=self.generator, device="cpu",
            )[:self.quota]

    @staticmethod
    def _select_rows(transitions, rows):
        return tuple(field.index_select(0, rows.to(field.device)) for field in transitions)

    def add(self, transitions):
        if len(transitions) != REPLAY_FIELDS:
            raise ValueError(f"Replay transition requires exactly {REPLAY_FIELDS} fields")
        candidate_count = int(transitions[0].shape[0])
        if any(field.shape[0] != candidate_count for field in transitions):
            raise ValueError("Replay transition fields must have matching row counts")
        if candidate_count == 0:
            return 0

        if not self.refreshing:
            remaining = self.capacity - len(self.replay)
            insert_count = min(candidate_count, max(remaining, 0))
            if insert_count == 0:
                return 0
            if insert_count == candidate_count:
                selected_rows = torch.arange(candidate_count, dtype=torch.long)
            else:
                selected_rows = torch.randperm(
                    candidate_count, generator=self.generator, device="cpu",
                )[:insert_count].sort().values
            self.replay.add(*self._select_rows(transitions, selected_rows))
            return insert_count

        # Standard online reservoir sampling. The one transient vector is bounded by
        # the current recv batch; transition observations are indexed only for writes.
        first_candidate = self.candidates_seen
        candidate_numbers = torch.arange(
            first_candidate + 1, first_candidate + candidate_count + 1,
            dtype=torch.float64,
        )
        random_values = torch.rand(
            candidate_count, generator=self.generator, dtype=torch.float64,
        )
        selected_slots = torch.floor(random_values * candidate_numbers).to(torch.long)
        initial_fill = candidate_numbers <= self.quota
        selected_slots[initial_fill] = candidate_numbers[initial_fill].to(torch.long) - 1
        eligible_rows = torch.nonzero(selected_slots < self.quota, as_tuple=False).flatten()
        self.candidates_seen += candidate_count
        if eligible_rows.numel() == 0:
            return 0

        eligible_slots = selected_slots[eligible_rows]
        # A batch can select one reservoir slot more than once. Keep the latest
        # sequential write for each slot so vectorized assignment matches the stream.
        sort_key = eligible_slots * candidate_count + eligible_rows
        order = torch.argsort(sort_key)
        ordered_slots = eligible_slots[order]
        ordered_rows = eligible_rows[order]
        keep_last = torch.ones(ordered_slots.numel(), dtype=torch.bool)
        keep_last[:-1] = ordered_slots[:-1] != ordered_slots[1:]
        destination_rows = self.destination_slots[ordered_slots[keep_last]]
        source_rows = ordered_rows[keep_last]
        self.replay.replace(
            destination_rows,
            *self._select_rows(transitions, source_rows),
        )
        return int(destination_rows.numel())

    def finish_cycle(self):
        if self.refreshing:
            return min(self.quota, self.candidates_seen)
        return len(self.replay) - self.cycle_start_size


def transfer_actor_and_verify(ppo_policy, sac_policy, observations, *, atol=1e-6, rtol=1e-5):
    """Strictly copy the compatible Drive actor and compare action probabilities."""
    if getattr(ppo_policy, "is_continuous", True):
        raise ValueError("Actor transfer requires a discrete PPO actor")
    if getattr(sac_policy, "action_count", None) != sum(ppo_policy.atn_dim):
        raise ValueError("PPO and SAC actor action counts do not match")
    sac_policy.actor_backbone.load_state_dict(ppo_policy.actor_backbone.state_dict(), strict=True)
    sac_policy.actor_head.load_state_dict(ppo_policy.actor_head.state_dict(), strict=True)

    device = next(sac_policy.parameters()).device
    sample_observations = torch.as_tensor(observations, device=device, dtype=torch.float32)
    if sample_observations.ndim == 1:
        sample_observations = sample_observations.unsqueeze(0)
    if sample_observations.shape[0] == 0:
        raise ValueError("Actor probability verification requires at least one observation")
    with torch.no_grad():
        ppo_hidden = ppo_policy.actor_backbone(sample_observations, ppo_policy.ego_dim)
        ppo_logits = ppo_policy.actor_head(ppo_hidden).float()
        sac_logits = sac_policy(sample_observations).float()
        ppo_probabilities = torch.softmax(ppo_logits, dim=-1)
        sac_probabilities = torch.softmax(sac_logits, dim=-1)
    if not torch.allclose(ppo_probabilities, sac_probabilities, atol=atol, rtol=rtol):
        max_error = (ppo_probabilities - sac_probabilities).abs().max().item()
        raise RuntimeError(f"Transferred actor probability mismatch; maximum absolute error {max_error}")
    return float((ppo_probabilities - sac_probabilities).abs().max().item())


class _PhaseLogger:
    def __init__(self, logger, prefix, *, preserve_environment=False):
        self._logger = logger
        self._prefix = prefix.rstrip("/")
        self._preserve_environment = preserve_environment

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def log(self, logs, step):
        phase_logs = {}
        for key, value in logs.items():
            if self._preserve_environment and (key.startswith("environment/") or key == "agent_steps"):
                phase_logs[key] = value
            else:
                phase_logs[f"{self._prefix}/{key}"] = value
        self._logger.log(phase_logs, step)


class WarmStartCoordinator:
    """Run full native PPO cycles and critic-only SAC updates over one vecenv."""

    def __init__(
        self,
        ppo_learner,
        sac_learner,
        *,
        handoff_steps,
        sac_steps,
        ppo_args=None,
        sac_args=None,
        logger=None,
        sac_rng=None,
        replay_generator=None,
    ):
        _positive_integer(handoff_steps, "handoff_steps")
        _positive_integer(sac_steps, "sac_steps")
        self.ppo = ppo_learner
        self.sac = sac_learner
        self.handoff_steps = handoff_steps
        self.sac_steps = sac_steps
        self.ppo_args = ppo_args
        self.sac_args = sac_args
        self.logger = logger if logger is not None else ppo_learner.logger
        if sac_rng is None:
            from pufferlib import pufferl

            sac_rng = IsolatedRNG(pufferl, 0)
        self.sac_rng = sac_rng
        self.replay_generator = replay_generator or torch.Generator(device="cpu")
        self.refresh = ReplayRefresh(
            sac_learner.replay,
            sac_learner.batch_size,
            sac_learner.refresh_count,
            self.replay_generator,
        )
        self.phase = "ppo_critic_pretrain"
        self.ppo_cycles = 0
        self.sac_phase_steps = 0
        self.cumulative_agent_steps = 0
        self.critic_phases = 0
        self.last_cycle_insert_count = 0
        self.last_cycle_candidate_count = 0
        self.last_cycle_critic_updates = 0
        self.boundary_agent_steps = None
        self._sac_checkpoint_interval = sac_learner.config.get("checkpoint_interval")
        self._target_policy = self._make_target_policy()

    @property
    def device(self):
        return self.sac.device

    def _make_target_policy(self):
        ppo_policy = self.ppo.uncompiled_policy

        def target_policy(observations):
            hidden = ppo_policy.actor_backbone(observations, ppo_policy.ego_dim)
            return ppo_policy.actor_head(hidden)

        return target_policy

    def observe_transition(
        self, *, observations, rewards, terminals, truncations, infos, env_ids, masks, actions,
    ):
        transitions, cleaned_infos = self.sac.observe_transition(
            observations=observations,
            rewards=rewards,
            terminals=terminals,
            truncations=truncations,
            infos=infos,
            env_ids=env_ids,
            masks=masks,
        )
        self.last_cycle_candidate_count += int(transitions[0].shape[0])
        self.refresh.add(transitions)
        self.sac.record_actions(observations, actions, env_ids)
        return cleaned_infos

    def _all_ranks_replay_full(self):
        return _all_ranks_true(len(self.sac.replay) == self.sac.batch_size, self.device)

    def _set_sac_step_counters(self, cumulative_agent_steps):
        self.sac.global_step = self.ppo.global_step
        self.sac.agent_steps = cumulative_agent_steps

    def run_ppo_cycle(self):
        if self.phase != "ppo_critic_pretrain":
            raise RuntimeError("PPO cycles can only run before actor handoff")
        self.refresh.begin_cycle(self._all_ranks_replay_full())
        self.last_cycle_candidate_count = 0
        if torch.device(self.ppo.config["device"]).type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        self.ppo.evaluate()
        if torch.device(self.ppo.config["device"]).type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        self.ppo.train()
        self.ppo_cycles += 1

        self.cumulative_agent_steps = _sum_across_ranks(self.ppo.global_step, self.device)
        self._set_sac_step_counters(self.cumulative_agent_steps)
        self.last_cycle_insert_count = self.refresh.finish_cycle()
        global_insert_count = _sum_across_ranks(self.last_cycle_insert_count, self.device)
        self.sac.valid_inserts += global_insert_count
        replay_ready = self._all_ranks_replay_full()
        self.last_cycle_critic_updates = 0
        if replay_ready:
            starting_updates = self.sac.learner_updates
            with self.sac_rng.activate():
                self.sac.train(update_policy=False, target_policy=self._target_policy)
            self.last_cycle_critic_updates = self.sac.learner_updates - starting_updates
            self.critic_phases += 1

        return {
            "ppo_cycle": self.ppo_cycles,
            "cumulative_agent_steps": self.cumulative_agent_steps,
            "replay_ready": replay_ready,
            "replay_size": len(self.sac.replay),
            "valid_replay_inserts": global_insert_count,
            "refresh_candidates": _sum_across_ranks(self.last_cycle_candidate_count, self.device),
            "critic_updates": _sum_across_ranks(self.last_cycle_critic_updates, self.device),
        }

    def handoff(self, observations=None):
        if self.phase != "ppo_critic_pretrain":
            raise RuntimeError(f"Cannot hand off from phase {self.phase!r}")
        if self.cumulative_agent_steps < self.handoff_steps:
            raise RuntimeError("Handoff threshold has not been reached")
        if self.last_cycle_critic_updates <= 0 or not self._all_ranks_replay_full():
            raise RuntimeError("Handoff requires a complete PPO cycle followed by a critic phase")
        if observations is None:
            verification_count = min(32, len(self.sac.replay))
            observations = self.sac.replay.observations[:verification_count]
        transfer_error = transfer_actor_and_verify(
            self.ppo.uncompiled_policy, self.sac.uncompiled_policy, observations,
        )
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            for parameter in self.sac.uncompiled_policy.parameters():
                torch.distributed.broadcast(parameter.data, src=0)

        self.boundary_agent_steps = self.cumulative_agent_steps
        self.sac_phase_steps = 0
        self.sac.config["total_timesteps"] = self.boundary_agent_steps + self.sac_steps
        self.sac.config["checkpoint_interval"] = self._sac_checkpoint_interval
        self._set_sac_step_counters(self.boundary_agent_steps)
        self.phase = "sac"
        return {
            "boundary_agent_steps": self.boundary_agent_steps,
            "actor_probability_max_abs_error": transfer_error,
            "replay_size": len(self.sac.replay),
        }


def _write_sidecar(path, state):
    path = Path(path)
    temporary_path = Path(str(path) + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def _make_logger(ppo_module, ppo_args, sac_args, rank):
    if rank != 0:
        return ppo_module.NoLogger(ppo_args)
    if ppo_args.get("neptune"):
        from pufferlib.sac_logging import SACNeptuneLogger

        return SACNeptuneLogger(sac_args)
    if ppo_args.get("wandb"):
        return ppo_module.WandbLogger(ppo_args)
    if ppo_args.get("tb"):
        return ppo_module.TensorBoardLogger(
            run_id=ppo_args["run_name"], experiment_dir=ppo_args["train"]["data_dir"],
        )
    return ppo_module.NoLogger(ppo_args)


def _prepare_run_directories(config, rank, world_size, ppo_module):
    run_dir = Path(config["warm_start"]["run_dir"]).expanduser().resolve()
    ppo_dir = run_dir / "ppo"
    sac_dir = run_dir / "sac"
    error_message = None
    if rank == 0:
        try:
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
            if ppo_dir.exists() or sac_dir.exists():
                raise FileExistsError(
                    f"Refusing to overwrite hybrid phase directories under {run_dir}"
                )
            if (run_dir / "hybrid_boundary_state.pt").exists():
                raise FileExistsError(f"Refusing to overwrite hybrid artifact in {run_dir}")
            ppo_dir.mkdir()
            sac_dir.mkdir()
        except BaseException as error:
            error_message = f"{type(error).__name__}: {error}"

    if world_size > 1:
        payload = [error_message]
        torch.distributed.broadcast_object_list(payload, src=0)
        error_message = payload[0]
    if error_message:
        raise RuntimeError(error_message)
    _dist_barrier()

    ppo_args = deepcopy(config["ppo"])
    sac_args = deepcopy(config["sac"])
    ppo_args["train"]["data_dir"] = str(ppo_dir)
    sac_args["train"]["data_dir"] = str(sac_dir)
    ppo_args["train"]["resume_state_path"] = None
    sac_args["train"]["resume_state_path"] = None
    if rank == 0:
        ppo_module._save_experiment_config(ppo_args, str(ppo_dir))
        ppo_module._save_experiment_config(sac_args, str(sac_dir))
    _dist_barrier()
    return run_dir, ppo_dir, sac_dir, ppo_args, sac_args


def _make_policy_ddp(policy, device):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return policy
    device = torch.device(device)
    if device.type == "cuda":
        local_rank = torch.cuda.current_device()
        policy = policy.to(local_rank)
        wrapped = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank,
        )
    else:
        wrapped = torch.nn.parallel.DistributedDataParallel(policy)
    wrapped.forward_eval = policy.forward_eval
    return wrapped


def _make_sac_rng_seed(train_seed, rank):
    return int(np.random.SeedSequence([int(train_seed), rank, SAC_ISOLATION_SEED_SALT]).generate_state(1)[0])


def _make_replay_seed(train_seed, rank):
    return int(np.random.SeedSequence([int(train_seed), rank, REPLAY_SEED_SALT]).generate_state(1)[0])


def _run_evaluation(ppo_module, args, actor, logger, epoch, global_step, run_dir, *, force=False):
    if not force and args["train"].get("evaluation_interval_epochs") is None:
        return False
    ppo_module.run_training_evaluation(
        env_name="puffer_drive",
        args=args,
        policy=actor,
        logger=logger,
        epoch=epoch,
        global_step=global_step,
        run_dir=str(run_dir),
    )
    return True


def _release_ppo(ppo_learner):
    if ppo_learner is None:
        return
    utilization = getattr(ppo_learner, "utilization", None)
    if utilization is not None:
        utilization.stop()
        utilization.join()
    for field in PPO_STORAGE_FIELDS:
        if hasattr(ppo_learner, field):
            setattr(ppo_learner, field, None)
    for field in ("lstm_h", "lstm_c"):
        if hasattr(ppo_learner, field):
            setattr(ppo_learner, field, None)
    for field in ("optimizer", "scheduler", "policy", "uncompiled_policy"):
        if hasattr(ppo_learner, field):
            setattr(ppo_learner, field, None)


def _cleanup_failed_run(vecenv, ppo_learner, sac_learner, process_group_owned):
    for learner in (ppo_learner, sac_learner):
        if learner is None:
            continue
        utilization = getattr(learner, "utilization", None)
        if utilization is not None:
            utilization.stop()
            utilization.join()
    if vecenv is not None:
        try:
            vecenv.close()
        except Exception:
            pass
    if process_group_owned and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _finish_logger_after_failure(logger):
    if logger is None:
        return
    if hasattr(logger, "finish"):
        logger.finish()
    elif hasattr(logger, "neptune"):
        logger.neptune.stop()
    elif hasattr(logger, "local_writer"):
        logger.local_writer.close()


def train(
    config,
    *,
    vecenv=None,
    ppo_policy=None,
    sac_policy=None,
    critic=None,
    replay=None,
    logger=None,
    ppo_learner=None,
    sac_learner=None,
):
    """Run the PPO-to-SAC coordinator using native learner operations."""
    from pufferlib import pufferl
    from pufferlib import pufferl_sac

    process_group_owned = False
    ppo_agent = ppo_learner
    sac_agent = sac_learner
    shared_vecenv = vecenv
    logger_to_close = logger
    run_dir = Path(config["warm_start"]["run_dir"]).expanduser().resolve()
    success = False
    try:
        owns_group, resolved_device, world_size, rank = pufferl_sac._initialize_process_group(
            config["ppo"]["train"]["device"], pufferl,
        )
        process_group_owned = owns_group
        validate_config(config, world_size)
        run_dir, ppo_dir, sac_dir, ppo_args, sac_args = _prepare_run_directories(
            config, rank, world_size, pufferl,
        )
        ppo_args["train"]["device"] = resolved_device
        sac_args["train"]["device"] = resolved_device
        if "LOCAL_RANK" in os.environ:
            ppo_args["train"]["device"] = resolved_device

        ppo_train_args = ppo_args["train"]
        global_ppo_schedule = ppo_train_args["total_timesteps"]
        ppo_train_args["total_timesteps"] //= world_size
        if ppo_train_args["total_timesteps"] <= 0:
            raise ValueError("PPO schedule horizon is smaller than the world size")
        train_seed = ppo_train_args.get("seed")
        if train_seed is None:
            train_seed = 0
            ppo_train_args["seed"] = train_seed
        torch_seed, env_seed = pufferl.derive_rank_seeds(
            ppo_args["vec"].get("seed"), train_seed, world_size, rank,
        )
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        ppo_args["train"]["data_dir"] = str(ppo_dir)
        sac_args["train"]["data_dir"] = str(sac_dir)
        ppo_args["train"]["total_timesteps"] = global_ppo_schedule // world_size
        ppo_args["git"] = pufferl._get_git_metadata()
        sac_args["git"] = ppo_args["git"]
        logger_to_close = logger or _make_logger(pufferl, ppo_args, sac_args, rank)

        if shared_vecenv is None:
            shared_vecenv = pufferl.load_env("puffer_drive", ppo_args, seed=env_seed)
        driver_env = getattr(shared_vecenv, "driver_env", None)
        if not getattr(driver_env, "capture_final_observations", False):
            raise ValueError("Hybrid training requires Drive final-observation transport")

        if ppo_policy is None:
            ppo_policy = pufferl.load_policy(ppo_args, shared_vecenv, "puffer_drive")
        ppo_policy = _make_policy_ddp(ppo_policy, resolved_device)
        ppo_train_config = dict(
            **ppo_args["train"], env="puffer_drive", eval=ppo_args.get("eval", {}),
            run_name=ppo_args["run_name"],
        )
        if ppo_agent is None:
            ppo_agent = pufferl.PuffeRL(
                ppo_train_config, shared_vecenv, ppo_policy, logger_to_close,
            )
        if ppo_agent.config.get("checkpoint_interval") is None:
            ppo_agent.config["checkpoint_interval"] = (
                ppo_agent.config["total_timesteps"] // ppo_agent.config["batch_size"] + 1
            )

        train_seed = sac_args["train"].get("seed", train_seed)
        sac_seed = _make_sac_rng_seed(train_seed, rank)
        sac_rng = IsolatedRNG(pufferl, sac_seed)
        replay_rng = torch.Generator(device="cpu").manual_seed(_make_replay_seed(train_seed, rank))
        sac_train = sac_args["train"]
        sac_checkpoint_interval = sac_train.get("checkpoint_interval")
        continuation_steps = int(config["warm_start"]["sac_steps"])
        initial_sac_budget = int(config["warm_start"]["handoff_steps"]) + continuation_steps
        sac_train["total_timesteps"] = initial_sac_budget
        sac_train["checkpoint_interval"] = None
        sac_train["resume_state_path"] = None
        sac_config = {
            key: value for key, value in sac_train.items()
            if key in pufferl_sac.SAC_TRAIN_KEYS
        }
        sac_config.update(env="puffer_drive", run_name=sac_args["run_name"])
        sac_config = pufferl_sac.resolve_sac_batch_sizes(sac_config, shared_vecenv)
        with sac_rng.activate():
            if sac_policy is None:
                sac_policy = pufferl_sac.load_policy(sac_args, shared_vecenv, "puffer_drive")
            if critic is None:
                from pufferlib.ocean.torch_sac import DriveSACCritic

                critic = DriveSACCritic(shared_vecenv.driver_env, **sac_args["policy"]).to(
                    device=torch.device(pufferl.torch_device(resolved_device)), dtype=torch.float32,
                )
            if replay is None:
                from pufferlib.sac_replay import TensorReplay

                replay_device = "cpu" if sac_config["cpu_offload"] else pufferl.torch_device(resolved_device)
                replay = TensorReplay(
                    sac_config["batch_size"], shared_vecenv.single_observation_space.shape,
                    device=replay_device,
                    pin_memory=sac_config["cpu_offload"] and torch.device(resolved_device).type == "cuda",
                )
            sac_config["action_count"] = sac_policy.action_count
            if sac_agent is None:
                sac_agent = pufferl_sac.PuffeSAC(
                    sac_config, shared_vecenv, sac_policy, critic, replay,
                    _PhaseLogger(logger_to_close, "hybrid/sac_pretrain") if rank == 0
                    else pufferl.NoLogger(sac_config),
                    reset_env=False,
                )

        coordinator = WarmStartCoordinator(
            ppo_agent, sac_agent,
            handoff_steps=int(config["warm_start"]["handoff_steps"]),
            sac_steps=continuation_steps,
            ppo_args=ppo_args,
            sac_args=sac_args,
            logger=logger_to_close,
            sac_rng=sac_rng,
            replay_generator=replay_rng,
        )
        coordinator._sac_checkpoint_interval = sac_checkpoint_interval
        ppo_agent.transition_observer = coordinator.observe_transition

        ppo_eval_scheduled = (
            ppo_args["train"].get("evaluation_interval_epochs") is not None
            and pufferl.drive_benchmark.validate_training_evaluation_config(ppo_args)
        )
        last_ppo_eval_epoch = None
        maximum_ppo_cycles = math.ceil(
            global_ppo_schedule / (ppo_agent.config["batch_size"] * world_size)
        ) + 1
        for _ in range(maximum_ppo_cycles):
            if ppo_agent.global_step >= ppo_args["train"]["total_timesteps"]:
                raise RuntimeError("PPO schedule horizon ended before hybrid handoff")
            cycle_result = coordinator.run_ppo_cycle()
            if rank == 0:
                logger_to_close.log({
                    "hybrid/ppo_critic_pretrain/cycles": coordinator.ppo_cycles,
                    "hybrid/ppo_critic_pretrain/replay_size_per_rank": len(sac_agent.replay),
                    "hybrid/ppo_critic_pretrain/refresh_candidates": cycle_result["refresh_candidates"],
                    "hybrid/ppo_critic_pretrain/critic_updates": cycle_result["critic_updates"],
                }, cycle_result["cumulative_agent_steps"])
            if ppo_eval_scheduled and (
                ppo_agent.epoch % ppo_args["train"]["evaluation_interval_epochs"] == 0
            ):
                last_ppo_eval_epoch = ppo_agent.epoch
                if rank == 0:
                    _run_evaluation(
                        pufferl, ppo_args, ppo_agent.uncompiled_policy, logger_to_close,
                        ppo_agent.epoch, cycle_result["cumulative_agent_steps"], ppo_dir,
                    )
                _dist_barrier()
            if (
                cycle_result["cumulative_agent_steps"] >= coordinator.handoff_steps
                and cycle_result["replay_ready"]
                and cycle_result["critic_updates"] > 0
            ):
                break
        else:
            raise RuntimeError("Hybrid PPO cycle limit reached before handoff")

        if ppo_eval_scheduled and last_ppo_eval_epoch != ppo_agent.epoch and rank == 0:
            _run_evaluation(
                pufferl, ppo_args, ppo_agent.uncompiled_policy, logger_to_close,
                ppo_agent.epoch, coordinator.cumulative_agent_steps, ppo_dir,
            )
        if rank == 0:
            ppo_agent.save_checkpoint()
        _dist_barrier()

        handoff_info = coordinator.handoff()
        smoke = bool(config["provenance"].get("smoke", False))
        if not smoke and rank == 0:
            _run_evaluation(
                pufferl, sac_args, sac_agent.uncompiled_policy, logger_to_close,
                sac_agent.epoch, coordinator.boundary_agent_steps, sac_dir, force=True,
            )
        _dist_barrier()

        with sac_rng.activate():
            boundary_sac_actor_path = sac_agent.save_checkpoint()
        _dist_barrier()
        boundary_sac_state_path = sac_dir / "trainer_state.pt"
        if rank == 0:
            immutable_sac_state_path = sac_dir / "boundary_trainer_state.pt"
            immutable_sac_actor_path = sac_dir / "boundary_actor.pt"
            shutil.copy2(boundary_sac_state_path, immutable_sac_state_path)
            shutil.copy2(boundary_sac_actor_path, immutable_sac_actor_path)
        if rank == 0:
            state = {
                "format": "pufferdrive-ppo-sac-warm-start-boundary-v1",
                "phase": "sac",
                "resume_supported": False,
                "resume_limitation": "Native checkpoints omit replay and live simulator state.",
                "ppo_checkpoint": str(ppo_dir / "trainer_state.pt"),
                "sac_checkpoint": str(sac_dir / "boundary_trainer_state.pt"),
                "sac_actor_checkpoint": str(sac_dir / "boundary_actor.pt"),
                "resolved_config": deepcopy(config),
                "world_size": world_size,
                "global_agent_steps": coordinator.boundary_agent_steps,
                "ppo_cycles": coordinator.ppo_cycles,
                "sac_phase_steps": 0,
                "replay_size_per_rank": len(sac_agent.replay),
                "valid_replay_inserts_global": sac_agent.valid_inserts,
                "critic_optimizer_updates_per_rank": sac_agent.learner_updates,
                "replay_samples_consumed_global": sac_agent.sampled_transitions,
                "actor_probability_max_abs_error": handoff_info["actor_probability_max_abs_error"],
                "ppo_rng_state": pufferl.capture_rng_state(),
                "sac_rng_state": sac_rng.state,
                "replay_rng_state": replay_rng.get_state(),
            }
            _write_sidecar(run_dir / "hybrid_boundary_state.pt", state)
            logger_to_close.log({
                "hybrid/handoff_agent_steps": coordinator.boundary_agent_steps,
                "hybrid/handoff_replay_size_per_rank": len(sac_agent.replay),
                "hybrid/handoff_actor_probability_max_abs_error": handoff_info[
                    "actor_probability_max_abs_error"
                ],
                "hybrid/ppo_cycles": coordinator.ppo_cycles,
            }, coordinator.boundary_agent_steps)
        _dist_barrier()

        _release_ppo(ppo_agent)
        ppo_policy = None
        coordinator._target_policy = None
        sac_agent.logger = _PhaseLogger(
            logger_to_close, "hybrid/sac", preserve_environment=True,
        ) if rank == 0 else pufferl.NoLogger(sac_agent.config)
        _dist_barrier()

        sac_args["train"]["total_timesteps"] = coordinator.boundary_agent_steps + continuation_steps
        sac_agent.config["total_timesteps"] = coordinator.boundary_agent_steps + continuation_steps
        sac_eval_scheduled = (
            sac_args["train"].get("evaluation_interval_epochs") is not None
            and pufferl.drive_benchmark.validate_training_evaluation_config(sac_args)
        )
        last_sac_eval_epoch = None
        maximum_sac_cycles = math.ceil(
            continuation_steps / (sac_agent.refresh_count * world_size)
        ) + 1
        for _ in range(maximum_sac_cycles):
            if sac_agent.agent_steps >= sac_agent.config["total_timesteps"]:
                break
            starting_steps = sac_agent.agent_steps
            if sac_agent.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            with sac_rng.activate():
                sac_agent.evaluate()
                sac_agent.train(update_policy=True)
            coordinator.sac_phase_steps = sac_agent.agent_steps - coordinator.boundary_agent_steps
            coordinator.cumulative_agent_steps = sac_agent.agent_steps
            if sac_eval_scheduled and sac_agent.epoch > 0 and (
                sac_agent.epoch % sac_args["train"]["evaluation_interval_epochs"] == 0
            ):
                last_sac_eval_epoch = sac_agent.epoch
                if rank == 0:
                    _run_evaluation(
                        pufferl, sac_args, sac_agent.uncompiled_policy, logger_to_close,
                        sac_agent.epoch, sac_agent.agent_steps, sac_dir,
                    )
                _dist_barrier()
            if sac_agent.agent_steps <= starting_steps:
                raise RuntimeError("SAC collection completed without receiving agent rows")
        else:
            raise RuntimeError("Hybrid SAC continuation exceeded its bounded collection-cycle limit")

        if sac_eval_scheduled and sac_agent.epoch > 0 and last_sac_eval_epoch != sac_agent.epoch:
            if rank == 0:
                _run_evaluation(
                    pufferl, sac_args, sac_agent.uncompiled_policy, logger_to_close,
                    sac_agent.epoch, sac_agent.agent_steps, sac_dir,
                )
            _dist_barrier()

        with sac_rng.activate():
            final_model_path = sac_agent.close()
        shared_vecenv = None
        logger_to_close.close(final_model_path, early_stop=False)
        success = True
        return {
            "run_dir": str(run_dir),
            "boundary_agent_steps": coordinator.boundary_agent_steps,
            "final_agent_steps": sac_agent.agent_steps,
            "sac_phase_steps": coordinator.sac_phase_steps,
            "ppo_cycles": coordinator.ppo_cycles,
            "replay_size_per_rank": len(sac_agent.replay),
            "final_model_path": final_model_path,
        }
    finally:
        if not success:
            _cleanup_failed_run(shared_vecenv, ppo_agent, sac_agent, process_group_owned)
            _finish_logger_after_failure(logger_to_close)
        elif process_group_owned and torch.distributed.is_initialized():
            _dist_barrier()
            torch.distributed.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved hybrid YAML config from prepare.py")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        with config_path.open("r") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid hybrid YAML config: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Hybrid YAML config must contain a mapping: {config_path}")
    train(config)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
