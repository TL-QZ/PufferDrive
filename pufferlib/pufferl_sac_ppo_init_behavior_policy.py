"""Native SAC with a frozen PPO collection policy and a received-step schedule."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


def validate_config(config):
    """Reject incompatible experiment inputs before creating a simulator."""
    behavior = config["behavior_policy"]
    for key in ("ppo_only_steps", "decay_steps", "total_steps"):
        value = behavior[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"behavior_policy.{key} must be a positive integer")
    if behavior["ppo_only_steps"] + behavior["decay_steps"] >= behavior["total_steps"]:
        raise ValueError("Schedule requires a SAC-only period after decay")
    if config["train"]["total_timesteps"] != behavior["total_steps"]:
        raise ValueError("SAC total_timesteps must equal behavior_policy.total_steps")
    if config.get("load_id") or config.get("load_model_path") or config["train"].get("resume_state_path"):
        raise ValueError("This experiment requires fresh SAC; teacher weights are separate")
    if config["train"].get("fresh_run") is not True:
        raise ValueError("Set train.fresh_run=true; exact behavior-policy resume is unsupported")
    seed = behavior["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("behavior_policy.seed must be a nonnegative integer")
    for key in ("checkpoint", "teacher_config"):
        if not Path(behavior[key]).is_file():
            raise FileNotFoundError(f"Teacher {key} not found: {behavior[key]}")
    with Path(behavior["teacher_config"]).open() as source:
        teacher_config = yaml.safe_load(source)
    if teacher_config["policy_name"] != "Drive" or teacher_config.get("rnn_name") is not None:
        raise ValueError("Teacher must be a feed-forward Drive PPO policy")
    if teacher_config["policy"].get("action_type") != "discrete":
        raise ValueError("Teacher must use discrete actions")
    # The model consumes observations produced by SAC's simulator. Equal tensor
    # dimensions alone do not ensure equal normalization or action semantics.
    semantic_keys = {
        "action_type", "dynamics_model", "dt", "reward_conditioning",
        "control_mode", "sdc_controller", "non_sdc_controller",
    }
    semantic_keys.update(key for key in teacher_config["env"] if key.startswith("obs_"))
    mismatches = [key for key in sorted(semantic_keys)
                  if teacher_config["env"].get(key) != config["env"].get(key)]
    if mismatches:
        raise ValueError(f"Teacher/environment representation mismatch: {', '.join(mismatches)}")
    return teacher_config


class FrozenPPOBehavior:
    """Select collection actions; never participate in SAC targets or gradients."""

    def __init__(self, learner, teacher, settings):
        self.learner = learner
        self.teacher = teacher.requires_grad_(False).eval()
        self.settings = deepcopy(settings)
        self.ppo_only_steps = settings["ppo_only_steps"]
        self.decay_end = self.ppo_only_steps + settings["decay_steps"]
        self.total_steps = settings["total_steps"]
        self.generator = torch.Generator(device=learner.device)
        self.generator.manual_seed(settings["seed"] + learner.rank)
        self.pending_sources = torch.zeros(learner.total_agents, device=learner.device, dtype=torch.bool)
        self.replay_sources = torch.zeros(learner.batch_size, dtype=torch.bool)
        self.last_sources = None
        self.beta = 1.0
        self.last_metrics = {}

    def begin_collection(self, learner):
        steps = learner.agent_steps
        self.beta = max(0.0, min(1.0, (self.decay_end - steps) / self.settings["decay_steps"]))
        self.teacher_actions = 0
        self.all_actions = 0
        self.teacher_inserts = 0
        self.all_inserts = 0
        self.collection_start = steps
        if steps < self.ppo_only_steps:
            return self.ppo_only_steps
        if steps < self.decay_end:
            return self.decay_end
        return self.total_steps

    @torch.no_grad()
    def sample_actions(self, learner, observations):
        row_count = observations.shape[0]
        if self.beta == 0:
            # No extra random draws once the teacher is retired.
            self.last_sources = torch.zeros(row_count, device=learner.device, dtype=torch.bool)
            with learner.amp_context:
                logits = learner.policy(observations)
        else:
            self.last_sources = (
                torch.ones(row_count, device=learner.device, dtype=torch.bool)
                if self.beta == 1 else
                torch.rand(row_count, generator=self.generator, device=learner.device) < self.beta
            )
            with learner.amp_context:
                # Drive's actor-only path avoids the unused PPO value forward.
                hidden = self.teacher.actor_backbone(observations, self.teacher.ego_dim)
                logits = self.teacher.actor_head(hidden)
                if self.beta != 1:
                    sac_logits = learner.policy(observations)
                    logits = torch.where(self.last_sources[:, None], logits, sac_logits)
        actions, _, _, continuous_actions = learner.pytorch.sample_logits(
            logits.float(), env_continuous=learner.env_continuous, policy=learner.uncompiled_policy,
        )
        self.teacher_actions += int(self.last_sources.sum().item())
        self.all_actions += row_count
        return actions, continuous_actions

    def transition_sources(self, env_ids, masks):
        # Completed transitions correspond to the previous send, not today's beta.
        return self.pending_sources[env_ids][self.learner.pending[env_ids] & masks].clone()

    def record_action_sources(self, env_ids):
        self.pending_sources[env_ids] = self.last_sources

    def record_replay_sources(self, sources, indices):
        cpu_sources = sources.detach().to(device="cpu", dtype=torch.bool)
        self.replay_sources[indices.cpu()] = cpu_sources
        self.teacher_inserts += int(cpu_sources.sum().item())
        self.all_inserts += cpu_sources.numel()

    def end_collection(self, learner):
        counters = torch.tensor([
            self.teacher_actions, self.all_actions, self.teacher_inserts, self.all_inserts,
            int(self.replay_sources[:len(learner.replay)].sum().item()), len(learner.replay),
        ], dtype=torch.int64, device=learner.device)
        if learner.distributed:
            torch.distributed.all_reduce(counters)
        teacher_actions, all_actions, teacher_inserts, all_inserts, teacher_replay, replay_size = counters.tolist()
        self.last_metrics = {
            "behavior/ppo_probability": self.beta,
            "behavior/ppo_action_fraction": teacher_actions / max(all_actions, 1),
            "behavior/ppo_insert_fraction": teacher_inserts / max(all_inserts, 1),
            "behavior/ppo_replay_fraction": teacher_replay / max(replay_size, 1),
            "behavior/ppo_actions": teacher_actions,
            "behavior/sac_actions": all_actions - teacher_actions,
            "behavior/ppo_replay_transitions": teacher_replay,
            "behavior/collection_start_steps": self.collection_start,
            "behavior/stage": 0 if self.collection_start < self.ppo_only_steps else
                (1 if self.collection_start < self.decay_end else 2),
        }

    def metrics(self):
        return self.last_metrics


def make_behavior_policy(learner, config):
    from pufferlib import pufferl

    teacher_config = validate_config(config)
    teacher_args = deepcopy(teacher_config)
    teacher_args.update(load_id=None, load_model_path=config["behavior_policy"]["checkpoint"])
    teacher_args["train"]["device"] = str(learner.device)
    rng_state = pufferl.capture_rng_state()
    try:
        teacher = pufferl.load_policy(teacher_args, learner.vecenv, "puffer_drive")
    finally:
        pufferl.restore_rng_state({"rng_state": rng_state})
    for table_name in ("action_table", "action_table_physical"):
        if not torch.equal(getattr(teacher, table_name), getattr(learner.uncompiled_policy, table_name)):
            raise ValueError(f"Teacher and SAC {table_name} differ")
    provider = FrozenPPOBehavior(learner, teacher, config["behavior_policy"])
    if learner.rank == 0:
        checkpoint = Path(config["behavior_policy"]["checkpoint"])
        with checkpoint.open("rb") as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            **config["behavior_policy"], "checkpoint_sha256": checkpoint_sha256,
            "teacher_frozen": True, "sac_initialization": "fresh",
            "schedule_units": "global received simulator agent rows",
            "teacher_config_snapshot": teacher_config,
        }
        output_dir = Path(config["train"]["data_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "behavior_policy_provenance.json").open("x") as output:
            json.dump(metadata, output, indent=2)
    return provider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    with Path(args.config).open() as source:
        config = yaml.safe_load(source)
    validate_config(config)
    output_dir = Path(config["train"]["data_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing learner directory: {output_dir}; prepare a fresh run")
    from pufferlib import pufferl_sac
    from project.soft_actor_critic_tuning.scripts.run_trial import TrialMonitor

    artifact_dir = output_dir.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrialMonitor(artifact_dir / "metrics.jsonl", artifact_dir / "stop_reason.json", None) \
        if int(os.environ.get("RANK", "0")) == 0 else None
    pufferl_sac.train(
        env_name="puffer_drive", args=config, early_stop_fn=monitor,
        behavior_policy_factory=make_behavior_policy,
    )


if __name__ == "__main__":
    main()
