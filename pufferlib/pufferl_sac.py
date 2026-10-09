"""Feed-forward discrete SAC using the same evaluate/train lifecycle as PuffeRL.

Drive networks, replay, pre-reset observations and Hydra startup are integrated.
Learner options, DDP, AMP/compile and evaluation workflows remain planned;
recurrence and replay/environment resume remain deferred.
"""

from collections import defaultdict
import copy
import glob
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import warnings

from hydra import compose, initialize_config_dir
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F


STATE_VERSION = 5
DEFAULT_ENTROPY_FRACTION = 0.98
FINAL_OBSERVATIONS_INFO_KEY = "_pufferlib_final_observations"
DEFAULT_CONFIG = dict(
    env="puffer_drive", device="cpu", seed=0, run_name="sac", batch_size=10_000,
    minibatch_size=256, max_minibatch_size=256, update_epochs=1,
    total_timesteps=None, gamma=0.99, learning_starts=10_000, replay_update_ratio=0.1,
    learning_rate=3e-4, critic_learning_rate=3e-4, alpha_learning_rate=3e-4,
    optimizer="adam", adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, adam_weight_decay=0.0,
    max_grad_norm=None,
    initial_alpha=0.2, target_entropy=None, tau=0.005, torch_deterministic=False,
    anneal_lr=False, checkpoint_interval=50, final_model_name="final_model.pt", resume_state_path=None,
)

SAC_TRAIN_KEYS = frozenset(DEFAULT_CONFIG) - {"env"} | {"data_dir"}
DEFAULT_MINIBATCH_SIZE = 256


def resolve_sac_batch_sizes(config, vecenv):
    """Resolve SAC replay capacity and logical batch sizes before replay allocation."""
    resolved = dict(config)
    batch_size = resolved.get("batch_size", DEFAULT_CONFIG["batch_size"])
    minibatch_size = resolved.get("minibatch_size", DEFAULT_CONFIG["minibatch_size"])
    learning_starts = resolved.get("learning_starts", DEFAULT_CONFIG["learning_starts"])
    max_minibatch_size = resolved.get("max_minibatch_size", DEFAULT_CONFIG["max_minibatch_size"])

    if not isinstance(learning_starts, int) or isinstance(learning_starts, bool) or learning_starts <= 0:
        raise ValueError("learning_starts must be a positive integer")
    if not isinstance(max_minibatch_size, int) or isinstance(max_minibatch_size, bool) or max_minibatch_size <= 0:
        raise ValueError("max_minibatch_size must be a positive integer")
    if batch_size != "auto" and (
            not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer or 'auto'")
    if minibatch_size != "auto" and (
            not isinstance(minibatch_size, int) or isinstance(minibatch_size, bool) or minibatch_size <= 0):
        raise ValueError("minibatch_size must be a positive integer or 'auto'")

    if minibatch_size == "auto":
        if batch_size == "auto":
            minibatch_size = DEFAULT_MINIBATCH_SIZE
        else:
            minibatch_size = min(DEFAULT_MINIBATCH_SIZE, batch_size)

    if batch_size == "auto":
        agents_per_batch = vecenv.agents_per_batch
        batch_size = max(learning_starts, agents_per_batch, minibatch_size)

    resolved["batch_size"] = batch_size
    resolved["minibatch_size"] = minibatch_size
    resolved["max_minibatch_size"] = max_minibatch_size
    if learning_starts > batch_size:
        raise ValueError("Require learning_starts <= batch_size replay capacity")
    return resolved


def load_config(env_name="puffer_drive", config_dir=None, overrides=None):
    """Compose the Drive SAC config and remove inherited PPO-only train settings."""
    if env_name != "puffer_drive":
        raise NotImplementedError(f"Drive SAC startup is only configured for puffer_drive, got {env_name!r}")

    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "config"
    if overrides is None:
        overrides = sys.argv[1:]

    for override in overrides:
        if override.startswith("--"):
            hint = override.lstrip("-").replace("-", "_").split("=")[0]
            import pufferlib
            raise pufferlib.APIUsageError(
                f"'{override}' uses the old flag syntax. Use Hydra overrides: {hint}=<value>"
            )
        override_path = override.lstrip("+~").split("=", 1)[0]
        if override_path == "train" or override_path.startswith("train."):
            train_key = override_path.partition(".")[2]
            if train_key not in SAC_TRAIN_KEYS:
                import pufferlib
                raise pufferlib.APIUsageError(
                    f"Unsupported SAC training override '{override_path}'. "
                    "PPO-only train settings are not used by SAC."
                )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=f"{env_name}_sac", overrides=list(overrides))

    # The SAC config inherits the Drive environment but has its own train schema.
    from pufferlib.config_schema import ENV_SCHEMAS
    env_schema = ENV_SCHEMAS.get(env_name)
    if env_schema is not None:
        cfg["env"] = OmegaConf.merge(OmegaConf.structured(env_schema), cfg["env"])

    args = defaultdict(dict, OmegaConf.to_container(
        cfg, resolve=True, enum_to_str=True, throw_on_missing=True,
    ))
    inherited_train = args["train"]
    args["train"] = {key: value for key, value in inherited_train.items() if key in SAC_TRAIN_KEYS}
    args["env_name"] = env_name
    return args


def load_policy(args, vecenv, env_name="puffer_drive"):
    """Build Drive SAC's FP32 actor, matching PPO's one-policy loader contract."""
    if env_name != "puffer_drive":
        raise NotImplementedError(f"Drive SAC policy loading is not configured for {env_name!r}")
    if args.get("rnn_name") is not None:
        raise NotImplementedError("SAC recurrence is deferred; set rnn_name=null")

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        from pufferlib import pufferl as ppo
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
    from pufferlib.ocean.torch_sac import DriveSAC

    device = ppo.torch_device(args["train"]["device"])
    actor = DriveSAC(vecenv.driver_env, **args["policy"]).to(device=device, dtype=torch.float32)
    return actor


class PuffeSAC:
    """Injected component contracts; transition tensors are learner-owned.

    policy(obs) -> logits [B,K]; critic(obs) -> (q1, q2), each [B,K].
    replay.add(obs, actions, rewards, next_obs, discounts) appends copied rows.
    replay.replace(indices, obs, actions, rewards, next_obs, discounts) copies rows
    into distinct existing slots without changing occupancy. Both use the same five
    transition fields; replay.get_batch(indices) returns those fields in that order.
    Indices passed to replace/get_batch are CPU int64 tensors.
    Observations are FP32 [B,...]; actions int64 [B]; rewards/discounts FP32 [B].
    vecenv uses Drive's same-step reset semantics: recv masks qualify the preceding
    action. Other backends need that same contract; reset-only rows are unsupported.
    Discounts include gamma. Truncated transitions need their pre-reset final
    observation, which Drive transports as a sparse private info record.
    batch_size is replay capacity. evaluate() fills remaining capacity with add(),
    then replaces a replay_update_ratio fraction of distinct slots per call. It sends
    every received row; global_step counts all received slots, including reset,
    masked, and valid rows discarded after the insertion quota is reached.
    """

    def __init__(self, config, vecenv=None, policy=None, critic=None, replay=None, logger=None):
        if vecenv is None or policy is None or critic is None or replay is None:
            raise NotImplementedError(
                "Supply vecenv, policy, critic and replay to PuffeSAC, "
                "or use train(...) to construct Drive components from configuration."
            )
        if config is None:
            raise ValueError("Supply a SAC training config dictionary")

        config = resolve_sac_batch_sizes({**DEFAULT_CONFIG, **config}, vecenv)
        if "data_dir" not in config or not config["data_dir"]:
            raise ValueError("SAC requires an explicit data_dir")
        supported_config = DEFAULT_CONFIG.keys() | {
            "action_count", "total_timesteps", "data_dir", "gamma", "tau",
        }
        if set(config) - supported_config:
            raise ValueError("Unsupported SAC config fields; use DEFAULT_CONFIG plus action_count and data_dir")
        for key in ("action_count", "total_timesteps", "update_epochs", "checkpoint_interval"):
            if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if config["action_count"] < 2:
            raise ValueError("Require action_count >= 2")
        for key in ("learning_rate", "critic_learning_rate", "alpha_learning_rate",
                    "initial_alpha", "replay_update_ratio", "tau", "adam_eps"):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if config["optimizer"] not in {"adam", "adamw", "muon"}:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        for key in ("adam_beta1", "adam_beta2"):
            if not math.isfinite(config[key]) or not 0 <= config[key] < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        if not math.isfinite(config["adam_weight_decay"]) or config["adam_weight_decay"] < 0:
            raise ValueError("adam_weight_decay must be finite and nonnegative")
        if config["max_grad_norm"] is not None and (
                not math.isfinite(config["max_grad_norm"]) or config["max_grad_norm"] <= 0):
            raise ValueError("max_grad_norm must be finite and positive or None")
        if not math.isfinite(config["gamma"]) or not 0 <= config["gamma"] <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if config["tau"] > 1:
            raise ValueError("Require tau <= 1")
        if config["replay_update_ratio"] > 1:
            raise ValueError("Require replay_update_ratio <= 1")
        if config["target_entropy"] is None:
            config["target_entropy"] = DEFAULT_ENTROPY_FRACTION * math.log(config["action_count"])
        if not 0 <= config["target_entropy"] <= math.log(config["action_count"]):
            raise ValueError("target_entropy must be in [0, log(action_count)]")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise NotImplementedError("SAC DDP is deferred; run one process")
        if set(policy.parameters()) & set(critic.parameters()):
            raise ValueError("Policy and critic optimizers must own separate parameters")

        # Import after preflight. PPO installs an immediate-exit SIGINT handler;
        # retain the caller's handler so construction can release workers safely.
        interrupt_handler = signal.getsignal(signal.SIGINT)
        try:
            from pufferlib import pufferl, pytorch, spaces
        finally:
            signal.signal(signal.SIGINT, interrupt_handler)

        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = not config["torch_deterministic"]
        torch.use_deterministic_algorithms(config["torch_deterministic"], warn_only=True)
        torch_seed, _ = pufferl.derive_rank_seeds(None, config["seed"], 1, 0)
        random.seed(torch_seed)
        np.random.seed(torch_seed)
        torch.manual_seed(torch_seed)

        self.helpers = pufferl
        self.pytorch = pytorch
        self.config = config
        self.device = pufferl.torch_device(config["device"])
        self.vecenv = vecenv
        self.batch_size = config["batch_size"]
        self.refresh_count = math.ceil(self.batch_size * config["replay_update_ratio"])
        if len(replay) > self.batch_size:
            raise ValueError("Initial replay length must not exceed batch_size capacity")
        self.env_continuous = isinstance(vecenv.single_action_space, spaces.Box)
        if self.env_continuous:
            if getattr(policy, "is_continuous", None) is not False or not callable(
                getattr(policy, "discrete_actions_to_continuous", None)
            ):
                raise NotImplementedError(
                    "Continuous environments require a discrete SAC policy with "
                    "is_continuous=False and discrete_actions_to_continuous()."
                )
        elif getattr(vecenv.single_action_space, "n", None) != config["action_count"]:
            raise NotImplementedError("Discrete SAC requires a Discrete environment action space matching action_count")

        self.total_agents = vecenv.num_agents
        self.minibatch_size = config["minibatch_size"]
        self.max_minibatch_size = config["max_minibatch_size"]
        self.observations = torch.zeros(
            self.total_agents, *vecenv.single_observation_space.shape, device=self.device, dtype=torch.float32
        )
        self.actions = torch.zeros(self.total_agents, device=self.device, dtype=torch.int64)
        self.pending = torch.zeros(self.total_agents, device=self.device, dtype=torch.bool)
        self.replay = replay

        self.policy = policy.to(self.device, dtype=torch.float32)
        self.uncompiled_policy = self.policy
        self.model_size = sum(parameter.numel() for parameter in self.policy.parameters() if parameter.requires_grad)
        self.critic = critic.to(self.device, dtype=torch.float32)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.log_alpha = torch.nn.Parameter(torch.tensor(
            math.log(config["initial_alpha"]), device=self.device, dtype=torch.float32
        ))
        self.optimizer = self._make_model_optimizer(self.policy.parameters(), config["learning_rate"])
        self.critic_optimizer = self._make_model_optimizer(self.critic.parameters(), config["critic_learning_rate"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["alpha_learning_rate"])
        scheduler_steps = max(1, math.ceil(config["total_timesteps"] / self.refresh_count))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_steps)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=scheduler_steps
        )
        self.total_epochs = scheduler_steps
        self.logger = logger if logger is not None else pufferl.NoLogger(config)
        self.profile = pufferl.Profile()
        self.global_step = 0
        self.agent_steps = 0
        self.epoch = 0
        self.last_log_step = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.valid_inserts = 0
        self.learner_updates = 0
        self.sampled_transitions = 0
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.last_stats_epoch = None
        self.losses = {}
        self.best_score = -float("inf")
        self.closed = False
        self.final_model_path = None
        self.utilization = pufferl.Utilization()
        try:
            vecenv.async_reset(config["seed"])
            self.print_dashboard(clear=True)
        except BaseException:
            self.utilization.stop()
            self.utilization.join()
            raise

    def _make_model_optimizer(self, parameters, learning_rate):
        optimizer_name = self.config["optimizer"]
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (self.config["adam_beta1"], self.config["adam_beta2"]),
            "eps": self.config["adam_eps"],
        }
        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameters, weight_decay=self.config["adam_weight_decay"], **optimizer_kwargs
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters, weight_decay=self.config["adam_weight_decay"], **optimizer_kwargs
            )

        import heavyball

        warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
        heavyball.utils.compile_mode = "default"
        muon_optimizer = getattr(heavyball, "ForeachMuon", None)
        if muon_optimizer is None:
            return heavyball.Muon(
                parameters,
                **optimizer_kwargs,
                heavyball_momentum=True,
                compile_step=False,
            )
        return muon_optimizer(
            parameters,
            **optimizer_kwargs,
            heavyball_momentum=True,
        )

    def load_training_state(self, path):
        """Restore learner state into a fresh environment and empty replay."""
        if (len(self.replay) or self.global_step or self.epoch or self.valid_inserts
                or self.learner_updates or self.sampled_transitions):
            raise ValueError("Restore requires a fresh trainer and empty replay")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (state.get("state_version") != STATE_VERSION or state.get("algorithm") != "sac"
                or state.get("scope") != "learner_only"):
            raise ValueError("Unsupported SAC checkpoint format; expected learner-only SAC state version 5")
        operational_config = {
            "device", "total_timesteps", "run_name", "data_dir", "final_model_name",
            "checkpoint_interval", "resume_state_path",
        }
        saved_config = {**DEFAULT_CONFIG, **state["config"]}
        for key, value in saved_config.items():
            if key not in operational_config and self.config[key] != value:
                raise ValueError(f"Checkpoint config mismatch: {key}")
        for name in ("policy", "critic", "target_critic"):
            model_state = self.helpers.clean_policy_state_dict(state[f"{name}_state_dict"])
            getattr(self, name).load_state_dict(model_state)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        for name in ("optimizer", "critic_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[f"{name}_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(state["critic_scheduler_state_dict"])
        self.total_epochs = self.scheduler.T_max
        for name in ("global_step", "epoch", "valid_inserts", "learner_updates", "sampled_transitions",
                     "best_score"):
            setattr(self, name, state[name])
        self.agent_steps = self.global_step
        self.pending.zero_()
        self.observations.zero_()
        self.actions.zero_()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.last_stats_epoch = None
        self.losses = {}
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = self.global_step
        self.profile.profiles.clear()
        self.profile.stack.clear()
        self.helpers.restore_rng_state(state)
        print(f"Resumed SAC learner from {path}: epoch={self.epoch}, global_step={self.global_step}")

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0
        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def _consume_final_observation_infos(self, infos, env_ids, required_rows):
        final_id_batches = []
        final_observation_batches = []
        metric_infos = []
        for info in infos:
            if not isinstance(info, dict) or FINAL_OBSERVATIONS_INFO_KEY not in info:
                metric_infos.append(info)
                continue

            payload = info[FINAL_OBSERVATIONS_INFO_KEY]
            final_id_batches.append(payload["env_ids"])
            final_observation_batches.append(payload["observations"])
            if len(info) > 1:
                metric_infos.append({key: value for key, value in info.items()
                                     if key != FINAL_OBSERVATIONS_INFO_KEY})

        if not final_id_batches:
            final_agent_ids = np.empty(0, dtype=np.int64)
            final_observations = None
        else:
            final_agent_ids = np.concatenate(final_id_batches)
            final_observations = np.concatenate(final_observation_batches)
            final_order = np.argsort(final_agent_ids)
            final_agent_ids = final_agent_ids[final_order]
            final_observations = final_observations[final_order]

        if required_rows:
            agent_ids = env_ids.detach().cpu().numpy()
            required_agent_ids = agent_ids[required_rows]
            final_positions = np.searchsorted(final_agent_ids, required_agent_ids)
            present = final_positions < final_agent_ids.size
            present_rows = np.flatnonzero(present)
            present[present_rows] = final_agent_ids[final_positions[present_rows]] == required_agent_ids[present_rows]
            if not present.all():
                missing_agent_ids = required_agent_ids[~present].tolist()
                raise ValueError(
                    "Missing pre-reset final observations for truncated transitions; "
                    f"agent IDs: {missing_agent_ids[:8]}"
                )
        return final_agent_ids, final_observations, metric_infos

    @torch.no_grad()
    def evaluate(self):
        """Collect transitions directly from vecenv, matching PuffeRL.evaluate."""
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)
        replay_size = len(self.replay)
        refresh_indices = None
        if replay_size < self.batch_size:
            insert_target = self.batch_size - replay_size
        else:
            insert_target = self.refresh_count
            refresh_indices = torch.randperm(replay_size, device="cpu")[:insert_target]
        received_budget = max(0, self.config["total_timesteps"] - self.global_step)
        agent_steps = 0
        valid_inserts = 0
        for _ in range(received_budget):
            profile("env", epoch)
            observations, rewards, terminals, truncations, infos, env_ids, masks = self.vecenv.recv()
            profile("eval_misc", epoch)
            observations = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
            terminals = torch.as_tensor(terminals, device=self.device, dtype=torch.bool)
            truncations = torch.as_tensor(truncations, device=self.device, dtype=torch.bool)
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            masks = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
            received_count = env_ids.numel()
            agent_steps += received_count
            self.global_step += received_count

            # The received mask qualifies the preceding action, not the new observation.
            completed = self.pending[env_ids] & masks
            unsupported_truncations = completed & truncations & ~terminals
            required_final_rows = torch.nonzero(unsupported_truncations, as_tuple=False).squeeze(-1)
            final_agent_ids, final_observations, infos = self._consume_final_observation_infos(
                infos, env_ids, required_final_rows.detach().cpu().tolist()
            )
            self.stats["obs/max"].append(observations.max().item())
            self.stats["obs/min"].append(observations.min().item())
            self.stats["obs/mean"].append(observations.mean().item())

            profile("eval_forward", epoch)
            actions, _, _, continuous_actions = self.pytorch.sample_logits(
                self.policy(observations), env_continuous=self.env_continuous, policy=self.policy
            )

            if completed.any():
                profile("eval_copy", epoch)
                completed_rows = torch.nonzero(completed, as_tuple=False).squeeze(-1)
                remaining_quota = insert_target - valid_inserts
                if completed_rows.numel() > remaining_quota:
                    selected_rows = torch.randperm(
                        completed_rows.numel(), device=completed_rows.device
                    )[:remaining_quota]
                    completed_rows = completed_rows[selected_rows]
                transition_ids = env_ids[completed_rows]
                terminal_transitions = terminals[completed_rows]
                next_observations = observations[completed_rows]
                truncated_transitions = truncations[completed_rows] & ~terminal_transitions
                if truncated_transitions.any():
                    truncated_agent_ids = transition_ids[truncated_transitions].detach().cpu().numpy()
                    final_positions = np.searchsorted(final_agent_ids, truncated_agent_ids)
                    final_batch = final_observations[final_positions]
                    next_observations[truncated_transitions] = torch.as_tensor(
                        final_batch, device=self.device, dtype=torch.float32
                    )
                # Terminal rows need no successor; use a finite, already-owned state.
                next_observations[terminal_transitions] = self.observations[
                    transition_ids[terminal_transitions]
                ]
                discounts = self.config["gamma"] * (~terminal_transitions).to(torch.float32)
                transition_rewards = rewards[completed_rows]
                transition_observations = self.observations[transition_ids]
                transition_actions = self.actions[transition_ids]
                if replay_size < self.batch_size:
                    self.replay.add(
                        transition_observations, transition_actions, transition_rewards,
                        next_observations, discounts,
                    )
                else:
                    replace_indices = refresh_indices[valid_inserts:valid_inserts + completed_rows.numel()]
                    self.replay.replace(
                        replace_indices, transition_observations, transition_actions, transition_rewards,
                        next_observations, discounts,
                    )
                valid_inserts += completed_rows.numel()

            profile("eval_misc", epoch)
            for info in infos:
                for key, value in self.helpers.pufferlib.unroll_nested_dict(info):
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if isinstance(value, (list, tuple)):
                        self.stats[key].extend(value)
                    else:
                        self.stats[key].append(value)

            # Copy before send can overwrite shared observations. Keep the final pending send.
            profile("eval_copy", epoch)
            self.observations[env_ids] = observations
            self.actions[env_ids] = actions.long()
            self.pending[env_ids] = True

            profile("env", epoch)
            if self.env_continuous:
                self.vecenv.send(continuous_actions.detach().cpu().numpy())
            else:
                self.vecenv.send(actions.detach().cpu().numpy())
            if valid_inserts >= insert_target or agent_steps >= received_budget:
                break

        self.valid_inserts += valid_inserts
        profile.end()
        return self.helpers.pufferlib.utils.reduce_environment_metrics(self.stats)

    def train(self):
        """Optimize from replay after evaluate(); one call increments epoch once."""
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        profile("train_misc", epoch, nest=True)
        losses = defaultdict(float)
        updates = self._train_sac_transition(losses, profile, epoch)
        profile("train_misc", epoch)
        if updates and self.config["anneal_lr"]:
            if self.scheduler.last_epoch < self.scheduler.T_max:
                self.scheduler.step()
            if self.critic_scheduler.last_epoch < self.critic_scheduler.T_max:
                self.critic_scheduler.step()
        profile.end()

        self.losses = losses
        self.epoch += 1
        done_training = self.global_step >= self.config["total_timesteps"]
        logs = None
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % self.config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"
        return logs

    @torch.no_grad()
    def _compute_targets(self, mb_rewards, mb_next_obs, mb_discounts):
        logits = self.policy(mb_next_obs)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        next_q1, next_q2 = self.target_critic(mb_next_obs)
        next_value = (probs * (torch.minimum(next_q1, next_q2) - self.alpha * logprobs)).sum(-1)
        return mb_rewards + mb_discounts * next_value

    def _critic_loss(self, mb_obs, mb_actions, mb_targets):
        q1, q2 = self.critic(mb_obs)
        q1 = q1.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
        q2 = q2.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q1, mb_targets) + F.mse_loss(q2, mb_targets)
        return loss, q1.detach().mean(), q2.detach().mean()

    def _policy_loss(self, mb_obs):
        logits = self.policy(mb_obs)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        with torch.no_grad():
            q1, q2 = self.critic(mb_obs)
        loss = (probs * (self.alpha * logprobs - torch.minimum(q1, q2))).sum(-1).mean()
        entropy = -(probs * logprobs).sum(-1).mean()
        return loss, entropy.detach()

    def _train_sac_transition(self, losses, profile, epoch):
        replay_size = len(self.replay)
        if replay_size < self.config["learning_starts"]:
            return 0
        minibatches_per_epoch = math.ceil(replay_size / self.minibatch_size)
        total_minibatches = self.config["update_epochs"] * minibatches_per_epoch
        total_sampled_transitions = replay_size * self.config["update_epochs"]
        replay_indices = None
        for logical_update_idx in range(total_minibatches):
            minibatch_idx = logical_update_idx % minibatches_per_epoch
            if minibatch_idx == 0:
                replay_indices = torch.randperm(replay_size, device="cpu")
            start_idx = minibatch_idx * self.minibatch_size
            mb_indices = replay_indices[start_idx:start_idx + self.minibatch_size]
            logical_batch_count = mb_indices.numel()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_value = 0.0
            q1_value = 0.0
            q2_value = 0.0
            target_value = 0.0
            for micro_start in range(0, logical_batch_count, self.max_minibatch_size):
                micro_indices = mb_indices[micro_start:micro_start + self.max_minibatch_size]
                micro_count = micro_indices.numel()
                profile("train_copy", epoch)
                mb_obs, mb_actions, mb_rewards, mb_next_obs, mb_discounts = (
                    tensor.to(self.device) for tensor in self.replay.get_batch(micro_indices)
                )
                profile("train_forward", epoch)
                mb_targets = self._compute_targets(mb_rewards, mb_next_obs, mb_discounts)
                critic_loss, q1, q2 = self._critic_loss(mb_obs, mb_actions, mb_targets)
                profile("learn", epoch)
                (critic_loss * (micro_count / logical_batch_count)).backward()
                critic_loss_value += critic_loss.detach().item() * micro_count / logical_batch_count
                q1_value += q1.item() * micro_count / logical_batch_count
                q2_value += q2.item() * micro_count / logical_batch_count
                target_value += mb_targets.mean().item() * micro_count / logical_batch_count

            if self.config["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config["max_grad_norm"])
            self.critic_optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            policy_loss_value = 0.0
            entropy_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            for micro_start in range(0, logical_batch_count, self.max_minibatch_size):
                micro_indices = mb_indices[micro_start:micro_start + self.max_minibatch_size]
                micro_count = micro_indices.numel()
                profile("train_copy", epoch)
                mb_obs = self.replay.get_batch(micro_indices)[0].to(self.device)
                profile("train_forward", epoch)
                policy_loss, entropy = self._policy_loss(mb_obs)
                profile("learn", epoch)
                (policy_loss * (micro_count / logical_batch_count)).backward()
                policy_loss_value += policy_loss.detach().item() * micro_count / logical_batch_count
                entropy_sum += entropy * micro_count

            if self.config["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()

            mean_entropy = entropy_sum / logical_batch_count
            alpha_loss = self.log_alpha * (mean_entropy - self.config["target_entropy"])
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            profile("train_misc", epoch)
            # Polyak-update all target parameters and buffers, including encoders.
            with torch.no_grad():
                online = dict(self.critic.named_parameters()) | dict(self.critic.named_buffers())
                targets = dict(self.target_critic.named_parameters()) | dict(self.target_critic.named_buffers())
                for key, target in targets.items():
                    if target.is_floating_point():
                        target.lerp_(online[key], self.config["tau"])
                    else:
                        target.copy_(online[key])

            self.learner_updates += 1
            self.sampled_transitions += logical_batch_count
            update_losses = dict(
                critic_loss=critic_loss_value, policy_loss=policy_loss_value,
                alpha_loss=alpha_loss.detach().item(), entropy=mean_entropy.item(), alpha=self.alpha.item(),
                q1=q1_value, q2=q2_value, target=target_value,
            )
            profile("train_misc", epoch)
            for key, value in update_losses.items():
                losses[key] += value * logical_batch_count / total_sampled_transitions
        return total_minibatches

    def mean_and_log(self):
        self.stats = self.helpers.pufferlib.utils.reduce_environment_metrics(self.stats)
        self.last_stats = dict(self.stats)
        self.last_stats_epoch = self.epoch
        self.agent_steps = self.global_step
        logs = {
            "SPS": self.sps,
            "agent_steps": self.global_step,
            "uptime": self.uptime,
            "epoch": self.epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "replay/size": len(self.replay),
            "replay/valid_inserts": self.valid_inserts,
            "replay/sampled_transitions": self.sampled_transitions,
            "sac/learner_updates": self.learner_updates,
            **{f"environment/{key}": value for key, value in self.stats.items()},
            **{f"losses/{key}": value for key, value in self.losses.items()},
            **{f"performance/{key}": value["elapsed"] for key, value in self.profile},
        }
        self.logger.log(logs, self.global_step)
        return logs

    def close(self):
        if self.closed:
            return self.final_model_path
        self.closed = True
        close_error = None
        try:
            self.vecenv.close()
        except BaseException as error:
            close_error = error
        try:
            self.utilization.stop()
            self.utilization.join()
        except BaseException as error:
            if close_error is None:
                close_error = error
        if close_error is not None:
            raise close_error

        model_path = self.save_checkpoint()
        final_path = Path(self.config["data_dir"]) / self.config["final_model_name"]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = str(final_path) + ".tmp"
        shutil.copyfile(model_path, temporary_path)
        os.replace(temporary_path, final_path)
        self.final_model_path = str(final_path)
        return self.final_model_path

    def save_checkpoint(self):
        """Atomically save the actor artifact and full learner-only state."""
        data_dir = Path(self.config["data_dir"])
        models_dir = data_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = models_dir / model_name
        model_temporary_path = str(model_path) + ".tmp"
        torch.save(self.uncompiled_policy.state_dict(), model_temporary_path)
        os.replace(model_temporary_path, model_path)

        score_is_fresh = self.last_stats_epoch == self.epoch
        current_score = self.last_stats.get("puffer_score", self.last_stats.get("score", -float("inf")))
        new_best = score_is_fresh and current_score > self.best_score
        previous_best_score = self.best_score
        best_state_file = data_dir / "best_models" / f"best_trainer_state_{self.epoch:06d}.pt"
        best_temporary_path = str(best_state_file) + ".tmp"
        if new_best:
            best_state_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_path, best_temporary_path)
            os.replace(best_temporary_path, best_state_file)
            self.best_score = current_score

        state = {
            "state_version": STATE_VERSION,
            "algorithm": "sac",
            "scope": "learner_only",
            "config": dict(self.config),
            "policy_state_dict": self.uncompiled_policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "valid_inserts": self.valid_inserts,
            "learner_updates": self.learner_updates,
            "sampled_transitions": self.sampled_transitions,
            "best_score": self.best_score,
            "model_name": model_name,
            "run_metadata": {"run_id": getattr(self.logger, "run_id", self.config["run_name"]),
                             "env": self.config["env"]},
            "rng_state": self.helpers.capture_rng_state(),
        }
        state_path = data_dir / "trainer_state.pt"
        state_temporary_path = str(state_path) + ".tmp"
        try:
            torch.save(state, state_temporary_path)
            os.replace(state_temporary_path, state_path)
        except BaseException:
            if new_best:
                self.best_score = previous_best_score
                if best_state_file.exists():
                    best_state_file.unlink()
            raise

        for old_model_path in glob.glob(str(models_dir / "model_*.pt")):
            if old_model_path != str(model_path):
                os.remove(old_model_path)
        if new_best:
            for old_best_path in glob.glob(str(data_dir / "best_models" / "best_trainer_state_*.pt")):
                if old_best_path != str(best_state_file):
                    os.remove(old_best_path)
            print(f"New best model saved at epoch {self.epoch} with score {self.best_score:.4f}")
        return str(model_path)

    def print_dashboard(self, clear=False):
        return self.helpers.PuffeRL.print_dashboard(self, clear=clear)


def train(env_name="puffer_drive", args=None, vecenv=None, policy=None, critic=None,
          replay=None, logger=None, early_stop_fn=None):
    """Run Drive SAC from Hydra config or from explicitly injected components."""
    pufferl = None
    early_stop = False
    last_logs = {}
    cleanup_logger = logger
    close_started = False
    try:
        if args is None or "train" in args:
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            try:
                from pufferlib import pufferl as ppo
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)

            startup_args = load_config(env_name) if args is None else args
            if env_name != "puffer_drive":
                raise NotImplementedError(f"Drive SAC startup is only configured for puffer_drive, got {env_name!r}")
            unsupported_train_keys = set(startup_args["train"]) - SAC_TRAIN_KEYS
            if unsupported_train_keys:
                unsupported_names = ", ".join(sorted(unsupported_train_keys))
                raise ValueError(
                    f"Unsupported SAC train settings: {unsupported_names}. "
                    "PPO-only train settings are not used by SAC."
                )
            if startup_args.get("rnn_name") is not None:
                raise NotImplementedError("SAC recurrence is deferred; set rnn_name=null")
            if startup_args["policy"].get("action_type") != "discrete":
                raise ValueError("SAC requires policy.action_type=discrete")
            if startup_args["policy"].get("shared_network") is not False:
                raise ValueError("SAC requires policy.shared_network=false for independent actor and critic networks")
            if startup_args.get("load_model_path") is not None or startup_args.get("load_id") is not None:
                raise NotImplementedError(
                    "SAC model loading by load_model_path/load_id is deferred; set both to null."
                )
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size != 1:
                raise NotImplementedError("SAC DDP is deferred; launch one process with WORLD_SIZE=1")

            train_args = startup_args["train"]
            run_name = startup_args.get("run_name", train_args.get("run_name", "sac"))
            train_seed = train_args.get("seed")
            if train_seed is None:
                train_seed = time.time_ns() & 0xFFFFFFFF
            train_args["seed"] = train_seed
            torch_seed, env_seed = ppo.derive_rank_seeds(
                startup_args["vec"].get("seed"), train_seed, world_size, 0,
            )
            random.seed(torch_seed)
            np.random.seed(torch_seed)
            torch.manual_seed(torch_seed)

            config = {key: value for key, value in train_args.items() if key in SAC_TRAIN_KEYS}
            config.update(
                env=env_name,
                run_name=run_name,
            )
            startup_args["run_name"] = run_name
            startup_args.setdefault("no_model_upload", False)
            startup_args["policy_name"] = "DriveSAC"
            startup_args["env"] = {
                **startup_args["env"],
                "capture_final_observations": True,
            }

            vecenv = vecenv or ppo.load_env(env_name, startup_args, seed=env_seed)
            resolved_sizes = resolve_sac_batch_sizes(config, vecenv)
            config.update(
                batch_size=resolved_sizes["batch_size"],
                minibatch_size=resolved_sizes["minibatch_size"],
            )
            train_args["batch_size"] = config["batch_size"]
            train_args["minibatch_size"] = config["minibatch_size"]
            driver_env = getattr(vecenv, "driver_env", None)
            if not getattr(driver_env, "capture_final_observations", False):
                raise NotImplementedError(
                    "SAC requires Drive final-observation transport. Set "
                    "env.capture_final_observations=True and use a Drive build with "
                    "the _pufferlib_final_observations info payload."
                )

            if policy is None:
                policy = load_policy(startup_args, vecenv, env_name)
            if critic is None:
                from pufferlib.ocean.torch_sac import DriveSACCritic
                critic = DriveSACCritic(vecenv.driver_env, **startup_args["policy"]).to(
                    device=ppo.torch_device(config["device"]), dtype=torch.float32,
                )
            if replay is None:
                from pufferlib.sac_replay import TensorReplay
                replay = TensorReplay(
                    config["batch_size"], vecenv.single_observation_space.shape, device="cpu",
                )
            config["action_count"] = policy.action_count

            startup_args["git"] = ppo._get_git_metadata()
            ppo._save_experiment_config(startup_args, config["data_dir"])

            if logger is None:
                if startup_args.get("neptune"):
                    logger = ppo.NeptuneLogger(startup_args)
                elif startup_args.get("wandb"):
                    logger = ppo.WandbLogger(startup_args)
                elif startup_args.get("tb"):
                    logger = ppo.TensorBoardLogger(
                        run_id=run_name,
                        experiment_dir=config["data_dir"],
                    )
            cleanup_logger = logger
        else:
            # Keep the original flat config + injected-component API intact.
            config = {**args, "env": env_name}

        pufferl = PuffeSAC(config, vecenv, policy, critic, replay, logger)
        cleanup_logger = pufferl.logger
        resume_state_path = pufferl.config["resume_state_path"]
        if resume_state_path is not None:
            pufferl.load_training_state(resume_state_path)
        remaining_steps = max(0, pufferl.config["total_timesteps"] - pufferl.global_step)
        for _ in range(remaining_steps):
            if pufferl.global_step >= pufferl.config["total_timesteps"]:
                break
            pufferl.evaluate()
            logs = pufferl.train()
            if logs is None:
                continue
            last_logs = logs
            if early_stop_fn is not None and early_stop_fn(logs):
                early_stop = True
                break
        model_path = pufferl.close()
        close_started = True
        pufferl.logger.close(model_path, early_stop)
        return last_logs
    except BaseException:
        if pufferl is not None and not pufferl.closed:
            pufferl.closed = True
            try:
                pufferl.vecenv.close()
            except BaseException:
                pass
            finally:
                try:
                    pufferl.utilization.stop()
                    pufferl.utilization.join()
                except BaseException:
                    pass
        elif pufferl is None:
            if vecenv is not None:
                try:
                    vecenv.close()
                except BaseException:
                    pass
        if cleanup_logger is not None and not close_started:
            if hasattr(cleanup_logger, "should_upload_model"):
                cleanup_logger.should_upload_model = False
            try:
                cleanup_logger.close(None, early_stop)
            except BaseException:
                pass
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != "train":
        raise NotImplementedError("SAC supports only: train [env_name] [Hydra overrides]")
    env_name = argv[1]
    return train(env_name=env_name, args=load_config(env_name, overrides=argv[2:]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
